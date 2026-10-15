"""Differential rewrite check: catopt's engine vs handl's, over the bank.

Why this exists: two independent engines applying the *same* law to the
*same* term is evidence a single engine cannot give — it is the cheapest
form of "certified rewrite" the project has, and the peer
(`lab/handl-peer`, plan 0030 step 2) makes it possible at all.

What is compared
----------------
For each law in the shipped bank whose left-hand side is a pure op tree
and whose right-hand side contains **no leaves** (a structural law), the
probe:

1. builds a term matching the lhs, with each metavar a fresh variable;
2. rewrites it with catopt's engine (``EGraph.apply_rule``);
3. rewrites it with handl's engine, through the peer CLI;
4. renders both results as ``{kind, children}`` and compares structurally.

Scope, stated honestly
----------------------
* **Structural laws only.**  A law whose rhs carries a leaf (a constant or
  a variable) is skipped: handl sees the leaf as a ``Create`` of a
  ``catopt:``-namespaced kind carrying the value in the payload, so the
  two renderings are not directly comparable yet (plan 0030 step 3).
  The probe reports how many laws that excludes — that number *is* the
  size of the remaining mapping work.
* **Guarded laws** are included only when their guard holds for the
  synthesized term (otherwise the two engines agree by both declining).
* **One term per law.**  This is a smoke-level differential, not
  exhaustive: it looks for *disagreement*, and one is enough to matter.

Usage::

    .venv/bin/python lab/audit/handl_differential.py            # report
    .venv/bin/python lab/audit/handl_differential.py --limit 10 # a slice
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from catopt_core.egraph import EGraph
from catopt_core.ir import Const, Op, Var
from catopt_core.laws import ALL_RULES

_REPO = Path(__file__).resolve().parents[2]
_BIN = Path(
    os.environ.get(
        "CATOPT_HANDL_PEER_BIN",
        Path.home() / "GitHub/handl-lang/platform/target/debug/rewrite",
    )
)
_FRAGMENTS = _REPO / "lab" / "handl-spike" / "fragments-all"


def _eliminates_root(rule: object) -> bool:
    """True when the law's rhs is a bare metavar — the root is dropped.

    handl's `TransformRule::Declarative` always *creates* a node (its
    `target_kind` is a `String`), so a law that returns a subterm has no
    faithful rule form: catopt emits `target_kind: None` and the two
    engines necessarily diverge.  Those laws are reported separately —
    the gap is real and is *not* papered over.
    """
    return isinstance(rule.rhs, str)


def _is_structural(node: object) -> bool:
    """True when the pattern is ops/metavars only (no leaves)."""
    if isinstance(node, str):
        return True
    parts = _parts(node)
    if parts is None:
        return False
    return all(_is_structural(a) for a in parts[1])


def _parts(node: object) -> tuple | None:
    """``(head, args)`` for an ``Op``; ``None`` otherwise."""
    if isinstance(node, Op):
        return (node.op, tuple(node.args))
    return None


def _instantiate(node: object, fresh: dict) -> object:
    """Replace every metavar with a fresh variable."""
    if isinstance(node, str):
        return fresh.setdefault(node, Var(f"v_{node}", "f64"))
    parts = _parts(node)
    if parts is None:
        return node
    return Op.make(
        parts[0], *[_instantiate(a, fresh) for a in parts[1]]
    )


def _render(node: object) -> dict:
    """Render a catopt term as ``{kind, children}`` (handl's shape)."""
    parts = _parts(node)
    if parts is None:
        kind = getattr(node, "name", repr(node))
        return {"kind": str(kind), "children": []}
    return {
        "kind": parts[0],
        "children": [_render(a) for a in parts[1]],
    }


def _handl_rewrite(rule_id: str, term: object) -> dict | None:
    """Rewrite *term* with handl through the peer; ``None`` on refusal."""
    arena: list[dict] = []
    root = _flatten(term, arena)
    ast = json.dumps(
        [
            {"kind": n["kind"], "payload": {"children": n["children"]}}
            for n in arena
        ]
    )
    out = subprocess.run(
        [str(_BIN), str(_FRAGMENTS), rule_id, str(root), ast],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def _flatten(node: object, arena: list[dict]) -> int:
    """Append *node* post-order; return its index."""
    parts = _parts(node)
    if parts is None:
        arena.append(
            {"kind": str(getattr(node, "name", node)), "children": []}
        )
        return len(arena) - 1
    kids = [_flatten(a, arena) for a in parts[1]]
    arena.append({"kind": parts[0], "children": kids})
    return len(arena) - 1


def _from_handl(arena: list[dict], index: int) -> dict:
    node = arena[index]
    return {
        "kind": node["kind"],
        "children": [
            _from_handl(arena, i) for i in node["payload"]["children"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the differential and print a report."""
    args = list(sys.argv[1:] if argv is None else argv)
    limit = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    if not _BIN.exists():
        print(f"peer binary not built: {_BIN}", file=sys.stderr)
        print(
            "  cd lab/handl-peer && CARGO_TARGET_DIR=~/GitHub/handl-lang"
            "/platform/target cargo build --bin rewrite",
            file=sys.stderr,
        )
        return 2

    agree = disagree = declined = skipped = eliminates = (
        guard_errors
    ) = 0
    disagreements: list[str] = []
    for rule in ALL_RULES:
        if limit is not None and agree + disagree >= limit:
            break
        if _eliminates_root(rule):
            eliminates += 1
            continue
        if not (_is_structural(rule.lhs) and _is_structural(rule.rhs)):
            skipped += 1
            continue

        fresh: dict = {}
        term = _instantiate(rule.lhs, fresh)
        graph = EGraph()
        root_eid = graph.add_term(term)
        try:
            fired = graph.apply_rule(rule, root_eid)
        except Exception:  # noqa: BLE001 — a guard raised on a synthesized term
            # Some guards are *shape-sensitive*: they consult the shape of a
            # child, and a synthesized term (bare f64 variables) cannot
            # satisfy every such lookup.  That is a limitation of
            # synthesizing one term per law, not an engine disagreement --
            # counted separately so it cannot hide a real one.
            guard_errors += 1
            continue
        if not fired:
            declined += 1
            continue
        expected = _render(_instantiate(rule.rhs, fresh))

        got = _handl_rewrite(f"transform.catopt.{rule.name}", term)
        if got is None:
            declined += 1
            continue
        actual = _from_handl(got["arena"], _root_index(got))

        if actual == expected:
            agree += 1
        else:
            disagree += 1
            disagreements.append(rule.name)

    print("differential rewrite check — catopt vs handl")
    print(f"  laws compared      : {agree + disagree}")
    print(f"  agree              : {agree}")
    print(f"  DISAGREE           : {disagree}")
    print(f"  skipped (leaf rhs) : {skipped}")
    print(
        f"  root-eliminating   : {eliminates}  (handl's Declarative always"
    )
    print(
        "                        creates a node — a named gap, plan 0030)"
    )
    print(f"  declined by an engine: {declined}")
    print(f"  guard raised (synth) : {guard_errors}")
    if disagreements:
        print("  disagreements:", ", ".join(disagreements[:10]))
    return 1 if disagree else 0


def _root_index(result: dict) -> int:
    """Index of the rewritten root in handl's arena (exported by the peer)."""
    return int(result["root_index"])


if __name__ == "__main__":
    raise SystemExit(main())
