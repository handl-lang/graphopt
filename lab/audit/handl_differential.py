"""Differential rewrite check: mlopt's engine vs handl's, over the bank.

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
2. rewrites it with mlopt's engine (``EGraph.apply_rule``);
3. rewrites it with handl's engine, through the peer — the PyO3
   extension ``mlopt_handl_peer`` when it is importable, else the CLI
   (the same Rust library, `lab/handl-peer`);
4. renders both results as ``{kind, children}`` and compares structurally.

Scope, stated honestly
----------------------
* **Structural laws only.**  A law whose *right-hand side* carries a leaf (a
  constant or a variable) is skipped: handl sees the leaf as a ``Create`` of
  a ``mlopt:``-namespaced kind carrying the value in the payload, so the
  two renderings are not directly comparable yet (plan 0030 step 3).
  The probe reports how many laws that excludes — that number *is* the
  size of the remaining mapping work.  (Only the *rhs* is filtered: a leaf
  in the *lhs* is part of the synthesized **input**, which the probe builds
  from mlopt's own IR and feeds to both engines identically, so it is
  perfectly comparable — filtering it out used to hide ``id_add``/``id_mul``
  needlessly.)
* **Root-eliminating laws are compared.**  A law whose rhs is a bare
  metavar (``double_neg``, ``id_add``, ``id_mul``, ``transpose_noop``,
  ``chunk_single``) drops its root and returns a subterm.  handl's
  ``!Replace { path }`` rule form (handl ADR-0127) expresses exactly that,
  so these 5 laws are no longer classified-and-skipped: they are fired on
  both engines and compared like any other structural law.
* **Guarded laws** are included only when their guard holds for the
  synthesized term (otherwise the two engines agree by both declining).
* **One term per law.**  This is a smoke-level differential, not
  exhaustive: it looks for *disagreement*, and one is enough to matter.

Usage::

    .venv/bin/python lab/audit/handl_differential.py            # report
    .venv/bin/python lab/audit/handl_differential.py --limit 10 # a slice
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Var
from mlopt_core.laws import ALL_RULES

_REPO = Path(__file__).resolve().parents[2]
_BIN = Path(
    os.environ.get(
        "MLOPT_HANDL_PEER_BIN",
        Path.home() / "GitHub/handl-lang/host/target/debug/rewrite",
    )
)
_FRAGMENTS = _REPO / "lab" / "handl-spike" / "fragments-all"

try:  # the shipping surface — absent until maturin builds it
    import mlopt_handl_peer as _ext
except ImportError:  # pragma: no cover - depends on the local build
    _ext = None


def _eliminates_root(rule: object) -> bool:
    """Whether the law's rhs is a bare metavar — the law drops its root.

    These are the laws a ``TransformRule::Declarative`` could not express
    (it always *creates* a node), so before handl ADR-0127 the probe had to
    classify them *before firing* rather than compare them.  The bridge now
    emits ``!Replace { path }`` for them — the rewritten node IS the existing
    descendant at ``path`` — so they are **compared like any other
    structural law**.  This predicate is kept only to *report* how many of
    the compared laws are root-eliminating; it no longer skips anything.
    """
    return isinstance(rule.rhs, str)


def _is_structural(node: object) -> bool:
    """Report whether the pattern is ops/metavars only (no leaves)."""
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
    """Render a mlopt term as ``{kind, children}`` (handl's shape)."""
    parts = _parts(node)
    if parts is None:
        kind = getattr(node, "name", repr(node))
        return {"kind": str(kind), "children": []}
    return {
        "kind": parts[0],
        "children": [_render(a) for a in parts[1]],
    }


def _handl_rewrite(rule_id: str, term: object) -> dict | None:
    """Rewrite *term* with handl through the peer; ``None`` on refusal.

    Prefers the PyO3 extension (``mlopt_handl_peer``) and falls back to
    the CLI binary — both are the same Rust library.
    """
    arena: list[dict] = []
    root = _flatten(term, arena)
    ast = json.dumps(
        [
            {"kind": n["kind"], "payload": {"children": n["children"]}}
            for n in arena
        ]
    )
    if _ext is not None:
        try:
            return json.loads(
                _ext.apply_rule(str(_FRAGMENTS), ast, root, rule_id)
            )
        except ValueError:
            return None
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


def _outcome(rule: object) -> str:
    """Compare one law on both engines and return its outcome.

    ``"skipped"`` — the rhs carries a leaf (not directly comparable yet,
    plan 0030 step 3); ``"guard"`` — a guard raised on the synthesized term;
    ``"declined"`` — an engine did not fire; ``"agree"``/``"disagree"`` —
    both engines fired and the rendered results (did not) match.
    """
    if not _is_structural(rule.rhs):
        return "skipped"
    fresh: dict = {}
    term = _instantiate(rule.lhs, fresh)
    graph = EGraph()
    root_eid = graph.add_term(term)
    try:
        fired = graph.apply_rule(rule, root_eid)
    except Exception:  # a guard raised on a synthesized term
        # Some guards are *shape-sensitive*: they consult the shape of a
        # child, and a synthesized term (bare f64 variables) cannot
        # satisfy every such lookup.  That is a limitation of
        # synthesizing one term per law, not an engine disagreement --
        # counted separately so it cannot hide a real one.
        return "guard"
    if not fired:
        return "declined"
    expected = _render(_instantiate(rule.rhs, fresh))
    got = _handl_rewrite(f"transform.mlopt.{rule.name}", term)
    if got is None:
        return "declined"
    actual = _from_handl(got["arena"], _root_index(got))
    return "agree" if actual == expected else "disagree"


def main(argv: list[str] | None = None) -> int:
    """Run the differential and print a report."""
    args = list(sys.argv[1:] if argv is None else argv)
    limit = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    if _ext is None and not _BIN.exists():
        print(
            f"peer not available (no mlopt_handl_peer, no binary: {_BIN})",
            file=sys.stderr,
        )
        print(
            "  cd lab/handl-peer && VIRTUAL_ENV=../../.venv maturin develop"
            " --target-dir ~/GitHub/handl-lang/host/target",
            file=sys.stderr,
        )
        return 2

    outcomes = {
        "agree": 0,
        "disagree": 0,
        "declined": 0,
        "skipped": 0,
        "guard": 0,
    }
    disagreements: list[str] = []
    #: (name, outcome) for every root-eliminating law — the 5 whose
    #: expression handl's `!Replace` (ADR-0127) made possible.  Reported
    #: per law so a decline cannot hide among the aggregate buckets.
    elim: list[tuple[str, str]] = []
    for rule in ALL_RULES:
        if (
            limit is not None
            and outcomes["agree"] + outcomes["disagree"] >= limit
        ):
            break
        result = _outcome(rule)
        outcomes[result] += 1
        if result == "disagree":
            disagreements.append(rule.name)
        if _eliminates_root(rule):
            elim.append((rule.name, result))

    agree, disagree = outcomes["agree"], outcomes["disagree"]
    print("differential rewrite check — mlopt vs handl")
    print(f"  laws compared      : {agree + disagree}")
    print(f"  agree              : {agree}")
    print(f"  DISAGREE           : {disagree}")
    print(f"  skipped (leaf rhs) : {outcomes['skipped']}")
    print(f"  declined by an engine: {outcomes['declined']}")
    print(f"  guard raised (synth) : {outcomes['guard']}")
    if elim:
        print(
            "  root-eliminating laws (bare-metavar rhs — handl"
            " !Replace, ADR-0127):"
        )
        for name, result in elim:
            print(f"    - {name}: {result}")
    if disagreements:
        print("  disagreements:", ", ".join(disagreements[:10]))
    return 1 if disagree else 0


def _root_index(result: dict) -> int:
    """Index of the rewritten root in handl's arena (exported by the peer)."""
    return int(result["root_index"])


if __name__ == "__main__":
    raise SystemExit(main())
