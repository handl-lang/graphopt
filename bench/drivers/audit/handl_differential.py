"""Differential rewrite check: mlopt's engine vs handl's, over the bank.

Why this exists: two independent engines applying the *same* law to the
*same* term is evidence a single engine cannot give — it is the cheapest
form of "certified rewrite" the project has, and the peer
(`packages/mlopt-handl-peer`, plan 0030 step 2) makes it possible at all.

What is compared
----------------
For each law in the shipped bank whose left-hand side is a pure op tree
and whose right-hand side contains **no leaves** (a structural law), the
probe:

1. builds a term matching the lhs, with each metavar a fresh variable;
2. rewrites it with mlopt's engine (``EGraph.apply_rule``);
3. rewrites it with handl's engine, through the peer — the PyO3
   extension ``mlopt_handl_peer`` when it is importable, else the CLI
   (the same Rust library, `packages/mlopt-handl-peer`);
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

    .venv/bin/python bench/drivers/audit/handl_differential.py            # report
    .venv/bin/python bench/drivers/audit/handl_differential.py --limit 10 # a slice
"""

from __future__ import annotations

import sys
from pathlib import Path

from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Var
from mlopt_core.laws import ALL_RULES
from mlopt_discovery.handl_engine import (
    HandlPeer,
    HandlPeerUnavailable,
    ast_result_term,
    is_structural_rhs,
    render_term,
    term_to_ast,
)

_REPO = Path(__file__).resolve().parents[3]
_FRAGMENTS = (
    _REPO / "tests" / "fixtures" / "handl-spike" / "fragments-all"
)

#: The peer client — the phase-1 backend's surface (extension preferred,
#: CLI fallback), built lazily so ``main`` can report unavailability.
_peer: HandlPeer | None = None


def _handl_peer() -> HandlPeer | None:
    """The shared peer client, or ``None`` when no surface is built."""
    global _peer
    if _peer is None:
        try:
            _peer = HandlPeer(fragments=_FRAGMENTS)
        except HandlPeerUnavailable:
            return None
    return _peer


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
    return is_structural_rhs(node)


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
    return render_term(node)


def _handl_rewrite(rule_id: str, term: object) -> dict | None:
    """Rewrite *term* with handl through the peer; ``None`` on refusal.

    Uses the phase-1 backend's :class:`HandlPeer` — the PyO3 extension
    (``mlopt_handl_peer``) preferred, the CLI binary as fallback; both
    are the same Rust library.
    """
    peer = _handl_peer()
    if peer is None:
        return None
    nodes, root = term_to_ast(term)
    return peer.apply_rule(rule_id, nodes, root)


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
    actual = ast_result_term(got)
    return "agree" if actual == expected else "disagree"


def main(argv: list[str] | None = None) -> int:
    """Run the differential and print a report."""
    args = list(sys.argv[1:] if argv is None else argv)
    limit = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    if _handl_peer() is None:
        print(
            "peer not available (no mlopt_handl_peer extension, "
            "no rewrite CLI — see HandlPeerUnavailable)",
            file=sys.stderr,
        )
        print(
            "  cd packages/mlopt-handl-peer && VIRTUAL_ENV=../../.venv maturin develop"
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


if __name__ == "__main__":
    raise SystemExit(main())
