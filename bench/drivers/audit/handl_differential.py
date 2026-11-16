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
* **Guarded laws** are instantiated with *attribute* metavars bound to
  a small candidate pool, so a law is exercised on both sides of its
  guard — the Phase-B decline-parity arm.  Each (law, instantiation)
  pair is classified by the *pair* of engine outcomes: ``agree`` /
  ``disagree`` (both fired), ``overfire`` (**mlopt declined, handl
  fired** — the bug class this arm exists to catch), ``underfire``
  (mlopt fired, faithful handl rule declined), ``refused`` (mlopt
  fired where the bridge emitted ``!Native`` — the honest gap), and
  ``both_declined`` (the parity evidence).  ``overfire`` and
  ``disagree`` fail the run; ``underfire``/``refused`` are reported.
* **A handful of terms per law.**  Structural laws get one
  instantiation; laws with attribute metavars get up to
  ``_MAX_ATTR_CANDIDATES``.  This is a smoke-level differential, not
  exhaustive: it looks for *disagreement* and *over-fire*, and one of
  either is enough to matter.

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
from kit_discovery.handl_engine import (
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


def _instantiate(
    node: object, fresh: dict, attrs: dict | None = None
) -> object:
    """Replace every metavar with a fresh variable — attrs included.

    ``attrs`` maps *attribute* metavars (the string values an ``Op``'s
    ``attrs`` dict carries in a pattern — mlopt binds them under
    ``"$attr:<name>"``) to the concrete values this candidate
    instantiation chose; unbound names stay literal strings, matching
    mlopt's own ``subst.get("$attr:" + v, v)`` fallback.  Literal attr
    values pass through untouched, so an lhs like ``eye(n=4)`` still
    requires ``n == 4`` on both engines.
    """
    if isinstance(node, str):
        return fresh.setdefault(node, Var(f"v_{node}", "f64"))
    parts = _parts(node)
    if parts is None:
        return node
    kwargs: dict = {}
    if isinstance(node, Op) and node.attrs:
        for k, v in node.attrs.items():
            if isinstance(v, str):
                kwargs[k] = (attrs or {}).get(v, v)
            else:
                kwargs[k] = v
    return Op.make(
        parts[0],
        *[_instantiate(a, fresh, attrs) for a in parts[1]],
        **kwargs,
    )


def _attr_metavars(node: object, out: set | None = None) -> set:
    """Collect the attribute-metavar names in an lhs pattern."""
    if out is None:
        out = set()
    if isinstance(node, str):
        return out
    parts = _parts(node)
    if parts is None:
        return out
    if isinstance(node, Op):
        for v in node.attrs.values():
            if isinstance(v, str):
                out.add(v)
    for a in parts[1]:
        _attr_metavars(a, out)
    return out


#: Candidate values for attribute metavars when synthesizing inputs.
#: Small ints cover the attr guards the bank actually carries
#: (``attr-eq CK 1``, dim/index comparisons); anything richer shows up
#: as a ``guard``/``skipped`` outcome, not a silent pass.
_ATTR_POOL = (0, 1, 2)

#: Cap on instantiations per law — enough to hit both sides of an
#: ``attr-eq``-style guard without a combinatorial sweep.
_MAX_ATTR_CANDIDATES = 8


def _attr_candidates(rule: object) -> list[dict]:
    """Yield up to ``_MAX_ATTR_CANDIDATES`` attr assignments for *rule*.

    Laws without attr metavars yield one empty assignment (the
    pre-attrs behavior).  Otherwise the product of ``_ATTR_POOL`` over
    the metavars is enumerated — deduplicated and truncated — which
    lands on both sides of literal guards like ``attr-eq CK 1`` for
    the values the pool contains.
    """
    import itertools

    mvars = sorted(_attr_metavars(rule.lhs))
    if not mvars:
        return [{}]
    out: list[dict] = []
    seen: set = set()
    for combo in itertools.product(_ATTR_POOL, repeat=len(mvars)):
        assign = dict(zip(mvars, combo, strict=True))
        key = tuple(assign[m] for m in mvars)
        if key in seen:
            continue
        seen.add(key)
        out.append(assign)
        if len(out) >= _MAX_ATTR_CANDIDATES:
            break
    return out


def _render(node: object) -> dict:
    """Render a mlopt term as ``{kind, children[, attrs]}`` (handl's shape)."""
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


def _handl_refused(rule: object) -> bool:
    """Whether the bridge emitted an explicit ``!Native`` refusal.

    A refused law declines on *every* input by construction — mlopt
    firing where handl refuses is the honest gap the refusal names,
    not an over-fire; it is bucketed ``refused`` rather than counted
    against parity.
    """
    from kit_discovery.handl import law_to_handl

    row = law_to_handl(rule, rule.name)
    rule_body = row.get("rule") or {}
    return "Native" in rule_body


def _outcome(rule: object, attrs: dict) -> str:
    """Compare one law on one instantiation; return the outcome pair's verdict.

    ``"skipped"`` — the rhs carries a leaf (not directly comparable yet,
    plan 0030 step 3); ``"guard"`` — a guard raised on the synthesized
    term; ``"overfire"`` — **mlopt declined and handl fired** (the
    Phase-B bug class: an unguarded projection rewriting where the law
    says no); ``"underfire"`` — mlopt fired and a *faithful* handl rule
    declined; ``"refused"`` — mlopt fired where the bridge emitted an
    explicit ``!Native``; ``"both_declined"`` — the decline-parity
    evidence; ``"declined"`` — mlopt fired and handl declined on a
    leaf-rhs-skipped path is unreachable, kept for clarity;
    ``"agree"``/``"disagree"`` — both engines fired and the rendered
    results (did not) match.
    """
    if not _is_structural(rule.rhs):
        return "skipped"
    fresh: dict = {}
    try:
        term = _instantiate(rule.lhs, fresh, attrs)
    except (TypeError, ValueError):
        # An attr assignment the op's ATTR_SCHEMA rejects — not a
        # term either engine could see; skip the candidate.
        return "skipped"
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
    got = _handl_rewrite(f"transform.mlopt.{rule.name}", term)
    handl_fired = got is not None
    if not fired:
        # mlopt declined — the decline-parity arm: handl must decline
        # too.  Firing here is the over-fire Phase B measured.
        return "overfire" if handl_fired else "both_declined"
    if not handl_fired:
        return "refused" if _handl_refused(rule) else "underfire"
    expected = _render(_instantiate(rule.rhs, fresh, attrs))
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
        "overfire": 0,
        "underfire": 0,
        "both_declined": 0,
        "refused": 0,
        "skipped": 0,
        "guard": 0,
    }
    disagreements: list[str] = []
    overfires: list[str] = []
    underfires: list[str] = []
    #: (name, outcome) for every root-eliminating law — the 5 whose
    #: expression handl's `!Replace` (ADR-0127) made possible.  Reported
    #: per law so a decline cannot hide among the aggregate buckets.
    elim: list[tuple[str, str]] = []
    #: Worst outcome seen per law across its attr candidates — a law
    #: only "passes" when no instantiation over-fires or disagrees.
    law_worst: dict[str, str] = {}
    _SEVERITY = {
        "agree": 0,
        "both_declined": 0,
        "skipped": 0,
        "guard": 0,
        "refused": 1,
        "underfire": 2,
        "disagree": 3,
        "overfire": 4,
    }
    for rule in ALL_RULES:
        if (
            limit is not None
            and outcomes["agree"] + outcomes["disagree"] >= limit
        ):
            break
        for attrs in _attr_candidates(rule):
            result = _outcome(rule, attrs)
            outcomes[result] += 1
            prev = law_worst.get(rule.name)
            if prev is None or _SEVERITY[result] > _SEVERITY[prev]:
                law_worst[rule.name] = result
            if result == "disagree" and rule.name not in disagreements:
                disagreements.append(rule.name)
            elif result == "overfire" and rule.name not in overfires:
                overfires.append(rule.name)
            elif result == "underfire" and rule.name not in underfires:
                underfires.append(rule.name)
        if _eliminates_root(rule):
            elim.append((rule.name, law_worst[rule.name]))

    agree, disagree = outcomes["agree"], outcomes["disagree"]
    print("differential rewrite check — mlopt vs handl")
    print(f"  laws compared      : {agree + disagree}")
    print(f"  agree              : {agree}")
    print(f"  DISAGREE           : {disagree}")
    print(f"  OVERFIRE (mlopt no, handl yes): {outcomes['overfire']}")
    print(
        f"  underfire (faithful rule declined): {outcomes['underfire']}"
    )
    print(f"  both declined      : {outcomes['both_declined']}")
    print(f"  refused (!Native)  : {outcomes['refused']}")
    print(f"  skipped (leaf rhs) : {outcomes['skipped']}")
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
    if overfires:
        print("  overfires:", ", ".join(overfires[:10]))
    if underfires:
        print("  underfires:", ", ".join(underfires[:10]))
    return 1 if (disagree or outcomes["overfire"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
