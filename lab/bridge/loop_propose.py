"""A ``loop``-proposing rule over the combinator presentation (ADR-0008 gate 3).

Gate 3 splits the presentation question in two:

* **as an expression** — PASS.  Gate 2 shows a hand-written
  ``loop(step, n)`` term lowers and verifies at 1e-7.
* **as a search space** — open.  The DAG bridge never *mints* ``loop``
  (the export erased the recurrence), so a ``loop``-headed term is only
  reachable if something **proposes** it.  Gate 3 asks whether a
  proposing rule exists and whether the search *selects* the result.

THE RULE
    ``unroll(step, n)  →  Loop(step, n)``

``unroll(step, n)`` is the presentation's *spelled* bounded scan: the
same ``step`` morphism applied ``n`` times, with the wire bookkeeping
made explicit (a ``perm`` before each step, one final ``perm``).  It is
the honest spelling of the recurrence *in the presentation* — the DAG
the search reaches has the same structure, but as a flat sequence of
``arr`` nodes.  ``Loop(step, n)`` is the combinator.  The two denote the
same morphism: ``comb_to_dag(unroll(step, n)) == comb_to_dag(Loop(step,
n))`` (checked in the tests), which is what makes the rule sound.

THE SEARCH
    :func:`saturate_comb` — a deliberately small term-level saturation
    over morphism terms: apply every rule at every node to a fixpoint,
    dedupe, extract the cheapest by a cost model.  No e-graph
    congruence (the presentation has no e-graph); the point is
    reachability + selection, not speed.  A real combinator search is
    ADR-0009 stage-4 work — this module measures whether that work has
    a target at all.

The measurement lives in ``lab/bridge/combinator_probe.py``
(``--propose-loop``); the honest reading of the gate-3 numbers is in
``project/adrs/0008-combinator-presentation.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mlopt_discovery.combinator import (
    Id,
    Loop,
    Par,
    Seq,
    _par,
    _perm,
    _seq,
    arity,
    morph_stats,
)

__all__ = [
    "EXPAND_LOOP",
    "PROPOSE_LOOP",
    "CombRule",
    "loop_headed",
    "propose_loop_report",
    "saturate_comb",
    "unroll",
]


# ---------------------------------------------------------------------------
#  The spelled recurrence: unroll(step, n)
# ---------------------------------------------------------------------------


def unroll(step: Any, n: int) -> Any:
    """Spell ``Loop(step, n)`` as an explicit ``n``-step morphism.

    ``step : S ⊗ X → S ⊗ Y`` with arity ``(1 + nx, 1 + ny)``.  The
    unrolled term consumes ``S ⊗ X^n`` (``1 + n·nx`` wires) and produces
    ``S' ⊗ Y^n`` (``1 + n·ny`` wires) — the same morphism
    :func:`~mlopt_discovery.combinator.comb_to_dag` builds by expanding
    ``Loop``, but written as a ``seq`` of ``perm``/``step`` segments so
    the shared ``step`` stays ONE subterm (that sharing is exactly what
    the flat DAG compilation loses).

    Wire bookkeeping: after segment ``i`` the bundle is ``[S_i, Y^i,
    Y^{i-1}, …, Y^1, X^{i+1}, …, X^n]``; segment ``i+1`` permutes
    ``X^{i+1}`` next to ``S_i``, applies ``step``, and a closing perm
    restores the ``[S_n, Y^1, …, Y^n]`` order ``Loop`` produces.
    """
    n_in, n_out = arity(step)
    nx, ny = n_in - 1, n_out - 1
    if n < 1:
        raise ValueError("unroll needs n >= 1")
    if nx < 1:
        raise ValueError("a scan step consumes the state plus an input")
    parts: list = []
    for i in range(1, n + 1):
        y = (i - 1) * ny  # Y wires already emitted (Y^{i-1} … Y^1)
        x0 = y + 1  # first wire of X^i
        order = [
            0,
            *range(x0, x0 + nx),  # X^i to the front
            *range(1, y + 1),  # Y^{i-1} … Y^1
            *range(x0 + nx, y + (n - i + 1) * nx + 1),  # X^{i+1} … X^n
        ]
        prm = _perm(order)
        if not isinstance(prm, Id):
            parts.append(prm)
        parts.append(_par(step, Id(y + (n - i) * nx)))
    # [S_n, Y^n … Y^1] -> [S_n, Y^1 … Y^n]
    fix = _perm(
        [
            0,
            *(
                j
                for blk in reversed(range(n))
                for j in range(blk * ny + 1, blk * ny + ny + 1)
            ),
        ]
    )
    if not isinstance(fix, Id):
        parts.append(fix)
    return _seq(*parts)


# ---------------------------------------------------------------------------
#  Subterm traversal over morphism terms
# ---------------------------------------------------------------------------


def _children(m: Any) -> tuple:
    """Return the direct subterms of a morphism node."""
    if isinstance(m, Loop):
        return (m.step,)
    if isinstance(m, (Seq, Par)):
        return tuple(m.parts)
    return ()


def _subterms(m: Any) -> list:
    """Return the distinct subterms of ``m`` (cycle-free tree)."""
    seen: set = set()
    out: list = []
    stack = [m]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        stack.extend(_children(t))
    return out


def _replace(m: Any, target: Any, new: Any) -> Any:
    """Rebuild ``m`` with every occurrence of ``target`` replaced by ``new``."""
    if m == target:
        return new
    if isinstance(m, Loop):
        return Loop(_replace(m.step, target, new), m.n, m.nx, m.ny)
    if isinstance(m, Seq):
        return Seq(tuple(_replace(p, target, new) for p in m.parts))
    if isinstance(m, Par):
        return Par(tuple(_replace(p, target, new) for p in m.parts))
    return m


def _in_arity(m: Any) -> int:
    """``arity`` of a term, or ``-1`` when it is not a morphism."""
    try:
        return arity(m)[0]
    except TypeError:
        return -1


# ---------------------------------------------------------------------------
#  The proposing rule (and its inverse)
# ---------------------------------------------------------------------------


def _folds(m: Any, max_n: int = 8) -> list[tuple[Any, Any, int]]:
    """Return every ``(spelled, step, n)`` with ``spelled == unroll(step, n)``.

    Candidate ``step``s are the subterms that consume at least two wires
    (a scan step takes the state plus at least one input); ``n`` starts
    at **2** — ``unroll(step, 1)`` is ``step`` itself, so allowing it
    would make *every* morphism a "fold" and the rule vacuous.  ``n`` is
    bounded by ``max_n`` (the ADR's "bounded scan", statically known
    iteration count).
    """
    subs = _subterms(m)
    subset = set(subs)
    out: list[tuple[Any, Any, int]] = []
    for step in subs:
        if _in_arity(step) < 2:
            continue
        for n in range(2, max_n + 1):
            spelled = unroll(step, n)
            if spelled in subset:
                out.append((spelled, step, n))
    return out


def loop_headed(m: Any) -> bool:
    """Return True when ``m`` carries a ``Loop`` node anywhere."""
    return any(isinstance(s, Loop) for s in _subterms(m))


@dataclass(frozen=True)
class CombRule:
    """A rewrite over morphism terms: a ``find``/``build`` pair.

    ``find(m, max_n)`` returns ``(matched_subterm, step, n)`` triples —
    the occurrences this rule rewrites.  ``build(step, n)`` returns the
    replacement, so one rule object folds *any* spelled scan rather than
    a single hand-picked one.
    """

    name: str
    find: Callable[[Any, int], list[tuple[Any, Any, int]]]
    build: Callable[[Any, int], Any]
    max_n: int = 8

    def apply(self, m: Any) -> list[Any]:
        """Return the terms ``m`` rewrites to (one occurrence per result)."""
        out: list[Any] = []
        for matched, step, n in self.find(m, self.max_n):
            new = self.build(step, n)
            if new == matched:
                continue
            out.append(_replace(m, matched, new))
        return out


def _to_loop(step: Any, n: int) -> Loop:
    """Build ``Loop(step, n)`` with the step's own per-step arity."""
    n_in, n_out = arity(step)
    return Loop(step, n, n_in - 1, n_out - 1)


def _find_loops(m: Any, max_n: int = 8) -> list[tuple[Any, Any, int]]:
    """Return every ``(loop, step, n)`` with ``loop == Loop(step, n, …)``."""
    out: list[tuple[Any, Any, int]] = []
    for s in _subterms(m):
        if isinstance(s, Loop):
            out.append((s, s.step, s.n))
    return out


#: The proposing rule: the spelled ``n``-step scan becomes a ``Loop``.
PROPOSE_LOOP = CombRule("loop_propose", _folds, _to_loop)

#: The inverse — a ``Loop`` re-spelled as its unrolled scan.  Co-resident
#: with :data:`PROPOSE_LOOP` the pair is a two-cycle, exactly like the
#: ``div``/``matmul`` naturalities the sequential-carrier probe has to
#: schedule apart; here it is what makes the closure a *search* (both
#: spellings are candidates and extraction picks).
EXPAND_LOOP = CombRule("loop_expand", _find_loops, unroll)


# ---------------------------------------------------------------------------
#  The search: a naive term-level saturation + cost extraction
# ---------------------------------------------------------------------------


def comb_cost(m: Any) -> int:
    """Cost of a morphism term: its node count (``Loop`` counts as one)."""
    return morph_stats(m).nodes


def saturate_comb(
    seed: Any,
    rules: list[CombRule],
    cost: Callable[[Any], int] = comb_cost,
    max_terms: int = 4000,
) -> dict[str, Any]:
    """Rewrite ``seed`` to a fixpoint; return the cheapest term found.

    Returns ``{"best", "cost", "loop_reachable", "loop_best_cost",
    "n_terms", "fired"}``.  ``loop_reachable`` is the gate's question:
    did *any* reachable term carry a ``Loop``?
    """
    seen: set = {seed}
    work = [seed]
    fired: dict[str, int] = {r.name: 0 for r in rules}
    best = seed
    best_cost = cost(seed)
    loop_best: Any = None
    loop_best_cost: int | None = None
    while work and len(seen) < max_terms:
        t = work.pop()
        for rule in rules:
            for nt in rule.apply(t):
                if nt in seen:
                    continue
                seen.add(nt)
                work.append(nt)
                fired[rule.name] += 1
                c = cost(nt)
                if c < best_cost:
                    best, best_cost = nt, c
                if loop_headed(nt) and (
                    loop_best_cost is None or c < loop_best_cost
                ):
                    loop_best, loop_best_cost = nt, c
    return {
        "best": best,
        "cost": best_cost,
        "loop_reachable": loop_best is not None,
        "loop_best": loop_best,
        "loop_best_cost": loop_best_cost,
        "n_terms": len(seen),
        "fired": fired,
    }


# ---------------------------------------------------------------------------
#  The measurement
# ---------------------------------------------------------------------------


def propose_loop_report(morph: Any, control: Any | None = None) -> None:
    """Print the gate-3 proposing-rule measurement for a projection.

    ``morph`` is the projection of the search's extracted winner (the
    term gate 3 must decide on).  ``control`` — when given — is the
    *authored* spelled scan (``unroll(retnet_step, k)``), which shows the
    rule is well-formed independently of whether the search reaches it.
    """
    folds = _folds(morph)
    ns = sorted({n for _s, _st, n in folds})
    print(
        f"   loop-propose: winner morph={morph_stats(morph).nodes} "
        f"spelled-scan subterms (n>=2) found={len(folds)} ns={ns}"
    )
    if folds:
        for _spelled, _step, n in folds[:4]:
            print(f"      match: unroll(step, {n})")
    else:
        print(
            "      no `unroll(step, n>=2)` subterm — the winner is a flat "
            "DAG compilation, not the presentation's spelled scan"
        )
    res = saturate_comb(morph, [PROPOSE_LOOP, EXPAND_LOOP])
    print(
        f"      saturation: terms={res['n_terms']} "
        f"fired={res['fired']}  loop-headed reachable="
        f"{res['loop_reachable']}  best_is_loop={loop_headed(res['best'])}"
        f"  best_cost={res['cost']}"
    )
    if control is not None:
        cfolds = _folds(control)
        cns = sorted({n for _s, _st, n in cfolds})
        cres = saturate_comb(control, [PROPOSE_LOOP, EXPAND_LOOP])
        print(
            f"   control (authored spelled scan): "
            f"spelled-scan subterms={len(cfolds)} ns={cns} "
            f"fired={cres['fired']}  loop-headed reachable="
            f"{cres['loop_reachable']}  "
            f"best_is_loop={loop_headed(cres['best'])}  "
            f"best_cost={cres['cost']} (spelled "
            f"{morph_stats(control).nodes})"
        )
