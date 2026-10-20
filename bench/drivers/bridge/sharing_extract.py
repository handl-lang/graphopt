"""Sharing-aware extraction — a principled tie-break for ADR-0008 gate 3.

THE DEFECT THIS ADDRESSES
    :meth:`~mlopt_core.egraph.EGraph.extract_best` is *per-class
    greedy*: each e-class picks its cheapest member independently, and
    equal-cost members are broken by ``nops`` then by the member sort
    order ``(op, children, repr(attrs))`` — i.e. **by op name**.  The
    tie-break is therefore *arbitrary with respect to sharing*: it
    cannot see that one of two cost-tied members reuses an e-class the
    *next* class also needs, and choosing it would charge that subtree
    once instead of twice.

    Measured on the gate-3 winner (``bench/drivers/probes/gate3_state_spelling_probe.py``):
    the state read is spelled two ways (the factored ``matmul(q, S)``
    and the unfactored re-expansion ``om_decayed(matmul(q, kᵀ), D, v)``);
    5 of the 7 read e-classes have members tied at the class minimum, and
    forcing the factored read moves the whole term by
    ``Δ ∈ {−528, 0, +16, +16, +144}`` FLOPs — indifferent to within
    0.03 %, except block 4, where the factored read is *strictly cheaper*
    because its state is shared with the next block's read.  So the
    greedy pass leaves FLOPs on the table and picks an arbitrary
    spelling.

THE FIX
    :func:`extract_sharing_aware` — a bounded *coordinated* extraction.
    Start from :meth:`~mlopt_core.egraph.EGraph.extract_best`; then
    sweep the extracted term's e-classes, and for each member of each
    class re-extract with that class **forced** to that member
    (``overrides``).  Keep any override that *lowers the whole term's
    DAG cost*, to a fixed point.

    This is principled in exactly the two ways the gate-3 brief asks
    for: it **removes an arbitrary choice** (the op-name tie-break) and
    it **adds a real cost signal** (the non-local sharing the greedy
    pass cannot see).  It is *monotone* — a step is kept only when the
    total cost strictly falls — so it can never manufacture a form at a
    price the cost model did not already prefer.  It is the same
    non-local-coordination idea as
    :meth:`~mlopt_core.egraph.EGraph.extract_paired`, specialised to
    the tie-break.

    It lives in ``bench/`` deliberately: the equivalent change inside
    :mod:`mlopt_core.egraph.extract` would alter extraction for *every*
    caller (the AGENTS rule is explicit that such a change needs the
    full suite), and the measured effect is 1 % on one workload.

THE READING
    On the gate-3 winner it recovers **4,624 FLOPs (1.01 %)** —
    455,752 → 451,128 through three overrides (two ``split`` classes and
    one ``matmul``) — numerically verified equivalent to machine
    epsilon (max|Δ| = 2.2e-16).  The gate-3 abstraction **still**
    reports ``irregular``, with the *same* first difference
    (``block1/block2/add[1]/mul[1]: op om_decayed vs add``), and the
    per-block read sequence is unchanged from the greedy term
    (``reexp|reexp|fold|fold|reexp|shared|shared``): the mix is a
    *cascade* of per-class ties, not one arbitrary choice.  So the mixed
    spelling is a genuine extraction defect, but fixing it does not
    close the gate.  See
    ``project/retros/gate3-extraction-state-spelling.md``.

    **Corrected 2026-10-15** (``project/retros/saturation-determinism.md``).
    This section previously read "23,056 FLOPs (5.06 %) — 455,752 →
    432,696, … five overrides", "unifies the large-chunk family's
    top-level state read" and a changed read sequence.  That 432,696 was
    an artifact of the member scan order: the descent kept the *first*
    strictly improving member of each class, and the scan ran over a
    ``set`` whose iteration order follows ``ENode``'s hash — so the
    "optimum" moved with ``PYTHONHASHSEED`` (432,696 with 5 overrides at
    ``PYTHONHASHSEED=0`` — a hash-order artifact; the canonical order
    gives 451,128 with 3 overrides in every process).  The descent now
    scans in canonical member order
    (:func:`mlopt_core.egraph.types._member_key`), and the reproducible
    reading is 451,128 with three overrides in every process.  The
    *qualitative* finding is unchanged — the greedy pass is off the cost
    its own model assigns, and a monotone coordination recovers it —
    only the size of the recovery, and the claim that it unified a read,
    were overstated.
"""
# ruff: noqa: RUF002

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlopt_core.cost import dag_cost
from mlopt_core.egraph.types import _member_key
from mlopt_core.ir import Op

__all__ = ["SharingResult", "extract_sharing_aware"]


@dataclass
class SharingResult:
    """The outcome of a sharing-aware extraction pass."""

    term: Any
    cost: float
    base_cost: float
    #: ``{class id: member op}`` for every override kept.
    overrides: dict[int, str] = field(default_factory=dict)
    #: ``(class id, member op, cost delta)`` per accepted step.
    trace: list[tuple[int, str, float]] = field(default_factory=list)
    n_trials: int = 0

    @property
    def delta(self) -> float:
        """FLOPs recovered over the greedy extraction."""
        return self.cost - self.base_cost


def _term_classes(eg: Any, term: Any) -> list[int]:
    """Return the e-class ids of ``term``'s distinct subterms (term order)."""
    out: list[int] = []
    seen: set[int] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if not isinstance(t, Op):
            continue
        eid, _en = eg._locate(t)
        if eid is None:
            continue
        eid = eg.find(eid)
        if eid in seen:
            continue
        seen.add(eid)
        out.append(eid)
        stack.extend(t.args)
    return out


def extract_sharing_aware(
    eg: Any,
    root_eid: int,
    cost_fn: Any,
    *,
    rounds: int = 4,
    max_classes: int = 400,
) -> SharingResult:
    """Extract the term minimising DAG cost under *cost_fn*, sharing-aware.

    A bounded coordinate-descent over the per-class member choice:
    begin at the greedy extraction, then repeatedly force a single
    e-class to one of its members and keep the override when the whole
    term's :func:`~mlopt_core.cost.dag_cost` strictly falls.  The
    result is never worse than ``eg.extract_best`` (the empty override
    map is always in the neighbourhood), and it is deterministic —
    classes are visited in term order, members in the e-class's
    CANONICAL member order (``_member_key``), and only strict
    improvements are kept.

    The member order matters: the descent keeps the *first* strictly
    improving member of a class, so a raw ``set`` order (``nodes`` is a
    ``set`` of :class:`~mlopt_core.egraph.ENode`, whose hash folds in
    the op name) would land on a different local optimum in every
    process — pre-fix it reached 432,696 (5 overrides) at
    ``PYTHONHASHSEED=0``; the canonical order gives 451,128 (3 overrides)
    in every process.
    """
    root_eid = eg.find(root_eid)
    best = eg.extract_best(root_eid, cost_fn)
    base_cost = dag_cost(best, cost_fn)
    cur, cur_cost = best, base_cost
    kept: dict[int, Any] = {}
    trace: list[tuple[int, str, float]] = []
    n_trials = 0
    for _ in range(rounds):
        improved = False
        for cls in _term_classes(eg, cur)[:max_classes]:
            if kept.get(cls) is not None:
                continue
            for node in sorted(
                eg.get_class(cls).nodes, key=_member_key
            ):
                if node.op == "leaf":
                    continue
                trial = dict(kept)
                trial[cls] = node
                n_trials += 1
                cand = eg.extract_best(
                    root_eid, cost_fn, overrides=trial
                )
                if cand is None:
                    continue
                c = dag_cost(cand, cost_fn)
                if c < cur_cost - 1e-9:
                    cur, cur_cost = cand, c
                    kept[cls] = node
                    trace.append((cls, node.op, c - base_cost))
                    improved = True
        if not improved:
            break
    return SharingResult(
        term=cur,
        cost=cur_cost,
        base_cost=base_cost,
        overrides={c: n.op for c, n in kept.items()},
        trace=trace,
        n_trials=n_trials,
    )
