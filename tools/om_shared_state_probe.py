# ruff: noqa: RUF002
"""Plan 0023 — the shared-state fold, *scheduled* and measured.

The reassociation family (``om_shared_state_laws()``) is the move that
turns the flat geometric cross sum ``Σ_{j<i} γ^{(i-j)C} ⊙ b_j`` into
the nested recurrence ``T_k = γ^C ⊙ T_{k-1} + b_{k-1}`` — whose inner
state is ONE e-class consumed by two row blocks, i.e. the O(T·C)
sequential carrier.  On the real chunked export it fires only at the
*scalar* sites, because the export's score is ``q(x)@k(x)ᵀ/√t``: the
scalar denominator sits OUTSIDE every cross contraction
(``div(matmul(q_i, B_j), c)``), so ``om_shared_mat_reassoc`` never sees
``mul(W, matmul(Q, B))`` and fires 0×.

The bridge that lifts the denominator inside,
``div(matmul(A,B), c) = matmul(div(A,c), B)``, is the *exact inverse*
of the shipped ``matmul_div_distribute`` — co-resident the two are a
two-cycle and the closure explodes (measured below).  This probe runs
the two families in SCHEDULED STAGES instead:

    stage A  reachability  — OM_LAWS + OM_CHUNKIFY + SPLIT_SCALAR_RULES
                             + the six naturality bridges + om_kway_laws(k)
    stage B  reassociation — stage A with ``matmul_div_distribute``
                             REPLACED by its inverse ``div_into_matmul``,
                             plus ``om_shared_state_laws()``

(both lists come from :func:`catopt_carriers.om.om_shared_state_stages`).
Staging is successive :class:`~catopt_core.egraph.EGraph` ``run`` calls
on the same graph, so the split costs nothing but the second pass.

Arms measured, all on the same seeded export in ONE process (a single
hash seed):

``reach``   stage A alone — the chunked carrier without the fold.
``core``    stage A + the reassociation family all at once (the current
            best; the div bridge is absent, so the mat variant is dead).
``staged``  stage A then stage B — the scheduled pair.

Per arm the probe reports the extracted ``dag_cost``, the e-graph size,
``shared`` — the number of carrier-state e-classes (``applyd`` /
``aff_diag`` members) reachable from the root with **≥2 distinct
parents**, the acceptance signature of the shared recurrence — whether
the extracted term itself carries a fold, and the reassociation fires.

Measured (seeded, ``iters=16``, kway budget 200, single process):

    T=512 K=8 C=64    extract      n_enodes   shared   secs
    reach            8,572,928       15,513       57    34
    core             7,643,656       25,855      196    57
    staged           6,943,236      124,005    5,820   429

``om_shared_mat_reassoc`` fires **112×** in the staged arm against
**0×** in ``core`` — the bridge is what revives it — and the extracted
cost lands 9.2 % below the current best.  At T=128/K=8 the drop is
25 % (1,279,448 → 954,772); at T=32/K=4 the staged arm equals ``core``
(the graph is too small for the fold to pay).

Co-residency is confirmed explosive: at T=128/K=8 running the two div
laws together (``matmul_div_distribute`` + ``div_into_matmul`` in one
stage) did **not** finish inside 800 s, against ~30 s for stage A alone.

Residual gap (honest): the extracted term still carries no fold, and
the reassociated state is not *i-independent*.  The export's cross
terms are ``mul(W, matmul(score_ij ⊙ η_j, v_j))`` — the score is
row-block dependent — so folding reassociates *q-bearing* terms and
each row block gets its own state.  The q-free factorization
(``matmul_scalar_factor`` + ``factor_matmul``, both in the set) is the
missing link between this and the hand-built reference
(``_shared_state_fold``: 1,405,952 at T=512/K=8, 589,952 at K=32).

Usage::

    .venv/bin/python tools/om_shared_state_probe.py --t 128 --k 8
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import torch
from catopt_carriers.om import (
    om_shared_state_budgets,
    om_shared_state_laws,
    om_shared_state_stages,
)
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import Op
from catopt_discovery.zoo import _ChunkedRetention
from catopt_torch.adapters import TorchSource


def _export(t: int):
    """Build the seeded ChunkedRetention export's IR root."""
    torch.manual_seed(0)
    model = _ChunkedRetention(16).double()
    x = torch.randn(2, t, 16, dtype=torch.float64)
    return TorchSource().to_ir(model, x)[0].root


def _saturate(eg, root, rules, iters, budgets):
    """Run one saturation stage on the shared e-graph."""
    eg.run(
        rules,
        root,
        max_iterations=iters,
        max_nodes=150_000,
        patience=8,
        rule_budgets=budgets,
    )


def _parent_count(eg, cid: int) -> int:
    """Count the distinct parent e-classes of ``cid``."""
    return len({eg.find(p) for p in eg._parents.get(cid, ())})


def _shared_state_census(eg, root):
    """Report the shared states reachable from the root class.

    A *shared state* is a class whose members include a carrier state
    (``applyd``/``aff_diag``) and which is consumed by ≥2 distinct
    parent classes — block ``i``'s output and block ``i+1``'s step, the
    acceptance signature of the O(T·C) recurrence.
    """
    root_cls = eg.find(root)
    # Reachable classes from the root (cycle-safe).
    seen: set[int] = set()
    stack = [root_cls]
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        for n in eg.get_class(cid).nodes:
            stack.extend(eg.find(c) for c in n.children)
    shared = [
        cid
        for cid in seen
        if _parent_count(eg, cid) >= 2
        and any(
            n.op in ("applyd", "aff_diag")
            for n in eg.get_class(cid).nodes
        )
    ]
    ops = Counter()
    for cid in shared:
        for n in eg.get_class(cid).nodes:
            ops[n.op] += 1
    return shared, ops


def _has_fold(term) -> bool:
    """Return True when ``term``'s DAG carries an ``applyd``/``aff_diag``."""
    if not isinstance(term, Op):
        return False
    seen: set[int] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if not isinstance(t, Op) or id(t) in seen:
            continue
        seen.add(id(t))
        if t.op in ("applyd", "aff_diag"):
            return True
        stack.extend(t.args)
    return False


def _run_arm(t, k, iters, mode, budgets):
    """Saturate one arm and return the measured row."""
    reach, reassoc = om_shared_state_stages(k)
    eg = EGraph()
    root = eg.add_term(_export(t))
    t0 = time.perf_counter()
    if mode == "reach":
        _saturate(eg, root, reach, iters, budgets)
    elif mode == "core":
        _saturate(
            eg, root, reach + om_shared_state_laws(), iters, budgets
        )
    else:  # staged
        _saturate(eg, root, reach, iters, budgets)
        _saturate(eg, root, reassoc, iters, budgets)
    secs = time.perf_counter() - t0
    best = eg.extract_best(eg.find(root), flops_cost)
    shared, ops = _shared_state_census(eg, root)
    fires = {
        a: b
        for a, b in eg.rule_fires.items()
        if "shared" in a or "div_into" in a
    }
    return {
        "cost": dag_cost(best, flops_cost),
        "nodes": eg.n_enodes,
        "secs": secs,
        "fires": fires,
        "n_shared": len(shared),
        "shared_ops": dict(ops.most_common(6)),
        "extracted_is_fold": _has_fold(best),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the arms and print the measured comparison."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument(
        "--arms",
        default="reach,core,staged",
        help="comma-separated subset of reach,core,staged",
    )
    args = ap.parse_args(argv)
    budgets = om_shared_state_budgets(args.k, kway=args.budget)
    print(
        f"T={args.t}  K={args.k}  C={args.t // args.k}  "
        f"iters={args.iters}  kway_budget={args.budget}",
        flush=True,
    )
    print(
        f"{'arm':>8} {'extract':>13} {'n_enodes':>9} {'shared':>7} "
        f"{'fold':>5} {'secs':>7}",
        flush=True,
    )
    for arm in args.arms.split(","):
        r = _run_arm(args.t, args.k, args.iters, arm, budgets)
        print(
            f"{arm:>8} {r['cost']:>13,.0f} {r['nodes']:>9} "
            f"{r['n_shared']:>7} {r['extracted_is_fold']!s:>5} "
            f"{r['secs']:>7.1f}",
            flush=True,
        )
        print(f"         fires: {r['fires']}", flush=True)
        print(
            f"         shared member ops: {r['shared_ops']}", flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
