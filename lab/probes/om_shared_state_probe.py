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

(both lists come from :func:`mlopt_carriers.om.om_shared_state_stages`).
Staging is successive :class:`~mlopt_core.egraph.EGraph` ``run`` calls
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

(and on the staged arm: ``reads`` = K−1 = 7 ``q_i @ S_{i-1}`` matmuls,
``chain`` = True — the folded carrier IS what extraction picks; see
the residual note below.)

``om_shared_mat_reassoc`` fires **112×** in the staged arm against
**0×** in ``core`` — the bridge is what revives it — and the extracted
cost lands 9.2 % below the current best.  At T=128/K=8 the drop is
25 % (1,279,448 → 954,772); at T=32/K=4 the staged arm equals ``core``
(the graph is too small for the fold to pay).

Co-residency is confirmed explosive: at T=128/K=8 running the two div
laws together (``matmul_div_distribute`` + ``div_into_matmul`` in one
stage) did **not** finish inside 800 s, against ~30 s for stage A alone.

Residual — RESOLVED, the earlier diagnosis was a metric artifact.
The ``extracted_is_fold`` column counts ``applyd``/``aff_diag`` ops in
the extracted term — but ``applyd`` is a COST-NEUTRAL respelling of
``add(mul(a,h),x)``, so extraction never prefers that spelling even
when it has selected exactly that structure.  Measured directly on the
extracted DAG, the staged arm DOES carry the shared recurrence: at
T=512/K=8 the winner has exactly K diagonal score matmuls feeding the
K intra ``om_decayed`` blocks, K−1 state contributions
``(k_j ⊙ η)ᵀ v_j`` each computed once, and K−1 fold reads
``q_i @ S_{i-1}`` where block ``i``'s state subterm literally nests
block ``i−1``'s (``S_i = γ^C·S_{i-1} + B_i`` — one shared chain).  The
``reads``/``chain`` columns below report that signature.

The nominal gap to the hand-built ``_shared_state_fold`` reference
(1,405,952 at T=512/K=8) is a like-for-like mismatch, not a search
residual: the reference runs on leaf-level unbatched ``q,k,v,s``, so
it prices neither the projections (1,572,864) nor the in-graph
diagonal score construction (2,097,152), and its intra carrier is
batch-1 (the export's B=2 doubles it to 2,162,688).  A hand-built
fold carrying the same work prices 7,082,496 — the extract lands
~2% BELOW the honest reference.

Usage::

    .venv/bin/python lab/probes/om_shared_state_probe.py --t 128 --k 8
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import torch
from mlopt_carriers.om import (
    om_shared_state_budgets,
    om_shared_state_laws,
    om_shared_state_stages,
)
from mlopt_core.cost import dag_cost, flops_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op
from mlopt_discovery.zoo import _ChunkedRetention
from mlopt_torch.adapters import TorchSource


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
    """Return True when ``term``'s DAG carries an ``applyd``/``aff_diag``.

    NB: extraction never prefers the carrier spelling (it prices
    exactly like the ``add(mul(a,h),x)`` it re-spells), so this column
    stays False even when the shared-state structure is selected —
    see :func:`_fold_signature` for the real signature.
    """
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


_Q_LEAVES = {"p_q_weight", "p_q_bias"}
_KV_LEAVES = {"p_k_weight", "p_k_bias", "p_v_weight", "p_v_bias"}


def _fold_signature(term) -> tuple[int, bool]:
    """Return the structural fold signature of an extracted term.

    ``reads`` counts ``matmul(q_i, S)`` members — left subtree free of
    the k/v leaves, right subtree free of q and carrying BOTH the k and
    v paths (the q-independent state).  ``chained`` reports whether one
    read's state subterm nests another's — the ``S_i = γ^C·S_{i-1} +
    B_i`` recurrence shared across row blocks rather than a per-pair
    respelling.
    """

    def leaves(t, acc=None):
        acc = set() if acc is None else acc
        if not isinstance(t, Op):
            acc.add(repr(t))
            return acc
        for a in t.args:
            leaves(a, acc)
        return acc

    reads: list[Op] = []
    seen: set[int] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if not isinstance(t, Op) or id(t) in seen:
            continue
        seen.add(id(t))
        if t.op == "matmul" and len(t.args) == 2:
            left, right = t.args
            if isinstance(left, Op) and isinstance(right, Op):
                ll, rl = leaves(left), leaves(right)
                if (
                    not (ll & _KV_LEAVES)
                    and {"p_k_weight", "p_v_weight"} <= rl
                    and not (rl & _Q_LEAVES)
                ):
                    reads.append(t)
        stack.extend(t.args)
    states = [r.args[1] for r in reads]

    def subtree_ids(t):
        ids: set[int] = set()
        st = [t]
        while st:
            x = st.pop()
            if not isinstance(x, Op) or id(x) in ids:
                continue
            ids.add(id(x))
            st.extend(x.args)
        return ids

    idsets = [subtree_ids(s) for s in states]
    chained = any(
        id(sj) in si_ids
        for i, si_ids in enumerate(idsets)
        for j, sj in enumerate(states)
        if i != j
    )
    return len(reads), chained


def _honest_reference_cost(t: int, k: int) -> float:
    """Price the hand-built fold carrying the export's real work.

    ``_shared_state_fold``-shaped program but like-for-like with the
    ``_ChunkedRetention`` export: batch-2 leaves, the q/k/v projections
    spelled as ``linear`` ops, and each intra block's score built
    in-graph as ``div(matmul(q_i, k_iᵀ), √t)`` — the work the leaf-level
    reference gets for free.  This is the floor a staged extraction
    should be compared against, not 1.4M.
    """
    from mlopt_core.ir import TensorType, Var

    def v_(name, *shape):
        return Var(name, TensorType(shape))

    b, d = 2, 16
    c_dim = t // k
    sz = (c_dim,) * k
    x = v_("x", b, t, d)
    wq, bq = v_("wq", d, d), v_("bq", d)
    wk, bk = v_("wk", d, d), v_("bk", d)
    wv, bv = v_("wv", d, d), v_("bv", d)
    g, a = v_("g"), v_("a")
    c = float(t) ** 0.5
    qfull = Op.make("linear", x, wq, bq)
    kfull = Op.make("linear", x, wk, bk)
    vfull = Op.make("linear", x, wv, bv)
    kt = Op.make("transpose", kfull, dim0=-2, dim1=-1)

    def blk(term, i, dim=-2):
        return Op.make("split", term, sizes=sz, dim=dim, index=i)

    eta = Op.make("decay_vec", g, t=c_dim, sign=-1)
    xi0 = Op.make("decay_vec", g, t=c_dim)
    state = None
    blocks = []
    for i in range(k):
        qi, ki, vi = blk(qfull, i), blk(kfull, i), blk(vfull, i)
        score = Op.make("div", Op.make("matmul", qi, blk(kt, i, -1)), c)
        intra = Op.make(
            "om_decayed", score, Op.make("decay", g, t=c_dim), vi
        )
        bj = Op.make(
            "matmul",
            Op.make(
                "transpose",
                Op.make("mul", ki, Op.make("unsqueeze", eta, dim=-1)),
                dim0=-2,
                dim1=-1,
            ),
            vi,
        )
        if state is None:
            state = bj
            blocks.append(intra)
            continue
        blocks.append(
            Op.make(
                "add",
                intra,
                Op.make(
                    "mul",
                    Op.make("unsqueeze", xi0, dim=-1),
                    Op.make("matmul", qi, state),
                ),
            )
        )
        if i < k - 1:
            state = Op.make("add", Op.make("mul", a, state), bj)
    out = blocks[-1]
    for block in reversed(blocks[:-1]):
        out = Op.make("concat", block, out, dim=-2)
    return dag_cost(out, flops_cost)


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
    reads, chained = _fold_signature(best)
    return {
        "cost": dag_cost(best, flops_cost),
        "nodes": eg.n_enodes,
        "secs": secs,
        "fires": fires,
        "n_shared": len(shared),
        "shared_ops": dict(ops.most_common(6)),
        "extracted_is_fold": _has_fold(best),
        "reads": reads,
        "chained": chained,
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
    ap.add_argument(
        "--ref",
        action="store_true",
        help="also print the like-for-like fold floor",
    )
    args = ap.parse_args(argv)
    budgets = om_shared_state_budgets(args.k, kway=args.budget)
    print(
        f"T={args.t}  K={args.k}  C={args.t // args.k}  "
        f"iters={args.iters}  kway_budget={args.budget}",
        flush=True,
    )
    if args.ref:
        print(
            f"{'ref':>8} {_honest_reference_cost(args.t, args.k):>13,.0f}"
            "   (hand fold + projections + in-graph scores, B=2)",
            flush=True,
        )
    print(
        f"{'arm':>8} {'extract':>13} {'n_enodes':>9} {'shared':>7} "
        f"{'fold':>5} {'reads':>6} {'chain':>6} {'secs':>7}",
        flush=True,
    )
    for arm in args.arms.split(","):
        r = _run_arm(args.t, args.k, args.iters, arm, budgets)
        print(
            f"{arm:>8} {r['cost']:>13,.0f} {r['nodes']:>9} "
            f"{r['n_shared']:>7} {r['extracted_is_fold']!s:>5} "
            f"{r['reads']:>6} {r['chained']!s:>6} "
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
