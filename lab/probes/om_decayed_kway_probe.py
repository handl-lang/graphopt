"""Plan 0023 construct 3 — the K-way chunk law, measured.

The decayed-attention fold reaches depth 1 on the real ChunkedRetention
export: ``OM_DECAYED_CHUNKIFY_ROWS``/``_KEYS`` are *pairwise* bisections,
so a row block only ever sees one prior block per split, and the dirty-
frontier rescan makes each deeper level costlier.  This probe answers
two questions with numbers:

1. **Does the bisection's cost-vs-depth curve pay?**  Sweep the chunk
   budget (and iteration cap) and record ``n_enodes`` and the extracted
   ``dag_cost``.  If depth saturates and the cost flattens, the K-way
   law buys nothing.

2. **What does the K-way law buy?**  ``om_decayed_kway_blocks(k)``
   reaches ``K`` chunks in ONE application, handing the fold laws all
   ``K-1`` prior blocks at once.  Sweep ``k`` and the law's enode budget
   and record the extracted cost, the e-graph's max fold depth
   (``applyd``/``affd_compose`` nesting), and whether extraction still
   selects the chunked carrier.

Reported for T=512 (``T/C-1 = 31`` for C=16).  All runs are seeded and
time-bounded; the host is 11 GB, so the enode budget is the knob.

Usage::

    .venv/bin/python lab/probes/om_decayed_kway_probe.py --t 512
"""

from __future__ import annotations

import argparse
import time

import torch
from mlopt_carriers.om import OM_CHUNKIFY, OM_LAWS, om_kway_laws
from mlopt_core.cost import dag_cost, flops_cost
from mlopt_core.egraph import EGraph
from mlopt_core.laws.tensor import (
    ASSOC_MATMUL_REV,
    DIV_ADD_FACTOR,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    MATMUL_DIV_DISTRIBUTE,
    MUL_DIV_SWAP,
    SPLIT_SCALAR_RULES,
)
from mlopt_discovery.zoo import _ChunkedRetention
from mlopt_torch.adapters import TorchSource

#: The naturality bridges the fold's last hops need (plan 0023: the
#: "fused still wins" reading used the limited set; these are
#: load-bearing for the carrier preference).
_BRIDGES = [
    MUL_DIV_SWAP,
    MATMUL_DIV_DISTRIBUTE,
    ASSOC_MATMUL_REV,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    DIV_ADD_FACTOR,
]

#: The carrier-fold ops whose nesting depth the plan's acceptance
#: names (``applyd`` at depth >= T/C-1, or ``affd_compose`` of that
#: arity).
_FOLD_OPS = ("applyd", "affd_compose")


def _max_fold_depth(eg: EGraph, root: int) -> int:
    """Max nesting of ``applyd``/``affd_compose``, cycle-safe DFS."""
    memo: dict[int, int] = {}

    def depth(eid: int, stack: set) -> int:
        eid = eg.find(eid)
        if eid in memo:
            return memo[eid]
        if eid in stack:
            return 0
        stack.add(eid)
        best = 0
        for node in eg._classes[eid].nodes:
            child = max(
                (depth(c, stack) for c in node.children), default=0
            )
            best = max(best, child + (node.op in _FOLD_OPS))
        stack.discard(eid)
        memo[eid] = best
        return best

    return depth(root, set())


def _max_split_len(eg: EGraph) -> int:
    """Longest ``split`` sizes tuple — the finest chunk count seen."""
    best = 0
    for cls in eg._classes.values():
        for node in cls.nodes:
            if node.op != "split":
                continue
            sizes = dict(node.attrs).get("sizes")
            if isinstance(sizes, (tuple, list)):
                best = max(best, len(sizes))
    return best


def _run(
    t: int,
    x: torch.Tensor,
    k: int | None,
    budget: int,
    iters: int,
    chunk_budget: int,
) -> dict:
    """Saturate one configuration and return its measured row."""
    rules = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + _BRIDGES
    )
    budgets = {
        "om_decayed_chunkify_rows": chunk_budget,
        "om_decayed_chunkify_keys": chunk_budget,
    }
    if k is not None:
        rules = rules + om_kway_laws(k)
        budgets[f"om_decayed_kway_blocks_{k}"] = budget
    model = _ChunkedRetention(16).double()
    term = TorchSource().to_ir(model, x)[0].root
    eg = EGraph()
    root = eg.add_term(term)
    t0 = time.perf_counter()
    eg.run(
        rules,
        root,
        max_iterations=iters,
        max_nodes=400_000,
        patience=8,
        rule_budgets=budgets,
    )
    secs = time.perf_counter() - t0
    rc = eg.find(root)
    best = eg.extract_best(rc, flops_cost)
    return {
        "k": k,
        "budget": budget if k is not None else chunk_budget,
        "n_enodes": eg.n_enodes,
        "classes": len(eg._classes),
        "cost": dag_cost(best, flops_cost),
        "root_op": best.op,
        "fold_depth": _max_fold_depth(eg, rc),
        "max_split": _max_split_len(eg),
        "secs": secs,
    }


def _row(row: dict) -> str:
    """Format one measured row for the report."""
    k = "-" if row["k"] is None else str(row["k"])
    return (
        f"{k:>3} {row['budget']:>7} {row['n_enodes']:>9} "
        f"{row['classes']:>8} {row['cost']:>12,.0f} "
        f"{row['root_op']:>9} {row['fold_depth']:>6} "
        f"{row['max_split']:>6} {row['secs']:>7.1f}"
    )


def _header() -> str:
    """Format the report's column header."""
    return (
        f"{'k':>3} {'budget':>7} {'n_enodes':>9} {'classes':>8} "
        f"{'cost':>12} {'root':>9} {'depth':>6} {'split':>6} "
        f"{'secs':>7}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the cost-vs-depth and K-way sweeps and print the table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-budget", type=int, default=60)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, args.t, 16, dtype=torch.float64)
    target = args.t // 16 - 1
    print(f"T={args.t}  C=16  fold-depth target {target}")
    print(_header(), flush=True)

    print("-- bisection: cost vs depth --", flush=True)
    for budget, iters in ((40, 12), (60, 16), (90, 18), (120, 22)):
        print(
            _row(_run(args.t, x, None, 0, iters, budget)),
            flush=True,
        )

    print("-- K-way: one application, K chunks --", flush=True)
    for k in (2, 4, 8):
        for budget, iters in ((100, 16), (200, 18)):
            try:
                row = _run(
                    args.t, x, k, budget, iters, args.chunk_budget
                )
            except Exception as exc:
                print(
                    f"{k:>3} {budget:>7}  ERROR "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            print(_row(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
