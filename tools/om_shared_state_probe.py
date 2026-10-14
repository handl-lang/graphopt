"""Plan 0023 — the shared-state reassociation, measured.

The K-way law hands the fold laws the full block decomposition, yet
extraction still keeps the *spelled* cross form: the cross sum is a
flat, left-nested add, so the affine-scan lifts fold a tree and the
O(T*C) sequential carrier is never reached.  ``om_shared_state_laws()``
adds the geometric reassociation

    W1*X + W2*Y = W2 * (gamma^(O1-O2)*X + Y)

which, applied bottom-up, produces the nested shared recurrence
``T_k = gamma^C*T_{k-1} + b_{k-1}``.

This probe measures what the law buys: the extracted ``dag_cost`` and
the e-graph size with and without ``om_shared_state_laws()``, for the
same seeded export.  Both arms run in ONE process so the comparison
sees a single hash seed.

Usage::

    .venv/bin/python tools/om_shared_state_probe.py --t 512 --k 8
"""

from __future__ import annotations

import argparse
import time

import torch
from catopt_carriers.om import (
    OM_CHUNKIFY,
    OM_LAWS,
    om_kway_laws,
    om_shared_state_laws,
)
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.laws.tensor import (
    ASSOC_MATMUL_REV,
    DIV_ADD_FACTOR,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    MATMUL_DIV_DISTRIBUTE,
    MUL_DIV_SWAP,
    SPLIT_SCALAR_RULES,
)
from catopt_discovery.zoo import _ChunkedRetention
from catopt_torch.adapters import TorchSource

#: The naturality bridges the fold's last hops need (plan 0023).
_BRIDGES = [
    MUL_DIV_SWAP,
    MATMUL_DIV_DISTRIBUTE,
    ASSOC_MATMUL_REV,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    DIV_ADD_FACTOR,
]


def _run(t: int, k: int, iters: int, budget: int, law: bool) -> tuple:
    """Saturate one arm and return ``(cost, n_enodes, secs)``."""
    torch.manual_seed(0)
    rules = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + _BRIDGES
        + om_kway_laws(k)
    )
    budgets = {
        "om_decayed_chunkify_rows": 60,
        "om_decayed_chunkify_keys": 60,
        f"om_decayed_kway_blocks_{k}": budget,
    }
    if law:
        shared = om_shared_state_laws()
        rules = rules + shared
        budgets.update({r.name: 400 for r in shared})
    model = _ChunkedRetention(16).double()
    x = torch.randn(2, t, 16, dtype=torch.float64)
    term = TorchSource().to_ir(model, x)[0].root
    eg = EGraph()
    root = eg.add_term(term)
    t0 = time.perf_counter()
    eg.run(
        rules,
        root,
        max_iterations=iters,
        max_nodes=150_000,
        patience=8,
        rule_budgets=budgets,
    )
    secs = time.perf_counter() - t0
    best = eg.extract_best(eg.find(root), flops_cost)
    return dag_cost(best, flops_cost), eg.n_enodes, secs


def main(argv: list[str] | None = None) -> int:
    """Run both arms and print the measured comparison."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t", type=int, default=512)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--budget", type=int, default=200)
    args = ap.parse_args(argv)
    print(
        f"T={args.t}  K={args.k}  C={args.t // args.k}  "
        f"iters={args.iters}  budget={args.budget}"
    )
    print(
        f"{'arm':>8} {'cost':>14} {'n_enodes':>10} {'secs':>8}",
        flush=True,
    )
    base = _run(args.t, args.k, args.iters, args.budget, law=False)
    law = _run(args.t, args.k, args.iters, args.budget, law=True)
    for name, (cost, nodes, secs) in (
        ("base", base),
        ("law", law),
    ):
        print(
            f"{name:>8} {cost:>14,.0f} {nodes:>10} {secs:>8.1f}",
            flush=True,
        )
    delta = (base[0] - law[0]) / base[0]
    print(f"cost drop: {delta:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
