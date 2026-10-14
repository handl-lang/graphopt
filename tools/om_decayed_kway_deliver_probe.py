"""Plan 0023 construct 3 — the K-way chunk law, delivered and timed.

The symbolic result: at T=512 the K-way law cuts the extracted
carrier's cost 4.4x (29.4M -> 6.70M FLOPs) and extraction *does*
pick the chunked carrier (``concat``).  What nobody had done is
lower that extracted program to a runnable module and time it
against the source.  This probe closes the loop:

1. Saturate the real ``_ChunkedRetention`` export with the full
   reachability law set plus ``om_kway_laws(k)``.
2. Extract the winner and report its root op, extracted ``dag_cost``
   and the fused ``om_decayed`` member's cost — the honest FLOP
   ratio, not a claimed one.
3. Lower the extracted term through the **ambient** ``TorchSink``
   (``om_decayed``/``decay``/``decay_vec``/``concat`` all carry torch
   bindings, so no minted handler is needed) and verify it against
   the source module in fp64 (``max_abs``).
4. Time it — eager baseline, ``torch.compile`` of the source, and the
   lowered carrier — under the exact ``gen_probe`` harness
   (``catopt_discovery.play._time_module``: 10 warm-ups, median of
   *reps*).

The measured row answers the product question: does the 4.4x FLOP
win survive as *time*?  It does — but only past a crossover size.
The chunked carrier is many small ops (splits, concats, ``decay_vec``
materialisations, per-block matmuls); at T=512 the fused quadratic is
too small to hide that launch overhead and the carrier *loses*.  By
T~=1024 it crosses over and by T=2048 it wins ~5x.  Both regimes are
printed so the crossover is visible, not hidden.

Usage::

    .venv/bin/python tools/om_decayed_kway_deliver_probe.py
    .venv/bin/python tools/om_decayed_kway_deliver_probe.py \
        --t 2048 --k 4
"""

from __future__ import annotations

import argparse
import time

import torch
from catopt_carriers.om import OM_CHUNKIFY, OM_LAWS, om_kway_laws
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op
from catopt_core.laws.tensor import (
    ASSOC_MATMUL_REV,
    DIV_ADD_FACTOR,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    MATMUL_DIV_DISTRIBUTE,
    MUL_DIV_SWAP,
    SPLIT_SCALAR_RULES,
)
from catopt_discovery.play import _time_module
from catopt_discovery.zoo import _ChunkedRetention
from catopt_torch.adapters import TorchSink, TorchSource

#: The naturality bridges the fold's last hops need (plan 0023) —
#: load-bearing for the carrier preference.
_BRIDGES = [
    MUL_DIV_SWAP,
    MATMUL_DIV_DISTRIBUTE,
    ASSOC_MATMUL_REV,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    DIV_ADD_FACTOR,
]


def _ops_of(term: object) -> set:
    """Every op name reachable in *term* (args-first)."""
    acc: set = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            acc.add(t.op)
            stack.extend(t.args)
    return acc


def _bisect_cost(term: object, iters: int, chunk: int) -> float:
    """Return the extracted cost of the *bisection* chunked form.

    No K-way law: the reference the ``4.4x`` headline is measured
    against is the pairwise-bisection chunked carrier (plan 0023's
    earlier form).
    """
    rules = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + _BRIDGES
    )
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(
        rules,
        root,
        max_iterations=iters,
        max_nodes=150_000,
        patience=8,
        rule_budgets={
            "om_decayed_chunkify_rows": chunk,
            "om_decayed_chunkify_keys": chunk,
        },
    )
    return dag_cost(
        eg.extract_best(eg.find(root), flops_cost), flops_cost
    )


def _saturate(
    t: int, k: int, iters: int, budget: int, chunk: int
) -> dict:
    """Saturate the real export; return the extraction + e-graph facts."""
    torch.manual_seed(0)
    rules = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + _BRIDGES
        + om_kway_laws(k)
    )
    budgets = {
        "om_decayed_chunkify_rows": chunk,
        "om_decayed_chunkify_keys": chunk,
        f"om_decayed_kway_blocks_{k}": budget,
    }
    model = _ChunkedRetention(16).double()
    x = torch.randn(2, t, 16, dtype=torch.float64)
    ir, leaves = TorchSource().to_ir(model, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    t0 = time.perf_counter()
    eg.run(
        rules,
        root,
        max_iterations=iters,
        max_nodes=200_000,
        patience=8,
        rule_budgets=budgets,
    )
    secs = time.perf_counter() - t0
    rc = eg.find(root)
    best = eg.extract_best(rc, flops_cost)
    fused = [n for n in eg._classes[rc].nodes if n.op == "om_decayed"]
    fused_cost = (
        dag_cost(
            eg.extract_best(rc, flops_cost, overrides={rc: fused[0]}),
            flops_cost,
        )
        if fused
        else float("nan")
    )
    return {
        "model": model,
        "x": x,
        "ir": ir,
        "leaves": leaves,
        "best": best,
        "cost": dag_cost(best, flops_cost),
        "fused_cost": fused_cost,
        "bisect_cost": _bisect_cost(ir.root, iters, chunk),
        "root_op": best.op,
        "n_enodes": eg.n_enodes,
        "sat_secs": secs,
    }


def _measure(sat: dict, *, reps: int, device: str) -> dict:
    """Lower, verify and time the extracted carrier vs eager/compiled."""
    sink = TorchSink()
    best = sat["best"]
    missing = _ops_of(best) - sink.supported_ops
    if missing:
        return {"delivered": False, "missing_ops": sorted(missing)}
    ir = sat["ir"]
    ex_ir = IR(
        root=best,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )
    module = sink.lower(ex_ir, params=sat["leaves"]).to(device)
    base = sat["model"].to(device)
    xin = sat["x"].to(device)
    rep = sink.verify(base, module, (xin,))

    cfg = torch._dynamo.config
    cfg.recompile_limit = max(int(cfg.recompile_limit), 256)
    t_base = _time_module(base, xin, reps)
    compiled = torch.compile(base)
    t_comp = _time_module(compiled, xin, reps)
    t_car = _time_module(module, xin, reps)
    return {
        "delivered": True,
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "baseline_us": t_base,
        "compiled_us": t_comp,
        "carrier_us": t_car,
        "speedup": round(t_base / t_car, 3),
        "vs_compiled": round(t_comp / t_car, 3),
        "device": device,
    }


def _row(t: int, k: int, sat: dict, m: dict) -> str:
    """Format one measured row."""
    fused = (
        sat["fused_cost"] / sat["cost"] if sat["cost"] else float("nan")
    )
    bisect = (
        sat["bisect_cost"] / sat["cost"]
        if sat["cost"]
        else float("nan")
    )
    if not m.get("delivered"):
        return (
            f"{t:>5} {k:>2} {sat['root_op']:>7} "
            f"{sat['cost']:>12,.0f} {fused:>5.2f} {bisect:>5.2f}  "
            f"LOWER FAILED {m.get('missing_ops')}"
        )
    return (
        f"{t:>5} {k:>2} {sat['root_op']:>7} "
        f"{sat['cost']:>12,.0f} {fused:>5.2f} {bisect:>5.2f} "
        f"{m['verified']!s:>5} {m['max_abs']:>9.2e} "
        f"{m['baseline_us']:>9.1f} {m['compiled_us']:>9.1f} "
        f"{m['carrier_us']:>9.1f} {m['speedup']:>7.3f} "
        f"{m['vs_compiled']:>7.3f}"
    )


def _header() -> str:
    """Format the report's column header."""
    return (
        f"{'T':>5} {'k':>2} {'root':>7} {'cost':>12} {'x_fused':>7} "
        f"{'x_bis':>5} {'ver':>5} {'max_abs':>9} {'base_us':>9} "
        f"{'comp_us':>9} {'car_us':>9} {'speedup':>7} {'vs_comp':>7}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the delivery + timing sweep and print the measured rows."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--t", type=int, default=0, help="0 = sweep default"
    )
    ap.add_argument(
        "--k", type=int, default=0, help="0 = sweep default"
    )
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=60)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("no CUDA device — the timing arm needs CUDA")
        return 1
    device = "cuda"
    pairs = (
        [(args.t, args.k)]
        if args.t and args.k
        else [(t, k) for t in (512, 1024, 2048) for k in (2, 4, 8)]
    )
    print(_header(), flush=True)
    for t, k in pairs:
        if t % k:
            continue
        sat = _saturate(t, k, args.iters, args.budget, args.chunk)
        m = _measure(sat, reps=args.reps, device=device)
        print(_row(t, k, sat, m), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
