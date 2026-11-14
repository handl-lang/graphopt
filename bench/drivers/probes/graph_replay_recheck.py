"""Post-8589162 recheck: ``CudaGraphRunner`` generic-path correctness.

Commit ``8589162`` fixed ``_GraphReplayModule.forward`` copying inputs
BACKWARDS: pre-fix, ``zip(xs, graph_ins)`` ran ``caller.copy_(graph_buf)``
— replay was frozen on capture-time inputs AND caller tensors were
silently overwritten.  Every prior generic-path (``IRModule``)
``CudaGraphRunner`` delivery was therefore silently wrong on fresh
inputs; carrier executors (``capture_cuda_graph``) were unaffected.
Timing claims survive by construction — the replay always launched the
same captured kernel sequence regardless of input freshness.

What did NOT survive: every ``graph_max_abs``/``graph_vs_compiled``
correctness check in the sweep/probe reports was vacuous — the compare
ran ``module(*xin)`` vs ``wrapped(*xin)`` on the SAME tensors used for
capture, and pre-fix ``wrapped`` mutated ``xin`` back to the capture
values before replaying them.

This driver re-runs the *delivery* half of ``gen_probe`` (no timing —
``_gen_board`` → ``_probe_play`` → ``board.deliverable`` →
``hybrid_sink.lower`` → ``sink.verify``) for the workloads whose
reports carry graph-arm claims, wraps the delivered module in
``CudaGraphRunner``, and compares replay vs eager-delivered vs the
source model on FRESH inputs (scaled/shifted and randn variants of the
capture feed — never the capture tensors).  It answers, per claim:
does delivered+graph produce correct outputs post-fix?

Claims covered (``docs/reports/``):

- ``t037_resweep_frontier.md`` — "graph arm beats torch.compile 7/8
  rows": ChunkedSSD, DeltaRuleBlock, GatedDeltaRule, DecoupledMLA at
  fp64 AND fp32.
- ``resweep_scan_route.md`` — the route-relevant five at both dtypes
  (adds GatedLinearAttention, MambaSelectiveScan; RealGatedLRU never
  produced a row — skipped).
- ``deliver_runner_arm.md`` — GatedDeltaRule fp32 CUDA (already in
  the T-037 fp32 set; the deliver-hook path uses the same generic
  capture).

Run::

    .venv/bin/python -u -m bench.drivers.probes.graph_replay_recheck \
        --dtype both --out /tmp/graph_recheck.jsonl
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

#: The route-relevant + T-037 graph-arm workloads at zoo(64, 32, 8).
DEFAULT_ONLY = (
    "ChunkedSSD",
    "DeltaRuleBlock",
    "GatedDeltaRule",
    "DecoupledMLA",
    "GatedLinearAttention",
    "MambaSelectiveScan",
)

DEFAULT_BUDGET = (
    900.0  # per (workload, dtype) — mint+profitable+lower+verify
)


class _OverBudget(Exception):
    """A single (workload, dtype) exceeded its wall-clock budget."""


@contextmanager
def _budget(seconds: float):
    """Abort the guarded block after ``seconds`` (0 or less = no limit)."""
    if seconds <= 0:
        yield
        return

    def _boom(_signum, _frame):
        raise _OverBudget(f"exceeded {seconds:g}s")

    prev = signal.signal(signal.SIGALRM, _boom)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _fresh_variants(xin: Any, k: int) -> list[Any]:
    """K fresh input tuples — same structure/shapes/dtypes as ``xin``.

    Two families per slot: a scaled+noised copy of the capture feed
    (stays in-distribution; non-negative feeds stay non-negative) and
    a pure ``randn_like`` draw.  Equality — not semantic validity — is
    what is under test: every arm sees the SAME fresh tensors.
    """

    def _map(t: torch.Tensor, variant: int) -> torch.Tensor:
        if variant % 2 == 0:
            scale = 0.5 + torch.rand((), device=t.device).item()
            out = t * scale + 0.05 * torch.randn_like(t)
            if bool((t >= 0).all()):
                out = out.clamp_min(0)
            return out
        return torch.randn_like(t) * (t.std() + 1e-3)

    def _build(obj: Any, variant: int) -> Any:
        if torch.is_tensor(obj):
            return _map(obj, variant)
        if isinstance(obj, tuple):
            return tuple(_build(x, variant) for x in obj)
        if isinstance(obj, list):
            return [_build(x, variant) for x in obj]
        return obj

    return [_build(xin, i) for i in range(k)]


def _clone_out(obj: Any) -> Any:
    """Clone tensors recursively — replay output is a static buffer."""
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone_out(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_out(x) for x in obj]
    return obj


def _delta(a: Any, b: Any) -> dict:
    """NaN-aware max |a-b| over tensors / (nested) tuples of them."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return {"max_abs": float("nan"), "nan_mismatch": -1}
        both_nan = torch.isnan(a) & torch.isnan(b)
        diff = (a - b).abs()
        diff = torch.where(both_nan, torch.zeros_like(diff), diff)
        finite = ~torch.isnan(diff)
        nan_mm = int((torch.isnan(a) != torch.isnan(b)).sum().item())
        return {
            "max_abs": float(diff[finite].max().item())
            if bool(finite.any())
            else 0.0,
            "nan_mismatch": nan_mm,
        }
    if (
        isinstance(a, (tuple, list))
        and isinstance(b, (tuple, list))
        and len(a) == len(b)
    ):
        parts = [_delta(x, y) for x, y in zip(a, b, strict=True)]
        vals = [
            p["max_abs"] for p in parts if p["max_abs"] == p["max_abs"]
        ]
        return {
            "max_abs": max(vals) if vals else float("nan"),
            "nan_mismatch": sum(p["nan_mismatch"] for p in parts),
        }
    return {"max_abs": float("nan"), "nan_mismatch": -1}


def _to_device(xin: Any, device: str) -> Any:
    """Mirror ``_gen_probe_impl``'s feed move (tuple elementwise)."""
    if isinstance(xin, (tuple, list)):
        return tuple(t.to(device) for t in xin)
    return xin.to(device)


def _deliver_module(case: Any) -> tuple[Any, Any, Any, dict]:
    """Replay ``_gen_probe_impl``'s delivery half; return the pieces.

    Returns ``(module, base_model, xin, meta)`` — the eager delivered
    module (pre-runner), the source model, the CUDA feed, and a meta
    dict (verified rep fields, case labels).  Raises on any decline.
    """
    from mlopt_core.ir import IR
    from mlopt_discovery import genkernel as gk
    from mlopt_discovery import play as play_mod
    from mlopt_discovery import shipped_primitives as sp
    from mlopt_discovery.meta_arena import _ops_of

    ex = case[5]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    board = play_mod._gen_board(case)
    best = play_mod._probe_play(board)
    meta: dict = {"claims": list(board.observe().claim_tags)}
    if best is None:
        raise RuntimeError("no extraction (best is None)")
    base_model = ex["model"].to(device)
    xin = _to_device(ex["input"], device)
    sink = gk.hybrid_sink(ex["gen_handlers"], compile_kernels=True)
    term = board.deliverable(best)
    prim_ops = sorted(_ops_of(term) & set(sp.SYNOP_SHIPPED))
    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    module = sink.lower(ir, params=ex["leaves"]).to(device)
    rep = sink.verify(
        base_model,
        module,
        xin if isinstance(xin, (tuple, list)) else (xin,),
    )
    meta.update(
        {
            "verified": bool(getattr(rep, "passed", False)),
            "max_abs": getattr(rep, "max_abs", None),
            "prim_ops": prim_ops,
        }
    )
    return module, base_model, xin, meta


def recheck_one(
    name: str,
    dtype: str,
    *,
    fresh: int,
    reps: int,
    shape: tuple[int, int, int] = (64, 32, 8),
) -> dict:
    """Deliver one zoo workload and re-verify its graph arm on fresh inputs."""
    from mlopt_cuda import CudaGraphRunner
    from mlopt_discovery import play as play_mod
    from mlopt_discovery import zoo as zoo_mod

    d, t, b = shape
    wl = next(w for w in zoo_mod.zoo(d, t, b) if w.name == name)
    case = play_mod._zoo_case(wl, dtype=dtype)
    module, base_model, xin, meta = _deliver_module(case)
    row: dict = {
        "case": f"zoo:{name}",
        "dtype": dtype,
        "shape": list(shape),
        "device": "cuda",
        **meta,
    }
    if not meta["verified"]:
        row["note"] = "delivered module failed verify — graph arm moot"
        return row
    xs = xin if isinstance(xin, (tuple, list)) else (xin,)
    if not (xs and torch.is_tensor(xs[0]) and xs[0].is_cuda):
        row["note"] = "inputs not CUDA tensors — capture declines"
        return row
    stats: dict = {}
    wrapped = CudaGraphRunner().apply(module, xin, stats)
    row["cuda_graph"] = bool(stats.get("cuda_graph"))
    row["capture_failure"] = stats.get("cuda_graph_failure")
    if not stats.get("cuda_graph"):
        row["note"] = "generic capture declined"
        return row

    # Fresh-input correctness: replay vs eager-delivered vs source.
    args = xin if isinstance(xin, (tuple, list)) else (xin,)
    checks = []
    with torch.no_grad():
        for i, fresh_in in enumerate(_fresh_variants(xin, fresh)):
            fargs = (
                fresh_in
                if isinstance(fresh_in, (tuple, list))
                else (fresh_in,)
            )
            out_eager = module(*fargs)
            out_graph = _clone_out(wrapped(*fargs))
            # the replay output is a static buffer — clone before the
            # next call overwrites it (module above already ran eager).
            out_base = base_model(*fargs)
            checks.append(
                {
                    "variant": i,
                    "graph_vs_delivered": _delta(out_graph, out_eager),
                    "delivered_vs_base": _delta(out_eager, out_base),
                }
            )
    row["fresh_checks"] = checks
    row["max_abs_graph_vs_delivered"] = max(
        c["graph_vs_delivered"]["max_abs"] for c in checks
    )
    row["max_nan_mismatch"] = max(
        c["graph_vs_delivered"]["nan_mismatch"] for c in checks
    )
    # Capture-input sanity: replay on the capture feed itself.
    with torch.no_grad():
        row["capture_input_delta"] = _delta(
            _clone_out(wrapped(*args)), module(*args)
        )
    # Light timing sanity (no torch.compile — timing claims survive
    # the bug by construction; this only confirms capture still pays).
    if reps > 0:
        t_mod = play_mod._time_module(module, xin, reps)
        t_wrap = play_mod._time_module(wrapped, xin, reps)
        row["deliver_us"] = round(t_mod, 1)
        row["graph_us"] = round(t_wrap, 1)
        row["graph_speedup"] = (
            round(t_mod / t_wrap, 3) if t_wrap else None
        )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        default=",".join(DEFAULT_ONLY),
        help="comma-separated zoo workload names",
    )
    ap.add_argument(
        "--dtype",
        choices=["fp32", "fp64", "both"],
        default="both",
    )
    ap.add_argument("--fresh", type=int, default=4)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    shape = (args.d, args.t, args.b)

    dtypes = ["fp32", "fp64"] if args.dtype == "both" else [args.dtype]
    names = [n for n in args.only.split(",") if n]
    out_path = args.out or Path(
        f"/tmp/graph_recheck_{args.dtype}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with out_path.open("a") as fh:
        for dtype in dtypes:
            for name in names:
                tag = f"{name}/{dtype}"
                try:
                    with _budget(args.budget):
                        row = recheck_one(
                            name,
                            dtype,
                            fresh=args.fresh,
                            reps=args.reps,
                            shape=shape,
                        )
                except _OverBudget:
                    row = {
                        "case": f"zoo:{name}",
                        "dtype": dtype,
                        "delivered": False,
                        "note": f"over budget: no result within "
                        f"{args.budget:g}s",
                    }
                except Exception as exc:  # honest per-row record
                    row = {
                        "case": f"zoo:{name}",
                        "dtype": dtype,
                        "delivered": False,
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                row["delivered"] = "note" not in row or bool(
                    row.get("cuda_graph", row.get("verified"))
                )
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
                n_done += 1
                print(
                    f"[{n_done}] {tag}: {row.get('note') or 'ok'}",
                    flush=True,
                )
                # free the GPU between deliveries — 4 GB shared box
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(f"wrote {n_done} rows -> {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
