"""Does the intake dtype change the sign of a delivered module?

`zoo_chain` sweeps the zoo in **fp64** — the intake convention, and the only
dtype where Triton cannot bind — so its ratios say nothing about the dtype a
user ships.  `RMSNormPerHead` is why that matters: worst row in the sweep at
**0.208x**, delivered body is a law fold to `aten::rms_norm`, whose fp64
kernel is pathological at head-width 16.  Same IR, fp32: a win.

This driver runs one workload through the **same arena and delivery path** as
`zoo_chain` (`_gen_board` → `_probe_play` → `board.deliverable` →
`hybrid_sink.lower`), with the dtype as the only variable, and prints the
verified ratio plus the delivered body's kernels for each dtype.  It is the
counterpart one level down from `zoo_kcount`: that one profiles *eager*, this
one profiles *what gets delivered*.

Run::

    .venv/bin/python -m bench.drivers.probes.zoo_dtype \\
        --only RMSNormPerHead --out docs/reports/zoo_dtype_rmsnorm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mlopt_core.ir import IR
from mlopt_discovery import genkernel as gk
from mlopt_discovery import lawdata
from mlopt_discovery import zoo as zoo_mod
from mlopt_discovery.meta_arena import _canon_concrete
from mlopt_discovery.play import (
    _gen_board,
    _mint_tiers,
    _probe_play,
    _time_module,
    _triton_kernel_for,
)
from mlopt_torch.adapters import TorchSource

_DTYPES = {"fp32": torch.float32, "fp64": torch.float64}


def _case(wl: Any, dtype: torch.dtype) -> Any:
    """Build the zoo case at ``dtype`` — `_zoo_case` with the dtype as a knob."""
    model, feed = wl.build()
    model = model.to(dtype).cuda()
    x = feed.to(dtype).cuda() if torch.is_tensor(feed) else feed
    xin = (
        tuple(t.to(dtype).cuda() for t in feed)
        if isinstance(feed, (tuple, list))
        else x
    )
    ir, leaves = TorchSource().to_ir(model, x)
    covered = {
        _canon_concrete(h["pattern"]) for h in lawdata.HANDLERS.values()
    }
    genh = _mint_tiers(ir.root, covered)
    env = dict(leaves)
    env.update(
        {
            v.name: t
            for v, t in zip(
                ir.inputs,
                (x,) if not isinstance(x, tuple) else x,
                strict=False,
            )
        }
    )
    genh = gk.profitable(
        genh, ir.root, env, reps=20, kernel_for=_triton_kernel_for
    )
    sink = gk.gen_sink(genh, compile_kernels=False)
    return (
        f"zoo:{wl.name}",
        ir.root,
        {"gen": len(genh), "minted": list(genh)},
        wl.name,
        sink.supported_ops,
        {
            "ir": ir,
            "leaves": leaves,
            "model": model,
            "input": feed,
            "sink": sink,
            "handlers": dict(lawdata.HANDLERS) | genh,
            "gen_handlers": genh,
            "xin": xin,
        },
    )


def _delivered(case: Any) -> Any:
    """Lower the arena's best extraction the way `gen_probe` does."""
    ex = case[5]
    board = _gen_board(case)
    best = _probe_play(board)
    if best is None:
        return None, None
    term = board.deliverable(best)
    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    sink = gk.hybrid_sink(ex["gen_handlers"], compile_kernels=True)
    return sink.lower(ir, params=ex["leaves"]).to("cuda"), sink


def _kernels(module: Any, xin: Any, reps: int) -> list[dict[str, Any]]:
    """The delivered body's device kernels — what actually costs the µs."""
    with torch.no_grad():
        for _ in range(5):
            module(xin)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(reps):
                module(xin)
            torch.cuda.synchronize()
    rows = [
        {
            "kernel": str(e.key)[:70],
            "count": e.count // reps,
            "device_us": round(e.self_device_time_total / reps, 2),
        }
        for e in prof.key_averages()
        if getattr(e, "self_device_time_total", 0) > 0
    ]
    return sorted(rows, key=lambda r: -r["device_us"])[:6]


def _one(wl: Any, name: str, reps: int) -> dict:
    """Measure one workload at ``name``'s dtype against its own eager arm."""
    case = _case(wl, _DTYPES[name])
    ex = case[5]
    xin, model = ex["xin"], ex["model"]
    base_us = _time_module(model, xin, reps)
    module, sink = _delivered(case)
    if module is None:
        return {
            "case": f"zoo:{wl.name}",
            "dtype": name,
            "delivered": False,
        }
    args = (xin,) if not isinstance(xin, (tuple, list)) else xin
    rep = sink.verify(model, module, args)
    gen_us = _time_module(module, xin, reps)
    return {
        "case": f"zoo:{wl.name}",
        "dtype": name,
        "delivered": True,
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "minted_kept": case[2]["minted"],
        "eager_us": round(base_us, 1),
        "gen_us": round(gen_us, 1),
        "speedup": round(base_us / gen_us, 3) if gen_us else None,
        "gen_kernels": _kernels(module, xin, max(reps, 50)),
    }


def measure(
    *, d: int, t: int, b: int, only: tuple[str, ...], reps: int
) -> list[dict]:
    """Run every requested workload at every dtype; one row per (case, dtype)."""
    want = set(only)
    rows: list[dict] = []
    for wl in zoo_mod.zoo(d, t, b):
        if want and wl.name not in want:
            continue
        for name in _DTYPES:
            try:
                row = _one(wl, name, reps)
            except (
                Exception
            ) as exc:  # a dtype that will not deliver is a row
                row = {
                    "case": f"zoo:{wl.name}",
                    "dtype": name,
                    "delivered": False,
                    "note": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print(f"zoo[{row['case']}][{name}]: {row}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--only", default="RMSNormPerHead")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print(
            "zoo_dtype: CUDA is required", file=__import__("sys").stderr
        )
        return 1
    only = tuple(x for x in args.only.split(",") if x)
    rows = measure(
        d=args.d, t=args.t, b=args.b, only=only, reps=args.reps
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
