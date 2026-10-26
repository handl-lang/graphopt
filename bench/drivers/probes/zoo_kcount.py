"""How many kernels does each zoo workload launch, and does that predict pay?

`zoo_chain` says *whether* a delivered module beats eager.  This driver asks
*why*: per workload it profiles the **eager** forward and records the kernel
count, the device time and the wall time.  The hypothesis under test is that
the delivered module wins exactly where eager is launch-bound — many kernels,
small each — because the delivered body launches fewer of them (measured on
`TimeCondConv`: 22 vs 34 kernels, 146 vs 171 µs device time).

It is cheap (no `torch.compile`, no IR, no delivery), so it runs over the
whole zoo in a couple of minutes and can be joined against a `zoo_chain`
sweep by case name.

Run::

    .venv/bin/python -m bench.drivers.probes.zoo_kcount \\
        --out docs/reports/zoo_kcount.json

Then join: `kernels/fwd`, `device_us`, `device_share` against each row's
`speedup` from `zoo_chain_sweep.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mlopt_discovery import zoo as zoo_mod


def _profile(wl, *, d: int, t: int, b: int, reps: int) -> dict:
    """Time and count the kernels of one eager forward, in fp64 on CUDA."""
    model, feed = wl.build()
    model = model.double().cuda()
    args = tuple(
        x.double().cuda()
        for x in (feed if isinstance(feed, (tuple, list)) else (feed,))
    )
    with torch.no_grad():
        for _ in range(5):
            model(*args)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            model(*args)
        end.record()
        torch.cuda.synchronize()
        wall_us = start.elapsed_time(end) * 1e3 / reps
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(reps):
                model(*args)
            torch.cuda.synchronize()
    events = [
        e
        for e in prof.key_averages()
        if getattr(e, "self_device_time_total", 0) > 0
    ]
    device_us = sum(e.self_device_time_total for e in events) / reps
    return {
        "case": f"zoo:{wl.name}",
        "kernels": len(events),
        "device_us": round(device_us, 2),
        "wall_us": round(wall_us, 2),
        "device_share": round(device_us / wall_us, 3)
        if wall_us
        else None,
        "us_per_kernel": round(device_us / len(events), 2)
        if events
        else None,
    }


def measure(
    *, d: int, t: int, b: int, only: tuple[str, ...], reps: int
) -> list[dict]:
    want = set(only)
    rows: list[dict] = []
    for wl in zoo_mod.zoo(d, t, b):
        if want and wl.name not in want:
            continue
        try:
            row = _profile(wl, d=d, t=t, b=b, reps=reps)
        except (
            Exception
        ) as exc:  # a workload that will not build is a row
            rows.append(
                {
                    "case": f"zoo:{wl.name}",
                    "note": f"profile failed: {type(exc).__name__}: {exc}",
                }
            )
            print(f"zoo[{wl.name}]: profile failed: {exc}", flush=True)
            continue
        rows.append(row)
        print(f"zoo[{row['case']}]: {row}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument(
        "--only", default="", help="comma-separated workloads"
    )
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print(
            "zoo_kcount: CUDA is required",
            file=__import__("sys").stderr,
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
