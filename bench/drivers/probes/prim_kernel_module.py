"""Module-level: do the prim folds pay when delivered by **kernels**?

The op-level probe shows a generated Triton body at ~0.37x the spec-walk.
The delivery probe showed the shipped family losing — but on fp64 CPU,
where Triton cannot run, so the two never met.  `measure_module` now takes
``device``/``dtype``; this driver joins them: the same module, the same
pipeline, two bodies for the prim ops (spec-walk vs generated kernel), and
the marginal reported for each.

Arms per body: ``spelled`` (the module), ``noprim`` (search+lower without
the prim folds), ``synop`` (with them).  The number that matters is
``synop / noprim`` — the fold's own effect on one executor — and
``synop / spelled`` for what a caller receives.

**Blocked (2026-10-25):** the plan executor allocates its intermediates
from a CPU ``like`` tensor, so a CUDA run raises "Expected all tensors to
be on the same device".  This driver is the harness for the fix; it will
run as written once ``torch_bridge.Plan.run_xs`` inherits the input's
device/dtype.

Run::

    .venv/bin/python -m bench.drivers.probes.prim_kernel_module \\
        --module RealGatedLRU --size large
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mlopt_discovery import shipped_primitives as sp

from bench.drivers.probes import prim_delivery_probe as probe
from bench.drivers.probes import prim_kernel_probe as kprobe


def _med(row: Any, arm: str) -> float | None:
    entry = row.times.get(arm)
    if not entry or "median" not in entry:
        return None
    return float(entry["median"])


def run(
    module: str = "RealGatedLRU",
    size: str = "large",
    *,
    device: str = "cuda",
    dtype: str = "float32",
    iters: int = 60,
    warmup: int = 20,
) -> dict[str, Any]:
    """Measure the same module with both prim bodies."""
    import mlopt_torch.torch_bridge as tb

    dt = torch.float64 if dtype == "float64" else torch.float32
    binds, notes = kprobe.kernels_for(sorted(kprobe.elementwise_ops(8)))
    orig = sp.register

    def register_with_kernels() -> int:
        n = orig()
        tb._IR_TO_TORCH.update(binds)
        return n

    kwargs = {
        "compile_arms": False,
        "iters": iters,
        "warmup": warmup,
        "device": device,
        "dtype": dt,
    }
    spec_walk = probe.measure_module(module, size, **kwargs)
    sp.register = register_with_kernels
    try:
        kerneled = probe.measure_module(module, size, **kwargs)
    finally:
        sp.register = orig

    rows = []
    for label, row in (("spec-walk", spec_walk), ("kernel", kerneled)):
        synop = _med(row, "synop")
        noprim = _med(row, "noprim")
        spelled = _med(row, "spelled")
        rows.append(
            {
                "body": label,
                "ms": {
                    "spelled": spelled,
                    "noprim": noprim,
                    "synop": synop,
                },
                "marginal": (synop / noprim)
                if synop and noprim
                else None,
                "e2e": (synop / spelled) if synop and spelled else None,
                "executors": dict(row.executors),
                "verified": row.verified.get("synop"),
                "allclose": row.allclose.get("synop"),
                "synops": list(row.synops),
            }
        )
    return {
        "module": module,
        "size": size,
        "device": device,
        "dtype": dtype,
        "kernels": sorted(binds),
        "declined": [n for n in notes if n["status"] != "bound"],
        "rows": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def summary(result: dict[str, Any]) -> str:
    """Render the two bodies side by side."""
    lines = [
        f"{result['module']} @ {result['size']} · {result['device']} "
        f"{result['dtype']} · kernels bound: {len(result['kernels'])}",
        "",
        "| body | spelled ms | noprim ms | synop ms | marginal | e2e | "
        "verified | equal |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        ms = row["ms"]
        lines.append(
            f"| {row['body']} | {_fmt(ms['spelled'])} | "
            f"{_fmt(ms['noprim'])} | {_fmt(ms['synop'])} | "
            f"{_fmt(row['marginal'])} | {_fmt(row['e2e'])} | "
            f"{row['verified']} | {row['allclose']} |"
        )
    lines += [
        "",
        "marginal = synop / noprim (same-executor fold effect); "
        "e2e = synop / spelled.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the module-level check and print the table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", default="RealGatedLRU")
    ap.add_argument("--size", default="large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="float32", choices=["float32", "float64"]
    )
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args(argv)
    result = run(
        args.module,
        args.size,
        device=args.device,
        dtype=args.dtype,
        iters=args.iters,
        warmup=args.warmup,
    )
    print(summary(result))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
