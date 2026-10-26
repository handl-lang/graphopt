"""Chain-level pay on the **real** zoo corpus: does `gen` deliver on modules?

`--probe` measures two fixed cases (`chainfuse`, `mulsoftplus`) and gets
2.87x.  That is a whole-chain kernel launched once over a synthetic tape.
The question this driver answers is whether the same machinery pays on the
held-out model zoo, at wide shapes, on real module boundaries.

Each workload goes through the shipped path unchanged: build -> fp64 ->
`TorchSource.to_ir` -> mint tiers -> `genkernel.profitable` (the measured
cost referee) -> `gen_sink` -> `play.gen_probe` (deliver, verify, time vs
eager and `torch.compile`).  No special casing, so the row is the number a
caller would get.

Wide shapes matter: the zoo's default is ``d16 t8 b2``, where the kernels
are too small for the launcher to matter.  ``--d 64 --t 32 --b 8`` is the
shape the module probe uses.

**Sample, not a sweep.**  `torch.compile` is the baseline, so each workload
costs tens of seconds at wide shapes and a handful time out on this box.
Select workloads with ``--only``; a partial table is reported as partial.

Run::

    .venv/bin/python -m bench.drivers.probes.zoo_chain \\
        --only GatedAttentionUnit,MLPMixerBlock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlopt_discovery import zoo as zoo_mod
from mlopt_discovery.play import _zoo_case, gen_probe

DEFAULT_ONLY = "GatedAttentionUnit,MLPMixerBlock,BlockSparseAttention,TimeCondConv,FNOBlock"


def measure(
    *, d: int, t: int, b: int, only: tuple[str, ...], reps: int
) -> list[dict]:
    """Run the shipped gen path over the named zoo workloads."""
    want = set(only)
    rows: list[dict] = []
    for wl in zoo_mod.zoo(d, t, b):
        if want and wl.name not in want:
            continue
        try:
            case = _zoo_case(wl)
        except Exception as exc:
            rows.append(
                {
                    "case": f"zoo:{wl.name}",
                    "delivered": False,
                    "note": f"export declined: {type(exc).__name__}: {exc}",
                }
            )
            print(f"zoo[{wl.name}]: export declined: {exc}", flush=True)
            continue
        row = gen_probe(case, reps=reps)
        rows.append(row)
        print(f"zoo[{row['case']}]: {row}", flush=True)  # stdout-compat
    return rows


def main(argv: list[str] | None = None) -> int:
    """Parse the shape / selection flags and print the rows."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument(
        "--only",
        default=DEFAULT_ONLY,
        help="comma-separated workload names; empty string = every workload",
    )
    ap.add_argument("--out", default="", help="optional JSON path")
    args = ap.parse_args(argv)

    only = tuple(x for x in args.only.split(",") if x)
    rows = measure(
        d=args.d, t=args.t, b=args.b, only=only, reps=args.reps
    )
    won = [
        r
        for r in rows
        if r.get("delivered") and (r.get("speedup") or 0) > 1.0
    ]
    print(
        f"\n{len(rows)} workloads: {len(won)} faster than eager, "
        f"{sum(1 for r in rows if not r.get('delivered'))} undeliverable"
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(rows, indent=1, default=str) + "\n"
        )
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
