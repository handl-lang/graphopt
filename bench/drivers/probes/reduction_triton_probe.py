"""Does the reduction Triton piece beat compiled-spelled *and* inductor?

The p31 reduction tier shipped ``red_*`` claims delivered as the
``torch.compile``d spelled body: 0/30 kept at zoo sizes, and at
64×64×8 two ``red_*`` keeps ran 1.206× vs eager but 0.892× vs the
whole-model ``torch.compile`` incumbent.  This driver re-measures the
same claim sweep after the reduction Triton piece landed
(``genkernel.reduction_triton_bindings`` — ``Tier("reduction", …,
reduction_triton_bindings)``): ``play._zoo_case``'s intake
``profitable`` now grades the generated kernel through
``_triton_kernel_for``, so ``TRITON_REFEREE_LOG`` records whether the
reduction tier bound and what it measured against spelled.

Per workload the row carries every ``red_*`` consult (tier attempts,
bound kernel's spelled/kernel times, keep vote) plus the kept set.
Workloads that keep a ``red_*`` then run ``play.gen_probe``
end-to-end for the delivered-module ratios vs eager and vs
whole-model ``torch.compile`` — the number the lever is priced on.

Run::

    .venv/bin/python -m bench.drivers.probes.reduction_triton_probe \\
        --d 64 --t 64 --b 8 --out docs/reports/red_triton_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mlopt_discovery import genkernel as gk
from mlopt_discovery import play
from mlopt_discovery import zoo as zoo_mod


def audit(wl: Any, dtype: str) -> dict:
    """Run one (workload, dtype) cell; return its red_* consults.

    Two logs join per handler: ``TRITON_REFEREE_LOG`` records the
    tier-attempt trail and bind/decline outcome, while
    ``gk.PROFITABLE_LOG`` carries the measured
    ``(name, t_spelled, t_kernel, window, keep_vote)`` samples the
    referee priced — a bound kernel that never reaches
    ``_kernel_pays`` (verify failure, raised call) contributes no
    rows there and shows up as ``bound`` with empty ``samples``.
    """
    n0 = len(play.TRITON_REFEREE_LOG)
    n1 = len(gk.PROFITABLE_LOG)
    case = play._zoo_case(wl, dtype=dtype)
    recs = [
        r
        for r in play.TRITON_REFEREE_LOG[n0:]
        if str(r.get("tag", "")).startswith("red_")
    ]
    samples: dict[str, list] = {}
    for name, ts, tk, window, vote in gk.PROFITABLE_LOG[n1:]:
        if str(name).startswith("red_"):
            samples.setdefault(name, []).append(
                {
                    "window": window,
                    "t_spelled_us": round(ts * 1e6, 3),
                    "t_kernel_us": round(tk * 1e6, 3),
                    "ratio_kernel_over_spelled": round(tk / ts, 4)
                    if ts > 0
                    else None,
                    "keep_vote": bool(vote),
                }
            )
    kept = sorted(
        k for k in case[5]["gen_handlers"] if k.startswith("red_")
    )
    rows = []
    for r in recs:
        rows.append(
            {
                "tag": r.get("tag"),
                "kernel": r.get("kernel"),
                "pattern_head": r.get("pattern_head"),
                "outcome": r.get("outcome"),
                "attempts": r.get("attempts"),
                "reason": r.get("reason"),
                "samples": samples.get(r.get("tag"), []),
                "kept_by_referee": r.get("tag") in kept,
            }
        )
    return {
        "workload": wl.name,
        "dtype": dtype,
        # ``gen_handlers`` post-dates ``profitable`` — only kept
        # claims remain, so the minted count is the consult count
        # (every measurable minted claim reaches ``kernel_for``).
        "red_specs_minted": len(recs),
        "red_kept": kept,
        "consults": rows,
        "case": case,
    }


def measure(
    *,
    d: int,
    t: int,
    b: int,
    dtype: str,
    probe: bool,
    workloads: set[str] | None = None,
) -> list[dict]:
    """Audit every zoo workload; optionally gen_probe the payers."""
    rows: list[dict] = []
    for wl in zoo_mod.zoo(d, t, b):
        if workloads is not None and wl.name not in workloads:
            continue
        try:
            row = audit(wl, dtype)
        except Exception as exc:
            row = {
                "workload": wl.name,
                "dtype": dtype,
                "error": f"{type(exc).__name__}: {exc}",
            }
        case = row.pop("case", None)
        if probe and row.get("red_kept") and case is not None:
            try:
                row["gen_probe"] = play.gen_probe(case, reps=100)
            except Exception as exc:
                row["gen_probe"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        rows.append(row)
        summ = ", ".join(
            f"{c['tag']}:{c['outcome']}"
            + (
                f"[{c['samples'][0]['ratio_kernel_over_spelled']}x]"
                if c.get("samples")
                else ""
            )
            + (" kept" if c.get("kept_by_referee") else " drop")
            for c in row.get("consults", [])
        )
        print(
            f"zoo:{wl.name}[{dtype}]: kept={row.get('red_kept')} {summ}",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--t", type=int, default=64)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--dtype", default="fp64", choices=("fp64", "fp32"))
    ap.add_argument(
        "--workloads",
        default=None,
        help="comma-separated workload names to audit (default: all)",
    )
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print(
            "reduction_triton_probe: CUDA is required",
            file=__import__("sys").stderr,
        )
        return 1
    workloads = (
        {w.strip() for w in args.workloads.split(",") if w.strip()}
        if args.workloads
        else None
    )
    rows = measure(
        d=args.d,
        t=args.t,
        b=args.b,
        dtype=args.dtype,
        probe=args.probe,
        workloads=workloads,
    )
    bound = sum(
        1 for r in rows for c in r.get("consults", []) if c["outcome"] == "bound"
    )
    kept = sum(len(r.get("red_kept", ())) for r in rows)
    print(f"\n{len(rows)} workloads: {bound} bound consults, {kept} red_* kept")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(rows, indent=1, default=str) + "\n"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
