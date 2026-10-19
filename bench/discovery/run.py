"""``python -m bench.discovery.run`` — the discovery benchmark CLI.

    .venv/bin/python -m bench.discovery.run --fast            # PR-gate lane
    .venv/bin/python -m bench.discovery.run --full            # measured lane
    .venv/bin/python -m bench.discovery.run --fast --cases conv_bn_fold
    .venv/bin/python -m bench.discovery.run --full --out bench/results

Writes ``<out>/scorecard.md`` + ``<out>/scorecard.json`` — default
``docs/`` (the committed, versioned scorecard).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.discovery.cases import load_cases
from bench.discovery.runner import run_suite
from bench.discovery.scorecard import write_outputs


def _print_row(row) -> None:
    if row.error:
        print(f"  {row.name:<20} ERROR — {row.error}", flush=True)
        return
    beat = {True: "BEAT", False: "tie", None: "—"}[row.beat_best]
    print(
        f"  {row.name:<20} found={row.found!s:<5} "
        f"cert={row.certified!s:<5} "
        f"cost {row.cost_before:,.0f} -> {row.cost_after:,.0f} "
        f"({row.axis})  beat={beat}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the suite, emit the scorecard."""
    p = argparse.ArgumentParser(prog="bench.discovery.run")
    lane = p.add_mutually_exclusive_group()
    lane.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="symbolic dag_cost only (seconds; the gate lane)",
    )
    lane.add_argument(
        "--full",
        action="store_true",
        help="measured pay — lower + verify + timed arms",
    )
    p.add_argument(
        "--cases",
        default="",
        help="comma-separated case-name filter",
    )
    p.add_argument(
        "--cases-dir",
        type=Path,
        default=None,
        help="override the TOML case directory",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs"),
        help="scorecard output directory (default docs/)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-run-time", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args(argv)

    cases = load_cases(args.cases_dir)
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.name in keep]
        missing = keep - {c.name for c in cases}
        if missing:
            print(f"unknown case(s): {', '.join(sorted(missing))}")
            return 2
    if args.seed is not None:
        from dataclasses import replace

        cases = [replace(c, seed=args.seed) for c in cases]

    lane_name = "full" if args.full else "fast"
    print(
        f"discovery benchmark — {len(cases)} case(s), lane={lane_name}",
        flush=True,
    )
    rows = run_suite(
        cases,
        lane=lane_name,
        device=args.device,
        min_run_time=args.min_run_time,
        warmup=args.warmup,
        progress=_print_row,
    )
    write_outputs(
        rows,
        lane=lane_name,
        md_path=args.out / "scorecard.md",
        json_path=args.out / "scorecard.json",
        device=args.device,
    )
    n_found = sum(r.found for r in rows)
    n_cert = sum(r.certified for r in rows)
    n_beat = sum(1 for r in rows if r.beat_best)
    print(
        f"scorecard: {n_found}/{len(rows)} found, "
        f"{n_cert}/{len(rows)} certified, "
        f"{n_beat} beat-best-registered",
        flush=True,
    )
    print(f"  -> {args.out / 'scorecard.md'}")
    print(f"  -> {args.out / 'scorecard.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
