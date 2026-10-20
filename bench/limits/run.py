"""``python -m bench.limits.run`` — the limits-table CLI.

    .venv/bin/python -m bench.limits.run --fast            # PR-gate lane
    .venv/bin/python -m bench.limits.run --full            # measured lane
    .venv/bin/python -m bench.limits.run --fast --cases conv_bn_fold
    .venv/bin/python -m bench.limits.run --full --baselines eager,paper_impl

Writes ``<out>/limits_table.md`` + ``<out>/limits_table.json`` —
default ``docs/`` (the committed, versioned limits table).  The
table crosses every registered baseline in ``bench/limits/`` with
the discovery corpus, so the "results — limits" rows in the
results docs are *generated*, not hand-copied.
"""

# ruff: noqa: RUF001 — math notation in prose.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.discovery.cases import load_cases
from bench.limits import registry
from bench.limits.runner import CaseLimits, run_suite
from bench.limits.table import write_outputs


def _print_case(cl: CaseLimits) -> None:
    """Progress lines for one case — one per baseline row."""
    if cl.error:
        print(f"  {cl.case.name:<20} ERROR — {cl.error}", flush=True)
        return
    for r in cl.rows:
        ratio = r.measured_ratio or r.ratio
        tag = f"{ratio:.2f}×" if ratio is not None else "—"
        mark = "" if r.available else " (skip)"
        print(
            f"  {cl.case.name:<18} {r.baseline:<14} {tag:>7}{mark}"
            f"  {r.caveat}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the lane, emit the limits table."""
    p = argparse.ArgumentParser(prog="bench.limits.run")
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
        "--baselines",
        default="",
        help="comma-separated baseline-name filter "
        "(default: every registered baseline each case supports)",
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
        help="limits-table output directory (default docs/)",
    )
    p.add_argument(
        "--pin",
        action="store_true",
        help="pin the table as bench/baselines/limits_table.json",
    )
    p.add_argument(
        "--baselines",
        type=Path,
        default=Path("bench/baselines"),
    )
    p.add_argument(
        "--no-ledger",
        action="store_true",
        help="skip the ledger append (bench/results/ledger.jsonl)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-run-time", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--list-baselines",
        action="store_true",
        help="print the registered roster and exit",
    )
    args = p.parse_args(argv)

    if args.list_baselines:
        for name in registry.names():
            s = registry.spec(name)
            cases = ", ".join(s.cases) if s.cases else "all"
            print(f"{name:<16} {s.kind:<8} {cases}")
        return 0

    cases = load_cases(args.cases_dir)
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.name in keep]
        missing = keep - {c.name for c in cases}
        if missing:
            print(f"unknown case(s): {', '.join(sorted(missing))}")
            return 2
    baselines = [
        x.strip() for x in args.baselines.split(",") if x.strip()
    ] or None
    if baselines is not None:
        unknown = [n for n in baselines if n not in registry.names()]
        if unknown:
            print(
                f"unknown baseline(s): {', '.join(unknown)} — "
                f"registered: {', '.join(registry.names())}"
            )
            return 2
    if args.seed is not None:
        from dataclasses import replace

        cases = [replace(c, seed=args.seed) for c in cases]

    lane_name = "full" if args.full else "fast"
    print(
        f"limits table — {len(cases)} case(s), "
        f"{len(registry.names())} registered baselines, "
        f"lane={lane_name}",
        flush=True,
    )
    rows = run_suite(
        cases,
        lane=lane_name,
        baselines=baselines,
        device=args.device,
        min_run_time=args.min_run_time,
        warmup=args.warmup,
        progress=_print_case,
    )
    write_outputs(
        rows,
        lane=lane_name,
        md_path=args.out / "limits_table.md",
        json_path=args.out / "limits_table.json",
        device=args.device,
    )
    flat = [r for cl in rows for r in cl.rows]
    ran = [r for r in flat if r.available]
    skipped = len(flat) - len(ran)
    from bench.benchkit import table as _t

    n_win = sum(
        1 for r in ran if (r.measured_ratio or r.ratio or 0) > 1.02
    )
    record = _t.table_record(
        "limits_table",
        (
            {
                "name": f"{r.case}/{r.baseline}",
                "metrics": {"ratio": r.measured_ratio or r.ratio},
            }
            for r in flat
            if r.available
        ),
        directions={"ratio": "down"},
        findings=[
            {
                "claim": "the delivered form beats the registered "
                "floor where structure exists",
                "verdict": "win" if n_win else "parity",
                "headline": f"{n_win}/{len(ran)} rows ahead",
            }
        ],
        provenance={"quick": lane_name == "fast"},
    )
    if not args.no_ledger:
        _t.append_record(record)
    if args.pin:
        print(
            f"  -> pinned {_t.write_table_baseline(record, args.baselines)}"
        )
    print(
        f"limits: {len(flat)} rows over {len(rows)} case(s) "
        f"({skipped} skipped with reasons)",
        flush=True,
    )
    print(f"  -> {args.out / 'limits_table.md'}")
    print(f"  -> {args.out / 'limits_table.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
