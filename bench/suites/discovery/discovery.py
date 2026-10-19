"""``discovery`` — the named-optimization scorecard as a bench suite.

Thin adapter over :mod:`bench.discovery` (the corpus + runner): runs
the case suite and packs the rows into a benchkit ``Report`` — one
``Finding`` per named optimization plus a summary line — so the
benchmark rides the same catalog/ledger/baseline machinery as every
other suite.

Direct flags for the lanes and corpus knobs live on the standalone
CLI::

    python -m bench.discovery.run --fast|--full [--cases a,b]

while the registered entry is::

    python -m bench run discovery            # fast lane
    python -m bench.suites.discovery.discovery --full
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.benchkit import (
    Case,
    Cell,
    Finding,
    Report,
    Variant,
    Verdict,
    collect_env,
)
from bench.discovery.cases import load_cases
from bench.discovery.runner import CaseResult, run_suite
from bench.discovery.scorecard import write_outputs

#: ``--quick`` (run-all) shrinks to the three cheapest cases.
QUICK = {"cases": "conv_bn_fold,mul_unsq_nat,weight_chain_fold"}


def _verdict(r: CaseResult) -> Verdict:
    if r.error:
        return Verdict.INCONCLUSIVE
    if r.beat_best:
        return Verdict.WIN
    if r.found:
        return Verdict.PARITY  # rediscovery — ties the registered floor
    return Verdict.NEGATIVE


def _finding(r: CaseResult) -> Finding:
    floors = [a.cost for a in r.baselines if a.cost is not None]
    floor = min(floors) if floors else None
    headline = (
        f"{r.axis} {r.cost_before:,.0f} → {r.cost_after:,.0f}"
        if r.cost_before is not None and r.cost_after is not None
        else (r.error or "no extraction")
    )
    return Finding(
        claim=f"{r.name}: {r.title}",
        verdict=_verdict(r),
        headline=headline,
        metric=f"{r.axis} dag_cost",
        value=r.cost_after,
        evidence={
            "found": r.found,
            "certified": r.certified,
            "verified": r.verified,
            "floor": floor,
            "beat_best": r.beat_best,
            "beat_best_measured": r.beat_best_measured,
            "signals": r.signals_seen,
        },
    )


def _cells(rows: list[CaseResult]) -> list[Cell]:
    """Reconstruct timed cells from the full-lane arm medians."""
    cells = []
    for r in rows:
        medians = r.detail.get("median_s")
        if not medians:
            continue
        arms = {a.name: a for a in r.baselines}
        variants = [
            Variant(
                name=n,
                stmt=lambda: None,
                note=arms.get(n).note
                if n in arms
                else "certified extraction",
            )
            for n in medians
        ]
        case = Case(
            name=r.name,
            params={"category": r.category},
            variants=variants,
            aux=dict(r.detail),
        )
        cells.append(
            Cell(case=case, medians=dict(medians), iqr={}, aux={})
        )
    return cells


def run_bench(args) -> Report:
    """Run the corpus and pack the rows into a benchkit ``Report``."""
    full = bool(getattr(args, "full", False))
    lane = "full" if full else "fast"
    cases = load_cases(getattr(args, "cases_dir", None))
    filt = getattr(args, "cases", None)
    if filt:
        keep = {c.strip() for c in filt.split(",") if c.strip()}
        cases = [c for c in cases if c.name in keep]
    rows = run_suite(
        cases,
        lane=lane,
        device=getattr(args, "device", None) or "cpu",
        min_run_time=getattr(args, "min_run_time", None) or 0.05,
        warmup=getattr(args, "warmup", None) or 3,
    )
    out = getattr(args, "out", None)
    if out is not None:
        out = Path(out)
        write_outputs(
            rows,
            lane=lane,
            md_path=out / "scorecard.md",
            json_path=out / "scorecard.json",
            device=getattr(args, "device", None) or "cpu",
        )
    n_found = sum(r.found for r in rows)
    n_cert = sum(r.certified for r in rows)
    n_beat = sum(1 for r in rows if r.beat_best)
    findings = [
        Finding(
            claim=(
                "The engine derives the registered named optimizations "
                "and prices them against the known-good floor"
            ),
            verdict=Verdict.WIN
            if n_found == len(rows) and n_cert == len(rows)
            else Verdict.NEGATIVE,
            headline=(
                f"{n_found}/{len(rows)} found, {n_cert} certified, "
                f"{n_beat} below the registered floor"
            ),
            metric="cases found/certified/beat",
            value=float(n_beat),
            evidence={"cases": [r.name for r in rows]},
        )
    ] + [_finding(r) for r in rows]
    return Report(
        suite="discovery",
        cells=_cells(rows),
        env=collect_env(getattr(args, "device", None)),
        title="Discovery benchmark — named optimizations",
        summary=(
            f"{len(rows)} cases, lane={lane}: {n_found} derived, "
            f"{n_cert} certified, {n_beat} below the floor."
        ),
        findings=findings,
        provenance={"lane": lane},
    )


def main(argv: list[str] | None = None) -> int:
    """Direct entry — ``python -m bench.suites.discovery.discovery``."""
    p = argparse.ArgumentParser(prog="bench.suites.discovery.discovery")
    p.add_argument("--full", action="store_true")
    p.add_argument("--cases", default="")
    p.add_argument("--out", type=Path, default=Path("bench/results"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-run-time", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args(argv)
    report = run_bench(args)
    print(report.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
