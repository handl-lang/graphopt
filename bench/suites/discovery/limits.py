"""``limits`` — the registered-baseline comparison as a bench suite.

Thin adapter over :mod:`bench.limits` (the baseline registry +
runner): crosses every registered opponent with the discovery corpus
and packs the rows into a benchkit ``Report`` — one ``Finding`` per
(case, baseline) row plus a summary line — so the lane rides the
same catalog/ledger/baseline machinery as every other suite.

Direct flags for the lanes and filters live on the standalone CLI::

    python -m bench.limits.run --fast|--full [--cases a,b]
        [--baselines eager,paper_impl]

while the registered entry is::

    python -m bench run limits             # fast lane
    python -m bench.suites.discovery.limits --full
"""

# ruff: noqa: RUF001, RUF002 — math notation in prose.

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
from bench.limits.runner import CaseLimits, run_suite
from bench.limits.table import write_outputs

#: ``--quick`` (run-all) shrinks to the three cheapest cases.
QUICK = {"cases": "conv_bn_fold,mul_unsq_nat,weight_chain_fold"}


def _row_verdict(ratio: float | None) -> Verdict:
    if ratio is None:
        return Verdict.INCONCLUSIVE
    if ratio > 1.02:
        return Verdict.WIN
    if ratio < 0.98:
        return Verdict.NEGATIVE
    return Verdict.PARITY


def _findings(cl: CaseLimits) -> list[Finding]:
    """One finding per baseline row — losses report NEGATIVE."""
    out = []
    for r in cl.rows:
        if not r.available:
            out.append(
                Finding(
                    claim=(f"{cl.case.name}: {r.baseline} unavailable"),
                    verdict=Verdict.INCONCLUSIVE,
                    headline=r.caveat,
                    metric="registered baseline",
                    evidence={"baseline": r.baseline, "kind": r.kind},
                )
            )
            continue
        ratio = r.measured_ratio or r.ratio
        out.append(
            Finding(
                claim=(
                    f"{cl.case.name}: delivered form vs {r.baseline}"
                ),
                verdict=_row_verdict(ratio),
                headline=(
                    f"{ratio:.2f}×"
                    if ratio is not None
                    else r.caveat or "unmeasured"
                ),
                metric=f"baseline/mlopt ({r.axis})",
                value=ratio,
                evidence={
                    "baseline": r.baseline,
                    "kind": r.kind,
                    "verify_passed": r.verify_passed,
                    "caveat": r.caveat,
                },
            )
        )
    return out


def _cells(out: list[CaseLimits]) -> list[Cell]:
    """Reconstruct timed cells from the full-lane arm medians."""
    cells = []
    for cl in out:
        medians = cl.detail.get("median_s")
        if not medians:
            continue
        variants = [Variant(name=n, stmt=lambda: None) for n in medians]
        case = Case(
            name=cl.case.name,
            params={"category": cl.case.category},
            variants=variants,
            aux=dict(cl.detail),
        )
        cells.append(
            Cell(case=case, medians=dict(medians), iqr={}, aux={})
        )
    return cells


def run_bench(args) -> Report:
    """Run the corpus × the baseline registry into a ``Report``."""
    full = bool(getattr(args, "full", False))
    lane = "full" if full else "fast"
    cases = load_cases(getattr(args, "cases_dir", None))
    filt = getattr(args, "cases", None)
    if filt:
        keep = {c.strip() for c in filt.split(",") if c.strip()}
        cases = [c for c in cases if c.name in keep]
    baselines = getattr(args, "baselines", None)
    if isinstance(baselines, str):
        baselines = [
            x.strip() for x in baselines.split(",") if x.strip()
        ] or None
    rows = run_suite(
        cases,
        lane=lane,
        baselines=baselines,
        device=getattr(args, "device", None) or "cpu",
        min_run_time=getattr(args, "min_run_time", None) or 0.05,
        warmup=getattr(args, "warmup", None) or 3,
    )
    out_dir = getattr(args, "out", None)
    if out_dir is not None:
        out_dir = Path(out_dir)
        write_outputs(
            rows,
            lane=lane,
            md_path=out_dir / "limits_table.md",
            json_path=out_dir / "limits_table.json",
            device=getattr(args, "device", None) or "cpu",
            seed=getattr(args, "seed", None),
        )
    flat = [r for cl in rows for r in cl.rows]
    ran = [r for r in flat if r.available]
    wins = sum(
        1 for r in ran if (r.measured_ratio or r.ratio or 0) > 1.02
    )
    losses = sum(
        1 for r in ran if (r.measured_ratio or r.ratio or 1) < 0.98
    )
    findings = [
        Finding(
            claim=(
                "The limits table is generated — every registered "
                "baseline priced against every case it covers"
            ),
            verdict=Verdict.WIN if losses == 0 else Verdict.PARITY,
            headline=(
                f"{len(ran)}/{len(flat)} baseline rows ran — "
                f"{wins} ahead, {losses} honest losses"
            ),
            metric="rows run / wins / losses",
            value=float(wins),
            evidence={"cases": [cl.case.name for cl in rows]},
        )
    ]
    for cl in rows:
        if cl.error:
            findings.append(
                Finding(
                    claim=f"{cl.case.name}: case failed",
                    verdict=Verdict.INCONCLUSIVE,
                    headline=cl.error,
                    metric="search/fixture",
                    evidence={},
                )
            )
            continue
        findings.extend(_findings(cl))
    return Report(
        suite="limits",
        cells=_cells(rows),
        env=collect_env(getattr(args, "device", None)),
        title="Limits table — delivered form vs registered baselines",
        summary=(
            f"{len(rows)} cases × the baseline registry, lane={lane}: "
            f"{len(ran)} rows ran, {wins} ahead, {losses} losses."
        ),
        findings=findings,
        provenance={"lane": lane},
    )


def main(argv: list[str] | None = None) -> int:
    """Direct entry — ``python -m bench.suites.discovery.limits``."""
    p = argparse.ArgumentParser(prog="bench.suites.discovery.limits")
    p.add_argument("--full", action="store_true")
    p.add_argument("--cases", default="")
    p.add_argument("--baselines", default="")
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
