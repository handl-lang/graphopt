"""The frontier — theoretical headroom vs what delivery captures.

The question this suite answers, before any further integration work:

    Given a measured row, what is the *most* an equality-preserving
    optimizer could win on it — and how much of that does the
    delivered body already take?

The bound.  A rewrite may only replace a term with an *equal* term, so
it cannot change the arithmetic a dense model must perform.  The only
headroom it can reach is the part of wall time that is not device
work: dispatch, launch gaps, materialization, dtype traffic.  If eager
spends a fraction ``F`` of wall time off-device, removing **all** of it
bounds the speedup at

    ceiling = 1 / (1 - F)          (Amdahl, on dispatch)

``F`` is not a guess: ``docs/reports/zoo_kcount.json`` measures
``device_share`` per case with the profiler, so ``F = 1 - device_share``
is the measured dispatch share of the spelled module.

The capture.  Against that ceiling, the delivered body's measured
speedup ``s`` gives

    captured = (s - 1) / (ceiling - 1)

the fraction of the available dispatch headroom the delivered form
actually takes.  ``captured <= 0`` means the delivery *loses* time
elsewhere (fp64 GEMMs, generic per-leaf lowering, materialized
intermediates) — the ceiling is not the binding constraint, delivery is.

Inputs (both already committed — this suite re-reads measurements; it
does not fabricate timings):

* ``docs/reports/zoo_chain_sweep.json`` — the zoo workloads, 3-5 rounds
  each, with ``runs[].{eager_us,gen_us,compiled_us}`` and ``verified``;
* ``docs/reports/zoo_kcount.json`` — per case ``kernels``, ``device_us``,
  ``wall_us``, ``device_share``.

Outputs:

* ``docs/reports/frontier.json`` — the canonical payload;
* ``docs/reports/frontier.md`` — the written report;
* a ``Report`` for the cross-suite dashboard.

    .venv/bin/python -m bench.suites.integration.frontier
    python -m bench run frontier
"""

from __future__ import annotations

import json
import statistics as stats
from dataclasses import dataclass
from pathlib import Path

from bench.benchkit.model import Case, Cell, Finding, Variant, Verdict
from bench.benchkit.report import Report

_REPO = Path(__file__).resolve().parents[3]

#: The measured inputs — re-read, never re-timed, by this suite.
DEFAULT_SWEEP = _REPO / "docs" / "reports" / "zoo_chain_sweep.json"
DEFAULT_KCOUNT = _REPO / "docs" / "reports" / "zoo_kcount.json"

#: Where the formal report lands (JSON + Markdown twins).
DOC_JSON = _REPO / "docs" / "reports" / "frontier.json"
DOC_MD = _REPO / "docs" / "reports" / "frontier.md"

#: A speedup at or above this is a win (matches docs/EVIDENCE.md §9).
WIN_TOL = 1.02

#: Below this much ceiling room the capture ratio divides by noise.
#: A row whose whole ceiling is 1.02x cannot say anything about how
#: much headroom delivery takes — a 2% timing wobble is ±100% of it.
CAPTURE_MIN_ROOM = 0.10

#: Regime edges on the measured dispatch share ``F``.
REGIMES: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.05, "compute-bound"),
    (0.05, 0.15, "mixed"),
    (0.15, 0.30, "launch-leaning"),
    (0.30, 1.01, "launch-bound"),
)


@dataclass(frozen=True)
class Row:
    """One measured workload joined against its profiled dispatch share.

    ``eager_us`` / ``delivered_us`` are medians over the sweep's rounds;
    ``speedup`` is recomputed from those medians rather than copied from
    the record's own ratio, so a re-measure with a different round count
    cannot disagree with itself.
    """

    case: str
    eager_us: float
    delivered_us: float
    device_share: float
    kernels: int
    verified: bool
    compiled_us: float | None = None

    @property
    def vs_compile(self) -> float | None:
        """Speedup over ``torch.compile`` (>1 wins).

        The free competitor: every PyTorch user has Inductor. A row that
        only beats eager has beaten nothing anyone had to install.
        """
        if not self.compiled_us:
            return None
        return self.compiled_us / self.delivered_us

    @property
    def beats_compile(self) -> bool | None:
        """Did the delivered form beat ``torch.compile`` past tolerance?"""
        ratio = self.vs_compile
        if ratio is None:
            return None
        return ratio >= WIN_TOL

    @property
    def speedup(self) -> float:
        """Delivered speedup over eager (>1 wins)."""
        return self.eager_us / self.delivered_us

    @property
    def dispatch_share(self) -> float:
        """``F`` — the fraction of eager wall time that is off-device."""
        return max(0.0, min(1.0, 1.0 - self.device_share))

    @property
    def ceiling(self) -> float:
        """The Amdahl bound on any dispatch-removing rewrite."""
        f = self.dispatch_share
        return 1.0 / (1.0 - f) if f < 1.0 else float("inf")

    @property
    def headroom_us(self) -> float:
        """The wall time a perfect dispatch elision would remove."""
        f = self.dispatch_share
        return self.eager_us * f if f < 1.0 else self.eager_us

    @property
    def captured(self) -> float | None:
        """Fraction of the headroom taken; ``None`` when none is usable.

        Only defined where the ceiling holds material room
        (``CAPTURE_MIN_ROOM``).  A row capped at 1.02x has a denominator
        of 0.02, so a 2% timing wobble would read as ±100% capture —
        that is noise, not a measurement, and it reports ``None``.

        ``> 1`` is impossible for a dispatch-only optimisation and is
        reported as-is when it happens — it means the delivered body
        won something the ceiling did not model (a cheaper algorithm,
        a better dtype), which is a finding about the model, not a bug.
        """
        room = self.ceiling - 1.0
        if room < CAPTURE_MIN_ROOM:
            return None
        return (self.speedup - 1.0) / room

    @property
    def regime(self) -> str:
        """Which dispatch regime this row belongs to."""
        f = self.dispatch_share
        for lo, hi, name in REGIMES:
            if lo <= f < hi:
                return name
        return "launch-bound"

    @property
    def is_win(self) -> bool:
        """Whether the delivered body beat the tolerance."""
        return self.speedup >= WIN_TOL

    def to_dict(self) -> dict:
        """The row as it is written to the report payload."""
        return {
            "case": self.case,
            "regime": self.regime,
            "eager_us": round(self.eager_us, 3),
            "delivered_us": round(self.delivered_us, 3),
            "speedup": round(self.speedup, 4),
            "device_share": round(self.device_share, 4),
            "dispatch_share": round(self.dispatch_share, 4),
            "ceiling": round(self.ceiling, 4),
            "headroom_us": round(self.headroom_us, 3),
            "captured": (
                None
                if self.captured is None
                else round(self.captured, 4)
            ),
            "kernels": self.kernels,
            "verified": self.verified,
            "compiled_us": (
                None if self.compiled_us is None else round(self.compiled_us, 3)
            ),
            "vs_compile": (
                None if self.vs_compile is None else round(self.vs_compile, 4)
            ),
            "beats_compile": self.beats_compile,
        }


def _median(values: list[float]) -> float:
    return float(stats.median(values))


def load_rows(
    sweep_path: Path = DEFAULT_SWEEP,
    kcount_path: Path = DEFAULT_KCOUNT,
) -> list[Row]:
    """Join the measured sweep against the profiled kernel counts.

    A case is kept only when it appears in *both* artifacts and carries
    at least one usable round — the sweep has cases the profiler never
    reached, and filling those with an assumed dispatch share would be
    the exact dishonesty this suite exists to prevent.
    """
    sweep = json.loads(Path(sweep_path).read_text())
    kcount = {
        r["case"]: r for r in json.loads(Path(kcount_path).read_text())
    }

    rows: list[Row] = []
    for rec in sweep:
        prof = kcount.get(rec["case"])
        runs = [r for r in rec.get("runs") or [] if r.get("eager_us")]
        if not prof or not runs:
            continue
        delivered = [r["gen_us"] for r in runs if r.get("gen_us")]
        if not delivered:
            delivered = [
                r["compiled_us"] for r in runs if r.get("compiled_us")
            ]
        if not delivered:
            continue
        compiled = [r["compiled_us"] for r in runs if r.get("compiled_us")]
        rows.append(
            Row(
                case=rec["case"],
                eager_us=_median([r["eager_us"] for r in runs]),
                delivered_us=_median(delivered),
                device_share=float(prof["device_share"]),
                kernels=int(prof.get("kernels") or 0),
                verified=bool(rec.get("verified")),
                compiled_us=_median(compiled) if compiled else None,
            )
        )
    return sorted(rows, key=lambda r: r.dispatch_share)


def _regime_stats(rows: list[Row]) -> list[dict]:
    """Per-regime summary: where the headroom is, and what is taken."""
    out = []
    for lo, hi, name in REGIMES:
        group = [r for r in rows if r.regime == name]
        if not group:
            continue
        caps = [r.captured for r in group if r.captured is not None]
        out.append(
            {
                "regime": name,
                "n": len(group),
                "dispatch_lo": lo,
                "dispatch_hi": min(hi, 1.0),
                "median_dispatch_share": round(
                    _median([r.dispatch_share for r in group]), 4
                ),
                "median_ceiling": round(
                    _median([r.ceiling for r in group]), 3
                ),
                "max_ceiling": round(max(r.ceiling for r in group), 3),
                "median_speedup": round(
                    _median([r.speedup for r in group]), 4
                ),
                "wins": sum(1 for r in group if r.is_win),
                "beat_compile": sum(
                    1 for r in group if r.beats_compile is True
                ),
                "with_compile": sum(
                    1 for r in group if r.vs_compile is not None
                ),
                "median_vs_compile": (
                    None
                    if not any(r.vs_compile for r in group)
                    else round(
                        _median(
                            [
                                r.vs_compile
                                for r in group
                                if r.vs_compile is not None
                            ]
                        ),
                        4,
                    )
                ),
                "median_captured": (
                    None if not caps else round(_median(caps), 4)
                ),
            }
        )
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r without numpy — both inputs are already floats."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / (sxx * syy) ** 0.5


def _verdict_ceiling(compute_bound: dict | None):
    """Is there a theoretical limit on whole-model rewrites?"""
    if compute_bound is None:
        return Verdict.INCONCLUSIVE, "no compute-bound rows measured"
    cap = compute_bound["median_ceiling"]
    if cap <= 1.10:
        return (
            Verdict.NEGATIVE,
            f"compute-bound rows are capped at {cap:.2f}x — equality-"
            "preserving rewrites cannot reach them",
        )
    return (
        Verdict.WIN,
        f"compute-bound rows still show {cap:.2f}x headroom — the cap "
        "assumption needs re-measuring",
    )


def _verdict_capture(rows: list[Row]):
    """Does the delivered body take the headroom the ceiling allows?"""
    caps = [r.captured for r in rows if r.captured is not None]
    if not caps:
        return Verdict.INCONCLUSIVE, "no finite-ceiling rows", None
    med = _median(caps)
    n = len(caps)
    if med <= 0.0:
        return (
            Verdict.NEGATIVE,
            f"median capture {med:+.2f} over {n} rows with material "
            "headroom — delivery loses more than the algebra gains; the "
            "binding constraint is the delivered body, not the ceiling",
            med,
        )
    if med < 0.5:
        return (
            Verdict.NEGATIVE,
            f"median capture {med:.2f} over {n} rows — less than half "
            "the available headroom is taken",
            med,
        )
    return (
        Verdict.WIN,
        f"median capture {med:.2f} over {n} rows — the delivered body "
        "takes most of the measured headroom",
        med,
    )


def build_findings(rows: list[Row]) -> list[Finding]:
    """The suite's conclusions, each carrying its number."""
    regimes = {r["regime"]: r for r in _regime_stats(rows)}
    cb = regimes.get("compute-bound")
    launch = regimes.get("launch-bound")

    v_ceiling, h_ceiling = _verdict_ceiling(cb)
    v_capture, h_capture, med_capture = _verdict_capture(rows)

    r_de: float | None = None
    if len(rows) >= 3:
        r_de = _pearson(
            [r.dispatch_share for r in rows],
            [r.speedup for r in rows],
        )

    return [
        Finding(
            claim=(
                "An equality-preserving rewrite cannot change a dense "
                "model's FLOPs, so its ceiling is the dispatch share, "
                "1/(1-F). Where F is small the ceiling is ~1x."
            ),
            verdict=v_ceiling,
            headline=h_ceiling,
            metric="median_ceiling_compute_bound",
            value=None if cb is None else cb["median_ceiling"],
            evidence={"compute_bound": cb, "launch_bound": launch},
        ),
        Finding(
            claim=(
                "The delivered body captures the headroom the ceiling "
                "allows: (speedup-1)/(ceiling-1)."
            ),
            verdict=v_capture,
            headline=h_capture,
            metric="median_captured_fraction",
            value=med_capture,
            evidence={"n": len(rows)},
        ),
        Finding(
            claim=(
                "Dispatch share predicts the sign of the win. (It does "
                "not always — the delivered body's own cost sets it.)"
            ),
            verdict=(
                Verdict.INCONCLUSIVE
                if r_de is None or abs(r_de) < 0.35
                else Verdict.WIN
            ),
            headline=(
                "r is not reported (too few rows)"
                if r_de is None
                else f"r(F, speedup) = {r_de:+.2f} — dispatch share "
                + (
                    "does not predict the sign alone"
                    if abs(r_de) < 0.35
                    else "tracks the sign"
                )
            ),
            metric="pearson_dispatch_vs_speedup",
            value=r_de,
        ),
        Finding(
            claim=(
                "The wins survive `torch.compile` — the free competitor "
                "every PyTorch user already has — not just eager."
            ),
            verdict=(
                Verdict.WIN
                if launch and launch["beat_compile"] >= launch["with_compile"] / 2
                else Verdict.NEGATIVE
            ),
            headline=(
                f"{launch['beat_compile']}/{launch['with_compile']} "
                f"launch-bound rows beat torch.compile (median "
                f"{launch['median_vs_compile']:.2f}x); compute-bound "
                f"{cb['beat_compile']}/{cb['with_compile']} at "
                f"{cb['median_vs_compile']:.2f}x"
                if launch and cb
                else "no rows carry a torch.compile arm"
            ),
            metric="beat_compile_fraction_launch_bound",
            value=(
                None
                if not launch or not launch["with_compile"]
                else launch["beat_compile"] / launch["with_compile"]
            ),
            evidence={"launch_bound": launch, "compute_bound": cb},
        ),
        Finding(
            claim=(
                "Every delivered row is verified equal to the spelled "
                "module — the frontier is measured on correct deliveries."
            ),
            verdict=(
                Verdict.WIN
                if rows and all(r.verified for r in rows)
                else Verdict.REGRESSION
            ),
            headline=(
                f"{sum(1 for r in rows if r.verified)}/{len(rows)} rows "
                "verified equal"
            ),
            metric="verified_fraction",
            value=(
                None
                if not rows
                else sum(1 for r in rows if r.verified) / len(rows)
            ),
        ),
    ]


def _cells(rows: list[Row]) -> list[Cell]:
    """One cell per row so the dashboard can slice the frontier."""

    def _noop() -> None:
        return None

    cells = []
    for r in rows:
        case = Case(
            name=r.case,
            params={
                "regime": r.regime,
                "dispatch_share": round(r.dispatch_share, 4),
                "ceiling": round(r.ceiling, 3),
            },
            variants=[
                Variant(name="eager", stmt=_noop),
                Variant(name="delivered", stmt=_noop),
            ],
            aux=r.to_dict(),
        )
        cells.append(
            Cell(
                case=case,
                medians={
                    "eager": r.eager_us / 1e6,
                    "delivered": r.delivered_us / 1e6,
                },
                iqr={},
                aux=r.to_dict(),
            )
        )
    return cells


def render_markdown(rows: list[Row], findings: list[Finding]) -> str:
    """The written report — the bound per regime, then the rows."""
    regime_stats = _regime_stats(rows)
    lines = [
        "# The frontier — theoretical headroom vs delivered capture",
        "",
        "**Bound.** A rewrite may only replace a term with an *equal* "
        "term, so it cannot change the arithmetic a dense model must "
        "do. The only headroom it can reach is the off-device share of "
        "wall time, so `ceiling = 1/(1-F)` with `F` the measured "
        "dispatch share (`1 - device_share`, profiler).",
        "",
        "**Capture.** `captured = (speedup-1)/(ceiling-1)`: the "
        "fraction of that headroom the delivered body actually takes. "
        "Negative means delivery costs more than the algebra saves.",
        "",
        f"- rows: {len(rows)} · verified: "
        f"{sum(1 for r in rows if r.verified)}/{len(rows)} · "
        f"win tolerance: ≥{WIN_TOL:.2f}x",
        "",
        "## Findings",
        "",
    ]
    for f in findings:
        lines += [
            f"### {f.verdict.badge} — {f.claim}",
            "",
            f.headline,
            "",
        ]
    lines += [
        "## By regime",
        "",
        "| regime | F range | n | median F | median ceiling | max "
        "ceiling | median speedup | wins | beat compile | median captured |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in regime_stats:
        med_cap = s["median_captured"]
        bc = (
            "—"
            if not s["with_compile"]
            else f"{s['beat_compile']}/{s['with_compile']}"
        )
        lines.append(
            f"| {s['regime']} | {s['dispatch_lo']:.2f}-"
            f"{s['dispatch_hi']:.2f} | {s['n']} | "
            f"{s['median_dispatch_share']:.3f} | "
            f"{s['median_ceiling']:.2f}x | {s['max_ceiling']:.2f}x | "
            f"{s['median_speedup']:.3f}x | {s['wins']}/{s['n']} | {bc} | "
            + ("—" if med_cap is None else f"{med_cap:+.2f}")
            + " |"
        )
    lines += [
        "",
        "## Rows",
        "",
        "| case | F | ceiling | speedup | captured | kernels | "
        "eager µs | delivered µs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cap = "—" if r.captured is None else f"{r.captured:+.2f}"
        lines.append(
            f"| `{r.case}` | {r.dispatch_share:.3f} | "
            f"{r.ceiling:.2f}x | {r.speedup:.3f}x | {cap} | "
            f"{r.kernels} | {r.eager_us:.1f} | {r.delivered_us:.1f} |"
        )
    lines += [
        "",
        "## What this bounds, and what it does not",
        "",
        "- It bounds **dispatch-removing** rewrites. It does not bound "
        "vocabulary growth: a *new* operation can change the arithmetic"
        " a module performs (blocked scans, sparsity, low rank), and "
        "that escapes this ceiling entirely.",
        "- It assumes the delivered body performs the same arithmetic "
        "as eager. A delivery that changes dtype upstream of verify "
        "(`fp32` where the contract allows) wins outside the model, "
        "which is why `captured > 1` is reported rather than clipped.",
        "- `F` is measured on the spelled module. A case whose eager "
        "form already fuses (a single SDPA call) has `F≈0` for the "
        "right reason: there is nothing off-device left to remove.",
        f"- `capture` is only defined where the ceiling holds at least "
        f"{CAPTURE_MIN_ROOM:.0%} room; below that the denominator is "
        "timing noise, so those rows report `—` rather than a ratio "
        "that a 2% wobble would swing by 100%.",
        "",
        "_Generated by `bench/suites/integration/frontier.py` from "
        "`docs/reports/zoo_chain_sweep.json` + `zoo_kcount.json`. "
        "Re-run; do not hand-edit._",
        "",
    ]
    return "\n".join(lines)


def run_bench(args) -> Report:
    """Join the artifacts, state the findings, write the twins."""
    only = getattr(args, "only", None)
    rows = load_rows(
        Path(getattr(args, "sweep", None) or DEFAULT_SWEEP),
        Path(getattr(args, "kcount", None) or DEFAULT_KCOUNT),
    )
    if only:
        keep = {c.strip() for c in only.split(",") if c.strip()}
        rows = [r for r in rows if r.case in keep]

    findings = build_findings(rows)

    out = getattr(args, "out", None)
    stem = getattr(args, "stem", None) or "frontier"
    base = Path(out) if out else DOC_MD.parent
    md_path = base / f"{stem}.md"
    json_path = base / f"{stem}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(rows, findings))
    json_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "suite": "frontier",
                "inputs": {
                    "sweep": str(DEFAULT_SWEEP),
                    "kcount": str(DEFAULT_KCOUNT),
                },
                "regimes": _regime_stats(rows),
                "findings": [
                    {
                        "claim": f.claim,
                        "verdict": f.verdict.value,
                        "headline": f.headline,
                        "metric": f.metric,
                        "value": f.value,
                    }
                    for f in findings
                ],
                "rows": [r.to_dict() for r in rows],
            },
            indent=2,
        )
        + "\n"
    )

    return Report(
        suite="frontier",
        cells=_cells(rows),
        title="The frontier — headroom vs capture",
        summary=(
            "Amdahl-on-dispatch bounds what an equality-preserving "
            "rewrite can win; the delivered body captures part of it. "
            "Compute-bound rows are capped near 1x by construction."
        ),
        findings=findings,
        provenance={
            "rows": len(rows),
            "inputs": [str(DEFAULT_SWEEP), str(DEFAULT_KCOUNT)],
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point (the CLI drives ``run_bench``)."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--kcount", default=None)
    ap.add_argument(
        "--only", default=None, help="comma-separated cases"
    )
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument(
        "--stem",
        default="frontier",
        help="report filename stem (default: frontier)",
    )
    args = ap.parse_args(argv)

    report = run_bench(args)
    for f in report.findings:
        print(f"[{f.verdict.badge}] {f.headline}")
    stem = getattr(args, "stem", None) or "frontier"
    print(f"\nwrote {stem}.md + {stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
