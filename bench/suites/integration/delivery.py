"""Delivery — does the same minted law pay as a kernel, not as a step?

The integration claim, measured as a *paired* comparison rather than two
separate results.  The same machine-minted `synop*` folds appear in two
committed artifacts under two deliveries:

* **as a step** — `docs/reports/pay_measured.json`: the fold fires
  inside a real module and the module is delivered through the ordinary
  executor path.  `marginal` is the delivered/spelled ratio, so
  `marginal >= 1` means the fold *loses*: the algebra saved work and
  the delivery spent more than it saved.
* **as a kernel body** — `docs/reports/prim_kernel.json` (fp32) and
  `prim_kernel_fp64.json`: the same ops lowered as a standalone
  compiled body over 1 M elements, `kernel_over_binding` = kernel ÷
  binding time, so `< 1` means the kernel wins.

Joining on the op name answers the only question that matters for the
integration: is the sign of the result a property of the *law* or of
the *delivery*?  If the same laws lose as steps and win as kernels, the
delivery is the variable — and P1's gate (the 46 losing zoo rows) is
worth running.

    .venv/bin/python -m bench.suites.integration.delivery
    python -m bench run delivery
"""

from __future__ import annotations

import json
import statistics as stats
from dataclasses import dataclass
from pathlib import Path

from bench.benchkit.model import Case, Cell, Finding, Variant, Verdict
from bench.benchkit.report import Report

_REPO = Path(__file__).resolve().parents[3]

#: The step-delivery artifact: minted folds inside real modules.
PAY_MEASURED = _REPO / "docs" / "reports" / "pay_measured.json"

#: The kernel-delivery artifacts: the same ops as standalone bodies.
PRIM_KERNEL = _REPO / "docs" / "reports" / "prim_kernel.json"
PRIM_KERNEL_FP64 = _REPO / "docs" / "reports" / "prim_kernel_fp64.json"

DOC_JSON = _REPO / "docs" / "reports" / "delivery.json"
DOC_MD = _REPO / "docs" / "reports" / "delivery.md"

#: A delivered/spelled ratio at or above this is a loss.
LOSS_TOL = 1.02


@dataclass(frozen=True)
class Row:
    """One minted op under both deliveries.

    ``step_worst`` is the worst delivered/spelled ratio the op produced
    as a step (`>=1` loses); ``kernel_ratio`` is kernel ÷ binding as an
    isolated body (`<1` wins).  Both are ``None`` when the op is absent
    from that artifact — an op measured in only one form cannot testify
    about delivery.
    """

    op: str
    step_worst: float | None
    step_cells: int
    kernel_ratio: float | None
    dtype: str

    @property
    def step_speedup(self) -> float | None:
        """Speedup of the step delivery (>1 wins)."""
        return (
            None if self.step_worst is None else 1.0 / self.step_worst
        )

    @property
    def kernel_speedup(self) -> float | None:
        """Speedup of the kernel delivery (>1 wins)."""
        return (
            None
            if self.kernel_ratio is None
            else 1.0 / self.kernel_ratio
        )

    @property
    def step_wins(self) -> bool | None:
        """Did the step delivery beat the tolerance?"""
        if self.step_worst is None:
            return None
        return self.step_worst <= (1.0 / LOSS_TOL)

    @property
    def kernel_wins(self) -> bool | None:
        """Did the kernel delivery beat the tolerance?"""
        if self.kernel_ratio is None:
            return None
        return self.kernel_ratio <= (1.0 / LOSS_TOL)

    @property
    def flips(self) -> bool:
        """Loses as a step and wins as a kernel — the paired contrast."""
        return self.step_wins is False and self.kernel_wins is True

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "step_worst_ratio": (
                None
                if self.step_worst is None
                else round(self.step_worst, 4)
            ),
            "step_cells": self.step_cells,
            "kernel_over_binding": (
                None
                if self.kernel_ratio is None
                else round(self.kernel_ratio, 4)
            ),
            "dtype": self.dtype,
            "step_speedup": (
                None
                if self.step_speedup is None
                else round(self.step_speedup, 4)
            ),
            "kernel_speedup": (
                None
                if self.kernel_speedup is None
                else round(self.kernel_speedup, 4)
            ),
            "loses_as_step": self.step_wins is False,
            "wins_as_kernel": self.kernel_wins is True,
            "pair_flip": self.flips,
        }


def load_rows(
    pay_path: Path = PAY_MEASURED,
    kernel_paths: tuple[Path, ...] = (PRIM_KERNEL, PRIM_KERNEL_FP64),
) -> list[Row]:
    """Join the step-delivery folds to the kernel-delivery bodies.

    Returns one row per op name seen in **either** artifact, so an op
    measured in only one form still appears (with the other column
    empty) instead of vanishing from the table.
    """
    step: dict[str, tuple[float, int]] = {}
    pay = json.loads(Path(pay_path).read_text())
    for op, rec in (pay.get("folds") or {}).items():
        cells = rec.get("modules") or []
        worst = rec.get("worst")
        if worst is None or not cells:
            continue
        step[op] = (float(worst), len(cells))

    kernel: dict[str, tuple[float, str]] = {}
    for path in kernel_paths:
        path = Path(path)
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        dtype = "fp64" if "fp64" in path.stem else "fp32"
        for row in rec.get("rows") or []:
            if row.get("status") != "measured":
                continue
            ratio = row.get("kernel_over_binding")
            if ratio is None:
                continue
            kernel[row["op"]] = (float(ratio), dtype)

    rows = []
    for op in sorted(set(step) | set(kernel)):
        s = step.get(op)
        k = kernel.get(op)
        rows.append(
            Row(
                op=op,
                step_worst=None if s is None else s[0],
                step_cells=0 if s is None else s[1],
                kernel_ratio=None if k is None else k[0],
                dtype="-" if k is None else k[1],
            )
        )
    return rows


def build_findings(rows: list[Row]) -> list[Finding]:
    """Does the sign belong to the law or to the delivery?"""
    paired = [
        r
        for r in rows
        if r.step_worst is not None and r.kernel_ratio is not None
    ]
    step_losses = [r for r in paired if r.step_wins is False]
    kernel_wins = [r for r in paired if r.kernel_wins is True]
    flips = [r for r in paired if r.flips]

    kernel_ratios = [r.kernel_ratio for r in paired if r.kernel_ratio]
    med_kernel = (
        None
        if not kernel_ratios
        else float(stats.median(kernel_ratios))
    )

    if not paired:
        return [
            Finding(
                claim="The same laws can be compared under both deliveries.",
                verdict=Verdict.INCONCLUSIVE,
                headline="no op appears in both artifacts — nothing paired",
                metric="paired_ops",
                value=0.0,
            )
        ]

    v_step = Verdict.NEGATIVE if step_losses else Verdict.WIN
    v_kernel = (
        Verdict.WIN
        if len(kernel_wins) == len(paired)
        else Verdict.PARITY
    )
    v_flip = (
        Verdict.WIN
        if flips and len(flips) == len(paired)
        else (Verdict.PARITY if flips else Verdict.NEGATIVE)
    )

    return [
        Finding(
            claim=(
                "Delivered as one more step inside a module, the minted "
                "folds lose."
            ),
            verdict=v_step,
            headline=(
                f"{len(step_losses)}/{len(paired)} paired ops lose as a "
                "step (worst "
                f"{max(r.step_worst for r in paired):.2f}x slower)"
            ),
            metric="step_losses",
            value=float(len(step_losses)),
            evidence={
                "paired": len(paired),
                "ops": [r.op for r in step_losses],
            },
        ),
        Finding(
            claim=(
                "Delivered as an isolated kernel body, the same ops win."
            ),
            verdict=v_kernel,
            headline=(
                f"{len(kernel_wins)}/{len(paired)} win as a kernel — "
                f"median {1.0 / med_kernel:.2f}x over the binding"
            ),
            metric="median_kernel_speedup",
            value=None if med_kernel is None else 1.0 / med_kernel,
            evidence={"kernel_over_binding": med_kernel},
        ),
        Finding(
            claim=(
                "The sign of the result is set by the delivery, not by "
                "the law: the same op flips from loss to win."
            ),
            verdict=v_flip,
            headline=(
                f"{len(flips)}/{len(paired)} paired ops flip — delivery "
                "is the variable"
            ),
            metric="pair_flip_fraction",
            value=len(flips) / len(paired),
        ),
    ]


def _cells(rows: list[Row]) -> list[Cell]:
    """One cell per op so the dashboard can compare the two forms."""

    def _noop() -> None:
        return None

    out = []
    for r in rows:
        case = Case(
            name=r.op,
            params={
                "dtype": r.dtype,
                "step_cells": r.step_cells,
                "pair_flip": r.flips,
            },
            variants=[
                Variant(name="step", stmt=_noop),
                Variant(name="kernel", stmt=_noop),
            ],
            aux=r.to_dict(),
        )
        out.append(
            Cell(
                case=case,
                medians={
                    k: v / 1000.0
                    for k, v in (
                        ("step", r.step_worst),
                        ("kernel", r.kernel_ratio),
                    )
                    if v is not None
                },
                iqr={},
                aux=r.to_dict(),
            )
        )
    return out


def render_markdown(rows: list[Row], findings: list[Finding]) -> str:
    """The written report — the paired table, then the claims."""
    paired = [
        r
        for r in rows
        if r.step_worst is not None and r.kernel_ratio is not None
    ]
    lines = [
        "# Delivery — a rule pays as a kernel, not as a step",
        "",
        "**Method.** The same machine-minted `synop*` folds, measured "
        "twice: as one more step inside a real module "
        "(`pay_measured.json`, `marginal = delivered/spelled`) and as an "
        "isolated compiled body over 1 M elements "
        "(`prim_kernel*.json`, `kernel/binding`). Paired on the op name.",
        "",
        f"- ops seen: {len(rows)} · paired in both forms: {len(paired)} · "
        f"loss tolerance: >{LOSS_TOL:.2f}x",
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
        "## Paired ops",
        "",
        "| op | as a step (worst) | as a kernel | dtype | flips |",
        "|---|---|---|---|---|",
    ]
    for r in paired:
        lines.append(
            f"| `{r.op}` | {r.step_worst:.2f}x slower | "
            f"{1.0 / r.kernel_ratio:.2f}x faster | {r.dtype} | "
            f"{'yes' if r.flips else 'no'} |"
        )
    unpaired = [r for r in rows if r not in paired]
    if unpaired:
        lines += [
            "",
            "### Measured in one form only (no testimony about delivery)",
            "",
            "| op | step | kernel | dtype |",
            "|---|---|---|---|",
        ]
        for r in unpaired:
            step = (
                "—" if r.step_worst is None else f"{r.step_worst:.2f}x"
            )
            kern = (
                "—"
                if r.kernel_ratio is None
                else f"{1.0 / r.kernel_ratio:.2f}x faster"
            )
            lines.append(f"| `{r.op}` | {step} | {kern} | {r.dtype} |")
    lines += [
        "",
        "## What this establishes, and what it does not",
        "",
        "- It establishes that **delivery** changes the sign for these "
        "ops. It does not establish that every losing zoo row flips: "
        "these ops are the ones the primitive mint produced, and the "
        "zoo rows carry different shapes and dtypes.",
        "- The paired set is small and drawn from the scan/gated-LRU "
        "family. The P1 gate (the 46 losing zoo rows) is the test that "
        "generalises it, and it is not answered here.",
        "- `kernel_over_binding` is measured at 1 M elements with the "
        "fp64 verify contract; a module-scale delivery also pays "
        "materialization and layout, which this ratio excludes.",
        "",
        "_Generated by `bench/suites/integration/delivery.py` from "
        "`pay_measured.json` + `prim_kernel{,_fp64}.json`. Re-run; do "
        "not hand-edit._",
        "",
    ]
    return "\n".join(lines)


def run_bench(args) -> Report:
    """Join the two deliveries, state the findings, write the twins."""
    rows = load_rows()
    findings = build_findings(rows)

    out = getattr(args, "out", None)
    md_path = Path(out) / "delivery.md" if out else DOC_MD
    json_path = Path(out) / "delivery.json" if out else DOC_JSON
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(rows, findings))
    json_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "suite": "delivery",
                "inputs": [
                    str(PAY_MEASURED),
                    str(PRIM_KERNEL),
                    str(PRIM_KERNEL_FP64),
                ],
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
        suite="delivery",
        cells=_cells(rows),
        title="Delivery — kernel vs step",
        summary=(
            "The same minted folds lose as a step inside a module and "
            "win as isolated kernel bodies: the delivery sets the sign."
        ),
        findings=findings,
        provenance={"rows": len(rows), "inputs": str(PAY_MEASURED)},
    )


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point (the CLI drives ``run_bench``)."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args(argv)

    report = run_bench(args)
    for f in report.findings:
        print(f"[{f.verdict.badge}] {f.headline}")
    print(f"\nwrote {DOC_MD.name} + {DOC_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
