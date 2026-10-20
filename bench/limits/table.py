"""Limits-table emission — ``docs/limits_table.md`` + the JSON side-file.

The Markdown is the human surface — one ``(case, baseline)`` row per
registered opponent, ratio columns in the ``docs/results.md``
convention (``baseline ÷ mlopt``: >1 ahead, ~1 parity, <1 an honest
loss) and a *caveat* column carrying the qualitative flags a bare
ratio hides — measured-only arms, skipped scratch deps, verify
failures, "no carrier claims this term".  The JSON carries the same
rows plus provenance (UTC timestamp, git sha/dirty, torch/device via
``benchkit.env.collect_env``) so a number is never quoted without
the context that produced it.  Both render from one
``list[CaseLimits]`` — see ``bench/discovery/scorecard.py`` for the
convention this mirrors.
"""

# ruff: noqa: RUF001, RUF002 — math notation in prose.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.benchkit.provenance import with_provenance
from bench.limits import registry
from bench.limits.runner import CaseLimits, LimitRow, _verdict

#: Limits-table payload version — bump on a schema break.
SCHEMA = 1


def _f(v: float | None, fmt: str = "{:,.0f}") -> str:
    return "—" if v is None else fmt.format(v)


def _x(ratio: float | None) -> str:
    """Render a ratio in the results-doc "×" convention."""
    return "—" if ratio is None else f"{ratio:.2f}×"


def _ms(s: float | None) -> str:
    return "—" if s is None else f"{s * 1e3:.3f}"


def _row_record(row: LimitRow) -> dict[str, Any]:
    """One ``(case, baseline)`` pair as a JSON-able record."""
    return {
        "case": row.case,
        "category": row.category,
        "baseline": row.baseline,
        "kind": row.kind,
        "axis": row.axis,
        "available": row.available,
        "mlopt_cost": row.mlopt_cost,
        "baseline_cost": row.baseline_cost,
        "ratio": row.ratio,
        "mlopt_s": row.mlopt_s,
        "baseline_s": row.baseline_s,
        "measured_ratio": row.measured_ratio,
        "verify_passed": row.verify_passed,
        "verify_max_abs": row.verify_max_abs,
        "peak_mib": row.peak_mib,
        "verdict": _verdict(row.measured_ratio or row.ratio),
        "caveat": row.caveat,
    }


def _case_record(cl: CaseLimits) -> dict[str, Any]:
    return {
        "name": cl.case.name,
        "title": cl.case.title,
        "category": cl.case.category,
        "citation": cl.case.citation,
        "axis": cl.case.cost,
        "cost_before": cl.cost_before,
        "cost_after": cl.cost_after,
        "mlopt_s": cl.mlopt_s,
        "verified": cl.verified,
        "verify_max_abs": cl.verify_max_abs,
        "error": cl.error,
        "detail": cl.detail,
        "baselines": [_row_record(r) for r in cl.rows],
    }


def render_markdown(
    out: list[CaseLimits], *, lane: str, env: dict | None = None
) -> str:
    """The human limits table — one row per (case, baseline)."""
    env = env or {}
    measured = lane == "full"
    lines = [
        "# Limits table — the delivered form vs the registered baselines",
        "",
        "Every registered opponent in `bench/limits/` crossed with "
        "the named-optimization corpus (`bench/discovery/cases/`).  "
        "**ratio** is `baseline ÷ mlopt` — the results-doc "
        "convention: >1× means the delivered form is ahead, ~1× is "
        "parity (a rediscovery of the known impl), <1× is an honest "
        "loss.  *caveat* carries what the number hides: measured-only "
        "arms, scratch-dep skips, verify failures.",
        "",
        f"- lane: `{lane}` "
        f"({'measured wall + verify' if measured else 'symbolic dag_cost'})",
        f"- date: {env.get('timestamp_utc', '—')} · "
        f"git: `{env.get('git_sha', 'unknown')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''} · "
        f"torch {env.get('torch', '?')} on {env.get('device', '?')}",
        "",
    ]
    head = (
        "| case | category | baseline | kind | mlopt | baseline | "
        "ratio | verdict | caveat |"
        if not measured
        else "| case | category | baseline | kind | mlopt ms | "
        "baseline ms | ratio | verify | caveat |"
    )
    lines += [head, "|---|---|---|---|---|---|---|---|---|"]
    for cl in out:
        if cl.error:
            lines.append(
                f"| `{cl.case.name}` | {cl.case.category} | — | — | "
                f"error — {cl.error} | | | | |"
            )
            continue
        for r in cl.rows:
            if measured:
                verify = {
                    True: "yes",
                    False: "no",
                    None: "—",
                }[r.verify_passed]
                lines.append(
                    f"| `{r.case}` | {r.category} | {r.baseline} "
                    f"| {r.kind} | {_ms(r.mlopt_s)} | "
                    f"{_ms(r.baseline_s)} | {_x(r.measured_ratio)} "
                    f"| {verify} | {r.caveat or '—'} |"
                )
            else:
                lines.append(
                    f"| `{r.case}` | {r.category} | {r.baseline} "
                    f"| {r.kind} | {_f(r.mlopt_cost)} | "
                    f"{_f(r.baseline_cost)} | {_x(r.ratio)} "
                    f"| {_verdict(r.ratio)} | {r.caveat or '—'} |"
                )
    lines += ["", "## The registered roster", ""]
    for name in registry.names():
        s = registry.spec(name)
        lines.append(
            f"- `{s.name}` ({s.kind}) — {s.title or s.name}"
            + (
                f" · cases: {', '.join(s.cases)}"
                if s.cases
                else " · all cases"
            )
            + (f" · record: {s.record}" if s.record else "")
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    out: list[CaseLimits],
    *,
    lane: str,
    md_path: Path,
    json_path: Path,
    device: str = "cpu",
    seed: Any = None,
) -> dict[str, Any]:
    """Write the Markdown + JSON limits table; return the payload."""
    payload = with_provenance(
        {
            "schema": SCHEMA,
            "lane": lane,
            "cases": [_case_record(cl) for cl in out],
        },
        seed=seed,
        device=device,
    )
    env = payload["env"]
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(out, lane=lane, env=env))
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return payload
