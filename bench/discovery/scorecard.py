"""Scorecard emission — ``docs/scorecard.md`` + the JSON side-file.

The Markdown table is the human surface; the JSON carries the same
rows plus provenance (UTC timestamp, git sha/dirty, torch/device via
``benchkit.env.collect_env``) so a number is never quoted without the
context that produced it.  Both render from one ``list[CaseResult]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.benchkit.provenance import with_provenance
from bench.discovery.runner import CaseResult

#: Scorecard payload version — bump on a schema break.
SCHEMA = 1


def _fmt_cost(v: float | None) -> str:
    if v is None:
        return "—"
    if v == float("inf"):
        return "∞"
    return f"{v:,.0f}"


def _fmt_delta(before: float | None, after: float | None) -> str:
    if before is None or after is None or not before:
        return "—"
    return f"{(after - before) / before:+.1%}"


def _bool(b: bool | None) -> str:
    return {True: "yes", False: "no", None: "—"}[b]


def _best_floor(row: CaseResult) -> tuple[str, float | None]:
    """The best (lowest-cost) registered baseline arm."""
    arms = [a for a in row.baselines if a.cost is not None]
    if not arms:
        return "—", None
    best = min(arms, key=lambda a: a.cost)
    return best.name, best.cost


def _row_record(row: CaseResult) -> dict[str, Any]:
    """One case as a JSON-able record."""
    best_name, best_cost = _best_floor(row)
    return {
        "name": row.name,
        "title": row.title,
        "category": row.category,
        "citation": row.citation,
        "found": row.found,
        "picked": row.picked,
        "certified": row.certified,
        "verified": row.verified,
        "verify_max_abs": row.verify_max_abs,
        "axis": row.axis,
        "cost_before": row.cost_before,
        "cost_after": row.cost_after,
        "best_baseline": best_name,
        "best_baseline_cost": best_cost,
        "beat_best": row.beat_best,
        "beat_best_measured": row.beat_best_measured,
        "signals_seen": row.signals_seen,
        "error": row.error,
        "detail": row.detail,
        "baselines": [
            {
                "name": a.name,
                "kind": a.kind,
                "available": a.available,
                "cost": a.cost,
                "median_s": a.median_s,
                "iqr_s": a.iqr_s,
                "verify_passed": a.verify_passed,
                "verify_max_abs": a.verify_max_abs,
                "note": a.note,
            }
            for a in row.baselines
        ],
    }


def render_markdown(
    rows: list[CaseResult], *, lane: str, env: dict | None = None
) -> str:
    """The human scorecard — one row per named optimization."""
    env = env or {}
    lines = [
        "# Discovery scorecard",
        "",
        "Can the engine derive *named* optimizations from the "
        "literature and real compilers — and can it beat the "
        "registered floor?  Each row is one case in "
        "`bench/discovery/cases/`; **found** = the expect spec held "
        "(signal observed + extraction contract), **cert** = the "
        "proof-carrying certificate replays standalone, **beat** = "
        "the certified form is *strictly below* the best registered "
        "baseline on the case's axis (a tie is a rediscovery, not a "
        "find).",
        "",
        f"- lane: `{lane}` "
        f"({'measured wall + verify' if lane == 'full' else 'symbolic dag_cost'})",
        f"- date: {env.get('timestamp_utc', '—')} · "
        f"git: `{env.get('git_sha', 'unknown')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''} · "
        f"torch {env.get('torch', '?')} on {env.get('device', '?')}",
        "",
        "| case | category | found | cert | cost (axis) | "
        "before → after | best baseline | beat |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        best_name, best_cost = _best_floor(r)
        best = f"{best_name} {_fmt_cost(best_cost)}"
        if r.error:
            lines.append(
                f"| `{r.name}` | {r.category} | error — {r.error} "
                "| | | | | |"
            )
            continue
        delta = _fmt_delta(r.cost_before, r.cost_after)
        lines.append(
            f"| `{r.name}` | {r.category} | {_bool(r.found)} "
            f"| {_bool(r.certified)} | {r.axis} | "
            f"{_fmt_cost(r.cost_before)} → {_fmt_cost(r.cost_after)} "
            f"({delta}) | {best} | {_bool(r.beat_best)} |"
        )
    if lane == "full":
        lines += [
            "",
            "## Measured (full lane)",
            "",
            "| case | verified | max\\|Δ\\| | mlopt ms | "
            "best baseline ms | beat (measured) |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            if r.error:
                continue
            best_ms = min(
                (
                    a.median_s
                    for a in r.baselines
                    if a.median_s is not None
                ),
                default=None,
            )
            mlopt_ms = r.detail.get("mlopt_ms")
            lines.append(
                f"| `{r.name}` | {_bool(r.verified)} | "
                f"{r.verify_max_abs if r.verify_max_abs is not None else '—'} | "
                f"{f'{mlopt_ms:.3f}' if mlopt_ms is not None else '—'} | "
                f"{f'{best_ms * 1e3:.3f}' if best_ms is not None else '—'} | "
                f"{_bool(r.beat_best_measured)} |"
            )
    lines += [
        "",
        "## Per-case detail",
        "",
    ]
    for r in rows:
        lines += [
            f"### `{r.name}` — {r.title}",
            "",
            f"{r.citation}",
            "",
            f"- signals: {', '.join(r.signals_seen) or 'none'}",
            "- baselines: "
            + "; ".join(
                f"{a.name} ({a.kind})"
                + (
                    f" {_fmt_cost(a.cost)}"
                    if a.cost is not None
                    else (" unavailable" if not a.available else "")
                )
                + (f" — {a.note}" if a.note else "")
                for a in r.baselines
            ),
            "",
        ]
    return "\n".join(lines) + "\n"


def write_outputs(
    rows: list[CaseResult],
    *,
    lane: str,
    md_path: Path,
    json_path: Path,
    device: str = "cpu",
    seed: Any = None,
) -> dict[str, Any]:
    """Write the Markdown + JSON scorecard; return the payload."""
    payload = with_provenance(
        {
            "schema": SCHEMA,
            "lane": lane,
            "cases": [_row_record(r) for r in rows],
        },
        seed=seed,
        device=device,
    )
    env = payload["env"]
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(rows, lane=lane, env=env))
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return payload
