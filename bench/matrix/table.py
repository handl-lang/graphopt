# ruff: noqa: RUF001 — math notation in prose.

"""Matrix emission — ``docs/matrix.md`` + the JSON side-file.

The Markdown is the human surface — one ``(player, referee)`` row per
board with admit counts, mean scored value and the cell's notes; a
*coverage* section marks every unrun cell with its reason (the
designed-subset contract — never a silent drop); a *Pareto* note per
board names the non-dominated arms.  The JSON carries the same cells
plus provenance.  Both render from one ``list[Cell]`` — the
scorecard/leaderboard convention.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.benchkit.env import collect_env
from bench.matrix.runner import Cell

#: Matrix payload version — bump on a schema break.
SCHEMA = 1

#: The Pareto objectives — (cell attr, higher-is-better).
_OBJECTIVES: tuple[tuple[str, bool], ...] = (
    ("admit_rate", True),
    ("reward_ep", True),
    ("wall_s", False),
)


def _cell_record(c: Cell) -> dict[str, Any]:
    """One cell as a JSON-able record."""
    return {
        "board": c.board,
        "player": c.player,
        "referee": c.referee,
        "run": c.run,
        "reason": c.reason,
        "claims": c.claims,
        "admitted": c.admitted,
        "admit_rate": c.admit_rate,
        "refuted": c.refuted,
        "value": c.value,
        "reward_ep": c.reward_ep,
        "evals_ep": c.evals_ep,
        "wall_s": c.wall_s,
        "notes": c.notes,
    }


def pareto_front(cells: list[Cell]) -> list[Cell]:
    """The non-dominated run cells under ``_OBJECTIVES``.

    A cell dominates another when it is at least as good on every
    objective and strictly better on one; cells missing a metric are
    dominated by anything that has it.  Empty-objective cells (no
    claims, no metrics) are dominated whenever another cell ran.
    """
    ran = [c for c in cells if c.run and c.claims]
    front = []
    for c in ran:
        dominated = False
        for other in ran:
            if other is c:
                continue
            better_on = False
            worse_on = False
            for attr, hi in _OBJECTIVES:
                a = getattr(other, attr)
                b = getattr(c, attr)
                if a is None or b is None:
                    continue
                if (a > b) == hi and a != b:
                    better_on = True
                elif (a < b) == hi and a != b:
                    worse_on = True
            if better_on and not worse_on:
                dominated = True
                break
        if not dominated:
            front.append(c)
    return front


def render_markdown(
    cells: list[Cell],
    *,
    coverage: list[Cell],
    env: dict | None = None,
    protocol: dict | None = None,
) -> str:
    """The human matrix — one table per board + coverage + Pareto."""
    env = env or {}
    protocol = protocol or {}
    lines = [
        "# Eval matrix — players × boards × referees",
        "",
        "Each cell answers: *the claims this player produces on this "
        "board — which survive this referee?*  Episode boards yield "
        "extraction claims (input ≡ delivered); `open` yields the "
        "candidates the player chose.  **admit** = claims the referee "
        "cleared; **value** = the mean `ScoringSpec` blend over "
        "measured claims (fraction of the launch bar — lower is "
        "better); **refuted** = picks the board's oracle called "
        "`useful` that the referee refuses — the strictness ablation "
        "as a row.",
        "",
        f"- date: {env.get('timestamp_utc', '—')} · "
        f"git: `{env.get('git_sha', 'unknown')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''} · "
        f"device {env.get('device', '?')} · seed {protocol.get('seed', '?')}",
    ]
    for key, val in protocol.items():
        if key != "seed":
            lines.append(f"- {key}: `{val}`")
    lines.append("")
    boards = []
    for c in cells:
        if c.board not in boards:
            boards.append(c.board)
    for board in boards:
        brows = [c for c in cells if c.board == board and c.run]
        lines += [
            f"## `{board}`",
            "",
            "| player | referee | claims | admit | refuted | value | "
            "reward/ep | wall s | notes |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in brows:
            rate = (
                "—" if c.admit_rate is None else f"{c.admit_rate:.2f}"
            )
            value = "—" if c.value is None else f"{c.value:.3f}"
            rew = "—" if c.reward_ep is None else f"{c.reward_ep:.4f}"
            lines.append(
                f"| `{c.player}` | `{c.referee}` | {c.claims} "
                f"| {rate} | {c.refuted} | {value} | {rew} "
                f"| {c.wall_s:.2f} | {c.notes or '—'} |"
            )
        front = pareto_front(brows)
        if front:
            names = ", ".join(
                f"`{c.player}`/`{c.referee}`" for c in front
            )
            lines += [
                "",
                f"**Pareto** (admit-rate ↑, reward ↑, wall ↓): {names}",
            ]
        lines.append("")
    unrun = [c for c in coverage if not c.run]
    if unrun:
        lines += [
            "## Coverage — the unrun cells",
            "",
            "| board | player | referee | reason |",
            "|---|---|---|---|",
        ]
        for c in unrun:
            lines.append(
                f"| `{c.board}` | `{c.player}` | `{c.referee}` "
                f"| {c.reason} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(
    cells: list[Cell],
    *,
    coverage: list[Cell],
    md_path: Path,
    json_path: Path,
    protocol: dict | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Write the Markdown + JSON matrix; return the payload."""
    env = collect_env(device)
    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(
            timespec="seconds"
        ),
        "env": env,
        "protocol": protocol or {},
        "cells": [_cell_record(c) for c in cells],
        "coverage": [_cell_record(c) for c in coverage],
    }
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_markdown(
            cells, coverage=coverage, env=env, protocol=protocol
        )
    )
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return payload
