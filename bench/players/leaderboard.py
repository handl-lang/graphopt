"""Leaderboard emission — ``docs/leaderboard_players.md`` + JSON.

The Markdown is the human surface — one table per board, a *notes*
column carrying the qualitative footnotes (drain artifacts, stalled
stops, unsupported boards, factory failures) the repo's results-doc
culture insists on; the JSON carries the same rows plus provenance
(UTC timestamp, git sha/dirty, device, the protocol parameters) so a
number is never quoted without the context that produced it.  Both
render from one ``list[Row]`` — see ``bench/discovery/scorecard.py``
for the convention this mirrors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.benchkit.provenance import with_provenance
from bench.players.runner import Row

#: Leaderboard payload version — bump on a schema break.
SCHEMA = 1


def _f(v: float | None, fmt: str = "{:.4f}") -> str:
    """Format a metric; ``—`` when the board does not report it."""
    return "—" if v is None else fmt.format(v)


def _row_record(row: Row) -> dict[str, Any]:
    """One row as a JSON-able record."""
    return row.as_record()


def _board_table(board: str, rows: list[Row]) -> list[str]:
    """Render one board's rows as a Markdown table."""
    head = (
        "| player | kind | reward/ep | evals/ep | useful/eval | "
        "rank AP | cost | ratio | wall s | notes |"
    )
    lines = [
        f"## `{board}`",
        "",
        head,
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cost = "—" if r.cost is None else f"{r.cost:,.2f}"
        ratio = "—" if r.ratio is None else f"{r.ratio:.3f}"
        lines.append(
            f"| `{r.player}` | {r.kind} | {_f(r.reward_ep)} "
            f"| {_f(r.evals_ep, '{:.2f}')} "
            f"| {_f(r.useful_per_eval)} | {_f(r.ap)} | {cost} "
            f"| {ratio} | {r.wall_s:.2f} | {r.notes or '—'} |"
        )
    return lines


def render_markdown(
    rows: list[Row],
    *,
    env: dict | None = None,
    protocol: dict | None = None,
    meta: dict | None = None,
) -> str:
    """The human leaderboard — one table per board."""
    env = env or {}
    protocol = protocol or {}
    meta = meta or {}
    lines = [
        "# Player leaderboard",
        "",
        "Every registered `players/` module on every board it can "
        "play, under one paired protocol: identical held-out cases, "
        "identical memoized oracle verdicts on `open` (per-candidate "
        "seeds replayed to every arm — see "
        "`bench/drivers/policy/policy_baseline.py`), identical contraction instances "
        "scored by one cost model.  **ratio** is the contraction "
        "board's cost/baseline figure; the *notes* column carries "
        "the qualitative flags (drain artifacts, premature stops, "
        "unsupported boards) that the numbers alone hide.",
        "",
        f"- date: {env.get('timestamp_utc', '—')} · "
        f"git: `{env.get('git_sha', 'unknown')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''} · "
        f"torch {env.get('torch', '?')} on {env.get('device', '?')} · "
        f"seed {protocol.get('seed', '?')}",
    ]
    for key, val in protocol.items():
        if key != "seed":
            lines.append(f"- {key}: `{val}`")
    for key, val in meta.items():
        lines.append(f"- {key}: `{val}`")
    lines.append("")
    boards = []
    for r in rows:
        if r.board not in boards:
            boards.append(r.board)
    for b in boards:
        lines += _board_table(b, [r for r in rows if r.board == b])
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(
    rows: list[Row],
    *,
    md_path: Path,
    json_path: Path,
    protocol: dict | None = None,
    meta: dict | None = None,
    device: str = "cpu",
    seed: Any = None,
) -> dict[str, Any]:
    """Write the Markdown + JSON leaderboard; return the payload."""
    if seed is None:
        seed = (protocol or {}).get("seed")
    payload = with_provenance(
        {
            "schema": SCHEMA,
            "protocol": protocol or {},
            "meta": meta or {},
            "rows": [_row_record(r) for r in rows],
        },
        seed=seed,
        device=device,
    )
    env = payload["env"]
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_markdown(rows, env=env, protocol=protocol, meta=meta)
    )
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return payload
