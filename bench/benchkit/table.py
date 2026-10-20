"""Pin-table support — docs-table payloads into the golden flow.

The docs tables (``scorecard`` / ``leaderboard_players`` /
``limits_table`` / ``matrix``) are not timing reports: their rows are
ratios, verdicts and counts, not medians.  This module projects a
table's rows into the same *pinnable record* shape the ledger carries
— one cell per row, a ``{metric: value}`` map — plus a ``directions``
table saying which way each metric is *bad* (``"up"`` = a rise is a
regression, e.g. cost; ``"down"`` = a drop is, e.g. a ratio or a
reward).  The records append to the same ``ledger.jsonl`` and pin to
the same ``baselines/<suite>.json``; ``compare_baseline`` /
``expectation_gaps`` in :mod:`bench.benchkit.compare` understand the
``kind: "table"`` payload — the golden flow extends to the tables
without forking it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bench.benchkit.jsonl import append_jsonl
from bench.benchkit.ledger import DEFAULT_LEDGER
from bench.benchkit.provenance import with_provenance

#: Table-record payload version — bump on a schema break.
SCHEMA = 1

#: Legal direction values — the way a metric is *bad*.
DIRECTIONS = ("up", "down")


def table_record(
    suite: str,
    cells: Iterable[Mapping[str, Any]],
    *,
    directions: Mapping[str, str],
    findings: Iterable[Mapping[str, Any]] = (),
    env: dict | None = None,
    provenance: dict | None = None,
    seed: Any = None,
    device: str | None = None,
) -> dict:
    """Build the pinnable table record — ledger and baseline share it.

    *cells* yields ``{"name": str, "metrics": {metric: value}}`` —
    non-finite and absent values are dropped per metric, and a cell
    with no surviving metrics is dropped whole (an honest row beats a
    vacuous comparison).  *directions* is ``{metric: "up"|"down"}`` —
    which way is bad; a metric missing from it compares as ``"up"``.
    *findings* are the same ``{claim, verdict, headline, ...}`` dicts
    a ``Report`` emits — the pinned-baseline contract requires a
    stated verdict.  The record is stamped through
    :func:`bench.benchkit.provenance.with_provenance`, so the
    date/sha/device/seed contract is enforced, not conventional.
    """
    for m, d in directions.items():
        if d not in DIRECTIONS:
            raise ValueError(
                f"direction of metric {m!r} must be one of "
                f"{DIRECTIONS} — got {d!r}"
            )
    out_cells = []
    for cell in cells:
        metrics = {
            k: float(v)
            for k, v in dict(cell.get("metrics", {})).items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        }
        if metrics:
            out_cells.append(
                {"name": str(cell["name"]), "metrics": metrics}
            )
    return with_provenance(
        {
            "schema": SCHEMA,
            "kind": "table",
            "suite": suite,
            "provenance": provenance or {},
            "directions": dict(directions),
            "findings": [dict(f) for f in findings],
            "cells": out_cells,
        },
        seed=seed,
        device=device,
        env=env,
    )


def append_record(
    record: dict, path: str | Path = DEFAULT_LEDGER
) -> Path:
    """Append a table record to the ledger; returns the ledger path."""
    return append_jsonl(path, record)


def write_table_baseline(record: dict, outdir: str | Path) -> Path:
    """Pin a table record as the golden baseline for its suite.

    Unlike ``Report.to_baseline`` there is no live spread to project
    away — the record's cells carry exactly the metrics
    ``compare_baseline`` gates on — so the pinned file is the record
    verbatim.
    """
    import json

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{record['suite']}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path
