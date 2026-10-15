"""Append-only run ledger + baseline comparison.

Every run appends one record per suite (env + cells + findings) to
``bench/results/ledger.jsonl``.  ``load`` reads it back as a polars
frame; ``latest``/``baseline`` pick records for the ``compare`` and
``gate`` commands.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.benchkit.report import Report

DEFAULT_LEDGER = Path("bench/results/ledger.jsonl")


def append(report: Report, path: str | Path = DEFAULT_LEDGER) -> Path:
    """Append a report to the ledger; returns the ledger path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = report.to_dict()
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return path


def load(path: str | Path = DEFAULT_LEDGER) -> list[dict]:
    """Read the ledger as a list of run records (one dict per run).

    Records are document-shaped — per-suite ``cells``/``fires``
    payloads differ in structure — so they are kept as dicts rather
    than forced into a polars frame (which raises on heterogeneous
    nested columns).
    """
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def latest(report_suite: str, path: str | Path = DEFAULT_LEDGER):
    """The most recent ledger record for ``report_suite`` (or ``None``)."""
    hits = [r for r in load(path) if r.get("suite") == report_suite]
    return hits[-1] if hits else None


def write_baseline(report: Report, outdir: str | Path) -> Path:
    """Pin a report as the golden baseline for its suite.

    Writes the *baseline projection* (``Report.to_baseline``) — the
    canonical payload minus the live-only spread — so a pinned
    reference carries exactly the quantity ``compare`` / ``gate``
    reads and nothing the timing contract can later invalidate.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{report.suite}.json"
    report.to_baseline(path)
    return path
