"""JSONL persistence — atomic append + torn-tail-tolerant read.

Two primitives shared by the run ledger (``benchkit.ledger`` /
``benchkit.table``) and the resume checkpoints
(``benchkit.checkpoint``):

* :func:`append_jsonl` — a single ``os.write`` under ``O_APPEND``.
  POSIX makes the seek-to-end+write atomic for regular files, so
  concurrent appenders (parallel workers, a crashed-and-resumed lane
  alongside a live one) can never interleave a record — no lockfile
  needed, which is why ``filelock`` was not adopted here.
* :func:`read_jsonl` — with ``tolerate_torn_tail`` a truncated final
  line (a crash mid-append) is dropped, never fatal: append is
  write-once, so the torn record is the run's unfinished cell.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    """Atomically append *record* as one JSON line; returns *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return path


def read_jsonl(
    path: str | Path, *, tolerate_torn_tail: bool = False
) -> list[dict[str, Any]]:
    """Read a JSONL file as a list of records ([] when absent).

    With ``tolerate_torn_tail`` a non-JSON line is dropped only when it
    is the file's last line — the signature of a crash mid-append;
    corruption anywhere else still raises.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_torn_tail and i == len(lines) - 1:
                break
            raise
    return out
