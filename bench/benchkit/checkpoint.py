"""Checkpoint/resume store for the matrix + leaderboard lanes.

The lanes persist each finished cell at the boundary where it is
produced — a ``(board, player, referee)`` matrix cell, a
``(board, player)`` leaderboard row, a ``(scale, instance)``
contraction accumulation — so a crash costs one cell, not the run
(plan 0032 phase 2b).

The file is append-only JSONL::

    {"_manifest": true, "lane": ..., "protocol": {...},
     "provenance": {git_sha, git_dirty, device}, ...}
    {"key": ["meta", "playbook", "naive"], "record": {...}, "ts": ...}
    ...

* The manifest makes resume **verifiable**: ``protocol`` must match
  exactly for entries to be reused (different seed/knobs/selection is
  a different run — a mismatch warns and restarts the file); a
  ``git_sha`` drift warns but still reuses (a dirty-tree resume is a
  common crash-recovery shape).
* Appends go through :func:`bench.benchkit.jsonl.append_jsonl` — one
  ``O_APPEND`` ``os.write``, so parallel lanes sharing a file cannot
  interleave a record.
* :meth:`open` tolerates a torn last line — the crash's unfinished
  cell — via :func:`bench.benchkit.jsonl.read_jsonl`.

Checkpoints are scratch state, not results: they live under
``bench/results/checkpoints/`` (gitignored).
"""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.benchkit.env import collect_env
from bench.benchkit.jsonl import append_jsonl, read_jsonl

#: Default checkpoint directory — inside the gitignored results tree.
DEFAULT_DIR = Path("bench/results/checkpoints")


def new_manifest(
    lane: str,
    protocol: Mapping[str, Any],
    *,
    device: str | None = None,
    env: dict | None = None,
) -> dict[str, Any]:
    """Build a run manifest: protocol + provenance + run identity."""
    e = env or collect_env(device)
    return {
        "_manifest": True,
        "lane": lane,
        "run_id": uuid.uuid4().hex[:12],
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": dict(protocol),
        "provenance": {
            "git_sha": e.get("git_sha", "unknown"),
            "git_dirty": e.get("git_dirty", False),
            "device": e.get("device", "?"),
        },
    }


def _default_warn(msg: str) -> None:
    print(msg, file=sys.stderr)


class Checkpoint:
    """A resume store for one lane run — see the module docstring."""

    def __init__(
        self,
        path: Path,
        manifest: dict[str, Any],
        entries: dict[tuple, dict] | None = None,
    ):
        self.path = Path(path)
        self.manifest = manifest
        self._entries: dict[tuple, dict] = dict(entries or {})

    @classmethod
    def open(
        cls,
        path: str | Path,
        manifest: dict[str, Any],
        *,
        env: dict | None = None,
        warn: Callable[[str], None] = _default_warn,
    ) -> Checkpoint:
        """Open *path* for resume; create it (with manifest) if absent.

        Reuses the stored entries when the stored manifest's
        ``protocol`` equals *manifest*'s.  A protocol mismatch or a
        missing manifest restarts the file (with a warning — the old
        progress is discarded, never silently merged).  A git-sha
        drift warns but reuses.
        """
        path = Path(path)
        if path.exists() and path.stat().st_size:
            records = read_jsonl(path, tolerate_torn_tail=True)
            stored = (
                records[0]
                if records and records[0].get("_manifest")
                else None
            )
            # The stored manifest came back through JSON — a tuple in
            # the caller's protocol reads as a list.  Normalize before
            # comparing so value-equal protocols match.
            want = json.loads(
                json.dumps(manifest.get("protocol"), default=str)
            )
            if stored is not None and stored.get("protocol") == want:
                cur = (env or collect_env()).get("git_sha")
                old = stored.get("provenance", {}).get("git_sha")
                if cur and old and cur != old:
                    warn(
                        f"checkpoint {path}: git sha moved "
                        f"{old} -> {cur} — reusing cells anyway"
                    )
                entries = {
                    tuple(r["key"]): r["record"]
                    for r in records[1:]
                    if "key" in r and "record" in r
                }
                return cls(path, stored, entries)
            warn(
                f"checkpoint {path}: protocol differs "
                "— restarting the checkpoint"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_line(manifest))
        return cls(path, manifest)

    # -- cell access ------------------------------------------------------

    def done(self, key: tuple) -> bool:
        """True when *key* has a recorded (complete) cell."""
        return tuple(key) in self._entries

    def get(self, key: tuple) -> dict | None:
        """The stored record for *key*, or ``None``."""
        return self._entries.get(tuple(key))

    def record(self, key: tuple, record: Mapping[str, Any]) -> None:
        """Persist one finished cell — atomic, crash-safe."""
        key = tuple(key)
        append_jsonl(
            self.path,
            {
                "key": list(key),
                "record": dict(record),
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        self._entries[key] = dict(record)

    @property
    def entries(self) -> dict[tuple, dict]:
        """The live ``{key: record}`` map (stored + recorded)."""
        return self._entries


def json_line(record: Mapping[str, Any]) -> str:
    """One JSON line — the manifest write format."""
    return json.dumps(record, default=str) + "\n"
