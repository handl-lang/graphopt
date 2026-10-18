#!/usr/bin/env python3
"""Show which Cline CLI sessions still need attention.

Cline stores one directory per session under ``~/.cline/data/sessions``;
the ``<id>.json`` file records the process lifecycle while
``<id>.messages.json`` holds the transcript.  The lifecycle ``status``
alone is misleading -- a killed turn is often recorded as ``completed``
with exit code 0 -- so this script classifies the transcript tail:

``live``
    The process is still alive.
``needs-you``
    The last assistant turn ends with a question or an offer to
    proceed, so the session is waiting for an answer.
``cut``
    The transcript stops mid-turn (last message is a user turn or a
    tool result, or the assistant's last block is an unanswered tool
    call).
``error``
    The last assistant text is a transport/quota error.
``done``
    The last assistant turn is a final answer with no pending work.

Sub-agent and team-task sessions (ids containing ``__``) are skipped:
their parent session owns them.  By default only sessions that are
not done are shown, newest first.  ``--resume`` prints the commands
needed to continue them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path.home() / ".cline" / "data"
NOT_DONE = {"live", "needs-you", "cut", "error"}

QUESTION_RE = re.compile(
    r"(?:\?|would you like|want me to|should i|shall i|"
    r"say the word|let me know|tell me (?:which|if|whether)|"
    r"your call)\s*$",
    re.IGNORECASE,
)
ERROR_RE = re.compile(
    r"(error 429|daily free limit|rate limit|quota exceeded|"
    r"getaddrinfo|etimedout|econn\w*|api\.cline|http 5\d\d|"
    r"service unavailable|internal server error)",
    re.IGNORECASE,
)
GREETING_RE = re.compile(
    r"(how can i help|what would you like|what can i (?:do|help))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Session:
    """One Cline session and the classification of its tail."""

    session_id: str
    state: str
    started_at: str
    cwd: str
    title: str

    @property
    def age(self) -> str:
        """Return the elapsed time since the session started."""
        return _age(self.started_at)

    @property
    def resume(self) -> str:
        """Return a shell command that resumes the session."""
        return (
            f"cd {_quote(self.cwd)} && cline --id {self.session_id} -i"
        )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty mapping on failure."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a message's content blocks, accepting plain strings."""
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _text_of(message: dict[str, Any]) -> str:
    """Join the text blocks of one message."""
    parts = [
        block.get("text") or ""
        for block in _blocks(message)
        if block.get("type") == "text"
    ]
    return "\n".join(parts).strip()


def _has_tool_use(message: dict[str, Any]) -> bool:
    """Return whether a message asks for a tool call."""
    return any(b.get("type") == "tool_use" for b in _blocks(message))


def _pid_alive(pid: object) -> bool:
    """Return whether a PID is still running."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _age(started_at: str) -> str:
    """Format the elapsed time since an ISO timestamp, compactly."""
    try:
        then = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return "?"
    elapsed = datetime.now(UTC) - then
    seconds = max(int(elapsed.total_seconds()), 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86_400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86_400}d"


def _title(meta: dict[str, Any]) -> str:
    """Return a human label for the session."""
    metadata = meta.get("metadata") or {}
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return " ".join(title.split())
    prompt = re.sub(r"<[^>]+>", " ", str(meta.get("prompt") or ""))
    return " ".join(prompt.split())


def classify(
    messages: list[dict[str, Any]], meta: dict[str, Any]
) -> str:
    """Classify a session from its metadata and transcript tail."""
    if not meta.get("ended_at") and _pid_alive(meta.get("pid")):
        return "live"
    if not messages:
        return "empty"
    last = messages[-1]
    if last.get("role") != "assistant" or _has_tool_use(last):
        return "cut"
    tail = " ".join(_text_of(last).split())[-200:]
    if not tail:
        return "cut"
    if GREETING_RE.search(tail):
        return "done"
    if QUESTION_RE.search(tail):
        return "needs-you"
    if ERROR_RE.search(tail):
        return "error"
    return "done"


def load_sessions(data_dir: Path) -> list[Session]:
    """Load and classify every top-level session under ``data_dir``."""
    sessions_dir = data_dir / "sessions"
    if not sessions_dir.is_dir():
        print(f"no Cline sessions at {sessions_dir}", file=sys.stderr)
        return []
    sessions: list[Session] = []
    for directory in sorted(sessions_dir.iterdir()):
        if not directory.is_dir():
            continue
        session_id = directory.name
        if "__" in session_id:
            continue
        meta = _read_json(directory / f"{session_id}.json")
        messages = _read_json(
            directory / f"{session_id}.messages.json"
        ).get("messages")
        if not isinstance(messages, list):
            messages = []
        sessions.append(
            Session(
                session_id=session_id,
                state=classify(messages, meta),
                started_at=str(meta.get("started_at") or ""),
                cwd=str(meta.get("cwd") or "?"),
                title=_title(meta),
            )
        )
    sessions.sort(key=lambda row: row.started_at, reverse=True)
    return sessions


def _shorten(text: str, width: int) -> str:
    """Clamp text to ``width`` characters."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _quote(path: str) -> str:
    """Quote a path for a POSIX shell."""
    home = str(Path.home())
    shown = "~" + path[len(home) :] if path.startswith(home) else path
    if re.search(r"[\s'\"$`\\]", shown):
        return "'" + shown.replace("'", "'\\''") + "'"
    return shown


def _print_table(rows: list[Session], total: int, label: str) -> None:
    """Print the compact session table."""
    header = (
        f"{'STATE':<10} {'AGE':>4}  {'SESSION':<20} {'DIR':<26} TITLE"
    )
    print(f"{len(rows)} {label} of {total} sessions")
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.state:<10} {row.age:>4}  "
            f"{_shorten(row.session_id, 20):<20} "
            f"{_shorten(_quote(row.cwd), 26):<26} "
            f"{_shorten(row.title, 60)}"
        )
    print()
    print(
        "resume: cd <dir> && cline --id <session> -i"
        "   (or rerun with --resume)"
    )


def _print_resume(rows: list[Session]) -> None:
    """Print ready-to-run resume commands."""
    for row in rows:
        print(f"# {row.state}: {_shorten(row.title, 70)}")
        print(row.resume)


def main(argv: list[str] | None = None) -> int:
    """Run the session report."""
    parser = argparse.ArgumentParser(
        description="List Cline CLI sessions that are not done."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Cline data directory (default: ~/.cline/data)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include done and empty sessions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="maximum rows to print (default: 100)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a table",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="print resume commands instead of a table",
    )
    args = parser.parse_args(argv)

    sessions = load_sessions(args.data_dir)
    rows = (
        sessions
        if args.all
        else [row for row in sessions if row.state in NOT_DONE]
    )
    rows = rows[: args.limit]

    if args.json:
        payload = [
            {
                "session_id": row.session_id,
                "state": row.state,
                "age": row.age,
                "started_at": row.started_at,
                "cwd": row.cwd,
                "title": row.title,
                "resume": row.resume,
            }
            for row in rows
        ]
        print(json.dumps(payload, indent=2))
    elif args.resume:
        _print_resume(rows)
    else:
        _print_table(
            rows,
            len(sessions),
            "shown" if args.all else "not done",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
