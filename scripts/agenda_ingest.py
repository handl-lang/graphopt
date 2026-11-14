#!/usr/bin/env python3
"""Ingest the mined research agenda into the myco ledger.

First loop-closure step, with the human audit gate kept explicit.

``bench/drivers/agenda/agenda_miner.py`` emits ranked proposals to
``docs/reports/agenda.jsonl`` (schema: ``kind == "agenda-proposal"``,
``id``, ``title``, ``signal``, ``target``, ``evidence``, ``lever``,
``headroom``, ``confidence``, ``score``, ``rank``).  This script turns
that file into *tracked work items* — ``myco todo add`` entries by
default, ``myco issue new`` with the evidence body under
``--kind issue`` — after validation, dedup, ranking, and capping.

The audit gate stays human: the default mode is a **dry run** that
prints exactly what *would* be created.  ``--execute`` is the only
flag that shells out to ``myco``; nothing here runs research, admits
laws, or touches the measurement surfaces.  Re-running is safe —
ingested proposal ids are recorded in a state file
(``docs/reports/agenda_ingest_state.json``) *and* matched against the
live ledger titles, so a proposal is never filed twice.

Usage::

    # dry run — print the ingest plan, change nothing (default)
    .venv/bin/python scripts/agenda_ingest.py
    .venv/bin/python scripts/agenda_ingest.py --cap 3 --kind issue
    .venv/bin/python scripts/agenda_ingest.py --since 12   # incremental

    # human-approved execute — files the todos/issues in project/
    .venv/bin/python scripts/agenda_ingest.py --execute --cap 5

Ledger items are titled ``agenda <proposal-id>: <title>`` so the id
is greppable for dedup and audit.  ``myco`` verbs run with
``--no-commit`` (the miner's convention): the per-verb commits are
left to ``myco check --fix`` / the human's own commit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_AGENDA = _REPO / "docs" / "reports" / "agenda.jsonl"
_DEFAULT_STATE = _REPO / "docs" / "reports" / "agenda_ingest_state.json"

_CONFIDENCE = {"high", "medium", "low"}

#: A runner takes an argv list plus optional stdin text and returns
#: (rc, stdout, stderr) — injectable so tests never shell out to myco.
Runner = Callable[[list[str], str | None], tuple[int, str, str]]


# ---------------------------------------------------------------------------
#  Load + validate
# ---------------------------------------------------------------------------


def load_proposals(path: Path) -> tuple[list[dict], list[str]]:
    """Parse the agenda JSONL; returns (valid, errors).

    Validation is deliberately strict — a malformed proposal is
    reported and skipped, never silently ingested.
    """
    valid: list[dict] = []
    errors: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        return [], [f"cannot read {path}: {exc}"]
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {i}: invalid JSON ({exc})")
            continue
        err = _validate(p)
        if err:
            errors.append(
                f"line {i} ({p.get('id', '?') if isinstance(p, dict) else '?'}): {err}"
            )
        else:
            valid.append(p)
    return valid, errors


def _validate(p: Any) -> str | None:
    """``None`` when *p* is a well-formed agenda proposal, else why not."""
    if not isinstance(p, dict):
        return "not a JSON object"
    if p.get("kind") != "agenda-proposal":
        return f"kind={p.get('kind')!r} (want 'agenda-proposal')"
    for key in ("id", "title", "signal", "lever"):
        v = p.get(key)
        if not isinstance(v, str) or not v.strip():
            return f"missing/empty {key!r}"
    if "\n" in p["title"] or "\r" in p["title"]:
        return "title contains a newline (myco titles are one line)"
    for key in ("score", "headroom"):
        v = p.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return f"{key} not numeric"
        if not 0.0 <= float(v) <= 1.0:
            return f"{key}={v} outside [0,1]"
    if p.get("confidence") not in _CONFIDENCE:
        return f"confidence={p.get('confidence')!r} not in {sorted(_CONFIDENCE)}"
    rank = p.get("rank")
    if rank is not None and (
        not isinstance(rank, int) or isinstance(rank, bool) or rank < 1
    ):
        return f"rank={rank!r} not a positive int"
    return None


# ---------------------------------------------------------------------------
#  Dedup + plan (pure — the testable core)
# ---------------------------------------------------------------------------


def proposal_marker(p: dict) -> str:
    """Return the dedup key embedded in every ledger title: the proposal id."""
    return str(p["id"])


def ledger_titles(existing: list[dict]) -> set[str]:
    """Titles of live ledger items (todos + issues), lowercased."""
    out: set[str] = set()
    for item in existing:
        t = item.get("title") if isinstance(item, dict) else None
        if isinstance(t, str):
            out.add(t.lower())
    return out


def _short_id(p: dict) -> str:
    """Proposal id minus its ``agenda-`` prefix — what lands in titles."""
    pid = proposal_marker(p)
    return pid[len("agenda-") :] if pid.startswith("agenda-") else pid


def is_ingested(p: dict, known_titles: set[str], state: dict) -> bool:
    """Return True when *p* is already tracked — state file or live title.

    Ledger titles carry the *short* id (``myco_title`` strips the
    ``agenda-`` prefix), so both spellings are tried as substrings.
    """
    if proposal_marker(p) in (state.get("ingested") or {}):
        return True
    needles = {proposal_marker(p).lower(), _short_id(p).lower()}
    return any(n and any(n in t for t in known_titles) for n in needles)


def myco_title(p: dict) -> str:
    """Ledger title: ``agenda <id>: <proposal title>`` (one line)."""
    # the id already starts with "agenda-"; drop the redundant prefix
    return f"agenda {_short_id(p)}: {p['title']}"


def myco_argv(p: dict, kind: str) -> list[str]:
    """Build the ``myco`` argv filing one proposal as a todo or an issue."""
    title = myco_title(p)
    pri = "high" if float(p["score"]) >= 0.5 else "med"
    labels = ["agenda"]
    if p["confidence"] == "low":
        labels.append("low-confidence")
    label_argv = [x for lab in labels for x in ("--label", lab)]
    argv = ["myco", "todo", "add", title, "--pri", pri, *label_argv]
    if kind == "issue":
        # `--body -` reads the markdown body from stdin (fed by
        # _file_one); `--body <FILE>`/`--text` are the alternatives.
        argv = [
            "myco",
            "issue",
            "new",
            title,
            "--pri",
            pri,
            *label_argv,
            "--body",
            "-",
        ]
    argv += ["--no-commit", "--json"]
    return argv


def issue_body(p: dict) -> str:
    """Markdown body for ``--kind issue``: evidence + lever + provenance."""
    lines = [
        f"Agenda proposal `{p['id']}` (signal `{p['signal']}`, "
        f"score {p['score']}, headroom {p['headroom']}, "
        f"confidence {p['confidence']}).",
        "",
        f"**Lever:** {p['lever']}",
        "",
        "**Target:** `" + json.dumps(p.get("target") or {}) + "`",
        "",
        "**Evidence:**",
    ]
    for src in (p.get("evidence") or {}).get("sources") or []:
        lines.append(f"- source: `{src}`")
    for k, v in (
        (p.get("evidence") or {}).get("metrics") or {}
    ).items():
        vs = json.dumps(v)
        lines.append(f"- {k}: {vs[:300]}{'…' if len(vs) > 300 else ''}")
    lines += [
        "",
        "Mined by `bench/drivers/agenda/agenda_miner.py`; filed by "
        "`scripts/agenda_ingest.py`. Proposals carry measured "
        "evidence — verify before acting.",
        "",
    ]
    return "\n".join(lines)


def plan_ingest(
    proposals: list[dict],
    known_titles: set[str],
    state: dict,
    *,
    since: int | None,
    cap: int,
    kind: str,
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Rank-order, dedup, cap.  Returns (to_file, skipped_with_reason).

    *to_file* entries carry the proposal plus the resolved argv/body.
    """
    # deterministic order independent of file order: score desc, id asc
    ordered = sorted(
        proposals, key=lambda p: (-float(p["score"]), p["id"])
    )
    seen_ids: set[str] = set()
    todo: list[dict] = []
    skipped: list[tuple[dict, str]] = []
    for p in ordered:
        rank = p.get("rank")
        if since is not None and rank is not None and rank <= since:
            skipped.append((p, f"rank {rank} <= --since {since}"))
            continue
        pid = proposal_marker(p)
        if pid in seen_ids:
            skipped.append((p, "duplicate id within agenda file"))
            continue
        seen_ids.add(pid)
        if is_ingested(p, known_titles, state):
            skipped.append((p, "already ingested (state or ledger)"))
            continue
        entry = dict(p)
        entry["_kind"] = kind
        if kind == "issue":
            entry["_body"] = issue_body(p)
        todo.append(entry)
        if len(todo) >= cap:
            break
    return todo, skipped


# ---------------------------------------------------------------------------
#  myco access (execute path only)
# ---------------------------------------------------------------------------


def _default_runner(
    argv: list[str], stdin_text: str | None = None
) -> tuple[int, str, str]:
    r = subprocess.run(
        argv,
        cwd=_REPO,
        input=stdin_text,
        capture_output=True,
        text=True,
    )
    return r.returncode, r.stdout, r.stderr


def _myco_json(argv: list[str], runner: Runner) -> list[dict]:
    """Run a myco list verb, return its JSON array ([] on any failure)."""
    rc, out, _ = runner([*argv, "--json"], None)
    if rc != 0:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def fetch_existing(runner: Runner) -> list[dict]:
    """Live ledger titles for dedup: todos + issues (open statuses)."""
    items = _myco_json(["myco", "todo", "list"], runner)
    items += _myco_json(["myco", "issue", "list"], runner)
    return items


def _file_one(p: dict, runner: Runner) -> tuple[str, str]:
    """File one planned proposal; returns (item_id_or_'?', detail)."""
    kind = p["_kind"]
    argv = myco_argv(p, kind)
    # `--body -` (issue kind) reads the markdown from stdin; todos pass None
    rc, out, err = runner(
        argv, p.get("_body") if kind == "issue" else None
    )
    item_id = "?"
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            item_id = str(data.get("id") or data.get("item") or "?")
        elif isinstance(data, list) and data:
            item_id = str(data[0].get("id") or "?")
    except json.JSONDecodeError:
        pass
    if rc != 0:
        return item_id, f"rc={rc} {err.strip()[:160]}"
    return item_id, out.strip()[:120]


# ---------------------------------------------------------------------------
#  State file
# ---------------------------------------------------------------------------


def load_state(path: Path) -> dict:
    """Load the ingest state file; ``{}`` when absent/corrupt."""
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(path: Path, state: dict) -> None:
    """Write the ingest state file (creates parents, stamps last_run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_run"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry — dry-run by default; ``--execute`` files via myco."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--agenda",
        type=Path,
        default=_DEFAULT_AGENDA,
        help="agenda JSONL (default docs/reports/agenda.jsonl)",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=5,
        help="max proposals to file this run (default 5)",
    )
    ap.add_argument(
        "--kind",
        choices=("todo", "issue"),
        default="todo",
        help="ledger kind: todo (default) or issue (body = evidence)",
    )
    ap.add_argument(
        "--since",
        type=int,
        default=None,
        metavar="RANK",
        help="incremental: only proposals with rank > RANK",
    )
    ap.add_argument(
        "--state",
        type=Path,
        default=_DEFAULT_STATE,
        help="ingest state file (default docs/reports/agenda_ingest_state.json)",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="actually call `myco` (default: dry-run plan only)",
    )
    args = ap.parse_args(argv)

    proposals, errors = load_proposals(args.agenda)
    for e in errors:
        print(f"agenda_ingest: INVALID {e}", file=sys.stderr)
    if not proposals:
        print("agenda_ingest: no valid proposals — nothing to do")
        return 1 if errors else 0

    state = load_state(args.state)
    # dedup against the live ledger; a myco failure degrades to
    # state-file dedup only (fetch_existing returns [] then).
    existing = fetch_existing(_default_runner)
    known = ledger_titles(existing)

    todo, skipped = plan_ingest(
        proposals,
        known,
        state,
        since=args.since,
        cap=args.cap,
        kind=args.kind,
    )

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(
        f"agenda_ingest [{mode}] agenda={args.agenda} "
        f"valid={len(proposals)} file={len(todo)} skipped={len(skipped)}"
    )
    for i, p in enumerate(todo, 1):
        pri = "high" if float(p["score"]) >= 0.5 else "med"
        print(f"  {i}. [{pri}] rank={p.get('rank')} {p['id']}")
        print(f"     {p['title'][:100]}")
    for p, why in skipped[:10]:
        print(f"  skip {p.get('id', '?')}: {why}")
    if len(skipped) > 10:
        print(f"  … {len(skipped) - 10} more skipped")
    for e in errors:
        print(f"  invalid: {e}", file=sys.stderr)

    if not args.execute:
        print(
            "dry-run only — re-run with --execute to file these in "
            "project/ (human approval gate)."
        )
        return 0

    ingested = state.setdefault("ingested", {})
    failures = 0
    for p in todo:
        item_id, detail = _file_one(p, _default_runner)
        pid = proposal_marker(p)
        if item_id != "?":
            ingested[pid] = {
                "item": item_id,
                "kind": p["_kind"],
                "rank": p.get("rank"),
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        ok = not detail.startswith("rc=")
        if not ok:
            failures += 1
        print(f"  filed {pid} -> {item_id} ({p['_kind']}) {detail}")
    save_state(args.state, state)
    print(
        f"state -> {args.state} ({len(ingested)} ingested total, "
        f"{failures} failures)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
