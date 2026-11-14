"""Tests for ``scripts/agenda_ingest.py`` — the agenda→ledger ingest.

The script is loaded by path (``scripts/`` is not a package, per the
repo convention that scripts are standalone tooling).  All myco
interaction goes through the injectable ``Runner`` — no test shells
out.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "agenda_ingest.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "agenda_ingest", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agenda_ingest"] = mod
    spec.loader.exec_module(mod)
    return mod


ai = _load()


def _prop(pid: str, score: float = 0.5, rank: int = 1, **kw) -> dict:
    p = {
        "id": pid,
        "kind": "agenda-proposal",
        "title": f"proposal {pid}",
        "target": {"domain": "gen"},
        "signal": "test_signal",
        "evidence": {
            "sources": ["docs/reports/x.json"],
            "metrics": {"n": 3},
        },
        "lever": "do the thing",
        "headroom": 0.7,
        "confidence": "high",
        "score": score,
        "rank": rank,
    }
    p.update(kw)
    return p


# ---------------------------------------------------------------------------
#  load + validate
# ---------------------------------------------------------------------------


def test_load_proposals_valid(tmp_path):
    f = tmp_path / "agenda.jsonl"
    f.write_text(json.dumps(_prop("agenda-a-1")) + "\n\n")
    valid, errors = ai.load_proposals(f)
    assert len(valid) == 1 and errors == []


@pytest.mark.parametrize(
    "mutate,frag",
    [
        ({"kind": "other"}, "kind"),
        ({"title": "two\nlines"}, "newline"),
        ({"score": 1.5}, "score"),
        ({"headroom": -0.1}, "headroom"),
        ({"confidence": "sure"}, "confidence"),
        ({"id": ""}, "id"),
        ({"rank": 0}, "rank"),
    ],
)
def test_load_proposals_rejects(tmp_path, mutate, frag):
    p = _prop("agenda-bad-1")
    p.update(mutate)
    f = tmp_path / "agenda.jsonl"
    f.write_text(json.dumps(p) + "\n")
    valid, errors = ai.load_proposals(f)
    assert valid == []
    assert errors and frag in errors[0]


def test_load_proposals_bad_json_and_missing_file(tmp_path):
    f = tmp_path / "agenda.jsonl"
    f.write_text("{not json\n")
    valid, errors = ai.load_proposals(f)
    assert valid == [] and "invalid JSON" in errors[0]
    valid, errors = ai.load_proposals(tmp_path / "nope.jsonl")
    assert valid == [] and "cannot read" in errors[0]


# ---------------------------------------------------------------------------
#  dedup + plan
# ---------------------------------------------------------------------------


def test_plan_orders_by_score_and_caps():
    props = [
        _prop("agenda-lo-1", score=0.2, rank=9),
        _prop("agenda-hi-1", score=0.9, rank=1),
        _prop("agenda-mid-1", score=0.5, rank=5),
    ]
    todo, skipped = ai.plan_ingest(
        props, set(), {}, since=None, cap=2, kind="todo"
    )
    assert [p["id"] for p in todo] == ["agenda-hi-1", "agenda-mid-1"]
    assert skipped == []  # cap truncates the tail silently


def test_plan_dedups_state_and_ledger():
    props = [
        _prop("agenda-x-1"),
        _prop("agenda-x-2"),
        _prop("agenda-x-3"),
    ]
    state = {"ingested": {"agenda-x-2": {"item": "T-001"}}}
    titles = {"agenda x-3: already filed proposal".lower()}
    todo, skipped = ai.plan_ingest(
        props, titles, state, since=None, cap=10, kind="todo"
    )
    assert [p["id"] for p in todo] == ["agenda-x-1"]
    reasons = {p["id"]: w for p, w in skipped}
    assert (
        "state" in reasons["agenda-x-2"]
        or "ingested" in reasons["agenda-x-2"]
    )
    assert "ingested" in reasons["agenda-x-3"]


def test_plan_since_filter():
    props = [_prop("agenda-s-1", rank=3), _prop("agenda-s-2", rank=8)]
    todo, skipped = ai.plan_ingest(
        props, set(), {}, since=5, cap=10, kind="todo"
    )
    assert [p["id"] for p in todo] == ["agenda-s-2"]
    assert skipped[0][1].startswith("rank 3")


def test_plan_duplicate_ids_in_file():
    props = [
        _prop("agenda-dup-1", score=0.9),
        _prop("agenda-dup-1", score=0.1),
    ]
    todo, skipped = ai.plan_ingest(
        props, set(), {}, since=None, cap=10, kind="todo"
    )
    assert len(todo) == 1
    assert any("duplicate" in w for _, w in skipped)


# ---------------------------------------------------------------------------
#  myco argv / titles
# ---------------------------------------------------------------------------


def test_myco_title_embeds_id():
    t = ai.myco_title(_prop("agenda-paymeasured_confound-abc"))
    assert t.startswith("agenda paymeasured_confound-abc: proposal")
    assert "\n" not in t


def test_myco_argv_todo_and_issue():
    p = _prop("agenda-z-1", score=0.9, confidence="low")
    argv = ai.myco_argv(p, "todo")
    assert argv[:3] == ["myco", "todo", "add"]
    assert "--pri" in argv and "high" in argv
    assert "low-confidence" in argv
    assert argv[-2:] == ["--no-commit", "--json"]
    argv_i = ai.myco_argv(p, "issue")
    assert argv_i[:3] == ["myco", "issue", "new"]
    assert argv_i[argv_i.index("--body") + 1] == "-"


def test_issue_body_carries_evidence():
    p = _prop("agenda-b-1")
    body = ai.issue_body(p)
    assert "agenda-b-1" in body and "do the thing" in body
    assert "docs/reports/x.json" in body


# ---------------------------------------------------------------------------
#  execute path (fake runner)
# ---------------------------------------------------------------------------


def _fake_runner_factory(calls):
    def runner(argv, stdin_text):
        calls.append((argv, stdin_text))
        if argv[:3] == ["myco", "todo", "list"] or argv[:3] == [
            "myco",
            "issue",
            "list",
        ]:
            return 0, "[]", ""
        return 0, json.dumps({"id": "T-999"}), ""

    return runner


def test_file_one_todo_and_issue():
    calls = []
    runner = _fake_runner_factory(calls)
    p = ai.plan_ingest(
        [_prop("agenda-f-1")], set(), {}, since=None, cap=1, kind="todo"
    )[0][0]
    item_id, _detail = ai._file_one(p, runner)
    assert item_id == "T-999"
    argv, stdin = calls[-1]
    assert argv[:3] == ["myco", "todo", "add"] and stdin is None

    p2 = ai.plan_ingest(
        [_prop("agenda-f-2")],
        set(),
        {},
        since=None,
        cap=1,
        kind="issue",
    )[0][0]
    item_id, _ = ai._file_one(p2, runner)
    assert item_id == "T-999"
    argv, stdin = calls[-1]
    assert (
        argv[:3] == ["myco", "issue", "new"] and "agenda-f-2" in stdin
    )


def test_file_one_failure():
    def runner(argv, stdin_text):
        return 1, "", "boom"

    p = ai.plan_ingest(
        [_prop("agenda-f-3")], set(), {}, since=None, cap=1, kind="todo"
    )[0][0]
    item_id, detail = ai._file_one(p, runner)
    assert item_id == "?" and detail.startswith("rc=1")


# ---------------------------------------------------------------------------
#  main(): dry-run default, execute gated
# ---------------------------------------------------------------------------


def test_main_dry_run_and_execute(tmp_path, monkeypatch, capsys):
    agenda = tmp_path / "agenda.jsonl"
    agenda.write_text(
        "\n".join(
            json.dumps(
                _prop(f"agenda-m-{i}", score=0.5 + i * 0.1, rank=i + 1)
            )
            for i in range(3)
        )
        + "\n"
    )
    state = tmp_path / "state.json"
    calls = []
    monkeypatch.setattr(ai, "fetch_existing", lambda runner: [])
    monkeypatch.setattr(
        ai, "_default_runner", _fake_runner_factory(calls)
    )

    rc = ai.main(
        ["--agenda", str(agenda), "--state", str(state), "--cap", "2"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "file=2" in out
    assert calls == []  # nothing shelled out
    assert not state.exists()

    rc = ai.main(
        [
            "--agenda",
            str(agenda),
            "--state",
            str(state),
            "--cap",
            "2",
            "--execute",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # score-desc order: m-2 (0.7) and m-1 (0.6) file first
    assert "EXECUTE" in out and "filed agenda-m-2" in out
    saved = json.loads(state.read_text())
    assert (
        "agenda-m-1" in saved["ingested"]
        and "agenda-m-2" in saved["ingested"]
    )

    # second run: state dedups the filed two; m-0 files
    rc = ai.main(
        [
            "--agenda",
            str(agenda),
            "--state",
            str(state),
            "--cap",
            "5",
            "--execute",
        ]
    )
    out = capsys.readouterr().out
    assert "file=1" in out and "agenda-m-0" in out


def test_main_since(tmp_path, monkeypatch, capsys):
    agenda = tmp_path / "agenda.jsonl"
    agenda.write_text(
        "\n".join(
            json.dumps(_prop(f"agenda-n-{i}", rank=i + 1))
            for i in range(4)
        )
        + "\n"
    )
    monkeypatch.setattr(ai, "fetch_existing", lambda runner: [])
    rc = ai.main(
        [
            "--agenda",
            str(agenda),
            "--state",
            str(tmp_path / "s.json"),
            "--since",
            "2",
            "--cap",
            "10",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "file=2" in out  # ranks 3 and 4
