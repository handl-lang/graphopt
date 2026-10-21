"""The overnight discovery driver's contract.

``bench/drivers/overnight.py`` is the loop the resumable infra was
built for: open-game episodes mint candidates, the referee gauntlet
admits them, the prior-art bank classifies the survivors, and every
outcome lands in a protocol-gated JSONL ledger.

These tests pin the operational contracts — torch-eager, seconds:

* **ledger schema** — a manifest line carrying the protocol and the
  provenance triple (``git_sha`` / ``git_dirty`` / ``device``), then
  ``{"key", "record", "ts"}`` cells: one ``("episode", i)`` per page,
  one ``("candidate", name)`` per minted rule (with the serialized
  law so resume never re-derives it), one ``("classify", name)`` per
  admitted law;
* **resume idempotence** — a ``--resume`` run over a complete store
  replays cells verbatim: the episode engine and the referee are
  never re-entered (spy-pinned, ``test_resilience.py`` style);
* **``--dry``** — two episodes, a tiny pool, the full referee: the
  run completes in seconds and emits both artifacts;
* **honest skip** — a player that never evaluates mints nothing: the
  ledger records zero candidates, the documents still emit, nothing
  crashes.
"""

from __future__ import annotations

import json
import time

import pytest

import bench.referees
from bench.benchkit import checkpoint as ck_mod
from bench.benchkit.env import collect_env
from bench.benchkit.jsonl import read_jsonl
from bench.drivers import overnight as on


def _argv(tmp_path, *extra):
    """A complete, seconds-scale argv pointed at *tmp_path*."""
    return [
        "--episodes",
        "2",
        "--page",
        "6",
        "--budget",
        "4",
        "--ckpt",
        str(tmp_path / "ck.jsonl"),
        "--out",
        str(tmp_path / "out.md"),
        *extra,
    ]


def _records(path):
    recs = read_jsonl(path)
    man, cells = recs[0], recs[1:]
    return man, cells


def _cells(cells, kind):
    return [c for c in cells if c["key"][0] == kind]


class TestDryRun:
    """The end-to-end smoke: a real (tiny) loop, seconds."""

    def test_dry_completes_and_emits(self, tmp_path):
        t0 = time.time()
        assert on.main(_argv(tmp_path, "--dry")) == 0
        assert (
            time.time() - t0 < 120
        )  # measured ~10 s; the box is generous
        md = tmp_path / "out.md"
        js = tmp_path / "out.json"
        assert md.exists() and js.exists()
        payload = json.loads(js.read_text())
        # the provenance contract
        assert payload["generated_utc"]
        assert "seed" in payload
        for f in ("git_sha", "git_dirty", "device", "timestamp_utc"):
            assert f in payload["env"]
        assert payload["episodes"] == 2
        assert payload["evaluated"] >= 0
        text = md.read_text()
        assert "Admission funnel" in text and "Novelty" in text

    def test_ledger_schema(self, tmp_path):
        on.main(_argv(tmp_path, "--dry"))
        man, cells = _records(tmp_path / "ck.jsonl")
        assert man["_manifest"] is True
        assert man["lane"] == "overnight"
        assert man["protocol"]["episodes"] == 2
        assert {"git_sha", "git_dirty", "device"} <= set(
            man["provenance"]
        )
        for c in cells:
            assert {"key", "record", "ts"} <= set(c)
        episodes = _cells(cells, "episode")
        assert len(episodes) == 2
        for e in episodes:
            rec = e["record"]
            assert {"evals", "reward", "steps"} <= set(rec)
            for ev in rec["evals"]:
                assert {"i", "name", "label", "verdict"} <= set(ev)
        for c in _cells(cells, "candidate"):
            rec = c["record"]
            assert rec["name"] == c["key"][1]
            assert rec["law"]["kind"] == "law"  # serialized, resumable
            assert isinstance(rec["admitted"], bool)
            assert isinstance(rec["stages"], list)
            for s in rec["stages"]:
                assert {"name", "passed", "detail"} <= set(s)
            assert rec["arena"]["verdict"] in (
                "useful",
                "true",
                "known",
                "false",
            )
        admitted = [
            c["key"][1]
            for c in _cells(cells, "candidate")
            if c["record"]["admitted"]
        ]
        classified = {c["key"][1] for c in _cells(cells, "classify")}
        assert classified == set(admitted)
        for c in _cells(cells, "classify"):
            assert c["record"]["verdict"] in (
                "machine-original",
                "known-art",
                "vacuous",
            )


class TestCorpusZoo:
    """``--corpus zoo`` — real modules as the episode boards."""

    def test_zoo_arena_pool_is_real_matched(self):
        """Every pooled candidate's pattern LHS binds the module."""
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.ir import Op
        from mlopt_core.meta import _positions

        case = {"zoo": "LoRAAdapter", "wseed": 0, "page": 6, "scan": 384}
        arena = on._ZooArena(case)
        assert not arena.error
        assert arena.pool, "no stream candidate matched the module"
        ir, _t, _f = on._zoo_export(case)
        subs = [s for _p, s in _positions(ir.root) if isinstance(s, Op)]
        for cand in arena.pool:
            pat = cand.as_rule().lhs
            assert any(
                _term_match(pat, s) is not None for s in subs
            ), f"{cand.label}: not a real match"

    def test_zoo_dry_completes_and_records_corpus(self, tmp_path):
        """The zoo arm end-to-end: episodes are real modules."""
        t0 = time.time()
        assert (
            on.main(_argv(tmp_path, "--dry", "--corpus", "zoo")) == 0
        )
        assert time.time() - t0 < 180  # exports + referee, seconds
        man, cells = _records(tmp_path / "ck.jsonl")
        # the corpus arm is protocol-pinned — resume can't mix arms
        assert man["protocol"]["corpus"] == "zoo"
        episodes = _cells(cells, "episode")
        assert len(episodes) == 2
        for e in episodes:
            assert "zoo" in e["record"]["case"]
        payload = json.loads((tmp_path / "out.json").read_text())
        assert payload["corpus"] == "zoo"
        assert "corpus: `zoo`" in (tmp_path / "out.md").read_text()

    def test_grammar_is_the_default_corpus(self, tmp_path):
        """Back-compat: a bare ``--dry`` runs the grammar arm."""
        assert on.main(_argv(tmp_path, "--dry")) == 0
        man, _cells = _records(tmp_path / "ck.jsonl")
        assert man["protocol"]["corpus"] == "grammar"


class TestResume:
    """A protocol-matched resume replays cells, never re-measures."""

    def test_resume_reuses_store(self, tmp_path, monkeypatch):
        argv = _argv(tmp_path, "--dry")
        assert on.main(argv) == 0
        _man1, cells1 = _records(tmp_path / "ck.jsonl")
        summary1 = json.loads((tmp_path / "out.json").read_text())

        def boom(*a, **k):
            raise AssertionError("recompute on resume")

        monkeypatch.setattr(on.engine, "run_episode", boom)
        monkeypatch.setattr(bench.referees, "judge", boom)
        monkeypatch.setattr(on, "verify_law", boom)
        assert on.main([*argv, "--resume"]) == 0
        _man2, cells2 = _records(tmp_path / "ck.jsonl")
        summary2 = json.loads((tmp_path / "out.json").read_text())
        # verbatim replay — no new cells, identical summary
        assert len(cells2) == len(cells1)
        for k in (
            "episodes",
            "evaluated",
            "minted",
            "admitted",
            "novelty",
            "leg_refusals",
        ):
            assert summary2[k] == summary1[k]

    def test_protocol_mismatch_restarts(self, tmp_path):
        assert on.main(_argv(tmp_path, "--dry")) == 0
        # a different protocol --resume must not merge old cells
        argv = _argv(tmp_path, "--budget", "3", "--resume")
        assert on.main(argv) == 0
        man, _cells = _records(tmp_path / "ck.jsonl")
        assert man["protocol"]["budget"] == 3


class TestHonestSkip:
    """No minted candidates — recorded, not a crash."""

    def test_empty_episodes(self, tmp_path):
        man = ck_mod.new_manifest(
            "overnight", {"protocol": "test"}, device="cpu"
        )
        ckpt = ck_mod.Checkpoint.open(tmp_path / "ck.jsonl", man)
        never = lambda _i: lambda _state: None  # noqa: E731
        summary = on.run(
            episodes=2,
            page=4,
            budget=4,
            seed=0,
            player="playbook",
            referee="full",
            ckpt=ckpt,
            arm_factory=never,
            log=lambda *_a: None,
        )
        assert summary["episodes"] == 2
        assert summary["evaluated"] == 0
        assert summary["minted"] == 0
        assert summary["admitted"] == 0
        assert summary["novelty"] == {}
        # the cells exist — the empty run is a recorded fact
        episodes = _cells(
            read_jsonl(tmp_path / "ck.jsonl")[1:], "episode"
        )
        assert len(episodes) == 2
        assert all(e["record"]["evals"] == [] for e in episodes)
        # and the documents still emit
        out = on.emit(
            summary,
            tmp_path / "out.md",
            env=collect_env("cpu"),
            seed=0,
        )
        assert out.exists()
        assert (
            "None." in out.read_text() or "**None.**" in out.read_text()
        )

    def test_page_beyond_stream_records_empty(self, tmp_path):
        man = ck_mod.new_manifest(
            "overnight", {"protocol": "test"}, device="cpu"
        )
        ckpt = ck_mod.Checkpoint.open(tmp_path / "ck.jsonl", man)

        def boom(_state):
            raise AssertionError("player consulted on a dead board")

        summary = on.run(
            episodes=1,
            page=4,
            budget=4,
            seed=0,
            player="playbook",
            referee="full",
            ckpt=ckpt,
            arm_factory=lambda _i: boom,
            # past the ~1.1k-candidate stream — an empty pool is a
            # done state, so the player is never consulted.
            cases=[{"ops": None, "offset": 10**9, "page": 4}],
            log=lambda *_a: None,
        )
        assert summary["episodes"] == 1
        assert summary["evaluated"] == 0
        assert summary["minted"] == 0
        episodes = _cells(
            read_jsonl(tmp_path / "ck.jsonl")[1:], "episode"
        )
        assert episodes[0]["record"]["evals"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
