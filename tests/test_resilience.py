"""Plan 0032 phase 2b — the resilience lane.

Pins the operational contracts:

* **parallel-safe stores** — the sqlite evidence store under
  concurrent writers/readers (WAL + busy_timeout), the dir-keyed
  rule cache under concurrent stores (per-writer tmp + atomic
  rename + orphan sweep), and the JSONL ledger/checkpoint append
  (one ``O_APPEND`` write — no lockfile);
* **checkpoint/resume** — the manifest-gated JSONL cell store the
  matrix and leaderboard lanes persist into: a crash costs a cell,
  a protocol-matched resume replays recorded rows verbatim, a
  protocol mismatch restarts loudly;
* **provenance** — every emitted artifact carries {date, git
  sha/dirty, seed, device} — enforced inside the benchkit emitters,
  not by convention;
* **determinism** — the same inputs emit identical artifacts modulo
  timestamp fields.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mlopt_discovery import evidence as ev  # noqa: E402
from mlopt_discovery.pipeline import Evidence, Proposal  # noqa: E402

from bench.benchkit.checkpoint import (  # noqa: E402
    Checkpoint,
    new_manifest,
)
from bench.benchkit.jsonl import append_jsonl, read_jsonl  # noqa: E402
from bench.benchkit.provenance import (  # noqa: E402
    ProvenanceError,
    require_provenance,
    with_provenance,
)

# ---------------------------------------------------------------------------
#  evidence store — concurrent writers + readers
# ---------------------------------------------------------------------------


def _meta(**kw):
    meta = {
        "corpus_hash": ev.corpus_hash(["bench:a:(4,4)"]),
        "rules_hash": ev.rules_hash(["k1"]),
        "code_rev": "abc123",
        "run_id": "run0",
        "holdout": "",
        "ts": "2024-01-01T00:00:00+00:00",
    }
    meta.update(kw)
    return meta


def _row(name, key):
    prop = Proposal(
        name=name, lhs=None, rhs=None, family="fam", sources=("s",)
    )
    evid = Evidence(
        proposal=prop,
        num_true=True,
        derivable=False,
        census_sites=3,
        relaxed=4,
        matches=2,
        example="add(x, y)",
        fires=5,
        fires_typed=5,
        fire_cases=("m1",),
        changed=1,
        paid=1,
        cost_drop=0.25,
        relation="new",
    )
    return ev.verdict_row(key, evid, "add(x, y)", "add(y, x)")


def test_evidence_store_concurrent_writers_and_readers(tmp_path):
    """N parallel writers + readers on one store — no SQLITE_BUSY."""
    db = str(tmp_path / "s.db")
    n_writers, n_rounds = 6, 10

    def write(w):
        conn = ev.connect(db)
        try:
            for r in range(n_rounds):
                ev.record_run(
                    conn,
                    _meta(run_id=f"run{w}", ts=f"t{w}:{r}"),
                    [
                        _row(f"law_{w}_{r}", f"K{w}_{r}"),
                        _row("shared", "SHARED"),
                    ],
                )
        finally:
            conn.close()

    stop = []
    reads = []

    def read():
        conn = ev.connect(db)
        try:
            for _ in range(
                2000
            ):  # bounded — no spin-forever on failure
                if stop:
                    break
                hits = ev.latest_verdicts(
                    conn,
                    _meta()["corpus_hash"],
                    _meta()["rules_hash"],
                    "abc123",
                )
                reads.append(len(hits))
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=n_writers + 2) as pool:
        readers = [pool.submit(read) for _ in range(2)]
        writers = [pool.submit(write, w) for w in range(n_writers)]
        try:
            for f in writers:
                f.result()  # propagates SQLITE_BUSY / corruption
        finally:
            stop.append(True)
        for f in readers:
            f.result()

    conn = ev.connect(db)
    hits = ev.latest_verdicts(
        conn,
        _meta()["corpus_hash"],
        _meta()["rules_hash"],
        "abc123",
    )
    conn.close()
    expected = {
        f"K{w}_{r}" for w in range(n_writers) for r in range(n_rounds)
    }
    expected.add("SHARED")
    assert set(hits) == expected
    assert reads and all(n <= len(expected) for n in reads)


def test_connect_retries_the_wal_transition(tmp_path):
    """The WAL pragma waits out a held exclusive lock, then succeeds."""
    import sqlite3
    import threading

    db = str(tmp_path / "s.db")
    # A rollback-journal db under a held EXCLUSIVE lock: the WAL
    # transition cannot proceed and does not honour busy_timeout —
    # connect retries until the lock clears.
    raw = sqlite3.connect(db, check_same_thread=False)
    raw.execute("CREATE TABLE t (x)")
    raw.commit()
    raw.execute("BEGIN EXCLUSIVE")
    raw.execute("INSERT INTO t VALUES (1)")

    t = threading.Timer(0.15, raw.commit)
    t.start()
    conn = ev.connect(db, busy_timeout_s=5.0)
    t.join()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()
    raw.close()


def test_connect_degrades_when_wal_stays_locked(tmp_path):
    """A lock past the deadline: WAL pragma is skipped, schema fails."""
    import sqlite3

    db = str(tmp_path / "s.db")
    raw = sqlite3.connect(db)
    raw.execute("CREATE TABLE t (x)")
    raw.commit()
    raw.execute("BEGIN EXCLUSIVE")
    raw.execute("INSERT INTO t VALUES (1)")
    try:
        with pytest.raises(sqlite3.OperationalError):
            ev.connect(db, busy_timeout_s=0.05)
    finally:
        raw.rollback()
        raw.close()


# ---------------------------------------------------------------------------
#  rule cache — concurrent stores, orphan sweep
# ---------------------------------------------------------------------------


def _toy_rule(name):
    from mlopt_core.egraph import Rewrite
    from mlopt_core.ir import Const, Op, TensorType, Var

    x = Var("x", TensorType((4,)))
    return Rewrite(
        name, Op.make("add", x, x), Op.make("mul", x, Const(2.0))
    )


def test_rulecache_concurrent_stores(tmp_path):
    """Parallel writers share no tmp file; every key loads cleanly."""
    from mlopt_core.rulecache import RuleCache

    cache = RuleCache(tmp_path / "rc")

    def store(k):
        RuleCache(tmp_path / "rc").store(
            f"key{k}", [_toy_rule(f"r{k}")]
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        # distinct keys plus same-key writers racing identical content
        list(pool.map(store, [*range(8), 0, 1, 2]))

    for k in range(8):
        loaded = cache.load(f"key{k}")
        assert loaded is not None and loaded[0].name == f"r{k}"
    assert not list(cache.dir.glob(".rules-*.tmp-*"))


def test_rulecache_sweeps_orphan_tmps(tmp_path):
    """Only *old* tmp files are swept — an in-flight write is safe."""
    import os

    from mlopt_core.rulecache import RuleCache

    cache = RuleCache(tmp_path / "rc")
    cache.dir.mkdir(parents=True)
    old = cache.dir / ".rules-dead.tmp-999-aa"
    old.write_text("partial")
    os.utime(old, (0, 0))  # crash leftover, not an in-flight write
    # An old directory under the tmp pattern raises OSError on
    # unlink — the sweep must swallow it, not fail the store.
    old_dir = cache.dir / ".rules-dead.tmp-888-bb"
    old_dir.mkdir()
    os.utime(old_dir, (0, 0))
    # A fresh tmp is a live writer's file — the sweep leaves it.
    fresh = cache.dir / ".rules-live.tmp-111-cc"
    fresh.write_text("in flight")

    cache.store("k", [_toy_rule("r")])
    assert not old.exists()
    assert old_dir.is_dir()
    assert fresh.exists()
    assert cache.load("k") is not None


# ---------------------------------------------------------------------------
#  JSONL — atomic append + torn-tail tolerance
# ---------------------------------------------------------------------------


def test_append_jsonl_concurrent_writers(tmp_path):
    """Interleaved appenders produce N intact lines — never a splice."""
    path = tmp_path / "ledger.jsonl"
    n = 60

    def emit(i):
        # pad records past the stdio buffer so a naive buffered
        # append would visibly splice
        append_jsonl(path, {"i": i, "pad": "x" * 512})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(emit, range(n)))

    records = read_jsonl(path)
    assert len(records) == n
    assert sorted(r["i"] for r in records) == list(range(n))


def test_read_jsonl_torn_tail(tmp_path):
    """A crash mid-append drops the torn last line, keeps the rest."""
    path = tmp_path / "c.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"a": 2})
    with path.open("a") as fh:
        fh.write('{"a": 3, "trun')
    records = read_jsonl(path, tolerate_torn_tail=True)
    assert [r["a"] for r in records] == [1, 2]
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


# ---------------------------------------------------------------------------
#  checkpoint — manifest gate, reuse, restart
# ---------------------------------------------------------------------------

_ENV = {
    "git_sha": "deadbeef",
    "git_dirty": False,
    "device": "cpu",
    "timestamp_utc": "2024-01-01T00:00:00+00:00",
}


def _manifest(protocol, env=None):
    return new_manifest("players", protocol, env=env or _ENV)


def test_checkpoint_records_and_reuses(tmp_path):
    path = tmp_path / "ck.jsonl"
    ck = Checkpoint.open(path, _manifest({"seed": 0}), env=_ENV)
    ck.record(("meta", "playbook"), {"reward_ep": 1.0})
    ck.record(("open", "random"), {"reward_ep": -0.5})

    again = Checkpoint.open(path, _manifest({"seed": 0}), env=_ENV)
    assert again.done(("meta", "playbook"))
    assert again.get(("meta", "playbook")) == {"reward_ep": 1.0}
    assert again.get(("open", "random")) == {"reward_ep": -0.5}
    assert not again.done(("meta", "random"))


def test_checkpoint_torn_tail_survives(tmp_path):
    path = tmp_path / "ck.jsonl"
    ck = Checkpoint.open(path, _manifest({"seed": 0}), env=_ENV)
    ck.record(("meta", "playbook"), {"reward_ep": 1.0})
    with path.open("a") as fh:  # a crash mid-append
        fh.write('{"key": ["meta", "li')
    again = Checkpoint.open(path, _manifest({"seed": 0}), env=_ENV)
    assert again.get(("meta", "playbook")) == {"reward_ep": 1.0}


def test_checkpoint_protocol_mismatch_restarts(tmp_path):
    """Different protocol = a different run; the file restarts loud."""
    path = tmp_path / "ck.jsonl"
    ck = Checkpoint.open(path, _manifest({"seed": 0}), env=_ENV)
    ck.record(("meta", "playbook"), {"reward_ep": 1.0})

    warned = []
    again = Checkpoint.open(
        path,
        _manifest({"seed": 1}),
        env=_ENV,
        warn=warned.append,
    )
    assert warned and "protocol differs" in warned[0]
    assert not again.entries
    # the manifest was rewritten — a third open matches the new run
    third = Checkpoint.open(
        path, _manifest({"seed": 1}), env=_ENV, warn=warned.append
    )
    assert not third.entries


def test_checkpoint_git_drift_warns_but_reuses(tmp_path):
    path = tmp_path / "ck.jsonl"
    ck = Checkpoint.open(path, _manifest({"seed": 0}), env=_ENV)
    ck.record(("meta", "playbook"), {"reward_ep": 1.0})
    warned = []
    again = Checkpoint.open(
        path,
        _manifest({"seed": 0}),
        env={"git_sha": "cafe", "git_dirty": True, "device": "cpu"},
        warn=warned.append,
    )
    assert any("git sha moved" in w for w in warned)
    assert again.done(("meta", "playbook"))


# ---------------------------------------------------------------------------
#  resume — the lanes replay recorded cells, never re-measure
# ---------------------------------------------------------------------------


def test_episode_board_resume_is_idempotent(tmp_path, monkeypatch):
    """A second run replays rows verbatim and never rebuilds arms."""
    from gamekit.players import registry as preg

    from bench.players.runner import run_episode_board

    path = tmp_path / "lb.jsonl"
    kw = dict(seed=0, cases=2, budget=4)
    m = _manifest({"lane": "players", **kw})

    ck = Checkpoint.open(path, m, env=_ENV)
    rows1 = run_episode_board(
        "meta", ["playbook", "random"], checkpoint=ck, **kw
    )

    calls = []
    real_make = preg.make

    def spy(*a, **k):
        calls.append(a)
        return real_make(*a, **k)

    monkeypatch.setattr(preg, "make", spy)
    ck2 = Checkpoint.open(path, m, env=_ENV)
    rows2 = run_episode_board(
        "meta", ["playbook", "random"], checkpoint=ck2, **kw
    )
    assert calls == []  # every row replayed
    assert [r.as_record() for r in rows2] == [
        r.as_record() for r in rows1
    ]


def test_episode_board_partial_resume(tmp_path, monkeypatch):
    """A crash after one player costs exactly that player's row."""
    from bench.players.runner import run_episode_board

    path = tmp_path / "lb.jsonl"
    kw = dict(seed=0, cases=2, budget=4)
    m = _manifest({"lane": "players", **kw})
    ck = Checkpoint.open(path, m, env=_ENV)
    rows1 = run_episode_board("meta", ["playbook"], checkpoint=ck, **kw)

    # resume with a wider selection — only the new player runs
    ck2 = Checkpoint.open(path, m, env=_ENV)
    rows2 = run_episode_board(
        "meta", ["playbook", "random"], checkpoint=ck2, **kw
    )
    assert rows2[0].as_record() == rows1[0].as_record()
    assert (
        rows2[1].player == "random" and rows2[1].reward_ep is not None
    )


def test_contraction_resume_replays_instances(tmp_path, monkeypatch):
    """Fully-cached instances never re-seed a bond network."""
    import mlopt_torch.contraction_policy as cp

    from bench.players.runner import run_contraction

    path = tmp_path / "lb.jsonl"
    kw = dict(seed=0, scales=(6,), instances=2, samples=2, device="cpu")
    m = _manifest({"lane": "players", **kw})

    ck = Checkpoint.open(path, m, env=_ENV)
    rows1 = run_contraction(["greedy", "random"], checkpoint=ck, **kw)

    def boom(*a, **k):
        raise AssertionError("instance re-measured on resume")

    monkeypatch.setattr(cp, "random_bond_network", boom)
    ck2 = Checkpoint.open(path, m, env=_ENV)
    rows2 = run_contraction(["greedy", "random"], checkpoint=ck2, **kw)
    assert [r.as_record() for r in rows2] == [
        r.as_record() for r in rows1
    ]


def test_matrix_resume_replays_cells(tmp_path, monkeypatch):
    """Done cells skip the episode engine entirely on resume."""
    from gamekit import engine

    from bench.matrix.runner import run_matrix

    path = tmp_path / "mx.jsonl"
    kw = dict(meta_cases=3, meta_budget=8)
    m = _manifest({"lane": "matrix", **kw})

    ck = Checkpoint.open(path, m, env=_ENV)
    cells1 = run_matrix(
        ["search"],
        ["playbook"],
        ["naive", "oracle"],
        checkpoint=ck,
        **kw,
    )

    def boom(*a, **k):
        raise AssertionError("episode re-run on resume")

    monkeypatch.setattr(engine, "run_episode", boom)
    ck2 = Checkpoint.open(path, m, env=_ENV)
    cells2 = run_matrix(
        ["search"],
        ["playbook"],
        ["naive", "oracle"],
        checkpoint=ck2,
        **kw,
    )
    assert [c.as_record() for c in cells2] == [
        c.as_record() for c in cells1
    ]


def test_row_cell_record_roundtrip():
    """The record codec is lossless — the resume seam's contract."""
    from bench.matrix.runner import Cell
    from bench.players.runner import Row

    row = Row(
        board="meta",
        player="playbook",
        kind="heuristic",
        reward_ep=1.5,
        evals_ep=3.0,
        ap=0.5,
        cost=12.0,
        ratio=0.9,
        wall_s=0.3,
        n=4,
        notes="n",
    )
    assert (
        Row.from_record(row.as_record()).as_record() == row.as_record()
    )

    cell = Cell(
        "open",
        "playbook",
        "pay",
        claims=3,
        admitted=2,
        refuted=1,
        value=0.4,
        reward_ep=1.0,
        evals_ep=2.0,
        wall_s=0.1,
        notes="n",
    )
    back = Cell.from_record(json.loads(json.dumps(cell.as_record())))
    assert back.as_record() == cell.as_record()
    assert back.admit_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
#  provenance — enforced, not conventional
# ---------------------------------------------------------------------------


def test_require_provenance_refuses_incomplete_docs():
    doc = with_provenance({"suite": "t"}, seed=0, device="cpu")
    require_provenance(doc)  # passes

    for missing in ("generated_utc", "seed", "env"):
        bad = dict(doc)
        bad.pop(missing)
        with pytest.raises(ProvenanceError, match=missing):
            require_provenance(bad, where="doc")

    for field in ("git_sha", "git_dirty", "device", "timestamp_utc"):
        bad = dict(doc)
        bad["env"] = {k: v for k, v in doc["env"].items() if k != field}
        with pytest.raises(ProvenanceError, match=field):
            require_provenance(bad)


def test_with_provenance_seed_must_be_declared():
    with pytest.raises(TypeError, match="seed"):
        with_provenance({"a": 1})
    doc = with_provenance({"a": 1}, seed=None, env=_ENV)
    assert doc["seed"] is None  # an unseeded lane declares null


def test_table_record_carries_the_contract():
    from bench.benchkit.table import table_record

    rec = table_record(
        "matrix",
        [{"name": "open/pay", "metrics": {"value": 0.5}}],
        directions={"value": "up"},
        findings=[{"claim": "c", "verdict": "parity", "headline": "h"}],
        seed=7,
    )
    require_provenance(rec)
    assert rec["seed"] == 7
    assert rec["env"]["git_sha"]


def test_all_emitters_carry_the_contract(tmp_path):
    """scorecard / leaderboard / limits / matrix all emit the fields."""
    from bench.discovery.cases import DiscoveryCase
    from bench.discovery.runner import CaseResult
    from bench.discovery.scorecard import write_outputs as sc_write
    from bench.limits.runner import CaseLimits
    from bench.limits.table import write_outputs as lim_write
    from bench.matrix.runner import Cell
    from bench.matrix.table import write_outputs as mx_write
    from bench.players.leaderboard import write_outputs as lb_write
    from bench.players.runner import Row

    row = Row(board="meta", player="random", reward_ep=-0.5)
    payload = lb_write(
        [row],
        md_path=tmp_path / "lb.md",
        json_path=tmp_path / "lb.json",
        seed=0,
        device="cpu",
    )
    require_provenance(payload)
    assert payload["seed"] == 0

    cell = Cell("search", "playbook", "naive", claims=1, admitted=1)
    payload = mx_write(
        [cell],
        coverage=[cell],
        md_path=tmp_path / "mx.md",
        json_path=tmp_path / "mx.json",
        protocol={"seed": 3},
        device="cpu",
    )
    require_provenance(payload)
    assert payload["seed"] == 3  # falls back to protocol["seed"]

    case = CaseResult(name="c", title="t", category="x", citation="cit")
    payload = sc_write(
        [case],
        lane="fast",
        md_path=tmp_path / "sc.md",
        json_path=tmp_path / "sc.json",
        seed=None,
        device="cpu",
    )
    require_provenance(payload)
    assert payload["seed"] is None

    cl = CaseLimits(
        case=DiscoveryCase(
            name="c",
            title="t",
            category="x",
            citation="cit",
            source="s",
        )
    )
    payload = lim_write(
        [cl],
        lane="fast",
        md_path=tmp_path / "lim.md",
        json_path=tmp_path / "lim.json",
        seed=None,
        device="cpu",
    )
    require_provenance(payload)


# ---------------------------------------------------------------------------
#  determinism — identical inputs emit identical artifacts modulo ts
# ---------------------------------------------------------------------------


def test_emitters_are_deterministic_modulo_timestamps(tmp_path):
    """Two emissions of the same rows differ only in the date fields."""
    from bench.players.leaderboard import render_markdown, write_outputs
    from bench.players.runner import Row

    rows = [
        Row(board="meta", player="playbook", reward_ep=1.0, wall_s=0.2),
        Row(board="meta", player="random", reward_ep=-0.5, wall_s=0.1),
    ]
    p1 = write_outputs(
        rows,
        md_path=tmp_path / "a.md",
        json_path=tmp_path / "a.json",
        protocol={"seed": 0},
        device="cpu",
    )
    p2 = write_outputs(
        rows,
        md_path=tmp_path / "b.md",
        json_path=tmp_path / "b.json",
        protocol={"seed": 0},
        device="cpu",
    )
    for p in (p1, p2):
        p.pop("generated_utc")
        p["env"].pop("timestamp_utc")
    assert p1 == p2

    # the Markdown surface is byte-identical under a fixed env
    env = dict(p1["env"])
    env["timestamp_utc"] = "fixed"
    md1 = render_markdown(rows, env=env, protocol={"seed": 0})
    md2 = render_markdown(rows, env=env, protocol={"seed": 0})
    assert md1 == md2


def test_table_record_deterministic_modulo_timestamps():
    from bench.benchkit.table import table_record

    kw = dict(
        directions={"v": "up"},
        findings=[{"claim": "c", "verdict": "parity", "headline": "h"}],
        seed=0,
        env=_ENV,
    )
    r1 = table_record("s", [{"name": "a", "metrics": {"v": 1.0}}], **kw)
    r2 = table_record("s", [{"name": "a", "metrics": {"v": 1.0}}], **kw)
    r1.pop("generated_utc")
    r2.pop("generated_utc")
    assert r1 == r2
