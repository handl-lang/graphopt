"""Chunked ``export --verify-run`` — segmenter, emitter, runner, CLI arm.

The chunked arm (``mlopt.verify_chunks`` + the ``_verify_run_export``
wiring) splits an over-budget term DAG into sequential ``hnl-run``
invocations with F64 safetensors handoffs.  These tests cover the
pure planner/emitter surface without ``hnl-run``, the handoff runner
over a fake runner, and — when the binary is present — the real
``_verify_run_export`` chunked path end to end.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from mlopt_core.ir import Op, Param, TensorType, Var

from mlopt.verify_chunks import (
    CHUNK_BYTES_ENV,
    ChunkPlanError,
    chunk_byte_budget,
    emit_chunk_programs,
    estimate_feed_bytes,
    plan_verify_chunks,
    run_chunk_programs,
)

T = TensorType


def _chain_term(n: int = 4):
    """``add(mul(w2, mul(w1, x)), mul(w3, x))``-style DAG over *n* weights."""
    ws = [Param(f"w{i}", T((4,))) for i in range(n)]
    x = Var("x", T((4,)))
    acc = Op.make("mul", ws[0], x)
    for w in ws[1:]:
        acc = Op.make("add", acc, Op.make("mul", w, x))
    return acc, ws, x


def _feed(names: list[str]) -> dict:
    return {n: {"shape": [4], "data": [1.0, 2.0, 3.0, 4.0]} for n in names}


# ---------------------------------------------------------------------------
#  Budget knob
# ---------------------------------------------------------------------------


def test_chunk_byte_budget_default() -> None:
    assert chunk_byte_budget() == (1 << 30)


def test_chunk_byte_budget_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHUNK_BYTES_ENV, "2048")
    assert chunk_byte_budget() == 2048
    monkeypatch.setenv(CHUNK_BYTES_ENV, "0")
    assert chunk_byte_budget() == 0
    monkeypatch.setenv(CHUNK_BYTES_ENV, "not-a-number")
    with pytest.raises(ValueError, match=CHUNK_BYTES_ENV):
        chunk_byte_budget()


# ---------------------------------------------------------------------------
#  Planner
# ---------------------------------------------------------------------------


def test_plan_verify_chunks_segments_under_budget() -> None:
    term, ws, x = _chain_term(4)
    feed = _feed([w.name for w in ws] + ["x"])
    segs = plan_verify_chunks(term, feed, budget=100)
    assert len(segs) > 1
    # Topo order: every segment is non-empty and the union covers the DAG.
    assert all(segs)
    flat = [t for s in segs for t in s]
    assert len(flat) == len(set(map(id, flat)))


def test_plan_verify_chunks_deterministic() -> None:
    term, ws, x = _chain_term(4)
    feed = _feed([w.name for w in ws] + ["x"])
    a = plan_verify_chunks(term, feed, budget=100)
    b = plan_verify_chunks(term, feed, budget=100)
    assert [[t.op for t in s] for s in a] == [[t.op for t in s] for s in b]


def test_plan_verify_chunks_single_node_declines() -> None:
    x = Var("x", T((4,)))
    p = Param("w", T((4,)))
    term = Op.make("mul", p, x)
    feed = _feed(["w", "x"])
    with pytest.raises(ChunkPlanError):
        plan_verify_chunks(term, feed, budget=1 << 20)


def test_plan_verify_chunks_node_over_budget_declines() -> None:
    term, ws, x = _chain_term(2)
    feed = _feed([w.name for w in ws] + ["x"])
    # One node binds two 4-elem leaves = 64 bytes; budget 8 can never hold it.
    with pytest.raises(ChunkPlanError, match="single node"):
        plan_verify_chunks(term, feed, budget=8)


def test_estimate_feed_bytes() -> None:
    feed = _feed(["w", "x"])
    kinds = [("w", "param"), ("x", "var")]
    # 4 elems * 8 B each = 64 total; hnl entries count 0.
    assert estimate_feed_bytes(feed, kinds) == 64
    feed2 = dict(feed)
    feed2["x"] = {"hnl": 'tensor.load("f.safetensors", "x")'}
    assert estimate_feed_bytes(feed2, kinds) == 32


# ---------------------------------------------------------------------------
#  Chunk program emission
# ---------------------------------------------------------------------------


def test_emit_chunk_programs_two_chunks() -> None:
    term, ws, x = _chain_term(4)
    feed = _feed([w.name for w in ws] + ["x"])
    specs = emit_chunk_programs(term, feed, sidecar=None, budget=100)
    assert len(specs) >= 2
    # Non-final chunks name their handoff; the final chunk's result is
    # the verify output (no out contract).
    assert specs[-1].out_names == ()
    assert all(s.out_names and s.out_file for s in specs[:-1])
    # Later chunks consume earlier chunks' safetensors handoffs.
    assert any("tensor.load" in s.source for s in specs[1:])
    # Every chunk is a self-contained do-block program.
    for s in specs:
        assert s.source.startswith('from "host-tensor" import tensor\ndo {')
        assert s.source.rstrip().endswith("}")


def test_emit_chunk_programs_single_segment_returns_empty() -> None:
    term, ws, x = _chain_term(2)
    feed = _feed([w.name for w in ws] + ["x"])
    assert emit_chunk_programs(term, feed, sidecar=None, budget=1 << 20) == []


def test_emit_chunk_programs_sidecar_uses_tensor_load(tmp_path: Path) -> None:
    n = 5000  # >= _SIDECAR_MIN_ELEMS so Params ride the sidecar channel
    ws = [Param(f"w{i}", T((n,))) for i in range(3)]
    x = Var("x", T((n,)))
    acc = Op.make("mul", ws[0], x)
    for w in ws[1:]:
        acc = Op.make("add", acc, Op.make("mul", w, x))
    feed = {w.name: [1.0] * n for w in ws}
    feed["x"] = [2.0] * n
    st = tmp_path / "w.safetensors"
    # Each mul node binds 2 leaves = 80 KB; 100 KB fits one mul per
    # segment plus the leaf-free adds, forcing a multi-chunk plan.
    specs = emit_chunk_programs(
        term=acc, feed=feed, sidecar=st, sidecar_dtype="F64", budget=100_000
    )
    assert specs
    # Every chunk program loads weights through tensor.load on the
    # sidecar basename — the same spelling the artifact carries.
    for s in specs:
        assert f'tensor.load("{st.name}"' in s.source
    assert st.exists()


def test_emit_chunk_programs_missing_feed_declines() -> None:
    term, ws, x = _chain_term(4)
    feed = _feed([w.name for w in ws])  # "x" missing
    with pytest.raises(ChunkPlanError, match="missing leaves"):
        emit_chunk_programs(term, feed, sidecar=None, budget=100)


# ---------------------------------------------------------------------------
#  Handoff runner (fake runner — no hnl-run needed)
# ---------------------------------------------------------------------------


def _wire(vals: list[float], shape=(4,)) -> dict:
    return {"__tensor__": {"shape": list(shape), "data": list(vals)}}


class _FakeRunner:
    """Returns canned chunk results in order; records the programs."""

    def __init__(self, results: list):
        self.results = list(results)
        self.programs: list[str] = []

    def run(self, src: str):
        self.programs.append(src)
        return self.results.pop(0)


def test_run_chunk_programs_writes_f64_handoffs(tmp_path: Path) -> None:
    term, ws, x = _chain_term(4)
    feed = _feed([w.name for w in ws] + ["x"])
    specs = emit_chunk_programs(term, feed, sidecar=None, budget=100)
    assert len(specs) == 2
    n_out = len(specs[0].out_names)
    assert n_out >= 1
    runner = _FakeRunner(
        [
            [_wire([1.0] * 4), _wire([2.0] * 4)][:n_out],
            _wire([3.0] * 4),
        ]
    )
    written: list[Path] = []
    out = run_chunk_programs(runner, specs, tmp_path, written)
    assert out == _wire([3.0] * 4)
    assert len(runner.programs) == 2
    assert len(written) == 1
    path = written[0]
    assert path.name == specs[0].out_file
    assert path.exists()
    # safetensors container: u64le header len + JSON header + payload.
    raw = path.read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + hlen])
    assert set(header) == set(specs[0].out_names)
    for meta in header.values():
        assert meta["dtype"] == "F64"
        assert meta["shape"] == [4]


def test_run_chunk_programs_wire_count_mismatch(tmp_path: Path) -> None:
    term, ws, x = _chain_term(4)
    feed = _feed([w.name for w in ws] + ["x"])
    specs = emit_chunk_programs(term, feed, sidecar=None, budget=100)
    runner = _FakeRunner([_wire([1.0] * 4)])  # one wire, expects >=1... force mismatch
    # Force a real mismatch: return a scalar count different from out_names.
    runner.results = [[_wire([1.0] * 4)]]  # list of 1; if out_names > 1 this raises
    if len(specs[0].out_names) == 1:
        pytest.skip("segmenter produced a single live-out here")
    from mlopt.emit import KitEvalError

    with pytest.raises(KitEvalError, match="expected"):
        run_chunk_programs(runner, specs, tmp_path)


# ---------------------------------------------------------------------------
#  _verify_run_export integration (real hnl-run)
# ---------------------------------------------------------------------------


def _verify_spec(torch):
    import types

    w1t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    w2t = torch.tensor([2.0, 1.0, 4.0, 3.0], dtype=torch.float64)
    w3t = torch.tensor([0.5, 0.25, 1.5, 2.5], dtype=torch.float64)
    # term: add(mul(w2, mul(w1, x)), mul(w3, x)) = x * (w1*w2 + w3)
    spec = types.SimpleNamespace(
        module=lambda x_: x_ * (w1t * w2t + w3t),
        inputs=(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),),
        kwargs={},
        source=None,
    )
    return spec


def _verify_term():
    w1 = Param("w1", T((4,)))
    w2 = Param("w2", T((4,)))
    w3 = Param("w3", T((4,)))
    x = Var("x", T((4,)))
    n1 = Op.make("mul", w1, x)
    n2 = Op.make("mul", w2, n1)
    return Op.make("add", n2, Op.make("mul", w3, x))


def test_verify_run_chunked_end_to_end(capsys, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Over-budget weights route through chunked hnl-run evals and the
    unchanged torch-reference compare still gates the final logits."""
    import torch

    from mlopt.cli import _verify_run_export
    from mlopt.emit import find_hnl_run

    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")

    monkeypatch.setenv(CHUNK_BYTES_ENV, "100")
    vals = [1.0, 2.0, 3.0, 4.0]
    # Must match _verify_spec's weights: w1=w2=vals, w3=[0.5,0.25,1.5,2.5].
    feed = {"w1": vals, "w2": [2.0, 1.0, 4.0, 3.0], "w3": [0.5, 0.25, 1.5, 2.5]}
    rc = _verify_run_export(_verify_term(), feed, _verify_spec(torch), None)
    out = capsys.readouterr()
    assert rc == 0
    assert "verify-run: PASS" in out.out
    assert "chunks=" in out.out
    assert "chunk 1/" in out.err


def test_verify_run_chunked_with_sidecar(capsys, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Chunked arm + safetensors weight channel: weights ride
    ``tensor.load`` and handoffs share the sidecar's directory."""
    import torch

    from mlopt.cli import _verify_run_export
    from mlopt.emit import find_hnl_run

    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")

    n = 5000  # >= _SIDECAR_MIN_ELEMS → Params sidecar
    monkeypatch.setenv(CHUNK_BYTES_ENV, "100000")
    ws = [Param(f"w{i}", T((n,))) for i in range(3)]
    x = Var("x", T((n,)))
    acc = Op.make("mul", ws[0], x)
    for w in ws[1:]:
        acc = Op.make("add", acc, Op.make("mul", w, x))
    feed = {w.name: {"shape": [n], "data": [1.0] * n} for w in ws}
    feed["x"] = {"shape": [n], "data": [2.0] * n}
    wt = [torch.ones(n, dtype=torch.float64) for _ in ws]
    spec = type(
        "S",
        (),
        {
            "module": staticmethod(lambda x_: x_ * (wt[0] + wt[1] + wt[2])),
            "inputs": (torch.full((n,), 2.0, dtype=torch.float64),),
            "kwargs": {},
            "source": None,
        },
    )()
    st = tmp_path / "w.safetensors"
    rc = _verify_run_export(acc, feed, spec, None, sidecar=st, sidecar_dtype="F64")
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "verify-run: PASS" in out.out
    assert "chunks=" in out.out
    # The weight sidecar was (re)written; handoff scratch is gone.
    assert st.exists()
    assert not list(tmp_path.glob("mlopt-verify-chunk*-out.safetensors"))


def test_verify_run_chunked_plan_failure_falls_back(capsys, monkeypatch) -> None:  # noqa: ANN001
    """A term that admits no partition still runs the monolithic eval —
    the status quo, never a silent partial verification."""
    import torch

    from mlopt.cli import _verify_run_export
    from mlopt.emit import find_hnl_run

    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")

    # Budget below any single node's leaf bytes → ChunkPlanError →
    # monolithic fallback (which still verifies this tiny term).
    monkeypatch.setenv(CHUNK_BYTES_ENV, "8")
    feed = {"w1": [1.0, 2.0, 3.0, 4.0], "w2": [2.0, 1.0, 4.0, 3.0], "w3": [0.5, 0.25, 1.5, 2.5]}
    rc = _verify_run_export(_verify_term(), feed, _verify_spec(torch), None)
    out = capsys.readouterr()
    assert rc == 0
    assert "verify-run: PASS" in out.out
    assert "chunked plan unavailable" in out.err
    assert "chunks=" not in out.out
