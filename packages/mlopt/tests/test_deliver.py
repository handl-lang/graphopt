"""``mlopt.deliver`` tests — runner selection, verify gating, report
fields, CLI parsing.  CPU-only: no CUDA benchmarking, fake runners."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mlopt.deliver import _alias_spec, _bench, _bench_paired, deliver, select_runner
from mlopt.optimize import OptimizeReport

# ---------------------------------------------------------------------------
#  Fakes
# ---------------------------------------------------------------------------


class _FakeRunner:
    name = "fake_runner"

    def __init__(self, module_out: Any = None) -> None:
        self.module_out = module_out
        self.calls: list[Any] = []

    def apply(self, module: Any, example_input: Any, stats: dict) -> Any:
        self.calls.append((module, example_input))
        stats["fake_ran"] = True
        return self.module_out if self.module_out is not None else module


class _FakeOptz:
    """``Optimizer`` stand-in: ``lower`` returns scripted LowerResults."""

    def __init__(self, lowers: list[Any]) -> None:
        self.lowers = list(lowers)
        self.runner_seen: list[Any] = []

    def lower(self, res: Any, x: Any, **kw: Any) -> Any:
        self.runner_seen.append(kw.get("runner"))
        return self.lowers.pop(0)


def _lowered(module: Any, *, passed: bool | None = True) -> Any:
    verified = None if passed is None else SimpleNamespace(passed=passed, max_abs=0.0, max_rel=0.0)
    return SimpleNamespace(module=module, stats={}, verified=verified)


def _res(term: Any = None) -> Any:
    return SimpleNamespace(term=term, param_values={}, ir=None, stats={})


class _Callable:
    """A bench-able module: ``module(*args)`` returns a constant."""

    def __init__(self, out: Any = 1.0) -> None:
        self.out = out
        self.n = 0
        #: Scripted ``(min_ms, med_ms)`` for the fake paired bench.
        self.ms: tuple[float, float] | None = None

    def __call__(self, *args: Any) -> Any:
        self.n += 1
        return self.out


# ---------------------------------------------------------------------------
#  select_runner — CPU / no-fusion declines
# ---------------------------------------------------------------------------


def test_select_runner_cpu_inputs_no_cuda() -> None:
    runner, notes = select_runner(module=object(), term=None, inputs=(1, 2, 3), meta={})
    assert runner is None
    assert any("no CUDA device" in n for n in notes)
    assert any("region_fuse" in n for n in notes)


def test_select_runner_meta_alias_specs_validated() -> None:
    # CPU inputs → no cuda runner regardless; spec validation is unit-level.
    spec, err = _alias_spec({"alias_inputs": [0, 2]}, "alias_inputs")
    assert spec == frozenset({0, 2})
    assert err is None
    spec, err = _alias_spec({"alias_inputs": "bogus"}, "alias_inputs")
    # a string is iterable-of-str → positions must be ints
    assert spec is None and err is not None
    spec, err = _alias_spec({"alias_inputs": 3}, "alias_inputs")
    assert spec is None and err is not None
    spec, err = _alias_spec({"alias_inputs": True}, "alias_inputs")
    assert spec is True and err is None


def test_fusible_requires_term_and_irmodule() -> None:
    runner, notes = select_runner(
        module=object(),  # not an IRModule
        term=object(),  # a term object (unused — module check first? no: term check first)
        inputs=(1,),
        meta={},
    )
    # term is not None, module is not IRModule → fusion declined honestly
    assert runner is None
    assert any("IRModule" in n or "term" in n for n in notes)


def test_fusible_no_term() -> None:
    runner, notes = select_runner(module=object(), term=None, inputs=(1,), meta={})
    assert runner is None
    assert any("no extracted term" in n for n in notes)


# ---------------------------------------------------------------------------
#  deliver() — verify gating + measurement
# ---------------------------------------------------------------------------


def test_deliver_no_runner_measures_eager_only() -> None:
    eager_mod = _Callable()
    optz = _FakeOptz([])  # no second lower expected
    out = deliver(
        optz=optz,
        res=_res(),
        lowered_eager=_lowered(eager_mod),
        inputs=(1, 2),
        bench_iters=4,
    )
    assert out.runner_name is None
    assert out.lowered.module is eager_mod
    assert out.eager_med_ms is not None and out.eager_med_ms > 0
    assert out.delivered_med_ms is None
    assert optz.runner_seen == []  # never re-lowered
    assert any("eager lowering only" in n for n in out.notes)


def test_deliver_runner_verified_ships_and_times() -> None:
    eager_mod = _Callable()
    delivered_mod = _Callable()
    runner = _FakeRunner(module_out=delivered_mod)
    optz = _FakeOptz([_lowered(delivered_mod, passed=True)])
    out = deliver(
        optz=optz,
        res=_res(),
        lowered_eager=_lowered(eager_mod),
        inputs=(1,),
        meta={},
        bench_iters=4,
        runner=runner,  # caller-supplied: skips selection
    )
    assert out.runner_name == "fake_runner"
    assert out.lowered.module is delivered_mod
    assert out.delivered_med_ms is not None
    assert out.eager_med_ms is not None
    # _FakeOptz.lower doesn't call runner.apply — the real Optimizer
    # does; here we assert the runner was PASSED to lower.
    assert optz.runner_seen == [runner]


def test_deliver_runner_verify_failure_keeps_eager() -> None:
    eager_mod = _Callable()
    delivered_mod = _Callable()
    runner = _FakeRunner(module_out=delivered_mod)
    optz = _FakeOptz([_lowered(delivered_mod, passed=False)])
    out = deliver(
        optz=optz,
        res=_res(),
        lowered_eager=_lowered(eager_mod),
        inputs=(1,),
        bench_iters=4,
        runner=runner,
        verify=True,
    )
    assert out.lowered.module is eager_mod  # verified eager ships
    # the runner was DECLINED — runner_name must not imply it shipped
    assert out.runner_name is None
    assert out.delivered_med_ms is None
    assert any("FAILED verify" in n for n in out.notes)


def test_deliver_missing_verify_report_declines_under_verify() -> None:
    """verify=True + NO verify report is a decline too (deliver_game rule)."""
    eager_mod = _Callable()
    optz = _FakeOptz([_lowered(_Callable(), passed=None)])
    out = deliver(
        optz=optz,
        res=_res(),
        lowered_eager=_lowered(eager_mod),
        inputs=(1,),
        bench_iters=2,
        runner=_FakeRunner(),
        verify=True,
    )
    assert out.lowered.module is eager_mod
    assert out.runner_name is None
    assert out.delivered_med_ms is None
    assert any("no verify report" in n for n in out.notes)


def test_deliver_no_verify_notes_unverified() -> None:
    eager_mod = _Callable()
    optz = _FakeOptz([_lowered(_Callable(), passed=None)])
    out = deliver(
        optz=optz,
        res=_res(),
        lowered_eager=_lowered(eager_mod),
        inputs=(1,),
        bench_iters=3,
        runner=_FakeRunner(),
        verify=False,
    )
    assert out.delivered_med_ms is not None
    assert any("WITHOUT the equivalence gate" in n for n in out.notes)


def test_deliver_lower_exception_keeps_eager() -> None:
    class _Boom:
        def lower(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("boom")

    eager_mod = _Callable()
    out = deliver(
        optz=_Boom(),
        res=_res(),
        lowered_eager=_lowered(eager_mod),
        inputs=(1,),
        bench_iters=2,
        runner=_FakeRunner(),
    )
    assert out.lowered.module is eager_mod
    assert any("lower failed" in n for n in out.notes)


def test_deliver_stats_ledger_notes() -> None:
    delivered = SimpleNamespace(
        module=_Callable(),
        stats={
            "cuda_graph": True,
            "cuda_graph_alias": True,
            "region_fuse": {
                "applied": True,
                "regions_fused": 3,
                "ops_fused": 27,
                "distinct_kernels": 2,
                "fused_vs_unfused": {"max_rel": 1e-6},
            },
        },
        verified=SimpleNamespace(passed=True, max_abs=0.0, max_rel=0.0),
    )
    out = deliver(
        optz=_FakeOptz([delivered]),
        res=_res(),
        lowered_eager=_lowered(_Callable()),
        inputs=(1,),
        bench_iters=2,
        runner=_FakeRunner(),
    )
    assert any("capture armed" in n and "alias-aware" in n for n in out.notes)
    assert any("region_fuse: applied" in n for n in out.notes)


def test_deliver_stats_ledger_notes_cuda_graph_skipped() -> None:
    """A compile+graph config whose inner compile won precedence
    records the skip — the notes must not imply an armed graph."""
    delivered = SimpleNamespace(
        module=_Callable(),
        stats={
            "compiled": True,
            "cuda_graph": False,
            "cuda_graph_skipped": "inner runner already compiled",
        },
        verified=SimpleNamespace(passed=True, max_abs=0.0, max_rel=0.0),
    )
    out = deliver(
        optz=_FakeOptz([delivered]),
        res=_res(),
        lowered_eager=_lowered(_Callable()),
        inputs=(1,),
        bench_iters=2,
        runner=_FakeRunner(),
    )
    assert any("cuda_graph: skipped" in n for n in out.notes)
    assert not any("capture armed" in n for n in out.notes)


# ---------------------------------------------------------------------------
#  _bench
# ---------------------------------------------------------------------------


def test_bench_min_median() -> None:
    mod = _Callable()
    mn, md = _bench(mod, (1, 2), iters=5, warmup=1)
    assert mn > 0 and md >= mn
    assert mod.n == 6  # warmup + iters


def test_bench_paired_alternates_rounds_and_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paired protocol: one eager + one config bench per round,
    order shuffled, warmup only in round 0; score = min over
    per-round mins, med = median of per-round meds."""
    import mlopt.deliver as dl

    calls: list[tuple[Any, int]] = []

    def fake_bench(module: Any, inputs: Any, iters: int, **kw: Any):
        calls.append((module, int(kw.get("warmup", 0))))
        return module.ms

    monkeypatch.setattr(dl, "_bench", fake_bench)
    eager, config = _Callable(), _Callable()
    eager.ms = (10.0, 11.0)
    config.ms = (4.0, 5.0)
    out = _bench_paired(eager, config, (1,), iters=3, rounds=4, warmup=2, seed=7)
    # 4 rounds x 2 sides = 8 timed blocks; each round pairs the two.
    assert len(calls) == 8
    seq = [m for m, _w in calls]
    for r in range(4):
        assert set(seq[2 * r : 2 * r + 2]) == {eager, config}
    # warmup rides only the first round's two blocks.
    assert [w for _m, w in calls] == [2, 2, 0, 0, 0, 0, 0, 0]
    assert out["eager"]["min_ms"] == 10.0
    assert out["eager"]["med_ms"] == 11.0
    assert out["config"]["min_ms"] == 4.0
    assert out["config"]["med_ms"] == 5.0
    assert len(out["eager"]["rounds"]) == 4
    assert out["eager"]["rounds"][0] == (10.0, 11.0)


def test_bench_paired_min_over_round_mins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A contention spike in one round can't sink the min score."""
    import mlopt.deliver as dl

    eager, config = _Callable(), _Callable()
    eager.ms = (10.0, 10.0)
    # config: round 0 slow (spike), later rounds fast.
    config_samples = iter([(9.0, 9.0), (3.0, 3.0), (3.0, 3.0), (3.0, 3.0)])

    def fake_bench(module: Any, inputs: Any, iters: int, **kw: Any):
        if module is config:
            return next(config_samples)
        return module.ms

    monkeypatch.setattr(dl, "_bench", fake_bench)
    out = _bench_paired(eager, config, (1,), iters=1, rounds=4, warmup=0, seed=0)
    assert out["config"]["min_ms"] == 3.0  # spike round doesn't count
    assert out["config"]["med_ms"] == 3.0


# ---------------------------------------------------------------------------
#  OptimizeReport.render
# ---------------------------------------------------------------------------


def test_report_render_deliver_line() -> None:
    r = OptimizeReport(model_ref="m", backend="torch")
    r.eager_ms = 10.0
    r.eager_min_ms = 9.5
    r.delivered_ms = 4.0
    r.delivered_min_ms = 3.9
    r.speedup = 2.5
    r.runner_name = "cuda_graph"
    out = r.render()
    assert "deliver:" in out
    assert "runner=cuda_graph" in out
    assert "speedup=2.50x" in out
    assert "eager only" not in out


def test_report_render_eager_only() -> None:
    r = OptimizeReport(model_ref="m", backend="torch")
    r.eager_ms = 3.0
    r.runner_name = None
    out = r.render()
    assert "delivered=eager only" in out
    assert "runner=none" in out


def test_report_render_no_deliver_unchanged() -> None:
    r = OptimizeReport(model_ref="m", backend="torch")
    out = r.render()
    assert "deliver:" not in out


# ---------------------------------------------------------------------------
#  CLI parsing
# ---------------------------------------------------------------------------


def test_parser_deliver_flags() -> None:
    from mlopt.cli import build_parser

    args = build_parser().parse_args(["optimize", "whisper:x", "--deliver"])
    assert args.deliver is True
    assert args.bench_iters == 50
    args = build_parser().parse_args(["optimize", "whisper:x", "--deliver", "--bench-iters", "7"])
    assert args.bench_iters == 7
    args = build_parser().parse_args(["optimize", "whisper:x"])
    assert args.deliver is False
