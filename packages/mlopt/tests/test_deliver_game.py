"""``mlopt.deliver_game`` + ``mlopt.play`` tests — board/move/referee
logic, registry seats, CLI parsing.  CPU-only: fake timers, fake
runners, no CUDA benchmarking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import mlopt.deliver_game as dg
import mlopt.play
from mlopt.deliver_game import (
    CONFIGS,
    DeliveryBoard,
    DeliveryState,
    GameUnavailable,
    _Context,
    _eligible,
    _evaluate,
    deliver_game,
    legal_actions,
)

# ---------------------------------------------------------------------------
#  Fakes
# ---------------------------------------------------------------------------


class _Mod:
    """A bench-able module; ``ms`` is what the fake timer reports."""

    def __init__(self, ms: tuple[float, float] = (1.0, 1.0)) -> None:
        self.ms = ms
        self.n = 0

    def __call__(self, *args: Any) -> Any:
        self.n += 1
        return 1.0


class _Runner:
    """Sentinel runner — ``name`` identifies the config under test."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOptz:
    """``Optimizer`` stand-in: lowers scripted per runner name."""

    def __init__(self, script: dict[str, Any]) -> None:
        # script: runner name -> (passed, ms) | Exception | "raise"
        self.script = script
        self.seen: list[str] = []

    def lower(self, res: Any, x: Any, **kw: Any) -> Any:
        runner = kw.get("runner")
        name = getattr(runner, "name", type(runner).__name__)
        self.seen.append(name)
        spec = self.script.get(name, (True, (2.0, 2.0)))
        if spec == "raise":
            raise RuntimeError("lower boom")
        if isinstance(spec, Exception):
            raise spec
        passed, ms = spec
        verified = (
            None if passed is None else SimpleNamespace(passed=passed, max_abs=0.0, max_rel=0.0)
        )
        return SimpleNamespace(module=_Mod(ms), stats={}, verified=verified)


def _res(term: Any = None) -> Any:
    return SimpleNamespace(term=term, param_values={}, ir=None, stats={})


def _lowered(module: Any, *, passed: bool | None = True) -> Any:
    verified = None if passed is None else SimpleNamespace(passed=passed, max_abs=0.0, max_rel=0.0)
    return SimpleNamespace(module=module, stats={}, verified=verified)


def _ctx(
    optz: Any,
    *,
    eager_ms: tuple[float, float] = (10.0, 10.0),
    verify: bool = True,
    **flags: Any,
) -> _Context:
    ctx = _Context(
        optz=optz,
        res=_res(),
        lowered_eager=_lowered(_Mod(eager_ms)),
        inputs=(1, 2),
        meta={},
        bench_iters=2,
        verify=verify,
    )
    ctx.fusible = flags.get("fusible", True)
    ctx.cuda = flags.get("cuda", True)
    ctx.compile_ok = flags.get("compile_ok", True)
    ctx.alias = flags.get("alias", frozenset({0}))
    ctx.mutable = flags.get("mutable")
    return ctx


def _fake_runner_for(_ctx: Any, name: str) -> Any:
    return _Runner(name)


def _fake_bench(module: Any, inputs: Any, iters: int, **kw: Any):
    return module.ms


def _fake_bench_paired(eager_mod: Any, config_mod: Any, inputs: Any, iters: int, **kw: Any):
    """Paired-protocol fake — both sides report their scripted ms,
    ``rounds`` pairs of per-round (min, med) for the audit trail."""
    e, c = eager_mod.ms, config_mod.ms
    rounds = int(kw.get("rounds", dg.PAIRED_ROUNDS))
    return {
        "eager": {"min_ms": e[0], "med_ms": e[1], "rounds": [(e[0], e[1])] * rounds},
        "config": {"min_ms": c[0], "med_ms": c[1], "rounds": [(c[0], c[1])] * rounds},
    }


@pytest.fixture
def game_env(monkeypatch: pytest.MonkeyPatch):
    """Patch the runner builder + timers so configs are pure fakes."""
    monkeypatch.setattr(dg, "_runner_for", _fake_runner_for)
    monkeypatch.setattr(dg, "_bench", _fake_bench)
    monkeypatch.setattr(dg, "_bench_paired", _fake_bench_paired)
    mlopt.play.register()
    return monkeypatch


# ---------------------------------------------------------------------------
#  Eligibility — the honest option space
# ---------------------------------------------------------------------------


def test_eligible_cpu_only_offers_compile() -> None:
    ctx = _Context(
        optz=None,
        res=_res(),
        lowered_eager=_lowered(object()),
        inputs=(1, 2),
        meta={},
        bench_iters=1,
        verify=True,
    )
    dg._probe(ctx)
    assert ctx.cuda is False
    assert ctx.fusible is False  # no term / not IRModule
    # kit_torch is installed → compile is the only honest CPU config
    assert _eligible(ctx) == ("compile",)


def test_eligible_full_space_with_alias() -> None:
    ctx = _ctx(None)
    assert _eligible(ctx) == CONFIGS
    # no alias spec → the two alias configs drop out
    ctx2 = _ctx(None, alias=None)
    assert "graph-alias" not in _eligible(ctx2)
    assert "fuse+graph-alias" not in _eligible(ctx2)


def test_legal_actions_pending_plus_stop() -> None:
    st = DeliveryState(pending=("fuse", "graph"), steps=1)
    assert legal_actions(st) == ("fuse", "graph", "stop")


# ---------------------------------------------------------------------------
#  Board mechanics — evaluate, decline, commit
# ---------------------------------------------------------------------------


def test_step_evaluates_config_and_records(game_env) -> None:
    optz = _FakeOptz({"compile": (True, (4.0, 4.0))})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=4)
    st, rep = board.step("compile")
    assert rep.reward == 0.0 and not rep.terminal
    assert st.results["compile"]["med_ms"] == 4.0
    assert st.results["compile"]["runner_name"] == "compile"
    assert optz.seen == ["compile"]


def test_step_unknown_action_declined(game_env) -> None:
    board = DeliveryBoard(_ctx(_FakeOptz({})), player_name="t", budget=4)
    st, rep = board.step("bogus")
    assert rep.applied is False
    assert st.results == {}


def test_verify_failure_is_decline_never_scored(game_env) -> None:
    # the fastest config FAILS verify — it must not appear as a winner
    optz = _FakeOptz({"compile": (False, (0.5, 0.5))})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=4)
    board.step("compile")
    st = board.state
    assert st.results["compile"].get("med_ms") is None
    assert "FAILED verify" in st.results["compile"]["reason"]
    assert st.best_config is None
    out = board.committed()
    assert out.lowered.module.ms == (10.0, 10.0)  # eager ships
    assert out.mechanism == "game"
    assert out.runner_name is None


def test_missing_verify_gate_is_decline_under_verify(game_env) -> None:
    # verified=None (no gate reported) + verify=True → cannot score
    optz = _FakeOptz({"compile": (None, (0.5, 0.5))})
    board = DeliveryBoard(_ctx(optz, verify=True), player_name="t", budget=4)
    board.step("compile")
    rec = board.state.results["compile"]
    assert rec.get("med_ms") is None
    assert "no verify report" in rec["reason"]


def test_commit_picks_fastest_verified(game_env) -> None:
    optz = _FakeOptz(
        {
            "fuse+graph-alias": (True, (8.0, 8.0)),
            "graph": (True, (3.0, 3.0)),
            "compile": (True, (5.0, 5.0)),
        }
    )
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=9)
    for name in ("fuse+graph-alias", "graph", "compile"):
        board.step(name)
    st, rep = board.step("stop")
    # reward = committed speedup − MOVE_COST·steps (3 config moves)
    assert rep.reward == pytest.approx(10.0 / 3.0 - dg.MOVE_COST * 3)
    assert rep.detail["steps"] == 3
    assert rep.detail["move_cost"] == pytest.approx(dg.MOVE_COST * 3)
    assert st.best_config == "graph"
    out = board.committed()
    assert out.runner_name == "graph"
    assert out.delivered_med_ms == 3.0
    assert out.eager_med_ms == 10.0
    assert out.mechanism == "game"
    assert any("mechanism=game" in n for n in out.notes)


def test_commit_eager_wins_ties_and_slowdowns(game_env) -> None:
    # every config verifies but is slower than eager → eager ships
    optz = _FakeOptz({})  # default (True, (2.0, 2.0)) — patch ms above eager?
    optz.script = {n: (True, (12.0, 12.0)) for n in CONFIGS}
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=9)
    for name in board.state.pending:
        board.step(name)
    board.step("stop")
    out = board.committed()
    assert out.runner_name is None
    assert out.delivered_med_ms is None
    assert any("does not beat eager" in n for n in out.notes)


def test_lower_exception_is_decline(game_env) -> None:
    optz = _FakeOptz({"compile": "raise"})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=4)
    board.step("compile")
    rec = board.state.results["compile"]
    assert rec.get("med_ms") is None and "lower failed" in rec["reason"]


# ---------------------------------------------------------------------------
#  CUDA-context poisoning — a sticky fault must not corrupt siblings
# ---------------------------------------------------------------------------


def test_is_context_poison_markers() -> None:
    assert dg._is_context_poison(RuntimeError("CUDA error: device-side assert triggered"))
    assert dg._is_context_poison(
        RuntimeError("CUDA error: an illegal memory access was encountered")
    )
    # a plain failure is NOT context-poisoning — siblings still run
    assert not dg._is_context_poison(RuntimeError("lower boom"))
    assert not dg._is_context_poison(RuntimeError("CUDA out of memory. Tried to allocate"))


def test_context_poison_skips_sibling_configs(game_env) -> None:
    """The arch:t5-small drain bug: a device-side assert poisons the
    CUDA context, so every LATER config's ``lower`` fails with the
    sticky error — recording bogus declines.  Once detected, siblings
    are recorded as explicit skips and never attempted."""
    poison = RuntimeError(
        "CUDA error: device-side assert triggered\nReturning 710 (CUDA_ERROR_ASSERT)"
    )
    optz = _FakeOptz(
        {
            "fuse+graph": poison,
            "graph": (True, (3.0, 3.0)),
            "compile": (True, (4.0, 4.0)),
        }
    )
    ctx = _ctx(optz)
    rec_poison = _evaluate(ctx, "fuse+graph")
    assert "lower failed" in rec_poison["reason"]
    assert ctx.poisoned_by == "fuse+graph"
    # siblings are skipped honestly — never lowered, never scored
    for name in ("graph", "compile"):
        rec = _evaluate(ctx, name)
        assert rec["skipped"] is True
        assert "poisoned" in rec["reason"]
        assert rec.get("med_ms") is None
    assert optz.seen == ["fuse+graph"]  # no sibling reached lower()


def test_non_poison_failure_does_not_skip(game_env) -> None:
    """An ordinary lower failure leaves the context usable — the next
    config still gets a real evaluation."""
    optz = _FakeOptz(
        {
            "fuse+graph": RuntimeError("verify declined"),
            "graph": (True, (3.0, 3.0)),
        }
    )
    ctx = _ctx(optz)
    _evaluate(ctx, "fuse+graph")
    assert ctx.poisoned_by is None
    rec = _evaluate(ctx, "graph")
    assert rec.get("med_ms") == 3.0
    assert "skipped" not in rec


def test_board_records_skip_not_bogus_decline(game_env) -> None:
    """End-to-end through the board: the poisoned config declines,
    later configs record ``skipped`` — the oracle's answer key keeps
    honest labels instead of a wall of inherited ``lower failed``."""
    poison = RuntimeError("CUDA error: device-side assert triggered")
    optz = _FakeOptz({"fuse+graph": poison})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=9)
    board.step("fuse+graph")
    board.step("graph")
    board.step("compile")
    st = board.state
    assert "lower failed" in st.results["fuse+graph"]["reason"]
    for name in ("graph", "compile"):
        assert st.results[name]["skipped"] is True
        assert st.results[name].get("med_ms") is None
    # and the pick logic sees no phantom measurements
    assert st.best_config is None


def test_unverified_config_flagged(game_env) -> None:
    optz = _FakeOptz({"compile": (None, (4.0, 4.0))})
    board = DeliveryBoard(_ctx(optz, verify=False), player_name="t", budget=4)
    board.step("compile")
    out = board.committed()
    assert out.runner_name == "compile"
    assert any("WITHOUT the equivalence gate" in n for n in out.notes)


def test_evaluate_directly(game_env) -> None:
    optz = _FakeOptz({"graph": (True, (2.0, 2.5))})
    ctx = _ctx(optz)
    rec = _evaluate(ctx, "graph")
    assert rec["med_ms"] == 2.5 and rec["verified"] is True


# ---------------------------------------------------------------------------
#  Paired-round protocol — eager re-measured per evaluation (autopsy §4)
# ---------------------------------------------------------------------------


def test_evaluate_records_paired_eager(game_env) -> None:
    """The referee record carries BOTH sides of the paired bench —
    the config's score AND the eager median from the same rounds."""
    optz = _FakeOptz({"graph": (True, (3.0, 3.5))})
    ctx = _ctx(optz)  # construction eager baseline (10, 10)
    rec = _evaluate(ctx, "graph")
    assert rec["min_ms"] == 3.0 and rec["med_ms"] == 3.5
    assert rec["eager_min_ms"] == 10.0 and rec["eager_med_ms"] == 10.0
    assert rec["bench"]["rounds"] == dg.PAIRED_ROUNDS
    assert len(rec["rounds"]) == dg.PAIRED_ROUNDS
    assert rec["rounds"][0] == {"eager": [10.0, 10.0], "config": [3.0, 3.5]}


def test_paired_eager_baseline_gates_commit(game_env, monkeypatch) -> None:
    """Drift defense: a config that beats the CONSTRUCTION eager but
    not its own paired-round eager must NOT ship — the comparison is
    against the re-measured baseline, not the stale one."""

    # construction baseline says eager=10ms; paired rounds say 8ms.
    def drifted_paired(eager_mod, config_mod, inputs, iters, **kw):
        out = _fake_bench_paired(eager_mod, config_mod, inputs, iters, **kw)
        out["eager"] = {"min_ms": 8.0, "med_ms": 8.0, "rounds": out["eager"]["rounds"]}
        return out

    monkeypatch.setattr(dg, "_bench_paired", drifted_paired)
    optz = _FakeOptz({"graph": (True, (9.0, 9.0))})  # 9ms: beats 10, not 8*margin
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=4)
    board.step("graph")
    st = board.state
    # 9.0 * 1.02 = 9.18 >= paired eager 8.0 → no committed win
    assert st.committed_speedup == 1.0
    out = board.committed()
    assert out.runner_name is None
    assert out.eager_med_ms == 8.0  # reported baseline is the paired one


def test_construction_eager_baseline_gates_commit(game_env, monkeypatch) -> None:
    """``eager_baseline="construction"`` scores every config against
    the ONE board-construction eager median — the de-noised referee
    metric.  Same drifted-paired setup as the paired-mode test, but
    here the config DOES ship: 9ms beats the shared 10ms baseline
    even though its own paired eager read 8ms."""

    def drifted_paired(eager_mod, config_mod, inputs, iters, **kw):
        out = _fake_bench_paired(eager_mod, config_mod, inputs, iters, **kw)
        out["eager"] = {"min_ms": 8.0, "med_ms": 8.0, "rounds": out["eager"]["rounds"]}
        return out

    monkeypatch.setattr(dg, "_bench_paired", drifted_paired)
    optz = _FakeOptz({"graph": (True, (9.0, 9.0))})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=4, eager_baseline="construction")
    board.step("graph")
    st = board.state
    # paired record still carries its own eager for audit …
    assert st.results["graph"]["eager_med_ms"] == 8.0
    # … but the gate/reward read the shared construction baseline.
    assert st.eager_baseline == "construction"
    assert st.committed_speedup == pytest.approx(10.0 / 9.0)
    out = board.committed()
    assert out.runner_name == "graph"
    assert out.eager_med_ms == 10.0  # reported baseline is the shared one


def test_eager_baseline_validation(game_env) -> None:
    with pytest.raises(ValueError, match="eager_baseline"):
        DeliveryBoard(_ctx(_FakeOptz({})), player_name="t", budget=1, eager_baseline="bogus")


def test_lowered_by_config_bounded_to_argmin(game_env) -> None:
    """Only the current argmin's lowered module is retained — each
    LowerResult pins weights, so the pool can't grow unbounded."""
    optz = _FakeOptz(
        {
            "fuse+graph-alias": (True, (8.0, 8.0)),
            "graph": (True, (3.0, 3.0)),
            "compile": (True, (5.0, 5.0)),
        }
    )
    ctx = _ctx(optz)
    board = DeliveryBoard(ctx, player_name="t", budget=9)
    board.step("fuse+graph-alias")
    assert set(ctx.lowered_by_config) == {"fuse+graph-alias"}
    board.step("graph")
    assert set(ctx.lowered_by_config) == {"graph"}  # argmin moved
    board.step("compile")  # slower than graph — still only graph kept
    assert set(ctx.lowered_by_config) == {"graph"}
    out = board.committed()
    assert out.runner_name == "graph"  # commit still finds its module


# ---------------------------------------------------------------------------
#  Players through the registry
# ---------------------------------------------------------------------------


def test_registry_seats_deliver_players() -> None:
    mlopt.play.register()
    from gamekit.players import registry

    for name in ("random", "playbook", "linear", "linear-greedy", "linear-cold"):
        assert "deliver" in registry.spec(name).boards
        arm = registry.make(name, "deliver", seed=0)
        player = arm(0)
        assert callable(player)


def test_playbook_prefers_config_order_and_stops_on_win(game_env) -> None:
    from gamekit.engine import run_episode
    from gamekit.players import registry

    optz = _FakeOptz({"fuse+graph-alias": (True, (9.0, 9.0))})
    board = DeliveryBoard(_ctx(optz), player_name="playbook", budget=9)
    traj = run_episode(board, registry.make("playbook", "deliver")(0), 9)
    # playbook drains in CONFIGS order: fuse+graph-alias first (9ms —
    # not >1% better than eager 10ms? 10/9=1.11 → stops after first win)
    assert traj.reports[0].action == "fuse+graph-alias"
    assert traj.reports[-1].terminal
    term = traj.terminal
    assert term is not None
    assert term.detail["best_config"] == "fuse+graph-alias"


def test_playbook_stops_when_pool_exhausted(game_env) -> None:
    from gamekit.engine import run_episode
    from gamekit.players import registry

    optz = _FakeOptz({})  # everything ties at 2ms — wait, beats eager!
    # make everything SLOW so playbook drains without early stop
    optz.script = {n: (True, (20.0, 20.0)) for n in CONFIGS}
    board = DeliveryBoard(_ctx(optz), player_name="playbook", budget=9)
    traj = run_episode(board, registry.make("playbook", "deliver")(0), 9)
    actions = [r.action for r in traj.reports]
    assert actions == [*CONFIGS, "stop"] or actions[-1] == "stop"
    assert set(actions[:-1]) == set(CONFIGS)


def test_random_arm_plays_legal(game_env) -> None:
    from gamekit.players import registry

    player = registry.make("random", "deliver", seed=3)(0)
    st = DeliveryState(pending=("fuse", "graph"), steps=0)
    for _ in range(20):
        a = player(st)
        assert a in ("fuse", "graph", "stop")


def test_linear_arm_runs_cold(game_env) -> None:
    from gamekit.players import registry

    for name in ("linear", "linear-greedy", "linear-cold"):
        player = registry.make(name, "deliver", seed=0)(0)
        st = DeliveryState(pending=("compile",), steps=0)
        a = player(st)
        assert a in ("compile", "stop")


# ---------------------------------------------------------------------------
#  deliver_game end-to-end + honest fallback
# ---------------------------------------------------------------------------


def test_deliver_game_end_to_end(game_env) -> None:
    optz = _FakeOptz({"compile": (True, (4.0, 4.0))})
    out = deliver_game(
        optz=optz,
        res=_res(),
        lowered_eager=_lowered(_Mod((10.0, 10.0))),
        inputs=(1,),
        bench_iters=2,
        verify=True,
        budget=6,
        player="playbook",
    )
    assert out.mechanism == "game"
    assert out.runner_name == "compile"
    assert out.delivered_med_ms == 4.0
    assert out.eager_med_ms == 10.0
    assert any("player=playbook" in n for n in out.notes)


def test_deliver_game_unknown_player_raises(game_env) -> None:
    with pytest.raises(GameUnavailable):
        deliver_game(
            optz=_FakeOptz({}),
            res=_res(),
            lowered_eager=_lowered(_Mod()),
            inputs=(1,),
            bench_iters=1,
            budget=2,
            player="no-such-player",
        )


def test_deliver_game_no_gamekit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dg, "_gamekit", lambda: False)
    with pytest.raises(GameUnavailable):
        deliver_game(
            optz=_FakeOptz({}),
            res=_res(),
            lowered_eager=_lowered(_Mod()),
            inputs=(1,),
            bench_iters=1,
            budget=2,
        )


# ---------------------------------------------------------------------------
#  optimize_model wiring + CLI parsing
# ---------------------------------------------------------------------------


def test_optimize_deliver_budget_runs_game(monkeypatch: pytest.MonkeyPatch) -> None:
    """--deliver-budget routes through the game; mechanism is named."""
    import mlopt_orchestrator.optimize as orch_optimize

    from mlopt.models import ModelSpec
    from mlopt.optimize import optimize_model

    class _Optz:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def search(self, *a: Any, **kw: Any) -> Any:
            return SimpleNamespace(term=None, n_enodes=0, param_values={})

        def lower(self, *a: Any, **kw: Any) -> Any:
            return _lowered(_Mod((5.0, 5.0)), passed=True)

    monkeypatch.setattr(orch_optimize, "Optimizer", _Optz)
    monkeypatch.setattr(dg, "_bench", _fake_bench)
    monkeypatch.setattr(dg, "_runner_for", _fake_runner_for)
    spec = ModelSpec(module=object(), inputs=(1,), meta={"ref": "t:x"})
    lowered, report = optimize_model(spec, deliver=True, bench_iters=2, deliver_budget=4)
    assert lowered is not None
    assert report.deliver_mechanism == "game"
    assert any("mechanism=game" in n for n in report.notes)
    assert "mechanism=game" in report.render()


def test_optimize_deliver_heuristic_flag_forces_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlopt_orchestrator.optimize as orch_optimize

    from mlopt.models import ModelSpec
    from mlopt.optimize import optimize_model

    class _Optz:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def search(self, *a: Any, **kw: Any) -> Any:
            return SimpleNamespace(term=None, n_enodes=0, param_values={})

        def lower(self, *a: Any, **kw: Any) -> Any:
            return _lowered(_Mod(), passed=True)

    monkeypatch.setattr(orch_optimize, "Optimizer", _Optz)
    spec = ModelSpec(module=object(), inputs=(1,), meta={"ref": "t:x"})
    _lowered_out, report = optimize_model(
        spec,
        deliver=True,
        bench_iters=2,
        deliver_budget=4,
        deliver_heuristic=True,
    )
    assert report.deliver_mechanism == "heuristic"


def test_parser_deliver_game_flags() -> None:
    from mlopt.cli import build_parser

    args = build_parser().parse_args(["optimize", "whisper:x", "--deliver"])
    assert args.deliver_budget == 0
    assert args.deliver_player == "playbook"
    assert args.deliver_heuristic is False
    args = build_parser().parse_args(
        [
            "optimize",
            "whisper:x",
            "--deliver",
            "--deliver-budget",
            "6",
            "--deliver-player",
            "random",
            "--deliver-heuristic",
        ]
    )
    assert args.deliver_budget == 6
    assert args.deliver_player == "random"
    assert args.deliver_heuristic is True


# ---------------------------------------------------------------------------
#  Probe-then-commit (v9) — screening moves, never committable
# ---------------------------------------------------------------------------


def test_probe_armed_legal_actions(game_env) -> None:
    board = DeliveryBoard(_ctx(_FakeOptz({})), player_name="t", budget=9, probe_armed=True)
    acts = legal_actions(board.state)
    assert "probe:fuse+graph-alias" in acts
    assert "probe:compile" in acts
    assert acts[-1] == "stop"
    # Unarmed board offers no probes.
    board2 = DeliveryBoard(_ctx(_FakeOptz({})), player_name="t", budget=9)
    assert not any(str(a).startswith("probe:") for a in legal_actions(board2.state))


def test_probe_move_screens_but_never_commits(game_env) -> None:
    from gamekit.engine import run_episode
    from gamekit.players import registry

    # ``graph`` is the fast config; everything else is slower than eager.
    optz = _FakeOptz({**{n: (True, (20.0, 20.0)) for n in CONFIGS}, "graph": (True, (4.0, 4.0))})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=12, probe_armed=True)
    traj = run_episode(board, registry.make("probe-scan", "deliver")(0), 12)
    st = board.state
    # Every pending config was probed; only the winner was confirmed.
    assert set(st.probes) == set(st.pending) | set(st.results)
    assert "graph" in st.results
    assert st.probe_steps == len(st.probes)
    # Probes are screening records — they never enter ``results``
    # unless confirmed, and the commit came from the full eval.
    term = traj.terminal
    assert term is not None
    assert term.detail["best_config"] == "graph"
    # Reward = speedup − MOVE_COST·steps − PROBE_MOVE_COST·probe_steps.
    expected = 2.5 - dg.MOVE_COST * 1 - dg.PROBE_MOVE_COST * len(st.probes)
    assert term.reward == pytest.approx(expected)


def test_probe_illegal_on_unarmed_board(game_env) -> None:
    board = DeliveryBoard(_ctx(_FakeOptz({})), player_name="t", budget=4)
    st, rep = board.step("probe:graph")
    assert rep.applied is False
    assert st.probe_steps == 1  # the wasted move is still priced
    assert not st.probes


def test_probe_double_probe_illegal(game_env) -> None:
    board = DeliveryBoard(_ctx(_FakeOptz({})), player_name="t", budget=9, probe_armed=True)
    board.step("probe:graph")
    st, rep = board.step("probe:graph")
    assert rep.applied is False
    assert list(st.probes) == ["graph"]


def test_probe_verify_failure_is_decline(game_env) -> None:
    optz = _FakeOptz({"graph": (False, (4.0, 4.0))})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=9, probe_armed=True)
    board.step("probe:graph")
    rec = board.state.probes["graph"]
    assert rec.get("med_ms") is None
    assert "verify" in rec["reason"].lower() or "FAILED" in rec["reason"]
    # A declined probe can never be confirmed into a win.
    assert board.state.best_config is None


def test_probe_arm_skips_confirm_when_screen_predicts_eager(game_env) -> None:
    from gamekit.engine import run_episode
    from gamekit.players import registry

    # Everything slower than eager → best probe speedup < margin.
    optz = _FakeOptz({n: (True, (20.0, 20.0)) for n in CONFIGS})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=12, probe_armed=True)
    traj = run_episode(board, registry.make("probe-commit", "deliver")(0), 12)
    st = board.state
    assert st.probe_steps == len(CONFIGS)
    assert not st.results  # no confirm was ever spent
    term = traj.terminal
    assert term is not None
    assert term.detail["best_config"] is None
    assert term.reward == pytest.approx(1.0 - dg.PROBE_MOVE_COST * len(CONFIGS))


def test_drain_arm_measures_all_then_stops(game_env) -> None:
    from gamekit.engine import run_episode
    from gamekit.players import registry

    optz = _FakeOptz({})  # default script: 2ms vs eager 10ms — all win
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=9)
    traj = run_episode(board, registry.make("drain", "deliver")(0), 9)
    actions = [r.action for r in traj.reports]
    assert actions == [*CONFIGS, "stop"]
    assert set(board.state.results) == set(CONFIGS)


def test_probe_arm_degrades_to_drain_on_unarmed_board(game_env) -> None:
    from gamekit.engine import run_episode
    from gamekit.players import registry

    optz = _FakeOptz({})
    board = DeliveryBoard(_ctx(optz), player_name="t", budget=9)  # NOT probe_armed
    traj = run_episode(board, registry.make("probe-commit", "deliver")(0), 9)
    actions = [r.action for r in traj.reports]
    assert actions == [*CONFIGS, "stop"]
    assert not board.state.probes
