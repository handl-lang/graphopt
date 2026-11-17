"""``mlopt.deliver_train`` + ``mlopt.weights`` tests — oracle picks,
imitation/REINFORCE mechanics, weight-table baking and warm-arm
loading.  CPU-only: fake timers, fake runners, flag-pinned boards."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import mlopt.deliver_game as dg
import mlopt.play
import mlopt.weights
from mlopt.deliver_game import CONFIGS, DeliveryState
from mlopt.deliver_train import (
    DeliveryCase,
    _block_means,
    _oracle_player,
    aggregate_rewards,
    baked_table,
    board_of,
    cold_train,
    held_out,
    oracle_drain,
    oracle_pick,
    train,
    weights_literal,
)

# ---------------------------------------------------------------------------
#  Fakes — same seams tests/test_deliver_game.py patches
# ---------------------------------------------------------------------------


class _Mod:
    """A bench-able module; ``ms`` is what the fake timer reports."""

    def __init__(self, ms: tuple[float, float] = (1.0, 1.0)) -> None:
        self.ms = ms

    def __call__(self, *args: Any) -> Any:
        return 1.0


class _Runner:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOptz:
    """``Optimizer`` stand-in: lowers scripted per runner name."""

    def __init__(self, script: dict[str, Any]) -> None:
        # script: runner name -> (passed, ms) | Exception | "raise"
        self.script = script

    def lower(self, res: Any, x: Any, **kw: Any) -> Any:
        runner = kw.get("runner")
        name = getattr(runner, "name", type(runner).__name__)
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


def _res() -> Any:
    return SimpleNamespace(term=None, param_values={}, ir=None, stats={})


def _lowered(module: Any, *, passed: bool | None = True) -> Any:
    verified = None if passed is None else SimpleNamespace(passed=passed, max_abs=0.0, max_rel=0.0)
    return SimpleNamespace(module=module, stats={}, verified=verified)


#: Full eligibility — every config offered (probe skipped via flags).
_FLAGS: dict[str, Any] = {
    "fusible": True,
    "cuda": True,
    "compile_ok": True,
    "alias": frozenset({0}),
    "mutable": None,
}


def _case(
    optz: Any,
    *,
    name: str = "case",
    eager_ms: tuple[float, float] = (10.0, 10.0),
    flags: dict[str, Any] | None = _FLAGS,
    verify: bool = True,
) -> DeliveryCase:
    return DeliveryCase(
        name=name,
        optz=optz,
        res=_res(),
        lowered_eager=_lowered(_Mod(eager_ms)),
        inputs=(1, 2),
        meta={},
        bench_iters=2,
        verify=verify,
        budget=9,
        flags=flags,
    )


def _all_slow() -> dict[str, Any]:
    return {n: (True, (20.0, 20.0)) for n in CONFIGS}


@pytest.fixture
def game_env(monkeypatch: pytest.MonkeyPatch):
    """Patch the runner builder + timers so configs are pure fakes."""
    monkeypatch.setattr(dg, "_runner_for", lambda _ctx, name: _Runner(name))
    monkeypatch.setattr(dg, "_bench", lambda module, inputs, iters, **kw: module.ms)

    def _paired(eager_mod, config_mod, inputs, iters, **kw):
        e, c = eager_mod.ms, config_mod.ms
        rounds = int(kw.get("rounds", dg.PAIRED_ROUNDS))
        return {
            "eager": {"min_ms": e[0], "med_ms": e[1], "rounds": [(e[0], e[1])] * rounds},
            "config": {"min_ms": c[0], "med_ms": c[1], "rounds": [(c[0], c[1])] * rounds},
        }

    monkeypatch.setattr(dg, "_bench_paired", _paired)
    return monkeypatch


# ---------------------------------------------------------------------------
#  board_of — fresh context per episode, flags skip the real probe
# ---------------------------------------------------------------------------


def test_board_of_flags_skip_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_ctx: Any) -> None:
        raise AssertionError("_probe must not run when flags are pinned")

    monkeypatch.setattr(dg, "_probe", boom)
    case = _case(_FakeOptz({}), flags={"cuda": False, "fusible": False, "compile_ok": False})
    board = board_of(case)
    st = board.observe()
    assert st.cuda is False and st.fusible is False
    assert "compile" in st.pending or "fuse" not in st.pending


def test_board_of_runs_probe_without_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(dg, "_probe", lambda ctx: seen.append(ctx))
    case = _case(_FakeOptz({}), flags=None)
    board_of(case)
    assert len(seen) == 1


def test_shape_facts_reach_state(game_env) -> None:
    """Workload shape (inputs/dtype/enodes/params) lands on the state."""

    class _P:
        ms = (10.0, 10.0)

        def parameters(self) -> Any:
            return [SimpleNamespace(numel=lambda: 7)]

    case = _case(_FakeOptz({}))
    case.lowered_eager = _lowered(_P())
    case.inputs = (SimpleNamespace(dtype="torch.bfloat16", device="cuda:0"),)
    case.res = SimpleNamespace(term=None, n_enodes=42)
    st = board_of(case).observe()
    assert st.n_inputs == 1
    assert st.lowprec is True
    assert st.n_enodes == 42
    assert st.n_params == 7


# ---------------------------------------------------------------------------
#  oracle_pick — the measured argmin
# ---------------------------------------------------------------------------


def test_oracle_pick_returns_fastest_verified(game_env) -> None:
    script = _all_slow()
    script["graph"] = (True, (3.0, 3.0))
    script["compile"] = (True, (4.0, 4.0))
    assert oracle_pick(_case(_FakeOptz(script))) == "graph"


def test_oracle_pick_none_when_eager_wins(game_env) -> None:
    assert oracle_pick(_case(_FakeOptz(_all_slow()))) is None


def test_oracle_pick_skips_verify_failure(game_env) -> None:
    script = _all_slow()
    script["graph"] = (False, (0.5, 0.5))  # fastest but fails verify
    script["compile"] = (True, (4.0, 4.0))
    assert oracle_pick(_case(_FakeOptz(script))) == "compile"


def test_oracle_pick_none_when_nothing_scores(game_env) -> None:
    script = {n: "raise" for n in CONFIGS}
    assert oracle_pick(_case(_FakeOptz(script))) is None


def test_oracle_drain_persists_per_config_records(game_env) -> None:
    """The answer key keeps its measurements — every config's
    min_ms/med_ms/verified/reason + paired eager baseline, JSON-safe."""
    script = _all_slow()
    script["graph"] = (True, (3.0, 3.5))
    script["compile"] = (False, (0.5, 0.5))  # fastest but fails verify
    pick, records = oracle_drain(_case(_FakeOptz(script)))
    assert pick == "graph"
    assert set(records) == set(CONFIGS)
    g = records["graph"]
    assert g["min_ms"] == 3.0 and g["med_ms"] == 3.5
    assert g["verified"] is True
    assert g["eager_med_ms"] == 10.0  # paired-round eager baseline
    assert g["runner_name"] == "graph"
    assert "rounds" in g and "bench" in g
    c = records["compile"]
    assert c.get("med_ms") is None and "FAILED verify" in c["reason"]
    # curated keys only — the runner stats blob never persists
    assert "stats" not in g
    import json

    json.dumps(records)  # the drain must be JSON-serializable as-is


def test_oracle_player_line(game_env) -> None:
    """pick → play it → stop; None → immediate stop."""
    st = DeliveryState(pending=("graph", "compile"), steps=0)
    go = _oracle_player("graph")
    assert go(st) == "graph"
    st2 = DeliveryState(pending=("compile",), steps=1, results={"graph": {}})
    assert go(st2) == "stop"
    assert _oracle_player(None)(st) == "stop"
    assert go(DeliveryState(done=True)) is None


# ---------------------------------------------------------------------------
#  train / cold_train — imitation + REINFORCE mechanics
# ---------------------------------------------------------------------------


def test_train_runs_and_bakes_weights(game_env) -> None:
    script_a = _all_slow()
    script_a["graph"] = (True, (3.0, 3.0))
    script_b = _all_slow()
    script_b["compile"] = (True, (4.0, 4.0))
    cases = [
        _case(_FakeOptz(script_a), name="a"),
        _case(_FakeOptz(script_b), name="b"),
    ]
    player, totals, imitation, picks, drains = train(
        cases, seed=0, epochs=1, imitate_epochs=1, budget=9
    )
    assert picks == {0: "graph", 1: "compile"}
    assert imitation["shown"] == 4  # 2 cases x (pick + stop)
    assert imitation["imitated"] == 4
    assert len(totals) == 2  # one REINFORCE episode per case
    assert all(t >= 0.0 for t in totals)
    # the persisted drain records ride along per case
    assert set(drains) == {0, 1}
    assert drains[0]["graph"]["med_ms"] == 3.0
    assert drains[1]["compile"]["med_ms"] == 4.0
    table = baked_table(player)
    assert table  # imitation moved at least one weight off zero
    assert set(table) <= set(player.FEATURES)
    literal = weights_literal(table)
    assert literal.startswith("DELIVERY_PLAYER_WEIGHTS")
    assert '"a:kind:graph"' in literal or '"a:' in literal


def test_train_empty_stream(game_env) -> None:
    player, totals, imitation, picks, drains = train(
        [], seed=0, epochs=1, imitate_epochs=1, budget=9
    )
    assert totals == [] and picks == {} and drains == {}
    assert player.weights_dict() == {n: 0.0 for n in player.FEATURES}


def test_train_imitates_only_oracle_positive_cases(game_env) -> None:
    """Autopsy §4: eager-argmin cases must not teach all-stop —
    imitation runs on oracle-positive cases only by default, while
    BOTH cases still feed REINFORCE and the drain cache."""
    pos_script = _all_slow()
    pos_script["graph"] = (True, (3.0, 3.0))
    cases = [
        _case(_FakeOptz(pos_script), name="pos"),
        _case(_FakeOptz(_all_slow()), name="neg"),  # eager argmin
    ]
    player, totals, imitation, picks, drains = train(
        cases, seed=0, epochs=1, imitate_epochs=1, budget=9
    )
    assert picks == {0: "graph", 1: None}
    # positive case demonstrates pick+stop (2 moves); the negative
    # case's bare-stop line is NOT imitated.
    assert imitation["shown"] == 2 and imitation["imitated"] == 2
    assert len(totals) == 2  # both cases still train REINFORCE
    assert set(drains) == {0, 1}
    # opt-out restores the old all-cases imitation stream
    _p, _t, imitation_all, _pk, _d = train(
        cases,
        seed=0,
        epochs=0,
        imitate_epochs=1,
        budget=9,
        imitate_positive_only=False,
    )
    # pos: pick+stop=2 moves; neg: bare stop=1 move
    assert imitation_all["shown"] == 3


def test_train_accepts_persisted_oracle_key(game_env, monkeypatch: pytest.MonkeyPatch) -> None:
    """``oracle_picks``/``oracle_drains`` skip the live drain — the
    corpus driver's persisted answer key drives imitation and RL."""
    import mlopt.deliver_train as dt

    def boom(_case: Any) -> Any:
        raise AssertionError("oracle_drain must not run with a persisted key")

    monkeypatch.setattr(dt, "oracle_drain", boom)
    script = _all_slow()
    script["graph"] = (True, (3.0, 3.0))
    cases = [_case(_FakeOptz(script), name="a"), _case(_FakeOptz(_all_slow()), name="b")]
    persisted = {"min_ms": 3.0, "med_ms": 3.0, "verified": True}
    player, totals, imitation, picks, drains = train(
        cases,
        seed=0,
        epochs=1,
        imitate_epochs=1,
        budget=9,
        oracle_picks={0: "graph", 1: None},
        oracle_drains={0: {"graph": dict(persisted)}, 1: {}},
    )
    assert picks == {0: "graph", 1: None}
    assert drains[0]["graph"]["med_ms"] == 3.0
    assert drains[1] == {}
    assert imitation["shown"] == 2  # only the positive case imitates
    assert len(totals) == 2
    assert baked_table(player)


def test_train_rejects_half_a_persisted_key(game_env) -> None:
    cases = [_case(_FakeOptz(_all_slow()))]
    with pytest.raises(ValueError, match="together"):
        train(cases, epochs=0, imitate_epochs=0, oracle_picks={0: None})


def test_cold_train_runs(game_env) -> None:
    script = _all_slow()
    script["graph"] = (True, (3.0, 3.0))
    cases = [_case(_FakeOptz(script))]
    player, totals = cold_train(cases, seed=0, epochs=2, budget=9)
    assert len(totals) == 2


def test_block_means() -> None:
    assert _block_means([1.0, 2.0, 3.0, 4.0], 2) == [1.5, 3.5]
    assert _block_means([], 4) == []


# ---------------------------------------------------------------------------
#  held_out — the arm table on a disjoint stream
# ---------------------------------------------------------------------------


def test_held_out_arm_table(game_env) -> None:
    script = _all_slow()
    script["graph"] = (True, (3.0, 3.0))
    cases = [_case(_FakeOptz(script), name="eval")]
    player, *_ = train(cases, seed=0, epochs=1, imitate_epochs=1, budget=9)
    means, errors, done = held_out(player, cases, seed=0, budget=9)
    table = means
    assert not errors
    assert all(n == 1 for n in done.values())
    for arm in (
        "random",
        "playbook",
        "cold",
        "shipped",
        "shipped-greedy",
        "trained",
        "trained-greedy",
    ):
        assert arm in table
    # playbook is deterministic under the fakes: drains in CONFIGS
    # order (4 moves to reach "graph"), stops right after it banks
    # 3.33x over eager 10ms; reward = speedup − MOVE_COST·steps
    assert table["playbook"] == pytest.approx(10.0 / 3.0 - dg.MOVE_COST * 4, abs=1e-3)
    # every arm produced a finite reward on the same case
    assert all(v >= 0.0 for v in table.values())


def test_held_out_survives_a_poisoned_cell(game_env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A context-poisoning cell must not kill the table — later arms
    are recorded as skipped, never scored."""
    import mlopt.deliver_train as dt

    script = _all_slow()
    script["graph"] = (True, (3.0, 3.0))
    cases = [_case(_FakeOptz(script), name="eval")]
    player, *_ = train(cases, seed=0, epochs=1, imitate_epochs=0, budget=9)

    real_board_of = dt.board_of
    calls = {"n": 0}

    def poison_once(case: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA error: device-side assert triggered")
        return real_board_of(case)

    monkeypatch.setattr(dt, "board_of", poison_once)
    means, errors, done = held_out(player, cases, seed=0, budget=9)
    # the first arm's only cell raised the poison signature; every
    # later arm was skipped without a single attempted cell.
    assert done["trained"] == 0
    assert all(n == 0 for n in done.values())
    assert "device-side assert" in errors["trained"]
    assert errors["random"].startswith("skipped: CUDA context poisoned")


# ---------------------------------------------------------------------------
#  Weight loading — cold today, warm when the literal lands
# ---------------------------------------------------------------------------


def test_weights_table_empty_and_spec_cold() -> None:
    """Honest cold fallback: {} → DELIVERY_SPEC.weights is None."""
    assert mlopt.weights.DELIVERY_PLAYER_WEIGHTS == {}
    assert mlopt.play.DELIVERY_SPEC.weights is None


def test_warm_spec_makes_linear_greedy_prefer_weighted_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A baked table flows through DELIVERY_SPEC into the arm."""
    from gamekit.players import DomainSpec

    warm = DomainSpec.build(
        features=mlopt.play.DELIVERY_FEATURES,
        legal=mlopt.play.DELIVERY_SPEC.legal,
        parts=mlopt.play.DELIVERY_SPEC.parts,
        weights={"a:kind:graph": 5.0},
    )
    monkeypatch.setattr(mlopt.play, "DELIVERY_SPEC", warm)
    mlopt.play.register()
    from gamekit.players import registry

    player = registry.make("linear-greedy", "deliver", seed=0)(0)
    st = DeliveryState(pending=("graph", "compile"), steps=0, cuda=True)
    assert player(st) == "graph"
    # the cold control ignores the prior entirely
    cold = registry.make("linear-cold", "deliver", seed=0)(0)
    assert cold(st) in ("graph", "compile", "stop")


def test_spec_weights_follow_weights_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``weights=dict(WEIGHTS) or None``: {} stays cold, a table warms."""
    from gamekit.players import DomainSpec

    def _spec() -> DomainSpec:
        return DomainSpec.build(
            features=mlopt.play.DELIVERY_FEATURES,
            legal=mlopt.play.DELIVERY_SPEC.legal,
            parts=mlopt.play.DELIVERY_SPEC.parts,
            weights=dict(mlopt.weights.DELIVERY_PLAYER_WEIGHTS) or None,
        )

    assert _spec().weights is None  # {} → honest cold start
    monkeypatch.setattr(mlopt.weights, "DELIVERY_PLAYER_WEIGHTS", {"a:stop": 0.25})
    assert _spec().weights == {"a:stop": 0.25}


# ---------------------------------------------------------------------------
#  aggregate_rewards — the opt-in de-noised collapses (v12)
# ---------------------------------------------------------------------------


def _rows() -> dict[str, list[dict[str, Any]]]:
    """Two arms x two cases; case "big" dominates raw means."""
    return {
        "good": [
            {"case": "flat", "reward": 1.0},
            {"case": "big", "reward": 8.0},
            {"case": "big", "reward": 8.0},  # median 8.0
        ],
        "bad": [
            {"case": "flat", "reward": 0.9},
            {"case": "big", "reward": 2.0},
            {"case": "big", "reward": 4.0},  # median 3.0 — noisy draw
        ],
    }


def test_aggregate_mean_is_the_shipped_default() -> None:
    out = aggregate_rewards(_rows(), mode="mean")
    # mean over ALL episode rows — the big case dominates.
    assert out["good"] == round((1.0 + 8.0 + 8.0) / 3, 3)
    assert out["bad"] == round((0.9 + 2.0 + 4.0) / 3, 3)


def test_aggregate_median_uses_per_case_medians() -> None:
    out = aggregate_rewards(_rows(), mode="median")
    # good: flat 1.0, big median 8.0 -> median(1.0, 8.0) = 4.5
    assert out["good"] == pytest.approx(4.5)
    # bad: flat 0.9, big median 3.0 -> median(0.9, 3.0) = 1.95
    assert out["bad"] == pytest.approx(1.95)


def test_aggregate_case_norm_normalizes_per_case() -> None:
    out = aggregate_rewards(_rows(), mode="case-norm")
    # flat: good/best = 1.0/1.0 = 1.0; bad = 0.9/1.0 = 0.9
    # big:  best = 8.0; good = 1.0, bad = 3.0/8.0 = 0.375
    # means: good (1.0+1.0)/2 = 1.0; bad (0.9+0.375)/2 = 0.6375
    assert out["good"] == pytest.approx(1.0)
    assert out["bad"] == pytest.approx(0.6375)


def test_aggregate_case_norm_flat_case_is_neutral() -> None:
    """A case where every arm ties contributes ~1.0 to all arms."""
    rows = {
        "a": [{"case": "flat", "reward": 1.0}, {"case": "head", "reward": 4.0}],
        "b": [{"case": "flat", "reward": 1.0}, {"case": "head", "reward": 2.0}],
    }
    out = aggregate_rewards(rows, mode="case-norm")
    # flat: both 1.0; head: a=1.0, b=0.5 -> a=1.0, b=0.75
    assert out["a"] == pytest.approx(1.0)
    assert out["b"] == pytest.approx(0.75)


def test_aggregate_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="aggregate"):
        aggregate_rewards(_rows(), mode="bogus")


def test_held_out_accepts_aggregate(game_env) -> None:
    """``held_out(aggregate=...)`` collapses through the same helper."""
    script = _all_slow()
    script["graph"] = (True, (3.0, 3.0))
    cases = [_case(_FakeOptz(script), name="eval")]
    player, *_ = train(cases, seed=0, epochs=1, imitate_epochs=0, budget=9)
    means, errors, done = held_out(player, cases, seed=0, budget=9, aggregate="case-norm")
    assert not errors
    assert all(n == 1 for n in done.values())
    # one case -> every arm is its own best or ties: scores in (0, 1].
    assert all(0.0 < v <= 1.0 for v in means.values())
    with pytest.raises(ValueError, match="aggregate"):
        held_out(player, cases, seed=0, budget=9, aggregate="bogus")
