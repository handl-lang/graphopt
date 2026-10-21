"""Tests for :class:`gamekit.advantage.AdvantagePlayer`.

The actor-critic variant of ``LearnedPlayer``: the REINFORCE baseline
is a learned state value ``V(s) = u·φ_state`` instead of the EMA
episode total, so each step's credit is ``G_t - V(s_t)`` — the
``stop`` action's credit isolates to ``-V(s_stop)`` instead of
inheriting the whole episode's return
(``docs/results_stop_policy.md``'s drowned-credit diagnosis).

Under test: the player decodes like ``LearnedPlayer``, the per-step
advantage math (``stop`` earns ``-V``), the V fit runs by the delta
rule on the episode's own rows, episode state resets, ``frozen``
copies, and the no-op paths (learn off, no state row injector).
"""

from __future__ import annotations

from gamekit import engine
from mlopt_discovery import open_game as og
from gamekit.advantage import AdvantagePlayer
from mlopt_discovery.meta_arena import Action
from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    OpOneHot,
    PlayHist,
    StateBroadcast,
    _dot,
)


class _Board:
    """A 4-step toy board: ``go`` pays +1, ``stop`` ends (0 reward)."""

    def __init__(self, n: int = 4) -> None:
        self.steps = 0
        self.done = False
        self.n = n

    def observe(self) -> _Board:
        return self

    def step(self, action: Action) -> tuple[_Board, engine.Report]:
        self.steps += 1
        reward = 1.0 if action.op == "go" else 0.0
        self.done = self.steps >= self.n or action.op == "stop"
        return self, engine.Report(
            action=action, reward=reward, terminal=self.done
        )


def _spec() -> DomainSpec:
    """A two-action spec over (bias, op:*, st:t)."""

    def legal(state: _Board) -> tuple[Action, ...]:
        return (Action("go", {}), Action("stop", {}))

    return DomainSpec.build(
        features=("bias", "op:go", "op:stop", "st:t"),
        legal=legal,
        parts=(
            Bias(),
            OpOneHot(frozenset({"op:go", "op:stop"})),
            StateBroadcast(lambda s: {"st:t": s.steps / 4.0}),
            ActionFeats(lambda a, r: {"x": 0.0} if a else {}),
        ),
        lr=0.1,
    )


def _player(seed: int = 0, **kw) -> AdvantagePlayer:
    spec = _spec()
    return AdvantagePlayer(
        seed,
        spec=spec,
        weights={},
        state_feats=lambda s, h: {
            "bias": 1.0,
            "st:t": s.steps / 4.0,
        },
        **kw,
    )


class TestEpisode:
    def test_decodes_and_learns(self):
        """Episodes run; weights and V both move off the cold start."""
        player = _player()
        totals = engine.train_policy(
            player, [0, 0, 0], lambda c: _Board(), budget=6
        )
        assert len(totals) == 3
        assert any(v != 0.0 for v in player.weights_dict().values())
        assert player._v  # V fit on the episodes' state rows
        assert isinstance(player._v_err, float)
        # Episode state resets between cases.
        assert player._ep == [] and player._states == []

    def test_frozen_shares_weights(self):
        """``frozen`` copies decode identically and never learn."""
        player = _player()
        engine.train_policy(player, [0], lambda c: _Board(), budget=6)
        q = player.frozen(seed=3)
        assert q.weights_dict() == player.weights_dict()
        traj = engine.run_episode(_Board(), q, 4)
        assert len(traj.reports) >= 1
        q.finish_episode(1.0, rewards=[1.0])  # learn=False → no-op

    def test_no_state_feats_is_no_baseline(self):
        """Without the injector V stays {} — flat g_t credit, no crash."""
        spec = _spec()
        player = AdvantagePlayer(
            0, spec=spec, weights={}, state_feats=None
        )
        engine.train_policy(player, [0, 0], lambda c: _Board(), budget=4)
        assert player._v == {}

    def test_stop_credit_isolation(self):
        """A high-V state makes ``stop``'s advantage ≈ -V(s).

        With ``go`` paying +1 for 4 steps, V should learn
        remaining-return ≈ n - t; stopping early then earns a
        negative per-step advantage — the mechanism the flat EMA
        baseline cannot express (there, an above-average episode
        credits *every* move, including a mistimed stop).
        """
        player = _player(v_passes=10)
        engine.train_policy(
            player, [0] * 24, lambda c: _Board(), budget=6
        )
        # V(bias) predicts the remaining return of a fresh state —
        # positive once fit (a full page of `go` returns 4.0; under
        # the near-uniform cold policy the expectation is ~2).
        fresh_v = _dot(player._v, {"bias": 1.0, "st:t": 0.0})
        assert fresh_v > 0.5


class TestFinishEpisode:
    def test_flat_return_fallback(self):
        """``rewards=None`` falls back to the flat episode total."""
        player = _player()
        engine.run_episode(_Board(), player, 4)
        player.finish_episode(2.0)  # no rewards vector
        assert player._v  # V still fit on flat-return targets

    def test_empty_episode_noop(self):
        """An episode with no sampled moves clears its buffers."""
        player = _player()
        player.finish_episode(0.0, rewards=[])
        assert player._ep == [] and player._states == []

    def test_v_learns_remaining_return(self):
        """The delta rule prices the state's remaining value."""
        player = _player(v_passes=10)
        engine.train_policy(
            player, [0] * 10, lambda c: _Board(), budget=6
        )
        # Late states predict less remaining reward than early ones.
        early = _dot(player._v, {"bias": 1.0, "st:t": 0.0})
        late = _dot(player._v, {"bias": 1.0, "st:t": 0.75})
        assert early > late

    def test_open_game_wiring(self):
        """The open domain binds: stop row as the V state summary."""
        spec = DomainSpec.build(
            features=og.OPEN_GAME_FEATURES,
            legal=og.legal_actions,
            parts=og.OPEN_PARTS,
            weights=None,
            rank=og._cand_feats,
            lr=og.OPEN_SPEC.lr,
        )
        player = AdvantagePlayer(
            0,
            spec=spec,
            weights={},
            state_feats=lambda s, h: spec.featurize(
                s, Action("stop", {}), h
            ),
        )
        case = og.gen_cases(0, 1, page=4)[0]
        totals = engine.train_policy(
            player, [case], og.OpenArena, budget=6
        )
        assert len(totals) == 1
        assert player._v
        assert "st:t" not in player._v  # open rows use st:* names
        assert "op:stop" in player._v


class TestHist:
    def test_hist_passed_to_state_feats(self):
        """The injector sees the episode's own play history."""
        seen: list[PlayHist] = []
        spec = _spec()
        player = AdvantagePlayer(
            0,
            spec=spec,
            weights={},
            state_feats=lambda s, h: seen.append(h)
            or {"bias": 1.0},
        )
        engine.run_episode(_Board(), player, 3)
        assert seen  # a hist object flowed to the injector
