"""Tests for the isolated stop channel — ``GatedPlayer`` + the hook.

``docs/results_stop_isolated.md`` (round 9): the quit decision is
delivered outside the ranking softmax — the engine consults the
player's ``forced_action(state)`` side channel before the policy's
own pick, and the channel fires the board's own ``stop`` move.

Under test: the engine hook (absent / abstain / force), the gated
player's score paths (callable, linear ``w·φ_stop``, unbound),
drain-terminal semantics, episode-reset bookkeeping, the learner
delegations, and the ``gated`` registry seat.
"""

from __future__ import annotations

from typing import Any

import pytest
from gamekit import engine
from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    GatedPlayer,
    LearnedPlayer,
    OpOneHot,
    PlayHist,
    StateBroadcast,
    _dot,
    registry,
)
from mlopt_discovery import open_game as og
from mlopt_discovery.meta_arena import Action


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


class _NoStopBoard(_Board):
    """A board whose legal set has no ``stop`` move."""

    def observe(self) -> _NoStopBoard:
        return self


def _legal(state: _Board) -> tuple[Action, ...]:
    return (Action("go", {}), Action("stop", {}))


def _nostop_legal(state: _NoStopBoard) -> tuple[Action, ...]:
    del state
    return (Action("go", {}),)


def _spec(legal=_legal) -> DomainSpec:
    """A two-action spec over (bias, op:*, st:t)."""
    return DomainSpec.build(
        features=("bias", "op:go", "op:stop", "st:t"),
        legal=legal,
        parts=(
            Bias(),
            OpOneHot(frozenset({"op:go", "op:stop"})),
            StateBroadcast(lambda s: {"st:t": s.steps / 4.0}),
            ActionFeats(lambda a, r: {}),
        ),
        lr=0.1,
    )


def _inner(seed: int = 0) -> Any:
    """An eval-only ranker — the gated arm's inner policy (always ``go``)."""
    del seed
    return lambda s: Action("go", {})


def _learned_inner(seed: int = 0) -> LearnedPlayer:
    """A real LearnedPlayer inner — the delegation tests' target."""
    return LearnedPlayer(
        seed,
        spec=_spec(),
        weights={},
        legal=lambda s: tuple(
            a for a in _legal(s) if a.op != "stop"
        ),
        learn=False,
    )


def _stop_feats(state: _Board, hist: PlayHist) -> dict:
    """A one-column stop row: the state's progress fraction."""
    return {"st:t": state.steps / 4.0}


class TestEngineHook:
    def test_no_hook_unchanged(self):
        """A plain callable player plays exactly as before."""
        traj = engine.run_episode(
            _Board(), lambda s: Action("go", {}), 4
        )
        assert len(traj.reports) == 4
        assert traj.total == 4.0

    def test_hook_fires_the_forced_move(self):
        """``forced_action`` non-None is the move the board plays."""

        class P:
            def forced_action(self, state):
                if state.steps == 1:
                    return Action("stop", {})
                return None

            def __call__(self, state):
                return Action("go", {})

        traj = engine.run_episode(_Board(), P(), 4)
        ops = [r.action.op for r in traj.reports]
        assert ops == ["go", "stop"]  # forced quit at step 1
        assert traj.terminal is not None

    def test_hook_abstains_then_policy_picks(self):
        """``forced_action`` returning None delegates to the pick."""
        seen: list[int] = []

        class P:
            def forced_action(self, state):
                seen.append(state.steps)
                return None  # abstain every step

            def __call__(self, state):
                return Action("go", {})

        traj = engine.run_episode(_Board(), P(), 3)
        assert len(traj.reports) == 3 and seen == [0, 1, 2]

    def test_hook_none_pick_still_ends(self):
        """A None pick inside the hook arm still ends the episode."""

        class P:
            def forced_action(self, state):
                return None

            def __call__(self, state):
                return None

        traj = engine.run_episode(_Board(), P(), 4)
        assert traj.reports == []


class TestGatedPlayer:
    def test_drains_when_unbound(self):
        """No scorer → never fires early; the terminal move closes."""
        player = GatedPlayer(_inner(), legal=_legal, w_stop=None)
        traj = engine.run_episode(_Board(), player, 6)
        ops = [r.action.op for r in traj.reports]
        # four `go`s drain the board; no stop report — done flag ends it
        assert ops == ["go"] * 4

    def test_stop_fires_at_kappa(self):
        """``w·φ_stop <= κ`` quits — the stop row never hits the softmax."""
        # stop row reads st:t = steps/4; κ=0.5 → fires once steps≥2
        player = GatedPlayer(
            _inner(),
            legal=_legal,
            stop_feats=_stop_feats,
            w_stop={"st:t": -1.0},
            kappa=-0.5,
        )
        traj = engine.run_episode(_Board(), player, 6)
        ops = [r.action.op for r in traj.reports]
        assert ops == ["go", "go", "stop"]
        assert traj.terminal is not None

    def test_score_callable_overrides_linear(self):
        """The injected ``score(state, hist)`` wins over ``w·φ``."""
        player = GatedPlayer(
            _inner(),
            legal=_legal,
            stop_feats=_stop_feats,
            w_stop={"st:t": 999.0},  # would never fire on its own
            score=lambda s, h: -1.0,
            kappa=0.0,
        )
        traj = engine.run_episode(_Board(), player, 6)
        assert [r.action.op for r in traj.reports] == ["stop"]

    def test_no_stop_move_abstains(self):
        """A domain without ``stop`` — the channel is inert."""
        player = GatedPlayer(
            _inner(),
            legal=_nostop_legal,
            score=lambda s, h: -1e9,
            kappa=1e9,
        )
        traj = engine.run_episode(_NoStopBoard(), player, 3)
        assert [r.action.op for r in traj.reports] == ["go"] * 3

    def test_call_falls_back_to_stop(self):
        """Inner pick None → the domain's stop move, not a bare end."""
        player = GatedPlayer(
            lambda s: None,  # inner never proposes
            legal=_legal,
            w_stop=None,
        )
        traj = engine.run_episode(_Board(), player, 4)
        assert [r.action.op for r in traj.reports] == ["stop"]
        assert traj.terminal is not None

    def test_drained_legal_set_forces_stop(self):
        """Only ``stop`` remains legal → the channel takes it."""
        player = GatedPlayer(
            _inner(),
            legal=lambda s: (Action("stop", {}),),
            w_stop=None,  # scorer unbound — drain still terminates
        )
        traj = engine.run_episode(_Board(), player, 4)
        assert [r.action.op for r in traj.reports] == ["stop"]
        assert traj.terminal is not None

    def test_episode_boundary_resets_hist(self):
        """A second episode re-zeroes the channel's play history."""
        player = GatedPlayer(
            _inner(),
            legal=_legal,
            stop_feats=_stop_feats,
            w_stop={"st:t": -1.0},
            kappa=-0.5,
        )
        for _ in range(2):
            traj = engine.run_episode(_Board(), player, 6)
            assert traj.reports[-1].action.op == "stop"
        assert player._hist.counts.get("stop", 0) == 1  # fresh per ep

    def test_finish_episode_delegates(self):
        """The learner hook reaches the inner policy when present."""
        inner = _learned_inner()
        player = GatedPlayer(inner, legal=_legal, w_stop=None)
        player.finish_episode(1.0, rewards=[1.0])
        assert inner._baseline != 0.0

    def test_finish_episode_no_inner_hook(self):
        """A bare-callable inner (no finish_episode) is tolerated."""
        player = GatedPlayer(lambda s: None, legal=_legal, w_stop=None)
        player.finish_episode(0.0, rewards=[])  # no raise

    def test_imitate_delegates(self):
        """``imitate`` forwards to the inner policy's CE step."""
        inner = _learned_inner()
        player = GatedPlayer(inner, legal=_legal, w_stop=None)
        ok = player.imitate(_Board(), Action("go", {}))
        assert ok is True
        assert any(
            v != 0.0 for v in inner.weights_dict().values()
        ) is False  # learn=False → no weight move
        player2 = GatedPlayer(lambda s: None, legal=_legal, w_stop=None)
        assert player2.imitate(_Board(), Action("go", {})) is False


class TestGatedRegistry:
    def test_roster_seat(self):
        """``gated`` is registered with the board it can play."""
        s = registry.spec("gated")
        assert s.kind == "gated" and "open" in s.boards
        assert s.record

    def test_make_open_plays_and_deposits_scorer(self):
        """The default (unbound) arm drains an open page honestly."""
        ctx: dict = {}
        mk = registry.make("gated", "open", seed=0, ctx=ctx)
        case = og.gen_cases(0, 1, page=4)[0]
        traj = engine.run_episode(og.OpenArena(case), mk(0), 6)
        assert len(traj.reports) >= 1
        assert callable(ctx["score_fn"])

    def test_make_armed_channel_stops(self):
        """``ctx['gated']`` binds the scorer — the channel fires."""
        spec0 = og.OPEN_SPEC
        ctx = {
            "gated": {
                "spec": spec0,
                "score": lambda s, h: -1.0,
                "kappa": 0.0,
            }
        }
        mk = registry.make("gated", "open", seed=0, ctx=ctx)
        case = og.gen_cases(0, 1, page=4)[0]
        traj = engine.run_episode(og.OpenArena(case), mk(0), 6)
        assert traj.reports[0].action.op == "stop"

    def test_make_linear_stop_head(self):
        """``ctx['gated']['weights']/kappa`` arm the linear head."""
        ctx = {
            "gated": {
                "spec": og.OPEN_SPEC,
                "weights": {"op:stop": 1.0},
                "kappa": 0.5,
            }
        }
        mk = registry.make("gated", "open", seed=0, ctx=ctx)
        case = og.gen_cases(0, 1, page=4)[0]
        player = mk(0)
        state = og.OpenArena(case).observe()
        # the head's stop row: w·φ = 1.0 > κ → abstain, eval flows
        assert player.forced_action(state) is None

    def test_make_other_domain_no_scorer(self):
        """On a domain without ``stop`` the channel is inert — drains."""
        mod = registry.module("gated")
        mk = mod.make("meta", spec=registry.spec("gated"), seed=0, ctx={})
        player = mk(0)
        assert callable(player)

    def test_make_refuses_specless_domain(self, monkeypatch):
        """A domain with no DomainSpec is rejected with a clear error."""
        from types import SimpleNamespace

        from mlopt_discovery import play

        monkeypatch.setitem(
            play.DOMAINS,
            "nospec",
            SimpleNamespace(spec=None),
        )
        mod = registry.module("gated")
        with pytest.raises(ValueError, match="ships no DomainSpec"):
            mod.make(
                "nospec", spec=registry.spec("gated"), seed=0, ctx={}
            )


class TestGatedOnOpen:
    def test_linear_score_reads_stop_row(self):
        """``_dot`` over the spec's stop row is the channel's head."""
        spec = og.OPEN_SPEC
        case = og.gen_cases(0, 1, page=4)[0]
        state = og.OpenArena(case).observe()
        stop = next(
            a for a in og.legal_actions(state) if a.op == "stop"
        )
        row = spec.featurize(state, stop, PlayHist())
        assert row["op:stop"] == 1.0
        assert _dot({"op:stop": -4.0}, row) == -4.0
