"""Tests for the open game — law-shape evaluation as a domain
(:mod:`mlopt_discovery.open_game`).

The board turns ``enumerate_candidates``' stream into episodes: one
``evaluate`` move per pending candidate, ``stop`` to end early, and
the reward carries the verdict tier (``OPEN_GAME_REWARD``) plus the
per-find ordering credit (``OPEN_GAME_EFFICIENCY / evals``) a useful
verdict earns.  These tests pin the board contract (``Board``
protocol via ``engine.run_episode``), the verdict ladder, the
order-sensitive credit, the featurizer schema, and the domain
registration in ``play.DOMAINS``.
"""

from __future__ import annotations

import dataclasses

import pytest
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_discovery import open_game as og
from gamekit.engine import run_episode, train_policy
from mlopt_discovery.meta_arena import Action
from gamekit.players import LinearPolicy


def _board(page: int = 12) -> og.OpenArena:
    return og.OpenArena({"ops": None, "offset": 0, "page": page})


class TestBoard:
    def test_observe_pending_and_pool(self):
        st = _board().observe()
        assert not st.done
        assert len(st.pending) == st.pool_size == 12
        assert len(st.pool) == 12
        assert st.n_seen == st.n_true == st.n_useful == 0

    def test_evaluate_records_verdict(self):
        board = _board()
        st = board.observe()
        act = og.legal_actions(st)[0]
        st2, rep = board.step(act)
        assert rep.applied and not rep.terminal
        assert rep.note.split(":")[0] in (
            "known",
            "false",
            "true",
            "useful",
        )
        assert st2.n_seen == 1 and len(st2.pending) == 11

    def test_step_reward_tiers(self):
        board = _board()
        st = board.observe()
        act = og.legal_actions(st)[0]
        _, rep = board.step(act)
        verdict = rep.note.split(":")[0]
        expected = (
            og.OPEN_GAME_REWARD["step"] + og.OPEN_GAME_REWARD[verdict]
        )
        if verdict == "useful":
            # a useful verdict also pays the per-find ordering credit
            expected += og.OPEN_GAME_EFFICIENCY / 1
        assert rep.reward == pytest.approx(expected)

    def test_stop_terminal(self):
        board = _board()
        _, rep = board.step(Action("stop", {}))
        assert rep.terminal and rep.reward == 0.0
        assert board.observe().done
        with pytest.raises(RuntimeError):
            board.step(Action("stop", {}))

    def test_useful_credit_is_order_sensitive(self):
        """The ordering credit pays a useful law found earlier more.

        The credit is ``OPEN_GAME_EFFICIENCY / evals``, so the same
        useful candidate (index 8 of page 0 — a ``unary-hom`` law)
        pays ``1.0`` when evaluated first and ``1/9`` when reached
        after eight non-paying evals.  Only the credit differs; the
        tier and the eval cost are identical, so the reward gap is
        exactly ``OPEN_GAME_EFFICIENCY·(1 - 1/9)``.
        """

        def reward_at(order: list[int], target: int) -> float:
            board = _board(page=16)
            for idx in order:
                st = board.observe()
                act = next(
                    a
                    for a in og.legal_actions(st)
                    if a.op == "evaluate" and a.params["i"] == idx
                )
                _, rep = board.step(act)
                if idx == target:
                    assert rep.note.split(":")[0] == "useful"
                    return rep.reward
            raise AssertionError("target not evaluated")

        early = reward_at([8], 8)
        late = reward_at([0, 1, 2, 3, 4, 5, 6, 7, 8], 8)
        assert early > late
        assert early - late == pytest.approx(
            og.OPEN_GAME_EFFICIENCY * (1 - 1 / 9)
        )

    def test_drains_to_done(self):
        board = _board(page=4)
        while not board.observe().done:
            st = board.observe()
            board.step(og.legal_actions(st)[0])
        assert board.observe().done

    def test_empty_pool_done(self):
        board = og.OpenArena({"ops": None, "offset": 10**9, "page": 4})
        assert board.observe().done


class TestPolicy:
    def test_legal_actions_bind_candidates(self):
        st = _board().observe()
        acts = og.legal_actions(st)
        assert len(acts) == len(st.pending) + 1
        assert acts[-1].op == "stop"
        assert all(a.op == "evaluate" for a in acts[:-1])
        assert all(a.params["cand"] is not None for a in acts[:-1])

    def test_featurize_schema(self):
        st = _board().observe()
        acts = og.legal_actions(st)
        for a in acts:
            feats = og.featurize(st, a)
            assert set(feats) <= set(og.OPEN_GAME_FEATURES)
        # evaluate moves carry candidate shape features
        f_eval = og.featurize(st, acts[0])
        assert "a:depth" in f_eval and "a:ops" in f_eval

    def test_learned_player_episode(self):
        player = og.OpenLearnedPlayer(seed=0)
        traj = run_episode(_board(page=8), player, budget=10)
        assert 1 <= len(traj.reports) <= 10

    def test_frozen_weights(self):
        player = og.OpenLearnedPlayer(seed=0)
        q = player.frozen(seed=1)
        assert q.weights_dict() == player.weights_dict()

    def test_train_policy_runs(self):
        totals = train_policy(
            og.OpenLearnedPlayer(seed=0),
            og.gen_cases(0, 2, page=6),
            og.OpenArena,
            budget=8,
        )
        assert len(totals) == 2


class TestRegistry:
    def test_play_domain_registered(self):
        from mlopt_discovery.play import DOMAINS

        d = DOMAINS["open"]
        assert d.board_of is og.OpenArena
        assert d.legal is og.legal_actions
        assert d.featurizer is og.featurize
        assert d.features == og.OPEN_GAME_FEATURES
        assert d.scripted is None

    def test_play_episode_generic(self):
        from mlopt_discovery.play import play

        player = LinearPolicy(
            seed=0,
            legal=og.legal_actions,
            featurizer=og.featurize,
            features=og.OPEN_GAME_FEATURES,
        )
        traj = play(
            "open", og.gen_cases(0, 1, page=6)[0], player, budget=8
        )
        assert len(traj.reports) >= 1


class TestPoolRelativeParts:
    """The pool-conditioned schema parts — registered in the parts
    vocabulary but not in the shipped spec's parts list — featurized
    directly against a real board state."""

    def test_pending_broadcasts(self):
        st = _board(page=6).observe()
        assert og._pending_top_feat(st)["st:pending_top"] >= 0.0
        assert og._pending_mean_feat(st)["st:pending_mean"] >= 0.0
        assert og._useful_ct_feat(st)["st:useful_ct"] == 0.0
        drained = dataclasses.replace(st, pending=())
        assert og._pending_top_feat(drained)["st:pending_top"] == 0.0
        assert og._pending_mean_feat(drained)["st:pending_mean"] == 0.0

    def test_stop_feats_rows(self):
        st = _board(page=6).observe()
        row = og._stop_feats(st, Action("stop", {}), {})
        assert set(row) == {
            "x:stop:marginal",
            "x:stop:evals_left",
            "x:stop:finds",
        }
        # a non-stop, non-evaluate action declines — nothing to price
        assert og._stop_feats(st, Action("noop", {}), {}) == {}
        cand = st.pool[st.pending[0]]
        row = og._stop_feats(
            st, Action("evaluate", {"cand": cand}), {}
        )
        assert "x:marginal" in row
        # …but a drained pending set has no quantile to place it in
        drained = dataclasses.replace(st, pending=())
        assert (
            og._stop_feats(
                drained, Action("evaluate", {"cand": cand}), {}
            )
            == {}
        )

    def test_next_value_feats(self):
        st = _board(page=6).observe()
        assert og._next_value_feats(st, Action("evaluate", {}), {}) == {}
        row = og._next_value_feats(st, Action("stop", {}), {})
        assert row["x:stop:remaining_frac"] == 1.0
        assert row["x:stop:best_score"] > 0.0
        drained = dataclasses.replace(st, pending=())
        row = og._next_value_feats(drained, Action("stop", {}), {})
        assert row == {
            "x:stop:best_score": 0.0,
            "x:stop:topk_score": 0.0,
            "x:stop:rank1_pred": 0.0,
            "x:stop:remaining_frac": 0.0,
            "x:stop:decay": 0.0,
        }

    def test_next_val_feats_injected_and_unfed(self):
        st = _board(page=6).observe()
        stop = Action("stop", {})
        assert og._next_val_feats(st, Action("evaluate", {}), {}) == {}
        try:
            og.set_value_preds(lambda _c: 2.0)
            row = og._next_val_feats(st, stop, {})
            assert row["x:stop:next_value"] == 2.0
            assert row["x:stop:value_frac"] == 1.0
        finally:
            og.set_value_preds(None)
        row = og._next_val_feats(st, stop, {})
        assert row["x:stop:next_value"] == 0.0

    def test_cand_feats_unknown_form_and_leaf_kinds(self):
        """An unregistered form emits no ``a:form:*`` one-hot; the
        walk counts Op nodes, collects Var metas and skips Const."""
        lhs = Op.make(
            "add", Var("x", TensorType((2, 4))), Const(1.0)
        )
        cand = og.lp.Candidate(
            strategy="synth", label="zzz:9", lhs=lhs, rhs=Const(0.0)
        )
        feats = og._cand_feats(cand)
        assert feats["a:metas"] == 1.0
        assert feats["a:ops"] == 1.0
        assert not any(k.startswith("a:form:") for k in feats)
