"""Tests for the self-posterior feature part
(:mod:`mlopt_discovery.open_self`).

The ``open:self`` part turns the *policy's own* softmax logits over
the pending ``evaluate`` moves into stop/eval-row columns — the
pending-value evidence the stop decision could not see
(``docs/results_stop_policy.md`` rounds 3–5).  The scorer is injected
(:func:`open_self.set_self_scores`) because a featurizer part cannot
see the policy's weights; a busy flag makes the scorer's own
featurizes treat the part as inert, so the posterior is over the
base schema and the part cannot recurse.

Contract under test: the registry entry, the emitted column names,
the zero-valued unfed/drained fallbacks, the fed posterior math, the
re-entrancy guard, and the spec-record round-trip.  No existing part
or schema changes — the part is registered but unshipped.
"""

from __future__ import annotations

import math

import pytest
from mlopt_discovery import open_game as og
from mlopt_discovery import open_self as oself
from gamekit.engine import run_episode
from mlopt_discovery.meta_arena import Action
from gamekit.players import PARTS, DomainSpec


def _spec() -> DomainSpec:
    """The incumbent spec plus the ``open:self`` part."""
    return DomainSpec.build(
        features=og.OPEN_GAME_FEATURES + oself.SELF_FEATURES,
        legal=og.legal_actions,
        parts=og.OPEN_PARTS + (PARTS["open:self"](),),
        weights=None,
        rank=og._cand_feats,
        lr=og.OPEN_SPEC.lr,
    )


@pytest.fixture(autouse=True)
def _clean_seams():
    """Both injected seams are restored after each test."""
    yield
    oself.set_self_scores(None)
    og.set_value_preds(None)


def _board(page: int = 6) -> og.OpenArena:
    return og.OpenArena({"ops": None, "offset": 0, "page": page})


class TestRegistry:
    def test_part_registered(self):
        """``open:self`` resolves through the parts registry."""
        assert PARTS["open:self"]() is oself.self_featurizer_part

    def test_unshipped(self):
        """The part is registered but absent from the shipped schema."""
        assert not set(oself.SELF_FEATURES) & set(og.OPEN_GAME_FEATURES)
        row = og.featurize(_board().observe(), Action("stop", {}))
        assert not set(oself.SELF_FEATURES) & set(row)

    def test_spec_record_round_trip(self):
        """A spec carrying the part reifies and rebuilds identically."""
        spec = _spec()
        rec = spec.to_record()
        assert rec["parts"][-1] == "open:self"
        spec2 = DomainSpec.build_from_record(rec)
        st = _board().observe()
        assert spec.featurize(st, Action("stop", {})) == spec2.featurize(
            st, Action("stop", {})
        )


class TestUnfed:
    def test_stop_row_zeros(self):
        """Unfed: the stop row emits its ``x:stop:self_*`` block at 0."""
        spec = _spec()
        row = spec.featurize(_board().observe(), Action("stop", {}))
        for f in oself.SELF_FEATURES:
            if f.startswith("x:stop:"):
                assert f in row and row[f] == 0.0
            else:
                assert f not in row  # eval-only columns stay off it

    def test_eval_row_zeros(self):
        """Unfed: every evaluate row emits the eval block at 0.0."""
        spec = _spec()
        st = _board().observe()
        for a in og.legal_actions(st):
            if a.op != "evaluate":
                continue
            row = spec.featurize(st, a)
            assert row["x:self:post"] == 0.0
            assert row["x:self:gap"] == 0.0

    def test_other_op_empty(self):
        """A foreign op emits nothing."""
        assert (
            oself.self_featurizer_part(
                _board().observe(), Action("noop", {}), None, {}
            )
            == {}
        )


class TestFed:
    def test_stop_row_math(self):
        """Index-logit scorer: the posterior math is exact."""
        spec = _spec()
        st = _board().observe()
        oself.set_self_scores(
            lambda s, a, h: float(a.params["i"])
            if a.op == "evaluate"
            else 0.0
        )
        row = spec.featurize(st, Action("stop", {}))
        k = len(st.pending)
        logits = [float(i) for i in st.pending]
        top = max(logits)
        exps = [math.exp(v - top) for v in logits]
        z = sum(exps)
        assert row["x:stop:self_best"] == pytest.approx(top / 16.0)
        assert row["x:stop:self_p1"] == pytest.approx(1.0 / z)
        assert row["x:stop:self_ev"] == pytest.approx(
            sum(e / z * v / 16.0 for e, v in zip(exps, logits))
        )
        assert row["x:stop:self_above"] == pytest.approx(
            sum(e / z > 1.0 / k for e in exps) / k
        )
        # The value table is unfed → the pay column reads 0.
        assert row["x:stop:self_pay"] == 0.0

    def test_eval_row_math(self):
        """An evaluate row reads its own share and margin."""
        spec = _spec()
        st = _board().observe()
        oself.set_self_scores(
            lambda s, a, h: float(a.params["i"])
            if a.op == "evaluate"
            else 0.0
        )
        i = st.pending[1]
        row = spec.featurize(
            st,
            Action("evaluate", {"i": i, "cand": st.pool[i]}),
        )
        logits = [float(j) for j in st.pending]
        top = max(logits)
        z = sum(math.exp(v - top) for v in logits)
        others = [v for v in logits if v != float(i)]
        assert row["x:self:post"] == pytest.approx(
            math.exp(i - top) / z
        )
        assert row["x:self:gap"] == pytest.approx(
            (float(i) - sum(others) / len(others)) / 16.0
        )

    def test_pay_uses_value_table(self):
        """``x:stop:self_pay`` — posterior mass times injected value."""
        spec = _spec()
        st = _board().observe()
        oself.set_self_scores(lambda s, a, h: 0.0)
        og.set_value_preds(lambda c: 1.0)
        row = spec.featurize(st, Action("stop", {}))
        # Uniform posterior x unit value = 1.0 expected yield.
        assert row["x:stop:self_pay"] == pytest.approx(1.0)

    def test_eval_not_pending_zeros(self):
        """An evaluate row for a spent member emits zeros."""
        spec = _spec()
        board = _board()
        st = board.observe()
        i = st.pending[0]
        board.step(Action("evaluate", {"i": i, "cand": st.pool[i]}))
        st2 = board.observe()
        oself.set_self_scores(
            lambda s, a, h: float(a.params["i"])
            if a.op == "evaluate"
            else 0.0
        )
        row = spec.featurize(
            st2,
            Action("evaluate", {"i": i, "cand": st2.pool[i]}),
        )
        assert row["x:self:post"] == 0.0
        assert row["x:self:gap"] == 0.0

    def test_single_pending_gap(self):
        """One pending member: no field to margin against — raw logit."""
        spec = _spec()
        board = _board(page=2)
        st = board.observe()
        i = st.pending[0]
        board.step(Action("evaluate", {"i": i, "cand": st.pool[i]}))
        st2 = board.observe()
        assert len(st2.pending) == 1
        j = st2.pending[0]
        oself.set_self_scores(
            lambda s, a, h: float(a.params["i"]) + 1.0
            if a.op == "evaluate"
            else 0.0
        )
        row = spec.featurize(
            st2,
            Action("evaluate", {"i": j, "cand": st2.pool[j]}),
        )
        assert row["x:self:post"] == pytest.approx(1.0)
        assert row["x:self:gap"] == pytest.approx((j + 1.0) / 16.0)

    def test_drained_pool_zeros(self):
        """A drained page emits the zero block on the stop row."""
        spec = _spec()
        board = _board(page=2)
        while not board.observe().done:
            st = board.observe()
            board.step(
                Action(
                    "evaluate",
                    {
                        "i": st.pending[0],
                        "cand": st.pool[st.pending[0]],
                    },
                )
            )
        st = board.observe()
        assert not st.pending
        oself.set_self_scores(lambda s, a, h: 9.0)
        row = spec.featurize(st, Action("stop", {}))
        for f in oself.SELF_FEATURES:
            if f.startswith("x:stop:"):
                assert row[f] == 0.0


class TestBusy:
    def test_nested_featurize_sees_base_rows(self):
        """The scorer's own featurizes get the part inert — no loop."""
        spec = _spec()
        st = _board().observe()
        seen: list[dict] = []

        def scorer(s, a, h):
            row = spec.featurize(s, a, h)  # re-entrant call
            seen.append(row)
            return float(a.params["i"])

        oself.set_self_scores(scorer)
        spec.featurize(st, Action("stop", {}))
        assert seen  # the scorer ran once per pending member
        for row in seen:
            assert "x:self:post" not in row
            assert "x:stop:self_best" not in row

    def test_busy_resets(self):
        """The guard clears — subsequent rows featurize normally."""
        spec = _spec()
        st = _board().observe()
        oself.set_self_scores(
            lambda s, a, h: spec.featurize(s, a, h)["bias"]
        )
        row = spec.featurize(st, Action("stop", {}))
        assert row["x:stop:self_p1"] > 0.0


class TestEpisode:
    def test_episode_runs_under_injection(self):
        """A spec with the part plays a real page end-to-end."""
        spec = _spec()
        player = og.OpenLearnedPlayer(seed=0, spec=spec, weights={})
        oself.set_self_scores(
            lambda s, a, h: spec.featurize(s, a, h).get("a:score", 0.0)
        )
        traj = run_episode(_board(page=4), player, budget=6)
        assert 1 <= len(traj.reports) <= 6
