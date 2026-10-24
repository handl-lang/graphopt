"""Tests for the shipped-prior feature part
(:mod:`mlopt_discovery.open_prior`).

The ``open:prior`` part prices the pending pool with the shipped
prior's own linear model — ``_dot(OPEN_SPEC.weights, _cand_feats)``
plus the full evaluate-row block for the logit-margin columns — so
the stop row carries "what the pending pool still offers" *in the
deployed weights' own units* with no ``set_value_preds`` /
``set_self_scores`` seam: the columns are computable at decode from
shipped tables alone (the round-10 serialization seam —
``bench/drivers/policy/stop_gate.py``).

Contract under test: the registry entry, the emitted column names,
the drained/unweighted fallbacks, the shipped-weights math, the
spec-record round-trip, and the part's inertness off the ``stop``
row.  No existing part or schema changes — the part is registered
but unshipped.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from mlopt_discovery import open_game as og
from mlopt_discovery import open_prior as opr
from gamekit.engine import run_episode
from mlopt_discovery.meta_arena import Action
from gamekit.players import PARTS, DomainSpec, _dot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "drivers" / "policy"))


def _spec() -> DomainSpec:
    """The incumbent spec plus the ``open:prior`` part."""
    return DomainSpec.build(
        features=og.OPEN_GAME_FEATURES + opr.PRIOR_FEATURES,
        legal=og.legal_actions,
        parts=(*og.OPEN_PARTS, PARTS["open:prior"]()),
        weights=None,
        rank=og._cand_feats,
        lr=og.OPEN_SPEC.lr,
    )


def _board(page: int = 6) -> og.OpenArena:
    return og.OpenArena({"ops": None, "offset": 0, "page": page})


class TestRegistry:
    def test_part_registered(self):
        """``open:prior`` resolves through the parts registry."""
        part = PARTS["open:prior"]()
        st = _board().observe()
        assert part(
            st, Action("stop", {}), None, {}
        ) == opr.prior_feats(st, Action("stop", {}), {})

    def test_unshipped(self):
        """The part is registered but absent from the shipped schema."""
        assert not set(opr.PRIOR_FEATURES) & set(og.OPEN_GAME_FEATURES)
        row = og.featurize(_board().observe(), Action("stop", {}))
        assert not set(opr.PRIOR_FEATURES) & set(row)

    def test_spec_record_round_trip(self):
        """A spec carrying the part reifies and rebuilds identically."""
        spec = _spec()
        rec = spec.to_record()
        assert rec["parts"][-1] == "open:prior"
        spec2 = DomainSpec.build_from_record(rec)
        st = _board().observe()
        assert spec.featurize(
            st, Action("stop", {})
        ) == spec2.featurize(st, Action("stop", {}))


class TestEmission:
    def test_non_stop_rows_emit_nothing(self):
        """Eval rows and foreign ops emit ``{}`` — the stop row only."""
        st = _board().observe()
        assert (
            opr.prior_feats(
                st,
                Action("evaluate", {"i": 0, "cand": st.pool[0]}),
                {},
            )
            == {}
        )
        assert opr.prior_feats(st, Action("noop", {}), {}) == {}

    def test_drained_pool_zeros(self):
        """A drained page emits the explicit zero block."""
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
        row = spec.featurize(board.observe(), Action("stop", {}))
        for f in opr.PRIOR_FEATURES:
            assert row[f] == 0.0

    def test_schema_carries_the_columns(self):
        """All six columns land on the stop row."""
        spec = _spec()
        row = spec.featurize(
            _board().observe(), Action("stop", {})
        )
        for f in opr.PRIOR_FEATURES:
            assert f in row


class TestMath:
    def test_pool_columns(self):
        """best/top3/pool/frac against a manual ``_dot`` pass."""
        spec = _spec()
        st = _board().observe()
        w = dict(og.OPEN_SPEC.weights)
        vals = [
            _dot(w, og._cand_feats(st.pool[i])) for i in st.pending
        ]
        row = spec.featurize(st, Action("stop", {}))
        assert row["x:stop:prior_best"] == pytest.approx(max(vals))
        assert row["x:stop:prior_top3"] == pytest.approx(
            sum(sorted(vals, reverse=True)[:3]) / 3
        )
        assert row["x:stop:prior_pool"] == pytest.approx(sum(vals))
        assert row["x:stop:prior_frac"] == pytest.approx(1.0)

    def test_frac_after_an_eval(self):
        """The share drops as the page drains."""
        spec = _spec()
        board = _board()
        st = board.observe()
        top = max(
            st.pending,
            key=lambda i: _dot(
                dict(og.OPEN_SPEC.weights),
                og._cand_feats(st.pool[i]),
            ),
        )
        board.step(
            Action("evaluate", {"i": top, "cand": st.pool[top]})
        )
        row = spec.featurize(board.observe(), Action("stop", {}))
        assert 0.0 <= row["x:stop:prior_frac"] < 1.0

    def test_margin_and_p1(self):
        """margin = max eval-logit − stop-logit; p1 = its softmax share."""
        spec = _spec()
        st = _board().observe()
        w = dict(og.OPEN_SPEC.weights)
        blocks = [
            og._score_rank_feat(
                st, Action("evaluate", {"i": i, "cand": st.pool[i]}), {}
            )
            for i in st.pending
        ]
        logits = [
            w.get("op:evaluate", 0.0)
            + _dot(
                w,
                {
                    **og._cand_feats(st.pool[i]),
                    **b,
                    **og._score_rel_feat(
                        st,
                        Action(
                            "evaluate", {"i": i, "cand": st.pool[i]}
                        ),
                        {},
                    ),
                    **og._score_x_depth_feat(
                        Action("evaluate", {"i": i, "cand": st.pool[i]}),
                        {},
                    ),
                    **og._score_x_ops_feat(
                        Action("evaluate", {"i": i, "cand": st.pool[i]}),
                        {},
                    ),
                },
            )
            for i, b in zip(st.pending, blocks, strict=True)
        ]
        top = max(logits)
        l_stop = w.get("op:stop", 0.0)
        row = spec.featurize(st, Action("stop", {}))
        assert row["x:stop:prior_margin"] == pytest.approx(top - l_stop)
        denom = sum(math.exp(l - top) for l in logits) + math.exp(
            l_stop - top
        )
        assert row["x:stop:prior_p1"] == pytest.approx(1.0 / denom)
        assert 0.0 < row["x:stop:prior_p1"] < 1.0


class TestEpisode:
    def test_episode_runs(self):
        """A spec with the part plays a real page end-to-end."""
        spec = _spec()
        player = og.OpenLearnedPlayer(seed=0, spec=spec, weights={})
        traj = run_episode(_board(page=4), player, budget=6)
        assert 1 <= len(traj.reports) <= 6
