"""Tests for the expected-yield feature part
(:mod:`mlopt_discovery.open_vlabel`) and the round-7 driver's
break-even gate (``bench/policy/stop_valuelabel.py``).

The ``open:vlabel`` part turns the injected per-candidate *expected
payoff* table (``open_game.set_value_preds`` — the same seam
``open:next-val`` reads) into the decision quantities the stop row
prices: remaining yield (``x:stop:vl_pool``), yield net of the
drain's eval costs (``x:stop:vl_net``), the next eval's margin
(``x:stop:vl_margin``), and the per-candidate ``x:vl:*`` columns.

Contract under test: the registry entry, the emitted column names,
the unfed/drained fallbacks (the ``-|step|`` cost terms survive even
unfed — with no positive predictions, draining does lose), the fed
payoff math, the spec-record round-trip, and the driver's analytic
``_EVGate`` break-even rules.  No existing part or schema changes —
the part is registered but unshipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mlopt_discovery import open_game as og
from mlopt_discovery import open_vlabel as ovl
from mlopt_discovery.engine import run_episode
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import PARTS, DomainSpec

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "policy"))

import stop_valuelabel as svl  # noqa: E402


def _spec() -> DomainSpec:
    """The incumbent spec plus the ``open:vlabel`` part."""
    return DomainSpec.build(
        features=og.OPEN_GAME_FEATURES + ovl.VL_FEATURES,
        legal=og.legal_actions,
        parts=(*og.OPEN_PARTS, PARTS["open:vlabel"]()),
        weights=None,
        rank=og._cand_feats,
        lr=og.OPEN_SPEC.lr,
    )


@pytest.fixture(autouse=True)
def _clean_seams():
    """The injected value table is restored after each test."""
    yield
    og.set_value_preds(None)


def _board(page: int = 6) -> og.OpenArena:
    return og.OpenArena({"ops": None, "offset": 0, "page": page})


class TestRegistry:
    def test_part_registered(self):
        """``open:vlabel`` resolves through the parts registry."""
        part = PARTS["open:vlabel"]()
        st = _board().observe()
        assert part(
            st, Action("stop", {}), None, {}
        ) == ovl.vlabel_feats(st, Action("stop", {}), {})

    def test_unshipped(self):
        """The part is registered but absent from the shipped schema."""
        assert not set(ovl.VL_FEATURES) & set(og.OPEN_GAME_FEATURES)
        row = og.featurize(_board().observe(), Action("stop", {}))
        assert not set(ovl.VL_FEATURES) & set(row)

    def test_spec_record_round_trip(self):
        """A spec carrying the part reifies and rebuilds identically."""
        spec = _spec()
        rec = spec.to_record()
        assert rec["parts"][-1] == "open:vlabel"
        spec2 = DomainSpec.build_from_record(rec)
        st = _board().observe()
        assert spec.featurize(
            st, Action("stop", {})
        ) == spec2.featurize(st, Action("stop", {}))


class TestUnfed:
    def test_stop_row_costs(self):
        """Unfed: yield columns read 0, the net/margin carry -|step|."""
        spec = _spec()
        st = _board().observe()
        row = spec.featurize(st, Action("stop", {}))
        n = len(st.pending)
        assert row["x:stop:vl_best"] == 0.0
        assert row["x:stop:vl_pool"] == 0.0
        assert row["x:stop:vl_net"] == pytest.approx(-ovl.EVAL_COST * n)
        assert row["x:stop:vl_margin"] == pytest.approx(-ovl.EVAL_COST)
        assert row["x:stop:vl_frac"] == 0.0
        # eval-only columns stay off the stop row
        assert "x:vl:exp" not in row
        assert "x:vl:net" not in row

    def test_eval_row_costs(self):
        """Unfed: every evaluate row emits v=0, net=-|step|."""
        spec = _spec()
        st = _board().observe()
        for a in og.legal_actions(st):
            if a.op != "evaluate":
                continue
            row = spec.featurize(st, a)
            assert row["x:vl:exp"] == 0.0
            assert row["x:vl:net"] == pytest.approx(-ovl.EVAL_COST)

    def test_other_op_empty(self):
        """A foreign op emits nothing."""
        assert (
            ovl.vlabel_feats(_board().observe(), Action("noop", {}), {})
            == {}
        )

    def test_eval_no_cand_zeros(self):
        """An evaluate row without a bound candidate emits zeros."""
        assert ovl.vlabel_feats(
            _board().observe(), Action("evaluate", {}), {}
        ) == {"x:vl:exp": 0.0, "x:vl:net": 0.0}


class TestFed:
    def test_stop_row_math(self):
        """Constant-value injection: the pool-priced block is exact."""
        spec = _spec()
        st = _board().observe()
        og.set_value_preds(lambda c: 0.1)
        row = spec.featurize(st, Action("stop", {}))
        n = len(st.pending)
        assert row["x:stop:vl_best"] == pytest.approx(0.1)
        assert row["x:stop:vl_pool"] == pytest.approx(0.1 * n)
        assert row["x:stop:vl_net"] == pytest.approx(
            (0.1 - ovl.EVAL_COST) * n
        )
        assert row["x:stop:vl_margin"] == pytest.approx(
            0.1 - ovl.EVAL_COST
        )
        # every candidate predicted equal — the whole page value stays
        assert row["x:stop:vl_frac"] == pytest.approx(1.0)

    def test_per_candidate_values(self):
        """Keyed injection: each candidate's own v lands in the sums."""
        spec = _spec()
        st = _board().observe()
        table = {
            og._value_key(c): float(i) * 0.01
            for i, c in enumerate(st.pool)
        }
        og.set_value_preds(table)
        row = spec.featurize(st, Action("stop", {}))
        vals = [float(i) * 0.01 for i in st.pending]
        pool = sum(vals)
        assert row["x:stop:vl_best"] == pytest.approx(max(vals))
        assert row["x:stop:vl_pool"] == pytest.approx(pool)
        assert row["x:stop:vl_net"] == pytest.approx(
            pool - ovl.EVAL_COST * len(vals)
        )
        assert row["x:stop:vl_frac"] == pytest.approx(1.0)

    def test_eval_row_math(self):
        """An evaluate row reads its own candidate's value, net."""
        spec = _spec()
        st = _board().observe()
        og.set_value_preds(
            lambda c: (
                0.5
                if og._value_key(c) == og._value_key(st.pool[0])
                else 0.0
            )
        )
        row = spec.featurize(
            st, Action("evaluate", {"i": 0, "cand": st.pool[0]})
        )
        assert row["x:vl:exp"] == pytest.approx(0.5)
        assert row["x:vl:net"] == pytest.approx(0.5 - ovl.EVAL_COST)
        i = st.pending[1]
        row2 = spec.featurize(
            st, Action("evaluate", {"i": i, "cand": st.pool[i]})
        )
        assert row2["x:vl:exp"] == 0.0
        assert row2["x:vl:net"] == pytest.approx(-ovl.EVAL_COST)

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
        st = board.observe()
        assert not st.pending
        og.set_value_preds(lambda c: 9.0)
        row = spec.featurize(st, Action("stop", {}))
        for f in ovl.VL_FEATURES:
            if f.startswith("x:stop:"):
                assert row[f] == 0.0


class TestEpisode:
    def test_episode_runs_under_injection(self):
        """A spec with the part plays a real page end-to-end."""
        spec = _spec()
        player = og.OpenLearnedPlayer(seed=0, spec=spec, weights={})
        og.set_value_preds(lambda c: 0.05)
        traj = run_episode(_board(page=4), player, budget=6)
        assert 1 <= len(traj.reports) <= 6


class TestEVGate:
    """The driver's break-even gate — ``stop_valuelabel._EVGate``."""

    def _state(self, page: int = 4) -> og.OpenState:
        return _board(page).observe()

    def _first_pending(self, state):
        """Inner stub: always evaluate the first pending member."""
        if not state.pending:
            return None
        i = state.pending[0]
        return Action("evaluate", {"i": i, "cand": state.pool[i]})

    def test_margin_stops_when_nothing_clears_cost(self):
        """All-zero values: the margin gate quits immediately."""
        st = self._state()
        gate = svl._EVGate(self._first_pending, ovl.EVAL_COST, "margin")
        assert gate(st).op == "stop"

    def test_margin_continues_above_cost(self):
        """One paying member: the margin gate keeps evaluating."""
        st = self._state()
        og.set_value_preds(lambda c: 0.05)
        gate = svl._EVGate(self._first_pending, ovl.EVAL_COST, "margin")
        act = gate(st)
        assert act.op == "evaluate"

    def test_pool_drains_below_cost_members(self):
        """Pool mode: many sub-cost members still net positive in sum."""
        st = self._state()
        # v = 0.001 < κ=0.002 each — margin stops, pool continues
        # while n·(v-κ) ... here negative too: pick v so pool differs.
        og.set_value_preds(lambda c: 0.001)
        gate_m = svl._EVGate(
            self._first_pending, ovl.EVAL_COST, "margin"
        )
        gate_p = svl._EVGate(self._first_pending, 0.0005, "pool")
        assert gate_m(st).op == "stop"
        assert gate_p(st).op == "evaluate"

    def test_pool_stops_when_drain_nets_negative(self):
        """Pool mode: Σ(v-κ) ≤ 0 quits even with members pending."""
        st = self._state()
        og.set_value_preds(lambda c: 0.001)
        gate = svl._EVGate(self._first_pending, ovl.EVAL_COST, "pool")
        assert gate(st).op == "stop"

    def test_empty_pending_stops(self):
        """No pending members: stop regardless of the table."""
        board = _board(page=1)
        board.step(
            Action(
                "evaluate",
                {
                    "i": board.observe().pending[0],
                    "cand": board.pool[board.observe().pending[0]],
                },
            )
        )
        st = board.observe()
        assert not st.pending
        gate = svl._EVGate(self._first_pending, ovl.EVAL_COST, "margin")
        assert gate(st).op == "stop"

    def test_inner_none_stops(self):
        """An inner with no pick falls back to stop."""
        st = self._state()
        og.set_value_preds(lambda c: 0.05)
        gate = svl._EVGate(lambda s: None, ovl.EVAL_COST, "margin")
        assert gate(st).op == "stop"
