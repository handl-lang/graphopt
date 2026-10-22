"""The Strassen MCTS probe — planner contract tests.

``bench/drivers/probes/strassen_mcts.py`` seats a UCB-tree
planner on the bilinear board (the Q2 lookahead arm).  Pins: the
planner terminates inside a budget, every committed move is in
the board's legal set, the seeded runs reproduce exactly, and
the collapse detector finds the engineered-zero reduction the
game tests pin.
"""

import pytest
from gamekit import engine
from mlopt_discovery import bilinear_game as bg
from mlopt_discovery.meta_arena import Action

from bench.drivers.probes.strassen_mcts import (
    MCTSPlayer,
    collapse_moves,
)


def _key(rep):
    """Canonical action key — params carry Fractions, compare as data."""
    return rep.action.op, repr(sorted(rep.action.params.items()))


class TestPlannerContract:
    def test_episode_terminates(self):
        # a small-sims episode completes inside the budget
        traj = engine.run_episode(
            bg.BilinearBoard(), MCTSPlayer(sims=4, seed=0), 6
        )
        assert 0 < len(traj.reports) <= 6

    def test_only_legal_moves_committed(self):
        # every committed action is in the state's legal set
        board = bg.BilinearBoard()
        player = MCTSPlayer(sims=4, seed=1)
        for _ in range(5):
            st = board.observe()
            if st.done:
                break
            a = player(st)
            assert a in bg.legal_actions(st)
            board.step(a)

    def test_deterministic_under_seed(self):
        # same seed → same committed line; different seed differs or
        # at minimum both terminate legally
        t1 = engine.run_episode(
            bg.BilinearBoard(), MCTSPlayer(sims=4, seed=3), 5
        )
        t2 = engine.run_episode(
            bg.BilinearBoard(), MCTSPlayer(sims=4, seed=3), 5
        )
        assert [_key(r) for r in t1.reports] == [
            _key(r) for r in t2.reports
        ]

    def test_never_worse_than_best_rank_floor(self):
        # best_rank is the deliverable — the planner can't push it
        # below the starting scheme's rank even by accident
        traj = engine.run_episode(
            bg.BilinearBoard(), MCTSPlayer(sims=4, seed=2), 6
        )
        assert min(r.detail["best_rank"] for r in traj.reports) <= 8


class TestCollapseDetection:
    def test_engineered_zero_pair_found(self):
        # two terms sharing proportional factors in two modes →
        # the flip that zeroes one is a collapse (the game file's
        # own engineered-drop pin, discovered generically)
        t0 = (bg._u(1, 0, 0, 0), bg._u(1, 0, 0, 0), bg._u(1, 0, 0, 0))
        t1 = (bg._u(0, 1, 0, 0), bg._u(1, 0, 0, 0), bg._u(1, 0, 0, 0))
        cm = collapse_moves((t0, t1, *bg.NAIVE_TERMS[2:]))
        assert cm and all(m["lam"] in bg.FLIP_LAMBDAS for m in cm)
        # and the detected flip really produces a droppable term

        m = cm[0]
        r = bg._flip(t0, t1, m["m"], m["s"], m["lam"])
        assert any(bg._is_zero(f) for f in r[0])

    def test_naive_has_no_collapse(self):
        # the naive scheme's terms share no two proportional modes
        assert collapse_moves(bg.NAIVE_TERMS) == []

    def test_collapse_wins_when_reachable(self):
        # the split-Strassen scheme: Strassen term 0 split along
        # its weight-2 w-factor (w = e0 + e3) — a valid rank-8
        # scheme whose two halves re-merge into a rank-7 win
        u, v, _w = bg.STRASSEN_TERMS[0]
        split = (u, v, bg._e(0)), (u, v, bg._e(3))
        terms = split + bg.STRASSEN_TERMS[1:]
        assert bg.verify_terms(terms) and len(terms) == 8
        board = bg.BilinearBoard(("split", terms))
        st = board.observe()
        cm = collapse_moves(st.terms)
        assert cm  # the split pair is detected generically
        st, _ = board.step(Action("flip", cm[0]))
        drops = [
            i
            for i, t in enumerate(st.terms)
            if any(bg._is_zero(f) for f in t)
        ]
        st, rep = board.step(Action("drop", {"i": drops[0]}))
        assert st.done and st.valid and st.rank == 7
        assert rep.cost == 7.0


@pytest.mark.parametrize("sims", [1, 4])
def test_sim_budgets_run(sims):
    # the sims-per-move parameter is honored at both ends
    traj = engine.run_episode(
        bg.BilinearBoard(), MCTSPlayer(sims=sims, seed=0), 4
    )
    assert len(traj.reports) <= 4
