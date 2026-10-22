"""The bilinear game — the Strassen rediscovery board.

Pins: the Brent-equation move set (cut / staged pick / flip /
drop / rescale) preserves or honestly breaks validity; the
referee verifies the 2×2×2 matmul tensor identity exactly *and*
numerically; the scripted Strassen line reaches verified rank 7
— the move set is not vacuous — and the domain seats players
through the registry seam.
"""

import pytest
from fractions import Fraction as F

from gamekit import engine
from gamekit.players import registry

from mlopt_discovery import bilinear_game as bg
from mlopt_discovery import play
from mlopt_discovery.meta_arena import Action


def _pick_seq(term):
    """The three pick actions committing *term* = (u, v, w)."""
    return [Action("pick", {"k": bg.FACTORS.index(f)}) for f in term]


class TestSchemeData:
    def test_naive_verifies_exact(self):
        assert bg.verify_terms(bg.NAIVE_TERMS)

    def test_strassen_verifies_exact(self):
        # the reachability target: a rank-7 correct scheme
        assert len(bg.STRASSEN_TERMS) == 7
        assert bg.verify_terms(bg.STRASSEN_TERMS)

    def test_strassen_verifies_numeric(self):
        assert bg.verify_numeric(bg.STRASSEN_TERMS, samples=4)

    def test_numeric_rejects_wrong_scheme(self):
        # swap two w factors — still 8 terms, wrong tensor
        bad = list(bg.NAIVE_TERMS)
        t0, t1 = bad[0], bad[2]
        bad[0] = (t0[0], t0[1], t1[2])
        bad[2] = (t1[0], t1[1], t0[2])
        assert not bg.verify_terms(bad)
        assert not bg.verify_numeric(bad, samples=4)

    def test_strassen_factors_in_dictionary(self):
        # the staged add can spell every Strassen factor
        for term in bg.STRASSEN_TERMS:
            for f in term:
                assert f in bg.FACTORS

    def test_residual_identity(self):
        r = bg.residual_of(bg.NAIVE_TERMS)
        assert all(x == 0 for x in r)
        r = bg.residual_of(bg.STRASSEN_TERMS)
        assert all(x == 0 for x in r)
        # one term only: residual = T − that term's tensor
        r = bg.residual_of(bg.NAIVE_TERMS[:1])
        assert sum(abs(x) for x in r) == 7


class TestBoardContract:
    def test_observe_initial(self):
        b = bg.BilinearBoard()
        st = b.observe()
        assert st.rank == 8 and st.valid and not st.done
        assert st.best_rank == 8 and st.steps == 0
        assert isinstance(b, engine.Board)

    def test_legal_vocabulary(self):
        st = bg.BilinearBoard().observe()
        acts = bg.legal_actions(st)
        ops = {a.op for a in acts}
        assert ops == {"cut", "flip", "rescale", "stop"}
        assert sum(1 for a in acts if a.op == "cut") == 8
        # no droppable term on the deal
        assert not any(a.op == "drop" for a in acts)
        # every offered flip really applies
        for a in acts:
            if a.op == "flip":
                assert (
                    bg._flip(
                        st.terms[a.params["i"]],
                        st.terms[a.params["j"]],
                        a.params["m"],
                        a.params["s"],
                        a.params["lam"],
                    )
                    is not None
                )

    def test_pending_picks_enumerated(self):
        b = bg.BilinearBoard()
        st, _ = b.step(Action("pick", {"k": 0}))
        acts = bg.legal_actions(st)
        assert {a.op for a in acts} == {"pick", "abort"}
        assert sum(1 for a in acts if a.op == "pick") == len(bg.FACTORS)

    def test_cut_then_readd_repairs(self):
        b = bg.BilinearBoard()
        st, rep = b.step(Action("cut", {"i": 0, "tnnz": 1}))
        assert st.rank == 7 and not st.valid
        assert rep.reward < 0  # the valley: residual mass costs
        for a in _pick_seq(bg.NAIVE_TERMS[0]):
            st, rep = b.step(a)
        assert st.valid and st.rank == 8
        assert rep.certificate_ok is True

    def test_flip_preserves_validity(self):
        b = bg.BilinearBoard()
        st = b.observe()
        a = next(a for a in bg.legal_actions(st) if a.op == "flip")
        st, rep = b.step(a)
        assert st.valid and st.rank == 8
        assert rep.reward == pytest.approx(0.0)

    def test_rescale_preserves_term(self):
        b = bg.BilinearBoard()
        st, _ = b.step(
            Action("rescale", {"i": 0, "m": 0, "n": 1, "lam": F(2)})
        )
        assert st.valid and st.terms[0][0] == tuple(
            x * 2 for x in bg.NAIVE_TERMS[0][0]
        )

    def test_illegal_moves_raise(self):
        b = bg.BilinearBoard()
        with pytest.raises(ValueError):
            b.step(Action("drop", {"i": 0}))  # no zero factor
        with pytest.raises(ValueError):
            b.step(Action("flip", {"i": 0, "j": 1, "m": 0, "s": 1,
                                   "lam": F(1)}))  # u0 ∦ u1
        with pytest.raises(ValueError):
            b.step(Action("warp", {}))

    def test_drop_after_engineered_zero(self):
        # a flip that zeroes a factor: two c1-paired terms whose
        # v-factors are aligned first — hand-drive via a scheme
        # where terms 0,1 share w *and* proportional v
        t0 = (bg._u(1, 0, 0, 0), bg._u(1, 0, 0, 0), bg._u(1, 0, 0, 0))
        t1 = (bg._u(0, 1, 0, 0), bg._u(1, 0, 0, 0), bg._u(1, 0, 0, 0))
        r = bg._flip(t0, t1, 2, 1, F(1))
        # f_v(t0) −= f_v(t1) → zero — then droppable
        assert r is not None and any(bg._is_zero(f) for f in r[0])
        # board-level: plant a zero and drop it
        b = bg.BilinearBoard()
        b.terms[0] = (bg._u(0, 0, 0, 0),) + bg.NAIVE_TERMS[0][1:]
        st = b.observe()
        drops = [a for a in bg.legal_actions(st) if a.op == "drop"]
        assert drops
        st, rep = b.step(drops[0])
        assert st.rank == 7 and st.valid and rep.cost == 7.0
        # and the win terminates the episode
        assert st.done


class TestStrassenLine:
    def test_scripted_reaches_rank_7(self):
        # THE pin: the move set expresses Strassen — without this
        # the rediscovery test is vacuous
        traj = engine.run_episode(
            bg.BilinearBoard(), bg.strassen_line(), 48
        )
        t = traj.terminal
        assert t is not None and t.terminal
        assert t.cost == 7.0 and t.certificate_ok is True
        assert t.detail["rank"] == 7 and t.detail["valid"]
        # reward telescopes to the rank delta: −8 → −7 = +1
        assert traj.total == pytest.approx(1.0)
        assert len(traj.reports) == 29  # 8 cuts + 7×3 picks

    def test_scripted_on_scrambled_case(self):
        # the line cuts whatever scheme it finds — case-agnostic
        case = bg.gen_cases(3, 4)[2]
        traj = engine.run_episode(
            bg.BilinearBoard(case), bg.strassen_line(), 48
        )
        t = traj.terminal
        assert t is not None and t.cost == 7.0
        assert t.certificate_ok is True


class TestMoves:
    def test_cases_valid_and_scrambled(self):
        cases = bg.gen_cases(0, 6)
        assert cases[0][1] == bg.NAIVE_TERMS
        seen = {cases[0][1]}
        for _name, terms in cases[1:]:
            assert bg.verify_terms(terms)
            assert len(terms) == 8
            assert not bg._droppable(terms)
            seen.add(terms)
        assert len(seen) > 1  # the scramble actually varies

    def test_play_registration(self):
        d = play.DOMAINS["bilinear"]
        assert d.spec is bg.BILINEAR_SPEC
        case = d.cases(0, 2)[1]
        st = d.board_of(case).observe()
        assert bg.legal_actions(st)
        # the spec's featurizer rows cover the schema's names
        row = d.featurizer(st, bg.legal_actions(st)[0], None)
        assert set(row) <= set(bg.BILINEAR_FEATURES)

    def test_registered_random_plays(self):
        mk = registry.make("random", "bilinear", seed=0, ctx={})
        traj = engine.run_episode(bg.BilinearBoard(), mk(0), 12)
        assert traj.reports  # a legal episode ran

    def test_registered_playbook_plays(self):
        mk = registry.make("playbook", "bilinear", seed=0, ctx={})
        traj = engine.run_episode(bg.BilinearBoard(), mk(0), 24)
        assert traj.reports
        # the greedy repair stays honest: every report's board was
        # reachable; the episode either wins or reports rank 8
        t = traj.terminal
        if t is not None:
            assert t.cost in (7.0, 8.0)

    def test_registered_linear_plays(self):
        mk = registry.make("linear", "bilinear", seed=0, ctx={})
        traj = engine.run_episode(bg.BilinearBoard(), mk(0), 12)
        assert traj.reports

    def test_abort_cancels_pick(self):
        b = bg.BilinearBoard()
        st, _ = b.step(Action("pick", {"k": 3}))
        st, _ = b.step(Action("abort", {}))
        assert not st.pending and st.rank == 8 and st.valid


_ZEROS = tuple(F(0) for _ in range(64))


def _state(**kw):
    """Fabricate a BilinearState for direct player calls."""
    base = {
        "terms": (),
        "residual": _ZEROS,
        "pending": (),
        "steps": 0,
        "done": False,
        "best_rank": 8,
    }
    base.update(kw)
    return bg.BilinearState(**base)


class TestEdges:
    def test_done_board_rejects_and_empties(self):
        b = bg.BilinearBoard()
        b.done = True
        assert bg.legal_actions(b.observe()) == ()
        with pytest.raises(RuntimeError):
            b.step(Action("stop", {}))

    def test_stop_ends_episode(self):
        b = bg.BilinearBoard()
        st, rep = b.step(Action("stop", {}))
        assert st.done and rep.terminal and rep.reward == 0.0

    def test_illegal_rescale(self):
        b = bg.BilinearBoard()
        with pytest.raises(ValueError):
            b.step(
                Action("rescale", {"i": 0, "m": 0, "n": 0, "lam": F(2)})
            )
        with pytest.raises(ValueError):
            b.step(
                Action(
                    "rescale", {"i": 0, "m": 0, "n": 1, "lam": F(0)}
                )
            )

    def test_norm_property(self):
        b = bg.BilinearBoard()
        st, _ = b.step(Action("cut", {"i": 0, "tnnz": 1}))
        assert st.norm == 1 and not st.valid

    def test_flip_none_on_zero_factor(self):
        # the zero-factor side is drop's business, not flip's
        z = (bg._u(0, 0, 0, 0), bg._u(1, 0, 0, 0), bg._u(1, 0, 0, 0))
        assert bg._flip(z, bg.NAIVE_TERMS[1], 0, 1, F(1)) is None

    def test_scripted_stops_on_valid_empty(self):
        go = bg.strassen_line()
        assert go(_state()).op == "stop"

    def test_greedy_stops_under_rank_7(self):
        p = bg._GreedyRepair()
        assert p(_state()).op == "stop"

    def test_greedy_aborts_when_plan_drained(self):
        p = bg._GreedyRepair()
        st = _state(pending=(bg.FACTORS[0],), steps=1)
        assert p(st).op == "abort"

    def test_greedy_repairs_residual(self):
        # residual = the whole tensor → the repair plan patches it
        p = bg._GreedyRepair()
        st = _state(residual=bg.TENSOR, steps=3)
        a = p(st)
        assert a.op == "pick" and len(p._plan) == 2

    def test_greedy_resets_between_episodes(self):
        p = bg._GreedyRepair()
        p(_state(terms=bg.NAIVE_TERMS, steps=5))
        # a fresh episode (steps back to 0) clears cut-exclusions
        a = p(_state(terms=bg.NAIVE_TERMS, steps=0))
        assert a.op == "cut" and p._cut == {bg.NAIVE_TERMS[0]}

    def test_greedy_drops_zero_terms(self):
        p = bg._GreedyRepair()
        z = (bg._u(0, 0, 0, 0), bg._u(1, 0, 0, 0), bg._u(1, 0, 0, 0))
        a = p(_state(terms=(*bg.NAIVE_TERMS, z), steps=2))
        assert a.op == "drop" and a.params["i"] == 8

    def test_greedy_pops_queued_plan(self):
        # a populated repair plan is consumed before re-planning
        p = bg._GreedyRepair()
        st = _state(residual=bg.TENSOR, steps=3)
        p(st)
        a = p(st)
        assert a.op == "pick" and len(p._plan) == 1

    def test_featurizer_pick_row(self):
        b = bg.BilinearBoard()
        st, _ = b.step(Action("pick", {"k": 0}))
        pick = next(a for a in bg.legal_actions(st) if a.op == "pick")
        row = bg.featurize(st, pick, None)
        assert row["a:stage"] > 0 and set(row) <= set(
            bg.BILINEAR_FEATURES
        )

    def test_gen_cases_scramble_zero(self):
        # scramble=0 deals the canonical board; seeds vary the rest
        cases = bg.gen_cases(0, 3, scramble=0)
        assert all(terms == bg.NAIVE_TERMS for _n, terms in cases)
        bg.gen_cases(5, 12, scramble=6)  # exercise the rescale arm
