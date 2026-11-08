"""The learned meta-arena player — Q3's arm.

Pins: the featurizer covers the schema and discriminates the paying
line's preconditions (a handleable spec, an unhandled declaration);
the player plays only legal unplayed moves, learns (weights move
under REINFORCE reward-to-go), freezes for eval; the driver's
case corpus mixes paying and non-paying boards.
"""


from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Op, TensorType, Var
from mlopt_core.laws import DEFAULT
from mlopt_discovery import lawdata
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import meta_player as mp
from gamekit import players


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _silu_board():
    """A board whose mined spec is handler-coverable."""
    x = _v("x", 4, 4)
    sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
    return ma.MetaArena(
        Op.make("mul", x, Op.make("sigmoid", x)),
        sans,
        cost_fn=count_cost,
    )


class TestFeaturize:
    def test_schema_subset(self):
        st = _silu_board().observe()
        for a in ma.legal_actions(st):
            feats = mp.featurize(st, a)
            assert set(feats) <= set(lawdata.META_ARENA_FEATURES)
            assert feats["bias"] == 1.0
            assert feats[f"op:{a.op}"] == 1.0

    def test_handleable_spec_detected(self):
        st = _silu_board().observe()
        assert mp._handleable(st) == 1
        # a declare row carries the precondition interaction
        d = [a for a in ma.legal_actions(st) if a.op == "declare"]
        assert d and mp.featurize(st, d[0])["x:handleable:declare"] > 0

    def test_control_board_has_no_handleable(self):
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        arena = ma.MetaArena(
            Op.make(
                "add", Op.make("mul", x, y), Op.make("matmul", x, y)
            ),
            DEFAULT,
            cost_fn=count_cost,
        )
        st = arena.observe()
        assert mp._handleable(st) == 0

    def test_action_params(self):
        st = _silu_board().observe()
        by_op = {}
        for a in ma.legal_actions(st):
            by_op.setdefault(a.op, []).append(a)
        sat_b = [
            a
            for a in by_op["saturate"]
            if a.params.get("budget") is not None
        ]
        sat_u = [
            a
            for a in by_op["saturate"]
            if a.params.get("budget") is None
        ]
        assert mp.featurize(st, sat_b[0])["a:budget"] > 0
        assert mp.featurize(st, sat_u[0])["a:unbounded"] == 1.0
        f = mp.featurize(st, by_op["fire"][0])
        assert any(k.startswith("a:r:") for k in f)

    def test_spec_index_parsing(self):
        named = ma.Action.declare(
            {"op": "fold", "params": {"name": "foldabs_7"}}
        )
        assert mp._spec_index(named) == 7 / 32.0
        unnamed = ma.Action.declare({"op": "fold", "params": {}})
        assert mp._spec_index(unnamed) == 0.5

    def test_hist_rates(self):
        h = mp._Hist()
        h.record("fire")
        h.record("fire")
        h.record("extract")
        assert h.rate("fire") == 2 / 3
        assert h.rate("declare") == 0.0


class TestPlayer:
    def test_plays_legal_unplayed(self):
        p = mp.MetaLearnedPlayer(0, learn=False)
        arena = _silu_board()
        played = set()
        for _ in range(8):
            st = arena.observe()
            if st.done:
                break
            a = p(st)
            assert a is not None and a in ma.legal_actions(st)
            assert mp._key(a) not in played
            played.add(mp._key(a))
            arena.step(a)

    def test_greedy_is_deterministic(self):
        a = mp.MetaLearnedPlayer(0, greedy=True, learn=False)
        b = mp.MetaLearnedPlayer(0, greedy=True, learn=False)
        st = _silu_board().observe()
        assert a(st) == b(st)

    def test_learn_moves_weights(self):
        p = mp.MetaLearnedPlayer(0)
        before = p.weights_dict()
        arena = _silu_board()
        traj = ma.run_episode(arena, p, 24)
        p.finish_episode(
            traj.total, rewards=[r.reward for r in traj.reports]
        )
        after = p.weights_dict()
        assert any(before[k] != after[k] for k in before)

    def test_finish_without_rewards_falls_back(self):
        p = mp.MetaLearnedPlayer(0)
        arena = _silu_board()
        traj = ma.run_episode(arena, p, 6)
        p.finish_episode(traj.total)  # no per-step rewards

    def test_frozen_does_not_learn(self):
        # the frozen copy shares the parent's weights and never moves
        # them — the parent may be the baked prior or the cold start
        p = mp.MetaLearnedPlayer(0)
        q = p.frozen(3, greedy=True)
        arena = _silu_board()
        traj = ma.run_episode(arena, q, 6)
        q.finish_episode(traj.total)
        assert q.weights_dict() == p.weights_dict()

    def test_epsilon_explores(self):
        # epsilon=1 forces the uniform branch; still legal, still plays
        p = mp.MetaLearnedPlayer(0, epsilon=1.0)
        arena = _silu_board()
        traj = ma.run_episode(arena, p, 6)
        assert traj.reports

    def test_drains_to_none(self):
        # a legal set of one: play it, then nothing remains
        def one(state):
            return (ma.Action.extract(),)

        p = mp.MetaLearnedPlayer(0, legal=one)
        arena = _silu_board()
        st = arena.observe()
        assert p(st) is not None
        arena.step(ma.Action.extract())
        st2 = arena.observe()
        assert p(st2) is None

    def test_episode_boundary_resets(self):
        p = mp.MetaLearnedPlayer(0, learn=False)
        arena = _silu_board()
        ma.run_episode(arena, p, 4)
        arena2 = _silu_board()
        st = arena2.observe()
        assert st.steps == 0
        # second board: the played mask must not carry over
        a = p(st)
        assert a is not None


class TestImitate:
    def test_imitates_legal_move_and_learns(self):
        p = mp.MetaLearnedPlayer(0)
        arena = _silu_board()
        st = arena.observe()
        move = ma.Action.saturate()
        before = p.weights_dict()
        assert p.imitate(st, move) is True
        after = p.weights_dict()
        assert any(before[k] != after[k] for k in before)

    def test_declines_move_outside_legal_set(self):
        p = mp.MetaLearnedPlayer(0)
        arena = _silu_board()
        st = arena.observe()
        move = ma.Action.saturate()
        assert p.imitate(st, move) is True
        # mid-episode the played mask still holds — a repeat offer
        # of an already-played move declines.  (A step-0 re-observe
        # with a non-empty mask now reads as a fresh board upstream,
        # so the decline is pinned past the boundary.)
        arena.step(move)
        st1 = arena.observe()
        assert p.imitate(st1, ma.Action.extract()) is True
        assert p.imitate(st1, move) is False

    def test_frozen_does_not_learn(self):
        p = mp.MetaLearnedPlayer(0, weights={}, learn=False)
        arena = _silu_board()
        st = arena.observe()
        assert p.imitate(st, ma.Action.saturate()) is True
        assert all(v == 0.0 for v in p.weights_dict().values())

    def test_imitation_warm_start_picks_expert_line(self):
        # expert: saturate → extract — after imitation the policy
        # should rank the demonstrated line top under greedy eval.
        # Cold start (``{}``): imitation is the only signal, so the
        # test measures the warm-start, not the shipped prior.
        p = mp.MetaLearnedPlayer(0, weights={})
        for _ in range(8):
            arena = _silu_board()
            st = arena.observe()
            assert p.imitate(st, ma.Action.saturate())
            arena.step(ma.Action.saturate())
            p.imitate(arena.observe(), ma.Action.extract())
        q = p.frozen(0, greedy=True)
        arena2 = _silu_board()
        a = q(arena2.observe())
        assert a is not None and a.op == "saturate"

    def test_imitate_policy_counts(self):
        from gamekit import engine as eng

        p = mp.MetaLearnedPlayer(0)
        res = eng.imitate_policy(
            p,
            lambda _i: ma.ScriptedPlayer(),
            cases=[None, None],
            board_of=lambda _c: _silu_board(),
            budget=4,
        )
        assert res["shown"] == 4
        assert res["imitated"] == 4

    def test_reset_episode_clears_stale_played_mask(self):
        # an episode ending on its first observed step leaves
        # ``prev_steps == 0`` — gamekit's ``_observe`` now detects
        # that boundary itself (a ``steps == 0`` re-observe with a
        # non-empty played mask is a fresh board), so the next
        # board's ``extract`` is NOT masked even without the local
        # ``reset_episode`` workaround, which stays as belt-and-
        # braces for older pins.
        w = {n: 0.0 for n in lawdata.META_ARENA_FEATURES}
        w["op:extract"] = 10.0
        p = mp.MetaLearnedPlayer(0, weights=w, greedy=True, learn=False)
        ma.run_episode(_silu_board(), p, 24)  # extract on move 1
        st2 = _silu_board().observe()
        assert p(st2).op == "extract"  # auto-cleared upstream
        p.reset_episode()
        assert p(st2).op == "extract"

    def test_train_player_resets_each_episode(self):
        p = mp.MetaLearnedPlayer(0, weights={}, learn=False)
        calls = []
        orig = p.reset_episode
        p.reset_episode = lambda: (calls.append(1), orig())
        mp.train_player(p, mp.gen_cases(0, 3), budget=4)
        assert len(calls) == 3


class TestOracle:
    def test_oracle_claims_paying_board(self):
        case = [
            c
            for c in mp.gen_cases(0, 12)
            if c[0].endswith(":silu")
        ][0]
        reward, line = mp.oracle_line(case, budget=8)
        # the offered macro compresses claim→extract into one
        # refereed step — strictly cheaper, so it wins the argmax
        assert [a.op for a in line] == ["macro"]
        assert reward > 0

    def test_oracle_declines_control_board(self):
        case = [
            c
            for c in mp.gen_cases(0, 12)
            if c[0].endswith(":control")
        ][0]
        reward, line = mp.oracle_line(case, budget=8)
        assert [a.op for a in line] == ["extract"]
        assert reward < 0  # the step cost — nothing pays

    def test_plan_oracles_and_curriculum_order(self):
        cases = mp.gen_cases(0, 12)
        plans = mp.plan_oracles(cases, budget=8)
        assert len(plans) == len(cases)
        ordered = mp.curriculum_order(list(zip(cases, plans)))
        ranks = [mp._plan_rank(p) for _c, p in ordered]
        assert ranks == sorted(ranks)
        # the claim-line boards sort before the decline boards
        first = ordered[0][1]
        assert mp._plan_rank(first) == 0
        # pairs stay aligned — plans travel with their cases
        for c, p in ordered:
            assert p in plans

    def test_imitate_oracle_clones_demonstrations(self):
        cases = mp.gen_cases(0, 8)
        plans = mp.plan_oracles(cases, budget=8)
        p = mp.MetaLearnedPlayer(0, weights={})
        res = mp.imitate_oracle(p, cases, plans, budget=8)
        assert res["shown"] > 0
        assert res["imitated"] == res["shown"]
        # the demonstrated claim move earns positive weight — full
        # consolidation is an RL-scale property (the measured run),
        # the unit pin is that the warm-start points the right way
        assert p.weights_dict()["op:claim"] > 0.0


class TestDriver:
    def test_gen_cases_kinds(self):
        cases = mp.gen_cases(0, 20)
        kinds = {c[0].split(":", 1)[1] for c in cases}
        assert len(cases) == 20
        assert {"silu", "control"} <= kinds

    def test_train_and_evaluate(self):
        cases = mp.gen_cases(1, 4)
        p = mp.MetaLearnedPlayer(1)
        totals = mp.train_player(p, cases, budget=8)
        assert len(totals) == 4
        arms = {
            "scripted": lambda i: ma.ScriptedPlayer(),
            "learned": lambda i: p.frozen(i, greedy=True),
        }
        table = mp.evaluate(arms, cases[:2], budget=8)
        assert set(table) == {"scripted", "learned"}
        assert len(table["scripted"]) == 2
        text = mp.player_table(table)
        assert "scripted" in text and "reward" in text

    def test_winning_line_reachable(self):
        # the hand-set policy proves the schema expresses the line:
        # declare+saturate+handle+extract beats scripted on a silu site
        w = {n: 0.0 for n in lawdata.META_ARENA_FEATURES}
        w.update(
            {
                "op:declare": 2.0,
                "op:saturate": 1.5,
                "a:unbounded": 1.0,
                "op:handle": 3.0,
                "op:extract": 0.5,
                "x:extract:improve": 8.0,
            }
        )
        case = [
            c
            for c in mp.gen_cases(7919, 10)
            if "silu" == c[0].split(":")[1]
        ][0]
        scripted = ma.MetaArena(case[1], case[2], cost_fn=count_cost)
        ma.run_episode(scripted, ma.ScriptedPlayer(), 24)
        arena = mp._board(case, count_cost, 20_000)
        p = mp.MetaLearnedPlayer(0, weights=w, greedy=True, learn=False)
        traj = ma.run_episode(arena, p, 24)
        t = traj.terminal
        assert t is not None and t.certificate_ok
        base = scripted.eg.extract_best(
            scripted.root, scripted.feasible_cost
        )
        from mlopt_core.cost.params import dag_cost

        assert t.cost < dag_cost(base, scripted.feasible_cost)

    def test_main(self, capsys, monkeypatch):
        real = mp.gen_cases
        monkeypatch.setattr(mp, "gen_cases", lambda s, n: real(0, 2))
        assert (
            mp.main(
                [
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "learned-greedy" in out

    def test_main_imitate_curriculum_flags(self, capsys, monkeypatch):
        real = mp.gen_cases
        monkeypatch.setattr(mp, "gen_cases", lambda s, n: real(0, 4))
        assert (
            mp.main(
                [
                    "--train-cases",
                    "4",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                    "--epochs",
                    "1",
                    "--imitate-epochs",
                    "1",
                    "--curriculum",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "imitation:" in out
        assert "oracle plans:" in out
        assert "claims=" in out

    def test_main_json(self, capsys, monkeypatch, tmp_path):
        real = mp.gen_cases
        monkeypatch.setattr(mp, "gen_cases", lambda s, n: real(0, 2))
        out_path = tmp_path / "run.json"
        assert (
            mp.main(
                [
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                    "--json",
                    str(out_path),
                ]
            )
            == 0
        )
        import json

        d = json.loads(out_path.read_text())
        assert "weights" in d and "table" in d


class TestInternals:
    def test_bucket_stable(self):
        assert mp._bucket("silu_fold", "rule") == mp._bucket(
            "silu_fold", "rule"
        )
        assert 0 <= mp._bucket("x", "handler") < 4

    def test_dot(self):
        assert players._dot({"a": 2.0}, {"a": 3.0, "b": 1.0}) == 6.0

    def test_grad_row(self):
        phi, ephi = players._grad_row(
            [{"a": 1.0}, {"a": 0.0, "b": 1.0}], [1.0, 1.0], 0
        )
        assert phi == {"a": 1.0}
        assert ephi == {"a": 0.5, "b": 0.5}

    def test_sample_tail_fallback(self):
        assert mp.MetaLearnedPlayer(0)._sample([]) == -1

    def test_foreign_action_gets_no_schema_keys(self):
        st = _silu_board().observe()
        feats = mp.featurize(st, ma.Action("bogus", {}))
        assert set(feats) <= set(lawdata.META_ARENA_FEATURES)
        assert not any(k.startswith("op:") for k in feats)

    def test_main_bumps_recursion_limit(self, capsys, monkeypatch):
        import sys

        real = mp.gen_cases
        bumped = []
        monkeypatch.setattr(sys, "getrecursionlimit", lambda: 100)
        monkeypatch.setattr(
            sys, "setrecursionlimit", lambda n: bumped.append(n)
        )
        monkeypatch.setattr(
            mp, "gen_cases", lambda s, n: real(0, 2)
        )
        assert mp.main(
            ["--train-cases", "1", "--eval-cases", "1", "--budget", "4"]
        ) == 0
        assert bumped == [40_000]


class TestSpelledLines:
    """The primitive declare→saturate→handle→extract candidates."""

    def _silu_case(self):
        return next(
            c for c in mp.gen_cases(0, 12) if c[0].endswith(":silu")
        )

    def _control_case(self):
        return next(
            c
            for c in mp.gen_cases(0, 12)
            if c[0].endswith(":control")
        )

    def test_spelled_candidates_on_silu(self):
        st = mp._board(self._silu_case(), count_cost, 20_000).observe()
        cands = mp.spelled_candidates(st)
        assert cands
        ops = [[a.op for a in line] for line in cands]
        assert ops == [
            ["declare", "saturate", "handle", "extract"]
        ] * len(cands)
        # the handle leg names the fold the declare minted
        assert all(
            line[2].params["object"].startswith("foldabs_")
            for line in cands
        )

    def test_spelled_candidates_empty_on_control(self):
        st = mp._board(self._control_case(), count_cost, 20_000).observe()
        assert mp.spelled_candidates(st) == []

    def test_plan_spelled_scores_and_declines(self):
        plans = mp.plan_spelled(
            [self._silu_case(), self._control_case()], budget=8
        )
        silu_plan, control_plan = plans
        assert silu_plan is not None
        reward, line = silu_plan
        assert [a.op for a in line] == [
            "declare",
            "saturate",
            "handle",
            "extract",
        ]
        assert reward > 0
        assert control_plan is None

    def test_oracle_spelled_never_beats_claim(self):
        # the spelled line is dominated wherever claim exists — and
        # the offered macro compresses the claim line into one step,
        # so the oracle argmax picks the macro, not the primitives
        _reward, line = mp.oracle_line(
            self._silu_case(), budget=8, spelled=True
        )
        assert [a.op for a in line] == ["macro"]
        assert mp._line_ops(line) == ["claim", "extract"]

    def test_candidate_lines_cover_new_shapes(self):
        # the extended oracle set: offered macros plus the
        # ordering/budget variants — all composed of moves the
        # legal enumerator itself offers
        st = mp._board(self._silu_case(), count_cost, 20_000).observe()
        lines = mp._candidate_lines(st)
        ops = [tuple(a.op for a in line) for line in lines]
        assert ("macro",) in ops
        assert ("claim", "extract") in ops
        assert ("claim", "saturate", "extract") in ops
        assert ("saturate", "claim", "extract") in ops
        # a bounded-saturate claim line is present (budget=256 arm)
        assert any(
            a.op == "saturate" and a.params.get("budget") == 256
            for line in lines
            for a in line
        )

    def test_candidate_lines_multi_claim_variants(self):
        case = next(
            c for c in mp.gen_cases(0, 30) if c[0].endswith(":swiglu")
        )
        st = mp._board(case, count_cost, 20_000).observe()
        lines = mp._candidate_lines(st)
        flats = [mp._line_ops(line) for line in lines]
        assert ["claim", "claim", "extract"] in flats
        assert ["claim", "claim", "saturate", "extract"] in flats
        assert ["saturate", "claim", "claim", "extract"] in flats
        # the reversed claim order is enumerated too
        assert flats.count(["claim", "claim", "extract"]) == 2

    def test_macro_line_terminal_and_partial(self):
        term = ma.Action(
            "macro",
            {"seq": [{"op": "claim", "params": {"handler": "silu"}},
                     {"op": "extract", "params": {}}]},
        )
        assert mp._macro_line(term) == [term]
        part = ma.Action(
            "macro", {"seq": [{"op": "saturate", "params": {}}]}
        )
        line = mp._macro_line(part)
        assert line == [part, ma.Action.extract()]
        # Action-typed (non-dict) seq entries read the same way
        mixed = ma.Action(
            "macro",
            {"seq": (ma.Action.claim("silu"), ma.Action.extract())},
        )
        assert mp._macro_line(mixed) == [mixed]
        # an empty seq is not terminal — extract is appended
        empty = ma.Action("macro", {})
        assert mp._macro_line(empty) == [empty, ma.Action.extract()]
        assert mp._line_ops([mixed]) == ["claim", "extract"]

    def test_case_availability_and_order(self):
        cases = mp.gen_cases(0, 12)
        pairs = [
            (c, mp.oracle_line(c, budget=8)) for c in cases
        ]
        avail = [mp.case_availability(c, max_nodes=20_000) for c in cases]
        # every generated board's claimable count matches its kind
        assert all(a[0] >= 0 and a[1] >= 0 for a in avail)
        ordered = mp.availability_order(pairs, max_nodes=20_000)
        keys = [
            mp.case_availability(c, max_nodes=20_000) for c, _p in ordered
        ]
        assert keys == sorted(keys, reverse=True)
        # pairs stay aligned — the same (case, plan) objects return
        assert len(ordered) == len(pairs)
        assert {id(cp) for cp in ordered} == {id(cp) for cp in pairs}


class TestPrevFeatures:
    """The V2 second-order arm — last move × board catalog."""

    def test_meta_hist_records_last_op(self):
        h = mp.MetaHist()
        h.record("declare")
        h.record("claim")
        assert h.last_op == "claim"
        assert h.rate("declare") == 0.5
        fresh = mp.MetaLearnedPlayer(0).fresh_hist()
        assert isinstance(fresh, mp.MetaHist)

    def test_seq_handler_tags(self):
        assert mp._seq_handler_tags(ma.Action.claim("silu")) == ["silu"]
        assert mp._seq_handler_tags(ma.Action.handle("o", "square")) == [
            "square"
        ]
        assert mp._seq_handler_tags(ma.Action.extract()) == []
        macro = ma.Action(
            "macro",
            {
                "seq": [
                    {"op": "claim", "params": {"handler": "silu"}},
                    {"op": "extract", "params": {}},
                ]
            },
        )
        assert mp._seq_handler_tags(macro) == ["silu"]

    def test_prev_move_feats_pairs(self):
        st = _silu_board().observe()
        h = mp.MetaHist()
        h.record("declare")
        row = mp.META_SPEC_V2.featurize(st, ma.Action.extract(), h)
        assert row["x:seq:declare>extract"] == 1.0
        assert "x:seq:declare>claim" not in row
        # a claim candidate also emits the handler-bucket term
        crow = mp.META_SPEC_V2.featurize(
            st, ma.Action.claim("silu"), h
        )
        assert crow["x:seq:declare>claim"] == 1.0
        assert any(k.startswith("x:seqh:declare:") for k in crow)
        # episode start reads as the ``start`` pseudo-predecessor
        srow = mp.META_SPEC_V2.featurize(
            st, ma.Action.extract(), mp.MetaHist()
        )
        assert srow["x:seq:start>extract"] == 1.0

    def test_v2_schema_covers_emissions(self):
        st = _silu_board().observe()
        h = mp.MetaHist()
        h.record("saturate")
        for a in ma.legal_actions(st):
            feats = mp.META_SPEC_V2.featurize(st, a, h)
            assert set(feats) <= set(mp.META_ARENA_FEATURES_V2)
            assert f"x:seq:saturate>{a.op}" in feats

    def test_v2_player_trains_and_freezes(self):
        cases = mp.gen_cases(0, 4)
        p = mp.MetaLearnedPlayer(
            0, spec=mp.META_SPEC_V2, weights={}, temperature=1.0
        )
        totals = mp.train_player(p, cases, budget=8)
        assert len(totals) == 4
        frozen = p.frozen(0, greedy=True)
        arena = mp._board(cases[0], count_cost, 20_000)
        traj = ma.run_episode(arena, frozen, 8)
        assert traj.total is not None
        # the second-order weights are live schema members
        assert "x:seq:start>extract" in p.weights_dict()

    def test_main_prev_features_flag(self, capsys, monkeypatch):
        real = mp.gen_cases
        monkeypatch.setattr(mp, "gen_cases", lambda s, n: real(0, 2))
        assert (
            mp.main(
                [
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                    "--prev-features",
                ]
            )
            == 0
        )


class TestAvailabilityFlags:
    """The claim-availability curriculum and spelled demo CLI arms."""

    def test_main_availability_and_oracle_spelled(self, capsys, monkeypatch):
        real = mp.gen_cases
        monkeypatch.setattr(mp, "gen_cases", lambda s, n: real(0, 4))
        assert (
            mp.main(
                [
                    "--train-cases",
                    "4",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                    "--epochs",
                    "1",
                    "--imitate-epochs",
                    "1",
                    "--availability",
                    "--oracle-spelled",
                    "--spelled-demo-epochs",
                    "1",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "spelled-pick=" in out
        assert "spelled-moves=" in out

    def test_main_spelled_demo_without_imitate(self, capsys, monkeypatch):
        real = mp.gen_cases
        monkeypatch.setattr(mp, "gen_cases", lambda s, n: real(0, 4))
        assert (
            mp.main(
                [
                    "--train-cases",
                    "4",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                    "--epochs",
                    "1",
                    "--spelled-demo-epochs",
                    "1",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "imitation:" in out
        capsys.readouterr()
