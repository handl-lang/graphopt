"""The generic game engine + the play tool.

Pins: the Board contract runs any conforming domain through one
driver; the joint domain joints forward+gradients into a legal
board program; the search adapter wraps SearchEnv; the learned arm
is domain-agnostic (featurizer injected, not domain-bound).
"""


import pytest

from mlopt_core.ir import Op, TensorType, Var
from gamekit import engine
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import meta_player as mp
from gamekit import players
from mlopt_discovery import play
from gamekit.players import LinearPolicy


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


class _ToyBoard:
    """A minimal Board: each step counts; 'stop' ends it."""

    def __init__(self):
        self.n = 0
        self._done = False

    def observe(self):
        return type(
            "S",
            (),
            {"steps": self.n, "done": self._done},
        )()

    def step(self, action):
        self.n += 1
        if action == "stop" or self.n >= 5:
            self._done = True
        return self.observe(), engine.Report(
            action, reward=-1.0, terminal=self._done
        )


class _DrainBoard:
    """A non-terminating Board: steps count, nothing ends it."""

    def __init__(self):
        self.n = 0

    def observe(self):
        return type(
            "S",
            (),
            {"steps": self.n, "done": False},
        )()

    def step(self, action):
        self.n += 1
        return self.observe(), engine.Report(action, reward=0.0)


def _toy_legal(state):
    return ("go", "go2", "stop") if not state.done else ()


class TestEngine:
    def test_board_protocol(self):
        assert isinstance(_ToyBoard(), engine.Board)

    def test_run_episode_budget(self):
        traj = engine.run_episode(_ToyBoard(), lambda s: "go", 10)
        assert len(traj.reports) == 5  # board's own done at n=5
        assert traj.total == -5.0
        assert traj.terminal is not None

    def test_run_episode_player_none(self):
        traj = engine.run_episode(_ToyBoard(), lambda s: None, 10)
        assert len(traj.reports) == 0
        assert traj.terminal is None

    def test_done_state_stops(self):
        traj = engine.run_episode(_ToyBoard(), lambda s: "stop", 10)
        assert len(traj.reports) == 1
        assert traj.terminal.terminal

    def test_train_policy_calls_finish(self):
        calls = []

        class P:
            def __call__(self, s):
                return "stop"

            def finish_episode(self, total, rewards=None):
                calls.append((total, list(rewards or ())))

        totals = engine.train_policy(
            P(), ["a", "b"], lambda c: _ToyBoard(), budget=4
        )
        assert len(totals) == 2 and len(calls) == 2
        assert calls[0][0] == -1.0 and calls[0][1] == [-1.0]

    def test_evaluate_paired(self):
        arms = {"stopper": lambda i: lambda s: "stop"}
        table = engine.evaluate(
            arms, [("c0",), ("c1",)], lambda c: _ToyBoard(), budget=4
        )
        assert len(table["stopper"]) == 2
        assert table["stopper"][0]["reward"] == -1.0


class TestLinearPolicy:
    def test_generic_domain_plays(self):
        # the same policy class drives a foreign board — featurizer
        # injected, nothing meta-arena-specific
        feats = ("bias",)
        p = LinearPolicy(
            0,
            legal=_toy_legal,
            featurizer=lambda s, a, h: {"bias": 1.0},
            features=feats,
            learn=False,
        )
        assert p.weights_dict() == {"bias": 0.0}
        traj = engine.run_episode(_ToyBoard(), p, 6)
        assert len(traj.reports) <= 3  # go, go2, stop then drain

    def test_learns_on_generic_board(self):
        p = LinearPolicy(
            0,
            legal=_toy_legal,
            featurizer=lambda s, a, h: {
                "bias": 1.0,
                f"act:{a}": 1.0,
            },
            features=("bias", "act:go", "act:go2", "act:stop"),
        )
        engine.train_policy(
            p, range(4), lambda c: _ToyBoard(), budget=6
        )
        w = p.weights_dict()
        assert any(v != 0.0 for v in w.values())

    def test_specless_policy_masks_played_moves(self):
        # regression: fresh_hist() returned None for a spec-less
        # LinearPolicy, and _observe read "no hist" as "new episode"
        # — _played.clear() ran on every call and the played mask
        # never engaged (play.main's learned arm, every domain)
        scored = []
        p = LinearPolicy(
            0,
            legal=_toy_legal,
            featurizer=lambda s, a, h: (
                scored.append(a) or {"bias": 1.0}
            ),
            features=("bias",),
            learn=False,
        )
        traj = engine.run_episode(_DrainBoard(), p, 10)
        keys = [players._action_key(r.action) for r in traj.reports]
        # the 3-move set drains to None — no replays within an episode
        assert len(keys) == len(set(keys)) == 3
        # 3+2+1 featurizations — a played move is never re-scored
        assert len(scored) == 6
        # a fresh board (steps back to 0) is a new episode — reset
        traj2 = engine.run_episode(_DrainBoard(), p, 10)
        assert len(traj2.reports) == 3

    def test_action_key_nonparams(self):
        assert players._action_key("plain") == ("call", "'plain'")


class TestDomains:
    def test_registry(self):
        assert set(play.DOMAINS) == {
            "meta",
            "joint",
            "search",
            "torch",
            "gen",
            "open",
            "player",
            "bilinear",
        }

    def test_torch_domain(self):
        # real modules export and board under the sink bound
        pytest.importorskip("torch")
        cases = play._torch_cases(0, 3)
        assert cases
        name, term, _r, _d, sup = cases[0][:5]
        assert isinstance(sup, frozenset)
        arena = play._torch_board(cases[0])
        assert isinstance(arena, ma.MetaArena)
        # the scripted arm plays the whole game
        traj = play.play(
            "torch", cases[0], ma.ScriptedPlayer(), budget=6
        )
        assert traj.terminal is not None

    def test_play_api_meta(self):
        traj = play.play(
            "meta",
            play.mp.gen_cases(0, 1)[0],
            ma.ScriptedPlayer(),
            budget=8,
        )
        assert traj.reports and traj.terminal is not None

    def test_joint_cases_and_board(self):
        cases = play._joint_cases(0, 4)
        assert len(cases) == 4
        arena = play._joint_board(cases[0])
        assert isinstance(arena, ma.MetaArena)
        # the joint program is a legal board citizen
        st, rep = arena.step(ma.Action.saturate())
        assert rep.applied

    def test_search_board_contract(self):
        board = play._search_board(play._search_cases(0, 1)[0])
        st = board.observe()
        assert not st.done and st.actions
        st2, rep = board.step(st.actions[0])
        assert isinstance(rep, engine.Report)
        assert st2.steps == 1

    def test_search_scripted(self):
        go = play._scripted_search()
        board = play._search_board(play._search_cases(0, 1)[0])
        st = board.observe()
        a = go(st)
        assert a in st.actions

    def test_main_each_domain(self, capsys):
        for domain in ("meta", "joint", "search"):
            # small real run — the registry must hold together
            rc = play.main(
                [
                    "--domain",
                    domain,
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                ]
            )
            assert rc == 0
            out = capsys.readouterr().out
            assert domain in out and "learned" in out


class TestEdges:
    def test_joint_all_archetypes(self):
        # sweep seeds until all three archetypes appear
        seen = set()
        for s in range(20):
            for c in play._joint_cases(s, 6):
                seen.add(c[0].split(":")[1])
        assert seen == {"0", "1", "2", "3"}

    def test_scripted_search_drains(self):
        go = play._scripted_search()
        board = play._search_board(play._search_cases(0, 1)[0])
        st = board.observe()
        for _ in range(len(st.actions) + 2):
            a = go(st)
            if a is None:
                break
            st, _ = board.step(a)
        assert go(st) is None

    def test_arms_without_scripted(self):
        d = play.Domain(
            cases=lambda s, n: [],
            board_of=lambda c: None,
            legal=_toy_legal,
            featurizer=lambda s, a, h: {},
        )
        arms = play._arms(
            d,
            LinearPolicy(
                0, legal=_toy_legal, featurizer=lambda s, a, h: {}
            ),
            0,
        )
        assert "scripted" not in arms and "learned" in arms

    def test_train_policy_no_learner(self):
        # a plain callable has no finish_episode — covered arm
        totals = engine.train_policy(
            lambda s: "stop", ["a"], lambda c: _ToyBoard(), budget=3
        )
        assert totals == [-1.0]

    def test_main_json(self, tmp_path, capsys):
        out_path = tmp_path / "run.json"
        rc = play.main(
            [
                "--domain",
                "search",
                "--train-cases",
                "2",
                "--eval-cases",
                "1",
                "--budget",
                "4",
                "--json",
                str(out_path),
            ]
        )
        assert rc == 0
        import json

        assert "table" in json.loads(out_path.read_text())

    def test_deliver_torch_artifact(self):
        # the product loop end to end: play -> lower -> verified module
        pytest.importorskip("torch")
        cases = play._torch_cases(0, 2)
        res = play.deliver(
            "torch", cases[0], ma.ScriptedPlayer(), budget=6
        )
        assert res["delivered"] and res["verified"]
        assert res["cert"] is True
        assert res["max_abs"] == 0.0

    def test_deliver_rejects_deliverless_domain(self):
        with pytest.raises(ValueError, match="no deliver hook"):
            play.deliver(
                "meta",
                mp.gen_cases(0, 1)[0],
                ma.ScriptedPlayer(),
                budget=4,
            )

    def test_torch_spelled_silu_delivers_fused(self):
        # a module spelling x*sigmoid(x) plays to a verified silu
        # module — shipped fold law or claim, the artifact is real
        pytest.importorskip("torch")
        for s in range(8):
            cases = play._torch_cases(s, 6)
            case = next(
                (c for c in cases if "spelled_silu" in c[0]), None
            )
            if case is not None:
                break
        assert case is not None
        res = play.deliver(
            "torch", case, ma.ScriptedPlayer(), budget=10
        )
        assert res["delivered"] and res["verified"]
        assert res["cost"] is not None and res["cost"] <= 1.0

    def test_deliver_no_member(self, monkeypatch):
        # extraction empty → deliver reports undelivered honestly
        pytest.importorskip("torch")
        cases = play._torch_cases(0, 1)
        board = play._torch_board(cases[0])
        monkeypatch.setattr(
            board.eg, "extract_best", lambda *a, **k: None
        )
        res = play._torch_deliver(
            board, cases[0], engine.Trajectory([])
        )
        assert res == {"delivered": False, "reward": 0.0}

    def test_main_deliver_flag(self, capsys):
        pytest.importorskip("torch")
        rc = play.main(
            [
                "--domain",
                "torch",
                "--train-cases",
                "1",
                "--eval-cases",
                "1",
                "--budget",
                "6",
                "--deliver",
            ]
        )
        assert rc == 0
        assert "deliver[" in capsys.readouterr().out

    def test_scripted_search_drains_at_all_played(self):
        # every rule played while the board lives → the playbook
        # stops, no dead moves (unreachable under the horizon
        # without a fabricated state — the horizon ends first)
        go = play._scripted_search()
        st = play._SearchState(
            actions=("a", "b"),
            ops=frozenset({"x"}),
            steps=0,
            done=False,
            cost=1.0,
            baseline=1.0,
        )
        assert go(st) in ("a", "b")
        assert go(st) in ("a", "b")
        assert go(st) is None

    def test_main_bumps_recursion_limit(self, capsys, monkeypatch):
        import sys

        bumped = []
        monkeypatch.setattr(sys, "getrecursionlimit", lambda: 100)
        monkeypatch.setattr(
            sys, "setrecursionlimit", lambda n: bumped.append(n)
        )
        rc = play.main(
            [
                "--domain",
                "search",
                "--train-cases",
                "1",
                "--eval-cases",
                "1",
                "--budget",
                "4",
            ]
        )
        assert rc == 0 and bumped == [40_000]
        capsys.readouterr()


class TestFusionSites:
    """``_fusion_sites`` boundary abstraction (plan: gen at non-ew leaves)."""

    def _site(self):
        # mul(y, relu(y)) where y = x@p — the arg is NOT elementwise
        x = Var("x", TensorType((4, 4)))
        p = Var("p", TensorType((4, 4)))
        y = Op.make("matmul", x, p)
        return Op.make("mul", y, Op.make("relu", y))

    def test_boundary_children_become_metavars(self):
        sites = play._fusion_sites(self._site())
        assert sites == [("mul", "X1", ("relu", "X1"))]

    def test_identical_subtrees_share_metavar(self):
        # two DIFFERENT spelled args → distinct metas, identical → same
        x = Var("x", TensorType((4, 4)))
        a = Op.make("matmul", x, Var("a", TensorType((4, 4))))
        b = Op.make("matmul", x, Var("b", TensorType((4, 4))))
        sites = play._fusion_sites(
            Op.make("add", Op.make("relu", a), Op.make("relu", b))
        )
        assert sites == [("add", ("relu", "X1"), ("relu", "X2"))]

    def test_const_leaf_stays_literal(self):
        x = Var("x", TensorType((4, 4)))
        spec = play._fusion_pattern(
            Op.make("mul", x, Op.make("relu", x)), {}
        )
        assert spec == ("mul", "X1", ("relu", "X1"))

    def test_non_elementwise_head_not_a_site(self):
        x = Var("x", TensorType((4, 4)))
        assert (
            play._fusion_sites(Op.make("matmul", x, Op.make("relu", x)))
            == []
        )

    def test_chainfuse_claim_binds_gen_over_mm(self):
        # the headline: claim(gen) on a metavar-boundary site —
        # delivered term is gen_*_k(folded mm)
        pytest.importorskip("torch")
        for s in range(12):
            for c in play._gen_cases(s, 8):
                if "chainfuse" not in c[0]:
                    continue
                board = play._gen_board(c)
                st = board.observe()
                claims = [
                    a for a in ma.legal_actions(st) if a.op == "claim"
                ]
                assert claims  # gen_0 site + silu + square
                for a in claims:
                    board.step(a)
                board.step(ma.Action.saturate())
                board.step(ma.Action.extract())
                best = board.eg.extract_best(
                    board.root, board.feasible_cost
                )
                from mlopt_core.ir import op_repr

                delivered = op_repr(board.deliverable(best))
                assert delivered.startswith("(gen_0_k (matmul")
                return
        raise AssertionError("no chainfuse case generated")


class TestZooProbe:
    """``zoo_probe`` — the held-out corpus driver."""

    def test_zoo_case_builds(self):
        pytest.importorskip("torch")
        from mlopt_discovery import zoo as _zoo

        wl = _zoo.zoo()[0]
        case = play._zoo_case(wl)
        assert (
            case[0].startswith("zoo:")
            and case[5]["gen_handlers"] is not None
        )

    def test_zoo_case_intake_dtype_is_a_knob(self):
        # fp64 is the convention, fp32 is what a caller ships; the IR is
        # the same either way, so only leaves and feeds may move.
        pytest.importorskip("torch")
        import torch

        from mlopt_discovery import zoo as _zoo

        wl = _zoo.zoo()[0]
        d64 = play._zoo_case(wl)
        d32 = play._zoo_case(wl, dtype="fp32")
        assert d64[5]["model"] is not d32[5]["model"]
        leaves64 = next(iter(d64[5]["leaves"].values()))
        leaves32 = next(iter(d32[5]["leaves"].values()))
        assert leaves64.dtype is torch.float64
        assert leaves32.dtype is torch.float32
        assert d32[5]["input"].dtype is torch.float32
        assert d32[1] == d64[1]  # same IR root, same structure

    def test_zoo_case_binds_input_vars_for_profitable(
        self, monkeypatch
    ):
        # the referee's env must include the feed vars ('x'/'q'…):
        # a param-only env makes every site touching the input
        # eval-fail and silently skip — decline by measurement, not
        # by accident.
        pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk
        from mlopt_discovery import zoo as _zoo

        seen: dict = {}
        real = gk.profitable

        def spy(handlers, term, var_env, **kw):
            seen["var_env"] = dict(var_env)
            return real(handlers, term, var_env, **kw)

        monkeypatch.setattr(gk, "profitable", spy)
        case = play._zoo_case(_zoo.zoo()[0])
        feeds = case[5]["input"]
        names = {
            v.name
            for v, _t in zip(
                case[5]["ir"].inputs,
                feeds if isinstance(feeds, tuple) else (feeds,),
            )
        }
        assert names <= set(seen["var_env"])

    def test_zoo_probe_records_declines(self, monkeypatch):
        pytest.importorskip("torch")
        from mlopt_discovery import intake

        calls = []

        class _W(intake.Workload):
            pass

        def _boom():
            raise RuntimeError("no trace")

        def _ok():
            import torch.nn as nn

            return nn.Linear(4, 4), __import__("torch").randn(
                2, 4
            ).double()

        monkeypatch.setattr(
            "mlopt_discovery.zoo.zoo",
            lambda *a: [_W("bad", _boom), _W("ok", _ok)],
        )
        monkeypatch.setattr(
            play,
            "gen_probe",
            lambda c, **k: calls.append(c[0]) or {"case": c[0]},
        )
        rows = play.zoo_probe()
        assert rows[0]["delivered"] is False
        assert "export declined" in rows[0]["note"]
        assert calls == ["zoo:ok"]


class TestScanSites:
    """The scan tier's site collector — carriers post-saturation."""

    def _folded_term(self):
        from mlopt_core.ir import Op, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        g = v("g")
        b0 = v("b0", 3, 6)
        b1 = v("b1", 3, 6)
        return Op.make(
            "add",
            Op.make("mul", Op.make("pow", g, 8), b0),
            Op.make("mul", Op.make("pow", g, 4), b1),
        )

    def test_scan_sites_harvests_applyd_members(self):
        pytest.importorskip("torch")
        sites = play._scan_sites(self._folded_term())
        assert sites, "the fold should mint an applyd site"
        from mlopt_discovery import genkernel as gk

        for s in sites:
            spec = s["spec"]
            assert spec[0] == "applyd"
            assert spec[-1] == {"kind": "unrolled"}
            # the state slot is abstracted to a metavar
            assert isinstance(spec[2], str)
            info = gk.scan_spec(spec)
            assert info is not None and info["kind"] == "unrolled"
            # every spec metavar carries its decoded leaf binding
            assert set(s["leaf_env"]) == set(gk._spec_metas(spec))

    def test_scan_site_leaf_env_evals_the_member(self):
        # the recorded env binds each metavar to its decoded
        # subterm; the spec spelled under it IS the member — which
        # saturation proved equal to the source term
        torch = pytest.importorskip("torch")
        import mlopt_torch.torch_bridge as tb

        from mlopt_discovery import genkernel as gk

        sites = play._scan_sites(self._folded_term())
        assert sites
        var_env = {
            "g": torch.tensor(0.9),
            "b0": torch.randn(3, 6),
            "b1": torch.randn(3, 6),
        }
        bindings = tb._IR_TO_TORCH
        for s in sites:
            vals = {
                m: gk._bound_val(t, var_env, bindings)
                for m, t in s["leaf_env"].items()
            }
            got = gk._eval_spec(s["spec"], vals, bindings)
            want = var_env["g"] ** 4 * var_env["b1"] + (
                var_env["g"] ** 8 * var_env["b0"]
            )
            torch.testing.assert_close(got, want)

    def test_scan_handlers_carry_leaf_env(self):
        pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk

        hs = gk.scan_handlers(play._scan_sites(self._folded_term()))
        assert hs
        for h in hs.values():
            assert set(h["leaf_env"]) == set(h["args"])

    def test_profitable_measures_scan_claims(self):
        # the referee binds the recorded env — a scan claim minted
        # post-saturation verifies against its spelled body, which
        # _site_env could never locate in the raw IR
        torch = pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk

        term = self._folded_term()
        hs = gk.scan_handlers(play._scan_sites(term))
        var_env = {
            "g": torch.tensor(0.9),
            "b0": torch.randn(3, 6),
            "b1": torch.randn(3, 6),
        }
        keep = gk.profitable(
            hs,
            term,
            var_env,
            reps=2,
            margin=1e9,  # huge margin: kept iff the env bound+verified
            kernel_for=play._triton_kernel_for,
        )
        assert keep == hs

    def test_profitable_scan_claim_measures_cuda_kernel(self):
        # on a CUDA host the referee times the generated Triton
        # carrier kernel itself — all-tensor metas reach it
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("needs cuda for triton")
        from mlopt_core.ir import Op, TensorType, Var

        from mlopt_discovery import genkernel as gk

        def v(n, *s):
            return Var(n, TensorType(s))

        a, x, h = v("a", 8), v("x", 8), v("h", 8)
        hs = {
            "s": {
                "pattern": (
                    "applyd",
                    ("aff_diag", "A", "X"),
                    "H",
                    {"kind": "unrolled"},
                ),
                "kernel": "sc_ref",
                "args": ("A", "X", "H"),
                "leaf_env": {"A": a, "X": x, "H": h},
            }
        }
        var_env = {
            "a": torch.rand(8) * 0.4 + 0.4,
            "x": torch.randn(8),
            "h": torch.randn(8),
        }
        keep = gk.profitable(
            hs,
            Op.make("applyd", Op.make("aff_diag", a, x), h),
            var_env,
            reps=2,
            margin=1e9,
            kernel_for=play._triton_kernel_for,
        )
        assert keep == hs

    def test_scan_sites_empty_without_carriers(self):
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, TensorType, Var

        term = Op.make(
            "mul", Var("a", TensorType((4,))), Var("b", TensorType((4,)))
        )
        assert play._scan_sites(term) == []

    def test_mint_tiers_includes_scan(self):
        """The registry walk reaches the scan tier's collector."""
        import mlopt_discovery.genkernel as gk

        assert "scan" in play.TIER_SITES
        assert play.TIER_SITES["scan"] is play._scan_sites
        assert any(t.name == "scan" for t in gk.TIERS)


class TestDecayedMint:
    """The decayed-tier mint path — sites, referee, zoo wiring."""

    def test_decayed_sites_finds_spelled_and_lifted(self):
        pytest.importorskip("torch")
        from mlopt_core.ir import Op

        spelled = Op.make("matmul", Op.make("mul", "a", "b"), "c")
        lifted = Op.make("om_decayed", "a", "b", "c")
        term = Op.make("add", spelled, lifted)
        sites = play._decayed_sites(term)
        assert ("matmul", ("mul", "S", "D"), "V") in sites
        assert ("om_decayed", "S", "D", "V") in sites

    def test_decayed_sites_ignores_other_matmuls(self):
        pytest.importorskip("torch")
        from mlopt_core.ir import Op

        term = Op.make("matmul", Op.make("add", "a", "b"), "c")
        assert play._decayed_sites(term) == []

    def test_triton_kernel_for_declines_without_cuda(self, monkeypatch):
        pytest.importorskip("torch")
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        h = {
            "pattern": ("matmul", ("mul", "S", "D"), "V"),
            "kernel": "decayed_0_k",
            "args": ("S", "D", "V"),
        }
        assert play._triton_kernel_for("decayed_0", h, {}) is None

    def test_triton_kernel_for_declines_on_binding_miss(
        self, monkeypatch
    ):
        pytest.importorskip("torch")
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(play, "_has_triton", lambda: True)
        from mlopt_discovery import genkernel as gk

        monkeypatch.setattr(
            gk, "decayed_triton_bindings", lambda handlers: {}
        )
        h = {
            "pattern": ("matmul", ("mul", "S", "D"), "V"),
            "kernel": "decayed_0_k",
            "args": ("S", "D", "V"),
        }
        assert play._triton_kernel_for("decayed_0", h, {}) is None

    def test_profitable_uses_kernel_for(self, monkeypatch):
        pytest.importorskip("torch")
        import torch

        from mlopt_discovery import genkernel as gk
        from mlopt_core.ir import Op

        from mlopt_core.ir import TensorType, Var

        site = Op.make(
            "matmul",
            Op.make(
                "mul",
                Var("s", TensorType((4, 4))),
                Var("d", TensorType((4, 4))),
            ),
            Var("v", TensorType((4, 4))),
        )
        handler = {
            "k": {
                "pattern": ("matmul", ("mul", "S", "D"), "V"),
                "kernel": "kk",
                "args": ("S", "D", "V"),
            }
        }
        vals = {
            "s": torch.randn(4, 4, dtype=torch.float64),
            "d": torch.randn(4, 4, dtype=torch.float64),
            "v": torch.randn(4, 4, dtype=torch.float64),
        }
        calls = []

        def kern(name, h, bound):
            calls.append(name)
            return lambda *a: (bound["S"] * bound["D"]) @ bound["V"]

        gk.profitable(
            handler,
            site,
            vals,
            reps=3,
            kernel_for=kern,
        )
        assert calls == ["k"]

    def test_zoo_case_mints_decayed_handlers(self, monkeypatch):
        pytest.importorskip("torch")
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        from mlopt_discovery import genkernel as gk
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_discovery.intake import Workload

        wl = Workload(
            "ChunkedRetention",
            lambda: (
                _ChunkedRetention(16),
                torch.randn(2, 8, 16, dtype=torch.float64),
            ),
            kind="zoo",
        )
        case = play._zoo_case(wl)
        decayed = gk.decayed_handlers(
            play._decayed_sites(case[5]["ir"].root)
        )
        assert "decayed_0" in decayed
        assert decayed["decayed_0"]["pattern"] == (
            "matmul",
            ("mul", "S", "D"),
            "V",
        )


class TestExtrasArms:
    """The ``--zoo``/``--scan-probe``/``--decay-probe`` report arms —
    printing drivers over the probe generators."""

    def test_zoo_rows_prints_each_probe_row(
        self, capsys, monkeypatch
    ):
        from types import SimpleNamespace

        monkeypatch.setattr(
            play,
            "zoo_probe",
            lambda d, t, b: iter([{"case": "c0", "ok": True}]),
        )
        play._zoo_rows(SimpleNamespace(zoo_d=16, zoo_t=8, zoo_b=2))
        assert "zoo[c0]:" in capsys.readouterr().out

    def test_kernel_probes_flag_arms(self, capsys, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(
            play, "scan_probe", lambda: iter([{"shape": (4, 4)}])
        )
        monkeypatch.setattr(
            play, "decay_probe", lambda: iter([{"shape": (2, 2)}])
        )
        play._kernel_probes(
            SimpleNamespace(scan_probe=True, decay_probe=True)
        )
        out = capsys.readouterr().out
        assert "scan[(4, 4)]:" in out and "decay[(2, 2)]:" in out
