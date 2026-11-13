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
            "fragment",
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

    def test_deliver_runner_arm_cpu_declines(self):
        # the CudaGraphRunner arm is additive: a CPU deliverable
        # declines honestly and keeps the eager module
        pytest.importorskip("torch")
        cases = play._torch_cases(0, 2)
        res = play.deliver(
            "torch", cases[0], ma.ScriptedPlayer(), budget=6
        )
        assert res["delivered"] and res["verified"]
        assert res["runner"] == "cuda_graph"
        assert res["plan_tape"] is True
        assert res["cuda_graph"] is False
        assert res["runner_note"] == "inputs not CUDA tensors"
        assert "graph_us" not in res

    @pytest.mark.requires_cuda
    def test_deliver_runner_arm_cuda_captures(self):
        # the T-034 case end to end: a verified zoo extraction on
        # CUDA delivers the graph-replay module with honest
        # eager-vs-graph timings and a zero output delta
        torch = pytest.importorskip("torch")
        pytest.importorskip("mlopt_cuda")
        from mlopt_discovery import zoo

        wl = next(
            w
            for w in zoo.zoo(64, 32, 8)
            if w.name == "GatedDeltaRule"
        )
        case = play._zoo_case(wl, dtype="fp32")
        ex = case[5]
        ex["model"] = ex["model"].to("cuda")
        feed = ex["input"]
        ex["input"] = (
            feed.to("cuda")
            if torch.is_tensor(feed)
            else tuple(t.to("cuda") for t in feed)
        )
        res = play.deliver("gen", case, ma.ScriptedPlayer(), budget=12)
        assert res["delivered"] and res["verified"]
        assert res["cuda_graph"] is True, res.get("runner_note")
        assert res["plan_tape"] is True
        assert res["graph_max_abs"] == 0.0
        assert res["graph_us"] < res["deliver_us"]
        assert res["graph_speedup"] > 1.0

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

    def test_chainfuse_claim_binds_gen_over_mm(self, monkeypatch):
        # the headline: claim(gen) on a metavar-boundary site —
        # delivered term is gen_*_k(folded mm).  The offer gate is
        # bypassed for the same reason as the twosite pin above —
        # this measures claim→deliver machinery, not offer policy.
        pytest.importorskip("torch")
        monkeypatch.setattr(play, "_claim_cost_gate", lambda case: None)
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

    def test_twosite_multi_claim_composes(self, monkeypatch):
        # T-010: two disjoint fusion sites (matmul-separated) compose
        # in ONE extraction — the delivered module calls both minted
        # kernels, and a claimed op can feed another claim's metavar.
        # The offer gate is bypassed here on purpose: this pins the
        # *composition machinery* (claims → saturate → extract →
        # deliver), which is offer-policy-independent — the measured
        # ``_claim_cost_gate`` may legitimately decline a marginal
        # island claim, and ``test_claim_cost_gate`` pins that side.
        pytest.importorskip("torch")
        import re

        monkeypatch.setattr(play, "_claim_cost_gate", lambda case: None)
        for s in range(12):
            for c in play._gen_cases(s, 12):
                if "twosite" not in c[0]:
                    continue
                board = play._gen_board(c)
                st = board.observe()
                claims = [
                    a for a in ma.legal_actions(st) if a.op == "claim"
                ]
                assert len(claims) >= 2  # both islands are claimable
                for a in claims:
                    board.step(a)
                board.step(ma.Action.saturate())
                board.step(ma.Action.extract())
                best = board.eg.extract_best(
                    board.root, board.feasible_cost
                )
                from mlopt_core.ir import op_repr

                delivered = op_repr(board.deliverable(best))
                kernels = set(re.findall(r"gen_\d+_k", delivered))
                assert len(kernels) >= 2, delivered
                return
        raise AssertionError("no twosite case generated")


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
        assert play.TIER_SITES["scan"] is play._all_scan_sites
        assert any(t.name == "scan" for t in gk.TIERS)


class TestSeqScanSites:
    """The seq-form collector — raw ``add(mul(·,·),·)`` chains."""

    def _recurrence(self, t_len: int = 3, *, reversed_idx: bool = False):
        """An unrolled ``h_t = a_t * h + b_t`` chain over selects."""
        from mlopt_core.ir import Op, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        A, B, U = (
            v("A", 2, t_len, 4),
            v("B", 2, t_len, 4),
            v("U", 2, t_len, 4),
        )
        h = v("H0", 2, 4)

        def sel(x, t):
            idx = t_len - 1 - t if reversed_idx else t
            return Op.make("select", x, dim=1, index=idx)

        for t in range(t_len):
            b_t = Op.make("mul", sel(B, t), sel(U, t))
            h = Op.make(
                "add", Op.make("mul", sel(A, t), h), b_t
            )
        return h

    def test_seq_sites_mints_fold_spec(self):
        pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk

        sites = play._seq_scan_sites(self._recurrence())
        assert len(sites) == 1
        s = sites[0]
        spec = s["spec"]
        assert spec[0] == "applyd"
        assert spec[1][0] == "aff_diag"
        assert spec[2] == "H"
        assert spec[3]["kind"] == "seq"
        assert spec[3]["axis"] == 1
        info = gk.scan_spec(spec)
        assert info is not None and info["kind"] == "seq"
        assert set(s["leaf_env"]) == set(gk._spec_metas(spec))
        assert s["axis"] == 1

    def test_seq_site_fold_equals_chain(self):
        # the minted fold must denote the unrolled chain's value —
        # the whole point of the collector's index/shape checks
        torch = pytest.importorskip("torch")
        import mlopt_torch.torch_bridge as tb

        from mlopt_discovery import genkernel as gk

        term = self._recurrence()
        sites = play._seq_scan_sites(term)
        assert len(sites) == 1
        s = sites[0]
        var_env = {
            "A": torch.randn(2, 3, 4),
            "B": torch.randn(2, 3, 4),
            "U": torch.randn(2, 3, 4),
            "H0": torch.randn(2, 4),
        }
        bindings = tb._IR_TO_TORCH
        vals = {
            m: gk._bound_val(t, var_env, bindings)
            for m, t in s["leaf_env"].items()
        }
        h = {
            "pattern": s["spec"],
            "args": tuple(gk._spec_metas(s["spec"])),
            "axis": s["axis"],
        }
        got = gk.handler_denotation(h)(
            *[vals[m] for m in h["args"]]
        )
        want = tb.eval_term(
            term,
            var_env=var_env,
            param_env=var_env,
            bindings=bindings,
            strict=True,
        )
        torch.testing.assert_close(got, want)

    def test_seq_handlers_carry_axis_and_leaf_env(self):
        pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk

        hs = gk.scan_handlers(play._seq_scan_sites(self._recurrence()))
        assert len(hs) == 1
        h = next(iter(hs.values()))
        assert h["axis"] == 1
        assert set(h["leaf_env"]) == set(h["args"])
        assert gk.handler_denotation(h) is not None

    def test_scan_handler_carries_chain_route(self):
        # the site record's chain root must reach the handler entry —
        # it is the deliver-time route's LHS (the ``applyd`` claim
        # pattern never occurs in raw IR, so delivery rewrites the
        # extracted chain's e-class to the kernel call instead)
        pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk

        term = self._recurrence()
        hs = gk.scan_handlers(play._seq_scan_sites(term))
        assert len(hs) == 1
        h = next(iter(hs.values()))
        assert h["chain"] is term

    def test_deliverable_routes_chain_to_scan_kernel(self):
        # end-to-end at term level: the board's ``deliverable``
        # rewrites the extracted recurrence chain to ``scan_<i>_k``
        # — the op every scan-aware sink binds — while a board
        # without the route delivers the spelled chain unchanged.
        pytest.importorskip("torch")
        from mlopt_core.ir import op_repr
        from mlopt_discovery import genkernel as gk

        term = self._recurrence()
        hs = gk.scan_handlers(play._seq_scan_sites(term))
        board = ma.MetaArena(term, handlers=hs)
        out = op_repr(board.deliverable(term))
        assert "scan_0_k" in out
        assert "applyd" not in out
        # no route data → the spelled chain delivers verbatim
        plain = ma.MetaArena(term)
        assert op_repr(plain.deliverable(term)) == op_repr(term)

    def test_route_matches_eclass_member_not_chain_object(self):
        # saturation rewrites inside the chain freely: the extracted
        # representative is usually a DIFFERENT member of the chain's
        # e-class (a folded variant, or a declared fold op holding
        # the whole class).  The route must fire on the class, not
        # the collector's ``chain`` object.
        pytest.importorskip("torch")
        from mlopt_core.egraph import EGraph, Rewrite
        from mlopt_core.ir import op_repr
        from mlopt_discovery import genkernel as gk

        term = self._recurrence()
        hs = gk.scan_handlers(play._seq_scan_sites(term))
        h = next(iter(hs.values()))
        from mlopt_core.cost import count_cost

        eg = EGraph()
        root = eg.add_term(term)
        # a stand-in member of the chain's class under a different
        # op name — the e-graph merges it; extraction picks it on
        # cost (one op beats the whole chain).  The route must still
        # recognise the class.
        proxy = Op.make("scan_proxy", next(iter(h["leaf_env"].values())))
        eg.apply_rule(
            Rewrite(
                name="test_proxy",
                lhs=h["chain"],
                rhs=proxy,
                law="test",
            ),
            root,
        )
        best = eg.extract_best(root, count_cost)
        assert best == proxy
        routed = ma._route_scan_chains(best, hs, eg)
        assert "scan_0_k" in op_repr(routed)

    def test_route_declines_incomplete_binding(self):
        # a handler whose leaf_env cannot instantiate every arg
        # keeps the spelled chain — never a partial kernel call
        pytest.importorskip("torch")
        from mlopt_core.ir import op_repr
        from mlopt_discovery import genkernel as gk

        term = self._recurrence()
        hs = gk.scan_handlers(play._seq_scan_sites(term))
        h = dict(next(iter(hs.values())))
        # present but binds only one metavar — the kernel call cannot
        # be instantiated, so the spelled chain must deliver
        h["leaf_env"] = dict(list(h["leaf_env"].items())[:1])
        board = ma.MetaArena(term, handlers={"scan_0": h})
        assert op_repr(board.deliverable(term)) == op_repr(term)

    def test_seq_sites_decline_wrong_index_order(self):
        # selects walking the axis backwards are not the fold's
        # step order — the site must decline, not relabel
        pytest.importorskip("torch")
        assert play._seq_scan_sites(
            self._recurrence(reversed_idx=True)
        ) == []

    def test_seq_sites_decline_non_chain(self):
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, TensorType, Var

        # a single-step carrier-shaped term is not a recurrence
        a = Var("a", TensorType((4,)))
        x = Var("x", TensorType((4,)))
        h = Var("h", TensorType((4,)))
        term = Op.make(
            "applyd", Op.make("aff_diag", a, x), h
        )
        assert play._seq_scan_sites(term) == []

    def _gla_recurrence(self, t_len: int = 3):
        """GLA-shaped chain: ``g_t⊙h + k_t v_tᵀ`` over unsqueezed selects.

        The matrix-state gated update — per-step ``unsqueeze`` views
        on ``(b,T,d)`` slices against a ``(b,d,d)`` state.  The view
        lift must grow the operand broadcast to include the state
        dims or the fold cannot denote the chain.
        """
        from mlopt_core.ir import Op, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        G, K, V = (
            v("G", 2, t_len, 4),
            v("K", 2, t_len, 4),
            v("V", 2, t_len, 4),
        )
        h = v("H0", 2, 4, 4)
        for t in range(t_len):
            a_t = Op.make(
                "unsqueeze", Op.make("select", G, dim=1, index=t), dim=-1
            )
            k_t = Op.make(
                "unsqueeze", Op.make("select", K, dim=1, index=t), dim=-1
            )
            v_t = Op.make(
                "unsqueeze",
                Op.make("select", V, dim=1, index=t),
                dim=-2,
            )
            h = Op.make(
                "add",
                Op.make("mul", a_t, h),
                Op.make("mul", k_t, v_t),
            )
        return h

    def test_seq_sites_lift_unsqueeze_views(self):
        # view-wrapped selects lift onto the sequence child: the
        # spec skeleton carries NO view op and scan_spec accepts
        pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk

        sites = play._seq_scan_sites(self._gla_recurrence())
        assert len(sites) == 1
        s = sites[0]
        spec = s["spec"]
        assert "unsqueeze" not in repr(spec)
        assert spec[1] == ("aff_diag", "Q1", ("mul", "Q2", "Q3"))
        info = gk.scan_spec(spec)
        assert info is not None and info["kind"] == "seq"
        # the lifted operands broadcast to the (b,T,d,d) frame —
        # state space (b,d,d) is what the (b,d,d) H binds
        torch = pytest.importorskip("torch")
        import mlopt_torch.torch_bridge as tb

        bindings = tb._IR_TO_TORCH
        env = {
            "G": torch.randn(2, 3, 4),
            "K": torch.randn(2, 3, 4),
            "V": torch.randn(2, 3, 4),
            "H0": torch.randn(2, 4, 4),
        }
        vals = {
            m: gk._bound_val(t, env, bindings)
            for m, t in s["leaf_env"].items()
        }
        shapes = {m: tuple(v.shape) for m, v in vals.items()}
        assert shapes["Q1"] == (2, 3, 4, 1)
        assert shapes["Q3"] == (2, 3, 1, 4)
        assert shapes["H"] == (2, 4, 4)

    def test_seq_sites_lifted_fold_equals_chain(self):
        # the lifted fold must still denote the unrolled chain
        # exactly — the view commute is only sound if it is
        torch = pytest.importorskip("torch")
        import mlopt_torch.torch_bridge as tb

        from mlopt_discovery import genkernel as gk

        term = self._gla_recurrence()
        sites = play._seq_scan_sites(term)
        assert len(sites) == 1
        s = sites[0]
        var_env = {
            "G": torch.randn(2, 3, 4),
            "K": torch.randn(2, 3, 4),
            "V": torch.randn(2, 3, 4),
            "H0": torch.randn(2, 4, 4),
        }
        bindings = tb._IR_TO_TORCH
        vals = {
            m: gk._bound_val(t, var_env, bindings)
            for m, t in s["leaf_env"].items()
        }
        h = {
            "pattern": s["spec"],
            "args": tuple(gk._spec_metas(s["spec"])),
            "axis": s["axis"],
        }
        got = gk.handler_denotation(h)(
            *[vals[m] for m in h["args"]]
        )
        want = tb.eval_term(
            term,
            var_env=var_env,
            param_env=var_env,
            bindings=bindings,
            strict=True,
        )
        torch.testing.assert_close(got, want)

    def test_seq_sites_decline_view_over_expr(self):
        # a view wrapping a COMPOUND step expr (not a bare select)
        # cannot lift — the skeleton keeps the view node and
        # scan_spec declines honestly at handler mint
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, TensorType, Var
        from mlopt_discovery import genkernel as gk

        def v(n, *s):
            return Var(n, TensorType(s))

        A, B = v("A", 2, 3, 4), v("B", 2, 3, 4)
        h = v("H0", 2, 4)
        for t in range(3):
            inner = Op.make(
                "mul",
                Op.make("select", A, dim=1, index=t),
                Op.make("select", B, dim=1, index=t),
            )
            a_t = Op.make("unsqueeze", inner, dim=-1)
            h = Op.make(
                "add",
                Op.make("mul", a_t, h),
                Op.make("select", B, dim=1, index=t),
            )
        sites = play._seq_scan_sites(h)
        # the collector may still emit the spine; the handler mint
        # is where the view skeleton must decline
        assert all(
            gk.scan_spec(s["spec"]) is None for s in sites
        ) or sites == []
        assert gk.scan_handlers(sites) == {}

    def test_seq_sites_decline_state_shaped_mul(self):
        # ``mul(state_expr, x) + y`` chains whose "map" factor is a
        # whole-tensor reference (no per-step select) cannot mint a
        # sequence operand — decline rather than guess
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        x, y = v("x", 2, 3, 4), v("y", 2, 4)
        h = v("h0", 2, 4)
        for t in range(3):
            h = Op.make(
                "add",
                Op.make("mul", x, h),
                Op.make("select", y, dim=1, index=t),
            )
        assert play._seq_scan_sites(h) == []

    def _proj_recurrence(self, t_len: int = 3):
        """In-step projections + loop-invariant leaves.

        ``h_t = a·h + sigmoid(x_t@W)·(x_t@V)`` — the bare-``Param`` grammar
        gap: ``x_t @ W`` is a ``matmul`` over a per-step ``select``
        and ``a`` is a whole-tensor ``Param``.  The collector pulls
        the select through the projection (the operand binds the
        hoisted ``x @ W`` sequence) and mints ``a`` as a ``"const"``
        meta bound whole — neither reaches the step skeleton.
        """
        from mlopt_core.ir import Op, Param, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        def p(n, *s):
            return Param(n, TensorType(s))

        leaves = {
            "X": v("X", t_len, 2, 4),
            "W": p("W", 4, 4),
            "V": p("V", 4, 4),
            "A": p("A", 4),
        }
        X, W, V, A = leaves["X"], leaves["W"], leaves["V"], leaves["A"]
        h = v("H0", 2, 4)
        for t in range(t_len):
            xt = Op.make("select", X, dim=0, index=t)
            u = Op.make(
                "mul",
                Op.make("sigmoid", Op.make("matmul", xt, W)),
                Op.make("matmul", xt, V),
            )
            h = Op.make("add", Op.make("mul", A, h), u)
        return h, leaves

    def test_seq_sites_lift_in_step_projections(self):
        # ``matmul(select(x,0,t), W)`` commutes with the slice on a
        # non-contracted axis — the operand binds ``matmul(x, W)``,
        # the projected sequence, and the bare ``Param`` decay mints
        # a ``"const"`` meta the fold reads whole every step.
        pytest.importorskip("torch")
        from mlopt_core.ir import Op
        from mlopt_discovery import genkernel as gk

        term, leaves = self._proj_recurrence()
        sites = play._seq_scan_sites(term)
        assert len(sites) == 1
        s = sites[0]
        spec = s["spec"]
        assert spec[0] == "applyd"
        assert spec[3]["kind"] == "seq"
        assert spec[3]["const"] == ("P1",)
        # the lifted operands bind the PROJECTED sequences — the
        # per-step ``x_t @ W`` spelling became the hoisted ``(x@W)_t``
        X, W, V = leaves["X"], leaves["W"], leaves["V"]
        assert s["leaf_env"]["Q1"] == Op.make("matmul", X, W)
        assert s["leaf_env"]["Q2"] == Op.make("matmul", X, V)
        assert s["leaf_env"]["P1"] == leaves["A"]
        info = gk.scan_spec(spec)
        assert info is not None and info["kind"] == "seq"
        assert info["seq"] == ["Q1", "Q2"]
        assert info["const"] == ["P1"]
        assert set(s["leaf_env"]) == set(gk._spec_metas(spec))

    def test_seq_site_projected_fold_equals_chain(self):
        # the minted fold must denote the unrolled chain whose steps
        # spell in-step projections — hoisting is only sound if the
        # per-step values agree bit-for-bit
        torch = pytest.importorskip("torch")
        import mlopt_torch.torch_bridge as tb

        from mlopt_discovery import genkernel as gk

        term, _leaves = self._proj_recurrence()
        sites = play._seq_scan_sites(term)
        assert len(sites) == 1
        s = sites[0]
        torch.manual_seed(0)
        var_env = {
            "X": torch.randn(3, 2, 4, dtype=torch.float64),
            "W": torch.randn(4, 4, dtype=torch.float64),
            "V": torch.randn(4, 4, dtype=torch.float64),
            "A": torch.randn(4, dtype=torch.float64),
            "H0": torch.randn(2, 4, dtype=torch.float64),
        }
        bindings = tb._IR_TO_TORCH
        vals = {
            m: gk._bound_val(t, var_env, bindings)
            for m, t in s["leaf_env"].items()
        }
        h = {
            "pattern": s["spec"],
            "args": tuple(gk._spec_metas(s["spec"])),
            "axis": s["axis"],
        }
        got = gk.handler_denotation(h)(
            *[vals[m] for m in h["args"]]
        )
        want = tb.eval_term(
            term,
            var_env=var_env,
            param_env=var_env,
            bindings=bindings,
            strict=True,
        )
        torch.testing.assert_close(got, want)

    def test_seq_sites_lift_linear_projection(self):
        # ``linear(select(x,0,t), W, b)`` lifts the same way —
        # ``F.linear`` contracts only the last axis of ``x``
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, Param, TensorType, Var
        from mlopt_discovery import genkernel as gk

        def v(n, *s):
            return Var(n, TensorType(s))

        def p(n, *s):
            return Param(n, TensorType(s))

        X = v("X", 3, 2, 4)
        W, Bs, A = p("W", 5, 4), p("Bs", 5), p("A", 5)
        h = v("H0", 2, 5)
        for t in range(3):
            xt = Op.make("select", X, dim=0, index=t)
            u = Op.make("linear", xt, W, Bs)
            h = Op.make("add", Op.make("mul", A, h), u)
        sites = play._seq_scan_sites(h)
        assert len(sites) == 1
        s = sites[0]
        assert s["leaf_env"]["Q1"] == Op.make("linear", X, W, Bs)
        info = gk.scan_spec(s["spec"])
        assert info is not None and info["kind"] == "seq"

    def test_seq_sites_decline_projected_contract_dim(self):
        # a ``select`` on the CONTRACTED axis does not commute with
        # the matmul — the lift declines, the spec keeps a per-step
        # ``matmul`` the elementwise kernel cannot render, and
        # ``scan_spec`` declines honestly at handler mint
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, Param, TensorType, Var
        from mlopt_discovery import genkernel as gk

        def v(n, *s):
            return Var(n, TensorType(s))

        def p(n, *s):
            return Param(n, TensorType(s))

        X = v("X", 3, 3, 3)
        W, A = p("W", 3, 3), p("A", 3)
        h = v("H0", 2, 3)
        for t in range(3):
            xt = Op.make("select", X, dim=2, index=t)
            u = Op.make("matmul", xt, W)
            h = Op.make("add", Op.make("mul", A, h), u)
        sites = play._seq_scan_sites(h)
        assert sites  # the spine still deindexes — the spec exists
        assert all(gk.scan_spec(s["spec"]) is None for s in sites)
        assert gk.scan_handlers(sites) == {}

    def test_seq_sites_decline_const_not_broadcasting(self):
        # a loop-invariant leaf that cannot broadcast into the
        # state shape is not a per-step operand the fold can read —
        # decline the whole site rather than emit a spec that
        # raises on call
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, Param, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        def p(n, *s):
            return Param(n, TensorType(s))

        X = v("X", 3, 2, 4)
        BIG = p("BIG", 3, 2, 4)  # step-axis extent — not a const
        h = v("H0", 2, 4)
        for t in range(3):
            xt = Op.make("select", X, dim=0, index=t)
            h = Op.make("add", Op.make("mul", BIG, h), xt)
        assert play._seq_scan_sites(h) == []


class TestSeqmSites:
    """The seqm collector — matrix-state delta-rule chains.

    ``h' = h + beta_t·k_t ⊗ (v_t - k_tᵀ h)`` puts the accumulator
    inside the step's ``b`` term (the ``kᵀh`` matvec read), a shape
    the diagonal ``seq`` grammar cannot express: the state is a
    direct ``add`` operand and the write is rank-1.  These tests pin
    the mint, the fold-equals-chain soundness, and the honest
    declines.
    """

    def _delta_recurrence(self, t_len: int = 3, *, gated: bool = False):
        """Unrolled ``S += beta·k ⊗ (v - [alpha·] kᵀS)`` chain.

        Shapes: ``beta (b,T,1,1)``, ``k (b,T,dk)``, ``v (b,T,dv)``,
        ``alpha (b,T,1)`` when gated, state ``(b,dk,dv)`` — the
        DeltaRuleBlock / GatedDeltaRule spellings.
        """
        from mlopt_core.ir import Op, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        b_, dk, dv = 2, 3, 4
        BETA = v("BETA", b_, t_len, 1, 1)
        K = v("K", b_, t_len, dk)
        V = v("V", b_, t_len, dv)
        ALPHA = v("ALPHA", b_, t_len, 1) if gated else None
        h = v("H0", b_, dk, dv)
        for t in range(t_len):
            k_t = Op.make("select", K, dim=1, index=t)
            kcol = Op.make("unsqueeze", k_t, dim=-1)  # (b,dk,1)
            krow = Op.make("unsqueeze", k_t, dim=-2)  # (b,1,dk)
            proj = Op.make(
                "squeeze",
                Op.make("matmul", krow, h),
                dim=1,
            )  # (b,dv)
            if gated:
                a_t = Op.make(
                    "select", ALPHA, dim=1, index=t
                )  # (b,1)
                proj = Op.make("mul", a_t, proj)
            resid = Op.make(
                "sub", Op.make("select", V, dim=1, index=t), proj
            )
            rrow = Op.make("unsqueeze", resid, dim=-2)  # (b,1,dv)
            beta_t = Op.make("select", BETA, dim=1, index=t)
            write = Op.make(
                "mul",
                Op.make("mul", beta_t, kcol),
                rrow,
            )
            h = Op.make("add", h, write)
        return h

    def _env(self, gated: bool = False):
        torch = pytest.importorskip("torch")
        t_len = 3
        env = {
            "BETA": torch.randn(2, t_len, 1, 1).double(),
            "K": torch.randn(2, t_len, 3).double(),
            "V": torch.randn(2, t_len, 4).double(),
            "H0": torch.randn(2, 3, 4).double(),
        }
        if gated:
            env["ALPHA"] = (
                torch.rand(2, t_len, 1).double() * 0.5 + 0.5
            )
        return env

    def test_seqm_sites_mints_matrix_fold_spec(self):
        pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk

        sites = play._seqm_scan_sites(self._delta_recurrence())
        assert len(sites) == 1
        s = sites[0]
        spec = s["spec"]
        assert spec[0] == "applyd"
        assert spec[1][0] == "aff_diag"
        assert spec[1][1] == 1  # the unit map — state is a direct add
        assert spec[2] == "H"
        assert spec[3]["kind"] == "seqm"
        assert spec[3]["axis"] == 1
        info = gk.scan_spec(spec)
        assert info is not None and info["kind"] == "seqm"
        assert info["k"] in info["seq"]
        assert "H" not in info["seq"]
        assert set(s["leaf_env"]) == set(gk._spec_metas(spec))
        assert s["axes"] == {m: 1 for m in info["seq"]}
        # the diagonal collector must NOT claim this chain — its
        # grammar has no state-inside-b form
        assert play._seq_scan_sites(self._delta_recurrence()) == []

    def test_seqm_site_fold_equals_chain(self):
        # the minted fold must denote the unrolled delta-rule chain
        # exactly — soundness before any pricing
        torch = pytest.importorskip("torch")
        import mlopt_torch.torch_bridge as tb

        from mlopt_discovery import genkernel as gk

        term = self._delta_recurrence()
        sites = play._seqm_scan_sites(term)
        assert len(sites) == 1
        s = sites[0]
        var_env = self._env()
        bindings = tb._IR_TO_TORCH
        vals = {
            m: gk._bound_val(t, var_env, bindings)
            for m, t in s["leaf_env"].items()
        }
        h = {
            "pattern": s["spec"],
            "args": tuple(gk._spec_metas(s["spec"])),
            "axis": s["axis"],
            "axes": s["axes"],
        }
        got = gk.handler_denotation(h)(
            *[vals[m] for m in h["args"]]
        )
        want = tb.eval_term(
            term,
            var_env=var_env,
            param_env=var_env,
            bindings=bindings,
            strict=True,
        )
        torch.testing.assert_close(got, want)

    def test_seqm_site_gated_fold_equals_chain(self):
        # the gated spelling carries a scalar ``a_t`` map AND the
        # in-residual ``alpha`` gate — both must survive the fold
        torch = pytest.importorskip("torch")
        import mlopt_torch.torch_bridge as tb

        from mlopt_discovery import genkernel as gk

        term = self._delta_recurrence(gated=True)
        sites = play._seqm_scan_sites(term)
        assert len(sites) == 1
        s = sites[0]
        var_env = self._env(gated=True)
        bindings = tb._IR_TO_TORCH
        vals = {
            m: gk._bound_val(t, var_env, bindings)
            for m, t in s["leaf_env"].items()
        }
        h = {
            "pattern": s["spec"],
            "args": tuple(gk._spec_metas(s["spec"])),
            "axis": s["axis"],
            "axes": s["axes"],
        }
        got = gk.handler_denotation(h)(
            *[vals[m] for m in h["args"]]
        )
        want = tb.eval_term(
            term,
            var_env=var_env,
            param_env=var_env,
            bindings=bindings,
            strict=True,
        )
        torch.testing.assert_close(got, want)

    def test_seqm_handlers_carry_axes(self):
        pytest.importorskip("torch")
        from mlopt_discovery import genkernel as gk

        hs = gk.scan_handlers(play._seqm_scan_sites(self._delta_recurrence()))
        assert len(hs) == 1
        h = next(iter(hs.values()))
        assert set(h["axes"]) == set(h["args"]) - {"H"}
        assert gk.handler_denotation(h) is not None
        assert gk.scan_spec(h["pattern"])["kind"] == "seqm"

    def test_seqm_sites_decline_state_free_write(self):
        # ``h' = h + k⊗v`` (no ``kᵀh`` read) is a rank-1 WRITE but
        # not state feedback — the seqm tier requires ``H`` inside
        # ``b``; this shape belongs to no fold tier
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        K, V = v("K", 2, 3, 3), v("V", 2, 3, 4)
        h = v("H0", 2, 3, 4)
        for t in range(3):
            k_t = Op.make("select", K, dim=1, index=t)
            v_t = Op.make("select", V, dim=1, index=t)
            write = Op.make(
                "mul",
                Op.make("unsqueeze", k_t, dim=-1),
                Op.make("unsqueeze", v_t, dim=-2),
            )
            h = Op.make("add", h, write)
        assert play._seqm_scan_sites(h) == []

    def test_seqm_sites_decline_wrong_index_order(self):
        # selects walking the step axis backwards are not the fold's
        # step order — decline, don't relabel
        pytest.importorskip("torch")
        from mlopt_core.ir import Op, TensorType, Var

        def v(n, *s):
            return Var(n, TensorType(s))

        b_, dk, dv, t_len = 2, 3, 4, 3
        BETA = v("BETA", b_, t_len, 1, 1)
        K = v("K", b_, t_len, dk)
        V = v("V", b_, t_len, dv)
        h = v("H0", b_, dk, dv)
        for t in range(t_len):
            idx = t_len - 1 - t
            k_t = Op.make("select", K, dim=1, index=idx)
            krow = Op.make("unsqueeze", k_t, dim=-2)
            proj = Op.make(
                "squeeze", Op.make("matmul", krow, h), dim=1
            )
            resid = Op.make(
                "sub", Op.make("select", V, dim=1, index=idx), proj
            )
            write = Op.make(
                "mul",
                Op.make("select", BETA, dim=1, index=idx),
                Op.make("unsqueeze", resid, dim=-2),
            )
            # multiply the column factor in explicitly so the write
            # stays rank-1 — k itself is the (b,dk,1) factor here
            write = Op.make(
                "mul",
                Op.make("unsqueeze", k_t, dim=-1),
                write,
            )
            h = Op.make("add", h, write)
        assert play._seqm_scan_sites(h) == []


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


class TestReductionSites:
    """The reduction tier's site collector + claim wiring."""

    def test_reduction_sites_finds_fusable_bodies(self):
        pytest.importorskip("torch")
        x = _v("x", 4, 8)
        body = Op.make("mul", x, Op.make("relu", x))
        term = Op.make("add", Op.make("mean", body, dim=(-1,), keepdim=True), x)
        sites = play._reduction_sites(term)
        assert (
            "mean",
            ("mul", "X1", ("relu", "X1")),
            {"dim": (-1,), "keepdim": True},
        ) in sites

    def test_reduction_sites_ignores_bare_and_nested(self):
        pytest.importorskip("torch")
        x = _v("x", 4, 8)
        bare = Op.make("sum", x, dim=(0,))
        nested = Op.make(
            "sum", Op.make("mean", x, dim=(1,)), dim=(0,)
        )
        assert play._reduction_sites(Op.make("add", bare, nested)) == []

    def test_mint_tiers_includes_reduction(self):
        import mlopt_discovery.genkernel as gk

        assert "reduction" in play.TIER_SITES
        assert play.TIER_SITES["reduction"] is play._reduction_sites
        assert any(t.name == "reduction" for t in gk.TIERS)

    def test_red_claims_are_measurable(self):
        h = {
            "pattern": (
                "mean",
                ("mul", "X1", "X2"),
                {"dim": (0,)},
            )
        }
        assert play._claim_measurable("red_0", h)
