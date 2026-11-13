"""Domain-spec files — Stage-2 domains-as-data, first cut.

Pins: spec loading/validation, the vocab→kind eligibility census,
the per-case vocab/rules enforcement, handler-subset boarding, and
the ``play --domain-spec`` registration path end to end.
"""

import json

import pytest
from mlopt_discovery import domain_spec as ds
from mlopt_discovery import lawdata, play
from mlopt_discovery import meta_player as mp


def _spec(**over):
    data = {
        "name": "t",
        "vocab": {"ops": ["add", "mul", "sigmoid", "tanh"]},
    }
    data.update(over)
    return ds._spec_from_dict(data)


class TestLoad:
    def test_packaged_name(self):
        spec = ds.load_spec("pointwise")
        assert spec.name == "pointwise"
        assert spec.board == "meta-arena"
        assert spec.incumbent == "scripted"
        assert spec.handler_tags == ("silu", "square")
        assert spec.rules_profile == "default"
        assert (
            spec.vocab_ops is not None
            and "matmul" not in spec.vocab_ops
        )

    def test_dict_source(self):
        spec = ds.load_spec({"name": "x", "player": "meta"})
        assert spec.name == "x"
        assert spec.corpus_kinds is None
        assert spec.vocab_ops is None

    def test_json_path(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps({"name": "pathed"}))
        assert ds.load_spec(p).name == "pathed"

    def test_missing(self):
        with pytest.raises(ValueError, match="not found"):
            ds.load_spec("no-such-spec-xyz")

    @pytest.mark.parametrize(
        "data,match",
        [
            ({"name": "Bad Name"}, "name"),
            ({"name": "x", "board": "bogus"}, "board"),
            ({"name": "x", "player": "nope"}, "player"),
            ({"name": "x", "incumbent": {"kind": "warp"}}, "incumbent"),
            (
                {"name": "x", "vocab": {"handler_tags": ["nope"]}},
                "handler_tags",
            ),
            ({"name": "x", "vocab": {"rules": "bogus"}}, "rules"),
            ({"name": "x", "corpus": {"kinds": "silu"}}, "kinds"),
            ({"name": "x", "shapes": []}, "shapes"),
        ],
    )
    def test_validation(self, data, match):
        with pytest.raises(ValueError, match=match):
            ds.load_spec(data)


class TestEligibility:
    def test_pointwise_kinds(self):
        spec = ds.load_spec("pointwise")
        kinds = ds._eligible_kinds(spec)
        # contraction/view/scan kinds are out; pointwise kinds in
        assert "law_mm" not in kinds
        assert "view_gap" not in kinds
        assert "scan" not in kinds
        assert {
            "law_silu",
            "ew_gap" if "ew_gap" in kinds else "mixed_gap",
        } & set(kinds)
        assert "silu" in kinds and "mixed_gap" in kinds

    def test_kinds_intersection(self):
        spec = _spec(corpus={"kinds": ["silu", "control"]})
        # 'control' spells matmul — outside the ops whitelist
        kinds = ds._eligible_kinds(spec)
        assert kinds == ["silu"]

    def test_no_eligible(self):
        spec = _spec(vocab={"ops": ["nonexistent_op"]})
        with pytest.raises(ValueError, match="no eligible"):
            ds._eligible_kinds(spec)

    def test_unknown_corpus_kind(self):
        spec = _spec(corpus={"kinds": ["bogus_kind"]})
        # footprint sampling never sees it → missing-kinds error or
        # empty eligibility; both are honest refusals
        with pytest.raises(ValueError):
            ds._eligible_kinds(spec)


class TestCases:
    def test_vocab_invariant(self):
        spec = ds.load_spec("pointwise")
        kinds = ds._eligible_kinds(spec)
        cases = ds._make_cases(spec, kinds)(0, 10)
        assert len(cases) == 10
        for name, term, *_ in cases:
            assert name.startswith("spec")
            ops, _attrs = ds._term_ops_attrs(term)
            assert ops <= spec.vocab_ops

    def test_deterministic(self):
        spec = _spec(corpus={"kinds": ["silu"]})
        kinds = ds._eligible_kinds(spec)
        a = ds._make_cases(spec, kinds)(3, 4)
        b = ds._make_cases(spec, kinds)(3, 4)
        assert [c[0] for c in a] == [c[0] for c in b]

    def test_rules_profile_default_excludes_scan(self):
        from mlopt_core.laws import DEFAULT
        from mlopt_core.laws.scan import SCAN_DIAG_LAWS, SCAN_LAWS

        spec = ds.load_spec("pointwise")
        kinds = ds._eligible_kinds(spec)
        cases = ds._make_cases(spec, kinds)(0, 30)
        scan_only = {
            r.name
            for r in (*SCAN_LAWS, *SCAN_DIAG_LAWS)
            if r.name not in {d.name for d in DEFAULT}
        }
        for _n, _t, rules, *_ in cases:
            names = (
                {r.name for r in rules} if rules is not None else set()
            )
            assert not (names & scan_only)


class TestBoard:
    def test_handler_subset(self):
        spec = ds.load_spec("pointwise")
        board_of = ds.BOARDS[spec.board](spec)
        case = mp.gen_cases(0, 4, kinds=["swiglu"])[0]
        board = board_of(case)
        assert set(board.handlers) == {"silu", "square"}

    def test_full_handlers_when_unset(self):
        spec = _spec()
        board_of = ds.BOARDS[spec.board](spec)
        case = mp.gen_cases(0, 2, kinds=["silu"])[0]
        board = board_of(case)
        assert set(board.handlers) == set(lawdata.HANDLERS)


class TestGenCasesKinds:
    def test_filter(self):
        cases = mp.gen_cases(0, 10, kinds=["silu"])
        assert {c[0].split(":", 1)[1] for c in cases} == {"silu"}

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown kinds"):
            mp.gen_cases(0, 2, kinds=["bogus"])


class TestRegister:
    def test_register_into_dict(self):
        domains: dict = {}
        spec = _spec(name="regtest")
        domain = ds.register_domain(spec, domains=domains)
        assert domains["regtest"] is domain
        assert domain.spec is not None
        cases = domain.cases(0, 2)
        assert len(cases) == 2

    def test_play_cli(self, capsys):
        try:
            rc = play.main(
                [
                    "--domain-spec",
                    "pointwise",
                    "--domain",
                    "pointwise",
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "1",
                    "--budget",
                    "6",
                ]
            )
            assert rc == 0
            out = capsys.readouterr().out
            assert "domain pointwise" in out
            assert "scripted" in out
        finally:
            # registration is the feature — don't leak it into the
            # shared registry other tests census
            play.DOMAINS.pop("pointwise", None)

    def test_play_cli_unknown_domain(self, capsys):
        rc = play.main(["--domain", "nope"])
        assert rc == 2
        assert "unknown domain" in capsys.readouterr().out


class TestSummaryAndCLI:
    def test_spec_summary(self):
        spec = ds.load_spec("pointwise")
        summary = ds.spec_summary(spec)
        assert summary["name"] == "pointwise"
        assert "silu" in summary["corpus"]["kinds_eligible"]
        assert summary["vocab"]["ops"] == 51
        assert summary["hardware"]["device"] == "cpu"

    def test_main(self, capsys):
        rc = ds.main(["--spec", "pointwise", "--cases", "4"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '"name": "pointwise"' in out
        assert "cases: 4" in out

    def test_main_torch_board(self, capsys):
        pytest.importorskip("torch")
        rc = ds.main(["--spec", "torch-silu", "--cases", "3"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '"board": "torch"' in out
        assert "cases: 3" in out


class TestPlayerPrior:
    def test_prior_none_parses(self):
        spec = ds.load_spec(
            {"name": "x", "player": {"kind": "meta", "prior": "none"}}
        )
        assert spec.player == "meta"
        assert spec.player_prior == "none"

    def test_prior_default_normalizes(self):
        spec = ds.load_spec(
            {"name": "x", "player": {"kind": "meta", "prior": "default"}}
        )
        assert spec.player_prior is None

    def test_prior_inline_object(self):
        spec = ds.load_spec(
            {
                "name": "x",
                "player": {"kind": "meta", "prior": {"op:extract": 1.5}},
            }
        )
        assert spec.player_prior == {"op:extract": 1.5}

    def test_prior_bad_kind(self):
        with pytest.raises(ValueError, match="player"):
            ds.load_spec({"name": "x", "player": {"kind": "nope"}})

    def test_load_prior_none(self):
        assert ds.load_prior("none") == {}

    def test_load_prior_packaged(self):
        w = ds.load_prior("pointwise")
        assert w and all(isinstance(v, float) for v in w.values())
        assert "op:extract" in w

    def test_load_prior_path_and_inline(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"bias": 0.25}))
        assert ds.load_prior(str(p)) == {"bias": 0.25}
        assert ds.load_prior({"bias": 1}) == {"bias": 1.0}

    def test_load_prior_errors(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            ds.load_prior("no-such-prior-xyz")
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([1, 2]))
        with pytest.raises(ValueError, match="object"):
            ds.load_prior(str(bad))
        bad2 = tmp_path / "bad2.json"
        bad2.write_text(json.dumps({"a": "x"}))
        with pytest.raises(ValueError, match="number"):
            ds.load_prior(str(bad2))
        with pytest.raises(ValueError, match="expected"):
            ds.load_prior(3)

    def test_resolve_player_prior_none(self):
        spec = ds.load_spec(
            {"name": "x", "player": {"kind": "meta", "prior": "none"}}
        )
        dom = ds.build_domain(spec)
        assert dom.spec.weights == {}

    def test_resolve_player_packaged_prior(self):
        spec = ds.load_spec(
            {"name": "x", "player": {"kind": "meta", "prior": "pointwise"}}
        )
        dom = ds.build_domain(spec)
        assert dom.spec.weights is not None
        assert dom.spec.weights.get("op:extract") == pytest.approx(
            ds.load_prior("pointwise")["op:extract"]
        )

    def test_default_prior_is_meta_spec(self):
        spec = ds.load_spec({"name": "x"})
        dom = ds.build_domain(spec)
        assert dom.spec is mp.META_SPEC


class TestRulesMinus:
    def test_minus_profile_parses(self):
        from mlopt_core.laws import DEFAULT

        spec = ds.load_spec(
            {
                "name": "x",
                "vocab": {
                    "rules": {"base": "default", "minus": ["select_mul"]}
                },
            }
        )
        assert spec.rules_profile[0] == "minus"
        allowed = ds._rule_allowlist(spec.rules_profile)
        assert "select_mul" not in allowed
        assert len(allowed) == len({r.name for r in DEFAULT}) - 1

    def test_minus_all_base(self):
        spec = ds.load_spec(
            {
                "name": "x",
                "vocab": {"rules": {"base": "all", "minus": ["select_mul"]}},
            }
        )
        allowed = ds._rule_allowlist(spec.rules_profile)
        assert allowed is not None and "select_mul" not in allowed

    def test_minus_bad_shape(self):
        with pytest.raises(ValueError, match="minus"):
            ds.load_spec(
                {"name": "x", "vocab": {"rules": {"base": "all", "minus": 3}}}
            )

    def test_minus_end_to_end(self):
        # subtract a scan-only law: DEFAULT-ruleset cases still pass,
        # scan-extended cases carrying it are filtered out
        from mlopt_discovery import corpus_cov
        from mlopt_core.laws import DEFAULT

        dflt = {r.name for r in DEFAULT}
        victim = next(
            r.name
            for r in corpus_cov.scan_rules()
            if r.name not in dflt
        )
        spec = ds.load_spec(
            {
                "name": "x",
                "vocab": {"rules": {"base": "all", "minus": [victim]}},
                "corpus": {"kinds": ["silu"]},
            }
        )
        kinds = ds._eligible_kinds(spec)
        cases = ds._make_cases(spec, kinds)(0, 6)
        assert cases
        for _n, _t, rules, *_ in cases:
            names = {r.name for r in rules} if rules is not None else set()
            assert victim not in names


class TestTorchBoard:
    def _tspec(self, **over):
        data = {
            "name": "t-torch",
            "board": "torch",
            "corpus": {"modules": ["lin"]},
            "shapes": {"widths": [8], "batch": 4},
        }
        data.update(over)
        return ds.load_spec(data)

    def test_packaged(self):
        pytest.importorskip("torch")
        spec = ds.load_spec("torch-silu")
        assert spec.board == "torch"
        assert spec.corpus_modules == ("spelled_silu", "mlp")
        dom = ds.build_domain(spec)
        cases = dom.cases(0, 4)
        assert len(cases) == 4
        assert dom.deliver is not None

    def test_cases_and_board(self):
        pytest.importorskip("torch")
        spec = self._tspec()
        dom = ds.build_domain(spec)
        cases = dom.cases(0, 3)
        assert all(c[0].startswith("torch") for c in cases)
        assert all(":lin" in c[0] for c in cases)
        board = dom.board_of(cases[0])
        assert isinstance(board.supported, frozenset)

    def test_vocab_ops_filter(self):
        pytest.importorskip("torch")
        # 'linear' is not a lifted IR op name — pick a real exclusion
        spec = ds.load_spec(
            {
                "name": "t-torch",
                "board": "torch",
                "corpus": {"modules": ["lin", "mlp"]},
                "shapes": {"widths": [8]},
                "vocab": {"ops": ["definitely_not_an_op"]},
            }
        )
        cases_fn = ds._make_torch_cases(spec)
        assert cases_fn(0, 4) == []

    def test_rules_profiles_rejected(self):
        pytest.importorskip("torch")
        # an empty explicit allowlist admits none of DEFAULT's names
        spec = self._tspec(vocab={"rules": []})
        with pytest.raises(ValueError, match="DEFAULT"):
            ds._make_torch_cases(spec)

    def test_shapes_hardware_validation(self):
        pytest.importorskip("torch")
        spec = ds.load_spec(
            {
                "name": "t-torch",
                "board": "torch",
                "shapes": {"widths": [0]},
            }
        )
        with pytest.raises(ValueError, match="widths"):
            ds._make_torch_cases(spec)
        spec = ds.load_spec(
            {
                "name": "t-torch",
                "board": "torch",
                "shapes": {"batch": -1},
            }
        )
        with pytest.raises(ValueError, match="batch"):
            ds._make_torch_cases(spec)
        spec = ds.load_spec(
            {
                "name": "t-torch",
                "board": "torch",
                "hardware": {"device": 3},
            }
        )
        with pytest.raises(ValueError, match="device"):
            ds._make_torch_cases(spec)

    def test_dtype_consumed(self):
        pytest.importorskip("torch")
        spec = ds.load_spec(
            {
                "name": "t-torch",
                "board": "torch",
                "corpus": {"modules": ["lin"]},
                "shapes": {"widths": [8]},
                "hardware": {"dtype": "float64"},
            }
        )
        cases = ds._make_torch_cases(spec)(0, 1)
        assert cases[0][5]["input"].dtype.__str__().endswith("float64")

    def test_kinds_rejected_on_torch(self):
        with pytest.raises(ValueError, match="corpus.kinds"):
            ds.load_spec(
                {
                    "name": "x",
                    "board": "torch",
                    "corpus": {"kinds": ["silu"]},
                }
            )

    def test_modules_rejected_on_meta_arena(self):
        with pytest.raises(ValueError, match="corpus.modules"):
            ds.load_spec(
                {
                    "name": "x",
                    "corpus": {"modules": ["lin"]},
                }
            )

    def test_unknown_module(self):
        pytest.importorskip("torch")
        spec = ds.load_spec(
            {
                "name": "t-torch",
                "board": "torch",
                "corpus": {"modules": ["bogus"]},
            }
        )
        with pytest.raises(ValueError, match="unknown torch builder"):
            ds._make_torch_cases(spec)(0, 1)

    def test_torch_cases_param_edges(self):
        pytest.importorskip("torch")
        with pytest.raises(ValueError, match="widths"):
            play._torch_cases(0, 1, widths=[0])
        with pytest.raises(ValueError, match="dtype"):
            play._torch_cases(0, 1, dtype="int4")
        cases = play._torch_cases(
            0, 2, modules=["lin"], widths=[8], batch=2,
            device="cpu", dtype="float64",
        )
        assert cases[0][5]["input"].shape == (2, 8)
        assert "float64" in str(cases[0][5]["input"].dtype)

    def test_deliver_field(self):
        pytest.importorskip("torch")
        # "none" seats no hook even on the torch board
        spec = self._tspec(deliver="none")
        assert ds.build_domain(spec).deliver is None
        # the named hook and "auto" both seat _torch_deliver
        for spelling in ("torch", "auto", True):
            spec = self._tspec(deliver=spelling)
            assert spec.deliver == ("auto" if spelling is True else spelling)
            assert ds.build_domain(spec).deliver is not None
        # meta-arena: auto is hookless (symbolic — meaningless),
        # the torch hook is refused at load
        spec = ds.load_spec({"name": "m", "deliver": False})
        assert ds.build_domain(spec).deliver is None
        with pytest.raises(ValueError, match="deliver 'torch'"):
            ds.load_spec({"name": "m2", "deliver": "torch"})
        with pytest.raises(ValueError, match="deliver 'bogus'"):
            ds.load_spec({"name": "m3", "deliver": "bogus"})
        # the census reports the field
        assert ds.spec_summary(self._tspec(deliver="none"))["deliver"] == "none"

    def test_deliver_smoke(self):
        pytest.importorskip("torch")
        from gamekit import engine

        from mlopt_discovery import meta_arena as ma

        spec = self._tspec()
        dom = ds.build_domain(spec)
        case = dom.cases(0, 6)
        silu = next((c for c in case if "spelled_silu" in c[0]), None)
        target = silu if silu is not None else case[0]
        traj = engine.run_episode(
            dom.board_of(target), ma.ScriptedPlayer(), budget=8
        )
        res = dom.deliver(dom.board_of(target), target, traj)
        assert res["delivered"] and res["verified"]


class TestBakePrior:
    def test_tiny_bake(self):
        spec = ds.load_spec("pointwise")
        w = ds.bake_prior(
            spec,
            cases=4,
            seed=0,
            budget=8,
            imitate_epochs=1,
            rl_epochs=1,
        )
        assert isinstance(w, dict) and w
        assert "op:extract" in w
        assert all(isinstance(v, float) for v in w.values())

    def test_bake_cold_init(self):
        spec = ds.load_spec({"name": "x", "corpus": {"kinds": ["silu"]}})
        w = ds.bake_prior(
            spec, cases=2, seed=0, budget=6, imitate_epochs=1, init="cold"
        )
        assert isinstance(w, dict)

    def test_bad_init(self):
        spec = ds.load_spec({"name": "x"})
        with pytest.raises(ValueError, match="init"):
            ds.bake_prior(spec, cases=1, init="bogus")


class TestCodeBoards:
    """Stage-3: the code-shaped boards as spec-able data."""

    def test_boards_registered(self):
        ds.load_spec({"name": "x"})  # seats the lazy registries
        for b in (
            "gen", "joint", "search", "fragment",
            "open", "player", "bilinear",
        ):
            assert b in ds.BOARDS and b in ds._EXTRA_CASES

    def test_board_defaults(self):
        cases = {
            "search": ("search", "search"),
            "fragment": ("fragment", "fragment"),
            "joint": ("meta", "scripted"),
            "open": ("open", "none"),
            "player": ("player", "player"),
            "bilinear": ("bilinear", "bilinear"),
            "gen": ("meta", "scripted"),
        }
        for board, (player, inc) in cases.items():
            spec = ds.load_spec({"name": f"d-{board}", "board": board})
            assert spec.player == player
            assert spec.incumbent == inc

    def test_cross_board_player_refused(self):
        with pytest.raises(ValueError, match="player spec"):
            ds.load_spec({"name": "x", "board": "search", "player": "meta"})
        with pytest.raises(ValueError, match="incumbent"):
            ds.load_spec(
                {"name": "x", "board": "search", "incumbent": "scripted"}
            )

    def test_vocab_refused_on_nonterm_boards(self):
        for board in ("open", "player", "bilinear"):
            with pytest.raises(ValueError, match="vocab"):
                ds.load_spec(
                    {"name": "x", "board": board, "vocab": {"ops": ["add"]}}
                )

    def test_corpus_key_refusals(self):
        with pytest.raises(ValueError, match="corpus.ops"):
            ds.load_spec({"name": "x", "board": "search", "corpus": {"ops": ["add"]}})
        with pytest.raises(ValueError, match="corpus.kinds"):
            ds.load_spec({"name": "x", "board": "open", "corpus": {"kinds": ["a"]}})
        with pytest.raises(ValueError, match="corpus.modules"):
            ds.load_spec({"name": "x", "board": "gen", "corpus": {"modules": ["mlp"]}})
        with pytest.raises(ValueError, match="corpus.page"):
            ds.load_spec({"name": "x", "board": "search", "corpus": {"page": 4}})

    def test_search_cases_and_rules(self):
        spec = ds.load_spec(
            {
                "name": "s",
                "board": "search",
                "corpus": {"kinds": ["silu", "addid"], "horizon": 6},
                "vocab": {"rules": ["silu_fold", "id_add"]},
            }
        )
        dom = ds.build_domain(spec)
        cases = dom.cases(0, 4)
        assert len(cases) == 4
        assert {c[0].split(":")[-1] for c in cases} <= {"silu", "addid"}
        board = dom.board_of(cases[0])
        assert {r.name for r in board.env.rules} == {"silu_fold", "id_add"}
        # an unknown pool name refuses on the first draw
        bad = ds.build_domain(
            ds.load_spec(
                {
                    "name": "s2",
                    "board": "search",
                    "corpus": {"kinds": ["bogus"]},
                }
            )
        )
        with pytest.raises(ValueError, match="pool"):
            bad.cases(0, 1)

    def test_search_empty_ruleset_refused(self):
        # a scan-only rule name parses (the scan universe knows it)
        # but seats nothing on the search board's DEFAULT ruleset
        spec = ds.load_spec(
            {"name": "s3", "board": "search", "vocab": {"rules": ["aff_lift"]}}
        )
        with pytest.raises(ValueError, match="leaves no rules"):
            ds.build_domain(spec)
        # a name outside the scan universe refuses when the
        # allowlist resolves (board build)
        with pytest.raises(ValueError, match="unknown rule names"):
            ds.build_domain(
                ds.load_spec(
                    {
                        "name": "s4",
                        "board": "search",
                        "vocab": {"rules": ["prim:nope"]},
                    }
                )
            )

    def test_fragment_board(self):
        spec = ds.load_spec(
            {
                "name": "f",
                "board": "fragment",
                "corpus": {"kinds": ["mulsig"], "max_moves": 32},
                "vocab": {"handler_tags": ["silu"]},
            }
        )
        dom = ds.build_domain(spec)
        cases = dom.cases(0, 3)
        assert all("mulsig" in c[0] for c in cases)
        board = dom.board_of(cases[0])
        assert set(board.handlers) == {"silu"}

    def test_joint_kinds_and_board(self):
        spec = ds.load_spec(
            {"name": "j", "board": "joint", "corpus": {"kinds": ["0", "2"]}}
        )
        dom = ds.build_domain(spec)
        cases = dom.cases(0, 6)
        assert cases
        assert {c[0].split(":")[-1] for c in cases} <= {"0", "2"}
        assert (
            type(dom.board_of(cases[0])).__name__ == "MetaArena"
        )

    def test_open_player_bilinear_cases(self):
        o = ds.build_domain(
            ds.load_spec(
                {
                    "name": "o",
                    "board": "open",
                    "corpus": {"ops": ["add", "mul"], "page": 8, "start": 1},
                }
            )
        )
        ocs = o.cases(0, 2)
        assert ocs[0]["page"] == 8 and ocs[0]["ops"] == ("add", "mul")
        assert type(o.board_of(ocs[0])).__name__ == "OpenArena"
        p = ds.build_domain(
            ds.load_spec(
                {"name": "p", "board": "player", "corpus": {"objective": "open", "max_evals": 4}}
            )
        )
        pcs = p.cases(0, 1)
        assert pcs[0]["max_evals"] == 4 and pcs[0]["objective"] == "open"
        assert type(p.board_of(pcs[0])).__name__ == "PlayerArena"
        b = ds.build_domain(
            ds.load_spec({"name": "b", "board": "bilinear", "corpus": {"scramble": 4}})
        )
        bcs = b.cases(0, 2)
        assert bcs[0][0] == "bilinear0:naive"
        assert type(b.board_of(bcs[0])).__name__ == "BilinearBoard"

    def test_gen_spec(self):
        pytest.importorskip("torch")
        spec = ds.load_spec("gen-fusion")
        assert spec.board == "gen" and spec.corpus_builders
        dom = ds.build_domain(spec)
        assert dom.deliver is not None  # "auto" → _torch_deliver
        cases = dom.cases(0, 2)
        assert len(cases) == 2
        assert {c[0].split(":")[-1] for c in cases} <= set(
            spec.corpus_builders
        )

    def test_gen_unknown_builder_refused_at_build(self):
        pytest.importorskip("torch")
        spec = ds.load_spec(
            {"name": "g", "board": "gen", "corpus": {"builders": ["nope"]}}
        )
        with pytest.raises(ValueError, match="corpus.builders"):
            ds.build_domain(spec)

    def test_gen_rules_profile_admission(self):
        pytest.importorskip("torch")
        spec = ds.load_spec(
            {
                "name": "g2",
                "board": "gen",
                "vocab": {"rules": ["silu_fold"]},
            }
        )
        with pytest.raises(ValueError, match="DEFAULT"):
            ds._EXTRA_CASES["gen"](spec)

    def test_bake_prior_refused_off_oracle_boards(self):
        spec = ds.load_spec({"name": "s", "board": "search"})
        with pytest.raises(ValueError, match="bake_prior"):
            ds.bake_prior(spec, cases=1)

    def test_packaged_specs_load(self):
        for name in ("gen-fusion", "search-folds", "attr-bounded"):
            spec = ds.load_spec(name)
            assert spec.name == name
            assert ds.spec_summary(spec)["board"] == spec.board


class TestAttrValues:
    """``vocab.attr_values`` — attribute-VALUE constraints (T-028)."""

    def _av(self, **over):
        data = {
            "name": "t-av",
            "vocab": {"attr_values": {"unsqueeze": {"dim": [0, 1]}}},
        }
        data.update(over)
        return ds.load_spec(data)

    def test_parse_normalizes(self):
        spec = self._av()
        assert spec.attr_values == {"unsqueeze": {"dim": (0, 1)}}
        spec = ds.load_spec(
            {"name": "x", "vocab": {"attr_values": {"clamp": {"min": 0.0}}}}
        )
        assert spec.attr_values == {"clamp": {"min": (0.0,)}}

    @pytest.mark.parametrize(
        "bad",
        [
            {"unsqueeze": 3},
            {"unsqueeze": {"dim": [{"nested": 1}]}},
            {"unsqueeze": {"dim": [[0, 1]]}},
            {"unsqueeze": {3: 0}},
            ["unsqueeze"],
        ],
    )
    def test_validation(self, bad):
        with pytest.raises(ValueError, match="attr_values"):
            ds.load_spec({"name": "x", "vocab": {"attr_values": bad}})

    def test_refused_on_nonterm_boards(self):
        for board in ("open", "player", "bilinear"):
            with pytest.raises(ValueError, match="vocab"):
                ds.load_spec(
                    {
                        "name": "x",
                        "board": board,
                        "vocab": {"attr_values": {"add": {"dim": 0}}},
                    }
                )

    def test_predicate_unit(self):
        from mlopt_core.ir import Op

        spec = self._av()
        av = spec.attr_values
        good = Op.make("unsqueeze", Op.make("add", 1, 2), dim=1)
        bad_dim = Op.make("unsqueeze", Op.make("add", 1, 2), dim=2)
        missing = Op.make("unsqueeze", Op.make("add", 1, 2))
        other = Op.make("add", 1, 2)  # unlisted op — unconstrained
        assert ds._attr_values_ok(good, av)
        assert not ds._attr_values_ok(bad_dim, av)
        assert not ds._attr_values_ok(missing, av)
        assert ds._attr_values_ok(other, av)
        # nested: a violating site anywhere fails the term
        nested = Op.make("mul", good, bad_dim)
        assert not ds._attr_values_ok(nested, av)

    def test_missing_attr_fails(self):
        # a listed attr absent on the node is a violation — "only
        # clamp(min=0) sites" does not admit attr-less clamps
        from mlopt_core.ir import Op

        spec = ds.load_spec(
            {
                "name": "x",
                "vocab": {"attr_values": {"clamp_min": {"min": 0.0}}},
            }
        )
        bare = Op.make("clamp_min", Op.make("add", 1, 2))
        spelled = Op.make("clamp_min", Op.make("add", 1, 2), min=0.0)
        assert not ds._attr_values_ok(bare, spec.attr_values)
        assert ds._attr_values_ok(spelled, spec.attr_values)

    def test_eligibility_drops_violating_kinds(self):
        spec = self._av()
        kinds = ds._eligible_kinds(spec)
        # the dim-2 view kind's footprint sample shows violations
        assert "law_unsq" not in kinds
        assert kinds  # honest: other kinds still draw

    def test_cases_invariant(self):
        spec = self._av(corpus={"kinds": ["silu", "square"]})
        kinds = ds._eligible_kinds(spec)
        cases = ds._make_cases(spec, kinds)(0, 8)
        assert cases
        for _n, term, *_ in cases:
            assert ds._attr_values_ok(term, spec.attr_values)

    def test_vacuous_constraint_keeps_kinds(self):
        spec = ds.load_spec(
            {
                "name": "x",
                "corpus": {"kinds": ["silu"]},
                "vocab": {"attr_values": {"no_such_op": {"a": 1}}},
            }
        )
        # constraining an op the silu kind never spells is vacuous —
        # kinds survive; the per-case filter still guards
        assert ds._eligible_kinds(spec) == ["silu"]

    def test_packaged_attr_bounded(self):
        spec = ds.load_spec("attr-bounded")
        assert spec.attr_values is not None
        assert spec.attr_values["clamp"] == {
            "min": (0.0,),
            "max": (1.0,),
        }
        kinds = ds._eligible_kinds(spec)
        assert kinds
        cases = ds._make_cases(spec, kinds)(0, 10)
        assert len(cases) == 10
        for _n, term, *_ in cases:
            assert ds._attr_values_ok(term, spec.attr_values)
            ops, _attrs = ds._term_ops_attrs(term)
            assert ops <= spec.vocab_ops

    def test_summary_reports_attr_values(self):
        spec = ds.load_spec("attr-bounded")
        av = ds.spec_summary(spec)["vocab"]["attr_values"]
        assert av["unsqueeze"] == {"dim": [0, 1]}
        assert av["clamp"]["min"] == [0.0]
