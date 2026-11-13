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
            ({"name": "x", "board": "gen"}, "board"),
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
