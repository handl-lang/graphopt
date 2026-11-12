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
            ({"name": "x", "board": "torch"}, "board"),
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
        assert summary["recorded_only"]["hardware"]["device"] == "cpu"

    def test_main(self, capsys):
        rc = ds.main(["--spec", "pointwise", "--cases", "4"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '"name": "pointwise"' in out
        assert "cases: 4" in out
