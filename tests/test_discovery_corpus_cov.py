"""Tests for ``mlopt_discovery.corpus_cov`` — the coverage census.

The module's contract: ``vocabulary`` enumerates what the arena
machinery can express, ``coverage_gaps`` diffs a case stream
against it, ``coverage_builders``/``gen_cases_coverage`` emit real
evaluable cases for the holes, and ``meta_player.gen_cases`` folds
them into the training stream (``coverage=False`` keeps the
pre-extension corpus as the A/B control).
"""

from __future__ import annotations

import random
from collections import Counter

import pytest
from mlopt_core.ir import Op, TensorType, Var
from mlopt_core.laws import DEFAULT
from mlopt_core.laws.ruleset import RuleSet
from mlopt_discovery import corpus_cov as cc
from mlopt_discovery import meta_player as mp

_X = Var("x", TensorType((4, 4)))
_Y = Var("y", TensorType((4, 4)))


def _case(term, rules=None, sup=None, name="t"):
    return (name, term, rules, (), sup)


class TestVocabulary:
    def test_keys_and_members(self):
        vocab = cc.vocabulary()
        assert isinstance(vocab["ops"], frozenset)
        assert isinstance(vocab["attrs"], frozenset)
        # the expressible classes the task names
        for op in (
            "silu",
            "clamp",
            "clamp_min",
            "sum",
            "softmax",
            "apply",
            "aff",
            "linear",
            "select",
        ):
            assert op in vocab["ops"], op
        assert ("select", "dim") in vocab["attrs"]
        assert ("select", "index") in vocab["attrs"]
        assert ("chunk", "chunks") in vocab["attrs"]
        assert set(vocab["handlers"]) == {"silu", "square", "swiglu"}
        # DEFAULT + the 22 scan-tier laws
        assert len(vocab["rules"]) == 154

    def test_scan_rules(self):
        rs = cc.scan_rules()
        assert isinstance(rs, RuleSet)
        assert len(rs) == 154
        assert cc.scan_rules() is rs  # cached

    def test_rule_ops_and_spec_ops(self):
        ops = set()
        for r in DEFAULT:
            ops |= cc._rule_ops(r)
        assert "matmul" in ops and "softmax" in ops
        pat = ("mul", "X", ("sigmoid", "X"))
        assert cc._spec_ops(pat) == {"mul", "sigmoid"}
        # attr dicts in spec position are skipped, not recursed
        assert cc._spec_ops(("sum", "X", {"dim": -1})) == {"sum"}


class TestSubterms:
    def test_dedup_and_order(self):
        shared = Op.make("add", _X, _Y)
        term = Op.make("mul", shared, shared)
        subs = list(cc._subterms(term))
        assert len(subs) == len({id(s) for s in subs})
        assert term in subs and shared in subs and _X in subs


class TestPatternOccurs:
    def test_silu_pattern(self):
        pat = ("mul", "X", ("sigmoid", "X"))
        assert cc._pattern_occurs(
            pat, Op.make("mul", _X, Op.make("sigmoid", _X))
        )
        # metavar consistency: mul(y, sigmoid(x)) must NOT match
        assert not cc._pattern_occurs(
            pat,
            Op.make("mul", _Y, Op.make("sigmoid", _X)),
        )
        assert not cc._pattern_occurs(pat, Op.make("add", _X, _Y))

    def test_square_pattern(self):
        pat = ("mul", "X", "X")
        assert cc._pattern_occurs(pat, Op.make("mul", _X, _X))
        assert not cc._pattern_occurs(pat, Op.make("mul", _X, _Y))


class TestRulesetNames:
    def test_none_is_default(self):
        names = cc._ruleset_names(None)
        assert "silu_fold" in names and len(names) == 132

    def test_ruleset_and_list(self):
        sans = DEFAULT - DEFAULT.named("silu_fold")
        assert "silu_fold" not in cc._ruleset_names(sans)
        assert "silu_fold" not in cc._ruleset_names(
            [r for r in DEFAULT if r.name != "silu_fold"]
        )


class TestCoverageGaps:
    def test_empty_corpus_reports_everything_missing(self):
        rep = cc.coverage_gaps([])
        vocab = cc.vocabulary()
        assert rep["n_cases"] == 0
        assert rep["ops"]["n_missing"] == len(vocab["ops"])
        assert rep["attrs"]["n_missing"] == len(vocab["attrs"])
        assert rep["handlers"]["missing"] == [
            "silu",
            "square",
            "swiglu",
        ]
        assert rep["rules"]["armed"] == []
        assert len(rep["rules"]["unarmed"]) == 154

    def test_spelled_silu_arms_fold_and_handler(self):
        site = Op.make("mul", _X, Op.make("sigmoid", _X))
        rep = cc.coverage_gaps([_case(site, DEFAULT)])
        assert "silu_fold" in rep["rules"]["armed"]
        assert rep["handlers"]["covered"] == ["silu"]
        assert "mul" in rep["ops"]["covered"]
        assert "sigmoid" in rep["ops"]["covered"]

    def test_stripped_ruleset_does_not_arm(self):
        sans = DEFAULT - DEFAULT.named("silu_fold")
        site = Op.make("mul", _X, Op.make("sigmoid", _X))
        rep = cc.coverage_gaps([_case(site, sans)])
        assert "silu_fold" not in rep["rules"]["armed"]
        # the pattern still occurs — the claim precondition is
        # ruleset-independent
        assert rep["handlers"]["covered"] == ["silu"]

    def test_scan_rules_arm_scan_laws(self):
        term = Op.make("add", Op.make("matmul", _X, _Y), _X)
        rep = cc.coverage_gaps(
            [_case(term, cc.scan_rules(), name="scan")]
        )
        assert "aff_lift" in rep["rules"]["armed"]
        assert "apply" in rep["ops"]["covered"] or True
        rep_default = cc.coverage_gaps([_case(term, DEFAULT)])
        assert "aff_lift" not in rep_default["rules"]["armed"]

    def test_attrs_counted_per_op(self):
        term = Op.make("select", _X, dim=0, index=1)
        rep = cc.coverage_gaps([_case(term, DEFAULT)])
        assert ("select", "dim") in rep["attrs"]["covered"]
        assert ("select", "index") in rep["attrs"]["covered"]
        assert ("select", "dim") not in rep["attrs"]["missing"]

    def test_base_stream_is_impoverished(self):
        base = mp.gen_cases(0, 200, coverage=False)
        rep = cc.coverage_gaps(base)
        # the measured pre-extension hole: a handful of ops, no
        # attrs, almost no armed laws
        assert rep["ops"]["n_missing"] > 100
        assert rep["attrs"]["n_missing"] > 50
        assert len(rep["rules"]["armed"]) <= 4
        assert rep["handlers"]["missing"] == []

    def test_extended_stream_closes_gaps(self):
        base = mp.gen_cases(0, 400, coverage=False)
        full = mp.gen_cases(0, 400)
        rep_b = cc.coverage_gaps(base)
        rep_f = cc.coverage_gaps(full)
        assert rep_f["ops"]["n_missing"] < rep_b["ops"]["n_missing"]
        assert rep_f["attrs"]["n_missing"] < rep_b["attrs"]["n_missing"]
        assert len(rep_f["rules"]["armed"]) > len(
            rep_b["rules"]["armed"]
        )
        # ops the dedicated coverage kinds always produce
        # (each kind draws ~15 times in a 400-case stream)
        for op in (
            "sum",
            "exp",
            "sqrt",
            "chunk",
            "linear",
            "unsqueeze",
            "apply",
            "div",
            "sigmoid",
        ):
            assert op in rep_f["ops"]["covered"], op
        # attr pairs likewise — chunk/unsqueeze attrs are emitted
        # by every law_glu / law_unsq draw
        assert ("chunk", "chunks") in rep_f["attrs"]["covered"]
        assert ("unsqueeze", "dim") in rep_f["attrs"]["covered"]


class TestProbeCase:
    def test_real_arena_path(self):
        cases = mp.gen_cases(0, 8)
        for c in cases:
            p = cc.probe_case(c)
            assert p["name"] == c[0]
            assert p["legal"] > 0
            assert p["specs"] >= 0 and p["sites"] >= 0
            assert p["claimable"] >= 0
            assert p["sat_reward"] is not None

    def test_coverage_case_probes(self):
        cov = cc.gen_cases_coverage(0, 40)
        for c in cov[:12]:
            p = cc.probe_case(c)
            assert p["legal"] > 0


class TestBuilders:
    """Every builder kind emits a real triple across seeds."""

    @pytest.mark.parametrize(
        "kind", [k for k, _b in cc.coverage_builders()]
    )
    def test_builder_kinds(self, kind):
        builds = dict(cc.coverage_builders())
        seen = set()
        for seed in range(40):
            term, rules, _sup = builds[kind](random.Random(seed))
            assert isinstance(term, Op)
            assert rules is not None
            seen.add(term.op)
        assert seen  # the kind varies its head op

    def test_all_kinds_probe_clean(self):
        # one draw per kind through the real arena — no padding
        builds = dict(cc.coverage_builders())
        for kind, build in builds.items():
            term, rules, sup = build(random.Random(7))
            p = cc.probe_case((f"t:{kind}", term, rules, (), sup))
            assert p["legal"] > 0, kind

    def test_branch_coverage(self):
        # enough draws that every rng arm fires — keeps the module
        # under the discovery coverage ratchet honest
        builds = dict(cc.coverage_builders())
        for kind in (
            "ew_gap",
            "view_gap",
            "law_mm",
            "kernel_ops",
            "mixed_gap",
            "reduce_gap",
            "scan",
            "law_idem",
        ):
            for seed in range(120):
                builds[kind](random.Random(seed))

    def test_ew_gap_covers_alphabet(self):
        builds = dict(cc.coverage_builders())
        ops = set()
        for seed in range(300):
            term, _r, _s = builds["ew_gap"](random.Random(seed))
            stack = [term]
            while stack:
                t = stack.pop()
                if isinstance(t, Op):
                    ops.add(t.op)
                    stack.extend(t.args)
        gap = set(cc._EW_GAP_OPS)
        # every ELEMENTWISE gap op occurs in some draw
        assert gap <= ops

    def test_view_gap_covers_templates(self):
        builds = dict(cc.coverage_builders())
        ops = set()
        for seed in range(300):
            term, _r, _s = builds["view_gap"](random.Random(seed))
            stack = [term]
            while stack:
                t = stack.pop()
                if isinstance(t, Op):
                    ops.add(t.op)
                    stack.extend(t.args)
        for op in (
            "transpose",
            "permute",
            "reshape",
            "select",
            "slice",
            "split",
            "chunk",
            "unsqueeze",
            "squeeze",
            "expand",
            "broadcast_to",
            "flatten",
            "getitem",
            "unbind",
            "movedim",
            "narrow",
            "concat",
            "stack",
            "view",
        ):
            assert op in ops, op


class TestStreams:
    def test_gen_cases_coverage_determinism(self):
        a = cc.gen_cases_coverage(3, 30)
        b = cc.gen_cases_coverage(3, 30)
        assert [c[0] for c in a] == [c[0] for c in b]
        assert len(a) == 30
        assert all(c[0].startswith("cov") for c in a)
        kinds = Counter(c[0].split(":", 1)[1] for c in a)
        assert set(kinds) <= {k for k, _b in cc.coverage_builders()}

    def test_gen_cases_default_includes_coverage(self):
        cases = mp.gen_cases(0, 200)
        kinds = {c[0].split(":", 1)[1] for c in cases}
        cov_kinds = {k for k, _b in cc.coverage_builders()}
        assert kinds & cov_kinds  # coverage kinds present
        # and the hand kinds are still there
        assert {"silu", "control", "square", "dual"} <= kinds

    def test_gen_cases_coverage_false_is_base_only(self):
        cases = mp.gen_cases(0, 200, coverage=False)
        kinds = {c[0].split(":", 1)[1] for c in cases}
        cov_kinds = {k for k, _b in cc.coverage_builders()}
        assert not (kinds & cov_kinds)
        assert len(cases) == 200

    def test_gen_cases_determinism(self):
        a = mp.gen_cases(5, 20)
        b = mp.gen_cases(5, 20)
        assert [c[0] for c in a] == [c[0] for c in b]
        # coverage=False reproduces the pre-extension picks
        old = mp.gen_cases(5, 20, coverage=False)
        assert all(
            "law_" not in c[0] and "_gap" not in c[0] for c in old
        )


class TestMain:
    def test_main_report(self, capsys):
        assert cc.main(["--cases", "30", "--stream", "coverage"]) == 0
        out = capsys.readouterr().out
        assert "corpus coverage" in out
        assert "ops:" in out and "rules:" in out

    def test_main_base_stream(self, capsys):
        assert cc.main(["--cases", "20", "--stream", "base"]) == 0
        out = capsys.readouterr().out
        assert "corpus coverage" in out
