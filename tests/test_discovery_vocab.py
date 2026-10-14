"""Tests for ``catopt_discovery.vocab`` — property-derived op classes.

The vocabulary is derived by *test*, not lookup: a view is arity-1,
value-preserving and shape-changing; pointwise means shape-preserving
and commuting with a battery of views; a reduction returns fewer,
non-preserved elements.  The classifiers run on synthetic term lists
so every verdict — True / False / untestable — is pinned.
"""

import json

import pytest
import torch
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op, TensorType, Var
from catopt_discovery import dspec_synth
from catopt_discovery import vocab as vc


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


_TERMS = [
    _p("add", _v("x", 4, 6), _v("y", 4, 6)),
    _p("mul", _v("x", 4, 6), _v("y", 4, 6)),
    _p("relu", _v("x", 4, 6)),
    _p("transpose", _v("x", 4, 6), dim0=0, dim1=1),
    _p("getitem", _v("x", 4, 6), index=0),
    _p("sum", _v("x", 4, 6), dim=0),
]


def test_leaf_rand_numel_helpers():
    assert vc._numel((2, 3)) == 6
    assert vc._numel(()) == 1
    t = vc._rand((2, 3))
    assert t.shape == (2, 3) and t.dtype == torch.float64
    leaf = vc._leaf((2, 3), "w")
    assert leaf.name == "w" and tuple(leaf.typ.shape) == (2, 3)


def test_concrete_shapes():
    assert vc._concrete((2, 3)) is True
    assert vc._concrete(()) is True
    assert vc._concrete((-1, 3)) is False
    assert vc._concrete((2, "x")) is False
    assert vc._concrete("nope") is False


def test_corpus_ops_mines_arities_and_occurrences():
    x, y = _v("x", 4, 6), _v("y", 4, 6)
    m = _p("mul", x, y)
    terms = [_p("add", m, m)]
    occ, arity = vc.corpus_ops(terms)
    assert arity == {"add": 2, "mul": 2}
    # The shared mul node is one occurrence, not two.
    assert len(occ["mul"]) == 1
    attrs, shapes = occ["add"][0]
    assert attrs == {} and shapes == [(4, 6), (4, 6)]


def test_value_preserving():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert vc._value_preserving(x, torch.tensor([3.0, 1.0])) is True
    assert vc._value_preserving(x, torch.tensor([4.0])) is False


def test_battery_is_shape_agnostic():
    small = vc._battery((4,))
    big = vc._battery((4, 6))
    assert len(small) == 4
    assert len(big) == 5
    assert [p[0] for p in big][-1] == "transpose"


def test_is_view_true_false_untestable():
    occ, _arity = vc.corpus_ops(_TERMS)
    assert vc._is_view("transpose", 1, occ) is True
    assert vc._is_view("getitem", 1, occ) is True
    # Arity != 1 is never a view.
    assert vc._is_view("add", 2, occ) is False
    # A computing unary op preserves shape/values? relu does not
    # preserve values (negatives clamp) — but on positive probes it
    # does; either way it never changes shape → not a view.
    assert vc._is_view("relu", 1, occ) is False
    # An op that cannot evaluate is untestable.
    occ2, _ = vc.corpus_ops([_p("not_a_real_op", _v("x", 2, 2))])
    assert vc._is_view("not_a_real_op", 1, occ2) is None


def test_is_pointwise_true_false_arity_gate():
    occ, _arity = vc.corpus_ops(_TERMS)
    assert vc._is_pointwise("add", 2, occ) is True
    assert vc._is_pointwise("mul", 2, occ) is True
    assert vc._is_pointwise("relu", 1, occ) is True
    # transpose changes shape — not pointwise.
    assert vc._is_pointwise("transpose", 1, occ) is False
    # sum returns a different shape — not pointwise.
    assert vc._is_pointwise("sum", 1, occ) is False
    # Arity 3+ is left undecided, and so is an unevaluable op.
    assert vc._is_pointwise("add", 3, occ) is None
    occ2, _ = vc.corpus_ops([_p("not_a_real_op", _v("x", 2, 2))])
    assert vc._is_pointwise("not_a_real_op", 1, occ2) is None


def test_is_reduction_true_false_untestable():
    occ, _arity = vc.corpus_ops(_TERMS)
    assert vc._is_reduction("sum", 1, occ) is True
    assert vc._is_reduction("add", 2, occ) is False
    # An op that cannot evaluate is untestable.
    occ2, _ = vc.corpus_ops([_p("not_a_real_op", _v("x", 2, 2))])
    assert vc._is_reduction("not_a_real_op", 1, occ2) is None


def test_commutes_compose_and_failure():
    # A genuinely pointwise op commutes with a reshape probe.
    assert (
        vc._commutes("neg", {}, [(2, 3)], ("reshape", {"shape": (6,)}))
        is True
    )
    # A malformed arity-1 "add" cannot evaluate on either side —
    # the probe is no evidence either way.
    assert (
        vc._commutes("add", {}, [()], ("reshape", {"shape": (1,)}))
        is None
    )


def test_classify_synthetic_corpus():
    v = vc.classify(_TERMS)
    classes = {c.op: c for c in v.classes}
    assert set(classes) == {
        "add",
        "mul",
        "relu",
        "transpose",
        "getitem",
        "sum",
    }
    assert classes["add"].pointwise is True
    assert classes["add"].arity == 2
    assert classes["relu"].pointwise is True
    assert classes["transpose"].view is True
    assert classes["getitem"].view is True
    assert classes["sum"].reduction is True
    assert classes["transpose"].attribute_carrying is True
    assert classes["getitem"].attribute_carrying is True
    assert classes["add"].attribute_carrying is False
    assert v.pointwise == ("add", "mul")
    assert v.unary_pointwise == ("relu",)
    assert set(v.views) == {"transpose", "getitem"}
    assert len(v.probes) == 5


def test_derived_set_and_vocabulary_views():
    vocab = vc.Vocabulary(
        classes=(
            vc.OpClass("add", 2, True, False, False, False),
            vc.OpClass("relu", 1, True, False, False, False),
            vc.OpClass("tr", 1, None, True, False, True),
        )
    )
    assert vocab.pointwise == ("add",)
    assert vocab.unary_pointwise == ("relu",)
    assert vocab.views == ("tr",)
    assert vc._derived_set(vocab, "view") == frozenset({"tr"})
    assert vc._derived_set(vocab, "binary") == frozenset({"add"})
    assert vc._derived_set(vocab, "unary") == frozenset({"relu"})
    assert vc._derived_set(vocab, "pointwise") == frozenset(
        {"add", "relu"}
    )


def test_validate_reports_agreement_and_corroboration():
    v = vc.classify(_TERMS)
    val = vc.validate(v)
    assert val["corpus_ops"] == sorted(
        {"add", "mul", "relu", "transpose", "getitem", "sum"}
    )
    assert len(val["rows"]) == 7
    rows = {r["table"]: r for r in val["rows"]}
    pipe = rows["pipeline._POINTWISE"]
    assert set(pipe["agree"]) == {"add", "mul"}
    pipe_views = rows["pipeline._VIEW_OPS"]
    assert "transpose" in pipe_views["agree"]
    # getitem is a view the pipeline table missed — corroborated by
    # the morphisms signature table, which does list it.
    assert "getitem" in pipe_views["derived_only"]
    assert "getitem" in pipe_views["corroborated"]
    sig = rows["signature._POINTWISE_OPS"]
    assert "mul" in sig["agree"]
    assert "add" in sig["derived_only"]


def test_print_report_and_dump(tmp_path, capsys):
    v = vc.classify(_TERMS)
    val = vc.validate(v)
    vc._print_report(v, val)
    out = capsys.readouterr().out
    assert "law_vocab" in out
    assert "pointwise (arity 2): add, mul" in out
    assert "corroborated" in out
    path = tmp_path / "vocab.json"
    vc._dump_json(str(path), v, val)
    payload = json.loads(path.read_text())
    assert [c["op"] for c in payload["classes"]] == sorted(
        c.op for c in v.classes
    )
    assert payload["pointwise"] == ["add", "mul"]
    assert "getitem" in payload["views"]
    assert "corpus_ops" in payload["validation"]


def test_derive_vocabulary_over_real_corpus():
    v = vc.derive_vocabulary()
    assert len(v.classes) > 40
    ops = {c.op for c in v.classes}
    assert {"add", "mul", "matmul"} <= ops
    assert "add" in v.pointwise


def test_main_derives_validates_dumps(tmp_path, capsys):
    out = tmp_path / "v.json"
    assert vc.main(["--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert len(payload["classes"]) > 40
    assert "corpus_ops" in payload["validation"]
    text = capsys.readouterr().out
    assert "derived vs hand tables" in text


# ---------------------------------------------------------------------------
#  Need detection — the vocabulary the derive bank LACKS (plan 0024 B)
# ---------------------------------------------------------------------------


def _rank_var(shape: tuple) -> Var:
    return Var("x", TensorType(shape))


def _rank_domain() -> list[dict]:
    """Five bindings whose ranks interleave — no single rank fits."""
    return [
        {"x": _rank_var(sh)}
        for sh in [
            (4,),
            (2, 4),
            (2, 3, 4),
            (2, 2, 3, 4),
            (2, 2, 2, 3, 4),
        ]
    ]


def _rank_of(bound: dict) -> int | None:
    sh = getattr(getattr(bound.get("x"), "typ", None), "shape", None)
    return len(sh) if sh else None


def _pow_rank_hook(bound: dict) -> dict | None:
    r = _rank_of(bound)
    return None if r is None else {"$attr:M": 2**r}


def _mod_rank_hook(bound: dict) -> dict | None:
    r = _rank_of(bound)
    return None if r is None else {"$attr:M": r % 2}


def _mul_rank_hook(bound: dict) -> dict | None:
    r = _rank_of(bound)
    return None if r is None else {"$attr:M": 2 * r}


def _str_hook(bound: dict) -> dict:
    return {"$attr:S": "r" + str(bound["$attr:a"])}


class TestCandidateRegistry:
    def test_candidate_lookup(self):
        assert vc._candidate("pow").arity == 2
        assert vc._candidate("mod").semantics == "a % b  (int % int)"
        with pytest.raises(ValueError):
            vc._candidate("nope")

    def test_candidate_value_is_the_bank_handler(self):
        assert vc._candidate_value("pow")(2, 3) == 8
        assert vc._candidate_value("mod")(7, 3) == 1
        # a decline surfaces as None, never an exception
        assert vc._candidate_value("pow")(2, -1) is None
        assert vc._candidate_value("mod")(7, 0) is None
        assert vc._candidate_value("pow")("x", 2) is None

    def test_helpers_uses_op_and_agreement(self):
        assert vc._uses_op(("pow", ("attr", "a"), 2), "pow") is True
        assert vc._uses_op(("add", ("pow", 1, 2), 3), "pow") is True
        assert vc._uses_op(("add", 1, 2), "pow") is False
        assert vc._uses_op("x", "pow") is False
        samples = [({}, 1), ({}, 2), ({}, 3)]
        assert vc._agreement((1, 2, 9), samples) == 2


class TestDetectNeed:
    def test_detects_pow_gap(self):
        need = vc.detect_need(_pow_rank_hook, _rank_domain())
        assert need is not None
        assert need.op == "pow"
        assert need.agreement_gain > 0
        assert need.rules_blocked == ("_pow_rank_hook",)
        assert need.before < need.after == need.samples

    def test_mod_derive_is_now_expressible(self):
        # ``r % 2`` needed ``mod``; the bank gained it (plan 0024 B),
        # so the derive is expressible and no need is reported — the
        # gap closed, the probe finds nothing.
        dom = _rank_domain()
        assert vc.detect_need(_mod_rank_hook, dom) is None
        spec = dspec_synth.synth_dspec_multi(_mod_rank_hook, dom)
        assert spec is not None
        assert "mod" in repr(spec["M"])

    def test_expressible_derive_needs_nothing(self):
        # ``2 * rank`` fits the shipped bank — no gap to report
        assert vc.detect_need(_mul_rank_hook, _rank_domain()) is None

    def test_thin_domain_refused(self):
        assert (
            vc.detect_need(_pow_rank_hook, _rank_domain()[:2]) is None
        )

    def test_no_candidate_op_helps(self):
        dom = [{"q": "Q", "$attr:a": i} for i in range(1, 7)]
        assert vc.detect_need(_str_hook, dom) is None

    def test_rule_reads_name_and_derive(self):
        rule = Rewrite(
            "synth_pow",
            Op.make("add", "x", "y"),
            Op.make("add", "x", "y"),
            derive=_pow_rank_hook,
        )
        need = vc.detect_need(rule, _rank_domain())
        assert need is not None and need.rules_blocked == ("synth_pow",)

    def test_not_a_rule_or_hook_raises(self):
        with pytest.raises(ValueError):
            vc.detect_need(42, [])

    def test_need_to_data_is_json_safe(self):
        need = vc.detect_need(_pow_rank_hook, _rank_domain())
        data = need.to_data()
        assert json.loads(json.dumps(data)) == data
        assert data["rules_blocked"] == ["_pow_rank_hook"]


class TestProposeAtom:
    def test_proposal_is_declare_shaped_and_not_admitted(self):
        need = vc.detect_need(_pow_rank_hook, _rank_domain())
        prop = vc.propose_atom(need)
        assert prop["op"] == "declare"
        assert prop["admitted"] is False
        atom = prop["params"]["construction"]["params"]
        assert atom["name"] == "pow"
        assert atom["arity"] == 2
        assert atom["kind"] == "derive-bank-op"
        assert "**" in atom["semantics"]
        assert atom["observable"][0] == {"in": [2, 3], "out": 8}
        assert prop["evidence"]["op"] == "pow"
        assert prop["evidence"]["agreement_gain"] == need.agreement_gain

    def test_observable_semantics_are_numeric(self):
        assert vc._observations("pow")[0] == {"in": [2, 3], "out": 8}
        assert vc._observations("mod")[1] == {"in": [3, 2], "out": 1}
        # a decline is carried as None, never dropped
        assert {"in": [5, 0], "out": 1} in vc._observations("pow")
        assert {"in": [5, 0], "out": None} in vc._observations("mod")

    def test_proposal_json_roundtrip(self):
        need = vc.detect_need(_pow_rank_hook, _rank_domain())
        prop = vc.propose_atom(need)
        assert json.loads(json.dumps(prop)) == prop

    def test_unknown_op_raises(self):
        bogus = vc.AtomNeed("nope", 1, ())
        with pytest.raises(ValueError):
            vc.propose_atom(bogus)


class TestScanNeeds:
    def _measure(self, dom):
        def measure(rule, corpus, limit):
            return [(env, "equal") for env in dom]

        return measure

    def test_ranks_by_unblock_then_gain(self):
        ranked = vc._rank_needs(
            [
                vc.AtomNeed("pow", 3, ("a",)),
                vc.AtomNeed("pow", 2, ("b",)),
                vc.AtomNeed("mod", 9, ("c", "d")),
            ]
        )
        # two full unblocks beat one, then gain breaks the tie
        assert [n.op for n in ranked] == ["mod", "pow"]
        assert ranked[0].rules_blocked == ("c", "d")
        assert ranked[1].rules_blocked == ("a", "b")
        assert ranked[1].agreement_gain == 5

    def test_scan_over_rules_with_injected_measure(self):
        dom = _rank_domain()
        need_rule = Rewrite(
            "needs_pow",
            Op.make("add", "x", "y"),
            Op.make("add", "x", "y"),
            derive=_pow_rank_hook,
        )
        expressible = Rewrite(
            "expressible",
            Op.make("add", "x", "y"),
            Op.make("add", "x", "y"),
            derive=_mul_rank_hook,
        )
        spec_rule = Rewrite(
            "has_dspec",
            Op.make("add", "x", "y"),
            Op.make("add", "x", "y"),
            derive={"M": ("attr", "a")},
        )
        no_derive = Rewrite(
            "no_derive",
            Op.make("add", "x", "y"),
            Op.make("add", "x", "y"),
        )
        ranked = vc.scan_needs(
            [need_rule, expressible, spec_rule, no_derive],
            measure=self._measure(dom),
        )
        assert [n.op for n in ranked] == ["pow"]
        assert ranked[0].rules_blocked == ("needs_pow",)

    def test_scan_bounds(self):
        dom = _rank_domain()
        rule = Rewrite(
            "needs_pow",
            Op.make("add", "x", "y"),
            Op.make("add", "x", "y"),
            derive=_pow_rank_hook,
        )
        assert vc.scan_needs([rule], max_rules=0) == []
        assert (
            vc.scan_needs(
                [rule], deadline=0.0, measure=self._measure(dom)
            )
            == []
        )
        assert vc.scan_needs([rule], measure=self._measure(dom)) != []
