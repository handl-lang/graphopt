"""Tests for ``catopt_discovery.enumerate`` — the law-shape enumerator.

The enumerator writes down every small *rewriting form* over the op
grammar (comm/assoc/factor/homomorphism/annihilator families) instead
of replaying hand-listed schemas.  These tests pin the grammar (shape
stratification, operand domains, the rewriting-form list), the honest
filters (tautology / unsynthesizable / missing-meta / domain / depth /
op-rooted LHS), the alpha-key dedup and library-relation annotation,
and the end-to-end flow through ``proposal.evaluate``: the enumerator
re-discovers the ``factor_left`` shape class (``new`` — the pointwise
factor rules are opt-in, not in ``ALL_RULES``), re-classifies shipped
laws as duplicates/inverses, and the numeric oracle rejects the
``exp(add) = add(exp,exp)`` false-shape counterexample.
"""

import itertools
import json

import pytest
from catopt_core.ir import Const, Op, Var, op_repr
from catopt_discovery import enumerate as en
from catopt_discovery import proposal as lp
from catopt_discovery.object_synthesis import term_from_spec

_OPS = ("add", "mul", "sub", "div", "neg", "exp", "square", "sqrt")
_OPS_POW = (*_OPS, "pow")


def _cands(ops=_OPS, **kw):
    return list(en.enumerate_candidates(ops, **kw))


def _by_note(cands, text):
    return [c for c in cands if text in c.note]


def _spec(op, *args):
    return term_from_spec((op, *args))


# ---------------------------------------------------------------------------
#  Op vocabulary / spec helpers
# ---------------------------------------------------------------------------


def test_split_resolves_arity_and_overrides():
    bin_, un_ = en._split(("add", "neg", "pow", "matmul", "square"))
    assert bin_ == ("add", "pow", "matmul")  # pow/matmul override
    assert un_ == ("neg", "square")
    with pytest.raises(ValueError, match="no usable arity"):
        en._split(("definitely_not_an_op",))


def test_depth_metas_used_helpers():
    spec = ("add", ("mul", "A", "B"), 0)
    assert en._depth(spec) == 2
    assert en._depth("A") == 0 and en._depth(1) == 0
    assert en._metas_used(spec) == {"A", "B"}
    assert en._metas_used(("add", "A", "A")) == {"A"}


# ---------------------------------------------------------------------------
#  Raw shape stream
# ---------------------------------------------------------------------------


def test_enumerate_shapes_stratified_order():
    shapes = list(
        en.enumerate_shapes(
            ("add", "neg"), max_depth=1, metas=("A", "B"), literals=(0,)
        )
    )
    # 3 leaves + neg(leaf)x3 + add(leaf,leaf)x9 = 15, leaves first.
    assert len(shapes) == 15
    assert shapes[:3] == ["A", "B", Const(0)]
    assert Op.make("neg", "A") in shapes
    assert Op.make("add", "A", "B") in shapes
    # Depth 1 only: every op child is a leaf.
    for s in shapes:
        if isinstance(s, Op):
            assert all(not isinstance(a, Op) for a in s.args)


def test_enumerate_shapes_deterministic_capped_and_default_ops():
    kw = dict(max_depth=2, metas=("A", "B"), literals=(0,))
    a = list(en.enumerate_shapes(("add", "neg"), **kw))
    b = list(en.enumerate_shapes(("add", "neg"), **kw))
    assert a == b
    # depth-2 adds op-over-depth-1 nodes (more than depth-1's 15).
    assert len(a) > 15
    capped = list(
        en.enumerate_shapes(("add", "neg"), max_shapes=5, **kw)
    )
    assert len(capped) == 5
    # The default op vocabulary works without an explicit ops arg.
    default = list(en.enumerate_shapes(max_depth=0, metas=("A",)))
    assert default == ["A", Const(0), Const(1), Const(2)]


# ---------------------------------------------------------------------------
#  Filters — sound-by-construction vs heuristic
# ---------------------------------------------------------------------------


def test_domain_ok_literal_and_tensor_children():
    assert en._domain_ok("A")
    assert en._domain_ok(("pow", "A", 2))
    # pow's exponent may not be an op node or a metavar — fixed-exp.
    assert not en._domain_ok(("pow", "A", ("neg", "A")))
    assert not en._domain_ok(("pow", "A", "B"))
    # matmul children may not be bare literals.
    assert not en._domain_ok(("matmul", "A", 0))
    assert not en._domain_ok(("matmul", 0, "A"))
    assert en._domain_ok(("matmul", "A", "B"))
    # Nested violations fail the whole spec.
    assert not en._domain_ok(("add", ("pow", "A", "B"), "A"))
    assert not en._domain_ok(("add", ("matmul", "A", 1), "A"))


def test_filter_reason_covers_every_rejection():
    assert en._filter_reason("A", "B", 2) == "lhs-not-op"
    assert (
        en._filter_reason(("add", "A", None), "B", 2) == "missing-meta"
    )
    assert (
        en._filter_reason(("add", "A", "B"), "C", 2)
        == "unsynthesizable"
    )
    assert en._filter_reason(("pow", "A", "B"), "A", 2) == "domain"
    deep = ("add", ("add", ("add", "A", "B"), "C"), "C")
    assert en._filter_reason(deep, "A", 2) == "deep"
    assert (
        en._filter_reason(("add", "A", "B"), ("add", "B", "A"), 2)
        is None
    )


def test_has_hole():
    assert en._has_hole(("add", "A", None))
    assert en._has_hole(None)
    assert not en._has_hole(("add", "A", ("neg", "B")))


# ---------------------------------------------------------------------------
#  The candidate stream — pairing, dedup, annotation
# ---------------------------------------------------------------------------


def test_candidates_instantiated_annotated_and_deduped():
    cands = _cands()
    assert cands, "expected a non-empty stream"
    assert all(c.strategy == "enumerate" for c in cands)
    # Alpha-keys are unique and no tautology survives the filter.
    keys = [lp._key(c.lhs, c.rhs) for c in cands]
    assert len(keys) == len(set(keys))
    assert all(k[0] != k[1] for k in keys)
    # Every candidate is instantiated (no str metavar leaves) and
    # carries its library relation in the note.
    for c in cands[:50]:
        assert "relation=" in c.note
        for leaf in lp._leaves(c.lhs) | lp._leaves(c.rhs):
            assert isinstance(leaf, (Var, Const))
    # comm emits exactly one oriented pair per binary op (the reverse
    # is alpha-equal to the forward pair, so dedup eats it).
    comms = [c for c in cands if c.label.startswith("comm:")]
    assert len(comms) == 4  # add, mul, sub, div


def test_stream_is_lazy_capped_and_interleaved():
    it = en.enumerate_candidates(_OPS)
    first = list(itertools.islice(it, 10))
    assert len(first) == 10  # no need to exhaust the stream
    capped = _cands(max_candidates=24)
    assert len(capped) == 24
    # Round-robin: a small cap still spans several form families.
    forms = {c.label.split(":")[0] for c in capped}
    assert len(forms) >= 4


def test_two_metas_cannot_spell_three_slot_forms():
    cands = _cands(metas=("A", "B"))
    assert cands, "two-slot forms still emit"
    blocked = {
        "assoc",
        "rotate",
        "rekey",
        "factor-left",
        "factor-right",
    }
    assert not {c.label.split(":")[0] for c in cands} & blocked
    # And no emitted term carries a hole.
    for c in cands:
        assert "None" not in op_repr(c.lhs) + op_repr(c.rhs)


def test_domain_pruning_in_the_stream():
    cands = _cands(_OPS_POW)
    for c in cands:
        for spec in (c.lhs, c.rhs):
            _assert_pow_exponents_literal(spec)
    with_mm = _cands(("add", "matmul"), metas=("A", "B"))
    assert with_mm
    for c in with_mm:
        for spec in (c.lhs, c.rhs):
            _assert_matmul_children_tensor(spec)
    # matmul commutativity is emitted — and it is not a library law.
    mm_comm = _by_note(with_mm, "(matmul 'A', 'B') = (matmul 'B', 'A')")
    assert len(mm_comm) == 1
    assert mm_comm[0].note.endswith("relation=new")


def _assert_pow_exponents_literal(term):
    if isinstance(term, Op):
        if term.op == "pow":
            assert isinstance(term.args[1], Const)
        for a in term.args:
            _assert_pow_exponents_literal(a)


def _assert_matmul_children_tensor(term):
    if isinstance(term, Op):
        if term.op == "matmul":
            assert all(isinstance(a, (Op, Var)) for a in term.args)
        for a in term.args:
            _assert_matmul_children_tensor(a)


# ---------------------------------------------------------------------------
#  Rediscovery — the enumerator re-finds the known forms
# ---------------------------------------------------------------------------


def test_factor_left_shape_class_is_new_and_useful():
    target = lp._key(
        _spec("add", ("mul", "A", "B"), ("mul", "A", "C")),
        _spec("mul", "A", ("add", "B", "C")),
    )
    hits = [c for c in _cands() if lp._key(c.lhs, c.rhs) == target]
    assert len(hits) == 1
    (out,) = lp.evaluate(hits)
    # POINTWISE_FACTOR_RULES is opt-in: not ALL_RULES, so "new".
    assert out.relation == "new"
    assert out.num_true is True
    assert out.useful is True
    assert out.cand_cost < out.base_cost


def test_shipped_laws_classify_as_duplicate_or_inverse():
    cands = _cands(_OPS_POW)
    rel = {
        note: c.note.rsplit("relation=", 1)[-1]
        for c in cands
        for note in [c.note.split(";")[0]]
    }
    dup = "(add 'A', 'B') = (add 'B', 'A')"  # comm_add
    inv = "(add (add 'A', 'B'), 'C') = (add 'A', (add 'B', 'C'))"
    assert rel[dup] == "duplicate"
    assert rel[inv] == "inverse"  # assoc_add runs rhs->lhs
    assert rel["(neg (neg 'A')) = 'A'"] == "duplicate"  # double_neg
    assert rel["(add 'A', 0) = 'A'"] == "duplicate"  # id_add
    assert rel["(mul 'A', 1) = 'A'"] == "duplicate"  # id_mul
    # sub_to_add, mul_square/square_expand, pow_to_square.
    assert rel["(sub 'A', 'B') = (add 'A', (neg 'B'))"] == "duplicate"
    assert rel["(mul 'A', 'A') = (square 'A')"] == "inverse"
    assert rel["(pow 'A', 2) = (square 'A')"] == "duplicate"


def test_false_shape_exp_add_rejected_numerically():
    hits = _by_note(
        _cands(),
        "(exp (add 'A', 'B')) = (add (exp 'A'), (exp 'B'))",
    )
    assert len(hits) == 1
    (out,) = lp.evaluate(hits)
    assert out.num_true is False
    assert out.useful is False


def test_true_new_family_neg_add_and_lit_self():
    cands = _cands()
    neg_add = _by_note(
        cands, "(add (neg 'A'), (neg 'B')) = (neg (add 'A', 'B'))"
    )
    double = _by_note(cands, "(add 'A', 'A') = (mul 'A', 2)")
    assert len(neg_add) == 1 and len(double) == 1
    outs = lp.evaluate([neg_add[0], double[0]])
    by = {o.candidate.label.split(":")[0]: o for o in outs}
    assert by["unary-hom"].num_true is True
    assert by["lit-self"].num_true is True


def test_evaluated_pool_feeds_the_yield_table():
    cands = _cands(max_candidates=8)
    outs = lp.evaluate(cands[:2])
    table = lp.yield_table(outs)
    assert table["enumerate"]["proposed"] == 2


# ---------------------------------------------------------------------------
#  Reporting + CLI
# ---------------------------------------------------------------------------


def test_report_helpers():
    cands = _cands(max_candidates=12)
    counts = en._relation_counts(cands)
    assert sum(counts.values()) == 12
    outs = lp.evaluate(cands[:2])
    text = en._top_ranked(outs, 2)
    assert "rel=" in text and "useful=" in text


def test_parse_csv():
    assert en._parse_csv("a,b,,c") == ("a", "b", "c")
    assert en._parse_csv("0,1") == ("0", "1")


def test_main_smoke_and_json(tmp_path, capsys):
    out = tmp_path / "enum.json"
    rc = en.main(
        [
            "--ops",
            "add,mul,neg,sub",
            "--max-candidates",
            "120",
            "--evaluate",
            "6",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "law-shape enumeration" in printed
    assert "enumerate" in printed and "prop" in printed
    payload = json.loads(out.read_text())
    assert payload["emitted"] == 120
    assert payload["n_shapes"] > 0
    assert len(payload["outcomes"]) == 6
    assert payload["yield"]["enumerate"]["proposed"] == 6


def test_main_without_json_still_returns_zero(capsys):
    rc = en.main(
        [
            "--ops",
            "add,neg",
            "--metas",
            "A,B",
            "--max-candidates",
            "20",
            "--evaluate",
            "3",
        ]
    )
    assert rc == 0
    assert "top-ranked" in capsys.readouterr().out
