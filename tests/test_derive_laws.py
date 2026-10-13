"""Tests for the declarative derive DSL (``catopt_core.laws.cond``'s
``dspec`` half).

A law's ``derive`` hook computes the RHS attributes the LHS cannot
bind — the one remaining piece of a rewrite that was opaque code
after ``cond`` landed.  A ``dspec`` is pure data — ``{NAME: expr}``
interpreted against the same ``bound`` environment ``derive`` sees —
and ``Rewrite.__post_init__`` folds it into ``derive`` so every
evaluation site keeps the single ``rule.derive`` convention.

Covered surface:

* every expr op, branch by branch — literals, attr/const/leaf reads,
  shape arithmetic, tuple construction, ``bcast`` — plus the
  strictness contract (any uncomputable expr vetoes the whole spec);
* the malformed-node contract (``ValueError``, not a decline);
* spec canonicalisation — dict / pair-list / ``$attr:``-prefixed keys
  all fold to the same sorted tuple-of-pairs, hashable and equal;
* the ``Rewrite`` fold — ``dspec=``, a spec handed to ``derive=``
  verbatim, and an ``as_derive`` partial are all recast to data;
* migration regression — the shipped spec-carrying rules mint
  exactly what the Python hooks minted, end-to-end in an e-graph,
  and every shipped dspec round-trips through JSON.
"""

from __future__ import annotations

import functools
import json

import pytest
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import ALL_RULES, SCAN_DIAG_LAWS
from catopt_core.laws.cond import (
    as_derive,
    compile_derive,
    derive_from_data,
    derive_to_data,
    eval_cond,
    eval_derive,
)
from catopt_core.laws.scan import (
    AFFD_LIFT,
    AFFD_LIFT_UNIT,
    _derive_affd_unit,
)
from catopt_core.laws.tensor import (
    QKV_FUSE_ASYM,
    RMS_NORM_FOLD,
    SOFTMAX_FOLD,
    _derive_rms_norm,
    _derive_scale_div,
    _derive_scale_mul,
    _derive_scale_one,
    _derive_softmax_dim,
    _derive_split_sizes,
)
from catopt_core.rulecache import ruleset_fingerprint


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _fires(rule, src, want):
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    return eg.find(root) == eg.find(eg.add_term(want))


# ---------------------------------------------------------------------------
#  eval_derive — literals, reads, and the decline contract
# ---------------------------------------------------------------------------


def test_literal_scalars_and_lit_op():
    spec = {
        "A": 1,
        "B": 2.5,
        "C": True,
        "D": None,
        "E": ("lit", "mode"),
    }
    assert eval_derive(spec, {}) == {
        "$attr:A": 1,
        "$attr:B": 2.5,
        "$attr:C": True,
        "$attr:D": None,
        "$attr:E": "mode",
    }
    # an empty spec derives an empty extra-map (not a veto)
    assert eval_derive({}, {"x": _v("x")}) == {}


def test_attr_and_attr0_reads():
    b = {"$attr:RD": (-1,), "$attr:K": 3, "$attr:W": "x"}
    assert eval_derive({"D": ("attr", "K")}, b) == {"$attr:D": 3}
    assert eval_derive({"D": ("attr0", "RD")}, b) == {"$attr:D": -1}
    assert eval_derive({"D": ("attr0", "K")}, b) == {"$attr:D": 3}
    assert eval_derive({"D": ("attr0", "W")}, b) == {"$attr:D": "x"}
    # declines: unbound attrs, and an empty tuple has no [0]
    assert eval_derive({"D": ("attr", "NOPE")}, b) is None
    assert eval_derive({"D": ("attr0", "NOPE")}, b) is None
    assert (
        eval_derive({"D": ("attr0", "E")}, {"$attr:E": ()}) is None
    )


def test_const_leaf_value():
    b = {"S": Const(0.125), "x": _v("x", 2)}
    assert eval_derive({"SC": ("const", "S")}, b) == {"$attr:SC": 0.125}
    # a non-Const binding has no .value; an unbound metavar declines
    assert eval_derive({"SC": ("const", "x")}, b) is None
    assert eval_derive({"SC": ("const", "NOPE")}, b) is None


def test_shape_dim_and_leaf_dim():
    b = {
        "w": _p("w", 6, 4),
        "v": _v("v", 2, 6),
        "unk": "freeleaf",
        "op": Op.make("add", _p("a", 2), _p("b", 2)),
    }
    assert eval_derive({"S": ("shape", "w")}, b) == {"$attr:S": (6, 4)}
    assert eval_derive({"D": ("dim", "w", 0)}, b) == {"$attr:D": 6}
    assert eval_derive({"D": ("dim", "w", -1)}, b) == {"$attr:D": 4}
    # ``dim`` reads the INFERRED shape — an Op member resolves
    assert eval_derive({"D": ("dim", "op", 0)}, b) == {"$attr:D": 2}
    # ``leaf-dim`` reads the DECLARED .typ shape — Op members decline
    assert eval_derive({"D": ("leaf-dim", "w", 0)}, b) == {
        "$attr:D": 6
    }
    assert eval_derive({"D": ("leaf-dim", "op", 0)}, b) is None
    # declines: unshaped, non-int index, out-of-range, None dims
    assert eval_derive({"S": ("shape", "unk")}, b) is None
    assert eval_derive({"D": ("dim", "unk", 0)}, b) is None
    assert eval_derive({"D": ("dim", "w", "x")}, b) is None
    assert eval_derive({"D": ("dim", "w", 5)}, b) is None
    assert eval_derive({"D": ("leaf-dim", "w", "x")}, b) is None
    assert eval_derive({"D": ("leaf-dim", "w", 5)}, b) is None
    half = _p("half", None, 4)
    assert eval_derive({"D": ("leaf-dim", "h", 0)}, {"h": half}) is None
    assert eval_derive({"D": ("dim", "h", 0)}, {"h": half}) is None
    # a scalar ()-shaped leaf has no dims to read
    assert eval_derive({"D": ("leaf-dim", "s", 0)}, {"s": Const(1)}) is None


def test_len_tuple_concat():
    b = {"$attr:SZ": (2, 4), "w": _p("w", 6, 4)}
    spec = {
        "N": ("len", ("attr", "SZ")),
        "T": ("tuple", 1, ("dim", "w", 0), ("attr", "SZ")),
        "C": ("concat", ("shape", "w"), ("tuple", 9)),
    }
    assert eval_derive(spec, b) == {
        "$attr:N": 2,
        "$attr:T": (1, 6, (2, 4)),
        "$attr:C": (6, 4, 9),
    }
    # len/concat on a non-sized value declines
    assert eval_derive({"N": ("len", ("dim", "w", 0))}, b) is None
    assert (
        eval_derive({"C": ("concat", ("dim", "w", 0), ("tuple", 1))}, b)
        is None
    )


def test_arithmetic_ops():
    b = {"$attr:K": 6}
    spec = {
        "A": ("add", ("attr", "K"), 2),
        "S": ("sub", ("attr", "K"), 1),
        "M": ("mul", ("attr", "K"), ("attr", "K")),
        "F": ("fdiv", 1.0, 4),
        "Q": ("floordiv", ("attr", "K"), 4),
        "N": ("neg", ("attr", "K")),
        "R": ("recip", 8.0),
        "G": ("float", 2),
        "I": ("int", 3.7),
    }
    assert eval_derive(spec, b) == {
        "$attr:A": 8,
        "$attr:S": 5,
        "$attr:M": 36,
        "$attr:F": 0.25,
        "$attr:Q": 1,
        "$attr:N": -6,
        "$attr:R": 0.125,
        "$attr:G": 2.0,
        "$attr:I": 3,
    }


def test_arithmetic_declines():
    b = {"S": Const("x")}
    # non-numeric const through float — the sdpa veto verbatim
    assert eval_derive({"SC": ("float", ("const", "S"))}, b) is None
    # zero division declines (the Python hook crashed; the spec vetoes)
    assert eval_derive({"R": ("recip", 0.0)}, b) is None
    assert eval_derive({"F": ("fdiv", 1.0, 0)}, b) is None
    # type errors decline
    assert eval_derive({"A": ("add", ("lit", "x"), 1)}, b) is None
    assert eval_derive({"N": ("neg", ("lit", "x"))}, b) is None
    assert eval_derive({"I": ("int", ("lit", "x"))}, b) is None


def test_bcast_op():
    h, x = _v("h", 4), _v("x", 4)
    wide = _v("w", 2, 4)
    b = {"h": h, "x": x, "w": wide}
    assert eval_derive({"US": ("bcast", "h", "x")}, b) == {
        "$attr:US": (4,)
    }
    assert eval_derive({"US": ("bcast", "w", "x")}, b) == {
        "$attr:US": (2, 4)
    }
    # an unshaped operand is the wildcard — the add's other shape wins
    assert eval_derive(
        {"US": ("bcast", "u", "x")}, {"u": "leaf", "x": x}
    ) == {"$attr:US": (4,)}
    # both unshaped → no concrete shape → veto
    assert eval_derive({"US": ("bcast", "u", "v")}, {}) is None
    # a None dim in the result vetoes
    nx = Var("nx", TensorType((None,)))
    assert eval_derive({"US": ("bcast", "u", "x")}, {"u": nx, "x": x}) is None
    # a provably ill-typed pair vetoes (broadcasting (4,) with (3,))
    z = _v("z", 3)
    assert eval_derive({"US": ("bcast", "z", "x")}, {"z": z, "x": x}) is None


def test_malformed_exprs_raise_valueerror():
    b = {"$attr:RD": (-1,)}
    for bad in (
        {"D": "RD"},  # bare string — probably a typo for ("attr","RD")
        {"D": ()},  # empty node
        {"D": {"x": 1}},  # a dict is not an expr
        {"D": ("nosuchop", "RD")},  # unknown op
    ):
        with pytest.raises(ValueError):
            eval_derive(bad, b)


# ---------------------------------------------------------------------------
#  Spec canonicalisation + the JSON-canonical form
# ---------------------------------------------------------------------------


def test_derive_spec_canonical_forms_agree():
    as_dict = {"SZ": ("tuple", ("leaf-dim", "Q", 0)), "D": 1}
    as_pairs = (("D", 1), ("SZ", ("tuple", ("leaf-dim", "Q", 0))))
    as_lists = json.loads(json.dumps(derive_to_data(as_dict)))
    canon = derive_from_data(as_dict)
    assert canon == derive_from_data(as_pairs)
    assert canon == derive_from_data(as_lists)
    assert canon == derive_from_data(canon)  # idempotent
    # sorted by name, exprs canonicalised to tuples
    assert canon[0][0] == "D" and canon[1][0] == "SZ"
    assert canon[1][1] == ("tuple", ("leaf-dim", "Q", 0))
    # $attr:-prefixed keys lose the prefix
    assert derive_from_data({"$attr:D": 1}) == (("D", 1),)
    # to_data emits a JSON-safe dict; None passes through
    assert derive_to_data(canon) == {
        "D": 1,
        "SZ": ["tuple", ["leaf-dim", "Q", 0]],
    }
    assert derive_to_data(None) is None
    assert derive_from_data(None) is None


def test_derive_spec_malformed_raises():
    for bad in (42, "SZ", {"D": 1, 2: 3}, (("D", 1, 2),), [1, 2]):
        with pytest.raises(ValueError):
            derive_from_data(bad)


# ---------------------------------------------------------------------------
#  compile_derive + as_derive — the derive-shaped views
# ---------------------------------------------------------------------------


def test_compile_derive_composition():
    spec = {"D": ("attr0", "RD")}
    g = compile_derive(spec, None)
    assert g({"$attr:RD": (-1,)}) == {"$attr:D": -1}
    assert g({"$attr:RD": ()}) is None  # spec veto
    # spec + procedural remainder: spec first, then code; keys merge
    calls: list[str] = []

    def extra(bound):
        calls.append("x")
        return {"$attr:E": bound["$attr:RD"][0] + 1}

    g2 = compile_derive(spec, extra)
    assert g2({"$attr:RD": (-1,)}) == {"$attr:D": -1, "$attr:E": 0}
    assert calls == ["x"]
    calls.clear()
    assert g2({"$attr:RD": ()}) is None  # spec vetoes before code
    assert calls == []
    # a vetoing remainder declines the composite
    g3 = compile_derive(spec, lambda b: None)
    assert g3({"$attr:RD": (-1,)}) is None


def test_as_derive_is_a_derive_shaped_partial():
    f = as_derive({"D": ("attr0", "RD")})
    assert isinstance(f, functools.partial)
    assert f.func is eval_derive
    assert f({"$attr:RD": (-1,)}) == {"$attr:D": -1}


# ---------------------------------------------------------------------------
#  The Rewrite fold — every spelling lands on dspec
# ---------------------------------------------------------------------------


def test_rewrite_post_init_folds_and_recasts():
    lhs = Op.make("add", "a", "b")
    # dspec= folds into a callable derive and canonicalises
    r = Rewrite(
        name="t_spec",
        lhs=lhs,
        rhs="a",
        dspec={"D": ("attr0", "RD")},
    )
    assert r.dspec == (("D", ("attr0", "RD")),)
    assert r.derive({"$attr:RD": (-1,)}) == {"$attr:D": -1}
    hash(r)  # canonical tuple pairs keep the frozen rule hashable
    # a spec handed to derive= is recast to dspec
    r2 = Rewrite(
        name="t_recast",
        lhs=lhs,
        rhs="a",
        derive={"D": ("attr0", "RD")},
    )
    assert r2.dspec == r.dspec
    assert r2.derive({"$attr:RD": (0,)}) == {"$attr:D": 0}
    # an as_derive partial carries its spec back into dspec
    r3 = Rewrite(
        name="t_partial",
        lhs=lhs,
        rhs="a",
        derive=as_derive({"D": ("attr0", "RD")}),
    )
    assert r3.dspec == r.dspec
    assert r3.derive({"$attr:RD": (0,)}) == {"$attr:D": 0}
    # spec + code conjoin through the same fold
    r4 = Rewrite(
        name="t_both",
        lhs=lhs,
        rhs="a",
        dspec={"D": ("attr0", "RD")},
        derive=lambda b: {"$attr:E": 1},
    )
    assert r4.derive({"$attr:RD": (0,)}) == {
        "$attr:D": 0,
        "$attr:E": 1,
    }
    # a spec twice is an error, not a silent choice
    with pytest.raises(ValueError):
        Rewrite(
            name="t_dup",
            lhs=lhs,
            rhs="a",
            dspec={"D": 1},
            derive={"E": 2},
        )
    # dspec-free rules are untouched
    r5 = Rewrite(name="t_none", lhs=lhs, rhs="a")
    assert r5.dspec is None and r5.derive is None
    # a partial that is NOT an eval_derive spec stays a callable hook
    other = functools.partial(lambda b, k: b.get(k), k="x")
    r6 = Rewrite(name="t_other", lhs=lhs, rhs="a", derive=other)
    assert r6.dspec is None and r6.derive is other
    # eval_cond partials (a cond view) are likewise NOT recast
    from catopt_core.laws.cond import eval_cond

    r7 = Rewrite(
        name="t_cond_partial",
        lhs=lhs,
        rhs="a",
        derive=functools.partial(eval_cond, ("scalar", "a")),
    )
    assert r7.dspec is None


def test_fingerprint_covers_dspec_data():
    """A dspec edit must invalidate the synthesis cache — the folded
    hook shares one closure signature across every spec-carrying rule."""
    lhs = Op.make("add", "a", "b")
    r1 = Rewrite(name="r", lhs=lhs, rhs="a", dspec={"D": ("attr0", "RD")})
    r2 = Rewrite(name="r", lhs=lhs, rhs="a", dspec={"D": ("attr0", "RE")})
    r3 = Rewrite(name="r", lhs=lhs, rhs="a")
    fp1, fp2, fp3 = (
        ruleset_fingerprint([r]) for r in (r1, r2, r3)
    )
    assert fp1 != fp2  # different spec → different fingerprint
    assert fp1 != fp3  # derived vs underived
    # same data re-spelled as lists fingerprints identically
    r1l = Rewrite(
        name="r", lhs=lhs, rhs="a", dspec=[("D", ["attr0", "RD"])]
    )
    assert ruleset_fingerprint([r1l]) == fp1


# ---------------------------------------------------------------------------
#  Migration regression — shipped specs behave like the Python hooks
# ---------------------------------------------------------------------------


def test_every_shipped_dspec_roundtrips_through_json():
    """All 22 spec-carrying rules' dspecs survive the store wire format."""
    seen = 0
    for rule in [*ALL_RULES, *SCAN_DIAG_LAWS]:
        if rule.dspec is None:
            continue
        seen += 1
        blob = json.dumps(derive_to_data(rule.dspec))
        assert derive_from_data(json.loads(blob)) == rule.dspec, rule.name
        # a rule rebuilt from data carries the same spec
        rebuilt = Rewrite(
            name=rule.name,
            lhs=rule.lhs,
            rhs=rule.rhs,
            dspec=json.loads(blob),
        )
        assert rebuilt.dspec == rule.dspec
    assert seen == 22


def test_migrated_alias_partials_are_the_same_data():
    """The ``_derive_*`` test aliases cannot drift — they ARE the spec."""
    assert _derive_softmax_dim.func is eval_derive
    assert _derive_softmax_dim.args == (SOFTMAX_FOLD.dspec,)
    assert _derive_split_sizes.args == (QKV_FUSE_ASYM.dspec,)
    assert _derive_affd_unit.args == (AFFD_LIFT_UNIT.dspec,)
    assert _derive_rms_norm.args == (RMS_NORM_FOLD.dspec,)
    # verdict parity on accept and decline bindings
    assert _derive_scale_mul({"S": Const(0.125)}) == {"$attr:SC": 0.125}
    assert _derive_scale_div({"S": Const(0.125)}) == {"$attr:SC": 8.0}
    assert _derive_scale_div({"S": Const("x")}) is None
    assert _derive_scale_one({}) == {"$attr:SC": 1.0}
    assert _derive_softmax_dim({"$attr:RD": (-1,)}) == {"$attr:SD": -1}
    assert _derive_softmax_dim({"$attr:RD": 1}) == {"$attr:SD": 1}
    assert _derive_split_sizes(
        {"Q": _p("Q", 6, 4), "K": _p("K", 2, 4), "V": _p("V", 2, 4)}
    ) == {"$attr:SZ": (6, 2, 2)}
    # the rms derive: tail-block spec + unwrapped Const eps
    rms_bound = {
        "u": _v("u", 4, 8),
        "EPS": Const(1e-5),
        "$attr:MD": (-1,),
    }
    assert _derive_rms_norm(rms_bound) == {
        "$attr:ND": (8,),
        "$attr:EP": 1e-5,
    }
    # a non-trailing reduce or a non-Const eps vetoes the spec
    assert (
        _derive_rms_norm({**rms_bound, "$attr:MD": (0,)}) is None
    )
    assert (
        _derive_rms_norm({**rms_bound, "EPS": _v("e")}) is None
    )


def test_migrated_rules_carry_dspec_and_folded_derive():
    by_name = {r.name: r for r in [*ALL_RULES, *SCAN_DIAG_LAWS]}
    for name in (
        "softmax_fold",
        "qkv_fuse_asym",
        *[f"sdpa_fold_add{s}{w}" for s in ("mul", "div", "") for w in ("", "_drop")],
        *[f"sdpa_fold_masked_fill{s}{w}" for s in ("mul", "div", "") for w in ("", "_drop")],
        "rms_norm_fold",
        "rms_norm_fold_nogain",
        "affd_lift_unit",
        "affd_lift_unit_post",
        "affd_lift_unit_step",
        "affd_lift_unit_step_post",
    ):
        r = by_name[name]
        assert r.dspec is not None, name
        assert callable(r.derive), name


def test_affd_unit_state_cond_verdicts():
    """The migrated unit-state guard keeps its verdicts as data."""
    h = _p("h0", 4)
    x = _v("x", 4)
    for t, want in (
        (h, True),  # leaf Param — the h0 state
        (x, True),  # free Var leaf
        (Const(1.0), False),  # scalar offset is not state
        (Op.make("add", h, x), True),
        (Op.make("sub", h, x), True),
        (Op.make("applyd", "f", h), True),
        (Op.make("mul", h, x), False),  # per-step term is not state
        (Op.make("select", h), False),
    ):
        assert AFFD_LIFT_UNIT.check({"h": t}) is want, t


def test_affd_state_cond_verdicts():
    """``_affd_state_like`` migrated identically: state ops or any
    leaf (Const offsets ARE admitted here — unlike the unit guard)."""
    h = _p("h0", 4)
    for t, want in (
        (h, True),
        (Const(1.0), True),  # any leaf is a state candidate
        (Op.make("add", h, h), True),
        (Op.make("sub", h, h), True),
        (Op.make("apply", "f", h), True),
        (Op.make("applyd", "f", h), True),
        (Op.make("mul", h, h), False),
        (Op.make("select", h), False),
    ):
        assert AFFD_LIFT.check({"h": t}) is want, t


# ---------------------------------------------------------------------------
#  End-to-end — spec-derived rules fire identically after a JSON hop
# ---------------------------------------------------------------------------


def test_softmax_fold_mints_derived_dim_end_to_end():
    u = _v("u", 4, 8)
    src = Op.make(
        "div",
        Op.make("exp", u),
        Op.make("sum", Op.make("exp", u), dim=(-1,), keepdim=True),
    )
    assert _fires(SOFTMAX_FOLD, src, Op.make("softmax", u, dim=-1))


def test_qkv_fuse_asym_mints_split_sizes_end_to_end():
    """The uneven (6,2,2) QKV match produces the split member — now
    through the spec, identically to the Python hook.  A rebuilt
    rule mints the same sizes."""
    from catopt_core.laws.serialize import law_from_data, law_to_data
    from catopt_core.laws.tensor import _head_v

    x = _v("x", 2, 4)
    Q, K, V = _p("Q", 6, 4), _p("K", 2, 4), _p("V", 2, 4)
    src = Op.make(
        "sdpa",
        _head_v(Op.make("linear", x, Q), (2, 2, 3, 2)),
        _head_v(Op.make("linear", x, K), (2, 2, 1, 2)),
        _head_v(Op.make("linear", x, V), (2, 2, 1, 2)),
        scale=0.5,
        enable_gqa=False,
    )
    rebuilt = law_from_data(
        json.loads(json.dumps(law_to_data(QKV_FUSE_ASYM)))
    )
    for rule in (QKV_FUSE_ASYM, rebuilt):
        eg = EGraph()
        root = eg.add_term(src)
        assert eg.apply_rule(rule, root)
        # the fused member exists somewhere: a split with the
        # derived (6,2,2) sizes on the concat-weight GEMM
        minted = any(
            n.op == "split" and dict(n.attrs).get("sizes") == (6, 2, 2)
            for ec in eg._classes.values()
            for n in ec.nodes
        )
        assert minted, rule.name


def test_affd_lift_unit_fires_and_mints_bcast_shape():
    h, x = _p("h0", 4), _v("x", 4)
    src = Op.make("add", h, x)
    want = Op.make(
        "applyd",
        Op.make(
            "aff_diag", Op.make("expand", Const(1.0), shape=(4,)), x
        ),
        h,
    )
    assert _fires(AFFD_LIFT_UNIT, src, want)


# ---------------------------------------------------------------------------
#  Grammar extensions — nth/take/sum reads, guard/if cond-gated exprs,
#  and the attr-cmp/attr-ints predicates they need
# ---------------------------------------------------------------------------


def test_nth_reads_and_declines():
    b = {"$attr:SZ": (8, 4, 2)}
    nth = ("nth", ("attr", "SZ"), ("attr", "I"))
    b1 = {**b, "$attr:I": 1}
    assert eval_derive({"X": nth}, b1) == {"$attr:X": 4}
    # negative indexing has Python reach
    assert eval_derive({"X": nth}, {**b, "$attr:I": -1}) == {
        "$attr:X": 2
    }
    # bool index — the isinstance(i, int) contract admits it
    assert eval_derive({"X": nth}, {**b, "$attr:I": True}) == {
        "$attr:X": 4
    }
    for bad in (
        {**b, "$attr:I": 3},  # out of range
        {**b, "$attr:I": -4},
        {**b, "$attr:I": "x"},  # non-int index
        {"$attr:SZ": 8, "$attr:I": 0},  # non-sequence
        {"$attr:SZ": "abc", "$attr:I": 0},  # str is not a seq here
        {"$attr:I": 0},  # SZ unbound
    ):
        assert eval_derive({"X": nth}, bad) is None, bad


def test_take_and_sum():
    b = {"$attr:SZ": (4, 5), "$attr:I": 1}
    assert eval_derive(
        {"P": ("take", ("attr", "SZ"), ("attr", "I"))}, b
    ) == {"$attr:P": (4,)}
    # oversized counts clamp like Python slicing
    assert eval_derive(
        {"P": ("take", ("attr", "SZ"), 9)}, b
    ) == {"$attr:P": (4, 5)}
    assert eval_derive(
        {"O": ("sum", ("take", ("attr", "SZ"), ("attr", "I")))}, b
    ) == {"$attr:O": 4}
    assert eval_derive({"O": ("sum", ("attr", "SZ"))}, b) == {
        "$attr:O": 9
    }
    # declines: non-sequence take/sum, non-int count, non-numeric elem
    assert eval_derive(
        {"P": ("take", ("attr", "SZ"), ("attr", "I"))},
        {"$attr:SZ": 4, "$attr:I": 1},
    ) is None
    assert eval_derive(
        {"P": ("take", ("attr", "SZ"), ("attr", "I"))},
        {"$attr:SZ": (4,), "$attr:I": "x"},
    ) is None
    assert eval_derive({"O": ("sum", ("attr", "SZ"))}, {}) is None
    assert eval_derive(
        {"O": ("sum", ("attr", "SZ"))}, {"$attr:SZ": 4}
    ) is None
    assert eval_derive({"O": ("sum", ("attr", "SZ"))}, {
        "$attr:SZ": (4, "x")
    }) is None


def test_guard_vetoes_and_if_branches():
    b = {"$attr:I": 1, "$attr:SZ": (4, 5)}
    spec = {
        "X": (
            "guard",
            ("attr-cmp", "I", "<", 2),
            ("nth", ("attr", "SZ"), ("attr", "I")),
        )
    }
    assert eval_derive(spec, b) == {"$attr:X": 5}
    assert eval_derive(spec, {**b, "$attr:I": 9}) is None
    # a malformed cond inside guard/if is a bug, not a decline
    with pytest.raises(ValueError):
        eval_derive({"X": ("guard", ("nosuch", "I"), 1)}, b)
    # if picks one arm — a declining untaken arm cannot veto
    spec_if = {
        "X": (
            "if",
            ("attr-eq-attr", "I", "SZ"),
            ("attr", "I"),
            ("attr", "MISSING"),  # would decline — not taken
        )
    }
    assert eval_derive(spec_if, b) is None  # cond False → else arm
    assert eval_derive(
        spec_if, {"$attr:I": 3, "$attr:SZ": 3}
    ) == {"$attr:X": 3}


def test_attr_cmp_and_attr_ints_predicates():
    # attr-cmp: names, literals, and inline exprs as operands
    b = {"$attr:T": 9, "$attr:SZ": (4, 5)}
    cond = ("attr-cmp", "T", "==", ("sum", ("attr", "SZ")))
    assert eval_cond(cond, b) is True
    assert (
        eval_cond(("attr-cmp", 0, "<=", "I"), {"$attr:I": 0}) is True
    )
    # a missing or uncomputable side declines
    assert eval_cond(("attr-cmp", "T", "<", "NOPE"), b) is False
    assert eval_cond(("attr-cmp", "NOPE", "<", "T"), b) is False
    assert (
        eval_cond(("attr-cmp", "T", "<", ("attr", "NOPE")), b) is False
    )
    # an unorderable pair declines, it does not raise
    b2 = {**b, "$attr:X": "abc"}
    assert eval_cond(("attr-cmp", "T", "<", "X"), b2) is False
    # attr-ints: tuple/list of ints, bool admitted, everything else no
    for v, want in (
        ((4, 5), True),
        ([4, 5], True),
        ((True, 5), True),  # isinstance(True, int) — the hook contract
        ((), True),
        ((4, "x"), False),
        ("45", False),
        (4, False),
    ):
        assert eval_cond(("attr-ints", "A"), {"$attr:A": v}) is want
    assert eval_cond(("attr-ints", "NOPE"), b) is False


# ---------------------------------------------------------------------------
#  Carrier migration regression — the migrated derives are pure data
# ---------------------------------------------------------------------------


def test_migrated_carrier_rules_carry_dspec():
    """Every migrated carrier derive is now declarative data.

    ``missing_hooks`` must not report ``"derive"`` — the folded hook
    is the spec-only closure — and the spec must survive the
    law-record JSON hop minting identical attrs.
    """
    from catopt_carriers import decode_geom, decode_laws, om, trace
    from catopt_core.laws.serialize import (
        law_from_data,
        law_to_data,
        missing_hooks,
    )

    migrated = [
        trace.TR_VANISH_SPLIT,
        trace.TR_VANISH_MERGE,
        trace.TR_SUPERPOSE_REV,
        trace.TR_SLIDE,
        trace.TR_SLIDE_REV,
        trace.TR_EXPAND,
        om.SPLIT_TRANSPOSE,
        om.DECAY_BLOCK_DIAG,
        om.DECAY_BLOCK_CROSS,
        decode_laws.INDEX_SELECT_DEDUP,
        decode_geom.CAT_NARROW_MERGE,
    ]
    for rule in migrated:
        assert rule.dspec is not None, rule.name
        assert "derive" not in missing_hooks(rule), rule.name
        blob = json.dumps(derive_to_data(rule.dspec))
        assert (
            derive_from_data(json.loads(blob)) == rule.dspec
        ), rule.name
        rebuilt = law_from_data(
            json.loads(json.dumps(law_to_data(rule)))
        )
        assert rebuilt.dspec == rule.dspec, rule.name


def test_migrated_carrier_derives_mint_identically():
    """The specs produce the same attr maps the Python hooks did."""
    from catopt_carriers import decode_geom, decode_laws, om, trace

    v = lambda n, *s: Var(n, TensorType(tuple(s)))  # noqa: E731
    # tr_vanish_split: usize (u,v) → two int dims
    assert trace.TR_VANISH_SPLIT.derive(
        {"f": v("f", 8, 8), "$attr:UV": (3, 2)}
    ) == {"$attr:DU": 3, "$attr:DV": 2}
    assert trace.TR_VANISH_SPLIT.derive({"$attr:UV": (3, 2, 1)}) is None
    # tr_slide / tr_slide_rev: the widened block axis
    g = v("g", 5, 7)
    assert trace.TR_SLIDE.derive({"g": g, "$attr:DU": 3}) == {
        "$attr:DY": 2
    }
    assert trace.TR_SLIDE.derive({"g": v("g2", 3, 7), "$attr:DU": 3}) is None
    assert trace.TR_SLIDE.derive({"g": v("g3", None, 7), "$attr:DU": 3}) is None
    assert trace.TR_SLIDE_REV.derive(
        {"g": v("g4", 7, 5), "$attr:DU": 3}
    ) == {"$attr:DX": 2}
    # tr_expand: (du, f_i − du) per axis
    assert trace.TR_EXPAND.derive({"f": v("f", 6, 8), "$attr:DU": 4}) == {
        "$attr:RS": (4, 2),
        "$attr:CS": (4, 4),
    }
    assert trace.TR_EXPAND.derive(
        {"f": v("f", 6, 8), "$attr:DU": 0}
    ) is None
    # tr_vanish_merge / tr_superpose_rev: the fused product size
    b = {"$attr:DU": 4, "$attr:DV": 2}
    assert trace.TR_VANISH_MERGE.derive(b) == {"$attr:UV": (4, 2)}
    assert trace.TR_SUPERPOSE_REV.derive(b) == {"$attr:UV": (4, 2)}
    # split_transpose: the pre-transpose axis pick
    dims = {"$attr:SD": -2, "$attr:D0": -2, "$attr:D1": -1}
    assert om.SPLIT_TRANSPOSE.derive(dims) == {"$attr:DP": -1}
    assert om.SPLIT_TRANSPOSE.derive(
        {**dims, "$attr:SD": -1}
    ) == {"$attr:DP": -2}
    assert om.SPLIT_TRANSPOSE.derive({**dims, "$attr:SD": 0}) == {
        "$attr:DP": 0
    }
    assert (
        om.SPLIT_TRANSPOSE.derive({**dims, "$attr:SD": "x"}) is None
    )
    # decay_block_diag: TC = SZ[I] on the (i,i) block
    ok = {"$attr:SZ": (8, 8), "$attr:I": 1, "$attr:T": 16}
    assert om.DECAY_BLOCK_DIAG.derive(ok) == {"$attr:TC": 8}
    assert om.DECAY_BLOCK_DIAG.derive({**ok, "$attr:I": 5}) is None
    assert om.DECAY_BLOCK_DIAG.derive({**ok, "$attr:SZ": (8, "x")}) is None
    assert om.DECAY_BLOCK_DIAG.derive({**ok, "$attr:T": 12}) is None
    # decay_block_cross: extents + the strict-lower offset
    ok2 = {"$attr:SZ": (4, 5), "$attr:T": 9, "$attr:I": 1, "$attr:J": 0}
    assert om.DECAY_BLOCK_CROSS.derive(ok2) == {
        "$attr:TI": 5,
        "$attr:TJ": 4,
        "$attr:OD": 4,
    }
    assert om.DECAY_BLOCK_CROSS.derive({**ok2, "$attr:J": 1}) is None
    assert om.DECAY_BLOCK_CROSS.derive({**ok2, "$attr:T": 8}) is None
    # index_select_dedup: uniques + position map
    assert decode_laws.INDEX_SELECT_DEDUP.derive(
        {"$attr:I": (7, 3, 7, 0, 3)}
    ) == {"$attr:U": (7, 3, 0), "$attr:VI": (0, 1, 0, 2, 1)}
    # cat_narrow_merge: the merged extent
    assert decode_geom.CAT_NARROW_MERGE.derive(
        {"$attr:L1": 2, "$attr:L2": 4}
    ) == {"$attr:LT": 6}
    assert (
        decode_geom.CAT_NARROW_MERGE.derive({"$attr:L1": "2"}) is None
    )
