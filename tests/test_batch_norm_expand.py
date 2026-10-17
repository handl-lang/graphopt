# ruff: noqa: RUF002 -- the docstrings spell the BN affine math
"""The batch_norm expansion family + the ``$leaf:`` compose seam.

ADR 0012 landed the ``$leaf:`` channel but named two limits: no law
LHS named ``batch_norm`` (content), and a ``$leaf:`` rule could not
auto-compose in ``meta.synthesize_rules`` (the ``@i:`` placeholder
machinery was attr-only).  Both are closed:

* ``BATCH_NORM_EXPAND*`` mint the eval-mode spelled work
  ``y = γ·((x−μ)·rsqrt(σ²+ε)) + β`` as *data* — the expose half of the
  conv+BN weight-space fold (``weightspace-fold.md`` gate 5).  The
  channel-broadcast reshape target and the ``eps`` leaf are derived:
  ``CH`` from ``x``'s rank and ``rm``'s extent, ``$leaf:EPS`` minting
  ``Const(eps)`` through the scalar side of the ``$leaf:`` channel.
* The ``@i:`` placeholder machinery now covers *term* metavars, so a
  ``$leaf:`` parent composes — measured here by composing
  ``batch_norm_expand`` (a real ``$leaf:`` rule) in the synthesizer.
"""

import torch
from catopt_core import meta
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import Const, Op, Param, TensorType, Var, op_repr
from catopt_core.laws import (
    BATCH_NORM_EXPAND,
    BATCH_NORM_EXPAND_NOBIAS,
    BATCH_NORM_EXPAND_PLAIN,
    BATCH_NORM_RULES,
    DEFAULT,
    _check_bn_eval,
    _derive_bn,
)
from catopt_core.laws.cond import derive_from_data, derive_to_data

# Registers the concrete-eval backend + op bindings for _eval_term.
from catopt_torch.torch_bridge import export_to_ir  # noqa: F401


def _T(*shape):
    return TensorType(tuple(shape) if shape else (4, 4))


def _bn(x, *stats, training=False, eps=1e-5):
    return Op.make(
        "batch_norm",
        x,
        *stats,
        training=training,
        momentum=0.1,
        eps=eps,
        cudnn_enabled=False,
    )


def _leaves(x_shape, c):
    return (
        Var("x", _T(*x_shape)),
        Param("w", _T(c)),
        Param("b", _T(c)),
        Param("rm", _T(c)),
        Param("rv", _T(c)),
    )


def _env(*leaves, seed=0):
    torch.manual_seed(seed)
    env = {}
    for leaf in leaves:
        v = torch.randn(*leaf.typ.shape, dtype=torch.float64)
        if leaf.name == "rv":
            v = v.abs() + 0.5  # a variance is positive
        env[leaf] = v
    return env


# ---------------------------------------------------------------------------
#  firing + guards
# ---------------------------------------------------------------------------


def test_batch_norm_expand_fires_affine():
    x, w, b, rm, rv = _leaves((1, 8, 8, 8), 8)
    out = meta.apply_rewrite_at(
        BATCH_NORM_EXPAND, _bn(x, w, b, rm, rv), ()
    )
    assert out is not None
    assert op_repr(out) == (
        "(add (mul (reshape w, shape=(1, 8, 1, 1)), "
        "(mul (sub x, (reshape rm, shape=(1, 8, 1, 1))), "
        "(rsqrt (add (reshape rv, shape=(1, 8, 1, 1)), 1e-05)))), "
        "(reshape b, shape=(1, 8, 1, 1)))"
    )


def test_batch_norm_expand_nobias_and_plain():
    x, w, _b, rm, rv = _leaves((2, 4), 4)
    out = meta.apply_rewrite_at(
        BATCH_NORM_EXPAND_NOBIAS, _bn(x, w, rm, rv), ()
    )
    assert out is not None
    assert "reshape w, shape=(1, 4)" in op_repr(out)
    assert "reshape b" not in op_repr(out)
    out2 = meta.apply_rewrite_at(
        BATCH_NORM_EXPAND_PLAIN, _bn(x, rm, rv), ()
    )
    assert out2 is not None
    assert op_repr(out2).startswith(
        "(mul (sub x, (reshape rm, shape=(1, 4)))"
    )


def test_batch_norm_expand_derives_target_per_rank():
    """One law covers ranks 2-6: the reshape target tracks x's rank."""
    for xshape, want in (
        ((2, 4), "(1, 4)"),
        ((2, 4, 3), "(1, 4, 1)"),
        ((1, 8, 8, 8), "(1, 8, 1, 1)"),
        ((1, 8, 2, 2, 2), "(1, 8, 1, 1, 1)"),
    ):
        x = Var("x", _T(*xshape))
        rm = Param("rm", _T(xshape[1]))
        rv = Param("rv", _T(xshape[1]))
        out = meta.apply_rewrite_at(
            BATCH_NORM_EXPAND_PLAIN, _bn(x, rm, rv), ()
        )
        assert out is not None, xshape
        assert f"shape={want}" in op_repr(out), (xshape, op_repr(out))


def test_batch_norm_expand_guard_vetoes():
    x, w, b, rm, rv = _leaves((1, 8, 8, 8), 8)
    # training mode computes batch stats — the eval-mode identity
    # does not apply
    assert (
        meta.apply_rewrite_at(
            BATCH_NORM_EXPAND, _bn(x, w, b, rm, rv, training=True), ()
        )
        is None
    )
    # stats extent must BE x's channel extent
    rm4 = Param("rm4", _T(4))
    assert (
        meta.apply_rewrite_at(
            BATCH_NORM_EXPAND, _bn(x, w, b, rm4, rv), ()
        )
        is None
    )
    # rank-1 x has no channel axis; rank >6 exceeds the ones-bank
    xr = Var("xr", _T(8))
    assert (
        meta.apply_rewrite_at(
            BATCH_NORM_EXPAND_PLAIN, _bn(xr, rm, rv), ()
        )
        is None
    )
    x7 = Var("x7", _T(1, 8, 1, 1, 1, 1, 1))
    assert (
        meta.apply_rewrite_at(
            BATCH_NORM_EXPAND_PLAIN, _bn(x7, rm, rv), ()
        )
        is None
    )
    # unshaped x cannot mint a concrete broadcast target
    xu = Var("xu", TensorType((None, None, None, None)))
    assert (
        meta.apply_rewrite_at(
            BATCH_NORM_EXPAND_PLAIN, _bn(xu, rm, rv), ()
        )
        is None
    )


def test_batch_norm_expand_in_egraph_keeps_fused_cheapest():
    """The expansion adds a member; extraction still prefers fused
    (priced 4 < the ~7-pass spelled subgraph)."""
    from catopt_core.cost.basic import flops_cost

    x, w, b, rm, rv = _leaves((1, 8, 8, 8), 8)
    term = _bn(x, w, b, rm, rv)
    eg = EGraph()
    root = eg.add_term(term)
    assert eg.apply_rule(BATCH_NORM_EXPAND, root)
    assert eg.extract_best(root, flops_cost) == term


def test_batch_norm_expand_numerically_exact():
    """fp64 agreement with the fused kernel through the torch bridge."""
    x, w, b, rm, rv = _leaves((2, 6, 4, 4), 6)
    term = _bn(x, w, b, rm, rv)
    env = _env(x, w, b, rm, rv)
    for rule in (BATCH_NORM_EXPAND,):
        out = meta.apply_rewrite_at(rule, term, ())
        assert out is not None
        assert meta._eval_allclose(
            meta._eval_term(term, env), meta._eval_term(out, env), 1e-10
        )
    # nobias + plain twins, same env trick (w/b unused where absent)
    t_nb = _bn(x, w, rm, rv)
    out_nb = meta.apply_rewrite_at(BATCH_NORM_EXPAND_NOBIAS, t_nb, ())
    assert meta._eval_allclose(
        meta._eval_term(t_nb, env), meta._eval_term(out_nb, env), 1e-10
    )
    t_pl = _bn(x, rm, rv)
    out_pl = meta.apply_rewrite_at(BATCH_NORM_EXPAND_PLAIN, t_pl, ())
    assert meta._eval_allclose(
        meta._eval_term(t_pl, env), meta._eval_term(out_pl, env), 1e-10
    )


def test_batch_norm_laws_are_data_and_opt_in():
    """The whole law serializes (cond+dspec are pure data) and stays
    out of DEFAULT pending the admission floor."""
    names = {r.name for r in DEFAULT.rules}
    assert not (
        {
            "batch_norm_expand",
            "batch_norm_expand_nobias",
            "batch_norm_expand_plain",
        }
        & names
    )
    for r in BATCH_NORM_RULES:
        assert derive_from_data(derive_to_data(r.dspec)) == r.dspec
        assert r.check is not None and r.derive is not None
    # the test-facing aliases read the same data the rules carry
    x, w, b, rm, rv = _leaves((1, 8, 8, 8), 8)
    bound = meta.match_pattern(
        BATCH_NORM_EXPAND.lhs, _bn(x, w, b, rm, rv), {}
    )
    assert bound is not None
    assert _check_bn_eval(bound)
    got = _derive_bn(bound)
    assert got["$attr:CH"] == (1, 8, 1, 1)
    assert got["EPS"] == Const(1e-5)


# ---------------------------------------------------------------------------
#  the "$leaf:" auto-compose seam (ADR 0012's named limit, closed)
# ---------------------------------------------------------------------------


def _leaf_mark_rule():
    """A ``$leaf:`` rule: mints a scalar Const leaf the LHS never binds."""
    return Rewrite(
        "r_leaf",
        Op.make("mark", "a"),
        Op.make("marked", "a", "eps"),
        dspec={"$leaf:eps": ("lit", 2.0)},
    )


def test_synthesizable_admits_leaf_rules():
    """The named refusal is gone: an unbound RHS *term* metavar is fine
    when the rule's derive produces it — the attr clause's twin."""
    r = _leaf_mark_rule()
    assert meta._synthesizable(r)
    # without a derive the same shape still refuses
    bare = Rewrite(
        "r_bare",
        Op.make("mark", "a"),
        Op.make("marked", "a", "eps"),
    )
    assert not meta._synthesizable(bare)


def test_fire_guarded_leaf_placeholder():
    """_fire_guarded carries a term-level derive output as an "@i:"
    leaf metavar and records the concrete leaf in conc."""
    v = Var("v", _T())
    out = meta._fire_guarded(
        _leaf_mark_rule(), Op.make("mark", v), (), "@1:"
    )
    assert out is not None
    _subst, rewritten, conc = out
    assert rewritten == Op.make("marked", v, "@1:eps")
    assert conc["@1:eps"] == Const(2.0)


def test_concretize_attrs_resolves_leaf_placeholders():
    """A placeholder leaf inside a bound term resolves via conc —
    r2's guards see the concrete minted leaf, never the string."""
    conc = {"@1:eps": Const(2.0)}
    assert meta._concretize_attrs("@1:eps", conc) == Const(2.0)
    t = Op.make("marked", "@1:eps", tag="@1:Q")
    got = meta._concretize_attrs(t, {**conc, "@1:Q": 7})
    assert got == Op.make("marked", Const(2.0), tag=7)
    # a non-placeholder string passes through untouched
    assert meta._concretize_attrs("plain", conc) == "plain"


def test_rhs_derive_placeholders_leaf():
    rhs = Op.make("sink", "a", "@1:eps", tag="@2:Q")
    assert meta._rhs_derive_placeholders(rhs) == {"@1:eps", "@2:Q"}
    assert meta._rhs_derive_placeholders(Op.make("sink", "a")) == set()


def test_synthesize_composes_leaf_rules():
    """End-to-end: a ``$leaf:`` parent auto-composes — the derived
    rule's RHS carries the "@i:" leaf metavar and its composite
    derive refills it at fire time."""
    r1 = _leaf_mark_rule()
    r2 = Rewrite(
        "r_sink",
        Op.make("marked", "u", "k"),
        Op.make("sink", "u", "k"),
    )
    v = Var("v", _T())
    out = meta.synthesize_rules(
        [r1, r2], [Op.make("mark", v)], fuel=400, numeric_check=False
    )
    hits = [
        d
        for d in out
        if "@1:eps" in meta.pattern_metavars(d.rhs)
        or "@2:eps" in meta.pattern_metavars(d.rhs)
    ]
    assert hits, "no emitted rule carries the $leaf: placeholder"
    d = hits[0]
    # the placeholder refills at fire time — on the witness and on a
    # fresh instance alike
    w = Var("w", _T())
    for src_leaf in (v, w):
        t0 = meta.instantiate_pattern(
            d.lhs,
            {"v0": src_leaf}
            if "v0" in meta.pattern_metavars(d.lhs)
            else {"a": src_leaf},
        )
        applied = meta.apply_rewrite_at(d, t0, ())
        assert applied is not None
        # the minted Const lands in the rewritten term — not a string
        leaves = [
            a for a in (applied.args if isinstance(applied, Op) else ())
        ]
        assert Const(2.0) in leaves or any(
            Const(2.0) in getattr(s, "args", ()) for s in leaves
        )
        assert meta._term_is_ground(applied)


def test_synthesize_leaf_rule_as_second_parent():
    """The seam works in both seats: a ``$leaf:`` rule as r2 mints
    "@2:"-namespaced leaf placeholders."""
    r1 = Rewrite(
        "r_src",
        Op.make("mark", "a"),
        Op.make("marked", "a", "a"),
    )
    r3 = Rewrite(
        "r_tag",
        Op.make("marked", "u", "k"),
        Op.make("sink", "u", "eps"),
        dspec={"$leaf:eps": ("lit", 3.0)},
    )
    v = Var("v", _T())
    out = meta.synthesize_rules(
        [r1, r3], [Op.make("mark", v)], fuel=400, numeric_check=False
    )
    hits = [d for d in out if "@2:eps" in meta.pattern_metavars(d.rhs)]
    assert hits, "no emitted rule carries the @2: leaf placeholder"
    d = hits[0]
    m = meta.match_pattern(d.lhs, Op.make("mark", v), {})
    assert m is not None
    t0 = meta.instantiate_pattern(d.lhs, m)
    applied = meta.apply_rewrite_at(d, t0, ())
    assert applied is not None and meta._term_is_ground(applied)


def test_synthesize_batch_norm_expand_auto_composes():
    """The real client: ``batch_norm_expand`` is a ``$leaf:`` rule —
    it now participates in composition instead of being filtered."""
    x, w, b, rm, rv = _leaves((1, 8, 8, 8), 8)
    assert meta._synthesizable(BATCH_NORM_EXPAND)
    seed = _bn(x, w, b, rm, rv)
    out = meta.synthesize_rules(
        list(BATCH_NORM_RULES),
        [seed],
        fuel=800,
        numeric_check=False,
    )
    # every emitted composite is well-formed: rhs metavars are
    # lhs-bound or "@i:" placeholders the composite derive refills
    for d in out:
        rhs_t = {
            m
            for m in meta.pattern_metavars(d.rhs)
            if not m.startswith("$attr:")
        }
        lhs_t = {
            m
            for m in meta.pattern_metavars(d.lhs)
            if not m.startswith("$attr:")
        }
        assert rhs_t <= lhs_t | set(
            meta._rhs_derive_placeholders(d.rhs)
        )
        m = meta.match_pattern(d.lhs, seed, {})
        if m is None:
            continue
        t0 = meta.instantiate_pattern(d.lhs, m)
        applied = meta.apply_rewrite_at(d, t0, ())
        assert applied is not None
        assert meta._term_is_ground(applied)
