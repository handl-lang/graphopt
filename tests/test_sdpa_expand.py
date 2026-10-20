"""Tests for the ``sdpa_expand`` family + ``sdpa_fold_mul_nomask``.

The fused-op decompose pair (``mlopt_core.laws.tensor``): the expansion
mints the kernel's own spelled form — ``matmul(softmax(mul(q@kᵀ, s),
dim=-1), v)`` — so the interior becomes an ordinary redex; the
mul-spelling fold is the shipped mediating inverse (re-fusion).

Coverage plan, mirroring ``test_attention_laws.py``:

* **fire + verify** — every benign key-set expands, every extractable
  member evaluates equal to the fused call at fp64;
* **decline** — masked / causal / train-dropout / GQA spellings never
  expand (exact-keyset + literal-attr guards);
* **round-trip** — expand mints a member ``sdpa_fold_mul_nomask``
  refolds to an explicit-``scale`` fused enode; extraction stays on a
  fused member;
* **reach** — inside the expanded interior the score-scale migration
  laws fire (the fused-op blindness the pair exists to cure);
* **certificates** — ``eg.certificate`` replays through the minted
  ``$leaf:`` Const;
* **wiring** — opt-in only: the family never leaks into ``DEFAULT``.
"""

from __future__ import annotations

import math

import pytest
import torch
from mlopt_core.cost import count_cost, flops_cost
from mlopt_core.egraph import EGraph, verify_certificate
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var
from mlopt_core.laws.ruleset import ATTENTION, DEFAULT
from mlopt_core.laws.tensor import (
    SDPA_EXPAND_RULES,
    SDPA_FOLD_MUL_NOMASK,
)
from mlopt_torch.torch_bridge import ir_to_torch_module

torch.manual_seed(0)

_EXPAND_BY_NAME = {r.name: r for r in SDPA_EXPAND_RULES}


def _v(name: str, *shape: int | None) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape: int | None) -> Param:
    return Param(name, TensorType(tuple(shape)))


def _saturate(term, rules, iters=8, max_nodes=100_000):
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(rules, eid, max_iterations=iters, max_nodes=max_nodes)
    return eg, eid


def _eval(term, feeds, params=None):
    xs = {k: v.to(torch.float64) for k, v in feeds.items()}
    ps = {k: v.to(torch.float64) for k, v in (params or {}).items()}
    inputs = [Var(k, TensorType(tuple(v.shape))) for k, v in xs.items()]
    pterms = {
        k: Param(k, TensorType(tuple(v.shape))) for k, v in ps.items()
    }
    mod = ir_to_torch_module(
        IR(
            root=term,
            inputs=inputs,
            input_names=set(xs),
            params=pterms,
        ),
        param_values=ps,
    )
    mod.eval()
    with torch.no_grad():
        return mod(*xs.values())


def _fired(eg, name):
    return eg.rule_fires.get(name, 0) > 0


def _members(eg, eid):
    """Every non-leaf member of the root class, each forced-extracted."""
    root = eg.find(eid)
    out = []
    for n in eg.get_class(eid).nodes:
        if n.op == "leaf":
            continue
        t = eg.extract_best(eid, count_cost, overrides={root: n})
        if t is not None:
            out.append(t)
    return out


def _check_all_members(eg, eid, lhs, feeds, ref, params=None):
    """Assert every member of the class evaluates equal to ``ref``."""
    torch.testing.assert_close(
        _eval(lhs, feeds, params), ref, rtol=0, atol=1e-12
    )
    for m in _members(eg, eid):
        got = _eval(m, feeds, params)
        torch.testing.assert_close(
            got, ref, rtol=0, atol=1e-12, msg=f"member {m!r} mismatch"
        )


def _contains(t, pred):
    return pred(t) or (
        isinstance(t, Op) and any(_contains(a, pred) for a in t.args)
    )


def _sdpa_ref(q, k, v, scale=None):
    with torch.no_grad():
        return torch.nn.functional.scaled_dot_product_attention(
            q.to(torch.float64),
            k.to(torch.float64),
            v.to(torch.float64),
            scale=scale,
        )


# ---------------------------------------------------------------------------
#  fires — canonical spellings
# ---------------------------------------------------------------------------


def test_sdpa_expand_fires_on_bare_sdpa():
    """``sdpa(q,k,v)`` (defaults elided — the canonical export) expands
    and mints ``mul(scores, 1/√d)`` — the implicit scale made explicit."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v)
    eg, eid = _saturate(lhs, [_EXPAND_BY_NAME["sdpa_expand"]])
    assert _fired(eg, "sdpa_expand")
    # the minted scale leaf is 1/√16 = 0.25, sitting under the mul
    scale_ok = lambda t: (  # noqa: E731
        isinstance(t, Op)
        and t.op == "mul"
        and isinstance(t.args[1], Const)
        and t.args[1].value == pytest.approx(1.0 / math.sqrt(d))
    )
    assert any(
        _contains(m, scale_ok) for m in _members(eg, eid)
    )
    feeds = {
        "q": torch.randn(B, H, T, d),
        "k": torch.randn(B, H, T, d),
        "v": torch.randn(B, H, T, d),
    }
    _check_all_members(eg, eid, lhs, feeds, _sdpa_ref(*feeds.values()))


def test_sdpa_expand_scale_fires_on_explicit_scale():
    """``sdpa(q,k,v,scale=s)`` — the scale attr is lifted verbatim."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v, scale=0.5)
    eg, eid = _saturate(lhs, [_EXPAND_BY_NAME["sdpa_expand_s"]])
    assert _fired(eg, "sdpa_expand_s")
    scale_ok = lambda t: (  # noqa: E731
        isinstance(t, Op)
        and t.op == "mul"
        and isinstance(t.args[1], Const)
        and t.args[1].value == 0.5
    )
    assert any(_contains(m, scale_ok) for m in _members(eg, eid))
    feeds = {
        "q": torch.randn(B, H, T, d),
        "k": torch.randn(B, H, T, d),
        "v": torch.randn(B, H, T, d),
    }
    _check_all_members(
        eg, eid, lhs, feeds, _sdpa_ref(*feeds.values(), scale=0.5)
    )


@pytest.mark.parametrize(
    "rulename,attrs",
    [
        ("sdpa_expand_d", {"dropout_p": 0.0}),
        ("sdpa_expand_dc", {"dropout_p": 0.0, "is_causal": False}),
        (
            "sdpa_expand_dcg",
            {
                "dropout_p": 0.0,
                "is_causal": False,
                "enable_gqa": False,
            },
        ),
        ("sdpa_expand_ds", {"dropout_p": 0.0, "scale": 0.25}),
        (
            "sdpa_expand_dcs",
            {"dropout_p": 0.0, "is_causal": False, "scale": 0.25},
        ),
    ],
)
def test_sdpa_expand_benign_attr_tails(rulename, attrs):
    """Retained-default key-sets expand — literal-attr values guard."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v, **attrs)
    eg, eid = _saturate(lhs, [_EXPAND_BY_NAME[rulename]])
    assert _fired(eg, rulename)
    feeds = {
        "q": torch.randn(B, H, T, d),
        "k": torch.randn(B, H, T, d),
        "v": torch.randn(B, H, T, d),
    }
    _check_all_members(
        eg,
        eid,
        lhs,
        feeds,
        _sdpa_ref(*feeds.values(), scale=attrs.get("scale")),
    )


@pytest.mark.parametrize(
    "attrs",
    [
        {"is_causal": True},  # causal — needs a minted mask; declined
        {"dropout_p": 0.1},  # train dropout — different math
        {"dropout_p": 0.0, "is_causal": True},
        {"enable_gqa": True},  # kv-repeat semantics — not the same op
        {
            "dropout_p": 0.0,
            "is_causal": False,
            "enable_gqa": True,
        },
    ],
)
def test_sdpa_expand_declines_nontrivial_attrs(attrs):
    """No benign pattern matches a causal/dropout/GQA spelling."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v, **attrs)
    eg, _ = _saturate(lhs, SDPA_EXPAND_RULES)
    assert not any(_fired(eg, r.name) for r in SDPA_EXPAND_RULES)


def test_sdpa_expand_declines_masked_arity():
    """The 4-operand masked form does not expand: bool keep-mask vs
    additive float mask is a dtype fact the IR cannot read."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    m = _v("m", B, H, T, T)
    lhs = Op.make("sdpa", q, k, v, m)
    eg, _ = _saturate(lhs, SDPA_EXPAND_RULES)
    assert not any(_fired(eg, r.name) for r in SDPA_EXPAND_RULES)


def test_sdpa_expand_declines_unshaped_q():
    """The implicit-scale mint vetoes honestly when q's head dim is
    unknown — a law never mints an attr it cannot verify."""
    B, H, T, d = 2, 4, 8, 16
    q = _v("q", B, H, T, None)
    k, v = _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v)
    eg, _ = _saturate(lhs, [_EXPAND_BY_NAME["sdpa_expand"]])
    assert not _fired(eg, "sdpa_expand")


# ---------------------------------------------------------------------------
#  the mediating fold — sdpa_fold_mul_nomask
# ---------------------------------------------------------------------------


def test_sdpa_fold_mul_nomask_fires_on_spelled():
    """``softmax(q kᵀ · s) v`` — the mul spelling people write —
    folds to the one kernel with ``scale`` minted from the Const."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1)
    )
    lhs = Op.make(
        "matmul",
        Op.make(
            "softmax", Op.make("mul", scores, Const(0.5)), dim=-1
        ),
        v,
    )
    eg, eid = _saturate(lhs, [SDPA_FOLD_MUL_NOMASK])
    assert _fired(eg, "sdpa_fold_mul_nomask")
    feeds = {
        "q": torch.randn(B, H, T, d),
        "k": torch.randn(B, H, T, d),
        "v": torch.randn(B, H, T, d),
    }
    _check_all_members(
        eg, eid, lhs, feeds, _sdpa_ref(*feeds.values(), scale=0.5)
    )
    best = eg.extract_best(eid, flops_cost)
    assert isinstance(best, Op) and best.op == "sdpa"
    assert best.attrs.get("scale") == 0.5


def test_sdpa_fold_mul_nomask_declines_nonconst_scale():
    """A non-Const scale operand declines — ``scale`` needs a float."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    s = _v("s", B, H, T, T)  # a full score-shaped scale — not a Const
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1)
    )
    lhs = Op.make(
        "matmul",
        Op.make("softmax", Op.make("mul", scores, s), dim=-1),
        v,
    )
    eg, _ = _saturate(lhs, [SDPA_FOLD_MUL_NOMASK])
    assert not _fired(eg, "sdpa_fold_mul_nomask")


# ---------------------------------------------------------------------------
#  round-trip — expand refolds; extraction stays fused
# ---------------------------------------------------------------------------


def test_expand_fold_round_trip_stays_fused():
    """``sdpa`` → spelled → ``sdpa(scale=1/√d)``: the mediating fold
    rebuilds the kernel; extraction picks a fused member."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v)
    rules = list(DEFAULT.rules) + list(SDPA_EXPAND_RULES)
    eg, eid = _saturate(lhs, rules)
    assert _fired(eg, "sdpa_expand")
    assert _fired(eg, "sdpa_fold_mul_nomask")
    best = eg.extract_best(eid, flops_cost)
    assert isinstance(best, Op) and best.op == "sdpa"
    # the refolded member carries the explicit scale (same value as
    # the implicit 1/√d the fused call had all along)
    assert best.attrs.get("scale") == pytest.approx(1.0 / math.sqrt(d))
    feeds = {
        "q": torch.randn(B, H, T, d),
        "k": torch.randn(B, H, T, d),
        "v": torch.randn(B, H, T, d),
    }
    torch.testing.assert_close(
        _eval(best, feeds),
        _sdpa_ref(*feeds.values()),
        rtol=0,
        atol=1e-12,
    )


def test_expand_fold_round_trip_scaled():
    """``sdpa(scale=s)`` expands and the fold's redex exists — the
    refold union lands on the already-present fused member (a no-op
    merge, so it does not count as a fire; the matcher is the proof)."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v, scale=0.5)
    rules = list(DEFAULT.rules) + list(SDPA_EXPAND_RULES)
    eg, eid = _saturate(lhs, rules)
    assert _fired(eg, "sdpa_expand_s")
    # the expanded member IS the fold's LHS redex — the pair rejoins
    assert list(eg.matches(SDPA_FOLD_MUL_NOMASK.lhs, eid))
    best = eg.extract_best(eid, flops_cost)
    assert isinstance(best, Op) and best.op == "sdpa"


# ---------------------------------------------------------------------------
#  reach — the fused op's interior is now a redex
# ---------------------------------------------------------------------------


def test_expand_exposes_score_scale_migration():
    """The measured fused-op blindness, cured: inside the expanded
    ``sdpa(scale=s)`` the score-scale migration fires — the (T,T)
    runtime mul reaches a compile-time weight fold that the fused
    call hid."""
    B, T, i, H, dh = 2, 8, 16, 4, 4
    o = H * dh
    x = _v("x", B, T, i)
    wq, wk, wv = _p("Wq", o, i), _p("Wk", o, i), _p("Wv", o, i)

    def head(t):
        return Op.make(
            "transpose",
            Op.make("reshape", t, shape=(B, T, H, dh)),
            dim0=1,
            dim1=2,
        )

    lhs = Op.make(
        "sdpa",
        head(Op.make("linear", x, wq)),
        head(Op.make("linear", x, wk)),
        head(Op.make("linear", x, wv)),
        scale=0.5,
    )
    rules = list(DEFAULT.rules) + list(ATTENTION.rules)
    eg, eid = _saturate(lhs, rules, iters=10)
    assert _fired(eg, "sdpa_expand_s")
    # the interior scale migrated: through the left matmul operand,
    # through the head views, into the projection weight
    assert _fired(eg, "naturality_scalar_left_rev")
    assert _fired(eg, "linear_out_scale")
    # a fused member carrying the weight-folded q exists — the fused
    # call was hiding this fold behind its boundary
    weight_folded = lambda t: (  # noqa: E731
        isinstance(t, Op)
        and t.op == "linear"
        and isinstance(t.args[1], Op)
        and t.args[1].op == "mul"
    )
    root = eg.find(eid)
    assert any(
        _contains(
            eg.extract_best(eid, count_cost, overrides={root: n}),
            weight_folded,
        )
        for n in eg.get_class(eid).nodes
        if n.op != "leaf"
    )
    # extraction still picks a fused sdpa — expansion is reach, not a
    # cost regression (the kernel is the cheap member on every axis)
    best = eg.extract_best(eid, flops_cost)
    assert isinstance(best, Op) and best.op == "sdpa"
    feeds = {"x": torch.randn(B, T, i)}
    params = {
        "Wq": torch.randn(o, i),
        "Wk": torch.randn(o, i),
        "Wv": torch.randn(o, i),
    }
    torch.testing.assert_close(
        _eval(best, feeds, params),
        _eval(lhs, feeds, params),
        rtol=0,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
#  certificates — the minted ``$leaf:`` Const replays
# ---------------------------------------------------------------------------


def test_certificate_replays_through_expansion():
    """The derivation stands alone: the expand + refold steps replay
    under ``verify_certificate`` without the e-graph."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v)
    rules = list(DEFAULT.rules) + list(SDPA_EXPAND_RULES)
    eg, eid = _saturate(lhs, rules)
    assert _fired(eg, "sdpa_expand")
    # certify the expanded member — its derivation records the
    # minted Const leaf (the ``$leaf:`` channel replays deterministically)
    expanded = next(
        n
        for n in eg.get_class(eid).nodes
        if n.op == "matmul"
    )
    term = eg.extract_best(
        eid, count_cost, overrides={eg.find(eid): expanded}
    )
    cert = eg.certificate(lhs, term)
    out = verify_certificate(lhs, cert)
    assert out is not None
    assert cert.stats.get("n_steps", 0) >= 1


# ---------------------------------------------------------------------------
#  wiring — opt-in status
# ---------------------------------------------------------------------------


def test_sdpa_expand_never_in_default():
    """The decompose direction is opt-in — nothing leaks into DEFAULT."""
    assert not (set(SDPA_EXPAND_RULES) & set(DEFAULT.rules))
    assert "sdpa_fold_mul_nomask" in {r.name for r in DEFAULT.rules}
    # the fold direction IS default — every pipeline refolds the mint
    names = {r.name for r in DEFAULT.rules}
    assert not any(n.startswith("sdpa_expand") for n in names)


def test_expand_family_rule_names():
    """The generated family covers the benign key-sets, one rule each."""
    assert {r.name for r in SDPA_EXPAND_RULES} == {
        "sdpa_expand",
        "sdpa_expand_d",
        "sdpa_expand_dc",
        "sdpa_expand_dcg",
        "sdpa_expand_s",
        "sdpa_expand_ds",
        "sdpa_expand_dcs",
        "sdpa_expand_dcgs",
    }
