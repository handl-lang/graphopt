"""Tests for the square/rsqrt canonicalization bridge laws —
``mul_square`` and the three reciprocal-root folds
``div_sqrt_to_rsqrt`` / ``pow_to_rsqrt`` / ``recip_sqrt_to_rsqrt``
(:data:`mlopt_core.laws.tensor`).

    mul(u, u)                    -> square(u)
    div(1, sqrt(u))              -> rsqrt(u)
    pow(u, Const(-0.5))          -> rsqrt(u)
    reciprocal(sqrt(u))          -> rsqrt(u)

All four are the recorded-miss bridges from
``project/retros/rms-norm-law.md``: the RMSNorm fold pins
``pow(u, 2)`` inside ``mean`` and ``rsqrt(·)`` outside it, so a graph
spelling x² as ``x·x`` or the reciprocal root as ``1/√(·)`` /
``(·)^-0.5`` could never match.  ``mul_square`` is
``square_expand``'s definitional inverse (a lemma); the rsqrt folds
canonicalize the spellings into the member the fold needs —
``reciprocal(sqrt(u))`` is what ``1/sqrt`` actually exports as (aten
lowers scalar-over-tensor division to ``reciprocal(t) * 1``, and
``id_mul`` strips the unit).

Preconditions live in the pattern or the ``cond`` DSL: the shared
``u`` metavariable binds both mul operands to the same e-class; the
``const-cmp`` predicates accept ``Const(1)``/``Const(1.0)`` numerators
and ``Const(-0.5)`` exponents numerically while a ``Var``/``Param``
binding declines.

Covered surface:

* registration/tagging and kernel kinds;
* term-level match/instantiate round-trips;
* numeric soundness on real fp64 tensors (lhs == rhs);
* firing + RHS-is-member;
* the declines — operand-mismatched mul, non-unit numerator,
  non-``-0.5`` exponent, non-sqrt ``reciprocal`` argument;
* the e-graph reach the laws exist for: every recorded miss spelling
  of a manual RMSNorm (``x·x`` square, ``1/√`` as reciprocal, and
  ``pow(·, -0.5)`` — built as terms AND exported from real modules)
  lands the ``rms_norm`` member in the root e-class under
  ``ALL_RULES`` and survives extraction + ``sink.verify``;
* the ``mul_square``/``square_expand`` inverse pair sharing one
  e-class;
* the laws firing inside the public default pipeline.
"""

from __future__ import annotations

import torch
from mlopt_core.cost import backend_cost, dag_cost, executor_cost_for
from mlopt_core.egraph import EGraph, verify_certificate
from mlopt_core.ir import Const, IR, Op, Param, TensorType, Var
from mlopt_core.laws import (
    ALL_RULES,
    DIV_SQRT_TO_RSQRT,
    MUL_SQUARE,
    POW_TO_RSQRT,
    RECIP_SQRT_TO_RSQRT,
    SQUARE_EXPAND,
)
from mlopt_orchestrator import Optimizer
from mlopt_orchestrator.optimize import _lower_extracted
from mlopt_torch.adapters import TorchSink
from mlopt_torch.backend import TorchBackend
from mlopt_torch.torch_bridge import export_to_ir

_SINK = TorchSink()


def _cost_fn():
    """The pipeline's selection model (roofline + per-dispatch term)."""
    return backend_cost(
        executor_cost_for(lowering="generic"), _SINK.supported_ops
    )


def _saturate(term, rules, iters=6, nodes=60_000):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(rules, root, max_iterations=iters, max_nodes=nodes)
    return eg, root, eg.extract_best(root, _cost_fn())


def _params_of(term):
    """Every ``Param`` leaf of *term* (DAG-aware)."""
    seen: set[int] = set()
    out: dict[str, Param] = {}
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Param):
            out[t.name] = t
        elif isinstance(t, Op):
            stack.extend(t.args)
    return out


def _ir_of(term, ir):
    """Wrap *term* as an ``IR`` reusing *ir*'s inputs."""
    return IR(
        root=term,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=_params_of(term),
    )


def _mean(x: Op | Var) -> Op:
    """``mean(x, dim=(-1,), keepdim=True)`` — the rms reduce."""
    return Op.make("mean", x, dim=(-1,), keepdim=True)


def _rms_reduces(u):
    """Every recorded-miss spelling of ``rsqrt(mean(u²) + eps)`` —
    plus the canonical spelling (the control: it folds with no
    bridges) and the doubly-missed spelling (``x·x`` AND ``1/√``)."""
    mean_pow = _mean(Op.make("pow", u, Const(2)))
    mean_mul = _mean(Op.make("mul", u, u))
    eps_pow = Op.make("add", mean_pow, Const(1e-6))
    eps_mul = Op.make("add", mean_mul, Const(1e-6))
    return {
        "canonical_rsqrt": Op.make("rsqrt", eps_pow),
        "mul_spelled_x2": Op.make("rsqrt", eps_mul),
        "div_one_sqrt": Op.make(
            "div", Const(1), Op.make("sqrt", eps_pow)
        ),
        "recip_sqrt": Op.make("reciprocal", Op.make("sqrt", eps_pow)),
        "pow_neghalf": Op.make("pow", eps_pow, Const(-0.5)),
        "both_missed": Op.make(
            "div", Const(1), Op.make("sqrt", eps_mul)
        ),
    }


def _manual_rms(rec: Op, w: Param) -> Op:
    """``mul(mul(u, rec), w)`` — the gained fold's LHS shape."""
    u = Var("u", TensorType((4, 8)))
    return Op.make("mul", Op.make("mul", u, rec), w)


class _ManualRmsRecip(torch.nn.Module):
    """``x · (1/√mean(x·x)+eps) · w`` — 1/√ spelled ``1.0 / sqrt``."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = (x * x).mean(-1, keepdim=True)
        return x * (1.0 / torch.sqrt(m + 1e-6)) * self.w


class _ManualRmsPow(torch.nn.Module):
    """``x · (mean(x·x)+eps)^{-0.5} · w`` — the pow(-0.5) spelling."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = (x * x).mean(-1, keepdim=True)
        return x * (m + 1e-6).pow(-0.5) * self.w


# ---------------------------------------------------------------------------
#  Registration / tagging / kernel kind
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  Match / instantiate round-trips
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
#  Numeric soundness
# ---------------------------------------------------------------------------


def test_bridge_laws_sound_fp64():
    """lhs == rhs on real fp64 tensors — to ~1e-16 (rsqrt's kernel
    rounds differently than sqrt+div), not bitwise."""
    u = torch.randn(4, 8, dtype=torch.float64).abs() + 0.25
    assert torch.allclose(u * u, torch.square(u), atol=1e-15, rtol=0)
    rsqrt = torch.rsqrt(u)
    assert torch.allclose(
        1.0 / torch.sqrt(u), rsqrt, atol=1e-15, rtol=0
    )
    assert torch.allclose(
        torch.reciprocal(torch.sqrt(u)), rsqrt, atol=1e-15, rtol=0
    )
    assert torch.allclose(
        torch.pow(u, -0.5), rsqrt, atol=1e-15, rtol=0
    )


# ---------------------------------------------------------------------------
#  Firing + the declines
# ---------------------------------------------------------------------------


def test_mul_square_fires_and_rhs_is_member():
    u = Var("u", TensorType((4, 8)))
    eg, root, _best = _saturate(Op.make("mul", u, u), [MUL_SQUARE])
    assert eg.rule_fires.get("mul_square", 0) > 0
    assert list(eg.matches(Op.make("square", u), eg.find(root)))


def test_mul_square_declines_on_operand_mismatch():
    """``mul(u, v)`` — the shared ``u`` metavariable vetoes; no check."""
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    eg, _root, _best = _saturate(Op.make("mul", u, v), [MUL_SQUARE])
    assert eg.rule_fires.get("mul_square", 0) == 0


def test_div_sqrt_to_rsqrt_fires_and_declines():
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    sq = Op.make("sqrt", u)
    # Const(1) and Const(1.0) numerators both fold (numeric compare).
    for one in (Const(1), Const(1.0)):
        eg, root, _best = _saturate(
            Op.make("div", one, sq), [DIV_SQRT_TO_RSQRT]
        )
        assert eg.rule_fires.get("div_sqrt_to_rsqrt", 0) > 0
        assert list(eg.matches(Op.make("rsqrt", u), eg.find(root)))
    # A non-unit numerator or a non-Const leaf declines; a bare
    # numerator-without-sqrt denominator has no redex at all.
    for term in (
        Op.make("div", Const(2), sq),
        Op.make("div", v, sq),
        Op.make("div", Const(1), u),
    ):
        eg, _root, _best = _saturate(term, [DIV_SQRT_TO_RSQRT])
        assert eg.rule_fires.get("div_sqrt_to_rsqrt", 0) == 0


def test_pow_to_rsqrt_fires_and_declines():
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    eg, root, _best = _saturate(
        Op.make("pow", u, Const(-0.5)), [POW_TO_RSQRT]
    )
    assert eg.rule_fires.get("pow_to_rsqrt", 0) > 0
    assert list(eg.matches(Op.make("rsqrt", u), eg.find(root)))
    for exp in (Const(0.5), Const(2), Const(-1), v):
        eg, _root, _best = _saturate(Op.make("pow", u, exp), [POW_TO_RSQRT])
        assert eg.rule_fires.get("pow_to_rsqrt", 0) == 0


def test_recip_sqrt_to_rsqrt_fires_and_declines():
    u = Var("u", TensorType((4, 8)))
    eg, root, _best = _saturate(
        Op.make("reciprocal", Op.make("sqrt", u)),
        [RECIP_SQRT_TO_RSQRT],
    )
    assert eg.rule_fires.get("recip_sqrt_to_rsqrt", 0) > 0
    assert list(eg.matches(Op.make("rsqrt", u), eg.find(root)))
    # reciprocal(u) with no sqrt operand — no match.
    eg, _root, _best = _saturate(
        Op.make("reciprocal", u), [RECIP_SQRT_TO_RSQRT]
    )
    assert eg.rule_fires.get("recip_sqrt_to_rsqrt", 0) == 0


# ---------------------------------------------------------------------------
#  The inverse pair shares one e-class
# ---------------------------------------------------------------------------


def test_mul_square_expand_share_one_eclass():
    """``square(u)`` gains the ``mul(u,u)`` member under
    ``square_expand``; the fold's match on that member is a no-op
    merge — the two directions coexist in one class."""
    u = Var("u", TensorType((4, 8)))
    term = Op.make("square", u)
    eg, root, _best = _saturate(term, [SQUARE_EXPAND, MUL_SQUARE])
    assert eg.rule_fires.get("square_expand", 0) > 0
    assert eg.find(root) == eg.find(eg.add_term(Op.make("mul", u, u)))


# ---------------------------------------------------------------------------
#  The reach the laws exist for — every miss spelling folds to rms_norm
# ---------------------------------------------------------------------------






def test_real_export_spellings_fold_and_verify():
    """The two real-export spellings — ``1/√`` (reciprocal + a mul-by-1
    wrapper) and ``pow(·, -0.5)`` — fold to ``rms_norm`` end to end:
    firing on the exported graph, the extracted member, the replayed
    certificate, and the lowered before/after modules agreeing."""
    x = torch.randn(4, 8, dtype=torch.float64)
    for model in (
        _ManualRmsRecip().eval().double(),
        _ManualRmsPow().eval().double(),
    ):
        ir, leaves = export_to_ir(model, x)
        cost_fn = _cost_fn()
        eg = EGraph()
        root = eg.add_term(ir.root)
        eg.run(ALL_RULES, root, max_iterations=6, max_nodes=60_000)
        assert eg.rule_fires.get("rms_norm_fold", 0) > 0
        best = eg.extract_best(root, cost_fn)
        assert best.op == "rms_norm"
        assert dag_cost(best, cost_fn) <= dag_cost(ir.root, cost_fn)
        cert = eg.certificate(ir.root, best, cost_fn=cost_fn)
        verify_certificate(ir.root, cert)
        m0 = _lower_extracted(
            ir.root, _ir_of(ir.root, ir), leaves, _SINK
        )
        m1 = _lower_extracted(best, _ir_of(best, ir), leaves, _SINK)
        assert _SINK.verify(m0, m1, (x,), rtol=1e-4).passed


def test_bridges_fire_in_default_pipeline():
    """The public default pipeline sees the canonicalization fire on
    the real ``pow(·, -0.5)``-spelled export and verifies."""
    model = _ManualRmsPow().eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x, max_iterations=6, max_enodes=100_000)
    assert res.stats["rule_fires"].get("pow_to_rsqrt", 0) > 0
    assert res.stats["rule_fires"].get("mul_square", 0) > 0
    low = opt.lower(res, x, verify=True)
    assert low.verified is not None and low.verified.passed
