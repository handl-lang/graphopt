"""Tests for the division-naturality bridge laws — ``mul_div_swap``
and ``matmul_div_distribute`` (:data:`mlopt_core.laws.tensor`).

    mul(div(a, b), c)      ->  div(mul(a, c), b)     (a/b)·c = (a·c)/b
    matmul(div(a, c), v)   ->  div(matmul(a, v), c)  (a/c)@v = (a@v)/c

Both are the spelled-score links in the sequential-carrier chain
(plan 0023): the export denominates the block score
``split(div(split(qkT), c))`` and the fold needs the scale out of
the way of the state-level contraction.  ``mul_div_swap`` is
unguarded — mul and div broadcast identically so the pointwise
identity holds for every shape; ``matmul_div_distribute`` carries
``("scalar", "c")`` — the linearity claim (a/c)@v = (a@v)/c is
exact only when the denominator is constant along the contracted
axis, and scalar is the spelling the export actually produces.

Covered surface:

* registration/tagging and kernel kinds;
* firing + RHS-is-member, fp64 numeric exactness;
* ``matmul_div_distribute`` declining a non-scalar denominator;
* membership in ``SIMPLIFICATION_RULES``/``ALL_RULES``.
"""

from __future__ import annotations

import torch
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import (
    ALL_RULES,
    DIV_ADD_FACTOR,
    MATMUL_DIV_DISTRIBUTE,
    MUL_DIV_SWAP,
    SIMPLIFICATION_RULES,
)
import mlopt_torch.torch_bridge as tb


def _eval(t, env):
    return tb.eval_term(t, var_env=env, bindings=tb._IR_TO_TORCH)




class TestMulDivSwap:
    def _term(self):
        a = Var("a", TensorType((2, 4)))
        b = Var("b", TensorType(()))
        c = Var("c", TensorType((2, 4)))
        return Op.make("mul", Op.make("div", a, b), c)

    def test_fires_and_rhs_member(self):
        eg = EGraph()
        r = eg.add_term(self._term())
        assert eg.apply_rule(MUL_DIV_SWAP, r)
        cls = eg._classes[eg.find(r)]
        assert {n.op for n in cls.nodes} == {"mul", "div"}

    def test_exact_fp64(self):
        a = Var("a", TensorType((2, 4)))
        b = Var("b", TensorType(()))
        c = Var("c", TensorType((2, 4)))
        env = {
            "a": torch.randn(2, 4, dtype=torch.float64),
            "b": torch.tensor(2.5, dtype=torch.float64),
            "c": torch.randn(2, 4, dtype=torch.float64),
        }
        lhs = _eval(Op.make("mul", Op.make("div", a, b), c), env)
        rhs = _eval(Op.make("div", Op.make("mul", a, c), b), env)
        assert torch.allclose(lhs, rhs, atol=0, rtol=1e-14)

    def test_broadcast_exact(self):
        a = Var("a", TensorType((2, 4)))
        b = Var("b", TensorType((2, 4)))
        c = Var("c", TensorType((4,)))
        env = {
            "a": torch.randn(2, 4, dtype=torch.float64),
            "b": torch.randn(2, 4, dtype=torch.float64) + 2.0,
            "c": torch.randn(4, dtype=torch.float64),
        }
        lhs = _eval(Op.make("mul", Op.make("div", a, b), c), env)
        rhs = _eval(Op.make("div", Op.make("mul", a, c), b), env)
        assert torch.allclose(lhs, rhs, atol=0, rtol=1e-14)


class TestMatmulDivDistribute:
    def _term(self, c):
        a = Var("a", TensorType((2, 3)))
        v = Var("v", TensorType((3, 4)))
        return Op.make("matmul", Op.make("div", a, c), v)

    def test_fires_scalar(self):
        eg = EGraph()
        r = eg.add_term(self._term(Var("c", TensorType(()))))
        assert eg.apply_rule(MATMUL_DIV_DISTRIBUTE, r)
        cls = eg._classes[eg.find(r)]
        assert {n.op for n in cls.nodes} == {"matmul", "div"}

    def test_declines_tensor_denominator(self):
        eg = EGraph()
        r = eg.add_term(self._term(Var("c", TensorType((2, 3)))))
        assert not eg.apply_rule(MATMUL_DIV_DISTRIBUTE, r)

    def test_exact_fp64(self):
        a = Var("a", TensorType((2, 3)))
        c = Var("c", TensorType(()))
        v = Var("v", TensorType((3, 4)))
        env = {
            "a": torch.randn(2, 3, dtype=torch.float64),
            "c": torch.tensor(4.0, dtype=torch.float64),
            "v": torch.randn(3, 4, dtype=torch.float64),
        }
        lhs = _eval(Op.make("matmul", Op.make("div", a, c), v), env)
        rhs = _eval(Op.make("div", Op.make("matmul", a, v), c), env)
        assert torch.allclose(lhs, rhs, atol=0, rtol=1e-14)


class TestDivAddFactor:
    """``x/c + y/c = (x+y)/c`` — the shared-denominator naturality.

    The block-sum spelling the sequential carrier needs: per-block
    score contributions arrive as ``div(matmul(s_i,·), √d)`` terms
    the e-graph must re-denominator before ``factor_matmul`` can
    collect the shared query.
    """

    def _term(self, c):
        x = Var("x", TensorType((2, 4)))
        y = Var("y", TensorType((2, 4)))
        return Op.make(
            "add",
            Op.make("div", x, c),
            Op.make("div", y, c),
        )

    def test_registration_and_kind(self):
        assert DIV_ADD_FACTOR in SIMPLIFICATION_RULES
        assert DIV_ADD_FACTOR in ALL_RULES
        assert DIV_ADD_FACTOR.kind == "axiom"
        assert not DIV_ADD_FACTOR.derivation

    def test_fires_scalar(self):
        eg = EGraph()
        r = eg.add_term(self._term(Var("c", TensorType(()))))
        assert eg.apply_rule(DIV_ADD_FACTOR, r)
        cls = eg._classes[eg.find(r)]
        assert {n.op for n in cls.nodes} == {"add", "div"}

    def test_declines_tensor_denominator(self):
        eg = EGraph()
        r = eg.add_term(self._term(Var("c", TensorType((2, 4)))))
        assert not eg.apply_rule(DIV_ADD_FACTOR, r)

    def test_exact_fp64(self):
        x = Var("x", TensorType((2, 4)))
        y = Var("y", TensorType((2, 4)))
        c = Var("c", TensorType(()))
        env = {
            "x": torch.randn(2, 4, dtype=torch.float64),
            "y": torch.randn(2, 4, dtype=torch.float64),
            "c": torch.tensor(4.0, dtype=torch.float64),
        }
        lhs = _eval(self._term(c), env)
        rhs = _eval(Op.make("div", Op.make("add", x, y), c), env)
        assert torch.allclose(lhs, rhs, atol=0, rtol=1e-14)
