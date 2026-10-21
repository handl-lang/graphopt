"""Tests for the matmul/linear factor-distribute rank guard.

``project/retros/law-gap-targeted-gen.md`` §3 exposed a shipped
unsoundness: ``matmul`` contracts a rank>=2 right operand's axis -2 but
a rank-1 operand's ONLY axis, while ``add`` broadcast-aligns TRAILING
axes.  On a mixed-rank binding the distributive law is genuinely false:

    x(16,) @ a(16,) + x(16,) @ b(16,16)   !=   x @ (a + b)

— the scalar ``x·a`` broadcasts across the matvec's output axis
(``allclose=False``, max diff ~13 on randn).  The shipped
``weight_factor_matmul``, ``factor_matmul`` and
``distribute_matmul_over_add`` (plus ``weight_distribute_matmul`` — the
coherence catalogue's 4-member equivalence class) were unconditional;
``_check_mm_rhs_addends`` / ``_check_mm_rhs_weights`` now carry the
side condition.  The same matcher-cannot-see-shapes class affected the
``swiglu_fuse`` / ``parallel_mul_fuse`` pair folds — ``concat`` +
``chunk(·, 2)`` recovers each projection only when the paired weights
share every dim — guarded by ``_check_fuse_pair``.

Covered surface:

* the check hooks unit-tested branch by branch (equal rank, the legal
  rank>=2 batch-broadcast mixture, the rank-1 x rank>=2 veto, scalar
  and unshaped declines);
* the counterexample binding evaluated under the torch oracle —
  ``allclose=False``, and ``True`` on the sound region;
* firing declines on the counterexample term for every guarded rule,
  and firing + RHS-is-member on the equal-rank binding;
* the 4-member equivalence class still holds under the guard: on a
  well-typed binding the pair reach each other's members;
* the ``swiglu_fuse``/``parallel_mul_fuse`` pair guards — decline on a
  broadcastable-but-unequal weight pair, fire on equal weights;
* the unguarded left-operand pair (``right_distribute`` /
  ``right_factor``) stays unguarded and sound — contraction there is
  the last axis, which IS the broadcast axis.
"""

from __future__ import annotations

import torch
from mlopt_core.cost import backend_cost, executor_cost_for
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Param, TensorType, Var
from mlopt_core.laws import (
    RIGHT_DISTRIBUTE,
    RIGHT_FACTOR,
)
from mlopt_torch.adapters import TorchSink
from mlopt_torch.meta_eval import _eval_allclose, _eval_term

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
    return eg, root


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _env_of(*terms):
    """Random fp64 tensors keyed by the leaves of *terms*."""
    leaves: dict = {}

    def collect(t):
        if isinstance(t, (Var, Param)):
            leaves[t.name] = t
        elif isinstance(t, Op):
            for a in t.args:
                collect(a)

    for t in terms:
        collect(t)
    return {
        lf: torch.randn(lf.typ.shape, dtype=torch.float64)
        for lf in leaves.values()
    }


# ---------------------------------------------------------------------------
#  The counterexample under the torch oracle
# ---------------------------------------------------------------------------


def test_counterexample_is_numerically_false():
    """``x@a + x@B`` with a vector, B a matrix: the shipped rewrite was
    wrong — the dot product broadcasts over the matvec's output axis."""
    torch.manual_seed(0)
    x, a, b = _v("x", 16), _p("a", 16), _p("b", 16, 16)
    lhs = Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )
    rhs = Op.make("matmul", x, Op.make("add", a, b))
    env = _env_of(lhs, rhs)
    lv, rv = _eval_term(lhs, env), _eval_term(rhs, env)
    assert lv.shape == rv.shape == (16,)  # same SHAPE, wrong VALUES
    assert not _eval_allclose(lv, rv, 1e-6)


def test_sound_region_is_numerically_true():
    """Same-rank (and both-rank>=2) addends still satisfy bilinearity."""
    torch.manual_seed(0)
    # equal rank
    x, a, b = _v("x", 5, 4), _p("a", 4, 3), _p("b", 4, 3)
    lhs = Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )
    rhs = Op.make("matmul", x, Op.make("add", a, b))
    env = _env_of(lhs, rhs)
    assert _eval_allclose(_eval_term(lhs, env), _eval_term(rhs, env))
    # mixed rank, both >= 2: broadcast only replicates the batch axis
    w, a2, b2 = _p("w", 5, 4), _p("a2", 4, 3), _p("b2", 2, 4, 3)
    lhs2 = Op.make(
        "add", Op.make("matmul", w, a2), Op.make("matmul", w, b2)
    )
    rhs2 = Op.make("matmul", w, Op.make("add", a2, b2))
    env2 = _env_of(lhs2, rhs2)
    assert _eval_allclose(
        _eval_term(lhs2, env2), _eval_term(rhs2, env2)
    )
    # both rank-1 (dot + dot)
    w3, a3, b3 = _p("w3", 5, 4), _p("a3", 4), _p("b3", 4)
    lhs3 = Op.make(
        "add", Op.make("matmul", w3, a3), Op.make("matmul", w3, b3)
    )
    rhs3 = Op.make("matmul", w3, Op.make("add", a3, b3))
    env3 = _env_of(lhs3, rhs3)
    assert _eval_allclose(
        _eval_term(lhs3, env3), _eval_term(rhs3, env3)
    )


# ---------------------------------------------------------------------------
#  check hooks — branch by branch
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
#  Firing: declines on the counterexample, fires on equal-rank bindings
# ---------------------------------------------------------------------------


def _counterexample_term():
    """The documented witness: x(16,)@a(16,) + x(16,)@b(16,16)."""
    x, a, b = _v("x", 16), _p("a", 16), _p("b", 16, 16)
    return Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )


def _legal_term():
    """x(5,4)@a(4,3) + x(5,4)@b(4,3) — the LoRA-merge shape."""
    x, a, b = _v("x", 5, 4), _p("a", 4, 3), _p("b", 4, 3)
    return Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )










def test_left_operand_pair_unguarded_and_sound():
    """``(a+b)@W = a@W + b@W`` is sound on every evaluable binding —
    the contraction axis IS the last (broadcast-aligned) axis, so a
    mixed-rank left binding stays exact.  No guard by design."""
    assert RIGHT_DISTRIBUTE.check is None
    assert RIGHT_FACTOR.check is None
    torch.manual_seed(0)
    a, b, w = _p("a", 4), _p("b", 5, 4), _p("w", 4, 3)
    lhs = Op.make(
        "add", Op.make("matmul", a, w), Op.make("matmul", b, w)
    )
    rhs = Op.make("matmul", Op.make("add", a, b), w)
    env = _env_of(lhs, rhs)
    assert _eval_allclose(_eval_term(lhs, env), _eval_term(rhs, env))
    # …and it still fires
    eg, root = _saturate(lhs, [RIGHT_FACTOR])
    assert eg.rule_fires.get("right_factor_matmul", 0) > 0
    assert eg.find(root) == eg.find(eg.add_term(rhs))


# ---------------------------------------------------------------------------
#  The same bug class in the product-structure fuses
# ---------------------------------------------------------------------------


def _swiglu_lhs(sa, sb):
    x = _v("x", 5, sa[1])
    a, b = _p("A", *sa), _p("B", *sb)
    return Op.make(
        "mul",
        Op.make("silu", Op.make("linear", x, a)),
        Op.make("linear", x, b),
    )








# ---------------------------------------------------------------------------
#  Guards do not regress the legal firing + certificate path
# ---------------------------------------------------------------------------




