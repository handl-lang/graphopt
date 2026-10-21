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
from mlopt_core import meta
from mlopt_core.egraph import EGraph, Rewrite
from mlopt_core.ir import Const, Op, Param, TensorType, Var, op_repr
from mlopt_core.laws import (
    BATCH_NORM_EXPAND,
    BATCH_NORM_EXPAND_NOBIAS,
    BATCH_NORM_EXPAND_PLAIN,
    BATCH_NORM_RULES,
    DEFAULT,
    _check_bn_eval,
    _derive_bn,
)
from mlopt_core.laws.cond import derive_from_data, derive_to_data

# Registers the concrete-eval backend + op bindings for _eval_term.
from mlopt_torch.torch_bridge import export_to_ir  # noqa: F401


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














