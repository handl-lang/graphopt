# ruff: noqa: RUF002 -- the docstrings spell the conv+BN affine math
"""The conv output-affine absorb family — ``CONV_FOLD_RULES``.

The conv twin of ``linear_channel_scale`` (and its shift sibling): the
``conv2d``-headed anchors ``synthesize_rules`` was structurally unable
to mint while no bank rule named the op — gate A of
``bench/drivers/probes/conv_bn_fold_synth_probe.py`` /
``project/retros/conv-bn-fold-synth.md``.

    conv2d(x, W, b) · ŝ  =  conv2d(x, W∘ŝ′, b∘s)     ŝ ~ (1, C, 1, 1)
    conv2d(x, W, b) ± ĉ  =  conv2d(x, W, b ± c)      ĉ ~ (1, C, 1, 1)

The family is generated per (attr-prefix × bias-arity × operand-order
× direction): conv2d's positional attr tail is a PREFIX of
``(stride, padding, dilation, groups)`` and the matcher requires an
exact attr-key set.  Opt-in — held out of ``ALL_RULES``/``DEFAULT``
with ``BATCH_NORM_RULES`` pending the admission floor.
"""

import torch
from mlopt_core import meta
from mlopt_core.ir import Op, Param, TensorType, Var
from mlopt_core.laws import (
    CONV_FOLD_RULES,
)

# Registers the concrete-eval backend + op bindings for _eval_term.
from mlopt_torch.torch_bridge import export_to_ir  # noqa: F401


def _T(*shape):
    return TensorType(tuple(shape))


_BY_NAME = {r.name: r for r in CONV_FOLD_RULES}


def _leaves(x_shape, cout, cin, k=3):
    return (
        Var("x", _T(*x_shape)),
        Param("W", _T(cout, cin, k, k)),
        Param("b", _T(cout)),
        Param("sv", _T(1, cout, 1, 1)),
        Param("cv", _T(1, cout, 1, 1)),
    )


def _conv(x, w, b=None, **attrs):
    return Op.make("conv2d", x, w, *(() if b is None else (b,)), **attrs)


def _env(*leaves, seed=0):
    torch.manual_seed(seed)
    return {
        leaf: torch.randn(*leaf.typ.shape, dtype=torch.float64)
        for leaf in leaves
    }


# ---------------------------------------------------------------------------
#  family census + data posture
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  firing + guards
# ---------------------------------------------------------------------------














def test_conv_fold_numerically_exact():
    """fp64 agreement with F.conv2d through the torch bridge."""
    x, w, b, sv, cv = _leaves((2, 4, 8, 8), 6, 4)
    env = _env(x, w, b, sv, cv)
    conv = _conv(x, w, b, stride=(1, 1), padding=(1, 1))
    for name, t in (
        ("conv_channel_scale_sp", Op.make("mul", conv, sv)),
        ("conv_channel_scale_l_sp", Op.make("mul", sv, conv)),
        ("conv_channel_shift_sp", Op.make("add", conv, cv)),
        ("conv_channel_shift_l_sp", Op.make("add", cv, conv)),
        ("conv_channel_shift_sub_sp", Op.make("sub", conv, cv)),
    ):
        out = meta.apply_rewrite_at(_BY_NAME[name], t, ())
        assert out is not None, name
        assert meta._eval_allclose(
            meta._eval_term(t, env), meta._eval_term(out, env), 1e-10
        ), name
    # the nobias arity + the groups tail
    conv_nb = _conv(x, w, stride=(2, 2), padding=(1, 1))
    out_nb = meta.apply_rewrite_at(
        _BY_NAME["conv_channel_scale_sp_nobias"],
        Op.make("mul", conv_nb, sv),
        (),
    )
    assert meta._eval_allclose(
        meta._eval_term(Op.make("mul", conv_nb, sv), env),
        meta._eval_term(out_nb, env),
        1e-10,
    )
    wg = Param("Wg", _T(6, 2, 3, 3))
    bg = Param("bg", _T(6))
    env_g = _env(x, wg, bg, sv, cv, seed=1)
    conv_g = _conv(
        x, wg, bg, stride=(1, 1), padding=(1, 1), dilation=(1, 1),
        groups=2,
    )
    t_g = Op.make("mul", sv, conv_g)
    out_g = meta.apply_rewrite_at(
        _BY_NAME["conv_channel_scale_l_spdg"], t_g, ()
    )
    assert out_g is not None
    assert meta._eval_allclose(
        meta._eval_term(t_g, env_g), meta._eval_term(out_g, env_g), 1e-10
    )


