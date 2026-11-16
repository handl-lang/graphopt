"""Creator-op dtype context — the ``dtype_ctx`` contract.

A float creator with no ``dtype`` attr (``arange`` with float
bounds, ``zeros``/``ones``/``empty``/``randn``/``rand``/``linspace``,
float-fill ``full``) mints at the *computation's* dtype — the same
*like* tensor ``_mint_const`` uses for Const scalars — instead of
torch's ambient default (fp32).  This is what keeps an fp64 eval
honest: an unpinned float ``arange`` inside an fp64 graph used to
mint f32 tables (the llama verify-run RoPE divergence, ~4.5e-7).

Int-valued spellings keep torch's int inference (int-bound
``arange``, int-fill ``full``), explicit ``dtype`` attrs still win,
and outside an eval window the ambient default applies.
"""

import torch
from mlopt_core.ir import Const, Op
from mlopt_torch.torch_bridge import _IR_TO_TORCH, eval_term

F64 = {"w": torch.ones(4, dtype=torch.float64)}
F32 = {"w": torch.ones(4, dtype=torch.float32)}


def _ev(term, env=None):
    return eval_term(
        term,
        var_env=env or {},
        param_env=F64,
        bindings=_IR_TO_TORCH,
        strict=True,
    )


def test_float_arange_follows_f64_like():
    """Float-bound ``arange`` mints f64 under an f64 computation."""
    t = Op("arange", (Const(0.0), Const(8.0), Const(1.0)))
    out = _ev(t)
    assert out.dtype == torch.float64
    assert out.tolist() == [float(i) for i in range(8)]


def test_float_arange_follows_f32_like():
    """The same term mints f32 under an f32 computation."""
    t = Op("arange", (Const(0.0), Const(8.0), Const(1.0)))
    out = eval_term(
        t,
        var_env={},
        param_env=F32,
        bindings=_IR_TO_TORCH,
        strict=True,
    )
    assert out.dtype == torch.float32


def test_int_arange_keeps_int64():
    """Int-bound ``arange`` keeps torch's int64 inference — the ctx
    fallback only fires on float bounds."""
    t = Op("arange", (Const(0), Const(8), Const(1)))
    out = _ev(t)
    assert out.dtype == torch.int64


def test_explicit_dtype_attr_wins():
    """An explicit ``dtype`` attr overrides the context."""
    t = Op(
        "arange",
        (Const(0.0), Const(8.0), Const(1.0)),
        {"dtype": "float32"},
    )
    out = _ev(t)
    assert out.dtype == torch.float32


def test_creator_outside_eval_keeps_ambient():
    """No *like* anywhere → torch's ambient default (f32)."""
    t = Op("arange", (Const(0.0), Const(8.0), Const(1.0)))
    out = eval_term(t, bindings=_IR_TO_TORCH, strict=True)
    assert out.dtype == torch.float32


def test_float_creators_follow_ctx():
    """``zeros``/``ones``/``empty``/``randn``/``rand``/``linspace``
    all mint at the computation dtype."""
    for name, term in (
        ("zeros", Op("zeros", (), {"shape": (2, 2)})),
        ("ones", Op("ones", (), {"shape": (2, 2)})),
        ("empty", Op("empty", (), {"shape": (2, 2)})),
        ("randn", Op("randn", (), {"shape": (2, 2)})),
        ("rand", Op("rand", (), {"shape": (2, 2)})),
        (
            "linspace",
            Op(
                "linspace",
                (),
                {"start": 0.0, "end": 1.0, "steps": 4},
            ),
        ),
    ):
        out = _ev(term)
        assert out.dtype == torch.float64, name


def test_full_float_fill_follows_ctx_int_fill_stays_int():
    """``full`` splits on the fill value: float fill → ctx dtype,
    int fill → torch's int64 inference."""
    float_fill = Op("full", (Const(1.5),), {"shape": (2,)})
    assert _ev(float_fill).dtype == torch.float64
    int_fill = Op("full", (Const(2),), {"shape": (2,)})
    assert _ev(int_fill).dtype == torch.int64


def test_ctx_resets_after_eval():
    """The context is scoped to the eval call — a later eval with a
    different *like* sees its own dtype, and a bare creator outside
    any eval still gets the ambient default."""
    t = Op("arange", (Const(0.0), Const(4.0), Const(1.0)))
    out32 = eval_term(
        t,
        var_env={},
        param_env=F32,
        bindings=_IR_TO_TORCH,
        strict=True,
    )
    assert out32.dtype == torch.float32
    out_bare = eval_term(t, bindings=_IR_TO_TORCH, strict=True)
    assert out_bare.dtype == torch.float32
