# ruff: noqa: RUF001, RUF003 — math notation (≡, σ, −, ·) in docstrings.
"""The codegen law bank — schedule-level rewrites over the IR.

Not a language: sixteen laws.  Fourteen fold spelled op-compositions
into *named fused-kernel ops* (``kf_*`` — the kernel vocabulary the
:class:`~mlopt_kernel.sink.KernelSink` lowers as one compiled
composite, in the repo's ``silu``/``sdpa``/``rms_norm`` fold-op
tradition); two are arithmetic strength reductions — the codegen
move that replaces an expensive elementary op with the cheap one a
device kernel emits anyway.

The fusion arm mirrors what Inductor actually emits: elementwise
gate fusions (``kf_silu``, ``kf_softsign``, ``kf_addmul``,
``kf_swiglu``), the exp/sum normalization a softmax really is
(``kf_softmax`` — the kernel twin of ``mlopt_core.laws``'
``softmax_fold``), the mean/var/rsqrt compositions the norm blocks
fuse into one pass (``kf_layernorm``, ``kf_rmsnorm`` — the gain-free
spelling of ``rms_norm_fold_nogain``'s site), and the matmul
epilogues a template kernel absorbs (``kf_mm_bias``,
``kf_mm_scale``, ``kf_mm_sb``, ``kf_mm_gelu``).  Two deeper folds
reach toward real-module chains — ``kf_softmax_mm`` (the
``softmax(qk)·v`` two-step attention chain) — and the
second-generation residue family the measured triton gap named:
``kf_addmul_tanh`` and ``kf_silu_sigmoid`` fold the pointwise op a
first-generation fusion *left surviving* into the fused op itself.
Side conditions are *data* where the repo's ``cond``/``dspec`` DSL
reaches (attr metavariables, numeric consts) — the same posture the
default bank took when ``softmax_fold`` / ``rms_norm_fold`` went
declarative.

A design note the e-graph forces: a *wrapper* marker
(``k_fused(X)``) merges into ``X``'s own e-class and points back at
it — a cyclic enode the extractor can never pick.  The fused ops
here take the region's *leaves* as children, so they are ordinary
non-cyclic members: extraction chooses between spelled and fused by
:func:`~mlopt_kernel.cost.kernel_cost` alone.

The bank deliberately stays out of ``mlopt_core.laws.DEFAULT``: a
change to a default rule changes extraction for every caller, and
these rules only pay under the kernel cost model.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op
from mlopt_core.laws.base import R

__all__ = [
    "FUSED_KERNEL_OPS",
    "KERNEL_RULES",
    "kern_div_recip",
    "kern_fuse_addmul",
    "kern_fuse_addmul_tanh",
    "kern_fuse_layernorm",
    "kern_fuse_mm_bias",
    "kern_fuse_mm_gelu",
    "kern_fuse_mm_scale",
    "kern_fuse_mm_scale_bias",
    "kern_fuse_rmsnorm",
    "kern_fuse_silu",
    "kern_fuse_silu_sigmoid",
    "kern_fuse_softmax",
    "kern_fuse_softmax_mm",
    "kern_fuse_softsign",
    "kern_fuse_swiglu",
    "kern_square_strength",
]

#: The kernel-vocabulary ops the laws mint — every ``kf_*`` name is a
#: single kernel launch in the sink's cost model and a compiled
#: composite in its binding table.
FUSED_KERNEL_OPS = frozenset(
    {
        "kf_silu",
        "kf_softsign",
        "kf_addmul",
        "kf_swiglu",
        "kf_silu_sigmoid",
        "kf_addmul_tanh",
        "kf_softmax",
        "kf_softmax_mm",
        "kf_layernorm",
        "kf_rmsnorm",
        "kf_mm_bias",
        "kf_mm_scale",
        "kf_mm_sb",
        "kf_mm_gelu",
    }
)

kern_fuse_silu = R(
    "kern_fuse_silu",
    Op.make("mul", "a", Op.make("sigmoid", "b")),
    Op.make("kf_silu", "a", "b"),
    law="Gate fusion: a·σ(b) lowers as one fused kernel — the "
    "elementwise product and the sigmoid in a single pass (SiLU is "
    "the a == b section).",
)

kern_fuse_softsign = R(
    "kern_fuse_softsign",
    Op.make("div", "a", Op.make("add", Op.make("abs", "b"), "c")),
    Op.make("kf_softsign", "a", "b", "c"),
    law="Softsign fusion: a/(|b|+c) lowers as one fused kernel — "
    "abs, add and div in a single pass (softsign(x) is a≡b, c≡1).",
)

kern_fuse_addmul = R(
    "kern_fuse_addmul",
    Op.make("add", Op.make("mul", "a", "b"), "c"),
    Op.make("kf_addmul", "a", "b", "c"),
    law="Fused multiply-add: a·b + c lowers as one fused kernel — "
    "the hardware FMA the elementwise pair would emit anyway.",
)

kern_fuse_swiglu = R(
    "kern_fuse_swiglu",
    Op.make("mul", Op.make("mul", "a", Op.make("sigmoid", "a")), "b"),
    Op.make("kf_swiglu", "a", "b"),
    law="SwiGLU fusion: (a·σ(a))·b lowers as one fused kernel — the "
    "silu gate and its multiply in a single pass (the deeper fusion "
    "beats mul(kf_silu, b) under the launch-dominated cost).",
)

# The gen-2 residue folds — a first-generation ``kf_*`` fusion leaves
# the pointwise op *next to* it spelled (``add(kf_addmul, tanh(c))``,
# ``mul(kf_silu, sigmoid(b))``), so the fused op absorbs the survivor
# too.  These mint the one-launch composite the measured triton arm
# already writes by hand.
kern_fuse_addmul_tanh = R(
    "kern_fuse_addmul_tanh",
    Op.make("add", Op.make("mul", "a", "b"), Op.make("tanh", "c")),
    Op.make("kf_addmul_tanh", "a", "b", "c"),
    law="Residue fusion: a·b + tanh(c) lowers as one fused kernel — "
    "the survivor tanh kern_fuse_addmul left spelled, absorbed into "
    "the epilogue (the measured fma-case gap).",
)

kern_fuse_silu_sigmoid = R(
    "kern_fuse_silu_sigmoid",
    Op.make(
        "mul",
        Op.make("mul", "a", Op.make("sigmoid", "a")),
        Op.make("sigmoid", "b"),
    ),
    Op.make("kf_silu_sigmoid", "a", "b"),
    law="Residue fusion: (a·σ(a))·σ(b) lowers as one fused kernel — "
    "the double-gate the swiglu fold reaches as "
    "``kf_swiglu(a, σ(b))`` (one fused op plus a spelled σ), folded "
    "whole (the measured mixed-case gap).",
)

# The softmax fold, kernel-side — ``div(exp(u), Σexp(u))`` lowers as
# one fused normalization kernel.  The side condition is the shipped
# ``softmax_fold``'s verbatim: ``keepdim`` must be True (a dropped
# axis cannot broadcast the Σ back over ``u``) and the reduce must
# cover exactly one axis (softmax has no multi-axis image); ``dim``
# unwraps a 1-tuple the same way the core fold does.
_COND_SOFTMAX_SITE = (
    "and",
    ("attr-is", "RK", True),
    (
        "or",
        ("attr-type", "RD", "int"),
        (
            "and",
            ("attr-type", "RD", "tuple"),
            ("attr-len", "RD", "==", 1),
        ),
    ),
)

kern_fuse_softmax = R(
    "kern_fuse_softmax",
    Op.make(
        "div",
        Op.make("exp", "u"),
        Op.make("sum", Op.make("exp", "u"), dim="RD", keepdim="RK"),
    ),
    Op.make("kf_softmax", "u", dim="SD"),
    cond=_COND_SOFTMAX_SITE,
    dspec={"SD": ("attr0", "RD")},
    law="Softmax normalization: exp(u)/Σexp(u) lowers as one fused "
    "kernel — a two-pass reduction the spelled div+exp+sum launches "
    "as three.  (The kernel twin of the library's softmax_fold.)",
)


kern_fuse_softmax_mm = R(
    "kern_fuse_softmax_mm",
    Op.make(
        "matmul",
        Op.make(
            "div",
            Op.make("exp", "u"),
            Op.make("sum", Op.make("exp", "u"), dim="RD", keepdim="RK"),
        ),
        "v",
    ),
    Op.make("kf_softmax_mm", "u", "v", dim="SD"),
    cond=_COND_SOFTMAX_SITE,
    dspec={"SD": ("attr0", "RD")},
    law="Attention chain: softmax(u)·v lowers as one fused kernel — "
    "the post-softmax value matmul a spelled attention block "
    "launches separately (``softmax(qk)`` gains a ``kf_softmax`` "
    "member too; the deeper fold wins on launch count).",
)


def _ln_core(u: str = "u") -> Op:
    """``sub(u, mean(u, MD, keepdim=MK))`` — the centered term, twice."""
    return Op.make(
        "sub",
        u,
        Op.make("mean", u, dim="MD", keepdim="MK"),
    )


# The layernorm block — ``c·rsqrt(mean(c²)+eps)`` with ``c`` the
# centered input.  The fused op needs no tail-block gate (unlike the
# library's ``rms_norm_fold``, whose ``F.rms_norm`` image demands a
# trailing reduce): ``kf_layernorm`` is *defined* as the composite —
# whatever axes ``mean`` reduces, the fused kernel computes.  The
# residual guards are ``keepdim`` (a dropped axis cannot broadcast
# ``c``'s norm back), ``eps`` a numeric ``Const`` (it becomes the
# kernel's float attr), and the exponent the literal 2.
_COND_LN_SITE = (
    "and",
    ("attr-is", "MK", True),
    ("const-num", "EPS"),
    ("const-cmp", "P", "==", 2),
)

kern_fuse_layernorm = R(
    "kern_fuse_layernorm",
    Op.make(
        "mul",
        _ln_core(),
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "mean",
                    Op.make("pow", _ln_core(), "P"),
                    dim="MD",
                    keepdim="MK",
                ),
                "EPS",
            ),
        ),
    ),
    Op.make("kf_layernorm", "u", dim="MD", eps="EP"),
    cond=_COND_LN_SITE,
    dspec={"EP": ("float", ("const", "EPS"))},
    law="Layernorm fusion: (x−μ)·rsqrt(mean((x−μ)²)+eps) lowers as "
    "one fused kernel — sub, two means, pow, add and rsqrt in a "
    "single pass over the row.",
)

# The gain-free RMSNorm block — ``x·rsqrt(mean(x²)+eps)`` — the
# spelled twin of the library's ``rms_norm_fold_nogain`` site.  Same
# front guards as layernorm (keepdim for the broadcast back, eps a
# numeric Const, the square literal); ``dim`` carries the mean's
# reduce spec verbatim — the fused op is *defined* as the composite.
kern_fuse_rmsnorm = R(
    "kern_fuse_rmsnorm",
    Op.make(
        "mul",
        "u",
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "mean",
                    Op.make("pow", "u", "P"),
                    dim="MD",
                    keepdim="MK",
                ),
                "EPS",
            ),
        ),
    ),
    Op.make("kf_rmsnorm", "u", dim="MD", eps="EP"),
    cond=_COND_LN_SITE,
    dspec={"EP": ("float", ("const", "EPS"))},
    law="RMSNorm fusion: x·rsqrt(mean(x²)+eps) lowers as one fused "
    "kernel — mul, mean, pow, add and rsqrt in a single pass over "
    "the row (the kernel twin of rms_norm_fold_nogain's site).",
)

kern_fuse_mm_bias = R(
    "kern_fuse_mm_bias",
    Op.make("add", Op.make("matmul", "a", "w"), "b"),
    Op.make("kf_mm_bias", "a", "w", "b"),
    law="Bias epilogue: matmul(a,w)+b lowers as one fused kernel — "
    "the addmm a template kernel absorbs into its epilogue where "
    "the spelled program launched mm then add.",
)

kern_fuse_mm_scale_bias = R(
    "kern_fuse_mm_scale_bias",
    Op.make(
        "add",
        Op.make("mul", Op.make("matmul", "a", "w"), "s"),
        "b",
    ),
    Op.make("kf_mm_sb", "a", "w", "s", "b"),
    law="Scale+bias epilogue: matmul(a,w)·s+b lowers as one fused "
    "kernel — the attention-scale spelling (QKᵀ·(1/√d)+bias) the "
    "template epilogue absorbs; beats kf_addmul's two launches.",
)

kern_fuse_mm_scale = R(
    "kern_fuse_mm_scale",
    Op.make("mul", Op.make("matmul", "a", "w"), "s"),
    Op.make("kf_mm_scale", "a", "w", "s"),
    law="Scale epilogue: matmul(a,w)·s lowers as one fused kernel — "
    "the bias-free attention-scale spelling (QKᵀ·(1/√d), no mask "
    "term) the template epilogue absorbs.",
)

kern_fuse_mm_gelu = R(
    "kern_fuse_mm_gelu",
    Op.make("gelu", Op.make("matmul", "a", "w")),
    Op.make("kf_mm_gelu", "a", "w"),
    law="Activation epilogue: gelu(matmul(a,w)) lowers as one fused "
    "kernel — the transformer-MLP activation a template kernel "
    "absorbs into the mm epilogue.",
)

kern_square_strength = R(
    "kern_square_strength",
    Op.make("pow", "x", Const(2)),
    Op.make("mul", "x", "x"),
    law="Strength reduction: pow(x, 2) lowers as one multiply — "
    "pow_to_square + square_expand composed, at codegen level.",
    derivation=("pow_to_square", "square_expand"),
)


def _numeric_const(bound: dict[str, Any]) -> bool:
    """``c`` bound to a nonzero numeric literal (reciprocal folds)."""
    c = bound["c"]
    return (
        isinstance(c, Const)
        and isinstance(c.value, (int, float))
        and not isinstance(c.value, bool)
        and c.value != 0
    )


def _recip(bound: dict[str, Any]) -> dict[str, Any] | None:
    """Mint ``r = 1/c`` — the folded constant ``kern_div_recip`` needs."""
    return {"r": Const(1 / bound["c"].value)}


kern_div_recip = R(
    "kern_div_recip",
    Op.make("div", "x", "c"),
    Op.make("mul", "x", "r"),
    check=_numeric_const,
    derive=_recip,
    law="Reciprocal folding: x / c → x · (1/c) for constant c — "
    "division is the expensive elementary op; a device kernel emits "
    "the multiply.",
)

#: The bank, in budget order — the board's action set.  The deeper
#: fusions (swiglu, the norm/epilogue/chain folds, the gen-2
#: residue ops) sit ahead of the strength reductions: they remove
#: more launches per fire.
KERNEL_RULES: tuple[Rewrite, ...] = (
    kern_fuse_silu,
    kern_fuse_softsign,
    kern_fuse_addmul,
    kern_fuse_swiglu,
    kern_fuse_silu_sigmoid,
    kern_fuse_addmul_tanh,
    kern_fuse_softmax,
    kern_fuse_softmax_mm,
    kern_fuse_layernorm,
    kern_fuse_rmsnorm,
    kern_fuse_mm_bias,
    kern_fuse_mm_scale,
    kern_fuse_mm_scale_bias,
    kern_fuse_mm_gelu,
    kern_square_strength,
    kern_div_recip,
)
