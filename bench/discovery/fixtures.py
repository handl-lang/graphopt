"""Spelled "before" fixtures — the corpus's input programs.

Each entry in :data:`SOURCES` is a builder ``size -> (nn.Module,
feed)`` spelling the program the way a practitioner writes it —
*before* the named optimization.  Builders run under the caller's
seed; modules are ``.eval().double()`` so the sink's fp64
equivalence gate applies unchanged.

``feed`` is a tensor or a positional-args tuple, matching the
``TorchSource`` / ``sink.verify`` convention.
"""

# ruff: noqa: RUF002, RUF003 — math notation (·, σ, ⊗, ᵀ) in docstrings
# is deliberate, per bench convention (mirrors refs/__init__.py).

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

_T64 = torch.float64


def _r(*shape: int) -> torch.Tensor:
    """A fresh fp64 example input."""
    return torch.randn(tuple(shape), dtype=_T64)


class _SwiGLU(nn.Module):
    """LLaMA-style SwiGLU spelled as two projections + a pointwise mul.

    ``silu(x@A.T) ⊙ (x@B.T)`` — the fused form (one widened GEMM +
    split) is what the pairing pass / ``swiglu_fuse`` reach for.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.a = nn.Linear(d, d, bias=False)
        self.b = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.a(x)) * self.b(x)


class _ManualAttention(nn.Module):
    """Causal softmax attention spelled by hand.

    ``softmax(q@kᵀ/√d + mask) @ v`` with the additive causal mask as a
    registered buffer — the spelling ``sdpa_fold_add*`` folds and the
    causal-mask specialization can then close to ``is_causal``.
    """

    def __init__(self, d: int, t: int) -> None:
        super().__init__()
        self.d = d
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        m = torch.zeros(t, t).masked_fill(
            ~torch.tril(torch.ones(t, t, dtype=torch.bool)), -math.inf
        )
        self.register_buffer("mask", m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.d)
        return torch.softmax(s + self.mask, dim=-1) @ v


class _DiagRecurrence(nn.Module):
    """``h_t = a ⊙ h_{t-1} + x_t`` unrolled T steps — a sequential scan.

    The spelled program reads the sequence one element at a time; the
    efficient form is the diagonal-affine scan monoid
    (Blelloch-style parallel prefix), which the ``decay_state_lift``
    family derives.
    """

    def __init__(self, d: int, t: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.full((d,), 0.9, dtype=_T64))
        self.t = t
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.shape[0], self.d, dtype=x.dtype)
        for i in range(self.t):
            h = self.a * h + x[:, i]
        return h


class _MulUnsq(nn.Module):
    """``a.unsqueeze(-1) * b.unsqueeze(-1)`` — both sides view-lifted.

    The DeltaRuleBlock ``β·k`` op-tuple the census minted
    ``mul_unsq_nat`` from: the fused form multiplies first and lifts
    the view once.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.unsqueeze(-1) * b.unsqueeze(-1)


class _ManualRMSNorm(nn.Module):
    """LLaMA-style RMSNorm spelled by hand: ``x·rsqrt(mean(x²)+eps)·w``.

    The canonical spelling the ``rms_norm_fold`` LHS pins — ``pow``
    for the square, ``rsqrt`` for the reciprocal root, the
    ``(x·rms)·w`` association the exporter writes.
    """

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, dtype=_T64))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.w


class _ManualRMSNormNoGain(nn.Module):
    """``x·rsqrt(mean(x²)+eps)`` — the weight-free RMSNorm spelling.

    ``rms_norm_fold_nogain``'s own redex (and the gained fold's inner
    ``mul``): ``F.rms_norm`` with ``weight=None``.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms


class _ManualRMSNormMisspelled(nn.Module):
    """The doubly-missed RMSNorm spelling: ``x·x`` and ``(·)^{-1/2}``.

    Both halves of the fold's pattern arrive non-canonical — the
    square as ``mul(x, x)`` and the reciprocal root as ``pow(m, -0.5)``
    — so ``rms_norm_fold`` can only reach it through the canonicalizer
    bridges ``mul_square → square_to_pow`` and ``pow_to_rsqrt`` (the
    recorded-miss spellings of ``rms-norm-law.md`` /
    ``test_square_rsqrt_laws``).
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, dtype=_T64))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = (x * x).mean(-1, keepdim=True)
        return x * (m + self.eps).pow(-0.5) * self.w


class _ManualSoftmax(nn.Module):
    """``exp(u) / Σ exp(u)`` — softmax spelled as a composed-then-
    reduced chain, exactly what ``softmax_fold``'s pattern-recognition
    mint saw in the census."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = x.exp()
        return e / e.sum(-1, keepdim=True)


class _ManualSiLU(nn.Module):
    """``x · σ(x)`` — SiLU spelled from its definition; ``silu_fold``
    (the definitional inverse of ``silu_expand``) folds it to the one
    kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class _ManualSoftsign(nn.Module):
    """``x / (|x| + 1)`` — softsign spelled in decomposed div/abs/add
    form; the minted ``softsign_fold`` collapses three ops to one."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.abs() + 1)


class _ManualGLU(nn.Module):
    """``a ⊗ σ(b)`` over the two halves of ``x`` — the manual GLU.

    ``F.glu`` IS this composition; ``glu_fold`` recognises the
    chunk-indexed gate spelling (``corpus-expansion-r2.md`` §6).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class _EncoderAttention(nn.Module):
    """``softmax(q@kᵀ/√d) @ v`` with no mask — encoder attention.

    ``nn.MultiheadAttention``'s default path exports exactly this
    spelling, which the add/masked_fill fold family cannot see;
    ``sdpa_fold_div_nomask`` is its row (promoted-laws).
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.d)
        return torch.softmax(s, dim=-1) @ v


class _SplitQKV(nn.Module):
    """MHA with three separate projections — the fused-QKV site.

    ``<q,k,v> : X -> V³`` is one map into the product space: the
    pairing pass offers each projection its section of a single
    widened GEMM (the ``MergedColumnParallelLinear`` /
    ``qkv_fuse``-subsumed transform).
    """

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.d = d
        self.h = h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        def head(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, self.h, d // self.h).transpose(1, 2)

        return F.scaled_dot_product_attention(
            head(self.q(x)), head(self.k(x)), head(self.v(x))
        )


class _ParallelLinearSum(nn.Module):
    """``linear(x, A) + linear(x, B)`` — the weight-merge site.

    Two parallel projections of the same input summed: the LoRA /
    model-soup merge ``weight_factor_linear`` performs —
    ``linear(x, A+B)`` with the parameter add folding at compile
    time.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.a = nn.Linear(d, d, bias=False)
        self.b = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.b(x)


class _SelectMul(nn.Module):
    """``select(a, d, i) ⊙ select(b, d, i)`` — the slice product.

    ``select_mul`` commutes the elementwise product past the select
    view (``sel(u) ⊙ sel(v) = sel(u ⊙ v)`` — one ``select`` instead
    of two).
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.select(1, 0) * b.select(1, 0)


class _AdditiveRecurrence(nn.Module):
    """``h_t = h_{t-1} + x_t`` unrolled T steps — the unit-decay corner.

    The pure prefix-sum recurrence: the ``a = 1`` section of the
    diagonal-affine monoid ``blelloch_scan`` exercises.  The spelled
    program still reads the sequence one element at a time; the
    ``affd_lift_unit`` family lifts it to the same
    ``aff_diag``/``applyd`` carrier with a constant-one decay —
    the scan/cumsum association in its cheapest spelling.
    """

    def __init__(self, d: int, t: int) -> None:
        super().__init__()
        self.t = t
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.shape[0], self.d, dtype=x.dtype)
        for i in range(self.t):
            h = h + x[:, i]
        return h


# ---------------------------------------------------------------------------
#  The source registry — ``size`` is each fixture's scale knob
# ---------------------------------------------------------------------------


def _conv_bn_eval(size: int):
    """``BatchNorm2d(Conv2d(x))`` in eval mode — the inference fold site.

    ``size`` is the conv's output-channel count (the minted region was
    measured at cout=6, kernel 3, pad 1 — ``tests/test_minted_laws``).
    """
    cout = max(2, size)
    model = nn.Sequential(
        nn.Conv2d(4, cout, 3, padding=1), nn.BatchNorm2d(cout)
    )
    return model.eval().double(), _r(1, 4, 8, 8)


def _retnet_chunked(size: int):
    """RetNet inner-chunk retention — decay-masked quadratic attention.

    ``size`` is the chunk length T.  The spelled module is the intake
    zoo's ``_ChunkedRetention`` (Sun et al. 2023, ``(QKᵀ ⊙ D) V`` with
    the decay matrix built in-graph).
    """
    from mlopt_discovery.zoo import _ChunkedRetention

    t = max(8, size)
    return _ChunkedRetention(16).eval().double(), _r(2, t, 16)


def _flash_attention(size: int):
    """Hand-spelled causal softmax attention; ``size`` is the length T."""
    t = max(4, size)
    return _ManualAttention(16, t).eval().double(), _r(2, t, 16)


def _blelloch_scan(size: int):
    """The unrolled diagonal recurrence; ``size`` is the length T."""
    t = max(4, size)
    return _DiagRecurrence(16, t).eval().double(), _r(2, t, 16)


def _mul_unsq_nat(size: int):
    """The two-sided unsqueeze product; ``size`` is the feature dim."""
    d = max(2, size)
    return _MulUnsq().eval().double(), (_r(4, d), _r(4, d))


def _weight_chain(size: int):
    """Two bias-free ``Linear``s back to back — the reassociation site."""
    d = max(4, size)
    model = nn.Sequential(
        nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
    )
    return model.eval().double(), _r(4, d)


def _swiglu_mlp(size: int):
    """The two-projection SwiGLU; ``size`` is the model width d."""
    d = max(4, size)
    return _SwiGLU(d).eval().double(), _r(4, d)


def _rms_norm_gained(size: int):
    """The gained manual RMSNorm; ``size`` is the normalized width d."""
    d = max(4, size)
    return _ManualRMSNorm(d).eval().double(), _r(4, d)


def _rms_norm_nogain(size: int):
    """The weight-free manual RMSNorm; ``size`` is the width d."""
    d = max(4, size)
    return _ManualRMSNormNoGain().eval().double(), _r(4, d)


def _rms_norm_misspelled(size: int):
    """The ``x·x`` / ``pow(-0.5)`` RMSNorm; ``size`` is the width d."""
    d = max(4, size)
    return _ManualRMSNormMisspelled(d).eval().double(), _r(4, d)


def _manual_softmax(size: int):
    """``exp/Σexp`` softmax over the last dim; ``size`` is the width."""
    d = max(4, size)
    return _ManualSoftmax().eval().double(), _r(4, d)


def _manual_silu(size: int):
    """``x·σ(x)`` on a (4, d) block; ``size`` is the width d."""
    d = max(4, size)
    return _ManualSiLU().eval().double(), _r(4, d)


def _manual_softsign(size: int):
    """``x/(|x|+1)`` on a (4, d) block; ``size`` is the width d."""
    d = max(4, size)
    return _ManualSoftsign().eval().double(), _r(4, d)


def _manual_glu(size: int):
    """``a ⊗ σ(b)`` on an even last dim; ``size`` is the width d."""
    d = max(4, size)
    if d % 2:
        d += 1
    return _ManualGLU().eval().double(), _r(4, d)


def _sdpa_nomask(size: int):
    """No-mask spelled attention; ``size`` is the length T."""
    t = max(4, size)
    return _EncoderAttention(16).eval().double(), _r(2, t, 16)


def _mha_qkv(size: int):
    """Split-QKV multi-head attention; ``size`` is the length T."""
    t = max(4, size)
    return _SplitQKV(16, 4).eval().double(), _r(2, t, 16)


def _weight_merge(size: int):
    """``linear(x,A) + linear(x,B)``; ``size`` is the width d."""
    d = max(4, size)
    return _ParallelLinearSum(d).eval().double(), _r(4, d)


def _gqa_repeat_kv(size: int):
    """GQA attention with materialised ``repeat_kv``; ``size`` is T.

    The llama2.c spelling — ``unsqueeze→expand→reshape`` duplicates
    each kv head in-graph before ``F.sdpa``; ``gqa_absorb_repeat``
    pushes the copy into the kernel's ``enable_gqa`` flag.
    """
    from mlopt_torch.models import RepeatKVAttention

    t = max(4, size)
    return (
        RepeatKVAttention(dim=32, n_heads=4, n_kv_heads=2)
        .eval()
        .double(),
        _r(2, t, 32),
    )


def _biased_chain(size: int):
    """Two biased ``Linear``s back to back — the affine compose site."""
    d = max(4, size)
    model = nn.Sequential(nn.Linear(d, d), nn.Linear(d, d))
    return model.eval().double(), _r(4, d)


def _select_mul(size: int):
    """``select(a,1,0) * select(b,1,0)``; ``size`` is the selected dim."""
    d = max(2, size)
    return _SelectMul().eval().double(), (_r(4, d), _r(4, d))


def _cumsum_scan(size: int):
    """The unrolled unit-decay recurrence; ``size`` is the length T."""
    t = max(4, size)
    return _AdditiveRecurrence(16, t).eval().double(), _r(2, t, 16)


# ---------------------------------------------------------------------------
#  Real-corpus sources — the held-out zoo's own modules
# ---------------------------------------------------------------------------
#
# These builders instantiate the *actual* ``mlopt_discovery.zoo``
# workload classes — the architectures the model-zoo-yield /
# human-bar retros measured — rather than a purpose-built respelling.
# The case TOML then replays the recorded fire so the scorecard's
# real-model yield is guarded by rows, not prose.


def _zoo_lora_adapter(size: int):
    """The zoo ``LoRAAdapter``; ``size`` is the model width d."""
    from mlopt_discovery.zoo import _LoRAAdapter

    d = max(8, size)
    return _LoRAAdapter(d, 4).eval().double(), _r(4, d)


def _zoo_time_cond_conv(size: int):
    """The zoo ``TimeCondConv`` diffusion block (fixed channels)."""
    from mlopt_discovery.zoo import _TimeCondConv

    del size  # the zoo pins ch=8, temb=4 — the measured cell
    return _TimeCondConv(8, 4).eval().double(), (_r(1, 8, 8, 8), _r(1))


def _zoo_gau(size: int):
    """The zoo ``GatedAttentionUnit``; ``size`` is the width d."""
    from mlopt_discovery.zoo import _GatedAttentionUnit

    d = max(8, size)
    return _GatedAttentionUnit(d).eval().double(), _r(2, 8, d)


def _zoo_spline_kan(size: int):
    """The zoo ``SplineKAN`` edge layer; ``size`` is the input dim."""
    from mlopt_discovery.zoo import _SplineKAN

    d = max(8, size)
    return _SplineKAN(d, 8).eval().double(), _r(4, d)


def _zoo_cosine_attention(size: int):
    """The zoo ``CosineAttention``; ``size`` is the width d."""
    from mlopt_discovery.zoo import _CosineAttention

    d = max(8, size)
    return _CosineAttention(d, 4).eval().double(), _r(2, 8, d)


def _zoo_mos_head(size: int):
    """The zoo ``MoSHead``; ``size`` is the feature width d."""
    from mlopt_discovery.zoo import _MoSHead

    d = max(8, size)
    return _MoSHead(d, 10, 3).eval().double(), _r(4, d)


def _zoo_adaln_block(size: int):
    """The zoo ``AdaLNBlock``; ``size`` is the width d."""
    from mlopt_discovery.zoo import _AdaLNBlock

    d = max(8, size)
    return (
        _AdaLNBlock(d, 8).eval().double(),
        (_r(2, 8, d), _r(2, 8)),
    )


def _zoo_delta_rule(size: int):
    """The zoo ``DeltaRuleBlock`` — the unrolled matrix-state write."""
    from mlopt_discovery.zoo import _DeltaRuleBlock

    d = max(8, size)
    return _DeltaRuleBlock(d, d, 4).eval().double(), _r(2, 4, d)


#: ``name -> builder(size)`` for the case corpus's ``source`` key.
SOURCES: dict[str, Callable[[int], tuple[nn.Module, object]]] = {
    "conv_bn_eval": _conv_bn_eval,
    "retnet_chunked": _retnet_chunked,
    "flash_attention": _flash_attention,
    "blelloch_scan": _blelloch_scan,
    "mul_unsq_nat": _mul_unsq_nat,
    "weight_chain": _weight_chain,
    "swiglu_mlp": _swiglu_mlp,
    "rms_norm_gained": _rms_norm_gained,
    "rms_norm_nogain": _rms_norm_nogain,
    "rms_norm_misspelled": _rms_norm_misspelled,
    "manual_softmax": _manual_softmax,
    "manual_silu": _manual_silu,
    "manual_softsign": _manual_softsign,
    "manual_glu": _manual_glu,
    "sdpa_nomask": _sdpa_nomask,
    "mha_qkv": _mha_qkv,
    "weight_merge": _weight_merge,
    "gqa_repeat_kv": _gqa_repeat_kv,
    "biased_chain": _biased_chain,
    "select_mul": _select_mul,
    "cumsum_scan": _cumsum_scan,
    "zoo_lora_adapter": _zoo_lora_adapter,
    "zoo_time_cond_conv": _zoo_time_cond_conv,
    "zoo_gau": _zoo_gau,
    "zoo_spline_kan": _zoo_spline_kan,
    "zoo_cosine_attention": _zoo_cosine_attention,
    "zoo_mos_head": _zoo_mos_head,
    "zoo_adaln_block": _zoo_adaln_block,
    "zoo_delta_rule": _zoo_delta_rule,
}
