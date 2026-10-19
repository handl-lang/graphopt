"""Spelled "before" fixtures — the corpus's input programs.

Each entry in :data:`SOURCES` is a builder ``size -> (nn.Module,
feed)`` spelling the program the way a practitioner writes it —
*before* the named optimization.  Builders run under the caller's
seed; modules are ``.eval().double()`` so the sink's fp64
equivalence gate applies unchanged.

``feed`` is a tensor or a positional-args tuple, matching the
``TorchSource`` / ``sink.verify`` convention.
"""

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


#: ``name -> builder(size)`` for the case corpus's ``source`` key.
SOURCES: dict[str, Callable[[int], tuple[nn.Module, object]]] = {
    "conv_bn_eval": _conv_bn_eval,
    "retnet_chunked": _retnet_chunked,
    "flash_attention": _flash_attention,
    "blelloch_scan": _blelloch_scan,
    "mul_unsq_nat": _mul_unsq_nat,
    "weight_chain": _weight_chain,
    "swiglu_mlp": _swiglu_mlp,
}
