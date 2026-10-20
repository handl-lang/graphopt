"""Known-good reference implementations — the registered floor.

Each entry in :data:`REFS` is a builder ``(model, feed) -> nn.Module``
producing the module a practitioner or a real compiler would ship —
the hand-folded conv, ``F.scaled_dot_product_attention``, the
chunked-recurrence form from the RetNet paper.  The scorecard's
*beat-best-registered* flag compares the engine's certified form
against the best of these, so the floor is code, not a number copied
from a results doc.

Larger ports (a whole FLA/kernel-level reference) grow into
``refs/<name>/`` subpackages; single-module impls live here.
"""

# ruff: noqa: RUF002, RUF003 — math notation (·, γ, √, −) in
# docstrings is deliberate, per bench convention.

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class _FoldedConv(nn.Module):
    """The eval-mode conv+BN fold, done by hand (PyTorch's own recipe).

    ``W' = W · γ/√(var+eps)`` per output channel,
    ``b' = (b − μ)·γ/√(var+eps) + β`` — one conv, no BN at runtime.
    """

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        w = conv.weight.detach().clone()
        scale = bn.weight.detach() / torch.sqrt(
            bn.running_var.detach() + bn.eps
        )
        w = w * scale.reshape(-1, 1, 1, 1)
        b = (conv.bias.detach() - bn.running_mean.detach()) * scale
        b = b + bn.bias.detach()
        self.conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
        )
        with torch.no_grad():
            self.conv.weight.copy_(w)
            self.conv.bias.copy_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _KnownSDPA(nn.Module):
    """``F.scaled_dot_product_attention`` — the fused kernel, spelled."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.q = src.q
        self.k = src.k
        self.v = src.v
        self.d = src.d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            self.q(x), self.k(x), self.v(x), is_causal=True
        )


class _ChunkedRetentionRef(nn.Module):
    """The paper's chunkwise-recurrent retention (Sun et al. 2023).

    Intra-chunk ``(QKᵀ ⊙ D)V`` plus a decayed cross-chunk state
    recurrence — the known-good chunked impl the carrier is measured
    against (fp64; no stability rescale, same as the probe's
    ``paper-eqn`` arm in ``bench/probes/retnet_baseline_probe.py``).
    """

    def __init__(self, src: nn.Module, chunk: int = 16) -> None:
        super().__init__()
        self.q = src.q
        self.k = src.k
        self.v = src.v
        self.gamma = src.gamma
        self.chunk = chunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        c = self.chunk
        n = t // c
        pos = torch.arange(c, dtype=x.dtype, device=x.device)
        g = torch.tensor(self.gamma, dtype=x.dtype, device=x.device)
        decay = torch.tril(
            g ** (pos[:, None] - pos[None, :]).clamp_min(0)
        )
        q_dec = g ** (pos + 1)  # decay of the cross-chunk read
        v_dec = g ** (c - 1 - pos)  # decay folded into the state
        q = (self.q(x) / math.sqrt(t)).view(b, n, c, d)
        k = self.k(x).view(b, n, c, d)
        v = self.v(x).view(b, n, c, d)
        inner = ((q @ k.transpose(-1, -2)) * decay) @ v
        kv_chunk = k.transpose(-1, -2) @ (
            v * v_dec[None, None, :, None]
        )
        state = torch.zeros(b, d, d, dtype=x.dtype, device=x.device)
        cross = torch.zeros_like(inner)
        for i in range(n):
            cross[:, i] = (q[:, i] * q_dec[None, :, None]) @ state
            state = state * g**c + kv_chunk[:, i]
        return (inner + cross).reshape(b, t, d)


class _PrefixScanRef(nn.Module):
    """Blelloch-style parallel prefix for ``h_t = a·h_{t-1} + x_t``.

    The closed form ``h_t = P_t · cumsum(x_t / P_t)`` with
    ``P_t = a^t`` — the cumulative-product factoring every
    divide-and-conquer scan compiles to (fp64 keeps it stable at the
    suite's sizes).
    """

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.a = src.a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        steps = torch.arange(1, t + 1, dtype=x.dtype, device=x.device)
        p = self.a.view(1, 1, -1) ** steps.view(1, t, 1)  # (1, t, d)
        return (x / p).cumsum(dim=1)[:, -1] * p[0, -1]


class _FusedChain(nn.Module):
    """The weight-folded chain: one ``Linear`` carrying ``W2 @ W1``."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        d = src[0].weight.shape[1]
        self.lin = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(
                src[1].weight.detach() @ src[0].weight.detach()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _FusedSwiGLU(nn.Module):
    """The widened-GEMM SwiGLU: one ``(2d, d)`` projection + split."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        d = src.a.weight.shape[1]
        self.d = d
        self.w = nn.Linear(d, 2 * d, bias=False)
        with torch.no_grad():
            self.w.weight.copy_(
                torch.cat(
                    [src.a.weight.detach(), src.b.weight.detach()],
                    dim=0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w(x)
        return F.silu(y[..., : self.d]) * y[..., self.d :]


class _UnsqFolded(nn.Module):
    """``(a * b).unsqueeze(-1)`` — multiply first, lift the view once."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).unsqueeze(-1)


class _KnownRMSNorm(nn.Module):
    """``F.rms_norm`` — the fused kernel, carrying the spelled gain."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.w = src.w
        self.eps = src.eps
        self.d = src.w.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.d,), self.w, self.eps)


class _KnownRMSNormNoGain(nn.Module):
    """``F.rms_norm`` with ``weight=None`` — the gain-free kernel."""

    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.d = d
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.d,), weight=None, eps=self.eps)


class _KnownSoftmax(nn.Module):
    """``torch.softmax`` — the fused normalization kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)


class _KnownSiLU(nn.Module):
    """``F.silu`` — the gated activation as one kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class _KnownSoftsign(nn.Module):
    """``F.softsign`` — the bounded rational activation as one kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softsign(x)


class _KnownGLU(nn.Module):
    """``F.glu`` — ``a ⊗ σ(b)`` over the halves, as one kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(x, dim=-1)


class _KnownSDPANoMask(nn.Module):
    """``F.scaled_dot_product_attention``, no mask, explicit scale."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.q = src.q
        self.k = src.k
        self.v = src.v
        self.d = src.d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            self.q(x), self.k(x), self.v(x), scale=1 / math.sqrt(self.d)
        )


class _MergedQKV(nn.Module):
    """The fused-QKV form: one ``(3d, d)`` GEMM, three free splits."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.d = src.d
        self.h = src.h
        self.qkv = nn.Linear(src.d, 3 * src.d, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(
                torch.cat(
                    [
                        src.q.weight.detach(),
                        src.k.weight.detach(),
                        src.v.weight.detach(),
                    ],
                    dim=0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def head(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, self.h, d // self.h).transpose(1, 2)

        return F.scaled_dot_product_attention(head(q), head(k), head(v))


class _KnownGQA(nn.Module):
    """``F.scaled_dot_product_attention`` with ``enable_gqa=True``.

    The kernel-side head broadcast — same projections, no
    materialised repeat chain.
    """

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.h, self.hk, self.dh = src.h, src.hk, src.dh
        self.wq, self.wk, self.wv = src.wq, src.wk, src.wv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(b, t, self.hk, self.dh).transpose(1, 2)
        v = self.wv(x).view(b, t, self.hk, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )
        return o.transpose(1, 2).reshape(b, t, -1)


class _MergedWeights(nn.Module):
    """``linear(x, A + B)`` — the two projections merged weights-first."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        d = src.a.weight.shape[1]
        self.lin = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(
                src.a.weight.detach() + src.b.weight.detach()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _FoldedAffineChain(nn.Module):
    """The affine compose folded by hand: ``W = B·A``, ``b = B·b1+b2``."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        a, b = src[0], src[1]
        d = a.weight.shape[1]
        self.lin = nn.Linear(d, d)
        with torch.no_grad():
            self.lin.weight.copy_(b.weight.detach() @ a.weight.detach())
            self.lin.bias.copy_(
                b.weight.detach() @ a.bias.detach() + b.bias.detach()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _SelectMulFolded(nn.Module):
    """``(a ⊙ b).select(d, i)`` — product first, select once."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).select(1, 0)


def _conv_bn_folded(model: nn.Module, feed: object) -> nn.Module:
    return _FoldedConv(model[0], model[1]).eval().double()


def _sdpa_known(model: nn.Module, feed: object) -> nn.Module:
    return _KnownSDPA(model).eval().double()


def _retnet_chunked(model: nn.Module, feed: object) -> nn.Module:
    return _ChunkedRetentionRef(model).eval().double()


def _blelloch_prefix(model: nn.Module, feed: object) -> nn.Module:
    return _PrefixScanRef(model).eval().double()


def _weight_chain_fused(model: nn.Module, feed: object) -> nn.Module:
    return _FusedChain(model).eval().double()


def _swiglu_fused(model: nn.Module, feed: object) -> nn.Module:
    return _FusedSwiGLU(model).eval().double()


def _mul_unsq_folded(model: nn.Module, feed: object) -> nn.Module:
    return _UnsqFolded().eval().double()


def _rms_norm_known(model: nn.Module, feed: object) -> nn.Module:
    return _KnownRMSNorm(model).eval().double()


def _rms_norm_nogain_known(model: nn.Module, feed: object) -> nn.Module:
    return (
        _KnownRMSNormNoGain(feed.shape[-1], model.eps).eval().double()
    )


def _softmax_known(model: nn.Module, feed: object) -> nn.Module:
    return _KnownSoftmax().eval().double()


def _silu_known(model: nn.Module, feed: object) -> nn.Module:
    return _KnownSiLU().eval().double()


def _softsign_known(model: nn.Module, feed: object) -> nn.Module:
    return _KnownSoftsign().eval().double()


def _glu_known(model: nn.Module, feed: object) -> nn.Module:
    return _KnownGLU().eval().double()


def _sdpa_nomask_known(model: nn.Module, feed: object) -> nn.Module:
    return _KnownSDPANoMask(model).eval().double()


def _qkv_merged(model: nn.Module, feed: object) -> nn.Module:
    return _MergedQKV(model).eval().double()


def _gqa_enable_gqa(model: nn.Module, feed: object) -> nn.Module:
    return _KnownGQA(model).eval().double()


def _weight_merged(model: nn.Module, feed: object) -> nn.Module:
    return _MergedWeights(model).eval().double()


def _affine_chain_folded(model: nn.Module, feed: object) -> nn.Module:
    return _FoldedAffineChain(model).eval().double()


def _select_mul_folded(model: nn.Module, feed: object) -> nn.Module:
    return _SelectMulFolded().eval().double()


#: ``name -> builder(model, feed)`` — the case corpus's ``ref`` key.
REFS: dict[str, Callable[[nn.Module, object], nn.Module]] = {
    "conv_bn_folded": _conv_bn_folded,
    "sdpa_known": _sdpa_known,
    "retnet_chunked": _retnet_chunked,
    "blelloch_prefix": _blelloch_prefix,
    "weight_chain_fused": _weight_chain_fused,
    "swiglu_fused": _swiglu_fused,
    "mul_unsq_folded": _mul_unsq_folded,
    "rms_norm_known": _rms_norm_known,
    "rms_norm_nogain_known": _rms_norm_nogain_known,
    "softmax_known": _softmax_known,
    "silu_known": _silu_known,
    "softsign_known": _softsign_known,
    "glu_known": _glu_known,
    "sdpa_nomask_known": _sdpa_nomask_known,
    "qkv_merged": _qkv_merged,
    "gqa_enable_gqa": _gqa_enable_gqa,
    "weight_merged": _weight_merged,
    "affine_chain_folded": _affine_chain_folded,
    "select_mul_folded": _select_mul_folded,
}
