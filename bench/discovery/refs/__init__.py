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


#: ``name -> builder(model, feed)`` — the case corpus's ``ref`` key.
REFS: dict[str, Callable[[nn.Module, object], nn.Module]] = {
    "conv_bn_folded": _conv_bn_folded,
    "sdpa_known": _sdpa_known,
    "retnet_chunked": _retnet_chunked,
    "blelloch_prefix": _blelloch_prefix,
    "weight_chain_fused": _weight_chain_fused,
    "swiglu_fused": _swiglu_fused,
    "mul_unsq_folded": _mul_unsq_folded,
}
