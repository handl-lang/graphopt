"""``triton`` — hand-written Triton kernels, when the stack exists.

The field baseline for kernel cases: a Triton JIT kernel spelling
the fused op the way a kernel engineer writes it.  Requires
``triton`` + CUDA; declines honestly via :class:`SkipBaseline` on
hosts without either (and on cases with no registered kernel).

Registered kernels cover the ``bench.discovery.kernel`` case
shapes: the elementwise gate fusions (silu, softsign, FMA, swiglu,
the mixed gate, the strength pair), the row reductions (softmax,
layernorm — one program per row, the tutorial idiom) and the tiled
matmul epilogues (bias, scale+bias).  Each builder returns a
callable ``fn(*feeds)`` matching the case's Var order
(``_vars_of`` DFS).
"""

from __future__ import annotations

import importlib.util
from typing import Any

from bench.limits.registry import (
    BaselineCtx,
    BaselineSpec,
    SkipBaseline,
)

SPEC = BaselineSpec(
    name="triton",
    kind="kernel",
    title="hand-written Triton kernel",
    requires=("triton",),
    needs_cuda=True,
    symbolic=False,
    note="kernel cases only — no symbolic form",
    record="kernelopt spike baseline (plan 0032 phase 3)",
)


def _pointwise_scope():
    """Return ``(tl, triton, wrap)`` for a 1-D elementwise kernel.

    The pointwise arms share their launch boilerplate (grid over
    ``numel``, masked tail) — *wrap* turns the caller's nested
    ``@triton.jit`` kernel into the ``fn(*feeds)`` callable (nested
    jit compiles fine under triton 3.x).
    """
    import torch
    import triton
    import triton.language as tl

    def wrap(kern):
        def run(*ts):
            out = torch.empty_like(ts[0])
            n = ts[0].numel()
            kern[(triton.cdiv(n, 1024),)](*ts, out, n, BLOCK=1024)
            return out

        return run

    return tl, triton, wrap


def _triton_silu():
    """A hand Triton ``x·σ(x)`` — the kf_silu practitioner's arm."""
    tl, triton, wrap = _pointwise_scope()

    @triton.jit
    def _k(A, OUT, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(A + off, mask=m)
        tl.store(OUT + off, a * tl.sigmoid(a), mask=m)

    return wrap(_k)


def _triton_softsign():
    """A hand Triton ``x/(|x|+1)`` — kf_softsign's a≡b, c≡1."""
    tl, triton, wrap = _pointwise_scope()

    @triton.jit
    def _k(A, OUT, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(A + off, mask=m)
        tl.store(OUT + off, a / (tl.abs(a) + 1.0), mask=m)

    return wrap(_k)


def _triton_fma():
    """A hand Triton ``a·b + tanh(b)`` — the *whole* expression fused."""
    tl, triton, wrap = _pointwise_scope()
    import triton.language.extra.libdevice as libdevice

    @triton.jit
    def _k(A, B, OUT, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(A + off, mask=m)
        b = tl.load(B + off, mask=m)
        tl.store(OUT + off, a * b + libdevice.tanh(b), mask=m)

    return wrap(_k)


def _triton_swiglu():
    """A hand Triton ``(a·σ(a))·b`` — kf_swiglu's composite."""
    tl, triton, wrap = _pointwise_scope()

    @triton.jit
    def _k(A, B, OUT, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(A + off, mask=m)
        b = tl.load(B + off, mask=m)
        tl.store(OUT + off, (a * tl.sigmoid(a)) * b, mask=m)

    return wrap(_k)


def _triton_strength():
    """A hand Triton ``a·a + b·0.25`` — the strength-reduced pair."""
    tl, triton, wrap = _pointwise_scope()

    @triton.jit
    def _k(A, B, OUT, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(A + off, mask=m)
        b = tl.load(B + off, mask=m)
        tl.store(OUT + off, a * a + b * 0.25, mask=m)

    return wrap(_k)


def _triton_mixed():
    """A hand Triton ``(a·σ(a))·σ(b)`` — the mixed-gate composite."""
    tl, triton, wrap = _pointwise_scope()

    @triton.jit
    def _k(A, B, OUT, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(A + off, mask=m)
        b = tl.load(B + off, mask=m)
        tl.store(OUT + off, (a * tl.sigmoid(a)) * tl.sigmoid(b), mask=m)

    return wrap(_k)


def _triton_softmax(*, cap: bool = False):
    """A hand Triton row-softmax — the max-shifted practitioner's form.

    ``cap=True`` applies ``tanh`` to the loaded row first — the
    soft-capped-logit spelling the ``kernel_softmax_mm`` case carries
    (softmax over a bounded score, so the spelled reference stays
    finite in fp32).
    """
    import torch
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as libdevice

    @triton.jit
    def _k(X, OUT, ncols, CAP: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < ncols
        x = tl.load(X + row * ncols + cols, mask=m, other=0.0)
        if CAP:
            x = libdevice.tanh(x)
        e = tl.exp(x - tl.max(x, 0))
        tl.store(OUT + row * ncols + cols, e / tl.sum(e, 0), mask=m)

    def run(x):
        rows, ncols = x.shape
        out = torch.empty_like(x)
        _k[(rows,)](
            x, out, ncols, CAP=cap,
            BLOCK=triton.next_power_of_2(ncols),
        )
        return out

    return run


def _triton_layernorm():
    """A hand Triton row-layernorm — mean/var fold, eps baked at 1e-5."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _k(X, OUT, ncols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < ncols
        x = tl.load(X + row * ncols + cols, mask=m, other=0.0)
        mu = tl.sum(x, 0) / ncols
        c = x - mu
        var = tl.sum(c * c, 0) / ncols
        tl.store(
            OUT + row * ncols + cols,
            c * tl.rsqrt(var + 1e-5),
            mask=m,
        )

    def run(x):
        rows, ncols = x.shape
        out = torch.empty_like(x)
        _k[(rows,)](
            x, out, ncols, BLOCK=triton.next_power_of_2(ncols)
        )
        return out

    return run


def _triton_mm(
    scale: float | None, *, bias: bool, gelu: bool = False
):
    """A tiled Triton matmul with the absorbed epilogue.

    The tutorial idiom: ``BM×BN`` tiles accumulate over K; the fused
    kernel multiplies by *scale* (when the case carries one), adds
    the row-bias (when *bias*) and applies the erf-gelu (when
    *gelu*) in the epilogue where the spelled program launched extra
    pointwise kernels.
    """
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _k(A, W, B, OUT, M, N, K,
           SC: tl.constexpr, HAS_B: tl.constexpr, GELU: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(
                A + rm[:, None] * K + (k0 + rk)[None, :],
                mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K),
                other=0.0,
            )
            w = tl.load(
                W + (k0 + rk)[:, None] * N + rn[None, :],
                mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, w, input_precision="ieee")
        acc = acc * SC
        if HAS_B:
            b = tl.load(B + rn, mask=rn < N, other=0.0)
            acc = acc + b[None, :]
        if GELU:
            acc = (
                0.5
                * acc
                * (1.0 + tl.math.erf(acc * 0.7071067811865476))
            )
        tl.store(
            OUT + rm[:, None] * N + rn[None, :],
            acc,
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )

    # ``SC`` is baked as a constexpr — the case's literal scale
    # (1.0 when the fused op is plain ``mm + b``).
    sc = 1.0 if scale is None else float(scale)

    def run(*feeds):
        a, w = feeds[0], feeds[1]
        b = feeds[2] if bias else a  # dead pointer when HAS_B=False
        m, k = a.shape
        n = w.shape[1]
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(m, 128), triton.cdiv(n, 64))
        _k[grid](
            a, w, b, out, m, n, k,
            BM=128, BN=64, BK=32, SC=sc, HAS_B=bias, GELU=gelu,
        )
        return out

    return run


def _triton_mm_bias():
    """The mm+bias arm — the addmm a template kernel absorbs."""
    return _triton_mm(None, bias=True)


def _triton_mm_scale_bias():
    """The mm·s+b arm — the attention-scale epilogue spelling."""
    return _triton_mm(0.5, bias=True)


def _triton_mm_scale():
    """The mm·s arm — the bias-free attention-scale epilogue."""
    return _triton_mm(0.125, bias=False)


def _triton_mm_gelu():
    """The mm→gelu arm — the MLP activation in the mm epilogue."""
    return _triton_mm(None, bias=False, gelu=True)


def _triton_rmsnorm():
    """A hand Triton row-rmsnorm — mean-of-squares, eps baked 1e-6."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _k(X, OUT, ncols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < ncols
        x = tl.load(X + row * ncols + cols, mask=m, other=0.0)
        ms = tl.sum(x * x, 0) / ncols
        tl.store(
            OUT + row * ncols + cols,
            x * tl.rsqrt(ms + 1e-6),
            mask=m,
        )

    def run(x):
        rows, ncols = x.shape
        out = torch.empty_like(x)
        _k[(rows,)](
            x, out, ncols, BLOCK=triton.next_power_of_2(ncols)
        )
        return out

    return run


def _triton_softmax_mm():
    """The honest pipeline: tiled mm → capped row-softmax → tiled mm.

    The practitioner's non-flash ``softmax(tanh(qk))·v``: three hand
    launches composed, where the case's fused composite emits two
    (spelled ``tanh(qk)`` + ``kf_softmax_mm``).  Each stage is the
    same hand kernel the single-op rows time.
    """
    mm = _triton_mm(None, bias=False)
    sm = _triton_softmax(cap=True)

    def run(x, w):
        return mm(sm(mm(x, w)), w)

    return run


def _triton_sdpa():
    """The honest decomposed-attention pipeline: mm·s → softmax → mm."""
    mm = _triton_mm(None, bias=False)
    mm_scale = _triton_mm(0.25, bias=False)
    sm = _triton_softmax()

    def run(x, w):
        return mm(sm(mm_scale(x, w)), w)

    return run


_KERNEL_BUILDERS = {
    "kernel_silu": _triton_silu,
    "kernel_softsign": _triton_softsign,
    "kernel_fma": _triton_fma,
    "kernel_swiglu": _triton_swiglu,
    "kernel_strength": _triton_strength,
    "kernel_mixed": _triton_mixed,
    "kernel_softmax": _triton_softmax,
    "kernel_layernorm": _triton_layernorm,
    "kernel_rmsnorm": _triton_rmsnorm,
    "kernel_mm_bias": _triton_mm_bias,
    "kernel_mm_scale_bias": _triton_mm_scale_bias,
    "kernel_mm_scale": _triton_mm_scale,
    "kernel_mm_gelu": _triton_mm_gelu,
    "kernel_softmax_mm": _triton_softmax_mm,
    "kernel_sdpa": _triton_sdpa,
}


def kernel_builder(name: str) -> Any:
    """Return the registered kernel builder for case *name* (or None)."""
    return _KERNEL_BUILDERS.get(name)


def build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Any:
    """Return the hand Triton arm for the case, or decline honestly."""
    del spec
    if importlib.util.find_spec("triton") is None:
        raise SkipBaseline("triton not installed")
    import torch

    if not torch.cuda.is_available():
        raise SkipBaseline("needs CUDA")
    mk = _KERNEL_BUILDERS.get(getattr(ctx.case, "name", ""))
    if mk is None:
        raise SkipBaseline(
            f"no triton kernel registered for {ctx.case.name!r}"
        )
    return mk()
