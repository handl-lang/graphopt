"""``triton`` — hand-written Triton kernels, when the stack exists.

The field baseline for kernel cases: a Triton JIT kernel spelling
the fused op the way a kernel engineer writes it.  Requires
``triton`` + CUDA; declines honestly via :class:`SkipBaseline` on
hosts without either (and on cases with no registered kernel).
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


def _triton_silu():
    """A hand Triton ``a·σ(b)`` kernel — the kf_silu practitioner's arm."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _k(A, B, OUT, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        a = tl.load(A + off, mask=m)
        b = tl.load(B + off, mask=m)
        tl.store(OUT + off, a * tl.sigmoid(b), mask=m)

    def run(a, b):
        out = torch.empty_like(a)
        n = a.numel()
        grid = (triton.cdiv(n, 1024),)
        _k[grid](a, b, out, n, BLOCK=1024)
        return out

    return run


_KERNEL_BUILDERS = {"kernel_silu": _triton_silu}


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
