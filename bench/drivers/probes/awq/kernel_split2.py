"""Two-stage split reduction — correct, deterministic, no atomics.

The split-k path in `kernel_v2` used `tl.atomic_add` into an fp16 buffer
and produced errors of 3.8–5.6 (see `p13_retraction_splitk.md`). Atomics
on half-precision output are the suspect: the partial sums either collide
or lose precision.

This replaces them with the standard two-stage form: each K-slice writes
its own row of a `[SPLIT, N]` fp32 buffer with a plain store, then a tiny
reduce kernel sums the slices. Deterministic, exact, and one extra launch
that costs microseconds against a 2.18 MB weight read.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stage1(
    x_ptr,
    qw_ptr,
    qz_ptr,
    sc_ptr,
    part_ptr,
    N,
    K,
    GS,
    SPLIT,
    s_qwk,
    s_qwn,
    s_qzk,
    s_qzn,
    s_sck,
    s_scn,
    s_pk,
    s_pn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per (output block, K slice); stores its partial."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        g = rk // GS
        wmask = kmask[:, None] & (rw < (N + 7) // 8)[None, :]
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=wmask,
            other=0,
            eviction_policy="evict_first",
        )
        q = ((words[:, :, None] >> shift[None, None, :]) & 0xF).reshape(
            BLOCK_K, BLOCK_N
        )
        zwords = tl.load(
            qz_ptr + g[:, None] * s_qzk + rw[None, :] * s_qzn,
            mask=wmask,
            other=0,
            eviction_policy="evict_first",
        )
        z = (
            (zwords[:, :, None] >> shift[None, None, :]) & 0xF
        ).reshape(BLOCK_K, BLOCK_N)
        s = tl.load(
            sc_ptr + g[:, None] * s_sck + rn[None, :] * s_scn,
            mask=kmask[:, None] & nmask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = (q.to(tl.float32) - z.to(tl.float32)) * s
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[:, None], axis=0)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _reduce(
    part_ptr, out_ptr, N, SPLIT, s_pk, s_pn, BLOCK: tl.constexpr
):
    """Sum the SPLIT partial rows into the fp16 output."""
    pid = tl.program_id(0)
    rn = pid * BLOCK + tl.arange(0, BLOCK)
    mask = rn < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in range(SPLIT):
        acc += tl.load(
            part_ptr + j * s_pk + rn * s_pn, mask=mask, other=0.0
        )
    tl.store(out_ptr + rn, acc.to(tl.float16), mask=mask)


def gemv_split2(
    x,
    qw,
    qz,
    sc,
    n,
    k,
    bn=64,
    bk=128,
    split=4,
    warps=2,
    group: int = 128,
    out=None,
):
    """Two-stage fused dequant GEMV; exact and deterministic."""
    parts = torch.empty((split, n), device="cuda", dtype=torch.float32)
    _stage1[(triton.cdiv(n, bn), split)](
        x,
        qw,
        qz,
        sc,
        parts,
        n,
        k,
        group,
        split,
        qw.stride(0),
        qw.stride(1),
        qz.stride(0),
        qz.stride(1),
        sc.stride(0),
        sc.stride(1),
        parts.stride(0),
        parts.stride(1),
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
    )
    if out is None:
        out = torch.empty((1, n), device="cuda", dtype=torch.float16)
    _reduce[(triton.cdiv(n, 128),)](
        parts,
        out,
        n,
        split,
        parts.stride(0),
        parts.stride(1),
        BLOCK=128,
    )
    return out
