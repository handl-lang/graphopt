"""P25 roofline arm — attack the unpack arithmetic, not the schedule.

The P24 record settled that `num_stages`/`pipeline(depth)` is flat: the
generated kernel is arithmetic-bound in the int4 unpack + `tl.sum`, not
latency-bound on the loads. This driver prototypes variants of the
emitted `_stage1` body on **synthetic standard-order operands** (the AWQ
checkpoint and vllm are not installed on this host) and measures each
against the torch dequant reference.

Operand layouts match the checkpoint convention the emitted kernel was
gated against (P22): ``qw`` int32 ``[K, N/8]`` standard-order nibbles
(``to_standard_order`` already applied), ``qz`` int32 ``[K/GS, N/8]``,
``sc`` fp16 ``[K/GS, N]``, ``x`` fp16 ``[K]``.

Correctness gate: cosine >= 0.9999 vs ``unpack_ref`` (the same denotation
the emitted term carries), checked *inside* the timing path — a kernel
that fails never gets timed.

    PYTHONPATH=<lib-kernel-ir>/src:<this dir> python roofline_arm.py
"""

from __future__ import annotations

import json
import statistics as stats
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/luis/GitHub/handl-lang/lib-kernel-ir/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from irtriton import cached_module  # noqa: E402
from lib_kernel_ir import qgemv  # noqa: E402

BW_GBPS = 89.1  # the measured roofline clock (kernel_v2.py)
GS = 128
OUT_JSON = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/p25_roofline_arm.json"
)

SHAPES = (
    ("down_proj", 896, 4864, 8),
    ("gate_proj", 4864, 896, 4),
)


# -- reference + timing (verbatim semantics of kernel_v2, minus the
# checkpoint/vllm deps this host lacks) -------------------------------


def unpack_ref(qw, qz, sc, gs=128):
    """Torch denotation of AWQ dequant on standard-order operands."""
    shifts = torch.arange(0, 32, 4, device=qw.device, dtype=torch.int32)
    q = ((qw.unsqueeze(-1) >> shifts) & 0xF).reshape(qw.shape[0], -1)
    z = ((qz.unsqueeze(-1) >> shifts) & 0xF).reshape(qz.shape[0], -1)
    z = z.repeat_interleave(gs, dim=0)
    s = sc.repeat_interleave(gs, dim=0)
    return ((q.float() - z.float()) * s.float()).t().contiguous()


def roofline_us(n, k, dtype_bytes=2):
    weight = k * n * 0.5
    act = k * dtype_bytes + n * dtype_bytes
    return (weight + act) / (BW_GBPS * 1e9) * 1e6


def timeit_graph(fn, calls: int = 20, replays: int = 5):
    """Pure GPU time per call via a captured graph (P21's referee)."""
    try:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / (replays * calls)
    except Exception:  # noqa: BLE001 - capture is best-effort
        return None


def timeit_sync(fn, reps=50, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return stats.median(ts) * 1e6


def synthesize(n, k, seed=0):
    """Standard-order AWQ operands with realistic value distributions."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    qw = torch.randint(
        -(2**31), 2**31, (k, n // 8), device="cuda", dtype=torch.int32,
        generator=g,
    )
    qz = torch.randint(
        0, 2**31, (k // GS, n // 8), device="cuda", dtype=torch.int32,
        generator=g,
    ).to(torch.int32)
    qz = qz & 0x77777777  # keep zeros in the AWQ 0..7-ish range
    sc = (
        torch.rand((k // GS, n), device="cuda", generator=g) * 0.01 + 0.005
    ).half()
    x = torch.randn(k, device="cuda", dtype=torch.float16, generator=g)
    return x, qw, qz, sc


# -- variant kernels ---------------------------------------------------
#
# All keep the emitted two-stage shape and the
# `run(x, qw, qz, sc, n, k, bn, bk, warps)` boundary. Stage 1 writes one
# fp32 partial row per k-slice; stage 2 sums them.


@triton.jit
def _stage1_baseline(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """The emitted body, verbatim (the 47%-of-roofline incumbent)."""
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
            mask=wmask, other=0, eviction_policy="evict_first",
        )
        q = ((words[:, :, None] >> shift[None, None, :]) & 0xF).reshape(
            BLOCK_K, BLOCK_N
        )
        zwords = tl.load(
            qz_ptr + g[:, None] * s_qzk + rw[None, :] * s_qzn,
            mask=wmask, other=0, eviction_policy="evict_first",
        )
        z = ((zwords[:, :, None] >> shift[None, None, :]) & 0xF).reshape(
            BLOCK_K, BLOCK_N
        )
        s = tl.load(
            sc_ptr + g[:, None] * s_sck + rn[None, :] * s_scn,
            mask=kmask[:, None] & nmask[None, :], other=0.0,
        ).to(tl.float32)
        w = (q.to(tl.float32) - z.to(tl.float32)) * s
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[:, None], axis=0)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_hoist(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Hoist z,s out of the k tile: `sum (q-z)s x = s*(sum qx) - s*z*sum x`.

    Requires every k-block to sit inside one quant group (true whenever
    BLOCK_K and GS are powers of two and k0 is a multiple of BLOCK_K).
    The z unpack and the s gather drop from O(BLOCK_K*BLOCK_N) per block
    to O(BLOCK_N); the inner loop keeps only shift/mask/convert/mul/sum.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        q = ((words[:, :, None] >> shift[None, None, :]) & 0xF).reshape(
            BLOCK_K, BLOCK_N
        )
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
        pqx = tl.sum(q.to(tl.float32) * x[:, None], axis=0)
        xs = tl.sum(x, axis=0)

        g = k0 // GS  # scalar: the whole block is one quant group
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw * s_qzn, mask=wmask_row, other=0,
        )
        z = ((zwords[:, None] >> shift[None, :]) & 0xF).reshape(
            BLOCK_N
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pqx - z * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_magic(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Hoisted + fp16 magic-number unpack.

    `(word >> shift & 0xF) | 0x6400` is `1024 + q` reinterpreted as fp16
    (one LOP3 after the shift, no int->float convert). Subtracting 1024
    gives q exactly in fp16; products stay fp16 and the k-reduction
    accumulates fp32. z gets the same treatment once per block.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        m = (
            ((words[:, :, None] >> shift[None, None, :]) & 0xF) | 0x6400
        ).reshape(BLOCK_K, BLOCK_N)
        q = m.to(tl.uint16).to(tl.float16, bitcast=True) - 1024.0
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        prod = (q * x[:, None]).to(tl.float32)
        pqx = tl.sum(prod, axis=0)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw * s_qzn, mask=wmask_row, other=0,
        )
        zm = ((zwords[:, None] >> shift[None, :]) & 0xF) | 0x6400
        z = (
            zm.reshape(BLOCK_N).to(tl.uint16).to(tl.float16, bitcast=True)
            - 1024.0
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pqx - z * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_magic_fold(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Magic unpack with the 1024 offset folded into the group correction.

    `sum (1024+q) x = sum qx + 1024 sum x`, so
    `sum (q-z) s x = s*(pmx - (1024+z)*xs)`. Skips the per-element fp16
    subtract; the fp16 product `m*x` carries a ~2^-11 relative rounding
    into a term that is then differenced — this is the variant whose
    cosine is at risk, and the gate decides.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        m = (
            ((words[:, :, None] >> shift[None, None, :]) & 0xF) | 0x6400
        ).reshape(BLOCK_K, BLOCK_N)
        m = m.to(tl.uint16).to(tl.float16, bitcast=True)
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        pmx = tl.sum((m * x[:, None]).to(tl.float32), axis=0)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw * s_qzn, mask=wmask_row, other=0,
        )
        zm = ((zwords[:, None] >> shift[None, :]) & 0xF) | 0x6400
        z = (
            zm.reshape(BLOCK_N).to(tl.uint16).to(tl.float16, bitcast=True)
            - 1024.0
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pmx - (1024.0 + z) * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_fma(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Magic unpack + fused (m*x - 1024*x) as ONE fp16 FMA.

    `m = 1024 + q`, so `q*x = fma(m, x, -1024*x)` — and an FMA rounds
    once, on the *result* `q*x` (~|q*x| <= 15|x|), not on the 1024-sized
    intermediate that killed `magic_fold`'s cosine. `-1024*x` is
    O(BLOCK_K) per block, so the per-element body is
    shift + LOP3 + trunc + HFMA (+ fp32 cvt/reduce).
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        m = (
            ((words[:, :, None] >> shift[None, None, :]) & 0xF) | 0x6400
        ).reshape(BLOCK_K, BLOCK_N)
        m = m.to(tl.uint16).to(tl.float16, bitcast=True)
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        nx = x * (-1024.0)
        prod = tl.fma(m, x[:, None], nx[:, None]).to(tl.float32)
        pqx = tl.sum(prod, axis=0)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw * s_qzn, mask=wmask_row, other=0,
        )
        zm = ((zwords[:, None] >> shift[None, :]) & 0xF) | 0x6400
        z = (
            zm.reshape(BLOCK_N).to(tl.uint16).to(tl.float16, bitcast=True)
            - 1024.0
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pqx - z * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_magic_h16(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Magic, but the k-reduction stays fp16 (no fp32 cvt per element)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        m = (
            ((words[:, :, None] >> shift[None, None, :]) & 0xF) | 0x6400
        ).reshape(BLOCK_K, BLOCK_N)
        q = m.to(tl.uint16).to(tl.float16, bitcast=True) - 1024.0
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        pqx = tl.sum(q * x[:, None], axis=0).to(tl.float32)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw * s_qzn, mask=wmask_row, other=0,
        )
        zm = ((zwords[:, None] >> shift[None, :]) & 0xF) | 0x6400
        z = (
            zm.reshape(BLOCK_N).to(tl.uint16).to(tl.float16, bitcast=True)
            - 1024.0
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pqx - z * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_i2f(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Hoisted, plain int->fp16 convert: is the magic constant needed?

    Per element: shift, and, I2F.F16 (vs magic's shift, LOP3, u16-trunc,
    fp16-sub). One fewer op if the convert pipe isn't the bottleneck.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        q = ((words[:, :, None] >> shift[None, None, :]) & 0xF).reshape(
            BLOCK_K, BLOCK_N
        ).to(tl.float16)
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        prod = (q * x[:, None]).to(tl.float32)
        pqx = tl.sum(prod, axis=0)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw * s_qzn, mask=wmask_row, other=0,
        )
        z = ((zwords[:, None] >> shift[None, :]) & 0xF).reshape(
            BLOCK_N
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pqx - z * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_magic_w64(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MAGIC: tl.constexpr,
):
    """Magic + 8-byte qw loads: each int64 yields two packed words.

    `qw_ptr` is the same tensor viewed as int64 [K, N/16] (strides in
    elements). Halves the load-instruction count on the dominant stream;
    the nibble arithmetic is unchanged. `MAGIC=0` swaps the magic
    unpack for a plain int->fp16 convert.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 16) + tl.arange(0, BLOCK_N // 16)
    nmask = rn < N
    wmask_row = rw < (N + 15) // 16

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        w64 = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        lo32 = w64.to(tl.uint64).to(tl.uint32)
        hi32 = (w64.to(tl.uint64) >> 32).to(tl.uint32)
        words = tl.interleave(
            lo32.to(tl.int32, bitcast=True), hi32.to(tl.int32, bitcast=True)
        )  # [BLOCK_K, BLOCK_N/8]
        nib = ((words[:, :, None] >> shift[None, None, :]) & 0xF).reshape(
            BLOCK_K, BLOCK_N
        )
        if MAGIC:
            q = ((nib | 0x6400).to(tl.uint16).to(tl.float16, bitcast=True)
                 - 1024.0)
        else:
            q = nib.to(tl.float16)
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        prod = (q * x[:, None]).to(tl.float32)
        pqx = tl.sum(prod, axis=0)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        rw32 = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw32 * s_qzn,
            mask=rw32 < (N + 7) // 8, other=0,
        )
        zm = ((zwords[:, None] >> shift[None, :]) & 0xF) | 0x6400
        z = (
            zm.reshape(BLOCK_N).to(tl.uint16).to(tl.float16, bitcast=True)
            - 1024.0
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pqx - z * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_magic_hint(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Magic + contiguity hints so the qw row vectorises to LDG.128."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    rw = tl.max_contiguous(tl.multiple_of(rw, 8), BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        m = (
            ((words[:, :, None] >> shift[None, None, :]) & 0xF) | 0x6400
        ).reshape(BLOCK_K, BLOCK_N)
        q = m.to(tl.uint16).to(tl.float16, bitcast=True) - 1024.0
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        prod = (q * x[:, None]).to(tl.float32)
        pqx = tl.sum(prod, axis=0)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw * s_qzn, mask=wmask_row, other=0,
        )
        zm = ((zwords[:, None] >> shift[None, :]) & 0xF) | 0x6400
        z = (
            zm.reshape(BLOCK_N).to(tl.uint16).to(tl.float16, bitcast=True)
            - 1024.0
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pqx - z * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


#: n-offset of interleaved pair element i inside a word: lo/hi halves of
#: `(w >> 4j) & 0x000F000F | 0x64006400` give nibbles j and j+4, so the
#: pair-extraction order is [0,4,1,5,2,6,3,7] — the AWQ pack order.
PAIR_PERM = (0, 4, 1, 5, 2, 6, 3, 7)


@triton.jit
def _stage1_pair(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Two fp16 per int32 op: the Marlin pair extraction.

    `(w >> 4j) & 0x000F000F | 0x64006400` packs `1024+q_j` and
    `1024+q_{j+4}` into one int32 (SHF + LOP3 for *two* values); one
    `mov.b32` unpacks the two fp16 halves. Columns come out in
    PAIR_PERM order, so the block's n-index vector is permuted — s, z
    and the store all use actual-n, only the tile layout changes.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    col = tl.arange(0, BLOCK_N)
    # actual-n index of each tile column. After interleave, column i of
    # each word holds nibble position p(i) = (i%8//2) + 4*(i%2):
    # [0,4,1,5,2,6,3,7] — the pair order.
    rn = pid_n * BLOCK_N + (col // 8) * 8 + (col % 8 // 2) + 4 * (col % 2)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift4 = tl.arange(0, 4) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        packed = (
            ((words[:, :, None] >> shift4[None, None, :]) & 0x000F000F)
            | 0x64006400
        )  # [BLOCK_K, NW, 4] int32, two fp16 each
        lo, hi = tl.inline_asm_elementwise(
            "mov.b32 {$0,$1}, $2;",
            "=h,=h,r",
            args=[packed],
            dtype=(tl.float16, tl.float16),
            is_pure=True,
            pack=1,
        )  # [BK, NW, 4] fp16 each
        m = tl.interleave(lo, hi).reshape(BLOCK_K, BLOCK_N)
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        nx = x * (-1024.0)
        prod = tl.fma(m, x[:, None], nx[:, None]).to(tl.float32)
        pqx = tl.sum(prod, axis=0)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        # zeros in the pair order: column c draws nibble position
        # zpos(c) of word wcol(c) — gather the word per column from
        # memory (O(BLOCK_N) per block, L1-resident).
        wcol = pid_n * (BLOCK_N // 8) + col // 8
        zw = tl.load(
            qz_ptr + g * s_qzk + wcol * s_qzn,
            mask=wcol < (N + 7) // 8, other=0,
        )
        zpos = (col % 8 // 2) + 4 * (col % 2)
        z = ((zw >> (zpos * 4)) & 0xF) | 0x6400
        z = (z.to(tl.uint16).to(tl.float16, bitcast=True) - 1024.0).to(
            tl.float32
        )
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pqx - z * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _stage1_dot(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr,
    N, K, GS, SPLIT,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn, s_pk, s_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tensor-core reduction: `tl.dot` on the *quantized* fp16 tile.

    Broadcast x into a [16, BLOCK_K] fp16 operand; `dot` produces
    [16, BLOCK_N] where every row is `sum q*x` (tensor cores multiply in
    fp32 and accumulate fp32, so the 1024 magic offset is exact — no
    per-element subtract needed). The dequant is still group-hoisted:
    `acc += s*(pmx - (1024+z)*xs)`.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    nmask = rn < N
    wmask_row = rw < (N + 7) // 8

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)

    shift = tl.arange(0, 8) * 4
    row0 = (tl.arange(0, 16) == 0).to(tl.float32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rw[None, :] * s_qwn,
            mask=kmask[:, None] & wmask_row[None, :],
            other=0, eviction_policy="evict_first",
        )
        m = (
            ((words[:, :, None] >> shift[None, None, :]) & 0xF) | 0x6400
        ).reshape(BLOCK_K, BLOCK_N)
        m = m.to(tl.uint16).to(tl.float16, bitcast=True)
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        a = tl.broadcast_to(x[None, :], (16, BLOCK_K))
        pmx16 = tl.dot(a, m)  # [16, BLOCK_N], fp32 acc — row 0 is sum m*x
        pmx = tl.sum(pmx16 * row0[:, None], axis=0)
        xs = tl.sum(x.to(tl.float32), axis=0)

        g = k0 // GS
        zwords = tl.load(
            qz_ptr + g * s_qzk + rw * s_qzn, mask=wmask_row, other=0,
        )
        zm = ((zwords[:, None] >> shift[None, :]) & 0xF) | 0x6400
        z = (
            zm.reshape(BLOCK_N).to(tl.uint16).to(tl.float16, bitcast=True)
            - 1024.0
        ).to(tl.float32)
        s = tl.load(
            sc_ptr + g * s_sck + rn * s_scn, mask=nmask, other=0.0
        ).to(tl.float32)
        acc += s * (pmx - (1024.0 + z) * xs)

    tl.store(part_ptr + pid_k * s_pk + rn * s_pn, acc, mask=nmask)


@triton.jit
def _reduce(part_ptr, out_ptr, N, SPLIT, s_pk, s_pn, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BLOCK + tl.arange(0, BLOCK)
    mask = rn < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in range(SPLIT):
        acc += tl.load(part_ptr + j * s_pk + rn * s_pn, mask=mask, other=0.0)
    tl.store(out_ptr + rn, acc.to(tl.float16), mask=mask)


_KERNELS = {
    "baseline": _stage1_baseline,
    "hoist": _stage1_hoist,
    "magic": _stage1_magic,
    "magic_fold": _stage1_magic_fold,
    "magic_h16": _stage1_magic_h16,
    "magic_hint": _stage1_magic_hint,
    "i2f": _stage1_i2f,
    "fma": _stage1_fma,
    "pair": _stage1_pair,
    "dot": _stage1_dot,
}


def make_run(kernel, split, bn=64, bk=128, warps=2):
    """A callable with the emitted module's `run` boundary."""

    def run(x, qw, qz, sc, n, k):
        assert bk <= GS, "hoisted variants need one group per k-block"
        parts = torch.empty((split, n), device="cuda", dtype=torch.float32)
        kernel[(triton.cdiv(n, bn), split)](
            x, qw, qz, sc, parts,
            n, k, GS, split,
            qw.stride(0), qw.stride(1),
            qz.stride(0), qz.stride(1),
            sc.stride(0), sc.stride(1),
            parts.stride(0), parts.stride(1),
            BLOCK_N=bn, BLOCK_K=bk, num_warps=warps,
        )
        out = torch.empty((1, n), device="cuda", dtype=torch.float16)
        _reduce[(triton.cdiv(n, 128),)](
            parts, out, n, split, parts.stride(0), parts.stride(1),
            BLOCK=128,
        )
        return out

    return run


def make_run_w64(split, bn=64, bk=128, warps=2, magic=1):
    """w64 arm: `qw` arrives int32 [K, N/8]; the kernel reads it as int64."""

    def run(x, qw, qz, sc, n, k):
        qw64 = qw.view(torch.int64)
        parts = torch.empty((split, n), device="cuda", dtype=torch.float32)
        _stage1_magic_w64[(triton.cdiv(n, bn), split)](
            x, qw64, qz, sc, parts,
            n, k, GS, split,
            qw64.stride(0), qw64.stride(1),
            qz.stride(0), qz.stride(1),
            sc.stride(0), sc.stride(1),
            parts.stride(0), parts.stride(1),
            BLOCK_N=bn, BLOCK_K=bk, MAGIC=magic, num_warps=warps,
        )
        out = torch.empty((1, n), device="cuda", dtype=torch.float16)
        _reduce[(triton.cdiv(n, 128),)](
            parts, out, n, split, parts.stride(0), parts.stride(1),
            BLOCK=128,
        )
        return out

    return run


def emitted_runner(label, k, split, bn=64, bk=128, warps=2):
    """The actual compile-triton output — the honest baseline."""
    mod = cached_module(qgemv(k=k, split=split), block_n=bn, block_k=bk)
    return lambda x, qw, qz, sc, n, kk: mod.run(
        x, qw, qz, sc, n, kk, bn, bk, warps
    )


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def main() -> int:
    torch.manual_seed(0)
    rows = []
    for label, n, k, split in SHAPES:
        x, qw, qz, sc = synthesize(n, k)
        w_ref = unpack_ref(qw, qz, sc, GS).half()
        ref = x.float() @ w_ref.float().t()
        roof = roofline_us(n, k)
        print(f"\n== {label} n={n} k={k} roof {roof:.1f} us ==")

        arms = {"emitted": emitted_runner(label, k, split)}
        for name, kern in _KERNELS.items():
            arms[name] = make_run(kern, split)
        arms["magic_w64"] = make_run_w64(split)
        # focused sweep on the `magic` body
        for bn, bk, wp in (
            (128, 64, 4), (128, 64, 2), (128, 32, 4), (64, 64, 4),
            (64, 64, 2), (64, 32, 2), (256, 64, 4), (128, 64, 8),
        ):
            arms[f"m_bn{bn}bk{bk}w{wp}"] = make_run(
                _stage1_magic, split, bn=bn, bk=bk, warps=wp
            )
        for sp in (4, 8, 16):
            arms[f"w64bn128bk64w4_s{sp}"] = make_run_w64(
                sp, bn=128, bk=64, warps=4
            )
            arms[f"w64bn64bk64w2_s{sp}"] = make_run_w64(
                sp, bn=64, bk=64, warps=2
            )
            arms[f"w64bn128bk128w4_s{sp}"] = make_run_w64(
                sp, bn=128, bk=128, warps=4
            )
        for bn, bk, wp in (
            (128, 128, 4), (64, 128, 2), (256, 64, 4), (128, 64, 8),
            (64, 64, 4),
        ):
            arms[f"w64bn{bn}bk{bk}w{wp}"] = make_run_w64(
                split, bn=bn, bk=bk, warps=wp
            )
        arms["w64i2f_bn128bk128w4"] = make_run_w64(
            split, bn=128, bk=128, warps=4, magic=0
        )
        arms["w64i2f_bn128bk64w4"] = make_run_w64(
            split, bn=128, bk=64, warps=4, magic=0
        )
        arms["i2f_bn128bk64w4"] = make_run(
            _stage1_i2f, split, bn=128, bk=64, warps=4
        )

        for name, fn in arms.items():
            try:
                got = fn(x, qw, qz, sc, n, k)
            except Exception as exc:  # noqa: BLE001
                print(f"{name:>14} FAILED to run: {exc}")
                rows.append(
                    {"shape": label, "variant": name, "error": str(exc)}
                )
                continue
            cos = cosine(got, ref)
            # verify the kernel actually ran, not a stale buffer
            nonzero = got.abs().sum().item() > 0
            if cos < 0.9999 or not nonzero:
                print(
                    f"{name:>14} REJECTED cos={cos:.6f} nonzero={nonzero}"
                )
                rows.append(
                    {
                        "shape": label, "variant": name,
                        "cosine": round(cos, 6), "rejected": True,
                    }
                )
                continue
            us = timeit_graph(lambda: fn(x, qw, qz, sc, n, k), calls=10,
                              replays=3)
            sus = timeit_sync(lambda: fn(x, qw, qz, sc, n, k),
                              reps=40, warm=8)
            eff = 100 * roof / us if us else None
            rows.append(
                {
                    "shape": label, "variant": name,
                    "graph_us": round(us, 2) if us else None,
                    "sync_us": round(sus, 2),
                    "pct_of_roofline": round(eff, 1) if eff else None,
                    "cosine": round(cos, 6),
                }
            )
            print(
                f"{name:>14} {us:7.2f} us graph ({eff:5.1f}% roof) "
                f"{sus:7.2f} us sync  cos={cos:.6f}"
            )

    OUT_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
