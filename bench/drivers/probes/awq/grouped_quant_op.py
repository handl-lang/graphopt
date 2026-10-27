"""GroupedQuantLinear as a drop-in op — and where it actually wins.

The zoo module dequantizes a per-group-quantized weight and does a
linear map:

    w = w_q * scale[..., None]        # (N, K), group grid over K
    y = x @ w.T + bias

Eager materializes `w` (an N×K float tensor) then calls addmm.
`torch.compile` fuses the dequant but still pays dispatch overhead.
This ships the whole thing as **one Triton kernel behind a
`torch.library` op**, so any stack can call it.

The interesting question is not "does it win at d=64" (the zoo probe
shape, where everything is overhead) but **where it wins** as the weight
grows: the fused kernel never materializes `w`, so its advantage should
grow with N×K.
"""

from __future__ import annotations

import json
import statistics as stats
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _dq_mm_kernel(
    x_ptr, wq_ptr, sc_ptr, b_ptr, out_ptr,
    M, N, K, GS,
    s_xm, s_xk, s_wn, s_wk, s_sn, s_sg, s_om, s_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr = False,
):
    """y = (x @ (w_q * scale).T) + bias, dequantized in registers."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        kmask = kk < K
        x = tl.load(
            x_ptr + rm[:, None] * s_xm + kk[None, :] * s_xk,
            mask=(rm[:, None] < M) & kmask[None, :], other=0.0,
        )
        wq = tl.load(
            wq_ptr + rn[:, None] * s_wn + kk[None, :] * s_wk,
            mask=(rn[:, None] < N) & kmask[None, :], other=0.0,
        )
        sc = tl.load(
            sc_ptr + rn[:, None] * s_sn + (kk // GS)[None, :] * s_sg,
            mask=(rn[:, None] < N) & kmask[None, :], other=0.0,
        )
        w = tl.trans(wq * sc)
        if IEEE:
            acc += tl.dot(x, w, input_precision='ieee')
        else:
            acc += tl.dot(x, w, input_precision='tf32')

    acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0)[None, :]
    tl.store(
        out_ptr + rm[:, None] * s_om + rn[None, :] * s_on,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )



@triton.jit
def _dq_gemv(
    x_ptr, wq_ptr, sc_ptr, b_ptr, out_ptr,
    N, K, GS, s_wn, s_wk, s_sn, s_sg,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """M=1 decode: one program per output-row block, no tensor cores."""
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        kmask = kk < K
        x = tl.load(x_ptr + kk, mask=kmask, other=0.0)
        wq = tl.load(
            wq_ptr + rn[:, None] * s_wn + kk[None, :] * s_wk,
            mask=(rn[:, None] < N) & kmask[None, :], other=0.0,
        )
        sc = tl.load(
            sc_ptr + rn[:, None] * s_sn + (kk // GS)[None, :] * s_sg,
            mask=(rn[:, None] < N) & kmask[None, :], other=0.0,
        )
        acc += tl.sum(wq * sc * x[None, :], axis=1)
    acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    tl.store(out_ptr + rn, acc, mask=rn < N)


def _gemv(x, w_q, scale, bias, ieee=True):
    M, K = x.shape
    N = w_q.shape[0]
    gs = K // scale.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    _dq_gemv[(triton.cdiv(N, 32),)](
        x, w_q, scale, bias, out, N, K, gs,
        w_q.stride(0), w_q.stride(1),
        scale.stride(0), scale.stride(1),
        BLOCK_N=32, BLOCK_K=64, num_warps=4,
    )
    return out


def grouped_quant_linear(x, w_q, scale, bias, ieee=False) -> torch.Tensor:
    """One fused launch: dequant + matmul + bias."""
    M, K = x.shape
    if M == 1:
        return _gemv(x, w_q, scale, bias, ieee)
    N = w_q.shape[0]
    gs = K // scale.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _dq_mm_kernel[grid](
        x, w_q, scale, bias, out,
        M, N, K, gs,
        x.stride(0), x.stride(1),
        w_q.stride(0), w_q.stride(1),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, IEEE=ieee,
    )
    return out


@torch.library.custom_op("mlopt::grouped_quant_linear", mutates_args=())
def grouped_quant_linear_op(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Grouped-quant dequant linear, fused into one kernel."""
    return grouped_quant_linear(x, w_q, scale, bias)


@torch.library.custom_op("mlopt::grouped_quant_linear_ieee", mutates_args=())
def grouped_quant_linear_ieee_op(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Same op, full fp32 accumulate (no TF32)."""
    return grouped_quant_linear(x, w_q, scale, bias, ieee=True)


@grouped_quant_linear_ieee_op.register_fake
def _(x, w_q, scale, bias):
    return x.new_empty((x.shape[0], w_q.shape[0]))


@grouped_quant_linear_op.register_fake
def _(x, w_q, scale, bias):
    return x.new_empty((x.shape[0], w_q.shape[0]))


def make_case(d: int, groups: int, tokens: int, seed: int = 0):
    """The zoo module's tensors: int-valued w_q, per-group scales, bias."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(d, d, generator=g).reshape(d, groups, d // groups)
    s = (w.abs().amax(-1) / 7.0).clamp_min(1e-3)
    w_q = torch.round(w / s.unsqueeze(-1)).reshape(d, d)
    scale = s.reshape(d, groups)
    bias = torch.zeros(d)
    x = torch.randn(tokens, d, generator=g)
    to = lambda t: t.cuda().float()  # noqa: E731
    return to(x), to(w_q), to(scale), to(bias)


def eager(x, w_q, scale, bias):
    """The spelled module: dequantize, reshape, linear."""
    groups = scale.shape[1]
    gs = w_q.shape[1] // groups
    w = (w_q.reshape(w_q.shape[0], groups, gs)
         * scale.unsqueeze(-1)).reshape(w_q.shape)
    return x @ w.transpose(0, 1) + bias


def timeit(fn, reps: int = 100, warm: int = 20) -> float:
    """Median wall time in µs, CUDA-synced per call."""
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


def main() -> int:
    rows = []
    print(f"{'d':>5} {'tok':>4} {'eager':>9} {'compile':>9} {'ours':>8} "
          f"{'vs eager':>9} {'vs compile':>11} {'max|Δ|':>9}")
    for d, groups in ((64, 4), (128, 4), (256, 8), (768, 8), (2048, 8)):
        x, w_q, scale, bias = make_case(d, groups, tokens=32)
        ref = eager(x, w_q, scale, bias)
        got = grouped_quant_linear_op(x, w_q, scale, bias)
        err = float((ref - got).abs().max())

        def eager_fn():
            return eager(x, w_q, scale, bias)

        def ours_fn():
            return grouped_quant_linear_op(x, w_q, scale, bias)

        comp_fn = torch.compile(eager_fn, dynamic=False)
        comp_fn()  # compile outside the timed region

        got_ieee = grouped_quant_linear_ieee_op(x, w_q, scale, bias)
        err_ieee = float((ref - got_ieee).abs().max())

        def ieee_fn():
            return grouped_quant_linear_ieee_op(x, w_q, scale, bias)

        e, c, o = timeit(eager_fn), timeit(comp_fn), timeit(ours_fn)
        oi = timeit(ieee_fn)
        rows.append(
            {
                "d": d, "groups": groups, "tokens": 32,
                "eager_us": round(e, 2), "compile_us": round(c, 2),
                "ours_us": round(o, 2),
                "vs_eager": round(e / o, 3),
                "vs_compile": round(c / o, 3),
                "max_abs_err": err,
                "ieee_us": round(oi, 2),
                "ieee_max_abs_err": err_ieee,
                "ieee_vs_compile": round(c / oi, 3),
            }
        )
        print(f"{d:>5} {32:>4} {e:>9.1f} {c:>9.1f} {o:>8.1f} "
              f"{e/o:>8.2f}x {c/o:>10.2f}x {err:>9.1e}  "
              f"ieee {oi:>7.1f}µs {c/oi:>6.2f}x {err_ieee:>8.1e}")

    out = Path(
        "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
        "p6_grouped_quant_op.json"
    )
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

