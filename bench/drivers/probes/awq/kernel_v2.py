"""Kernel v2 — close the roofline gap.

v1 losses and their fixes here:

* each packed word loaded 8x (once per output feature) -> load the word
  once and unpack all 8 nibbles from it,
* 32-bit scalar loads -> contiguous word blocks, so Triton vectorises,
* fp32 in / fp32 out -> fp16 in / fp16 out with fp32 accumulate (halves
  the write traffic and removes the per-call dtype copies that made the
  end-to-end test flat),
* split-k everywhere -> split only when K is large.

Every result is reported as a fraction of the measured 89 GB/s roofline.
"""

from __future__ import annotations

import json
import statistics as stats
import time

import torch
import triton
import triton.language as tl
from safetensors import safe_open

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)
BW_GBPS = 89.1


@triton.jit
def _gemv_v2(
    x_ptr,
    qw_ptr,
    qz_ptr,
    sc_ptr,
    out_ptr,
    N,
    K,
    GS,
    s_qwk,
    s_qwn,
    s_qzk,
    s_qzn,
    s_sck,
    s_scn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """One word, eight outputs: unpack words in registers."""
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
        )
        q = ((words[:, :, None] >> shift[None, None, :]) & 0xF).reshape(
            BLOCK_K, BLOCK_N
        )
        zwords = tl.load(
            qz_ptr + g[:, None] * s_qzk + rw[None, :] * s_qzn,
            mask=wmask,
            other=0,
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

    tl.store(out_ptr + rn, acc.to(tl.float16), mask=nmask)


def gemv_v2(x, qw, qz, sc, n, k, bn=128, bk=64, split=1, warps=4):
    out = torch.empty((1, n), device="cuda", dtype=torch.float16)
    _gemv_v2[(triton.cdiv(n, bn), split)](
        x,
        qw,
        qz,
        sc,
        out,
        n,
        k,
        128,
        qw.stride(0),
        qw.stride(1),
        qz.stride(0),
        qz.stride(1),
        sc.stride(0),
        sc.stride(1),
        BLOCK_N=bn,
        BLOCK_K=bk,
        SPLIT=split,
        num_warps=warps,
    )
    return out


def unpack_ref(qw, qz, sc, gs=128):
    shifts = torch.arange(0, 32, 4, device=qw.device, dtype=torch.int32)
    q = ((qw.unsqueeze(-1) >> shifts) & 0xF).reshape(qw.shape[0], -1)
    z = ((qz.unsqueeze(-1) >> shifts) & 0xF).reshape(qz.shape[0], -1)
    z = z.repeat_interleave(gs, dim=0)
    s = sc.repeat_interleave(gs, dim=0)
    return ((q.float() - z.float()) * s.float()).t().contiguous()


def timeit(fn, reps=80, warm=15) -> float:
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


def roofline_us(n, k, dtype_bytes=2):
    """Weights + activation + output, at the measured bandwidth."""
    weight = k * n * 0.5
    act = k * dtype_bytes + n * dtype_bytes
    return (weight + act) / (BW_GBPS * 1e9) * 1e6


def main() -> int:
    import vllm._custom_ops as cops

    rows = []
    with safe_open(CKPT, "pt") as f:
        for pre, name in (
            ("model.layers.0.mlp.down_proj", "down_proj"),
            ("model.layers.0.self_attn.q_proj", "q_proj"),
            ("model.layers.0.self_attn.k_proj", "k_proj"),
            ("model.layers.0.mlp.gate_proj", "gate_proj"),
        ):
            qw = f.get_tensor(f"{pre}.qweight").cuda()
            qz = f.get_tensor(f"{pre}.qzeros").cuda()
            sc = f.get_tensor(f"{pre}.scales").cuda()
            n, k = sc.shape[1], qw.shape[0]
            w_ref = unpack_ref(qw, qz, sc).half()
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)
            ref = x.float() @ w_ref.float().t()
            roof = roofline_us(n, k)

            best = None
            for bn in (32, 64, 128, 256):
                if bn > n:
                    continue
                for bk in (64, 128, 256):
                    for split in (1, 2, 4):
                        for wp in (2, 4, 8):
                            try:
                                got = gemv_v2(
                                    x,
                                    qw,
                                    qz,
                                    sc,
                                    n,
                                    k,
                                    bn,
                                    bk,
                                    split,
                                    wp,
                                )
                                err = float(
                                    (ref - got.float()).abs().max()
                                )
                                us = timeit(
                                    lambda: gemv_v2(
                                        x,
                                        qw,
                                        qz,
                                        sc,
                                        n,
                                        k,
                                        bn,
                                        bk,
                                        split,
                                        wp,
                                    ),
                                    reps=40,
                                    warm=8,
                                )
                            except Exception:
                                continue
                            if best is None or us < best[0]:
                                best = (us, bn, bk, split, wp, err)

            us, bn, bk, split, wp, err = best
            t_vllm = timeit(lambda: cops.awq_gemm(x, qw, sc, qz, 1))
            eff = 100 * roof / us
            rows.append(
                {
                    "layer": name,
                    "out": n,
                    "in": k,
                    "config": f"bn={bn} bk={bk} split={split} w={wp}",
                    "roofline_us": round(roof, 1),
                    "ours_us": round(us, 1),
                    "vllm_us": round(t_vllm, 1),
                    "pct_of_roofline": round(eff, 1),
                    "vs_vllm": round(t_vllm / us, 2),
                    "max_abs_err": err,
                }
            )
            print(
                f"{name:>10} {n:>5}x{k:<5} roof {roof:6.1f}  "
                f"ours {us:6.1f} ({eff:4.1f}% roof)  "
                f"vllm {t_vllm:6.1f}  -> {t_vllm / us:.2f}x"
            )

    path = (
        "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
        "p9_kernel_v2.json"
    )
    open(path, "w").write(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
