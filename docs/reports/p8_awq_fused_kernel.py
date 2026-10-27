"""Unpack int4 inside the kernel, and race vLLM's awq_gemm.

Three arms on the same real AWQ layer:

1. **torch**  — unpack to float, then matmul (the portable path),
2. **ours**   — one Triton kernel that reads packed int4, unpacks in
   registers, multiplies, never builds a float weight,
3. **vllm**   — `awq_gemm`, the hand-tuned 4-bit kernel, the real
   competitor.

Decode (M=1) is where the earlier win lived, so it gets the GEMV path.
"""

from __future__ import annotations

import json
import statistics as stats
import sys
import time

import torch
import triton
import triton.language as tl
from safetensors import safe_open

sys.path.insert(0, "/tmp")

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)


@triton.jit
def _awq_gemv(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, out_ptr,
    N, K, GS,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y[n] = sum_k x[k] * (q[k,n] - z[k//g,n]) * s[k//g,n], int4 unpacked here."""
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_p = rn // 8
    shift = (rn % 8) * 4
    nmask = rn < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        g = rk // GS
        mask = kmask[:, None] & nmask[None, :]
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0)
        w_word = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rn_p[None, :] * s_qwn,
            mask=mask, other=0,
        )
        q = (w_word >> shift[None, :]) & 0xF
        z_word = tl.load(
            qz_ptr + g[:, None] * s_qzk + rn_p[None, :] * s_qzn,
            mask=mask, other=0,
        )
        z = (z_word >> shift[None, :]) & 0xF
        s = tl.load(
            sc_ptr + g[:, None] * s_sck + rn[None, :] * s_scn,
            mask=mask, other=0.0,
        ).to(tl.float32)
        w = (q.to(tl.float32) - z.to(tl.float32)) * s
        acc += tl.sum(w * x[:, None], axis=0)

    tl.store(out_ptr + rn, acc, mask=nmask)


def ours_gemv(x, qw, qz, sc, gs=128, block_n=32, block_k=64):
    k, _ = qw.shape
    n = sc.shape[1]
    out = torch.empty((1, n), device=x.device, dtype=torch.float32)
    _awq_gemv[(triton.cdiv(n, block_n),)](
        x.reshape(-1).contiguous(), qw, qz, sc, out,
        n, k, gs,
        qw.stride(0), qw.stride(1),
        qz.stride(0), qz.stride(1),
        sc.stride(0), sc.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4,
    )
    return out


def unpack_ref(qw, qz, sc, gs=128):
    """Float weight from packed int4 — what torch-based stacks build."""
    shifts = torch.arange(0, 32, 4, device=qw.device, dtype=torch.int32)
    q = ((qw.unsqueeze(-1) >> shifts) & 0xF).reshape(qw.shape[0], -1)
    z = ((qz.unsqueeze(-1) >> shifts) & 0xF).reshape(qz.shape[0], -1)
    z = z.repeat_interleave(gs, dim=0)
    s = sc.repeat_interleave(gs, dim=0)
    return ((q.float() - z.float()) * s.float()).t().contiguous()


def timeit(fn, reps=100, warm=20) -> float:
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
    import vllm._custom_ops as cops

    out = {"ckpt": "Qwen2.5-0.5B-AWQ", "rows": []}
    with safe_open(CKPT, "pt") as f:
        for layer in ("self_attn.q_proj", "mlp.down_proj"):
            pre = f"model.layers.0.{layer}"
            qw = f.get_tensor(f"{pre}.qweight").cuda()
            qz = f.get_tensor(f"{pre}.qzeros").cuda()
            sc = f.get_tensor(f"{pre}.scales").cuda()
            n, k = sc.shape[1], qw.shape[0]
            w_ref = unpack_ref(qw, qz, sc)          # [out, in] float

            x32 = torch.randn(1, k, device="cuda")
            ref = x32 @ w_ref.t()
            got = ours_gemv(x32, qw, qz, sc)
            err = float((ref - got).abs().max())

            t_torch = timeit(lambda: x32 @ w_ref.t())
            t_ours = timeit(lambda: ours_gemv(x32, qw, qz, sc))

            x16 = x32.half()
            t_vllm = None
            try:
                cops.awq_gemm(x16, qw, sc, qz, 1)
                t_vllm = timeit(
                    lambda: cops.awq_gemm(x16, qw, sc, qz, 1)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  awq_gemm unavailable: {type(exc).__name__}")

            row = {
                "layer": layer, "out": n, "in": k,
                "torch_us": round(t_torch, 1),
                "ours_us": round(t_ours, 1),
                "vllm_us": None if t_vllm is None else round(t_vllm, 1),
                "vs_torch": round(t_torch / t_ours, 3),
                "vs_vllm": (
                    None if t_vllm is None else round(t_vllm / t_ours, 3)
                ),
                "max_abs_err": err,
            }
            out["rows"].append(row)
            v = "n/a" if t_vllm is None else f"{t_vllm:7.1f}"
            vs = "n/a" if t_vllm is None else f"{t_vllm/t_ours:.2f}x"
            print(f"{layer:>18} {n}x{k}  torch {t_torch:7.1f}  "
                  f"ours {t_ours:7.1f} ({t_torch/t_ours:.2f}x)  "
                  f"vllm {v} ({vs})  err {err:.1e}")

    path = (
        "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
        "p8_awq_fused.json"
    )
    open(path, "w").write(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
