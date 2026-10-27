"""Split-k for the int4 GEMV: close the `gate_proj` loss to awq_gemm.

`awq_gemm` wins the large-output shapes because it splits the reduction
across programs. Same idea here: launch `SPLIT` programs along K, each
accumulating a partial dot product, combined with `atomic_add` into the
output. Then sweep split x block config per shape.
"""

from __future__ import annotations

import json
import sys

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from awq_fused import timeit, unpack_ref  # noqa: E402

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)


@triton.jit
def _gemv_splitk(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, out_ptr,
    N, K, GS,
    s_qwk, s_qwn, s_qzk, s_qzn, s_sck, s_scn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT: tl.constexpr,
):
    """One program does a K-slice; partials are atomically summed."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_p = rn // 8
    shift = (rn % 8) * 4
    nmask = rn < N

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
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
        acc += tl.sum(((q.to(tl.float32) - z.to(tl.float32)) * s)
                      * x[:, None], axis=0)

    tl.atomic_add(out_ptr + rn, acc, mask=nmask)


def splitk(x, qw, qz, sc, n, k, bn, bk, split, warps):
    out = torch.zeros((1, n), device="cuda", dtype=torch.float32)
    _gemv_splitk[(triton.cdiv(n, bn), split)](
        x, qw, qz, sc, out, n, k, 128,
        qw.stride(0), qw.stride(1),
        qz.stride(0), qz.stride(1),
        sc.stride(0), sc.stride(1),
        BLOCK_N=bn, BLOCK_K=bk, SPLIT=split, num_warps=warps,
    )
    return out


def main() -> int:
    import vllm._custom_ops as cops

    rows = []
    with safe_open(CKPT, "pt") as f:
        for pre, name in (
            ("model.layers.0.mlp.gate_proj", "gate_proj"),
            ("model.layers.0.self_attn.k_proj", "k_proj"),
            ("model.layers.0.mlp.down_proj", "down_proj"),
        ):
            qw = f.get_tensor(f"{pre}.qweight").cuda()
            qz = f.get_tensor(f"{pre}.qzeros").cuda()
            sc = f.get_tensor(f"{pre}.scales").cuda()
            n, k = sc.shape[1], qw.shape[0]
            w_ref = unpack_ref(qw, qz, sc)
            x = torch.randn(1, k, device="cuda")
            xf = x.reshape(-1).contiguous()
            ref = x @ w_ref.t()
            x16 = x.half()
            t_vllm = timeit(lambda: cops.awq_gemm(x16, qw, sc, qz, 1))

            best = None
            for bn in (16, 32, 64, 128):
                if bn > n:
                    continue
                for bk in (64, 128, 256):
                    for split in (1, 2, 4, 8):
                        for wp in (1, 2, 4, 8):
                            try:
                                got = splitk(xf, qw, qz, sc, n, k, bn, bk,
                                             split, wp)
                                err = float((ref - got).abs().max())
                                us = timeit(
                                    lambda: splitk(xf, qw, qz, sc, n, k,
                                                   bn, bk, split, wp),
                                    reps=30, warm=6,
                                )
                            except Exception:  # noqa: BLE001
                                continue
                            if best is None or us < best[0]:
                                best = (us, bn, bk, split, wp, err)
            us, bn, bk, split, wp, err = best
            rows.append(
                {
                    "layer": name, "in": k, "out": n,
                    "config": f"bn={bn} bk={bk} split={split} w={wp}",
                    "vllm_us": round(t_vllm, 1),
                    "ours_us": round(us, 1),
                    "vs_vllm": round(t_vllm / us, 3),
                    "max_abs_err": err,
                }
            )
            print(f"{name:>10} {n:>5}x{k:<5} split={split} bn={bn} bk={bk} "
                  f"w={wp}  ours {us:7.1f}  vllm {t_vllm:7.1f}  "
                  f"-> {t_vllm/us:.2f}x")

    path = (
        "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
        "p8c_splitk.json"
    )
    open(path, "w").write(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
