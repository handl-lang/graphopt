"""All 168 quantized linears of Qwen2.5-0.5B-AWQ: win/loss distribution.

For each unique geometry, the machine sweeps its own kernel configs
(BLOCK_N x BLOCK_K x warps) and picks the best by measurement — the
choice is not mine. Then three arms are compared at one token:

    torch  — float weight pre-built, plain matmul
    ours   — packed int4 unpacked in-kernel (best config found)
    vllm   — awq_gemm, the hand-tuned production kernel
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import torch
import triton
from safetensors import safe_open

sys.path.insert(0, "/tmp")
from awq_fused import _awq_gemv, timeit, unpack_ref  # noqa: E402

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)

GRID = [
    (bn, bk, w)
    for bn in (16, 32, 64, 128)
    for bk in (64, 128, 256)
    for w in (1, 2, 4, 8)
]


def run_cfg(x, qw, qz, sc, n, k, bn, bk, warps):
    out = torch.empty((1, n), device="cuda", dtype=torch.float32)
    _awq_gemv[(triton.cdiv(n, bn),)](
        x, qw, qz, sc, out, n, k, 128,
        qw.stride(0), qw.stride(1),
        qz.stride(0), qz.stride(1),
        sc.stride(0), sc.stride(1),
        BLOCK_N=bn, BLOCK_K=bk, num_warps=warps,
    )
    return out


def main() -> int:
    import vllm._custom_ops as cops

    # one representative tensor set per geometry, plus how many layers use it
    wanted = {
        (4864, 896): ("model.layers.0.mlp.down_proj", "down_proj"),
        (896, 4864): ("model.layers.0.mlp.gate_proj", "gate_proj"),
        (896, 128): ("model.layers.0.self_attn.k_proj", "k_proj"),
        (896, 896): ("model.layers.0.self_attn.q_proj", "q_proj"),
    }
    counts = Counter()
    rows = []
    with safe_open(CKPT, "pt") as f:
        for key in f.keys():
            if not key.endswith(".qweight"):
                continue
            sh = f.get_slice(key).get_shape()
            counts[(sh[0], sh[1] * 8)] += 1

        for (k, n), (pre, name) in wanted.items():
            qw = f.get_tensor(f"{pre}.qweight").cuda()
            qz = f.get_tensor(f"{pre}.qzeros").cuda()
            sc = f.get_tensor(f"{pre}.scales").cuda()
            w_ref = unpack_ref(qw, qz, sc)
            x = torch.randn(1, k, device="cuda")
            xf = x.reshape(-1).contiguous()
            ref = x @ w_ref.t()

            best = None
            for bn, bk, wp in GRID:
                if bn > n:
                    continue
                try:
                    got = run_cfg(xf, qw, qz, sc, n, k, bn, bk, wp)
                    err = float((ref - got).abs().max())
                    us = timeit(
                        lambda: run_cfg(xf, qw, qz, sc, n, k, bn, bk, wp),
                        reps=40, warm=8,
                    )
                except Exception:  # noqa: BLE001
                    continue
                if best is None or us < best[0]:
                    best = (us, bn, bk, wp, err)

            t_torch = timeit(lambda: x @ w_ref.t(), reps=60, warm=10)
            x16 = x.half()
            t_vllm = timeit(
                lambda: cops.awq_gemm(x16, qw, sc, qz, 1), reps=60, warm=10
            )

            us, bn, bk, wp, err = best
            layers = counts[(k, n)]
            rows.append(
                {
                    "layer": name, "in": k, "out": n, "layers": layers,
                    "config": f"bn={bn} bk={bk} w={wp}",
                    "torch_us": round(t_torch, 1),
                    "ours_us": round(us, 1),
                    "vllm_us": round(t_vllm, 1),
                    "vs_torch": round(t_torch / us, 3),
                    "vs_vllm": round(t_vllm / us, 3),
                    "max_abs_err": err,
                }
            )
            print(f"{name:>10} {n:>5}x{k:<5} x{layers:<3} best[{bn},{bk},{wp}] "
                  f"torch {t_torch:7.1f}  ours {us:7.1f} "
                  f"({t_torch/us:5.2f}x)  vllm {t_vllm:7.1f} "
                  f"({t_vllm/us:5.2f}x)")

    wins_v = sum(r["layers"] for r in rows if r["vs_vllm"] >= 1.0)
    wins_t = sum(r["layers"] for r in rows if r["vs_torch"] >= 1.0)
    total = sum(r["layers"] for r in rows)
    print(f"\nlayers beating torch : {wins_t}/{total}")
    print(f"layers beating awq_gemm: {wins_v}/{total}")

    path = (
        "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
        "p8b_awq_sweep.json"
    )
    open(path, "w").write(
        json.dumps(
            {"rows": rows, "beat_torch": wins_t, "beat_vllm": wins_v,
             "total": total},
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
