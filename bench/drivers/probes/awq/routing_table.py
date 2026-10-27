"""Routing table under the deploy regime — emitted, not typed.

Two regimes, because they disagree and only one is deployment:

* **cold** — weights evicted, one call timed. This is a transformer
  forward: a layer's 2.18 MB of weights cannot survive a 2 MB L2 across
  the 24 layers that run before it is used again. **This decides the
  route.**
* **reuse** — the same kernel called again immediately (a benchmark
  loop, not a model). Reported for contrast: it flatters every arm.

Arms: ``ours`` (int4 GEMV), ``vllm`` (``awq_gemm``), ``torch`` (float
weight pre-built, plain matmul). A route that only wins warm is not a
route, so the winner is the lowest cold median and disagreement between
regimes is flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import triton
from safetensors import safe_open

from bench.benchkit.regime import blocked_paired, roofline_us

from .kernel_v2 import _gemv_v2, unpack_ref

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)
OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p12_routing_table.json"
)
BW_GBPS = 89.1
CONFIG = {
    (4864, 896): (64, 128, 4, 2),
    (896, 896): (32, 128, 4, 2),
    (896, 4864): (64, 64, 4, 2),
    (896, 128): (32, 256, 4, 4),
}


def main() -> int:
    import vllm._custom_ops as cops

    shapes = (
        ("down_proj", "model.layers.0.mlp.down_proj"),
        ("q_proj", "model.layers.0.self_attn.q_proj"),
        ("gate_proj", "model.layers.0.mlp.gate_proj"),
        ("k_proj", "model.layers.0.self_attn.k_proj"),
    )
    rows = []
    with safe_open(CKPT, "pt") as f:
        for name, pre in shapes:
            qw = f.get_tensor(f"{pre}.qweight").cuda()
            qz = f.get_tensor(f"{pre}.qzeros").cuda()
            sc = f.get_tensor(f"{pre}.scales").cuda()
            n, k = sc.shape[1], qw.shape[0]
            w_ref = unpack_ref(qw, qz, sc).half()
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)
            bn, bk, split, wp = CONFIG[(k, n)]

            def ours():
                out = torch.empty(
                    (1, n), device="cuda", dtype=torch.float16
                )
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
                    num_warps=wp,
                )
                return out

            arms = {
                "ours": ours,
                "vllm": lambda: cops.awq_gemm(x, qw, sc, qz, 1),
                "torch": lambda: x @ w_ref.t(),
            }
            cold = blocked_paired(
                arms,
                evict_passes=4,
                rounds=25,
                warm_per_block=0,
                inner_reps=1,
            )
            reuse = blocked_paired(
                arms,
                evict_passes=4,
                rounds=25,
                warm_per_block=2,
                inner_reps=4,
            )
            w_cold = min(cold.medians, key=cold.medians.get)
            w_reuse = min(reuse.medians, key=reuse.medians.get)
            roof = roofline_us(k * n * 0.5 + (k + n) * 2, BW_GBPS)
            rows.append(
                {
                    "layer": name,
                    "out": n,
                    "in": k,
                    "cold_us": {
                        a: round(v, 1) for a, v in cold.medians.items()
                    },
                    "reuse_us": {
                        a: round(v, 1) for a, v in reuse.medians.items()
                    },
                    "cold_sd_us": {
                        a: round(v, 1) for a, v in cold.stdevs.items()
                    },
                    "winner_cold": w_cold,
                    "winner_reuse": w_reuse,
                    "regimes_agree": w_cold == w_reuse,
                    "roofline_us": round(roof, 1),
                    "our_pct_of_roofline_cold": round(
                        100 * roof / cold.medians["ours"], 1
                    ),
                    "vllm_over_ours_cold": round(
                        cold.medians["vllm"] / cold.medians["ours"], 2
                    ),
                }
            )
            OUT.write_text(json.dumps(rows, indent=2) + "\n")
            print(
                f"{name:>10} {n:>5}x{k:<5} "
                f"COLD ours {cold.medians['ours']:6.1f} "
                f"vllm {cold.medians['vllm']:6.1f} "
                f"torch {cold.medians['torch']:6.1f} -> {w_cold:>5s} | "
                f"reuse ours {reuse.medians['ours']:6.1f} "
                f"vllm {reuse.medians['vllm']:6.1f} -> {w_reuse}",
                flush=True,
            )

    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
