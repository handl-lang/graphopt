"""Overnight: the deploy-regime benchmark.

Every previous number was measured in a warm cache, and the weights of the
big layers (2.18 MB) do not survive a 2 MB L2 between uses. Deployment is
the **cold** regime, so the ranking we have may be wrong.

Part A — regime curve: each arm at increasing eviction pressure (0, 1, 4,
16 passes over a 64 MB buffer). Quantifies how much of every speedup was
cache.

Part B — the true ranking at deployment-like pressure (4 passes),
blocked-randomized so drift cancels.

Written incrementally: a crash keeps the partial table. Clocks and
temperature are logged with every block, because a lock that does not
hold is itself a result.
"""

from __future__ import annotations

import json
import statistics as stats
import subprocess
import time
from pathlib import Path

import torch
from safetensors import safe_open

from .kernel_v2 import _gemv_v2, roofline_us, unpack_ref

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)
OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p11_deploy_regime.json"
)

CONFIG = {
    (4864, 896): (64, 128, 4, 2),
    (896, 896): (32, 128, 4, 2),
    (896, 4864): (64, 64, 4, 2),
    (896, 128): (32, 256, 4, 4),
}
EVICT_MB = 64
EVICT_LEVELS = (0, 1, 4, 16)
ROUNDS = 30


def clocks() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        sm, temp, pwr = [x.strip() for x in out.split(",")]
        return {
            "sm_mhz": int(float(sm)),
            "temp_c": int(float(temp)),
            "power_w": float(pwr),
        }
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    import triton
    import vllm._custom_ops as cops

    evict = torch.empty(
        EVICT_MB * 1024 * 1024 // 4, dtype=torch.float32, device="cuda"
    )

    def flush(passes: int) -> None:
        for _ in range(passes):
            evict.mul_(1.0000001)

    shapes = (
        ("down_proj", "model.layers.0.mlp.down_proj"),
        ("q_proj", "model.layers.0.self_attn.q_proj"),
        ("gate_proj", "model.layers.0.mlp.gate_proj"),
        ("k_proj", "model.layers.0.self_attn.k_proj"),
    )
    result: dict = {"part_a": [], "part_b": [], "env": []}

    def save() -> None:
        OUT.write_text(json.dumps(result, indent=2) + "\n")

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
            for fn in arms.values():
                for _ in range(3):
                    fn()
            torch.cuda.synchronize()

            roof = roofline_us(n, k)
            for lvl in EVICT_LEVELS:
                buckets = {a: [] for a in arms}
                for _ in range(ROUNDS):
                    for a in ("ours", "vllm", "torch"):
                        flush(lvl)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        arms[a]()
                        torch.cuda.synchronize()
                        buckets[a].append(
                            (time.perf_counter() - t0) * 1e6
                        )
                for a in ("ours", "vllm", "torch"):
                    us = stats.median(buckets[a])
                    result["part_a"].append(
                        {
                            "layer": name,
                            "out": n,
                            "in": k,
                            "arm": a,
                            "evict_passes": lvl,
                            "median_us": round(us, 1),
                            "sd_us": round(stats.stdev(buckets[a]), 1),
                            "pct_of_roofline": round(
                                100 * roof / us, 1
                            ),
                        }
                    )
                result["env"].append(
                    {"layer": name, "evict": lvl, **clocks()}
                )
                save()
                o = stats.median(buckets["ours"])
                v = stats.median(buckets["vllm"])
                t = stats.median(buckets["torch"])
                print(
                    f"{name:>10} evict={lvl:<3} ours {o:7.1f} "
                    f"vllm {v:7.1f} torch {t:7.1f}  ours/vllm "
                    f"{v / o:5.2f}x",
                    flush=True,
                )

            bb = {a: [] for a in arms}
            for _ in range(ROUNDS):
                for a in ("ours", "vllm", "torch"):
                    flush(4)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    arms[a]()
                    torch.cuda.synchronize()
                    bb[a].append((time.perf_counter() - t0) * 1e6)
            mo, mv, mt = (
                stats.median(bb[a]) for a in ("ours", "vllm", "torch")
            )
            result["part_b"].append(
                {
                    "layer": name,
                    "out": n,
                    "in": k,
                    "ours_us": round(mo, 1),
                    "vllm_us": round(mv, 1),
                    "torch_us": round(mt, 1),
                    "vs_vllm": round(mv / mo, 2),
                    "vs_torch": round(mt / mo, 2),
                    "wins_vs_both": bool(mo < mv and mo < mt),
                }
            )
            save()
            print(
                f"  DEPLOY {name}: ours {mo:.1f} vllm {mv:.1f} "
                f"torch {mt:.1f} -> {mv / mo:.2f}x vs vllm",
                flush=True,
            )

    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
