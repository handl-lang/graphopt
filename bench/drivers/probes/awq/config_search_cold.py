"""Cold-regime config search — the search the warm one should have been.

Every previous config sweep scored candidates in a warm cache, and the
cold regime ranks them differently (P12). This repeats the sweep with the
weights evicted before each call, so the winner is the one that wins in
deployment.

Candidates are scored by ``bench.benchkit.regime.blocked_paired`` with
``warm_per_block=0`` (cold, one call) and order shuffled per round.
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
    "p13_cold_config_search.json"
)
BW_GBPS = 89.1

GRID = [
    (bn, bk, split, wp)
    for bn in (32, 64, 128)
    for bk in (64, 128)
    for split in (2, 4, 8, 16)
    for wp in (2, 4)
]


def main() -> int:
    shapes = (
        ("down_proj", "model.layers.0.mlp.down_proj"),
        ("gate_proj", "model.layers.0.mlp.gate_proj"),
    )
    rows = []
    with safe_open(CKPT, "pt") as f:
        for name, pre in shapes:
            qw = f.get_tensor(f"{pre}.qweight").cuda()
            qz = f.get_tensor(f"{pre}.qzeros").cuda()
            sc = f.get_tensor(f"{pre}.scales").cuda()
            n, k = sc.shape[1], qw.shape[0]
            ref = x_ref = torch.randn(
                1, k, device="cuda", dtype=torch.float16
            )
            ref = x_ref.float() @ unpack_ref(qw, qz, sc).t()

            cands: dict[str, object] = {}
            for bn, bk, split, wp in GRID:
                if bn > n:
                    continue

                def make(bn=bn, bk=bk, split=split, wp=wp):
                    def run():
                        out = torch.empty(
                            (1, n), device="cuda", dtype=torch.float16
                        )
                        _gemv_v2[(triton.cdiv(n, bn), split)](
                            x_ref,
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

                    return run

                fn = make()
                try:
                    got = fn()
                except Exception:  # noqa: BLE001
                    continue
                if float((ref - got.float()).abs().max()) > 5e-2:
                    continue
                cands[f"bn{bn}_bk{bk}_s{split}_w{wp}"] = fn

            res = blocked_paired(
                cands,
                evict_passes=4,
                rounds=15,
                warm_per_block=0,
                inner_reps=1,
            )
            ranked = sorted(res.medians.items(), key=lambda kv: kv[1])
            best_name, best_us = ranked[0]
            roof = roofline_us(k * n * 0.5 + (k + n) * 2, BW_GBPS)
            rows.append(
                {
                    "layer": name,
                    "out": n,
                    "in": k,
                    "candidates": len(cands),
                    "cold_best": best_name,
                    "cold_us": round(best_us, 1),
                    "pct_of_roofline": round(100 * roof / best_us, 1),
                    "top5": [
                        {"config": c, "us": round(v, 1)}
                        for c, v in ranked[:5]
                    ],
                    "worst_us": round(ranked[-1][1], 1),
                }
            )
            OUT.write_text(json.dumps(rows, indent=2) + "\n")
            print(
                f"{name:>10} {n}x{k} cold-best {best_name} "
                f"{best_us:6.1f} us ({100 * roof / best_us:.0f}% roof) "
                f"of {len(cands)} cands",
                flush=True,
            )
            for c, v in ranked[:5]:
                print(f"    {c:>22} {v:7.1f}", flush=True)

    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
