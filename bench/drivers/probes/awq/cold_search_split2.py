"""Cold-regime search over the *correct* split kernel.

`kernel_split2` replaces the broken atomics with a two-stage reduction and
is exact to 1e-3 for every split (verified in `p13`). This sweeps its
configs under the deploy regime and races `awq_gemm` and plain matmul.

Every candidate is exactness-gated immediately before it is timed: a
kernel that computes the wrong answer cannot win a speed comparison
again.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from bench.benchkit.regime import blocked_paired, roofline_us

from .kernel_split2 import gemv_split2
from .kernel_v2 import unpack_ref

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)
OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p14_correct_splitk.json"
)
BW_GBPS = 89.1
TOL = 5e-2

GRID = [
    (bn, bk, split, wp)
    for bn in (32, 64, 128)
    for bk in (64, 128)
    for split in (1, 2, 4, 8)
    for wp in (2, 4)
]


def main() -> int:
    import vllm._custom_ops as cops

    shapes = (
        ("down_proj", "model.layers.0.mlp.down_proj"),
        ("gate_proj", "model.layers.0.mlp.gate_proj"),
        ("q_proj", "model.layers.0.self_attn.q_proj"),
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
            ref = x.float() @ w_ref.float().t()

            cands: dict[str, object] = {}
            for bn, bk, split, wp in GRID:
                if bn > n:
                    continue

                def make(bn=bn, bk=bk, split=split, wp=wp):
                    def run():
                        return gemv_split2(
                            x,
                            qw,
                            qz,
                            sc,
                            n,
                            k,
                            bn=bn,
                            bk=bk,
                            split=split,
                            warps=wp,
                        )

                    return run

                fn = make()
                try:
                    err = float((ref - fn().float()).abs().max())
                except Exception:  # noqa: BLE001
                    continue
                if err > TOL:  # exactness gate, per candidate
                    continue
                cands[f"bn{bn}_bk{bk}_s{split}_w{wp}"] = fn

            res = blocked_paired(
                cands,
                evict_passes=4,
                rounds=15,
                warm_per_block=0,
                inner_reps=1,
            )
            base = blocked_paired(
                {
                    "vllm": lambda: cops.awq_gemm(x, qw, sc, qz, 1),
                    "torch": lambda: x @ w_ref.t(),
                },
                evict_passes=4,
                rounds=15,
                warm_per_block=0,
                inner_reps=1,
            )
            ranked = sorted(res.medians.items(), key=lambda kv: kv[1])
            best_cfg, best_us = ranked[0]
            roof = roofline_us(k * n * 0.5 + (k + n) * 2, BW_GBPS)
            rows.append(
                {
                    "layer": name,
                    "out": n,
                    "in": k,
                    "candidates": len(cands),
                    "best_config": best_cfg,
                    "ours_us": round(best_us, 1),
                    "vllm_us": round(base.medians["vllm"], 1),
                    "torch_us": round(base.medians["torch"], 1),
                    "vs_vllm": round(base.medians["vllm"] / best_us, 2),
                    "vs_torch": round(
                        base.medians["torch"] / best_us, 2
                    ),
                    "pct_of_roofline": round(100 * roof / best_us, 1),
                    "top3": [
                        {"config": c, "us": round(v, 1)}
                        for c, v in ranked[:3]
                    ],
                }
            )
            OUT.write_text(json.dumps(rows, indent=2) + "\n")
            r = rows[-1]
            print(
                f"{name:>10} {n}x{k} best {best_cfg:>20} "
                f"{best_us:6.1f}us ({r['pct_of_roofline']}% roof)  "
                f"vllm {r['vllm_us']:6.1f} torch {r['torch_us']:6.1f}  "
                f"-> {r['vs_vllm']}x vs vllm",
                flush=True,
            )

    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
