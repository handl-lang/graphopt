"""M2's question: does a `pipeline(depth)` annotation buy anything?

The review's objection to the nine-op IR was that it cannot state a
schedule, so it can only express naive schedules. `pipeline` is the
minimum addition (ADR-0166) and lowers to Triton's `num_stages`. This
sweeps it against `split_k` and `BLOCK_K` on both geometries and compares
the best to P23's baseline (52.3 us on `down_proj`, 52.4 on `gate_proj`,
both at depth 1).

If nothing beats depth=1, the annotation is decoration and that is the
finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from bench.benchkit.equality import cosine_gate, require
from lib_kernel_ir import qgemv

from .awq_packing import to_standard_order
from .irtriton import cached_module
from .kernel_v2 import BW_GBPS, roofline_us, timeit_graph

OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p24_m2_schedule.json"
)

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)

SHAPES = (
    ("down_proj", "model.layers.0.mlp.down_proj", 896, 4864, 52.3),
    ("gate_proj", "model.layers.0.mlp.gate_proj", 4864, 896, 52.4),
)

DEPTHS = (1, 2, 3, 4)
SPLITS = (2, 4, 8)
BLOCK_KS = (128, 256)
BLOCK_N, WARPS = 64, 2


def main() -> int:
    from vllm import _custom_ops as cops

    rows = []
    with safe_open(CKPT, "pt") as f:
        for label, prefix, n, k, baseline in SHAPES:
            qw = f.get_tensor(f"{prefix}.qweight").cuda()
            qz = f.get_tensor(f"{prefix}.qzeros").cuda()
            sc = f.get_tensor(f"{prefix}.scales").cuda()
            qw_std, qz_std = to_standard_order(qw), to_standard_order(qz)
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)
            xf = x.reshape(-1)
            incumbent = cops.awq_gemm(x, qw, sc, qz, 1)
            roof = roofline_us(n, k)
            best = (1e9, None)

            for depth in DEPTHS:
                for split in SPLITS:
                    for bk in BLOCK_KS:
                        term = qgemv(k=k, split=split, depth=depth)
                        mod = cached_module(term, block_n=BLOCK_N, block_k=bk)

                        def gen(mod=mod, bk=bk):  # noqa: ANN001, ANN202
                            return mod.run(
                                xf,
                                qw_std,
                                qz_std,
                                sc,
                                n,
                                k,
                                BLOCK_N,
                                bk,
                                WARPS,
                            )

                        try:
                            out = gen()
                        except Exception as exc:  # noqa: BLE001
                            rows.append(
                                {
                                    "shape": label,
                                    "depth": depth,
                                    "split": split,
                                    "block_k": bk,
                                    "error": type(exc).__name__,
                                }
                            )
                            continue
                        verdict = cosine_gate(
                            f"{label}/d{depth}s{split}bk{bk}",
                            out,
                            incumbent=incumbent,
                        )
                        if not verdict.passed:
                            rows.append(
                                {
                                    "shape": label,
                                    "depth": depth,
                                    "split": split,
                                    "block_k": bk,
                                    "cosine": round(verdict.cosine, 7),
                                    "error": "gate",
                                }
                            )
                            continue
                        us = timeit_graph(gen, calls=10, replays=3)
                        rows.append(
                            {
                                "shape": label,
                                "depth": depth,
                                "split": split,
                                "block_k": bk,
                                "us": round(us, 1),
                                "cosine": round(verdict.cosine, 7),
                                "pct_of_roofline": round(100 * roof / us, 1),
                            }
                        )
                        if us < best[0]:
                            best = (us, (depth, split, bk))

            d, s, bk = best[1]
            print(
                f"{label}: best {best[0]:.1f} us at depth={d} split={s} "
                f"bk={bk}  (P23 baseline {baseline:.1f} us, "
                f"delta {100 * (best[0] / baseline - 1):+.1f}%)"
            )
            require(
                cosine_gate(
                    f"{label}/best", gen(), incumbent=incumbent
                )
            )

    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {OUT}  basis {BW_GBPS} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
