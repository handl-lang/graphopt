"""The search driver, run for real: IR terms -> gate -> referee -> row.

The driver's own tests use injected candidates, so its *plumbing* has never
met a GPU. Here the candidates are the generated kernels from
`compile-triton`, the incumbent is `awq_gemm`, the gate is
`bench.benchkit.equality`, and the clock is a captured graph.

    PYTHONPATH=$PWD:<lib-kernel-ir>/src:<kernelopt>/src:<eqsat>/mlopt-core/src \
      python -m bench.drivers.probes.awq.search_real
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from lib_kernel_ir import qgemv
from mlopt_kernel.search import Candidate, search

from .awq_packing import to_standard_order
from .irtriton import cached_module
from .kernel_v2 import BW_GBPS, roofline_us, timeit_graph

OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/p24_search_real.json"
)

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)

SHAPES = (
    ("down_proj", "model.layers.0.mlp.down_proj", 896, 4864),
    ("gate_proj", "model.layers.0.mlp.gate_proj", 4864, 896),
)

SPLITS = (1, 2, 4, 8)
BLOCK_N, BLOCK_K, WARPS = 64, 128, 2


def main() -> int:
    from vllm import _custom_ops as cops

    routes = []
    with safe_open(CKPT, "pt") as f:
        for label, prefix, n, k in SHAPES:
            qw = f.get_tensor(f"{prefix}.qweight").cuda()
            qz = f.get_tensor(f"{prefix}.qzeros").cuda()
            sc = f.get_tensor(f"{prefix}.scales").cuda()
            qw_std, qz_std = to_standard_order(qw), to_standard_order(qz)
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)
            xf = x.reshape(-1)
            awq_fn = lambda: cops.awq_gemm(x, qw, sc, qz, 1)  # noqa: E731

            candidates = []
            for split in SPLITS:
                term = qgemv(k=k, split=split)
                mod = cached_module(
                    term, block_n=BLOCK_N, block_k=BLOCK_K
                )

                def run(mod=mod):  # noqa: ANN001, ANN202
                    return mod.run(
                        xf,
                        qw_std,
                        qz_std,
                        sc,
                        n,
                        k,
                        BLOCK_N,
                        BLOCK_K,
                        WARPS,
                    )

                candidates.append(
                    Candidate(
                        term=term,
                        label=f"split{split}",
                        build=run,
                        measure=lambda run=run: timeit_graph(
                            run, calls=10, replays=3
                        ),
                    )
                )

            result = search(
                operator=f"qgemv/{label}",
                candidates=candidates,
                incumbent=awq_fn,
                bytes_moved=n * k * 0.5 + 2 * k + 2 * n,
                bandwidth_gbps=BW_GBPS,
                measure_incumbent=lambda: timeit_graph(
                    awq_fn, calls=10, replays=3
                ),
            )
            row = result.to_dict()
            row["roofline_us"] = round(roofline_us(n, k), 2)
            routes.append(row)
            print(json.dumps(row, indent=2))
            if result.chosen is None:
                print(f"!! {label}: nothing survived the gate")

    OUT.write_text(json.dumps(routes, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    ok = all(r["term"] for r in routes)
    print(f"every operator routed: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
