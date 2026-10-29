"""M1's definition of done: four `split_k` variants, generated from terms.

The build list says the gate is that the variants are *generated from IR
terms* (not hand-written), all agree with `awq_gemm` (cosine >= 0.9999),
and at least one matches the hand-written kernel's time. Two geometries
are exercised — `down_proj` (n=896, k=4864) and `gate_proj` (n=4864,
k=896) — so the lowering cannot overfit one shape.

The kill criterion is pre-registered: if generated code cannot match the
hand kernel, generalising is wasted work and this is where it gets said.

    PYTHONPATH=$PWD:... python -m bench.drivers.probes.awq.m1_verdict
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from bench.benchkit.equality import cosine_gate, require
from lib_kernel_ir import qgemv

from .awq_packing import to_standard_order
from .irtriton import compile_source, cached_module
from .kernel_split2 import gemv_split2
from .kernel_v2 import BW_GBPS, roofline_us, timeit_graph

OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/p23_m1_verdict.json"
)

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)

#: (label, output features, contraction) — two opposite geometries.
SHAPES: tuple[tuple[str, str, int, int], ...] = (
    ("down_proj", "model.layers.0.mlp.down_proj", 896, 4864),
    ("gate_proj", "model.layers.0.mlp.gate_proj", 4864, 896),
)

SPLITS = (1, 2, 4, 8)
BLOCK_N, BLOCK_K, WARPS = 64, 128, 2


def main() -> int:
    from vllm import _custom_ops as cops

    rows = []
    with safe_open(CKPT, "pt") as f:
        for label, prefix, n, k in SHAPES:
            qw = f.get_tensor(f"{prefix}.qweight").cuda()
            qz = f.get_tensor(f"{prefix}.qzeros").cuda()
            sc = f.get_tensor(f"{prefix}.scales").cuda()
            qw_std, qz_std = to_standard_order(qw), to_standard_order(qz)
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)
            xf = x.reshape(-1)
            incumbent = cops.awq_gemm(x, qw, sc, qz, 1)
            roof = roofline_us(n, k)

            for split in SPLITS:
                term = qgemv(k=k, split=split)
                source = compile_source(
                    term, block_n=BLOCK_N, block_k=BLOCK_K
                )
                mod = cached_module(
                    term, block_n=BLOCK_N, block_k=BLOCK_K
                )

                def gen() -> torch.Tensor:
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

                out = gen()
                verdict = cosine_gate(
                    f"{label}/split{split}", out, incumbent=incumbent
                )
                require(verdict)

                def hand() -> torch.Tensor:
                    return gemv_split2(
                        xf,
                        qw_std,
                        qz_std,
                        sc,
                        n,
                        k,
                        BLOCK_N,
                        BLOCK_K,
                        split,
                        WARPS,
                    )

                hand_gate = cosine_gate(
                    f"{label}/hand-split{split}",
                    hand(),
                    incumbent=incumbent,
                )
                require(hand_gate)

                us_gen = timeit_graph(gen, calls=10, replays=3)
                us_hand = timeit_graph(hand, calls=10, replays=3)
                rows.append(
                    {
                        "shape": label,
                        "n": n,
                        "k": k,
                        "split": split,
                        "generated_us": round(us_gen, 1),
                        "hand_us": round(us_hand, 1),
                        "generated_over_hand": round(us_gen / us_hand, 2),
                        "cosine_vs_awq": round(verdict.cosine, 7),
                        "cosine_hand_vs_awq": round(hand_gate.cosine, 7),
                        "pct_of_roofline": round(100 * roof / us_gen, 1),
                        "source_lines": source.count("\n"),
                        "source_has_atomics": "atomic" in source,
                    }
                )
                print(
                    f"{label:>10} split={split} "
                    f"gen {us_gen:6.1f} us  hand {us_hand:6.1f} us  "
                    f"gen/hand {us_gen / us_hand:4.2f}x  "
                    f"cos {verdict.cosine:.6f}  "
                    f"({100 * roof / us_gen:4.1f}% roof)"
                )

    OUT.write_text(json.dumps(rows, indent=2) + "\n")

    # -- the gate, stated once ------------------------------------
    best: dict[str, tuple[float, float]] = {}
    for r in rows:
        shape = r["shape"]
        g, h = best.get(shape, (1e9, 1e9))
        best[shape] = (min(g, r["generated_us"]), min(h, r["hand_us"]))
    print()
    ok = True
    for shape, (g, h) in best.items():
        within = g <= h * 1.05
        ok = ok and within
        print(
            f"{shape}: best generated {g:.1f} us vs best hand {h:.1f} us "
            f"-> {'MATCHES' if within else 'LOSES'} "
            f"({g / h:.2f}x, tolerance 1.05x)"
        )
    all_gated = all(r["cosine_vs_awq"] >= 0.9999 for r in rows)
    clean = not any(r["source_has_atomics"] for r in rows)
    print(
        f"\nvariants gated at cosine >= 0.9999: {all_gated}\n"
        f"no generated source uses atomics: {clean}\n"
        f"M1 GATE (generated matches hand on both geometries): {ok}\n"
        f"bandwidth basis {BW_GBPS} GB/s\nwrote {OUT}"
    )
    return 0 if (ok and all_gated) else 1


if __name__ == "__main__":
    raise SystemExit(main())
