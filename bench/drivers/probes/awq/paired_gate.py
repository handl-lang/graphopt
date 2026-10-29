"""P22 — P9/P10 re-derived: gate against `awq_gemm`, time with the referee.

Two instruments were wrong in P9/P10/P14:

* the reference shared the kernel's nibble order, so "exact" meant
  "wrong in the same way" (cosine 0.21 vs `awq_gemm`, P21),
* configs were ranked with sync-per-call timing, which at these sizes
  measures launch overhead (up to 4.7x off, P21).

This measures every shape twice — `blocked_paired` (the deploy referee,
drift cancelled, L2 evicted) and a captured CUDA graph (what a CUDA-graph
serving loop actually executes) — and gates every output against
`awq_gemm` with `bench.benchkit.equality`.

Cross-run comparisons are not valid on this card: identical code measured
`awq_gemm` at 38 µs and 285 µs minutes apart. Only ratios measured inside
one run mean anything, which is why the incumbent is timed alongside each
candidate rather than quoted from an earlier report.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from bench.benchkit.equality import cosine_gate, require
from bench.benchkit.regime import blocked_paired, evictor, roofline_us

from .awq_packing import to_standard_order
from .kernel_split2 import gemv_split2
from .kernel_v2 import BW_GBPS, timeit_graph

OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/p22_paired.json"
)

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)

L0 = "model.layers.0"

GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "qkv",
        (
            f"{L0}.self_attn.q_proj",
            f"{L0}.self_attn.k_proj",
            f"{L0}.self_attn.v_proj",
        ),
    ),
    ("o_proj", (f"{L0}.self_attn.o_proj",)),
    ("gate_up", (f"{L0}.mlp.gate_proj", f"{L0}.mlp.up_proj")),
    ("down_proj", (f"{L0}.mlp.down_proj",)),
    ("k_proj", (f"{L0}.self_attn.k_proj",)),
)


def _load_group(f, names):  # noqa: ANN001, ANN201
    parts = [
        (
            f.get_tensor(f"{nm}.qweight").cuda(),
            f.get_tensor(f"{nm}.qzeros").cuda(),
            f.get_tensor(f"{nm}.scales").cuda(),
        )
        for nm in names
    ]
    qw = torch.cat([p[0] for p in parts], dim=1).contiguous()
    qz = torch.cat([p[1] for p in parts], dim=1).contiguous()
    sc = torch.cat([p[2] for p in parts], dim=1).contiguous()
    return qw, qz, sc


def _best_gated_config(x, qw_std, qz_std, sc, n, k, incumbent):
    """Smallest graph time among configs that clear the gate."""
    best = None
    for bn in (32, 64, 128):
        if bn > n:
            continue
        for split in (1, 2, 4):
            for warps in (2, 4):

                def fn(bn=bn, split=split, warps=warps):  # noqa: ANN202
                    return gemv_split2(
                        x.reshape(-1),
                        qw_std,
                        qz_std,
                        sc,
                        n,
                        k,
                        bn,
                        128,
                        split,
                        warps,
                    )

                try:
                    out = fn()
                except Exception:
                    continue
                name = f"bn={bn} split={split} w={warps}"
                verdict = cosine_gate(name, out, incumbent=incumbent)
                if not verdict.passed:
                    continue
                us = timeit_graph(fn, calls=10, replays=3)
                if us is None:
                    continue
                if best is None or us < best[0]:
                    best = (us, bn, split, warps, verdict, fn)
    return best


def main() -> int:
    from vllm import _custom_ops as cops

    rows = []
    with safe_open(CKPT, "pt") as f:
        for label, names in GROUPS:
            qw, qz, sc = _load_group(f, names)
            n, k = int(sc.shape[1]), int(qw.shape[0])
            qw_std, qz_std = (
                to_standard_order(qw),
                to_standard_order(qz),
            )
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)

            # The incumbent is the only definition of "correct" here.
            incumbent = cops.awq_gemm(x, qw, sc, qz, 1)
            awq_fn = lambda: cops.awq_gemm(x, qw, sc, qz, 1)  # noqa: E731

            best = _best_gated_config(
                x, qw_std, qz_std, sc, n, k, incumbent
            )
            if best is None:
                raise SystemExit(f"{label}: no config cleared the gate")
            _us, bn, split, warps, verdict, ours_fn = best
            require(verdict)

            # DEFAULT_EVICT_MB is 64 — 32x this card's L2 and ~72 % of the
            # free VRAM after our tensors. At 4 GB it makes every call in the
            # block pay allocator pressure: both candidates measured ~400 us
            # against graph times of 5-50 us.
            regime = blocked_paired(
                {"ours": ours_fn, "awq": awq_fn},
                evict_passes=4,
                rounds=20,
                flush=evictor(evict_mb=8),
            )
            g_ours = timeit_graph(ours_fn)
            g_awq = timeit_graph(awq_fn)
            roof = roofline_us(n * k * 0.5 + 2 * k + 2 * n, BW_GBPS)

            row = {
                "layer": label,
                "out": n,
                "in": k,
                "config": f"bn={bn} split={split} w={warps}",
                "cosine_vs_awq": round(verdict.cosine, 7),
                "relative_worst": round(verdict.relative_worst, 5),
                "roofline_us": round(roof, 2),
                "deploy_ours_us": round(regime.medians["ours"], 1),
                "deploy_awq_us": round(regime.medians["awq"], 1),
                "deploy_ratio": round(regime.ratio("awq", "ours"), 2),
                "graph_ours_us": round(g_ours, 1),
                "graph_awq_us": round(g_awq, 1),
                "graph_ratio": round(g_awq / g_ours, 2),
                "graph_pct_roofline": round(100 * roof / g_ours, 1),
            }
            rows.append(row)
            print(
                f"{label:>10} {n:>5}x{k:<5} cos {verdict.cosine:.6f}  "
                f"deploy ours {row['deploy_ours_us']:6.1f} "
                f"awq {row['deploy_awq_us']:6.1f} "
                f"-> {row['deploy_ratio']:.2f}x  "
                f"| graph ours {row['graph_ours_us']:6.1f} "
                f"awq {row['graph_awq_us']:6.1f} "
                f"-> {row['graph_ratio']:.2f}x "
                f"({row['graph_pct_roofline']:4.1f}% roof)"
            )

    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
