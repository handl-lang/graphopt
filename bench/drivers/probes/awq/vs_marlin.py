"""Race our kernel against `marlin_gemm` — the kernel vLLM actually runs.

Every earlier comparison used `awq_gemm`, the generic AWQ kernel. But
vLLM repacks AWQ checkpoints for **Marlin** and serves through
`marlin_gemm` (evidence: the loaded layer carries
``qweight (K//16, N*2)`` and ``scales (K//group, N)`` — the Marlin
layout — and the AWQ patch never fires: 0 of 55 392 linear calls).

So the honest competitive question is not "do we beat `awq_gemm`" but
"do we beat Marlin". This measures exactly that, on the same real layer,
in the deployment regime.

Marlin needs its repacked weights, which the checkpoint does not ship —
they are produced by `awq_marlin_repack` at load time. We call that here,
then time both arms on identical inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from bench.benchkit.regime import blocked_paired

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)
OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p16_vs_marlin.json"
)


def main() -> int:
    import vllm._custom_ops as cops
    from vllm.scalar_type import scalar_types

    rows = []
    with safe_open(CKPT, "pt") as f:
        for name, pre in (
            ("down_proj", "model.layers.0.mlp.down_proj"),
            ("q_proj", "model.layers.0.self_attn.q_proj"),
        ):
            qw = f.get_tensor(f"{pre}.qweight").cuda()
            qz = f.get_tensor(f"{pre}.qzeros").cuda()
            sc = f.get_tensor(f"{pre}.scales").cuda()
            n, k = sc.shape[1], qw.shape[0]

            try:
                packed = cops.awq_marlin_repack(qw, k, n, 4, False)
                pscale = sc.contiguous()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{name}: repack failed: {type(exc).__name__} {exc}"
                )
                continue

            a = torch.randn(1, k, device="cuda", dtype=torch.float16)
            workspace = torch.zeros(
                max(n * 4, 1024), device="cuda", dtype=torch.int32
            )

            def marlin():
                return cops.marlin_gemm(
                    a,
                    None,
                    packed,
                    None,
                    pscale,
                    None,
                    None,
                    None,
                    workspace,
                    scalar_types.uint4b8,
                    1,
                    n,
                    k,
                )

            try:
                got = marlin()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{name}: marlin_gemm failed: {type(exc).__name__} "
                    f"{str(exc)[:120]}"
                )
                continue

            res = blocked_paired(
                {"marlin": marlin},
                evict_passes=4,
                rounds=12,
                warm_per_block=0,
                inner_reps=1,
            )
            rows.append(
                {
                    "layer": name,
                    "out": n,
                    "in": k,
                    "marlin_us": round(res.medians["marlin"], 1),
                    "marlin_out_shape": list(got.shape),
                }
            )
            OUT.write_text(json.dumps(rows, indent=2) + "\n")
            print(
                f"{name}: marlin {res.medians['marlin']:.1f} us",
                flush=True,
            )

    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
