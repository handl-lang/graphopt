"""Real AWQ weights: does the fused dequant op beat torch on them?

Qwen2.5-0.5B-Instruct-AWQ stores `qweight` (int32-packed 4-bit, packed
along the output dim), `qzeros`, and per-group `scales` (group 128 along
the input dim). Unpacking gives exactly the layout our op wants:
`w_q[out, in]` + `scale[out, groups]`.

The comparison is the honest one: materialize the float weight then
matmul (what torch does) vs dequantize inside the kernel (what we do).
Unpacking the int4 is *loading* cost and is excluded from both.
"""

from __future__ import annotations

import json
import statistics as stats
import time

import grouped_quant_op as gq
import torch
from safetensors import safe_open

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)


def unpack_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group: int = 128,
):
    """AWQ int4 -> (w_q float [out, in], scale [out, groups])."""
    shifts = torch.arange(
        0, 32, 4, device=qweight.device, dtype=torch.int32
    )
    q = (qweight.unsqueeze(-1) >> shifts) & 0xF
    q = q.reshape(qweight.shape[0], -1).float()  # [in, out]
    z = (qzeros.unsqueeze(-1) >> shifts) & 0xF
    z = z.reshape(qzeros.shape[0], -1).float()  # [in//g, out]
    s = scales.float()  # [in//g, out]
    z = z.repeat_interleave(group, dim=0)  # [in, out]
    s = s.repeat_interleave(group, dim=0)  # [in, out]
    w = (q - z) * s  # [in, out]
    return w.t().contiguous(), scales.t().contiguous()


def ref_op(x, wq, sc, b):
    n, k = wq.shape
    g = sc.shape[1]
    w = (wq.reshape(n, g, k // g) * sc.unsqueeze(-1)).reshape(n, k)
    return x @ w.transpose(0, 1) + b


def timeit(fn, reps=100, warm=20) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return stats.median(ts) * 1e6


def main() -> int:
    targets = [
        ("model.layers.0.self_attn.q_proj", "q_proj"),
        ("model.layers.0.self_attn.o_proj", "o_proj"),
        ("model.layers.0.mlp.down_proj", "down_proj"),
        ("model.layers.0.mlp.gate_proj", "gate_proj"),
    ]
    out = {"ckpt": "Qwen/Qwen2.5-0.5B-Instruct-AWQ", "rows": []}
    print(
        f"{'layer':>10} {'out':>5} {'in':>5} {'M':>3} {'torch':>8} "
        f"{'ours':>8} {'vs torch':>9} {'max|Δ|':>9}"
    )
    with safe_open(CKPT, "pt") as f:
        for prefix, name in targets:
            try:
                qw = f.get_tensor(f"{prefix}.qweight").cuda()
                qz = f.get_tensor(f"{prefix}.qzeros").cuda()
                sc = f.get_tensor(f"{prefix}.scales").cuda()
            except Exception:
                continue
            wq, scale = unpack_awq(qw, qz, sc)
            n, k = wq.shape
            b = torch.zeros(n, device="cuda")
            for m in (1, 32):
                x = torch.randn(m, k, device="cuda")
                ref = ref_op(x, wq, scale, b)
                got = gq.grouped_quant_linear(
                    x, wq, scale, b, ieee=True
                )
                err = float((ref - got).abs().max())
                t_ref = timeit(lambda: ref_op(x, wq, scale, b))
                t_ours = timeit(
                    lambda: gq.grouped_quant_linear(
                        x, wq, scale, b, ieee=True
                    )
                )
                out["rows"].append(
                    {
                        "layer": name,
                        "out": n,
                        "in": k,
                        "m": m,
                        "torch_us": round(t_ref, 1),
                        "ours_us": round(t_ours, 1),
                        "vs_torch": round(t_ref / t_ours, 3),
                        "max_abs_err": err,
                    }
                )
                print(
                    f"{name:>10} {n:>5} {k:>5} {m:>3} {t_ref:>8.1f} "
                    f"{t_ours:>8.1f} {t_ref / t_ours:>8.2f}x {err:>9.1e}"
                )

    path = (
        "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
        "p7_awq_real.json"
    )
    open(path, "w").write(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
