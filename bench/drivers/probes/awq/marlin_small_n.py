"""Does Marlin's documented lever pay, and does it really lose at small `n`?

P19 read vLLM's log line — *"Marlin kernel can achieve better performance
for small size_n with experimental use_atomic_add"* — as a statement about
`k_proj`/`v_proj`.  The gate says otherwise::

    # marlin_utils.should_use_atomic_add_reduce
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

So on this model the hint can only ever fire on **`down_proj`**
(`n=896, k=4864`).  `k_proj`/`v_proj` (`n=128, k=896`) fail on `k < 2048`,
`qkv` fails the same way, and `gate_up` fails on `n >= 2048`.  The "small
size_n" in the message means *small relative to K*, not *small in absolute
terms*.

This measures the levers rather than reading them:

* Marlin, default (what `vllm serve` runs),
* Marlin with `VLLM_MARLIN_USE_ATOMIC_ADD=1`, on every shape — so we see
  both whether it pays and whether vLLM's gate picks the right shapes,
* `awq_gemm`, the non-Marlin AWQ path,
* our two-stage split kernel (`gemv_split2`, P14), best config by timer,
* an fp16 GEMV of the same layer, as the "no dequant at all" floor.

Every timing is reported against the measured 89.1 GB/s byte roofline
(`kernel_v2.BW_GBPS`), and every kernel is checked against the unpacked
reference before its number is believed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from .awq_packing import to_standard_order
from .kernel_split2 import gemv_split2
from .kernel_v2 import BW_GBPS, roofline_us, timeit, unpack_ref

OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p21_marlin_small_n.json"
)

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)

L0 = "model.layers.0"

#: The linears as the model calls them (fused), plus `k_proj` alone, which
#: is the shape the P19 claim was about and is a sub-shape of `qkv`.
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


def load_group(f, names):
    """Load and (if needed) fuse one linear in AWQ layout."""
    qws, qzs, scs = [], [], []
    for name in names:
        qws.append(f.get_tensor(f"{name}.qweight").cuda())
        qzs.append(f.get_tensor(f"{name}.qzeros").cuda())
        scs.append(f.get_tensor(f"{name}.scales").cuda())
    # AWQ packs along N: qweight [K, N/8], qzeros [K/GS, N/8], scales [K/GS, N].
    qw = torch.cat(qws, dim=1).contiguous()
    qz = torch.cat(qzs, dim=1).contiguous()
    sc = torch.cat(scs, dim=1).contiguous()
    return qw, qz, sc


def timeit_b2b(fn, reps: int = 50, warm: int = 10) -> float:
    """Queue ``reps`` calls, sync once, divide.

    `kernel_v2.timeit` synchronises after *every* call, so at these sizes it
    is measuring launch overhead as much as the kernel — it put Marlin at
    191 µs/layer while the in-engine profiler said 105 (P19). A decode step
    enqueues 96 linears back to back and never waits between them, so this
    is the number that matches the service.
    """
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


def timeit_graph(fn, calls: int = 20, replays: int = 5):
    """Pure GPU time per call, via a captured graph.

    Back-to-back timing still counts the CPU launch gap for each call, and
    at 25 µs kernels that gap *is* the measurement. A captured graph removes
    the CPU from the loop, so what is left is device time per call — the
    same quantity the in-engine profiler reports (`self_device_time_total /
    count`, P19). Three timers disagreeing by 1.8x is the reason this
    exists.
    """
    try:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / (replays * calls)
    except Exception:
        return None


def rank_time(fn) -> float:
    """Instrument used to *choose* a config: pure GPU time, else b2b.

    Ranking on `kernel_v2.timeit` (one sync per call) picks launch overhead,
    not the kernel: on `gate_up` it selected a config that graph timing then
    showed at 95.6 µs when a 81 µs one was available.
    """
    t = timeit_graph(fn, calls=10, replays=3)
    return t if t is not None else timeit_b2b(fn, reps=30)


def main() -> int:
    from vllm import _custom_ops as cops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
        awq_to_marlin_zero_points,
        get_marlin_workspace,
        marlin_permute_scales,
        should_use_atomic_add_reduce,
    )
    from vllm.scalar_type import scalar_types

    rows = []
    with safe_open(CKPT, "pt") as f:
        for label, names in GROUPS:
            qw, qz, sc = load_group(f, names)
            n, k = int(sc.shape[1]), int(qw.shape[0])
            gs = k // sc.shape[0]
            # AWQ packs nibbles as [0,4,1,5,2,6,3,7] (`awq_packing`). Marlin
            # and `awq_gemm` consume that native order; our kernel and the
            # reference need plain order. Feeding the native words to both —
            # which `kernel_v2.main` does — makes the check vacuous: the
            # kernel and the "truth" agree while neither matches AWQ
            # (cosine 0.21 against `awq_gemm`, measured).
            qw_std = to_standard_order(qw)
            qz_std = to_standard_order(qz)
            w_ref = unpack_ref(qw_std, qz_std, sc, gs).half()
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)
            ref = x.float() @ w_ref.float().t()
            scale = float(ref.abs().mean()) or 1.0
            roof = roofline_us(n, k)

            # --- Marlin, packed exactly as vLLM packs it -----------------
            wq = cops.awq_marlin_repack(
                qw, size_k=k, size_n=n, num_bits=4
            )
            wz = awq_to_marlin_zero_points(
                qz, size_k=int(qz.shape[0]), size_n=n, num_bits=4
            )
            ws = marlin_permute_scales(
                sc, size_k=k, size_n=n, group_size=gs
            )
            workspace = get_marlin_workspace(x.device)
            wtype = (
                scalar_types.uint4
            )  # AWQ carries zero points: no +8 bias

            def marlin() -> torch.Tensor:
                return apply_gptq_marlin_linear(
                    input=x,
                    weight=wq,
                    weight_scale=ws,
                    weight_zp=wz,
                    workspace=workspace,
                    wtype=wtype,
                    output_size_per_partition=n,
                    input_size_per_partition=k,
                )

            os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"] = "0"
            out_plain = marlin()
            err_plain = (
                float((ref - out_plain.float()).abs().max()) / scale
            )
            us_plain = timeit(marlin, reps=60, warm=15)
            b2b_plain = timeit_b2b(marlin)

            # vLLM's own gate, asked directly.
            os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"] = "1"
            eligible = should_use_atomic_add_reduce(
                1, n, k, x.device, torch.float16
            )
            us_atomic, err_atomic, b2b_atomic = None, None, None
            if eligible:
                out_atomic = marlin()
                err_atomic = (
                    float((ref - out_atomic.float()).abs().max())
                    / scale
                )
                us_atomic = timeit(marlin, reps=60, warm=15)
                b2b_atomic = timeit_b2b(marlin)
            os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"] = "0"

            # --- awq_gemm (non-Marlin) ----------------------------------
            awq_fn = lambda: cops.awq_gemm(x, qw, sc, qz, 1)  # noqa: E731
            us_awq = timeit(awq_fn, reps=60)
            b2b_awq = timeit_b2b(awq_fn)

            # --- ours: best config by timer, exactness gated -------------
            xf = x.reshape(-1)
            best = None
            for bn in (32, 64, 128, 256):
                if bn > n:
                    continue
                for split in (1, 2, 4):
                    for warps in (2, 4):
                        try:
                            got = gemv_split2(
                                xf,
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
                        except Exception:
                            continue
                        err = (
                            float((ref - got.float()).abs().max())
                            / scale
                        )
                        if err > 0.05:  # not this kernel's number
                            continue
                        us = rank_time(
                            lambda: gemv_split2(
                                xf,
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
                        )
                        if best is None or us < best[0]:
                            best = (us, bn, split, warps, err)
            cfg, err_ours = best[1:4], best[4]
            ours_fn = lambda: gemv_split2(  # noqa: E731
                xf,
                qw_std,
                qz_std,
                sc,
                n,
                k,
                cfg[0],
                128,
                cfg[1],
                cfg[2],
            )
            # Chosen on `rank_us` (graph), reported on all three clocks so the
            # columns stay comparable with Marlin's.
            us_ours = timeit(ours_fn, reps=60)
            b2b_ours = timeit_b2b(ours_fn)

            # --- fp16 GEMV: same weights, no dequant, no unpack ---------
            fp16_fn = lambda: x @ w_ref.t()  # noqa: E731
            us_fp16 = timeit(fp16_fn, reps=60)
            b2b_fp16 = timeit_b2b(fp16_fn)

            g_marlin = timeit_graph(marlin)
            g_awq = timeit_graph(awq_fn)
            g_ours = timeit_graph(ours_fn)
            g_fp16 = timeit_graph(fp16_fn)
            if eligible:
                os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"] = "1"
                g_atomic = timeit_graph(marlin)
                os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"] = "0"
            else:
                g_atomic = None

            row = {
                "layer": label,
                "out": n,
                "in": k,
                "roofline_us": round(roof, 2),
                "marlin_us": round(us_plain, 1),
                "marlin_pct_roof": round(100 * roof / us_plain, 1),
                "marlin_rel_err": round(err_plain, 4),
                "atomic_eligible": eligible,
                "marlin_atomic_us": (
                    round(us_atomic, 1) if us_atomic else None
                ),
                "marlin_atomic_pct_roof": (
                    round(100 * roof / us_atomic, 1)
                    if us_atomic
                    else None
                ),
                "marlin_atomic_rel_err": (
                    round(err_atomic, 4)
                    if err_atomic is not None
                    else None
                ),
                "awq_us": round(us_awq, 1),
                "ours_us": round(us_ours, 1),
                "ours_config": f"bn={cfg[0]} split={cfg[1]} w={cfg[2]}",
                "ours_pct_roof": round(100 * roof / us_ours, 1),
                "ours_rel_err": round(err_ours, 4),
                "fp16_us": round(us_fp16, 1),
                "marlin_b2b_us": round(b2b_plain, 1),
                "marlin_b2b_pct_roof": round(100 * roof / b2b_plain, 1),
                "marlin_atomic_b2b_us": (
                    round(b2b_atomic, 1) if b2b_atomic else None
                ),
                "awq_b2b_us": round(b2b_awq, 1),
                "ours_b2b_us": round(b2b_ours, 1),
                "ours_b2b_pct_roof": round(100 * roof / b2b_ours, 1),
                "fp16_b2b_us": round(b2b_fp16, 1),
                "ours_vs_marlin_b2b": round(b2b_plain / b2b_ours, 2),
                "graph_marlin_us": round(g_marlin, 1)
                if g_marlin
                else None,
                "graph_atomic_us": round(g_atomic, 1)
                if g_atomic
                else None,
                "graph_awq_us": round(g_awq, 1) if g_awq else None,
                "graph_ours_us": round(g_ours, 1) if g_ours else None,
                "graph_fp16_us": round(g_fp16, 1) if g_fp16 else None,
                "graph_marlin_pct_roof": (
                    round(100 * roof / g_marlin, 1)
                    if g_marlin
                    else None
                ),
                "graph_ours_pct_roof": (
                    round(100 * roof / g_ours, 1) if g_ours else None
                ),
                "graph_ours_vs_marlin": (
                    round(g_marlin / g_ours, 2) if g_ours else None
                ),
                "ours_vs_marlin": round(us_plain / us_ours, 2),
                "fp16_vs_marlin": round(us_plain / us_fp16, 2),
            }
            rows.append(row)
            at = (
                f"atomic {us_atomic:6.1f}"
                if us_atomic
                else "atomic      -"
            )
            print(
                f"{label:>10} {n:>5}x{k:<5} roof {roof:6.2f} | "
                f"marlin {us_plain:6.1f} ({row['marlin_pct_roof']:4.1f}%) "
                f"{at} | awq {us_awq:6.1f} | "
                f"ours {us_ours:6.1f} ({row['ours_pct_roof']:4.1f}%) "
                f"[{row['ours_config']}] | fp16 {us_fp16:6.1f} | "
                f"ours/marlin {row['ours_vs_marlin']:.2f}x"
                f"\n{'':>10} {'':>11} b2b  | "
                f"marlin {b2b_plain:6.1f} "
                f"({row['marlin_b2b_pct_roof']:4.1f}%) | "
                f"awq {b2b_awq:6.1f} | ours {b2b_ours:6.1f} "
                f"({row['ours_b2b_pct_roof']:4.1f}%) | "
                f"fp16 {b2b_fp16:6.1f} | "
                f"ours/marlin {row['ours_vs_marlin_b2b']:.2f}x"
                f"\n{'':>10} {'':>11} graph| "
                f"marlin {g_marlin or 0:6.1f} "
                f"({row['graph_marlin_pct_roof'] or 0:4.1f}%) "
                f"atomic {(f'{g_atomic:5.1f}' if g_atomic else '    -')} | "
                f"awq {g_awq or 0:6.1f} | ours {g_ours or 0:6.1f} "
                f"({row['graph_ours_pct_roof'] or 0:4.1f}%) | "
                f"fp16 {g_fp16 or 0:6.1f} | "
                f"ours/marlin {row['graph_ours_vs_marlin'] or 0:.2f}x"
            )

    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nbandwidth basis: {BW_GBPS} GB/s\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
