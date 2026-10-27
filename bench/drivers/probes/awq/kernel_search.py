"""Phase 3 — the machine picks structure, not just config.

Two structural variants and a config grid, scored by measurement:

* ``per_feature`` — v1: index the packed word once per output feature
  (8 redundant loads of every word),
* ``word_once``   — v2: load the word once, unpack all 8 nibbles.

plus ``split`` and the tile shape. The referee is the timer, the gate is
exactness (`max|Δ|` vs the unpacked reference), and the output says
whether the machine's choice agrees with the config I picked by hand.

This is search over an authored template space — not synthesis. The
action space is mine; the choice is the machine's.
"""

from __future__ import annotations

import json
import statistics as stats
import time

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from .kernel_v2 import _gemv_v2, roofline_us, unpack_ref

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)


@triton.jit
def _gemv_per_feature(
    x_ptr,
    qw_ptr,
    qz_ptr,
    sc_ptr,
    out_ptr,
    N,
    K,
    GS,
    s_qwk,
    s_qwn,
    s_qzk,
    s_qzn,
    s_sck,
    s_scn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """v1 structure: one word read per output feature (redundant)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_p = rn // 8
    shift = (rn % 8) * 4
    nmask = rn < N

    per = tl.cdiv(K, SPLIT * BLOCK_K) * BLOCK_K
    k_lo = pid_k * per
    k_hi = tl.minimum(k_lo + per, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        g = rk // GS
        mask = kmask[:, None] & nmask[None, :]
        words = tl.load(
            qw_ptr + rk[:, None] * s_qwk + rn_p[None, :] * s_qwn,
            mask=mask,
            other=0,
        )
        q = (words >> shift[None, :]) & 0xF
        zwords = tl.load(
            qz_ptr + g[:, None] * s_qzk + rn_p[None, :] * s_qzn,
            mask=mask,
            other=0,
        )
        z = (zwords >> shift[None, :]) & 0xF
        s = tl.load(
            sc_ptr + g[:, None] * s_sck + rn[None, :] * s_scn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(x_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
        acc += tl.sum(
            (q.to(tl.float32) - z.to(tl.float32)) * s * x[:, None],
            axis=0,
        )

    tl.store(out_ptr + rn, acc.to(tl.float16), mask=nmask)


def _run_per_feature(x, qw, qz, sc, n, k, bn, bk, split, wp):
    out = torch.empty((1, n), device="cuda", dtype=torch.float16)
    _gemv_per_feature[(triton.cdiv(n, bn), split)](
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


def timeit(fn, reps=25, warm=6) -> float:
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


HAND_PICKED = {
    (4864, 896): ("word_once", 64, 128, 4, 2),
    (896, 896): ("word_once", 32, 128, 4, 2),
    (896, 4864): ("word_once", 64, 64, 4, 2),
    (896, 128): ("word_once", 32, 256, 4, 4),
}


def search_shape(qw, qz, sc, n, k, ref, x, grid) -> dict:
    """Round-robin over candidates so clock drift hits every one equally.

    Measuring candidate A fully, then B, then C lets a drifting GPU
    favour whoever ran first — that is how the first sweep picked a
    config 40 % slower than the hand-picked one. Here every round times
    *all* candidates once, and the per-candidate median is taken across
    rounds, so drift cancels by construction.
    """
    cands = []
    for variant in ("word_once", "per_feature"):
        for bn in grid["bn"]:
            if bn > n:
                continue
            for bk in grid["bk"]:
                for split in grid["split"]:
                    for wp in grid["warps"]:
                        cands.append((variant, bn, bk, split, wp))

    live = []
    for variant, bn, bk, split, wp in cands:
        try:
            fn = (
                (
                    lambda: _run_word(
                        x, qw, qz, sc, n, k, bn, bk, split, wp
                    )
                )
                if variant == "word_once"
                else (
                    lambda: _run_per_feature(
                        x, qw, qz, sc, n, k, bn, bk, split, wp
                    )
                )
            )
            got = fn()
            err = float((ref - got.float()).abs().max())
        except Exception:  # noqa: BLE001
            continue
        if err > 5e-2:  # exactness gate before any timing
            continue
        live.append(
            {
                "variant": variant,
                "bn": bn,
                "bk": bk,
                "split": split,
                "warps": wp,
                "fn": fn,
                "max_abs_err": err,
                "times": [],
            }
        )

    for c in live:  # JIT-compile everything first
        c["fn"]()
    torch.cuda.synchronize()

    # Blocked randomization: each round visits every candidate in a fresh
    # shuffled order, warming that kernel's weights before the timed
    # block. Warm-cache deploy regime + drift averaged across rounds.
    rounds = 5
    gen = torch.Generator().manual_seed(0)
    for _ in range(rounds):
        for i in torch.randperm(len(live), generator=gen).tolist():
            c = live[i]
            for _ in range(2):
                c["fn"]()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(4):
                c["fn"]()
            torch.cuda.synchronize()
            c["times"].append((time.perf_counter() - t0) / 4)

    for c in live:
        c["us"] = stats.median(c["times"]) * 1e6
    live.sort(key=lambda c: c["us"])
    best = live[0] if live else None
    if best is not None:
        best = {k: v for k, v in best.items() if k != "fn"}
        best["times"] = [round(t * 1e6, 1) for t in best["times"]]
    return {"tried": len(live), "best": best}


def _run_word(x, qw, qz, sc, n, k, bn, bk, split, wp):
    out = torch.empty((1, n), device="cuda", dtype=torch.float16)
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


def main() -> int:
    import vllm._custom_ops as cops

    grid = {
        "bn": (32, 64, 128, 256),
        "bk": (64, 128, 256),
        "split": (1, 2, 4, 8),
        "warps": (2, 4, 8),
    }
    shapes = (
        ("down_proj", "model.layers.0.mlp.down_proj"),
        ("q_proj", "model.layers.0.self_attn.q_proj"),
        ("gate_proj", "model.layers.0.mlp.gate_proj"),
        ("k_proj", "model.layers.0.self_attn.k_proj"),
    )
    rows = []
    with safe_open(CKPT, "pt") as f:
        for name, pre in shapes:
            qw = f.get_tensor(f"{pre}.qweight").cuda()
            qz = f.get_tensor(f"{pre}.qzeros").cuda()
            sc = f.get_tensor(f"{pre}.scales").cuda()
            n, k = sc.shape[1], qw.shape[0]
            ref = unpack_ref(qw, qz, sc).half().float()
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)
            ref = x.float() @ ref.t()

            best = search_shape(qw, qz, sc, n, k, ref, x, grid)
            t_vllm = timeit(lambda: cops.awq_gemm(x, qw, sc, qz, 1))
            hand = HAND_PICKED.get((k, n))
            b = best["best"]
            rows.append(
                {
                    "layer": name,
                    "out": n,
                    "in": k,
                    "roofline_us": round(roofline_us(n, k), 1),
                    "chosen": (
                        None
                        if b is None
                        else f"{b['variant']} bn={b['bn']} bk={b['bk']} "
                        f"split={b['split']} w={b['warps']}"
                    ),
                    "chosen_us": None
                    if b is None
                    else round(b["us"], 1),
                    "pct_of_roofline": (
                        None
                        if b is None
                        else round(100 * roofline_us(n, k) / b["us"], 1)
                    ),
                    "vs_vllm": (
                        None
                        if b is None
                        else round(t_vllm / b["us"], 2)
                    ),
                    "hand_picked": (
                        None
                        if hand is None
                        else f"{hand[0]} bn={hand[1]} bk={hand[2]} "
                        f"split={hand[3]} w={hand[4]}"
                    ),
                    "agrees_with_hand": (
                        None
                        if (b is None or hand is None)
                        else (
                            b["variant"],
                            b["bn"],
                            b["bk"],
                            b["split"],
                            b["warps"],
                        )
                        == hand
                    ),
                    "candidates": best["tried"],
                    "vllm_us": round(t_vllm, 1),
                }
            )
            print(
                f"{name:>10} {n:>5}x{k:<5} candidates={best['tried']:<4} "
                f"best {rows[-1]['chosen']}  {b['us']:6.1f} us "
                f"({rows[-1]['pct_of_roofline']}% roof)  vs vllm "
                f"{rows[-1]['vs_vllm']}x  agrees_with_hand="
                f"{rows[-1]['agrees_with_hand']}"
            )

    path = (
        "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
        "p10_kernel_search.json"
    )
    open(path, "w").write(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
