"""e2e with the *original* checkpoint tensors (P17 fix).

vLLM converts AWQ into a GPTQ-like layout and pads it in
`process_weights_after_loading`, so a kernel written for the checkpoint
layout reads the wrong numbers (P17: divergence at token 1). This
snapshots `qweight`/`qzeros`/`scales` **before** that conversion and
drives our kernel from the snapshot.

The run reports token agreement first and timing second: a timing number
attached to wrong output is not a result.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time

import torch

from .awq_packing import to_standard_order
from .kernel_split2 import gemv_split2

USE_OURS = {"on": False}
STASH: dict[int, tuple] = {}
COUNTS = {"ours": 0, "stock": 0, "no_cfg": 0, "not_m1": 0}

#: (in, out) -> (bn, bk, split, warps) from the cold search (P14)
ROUTE: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (4864, 896): (64, 64, 8, 2),  # down_proj
    (896, 896): (32, 128, 4, 2),  # q/o_proj
    (896, 128): (32, 64, 4, 2),  # k/v_proj
}


def install_patch() -> None:
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQLinearMethod,
    )

    orig_pwal = AutoAWQLinearMethod.process_weights_after_loading
    orig_apply = AutoAWQLinearMethod.apply

    def pwal(self, layer):  # noqa: ANN001
        # snapshot the checkpoint tensors before vLLM repacks and pads
        if getattr(layer, "qweight", None) is not None:
            STASH[id(layer)] = (
                to_standard_order(layer.qweight.detach()),
                to_standard_order(layer.qzeros.detach()),
                layer.scales.detach().clone(),
            )
        return orig_pwal(self, layer)

    def apply(self, layer, x, bias=None):  # noqa: ANN001
        snap = STASH.get(id(layer))
        if not USE_OURS["on"] or snap is None:
            COUNTS["stock"] += 1
            return orig_apply(self, layer, x, bias)
        qw, qz, sc = snap
        n, k = sc.shape[1], qw.shape[0]
        cfg = ROUTE.get((k, n))
        if cfg is None:
            COUNTS["no_cfg"] += 1
            return orig_apply(self, layer, x, bias)
        if x.shape[0] != 1:
            COUNTS["not_m1"] += 1
            return orig_apply(self, layer, x, bias)
        COUNTS["ours"] += 1
        bn, bk, split, wp = cfg
        out = gemv_split2(
            x.reshape(-1).to(torch.float16).contiguous(),
            qw,
            qz,
            sc,
            n,
            k,
            bn,
            bk,
            split,
            wp,
        )
        if bias is not None:
            out = out + bias.to(torch.float16)
        return out

    AutoAWQLinearMethod.process_weights_after_loading = pwal
    AutoAWQLinearMethod.apply = apply


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--frac", type=float, default=0.45)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    install_patch()
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        quantization="awq",
        tokenizer="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.frac,
        max_model_len=args.ctx,
        seed=0,
        disable_log_stats=True,
    )
    torch.manual_seed(0)
    ids = torch.randint(1, 151936, (1, args.ctx // 2)).tolist()
    sp = SamplingParams(
        temperature=0.0, max_tokens=args.gen, ignore_eos=True
    )

    def call(on: bool):
        USE_OURS["on"] = on
        t0 = time.perf_counter()
        out = llm.generate(ids, sampling_params=sp, use_tqdm=False)
        return time.perf_counter() - t0, list(
            out[0].outputs[0].token_ids
        )

    # agreement FIRST: a timing attached to wrong output is not a result
    base_toks = call(False)[1]
    ours_toks = call(True)[1]
    first = next(
        (
            i
            for i, (a, b) in enumerate(
                zip(base_toks, ours_toks, strict=True)
            )
            if a != b
        ),
        None,
    )
    agree = base_toks == ours_toks
    print(f"token agreement: {agree} (first divergence at {first})")
    print(f"routing: {COUNTS}")

    for _ in range(3):
        call(False)
        call(True)

    base, ours = [], []
    for i in range(args.rounds):
        for on in (False, True) if i % 2 == 0 else (True, False):
            dt, _ = call(on)
            (ours if on else base).append(dt)
    USE_OURS["on"] = False

    mb, mo = stats.median(base), stats.median(ours)
    wins = sum(1 for a, b in zip(ours, base, strict=True) if a < b)
    print(
        f"stock {mb * 1e3:7.1f} ms | ours {mo * 1e3:7.1f} ms | "
        f"{100 * (mo - mb) / mb:+.1f}% | wins {wins}/{args.rounds}"
    )
    if args.out:
        open(args.out, "w").write(
            json.dumps(
                {
                    "identical_tokens": agree,
                    "first_divergence": first,
                    "counts": COUNTS,
                    "rounds": args.rounds,
                    "stock_ms": round(mb * 1e3, 2),
                    "ours_ms": round(mo * 1e3, 2),
                    "delta_pct": round(100 * (mo - mb) / mb, 2),
                    "ours_wins": wins,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
