"""End-to-end: replace vLLM's AWQ linear where we win, measure decode.

Routing comes from measurement, per geometry:

    down_proj (4864->896)  ours   (3.32x over awq_gemm)
    q/o_proj  (896->896)   ours   (1.21x)
    gate/up   (896->4864)  awq_gemm  (ours 0.89x)
    k/v_proj  (896->128)   torch    (plain matmul fastest at this size)

Two arms toggled per round in one engine: stock vLLM vs routed-ours.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time

import torch

from .awq_splitk import splitk

USE_OURS = {"on": False}
ROUTE: dict[tuple[int, int], tuple] = {
    (4864, 896): (64, 64, 8, 1),  # down: bn, bk, split, warps
    (896, 896): (32, 256, 1, 4),  # q/o
}


def install_patch() -> None:
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQLinearMethod as AWQLinearMethod,
    )

    orig = AWQLinearMethod.apply

    def apply(self, layer, x, bias=None):
        qw = getattr(layer, "qweight", None)
        if not USE_OURS["on"] or qw is None:
            return orig(self, layer, x, bias)
        sc = layer.scales
        n, k = sc.shape[1], qw.shape[0]
        cfg = ROUTE.get((k, n))
        if cfg is None or x.shape[0] != 1:
            return orig(self, layer, x, bias)
        bn, bk, split, wp = cfg
        out = splitk(
            x.reshape(-1).float().contiguous(),
            qw,
            layer.qzeros,
            sc,
            n,
            k,
            bn,
            bk,
            split,
            wp,
        )
        if bias is not None:
            out = out + bias.float()
        return out

    AWQLinearMethod.apply = apply


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    install_patch()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
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
        return time.perf_counter() - t0, [
            o.outputs[0].token_ids for o in out
        ]

    for _ in range(args.warm):
        call(False)
        call(True)

    base, ours, toks = [], [], {}
    for i in range(args.rounds):
        for on in (False, True) if i % 2 == 0 else (True, False):
            dt, tk = call(on)
            (ours if on else base).append(dt)
            toks.setdefault("ours" if on else "base", tk)
    USE_OURS["on"] = False

    mb, mo = stats.median(base), stats.median(ours)
    wins = sum(1 for a, b in zip(ours, base, strict=True) if a < b)
    same = toks["ours"] == toks["base"]
    print(f"\n{args.rounds} paired rounds")
    print(
        f"  stock vLLM : {mb * 1e3:7.1f} ms  sd {stats.stdev(base) * 1e3:5.1f}"
        f"  {args.gen / mb:6.1f} tok/s"
    )
    print(
        f"  routed ours: {mo * 1e3:7.1f} ms  sd {stats.stdev(ours) * 1e3:5.1f}"
        f"  {args.gen / mo:6.1f} tok/s"
    )
    print(
        f"  -> ours {100 * (mo - mb) / mb:+.1f}% · wins {wins}/{args.rounds}"
        f" · identical tokens: {same}"
    )

    if args.out:
        open(args.out, "w").write(
            json.dumps(
                {
                    "rounds": args.rounds,
                    "stock_ms": round(mb * 1e3, 2),
                    "ours_ms": round(mo * 1e3, 2),
                    "delta_pct": round(100 * (mo - mb) / mb, 2),
                    "ours_wins": wins,
                    "identical_tokens": same,
                    "route": {
                        f"{k}:{v}": c for (k, v), c in ROUTE.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
