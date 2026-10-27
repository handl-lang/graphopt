"""W4-style quantized Llama in vLLM: torch dequant vs our fused op.

Every `Linear` in the model is quantized per-group (W4-ish: group 128,
scale = max|w|/7).  Two dequant paths are then compared **inside the same
engine**, toggled per round:

* baseline — ``x @ (w_q * scale).T + bias``: materializes the float
  weight every forward (the realistic torch spelling),
* ours      — ``mlopt::grouped_quant_linear_ieee``: one Triton kernel,
  dequant in registers, weight never materialized.

Same weights, same math, only the dequant path differs.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time
from pathlib import Path

import torch

import grouped_quant_op as gq  # noqa: E402

USE_OURS = {"on": False}
QUANT: dict[int, tuple] = {}


def quant_for(layer) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-group W4-quantize a layer's weight; cached per module."""
    key = id(layer)
    if key in QUANT:
        return QUANT[key]
    w = layer.weight.detach().float()
    if w.dim() != 2:
        QUANT[key] = None
        return None
    n, k = w.shape
    groups = max(1, k // 128)
    if k % groups:
        QUANT[key] = None
        return None
    wg = w.reshape(n, groups, k // groups)
    s = (wg.abs().amax(-1) / 7.0).clamp_min(1e-6)
    wq = torch.round(wg / s.unsqueeze(-1)).reshape(n, k).contiguous()
    s = s.contiguous()
    QUANT[key] = (wq, s, groups)
    return QUANT[key]


def reference(x, wq, scale, groups, bias):
    """Eager dequant path: build the float weight, then matmul."""
    n, k = wq.shape
    w = (wq.reshape(n, groups, k // groups) * scale.unsqueeze(-1)).reshape(
        n, k
    )
    return x @ w.transpose(0, 1) + bias


def install_patch() -> None:
    """Route vLLM's unquantized linear through the chosen dequant path."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    orig = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):  # noqa: ANN001
        q = quant_for(layer)
        if q is None:
            return orig(self, layer, x, bias)
        wq, scale, groups = q
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        b = (
            bias.float()
            if bias is not None
            else torch.zeros(wq.shape[0], device=x.device)
        )
        if USE_OURS["on"]:
            y = gq.grouped_quant_linear_ieee_op(flat, wq, scale, b)
        else:
            y = reference(flat, wq, scale, groups, b)
        return y.reshape(*shape[:-1], wq.shape[0])

    UnquantizedLinearMethod.apply = apply


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    install_patch()

    cfg = json.loads(Path(args.model, "config.json").read_text())
    torch.manual_seed(0)
    ids = torch.randint(1, int(cfg["vocab_size"]), (1, args.ctx // 2)).tolist()
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen, ignore_eos=True)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="float32",
        enforce_eager=True,
        gpu_memory_utilization=args.frac,
        max_model_len=args.ctx,
        seed=0,
        disable_log_stats=True,
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
        for on in ((False, True) if i % 2 == 0 else (True, False)):
            dt, tk = call(on)
            (ours if on else base).append(dt)
            toks.setdefault("ours" if on else "base", tk)
    USE_OURS["on"] = False

    mb, mo = stats.median(base), stats.median(ours)
    wins = sum(1 for a, b in zip(ours, base, strict=True) if a < b)
    print(f"\n{args.rounds} paired rounds · {len(QUANT)} linears quantized")
    print(f"  torch dequant : {mb*1e3:7.1f} ms  sd {stats.stdev(base)*1e3:5.1f}"
          f"  {args.gen/mb:6.1f} tok/s")
    print(f"  our fused op  : {mo*1e3:7.1f} ms  sd {stats.stdev(ours)*1e3:5.1f}"
          f"  {args.gen/mo:6.1f} tok/s")
    print(f"  paired: ours {100*(mo-mb)/mb:+.1f}% · "
          f"ours faster in {wins}/{args.rounds} · "
          f"identical tokens: {toks['ours'] == toks['base']}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "rounds": args.rounds,
                    "linears": len(QUANT),
                    "torch_ms": round(mb * 1e3, 2),
                    "ours_ms": round(mo * 1e3, 2),
                    "delta_pct": round(100 * (mo - mb) / mb, 2),
                    "ours_wins": wins,
                    "identical_tokens": toks["ours"] == toks["base"],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
