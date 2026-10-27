"""Time the 24 quantized linears alone, inside the live vLLM model.

The end-to-end test diluted our op into attention, sampling and the
scheduler. This loads the model, lets one generate populate the
per-group-quantized weights, then times **just the linear layers** on
their real weights and shapes:

    baseline : x @ (w_q * scale).T + bias   (materializes the weight)
    ours     : mlopt::grouped_quant_linear_ieee  (one Triton kernel)
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time

import torch

import grouped_quant_op as gq  # noqa: E402

QUANT: dict[int, tuple] = {}
LAYERS: dict[int, object] = {}


def quant_for(layer):
    key = id(layer)
    if key in QUANT:
        return QUANT[key]
    w = layer.weight.detach().float()
    if w.dim() != 2 or w.shape[1] % 128:
        QUANT[key] = None
        return None
    n, k = w.shape
    groups = max(1, k // 128)
    wg = w.reshape(n, groups, k // groups)
    s = (wg.abs().amax(-1) / 7.0).clamp_min(1e-6)
    wq = torch.round(wg / s.unsqueeze(-1)).reshape(n, k).contiguous()
    QUANT[key] = (wq, s.contiguous(), groups)
    LAYERS[key] = layer
    return QUANT[key]


def install_patch() -> None:
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
        n, k = wq.shape
        w = (wq.reshape(n, groups, k // groups) * scale.unsqueeze(-1)).reshape(
            n, k
        )
        return (flat @ w.transpose(0, 1) + b).reshape(*shape[:-1], n)

    UnquantizedLinearMethod.apply = apply


def timeit(fn, reps=50, warm=10) -> float:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, action="append", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tokens_list = args.tokens or [1, 32]

    from vllm import LLM, SamplingParams

    install_patch()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="float32",
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        max_model_len=64,
        seed=0,
        disable_log_stats=True,
    )
    llm.generate(
        [[1, 2, 3, 4]], sampling_params=SamplingParams(max_tokens=2),
        use_tqdm=False,
    )
    print(f"quantized linears discovered: {len(QUANT)}")

    results = {}
    for tokens in tokens_list:
        tot_ref = tot_ours = 0.0
        for key, q in QUANT.items():
            if q is None:
                continue
            wq, scale, groups = q
            n, k = wq.shape
            b = torch.zeros(n, device="cuda")
            x = torch.randn(tokens, k, device="cuda")

            def ref():
                w = (
                    wq.reshape(n, groups, k // groups)
                    * scale.unsqueeze(-1)
                ).reshape(n, k)
                return x @ w.transpose(0, 1) + b

            def ours():
                return gq.grouped_quant_linear_ieee_op(x, wq, scale, b)

            tot_ref += timeit(ref)
            tot_ours += timeit(ours)

        results[tokens] = {
            "tokens": tokens,
            "linears": sum(1 for q in QUANT.values() if q),
            "ref_us": round(tot_ref, 1),
            "ours_us": round(tot_ours, 1),
            "vs_ref": round(tot_ref / tot_ours, 3),
        }
        print(
            f"tokens={tokens:3d}  torch dequant {tot_ref:8.1f} µs   "
            f"ours {tot_ours:8.1f} µs   -> {tot_ref/tot_ours:.2f}x"
        )

    if args.out:
        open(args.out, "w").write(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
