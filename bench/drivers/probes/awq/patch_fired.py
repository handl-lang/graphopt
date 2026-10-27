"""Did the e2e patch actually fire?

P15 measured a flat model and the profile (P16) says the quantized gemms
are 51.6 % of device time — so a 3× kernel win should have shown ~34 %.
A flat result with that much headroom means the substitution probably
never happened. This counts calls: stock path, our kernel, and fallbacks,
per generate.
"""

from __future__ import annotations

import torch

COUNT = {"ours": 0, "stock": 0, "no_cfg": 0, "not_m1": 0}

from .kernel_split2 import gemv_split2  # noqa: E402
from .vllm_awq_e2e_fp16 import ROUTE  # noqa: E402

USE_OURS = {"on": True}


def install_counter_patch() -> None:
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQLinearMethod,
    )

    orig = AutoAWQLinearMethod.apply

    def apply(self, layer, x, bias=None):  # noqa: ANN001
        qw = getattr(layer, "qweight", None)
        if not USE_OURS["on"] or qw is None:
            COUNT["stock"] += 1
            return orig(self, layer, x, bias)
        sc = layer.scales
        n, k = sc.shape[1], qw.shape[0]
        cfg = ROUTE.get((k, n))
        if cfg is None:
            COUNT["no_cfg"] += 1
            return orig(self, layer, x, bias)
        if x.shape[0] != 1:
            COUNT["not_m1"] += 1
            return orig(self, layer, x, bias)
        COUNT["ours"] += 1
        bn, bk, split, wp = cfg
        out = gemv_split2(
            x.reshape(-1).to(torch.float16).contiguous(),
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
            out = out + bias.to(torch.float16)
        return out

    AutoAWQLinearMethod.apply = apply


def main() -> int:
    from vllm import LLM, SamplingParams

    install_counter_patch()
    llm = LLM(
        quantization="awq",  # force the plain AWQ path (not Marlin,
        # which repacks the weights and has its own apply())
        model="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        tokenizer="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=128,
        seed=0,
        disable_log_stats=True,
    )
    torch.manual_seed(0)
    ids = torch.randint(1, 151936, (1, 32)).tolist()
    sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)

    for k in COUNT:
        COUNT[k] = 0
    llm.generate(ids, sampling_params=sp, use_tqdm=False)
    total = sum(COUNT.values())
    print(f"linear calls in one 8-token generate: {total}")
    for k, v in COUNT.items():
        share = 100 * v / total if total else 0
        print(f"  {k:<8} {v:6d}  ({share:5.1f}%)")
    routed = COUNT["ours"] / total if total else 0.0
    print(
        f"\nrouted to our kernel: {100 * routed:.1f}% of linear calls"
    )
    print("expected if the patch works: ~75% (3 of 4 geometries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
