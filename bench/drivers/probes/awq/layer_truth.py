"""Layer-level truth: our kernel vs awq_gemm on the same x.

P18 diverges at token 1 even when only `down_proj` (no bias) is routed,
so the disagreement is numeric, not plumbing. This computes both on the
same input for the same layer and prints the difference — the smallest
experiment that distinguishes "rounding" from "wrong function".
"""

from __future__ import annotations

import torch

from .awq_packing import to_standard_order
from .kernel_split2 import gemv_split2

STASH: dict[int, tuple] = {}
DONE = {"n": 0}


def install() -> None:
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQLinearMethod,
    )

    orig_pwal = AutoAWQLinearMethod.process_weights_after_loading
    orig_apply = AutoAWQLinearMethod.apply

    def pwal(self, layer):  # noqa: ANN001
        if getattr(layer, "qweight", None) is not None:
            STASH[id(layer)] = (
                to_standard_order(layer.qweight.detach()),
                to_standard_order(layer.qzeros.detach()),
                layer.scales.detach().clone(),
            )
        return orig_pwal(self, layer)

    def apply(self, layer, x, bias=None):  # noqa: ANN001
        snap = STASH.get(id(layer))
        out = orig_apply(self, layer, x, bias)
        if snap is not None and DONE["n"] < 2 and x.shape[0] == 1:
            qw, qz, sc = snap
            n, k = sc.shape[1], qw.shape[0]
            if (k, n) in ((4864, 896), (896, 896)):
                ours = gemv_split2(
                    x.reshape(-1).to(torch.float16).contiguous(),
                    qw,
                    qz,
                    sc,
                    n,
                    k,
                    64,
                    64,
                    8,
                    2,
                )
                diff = (out.float() - ours.float()).abs()
                DONE["n"] += 1
                print(
                    f"[{k}x{n}] max|d| {float(diff.max()):.4f}  "
                    f"mean|d| {float(diff.mean()):.5f}  "
                    f"|stock| mean {float(out.abs().mean()):.3f}  "
                    f"cos {torch.nn.functional.cosine_similarity(out.flatten().float(), ours.flatten().float(), dim=0).item():.4f}"
                )
        return out

    AutoAWQLinearMethod.process_weights_after_loading = pwal
    AutoAWQLinearMethod.apply = apply


def main() -> int:
    import vllm
    from vllm import LLM

    install()
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        quantization="awq",
        tokenizer="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        max_model_len=64,
        seed=0,
        disable_log_stats=True,
    )
    torch.manual_seed(0)
    ids = torch.randint(1, 151936, (1, 32)).tolist()
    llm.generate(
        ids,
        sampling_params=vllm.SamplingParams(
            max_tokens=4, ignore_eos=True
        ),
        use_tqdm=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
