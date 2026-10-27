"""Where do the two arms diverge? (the -28.5% claim depends on this)

P17 measured 30 paired rounds at -28.5 % with 30/30 wins, but the greedy
token sequences were not identical — so before any speed claim stands,
the divergence has to be characterised: how many tokens differ, from
which position, and is it a rounding difference or a bug.
"""

from __future__ import annotations

import torch

from .vllm_awq_e2e_fp16 import USE_OURS, install_patch


def main() -> int:
    from vllm import LLM, SamplingParams

    install_patch()
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        quantization="awq",
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
    sp = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)

    def run(on: bool):
        USE_OURS["on"] = on
        out = llm.generate(ids, sampling_params=sp, use_tqdm=False)
        return list(out[0].outputs[0].token_ids)

    for _ in range(2):  # warm
        run(False)
        run(True)

    stock = run(False)
    ours = run(True)
    USE_OURS["on"] = False

    first = next(
        (
            i
            for i, (a, b) in enumerate(zip(stock, ours, strict=True))
            if a != b
        ),
        None,
    )
    same = sum(1 for a, b in zip(stock, ours, strict=True) if a == b)
    print(
        f"tokens: {len(stock)} | identical: {same} | "
        f"first divergence at index: {first}"
    )
    if first is not None:
        lo = max(0, first - 2)
        print(f"  stock[{lo}:{first + 4}] = {stock[lo : first + 4]}")
        print(f"  ours [{lo}:{first + 4}] = {ours[lo : first + 4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
