"""Where does a decode token's time go?

The end-to-end test is flat (P15). Two explanations fit: dilution (the
linears are a small slice of a token) or something else eating the gain.
This measures it instead of guessing — a CUDA-activity profile of real
generation, bucketed by what the kernel is doing.

The output is the ceiling for kernel work on this workload: if the
quantized linears are X % of a token and we make them k× faster, the best
possible end-to-end gain is X·(1 − 1/k).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p16_token_time.json"
)

#: bucket by substring, first match wins
BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "gemm/quant (the part we optimize)",
        ("awq", "gemm", "gemv", "matmul", "cutlass", "w4a16"),
    ),
    ("attention", ("attention", "flash", "fmha", "softmax", "paged")),
    ("norm/act", ("rms", "norm", "silu", "gelu", "mul")),
    ("sampling/argmax", ("sampl", "argmax", "topk", "sort", "random")),
    ("memory/copies", ("memcpy", "elementwise", "copy", "cast")),
)


def bucket_of(name: str) -> str:
    low = name.lower()
    for label, keys in BUCKETS:
        if any(k in low for k in keys):
            return label
    return "other"


def main() -> int:
    from vllm import LLM, SamplingParams

    llm = LLM(
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
    sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)

    for _ in range(3):  # warm caches and allocator
        llm.generate(ids, sampling_params=sp, use_tqdm=False)

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        for _ in range(5):
            llm.generate(ids, sampling_params=sp, use_tqdm=False)

    totals: dict[str, float] = defaultdict(float)
    top: list[dict] = []
    for evt in prof.key_averages():
        us = getattr(evt, "self_device_time_total", 0.0) or 0.0
        if us <= 0:
            continue
        totals[bucket_of(evt.key)] += us
        top.append({"name": evt.key, "us": round(us, 1)})

    grand = sum(totals.values()) or 1.0
    rows = [
        {
            "bucket": b,
            "us": round(v, 1),
            "pct": round(100 * v / grand, 1),
        }
        for b, v in sorted(totals.items(), key=lambda kv: -kv[1])
    ]
    top.sort(key=lambda r: -r["us"])

    payload = {
        "tokens": 5 * 32,
        "total_device_us": round(grand, 1),
        "buckets": rows,
        "top_kernels": top[:15],
        "ceiling_if_3x_on_gemm": None,
    }
    gemm = next(
        (r for r in rows if r["bucket"].startswith("gemm")),
        {"pct": 0.0},
    )
    payload["ceiling_if_3x_on_gemm"] = round(
        gemm["pct"] * (1 - 1 / 3), 1
    )
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"device time over {payload['tokens']} tokens: "
        f"{grand / 1e3:.1f} ms"
    )
    for r in rows:
        print(f"  {r['bucket']:<38} {r['pct']:5.1f}%")
    print(
        f"\nceiling for a 3x gemm win: "
        f"{payload['ceiling_if_3x_on_gemm']}% of a token"
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
