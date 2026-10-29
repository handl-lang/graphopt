"""Per-call cost of Marlin (the default kernel) — counts, not averages.

P16 reported kernel totals, so "Marlin ≈ 105 µs per linear" was a division
by an assumed call count. This records `self_device_time_total / count`
per kernel, which is the number needed to compare against our kernel's
58.9 µs cold on `down_proj`.

Runs the **default** path (Marlin), so the number is what a user gets from
`vllm serve` — not the batch-invariant path the 29 % claim came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

OUT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p19_marlin_per_call.json"
)


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

    for _ in range(3):
        llm.generate(ids, sampling_params=sp, use_tqdm=False)

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        for _ in range(5):
            llm.generate(ids, sampling_params=sp, use_tqdm=False)

    rows = []
    for evt in prof.key_averages():
        us = getattr(evt, "self_device_time_total", 0.0) or 0.0
        if us <= 0:
            continue
        count = int(getattr(evt, "count", 0) or 0)
        rows.append(
            {
                "name": evt.key.split("<")[0].strip()[:60],
                "us_total": round(us, 1),
                "calls": count,
                "us_per_call": round(us / count, 2) if count else None,
            }
        )
    rows.sort(key=lambda r: -r["us_total"])

    payload = {
        "tokens": 5 * 32,
        "kernels": rows[:12],
        "marlin_us_per_call": [
            r["us_per_call"]
            for r in rows
            if "marlin" in r["name"].lower()
        ],
        "gemv2t_us_per_call": next(
            (
                r["us_per_call"]
                for r in rows
                if "gemv2t" in r["name"].lower()
            ),
            None,
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'kernel':<40}{'total ms':>10}{'calls':>8}{'µs/call':>10}")
    for r in rows[:10]:
        print(
            f"{r['name']:<40}{r['us_total'] / 1e3:>10.1f}"
            f"{r['calls']:>8}{r['us_per_call']:>10.1f}"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
