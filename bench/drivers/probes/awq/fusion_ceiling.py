"""Ceiling check: how much of a vLLM decode step is the MLP activation?

Before building a fused ``silu_and_mul + down_proj`` kernel, measure the
fraction of a decode step that any such fusion could possibly remove:

* the activation op itself (vLLM's fused CUDA kernel) at decode and
  prefill shapes,
* one kernel launch (an empty Triton kernel), which is the floor for
  any fusion that removes a launch,
* total decode time per token from the measured engine run.

If the op is ~1% of the token, fusing it cannot return more than ~1%.
"""

from __future__ import annotations

import json
import statistics as stats
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
import vllm._custom_ops  # noqa: F401 — registers torch.ops._C.*


@triton.jit
def _noop(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        x_ptr + offs,
        tl.load(x_ptr + offs, mask=offs < n),
        mask=offs < n,
    )


def timeit(fn, reps: int = 200, warm: int = 20) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return stats.median(ts) * 1e6  # µs


def main() -> int:
    cfg = json.loads(
        Path(
            "/home/luis/GitHub/handl-lang/graphopt/bench/results/"
            "vllm_gate3/hf_stories15m/eager/config.json"
        ).read_text()
    )
    hidden = int(cfg["intermediate_size"])
    out = {}

    op = torch.ops._C.silu_and_mul
    for m in (1, 8, 64, 256):
        x = torch.randn(
            m, 2 * hidden, device="cuda", dtype=torch.float32
        )
        dst = torch.empty(m, hidden, device="cuda", dtype=torch.float32)
        us = timeit(lambda: op(dst, x))
        out[f"silu_and_mul_m{m}_us"] = round(us, 2)
        print(f"silu_and_mul  M={m:4d}  {us:7.2f} µs")

    buf = torch.zeros(1024, device="cuda", dtype=torch.float32)
    us = timeit(lambda: _noop[(1,)](buf, 1024, BLOCK=1024))
    out["empty_launch_us"] = round(us, 2)
    print(f"empty triton launch      {us:7.2f} µs")

    # decode step budget measured on this box: 111.6 ms / 32 tokens
    token_ms = 111.6 / 32
    layers = int(cfg["num_hidden_layers"])
    act = out["silu_and_mul_m1_us"]
    share = 100 * layers * act / (token_ms * 1000)
    out["layers"] = layers
    out["token_ms"] = round(token_ms, 2)
    out["activation_share_pct"] = round(share, 2)
    print(
        f"\ndecode step {token_ms:.2f} ms · {layers} layers · "
        f"activation {act:.1f} µs × {layers}"
    )
    print(f"-> the whole activation is {share:.2f}% of a token")
    print(
        "-> a fused silu+mul+down kernel can return at most that much "
        "(less, minus its own cost)"
    )

    Path(
        "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
        "p5_vllm_fusion_ceiling.json"
    ).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
