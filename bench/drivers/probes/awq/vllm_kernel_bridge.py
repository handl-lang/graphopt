"""Can our kernel live inside vLLM? SwiGLU, swapped in the live model.

vLLM's ``SiluAndMul`` calls a hand-written CUDA kernel
(``torch.ops._C.silu_and_mul``).  This replaces its ``forward_cuda`` with
a **Triton kernel** and toggles between the two per round, so both arms
run in the *same engine, same memory, same model* — only the kernel
differs.  That is the strongest paired design available on one GPU.

    python /tmp/vllm_kernel_bridge.py --ctx 64 --rounds 30
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

USE_OURS = {"on": False}


@triton.jit
def _swiglu_kernel(x_ptr, out_ptr, total, d, BLOCK: tl.constexpr):
    """silu(x[..., :d]) * x[..., d:] in one launch, no copies."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total
    row = idx // d
    col = idx - row * d
    base = row * 2 * d + col
    g = tl.load(x_ptr + base, mask=mask, other=0.0)
    u = tl.load(x_ptr + base + d, mask=mask, other=0.0)
    y = (g * tl.sigmoid(g)) * u
    tl.store(out_ptr + idx, y, mask=mask)


def our_swiglu(x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for ``SiluAndMul``."""
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    total = out.numel()
    grid = (triton.cdiv(total, 1024),)
    _swiglu_kernel[grid](x, out, total, d, BLOCK=1024)
    return out


def install_patch() -> None:
    """Route vLLM's SiluAndMul through ``our_swiglu`` when the flag is on."""
    from vllm.model_executor.layers.activation import SiluAndMul

    orig = SiluAndMul.forward_cuda

    def forward_cuda(self, x):  # noqa: ANN001
        if not USE_OURS["on"]:
            return orig(self, x)
        return our_swiglu(x)

    SiluAndMul.forward_cuda = forward_cuda


def verify(d: int, tokens: int = 4096) -> float:
    """Max abs error of our kernel vs the spelled reference."""
    x = torch.randn(tokens, 2 * d, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    got = our_swiglu(x)
    return float((ref - got).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, action="append", required=True)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.model, "config.json").read_text())
    d_mlp = int(cfg["intermediate_size"])
    err = verify(d_mlp)
    print(f"kernel verified: max|Δ| = {err:.3e} on d={d_mlp}")

    from vllm import LLM, SamplingParams

    install_patch()

    results = {}
    for ctx in args.ctx:
        torch.manual_seed(0)
        ids = torch.randint(
            1, int(cfg["vocab_size"]), (1, ctx // 2)
        ).tolist()
        sp = SamplingParams(
            temperature=0.0, max_tokens=args.gen, ignore_eos=True
        )
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            dtype="float32",
            enforce_eager=True,
            gpu_memory_utilization=args.frac,
            max_model_len=ctx,
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

        base_t, ours_t, toks = [], [], {}
        for i in range(args.rounds):
            for on in ((False, True) if i % 2 == 0 else (True, False)):
                dt, tk = call(on)
                (ours_t if on else base_t).append(dt)
                toks.setdefault("ours" if on else "base", tk)

        diffs = [o - b for o, b in zip(ours_t, base_t, strict=True)]
        med_b, med_o = stats.median(base_t), stats.median(ours_t)
        ours_wins = sum(1 for x in diffs if x < 0)
        use_ours = USE_OURS["on"]
        USE_OURS["on"] = False

        print(f"\nctx={ctx}  {args.rounds} paired rounds, one engine")
        print(f"  vLLM CUDA kernel : {med_b*1e3:7.1f} ms  "
              f"sd {stats.stdev(base_t)*1e3:5.1f}  {args.gen/med_b:6.1f} tok/s")
        print(f"  our Triton kernel: {med_o*1e3:7.1f} ms  "
              f"sd {stats.stdev(ours_t)*1e3:5.1f}  {args.gen/med_o:6.1f} tok/s")
        print(f"  paired: ours {100*(med_o-med_b)/med_b:+.1f}% · "
              f"ours faster in {ours_wins}/{args.rounds} rounds · "
              f"identical tokens: {toks['ours'] == toks['base']}")

        results[f"{ctx}"] = {
            "vllm_cuda_ms": round(med_b * 1e3, 2),
            "ours_triton_ms": round(med_o * 1e3, 2),
            "sd_cuda_ms": round(stats.stdev(base_t) * 1e3, 2),
            "sd_ours_ms": round(stats.stdev(ours_t) * 1e3, 2),
            "delta_pct": round(100 * (med_o - med_b) / med_b, 2),
            "ours_wins": ours_wins,
            "rounds": args.rounds,
            "identical_tokens": toks["ours"] == toks["base"],
            "kernel_max_abs_err": err,
        }
        del llm
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
