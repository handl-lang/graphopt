> **WITHDRAWN — see [p13_retraction_splitk.md](p13_retraction_splitk.md).** The split-k path used here
> accumulates incorrectly (max abs error 3.8-5.6 vs reference); every speedup
> in this report is invalid. The regime analysis stands; the numbers do not.

# P8b — the whole model, machine-picked configs

All four distinct geometries in Qwen2.5-0.5B-AWQ (168 quantized linears).
For each shape the machine sweeps its own kernel configs — 48
combinations of `BLOCK_N × BLOCK_K × num_warps` — and keeps the fastest
by measurement. The config choice is the machine's, not mine.

One token (decode). `torch` = float weight pre-built, plain matmul;
`vllm` = `awq_gemm`.

| layer | shape | layers | best config | torch | ours | `awq_gemm` | vs torch | vs vLLM |
|---|---|---|---|---|---|---|---|---|
| `down_proj` | 896×4864 in | 24 | bn32 bk256 w4 | 197.4 | **87.1** | 187.8 | **2.27×** | **2.16×** |
| `gate_proj` | 4864×896 in | 48 | bn64 bk64 w1 | 197.0 | **63.0** | 53.6 | **3.13×** | 0.85× |
| `k_proj` | 128×896 in | 48 | bn16 bk256 w4 | **16.3** | 25.2 | 38.5 | 0.65× | 1.53× |
| `q_proj` | 896×896 in | 48 | bn32 bk256 w4 | 45.5 | **31.7** | 38.3 | **1.43×** | **1.21×** |

**120 of 168 layers beat the plain matmul; 120 of 168 beat `awq_gemm`.**

## What tuning changed

The untuned run (`p8_awq_fused.md`) lost `q_proj` by 5 % and won
`down_proj` by 1.25×. Letting the machine pick per-shape configs turned
those into **1.21×** and **2.16×**. The kernel did not change — only the
config. That is the cheapest kind of win and exactly what an autotuning
referee is for.

## Where it loses, and why that is fine

- **`gate_proj` (0.85× vs `awq_gemm`)** — a large-output shape where
  vLLM's split-k schedule is genuinely better. It still beats plain
  torch by 3.13×.
- **`k_proj` (0.65× vs torch)** — 128 outputs is a tiny, launch-bound
  matmul; the plain matmul is faster than any dequant kernel, ours or
  vLLM's (`awq_gemm` is 2.4× slower still). The right answer here is
  **not to use our kernel** — a dispatch decision, not a failure.

That is the honest shape of the result: a per-shape win with a
per-shape *routing* rule, which is what a real compiler does.

## The gap that remains (question 3)

The machine chose the **configs**. It did not choose the **algorithm** —
I hand-wrote the int4 GEMV. So the evidence is: *the substrate can host
and tune a kernel that beats a hand-tuned production kernel*, not yet
*the loop discovers such a kernel unaided*. Closing that gap is the
minting-loop experiment, and it is the one that would prove the RL claim.
