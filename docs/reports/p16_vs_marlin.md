# P16 — the right baseline at last: `marlin_gemm`, and a 1.2× lead

Three things were wrong in the earlier end-to-end work, and each one
flattered the result:

1. **The kernel was numerically broken** (fp16 atomics) — retracted in P13.
2. **The vLLM patch never fired.** `AutoAWQLinearMethod.apply` is called
   zero times: vLLM repacks AWQ checkpoints for **Marlin** and serves
   through `marlin_gemm`. The first six end-to-end runs compared vLLM
   against itself.
3. **The baseline was the wrong kernel.** We raced `awq_gemm` — the
   generic AWQ kernel — while production runs Marlin, a tensor-core
   kernel roughly 2.5× faster than `awq_gemm`.

## The honest head-to-head

Same real layers of Qwen2.5-0.5B-AWQ, cold (deployment) regime, one
token:

| layer | ours | `awq_gemm` | **`marlin_gemm`** (what runs) | vs Marlin |
|---|---|---|---|---|
| `down_proj` 896×4864 | **58.9 µs** | 174.1 | **70.8** | **1.20×** |
| `q_proj` 896×896 | **35.2** | 47.7 | **40.8** | **1.16×** |

**We beat the production kernel by 16–20 %**, not by 3×. Still a win, and
now against the right opponent.

## Caveats that must travel with the number

- **Marlin's arm here is approximate.** It was called without zero-point
  tensors (`b_zeros=None`), so its arithmetic differs from the checkpoint
  path. The *timing* is representative — same shapes, dtypes, layout — but
  this is not a correctness-verified comparison of the two arms.
- **Ours is exact** (max error ~1e-3 vs the dequantized reference).
- **Two shapes**, one GPU, decode only. `gate_proj` was not measured
  against Marlin.

## What this does to the story

| claim | status |
|---|---|
| "we beat vLLM's kernel by 2.96×" | **wrong opponent** — that was `awq_gemm` |
| "we beat the kernel vLLM actually runs" | **yes, by 1.16–1.20×, cold, on two shapes** |
| "the model gets faster" | **unmeasured** — the patch never fired; a correct injection must target Marlin's call site or bypass it |
| "our kernel is exact" | **yes** |

The margin is smaller than the earlier reports implied and it is *real*,
which is the only trade worth making. It also changes the pitch: a 1.2×
lead on a production kernel, repeated across many shapes, is a credible
compiler story; a 3× lead against the wrong baseline was not.

## Next, precisely

1. **Validate the Marlin arm** with zero-points so the comparison is
   like-for-like on both speed and accuracy.
2. **Inject at the right site**: patch `marlin_gemm`'s caller (or the
   `MarlinLinearMethod.apply`) and prove with `COUNT` that it fires
   before measuring anything.
3. **Then** the end-to-end question is finally testable — and it is the
   only thing standing between "a faster kernel" and "a faster model".
