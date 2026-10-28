# P19 — Marlin is the default, and it is near the roofline

Per-call costs from a live default-path run (vLLM, `marlin`, 160 tokens):

| kernel | calls/token | µs/call |
|---|---|---|
| `gemv2T` (LM head, 896×151936) | 1 | **2929** |
| `marlin::Marlin` | 96 | **26.2** |

Per token: LM head **2.93 ms (49 %)**, all 24 quantized linears **2.53 ms (42 %)**.

## What this corrects

- The earlier “Marlin ≈ 105 µs per linear” was my error (divided by 24
  calls/token instead of 96). **This model has 24 layers, not 6.**
- At ~26 µs/call Marlin is **~2× faster than our kernel** (35–61 µs across
  the four shapes), not slower.
- The 29.2 % end-to-end win was against `VLLM_BATCH_INVARIANT=1`, a
  globally degraded path (plain AWQ). Default vLLM is ~87–95 tok/s; ours
  on that degraded path was 38.8.

## What the literature says

The Marlin README states the physics and the achievement:

> “less than 25-50 … MACs per 4-bit weight … theoretically … near ideal
> **4× speedup** over FP16”

> “close to **maximum (4×)** … up to batchsizes of 16-32”

And the maintainers, in [issue #2](https://github.com/IST-DASLab/marlin/issues/2):
“for autoregressive decoding marlin would be the **best in class kernel
available**”, tuned for the **medium batch range** (above 64 it replays the
batch-64 kernel).

So the 4× budget is already almost entirely spent. **Beating Marlin on its
tuned shapes is blocked by physics, not by effort.**

## Where a win is still credible

1. **Small `size_n`.** vLLM's own log: *“Marlin kernel can achieve better
   performance for small size_n with experimental use_atomic_add”* — and
   `k_proj`/`v_proj` are N=128. A documented compromise, not a secret.
2. **Unattended shapes and formats** — the same argument that produced the
   `awq_gemm` margin.
3. **Above batch 64**, where Marlin replays a batch-64 kernel.

## Consequence for the plan

M1's gate (“match hand-written Triton, parity on `gate_proj`”) is right and
achievable. **“Beat Marlin on its tuned shapes” should not be a goal** — it
is a claim physics refuses. The honest target set is: parity where Marlin is
tuned, wins where its tuning is a compromise.
