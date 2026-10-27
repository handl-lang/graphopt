# P14 — the correct split kernel: 3 of 4 shapes beat `awq_gemm`

After the P13 retraction, the split path was rebuilt as a **two-stage
reduction** (each K-slice stores its partial into a `[SPLIT, N]` fp32
buffer, a tiny kernel sums them) instead of fp16 atomics. Verified exact
to 1e-3 for split = 1, 2, 4, 8.

Measured **cold** (deployment regime: weights evicted, one call, medians
over 15 shuffled rounds), configs searched per shape with the exactness
gate applied *before* the timing:

| layer | ours | `awq_gemm` | torch | vs vLLM | vs torch | % of roofline |
|---|---|---|---|---|---|---|
| `down_proj` 896×4864 | **58.9 µs** | 174.1 | 120.2 | **2.96×** | **2.04×** | 41.8 % |
| `q_proj` 896×896 | **35.2** | 47.7 | 41.1 | **1.35×** | **1.17×** | 12.9 % |
| `k_proj` 128×896 | **38.5** | 58.7 | 31.2 | **1.52×** | 0.81× | 1.7 % |
| `gate_proj` 4864×896 | 60.6 | **57.8** | 115.9 | 0.95× | **1.91×** | 40.6 % |

**Three of four shapes beat the hand-written production kernel in the
deployment regime, with exact arithmetic.** The fourth is a 5 % loss to
`awq_gemm` but a 1.9× win over plain matmul.

Best configs were all split-K (4–8) with `bn=32–64`, which is what the
broken version had been *pretending* to do — the atomics were both wrong
and, oddly, not much faster.

## Why this is now credible where P8–P12 were not

1. **Correctness is gated per candidate, immediately before timing.** The
   retraction's root cause was a gate applied to a different code path.
2. **The regime is the deployment one** — eviction, one call, shuffled
   order — not a warm loop.
3. **Every row has a roofline reference.** `down_proj` at 41.8 % of
   24.6 µs says the number is physically plausible. A 2.96× win that was
   also 5× under the roofline would be a bug signal, not a result.

## What is still not good enough

- **41.8 % of roofline** on the winning shape: 2.4× is still on the table.
- **`k_proj` at 1.7 %** — launch-bound; 38.5 µs for 57 KB of weights is
  almost all overhead, and plain matmul (31.2 µs) is still better.
- **`gate_proj` loses** where N is large and K small; the tile shape that
  wins elsewhere is wrong there.
- **The end-to-end model number is still unmeasured with a correct
  kernel.** Every e2e run so far used the broken path, so the flat result
  says nothing yet and must be redone.
