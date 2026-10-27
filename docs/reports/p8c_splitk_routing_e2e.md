> **WITHDRAWN — see [p13_retraction_splitk.md](p13_retraction_splitk.md).** The split-k path used here
> accumulates incorrectly (max abs error 3.8-5.6 vs reference); every speedup
> in this report is invalid. The regime analysis stands; the numbers do not.

# P8c — split-k, routing, and the end-to-end answer

## Split-k closed most of the gap

Splitting the K reduction across programs (atomic accumulate) with a
per-shape config sweep:

| layer | best config | ours | `awq_gemm` | vs vLLM |
|---|---|---|---|---|
| `down_proj` 896×4864 | bn64 bk64 split8 w1 | **55.9 µs** | 186.0 | **3.32×** |
| `k_proj` 128×896 | bn32 bk128 split8 w2 | 22.6 | 37.9 | 1.68× |
| `gate_proj` 4864×896 | bn128 bk64 split4 w2 | 60.0 | 53.5 | **0.89×** |

`down_proj` went from 2.16× to **3.32×** on config alone. `gate_proj`
stays a loss — vLLM's schedule is genuinely better for that shape.

## The routing table (measured, not assumed)

| geometry | layers | route | why |
|---|---|---|---|
| `down_proj` 4864→896 | 24 | **ours** | 3.32× over `awq_gemm` |
| `q_proj`/`o_proj` 896→896 | 48 | **ours** | 1.21× over `awq_gemm` |
| `gate_proj`/`up_proj` 896→4864 | 48 | `awq_gemm` | ours 0.89× |
| `k_proj`/`v_proj` 896→128 | 48 | **plain matmul** | 16.3 µs vs our 22.6 vs vLLM 37.9 |

## End-to-end: flat

Qwen2.5-0.5B-AWQ served by vLLM, 20 paired rounds, one engine, routing
applied inside `AutoAWQLinearMethod.apply`:

| arm | time | tok/s | |
|---|---|---|---|
| stock vLLM | 314.4 ms ±24.9 | 101.8 | |
| routed ours | 317.6 ms ±28.4 | 100.7 | **+1.0%, wins 12/20** |

**Token-identical.** No measurable speed difference.

## Why the kernel wins did not become a model win

1. **Dtype conversion.** The model runs fp16; our kernel takes fp32 and
   returns fp32, so every call pays `x.half→float` and `out.float→half`
   copies. That cost is not in the isolated benchmarks and it eats the
   kernel gain.
2. **Dilution.** Two of four geometries route to us; attention, sampling,
   KV cache and scheduling dominate the rest of the token.
3. **Precision mismatch was the warning.** I flagged in P8 that
   comparing fp32-ours against fp16-`awq_gemm` was not a fair fight. In a
   real model it isn't a fairness question any more — it's a cost.

**The fix is known and not done:** make the kernel fp16-in/fp16-out
(accumulate in fp32). That is what `awq_gemm` does, and until then the
end-to-end number cannot move.
