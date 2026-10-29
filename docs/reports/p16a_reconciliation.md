# P16a — reconciliation with P19/P22 (superseding parts of this report)

Written after P19–P22 landed from a parallel session. Three of this
report's statements do not survive; two do.

## Do not survive

1. **"Exact to 1e-3"** — the reference (`unpack_ref`) and the kernel used
   the *same* native AWQ nibble order, so they agreed with each other and
   with neither `awq_gemm` (cosine **0.21**). P22 repaired it with
   `to_standard_order` plus a cosine gate against `awq_gemm`; every
   re-derived output is cosine 1.000000. **My exactness claim was
   circular** and is withdrawn.
2. **"We beat the kernel vLLM actually runs by 1.16–1.20×"** — the Marlin
   arm here was a *hand-constructed* call (no zero-points, my own
   workspace sizing) at 70.8 µs (`down_proj`) and 40.8 µs (`q_proj`). P19
   profiled the real default path: **96 Marlin calls/token at 26.2 µs
   average**, and the model has **24 layers, not 6** as I had assumed.
   A shape-matched comparison against the live path has **not been done**,
   so no lead over Marlin can be claimed.
3. **"1.20× over production" in the README** — same reason; now pointed at
   this note.

## Survive

- **The reflex to distrust a good number.** P13 (atomics), P16 (patch
  never fired) and this note are three instances where the result was
  wrong in the flattering direction, and each was caught by checking the
  instrument rather than the claim.
- **The referee module** (`bench/benchkit/regime.py`) — P22 kept the
  eviction + blocked-randomization design and paired it with a CUDA-graph
  timer and a cosine gate; the module is in the same lineage.

## What is now the authoritative record

| question | answer | source |
|---|---|---|
| is our kernel correct? | yes — cosine 1.000000 vs `awq_gemm` | `p22_record_repaired.md` |
| how fast vs `awq_gemm`? | **3.22× deploy / 3.73× graph** on `down_proj`; **0.60×** on `gate_up`; 1.05×/2.14× on `qkv` | p22 |
| how fast vs Marlin? | **unmeasured, shape-matched** — Marlin averages 26.2 µs/call live; our per-shape numbers are not comparable | `p19_marlin_is_the_default.md` |
| can we beat Marlin on its tuned shapes? | **blocked by physics** — the 4× fp16→int4 budget is nearly spent | p19 |

## The honest one-liner

Our kernel is correct and beats the *generic* AWQ kernel by 3.2× on the
shape where memory dominates; whether it beats the kernel vLLM actually
ships is **still an open measurement**, and the physics suggests the
headroom there is small — the credible targets are the shapes Marlin
tunes badly (small `size_n`, unattended formats, batch > 64).
