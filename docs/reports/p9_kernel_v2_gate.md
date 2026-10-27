# P9 — kernel v2: small shapes better, big shape unchanged (gate failed)

The Phase 1 gate was `down_proj` < 35 µs (≥70 % of the 24.6 µs roofline).
**It did not pass.** v2 fixes were: load each packed word once and unpack
all 8 nibbles, contiguous word blocks, fp16 in/out with fp32 accumulate,
split-k only where K is large.

| layer | roof | v1 | **v2** | vLLM | v2 % of roof | gate |
|---|---|---|---|---|---|---|
| `down_proj` 896×4864 | 24.6 µs | 55.9 | **58.2** | 196.9 | **42.3 %** | **FAIL** (<35 µs) |
| `q_proj` 896×896 | 4.5 | 31.7 | **28.7** | 35.1 | 15.8 % | improved |
| `k_proj` 128×896 | 0.7 | 22.6 | **20.4** | 35.3 | 3.3 % | improved |
| `gate_proj` 4864×896 | 24.6 | 60.0 | **55.7** | 52.1 | 44.1 % | still 0.94× vs vLLM |

fp16 in/out and the word-once load helped the small shapes (q −9 %,
k −10 %) and did **nothing** for the big one. The 3.38× over `awq_gemm`
survives, but it is 3.38× over a kernel that is at 13 % of roofline —
not a claim about being near the physics.

## What the failure rules out

- **Occupancy** is not it: the winning `down_proj` config is
  `bn=64 split=4` → 56 CTAs on a 16-SM card.
- **Bandwidth** is not it either, or the word-once load would have moved
  it.
- **The remaining cost is the unpack arithmetic and the reduction**:
  per element we do two shifts, two masks, a subtract, two multiplies
  and a `tl.sum` across `BLOCK_K`. That is ~6 ops on 4.36 M elements,
  which should be single-digit µs — we are at 58.

Most likely cause: the 3-D `reshape(BLOCK_K, BLOCK_N)` after the shift
generates poor code, and `tl.sum` over the K axis each iteration
serialises the reduction instead of accumulating a dot per lane.

## Per the pre-registered criterion

The plan says: *"hand-tuning cannot reach 70 % of roofline → the whole
thread is dead"*. We are at 42 %, so the honest reading is that **this
kernel shape is not the road to the roofline**. Either the unpack needs a
fundamentally different formulation (8 outputs per lane, no 3-D reshape,
vectorised 128-bit word loads), or the GEMV formulation itself is wrong
for this hardware and the tiled path should be revisited at fp16.

What is *not* affected: the competitive result (3.3× over `awq_gemm`),
the exactness, and the fp16 fix that the end-to-end test needed. What is
affected: the claim that we are close to the hardware limit — we are not,
and neither is vLLM.
