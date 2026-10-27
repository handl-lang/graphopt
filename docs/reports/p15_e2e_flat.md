# P15 — end-to-end with the corrected kernel: flat, and now we know why

The kernel is correct now (two-stage reduction, exact to 1e-3) and wins
three of four layer shapes cold. So the model-level test was finally
fair. Three paired runs, same model, same routing, increasing power:

| rounds | stock | ours | delta | ours wins | tokens |
|---|---|---|---|---|---|
| 20 (broken kernel) | 314.3 ms | 307.7 | −2.1 % | 9/20 | identical |
| 24 (corrected) | 337.0 | 326.5 | **−3.1 %** | 15/24 | identical |
| **50 (corrected)** | **367.4** | **374.4** | **+1.9 %** | **26/50** | identical |

**26 of 50 is a coin flip.** The promising −3.1 % at 24 rounds did not
survive doubling the power; the estimate converges to zero. End-to-end
is flat, and it is flat for reasons that are now identifiable rather than
mysterious.

## Why flat, measured rather than guessed

1. **Coverage.** Three of four geometries route to us, but `down_proj`
   (the 2.96×) is 24 of 168 linears, and `gate/up` (96 layers) go to
   `awq_gemm` because we lose there.
2. **Dilution.** A 32-token decode on a 0.5B model spends most of its
   time in attention, the KV-cache path, sampling and the scheduler, not
   in the linear layers we replaced.
3. **Thermal reality.** Clocks read 1327 MHz against a 2100 MHz ceiling
   for every run in this block, with 8–9 % within-arm spread. A 2 % effect
   is not resolvable on this box even when it exists.

## What this closes

The product hypothesis "*kernel wins compound into a faster served
model, at this scale*" is **not supported**. Anyone quoting a model-level
speedup from this work would be quoting noise — as the 24-round run
nearly did.

## What it leaves standing

- **Kernel-level margin is real and now trustworthy:** 2.96× / 1.52× /
  1.35× over a production hand-tuned kernel, exact, cold, three of four
  shapes.
- **The referee is the asset that produced both the win and the
  retraction** — it caught a fake 3.3× and it caught a fake −3.1 %.
- **The honest product shape is a kernel library for unattended shapes**,
  not a faster serving engine for this model: the win has to be delivered
  where the linear layers dominate (prefill-heavy batches, larger models,
  or models whose graphs are not already fused) and that is unmeasured.
