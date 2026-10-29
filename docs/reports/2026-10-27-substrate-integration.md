# Integration plan — making the int4 kernel a substrate citizen

**Date:** 2026-10-27
**Status:** design + first steps. The kernel today is a demo: it bypasses
`kernelopt` entirely (no `Rewrite`, no `KernelSink`, no cost model, no
certificate, no sieve).

## What exists, and what it costs us

| piece | today | consequence |
|---|---|---|
| int4 GEMV | a Triton function in `bench/drivers/probes/awq/` | not minted, not priced, not certified |
| routing | a hand-typed dict of `(in, out) -> config` | does not generalise to new shapes |
| referee | `bench/benchkit/regime.py` (shipped, tested) | the one piece already production-shaped |
| cost model | `mlopt_core` roofline/profile, calibrated per machine | unused by the kernel path |
| sieve | `kernelopt` derivability + novelty | never sees these kernels |

## Step 1 — name the operations

Add int4 primitives to the kernel vocabulary so a kernel is a *term*, not
a function:

```
op:packed_load(in, out/8) -> i32[in, out/8]
op:dequant    (i32, i32 qzeros, f16 scales, group) -> f16[in, out]
op:gemv_int4  (f16[1,in], dequant) -> f16[1,out]
```

With these, the kernel becomes a composition the sink can price and the
sieve can compare against `KERNEL_PRIOR_ART_RULES` — which is what makes
"is this novel" answerable at all.

## Step 2 — price it with the machine, not a constant

`scripts/calibrate_profile.py` already measures bandwidth and launch
overhead per machine. The dispatch decision must be:

```
route(shape) = argmin over {ours, marlin, awq_gemm, matmul}
                 of predicted_us(shape, arm, profile)
```

with `predicted_us` = bytes ÷ measured bandwidth + overheads. Today's
routing table is a hardcoded dict measured on one laptop; it should be
*emitted* from the calibrated cost model, and re-emitted per machine.

## Step 3 — gate the search on the referee

The search must score candidates with `regime.blocked_paired` (eviction,
shuffled rounds) and reject any candidate whose `max|Δ|` exceeds
tolerance **immediately before timing**. Both rules exist because we lost
a week to their absence: a broken kernel looked 3× fast, and a patch that
never fired looked flat.

## Step 4 — let the mint propose the structure

The template space (GEMV / tiled / split-k × load width × tiles ×
output dtype) is authored by hand; the search picks within it. Getting
the *machine* to propose kernels means the vocabulary from Step 1 plus a
cost model that can rank structures it has never run — that is the
research question, and it is not answered here.

## What I would not do next

- More config sweeps. Every sweep lands at ~42 % of roofline; the
  remaining 2.4× is unpack codegen, not a parameter.
- More end-to-end runs on this box. 100 rounds at 13 % spread cannot
  resolve a 2 % effect; the last three runs said −3.2 %, −4.1 %, −2.0 %
  with sign tests of 4/6, 16/30, 46/100 — noise.
