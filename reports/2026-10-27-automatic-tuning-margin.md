# The margin, and what it means

**Date:** 2026-10-27
**Status:** conclusion of the P5–P8 experiment block
**One line:** hand-tuning is expensive and therefore only bought for
popular shapes; automating it collapses the cost per experiment, and the
first measured margin over a hand-written production kernel came from
exactly a shape nobody had tuned.

## What we actually measured

Against `awq_gemm` — vLLM's hand-written, production 4-bit CUDA kernel —
on the real layers of Qwen2.5-0.5B-Instruct-AWQ, one token at a time:

| geometry | layers | ours | `awq_gemm` | result |
|---|---|---|---|---|
| `down_proj` 4864→896 | 24 | **55.9 µs** | 186.0 µs | **3.32× faster** |
| `q_proj`/`o_proj` 896→896 | 48 | 31.7 | 38.3 | **1.21× faster** |
| `gate_proj`/`up_proj` 896→4864 | 48 | 60.0 | 53.5 | 0.89× (loss) |
| `k_proj`/`v_proj` 896→128 | 48 | 22.6 | 37.9 | 1.68× over vLLM, but plain matmul is 16.3 — route away |

**120 of 168 linears beat `awq_gemm`.** Every number is fp32-exact to
~1e-6 against the dequantized reference.

The kernel was written in an afternoon and never hand-tuned. The configs
were chosen by a measured sweep (48 combinations per shape) — the machine
picked them, not a person.

## Why the margin exists

Not because we out-thought the kernel engineers. `awq_gemm` is tuned for
the shapes their customers actually run. A 896×4864 matmul inside a
0.5B model on a 4 GB laptop GPU is **an unattended shape** — it is not
worth a specialist's week.

That is the whole economic argument:

> Human tuning has a high fixed cost per shape, so it is only bought
> where the shape is popular. Automation collapses that cost, so the
> long tail becomes worth optimizing. The world has vastly more model
> shapes than any kernel team can hand-tune.

The margin is **arbitrage on unattended cases**, not a breakthrough of a
physical ceiling. That distinction matters, because it tells you where
the next margins are: everywhere nobody looked.

## What scales, and what does not

**Does not scale: per-kernel speed.** Kernel time is bounded by memory
bandwidth and FLOPs. The gap between "well hand-tuned" and "optimal" is
typically 1.1–2×, occasionally 3–5×. No amount of search produces a 100×
kernel, because the hardware is the ceiling.

**Does scale: coverage.** If automation optimizes 50,000 shapes instead
of 50, the aggregate value is enormous even when each win is 1.3×. The
exponential lives in breadth, and in how often models and formats change
— not in a single miracle number.

**Also scales: the cost per experiment.** Seconds instead of days. That
is the compounding asset; every other benefit is downstream of it.

## Where novelty can come from (measured, not assumed)

The strongest result in this project is about *where* new structure
enters:

| arm | machine-original admitted |
|---|---|
| compose rules the bank already has | **0 / 39** |
| allowed to name new operations | **260 / 262** |

Novelty comes from **growing the vocabulary**, not from searching harder
inside a fixed one. "Keep exploring" only produces new algorithms if the
exploration may invent concepts. Everything else is rearrangement.

## What this does *not* prove

**The machine did not invent the kernel.** I wrote the int4 GEMV; the
machine swept its configs. So the evidence is *the substrate can host and
tune a kernel that beats a hand-tuned production kernel* — not *the loop
discovers such a kernel unaided*. That gap is the RL claim, and it is
still open.

**The model did not get faster.** End-to-end on the same AWQ model in
vLLM, routed per shape: 314.4 ms stock vs 317.6 ms ours (+1.0 %, 12/20
rounds, tokens identical). Flat. The kernel wins are real and the model
win is not, for two known reasons:

1. **dtype conversion** — the model runs fp16 and our kernel is
   fp32-in/fp32-out, so each call pays two copies the isolated benchmarks
   never saw;
2. **dilution** — two of four geometries route to us; attention,
   sampling and scheduling dominate the rest of a token.

Both have known fixes (fp16 kernel, more coverage). Neither is done.

**Precision was not matched.** We accumulate fp32 and return fp32;
`awq_gemm` is fp16 in and out. The isolated comparison flattered us, and
in a real model it turned from a fairness question into a cost.

## What the session also established (negatives worth keeping)

- **Whole Llama through vLLM: our weights are speed-neutral.** Token
  exact, no speed. The transfer path "hand them weights" carries almost
  none of the optimization.
- **Fusing the MLP chain inside vLLM is worth at most 1.29 %.** Measured
  before building it: the activation is 7.5 µs and a launch is 9.5 µs on
  a 3,490 µs token. The graph is already fused; there is nothing there.
- **Compute-bound shapes are capped at 1.03× by construction** — an
  equality-preserving rewrite cannot change the FLOPs a dense model must
  do. 28 of 65 zoo rows sit in that regime.
- **Where the headroom is:** 25 of 65 zoo rows are launch-bound, median
  ceiling 2.23×, max 5.95×.

## What to do next, in order

1. **fp16 kernel** (fp32 accumulate). Without it the end-to-end number
   cannot move, no matter how good the isolated numbers get.
2. **Let the loop propose the kernel**, not just its config. That is the
   experiment that turns this report's optimism into evidence.
3. **Coverage**: apply the same search to more shapes and formats — the
   long tail is the actual market.
4. **An honest referee at scale.** On this laptop, run-to-run noise
   sometimes reached ±50 %, ten times the effect. Ten thousand
   experiments measured that way produce ten thousand false positives.
   The measurement discipline — negatives written down next to wins — is
   the asset that makes any of this trustworthy.

## The judgment

The project began as a bet that looked implausible. It now has a
measured, reproducible margin over a hand-written production kernel, on
real weights, from a kernel nobody tuned by hand.

That is not "impossible became possible." It is narrower and more useful:
**the unattended corners of the ML stack are large, and the cost of
attending to them just collapsed.** Per-shape wins are modest (1.2–3×);
the number that matters is that they can now exist everywhere.

