# P12 — the routing table, under the regime that actually runs

Everything before this measured kernels in a warm cache. A transformer
does not run that way: a layer's 2.18 MB of int4 weights cannot survive a
2 MB L2 across the 24 layers that execute before it is used again. So the
deployment regime is **cold**, and the earlier rankings were partly
fiction.

Measured with `bench.benchkit.regime.blocked_paired` — weights evicted,
candidates shuffled per round, medians over 25 rounds.

## The table (cold = deployment; reuse = the same kernel called again)

| layer | cold ours | cold `awq_gemm` | cold torch | **cold winner** | reuse winner |
|---|---|---|---|---|---|
| `down_proj` 896×4864 | **76.5 µs** | 179.9 | 123.7 | **ours** (2.35× vLLM) | ours |
| `q_proj` 896×896 | 52.0 | 58.4 | **51.6** | torch | ours |
| `gate_proj` 4864×896 | 79.7 | **66.8** | 127.7 | vLLM | ours |
| `k_proj` 128×896 | 53.5 | 61.2 | **34.1** | torch | torch |

**The two regimes disagree on three of four shapes.** A routing table
built on warm numbers would send `q_proj` and `gate_proj` to us; in
deployment both lose. This is the single most important methodological
result of the block: *the objective has to be the regime that runs.*

## What streaming hints bought

Adding `eviction_policy="evict_first"` to the packed-weight loads — the
weights are read once and never reused within a call, so they should not
be cached — improved the cold numbers by 17–39 %:

| layer | cold before | cold after |
|---|---|---|
| `down_proj` | 92.3 | **76.5** |
| `q_proj` | 58.3 | **52.0** |
| `gate_proj` | 96.9 | **79.7** |
| `k_proj` | 87.6 | **53.5** |

`down_proj` is now at **32 % of its roofline** (76.5 vs 24.6 µs) — still
3× off the physics, but the hint alone closed a fifth of the gap, which
is a hint about where the rest lives.

## The routing that survives

```
down_proj   4864->896   ours      (2.35x over awq_gemm, 1.62x over torch)
q_proj/o_proj 896->896  torch     (ours 0.99x — a tie, and torch has no dequant)
gate/up     896->4864   awq_gemm  (ours 0.84x)
k/v_proj    896->128    torch     (ours 0.64x)
```

**24 of 168 layers route to us.** That is why the end-to-end test is
flat: the one geometry we win is a quarter of the work, and within it we
are 2.35× — but a token is not made only of that layer.

## Honest position

- The kernel is **real**: 2.35× over a hand-written production kernel, in
  the deployment regime, on the shape where memory traffic dominates.
- Our coverage is **one of four geometries**, and that is measured, not
  guessed.
- The roofline still has **3× above us** on the winning shape, so the
  kernel is not done either.
- The next lever is not another config sweep; it is why 76 µs of work
  reads 2.18 MB at 32 % of the achievable bandwidth, when the hint
  experiment shows the memory system responds to being told what to do.
