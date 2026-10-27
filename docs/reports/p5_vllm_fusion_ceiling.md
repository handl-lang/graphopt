# P5c — the fusion ceiling inside vLLM (why not to build it)

**Question:** should we fuse `silu_and_mul + down_proj` inside vLLM's
Llama, where our 2.6–2.9× chain wins came from?

**Answer: no. By arithmetic, not by implementation quality.**

## Measured on this box

| what | time |
|---|---|
| `silu_and_mul`, M=1 (decode) | **7.49 µs** |
| `silu_and_mul`, M=8 | 7.68 µs |
| `silu_and_mul`, M=64 | 8.41 µs |
| `silu_and_mul`, M=256 (prefill-ish) | 31.70 µs |
| one empty Triton launch | **9.46 µs** |
| a decode token (whole model, 6 layers) | 3,490 µs |

A token spends `6 × 7.49 µs = 45 µs` in the activation → **1.29 %** of
the step. And a launch costs more than the op it would replace
(9.46 µs > 7.49 µs), so a fused `silu+mul+down` kernel can return
**at most ~1.3 %, realistically ~0 %**.

## Why this is the correct answer, and not a disappointment

vLLM's Llama graph is already fused: `rms_norm`, `fused_add_rms_norm`,
`silu_and_mul`, paged attention and the KV-cache path are each one
kernel. There is nothing per-op left to remove — the graph is at the
point where every remaining op is a big GEMM or a fused kernel.

Our 2.6–2.9× came from a different regime: **chains of many small
elementwise ops over big tensors**, where each op costs a launch and the
intermediate is materialized. Llama-in-vLLM is not that regime.

## What this redirects the work to

The fusion that pays is the fusion of *exposed* graphs — research models
written as spelled elementwise chains, which is exactly the
dispatch-bound half of the zoo (25 of 65 rows have F ≥ 0.30, median
ceiling 2.23×). The next experiment therefore is:

> take a dispatch-bound zoo module, fuse its elementwise chain into one
> Triton kernel exposed as a **plain `torch.library` op**, and measure it
> against eager and `torch.compile` — a drop-in that any stack
> (including vLLM's `CustomOp`) can call.

That is the version of "get into industry" that has headroom. Fusing
one activation inside a already-fused server does not.
