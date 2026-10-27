# P5 — the vLLM speed claim, re-measured (and withdrawn)

**Status: the +8% decode-tok/s number does not reproduce under a
controlled measurement. Token agreement does.**

Model: llama2.c stories15M (6 layers, fp32), vLLM 0.32.0,
`enforce_eager=True`, RTX 2050 (4 GB), greedy, gen 32 tokens.
Arms: `eager` = unoptimized HF export, `mlopt` = mlopt-optimized HF
export. Both arms produce **identical tokens in every configuration**.

## What the original suite reported

Best-of-3 `generate()` calls, one warmup call, subtract a separately
measured prefill:

| arm | decode tok/s |
|---|---|
| eager | 604.6 |
| mlopt | **653.5** (+8.1%) |

The per-call times show why this is unsafe: eager's three gen calls were
`0.465 → 0.125 → 0.107 s` — still falling on the last one. The estimator
compared a warmed arm against one that had not reached steady state.

## Sequential re-measurement — and it flips with order

10 reps after 5 warmup calls, median of whole-call wall time:

| run order | ctx | eager | mlopt | winner |
|---|---|---|---|---|
| eager first | 64 | 99.0 ms | 107.5 ms | eager by 7.9% |
| eager first | 256 | 96.8 ms | 120.5 ms | eager by 19.7% |
| mlopt first | 64 | 107.9 ms | 92.1 ms | mlopt by 14.6% |
| mlopt first | 256 | 94.9 ms | 92.4 ms | mlopt by 2.6% |

**Each arm is ~15% faster when it runs first.** The ordering effect is
larger than the effect under test (thermal drift / allocator state), so
neither direction is evidence.

## Paired A/B — both engines loaded, orders alternated

12 rounds, A/B order alternating each round, median over rounds:

| arm | median | sd | tok/s |
|---|---|---|---|
| eager | 99.9 ms | ±10.2 | 320.3 |
| mlopt | 95.2 ms | ±9.3 | 336.2 |

mlopt +5.0% on the median — but **eager was faster in 5 of 12 rounds**
and the within-arm sd (~10%) is twice the between-arm difference.
At ctx 256 the paired design could not run: two engines need 1.31 GiB
each and only 1.1 GiB was free on a 4 GB card.

## Locked clocks — the decisive runs

`sudo nvidia-smi -lgc`, 12 then 30 alternating rounds per context.

**At 210 MHz (idle clock, max is 2100):** eager 111.4 ms ±3.5, mlopt
114.7 ms ±6.6 → mlopt 3.4% slower, eager won **8/12**.

**At 1500 MHz, 12 rounds:** mlopt looked faster — ctx 64 +6.5% (8/12),
ctx 256 +5.3% (9/12).

**At 1500 MHz, 30 rounds — the 12-round win evaporates:**

| ctx | eager | mlopt | difference | mlopt wins |
|---|---|---|---|---|
| 64 | 99.7 ms ±5.2 | 101.5 ms ±7.1 | −1.8% | **15 / 30** |
| 256 | 117.9 ms ±17.2 | 112.2 ms ±13.4 | +5.1% | **17 / 30** |

Both sign tests sit at chance (15/30 and 17/30; 32/60 combined). The
point estimates point in opposite directions across the two context
lengths. **There is no measurable speed difference.**

## Final verdict

> Optimized weights are served **token-identically** by vLLM (1.000
> agreement, every configuration) and are **speed-neutral** on this
> hardware: no reproducible difference, effect below ±3%, sign tests at
> coin-flip with 30 rounds.

That is consistent with the mechanism: vLLM does not consume mlopt's
executors, so only the *weights* cross over — and at 6 layers in fp32
the folded weights remove no work vLLM was paying for.

**Lesson kept:** three separate numbers (+8%, +6.5%, −1.8%) came out of
the same two models depending on protocol. Only the locked-clock,
30-round, order-alternating design was stable — and it said *no
difference*. The earlier numbers were protocol, not product.


## Why this is worth keeping

The 1.000 token agreement is the load-bearing P5 result and is
unaffected: optimized weights survive export, a HF checkpoint, and a
production server without changing a single output token. The speed
claim is withdrawn rather than published with a caveat — a number this
sensitive to arm order should not leave the building.
