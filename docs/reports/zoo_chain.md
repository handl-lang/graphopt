# Chain-level pay on the real zoo corpus

**Question.** `--domain gen --probe` reports **2.87×** over `torch.compile` on a
synthetic 6-op chain. Does the same machinery pay when it has to deliver a
**real module**, at wide shapes, through the shipped path?

**Answer: on 15 of 65 delivered modules, and for a different reason than the
2.87×.** Measured over the **whole zoo** (68 workloads, 3 rounds each):
`TernaryWeightLinear` **1.94×**, `RotaryPair` **1.52×**, `PartialRotary`
**1.47×**, `GroupedQuantLinear` **1.43×**, `DeepEquilibrium` **1.30×**,
`LoRAAdapter` **1.28×** — every one faster than eager in **all three rounds**.
The mechanism is dispatch: a delivered `TimeCondConv` runs 1.24× eager by
launching **22 kernels instead of 34**. It does not transfer to
kernel-bound modules — 44 of 65 lose, median **0.972×** (31 of them by <10 %).
`FNOBlock` is the outlier direction: **1.77× `torch.compile`**, tying eager.
No win comes from a *minted* claim — the one workload whose search found
claims (`GatedAttentionUnit`, 2) is 0.95× eager. Whole-chain supremacy does
not transfer; a narrower per-workload effect does, and it is now counted
rather than asserted.

## Setup

`bench/drivers/probes/zoo_chain.py` runs the unmodified shipped path per
workload — build → fp64 → `TorchSource.to_ir` → mint tiers →
`genkernel.profitable` (measured-cost referee) → `gen_sink` →
`play.gen_probe` (deliver, verify, time against eager and `torch.compile`).

Shapes `d=64 t=32 b=8` — the module probe's `large`, not the zoo default
`d16 t8 b2` where every kernel is below the launcher's noise floor.
Box: i5-12500H / RTX 2050, fp64, medians over 100 reps per round, **3 rounds
per workload**, ratio computed per round and the median of the three
reported. Every row's `max_abs` is `0.0` — the delivered module computes the
same function as the source module.

```bash
.venv/bin/python -m bench.drivers.probes.zoo_chain --rounds 3 \
    --out docs/reports/zoo_chain.json
```

## Result (2026-10-26)

| workload | claims | gen µs (3 rounds) | eager µs | `torch.compile` µs | gen ÷ eager | median | gen ÷ compile |
|---|---|---|---|---|---|---|---|
| `GatedAttentionUnit` | 2 | 303.1 / 293.0 / 229.6 | 279.8 / 281.2 / 218.2 | 312.1 / 318.5 / 253.9 | 0.923 / 0.960 / 0.950 | **0.95** | 1.03 |
| `MLPMixerBlock` | 0 | 318.7 / 316.5 / 316.6 | 312.7 / 313.1 / 311.6 | 294.4 / 288.5 / 289.4 | 0.981 / 0.989 / 0.984 | **0.98** | 0.91 |
| `BlockSparseAttention` | 0 | 407.8 / 407.4 / 317.1 | 390.7 / 392.6 / 301.3 | 400.1 / 397.4 / 313.7 | 0.958 / 0.964 / 0.950 | **0.96** | 0.98 |
| `TimeCondConv` | 0 | 122.4 / 135.8 / 134.7 | 164.1 / 181.9 / 178.9 | 110.8 / 114.7 / 111.3 | 1.341 / 1.340 / 1.328 | **1.34** | 0.82 |
| `FNOBlock` | 0 | 115.0 / 103.3 / 106.6 | 94.9 / 101.6 / 99.8 | 194.3 / 189.8 / 188.2 | 0.825 / 0.984 / 0.935 | **0.93** | **1.77** |

Read it as two rows, not five:

- **`TimeCondConv` pays against eager, loses to `torch.compile`.** Every round
  in every session lands above 1 (18 clean rounds, **1.15–1.42**, session
  medians 1.18–1.34): a real, reproducible win over the framework's own
  dispatch — and inductor still does better. The absolute µs drift between
  sessions (eager 164 µs in the table above, 190–205 µs in a later run) moves
  the ratio more than anything in the delivered module does.
- **`FNOBlock` pays against `torch.compile`, ties eager.** 1.77× on all three
  rounds: `torch.compile` regresses the fft chain to ~190 µs where eager holds
  ~100 µs, and the delivered module holds eager's cost.

Everything else is a 2–5 % loss with the spread inside 4 % — tight enough that
no single round is carrying the verdict.

This is the **sample**: five workloads chosen for range. It is kept because its
per-round µs are what the `TimeCondConv` kernel-count argument and the
`FNOBlock` `torch.compile` comparison are built on. The corpus-wide answer is
the next section.

## The sweep: all 68 workloads (2026-10-26)

Same shipped path, same shape, 3 rounds each, one process per workload
(`--isolate`) with a 150 s budget, so a module whose `torch.compile` never
returns costs one row instead of the table:

```bash
.venv/bin/python -m bench.drivers.probes.zoo_chain --isolate --rounds 3 \
    --only '' --budget 150 --out docs/reports/zoo_chain_sweep.json
```

**65 of 68 delivered and verified** (`max_abs 0.0` on every row), **3 over
budget** (`MambaSelectiveScan`, `GatedLinearAttention`, `RealGatedLRU` — all
three do deliver in the `prim_kernel` probes at `large`, so this is a cost
fact about the sweep shape, not an export failure). **15 beat eager in all
three rounds**; 6 more have a median above 1 with a round below it; 44 lose.

| workload | eager µs | gen µs | ratio (3 rounds) | median |
|---|---|---|---|---|
| `TernaryWeightLinear` | 94 | 49 | 1.879 / 2.361 / 1.936 | **1.94×** |
| `RotaryPair` | 406 | 267 | 1.845 / 1.485 / 1.520 | **1.52×** |
| `PartialRotary` | 612 | 416 | 1.745 / 1.473 / 1.373 | **1.47×** |
| `GroupedQuantLinear` | 32 | 22 | 1.256 / 1.425 / 1.449 | **1.43×** |
| `DeepEquilibrium` | 280 | 215 | 1.261 / 1.321 / 1.305 | **1.30×** |
| `LoRAAdapter` | 92 | 72 | 1.280 / 1.293 / 1.162 | **1.28×** |
| `NormedSoftcapHead` | 99 | 78 | 1.278 / 1.265 / 1.218 | **1.26×** |
| `TimeCondConv` | 170 | 136 | 1.244 / 1.234 / 1.250 | **1.24×** |
| `GatedConvUnit` | 78 | 64 | 1.049 / 1.212 / 1.223 | **1.21×** |
| `ManualGroupNorm` | 72 | 60 | 1.271 / 1.173 / 1.191 | **1.19×** |
| `ChunkedRetention` | 308 | 270 | 1.143 / 1.090 / 1.395 | **1.14×** |
| `HyenaLongConv` | 252 | 227 | 1.110 / 1.078 / 1.350 | **1.11×** |
| `STEQuantLinear` | 128 | 121 | 3.659 / 1.060 / 1.053 | **1.06×** |
| `RelPosTableAttention` | 314 | 300 | 1.028 / 1.344 / 1.043 | **1.04×** |
| `SlidingWindowAttention` | 359 | 357 | 1.005 / 1.007 / 1.002 | **1.00×** |

µs columns are the round whose ratio is the median, so the pair and the ratio
are always from the same measurement.

**What the winners have in common — measured, and weaker than the story.**
`bench/drivers/probes/zoo_kcount.py` profiles each workload's **eager**
forward (kernel count, device µs, wall µs) so the hypothesis can be tested
rather than told. Joined against the sweep's 65 delivered rows:

| eager's device share of wall time | rows | wins | win rate |
|---|---|---|---|
| < 0.5 (mostly dispatch) | 17 | 7 | **41 %** |
| 0.5 – 0.7 | 8 | 2 | 25 % |
| 0.7 – 0.9 | 10 | 1 | 10 % |
| ≥ 0.9 (mostly device) | 30 | 5 | 17 % |

So the *direction* holds — where eager spends more than half its wall time
outside the kernels, 41 % of modules pay; where the kernels dominate, 10–17 %.
But the hypothesis is **not a predictor**: correlation of the ratio with
`kernels/fwd` is **0.01**, with device share **−0.17**, with µs-per-kernel
**−0.07**. Kernel *count* is useless as a signal (`GroupedQuantLinear` wins
1.43× on 3 kernels; `TaylorLinearAttention` launches 11 and loses nothing).
And the worst row sits in the bucket the hypothesis says should win:
`RMSNormPerHead` has device share 0.30 and 5 kernels — eager 49 µs wall,
14.7 µs device — and the delivered body takes **231 µs**, i.e. 0.208×. The
delivered module's own cost, not eager's launch profile, is what decides the
sign, which is what `genkernel.profitable` exists to referee and did not here.

**Six rows are noise, not wins.** `PerChannelQuantMatmul` [0.31 / 1.247 /
1.397], `SoftcapLogitHead` [0.278 / 2.424 / 1.015], `WaveNetGate` [0.874 /
1.160 / 1.186], `DynamicTanhBlock` [0.971 / 1.030 / 1.038], `DecoupledMLA`
[1.105 / 1.017 / 0.952], `CapsuleRouting` [0.921 / 1.005 / 1.031] each have a
round where one arm moved by tens of percent — the same contamination that
produced the 0.42× `TimeCondConv` reading. They are listed as noisy and
excluded from the count above. `STEQuantLinear` is the flattering version of
the same artifact (one round read 3.66× while its other two read 1.06 and
1.05); it stays in the table at its median, 1.06×, which is also its floor.

**One row is a real regression worth its own look.** `RMSNormPerHead` is
**0.208×** — eager 49 µs, delivered 231 µs, verified equal — on a module whose
eager forward launches 5 kernels and spends 14.7 µs on device. Something in
the delivered body costs ~180 µs that the referee did not see, and it is the
largest loss in the corpus by a factor of three over the next one.

## Why `TimeCondConv` is faster (measured, not asserted)

Torch profiler, per forward, same process:

| | GPU kernels / iter | device time / iter |
|---|---|---|
| eager module | 34 | 171 µs |
| delivered module | **22** | **146 µs** |

The delivered module drops 12 small elementwise kernels — the fold replaces
them with fewer, wider ones — and device time falls 15 %, which is the same
~1.3× once dispatch is counted once eager's 34-way dispatch is added on top. The win is
**kernel-count reduction** reaching module scale because this module is
dispatch-bound, not kernel-bound.

## Correction: the "unusable" verdict was my measurement, not the code

An earlier revision of this report recorded `TimeCondConv` as **unusable** —
gen 130–431 µs across five *identical* runs (0.42–1.44×) while eager held
±3 % — and blamed a bimodal slow path in the delivered module. That
conclusion does not survive a controlled repeat:

| protocol | gen ÷ eager |
|---|---|
| 5 single-run samples, box also running a pytest suite / another probe | 1.03, 1.44, 1.29, 1.24, **0.42** |
| 6 consecutive rounds, nothing else running | 1.281, 1.271, 1.235, 1.192, 1.230, 1.299 |
| 3-round driver run | 1.264, 1.272, 1.419 |
| 3-round driver run (table above) | 1.341, 1.340, 1.328 |
| 3-round driver run, later session | 1.230, 1.184, 1.154 |

Eighteen clean round-readings land in **1.15–1.42**, every one above 1. The
single 0.42× came from a sample taken while other jobs shared the box, and two
attempts to reproduce it deliberately failed: four CPU spinners left both arms
unchanged (eager 224 µs, gen 167 µs), and it never recurred in any isolated
run. The cause of that one outlier is therefore **unidentified**, and it is
recorded that way rather than as a property of the code.

What survives is a methodological rule, now enforced by the driver: **a
module-level ratio needs repeated rounds on an idle box**, because the two
arms are timed at different moments. `--rounds` (default 3) reports a median
and keeps the spread; a single round is no longer publishable.

## What this does and does not close

The 2.87× is real, but it is a **whole-chain kernel launched once** over a
synthetic tape. Delivered into real modules, the same machinery pays on **15 of
65** and loses on 44 (median 0.972×). Combined with `prim_kernel.md` §module
level (folds 3.17 → 3.34 at `large`, 3.23 → 2.67 at `wide`), the pattern across
both delivery mechanisms is now precise:

> A generated kernel pays **as an op** (2.6×), or as the **whole chain**
> (2.87× on a synthetic tape), or where the delivered module launches **fewer
> kernels than eager** (`TimeCondConv`, 22 vs 34; and 41 % of the modules
> where eager leaves >50 % of wall time in dispatch). It does not pay as a
> step inside a module that is already kernel-bound. The per-module sign is
> **not** predictable from eager's profile — that is `genkernel.profitable`'s
> job, and it currently gets it wrong at least once (`RMSNormPerHead`).

## Limits, stated

- **65 of 68 workloads, one box, one shape.** Three rows exceed the 150 s
  per-workload budget at `d64 t32 b8` (`MambaSelectiveScan`,
  `GatedLinearAttention`, `RealGatedLRU`) and are reported as such. All 65
  delivered rows verify bit-equal (`max_abs 0.0`). Single GPU (RTX 2050) — the
  ratios are dispatch-sensitive, so absolute µs will move on other hardware
  and the *set* of winners may too.
- **Six rows are not resolved.** `PerChannelQuantMatmul`, `SoftcapLogitHead`,
  `WaveNetGate`, `DynamicTanhBlock`, `DecoupledMLA`, `CapsuleRouting` have a
  round where one arm moved by tens of percent. Their medians are printed, but
  they are excluded from the win count until re-measured on an idle box.
- **fp64.** Triton cannot run fp64, so the gen kernels here are the fallback
  bodies; the fp32/CUDA comparison is `prim_kernel.md`'s job, not this one's.
- **The claim-bearing workload loses.** `GatedAttentionUnit` is the only
  workload where the cost referee found a site worth replacing (2 claims), and
  the module still comes out 0.95× eager. Every win in the table comes from
  folds the delivery path applies unconditionally (or from fewer launches),
  not from the search's inventions.
- **`fft_*` is now bound.** `FNOBlock` was undeliverable (no binding for
  `fft_rfft2`); the bridge now binds the fft family plus `real`/`imag`, and the
  exporter records an fft output shape as `s` rather than `dim`. That is what
  turns the row from missing into measurable.

## Next

1. **`RMSNormPerHead` at 0.208×** is the one row that is neither noise nor a
   small loss. Profile the *delivered* body (the `zoo_kcount` counterpart one
   level down): 5 eager kernels, 14.7 µs device, 49 µs wall against 231 µs
   delivered means the delivered body is not launching 5 kernels' worth of
   work. Until it is explained, the referee's silence on it is unexplained.
2. **Make the referee per-module.** `genkernel.profitable` prices claims
   site-by-site, but the sign of a module is set by the whole delivered body.
   A delivery-time check that times the delivered module against eager once and
   *declines* when it loses would have caught `RMSNormPerHead` and cost one
   measurement per workload — the sweep shows that measurement exists.
3. **Re-run the six noisy rows** on an idle box (`--only`, 5 rounds). Their
   spread is evidence about the box, not about the code, and the win count is
   currently conservative because they are excluded.
