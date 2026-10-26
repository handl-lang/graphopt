# Chain-level pay on the real zoo corpus

**Question.** `--domain gen --probe` reports **2.87×** over `torch.compile` on a
synthetic 6-op chain. Does the same machinery pay when it has to deliver a
**real module**, at wide shapes, through the shipped path?

**Answer: on 17 of 65 delivered modules, and for a different reason than the
2.87×.** Measured over the **whole zoo** (68 workloads, 3 rounds each, six rows
re-measured at 5): `TernaryWeightLinear` **1.94×**, `RotaryPair` **1.52×**,
`PartialRotary` **1.47×**, `GroupedQuantLinear` **1.43×**, `DeepEquilibrium`
**1.30×**, `LoRAAdapter` **1.28×** — every one faster than eager in **every
round**. The mechanism is dispatch: a delivered `TimeCondConv` runs 1.24× eager
by launching **22 kernels instead of 34**. It does not transfer to kernel-bound
modules — 46 of 65 lose, median **0.969×** (33 of them by <10 %).
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

The six rows the 3-round pass flagged as noisy were then re-measured at 5
rounds on an idle box, and those rows replaced in the corpus above:

```bash
.venv/bin/python -m bench.drivers.probes.zoo_chain --isolate --rounds 5 \
    --only 'PerChannelQuantMatmul,SoftcapLogitHead,WaveNetGate,DynamicTanhBlock,DecoupledMLA,CapsuleRouting' \
    --budget 150 --out docs/reports/zoo_chain_sweep_round5.json
```

**65 of 68 delivered and verified** (`max_abs 0.0` on every row), **3 over
budget** (`MambaSelectiveScan`, `GatedLinearAttention`, `RealGatedLRU` — all
three do deliver in the `prim_kernel` probes at `large`, so this is a cost
fact about the sweep shape, not an export failure). **17 beat eager in every
round** (two of them after a 5-round re-measure); 2 have a median above
1 with a round below it and stay out of the count; 46 lose.

| workload | eager µs | gen µs | ratio (rounds) | median |
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
| `PerChannelQuantMatmul` | 29 | 24 | 1.158 / 1.204 / 1.322 / 1.333 / 1.199 | **1.20×** |
| `ManualGroupNorm` | 72 | 60 | 1.271 / 1.173 / 1.191 | **1.19×** |
| `ChunkedRetention` | 308 | 270 | 1.143 / 1.090 / 1.395 | **1.14×** |
| `HyenaLongConv` | 252 | 227 | 1.110 / 1.078 / 1.350 | **1.11×** |
| `STEQuantLinear` | 128 | 121 | 3.659 / 1.060 / 1.053 | **1.06×** |
| `RelPosTableAttention` | 314 | 300 | 1.028 / 1.344 / 1.043 | **1.04×** |
| `WaveNetGate` | 83 | 81 | 1.019 / 1.028 / 1.039 / 1.078 / 1.015 | **1.03×** |
| `SlidingWindowAttention` | 359 | 357 | 1.005 / 1.007 / 1.002 | **1.00×** |

µs columns are the round whose ratio is the median, so the pair and the ratio
are always from the same measurement. The two 5-round rows are the ones the
first pass flagged as noisy; they were re-measured on an idle box
(`zoo_chain_sweep_round5.json`) and both are wins at every round.

**What the winners have in common — measured, and weaker than the story.**
`bench/drivers/probes/zoo_kcount.py` profiles each workload's **eager**
forward (kernel count, device µs, wall µs) so the hypothesis can be tested
rather than told. Joined against the sweep's 65 delivered rows:

| eager's device share of wall time | rows | wins | win rate |
|---|---|---|---|
| < 0.5 (mostly dispatch) | 17 | 8 | **47 %** |
| 0.5 – 0.7 | 8 | 3 | 38 % |
| 0.7 – 0.9 | 10 | 1 | 10 % |
| ≥ 0.9 (mostly device) | 30 | 5 | 17 % |

So the *direction* holds — where eager spends more than half its wall time
outside the kernels, 47 % of modules pay; where the kernels dominate, 10–17 %.
But the hypothesis is **not a predictor**: correlation of the ratio with
`kernels/fwd` is **0.01**, with device share **−0.17**, with µs-per-kernel
**−0.07**. Kernel *count* is useless as a signal (`GroupedQuantLinear` wins
1.43× on 3 kernels; `TaylorLinearAttention` launches 11 and loses nothing).
The delivered body's own cost, not eager's launch profile, decides the sign.
`genkernel.profitable` is the thing that prices delivered bodies, and the
dtype section below is the case where the sign comes from something it does
not price at all — a *law* fold, applied unconditionally.

**Six rows were noise; re-measuring resolved four.** `PerChannelQuantMatmul`
[0.31 / 1.247 / 1.397] and `WaveNetGate` [0.874 / 1.160 / 1.186] each had a
round with a wild arm; at 5 rounds on an idle box they read
[1.158 – 1.333] and [1.015 – 1.078], so **both are wins** and both are in the
table above. `DecoupledMLA` [1.105 / 1.017 / 0.952] and `SoftcapLogitHead`
[0.278 / 2.424 / 1.015] went the other way — at 5 rounds they read
[0.901 – 0.979] and [0.917 – 0.926] with a single 2.31× outlier, so **both are
losses** and their earlier medians were the artifact. `CapsuleRouting`
[1.146 / 1.014 / 1.64 / 1.027 / 0.891] and `DynamicTanhBlock` [0.952 – 1.091]
are still unresolved and stay out of the count, which is why it says 17 and
not 19. `STEQuantLinear` is the flattering version of the same artifact (one
round read 3.66× while its other two read 1.06 and 1.05); it stays at its
median, 1.06×, which is also its floor.

**The worst row is a dtype artifact, measured.** `RMSNormPerHead` reads
**0.208×** (eager 49 µs, delivered 231 µs, verified) — and the cause is
not the fold, it is **fp64**. A law fold rewrites the spelled chain into
`aten::rms_norm`, and PyTorch's fused RMSNorm is pathological in double
precision at this shape: on its own, `rms_norm` over `(b,t,h,16)` costs
**228.6 µs in fp64 against 32.8 µs for the spelled chain**, while in fp32 the
same call costs **14.3 µs against 32.5 µs**. Running the module through the
same arena path with dtype as the only variable, no minted kernel admitted in
either arm (the referee declines all three in both):

| dtype | eager µs | delivered µs | ratio | max_abs |
|---|---|---|---|---|
| fp32 | 39.5 | **28.6** | **1.38×** | 3.6e-07 |
| fp64 | 58.7 | **233.9** | **0.25×** | 8.9e-16 |

(`bench/drivers/probes/zoo_dtype.py --only RMSNormPerHead`,
`docs/reports/zoo_dtype_rmsnorm.json`; a second run read 1.45× / 0.27×, so the
direction is stable and the magnitude is within ~5 %.)

So the module is a **1.38× win in fp32** and a 4× loss in fp64, from the same
IR and the same fold, with no minted kernel admitted in either arm. The sweep
runs fp64 because that is the intake convention — the dtype where Triton
cannot bind either — so this row is the probe's dtype showing through, not a
defect in what the fold ships. The delivered body names the cost outright
(`bench/drivers/probes/zoo_dtype.py`, `docs/reports/zoo_dtype_rmsnorm.json`):
one `RowwiseMomentsCUDAKernel<double>` at **205.8 µs**, against a
`vectorized_layer_norm_kernel<float>` at **11.8 µs** in fp32 — a 17× difference
between two backend kernels for the same fold.

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
synthetic tape. Delivered into real modules, the same machinery pays on **17 of
65** and loses on 46 (median 0.969×). Combined with `prim_kernel.md` §module
level (folds 3.17 → 3.34 at `large`, 3.23 → 2.67 at `wide`), the pattern across
both delivery mechanisms is now precise:

> A generated kernel pays **as an op** (2.6×), or as the **whole chain**
> (2.87× on a synthetic tape), or where the delivered module launches **fewer
> kernels than eager** (`TimeCondConv`, 22 vs 34; and 47 % of the modules
> where eager leaves >50 % of wall time in dispatch). It does not pay as a
> step inside a module that is already kernel-bound. The per-module sign is
> **not** predictable from eager's profile, so it has to be measured per
> module — which is `genkernel.profitable`'s job, and it prices *claims*
> only: every win in the table comes from an unconditional fold or from fewer
> launches, and the largest single loss (`RMSNormPerHead`, 0.208×) is a fold
> whose sign is set by the backend kernel at the intake dtype. The referee
> declining all three claims there was right; nothing was pricing the fold.

## Limits, stated

- **65 of 68 workloads, one box, one shape.** Three rows exceed the 150 s
  per-workload budget at `d64 t32 b8` (`MambaSelectiveScan`,
  `GatedLinearAttention`, `RealGatedLRU`) and are reported as such. All 65
  delivered rows verify bit-equal (`max_abs 0.0`). Single GPU (RTX 2050) — the
  ratios are dispatch-sensitive, so absolute µs will move on other hardware
  and the *set* of winners may too.
- **Two rows are not resolved.** `CapsuleRouting` [1.146 / 1.014 / 1.64 /
  1.027 / 0.891] and `DynamicTanhBlock` [0.952 – 1.091] still straddle 1 after
  a 5-round re-measure, so they are excluded from the win count. The other
  four noisy rows resolved: two were wins (`PerChannelQuantMatmul` 1.20×,
  `WaveNetGate` 1.03×) and two were losses (`DecoupledMLA` 0.93×,
  `SoftcapLogitHead` 0.92×).
- **fp64, and that is a real constraint on the numbers.** Triton cannot run
  fp64, so the gen kernels here are the fallback bodies; the fp32/CUDA
  comparison is `prim_kernel.md`'s job, not this one's. The dtype is not
  neutral for the *folds* either: `RMSNormPerHead` is 0.25× in fp64 and
  **1.38× in fp32** from the same IR, because PyTorch's fused `rms_norm` is
  slow in double at head-width 16. Whatever else is pathologically slow in
  fp64 is invisible to this sweep — it can only make the corpus look worse
  than its fp32 behaviour, so 17/65 is a floor for the fold-and-launch
  mechanism, not an estimate of it.
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

1. **Run the sweep in fp32.** The corpus above is fp64 — the dtype where
   Triton cannot bind *and* where at least one folded body is 4× slower than
   the chain it replaces. `RMSNormPerHead` shows the two dtypes disagree by
   5.5× in ratio on identical IR (0.25× → 1.38×), so the honest next number is
   the same 68 workloads in fp32, where the delivered body is also the one
   users ship. That is a `--dtype` flag on the driver plus a second
   `zoo_chain_sweep_fp32.json`; until it exists, 17/65 is a floor.
2. **Price the folds, not only the claims.** `genkernel.profitable` prices
   minted claims site-by-site; every win in the table and the worst loss come
   from *law* folds applied unconditionally, with no cost gate at all. A
   delivery-time check that times the delivered module against eager once and
   declines when it loses would have caught the fp64 `RMSNormPerHead` at one
   measurement per workload — and would have caught it as a dtype, not a
   regression.
3. **Two rows still straddle 1** (`CapsuleRouting`, `DynamicTanhBlock`). Both
   have one wild round in five; a 9-round run on an idle box, or the same
   fp32 sweep in (1), should settle them. They cost 2 rows of the count, not
   its direction.
