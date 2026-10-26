# Chain-level pay on the real zoo corpus

**Question.** `--domain gen --probe` reports **2.87×** over `torch.compile` on a
synthetic 6-op chain. Does the same machinery pay when it has to deliver a
**real module**, at wide shapes, through the shipped path?

**Answer: on 23 of 65 delivered modules in fp32 (17 of 65 in fp64), and for a
different reason than the 2.87×.** Measured over the **whole zoo** (68
workloads, 3 rounds each, six rows re-measured at 5, both intake dtypes):
`TernaryWeightLinear` **3.05×**, `STEQuantLinear` **1.68×**,
`ALiBiSlopeAttention` **1.64×**, `RelPosTableAttention` **1.58×**,
`RMSNormPerHead` **1.53×**, `GroupedQuantLinear` **1.52×** — every one faster
than eager in **every round** at the dtype a caller ships. The mechanism is
dispatch: a delivered `TimeCondConv` runs 1.36× eager by launching **22
kernels instead of 34**. It does not transfer to kernel-bound modules — 38 of
65 lose in fp32, median **0.979×**. `FNOBlock` is the outlier direction:
**1.77× `torch.compile`**, tying eager. No win comes from a *minted* claim —
the one workload whose search found claims (`GatedAttentionUnit`, 2) is 0.76×
eager in fp32, 0.95× in fp64. Whole-chain supremacy does not transfer; a
narrower per-workload effect does, and it is now counted rather than asserted
— **in both dtypes, because the sign of a folded body is a backend-kernel
fact**: 9 modules flip from loss to every-round win between fp64 and fp32, and
`RMSNormPerHead` flips 0.21× → 1.53× on identical IR.

## Setup

`bench/drivers/probes/zoo_chain.py` runs the unmodified shipped path per
workload — build → intake dtype (`--dtype fp64` default, `fp32` = what a
caller ships) → `TorchSource.to_ir` → mint tiers → `genkernel.profitable`
(measured-cost referee) → `gen_sink` → `play.gen_probe` (deliver, verify, time
against eager and `torch.compile`).

Shapes `d=64 t=32 b=8` — the module probe's `large`, not the zoo default
`d16 t8 b2` where every kernel is below the launcher's noise floor.
Box: i5-12500H / RTX 2050, fp64, medians over 100 reps per round, **3 rounds
per workload**, ratio computed per round and the median of the three
reported. Every row's `max_abs` is `0.0` — the delivered module computes the
same function as the source module. The fp32 sweep (`--dtype fp32`, below)
repeats the protocol with the dtype as the only change.

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

That is the **fp64** sweep — the intake convention, kept because it is the
corpus every earlier table in this report was measured on. The number a caller
ships is the **fp32** one: **23 of 65**, in "The same sweep in fp32" below.

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
[1.146 / 1.014 / 1.64 / 1.027 / 0.891] is still unresolved in fp64 and stays out
of the count, which is why it says 17 and not 18. `DynamicTanhBlock`
[0.952 – 1.091] was the other straddler and the fp32 sweep settles it: **1.13×
in every round**. `STEQuantLinear` is the flattering version of the same
artifact (one round read 3.66× while its other two read 1.06 and 1.05); it stays
at its
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
direction is stable and the magnitude is within ~5 %. The sweep's own
module-level protocol reads **1.53×** for the same row at 100 reps — same
direction, wider rep count, so the probe's 1.38× is the conservative one.)

So the module is a **1.38× win in fp32** and a 4× loss in fp64, from the same
IR and the same fold, with no minted kernel admitted in either arm. The sweep
runs fp64 because that is the intake convention — every table in this report
was measured at it — so this row is the probe's dtype showing through, not a
defect in what the fold ships. The delivered body names the cost outright
(`bench/drivers/probes/zoo_dtype.py`, `docs/reports/zoo_dtype_rmsnorm.json`):
one `RowwiseMomentsCUDAKernel<double>` at **205.8 µs**, against a
`vectorized_layer_norm_kernel<float>` at **11.8 µs** in fp32 — a 17× difference
between two backend kernels for the same fold.

## The same sweep in fp32 — the dtype a caller ships

`--dtype` was added to the driver for this; same shape, same protocol, same
shipped path, isolated processes, 150 s budget:

```bash
.venv/bin/python -m bench.drivers.probes.zoo_chain --isolate --rounds 3 \
    --only '' --dtype fp32 --budget 150 \
    --out docs/reports/zoo_chain_sweep_fp32.json
```

**23 of 65 delivered modules beat eager in every round, 4 straddle 1, 38 lose,
median 0.979×.** The same three rows are over budget in both dtypes
(`MambaSelectiveScan`, `GatedLinearAttention`, `RealGatedLRU`) — that is
compile cost, not arithmetic — and all 65 delivered rows verify (`max_abs ≤
1e-6`, floating-point reassociation; 15 of 65 are nonzero, the other 50 are
bit-equal).

| | fp64 (convention) | fp32 (shipped) |
|---|---|---|
| delivered / verified | 65 | 65 |
| wins, every round | 17 | **23** |
| straddling 1 (excluded) | 2 | 4 |
| losses | 46 | 38 |
| median ratio | 0.969× | 0.979× |
| losses inside 10 % | 33 | 15 |

The 23 fp32 winners, µs from the round whose ratio is the median:

| workload | eager µs | gen µs | ratio (rounds) | median | fp64 median |
|---|---|---|---|---|---|
| `TernaryWeightLinear` | 53 | 17 | 2.72 / 3.05 / 3.46 | **3.05×** | 1.94× |
| `STEQuantLinear` | 99 | 59 | 1.29 / 1.68 / 1.70 | **1.68×** | 1.06× |
| `ALiBiSlopeAttention` | 202 | 124 | 1.54 / 1.64 / 1.64 | **1.64×** | 0.58× |
| `RelPosTableAttention` | 220 | 139 | 1.48 / 1.58 / 1.64 | **1.58×** | 1.04× |
| `RMSNormPerHead` | 46 | 30 | 1.53 / 1.54 / 1.33 | **1.53×** | 0.21× |
| `GroupedQuantLinear` | 32 | 21 | 1.24 / 1.56 / 1.52 | **1.52×** | 1.43× |
| `TimeCondConv` | 200 | 147 | 1.08 / 1.46 / 1.36 | **1.36×** | 1.24× |
| `PerChannelQuantMatmul` | 28 | 21 | 1.21 / 1.35 / 1.34 | **1.34×** | 1.20× |
| `PartialRotary` | 433 | 328 | 1.58 / 1.27 / 1.32 | **1.32×** | 1.47× |
| `ManualGroupNorm` | 63 | 49 | 1.27 / 1.30 / 1.37 | **1.30×** | 1.19× |
| `NormedSoftcapHead` | 76 | 59 | 1.16 / 1.31 / 1.28 | **1.28×** | 1.27× |
| `DecoupledMLA` | 419 | 327 | 1.35 / 1.28 / 1.22 | **1.28×** | 0.93× |
| `DeepEquilibrium` | 132 | 104 | 1.23 / 1.29 / 1.28 | **1.28×** | 1.31× |
| `RotaryPair` | 350 | 286 | 1.44 / 1.23 / 1.21 | **1.23×** | 1.52× |
| `LoRAAdapter` | 53 | 45 | 1.18 / 1.18 / 1.25 | **1.18×** | 1.28× |
| `GatedConvUnit` | 63 | 55 | 1.06 / 1.16 / 1.17 | **1.16×** | 1.21× |
| `DynamicTanhBlock` | 92 | 81 | 1.01 / 1.13 / 1.14 | **1.13×** | 1.02 (straddled 1) |
| `ChunkedRetention` | 158 | 148 | 1.07 / 1.14 / 1.04 | **1.07×** | 1.14× |
| `KVCacheAttention` | 150 | 141 | 1.02 / 1.06 / 1.25 | **1.06×** | 0.76× |
| `DiffAttention` | 266 | 251 | 1.02 / 1.06 / 1.07 | **1.06×** | 0.87× |
| `MLPMixerBlock` | 113 | 108 | 1.49 / 1.05 / 1.04 | **1.05×** | 0.98× |
| `RWKVMixBlock` | 486 | 467 | 1.04 / 1.03 / 1.04 | **1.04×** | 0.96× |
| `DeltaRuleBlock` | 328 | 318 | 1.06 / 1.03 / 1.01 | **1.03×** | 0.97× |

The nine modules that flip from loss to every-round win are the whole
difference: `RMSNormPerHead` 0.21× → **1.53×**, `ALiBiSlopeAttention` 0.58× →
**1.64×**, `KVCacheAttention` 0.76× → 1.06×, `DiffAttention` 0.87× → 1.06×,
`DecoupledMLA` 0.93× → 1.28×, `RWKVMixBlock` 0.96× → 1.04×,
`DeltaRuleBlock` 0.97× → 1.03×, `MLPMixerBlock` 0.98× → 1.05×,
`DynamicTanhBlock` 1.02× → **1.13×** (one of the two fp64 rows that straddled
1; it resolves here). Three go the other way and are all within noise of 1
(`HyenaLongConv` 1.11× → 1.09×, `SlidingWindowAttention` 1.005× → 0.97×,
`WaveNetGate` 1.03× → [0.952 / 1.094 / 1.047], now excluded). 14 rows win in
both dtypes, so the overlap is 61 % of the fp32 count and the flips run one
way: **fp64 understates**.

Two things this settles, and one it does not:

- **The fold's sign is a backend-kernel fact.** Same IR, same unconditional
  fold, opposite verdicts, one dtype apart. `profitable` cannot see it: it
  times minted claims against spelled sites, and a fold replaces a spelled
  site without asking.
- **The referee is not the thing that changes with dtype.** fp64 admitted
  *more* minted claims than fp32 on this corpus (`DeltaRuleBlock` 2 vs 1,
  `GatedDeltaRule` 4 vs 2, `RWKVMixBlock` 2 vs 0, `GEGLUBlock` 1 vs 0). It
  compares a compiled body against the spelled site *at the same dtype*, so
  both sides move together and the two sweeps do not explain the direction —
  only that claims are not where the dtype story lives. Folds are.
- **Not a general "fp32 is faster" claim.** `TernaryWeightLinear` gains most
  (1.94× → 3.05×) and the biggest drops are `RotaryPair` (1.52× → 1.23×) and
  `SigmoidAttention` (1.05× → 0.72×) — same machinery, different backend
  kernels underneath. Which dtype a row wins in is not predictable from here;
  both sweeps are needed to state the corpus.

**Correction to an earlier framing in this report.** The fp64 rows were once
described as "the dtype where Triton cannot bind". Triton itself is fine with
fp64 — a `@triton.jit` kernel over `float64` loads, stores and arithmetic runs
on this box — but nothing in either sweep rests on that either way: for every
minted handler on the four claim-bearing workloads probed here, in **both**
dtypes, `_triton_kernel_for` returned no binding, so the callable the referee
timed was the **compiled spelled body** every time (8 of 8). So the "no gen
kernel" part of the old framing was right by accident, and the reason was
wrong: what fp64 changes is which spelled-body kernels PyTorch runs, and that
is enough to move 9 of 65 signs.

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
  `GatedLinearAttention`, `RealGatedLRU`) and are reported as such. In fp64 all
  65 delivered rows verify bit-equal (`max_abs 0.0`); in fp32 they verify to
  `max_abs ≤ 1e-6` (15 rows nonzero, 50 bit-equal — reassociation, not a
  different function). Single GPU (RTX 2050) — the ratios are
  dispatch-sensitive, so absolute µs will move on other hardware and the *set*
  of winners may too.
- **Rows that straddle 1 are excluded, and there are 2 in fp64, 4 in fp32.**
  fp64: `CapsuleRouting` [1.146 / 1.014 / 1.64 / 1.027 / 0.891] and
  `DynamicTanhBlock` [0.952 – 1.091] after a 5-round re-measure. fp32:
  `WaveNetGate`, `HyenaLongConv`, `SkipInitNet`, `DilatedBlockAttention` — each
  with one round on the wrong side of 1 at 3 rounds. `DynamicTanhBlock` is
  settled by the fp32 run: **1.13× in every round**, so it is a win in the
  dtype that ships and unresolved only in fp64.
- **Two sweeps, and neither is the whole truth.** fp64 is the intake
  convention every earlier table used; fp32 is what a caller ships. They
  disagree on 12 of 65 modules (9 up, 3 down) and there is no rule here for
  which dtype a given module wins in. So the corpus is stated twice — **23/65
  wins, median 0.979× in fp32**, 17/65 and 0.969× in fp64 — and any single
  number quoted without its dtype is quote-mining. What is not
  dtype-dependent: the three over-budget rows, which are compile cost
  (`torch.compile` on them exceeds 150 s in both).
- **Claim-bearing rows win less often, in both dtypes.** 11 of 65 rows admit at
  least one claim in fp64 (9 in fp32), and **2 of 11** of them beat eager in
  every round, against **15 of 54** rows with no claims (fp32: 3 of 9 vs 20 of
  56). The direction is consistent and the samples are small — read it as "no
  evidence that claims help", not as "claims hurt". Every win in both tables
  comes from folds the delivery path applies unconditionally, or from fewer
  launches.
- **`fft_*` is now bound.** `FNOBlock` was undeliverable (no binding for
  `fft_rfft2`); the bridge now binds the fft family plus `real`/`imag`, and the
  exporter records an fft output shape as `s` rather than `dim`. That is what
  turns the row from missing into measurable.

## Next

1. **Price the folds, not only the claims.** `genkernel.profitable` prices
   minted claims site-by-site; every win in both sweeps and the worst loss
   come from *law* folds applied unconditionally, with no cost gate at all. A
   delivery-time check that times the delivered module against eager once and
   declines when it loses would have caught the fp64 `RMSNormPerHead` at one
   measurement per workload — and would have caught it as a dtype, not as a
   regression. It would also have caught `SigmoidAttention` in fp32 (0.72×),
   which is currently shipped as-is.
2. **Settle the four straddling rows** — `CapsuleRouting` (fp64, one round in
   five at 0.891), and in fp32 `WaveNetGate` [0.952 / 1.094 / 1.047],
   `HyenaLongConv` [0.926 / 1.103 / 1.09], `SkipInitNet` [0.989 / 1.372 /
   1.156], `DilatedBlockAttention` [0.957 / 1.084 / 1.134]. A 9-round run on
   an idle box settles each; together they are 5 rows of a 65-row count, not
   its direction.
3. **Get the three compile-bound rows under the budget** (`MambaSelectiveScan`,
   `GatedLinearAttention`, `RealGatedLRU` — over 150 s in both dtypes). Either
   a bigger budget or a `--max-autotune 0`-style compile setting; until then
   the corpus is 65 of 68 in both sweeps.
4. **Find why the referee never times a Triton kernel here.** On the four
   claim-bearing workloads, in both dtypes, `_triton_kernel_for` returned no
   binding for every minted handler (8 of 8), so `profitable` has only ever
   graded compiled-spelled bodies on this corpus. Either the tiers' `triton`
   builders are declining these patterns or the codegen has no path for them;
   the guards are now the only part of the claim story with no measurement
   attached.
