# The frontier — theoretical headroom vs delivered capture

**Bound.** A rewrite may only replace a term with an *equal* term, so it cannot change the arithmetic a dense model must do. The only headroom it can reach is the off-device share of wall time, so `ceiling = 1/(1-F)` with `F` the measured dispatch share (`1 - device_share`, profiler).

**Capture.** `captured = (speedup-1)/(ceiling-1)`: the fraction of that headroom the delivered body actually takes. Negative means delivery costs more than the algebra saves.

- rows: 65 · verified: 65/65 · win tolerance: ≥1.02x

## Findings

### NEGATIVE — An equality-preserving rewrite cannot change a dense model's FLOPs, so its ceiling is the dispatch share, 1/(1-F). Where F is small the ceiling is ~1x.

compute-bound rows are capped at 1.03x — equality-preserving rewrites cannot reach them

### NEGATIVE — The delivered body captures the headroom the ceiling allows: (speedup-1)/(ceiling-1).

median capture -0.03 over 35 rows with material headroom — delivery loses more than the algebra gains; the binding constraint is the delivered body, not the ceiling

### INCONCLUSIVE — Dispatch share predicts the sign of the win. (It does not always — the delivered body's own cost sets it.)

r(F, speedup) = +0.08 — dispatch share does not predict the sign alone

### WIN — The wins survive `torch.compile` — the free competitor every PyTorch user already has — not just eager.

13/25 launch-bound rows beat torch.compile (median 1.11x); compute-bound 13/28 at 0.99x

### WIN — Every delivered row is verified equal to the spelled module — the frontier is measured on correct deliveries.

65/65 rows verified equal

## By regime

| regime | F range | n | median F | median ceiling | max ceiling | median speedup | wins | beat compile | median captured |
|---|---|---|---|---|---|---|---|---|
| compute-bound | 0.00-0.05 | 28 | 0.028 | 1.03x | 1.05x | 0.983x | 8/28 | 13/28 | — |
| mixed | 0.05-0.15 | 5 | 0.106 | 1.12x | 1.17x | 0.877x | 2/5 | 3/5 | -0.74 |
| launch-leaning | 0.15-0.30 | 7 | 0.220 | 1.28x | 1.39x | 0.907x | 0/7 | 2/7 | -0.29 |
| launch-bound | 0.30-1.00 | 25 | 0.552 | 2.23x | 5.95x | 1.006x | 12/25 | 13/25 | +0.01 |

## Rows

| case | F | ceiling | speedup | captured | kernels | eager µs | delivered µs |
|---|---|---|---|---|---|---|---|
| `zoo:TaylorLinearAttention` | 0.010 | 1.01x | 0.996x | — | 11 | 1039.6 | 1043.3 |
| `zoo:TalkingHeadsAttention` | 0.015 | 1.02x | 1.258x | — | 5 | 600.5 | 477.2 |
| `zoo:MLPMixerBlock` | 0.015 | 1.02x | 0.984x | — | 8 | 405.7 | 412.2 |
| `zoo:LearnedPosSlice` | 0.016 | 1.02x | 0.980x | — | 4 | 170.3 | 173.7 |
| `zoo:GEGLUBlock` | 0.017 | 1.02x | 0.962x | — | 7 | 258.8 | 269.0 |
| `zoo:SquaredReluFFN` | 0.019 | 1.02x | 0.951x | — | 6 | 240.6 | 252.9 |
| `zoo:AdditiveAttention` | 0.022 | 1.02x | 1.243x | — | 9 | 330.2 | 265.6 |
| `zoo:AttnLogitSoftcap` | 0.022 | 1.02x | 1.237x | — | 7 | 347.4 | 280.8 |
| `zoo:GatedAttentionUnit` | 0.024 | 1.02x | 0.973x | — | 7 | 279.8 | 287.7 |
| `zoo:AdaLNBlock` | 0.024 | 1.02x | 1.116x | — | 8 | 308.9 | 276.7 |
| `zoo:RWKVMixBlock` | 0.024 | 1.02x | 0.945x | — | 21 | 1067.5 | 1130.1 |
| `zoo:SkipInitNet` | 0.024 | 1.02x | 0.991x | — | 4 | 293.8 | 296.6 |
| `zoo:SigmoidAttention` | 0.025 | 1.03x | 0.962x | — | 6 | 301.7 | 313.5 |
| `zoo:DynamicTanhBlock` | 0.027 | 1.03x | 1.000x | — | 6 | 221.5 | 221.5 |
| `zoo:FiLMHead` | 0.029 | 1.03x | 0.824x | — | 5 | 120.5 | 146.2 |
| `zoo:SoftSlotMoE` | 0.029 | 1.03x | 0.974x | — | 7 | 196.0 | 201.2 |
| `zoo:SlidingWindowAttention` | 0.029 | 1.03x | 1.007x | — | 11 | 336.6 | 334.3 |
| `zoo:DeepEquilibrium` | 0.030 | 1.03x | 1.301x | — | 3 | 280.0 | 215.2 |
| `zoo:CosineAttention` | 0.031 | 1.03x | 0.989x | — | 8 | 351.9 | 355.7 |
| `zoo:RelPosTableAttention` | 0.033 | 1.03x | 1.297x | — | 11 | 389.7 | 300.4 |
| `zoo:ALiBiSlopeAttention` | 0.034 | 1.04x | 0.578x | — | 8 | 335.6 | 580.2 |
| `zoo:BlockSparseAttention` | 0.037 | 1.04x | 0.957x | — | 15 | 350.3 | 366.2 |
| `zoo:SinkWindowAttention` | 0.037 | 1.04x | 0.982x | — | 15 | 375.3 | 382.2 |
| `zoo:DilatedBlockAttention` | 0.038 | 1.04x | 1.055x | — | 12 | 357.7 | 339.2 |
| `zoo:LoRAAdapter` | 0.040 | 1.04x | 1.288x | — | 4 | 93.0 | 72.2 |
| `zoo:DiffAttention` | 0.043 | 1.04x | 0.864x | — | 10 | 319.7 | 370.0 |
| `zoo:DeepSeekSigMoE` | 0.046 | 1.05x | 0.976x | — | 14 | 335.6 | 343.8 |
| `zoo:SplineKAN` | 0.047 | 1.05x | 0.941x | — | 6 | 146.2 | 155.3 |
| `zoo:HyenaLongConv` | 0.050 | 1.05x | 1.082x | — | 9 | 245.1 | 226.5 |
| `zoo:HighwayGate` | 0.056 | 1.06x | 0.877x | — | 6 | 99.4 | 113.3 |
| `zoo:MoSHead` | 0.106 | 1.12x | 0.826x | -1.47 | 8 | 105.2 | 127.4 |
| `zoo:ChunkedRetention` | 0.128 | 1.15x | 1.147x | +1.00 | 11 | 309.5 | 269.8 |
| `zoo:SharedSwiGLUMoE` | 0.143 | 1.17x | 0.876x | -0.75 | 17 | 352.2 | 402.2 |
| `zoo:AffineCoupling` | 0.176 | 1.21x | 0.936x | -0.30 | 7 | 91.7 | 98.0 |
| `zoo:SoftcapLogitHead` | 0.206 | 1.26x | 0.929x | -0.27 | 4 | 54.9 | 59.1 |
| `zoo:PagedKVAttention` | 0.206 | 1.26x | 0.805x | -0.75 | 7 | 179.2 | 222.5 |
| `zoo:GCNLayer` | 0.220 | 1.28x | 0.899x | -0.36 | 6 | 131.5 | 146.2 |
| `zoo:TopKRouterMoE` | 0.245 | 1.32x | 0.905x | -0.29 | 16 | 283.5 | 313.3 |
| `zoo:ExpertChoiceMoE` | 0.262 | 1.36x | 0.907x | -0.26 | 11 | 163.5 | 180.3 |
| `zoo:DecoupledMLA` | 0.280 | 1.39x | 0.907x | -0.24 | 17 | 500.3 | 551.7 |
| `zoo:GRUFusedGate` | 0.319 | 1.47x | 0.716x | -0.61 | 8 | 122.3 | 170.8 |
| `zoo:ConvNeXtStage` | 0.328 | 1.49x | 0.978x | -0.05 | 8 | 105.5 | 107.9 |
| `zoo:PartialRotary` | 0.359 | 1.56x | 1.473x | +0.84 | 15 | 612.2 | 415.6 |
| `zoo:KVCacheAttention` | 0.364 | 1.57x | 0.757x | -0.43 | 6 | 191.5 | 253.1 |
| `zoo:TernaryWeightLinear` | 0.405 | 1.68x | 1.936x | +1.38 | 7 | 94.5 | 48.8 |
| `zoo:WaveNetGate` | 0.419 | 1.72x | 1.045x | +0.06 | 7 | 85.7 | 82.0 |
| `zoo:CovarianceHead` | 0.472 | 1.89x | 0.828x | -0.19 | 4 | 33.3 | 40.2 |
| `zoo:AdaGNConv` | 0.472 | 1.89x | 0.928x | -0.08 | 9 | 145.4 | 156.6 |
| `zoo:CapsuleRouting` | 0.504 | 2.02x | 1.070x | +0.07 | 13 | 279.1 | 260.9 |
| `zoo:DeltaRuleBlock` | 0.528 | 2.12x | 1.006x | +0.01 | 9 | 309.5 | 307.6 |
| `zoo:RotaryPair` | 0.534 | 2.15x | 1.571x | +0.50 | 12 | 419.8 | 267.3 |
| `zoo:MBConvSEBlock` | 0.552 | 2.23x | 0.938x | -0.05 | 10 | 137.0 | 146.1 |
| `zoo:NormedSoftcapHead` | 0.552 | 2.23x | 1.265x | +0.22 | 9 | 98.7 | 78.0 |
| `zoo:TimeCondConv` | 0.570 | 2.33x | 1.250x | +0.19 | 18 | 182.7 | 146.2 |
| `zoo:STEQuantLinear` | 0.589 | 2.43x | 1.108x | +0.08 | 6 | 133.6 | 120.6 |
| `zoo:GatedDeltaRule` | 0.605 | 2.53x | 0.951x | -0.03 | 12 | 431.0 | 453.2 |
| `zoo:GatedConvUnit` | 0.649 | 2.85x | 1.211x | +0.11 | 5 | 78.2 | 64.6 |
| `zoo:ChunkedSSD` | 0.669 | 3.02x | 0.876x | -0.06 | 16 | 2618.2 | 2989.8 |
| `zoo:RMSNormPerHead` | 0.700 | 3.33x | 0.207x | -0.34 | 5 | 47.9 | 230.9 |
| `zoo:InstanceNormBroadcast` | 0.743 | 3.89x | 0.963x | -0.01 | 6 | 77.6 | 80.6 |
| `zoo:SqueezeExcite` | 0.757 | 4.12x | 0.947x | -0.02 | 5 | 68.4 | 72.2 |
| `zoo:GroupedQuantLinear` | 0.757 | 4.12x | 1.414x | +0.13 | 3 | 32.1 | 22.7 |
| `zoo:PerChannelQuantMatmul` | 0.773 | 4.41x | 1.246x | +0.07 | 3 | 28.9 | 23.2 |
| `zoo:ManualGroupNorm` | 0.776 | 4.46x | 1.192x | +0.06 | 6 | 72.1 | 60.5 |
| `zoo:FNOBlock` | 0.832 | 5.95x | 0.886x | -0.02 | 11 | 117.2 | 132.3 |

## What this bounds, and what it does not

- It bounds **dispatch-removing** rewrites. It does not bound vocabulary growth: a *new* operation can change the arithmetic a module performs (blocked scans, sparsity, low rank), and that escapes this ceiling entirely.
- It assumes the delivered body performs the same arithmetic as eager. A delivery that changes dtype upstream of verify (`fp32` where the contract allows) wins outside the model, which is why `captured > 1` is reported rather than clipped.
- `F` is measured on the spelled module. A case whose eager form already fuses (a single SDPA call) has `F≈0` for the right reason: there is nothing off-device left to remove.
- `capture` is only defined where the ceiling holds at least 10% room; below that the denominator is timing noise, so those rows report `—` rather than a ratio that a 2% wobble would swing by 100%.

_Generated by `bench/suites/integration/frontier.py` from `docs/reports/zoo_chain_sweep.json` + `zoo_kcount.json`. Re-run; do not hand-edit._
