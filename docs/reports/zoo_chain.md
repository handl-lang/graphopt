# Chain-level pay on the real zoo corpus

**Question.** `--domain gen --probe` reports **2.87×** over `torch.compile` on a
synthetic 6-op chain. Does the same machinery pay when it has to deliver a
**real module**, at wide shapes, through the shipped path?

**Answer: on 1 of 5 workloads, for a different reason than the 2.87×.** A
delivered `TimeCondConv` runs **1.34× eager** (3/3 rounds, `max_abs 0.0`) by
launching **22 kernels instead of 34**; a delivered `FNOBlock` runs **1.77×
`torch.compile`** (now that `fft_*` has a binding) while tying eager. Neither
win comes from a *minted* claim — the one workload whose search found claims
(`GatedAttentionUnit`, 2) is 0.95× eager. Whole-chain supremacy does not
transfer; a narrower per-workload effect does.

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

- **`TimeCondConv` pays against eager, loses to `torch.compile`.** Three rounds
  agree to 1 % (1.328–1.341): a real, reproducible win over the framework's own
  dispatch — and inductor still does better.
- **`FNOBlock` pays against `torch.compile`, ties eager.** 1.77× on all three
  rounds: `torch.compile` regresses the fft chain to ~190 µs where eager holds
  ~100 µs, and the delivered module holds eager's cost.

Everything else is a 2–5 % loss with the spread inside 4 % — tight enough that
no single round is carrying the verdict.

## Why `TimeCondConv` is faster (measured, not asserted)

Torch profiler, per forward, same process:

| | GPU kernels / iter | device time / iter |
|---|---|---|
| eager module | 34 | 171 µs |
| delivered module | **22** | **146 µs** |

The delivered module drops 12 small elementwise kernels — the fold replaces
them with fewer, wider ones — and device time falls 15 %, which is the 1.34×
wall-clock once eager's 34-way dispatch is added on top. The win is
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

Sixteen clean round-readings land in **1.19–1.42**. The single 0.42× came from
a sample taken while other jobs shared the box, and two attempts to reproduce
it deliberately failed: four CPU spinners left both arms unchanged (eager
224 µs, gen 167 µs), and it never recurred in any isolated run. The cause of
that one outlier is therefore **unidentified**, and it is recorded that way
rather than as a property of the code.

What survives is a methodological rule, now enforced by the driver: **a
module-level ratio needs repeated rounds on an idle box**, because the two
arms are timed at different moments. `--rounds` (default 3) reports a median
and keeps the spread; a single round is no longer publishable.

## What this does and does not close

The 2.87× is real, but it is a **whole-chain kernel launched once** over a
synthetic tape. Delivered into a real module, the same machinery gives a
2–5 % loss on four workloads and a 1.34× win on the one that was
dispatch-bound. Combined with `prim_kernel.md` §module level (folds 3.17 → 3.34
at `large`, 3.23 → 2.67 at `wide`), the pattern across both delivery
mechanisms is now precise:

> A generated kernel pays **as an op** (2.6×), or as the **whole chain**
> (2.87× on a synthetic tape), or where the delivered module is
> **dispatch-bound** (`TimeCondConv`, 22 vs 34 kernels). It does not pay as a
> step inside a module that is already kernel-bound.

## Limits, stated

- **5 of 68 workloads.** `torch.compile` is the baseline, so each row costs
  tens of seconds at wide shapes; `ConvNeXtStage` and `MBConvSEBlock` did not
  finish within the probe budget on this box. The table is a sample, and it is
  reported as one.
- **fp64.** Triton cannot run fp64, so the gen kernels here are the fallback
  bodies; the fp32/CUDA comparison is `prim_kernel.md`'s job, not this one's.
- **The claim-bearing workload loses.** `GatedAttentionUnit` is the only
  workload where the cost referee found a site worth replacing (2 claims), and
  the module still comes out 0.95× eager. Both wins come from folds the
  delivery path applies unconditionally, not from the search's inventions.
- **`fft_*` is now bound.** `FNOBlock` was undeliverable (no binding for
  `fft_rfft2`); the bridge now binds the fft family plus `real`/`imag`, and the
  exporter records an fft output shape as `s` rather than `dim`. That is what
  turns the row from missing into measurable.

## Next

1. Sweep the remaining 63 workloads overnight — per-row cost is now
   predictable (3 rounds ≈ 20 s), and a 1.34× here suggests a family
   (`TimeCondConv` is conditioning-heavy and dispatch-bound) rather than a
   single hit.
2. Test the dispatch-bound hypothesis directly: pick a workload whose eager
   kernel count is high and whose kernels are small, and check whether the win
   tracks the kernel-count ratio rather than the op count.
