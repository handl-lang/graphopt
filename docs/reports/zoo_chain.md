# Chain-level pay on the real zoo corpus

**Question.** `--domain gen --probe` reports **2.87×** over `torch.compile` on a
synthetic 6-op chain. Does the same machinery pay when it has to deliver a
**real module**, at wide shapes, through the shipped path?

**Answer: no, and one workload's timing is not even reproducible.**

## Setup

`bench/drivers/probes/zoo_chain.py` runs the unmodified shipped path per
workload — build → fp64 → `TorchSource.to_ir` → mint tiers →
`genkernel.profitable` (measured-cost referee) → `gen_sink` →
`play.gen_probe` (deliver, verify, time against eager and `torch.compile`).

Shapes `d=64 t=32 b=8` — the module probe's `large`, not the zoo default
`d16 t8 b2` where every kernel is below the launcher's noise floor.
Box: i5-12500H / RTX 2050, fp64, medians over 100 reps.

```bash
.venv/bin/python -m bench.drivers.probes.zoo_chain --out docs/reports/zoo_chain.json
```

## Result (2026-10-26)

| workload | claims minted | verified | eager µs | `torch.compile` µs | gen µs | gen / eager | gen / compile |
|---|---|---|---|---|---|---|---|
| `GatedAttentionUnit` | 2 (`decayed_0`, `square`) | yes, `max_abs 0.0` | 280.4 | 320.2 | 299.4 | **0.94** | 1.07 |
| `MLPMixerBlock` | 0 | yes | 405.3 | 354.2 | 410.7 | **0.99** | 0.86 |
| `BlockSparseAttention` | 0 | yes | 394.2 | 401.4 | 412.6 | **0.96** | 0.97 |
| `TimeCondConv` | 0 | yes | 187.2 | 118.0 | 130.3 | **1.44** — see below | 0.91 |
| `FNOBlock` | 0 | — | 96.5 | — | — | **undeliverable** | — |

Repeating the whole table gives the same verdict for the three attention /
mixer rows (`GatedAttentionUnit` 0.90 / 0.94, `MLPMixerBlock` 0.99 / 0.99,
`BlockSparseAttention` 0.96 / 0.96). Delivering the module never beats eager.

## The one workload that "wins" does not reproduce

`TimeCondConv`, five independent runs of the same command:

| run | eager µs | gen µs | gen / eager |
|---|---|---|---|
| 1 | 181.6 | 177.2 | 1.03 |
| 2 | 187.2 | 130.3 | 1.44 |
| 3 | 188.7 | 146.2 | 1.29 |
| 4 | 193.8 | 156.2 | 1.24 |
| 5 | 181.7 | 431.1 | **0.42** |

Eager is stable to ±3 %; **gen swings 130–431 µs (3.3×)**. The delivered
module is bimodal — it sometimes takes a slow path that the other runs do
not. Until that path is identified, a single-run speedup for this workload
is not evidence of anything, and it is recorded as **unusable**, not a win.

## What this means

The 2.87× is real, but it is a **whole-chain kernel launched once** over a
synthetic tape. On a real module the same kernels are delivered as part of
a larger Python graph, and the result is 0.94–0.99× eager — the same verdict
the fp64 module probe reaches for the shipped folds, now on the held-out
corpus with the gen path.

Combined with `prim_kernel.md` §module level (3.17 → 3.34 at `large`,
3.23 → 2.67 at `wide` for the folds), the pattern is consistent across both
delivery mechanisms: **a generated kernel pays as an op, not as a step
inside a delivered module.**

## Limits, stated

- **5 of 68 workloads.** `torch.compile` is the baseline, so each row costs
  tens of seconds at wide shapes; `ConvNeXtStage` and `MBConvSEBlock` did not
  finish within the probe budget on this box. The table is a sample, and it
  is reported as one.
- **Two blocked artefacts.** `FNOBlock` cannot be delivered at all — there is
  no torch binding for `fft_rfft2`. `GatedAttentionUnit` mints 2 claims, which
  is the one workload where the referee found a site worth replacing — and
  the module still comes out 0.94×.
- **fp64.** Triton cannot run fp64, so the gen kernels here are the fallback
  bodies; the fp32/CUDA comparison is `prim_kernel.md`'s job, not this one's.

## Next

1. Find the bimodal slow path in `TimeCondConv`'s delivery — the only
   workload whose delivered form is sometimes 1.4× and sometimes 0.4×.
2. Give `fft_*` a torch binding so `FNOBlock` is measurable rather than
   missing.
3. Sweep the remaining 63 workloads in the background overnight, once each
   row's cost is predictable.
