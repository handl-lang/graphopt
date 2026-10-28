# Proposal — a vocabulary for int4 kernels, so the machine can find them

**Status:** proposed
**Problem it solves:** today the int4 GEMV was *written by hand*; the
machine only swept 48 configs of my design. To make "the machinery finds
kernels" true, the design space has to be data, not code — and the
correctness claim has to be checkable, not asserted.

## The split: meaning in handl, delivery in kernelopt

Kernels have two vocabularies and only one belongs in the arrow calculus.

| layer | what it says | where it lives |
|---|---|---|
| **semantics** | `qgemv(x, w_q, z, s) ≡ matmul(x, dequant(w_q, z, s))` | handl substrate: an arrow law, certifiable |
| **schedule** | tiles, split-K, load width, register unpack, reduction | kernelopt: IR + op table + cost model |

Arrows compose meaning well and schedules badly (shared memory and
occupancy are not composition). Keeping them apart lets the certificate
speak about *meaning* while the search varies *delivery* — the same
"one calculus, many implementations" pattern the platform already uses
for engines.

## What gets added

**1. Ops, with costs (kernelopt op table).**
`load_i4`, `dequant_group`, `gemv_dot`, `stage_partial`, `reduce_add`,
`epilogue_bias`. Each entry carries its byte count and a calibrated cost,
so a candidate can be *priced* against the roofline before it is built.

**2. A schedule record (data, not code).**

```python
KernelTemplate(
    layout="packed_int4",          # vs float
    delivery="gemv",               # vs tiled
    split_k=8,                     # 1, 2, 4, 8, 16
    reduce="two_stage",           # vs atomics (retired: wrong answers)
    block_n=64, block_k=64, warps=2,
    load_bits=128,                # vector width
    out_dtype="fp16", acc="fp32",
    epilogue=("bias",),           # optional fused tail
)
```

A generator turns any instance into Triton source. Nothing about the
*structure* is named by a person at search time.

**3. A search that prunes before it measures.**
legality (shape/group/alignment) → roofline bound from the cost model →
exactness against the incumbent → only then the deploy-referee timing.

**4. The gate that makes the claim.**
> The machine must reproduce or beat the hand-written baseline
> (`2.96×` / `1.52×` / `1.35×` over `awq_gemm`) on ≥ 3 geometries,
> **without the structure being specified by a human**, and with every
> timed candidate exact (cosine ≥ 0.9999 vs the incumbent).

Beating **42 % of roofline** on `down_proj` is the stretch target: my hand
kernel is at 42 %, so anything above it was found, not transcribed.

## Milestones

| # | deliverable | proves |
|---|---|---|
| **M1** | template family + generator + exactness gate; search reproduces my kernel | the space is expressible |
| **M2** | cost-model pruning (measure 1 in 10) | the search is affordable |
| **M3** | add `tiled` delivery and `bias`/epilogue ops | coverage of `gate_proj` and prefill (M>1) |
| **M4** | emit the routing table from the search; package the winner | it ships |

## Non-goals

- Not deriving Triton from arrow terms. Arrows state the claim; the
  schedule states the implementation.
- Not cross-GPU autotuning yet — the cost model is calibrated per machine
  (`scripts/calibrate_profile.py`), and every number here is one 4 GB card.
- Not supporting Marlin's repacked layout; the substitution targets the
  plain AWQ path.

## Risks and their counters

- **Search cost** (compile time dominates): the cost model prunes to ~10 %
  of candidates before any compile.
- **Noise**: deploy-regime referee, blocked rounds, medians — the protocol
  that already caught three bad numbers of mine.
- **Correctness leakage**: the AWQ nibble-order bug survived because my
  reference shared my assumption. The gate compares against the
  **incumbent kernel**, not against our own reimplementation.
- **Overfitting to one GPU**: routing tables ship with the calibration
  they were measured on, and are regenerated per machine.
