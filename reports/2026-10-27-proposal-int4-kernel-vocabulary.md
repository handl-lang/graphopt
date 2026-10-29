---
title: Proposal — a vocabulary for int4 kernels, so the machine can find them
type: report
date: 2026-10-27
---

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
| **schedule** | tiles, split-K, load width, register unpack, reduction | a **grade** on the op, priced by the cost model |

Graded arrows, enriched hom-objects and comonads cover scheduling as a
resource statement, so these two layers are the *same* calculus at
different grades rather than two systems: the law fixes the meaning, the
grade prices the delivery.

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

## How this uses handl — and what it expands

Nothing above needs a new platform concept. It uses three mechanisms the
platform already has, and expands exactly one of them: the vocabulary.

### 1. The op surface is a **kit** (data, not code)

A kit is authored YAML merged into a `Language` (`kit-prelude-lang`
pattern). A new `kit-kernel-ir` (M2) declares the operators and their grades:

```yaml
# kit-kernel-ir/ (M2)fragments/qgemv.yaml
op: qgemv
signature: (Vec, PackedI4, ZeroPoints, Scales) ~[(Op, Tensor)]~> Vec
backend: cuda
capabilities: [f16, f32-acc]
cost: { bytes: "k*n/2 + (k+n)*2", launches: 1 }
laws: [qgemv_denotes_matmul]      # the certificate's claim
```

The engine sees an ordinary graded `Term::Op`; it has no idea a GPU is
involved. This is the existing "language as data" boundary — a new op is a
data change, not an engine change, which is the property
`explanation/language-as-data.md` already claims.

### 2. The implementation is a **backend library** (`lib-*-c` pattern)

`lib-triton-qgemv` is a cdylib discovered by the host through the ADR-0149
extension ABI (`libhnl_ext_*.so` + manifest), exactly like `lib-intl-c`,
`lib-unicode-c` and the other native adapters. It exports the Triton
kernel and binds it to the `qgemv` capability. The port/adapter split the
`lib-*` repos already use applies unchanged: `lib-triton` is the contract,
`lib-triton-*` are implementations.

### 3. The schedule is a grade — not an excuse

Earlier drafts of this proposal said "arrows cannot say tiling". That was
wrong, and it is worth being precise about what the theory actually gives:

- **Graded arrows** (already the platform's core) attach a resource row to
  an arrow. A schedule is a resource statement: bytes read, launches,
  occupancy class, residency. That is a grade, not a new concept.
- **Enriched categories** put structure on the hom-object: cost as
  enrichment is the standard "quantitative semantics" move, and it makes
  "find the cheapest derivation" a well-formed question rather than a
  benchmark.
- **Comonads** model dataflow/scheduling structure (Uustalu–Vene), which is
  what a tiled pipeline is.
- **Operads** model hierarchical, parallel composition — the natural shape
  of a tiled kernel decomposed into blocks.
- **Traced monoidal structure** (the `loop` combinator already in the
  kernel) covers the accumulation reduction.

So expressibility is not the obstacle. Two honest obstacles remain:

1. **The kernel body is opaque.** A `triton.jit` function is not a term;
   only its *interface and cost* can be modelled in the calculus. The body
   stays a backend binding (`lib-triton-*`), and the law relates the op's
   meaning to the spelled computation — not the text of the kernel.
2. **Whether grade-encoding pays is empirical.** Encoding schedules as
   grades buys priced search over derivations; encoding them as a plain
   data template buys cheap instantiation. Which is better for finding
   kernels is testable, and the cheap test is one axis: encode `split_k`
   as a grade, run the search, and compare against the same space
   enumerated as data.

That is the design consequence: **the schedule is a grade vector on the
op** — `(bytes, launches, occupancy)` — with the cost model as the
enrichment, and the Triton body as an opaque backend. The search then
minimises cost over derivations rather than over a hand-written grid,
which is the thing the platform was built to do.

### What it expands

The vocabulary, in two directions, and both are the *point*:

- **New ops become namable.** `qgemv`, `dequant_group`, `epilogue_bias`
  are things the calculus could not say before. In the measured record,
  novelty only appears when the machine may name a new operation
  (**0/39** machine-original when composing known rules, **260/262** when
  it could mint new ones). This proposal is how a kernel family becomes
  nameable at all.
- **A new backend family.** CUDA joins the existing port/adapter
  hierarchy, so "the same op, delivered differently" is the same pattern
  as `lib-str-handl` vs `lib-str-c`, not a special case.

### The seam, stated plainly

| layer | artifact | who owns it |
|---|---|---|
| meaning | `qgemv ≡ matmul∘dequant`, a law | handl (`kernelopt`/`eqsat` certificate) |
| naming | `kit-kernel-ir` (M2) YAML declaration | handl kit |
| binding | `lib-triton-qgemv` cdylib + manifest | host plugin ABI |
| delivery | `KernelTemplate` → Triton source | kernelopt IR |

If a later experiment cannot be expressed through these four, that is a
finding about the platform, not a gap in this proposal.


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
