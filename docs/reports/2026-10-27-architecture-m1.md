# Architecture & layout — M1 (for review)

## Corrections from review (accepted)

**1. `kit-cuda-lang` was the wrong noun.** A **kit** is a curated manifest
of handl libs; a **lib** is a package. So the vocabulary is
**`lib-cuda-ir`** (a handl library: ops, grades, costs, denotations), and
**`kit-cuda`** is the curated kit that includes it alongside the base
types. Two artefacts, two nouns, as every other kit/lib pair in the org.

**2. `compile-triton` is Rust.** Every engine in the `compile-*` family is
Rust; emitting Triton is string generation over a term, so it belongs
there too — deterministic, no Python runtime, part of the engine family.
Executing the emitted source needs Python, but that is the runtime binding
(`lib-cuda-*`) or an ext, never the compiler.

**3. There is no new schema. The IR *is* `Term`.** This is the correction
that matters: handl already represents computation as data —
`Term::{Arr, Seq, Op, Handle, Loop, App}` with a dimension/grade slot. A
schedule does not need a parallel IR format; it needs **ops the kit
declares** (`tile_load`, `split`, `dot`, …), carried in `Term::Op` with a
grade row, and **an engine that interprets them**. So:

| before (wrong) | after (right) |
|---|---|
| new `KernelIR` schema in `kit-cuda-lang` | **`Term` values** with kit-declared ops |
| IR → Triton lowering in a bespoke module | **`compile-triton`** interprets `Term` |
| denotations in the kit | denotations in **`lib-cuda-ir`** tests + laws |

What this buys: the body is transparent **in the platform's own
representation**, so the certificate, the sink, the sieve and the cost
model all apply unchanged — no second universe to keep in sync. Where a
schedule cannot be expressed as `Term`, that is a finding about the
calculus, and the platform's rule applies: it enters by data (an op), a
desugaring, or a handler — never by a new combinator.


```
                 ┌─────────────────────────────────────────────┐
                 │  lib-cuda-ir      (DATA: the IR vocabulary)  │
                 │  ops: tile_load unpack_i4 dequant dot split │
                 │        stage reduce store bias              │
                 │  each: signature · grade row · cost · law   │
                 └───────────────────┬─────────────────────────┘
                                     │  IR terms (schedules as data)
                 ┌───────────────────▼─────────────────────────┐
                 │  kernelopt   (SEARCH + CERTIFICATE)         │
                 │  enumerate terms → legality → roofline floor│
                 │  → equality gate → deploy-referee timing    │
                 └──────┬───────────────────────────┬──────────┘
                        │ term                      │ chosen term
        ┌───────────────▼────────────┐   ┌──────────▼───────────────┐
        │ compile-triton (LOWERING)  │   │ graphopt/bench (REFEREE) │
        │ term → Triton source       │   │ regime.py: eviction,     │
        │ one op, 4 split_k variants │   │ blocked rounds, roofline │
        └───────────────┬────────────┘   └──────────────────────────┘
                        │ source
        ┌───────────────▼────────────┐
        │ lib-cuda / lib-cuda-qgemv  │   (M2 — needed to ship,
        │ runtime binding + manifest │    not to find)
        └────────────────────────────┘
```

**Two-stage schedule, stated once:** our IR decides blocks and split; Triton
decides pipelining, vectorization and layouts. Reports must say which stage
chose what, or a win cannot be attributed.

## Layout

```
handl-lang/
├── host/project/adrs/
│   └── 01XX-kernel-ir-as-language.md      ← (1) ADR: IR is data, kernel stays closed
│
├── kit-cuda-lang/                          ← (2) NEW · the vocabulary (data)
│   ├── fragments/
│   │   ├── qgemv.yaml                     op · signature · grades · cost
│   │   └── _ops.yaml                      the 8 core ops
│   ├── denotations/                       (3) torch reference per op
│   │   ├── qgemv.py
│   │   └── _shared.py
│   └── tests/
│       └── test_denotations.py            every op has a reference + a shape
│
├── kit-cuda/                               ← curated kit: includes lib-cuda-ir
│
├── compile-triton/                         ← (4) NEW · Rust engine (lowering)
│   ├── src/
│   │   ├── lower.py                       term → Triton source
│   │   └── variants.py                    split_k ∈ {1,2,4,8}
│   └── tests/
│       └── test_lower_qgemv.py            compile + run + compare
│
├── kernelopt/                              ← (6) SEARCH (exists, extend)
│   └── packages/mlopt-kernel/src/mlopt_kernel/
│       ├── schedules.py                   (NEW) IR terms as data
│       └── search.py                      (NEW) enumerate → gate → measure
│
├── graphopt/bench/                         ← (5) REFEREE (exists)
│   ├── benchkit/regime.py                 eviction · blocked rounds · roofline
│   └── drivers/probes/awq/                incumbents: hand kernel, awq_gemm,
│                                          the probes that produced P13–P18
│
└── lib-cuda/ , lib-cuda-qgemv/             ← M2 · runtime binding (deferred)
```

## What each piece owns

| layer | owns | does not own |
|---|---|---|
| `lib-cuda-ir` (+ `kit-cuda`) | op vocabulary, grades, costs, denotations | code generation, measurement |
| `compile-triton` | term → Triton, one op at a time | choosing which term |
| `kernelopt` | enumeration, legality, certificates, the choice | the lowering, the referee |
| `graphopt/bench` | the honest number | the search, the semantics |
| `lib-cuda-*` | running the winner in a host | finding it |

## M1 boundary

**In:** ADR · `lib-cuda-ir` with 8 ops + denotations · `kit-cuda` manifest · `compile-triton` for `qgemv`
with 4 `split_k` variants · equality gate vs `awq_gemm` · search driver
emitting a best kernel and a routing row.

**Out:** `lib-cuda` binding, M>1 tiling, FP8/GPTQ, compile cache,
`compile-cuda`, cost-model *ranking* (bound only).

**Gate:** all four variants cosine ≥ 0.9999 vs `awq_gemm`; at least one
within the hand kernel's 58.9 µs cold on `down_proj`.
