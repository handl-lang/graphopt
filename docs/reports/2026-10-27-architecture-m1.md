# Architecture & layout — M1 (for review)

## Data flow

```
                 ┌─────────────────────────────────────────────┐
                 │  kit-cuda-lang   (DATA: the IR vocabulary)  │
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
├── compile-triton/                         ← (4) NEW · lowering (engine)
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
| `kit-cuda-lang` | op vocabulary, grades, costs, denotations | code generation, measurement |
| `compile-triton` | term → Triton, one op at a time | choosing which term |
| `kernelopt` | enumeration, legality, certificates, the choice | the lowering, the referee |
| `graphopt/bench` | the honest number | the search, the semantics |
| `lib-cuda-*` | running the winner in a host | finding it |

## M1 boundary

**In:** ADR · kit with 8 ops + denotations · `compile-triton` for `qgemv`
with 4 `split_k` variants · equality gate vs `awq_gemm` · search driver
emitting a best kernel and a routing row.

**Out:** `lib-cuda` binding, M>1 tiling, FP8/GPTQ, compile cache,
`compile-cuda`, cost-model *ranking* (bound only).

**Gate:** all four variants cosine ≥ 0.9999 vs `awq_gemm`; at least one
within the hand kernel's 58.9 µs cold on `down_proj`.
