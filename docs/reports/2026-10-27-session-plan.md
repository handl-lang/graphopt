# This session's plan — one page

**Goal:** make the machine find the kernel instead of me. Today I wrote it;
the machine only tuned 48 settings of my design.

**Baseline already proven** (no new work needed): our int4 GEMV is
**2.96× / 1.52× / 1.35×** over `awq_gemm` on three of four layer shapes,
cold and exact; dropped into vLLM it is **token-identical and 29.2 %
faster**. Everything below is about making that automatic.

## Phases

| phase | work | gate | if it fails |
|---|---|---|---|
| **M1a** | ADR-0160 — the IR is `Term` + kit-declared ops | accepted | stop; nothing builds on sand |
| **M1b** | `lib-kernel-ir` (public) — nine ops: signature, grade, cost, torch denotation | every op has all four | IR is not expressive enough — finding about the calculus |
| **M1c** | equality gate vs `awq_gemm` in `graphopt/bench` | cosine ≥ 0.9999, inside the timing path | fix the gate before any codegen |
| **M1d** | `compile-triton` (Rust) — `Term` → Triton, `qgemv` + `gate_proj`, 4 `split_k` each | 8 variants compile, run, and pass the gate | **kill:** if generated code can't match hand-written Triton, keep hand-writing |
| **M1e** | search driver in `opt-kernel` | emits best kernel per shape + routing row | search has no objective |
| **M1.5** | compile cache + roofline pruning | 1 in 10 candidates ever compiled | breadth is rate-limited by `triton.jit` |
| **M2** | `lib-triton` binding + `kit-kernel-ir` | a found kernel runs in a host | — |
| **M3** | breadth: GPTQ/FP8, more shapes | routing table with no hand entries | — |
| **M4** | dissolve into `ext-opt-*` (ADR-0159 triggers) | a non-researcher caller | — |

## Order, with dependencies

```
ADR-0160 ─▶ lib-kernel-ir ─▶ equality gate ─▶ compile-triton ─▶ search ─▶ M1 gate
                    └─────────── denotations are the gate's reference ────┘
```

The gate comes **before** the lowerer on purpose: three fake results this
week came from checking against my own reimplementation, or checking
nothing.

## Decisions already made

- **Lab-form** (ADR-0159): repos and drivers, no `ext` wrapper, no FFI,
  until three triggers fire.
- **Naming:** `opt-graph` (today `graphopt`), `opt-kernel` (today
  `kernelopt`); third member only on the rule of three.
- **Visibility** (ADR-0150): `lib-kernel-ir` public — the vocabulary is the
  public artefact; `compile-triton`, the binding and any `ext` stay
  private until done.
- **The IR is `Term`** — no new schema; the six combinators stay closed.
- **Triton first**, CUTLASS only on a measured gap.

## Needs you

1. **Repo creation**: `lib-kernel-ir`, `compile-triton` (say go, or create
   them and I fill them).
2. **Repo ownership** for this session, so we don't overwrite each other.
3. **GPU window** — I stay off it while you benchmark.
4. **ADR-0160 acceptance** — the decision should carry your name.
