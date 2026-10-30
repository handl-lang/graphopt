---
title: Evaluation is an independent dimension
type: adr
date: 2026-10
status: accepted
---

# ADR 0003 — Evaluation is an independent dimension

## Decision


Adopt **four independent dimensions** with explicit ports, and keep
them independent.

```text
SEMANTICS ──▶ SEARCH ──▶ EVALUATION ──▶ EXECUTION
   │            │             │              │
 "equivalent?" "explore?"  "how does it   "what actually
                            look / run?"    ran?"
```

**Rule.** Semantics, search, evaluation and execution are independent
dimensions; no layer may become responsible for another's question.

Each dimension answers exactly one question:

| Dimension | Question | Owns | Must not |
|---|---|---|---|
| Semantics | Is it equivalent? | `EGraph`, `verify_certificate` | price or time anything |
| Search | What should we explore? | `Engine`, policy | decide equivalence |
| Evaluation | What does it look like / what might run well? | profiler, cost, performance model | change the function |
| Execution | What actually ran? | `Sink`, `Meter`, hardware adapter | define semantics |

### Decisions in detail

1. **Evaluation splits from execution.**  A `Profiler` observes a
   program *without running it on the target*; an `Executor` runs it;
   the existing `Meter` measures a run — there is **no separate
   `Benchmark` port**; `Meter` gains provenance and failure
   classification.  **Profiling does not imply a GPU** — the static
   half is torch-free, in core.
2. **Keep the frontier; do not scalarize early.**  The candidate set
   is what `discover_alternatives` already enumerates
   (`SearchResult.alternatives`); selection keeps the non-dominated
   subset over named dimensions.  A scalar `best()` is a convenience
   view, not the internal contract — today's `criteria={...}` weighted
   blend is that view.  Note: `extract_best` minimizes one scalar by
   construction, so true multi-objective *extraction* is new work, not
   a re-view of the existing one.
3. **One reference hardware backend, not one architecture.**  The
   first GPU profiler/benchmarker is the first *instance* behind the
   ports.  `cuda` / `rocm` / `tpu` names never enter core; the
   dependency arrow points core ◀ adapter, never core ▶ CUDA.
4. **Policy is a replaceable port, RL included.**  RL is a *policy*
   over an already-structured space — never the mechanism of reach.
   The first policies are random / greedy / heuristic / existing; RL
   is one more conforming value.  `Policy.choose(state, actions)`
   threads through the existing `Engine.run` schedule / `stop` hook —
   it is not a parallel search mechanism.  This **amends ADR 0002
   Rule 3**.
5. **The performance model is a port.**  Initially analytical, later
   learned; it *predicts*, it never decides correctness and never
   bounds the semantic space.
6. **Benchmarks are last.**  Rewrite benchmarks only after the
   boundaries settle, around research questions, not just speed
   numbers.

**Feasibility is hard; performance is soft.**  Two mechanisms already
bound the search and must stay distinct: `supported_ops` is a *hard*
feasibility bound (a form using an unlowerable op prices at `+inf` and
is never selected); a performance model is a *soft* ranking (it orders
feasible members and may never drop one).  Invariant 8 makes this
binding.


## Why

CatOpt today proves *reach* and prices it with a scalar, static, calibrated cost model.  The pipeline is lift → saturate → extract → certify → lower.  Pricing is a `CostFn` (selected by a `Criterion`) over a `TargetProfile`'s measured constants.  Measurement exists, but only as an autotune *selector* inside one strategy (`Autotuned` via the `Meter` port), and its numbers are folded back into the cost model as multiplicative corrections.

## Not doing

- **Keep the scalar cost model + autotune (status quo).**  Rejected:
- **Start with RL.**  Rejected: without a profiler, a frontier and a
- **A new top-level `catopt` façade for the "clean" API.**  Rejected:
- **Put program features in `catopt_core.profile`.**  Rejected: name
- **Make the GPU implementation the abstraction.**  Rejected: CUDA is

## Consequences

- The engine gains an explicit `Policy` seam below `Strategy`, so
- `discover_alternatives` becomes the entry to a Pareto frontier, not
- `Meter` gains provenance and failure classification; `Autotuned`
- New protocols are new surface: they must be typed (`ty`), tested
- The road to a learned performance model is opened without changing

## Enforcement

- Code and tests are named in the design doc.

## Design

The full record (context, tables, phases, trade-offs): [`design/0003-evaluation-is-an-independent-dimension.md`](../design/0003-evaluation-is-an-independent-dimension.md).
