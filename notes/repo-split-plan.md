---
title: Note — repo split plan (proposal only, not enacted)
type: note
date: 2026-10-27
---

# Note — repo split plan (proposal only, not enacted)

2026-10-19.  Target layout once the Board protocol stabilizes.

## The four repos

| repo | owns |
|---|---|
| `gamekit` | Board protocol, players package, episode/train/eval drivers, calibrated-gate machinery (open_self/advantage/vlabel/calibrate), benchkit |
| `eqsat` | the shared substrate: e-graph, certificates, admission gauntlet, `dag_cost`, Sink port, `mlopt-native` (Rust core) |
| `graphopt` (was research-mlopt) | tensor law bank, discovery machinery, torch/onnx/handl sinks, cases, paper |
| `kernelopt` | codegen rules, kernel sinks (Triton/executor), kernel board, paper |

## Reuse boundary

- **Shared machinery lives in `eqsat`** — mint/verify/
  cost/lower+verify is domain-generic; both research repos depend on
  it as a versioned package.
- **Bank contents stay per-repo** — tensor laws ≠ codegen laws; each
  research repo owns its bank, cases, and domain registration.
- **The Sink port is the shared contract** — a `KernelSink` implements
  the same lower+verify interface ONNX already proved lands with zero
  core edits.
- **`gamekit` depends on nothing below the Board level** — boards in
  either research repo register against it.

## Sequencing (trigger, not schedule)

1. Freeze the Board/player interfaces behind a versioned contract
   (in-repo, worthwhile regardless).
2. Extract `eqsat` + `gamekit` when a domain change needs
   zero substrate changes for a few releases.
3. Each research repo gets `paper/` — kopt reuses the catopt quarto
   scaffold (theme, bib conventions, honest-results register).

## Deferred / open

- Names settled: `gamekit` / `eqsat` / `graphopt` /
  `kernelopt` (parallel -opt pair).
- Internal package renames queued with the split: `mlopt_core`→`egraph_core`,
  `mlopt_discovery`→`lawmint`, `bench/discovery`→`scorecard`,
  `mlopt_orchestrator`→`pipeline`, `players`→`gamekit`.
- Whether `gamekit` merges into `eqsat` (one shared lib)
  vs stays separate — decide at extraction; lean separate (game
  protocol is orthogonal to rewriting).
- CI/gate duplication cost per repo — accept at extraction time.
