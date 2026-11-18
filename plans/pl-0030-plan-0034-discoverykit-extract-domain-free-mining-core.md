+++
id = "PL-030"
title = "Plan 0034 — discoverykit: extract the domain-free mining core"
status = "open"
priority = "high"
+++

---
title: 'Plan 0034 — discoverykit: extract the domain-free mining core'
type: plan
date: 2026-11-19
---

# Plan 0034 — discoverykit: extract the domain-free mining core

Origin: architecture review (2026-11 session). The self-improvement
machinery — corpus mining, antiunification, handler synthesis, the
meta-arena, law minting — lives inside `mlopt_discovery` in graphopt,
but its core operates on `mlopt_core` terms with zero torch dependency.
A second domain (kernelopt, host-tensor/lib-tensor) cannot reuse it
without inheriting torch-shaped collectors. This plan extracts the
generic half into a sibling repo, mirroring the gamekit extraction
(plan 0032 lane A).

## What moves (domain-free, term-generic)

- `antiunify`, `handler_synth` — candidate law mining
- `combinator`, `form_propose`, `state_reindex`, `index_abstract`,
  `loop_propose` — already promoted prod modules
- Meta-arena core (`meta_arena` machinery minus torch collectors),
  `ProbeBounds` (landing now), skip-stat accounting
- `agenda_miner` generic scanning, overnight-loop machinery
- law-minting gauntlet drivers that are not torch-shaped

## What stays in graphopt

- `census`, `model_corpus`, `workload_gen`, torch site collectors
- the `torch` domain registry entry and torch-shaped play arms
- minted-store records that name torch ops

## Phases

**Phase 0 — dependency audit (gate).** Prove the move set is
 torch-free: `import mlopt_discovery.<mod>` under a torch-less env,
or static import-linter contract. Any hidden torch import blocks the
move or forces a shim boundary. Output: the definitive move list.

**Phase 1 — repo + package.** New `handl-lang/discoverykit` repo,
`packages/discoverykit` (stdlib + mlopt-core dep only, same contract
style as gamekit). Move modules; leave re-exporting shims at the old
`mlopt_discovery.*` paths (the gamekit playbook — zero downstream
breakage).

**Phase 2 — pin + ratchet split.** graphopt + kernelopt pin
discoverykit git-tag; the 99% coverage ratchet splits per repo like
gamekit's did. `uv.lock` refresh is deliberate and loud.

**Phase 3 — second-domain seat.** kernelopt (or a lib-tensor domain)
consumes discoverykit for its own vocabulary — the proof the
extraction was real, not cosmetic. Success = a minted law on a
non-torch domain through the shared machinery.

## Hard gates

- Phase 0 audit clean: zero torch imports in the move set.
- Shims keep every existing import path working (tests unchanged).
- Coverage floors re-measured per repo; tighten, never lower.
- kernelopt Phase-3 mint produces ≥1 gauntlet-admitted law or an
  honest documented negative.

## Dependencies / adjacency

- `ProbeBounds` (in flight) lands first — it is the typed-budget
  object discoverykit ships.
- `mlopt_core.traversal` (eqsat `v0.1.0`) is the shared walker util
  the moved modules already use.
- Plan 0035 (typed ports) defines the cross-package protocols
  discoverykit's public surface must satisfy.
