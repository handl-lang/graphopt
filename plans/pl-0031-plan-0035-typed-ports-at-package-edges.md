+++
id = "PL-031"
title = "Plan 0035 — typed ports at package edges"
status = "done"
priority = "med"
+++

---
title: 'Plan 0035 — typed ports at package edges'
type: plan
date: 2026-11-19
---

# Plan 0035 — typed ports at package edges

Origin: architecture audits (2026-11 session, all five reports).
Composition works inside packages; boundaries between them are
convention — stringly-typed dicts, bare tuples, scattered caps.
`ProbeBounds` (in flight) is the template; this plan generalizes it.

## The pattern

Every cross-package protocol becomes a typed object:

- `ProbeBounds`/`Budget` — spec caps, key caps, claim/time budgets,
  skip-stat counters (landing via the bounds refactor)
- `ex[...]` episode state → typed `SearchState`/`EpisodeState`
- `TORCH_BINDINGS`/`_IR_TO_TORCH` → generated `aten_classify`
  records (torch_port_gen, in flight)
- referee dicts → `ScoringSpec`-style objects (gamekit pattern,
  extended to discovery)
- skip counters → fields on the bounds object, not scattered ints

## Phases

**Phase 0 — inventory.** Enumerate every `ex[`-style dict, `*_BINDINGS`
table, and `**kwargs`-shaped cross-module call in mlopt_discovery,
mlopt_torch, mlopt_carriers, gamekit seams. Output: boundary list
with owner + proposed type.

**Phase 1 — ProbeBounds lands** (in flight). Generalize: the same
treatment for `SearchState` and the binding records.

**Phase 2 — migrate call sites** behind the typed objects; old dict
access becomes properties/shims during transition.

**Phase 3 — lint rule.** New stringly-typed cross-package state
fails review; document the rule in AGENTS.md.

## Hard gates

- No behavior change: identical search/extraction results on the
  regression corpus (tests + a real-model smoke).
- Each migration is independently revertible (shim layer).
- The dual-metric `spec_node_cap` bug (tree-size vs op-expansion)
  is resolved by the bounds object — one metric, named honestly.

## Dependencies / adjacency

- Feeds plan 0034 (discoverykit ships typed surfaces, not dicts).
- `mlopt_core.traversal` + `RuleSet.scoped()` (eqsat v0.1.0) are the
  substrate precedents this plan extends to the domain layer.
