+++
id = "ADR-0002"
title = "ADR 0002 — The categorical re-expression thesis"
status = "open"
priority = "medium"
+++

---
title: The categorical re-expression thesis
type: adr
date: 2026-09
status: accepted
---

# ADR 0002 — The categorical re-expression thesis

## Decision


Adopt one thesis; make every capability answer to it.

**Thesis.** A tensor program is a morphism in a (hyper)graph
monoidal category. Optimization is search in the quotient of the
free category on the IR's generators, by the laws of the structures
the program instantiates — monoids, comonoids, traces, contraction.
Op-level compilers rewrite *generators*; this rewrites *morphisms up
to structure*, which is why it reaches programs they provably
cannot.

The interpretation is **not unique**, and choosing it is the creative
act: the same graph reads as plain composition (nothing to do), a
shared-input comonoid (pairing), a low-rank approximation (bounded),
or a contraction problem — each exposing different laws. So
**discovery is choosing the richest interpretation**, not applying
laws inside a fixed one. Reach is the **closure** of the generators:
the law library is the generating cells, and everything they
generate is the reachable set.

**Goal.** Given a model and its weights, automatically produce a
cheaper program that computes the same function — or one provably
within an `error_budget` — and return it with a replayable
certificate. Weights are source material: reused exactly (tying,
folding, GQA replication) or derived within the bound (KV sharing,
head merging, factoring). The **function** is fixed, not the
weights.

**Mechanism.** Lift the model to the categorical structure — choose
the richest structural interpretation — search structural moves on
that small structure, and verify each candidate by certificate
before delivery. Construction from recognized structure is the
primary path; saturation is the fallback.

**Objective.** Minimize cost (memory, compute, decode latency)
subject to the certificate constraint. The constraint is a hard
filter — accept/reject — not a soft reward.

Six rules follow and are binding:

1. **Name your structure.** Every capability must name the
   categorical structure it exploits. A feature that cannot name
   one is not in the thesis and does not ship under it.
2. **The certificate is the artifact.** The deliverable is a
   cheaper program *and* its proof — exact equivalence or a
   certified bound, never silent approximation.
3. **Directed search, not learning.** You never enumerate the
   space: evaluate states (verifier + cost) and prune (cost bound,
   decomposition). That handles exponential joint move spaces — the
   Catalan bracketing search already does. RL is a non-goal:
   structure is what makes search beat it.  *(Amended by ADR 0003:
   this holds for **reach**; a learned **policy** selecting among the
   structure's rewrites is now permitted.)*
4. **Construction primary, saturation fallback.** Where the
   structure names the target, build the member directly
   (`lift_scan_to_applyd`); saturation is the completeness
   fallback, and its truncation (budgets) can miss, never
   mis-prove.
5. **Exact and bounded are one object.** Exact rewrites are the
   zero-error elements of a category enriched over (cost, error);
   bounded rewrites are lax morphisms carrying a 2-cell. The two
   modes are not separate features.
6. **Scope claims to structure.** Parity where Inductor fuses
   well; wins where work is reduced or Inductor cannot codegen.
   Honest declines, never a wrong answer.


## Why

The project's capabilities grew independently — carrier monoids, non-local lifts, the morphism engine, bounded-error rewrites, exact weight sharing — each justified on its own and each documented in a different place (README, the since-retired report and writeup, plans 0009–0013). Two things now force a single anchor:

## Not doing

- **Reachability only.** Keep the writeup's framing (structure
- **Weight-space compression** (the original bet). Falsified by
- **Higher category as the public headline.** Rejected: it must do
- **Three independent generality mechanisms** (0009/0011/0013).

## Consequences

- README and plans 0009–0013 cite this ADR as the anchor (the retired
- 0009 → 0011 → 0013 is one arc (term laws → block laws → block
- Bounded mode (0012) is restated as the enrichment's lax case,
- New features must name their structure; structure-less features
- Plan 0014 stages the conversion path this thesis implies.

## Enforcement

- Code and tests are named in the design doc.

## Design

The full record (context, tables, phases, trade-offs): [`design/0002-categorical-re-expression-thesis.md`](../design/0002-categorical-re-expression-thesis.md).
