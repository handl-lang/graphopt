---
title: Combinator presentation of the IR (monoidal + trace)
type: adr
date: 2026-10-14
status: proposed
---

# ADR 0008 — Combinator presentation of the IR (monoidal + trace)

## Decision


**Add a combinator presentation layer over the IR — parallel, not a
replacement.**  The op-graph IR stays the lowering surface (sinks,
executors, `pack`); the combinator presentation is the
*discovery and semantics* surface.

### The presentation

- Terms are morphisms of a symmetric monoidal category with Δ
  (sharing is a morphism, not a pointer convention) plus `loop`
  (typed trace — state shape fixed).
- A bridge, both directions: `dag → combinator` (fan-out becomes
  explicit `Δ`; `pack` becomes `⊗`) and `combinator → dag` (lower
  via expansion of structure morphisms).
- Laws remain equations — now between combinator expressions.
  The same admission gauntlet applies (ADR 0004/0007): oracle →
  typed gate → certificate → adversarial sweep → measured pay.

### What becomes definable (the payoff)

- **Carrier atoms as `loop` instances.**  `applyd`/`om_decayed`/
  `decay_vec` cease to be primitives — the RetNet carrier is a
  spelled trace term.  The authored-atom problem dissolves.
- **Weight-space laws as ordinary rewrites.**  Params as morphisms
  → conv+BN folding, reparameterization, linear-chain folding are
  equations in the same object language as everything else.
- **Sharing as structure.**  `Δ` makes fan-out explicit — the
  DAG-vs-tree ambiguity class of bugs is impossible by construction.
- **A uniform discovery space.**  Proposers enumerate morphism
  shapes of the free category — naturality/factorization families
  stop being separately-coded generators.


## Why

The recurring user question this week — *"why is X hardcoded?"* — has a single root: **the object language is an op-graph, so anything that isn't an op-node needs a hand-minted atom.**  Carrier atoms (`om_decayed`, `applyd`, `decay_vec`, `aff_diag`) are authored primitives; multi-output needed a bespoke `pack`; sharing is implicit in the DAG and has bitten us (the inf-norm sharing divergence); weight-space rewrites need a new law family because parameters are leaves, not morphisms.  ADR-0007 retires part of this by minting atoms mechanically — but minted atoms are still *atoms*: the object language has no combinator structure of its own.

## Not doing

- none recorded — legacy record; the design doc lists what was deferred.

## Consequences

- **The authored atom set shrinks to the combinator basis** — the
- **Discovery targets morphisms, not op-tuples** — proposer
- **A runtime is a diagram family.**  The ADR-0007 emergent-runtime
- **Both surfaces coexist deliberately** — the op-graph stays for

## Enforcement

- Code and tests are named in the design doc.

## Design

The full record (context, tables, phases, trade-offs): [`design/0008-combinator-presentation.md`](../design/0008-combinator-presentation.md).
