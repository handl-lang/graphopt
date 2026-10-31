+++
id = "ADR-0009"
title = "ADR-0009 — The machinery is domain-general; tensor programs are the first domain"
status = "open"
priority = "medium"
+++

---
title: The machinery is domain-general; tensor programs are the first domain
type: adr
date: 2026-10-19
status: proposed
status_note: proposed (claim articulated in deck; porting work untested)
---

# ADR-0009 — The machinery is domain-general; tensor programs are the first domain

## Context

The deck now claims the project is a general machine — rules as data on a
categorical substrate, certification as the referee, cost as the score,
discovery closed into a well-defined game with learned players. Nothing in
that loop is tensor-specific. This ADR records what transfers to other
domains and what each new domain must supply.

## Decision / thesis

The machinery is domain-general. Domain-specificity lives in *data*, not
mechanism:

- the substrate is a general arrow calculus (`kernel`/`host`)
- the discovery loop (enumerate → measure → truth → novelty → pay →
  closure → cert) operates on terms and laws, not on tensor ops
- the game layer's boards are injected per-domain (`DomainSpec` = feature
  schema + featurizer parts + legal-move enumerator — a domain is data,
  not a subclass)

A new domain supplies exactly:

1. its operations as host functions (the leaf vocabulary),
2. its laws expressible in the calculus (the base alphabet),
3. a cost signal (the score).

The hard-earned pieces — certifying machine-authored rules, playing the
search as a game — transfer unchanged.

## Candidate domains (named, not yet built)

- **Compiler IRs beyond tensors** — instruction selection, algebraic
  simplification; any IR with equational laws.
- **Query optimization** — relational algebra is an equational theory;
  joins/filters are morphisms; cost models are mature.
- **Signal processing** — filter composition, FFT-algebra — linear
  algebra of another flavor.
- **Protocols / crypto** — equivalence-checking domains where the
  correctness gate is the load-bearing piece.

## Consequences / open work

- The claim is currently *design-level*: no second domain has been
  ported. The honest next test is one small non-tensor domain end-to-end
  (host ops + base laws + cost), measuring how much machinery actually
  reuses vs how much silently assumed tensors.
- `host-tensor`/`lib-tensor` are the tensor domain's data; an analogous
  `host-<domain>`/`lib-<domain>` pair is the porting unit.
- Open question carried in the deck: whether the learned players
  transfer across domains (policy trained on `meta`/`search` boards
  playing a new domain's board) is unmeasured.
