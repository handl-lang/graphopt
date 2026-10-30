---
title: 'Semantic atoms: how new *meaning* enters the object language'
type: adr
date: 2026-10-14
status: proposed
---

# ADR 0007 — Semantic atoms: how new *meaning* enters the object language

## Decision


**Minting a semantic atom becomes a pipeline, not an event** — four
layers, each with its own evidence requirement, all admitted through
the ADR-0004 gauntlet (oracle → typed gate → certificate replay →
adversarial sweep → measured pay).

### Layer 1 — definitional atoms (compression; sound by construction)

Anti-unify recurring subgraphs in saturated e-graphs; mint a named
atom whose semantics IS its expansion (macro / definitional extension,
Babble-style library learning).  No new meaning is claimed — the atom
unfolds to a known term — so soundness is free.  The value is
*compression of the reachable set*: named structure makes deeper
compositions fit the iteration budget.  This is the engine ADR 0004
assumed.

### Layer 2 — handler-interpreted atoms (semantics by denotation)

An atom minted with a proposed *handler body* — code drawn from a
structured grammar of tensor primitives (loop-shaped, fold-shaped,
two-axis-contraction templates — NOT free-form Python; free-form is
where the search explodes).  `genkernel` is the elementwise special
case; this generalizes the proposal site to non-elementwise skeletons.
Evidence: the handler is verified numerically against the pre-mint
subterm it compresses, on the full measured domain — same bar as a
law.

### Layer 3 — law-theoretic atoms (equational grounding)

The algebraic level: mint an atom with **no body**, only the laws it
is hypothesized to satisfy.  This is how primitives legitimately enter
mathematics — "scan" is defined by the fold law, not by an
implementation.  Consistency is checkable: does the induced equational
theory stay coherent under saturation, and does any paying extraction
result?  The closing loop already exists in pieces:
**handler → measured bindings → `dspec_synth`/`auto_cond` synthesize
the laws the atom satisfies → `verify_law` certifies them.**  A
runtime *self-assembles* when that loop runs over a coherent atom set.

### Layer 4 — induced atoms (approximate semantics)

A handler *learned* to match observed computation — the only layer
that breaks exactness, so it is **confined to the certified-error
lane**: admitted only as a bounded member with a carried
`error_budget`, never as a semantic equivalence.  ADR 0003 is the
guardrail: a fit handler is an *execution* candidate, never a
*semantics* claim.

### The emergent-runtime thesis

When layers 2+3 run to fixpoint over a domain, the output IS a
runtime: {minted atoms, their handlers, their certified laws} — the
shape `catopt-carriers` already has, assembled by measurement instead
of by hand.  `vocab`'s refusal-driven need-detection supplies the
proposal frontier (it already knows which atoms are wanted);
`enumerate`/`forms` supply the law candidates around them.


## Why

The e-graph composes and folds exactly what its vocabulary can spell. Measured so far: the RetNet fold is a *rediscovery* — the carrier the machine extracted is numerically identical (9.8e-9 fp64) to the paper's published chunkwise duality (`lab/probes/retnet_baseline_probe.py`), and hand-tuned baselines still beat the delivered artifact.  The discovery machinery can mint new *spellings* (law instances, guards, derives, paying forms — `square(X)=square(neg X)` was non-derivable), and new *operators-as-implementations* (`genkernel` binds a measured elementwise pattern to a generated kernel).  What it cannot yet do is mint a new **semantic atom** — an op whose *meaning* did not exist in the library before.

## Not doing

- none recorded — legacy record; the design doc lists what was deferred.

## Consequences

- **The library stops being the bottleneck.**  Carrier atoms today are
- **Discovery changes claim.**  From "rediscovered a published
- **Everything still passes the same gauntlet.**  A minted atom is a
- **Runtimes become artifacts.**  A coherent {atoms, handlers, laws}

## Enforcement

- Code and tests are named in the design doc.

## Design

The full record (context, tables, phases, trade-offs): [`design/0007-semantic-atom-minting.md`](../design/0007-semantic-atom-minting.md).
