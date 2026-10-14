# ADR 0007 — Semantic atoms: how new *meaning* enters the object language

**Status**: proposed.
**Date**: 2026-10-14.
**Depends on**: ADR 0004 (abstraction as move — the `ConstructedObject`
record and its admission gauntlet), ADR 0005 (the open game), ADR 0003
(four independent dimensions).
**Related**: `genkernel` (minted Triton kernels — the existing
layer-2 precedent), `dspec_synth` / `auto_cond` (law synthesis from
measured bindings), `vocab` need-detection (names missing atoms from
refusals), the `error_budget` machinery (`bounded_e2e` — certified
approximate members), the carrier package (`catopt-carriers` — a
hand-built runtime: atoms + handlers + laws).

## Context

The e-graph composes and folds exactly what its vocabulary can spell.
Measured so far: the RetNet fold is a *rediscovery* — the carrier the
machine extracted is numerically identical (9.8e-9 fp64) to the
paper's published chunkwise duality (`tools/retnet_baseline_probe.py`),
and hand-tuned baselines still beat the delivered artifact.  The
discovery machinery can mint new *spellings* (law instances, guards,
derives, paying forms — `square(X)=square(neg X)` was non-derivable),
and new *operators-as-implementations* (`genkernel` binds a measured
elementwise pattern to a generated kernel).  What it cannot yet do is
mint a new **semantic atom** — an op whose *meaning* did not exist in
the library before.

That is the line between rediscovery and creation.  Everything this
ADR needs already exists in pieces; the decision is to make the
missing piece a first-class, gate-guarded mechanism rather than an
implicit capability of hand-written packages.

The two enabling observations (proposed in discussion, verified
against the code):

- **Handlers are first-class interpretation.**  An atom's semantics is
  a registered cell — `Op` carries no meaning; the sink's executor
  table does.  An atom minted with a handler *has* semantics by
  interpretation, not by expansion.
- **The kernel layer is a VM.**  `Sink.lower` produces an executor:
  op → handler dispatch over tensors + memory.  A *runtime* is a
  coherent set {atoms, handlers, laws} — `catopt-carriers` is one such
  package, built by hand.  Nothing in the architecture requires it to
  be hand-built.

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

## Falsification path

The direction fails, measurably, if any of:

1. **Compression never pays** — layer-1 atoms mint but no deeper
   extraction or lower cost follows (the claim that naming structure
   expands reach is directly checkable against the shared-state gap).
2. **No synthesized law survives the gauntlet** — `dspec_synth` over
   minted-handler bindings yields only coincidence-fits or unfaithful
   specs; the audit's unfaithful category (already measured: 4 of 73)
   would then bound the whole direction.
3. **Handler grammar too weak** — no non-elementwise handler verifies
   against its pre-mint subterm within the bounded search; then "the
   machine can't write a scan" is the honest negative.
4. **The runtime never coheres** — minted atoms acquire no consistent
   law set; induced theory collapses or every candidate is
   paying-free.

Any of these closes a layer; the ADR requires one layer to pay on a
held-out task to stay open.

## Honest scars carried forward

- **`square(X)=square(neg X)` is trivial.**  The one non-derivable
  paying form to date has no substance — minting machinery that works
  is not minting machinery that matters.  This ADR does not change
  that measurement; it changes what the machinery can reach.
- **Free-form handler search is unbounded.**  Restricting to
  structured grammars is the decision, not a limitation to note later.
- **Approximate atoms threaten the exactness brand.**  Layer 4 exists
  because `error_budget` already certifies bounded members — but it
  stays fenced: an approximate atom may never appear on an exact
  law's RHS.
- **Serialization debt.**  Minted atoms need the `ConstructedObject`
  record generalized to carry handler provenance (source grammar,
  verified subterm, certified laws) — the admission format ADR 0004
  anticipated.

## Consequences

- **The library stops being the bottleneck.**  Carrier atoms today are
  hand-authored per domain; the minted-atom pipeline makes vocabulary
  a product of measurement.
- **Discovery changes claim.**  From "rediscovered a published
  duality" to "the object language is itself discovered" — the
  strongest defensible version of the project's thesis.
- **Everything still passes the same gauntlet.**  A minted atom is a
  search artifact that must clear the semantics gate like a law —
  the four dimensions hold, including for the atoms the machine
  invents.
- **Runtimes become artifacts.**  A coherent {atoms, handlers, laws}
  set is serializable, versionable, and reviewable — a generated
  package, not a hand-grown one.
