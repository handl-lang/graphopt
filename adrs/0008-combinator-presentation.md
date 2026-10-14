# ADR 0008 — Combinator presentation of the IR (monoidal + trace)

**Status**: proposed.
**Date**: 2026-10-14.
**Depends on**: ADR 0004 (abstraction as move — minted atoms become
*definable* under this presentation), ADR 0005 (the open game),
ADR 0007 (semantic-atom minting — the runtime-assembly loop runs over
combinator terms).
**Related**: `pack` (already the proto-monoidal product), the carrier
atoms (`om_decayed`, `applyd`, `decay_vec` — hand-minted; this ADR
makes them definable), the sharing-ambiguity scars
(DAG-vs-tree semantics under e-graph congruence).
Literature basis: the categorical calculus for neural architectures
(Gavranović, Lessard, McCourt, de Jong van Lier, Sanada et al.,
"Categorical Deep Learning") — architectures as morphisms of a
monoidal category, parameters via `Param`, gradients via lenses —
plus the arrow/``loop`` line (Hughes) for first-class recurrence.

## Context

The recurring user question this week — *"why is X hardcoded?"* — has
a single root: **the object language is an op-graph, so anything that
isn't an op-node needs a hand-minted atom.**  Carrier atoms
(`om_decayed`, `applyd`, `decay_vec`, `aff_diag`) are authored
primitives; multi-output needed a bespoke `pack`; sharing is implicit
in the DAG and has bitten us (the inf-norm sharing divergence);
weight-space rewrites need a new law family because parameters are
leaves, not morphisms.  ADR-0007 retires part of this by minting
atoms mechanically — but minted atoms are still *atoms*: the object
language has no combinator structure of its own.

The categorical presentation offers the minimal universal grammar:
objects are types, morphisms are terms, and a small combinator basis
— composition `>>>`, monoidal product `⊗`, structural morphisms
(`id`, swap, assoc, unitors), diagonal `Δ` for sharing — presents
every computation graph.  Two extensions cover the neural specifics:

- **`Param` morphisms** (CDL): a weight is a first-class component of
  the morphism, not an input leaf — conv+BN folding becomes an
  ordinary equation between morphism expressions, no new law family.
- **`loop`/`trace`** (Hughes): recurrence is first-class — a scan is
  `loop (arr step)`, and the carrier atoms become *spellings* of the
  trace construction rather than primitives.

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

## What this does NOT buy — stated so we don't overclaim

- **Search gets uniform, not smaller.**  A universal substrate
  enumerates an enormous space; proposer/guidance machinery remains
  the bottleneck.  The menus stop being authored *op-patterns* but
  guidance (which morphism families to enumerate, budget policy,
  measured need) is still a designed component.
- **`loop` has real typing subtleties.**  Trace requires a fixed
  state shape; dynamically-shaped recurrence is out of scope
  initially (constant shape per term is the checkable contract).
- **Verification machinery is unchanged.**  The oracle/cert/sweep
  gauntlet is representation-agnostic; the presentation changes what
  terms look like, not what "sound" means.

## Falsification path

The direction fails, measurably, if:

1. **The bridge loses structure.**  `dag → combinator → dag` does
   not round-trip on the corpus (survivors verified end-to-end),
   or the Δ-expansion blows term size past the enode budget on
   representative exports.
2. **Nothing authored becomes definable.**  If `applyd`/`om_decayed`
   cannot be respelled as trace terms within the basis (measured:
   a corpus workload saturates to the carrier *through* `loop`
   terms), the claimed payoff is empty.
3. **The combinator space is unusably large for discovery.**  If
   enumerating morphism shapes to depth ~6 yields nothing firing
   that the op-pattern enumerators already find cheaper, uniformity
   bought us elegance, not reach.
4. **Two-way drift.**  If maintaining the dual presentation forces
   semantic fixes to land twice or diverge, the parallel layer is
   worse than the problem.

## Honest scars carried forward

- **Authoring moves, not disappears.**  The combinator basis itself,
  the Δ/`loop` law sets, and the bridge are authored — same
  bootstrapping disclaimer as the law bank.  The claim is that the
  authored surface gets *smaller and more uniform*, not that it
  vanishes.
- **An authored thing that retires must retire measured.**  Carrier
  atoms remain until a corpus workload reaches the carrier via
  `loop` terms end-to-end — parity-or-better, verified.  Until then
  both spellings coexist (provenance-marked).
- **The sharing fix is the strongest immediate argument.**  If `Δ`
  only eliminates one historical bug class and nothing else follows,
  that alone may not justify the layer.
- **Representation upgrades have a graveyard.**  The prior exposure
  screens keep paying because they *measured*; this ADR's value is
  bounded by its falsification checks running, not by the elegance
  of the substrate.

## Consequences

- **The authored atom set shrinks to the combinator basis** — the
  smallest hand-written surface the system can have while remaining
  an object language at all.
- **Discovery targets morphisms, not op-tuples** — proposer
  grammars become "which combinator shapes," strictly more uniform.
- **A runtime is a diagram family.**  The ADR-0007 emergent-runtime
  loop mints combinator terms; a generated package = {basis
  extensions, laws, verified handlers} all in one representation.
- **Both surfaces coexist deliberately** — the op-graph stays for
  lowering (sinks already speak it); the combinator layer is where
  semantics, laws, and discovery live.  The bridge is the seam.
