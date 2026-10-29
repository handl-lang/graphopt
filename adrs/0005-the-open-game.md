---
title: 'The open game: human-free discovery inside anchored anchors'
type: adr
date: 2026-10
status: accepted
status_note: |-
  **Accepted (2026-10-15)** — the phase-1 measurement (carrier vs
  fused on a real export at large T) passed: 1.82–2.33× at T=512,
  with extraction preferring the derived carrier.  The research bet
  (proposer + constraint-synthesized guards) has its first
  witnesses: the game minted a conditional law and the synthesizer
  names its own gaps.  The open question has moved up a level —
  **the form list and the mint target**, planned as 0024.

  The architecture's fixed pieces are settled; what remains is the
  admission rate of the synthesized surface at the form level,
  which only evidence settles.
---

# ADR 0005 — The open game: human-free discovery inside anchored anchors

planned (plan 0024).
**Date**: 2026-10 (updated 2026-10-15).
**Depends on**: ADR 0002 (the thesis), ADR 0003 (the four
dimensions), ADR 0004 (abstraction as move).
**Related**: `catopt_discovery.cellstore` (the store everything
is), `catopt_discovery.genkernel` (the runtime minter),
plan 0017 (serialization), the tier registry.

## Context

ADR 0004 opened the move set: constructed objects — folds, bridges,
carriers — are admissible *data*, not pre-approved Python.  Since
then the machinery closed a demonstrable loop: on
`torch.export(ChunkedRetention)`, equality saturation under a
guarded law set produces `decay`, `decay_vec`, and folded
`applyd`/`aff_diag` carriers unaided; the scan tier lowers those
carriers into real `triton.jit` kernels verified to ~1e-16 on
CUDA; minted composites land in the cell store as `op` cells the
registry walks uniformly.

The pressure this ADR answers: **who authors the rules of play?**
Today the honest surface is threefold — the axiom the alphabet
lacks (each missing structure was one guarded `R(...)`), the
guard/derive predicates inside it, and the candidate
enumerator.  If those stay human, the system is an optimizer with
a very good closure engine; it is not the optimizer that
*discovers*.  The question is which pieces of the game are
anchors — things that cannot be self-minted without voiding the
game — and which are merely *current capabilities*.

The measured boundary so far:

1. **Closure is machine-owned.**  Given the vocabulary, the
   e-graph composes, folds, verifies, extracts, and lowers
   without naming anything — the RetNet recurrence is the
   existence proof.
2. **Invention is partially machine-owned.**  Composite ops mint
   today (`declare` emits `op`+`law`+`handle` cells); handlers
   mint (generated Triton runtimes); verified candidate
   equivalences mint (the compose sweep admits on a small
   space).
3. **Guards, enumeration, and derives are now partially
   machine-owned** (updated 2026-10-15):
   - guards synthesize — `auto_cond_object` mints the smallest
     declarative `cond` covering the measured-equal sites; the
     game's first *conditional* law (`xy + xz = x(y+z)` iff
     `x ≡ y`) came from it, not from a human;
   - candidates enumerate — `catopt_discovery.enumerate`
     instantiates 27 rewriting forms over the op grammar (1,160
     candidates at depth ≤2, five paying laws auto-found), and
     `open_game` plays the stream as a board whose learned prior
     beats random on held-out pages;
   - derives synthesize — `dspec_synth` mints a declarative
     `dspec` from a procedural hook's measured outputs, refusing
     (with the missing bank op named) when the grammar cannot
     express it.

   What remains human is the **form list** (the 27 skeletons)
   and the **mint target** (which atom to propose) — one level
   up, planned as plan 0024.  ADR 0004's standing lesson —
   direct law invention measured ~10× worse than structured
   construction — applies unchanged.
4. **The alphabet is open, not closed.**  A minted composite is
   new vocabulary; a declared op + generated kernel is an
   *invented atom* whose semantics is its code, verified
   observationally.  The floor is substrate-expressibility, not
   the shipping op list.

## Decision

**The optimizer is an open game: every rule, object, vocabulary,
runtime, and pricing inside it is machine-made data admitted by
anchored machinery — and the only fixed pieces are the substrate,
the correctness oracle, and the clock.**

Concretely:

1. **Three anchors, nothing else.**  (a) *The substrate* — torch,
   GPU, the Triton compiler; the machine cannot mint the thing it
   runs on.  (b) *The correctness oracle* — observational
   equivalence of outputs on real tensors plus certificate
   replay; a machine that could redefine equivalence voids every
   certificate, so the checker is machinery, not mintable data.
   (c) *The measured objective* — wall-clock time and memory on
   the substrate; a self-minted objective is a rigged race.
   Everything else — laws, guards, ops, carriers, runtimes, cost
   models, the proposal grammar — is inside the game.
2. **The VM picture is already the architecture.**  Ops are
   machine *descriptions* (`op` cells); handlers are machine
   *instantiations* (`handle` cells); the codegen tiers emit the
   runtimes (generated kernels are small minted VMs); carriers
   are state machines (`applyd` = `h ← a⊙h+x`).  A cost model is
   a handler into flop/byte space — therefore mintable; only the
   *measured* referee stays anchored.
3. **The alphabet stays open by construction.**  New vocabulary
   enters two ways: composites (closed-form — `decay_vec` =
   `pow(γ, arange)`, retractible as a *declared* atom) and
   declared-op/generated-kernel atoms (open-form — semantics is
   the emitted code, verified observationally).  The residual
   floor is whatever the substrate cannot express — universal
   computation is in reach; silicon is not.
4. **The learned prior guides a verified search — RL does not
   replace it.**  The verifier is perfect (exact + observational,
   no fuzzy reward), the cell store gives curriculum, and
   imitation on claiming traces already warm-starts the
   policy.  What RL does not fix: candidate density (~10⁹ law
   shapes, most invalid), cold start, and credit assignment
   across pattern + guard + derive.  Guard synthesis is
   constraint-solving — conds are already data — not a policy
   problem.  Quality-diversity beats raw RL for frontier
   coverage.
5. **The four dimensions hold at the meta level too.**  A
   synthesized law/guard/runtime is a *search* artifact that must
   pass the same semantics gate a shipped `R(...)` passes —
   numeric oracle → typed-pay gate → certificate replay →
   adversarial sweep of the guard's false region.  Admission is
   never a vote the machine casts for itself.

## The categorical spine — why the game is well-posed

The ontology is the theory of *why* this architecture works; the
machinery computes equational closure regardless.  Its content:

1. **Legality is derivable, not chosen.**  A move must be a
   structure-preserving map — folds, bridges, carriers are
   *homomorphisms*, the unique maps out of free structure.  The
   proposer's space is defined by the algebra's shape, not a
   policy's imagination.
2. **Correctness is a formal object.**  Equivalence = same
   morphism in the semantic category; the e-graph is congruence
   closure — the free category modulo equations.  A certificate
   is an algebraic proof the checker replays, not a heuristic.
3. **Engine-genericity is interpretation-as-functor.**  Evaluate,
   cost, verify, lower are *the same fold to different target
   algebras* — one traversal, four purposes.  The
   four-dimension split (ADR 0003) is the consequence, not the
   choice.
4. **Chunking is a theorem, not a trick.**  `fold(concat x y) =
   fold x ◇ fold y` — the scan monoid homomorphism — is *why*
   the derived RetNet recurrence is exact.  The machine produced
   the fold; the theorem guarantees legality.  No tensor-level
   optimizer can state this claim.
5. **Knowledge is structural, so it transfers.**  Laws are
   equations on morphisms; carriers are state-machine morphisms;
   costs are denotations into ℝ.  A minted cell carries
   structure, not tensors — which is why admitted knowledge
   applies to the next model unchanged.

The honest caveat: this is CT's shallow end — Lawvere theories,
equational closure, monoidal structure.  No 2-cells, no
string-diagram machinery, and the code never invokes it.  Its
role: the language the theorems are written in — what the game
may mint, what admission certifies, and the formal spine any
publishable claim stands on.

## Consequences

- **The repository shrinks to machinery + the declaration
  kernel.**  Per-architecture knowledge — axioms, carriers,
  bespoke handlers — migrates into generated cells; dead
  knowledge dies in the evidence store without deprecation.
  The codebase gets *fewer abstractions*, not none: one generic
  game replaces N hand-authored optimizations.
- **The measured win is the phase-1 gate.**  Nothing is
  defensible until a minted carrier beats the incumbent on a real
  model.  **Passed (2026-10-15):** on the spelled
  `ChunkedRetention` export at T=512, extraction prefers the
  derived chunked carrier (29.4M vs 39.1M FLOPs) and the
  end-to-end probe measures **1.82–2.33×** against the fused
  baseline (866–1113 µs vs 2018–2028 µs), fp64 max-abs ≈1.3e-8.
  Honest scope: the paying kernels are the decayed fused +
  elementwise ones; the scan kernels are correctly declined as
  unloweable.  The K-way construct (`om_kway_laws`) then cut the
  extracted cost a further 4.4× (6.70M, fold depth 1→7), with
  the O(T·C) floor (~2.59M) approached but not reached —
  extraction still prefers the spelled cross form.
- **Guard synthesis is the research bet.**  If proposer +
  constraint-synthesized guards produce non-trivial admissions
  at useful rates, the claim is real — *the rule set as a
  synthesized, measured, admitted object* is the novel
  contribution.  If not, the system remains: humans write axioms,
  machines compute closure — a solid optimizer, not the vision.
- **Honest scars carried forward** (updated 2026-10-15; closed
  items struck through).  ~~Post-saturation site binding~~
  (closed — `leaf_env` travels harvest → handler → `profitable`,
  and a raising kernel now declines instead of aborting the
  sweep); ~~`derive=`-as-data~~ (closed where the grammar
  reaches — `dspec_synth` mints specs from measured bindings,
  refusals naming the missing bank op); ~~candidate enumeration
  and the learned prior~~ (built — `enumerate` + `open_game`;
  the prior is now **trained and baked** and its ordering beats
  the cost-delta heuristic on held-out pages, AP 0.133 vs 0.045).  **Open:** extraction
  prefers the *folded* carrier (the ~2.59M floor); the **form
  list** and the **mint target** are still human (plan 0024);
  the falsified naive-RL path — enumerated, not reopened.
- **What this is not.**  Not RL solving search; not an LLM
  writing rewrites; not a new op list.  It is a game whose fixed
  pieces are silicon, the oracle, and the clock — the claim is
  *human-free optimization discovery within a declared algebra*.
