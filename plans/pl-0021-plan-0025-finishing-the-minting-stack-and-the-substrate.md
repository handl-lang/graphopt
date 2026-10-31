+++
id = "PL-021"
title = "Plan 0025 — finishing the minting stack and the substrate"
status = "open"
priority = "medium"
+++

---
title: Plan 0025 — finishing the minting stack and the substrate
type: plan
date: 2026-10-14
---

# Plan 0025 — finishing the minting stack and the substrate

Depends on: ADR 0007 (semantic-atom minting — measured),
ADR 0008 (combinator presentation — bridge gates 1–2 PASS, gate 3
open), plan 0024 (form-level proposer / vocabulary minting),
plan 0012 (bounded-error rewrites).

## Motivation

This cycle closed the minting loop far enough to ship: `mul_unsq_nat`
is the first law minted from a *corpus op-tuple* and it is in the
default rule set with a synthesized guard that survived an adversarial
sweep.  The same machinery honestly refused its apparent siblings, its
first composed atom (pay-null), and its own compression layer
(reach-null).  Three measured blockers now stand between "mints and
ships" and "mints, ships, and *pays at scale*":

1. **Enumeration ordering.**  The paying `fold_emit` descriptor sits
   >200k deep in `propose_bodies`' round-robin; it verifies at
   6.7e-16 when reached directly.  The gap is ordering, not
   expressiveness (`feat/enum-order` in flight).
2. **Grammar dead space and gaps.**  `fold`/`fold_emit` empty-side
   const fillers always raise in `eval_body` (candidates that
   structurally cannot evaluate); reductions/softmax have zero
   candidates; the `a≡1` side is only spellable as `div(X,X)`, which
   the disjoint a/b partition can never propose; `_family_gens`
   hard-gates fold/fold_emit at ≥3 metas.
3. **Proposal space.**  Yield in the exposed corpus is ~1.5%, and the
   form list is still hand-written (plan 0024's remaining half).  The
   combinator presentation (ADR 0008) is the uniform space, but gate 3
   — whether search *reaches* `loop` forms — is unmeasured.

## What exists (don't rebuild)

- `catopt_discovery.handler_synth` — the structured handler grammar +
  `verify_body` + `mint_handlers` + record round-trip (ADR 0007 L2).
- `lab/mint/atom_law_loop.py` — per-atom law synthesis (25 certified
  rows, 4 rejected, 0 refused) (ADR 0007 L3).
- `lab/mint/atom_synth_loop.py` — the composed need → handler → laws →
  admission → pay loop, with the stage-level funnel.
- `lab/bridge/combinator.py` / `combinator_probe.py` — the bridge,
  `BASIS` as data, both gates + the corpus round-trip driver.
- `catopt_discovery.enumerate` / `forms` (plan 0024) — instance and
  skeleton enumeration; the attr-spelling fix (`feat/enum-attr-gap`,
  0→198 fires, 0 paying).
- The admission gauntlet — oracle → typed pay gate → certificate
  replay → adversarial sweep → measured pay; unchanged throughout.

## The work

### A. Make the paying descriptor reachable (in flight)

Canonical emitted-scan seed in `propose_bodies`; stop emitting
candidates whose const fillers cannot evaluate; re-measure the
`SelectiveScan` funnel.  **Success = the minted atom's law set fires
on real corpus terms** (the pay question), or a named next blocker.

### B. Grammar extension, measured

Extend the handler grammar by the *refusals*, not by taste:
reductions/softmax (0 candidates today), an `a≡1` spelling that does
not need `div(X,X)`, and an emit tree deeper than 1 non-acc op.
Each extension lands with its refusal count dropping — measured.

### C. Gate 3 — does search reach `loop` forms?

The substrate's open question.  On a corpus workload whose export
erased a recurrence (`lstm.input`, `cumsum`, `cumprod`), run
saturation over the *combinator* presentation and ask whether a
`loop`-headed member appears and is extractable.  If yes: the
substrate is a discovery surface.  If no: the fold must be proposed
(the proposer carries the burden), and ADR 0008's cost is a
presentation-only win — recorded either way.

### D. The missing law families (BiRefNet's lesson)

The measured boundary: models whose math is already fused expose
nothing to the op-algebra laws.  Three families are named and
untried — take them in cost order:

1. **Precision** — the bounded lane is *built* and was a certified
   negative on BiRefNet for a measured reason (thin dedup: the
   bounded member priced 1.32× worse than dense).  Next target: a
   model where the dedup is thick (a wide-MLP checkpoint), and
   weight-space members rather than row-gather ones.
2. **Data movement** — hoisting invariant structure (Swin masks are
   recomputed per layer, identical).  Compete with Inductor here at
   your peril: measure *before* building.
3. **Weight space** — conv+BN fusion targets exist (43
   `BatchNorm2d` modules measured in real BiRefNet).  Per the
   no-hardcoding rule, the work is **derive-or-refuse**: feed
   measured `(W,b,μ,σ,γ,β) → (W′,b′)` bindings to `dspec_synth` and
   let the machinery attempt the fold spec; whatever it cannot
   express becomes a named grammar gap (extend the grammar, never
   hand-write the law).

## Falsification

- A: if the canonical seed is reached and the laws still fire nowhere
  real, the pay-null is semantic (the corpus's usages are in the
  guarded-out region) — record it and stop chasing the atom.
- B: an extension that adds candidates without dropping a refusal
  count is churn — revert it.
- C: gate 3 negative closes the substrate's *discovery* value; the
  presentation value (uniformity, explicit Δ) must then carry the
  ADR alone.
- D: each family is killed by its own measurement (no pays → the
  family is closed with numbers, like the bounded lane's BiRefNet
  negative).

## Order

A (in flight) → B → D1 (cheapest, machinery exists) → C → D3
(derive-or-refuse) → D2 (only with a measured reason).
