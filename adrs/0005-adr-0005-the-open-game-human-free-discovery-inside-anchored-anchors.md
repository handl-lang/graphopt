+++
id = "ADR-0005"
title = "ADR 0005 — The open game: human-free discovery inside anchored anchors"
status = "open"
priority = "medium"
+++

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


## Why

ADR 0004 opened the move set: constructed objects — folds, bridges, carriers — are admissible *data*, not pre-approved Python.  Since then the machinery closed a demonstrable loop: on `torch.export(ChunkedRetention)`, equality saturation under a guarded law set produces `decay`, `decay_vec`, and folded `applyd`/`aff_diag` carriers unaided; the scan tier lowers those carriers into real `triton.jit` kernels verified to ~1e-16 on CUDA; minted composites land in the cell store as `op` cells the registry walks uniformly.

## Not doing

- none recorded — legacy record; the design doc lists what was deferred.

## Consequences

- **The repository shrinks to machinery + the declaration
- **The measured win is the phase-1 gate.**  Nothing is
- **Guard synthesis is the research bet.**  If proposer +
- **Honest scars carried forward** (updated 2026-10-15; closed
- **What this is not.**  Not RL solving search; not an LLM

## Enforcement

- Code and tests are named in the design doc.

## Design

The full record (context, tables, phases, trade-offs): [`design/0005-the-open-game.md`](../design/0005-the-open-game.md).
