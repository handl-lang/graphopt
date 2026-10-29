---
title: Plan 0015 — task-aware certificates
type: plan
date: 2026-10-14
---

# Plan 0015 — task-aware certificates

Status: draft — after 0013 groundwork lands.

## Idea

Today a certificate proves `out_opt(x) ≈ out_ref(x)` pointwise
(exact, or within a propagated bound).  But the user's real contract
is the *task*: for an LLM the logits' ranking matters, not the raw
activations; for a classifier the argmax; for retrieval the
cosine-ranking of embeddings.  A bounded rewrite that preserves the
task metric but violates a pointwise bound is wrongly declined today
(we measured exactly this in bound_amplification: τ=1e-2 keeps
top-1=100% while the propagated bound is loose).

## Extension

- `TaskMetric` port (backend-neutral, `catopt_core.ports`): given
  (ref_out, opt_out, context) → a scalar distance + pass/fail at a
  task tolerance.  Shipped metrics: `TopKAgreement(k)`, `LogitKL`,
  `ArgmaxStability`, `CosineRanking`, `MaxRel` (the current default).
- `search(..., task=TopKAgreement(5), task_tol=...)` — opt-in;
  verify gates on the task metric instead of (in addition to) the
  pointwise bound.  Certificate records which task metric accepted,
  its tolerance, and the measured value — the provenance is still a
  replayable bound, just expressed in task units.
- The bound ledger stays honest: a task-accepted rewrite records
  `accepted_by="task"`, the pointwise bound is *also* reported —
  the two-sided honesty ("certified under task tolerance T, drift
  measured X") is the claim.

## Why it matters

The amplification study showed the pointwise gate is ~13×
conservative — real models have far more exploitable slack than a
max_rel gate admits.  Task-aware certification is the principled
bridge: exploit the slack the task can actually tolerate while
staying certified (the certificate says exactly under which task
contract the program is equivalent).  This is the honest version of
"compression knows what matters".

## Risks

- Task metrics are input-distribution dependent — the certificate
  is conditioned on a calibration set (documented, shipped in the
  manifest); distribution shift is the honest caveat.
- Never silent: `stats["verify_metric"]`, manifest records which
  task metric gated acceptance.
