---
title: Plan 0012 — Certified bounded-error rewrites (eps machinery)
type: plan
date: 2026-10-14
status_note: |-
  draft — justified by literature (xKV, CommonKV, CLA, CHAI,
  ReLU-sparsity): real models carry structure *approximately* —
  exact-factor gates decline them, approximate literature compresses
  them without certificates.  CatOpt's differentiator: rewrites that
  carry a proven error envelope.
---

# Plan 0012 — Certified bounded-error rewrites (eps machinery)

## Idea

Extend `Rewrite`/`_offer_witness` with `error_bound`: a rewrite may
apply when lhs ≈ rhs within a *certified* bound propagated to the
delivered module's verify report — not fp64-exact, but proven
`max_rel ≤ bound` against the original on real inputs, recorded in
`stats["error_bounds"]` and in the exported manifest.

## Targets (ranked by evidence)

1. **Approximate KV latent sharing** — xKV: cross-layer KV singular
   vectors align (8x compression training-free).  Our KVLatentShare
   fires only near-exactly today; bounded-error accepts the aligned
   but imperfect case with bound = reconstruction error + propagated
   attention-path sensitivity.
2. **Attention-score head sharing** — CHAI/head-clustering: heads
   with correlated QK share compute.  Exact case already covered by
   WeightTie; bounded case = near-shared weights.
3. **Low-rank weight folding** — FFN low-rank (2.6x at 32% params):
   W ≈ A·B with bound = discarded singular mass.
4. **Dead-neuron elision** — structurally-zero rows exact; nearly-
   dead rows bounded.

## Machinery (mostly exists)

- `eps.py`/`ibp.py` — interval-bound propagation already in core.
- `verify` accepts rtol — a bounded rewrite verifies with its bound
  as tolerance AND records the bound (two-sided: measured max_rel +
  certified bound ≤ user budget).
- `error_budget=` kwarg on search/optimize: default None (exact
  only — behavior unchanged); opt-in float enables bounded rewrites.

## Honest position

This is lossy-but-proven — different from compression papers
(they give perplexity deltas, we give input-level error envelopes
on the delivered function).  Never silent: stats + manifest always
report which bounded rewrites fired and their bounds.
