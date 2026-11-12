---
title: 'Retro: the last five model verdicts — attr metavars bound, 109/109 handl-shipped'
type: retro
date: 2026-11-12
---

# Retro: the last five model verdicts — attr metavars bound, 109/109 handl-shipped

Phase 2's residual closed.  `vocab/models.yaml` went from 104
`handl-shipped` / 5 `python-shipped` to **109/109** — the five
handler-refused prims (`denotation:synop_{4c5296b3ba, 4c657cb2c0,
976f9836d7, 98a002d64b, f08a53f9a0}`) promoted through the real fp64
gate, worst `max_rel` `3.2e-13`.

Two distinct gaps had to close, and they were different kinds of
missing data:

- **Domain provenance** was *recorded elsewhere*: the v2 mint's
  refused handlers carried no `provenance.domain`, but the v4
  `--prim-vocab` rerun minted the same content-hashed shapes and
  kept the measured bound-site domain.  `model_domains.
  DOMAIN_PROVENANCE` backfills it verbatim — no re-mint needed
  because the measurement already existed in a committed report.
- **Attr-metavariable binding** was a *machinery gap*, not missing
  data: `modelproj` rendered `{"dim": "a97"}` as the string literal
  `"a97"` and had no binding channel at all.  The T-026 attr-site
  work (`spec_bind`/`freeze_spec`/`attr_mvars`) already covered the
  runtime side; the projection side now mirrors it —
  `spec_attr_mvars` collects, `spec_to_hnl` emits names,
  `hnl_program` binds `name :: <lit>` prelude lines, `verify_model`
  `spec_bind`s the torch ref and records `attr_bindings` in the
  verdict.  Unbound metavars refuse loudly.

The honest caveat that survives: attr bindings are the
representative-site freeze (`sites[0]`), the same convention shipped
minted handlers verify under — per-site `$attr:` values were never
persisted, so parity is proven at the recorded binding, not over
the attr's full range.  Persisting `$attr:` in `prim_domain` is the
documented widening path.

Full write-up: `docs/reports/fragment_tower_phase2_verdicts.md`.
host-tensor needed nothing — `tensor.select`/`tensor.linear` were
already shipped (`85c68d3`); the blocker was entirely graphopt-side.
