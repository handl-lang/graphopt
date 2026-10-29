---
title: 'Plan 0033 — the fragment tower: mlopt vocabularies as `Language` data'
type: plan
date: 2026-10-27
---

# Plan 0033 — the fragment tower: mlopt vocabularies as `Language` data

Origin: ADR-0157 (`handl-lang/host`). The discovery stack measured its
own boundary — every minted original came from vocabulary *growth*,
never composition inside it. The hand-drawn generator space is the
cap. This plan is the phased migration; each phase ships standalone
value.

## Phase 0 — canonical fragment vocabulary

One fragment directory as the single source of truth for the
discovery vocabulary. Today the same facts live as Python objects in
five places (`op_def` registries, `law_to_data` rows,
`_FUSION_SCHEMAS`, `handler_synth` family tables,
`shipped_primitives._RECORDS`).

Workstream 0a — schema:

```
vocab/
  ops.yaml          # op name, arity, attrs, denotation-ref
  laws.yaml         # law_to_data rows (patterns, conds, provenance)
  handlers.yaml     # minted handler bodies + domains
  schemas.yaml      # fold-class schemas (the _FUSION_SCHEMAS table)
  prims.yaml        # shipped_primitives _RECORDS verbatim
  families.yaml     # handler grammar family tables
```

- Emit each from its current owner (one emitter per registry —
  `scripts/emit_vocab_fragments.py`), with a `schema:` version pin.
- `lawdata`/registries read the dir at import (or are generated from
  it) — the Python objects become projections, not the source.
- Exit gate: `mlopt-handl-peer`'s `transform_ids(dir)` loads the same
  dir `pipeline.propose` consumes; a fragment added by hand or by
  machine is visible to both engines identically.

Workstream 0b — fragment emit API for the mint: `minted op` →
`ops.yaml` row; `minted handler` → `handlers.yaml` row. Today these
land in a Python-side `shipped_primitives` registry; after 0b they
land as fragments and the registry projects them. **This is the
generator-fix landing**: emitted vocab re-enters the next mint's
search space by construction.

## Phase 1 — `engine="handl"` seat

`mlopt-handl-peer` graduates from spike to engine: `search(...,
engine="handl")` routes law application/saturation through
`handl-egraph` + `RewriteEngine`; the Python `EGraph` stays the
differential oracle — every equivalence checked twice, disagreement
refuses loudly. Overnight arm: `--engine handl` records parity stats
beside the funnel.

## Phase 2 — denotations as Models

`synop_*` bodies + minted handlers as handl handler records over
`host-tensor`. Gate: fp64 max_rel ≤ the shipped verify floor on every
bound site before a Model is "handl-shipped"; `TorchSink` stays the
delivery oracle until then.

## Phase 3 — mint emits fragments (the principled fix)

- `zoo-prim` arm: mined shape → op row + denotation fragment merged
  into the ambient vocabulary → next mint sees the op as a mintable
  atom (depth-2 primitives).
- refused-spec analysis: emits handler-*family* fragments (the
  index/gather family that today is hand-authored).
- antiunify over oracle-failed mints: emits schema fragments
  (corpus-derived `fold:*`-class replacements).
- Loop-closure test: run mint → emit → re-mint; a depth-2 synop must
  appear for the bootstrap to be real.

## Phase 4 — play + saturation + typed guards

`Board` seat over fragment-application moves; `handl-egraph` becomes
the saturation engine when the perf-parity gate passes; `check`/
`derive`/`cond` hooks migrate toward typecheck-nbe predicates as the
dependent surface lands.

## Hard gates (measured, not aspirational)

- `host-tensor` fp64 parity with `TorchSink` on the prim domains.
- `handl-egraph` saturation time/memory within 2× of `EGraph` on the
  68-module corpus.
- Fragment schema version bumps ride the `uv.lock` pin loudly — a
  silent schema drift is a release blocker.

## Dependencies / adjacency

- `engine-typecheck-nbe` work (parallel session) supplies the
  guard-as-type surface Phase 4 wants.
- `mlopt-handl-peer` already proves the load-and-fire seam; Phase 0
  is the only phase with no external dependency.
- kernelopt follows identical phases on its own vocabulary (private
  fragments per ADR-0150).
