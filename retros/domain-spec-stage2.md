---
title: 'Retro: domains-as-data — Stage-2 self-driving, first cut'
type: retro
date: 2026-11-12
---

# Retro: domains-as-data — Stage-2 self-driving, first cut

Context: Stage 1 made the corpus and the law bank mechanical
(`corpus_cov` gap-driven case generation; `overnight --auto-admit`
+ eqsat `AUTO_TIER` auto-admission).  Stage 2's question was
whether a whole *domain* — corpus + board + player vocabulary +
incumbent — can be emitted from data.  This retro records the
first cut: yes, for the meta-arena board family, via a JSON spec
format and a `gen_cases(kinds=)` corpus seam.

## The shape of the answer

The `play` registry contract turned out smaller than feared: a
`Domain` is `cases(seed,n)`, `board_of(case)`, a player
`DomainSpec` (legal/featurizer/features), an optional `scripted`
arm and an optional `deliver` hook.  For the meta-arena family all
of those are already data-driven — `MetaArena` takes its ruleset,
handler table and supported bound per case, and `META_SPEC` is a
composable featurizer spec.  So "spec → domain" decomposes into
three mechanical steps:

1. **corpus selection** — `gen_cases(kinds=)` filters the builder
   table (base + `corpus_cov` coverage kinds share one namespace);
2. **vocab enforcement** — per-kind op/attr *footprints* (sampled)
   decide eligibility, and a per-case post-filter guarantees the
   invariant on every emitted term;
3. **board assembly** — `MetaArena(handlers=<subset>)` seats the
   handler whitelist directly.

The interesting design choice is the two-layer vocab check:
footprints are cheap and decide what may *draw*; the post-filter
is the honest guarantee.  A kind whose rare draws escape its
sampled footprint is dropped, not padded — yield suffers,
correctness doesn't.

## What the demo proved

`domains/pointwise.json` — 51 ops (elementwise + reduction heads
+ kernel spellings), 2 of 3 handler tags, `rules="default"` —
resolved to 18 eligible kinds and ran the full `play` pipeline:
train → eval table with scripted/random/learned/learned-greedy
arms.  The domain did not exist before the spec did; nothing in
the pipeline knew the name `pointwise`.

## What it honestly did not do

* Only `board="meta-arena"`.  Torch-backed domains need module
  builders a JSON file can't name — `board` is the reserved seam.
* `shapes`/`hardware` are recorded, not consumed: the board is
  symbolic, no tensor is materialized.  They exist so a
  tensor-backed board binds them later without a schema break.
* The learned arm reuses `META_SPEC` + its full-corpus baked
  weights unmodified — features that never fire are inert, but
  the prior is mismatched to a narrow vocab.  Per-spec retraining
  is future work.
* `attrs` whitelist is `(op, attr-name)` only — attr values
  unconstrained.

## Gotcha worth remembering

`python -m mlopt_discovery.play` runs `play.py` as `__main__` — a
distinct module object from `mlopt_discovery.play`.  Any
registration that mutates `play.DOMAINS` through the import seats
the domain in the *other* registry.  `register_domain(domains=)`
takes the live dict explicitly; `play.main` passes its own.  The
same hazard applies to kernelopt-style plugin registration under
`-m`.

## Files

`mlopt_discovery/domain_spec.py` (new), `domains/pointwise.json`
(new packaged spec), `meta_player.gen_cases(kinds=)`,
`play.py --domain-spec`, `tests/test_domain_spec.py` (28 tests),
report `docs/reports/2026-11-12-domain-spec-domains-as-data.md`.

Also noticed (pre-existing, not this change):
`test_discovery_corpus_cov` hardcodes 131/153 rule counts; the
eqsat bank now ships 132+22=154 — four assertions drifted on
`main`.
