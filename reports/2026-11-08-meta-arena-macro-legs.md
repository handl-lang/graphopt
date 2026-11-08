# T-009 — macro move-shape legs + handler-bucket fix

Date: 2026-11-08.  Follows
`2026-11-07-meta-arena-corpus-coverage.md`, whose verdict named the
residual precisely: on `dual`/`tri` boards the oracle argmax is the
spelled `declare→saturate→handle→extract` macro while the policy
plays a `claim_*_extract` compression, and the two offers differed
only in `x:macro:len`/`x:macro:score`/handler-bucket legs — plus a
recorded latent bug: `silu`/`square` collide in handler bucket 3
of `META_ARENA_HASH_BUCKETS` (width 4), so literal `claim` rows
cannot distinguish the two tags.

This round attacked both levers in `meta_player`.

## Lever A — spelled-vs-claim feature legs

`_macro_feats` now emits, per `macro` action's seq:

- `x:macro:spelled` — seq contains `declare`;
- `x:macro:saturating` — seq contains `saturate`;
- `x:macro:nclaims` — claim-leg count / 4;
- `x:macro:reach_sites` / `x:macro:reach_claimable` —
  `spelled ∧ saturating` × the board's `st:sites` / `st:claimable`
  broadcasts (the "the unbounded saturate reaches ≥2 sites"
  signal; zero on non-spelled offers so the product stays
  candidate-varying);
- `x:mc:<bucket>` / `x:mh:<bucket>` — handler identity split by
  *role*: a claim-leg bind and a handle-leg bind are different
  moves that `a:h:*` merges.

Schema: new `meta_player.META_FEATURES` =
`lawdata.META_ARENA_FEATURES` + `a:h:4..7` + the legs above;
`META_PARTS`/`META_SPEC`/`META_SPEC_V2` all build on it (V2's
`x:seqh:*` block widened to the same handler width).  Domain-side
rather than lawdata/gamekit because the legs name this featurizer's
own parts — the same seam `META_ARENA_FEATURES_V2` already used.

## Lever B — `silu`/`square` bucket collision

`_META_HANDLER_BUCKETS = 8` overrides the `"handler"` width in
`_bucket` (lawdata's table stays 4 for the other kinds).  New
mapping: `silu→7`, `square→3`, `swiglu→0` — all three shipped tags
separated, headroom for minted tags.  Because the `a:h:*` *bucket
indices* remapped, the old weight table would misprice every
claim/handle row — **a re-bake was forced by this fix regardless
of Lever A's payoff.**

## Measurement — same bake recipe, 10-kind corpus

```
.venv/bin/python -m mlopt_discovery.meta_player \
    --seed 0 --train-cases 400 --eval-cases 40 --budget 24 \
    --temperature 1.2 --epsilon 0.25 \
    --imitate-epochs 3 --epochs 5 --json run.json
```

Learned-greedy held-out (sampled in parens), seed-0 bake:

| arm | offset 7919 | offset 31337 |
|---|---:|---:|
| oracle ceiling | +0.352 | +0.349 |
| previous bake (2026-11-07) | +0.32 (+0.30) | +0.33 (+0.29) |
| **this bake — Lever A+B, seed 0** | **+0.33 (+0.30)** | **+0.33 (+0.31)** |
| replication bake, seed 23418 | +0.34 (+0.29) | +0.33 (+0.31) |

Per-offset deltas vs the previous bake: +0.01 greedy / +0.00
sampled at 7919; +0.00 greedy / +0.02 sampled at 31337 (seed 0);
the seed-23418 replication lands +0.34/+0.33 greedy — same band.
Within run noise — **Lever A measured FLAT**: the ~0.03
`dual`/`tri` residual did not close.  The learned weights do use the new legs
(`x:macro:spelled` −0.85, `x:macro:saturating` −0.85,
`reach_sites` +2.23, `reach_claimable` +2.55, `x:mc:7` +1.00,
`x:mh:7` +0.79, `a:h:7` +1.80) — the policy can now *name* the
spelled-vs-claim distinction, it just doesn't convert it to
held-out reward on this corpus.

## Verdict

- **Lever A (feature legs): honest negative on payoff, kept.**  The
  schema now expresses the distinction the oracle argmax needs;
  400 training boards at budget 24 did not find a weight vector
  that converts it.  Recorded as measured-flat, lever closed —
  further gains likely need either more multi-site training mass
  or a nonlinear policy, not more linear legs of the same shape.
- **Lever B (bucket fix): shipped as correctness.**  `silu`→7 /
  `square`→3 / `swiglu`→0 verified; literal claim rows and
  role-qualified macro legs are now tag-distinct.  Held-out
  impact unmeasurable today (square still declines everywhere —
  the collision was latent, exactly as recorded), but the next
  paying square board would have been invisible to the old table.
- `meta_prior.META_ARENA_PLAYER_WEIGHTS` re-baked from the seed-0
  run (53 nonzero rows, 4-dp convention) — required by the bucket
  remap; performance-neutral vs the superseded bake.

Artifacts: `t009_macrolegs_s0.json` / `t009_macrolegs_s23418.json`
(bake + weights + imitation record per seed; seed 0 is the shipped
bake per convention).  Code uncommitted on `main` per task
convention.
