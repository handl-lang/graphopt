# Retro — `BATCH_NORM_RULES` admission sweep: truth green, pay refuses

**Date**: 2026-10-18.  **Object**: the three eval-mode expansion laws
(`batch_norm_expand`, `batch_norm_expand_nobias`,
`batch_norm_expand_plain`, `catopt_core/laws/tensor.py`), held out of
`DEFAULT` pending the guarded sweep — the same gauntlet the promoted
laws cleared (`admission-gauntlet.md`, `mul-unsq-nat.md`).

**Verdict: NOT admitted.**  The sweep ran green through stage 4
(truth) and refused at stage 6 (typed-pay).  `DEFAULT` unchanged —
the sweep is the floor and it refused.

## How it ran

The gauntlet operates on *stored* objects, so each rule was stored
verbatim (`store_object`, `kind="law"`, `cert=None` — all three
serialize fully: `missing_hooks` empty, `serializable: True`) into a
scratch store `catopt_discovery.cache_dir()/bn_admission.db`, then
`run_gauntlet(key, corpus=default_gauntlet_corpus())`.  The real
corpus assembled in 10.1 s (364 real terms / 245 probe cases / 83
base rules); each gauntlet ran 0.6–9.9 s — the sweep is cheap on this
box, no bounding needed.

## Stage account (all three)

| stage | `batch_norm_expand` | `…_nobias` | `…_plain` |
|---|---|---|---|
| reconstruct | pass | pass | pass |
| full-data | pass | pass | pass |
| measure | matches=10 census=3 | matches=0 | matches=0 |
| truth | synth 16eq/0ne/0rerr (344 declined, 360 envs); real 10eq/0ne | synth 16eq/0ne; real — | synth 16eq/0ne; real — |
| novelty | new | new | new |
| **typed-pay** | **FAIL — fires=8 typed=8 ill=0 paid=0** | **FAIL — fires=0** | **FAIL — fires=0** |

## Why the refusal is the honest one

The truth gate is exactly what the family needed to clear: every
guard-accepted evaluable binding is `equal` (16/16 synthesized, 10/10
real for the full variant; the two short arities accept nothing real
because aten exports always carry the affine operands — their guarded
region is corpus-empty but the sweep is not vacuous: 16 equal synth
sites each).

The refusal is `typed-pay` (`evd.paid > 0` required): the expand
direction writes the fused kernel's *spelled* work into the e-graph —
a member that is never the cheapest (`batch_norm` prices 4 vs the
~7-pass spelled subgraph, `tensor.py`'s own header comment).  On the
8 corpus sites the full variant fires, extraction never selects the
member — `paid=0`.  `nobias`/`plain` match no corpus arity at all
(`fires=0`).

An expose/infrastructure law that cannot win extraction anywhere is
not a `DEFAULT` candidate under this floor — the promoted laws each
paid ≥1 site.  Its value is *enabling*: it gives downstream
weight-space laws the spelled per-channel scale/shift to see
(`conv-bn-fold-synth.md`'s anchor story).  Admission routes that
recognize enablement rather than extraction pay would be a different,
unbuilt floor — recording the gate's verdict, not working around it.

## What stays open

- `BATCH_NORM_RULES` remains opt-in (`search(..., rules=DEFAULT +
  BATCH_NORM_RULES)`); re-run the gauntlet if a corpus grows
  `batch_norm`-free export arities or an enablement-aware admission
  floor is built.
- The `batch_norm_expand` matches (10) live in the corpus — the real
  BN sites exist (`ResNetBlock` et al.), so the family's *truth* was
  measured on real instances, not only synthesized ones.
