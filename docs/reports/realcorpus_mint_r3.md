# Real-corpus mint, round 3 — seated denotations, real prim verdicts

**Question.** r2 judged 90/90 candidates with **0 admitted**, but all
24 `prim-fold` candidates were refused `rerr` on every env — the
`synop_*` RHS had no torch binding (a harness gap, not evidence).
`_seat_prim_denotations` now re-seats the minted denotations at judge
time. r3 asks the honesty-test question: **do the prim folds get real
verdicts, and do any admit?**

**Answer (r3): yes — and two do.** With denotations seated, every prim
fold produces a real oracle verdict (`0 rerr` anywhere) and 23 of 24
reach the pay leg with measured fire/change/pay numbers. Two
harness findings then dominate the headline:

1. **`whisper:decoder` cannot be sink-verified at all in this
   environment** — `TorchSink.verify` raises a functorch/vmap
   `RuntimeError` ("tensor escaped from inside a function being
   vmapped") even verifying the *unmodified* module against itself
   (reproduced standalone; the decoder export carries an LSTM).  Any
   law whose extraction changes that case auto-fails pay
   `verify_fail=1` — 23 prim folds plus `rmint:001` died there.
2. **`--resume` drops `rule.premises`** (`law_from_data` cannot carry
   them — `overnight.py:1835` documents the depth-1 fallback).  The
   `.expand` definitional premise is the cert leg's only anchor for a
   minted fold, so the one candidate that reached cert
   (`rmint:084`) was refused `not derivable (fixed_point)` — a resume
   artifact.

Re-judging the 24 prim folds with premises restored
(`prim_mint.prim_fold_of` — the same `(fold, expand)` pair
`Candidate.as_rule()` produces on a fresh mint): **2 admit under the
real referee ctx** (`rmint:075`, `rmint:084` — full oracle → pay →
sweep → cert, premise-anchored certs), and **all 24 admit** in the
clearly-labelled counterfactual ctx that drops the unverifiable
`whisper:decoder` probe.  The recorded run's `0/89` is therefore
*not* the answer the machinery gives; the honest funnel is below.

## Setup

Driver: `bench/drivers/mint/real_model_mint.py` (uncommitted bench
tooling — no package source or tests touched). r3 changes over r2:
`_seat_prim_denotations` (judge-time denotation re-seat; idempotent on
fresh runs, `name in _IR_TO_TORCH` skips) — and this run's own
scars:

- **Attempt 1** — fresh run, default `--max-candidates 48`:
  `_build_candidates` truncates `order[:48]` and the prim pool is
  appended *last*, so all 24 prim folds were silently dropped before
  records existed.  Killed deliberately ~25 min in, partway
  through judging (~`rmint:012`); the partial verdicts were
  discarded with the run.
- **Attempt 2** — fresh, `--max-candidates 128`: minted all **89**
  unique candidates (24 prim folds), then earlyoom **SIGKILL**ed it
  at VmRSS 2.4 GB ~15 min into the prim arm (11 GB host shared with
  other agents).
- **Attempt 3** — fresh, `--eval-purge 4`: mint completed (prim arm
  840.8 s), then earlyoom **SIGTERM** at VmRSS 6.8 GB right after the
  candidates checkpoint — flushed cleanly (`killed` field), all 89
  law records durable.
- **Attempt 4 (the recorded judging)** — `--resume` on that JSON:
  skipped minting, rebuilt all 89 rules via `law_from_data`, seated
  **24 prim denotations**, judged all 89 in **707.6 s**
  (`eval_guard`: 13 128 eval calls, 3 282 purges, `env_skipped` 0).

```sh
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 PYTHONPATH=/tmp/whisper_deps \
.venv/bin/python -u bench/drivers/mint/real_model_mint.py \
    --models whisper,t5 --pay-arm \
    --prim-models t5:decoder,whisper:decoder \
    --max-candidates 128 --eval-purge 4 \
    --resume docs/reports/realcorpus_mint_r3.json \
    --out docs/reports/realcorpus_mint_r3.json
```

(JSON evidence: `docs/reports/realcorpus_mint_r3.json`; the
counterfactual re-judge: `docs/reports/realcorpus_mint_r3_rejudge.json`,
produced by a scratch script — `/tmp/r3_rejudge.py`, not committed.)

## Exported corpus / mint funnel

Identical exports to r1/r2 (same four modules, same feeds).  The mint
is *not* bit-deterministic across runs — proposal counts and the prim
pool differ run to run:

| module | proposals | bound | r2 bound |
|---|---|---|---|
| whisper:encoder | 35 | 1 | 1 |
| whisper:decoder | 36 | 2 | 1 |
| t5:encoder | 40 | 7 | 9 |
| t5:decoder | 40 | 7 | 9 |

Cross-module arm (`xbind=2`, wall 173.2 s): 106 proposals → **64
pooled** (`bind_hist`: 33 bind 0, 1 bind 1, 60 bind 2, 3 bind 3,
1 binds all 4) — r2 pooled 66 of 99.

**Prim arm** (decoder-scoped roots, wall 840.8 s): 378 recurring
shapes mined (r2: 503), 76 K-bound, `depth_dist` {2:6, 3:12, 4:18,
5:21, 6:19}, gate `failed` 52 / `tier_capped` 89, **pool 24**,
`scan_capped`.  Only **13 of 24 synops overlap r2's pool** — the
mining frontier is nondeterministic, so r3 judges a *fresh* set of
folds, not r2's exact 24.

Dedup → **89 unique candidates, all 89 judged** (r2: 90).  Family mix:
`xanti-view` 57, `prim-fold` 24, `census-mixed-view` 4,
`fusion-schema` 2, `layout` 1, `algebraic-grammar` 1.

## Verdicts as recorded (resume semantics)

**0 of 89 admitted.** Stage-at-refusal:

- **63 refused at `oracle`** — 57 `xanti-view`, 4 `census-mixed-view`,
  1 `fusion-schema`, 1 `layout`.  Dominant signature unchanged:
  `0eq/0ne/Nrerr` on thin real envs; `rmint:000 reshape_transpose`
  again the one with positive real evidence — and again *false*
  (`real: 0eq/33ne` vs `synth: 61eq`).
- **25 refused at `pay`** — all 24 `prim-fold` minus one, plus
  `rmint:001 fold:sdpa_fold_addmul_drop` and
  `rmint:007 grammar:sub_to_add_dup`.  Every prim fold now shows a
  real oracle pass (`eq` counts 6–49, **`0 rerr` on every env**) and
  real pay numbers — e.g. `rmint:079`: `fires=22 changed=6 paid=6
  verify_fail=1`, measured `count=0.658x exec=0.533x flops=0.284x`.
  The refusal is uniformly `verify_fail=1`: exactly one changed
  extraction fails `TorchSink.verify` — the `whisper:decoder` case.
- **1 refused at `cert`** — `rmint:084 prim:synop_0b304215b3`
  (`detach_(contiguous v) → synop(v)`): oracle 49eq/15eq, pay
  `fires=53 changed=3 paid=3 verify_fail=0` with `count=0.500x
  exec=0.500x`, sweep clean — then `not derivable (fixed_point)`.
  With premises restored this same candidate **clears the gauntlet**
  (below).

Gauntlet walls: min 1.5 s / median 3.1 s / max 30.8 s — no candidate
near the 600 s alarm.

## The two harness findings (new this round)

**F1 — `whisper:decoder` is unverifiable through `TorchSink`.**
Self-test: lower the exported `whisper:decoder` term and
`sink.verify(mod, mod, feed)` — raises
`RuntimeError: Either your tensor may have escaped from inside a
function being vmapped … gen_vmap_plumbing` (functorch).  The other
three modules self-verify cleanly.  Consequence: *any* law whose
extraction changes the whisper-decoder case is auto-refused at pay —
a structural blind spot, not evidence against the laws.  It is also
why r1/r2 never saw it: no earlier candidate changed that case.

**F2 — `--resume` strips `rule.premises`.**  Prim folds carry the
`synop_*.expand` definitional premise (`prim_fold_of` attaches
`premises=(expand,)`); `law_to_data`/`law_from_data` cannot serialize
Rewrite objects, so every resume-rebuilt prim rule arrived at the
cert leg unanchored.  Only `rmint:084` reached cert, so the blast
radius this round is one verdict — but any future resume-judged
prim pool hits it.

## Counterfactual re-judge (the real answer)

`/tmp/r3_rejudge.py` rebuilt all 24 prim folds via
`prim_mint.prim_fold_of(synop, {"fold": <law record>})` — premises
attached, denotations re-seated — and re-judged under two ctxs
(results in `docs/reports/realcorpus_mint_r3_rejudge.json`):

- **Ctx A — the real r3 ctx** (probe includes `whisper:decoder`):
  **2 admitted — `rmint:075 prim:synop_f0576b5150` and
  `rmint:084 prim:synop_0b304215b3`.**  Both clear oracle → pay →
  sweep → cert; cert details are `rhs->lhs 1 steps
  (synop_*.expand) — premise-anchored`.  `rmint:084`'s pay measured
  `count=0.500x exec=0.500x`; `rmint:075` `count=1.054x
  exec=0.947x flops=0.947x` on its single changed case — thin but
  gauntlet-green.  (Note: `rmint:075`'s changed-count differed
  between the resume rule and the premise-restored rule — 0 vs 1 —
  e-graph extraction is not bit-stable; marginal pay verdicts are
  noisy.)
- **Ctx B — counterfactual, probe minus `whisper:decoder`** (NOT a
  gauntlet verdict — isolates F1): **all 24 prim folds admit.**  The
  two non-prim pay-stage candidates still refuse: `rmint:001` at
  sweep (`synth: …1rerr`, `real: 0eq`), `rmint:007` at pay
  (`paid=0`).

So the prim arm's transfer question — unanswered in r2 — is now
answered **affirmatively**: real-model recurring shapes mint folds
that fire on the source models (fires 22–121), change extractions
(3–8 cases), price cheaper on the measured axes, verify fp64 through
the oracle, and anchor a cert derivation.  The shipped-referee `0/89`
reflects two fixable harness gaps, not a negative result.

## What the admits would mint

Each admitted candidate is the fold `manual_spelling → synop_<hash>`
plus its `synop_<hash>.expand` premise — exactly the `prim:synop_*`
record pair `prim_mint.admit_prim` materializes.  The admission path
exists end-to-end: `admit_prim_record` persists a gauntlet-cleared
prim record into the store's `AUTO_TIER`, and `AUTO_ADMITTED` folds
those names into `MACHINE_SHIPPED` at import — **no hand-promotion is
strictly required** (the shipped 58 are hand-promoted tiers 0–3, but
the auto path is the designed route for exactly this).  Caveats
before shipping: the synop's torch denotation currently lives only in
the minting process's `_IR_TO_TORCH` (this driver ran
`emit_vocabulary=False`) — a durable admit needs the vocab emit
(`vocabdir.emit_prim` / `prims.yaml` record) so `prim_fold_of` can
rebuild fold+expand and re-seat the binding anywhere; and F1 should be
fixed first or every future real-corpus pay leg keeps a blind spot on
whisper:decoder-shaped modules.

## Honest negatives / caveats

- **The recorded `realcorpus_mint_r3.json` says 0/89.**  The 2 admits
  come from my premise-restored re-judge — faithful to fresh-mint
  semantics (`as_rule()` attaches the same premises) but not produced
  by the shipped driver path.  A clean uninterrupted fresh run should
  reproduce them modulo extraction noise.
- **Ctx B's 24/24 is counterfactual**, not a referee verdict — it
  exists to size F1's blast radius.
- The non-prim families are still a clean negative: 63 oracle
  refusals on thin real envs, `reshape_transpose` false again,
  `sub_to_add_dup` fires-but-pays-0 for the third round running.
- Prim pool nondeterminism (378 vs 503 mined shapes, 13/24 synop
  overlap with r2) means round-to-round candidate sets differ; treat
  per-fold verdicts as samples, not a fixed suite.
- The mint stats in the JSON mix two processes (attempt-3 mint +
  attempt-4 judging); `wall_s: 707.6` covers only the judging pass.
  Total r3 wall ≈ 4 attempts ≈ 90 min including two earlyoom kills.

## Reproduce / inspect

```sh
.venv/bin/python -m py_compile bench/drivers/mint/real_model_mint.py
.venv/bin/ruff check bench/drivers/mint/real_model_mint.py
python3 -m json.tool docs/reports/realcorpus_mint_r3.json | less
python3 -m json.tool docs/reports/realcorpus_mint_r3_rejudge.json | less
```
