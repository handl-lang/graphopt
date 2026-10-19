# Macro-aware features — the bake re-run the blindness doc prescribed

**Date:** 2026-10-19 · **Scripts:** `lab/mint/macro_moves.py`
(`instantiate` now carries the miner's evidence on
`params["macro"]`), `mlopt_discovery/meta_player.py` (the new
`meta:macro` featurizer part), `lab/train/train_meta_macro_prior.py`
· **Seed:** 0 throughout · **Runner:**

```bash
timeout 1800 .venv/bin/python lab/train/train_meta_macro_prior.py \
    --epochs 30 --json /tmp/meta_macro_30.json   # ~25 s
timeout 1800 .venv/bin/python lab/train/train_meta_macro_prior.py \
    --epochs 120 --json /tmp/meta_macro_120.json # ~90 s
```

## What changed since the failed retrain

`docs/results_macro_moves.md`'s closing diagnosis: the macro
retrain could learn *when* a macro pays but never *which* — every
offer featurized to the same broadcast row because the schema had
no `op:macro` and no per-offer `a:*`/`x:*` identity.  The appendix
named the fix: "a macro-aware feature — `op:macro` in the schema,
or a per-offer `a:*` identity (the mined `score`/`ops` rank as an
action feature)".  This change implements it:

* **`instantiate` attaches the evidence.**  `Action("macro", …)`
  now carries `params["macro"] = {"name", "count", "certified",
  "mean_terminal", "score"}` — the `Macro` record's stats as plain
  scalars (the record itself is unhashable — its `seq` holds dicts —
  so plain data is what keeps `_action_key`/played-mask hashing and
  reprs deterministic).  The referee still reads only
  `params["seq"]`; replay is untouched.
* **A new `meta:macro` part** (`ActionFeats`) emits, on macro rows
  only: `x:macro:len` (`len(seq)/8`), `x:macro:certified`,
  `x:macro:score` (`score/8`), the `a:h:<bucket>` handler-identity
  channel for each claim/handle leg (the *same* buckets a spelled
  claim uses — `claim(silu)` and `claim(swiglu)` now price
  differently), and `x:claimable:claim`/`x:handleable:claim` when
  the seq contains a claim, so macro rows inherit the board-fit
  pricing.  `op:macro` comes free through `OpOneHot`; `a:macro` is
  deliberately absent (it would be the same flag twice).  The part
  is registered in `PARTS`/`META_PARTS`/`META_PART_TAGS`;
  `to_record` round-trips, and `player_arena`'s meta `requires`
  gained `("meta:state", "meta:macro")` (it reads `st:*` keys).
* **Schema:** `META_ARENA_FEATURES` grew `op:macro`,
  `x:macro:len`, `x:macro:certified`, `x:macro:score` (the
  `a:h:*`/`x:claimable:claim` columns already existed).

## Retrains — same protocol as the failed run

Mining is unchanged (8 macros off `gen_cases(0, 24)` — the same
offer set, `claim(silu)→extract` at rank 1).  Cold
`MetaLearnedPlayer`, `legal_with_macros(cap=8)`, 40 train cases,
budget 12; held-out `gen_cases(0+7919, 16)`.

### 30 epochs (the protocol budget)

Train block means `[-0.348, -0.095, -0.067, -0.026, -0.018,
-0.011]`; train macro picks **892/6,925 (rate 0.129 — nearly 2x the
0.069 of the feature-blind run)**.

| arm | reward/ep | cost | steps | cert | decl | macro picks |
|---|---|---|---|---|---|---|
| random+macros | −0.532 | 3.00 | 11.12 | 4 | 1 | 1 |
| cold+macros | −0.419 | 2.43 | 10.25 | 7 | 4 | 5 |
| **v1-shipped** | **−0.056** | 2.94 | 1.12 | 16 | 0 | 0 |
| v2-base *(V2 weights, base legal)* | **−0.348** | 2.88 | 7.69 | 8 | 2 | — |
| v2+macros (sampled) | −0.056 | 2.33 | 5.19 | 12 | 9 | 12/7c |
| **v2+macros (greedy)** | **+0.117** | 2.50 | 1.00 | 16 | 7 | 7/7c |
| scripted | −0.100 | 2.94 | 2.00 | 16 | 0 | 0 |
| **scripted+macros** | **+0.088** | 2.50 | 1.56 | 16 | 7 | 7 |

### 120 epochs (the arm that crossed +0.108 greedy before)

Train block means `[-0.134, -0.011, 0.011, 0.023, 0.03, 0.031]`;
train macro picks **3,612/21,544 (rate 0.168)**.

| arm | reward/ep | cost | steps | cert | decl | macro picks |
|---|---|---|---|---|---|---|
| v2-base | **−0.096** | 2.53 | 4.81 | 15 | 9 | — |
| v2+macros (sampled) | +0.001 | 2.50 | 4.44 | 16 | 13 | 12/7c |
| **v2+macros (greedy)** | **+0.117** | 2.50 | 1.00 | 16 | 7 | 7/7c |

### The learned weights are legible

`op:macro` +1.87, `x:macro:certified` +1.87, `x:macro:score` +1.22,
`a:h:3` +2.66 (the claim-tag bucket) at 120 ep — the policy prices
the miner's evidence directly, and the identity channel works: the
greedy arm wins *both* swiglu cases at cost 1.0 where the rank-1
`claim(silu)` line stalls at 2.0 — the exact failure mode the
mining doc's honest notes called "viability is legality, not
payoff".

## Verdict: the features fixed the blindness — but the bake is still not earned

**`META_ARENA_PLAYER_WEIGHTS` stays.**  The headline the prior doc
predicted is real: the greedy decode **clears both anchors at both
budgets** (+0.117 vs shipped −0.056 and scripted+macros +0.088,
7/7 macro cases, cert 16) — macro-aware features bought the
crossing.  Two things still block the bake:

* **The sampled decode never reaches scripted+macros.**  −0.056 at
  30 ep, +0.001 at 120 ep — the sampled arm beats V1 at the longer
  budget but stays under +0.088 at both.  It out-picks the scripted
  rank-1 only when it commits (greedy); under sampling it still
  splits mass between macro offers and the spelled moves.
* **`v2-base` regresses — shipping these weights would break every
  caller that never offers macros.**  −0.348 at 30 ep, −0.096 at
  120 ep vs V1's −0.056.  Retraining against the extended legal set
  scrambles the base-op weights (`op:extract` collapses +3.15 →
  +0.74 at 120 ep; `a:h:3` inflates to +2.66 because claim-tags now
  arrive via macros too).  `legal_with_macros` lives in `lab/` —
  no shipped caller injects offers, so the baked default *is*
  `v2-base`, and it is worse than V1 at every budget tried.

**Macro pick-rate** (the asked-for number): train 0.129 (30 ep) /
0.168 (120 ep); held-out sampled **12 picks in 7 of 16 cases**,
greedy **7 picks in 7 cases** — vs 6 picks/6 cases in the blind
run.  `scripted+macros` picks 7; random/cold pick ~1-5 by accident.
The learned arm now uses macros decisively — more often than the
blind trained arm, and *selectively* (0 picks on `softsign`
controls where `viable` offers nothing bindable).

**What would earn the bake:** the crossing needs a training-time
change, not more features — e.g. train against the *mixed* legal
set but measure the shipped decode on the *base* set inside the
loop (a `v2-base ≥ V1` gate in the driver's bake check), or
bake only if a paired `v2-base` run is non-regressing; the feature
layer this doc added is exactly what that next run needs.

## Promoted: macros are the shipped legal set

**Date:** 2026-10-19 · **Change:** the macro layer moved to prod —
`mlopt_discovery/macro_moves.py` (the `"macro"` handler, `Macro`,
`instantiate`/`viable`, `offered`, `legal_with_macros`), the minted
8-row table baked as `lawdata.MACROS`
(`packages/mlopt-discovery/src/mlopt_discovery/lawdata/macros.py`,
regenerate with `lab/mint/macro_miner.py --emit`), and
`meta_arena.legal_actions` itself offers the viable minted lines to
**every** player — scripted, random, greedy, learned.  The lab-only
`legal=` injection is gone: `v2-base`'s old failure mode ("the baked
default is what macro-free callers get") can no longer ship a
macro-blind policy by omission.  `legal_with_macros` survives as the
wrapper for lab tables that differ from the shipped one (dedup keys
on the seq, not the stat payload — a re-minted copy of a shipped
line is the same move).  `tests/test_discovery_macro_moves.py`
covers the promoted module at 100% (38 tests).

Two prod-only fixes the lab context never exercised:

* `viable`'s site set held `_canon_spec` outputs in a `set` — spec
  attr dicts are unhashable; on joint-domain boards (`legal_actions`
  is the joint domain's enumerator too) the offer crashed with
  `TypeError`.  Now an equality-compared list.
* A done board offered macros — `offered` early-returns `()`.

**Gate rows, `mp.gen_cases(0+7919, 16)`, budget 12, cold retrain
on the prod set** (`scripts/train_meta_macro_prior.py`; the mined
table re-ran clean — same 8 ops-lines, `m1` stats drifted 15→14
occurrences / score 4.333→4.133 because mining now plays the
macro-offering legal set itself; the shipped table carries the
fresh numbers):

| gate | 30 ep | 120 ep | bar | verdict |
|---|---|---|---|---|
| v2+macros greedy | **+0.117** | **+0.117** | ≥ −0.056 & ≥ +0.088 | ✓ |
| v2+macros sampled | −0.056 | +0.007 | ≥ −0.056 & ≥ +0.088 | ✗ sampled |
| v2-base (`_base_legal`) | −0.348 | −0.072 | ≥ V1 (−0.056) | ✗ |
| scripted+macros | +0.088 | +0.088 | context | — |

**Verdict: the bake is still not earned — `META_ARENA_PLAYER_WEIGHTS`
stays.**  Under the prod legal set the picture is unchanged in kind
and gentler in degree: greedy clears both anchors at both budgets
(+0.117, 7/7 macro cases, cert 16); the sampled decode never reaches
scripted+macros (−0.056 → +0.007); and `v2-base` — the macro-free
residual a suppressing caller gets — improved from −0.096 to −0.072
but still trails V1's −0.056.  The sampled-decode gap is the same
mass-splitting the blind doc diagnosed; the features told the policy
*which* offer to take, not how often to commit under sampling.

Macro pick-rate: train 0.129 (30 ep) / 0.168 (120 ep); held-out
greedy 7 picks in 7 cases at both budgets, sampled 12 picks in 7 of
16 cases — selective (0 picks where `viable` offers nothing
bindable).

**What would earn it now:** a decode-side answer to the split-mass
failure — e.g. a `greedy`-default bake gated on the sampled arm, a
higher-capacity policy class (the linear softmax splits uniformly
over same-schema rows), or a distillation/imitation stage against
the scripted+macros line.
