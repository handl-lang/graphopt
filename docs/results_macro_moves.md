# Macro-action minting — new *moves* the way laws are minted

**Date:** 2026-10-19 · **Scripts:** `lab/mint/macro_moves.py` (the
`"macro"` handler + offer seam), `lab/mint/macro_miner.py` (miner +
eval) · **Seed:** 0 throughout · **Runner:**

```bash
.venv/bin/python lab/mint/macro_miner.py --json /tmp/macro.json
```

The same move the discovery loop makes on laws, run on the action
space: collect `Trajectory` reports from played episodes, count
canonical action n-grams, rank by `count x payoff` (the mean
terminal-step reward of the episodes the gram rode in — "did the
subsequence precede a *paying* certified extract?"), and mint the
top grams as `Macro` templates. A minted macro is replayed as
`Action("macro", {"seq": [...]})` through the refereed `macro`
handler — each sub-action dispatches through `ACTIONS`, declares are
accepted two-phase, a declined sub-move declines the whole macro, a
terminal one propagates its certificate. Soundness is the antiunify
argument: a macro mints nothing a player could not already spell —
it compresses the line into **one** step (one `step` charge, one
budget tick).

## Mining — 206 trajectories over `gen_cases(0, 24)`, budget 24

Arms played per case: scripted, greedy, 2 seeded randoms, the frozen
shipped prior (sampled + greedy), `claim→extract` per claimable
handler tag, and one declare-line episode per mined candidate
(`declare→saturate→handle*→extract`). N-grams are counted over the
*applied* action stream, keyed by `(op, identity-params)` — declare
by its spelled spec's coverage skeleton, handle/claim by handler
tag, fire by rule — so the silu and swiglu lines mint as distinct
macros and fresh fold names merge.

| rank | seq | freq | cert% | term-rw | score |
|---|---|---|---|---|---|
| 1 | `claim(silu)→extract` | 15 | 1.00 | 0.289 | 4.333 |
| 2 | `declare(silu-spec)→saturate` | 13 | 1.00 | 0.296 | 3.850 |
| 3 | `declare(silu-spec)→saturate→handle(silu)` | 13 | 1.00 | 0.296 | 3.850 |
| 4 | **`declare(silu-spec)→saturate→handle(silu)→extract`** | 13 | 1.00 | 0.296 | 3.850 |
| 5 | `handle(silu)→extract` | 13 | 1.00 | 0.296 | 3.850 |
| 6 | `saturate→handle(silu)` | 13 | 1.00 | 0.296 | 3.850 |
| 7 | `saturate→handle(silu)→extract` | 13 | 1.00 | 0.296 | 3.850 |
| 8 | `saturate→handle(swiglu)` | 7 | 0.86 | 0.529 | 3.700 |

Just below the top-8 offer cap (score 3.70): `claim(swiglu)→extract`
(freq 6, term-rw 0.617) and the swiglu-spec `declare→saturate→
handle→extract` line. The base playbook `saturate→extract` itself
mines at freq 48 but **score 0** — its mean terminal reward is
−0.05, the correct ranking signal: the gram precedes certified
extracts that pay nothing.

**Rediscovery: YES** — the smoke test surfaces the known winning
line in *both* forms: the spelled `declare→saturate→handle→extract`
(rank 4) and its claim-fused form `claim→extract` (rank 1).

## Held-out eval — `mp.gen_cases(0 + 7919, 16)`, budget 12

Macro offers enter through the players' `legal=` seam
(`legal_with_macros`: ranked, `viable`-filtered — a `claim` needs
its tag in `state.claim_tags`, a `declare` needs its spec's skeleton
among the occurring sites — deduplicated, capped at 8). The trained
arms use the **frozen** shipped prior — weights untouched; a macro
featurizes to the generic broadcast row (no `op:macro` weight).

| arm | reward/ep | cost | steps | cert | decl |
|---|---|---|---|---|---|
| scripted | −0.100 | 2.94 | 2.00 | 16 | 0 |
| **scripted+macros** | **+0.088** | **2.50** | 1.56 | 16 | 7 |
| greedy | −0.600 | — | 12.00 | 0 | 0 |
| greedy+macros | −0.600 | — | 12.00 | 0 | 0 |
| trained-linear | −0.056 | 2.94 | 1.12 | 16 | 0 |
| trained+macros | −0.056 | 2.94 | 1.12 | 16 | 0 |

**scripted+macros vs scripted: 7 wins / 0 losses / 9 ties, all 7
cost wins.** Every case where the base playbook stalled at baseline
(a silu/swiglu site under the stripped ruleset) was won by the
minted `claim(silu)→extract` — one refereed step replacing a line
the scripted arm never plays:

| case | scripted → +macros | cost → | declared |
|---|---|---|---|
| gen1:silu_nested | −0.100 → +0.200 | 4.0 → 3.0 | `claim_silu_0` |
| gen2:swiglu | −0.100 → +0.283 | 3.0 → 2.0 | `claim_silu_0` |
| gen5:silu | −0.100 → +0.450 | 2.0 → 1.0 | `claim_silu_0` |
| gen6:silu | −0.100 → +0.450 | 2.0 → 1.0 | `claim_silu_0` |
| gen7:swiglu | −0.100 → +0.283 | 3.0 → 2.0 | `claim_silu_0` |
| gen12:silu_nested | −0.100 → +0.200 | 4.0 → 3.0 | `claim_silu_0` |
| gen14:silu | −0.100 → +0.450 | 2.0 → 1.0 | `claim_silu_0` |

The macro player is also the *first* arm in this protocol to beat
the trained prior on mean reward (+0.088 vs −0.056).

## Honest notes

* **`trained+macros` ≡ `trained-linear`.** 0 macro picks in 18
  sampled steps: the frozen policy scores a macro on the generic
  broadcast row (logit ≈ 0) while `op:extract` sits at +3.15 and
  ends episodes in ~1 step. Macro offers are invisible to a policy
  that was not trained on them — the inventory pays a player that
  already knows to look (scripted+macros), not a frozen one.
* **`greedy+macros` ≡ `greedy`.** 0/192 steps — greedy grinds the
  `-step`-priced fire list first; it never reaches the offers.
* **Rank order leaves value on the table.** On swiglu cases the
  rank-1 `claim(silu)` macro fires (the silu subterm occurs) and
  extracts at cost 2.0 where `claim(swiglu)` would reach 1.0 —
  the scripted arm plays the first *viable* macro, and viability is
  legality, not payoff. A case-aware picker is future work.
* **Off-case macros are declined, not wrong.** `viable` keeps the
  minted lines off boards they cannot bind; where it lets one
  through that mid-sequence declines (e.g. a `handle` whose object
  never got declared), the macro scores 0 — the referee's honest
  refusal, no rollback needed because a declined sub-move leaves the
  board exactly as its refereed effects dictated.

## Retraining the prior on the macro-extended board — does the learned arm pick macros?

**Date:** 2026-10-20 · **Script:** `lab/train/train_meta_macro_prior.py`
(mines off `gen_cases(0, 24)` train cases only, then retrains cold via
`train_meta_prior`'s loop against `legal_with_macros(cap=8)` — the
shipped prior's weights untouched during the run) · **Seed:** 0 ·
**Runner:**

```bash
timeout 1800 .venv/bin/python lab/train/train_meta_macro_prior.py \
    --json /tmp/meta_macro.json      # ~16 s at the stated budget
```

The follow-up the table above invites: `trained+macros` was blind
because the *frozen* prior scores every macro on the same broadcast
row.  A policy *trained* against the offers is a different question —
and the answer is **yes, the trained arm picks macros**: 507/7,330
train steps (rate 0.069), 6 picks in 6 of 16 held-out cases.  But at
the prescribed budget it does **not** earn the bake:

| arm | reward/ep | cost | steps | cert | decl | macro picks |
|---|---|---|---|---|---|---|
| random+macros | −0.532 | 3.00 | 11.12 | 4 | 1 | 1 |
| cold+macros | −0.419 | 2.43 | 10.25 | 7 | 4 | 5 |
| **v1-shipped** | **−0.056** | 2.94 | 1.12 | 16 | 0 | 0 |
| v2-base *(V2 weights, base legal)* | −0.052 | 2.75 | 2.50 | 16 | 3 | — |
| **v2+macros (sampled)** | **+0.045** | 2.56 | 2.00 | 16 | 6 | 6/6c |
| v2+macros (greedy) | −0.050 | 2.94 | 1.00 | 16 | 0 | 0 |
| scripted | −0.100 | 2.94 | 2.00 | 16 | 0 | 0 |
| **scripted+macros** | **+0.088** | 2.50 | 1.56 | 16 | 7 | 7 |

(30 epochs × 40 cases, cold start; block means `−0.432 → −0.051`.)

**Verdict: the bake is not earned — `META_ARENA_PLAYER_WEIGHTS`
stays.**  The trained arm *does* pick macros and its sampled decode
clears the shipped prior (+0.045 vs −0.056 — the frozen-prior
blindness is gone), but it does not reach scripted+macros (+0.088).
The bar was ≥ both.  Per-case: the learned policy *out-picks* the
scripted rank-1 on `gen7:swiglu` (+0.415 vs +0.283 — it plays
`claim(swiglu)` to cost 1.0 where rank-1 `claim(silu)` stalls at
2.0), but gives it back on `gen2:swiglu` (sampling missed every
offer — greedy argmax never does, see below) and on control cases,
where the sampled decode burns ~2 extra moves before extracting.

**Why the gap is structural, not budgetary.**  Every macro row is
the *same* broadcast row — no `op:macro` in the schema, no per-macro
`a:*` identity — so the softmax must split its mass **uniformly**
across however many viable offers sit in the legal set.  The policy
can learn *when* a macro pays (the `st:*`/`h:*` context does that —
it never fires one on `softsign`, where `viable` offers nothing it
can bind) but it cannot learn *which* — rank preference among the
offers is inexpressible.  Epoch sensitivity says the same thing:

| epochs | v2+macros sampled | v2+macros greedy | v2-base |
|---|---|---|---|
| 30 | +0.045 | −0.050 | −0.052 |
| 60 | +0.057 | +0.027 | −0.017 |
| 120 | −0.040 | **+0.108** | −0.073 |

At 4× the protocol budget the *greedy* arm does cross the bar
(+0.108, 7/7 macro cases, decl=16 — it wins both swiglu cases
outright at cost 1.0), while the sampled arm degrades (macro
over-picks plus control churn) and `v2-base` regresses below V1
(−0.073 vs −0.056).  So the crossing exists but does not survive a
bake: the greedy-only gain at 120 ep ships a *worse* default to
every caller that never offers macros, and the stated 30-epoch
protocol's best decode is +0.045 either way.

**What would earn it:** a macro-aware feature — `op:macro` in the
schema, or a per-offer `a:*` identity (the mined `score`/`ops`
rank as an action feature — rank-1-ness is legality-visible data)
— so the policy can price offers against each other instead of
splitting mass uniformly; then re-run this driver.  The
`weights_dict()` → `spec.to_record()` → `spec_from_record` →
`weights_dict()` round-trip was verified on the trained table, so
the bake path itself is sound when the measurement earns it.
