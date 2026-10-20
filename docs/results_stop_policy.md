# The stopping policy — can the learned arm learn *when to quit*

**Date:** 2026-10-19 · **Script:** `bench/policy/stop_policy.py` (new —
same memoized-verdict protocol as `bench/policy/richer_features.py`) ·
**Seed:** 0 · **Runner:**

```bash
timeout 1800 .venv/bin/python bench/policy/stop_policy.py \
    --json /tmp/stop.json            # ~4 min
```

## Motivation

`docs/results_richer_features.md`'s flat axis: reward sits at
~0.27 across every schema arm while AP jumped 0.133 → 0.210 — the
features sharpen *ordering* but nothing trains the `stop` timing.
The baked prior drains ~15 of 16 evals/page.  Two ways to close
that were priced: a second decision head (invasive — the
`LinearPolicy` softmax scores actions, not heads) or new *feature
columns* the existing softmax can weight.  This run does the
feature-column version.

## The columns (the `open:stop` part)

A pure `st:*` broadcast cannot teach stopping — it shifts every
logit together and cancels in the softmax (the measured dead weight
of `open:pending-top`/`open:pending-mean`).  So the signal rides the
*action rows* via a `RelFeats` part:

| feature | emitted on | what it encodes |
|---|---|---|
| `x:marginal` | `evaluate` rows | candidate's pending rank quantile × the payoff a useful find collects *this* step (`1 + efficiency/(steps+1)`) — decays as the episode ages: the "a marginal eval late is worth less" gradient |
| `x:stop:marginal` | `stop` row | same payoff scaled to the *best* still-pending score — the opportunity cost of quitting now |
| `x:stop:evals_left` | `stop` row | evals the page still affords, /pool |
| `x:stop:finds` | `stop` row | useful finds already banked, /pool |

(`st:evals_left`/`st:finds_so_far`/`a:stop` as broadcast or flag
names were the sketch's suggestion; broadcasts cancel and `a:stop`
duplicates `op:stop`, so the action-level `x:stop:*` names shipped
instead.)  The part is registered in `PARTS` +
`OPEN_EXTRA_FEATURES`/`OPEN_EXTRA_PART_TAGS` — nameable by spec
records, not in the shipped schema.

## Protocol

Identical to the richer-features run: cold retrain per arm, **5
epochs × 36 even pages** of `engine.train_policy` under memoized
per-candidate verdicts; held-out odd pages, budget 17; page-aware
AP (fresh-page `w·featurize(evaluate)` rows pooled — `rank_score`
cannot see the pool-relative columns).

## Results

| arm | reward/ep | greedy | use/eval | evals/ep | stops | AP |
|---|---|---|---|---|---|---|
| incumbent (retrained) | 0.2728 | 0.3590 | 0.0115 | 14.44 | 19 | 0.2101 |
| **inc+stop** | **0.2444** | 0.3613 | 0.0091 | 15.28 | 15 | **0.2149** |
| — anchors — |||||||
| baked-prior | 0.2758 | — | 0.0106 | 13.06 | — | 0.2101 |
| heuristic drain | 0.3595 | — | 0.0104 | 16.00 | — | 0.0450 |
| random | 0.1339 | — | 0.0096 | 8.67 | — | 0.0123 |

## Verdict: not earned — stopping stays untrained

**`OPEN_GAME_PLAYER_WEIGHTS` stays** (bake rule: reward AND AP
both ≥ incumbent — 0.273/0.210).  The stop columns moved the
wrong way on the priced axis: sampled reward *dropped* 0.2728 →
0.2444, evals/ep *rose* 14.4 → 15.3, and the arm plays `stop`
*less* (15 vs 19 picks — it drains more, not less).  AP ticked up
0.2149 vs 0.2101 (+0.005, inside the ±0.01 noise band) — the
`x:marginal` column sharpens the fresh-page ordering marginally,
but the episode behavior it was built for does not appear.

Honest read: the linear map prices `x:stop:marginal` against
`x:marginal` fine at eval time, but the REINFORCE gradient has no
reason to discover quitting — the `stop` action's return is the
episode total it *ends*, and a policy that has learned the ordering
already collects most of the page's payoff before stopping.  The
remaining ~0.03 of headroom (draining duds at −0.002 each plus the
ordering-credit tail) may simply be too small for the
episode-return signal to isolate — a per-step advantage or a real
second head is the next arm, not more columns.

---

# Round 2 — the explicit stop head (same script, extended)

**Date:** 2026-10-19 (same day, second run) · **Runner:** identical —
`timeout 1800 .venv/bin/python bench/policy/stop_policy.py --json /tmp/stop.json`
(~5 min).

## Mechanisms priced

The round-1 negative isolated two candidate mechanisms; both were
implemented and measured, plus a bakeable delivery of the head's
signal:

* **(a) features + more training** — `inc+stop-long`: the `open:stop`
  part at 20 epochs (4×), testing the undertraining hypothesis.
* **(b) explicit stop head** — `head` / `head+stop`: the ranking
  softmax keeps the `evaluate` picks (the `stop` action is removed
  from the inner player's legal set); a *second* linear scorer over
  the spec's `stop`-row features gates each step: `stop` iff
  `w·φ_stop(state) ≥ τ`.  The head is trained **supervised**: walk
  each train page in the ranker's own greedy order, and label every
  visited state with the *oracle-optimal* continuation decision —
  "does the best-prefix value of draining what remains net > 0",
  computed exactly from the memoized verdicts (step cost + verdict
  tier + the `1/steps` ordering credit at its real eval index).
  τ is then calibrated on the train pages by actually playing the
  gated arm — the threshold is chosen on the metric it must earn.
* **(b, foldable)** — `imitate-stop` / `imitate-cold`: the same
  oracle labels delivered through `engine.imitate_policy` — a
  cross-entropy step toward the expert's action each state, so the
  timing lands in the *single* weight vector (`op:stop`,
  `x:stop:*`) and could serialize as `OPEN_GAME_PLAYER_WEIGHTS`.
  `imitate-cold` is the no-RL control (does the supervision alone
  carry the episode?).

Mechanism (c) — reshaping `OPEN_GAME_REWARD` to price remaining
expected yield at `stop` — was **not** run: the diagnostics below
show the blocker is observability (no feature separates "pending
still pays" from "pending is duds"), not the reward's incentive
price.  A stop bonus would re-price the same unlearnable decision
and change the game's economics for every arm.

## Results

| arm | mechanism | reward/ep | greedy | use/eval | evals/ep | stops | AP |
|---|---|---|---|---|---|---|---|
| incumbent | rl 5ep | 0.2728 | 0.3590 | 0.0115 | 14.44 | 19 | 0.2101 |
| inc+stop | rl 5ep (r1) | 0.2444 | 0.3613 | 0.0091 | 15.28 | 15 | 0.2149 |
| **inc+stop-long** | (a) rl 20ep | 0.2355 | 0.2691 | 0.0090 | 15.50 | 14 | **0.0102** |
| **imitate-stop** | (b) rl 5ep + 6 imitate passes | 0.0142 | 0.0086 | 0.0000 | 2.19 | 34 | 0.2197 |
| **imitate-cold** | (b) imitate only | 0.0152 | −0.0013 | 0.0000 | 1.72 | 36 | 0.0096 |
| **head** | (b) incumbent schema + gate | 0.2700 | 0.3501 | 0.0109 | 15.22 | 2 | 0.2101 |
| **head+stop** | (b) `open:stop` schema + gate | **0.2838** | 0.3613 | 0.0104 | 16.00 | 0 | **0.2149** |
| — anchors — ||||||||
| oracle-stop ceiling* | best-prefix under the oracle | 0.3877 | — | 0.0444 | 3.75 | 36 | — |
| baked-prior | | 0.2758 | — | 0.0106 | 13.06 | — | 0.2101 |
| heuristic drain | | 0.3595 | — | 0.0104 | 16.00 | — | 0.0450 |
| random | | 0.1339 | — | 0.0096 | 8.67 | — | 0.0123 |

\* diagnostic: the oracle expert (the ranker's greedy order + stop
the moment the best remaining prefix is non-positive) played on the
held-out pages under the memoized verdicts — 0.3877 reward/ep at
**3.75 evals/ep** (incumbent-order variant 0.3857 at 3.61; a
heuristic ordering earns 0.3845 at 3.50 — under *optimal* stopping
the ordering quality barely matters).

## What the stop decision actually learned

* **(a) is dead, and overtraining is actively harmful.**  20 epochs
  did not slowly find the timing — it *collapsed the ranking*: AP
  0.2149 → 0.0102 (random level), reward 0.2444 → 0.2355.  The
  train curve oscillates 0.23–0.37 with no trend — past ~5 epochs
  REINFORCE just randomizes the weight vector.  The `x:stop:*`
  weights it did learn were directionally sane but useless:
  `x:stop:marginal` −0.25/−0.36 ("don't stop while the board still
  offers"), `x:stop:evals_left` −0.06 — yet the sampled arm still
  drains ~15.5 evals, because a shared-softmax stop logit competes
  against 16 evaluate logits whose `a:*` masses dwarf it.

* **(b)'s supervised head learned the real structure — and it says
  "drain".**  Logistic fit on 576 labelled states, base rate 15.5%
  continue (the oracle wants ~2.5 more evals/page on average), loss
  0.34–0.38 vs the ~0.43 constant-rate floor — *some* signal is
  extracted (`st:useful_rate`/`st:true_rate` positive-ish,
  `st:seen_frac`/`x:stop:finds` negative, `x:stop:marginal` −0.30:
  "stop once finds are banked and the board looks thin").  But the
  threshold calibration is the honest verdict: the reward-maximizing
  τ on train pages is the **never-stop** end of the grid (cal
  reward 0.36 ≈ drain; `head` stops twice in 36 pages, `head+stop`
  never).  Under an asymmetric cost table — a missed `true`/`useful`
  forfeits +0.2/+1.0, a wasted eval costs −0.002 — a weakly
  predictive head's Bayes-optimal threshold *is* extreme: keep
  spending unless nearly certain nothing pays.  The features cannot
  certify that, so the learned stop policy degenerates to the null
  rule.

* **The bakeable delivery fails hardest.**  `imitate-stop`'s folded
  weights are legible and wrong-signed relative to intent:
  `op:stop` +1.07 (positive propensity — the policy *wants* to
  stop), `x:stop:marginal` +2.88, `x:stop:evals_left` +0.82 —
  "stop *more* when the board still offers and evals remain", i.e.
  the cross-entropy pass learned the *marginal* propensity
  (~85% of states are stop-labelled → stop early everywhere), not
  the conditional timing.  Held-out: stops at 2.19 evals/ep,
  reward 0.014, finds zero — the labels' stop mass lands before the
  finds it can't see.  `imitate-cold` confirms supervision alone
  carries nothing (AP 0.0096 — the expert's eval order under cold
  weights is arbitrary, so nothing transferable is demonstrated).

* **Why the headroom stays locked.**  The oracle ceiling shows
  stopping is worth ~+0.026 reward/ep *and* ~4.3× yield-per-eval
  (0.0444 vs 0.0104 useful/eval) — but reaching it requires knowing
  *which* pending candidates still pay, and no row feature sees
  that: the heuristic score's own AP is 0.045, so
  `x:stop:marginal`'s "best pending score" is nearly uninformative
  about hidden usefulness.  Stop timing is a prediction problem the
  feature set structurally cannot answer; the reward table then
  rationally resolves it to "drain".

## Verdict (round 2): still not earned — `OPEN_GAME_PLAYER_WEIGHTS` stays

`head+stop` nominally clears both bars (0.2838 ≥ 0.273, 0.2149 ≥
0.210) — but it does so by *degenerating*: the calibrated head
never stops, which as a weights-bake is `op:stop → −∞`, i.e. the
drain confound formalized.  Its pure-weights fold (inc+stop schema
+ `op:stop` −4.0) measures **0.2730 sampled / 0.3613 greedy /
0.2149 AP** — statistically indistinguishable from the incumbent
retrain (0.2728) and *below* the current bake's own 0.2758 anchor
on the same protocol.  Shipping a schema extension (+4 features)
whose only earned behavior is "never stop" — at flat-to-worse
useful/eval (0.0104 vs 0.0115) — is not a stopping policy; it is
the confound the docs keep honest.  **No bake.**

The honest summary across both rounds: the open game's `stop` axis
has real but small headroom (+0.026 reward, ~4× yield-per-eval),
the head supervision proves the labels are computable and partly
predictable, and every learnable delivery — softmax columns,
longer RL, CE-folded expert labels, or a calibrated second head —
converges to one of two degenerate points (stop-on-average /
drain-always).  The missing piece is not a mechanism but
*evidence*: a feature that actually predicts pending-candidate
usefulness (the `a:score`/`x:stop:marginal` proxy has AP 0.045).
Until then the game's economics rationally say keep spending.

---

# Round 3 — the next-candidate-value features (the missing variable, built)

**Date:** 2026-10-19 (same day, third run) · **Runner:** identical —
`timeout 1800 .venv/bin/python bench/policy/stop_policy.py --json /tmp/stop.json`
(~3.5 min: 148 s pool memoization + 12 arms).

Round 2's closing diagnosis named the missing variable: nothing
prices the *remaining* pool — "expected next find" had no feature.
This round builds it.  The `open:next-value` part (registered,
unshipped — `RelFeats`, pool-visible) emits five columns on the
`stop` row:

| feature | what it encodes |
|---|---|
| `x:stop:best_score` | max `a:score` still pending — the heuristic's own view of what's left |
| `x:stop:topk_score` | mean of the top-3 pending scores — paying *mass*, not just the peak |
| `x:stop:rank1_pred` | softmax(`a:score`) probability on the rank-1 pending member — the policy's next-pick confidence is what is wanted, but a `RelFeats` part cannot see the weights, so this is the linear-score rank proxy |
| `x:stop:remaining_frac` | pending / pool — drain progress (identical to `x:stop:evals_left`; kept for the probe) |
| `x:stop:decay` | `best_score × remaining_frac` — combined *value left* |

Five new arms rerun the mechanisms on the enriched stop row:
`inc+nextval` / `inc+stop+nextval` (cold RL, bake recipe),
`inc+nextval-long` (the 20-epoch undertraining/collapse check), and
`head+nextval` / `head+stop+nextval` (the supervised `_StopGate`
retrained over the new columns).  Each head arm also reports a
**per-feature probe**: the column's solo AP against the held-out
oracle continue-labels, and its leave-one-out cost in a refit head.

## Results (new arms; round-1/2 rows repeat bit-identically)

| arm | mechanism | reward/ep | greedy | use/eval | evals/ep | stops | AP |
|---|---|---|---|---|---|---|---|
| inc+nextval | rl 5ep | 0.2745 | 0.2910 | 0.0090 | 15.50 | 16 | **0.0215** |
| inc+stop+nextval | rl 5ep | 0.2711 | 0.3685 | 0.0107 | 15.56 | 20 | 0.1139 |
| inc+nextval-long | rl 20ep | 0.2307 | 0.2700 | 0.0089 | 15.67 | 8 | **0.0123** |
| **head+nextval** | gate over `open:next-value` | **0.3036** | 0.2807 | 0.0107 | 15.61 | 1 | 0.0215 |
| **head+stop+nextval** | gate over both parts | **0.2858** | 0.3664 | 0.0104 | 16.00 | 0 | 0.1139 |

## The features carry real signal — the head got sharper

Held-out decision-row AP (does the head score continue-labelled
states above stop-labelled ones; base rate ≈ pos rate):

| head arm | schema | head eval AP | pos rate |
|---|---|---|---|
| head | incumbent | 0.3694 | 0.165 |
| head+stop | `open:stop` | 0.4279 | 0.174 |
| head+nextval | `open:next-value` | 0.3793 | 0.163 |
| **head+stop+nextval** | both | **0.5494** | 0.196 |

The combined schema's head discriminates at 0.55 — a real jump over
round 2's 0.43/0.37.  The probe shows where it comes from
(`head+stop+nextval` rows, base rate 0.196):

| feature | solo AP | LOO AP | LOO Δ |
|---|---|---|---|
| `x:stop:remaining_frac` | **0.3287** | 0.5494 | −0.000 |
| `x:stop:decay` | 0.2062 | 0.5464 | −0.003 |
| `x:stop:best_score` | 0.2007 | 0.5609 | +0.011 |
| `x:stop:topk_score` | 0.1936 | 0.5567 | +0.007 |
| `x:stop:rank1_pred` | 0.1492 | 0.5595 | +0.010 |

**`x:stop:remaining_frac` carries the timing** — the only column
whose solo AP clears the base rate and the only one whose LOO does
not *improve* the refit head.  The honest caveat: it is a
drain-progress counter (identical to `x:stop:evals_left`), not a
value signal.  Among the genuinely pool-priced columns only
`x:stop:decay` helps at all (solo 0.206, LOO −0.003); `best_score`/
`topk_score` sit at base rate, and `rank1_pred` is *anti*-predictive
solo (0.149 < 0.196 — softmax concentration is highest on small,
uniformly-negative pools, i.e. late states, so it reads as a stop
signal inverted).  All LOO deltas are ≈0 — the family is collinear
with the progress features; the head's 0.55 comes mostly from
`x:stop:finds`/`st:useful_rate`/`st:true_rate` (weights −151/+60/+20)
plus progress.

## Why the gate still fails — the honest negative replicates, sharper

* **Every head arm degenerates to drain.**  Even at head eval AP
  0.55 — well above round 2's — reward-max τ lands at the
  never-stop end again (`head+stop+nextval` τ=71.3, 0 stops;
  `head+nextval` τ=17.5, 1 stop in 36 pages).  Its 0.2858/0.3036
  rewards are the same confound round 2 flagged: never-stopping
  *earns*, it is not a stopping policy.  Prediction improved; the
  decision did not — the value columns are still the heuristic's
  view (candidate AP 0.045), so the head cannot certify "nothing
  pending pays", and under the asymmetric table (+1.0 missed find
  vs −0.002 wasted eval) EV stays positive until near-certainty.
  Better discrimination of the *label* does not move the
  reward-optimal *threshold*.

* **The RL arms got worse, not better.**  `inc+nextval`'s sampled
  reward (0.2745) nominally clears 0.273 but its page-aware AP
  *collapsed to 0.0215 at 5 epochs* — worse than round 1's `inc+stop`
  (0.2149).  The stop row's logits now vary state-to-state, so the
  softmax mass on `stop` shifts every step and the shared-weight
  REINFORCE term randomizes the `a:*` ranking columns (the train
  curve oscillates 0.24–0.37, no trend).  `inc+stop+nextval` lands
  mid-collapse (AP 0.1139).

* **The 20-epoch collapse reproduces.**  `inc+nextval-long`: AP
  0.0123, reward 0.2307 — same signature as `inc+stop-long`.  Longer
  RL on stop-row columns is confirmed dead, on a second feature
  family.

* **The learned `x:stop` weights are sane and inert.**  All five
  next-value columns train negative (−0.02…−0.23 — "don't stop
  while value remains"), but at magnitudes far too small to move a
  softmax whose 16 `evaluate` logits dwarf the `stop` row — the arm
  still drains ~15.5 evals/ep and plays `stop` no more than the
  incumbent.

## Verdict (round 3): not earned — `OPEN_GAME_PLAYER_WEIGHTS` stays

Gate: held-out reward ≥ 0.273 **and** AP ≥ 0.210, honestly.  No arm
clears both: the reward-clearing arms are the never-stop confound
(`head+nextval` 0.3036 / `head+stop+nextval` 0.2858, ≤1 stop) *and*
fail AP outright (0.02/0.11); the best RL arm's marginal reward
clear (0.2745) rides a collapsed ranking (AP 0.0215).  **No bake.**

The three-round summary is now closed: the missing-evidence
hypothesis was tested directly — the next-candidate-value columns
*do* sharpen the supervised head (0.37 → 0.55 held-out label AP)
and the probe isolates the carrier (`x:stop:remaining_frac` — a
progress counter; the pool-priced columns are individually weak and
collectively redundant).  But sharper stop-*label* prediction does
not change the stop *decision*: the features cannot certify an
empty-of-value pool, so both learnable deliveries still converge to
drain.  The remaining headroom (+0.026 reward, ~4× yield-per-eval)
needs evidence this feature family structurally cannot provide — a
pending-usefulness signal better than the heuristic's AP-0.045 view
(the `rank1_pred` note stands: a `RelFeats` part cannot see the
policy's own next-pick probability, so even the confidence column
is only the linear-score proxy).

---

# Round 4 — the learned value signal + a gate correction that reopens the head

**Date:** 2026-10-19 (same day, fourth run) · **Runner:** identical —
`timeout 1800 .venv/bin/python bench/policy/stop_policy.py --json /tmp/stop.json`
(~4 min: 148 s pool memoization + 3 s net + 18 arms).

Round 3 closed on the missing variable: nothing predicts *which*
pending candidates still pay — the heuristic view tops out at
candidate AP 0.045.  That prediction is `RuleValueNet`'s own job,
and it reaches AP ~0.041 on-distribution — comparable to the
heuristic.  This round trains it on-distribution and feeds its
per-candidate prediction to the stop arm — **and finds that the
stop mechanism itself was broken the whole time**.

## Correction: the `_StopGate` comparison was inverted (rounds 2–3 head verdicts were forced)

The supervised head is fit on *continue* labels — high `w·φ_stop`
means "draining what remains still pays".  The gate then read
`stop iff w·φ ≥ τ`: it quits **first exactly where continuing is
worth most**.  Under a perfect head every continue-state scores at
the top, so the grid's only non-degenerate point is the never-stop
end — the "the reward-optimal τ is never-stop" result that drove
rounds 2 and 3's verdicts was **forced by construction**, not
measured.  The asymmetric-cost-table argument was applied to a
mechanism that could never express the correct rule.  The gate is
now `stop iff w·φ ≤ τ` — quit when the head's continue-score drops
below the threshold — and every head arm in the table below is
re-measured under the corrected comparison.  (Round-2/3 head rows
in their own sections remain as recorded under the inverted gate.)

## The predictor

`train_rule_value` on `RuleSample` rows built from the 576 even-page
candidates' **memoized** verdicts (BCE `delta_cost > 0`, 300 epochs,
hidden 32 — the `bench/policy/policy_baseline.py` neural path), sigmoid
predictions injected into the new `open:next-val` part's lookup
(`og.set_value_preds` — keyed by the same structural
`(label, repr(lhs), repr(rhs))` identity the verdict memo uses, so
predictions survive re-enumeration; the part emits zeros unfed):

* the predictor's **own** power, held-out: AP vs its training label
  (`delta_cost > 0`) is **0.9320** — it learned the quantity it was
  asked for and it transfers across pages.  AP vs the `useful`
  verdict is **0.0284** — *below* the heuristic's 0.045, and below
  the 0.0412 the same model's raw logits earn: 300 epochs of BCE
  saturates the sigmoid (85% of the pool lands at exactly 0.0/1.0),
  so the [0,1] features are effectively binary.  Two honest caveats:
  (a) `delta > 0` — "firing this rule on its own lhs lowers the
  cost" — is *not* the `useful` verdict (which also requires truth
  and novelty); the net answers an easier question.  (b) train-page
  preds are in-sample (~memorized), so the τ calibration sees a
  near-oracle signal that held-out pages do not have — priced below.
* **`x:stop:next_value`** — max pred over pending (the
  `x:stop:best_score` analogue in predicted-pays units);
  **`x:stop:pool_value`** — Σ preds, the remaining expected yield;
  **`x:stop:value_frac`** — pool_value / the page's starting value.

The head arms now hold the *ranker* at the incumbent schema
(`rank_tags=()`) — round 3 showed RL over stop-varying columns can
collapse the shared ranking (AP 0.0215), a confound outside the
stop-decision mechanism under test.  `imitate-nv` is the only
bakeable delivery of a working stop rule; `head+nvh` is the
ablation — the round-3 heuristic-priced columns under the same
incumbent ranker.

## Results (head arms re-measured under the corrected gate)

| arm | mechanism | reward/ep | greedy | use/eval | evals/ep | stops | AP |
|---|---|---|---|---|---|---|---|
| incumbent | rl 5ep | 0.2728 | 0.3590 | 0.0115 | 14.44 | 19 | 0.2101 |
| head | gate, incumbent schema | **0.2878** | 0.3508 | 0.0109 | 15.31 | 26/27 | 0.2101 |
| head+stop | gate over `open:stop` | 0.2735 | 0.3619 | 0.0106 | 15.69 | 4/3 | 0.2149 |
| head+nextval | gate over `open:next-value` | 0.2641 | 0.2649 | 0.0091 | 15.25 | 27/30 | 0.0215 |
| head+stop+nextval | gate over both | **0.2876** | 0.3573 | 0.0104 | 15.97 | 31/32 | 0.1139 |
| inc+nv | rl 5ep | 0.2316 | 0.2682 | 0.0092 | 15.17 | 20 | **0.0098** |
| inc+stop+nv | rl 5ep | 0.2386 | 0.2682 | 0.0090 | 15.50 | 16 | **0.0098** |
| imitate-nv | rl + 6 oracle-imitate | 0.0307 | 0.0200 | 0.0000 | 2.28 | 35/35 | 0.0096 |
| head+nv | gate over `open:next-val` | 0.2450 | 0.3502 | **0.0129** | **10.78** | 32/31 | 0.2101 |
| head+nvh | gate over `open:next-value`, inc ranker | **0.2835** | 0.3571 | 0.0111 | 14.97 | 36/36 | 0.2101 |
| head+stop+nv | gate over both families | 0.2435 | 0.3454 | 0.0100 | 13.94 | 19/8 | 0.2101 |

## What the corrected mechanism shows

* **The gate works now — on every schema.**  All six head arms
  produce real stopping policies (4–36 stops vs ≤2 before): `head`
  on incumbent features alone earns 0.2878 (+0.015 over incumbent,
  26 stops at 15.3 evals — mostly tail-dud savings); `head+nvh`
  0.2835 at 36/36 stops; `head+stop+nextval` 0.2876.  The
  never-stop "economic verdict" was the inversion, not the cost
  table.
* **The value signal is real and used.**  `head+nv` is the first
  arm to stop early at scale — 10.78 evals/ep (vs ~15.3 everywhere
  else), the best learned yield-per-eval (0.0129), 32 sampled
  stops.  The per-column probe confirms it: `x:stop:next_value`
  solo AP **0.3217** vs the oracle continue-labels — the first
  genuinely pool-priced column matching the progress counter's
  0.3287 (`x:stop:value_frac` 0.2846 is second; `x:stop:pool_value`
  is base-rate solo but LOO −0.055, contributing in context).
  Head eval AP 0.3694 → **0.4458** over incumbent — the sharpest
  value-family discrimination yet.
* **…but it over-stops on held-out pages.**  `head+nv` earns
  0.2450 — −0.028 vs incumbent: the τ calibrated under near-oracle
  in-sample train preds (cal 0.3398) stops at 10.78 evals held-out,
  forfeiting the banked true-tier tail (+0.2 each).  The oracle
  moves *more* aggressively (3.75 evals) yet earns *more* (0.3877)
  — so the deficit is stop *placement*, not stopping per se: at
  per-candidate AP ~0.03–0.04 vs `useful`, the predictor can rank
  pages but cannot certify which pending member still pays.  The
  heuristic-column ablation `head+nvh` (0.2835 @ 14.97 evals) earns
  more precisely because its weaker signal stops less.
* **Both serializable deliveries still fail.**  `inc+nv`/`inc+stop+nv`
  collapse a third time (AP 0.0098 — the shared-softmax stop-row
  gradient noise is structural, now on three feature families);
  `imitate-nv` folds the same stop-on-average confound as round 2
  (0.0307 @ 2.28 evals, 35 stops, AP 0.0096).

## Verdict (round 4): not earned — `OPEN_GAME_PLAYER_WEIGHTS` stays

Gate: held-out reward ≥ 0.273 **and** AP ≥ 0.210, honestly.  Three
corrected head arms clear reward (`head` 0.2878, `head+nvh` 0.2835,
`head+stop+nextval` 0.2876) and two of those also clear AP — **but a
gated head is not a weight table**: it ships a second scorer + a
calibrated threshold + a restricted legal set, none of which
serializes into `OPEN_GAME_PLAYER_WEIGHTS`.  Every weights-typed
delivery (softmax columns, CE-folded expert) collapses.  **No bake.**

The blocker has now moved twice.  Rounds 1–3: no feature could price
the pending pool — tested and confirmed, then a gate inversion hid
that the head mechanism itself was never exercised.  Round 4: the
evidence exists (`x:stop:next_value`, solo 0.32, used — earliest real
stopping) and the mechanism works (corrected gate, three
reward-clearing arms).  What remains is *delivery*: a serializable
arm still cannot carry the timing — the softmax stop-row corrupts
the ranking, and the imitation fold learns marginal propensity not
conditional timing — plus *precision*: a per-candidate AP ~0.03–0.04
`useful` predictor places stops ~3× past the oracle point.  The two
open threads, in order: (a) a *verdict*-trained predictor (`useful`
labels, not `delta > 0` — the net nails its own label at 0.93 and
still misses `useful` at 0.03, so the label gap is the binding
constraint); (b) a serializable gate — e.g. the head folded as a
*pairwise* stop-vs-top-eval comparison the softmax can express.

---

# Round 5 — the `useful`-label predictor, honest calibration, and the serializable fold

**Date:** 2026-10-19 (same day, fifth run) · **Runner:** identical —
`timeout 1800 .venv/bin/python bench/policy/stop_policy.py
--json /tmp/stop5_<label>.json --value-label <label>` (~5 min:
173 s pool memoization + ~25 s nets + 18 arms with folds).  Three
runs, one per injected label.

Round 4's open thread (a) is tested directly: the identical
`RuleValueNet` retrained on the *verdict* label — **`useful`** (truth
∧ strictly-cost-lower ∧ novelty, the real verdict — 7/576 train
positives, 1.2%) with the balanced `pos_weight` (81.3) added as a
`train_rule_value(labels=…, pos_weight=…)` seam for exactly this
rare-positive regime — plus **`pays`** (useful|true — the tiers the
stop decision actually prices, 26 positives) and **`delta`** kept as
the round-4 control.  Two protocol fixes land with it: (i) train-page
preds are now **out-of-fold** (6 folds — every page's preds come from
a net that never saw it), removing the in-sample optimism round 4
priced (`head+nv` over-stopped at 10.78 evals); (ii) every head arm
ships a **fold** variant — the head's stop-only `x:stop:*` columns
plus a calibrated `op:stop` intercept as an ordinary weight table,
the only delivery of the gate's timing that
`OPEN_GAME_PLAYER_WEIGHTS` could serialize (its stop logit is
`w_rank·φ_stop − w_head[x:stop:*]·φ + c` — a *relative* threshold,
not the gate's absolute one).

## The predictor — the label was *not* the constraint

Held-out AP per trained net vs each label (identical across the
three runs — same nets):

| net | pos | pos_w | AP vs `useful` sigmoid | AP vs `useful` logit | AP vs `delta` logit | AP vs `pays` logit | AP in-sample |
|---|---|---|---|---|---|---|---|
| `delta` (round-4 control) | 271 | — | 0.0284 | 0.0432 | **0.9313** | 0.0440 | 1.0000 |
| **`useful`** | 7 | 81.3 | 0.0162 | 0.0167 | 0.4647 | **0.0959** | 1.0000 |
| `pays` | 26 | 21.1 | 0.0173 | 0.0168 | 0.4103 | 0.0643 | 1.0000 |

**The label-swap hypothesis fails.**  The `useful`-label net does
not beat the round-4 net on `useful` — 0.0167 logit AP vs 0.0432,
worse than the heuristic's 0.045 too.  Its in-sample AP is 1.0000
within ~20 epochs at every recipe tried (pos_weight 1–162, hidden
8–32, epochs 20–300, seeds 0–2): with 7 positives the MLP *memorizes*
them and the residual — `useful` additionally requires *truth* of an
unproven candidate and *novelty* against the library — is not in the
(program features ⊕ rule shape) vocabulary at all.  `delta > 0`
(pure cost-movement) is learnable at AP 0.93; the truth+novelty
conjunction is not.  The round-4 read stands, sharper: the predictor
class is fine and the label was wrong — *and the new label is not
learnable either*, so the gap is evidence, not targets.

One genuine inversion worth reporting: on the *decision rows* the
`useful` net is the best signal — `x:stop:next_value` solo AP vs the
oracle continue-labels is **0.3818** under the useful-label
injection (delta 0.3217, `pays` 0.1706 — the nominally-right label
is worst).  Pool aggregation (max/Σ over 16 pending preds) forgives
per-candidate noise: the worst ranker produces the best stop-state
feature.

## Results (`head` arms under OOF calibration; fold variants new)

| arm | mechanism | reward/ep | greedy | use/eval | evals/ep | stops | AP |
|---|---|---|---|---|---|---|---|
| incumbent | rl 5ep | 0.2728 | 0.3590 | 0.0115 | 14.44 | 19 | 0.2101 |
| `head+nv` **delta** | gate, OOF cal | **0.2866** | 0.3504 | 0.0106 | 15.67 | 18/20 | 0.2101 |
| `head+nv` **useful** | gate, OOF cal | **0.2850** | 0.3591 | 0.0104 | 16.00 | 1/— | 0.2101 |
| `head+nv` **pays** | gate, OOF cal | 0.2413 | 0.3611 | 0.0146 | 9.50 | 31/29 | 0.2101 |
| `head+stop+nv` **pays** | gate, OOF cal | **0.2735** | 0.3629 | 0.0121 | 13.72 | 12/8 | 0.2101 |
| `head+stop+nv` **delta** | gate, OOF cal | 0.2726 | 0.3465 | 0.0115 | 14.50 | 24/17 | 0.2101 |
| `head+stop+nv` **useful** | gate, OOF cal | 0.2425 | 0.3087 | 0.0151 | 9.22 | 31/— | 0.2101 |
| `head+nv-fold` **delta** | weight table | 0.2726 | 0.3601 | 0.0114 | 14.67 | 22/13 | 0.2101 |
| `head+nv-fold` **useful** | weight table | **0.2850** | 0.3590 | 0.0104 | 16.00 | 0/0 | 0.2101 |
| `head+nv-fold` **pays** | weight table | 0.2558 | 0.3607 | 0.0116 | 14.33 | 16/3 | 0.2101 |
| `head+stop+nv-fold` **delta** | weight table | 0.2640 | 0.3486 | 0.0106 | 15.50 | 4/1 | 0.2101 |
| `inc+nv` **delta** | rl 5ep | **0.3008** | 0.3810 | 0.0109 | 15.31 | 16 | **0.1042** |
| `inc+nv` **useful** | rl 5ep | 0.2594 | 0.3484 | 0.0094 | 14.81 | 18 | 0.0650 |
| `imitate-nv` **pays** | rl + imitate | 0.1809 | 0.0454 | 0.0412 | 2.69 | 36 | 0.0314 |
| `head` / `head-fold` | no value signal | 0.2878 / 0.2850 | 0.3508 / 0.3590 | | 15.31 / 16.00 | 26 / 0 | 0.2101 |

(Non-nv arms replicate round 4 bit-identically; `head+nvh`/`head+nvh-fold`
and the `nextval` family are unchanged and omitted for space —
`head+nvh` still 0.2835 @ 36 stops.)

## What the corrected calibration shows

* **The over-stop was optimism, and it is fixed.**  Under in-sample
  train preds round 4's `head+nv` calibrated to a τ that stops at
  10.78 evals held-out (−0.028).  Under out-of-fold train preds the
  same arm lands τ mid-grid and stops at 15.67 evals for **0.2866**
  — +0.014 over incumbent, the first value-fed gate that both earns
  *and* still stops (18/36 episodes).  The signal is real but thin:
  the gate mostly declines to stop early because the predictor
  cannot certify an empty pool — honest calibration now prices that
  honestly instead of being fooled by memorized train preds.
* **The injected label decides *which* degenerate point.**  `useful`
  preds → τ at the drain end (1 stop, 0.2850 — the never-stop
  confound); `pays` preds → over-stop (9.5–9.2 evals, 0.2413/0.2425 —
  the pos_weight-inflated probs keep expected yield high everywhere,
  so `pool_value` drains slowly and the head quits early); `delta`
  preds → the only profitable mid-point.  `head+stop+nv` under
  `pays` is the second honest clear (0.2735, 12 real stops — the
  stop-schema columns discipline where the `pays` signal is used).
* **The fold comes within 0.0004 — and still misses.**  The
  serializable `head+nv-fold` under `delta` preds plays a *real*
  stopping policy in pure weight-table form: 22 sampled stops at
  14.67 evals, reward **0.2726** — below the 0.273 bar by four
  ten-thousandths, and its `op:stop` intercept calibration is a
  knife-edge (the train-reward landscape is flat: neighbouring c
  cells range 0.264–0.285 and the best cells are the never-stop
  end).  The fold variants that *do* clear reward (`head+nv-fold`
  under `useful`, 0.2850) stop **zero times** — the documented
  confound, not a stopping policy.  The structural reason stands:
  the gate's rule is an *absolute* threshold on the continue-score;
  the softmax fold can only express "stop beats the best eval
  logit", so the threshold drifts with eval-logit scale — the
  relative-vs-absolute gap the fold's docstring prices.
* **The RL deliveries still collapse.**  `inc+nv` under `delta`
  posts the best sampled reward in the whole run (0.3008) at AP
  0.1042 — the shared-softmax corruption, a fourth time; `inc+nv`
  under `useful` AP 0.0650.  `imitate-nv` under `pays` finally moves
  off the floor (0.1809, 2.69 evals — stops on average, finds
  nothing transferable, AP 0.0314).

## Verdict (round 5): not earned — `OPEN_GAME_PLAYER_WEIGHTS` stays

Gate: held-out reward ≥ 0.273 **and** AP ≥ 0.210, honestly —
never-stop or AP-collapse is an honest negative.  Two *gated* arms
clear both bars with real stopping (`head+nv` under `delta`
0.2866 @ 18 stops; `head+stop+nv` under `pays` 0.2735 @ 12 stops),
but a gate ships a second scorer + calibrated τ — not a weight
table.  Every *serializable* arm either misses (`head+nv-fold`
0.2726 — 0.0004 short), clears by never stopping (`head+nv-fold`
`useful`, 0.2850 @ 0 stops), or collapses the ranking (`inc+nv`,
AP 0.1042).  And a fold bake would still be dishonest in a second
way: its stop behavior depends on `set_value_preds` being fed at
runtime — unfed, the `x:stop:*` columns are zeros and the fold
degenerates to a constant `op:stop` propensity, so it is not a pure
weights table regardless.  **No bake.**

The five-round arc closes: the stop axis's ~+0.026 headroom is real
(oracle 0.3877), the gate mechanism works (corrected comparison,
reward-calibrated τ), the calibration is now honest (OOF preds), and
a per-candidate value signal helps the *decision rows* even at
candidate-AP 0.02 (solo 0.38) — but the two missing pieces are
structural: (a) a predictor for `useful` cannot exist on these
features (7 positives memorize; truth ∧ novelty is invisible to
program-shape ⊕ rule-shape), so signal quality caps the attainable
τ; (b) the gate's absolute threshold does not survive serialization
into a softmax weight table, and every weight-typed delivery tested
— RL columns ×4 feature families, CE imitation, and now the
head-fold — lands one of the two degenerate points.  A bakeable
stop policy needs either a fundamentally better usefulness signal
(verdict-conditioned features the verifier can see, not just term
shape) or a stop decision outside the shared softmax entirely.

---

# Round 6 — the self-posterior + per-step advantage (appendix; full doc `results_stop_advantage.md`)

**Date:** 2026-10-19 · **Script:** `bench/policy/stop_advantage.py`
(new — same protocol, reuses this driver's gate machinery) ·
**Runner:** `timeout 1800 .venv/bin/python
bench/policy/stop_advantage.py --json /tmp/adv6.json` (~3.5 min:
12 arms + 9 unfed replays; re-run reproduced every committed row
bit-identically).

Round 5 closed on two missing pieces — *evidence* (no feature
prices the pending pool) and *delivery* (every weights-table stop
delivery lands degenerate).  Round 6 prices both new levers:
**`open:self`** (`mlopt_discovery/open_self` — the policy's own
softmax over pending evaluates injected as `x:stop:self_*` /
`x:self:*` columns via `set_self_scores`) and **per-step
advantage** (`mlopt_discovery/advantage.AdvantagePlayer` — a
learned `V(s)=u·φ_stop` NLMS critic, so `stop`'s credit isolates
to `−V(s)` instead of the episode total).

## Feature power — self-posterior solo APs vs the oracle continue-labels

Held-out decision-row probe (`head+self`, base rate 0.155):

| column | solo AP | | reference columns | solo AP |
|---|---|---|---|---|
| `x:stop:self_topk` | **0.3335** | | `x:stop:remaining_frac` (r3 best) | 0.3287 |
| `x:stop:self_best` | 0.3312 | | `x:stop:next_value` (r4, delta net) | 0.3217 |
| `x:stop:self_ev` | 0.3309 | | `x:stop:next_value` (r5, useful net) | **0.3818** |
| `x:stop:self_above` | 0.2802 | | `x:stop:best_score` (heuristic) | 0.2007 |
| `x:stop:self_pay` | 0.1245 | | `x:stop:rank1_pred` (heuristic) | 0.1492 |
| `x:stop:self_p1` | 0.1110 | | | |

The self-posterior is a real pool-priced signal at **zero
predictor cost** — it matches the progress counter and the
delta-net injection and decisively beats the heuristic columns,
but does **not** out-score the useful-label preds (0.38), and its
LOO deltas are ≈0 (collinear with what exists; combined head eval
AP 0.40 < round-3's 0.55).  It is a *ranking* signal, still not a
*will-it-pay* one.

## Results (held-out odd pages, budget 17, `delta`-label injection)

| arm | mechanism | reward/ep | greedy | evals/ep | stops | AP |
|---|---|---|---|---|---|---|
| incumbent | rl 5ep | 0.2728 | 0.3590 | 14.44 | 19 | 0.2101 |
| ac | actor-critic | 0.2761 | 0.3659 | 14.44 | 20 | 0.1098 |
| ac+self | ac + `open:self` | **0.3009** | 0.3856 | 15.28 | 17 | 0.1166 |
| nv (r5 control) | rl + `open:next-val` | 0.3008 | 0.3810 | 15.31 | 16 | 0.1042 |
| **ac+nv** | ac + `open:next-val` | **0.2764** | 0.3555 | 15.36 | 12 | **0.2074** |
| head+self | gate over `open:self` | **0.2816** | 0.3559 | 15.89 | 36 | 0.2101 |
| head+self+nv | gate over both | **0.2813** | 0.3553 | 15.97 | 33 | 0.2101 |
| head+self+stop | gate + `open:stop` | 0.2698 | 0.3366 | 13.47 | 36 | 0.2101 |
| head+self-fold / +nv / +stop | weight tables | 0.2671 / 0.2650 / 0.2630 | | ~15 | 30/23/30 | 0.2101 |

## The unfed replay — delivery checked as `OPEN_GAME_PLAYER_WEIGHTS` would run it

Every row above was measured with the injected seams bound; the
weights table serializes without them, so each candidate is also
replayed with `set_self_scores(None)` + `set_value_preds(None)`:

| candidate | bound → unfed reward | bound → unfed AP |
|---|---|---|
| **ac+nv** | 0.2764 → **0.2772** | 0.2074 → 0.2074 |
| ac+self | 0.3009 → 0.3013 | 0.1166 → 0.1086 |
| self+nv / ac+self+nv | 0.2834 / 0.2954 → 0.2856 / 0.2978 | 0.1453 / 0.1174 → 0.1406 / 0.1095 |
| head+self-fold | 0.2671 → 0.2691 (stops 30→5) | 0.2101 → 0.2101 |
| head+self+nv-fold | 0.2650 → 0.2694 (stops 23→7) | 0.2101 → 0.2101 |
| head+self+stop-fold | 0.2630 → 0.2585 (stops 30→4) | 0.2101 → 0.2101 |

The injected columns are a *training-time teacher*: at the drain
operating point they change ~nothing at decode (≤0.003 reward),
so `ac+nv`'s near-miss is intrinsic — but the folds' stopping
*was* the seam (bound 23-30 stops → unfed 4-7).

## Bake gate — round 6

| candidate | reward ≥ 0.273 | AP ≥ 0.210 | serializable | verdict |
|---|---|---|---|---|
| `ac+nv` (bound or unfed) | ✓ 0.2764/0.2772 | ✗ 0.2074 (−0.003) | ✓ pure table + schema ext | **miss** |
| `ac+self` | ✓ 0.3009 | ✗ 0.1166 | ✓ modulo seam | AP collapse |
| `head+self` / `head+self+nv` | ✓ | ✓ | ✗ gate (scorer + τ) | not a weight table |
| `head+self+stop` | ✗ 0.2698 | ✓ | ✗ gate | miss |
| all `-fold` tables | ✗ ≤0.2694 | ✓ | ~ (unfed: stops vanish) | miss |

**Gate: not passed — `OPEN_GAME_PLAYER_WEIGHTS` stays.**  Honest
verdict on `open_self`: it is *not* the same confounds under a new
name — the evidence half genuinely moved (the first zero-cost
pending-pool view, matching the best injected columns) and the
per-step advantage is a real credit mechanism (`ac+nv` posts the
closest serializable result in six rounds, bound *or* unfed).  But
the gate is unforgiving in the same place: the posterior ranks the
pool, it cannot certify it, so reward-clearing weight tables still
ride collapsed rankings (AP ≤0.12), the earning deliveries are
still gates (non-serializable second scorer + τ), and the serial
folds still miss — `ac+nv` by 0.003 on AP, exactly where round 5's
`head+nv-fold` missed by 0.0004 on reward.  The remaining blocker
is unchanged: nothing in the state certifies "pending is duds"
better than ~0.4 decision-row AP, and the game's economics keep
rationally resolving that uncertainty to "drain".
