# The stopping policy — can the learned arm learn *when to quit*

**Date:** 2026-10-19 · **Script:** `lab/stop_policy.py` (new —
same memoized-verdict protocol as `lab/richer_features.py`) ·
**Seed:** 0 · **Runner:**

```bash
timeout 1800 .venv/bin/python lab/stop_policy.py \
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
`timeout 1800 .venv/bin/python lab/stop_policy.py --json /tmp/stop.json`
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
