# The stopping policy, round 6 — the observability attack on both axes

**Date:** 2026-10-19 · **Script:** `bench/policy/stop_advantage.py` (new
— same memoized-verdict protocol as `bench/policy/stop_policy.py`,
whose gate machinery it reuses) · **Seed:** 0 · **Runner:**

```bash
timeout 1800 .venv/bin/python bench/policy/stop_advantage.py \
    --json /tmp/adv.json            # ~6 min (148 s pool memoization)
```

## Motivation

`docs/results_stop_policy.md`'s five rounds closed on **two** missing
pieces, not one:

* **evidence** — no feature prices the *pending* pool: the
  heuristic's view tops out at candidate AP 0.045 and the
  verdict-label `RuleValueNet` memorizes its 7 positives, so
  "pending still pays" vs "pending is duds" is invisible to every
  learnable row;
* **delivery** — the corrected `_StopGate` earns (+0.015 over
  incumbent) but ships a second scorer + calibrated τ: every
  *weight-table* delivery — softmax columns, CE imitation, the head
  fold — lands a degenerate point, and rounds 3-5 showed the
  shared-softmax gradient *randomizes the ranking* (AP collapses to
  0.01-0.11) whenever the stop row carries real signal.

This round prices the two remaining levers named by the round-5
verdict and the report's own forward pointer ("a per-step advantage
or a real second head is the next arm"):

1. **the policy's own posterior as evidence** — the `open:self` part
   (`mlopt_discovery.open_self`, registered-but-unshipped).  Round 3
   named the blocker: a `RelFeats` part cannot see the weights, so
   `x:stop:rank1_pred` shipped the linear-score proxy.  The part
   lifts that barrier by *injection*: `set_self_scores` binds the
   model's own `w·φ` over the pending `evaluate` rows (live — the
   columns track the training weights; a busy flag makes the
   scorer's nested featurizes see the part as inert, so the
   posterior is over the base schema and no recursion is possible).
   Six `x:stop:self_*` columns (best/top-3 pending logit, rank-1
   share, posterior-weighted logit, posterior-weighted injected
   value, above-uniform count) plus two per-candidate columns on
   `evaluate` rows (`x:self:post` — this candidate's posterior
   share; `x:self:gap` — its margin over the field): the
   learned-units analogue of `a:score_rank`/`a:score_rel`.
2. **per-step advantage** — `mlopt_discovery.advantage.
   AdvantagePlayer`: the REINFORCE EMA-total baseline becomes a
   learned `V(s) = u·φ_stop(s)` fit online by the *normalized* LMS
   rule on the episode's own reward-to-go rows (NLMS — the raw delta
   rule diverged on the growing `x:stop:self_*` columns, measured
   `fsum` overflow mid-run).  The `stop` action's credit then
   isolates to `-V(s)` — the decision quantity the flat baseline
   drowns — and the `evaluate` rows' credit is state-conditioned
   too, the variance reduction the AP collapses priced.

## Protocol

Identical to `stop_policy.py`: cold retrain per arm
(`engine.train_policy`), even pages train / odd pages held out
(page 16, 36 pages, budget 17), page-aware AP, the same
`_measure_pools` memoized verdicts.  `nv` arms inject the
out-of-fold `delta`-label `RuleValueNet` table (round 5's only
earning label); head arms keep the round-4 protocol
(`rank_tags=()` — incumbent ranker) so the stop decision is
measured apart from the ranking.  `nv` replicates round 5's
`inc+nv`-delta row bit-identically (0.3008 / AP 0.1042) — the
harness port is exact.

## Results

| arm | mechanism | reward/ep | greedy | use/eval | evals/ep | stops | AP |
|---|---|---|---|---|---|---|---|
| incumbent | rl 5ep | 0.2728 | 0.3590 | 0.0115 | 14.44 | 19 | 0.2101 |
| ac | actor-critic | 0.2761 | 0.3659 | 0.0115 | 14.44 | 20 | 0.1098 |
| ac+stop | ac + `open:stop` | 0.2330 | 0.2695 | 0.0094 | 14.81 | 19 | 0.0121 |
| self | rl + `open:self` | 0.2299 | 0.3468 | 0.0076 | 14.61 | 18 | 0.0538 |
| ac+self | ac + `open:self` | **0.3009** | 0.3856 | 0.0109 | 15.28 | 17 | 0.1166 |
| nv | rl + `open:next-val` (r5 control) | 0.3008 | 0.3810 | 0.0109 | 15.31 | 16 | 0.1042 |
| **ac+nv** | ac + `open:next-val` | **0.2764** | 0.3555 | 0.0090 | 15.36 | 12 | **0.2074** |
| self+nv | rl + both | 0.2834 | 0.3590 | 0.0107 | 15.58 | 11 | 0.1453 |
| ac+self+nv | ac + both | 0.2954 | 0.3856 | 0.0109 | 15.25 | 17 | 0.1174 |
| head+self | gate over `open:self` | **0.2816** | 0.3559 | 0.0105 | 15.89 | 36/36 | 0.2101 |
| head+self-fold | weight table | 0.2671 | 0.3558 | 0.0112 | 14.86 | 30/36 | 0.2101 |
| head+self+nv | gate over both | **0.2813** | 0.3553 | 0.0104 | 15.97 | 33/36 | 0.2101 |
| head+self+nv-fold | weight table | 0.2650 | 0.3554 | 0.0108 | 15.50 | 23/36 | 0.2101 |
| — anchors — ||||||||
| oracle-stop ceiling* | best-prefix under the oracle | 0.3877 | — | 0.0444 | 3.75 | 36 | — |
| baked-prior | | 0.2758 | — | 0.0106 | 13.06 | — | 0.2101 |
| heuristic drain | | 0.3595 | — | 0.0104 | 16.00 | — | 0.0450 |
| random | | 0.1339 | — | 0.0096 | 8.67 | — | 0.0123 |

\* the round-2 diagnostic — unchanged.

## What the mechanisms showed

* **The self-posterior columns carry real signal — at the progress
  counter's level, not above it.**  Held-out decision-row probe
  (`head+self`, oracle continue-labels, base rate 0.155):
  `x:stop:self_best` solo AP 0.3312, `x:stop:self_topk` 0.3335,
  `x:stop:self_ev` 0.3309 — all ≈ the previous single-column best
  (`x:stop:remaining_frac` 0.3287), decisively beating the
  heuristic-priced value columns (`best_score` 0.2007,
  `rank1_pred` 0.1492) and matching the injected net's
  `next_value` (0.3217) *without* a trained predictor.  But the
  LOO deltas are ≈0 — the family is collinear with the progress
  features — and the head's total discrimination lands at eval AP
  **0.401**, *below* the round-3 combined peak (0.5494) and
  round-4's `head+nv` (0.4458).  The policy's own posterior is a
  real value view; it is not a better one than what exists.
  **Artifact caveat:** `x:self:post`/`x:self:gap` "solo AP 0.4593"
  is a constant-column artifact — those columns are eval-row only
  (absent on the stop row), so the probe reads 0.0 everywhere and
  the "AP" is the chronological-label sort, not a feature reading.
* **Per-step advantage rescues the ranking — halves the collapse,
  does not end it.**  AC-vs-RL twins: `ac+nv` AP 0.2074 vs `nv`
  0.1042 (+0.10); `ac+self` 0.1166 vs `self` 0.0538 (+0.06);
  `ac+self+nv` 0.1174 vs `self+nv` 0.1453 (−0.03).  Reward moves
  the other way on nv (0.2764 vs 0.3008, −0.024) and up on self
  (0.3009 vs 0.2299, +0.071).  The V fits are real and legible —
  `ac+nv`'s critic puts `x:stop:value_frac` +0.34 /
  `st:pending_frac` +0.36 / `st:seen_frac` −0.25 ("remaining return
  tracks remaining predicted value", the right sign) at v_err
  ~0.03 vs the ~0.3 return scale.
* **`ac+nv` is the closest serializable result yet — and still not
  a stopping policy.**  0.2764 ≥ 0.273 *and* above the current
  bake's own 0.2758 anchor on reward, at AP 0.2074 — missing the
  0.210 bar by 0.003.  Prior serializable misses were worse
  (round-5 `head+nv-fold` missed reward by 0.0004; every RL-column
  arm collapsed AP ≤0.11).  But it earns it by draining (15.36
  evals/ep, 12 stops at tail-dud savings) — the same profile as
  every near-miss, not the oracle's 3.75-eval operating point.
* **The gate still earns; the fold still misses.**  `head+self`
  0.2816 @ 36/36 stops and `head+self+nv` 0.2813 @ 33 — the
  corrected-gate pattern replicates on a new feature family
  (reward-clearing, AP-preserving, non-serializable, and again
  stopping only at ~16 evals — tail savings, not early quitting).
  The folds land 0.2671/0.2650 — real stopping (30, 23) but
  short of the bar.
* **`ac+stop` is the worst of both worlds** — 0.2330/AP 0.0121:
  the stop-row `x:stop:*` columns under per-step credit still
  collapse the ordering, and a first-run pathology is worth
  noting: before the NLMS fix the critic could reach a degenerate
  fixed point (stop-early episodes → V sees ~0 returns → flat
  credit forever), and `ac+stop` still lands nearest it.

## Verdict: not earned — `OPEN_GAME_PLAYER_WEIGHTS` stays

Gate: held-out reward ≥ 0.273 **and** AP ≥ 0.210, honestly — the
never-stop confound or an AP collapse is an honest negative.  No
weights-table arm clears both: `ac+nv` misses AP by 0.003
(0.2074); `ac`/`ac+self`/`ac+self+nv` clear reward on collapsed
rankings; both folds miss.  The two *gated* arms clear both bars
but are gates (a second scorer + τ — not `OPEN_GAME_PLAYER_WEIGHTS`
shape).  **No bake.**

The honest six-round summary: the blocker narrowed on both axes
but closed on neither.

* **Evidence:** the self-posterior part delivers the model's own
  pending-pool view for the first time — signal quality ≈ the
  progress counter's, above the heuristic's and on par with the
  injected net's, at zero predictor cost.  It is now the cheapest
  pending-value signal available — and it is still a *ranking*
  signal, not a *will-it-pay* one: the posterior says which
  candidate the model prefers, not whether evaluating it nets
  positive.
* **Credit:** per-step advantage is a genuine mechanism gain —
  `ac+nv` posts the closest serializable result in six rounds and
  halves the AP corruption — but the stop timing still does not
  appear: every arm operates at 14-16 evals/ep because the
  *decision* to quit early still has no observable that certifies
  an empty pool, and V learns the same proxy features that
  couldn't certify it before.

The remaining headroom (+0.026 reward, ~4× yield-per-eval at the
oracle point) is now bounded from a new direction: not the credit,
not the features' ranking content — the *label*: nothing in the
state (however posed — heuristic, injected net, or the policy's own
posterior) can certify "pending is duds" better than AP ~0.33-0.55,
and the reward table's asymmetry prices that uncertainty as
"drain".  A bakeable stop policy needs either a fundamentally
different evidence source (verdict-conditioned or verifier-visible
features, as the round-5 close said) or the decision moved outside
the shared softmax — the gate arms keep earning exactly because
they already live there.
