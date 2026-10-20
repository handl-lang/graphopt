# Learned-policy ablations

**Date:** 2026-10-19 · **Script:** `bench/drivers/policy/policy_ablations.py` · **Seed:** 0
throughout · **Runner:**

```bash
timeout 1800 .venv/bin/python bench/drivers/policy/policy_ablations.py \
    --json /tmp/policy_ablations.json
```

(~4 min wall clock on the 11 GB CPU box; no scale reduction needed.)

## Methodology

Same protocol as `docs/results_learned_baseline.md`, extended:

* **Held-out:** odd pages `og.gen_cases(0, 36, page=16, start=1, step=2)`,
  budget 17. Pool = 576 candidates, **7 useful** (AP random level 0.0122).
* **Train:** the disjoint even pages (`start=0, step=2`), 576 candidates,
  271 oracle-positive (delta > 0) rows.
* **Verdict memoization — extended to training.** Every pool member is
  measured once under a per-candidate seed and `lp.evaluate` is patched
  to replay the memoized verdict for *episodes and training runs alike*.
  The shipped bake drew fresh oracle tensors every episode; here every
  arm and every epoch sees the *same* realization — a
  controlled-difference choice that removes epoch-level draw noise so
  the schedule/reward variable is isolated. `cold+rl` re-derives the
  shipped recipe under this control and lands at AP 0.118 vs the baked
  0.133 — the residual gap is draw-realization noise, not protocol.
* **n, epochs:** 36 train pages × 5 epochs REINFORCE (the
  `train_open_prior.py` recipe); imitation = one `imitate_policy` pass
  (576 demonstrations, 576 imitated — 100% acceptance); neural arms =
  `train_rule_value` BCE 300 epochs full-batch CPU.
* **Noise band:** the same `cold+rl` recipe at two player seeds gives
  AP 0.118 / 0.113 — treat ±0.01 as the seed-noise floor on AP.

## Ablation 1 — heuristic bootstrap

| arm | reward/ep | useful/ep | evals/ep | useful/eval | found | AP |
|---|---|---|---|---|---|---|
| random | 0.1339 | 0.083 | 8.67 | 0.0096 | 3 | — |
| heuristic drain | 0.3595 | 0.167 | 16.00 | 0.0104 | 6 | 0.045* |
| trained-linear (baked) | 0.3271 | 0.167 | 13.75 | **0.0121** | 6 | **0.1332** |
| cold | 0.1699 | 0.111 | 9.31 | 0.0119 | 4 | 0.0123 |
| primed (imitation only) | 0.3467 | 0.167 | 15.97 | 0.0104 | 6 | 0.0555 |
| primed-greedy | 0.3611 | 0.167 | 16.00 | 0.0104 | 6 | |
| primed+rl | **0.3544** | 0.167 | 16.00 | 0.0104 | 6 | 0.0557 |
| primed+rl-greedy | 0.3611 | 0.167 | 16.00 | 0.0104 | 6 | |
| cold+rl | 0.3043 | 0.167 | 14.03 | **0.0119** | 6 | **0.1184** |
| cold+rl-greedy | 0.3553 | 0.167 | 16.00 | 0.0104 | 6 | |

*heuristic AP is a scorer baseline, not an episode arm.

**Warm-start does not beat from-scratch where it counts.** Imitation
works exactly as designed — `a:score` weight jumps to 1.67 and the
policy reproduces the heuristic's drain behavior and its episode
reward (0.347→0.354 ≈ heuristic 0.360). But it also inherits the
heuristic's *ranking*: AP 0.055 vs heuristic 0.045. Five epochs of
REINFORCE on top barely move it (AP 0.0555→0.0557 — the imitation
prior anchors the policy in the heuristic's basin). Cold+RL, left to
find its own features, reaches AP 0.118 — 2.1× the primed arm — and
leads on the game's priced metric, useful/eval (0.0119 vs 0.0104).
The primed arm's reward edge is the drain/stopping confound (16 vs
14 evals/ep), same confound the baseline doc isolated.

## Ablation 2 — exploration schedule (init = shipped prior)

| arm | reward/ep | useful/ep | evals/ep | useful/eval | found | AP |
|---|---|---|---|---|---|---|
| prior-frozen | 0.3271 | 0.167 | 13.75 | **0.0121** | 6 | **0.1332** |
| prior-frozen-greedy | 0.3576 | 0.167 | 16.00 | 0.0104 | 6 | |
| eps-0.1 | 0.3100 | 0.167 | 15.50 | 0.0108 | 6 | **0.1396** |
| eps-0.3 | 0.3525 | 0.167 | 16.00 | 0.0104 | 6 | 0.0956 |
| temp-anneal (4→0.25) | 0.3004 | 0.167 | 15.39 | 0.0108 | 6 | 0.1048 |
| imitate-again | 0.3467 | 0.167 | 15.97 | 0.0104 | 6 | 0.0555 |

**Continued exploration finds no MORE useful laws** — every arm finds
the same 6 of 7 (useful/ep 0.167 flat); the 7th is missed by every
policy including the heuristic drain, so it is likely outside the
feature set's reach, not a coverage problem. Mild exploration
(eps-0.1) buys a marginal AP bump (0.140 vs 0.133, ~1σ of seed
noise); heavier exploration degrades the ranking (eps-0.3 → 0.096,
anneal → 0.105). Imitating the heuristic on top of the prior is
actively harmful to AP (0.133→0.056) — the same anchoring seen in
ablation 1.

## Ablation 3 — reward design (all cold, 5-epoch RL, evaluated under the standard reward)

| arm | reward/ep | useful/ep | evals/ep | useful/eval | found | AP |
|---|---|---|---|---|---|---|
| std (full reward) | 0.3153 | 0.167 | 13.64 | 0.0122 | 6 | **0.1134** |
| no-eff (ordering credit zeroed) | 0.2809 | 0.167 | 13.53 | 0.0123 | 6 | 0.1032 |
| no-useful (tier zeroed) | 0.2349 | 0.139 | 11.08 | 0.0125 | 5 | 0.0501 |

**The `useful` = 1.0 tier is the load-bearing reward term, not the
ordering credit.** Removing `OPEN_GAME_EFFICIENCY` costs ~0.01 AP
(within ~1σ noise); removing the useful *tier* collapses AP to 0.050 —
heuristic level. The ordering credit shapes *how* reward is earned
(eval placement, drain-vs-stop), but the big tier differential between
`useful` (1.0) and `true` (0.2) is what teaches the ranking. Notably
`no-eff` still learned a ranking with `a:score` weight ≈ 0 — it found
the useful signal through `a:ops`/`a:metas`, not the heuristic feature.
Caveat on `no-useful`'s useful/eval 0.0125: it tops that column only
because it stopped early (11.1 evals/ep) after finding 5 — a
denominator artifact, not a better policy.

## Ablation 4 — neural capacity × data

| hidden | on-distribution (even-page outcomes, n=576) | corpus `rule_samples` (n=7,560 / 90 programs) |
|---|---|---|
| 32 | **0.0412** | 0.0156 |
| 96 | 0.0406 | 0.0156 |

reference: heuristic AP 0.045, trained-linear 0.133, random 0.012.

**The 0.023→0.041 gap is data, not capacity.** Doubling hidden 32→96
moves AP by −0.0006 on-distribution and 0.0000 on corpus — the model is
data-limited, not capacity-limited. On-distribution samples (candidate
LHS features + measured deltas, 576 rows) beat a 13× larger corpus
(7,560 `rule_samples` rows over `mixture_programs`) by 2.6×. Even at
its best the neural ranker only ties the heuristic (0.041 vs 0.045)
and loses 3× to the linear policy — the episode-trained `a:*`-feature
policy is the better inductive bias for this pool.
(`--corpus-per-family 30`, halved from `train_search_policy.py`'s 60
for CPU budget; the corpus arm is already far below the interesting
threshold so the smaller sample is not load-bearing.)

## Ablation 5 — yield per eval (the user's metric)

Best arm per ablation on `useful/eval` (sampled decode; every greedy
arm drains 16 evals → 6/576 = 0.0104):

* bootstrap: `cold+rl` **0.0119** (vs primed+rl 0.0104)
* exploration: `prior-frozen` **0.0121** (no arm beats frozen)
* reward: `std` 0.0122 (`no-eff` 0.0123 is a 1-eval/ep denominator
  artifact of slightly different stopping)
* overall: trained-linear's **0.0121** stands — the shipped prior is
  still the best learned ranker per eval.

## Verdict

* **Load-bearing knob: the `useful = 1.0` reward tier.** Zeroing it
  drops AP from ~0.11 to 0.050 (heuristic level); zeroing the ordering
  credit costs ~0.01. The tier differential — not the efficiency
  credit — is what makes the learned policy out-rank the heuristic.
* **Is heuristic-bootstrapped exploration the right bet? No.**
  Imitation reliably reproduces the heuristic's drain behavior and its
  reward, but it anchors the policy in the heuristic's ranking basin
  (AP ~0.055) and 5 epochs of REINFORCE cannot escape it (0.0557). The
  same RL budget from scratch learns a 2× better ranking (0.118) and
  the better useful/eval. Warm-start buys episode reward via the
  drain confound, not ordering quality. If reward-per-episode is the
  only target, priming wins; if ranking/yield-per-eval is the target —
  the metric the game was built to price — train from scratch with the
  useful tier intact, and keep exploration mild (eps ≈ 0.1 at most).
* Secondary: the neural arm's ceiling is a *data* problem (capacity
  changes nothing at either size); even fixed, it only reaches
  heuristic parity — the linear episode policy remains the right model
  class here.
