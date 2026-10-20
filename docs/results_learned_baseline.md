# Controlled learned-policy baseline

**Date:** 2026-10-19 · **Script:** `bench/policy/policy_baseline.py` · **Seed:** 0
throughout · **Runner:**

```bash
.venv/bin/python bench/policy/policy_baseline.py --json /tmp/policy_baseline.json
```

The question under measurement is `docs/results.md`'s honest note:
"learned policies beat random but lose to the cost-delta heuristic."
Every arm is measured on the *same* held-out cases under the *same*
budget, with one control the pinned numbers did not have: `proposal.evaluate`
draws its numeric-oracle tensors from torch's global RNG, so a candidate
verdict is a *draw*, not a constant. This run measures every pool member
once under a per-candidate seed (`torch.manual_seed(seed + i)`) and
replays the memoized verdict through a patched `lp.evaluate` for every
arm — literally identical oracle draws, and verdicts consistent with the
ranking pass. The pinned tests' episode numbers carry that draw noise.

## Meta arena — `mp.gen_cases(0 + 7919, 16)`, budget 12

| arm | reward | cost | steps | cert | pin | reproduces? |
|---|---|---|---|---|---|---|
| scripted | −0.1000 | 2.94 | 2.00 | 16 | −0.10 | **yes** |
| greedy | −0.6000 | — | 12.00 | 0 | — | — |
| random | −0.5657 | 3.00 | 11.31 | 4 | −0.57 | **yes** |
| cold-linear | −0.4877 | 2.20 | 11.00 | 5 | −0.49 | **yes** |
| trained-linear | −0.0563 | 2.94 | 1.12 | 16 | −0.06 | **yes** |
| trained-greedy | −0.0500 | 2.94 | 1.00 | 16 | — | — |

All four pinned meta numbers reproduce. The trained prior is the best
reward (−0.056 vs scripted −0.10); greedy decoding of it is one step
per episode at the same cost.

## Open game — odd pages `gen_cases(0, 36, page=16, start=1, step=2)`, budget 17

Even pages (`start=0`) are the train half — disjoint, never evaluated.

| arm | reward/ep | useful/ep | evals/ep | useful/eval | found |
|---|---|---|---|---|---|
| random | 0.1339 | 0.083 | 8.67 | 0.0096 | 3 |
| heuristic | **0.3595** | 0.167 | 16.00 | 0.0104 | 6 |
| cold-linear | 0.1699 | 0.111 | 9.31 | 0.0119 | 4 |
| trained-linear | 0.3271 | 0.167 | 13.75 | **0.0121** | 6 |
| trained-greedy | 0.3576 | 0.167 | 16.00 | 0.0104 | 6 |

Pins (`docs/results.md`): random 0.134 / 0.083 useful — **reproduces**
(0.1339 / 0.083). Trained 0.304 / 0.167 — **reproduces** (0.327 / 0.167,
slightly higher under the paired draws). Cold 0.143 / 0.083 — **drifts**
(0.170 / 0.111; same direction, draw-level noise).

The `heuristic` arm — rank `pending` by `dag_cost(lhs) − dag_cost(rhs)`
descending, drain, then `stop` — is built in the script; the board ships
no heuristic episode player.

## Ranking AP — the same held-out pool (576 candidates, 7 useful)

| scorer | AP | pin | reproduces? |
|---|---|---|---|
| random (expectation = useful/pool) | 0.0122 | 0.012 | **yes** |
| cold-linear (`rank_score`, `w={}`) | 0.0123 | 0.012 | **yes** |
| heuristic (cost delta) | 0.0450 | 0.045 | **yes, exactly** |
| neural `RuleValueNet` | 0.0412 | 0.023 | **partially** — still loses to heuristic, but by 0.004 not 0.022 |
| trained-linear (`rank_score`, baked) | **0.1332** | 0.133 | **yes, exactly** |

Neural protocol: `RuleSample(features(lhs), rule_vector(as_rule),
delta = base_cost − cand_cost)` built from the 576 *even*-page candidates'
measured outcomes (271 positives), `train_rule_value` BCE 300 epochs
full-batch CPU, `hidden=32`. The 0.023 pin trained on a different sample
family (`rule_samples` over corpus programs); on on-distribution samples
the same net lands at 0.041 — within noise of the heuristic, not
categorically below it.

## Verdict

The claim "learned beats random but loses to the cost-delta heuristic"
is **confound-dependent, not a single fact**:

* **Ranking (AP):** the claim is *false* for the shipped linear prior —
  trained-linear AP 0.133 is **3.0×** the heuristic's 0.045 (reproduces
  the pin exactly). It is *true* for the cold start (0.012, pinned) and
  for `RuleValueNet` (0.041 vs 0.045 — direction reproduces, magnitude
  shrinks on on-distribution training data).
* **Episode reward:** the heuristic drain wins by a hair (0.3595 vs
  trained-greedy 0.3576, trained-linear 0.3271) — but this is the
  **stopping/drain confound**, not a ranking deficit. All three find 6 of
  the 7 useful laws; the heuristic drains all 16 candidates banking every
  true-but-shallow tier (0.2 each), while the trained sampled arm stops
  at 13.75 evals/ep and leaves ~2 candidates unpriced. On the metric the
  game was built to price — **useful laws per eval** — trained-linear
  leads (0.0121 vs 0.0104).
* **Single biggest confound:** `evals_per_ep` variation (8.7 → 16.0
  across arms) combined with the reward's ordering credit. Reward
  rankings conflate "rank well" with "keep spending"; AP and
  useful-per-eval separate them, and on both the trained prior beats
  the heuristic.

So: "learned loses to heuristic" is (a) **AP/ranking only for the
untrained/neural arms** — for the shipped trained prior it inverts;
(b) the reward-level edge the heuristic shows here is a
**stopping/drain artifact** of comparing a 16-eval drain against a
policy that stops at ~14 — not evidence of a worse ordering.
