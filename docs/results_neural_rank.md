# Neural-arm ranking attack — pairwise/listwise objectives on the grown pool

**Date:** 2026-10-20 · **Script:** `bench/drivers/policy/neural_rank.py` ·
**Seed:** 0 throughout · **Runner:**

```bash
.venv/bin/python bench/drivers/policy/neural_rank.py \
    --json /tmp/nr.json --cache /tmp/neural_corpus_cache.pt
```

(~90 s wall — every verdict replays from the `neural_corpus`
checkpoint; no new `lp.evaluate` calls were needed.)

## The question

`docs/results_neural_corpus.md` located the residual neural gap in
**ordering credit, not coverage**: on the grown ondist+wide pool the
BCE-trained `RuleValueNet` finds the same 6 of 7 useful candidates as
the heuristic drain (16 evals/ep each) yet earns 0.309–0.321 vs
0.360 reward/episode — it ranks useful candidates lower *within a
page*. BCE is a pointwise objective: it scores each candidate against
a fixed threshold and never compares two candidates on the same
slate. This run keeps everything fixed — `RuleValueNet` (hidden 32),
the featurization, the 2,255-row ondist+wide pool (even pages + tail
+ wide), 300-epoch full-batch Adam @ lr 1e-2, seed 0 — and swaps only
the loss for per-page ranking objectives.

## Ranking groups

Losses are computed *within the pages the candidates arrived on* —
the structure the episode reward actually pays:

* even pages: the protocol's own 36 × 16 structure (train half);
* stream tail: one group of 8;
* wide grammar: the flat stream chunked into 16s.

142 groups over 2,255 rows; per-group losses averaged (a page counts
once regardless of pair count).

## The arms — a label × loss-form grid

| arm | label | loss |
|---|---|---|
| `bce-ondist+wide` | delta > 0 | pointwise BCE (the control — must reproduce 0.0595) |
| `bce-useful` | useful | pointwise BCE, `pos_weight` = neg/pos ≈ 187 |
| `rank-delta` | delta > 0 | pairwise RankNet (`softplus(-(s_i − s_j))` per ordered pair) |
| `rank-useful` | useful | pairwise RankNet |
| `rank-graded` | useful=2 > delta>0=1 > rest=0 | pairwise RankNet |
| `listnet-graded` | 3-tier | listwise ListNet top-1 softmax CE |

`bce-useful` + `rank-useful` complete the 2×2 that separates the
*label* change from the *loss-form* change.

## Held-out AP — pool 576, useful 7

| arm | AP |
|---|---|
| **rank-useful** | **0.1941** |
| listnet-graded | 0.0687 |
| bce-ondist+wide (control) | 0.0595 ✓ reproduces |
| rank-graded | 0.0550 |
| rank-delta | 0.0441 |
| bce-useful | 0.0254 |

Anchors (this run, same pool — identical to the corpus run):
heuristic **0.0450**, trained-linear `rank_score` **0.1284**, random
0.0122.

## Episodes — reward/episode, same held-out pages, memoized verdicts

Every neural arm plays its net as a drain-ranker (argmax over
`state.pending`, drain, `stop`). Coverage is identical across all
arms — 6 of 7 useful found, 16.00 evals/ep — so every reward delta is
pure ordering credit. (The 7th useful, `bin-unary-right:859`, is
`derivable=True` — unreachable for every policy per ablation 2.)

| arm | reward/ep | useful/ep | evals/ep | found |
|---|---|---|---|---|
| heuristic drain | 0.3595 | 0.167 | 16.00 | 6 |
| **neural-listnet-graded** | **0.3549** | 0.167 | 16.00 | 6 |
| neural-rank-useful | 0.3368 | 0.167 | 16.00 | 6 |
| neural-bce (ondist+wide) | 0.3208 | 0.167 | 16.00 | 6 |
| neural-rank-delta | 0.3185 | 0.167 | 16.00 | 6 |
| neural-rank-graded | 0.3134 | 0.167 | 16.00 | 6 |
| neural-bce-useful | 0.2901 | 0.167 | 16.00 | 6 |
| trained-linear (baked) | 0.2758 | 0.139 | 13.06 | 5 |
| random | 0.1339 | 0.083 | 8.67 | 3 |

## The diagnostic: in-page rank of the 7 useful held-out candidates

AP is pool-global; the reward only pays for order *inside* the page.
Each entry is `1 +` the number of page-mates scoring strictly higher.
Columns are the held-out pool indices (index 427 is the derivable
useful no policy can reach; the other six are the reachable finds):

| arm | mean | top-3 | 204 | 209 | 212 | 246 | 334 | 427† | 546 |
|---|---|---|---|---|---|---|---|---|---|
| heuristic | 2.71 | 4 | 1 | 1 | 4 | 2 | 5 | 5 | 1 |
| listnet-graded | 3.86 | 6 | 3 | 1 | 3 | **16** | 1 | 2 | 1 |
| bce-ondist+wide | 3.86 | 4 | 1 | 3 | 2 | 6 | 6 | 6 | 3 |
| rank-delta | 4.00 | 3 | 1 | 6 | 4 | 2 | 6 | 6 | 3 |
| rank-graded | 4.00 | 2 | 1 | 5 | 4 | 5 | 4 | 6 | 3 |
| rank-useful | 5.14 | 4 | 3 | 2 | 1 | 10 | 1 | 12 | 7 |
| bce-useful | 7.29 | 2 | 5 | 14 | 2 | 6 | 3 | 13 | 8 |

† index 427 = `bin-unary-right:859`, `derivable=True` — unreachable
for every policy (ablation 2); its in-page rank cannot pay out.

listnet-graded's single deep miss is index 246 (`rekey:502`, page 15)
— a *reachable* useful it drains last, earning ~zero ordering credit
on that find yet still posting the best neural reward: it wins the
other five reachable finds at in-page ranks 1–3.

## Verdict — ranking closes the gap, but it is the *interaction*

* **The label × form interaction is the finding.** On the old
  `delta > 0` label, swapping BCE for pairwise RankNet *hurts*
  (0.0595 → 0.0441): the form alone buys nothing, because most
  `delta > 0` rows are merely-positive non-usefuls — ordering them
  above negatives doesn't order usefuls. On the `useful` label the
  picture inverts completely: pointwise BCE collapses (0.0254 — 12
  positives in 2,255 rows is a brutal pointwise fit even at
  `pos_weight` ≈ 187) while RankNet turns the same 12 rows into ~180
  in-page comparisons and reaches **0.1941** — 3.3× the BCE arm,
  above the trained-linear anchor (0.1284), just under the page-aware
  linear's 0.2101. Rare-positive data is exactly the regime where a
  pairwise objective out-extracts a pointwise one: each useful row is
  one BCE term but ~15 pair terms.
* **AP and reward decouple — the ordering gap is in-page, not
  global.** rank-useful's 0.1941 AP coexists with in-page ranks
  [1,1,2,3,7,10,12]: it orders usefuls near the top of the *global*
  pool yet buries two reachable finds mid-page (10 and 7; the rank-12
  is the unreachable derivable).
  listnet-graded is the inverse: a modest 0.0687 AP but five of six
  reachable finds at in-page rank 1–3 and the best neural reward ever
  measured, 0.3549 vs the heuristic's 0.3595 (−0.005, ~inside the
  ±0.01 seed-noise floor — call it parity).
* **Honest limits.** No neural arm *beats* the heuristic on reward —
  the cost-delta heuristic's in-page mean 2.71 is still the best. And the ordering hypothesis is confirmed in both
  directions: reward tracks in-page rank (listnet 0.3549 > rank-useful
  0.3368 despite 3× worse AP), not global AP. For the *play* metric,
  training on within-page structure with the metric's own label is
  what matters — not rank quality in the abstract.
* **Where this leaves the neural arm.** The ordering-credit diagnosis
  from the corpus run is confirmed and now mechanized: give the net a
  page-local objective on the reward's label and the residual reward
  gap shrinks from −0.039 (BCE ondist+wide, 0.321) to −0.005
  (listnet-graded, 0.355). The remaining gap to the heuristic, and
  the whole gap to the episode-trained linear policy's *stopping*
  behavior (it earns 0.276 in only 13.06 evals — efficiency, not
  ordering), is no longer a ranking-loss question.
