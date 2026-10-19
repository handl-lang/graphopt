# Richer open-domain features — which new features move AP

**Date:** 2026-10-19 · **Script:** `lab/richer_features.py` · **Seed:** 0
throughout · **Hardware:** the 11 GB CPU box · **Runner:**

```bash
timeout 1800 .venv/bin/python lab/richer_features.py \
    --json /tmp/richer.json          # ~10 min
```

## Question

`docs/results_schema_leaderboard.md` found the shipped open schema's
load-bearing part is `action` (the `a:*` block) — but every schema it
enumerated was a subset of the *same* shipped vocabulary: the search
could only drop information, never add any.  This run grows the
vocabulary with seven new registered parts, each emitting one feature
the incumbent schema cannot express, and asks which — alone, and
leave-one-out from the full set — move held-out AP and reward.

## The new vocabulary

| part tag | emits | reads | what it encodes |
|---|---|---|---|
| `open:score-rank` | `a:score_rank` | state+action (`RelFeats`) | the candidate's heuristic-score rank quantile within the *pending* pool — 1.0 for the best option left |
| `open:score-rel` | `a:score_rel` | state+action (`RelFeats`) | candidate score minus the pending-pool mean, in `a:score` units |
| `open:score-x-depth` | `x:score:depth` | action (`ActionFeats`) | `a:score × a:depth / 4` — does the heuristic pay more for deep shapes |
| `open:score-x-ops` | `x:score:ops` | action (`ActionFeats`) | `a:score × a:ops / 8` — same for operator count |
| `open:pending-top` | `st:pending_top` | state (`StateBroadcast`) | best heuristic score still pending — what the board still offers |
| `open:pending-mean` | `st:pending_mean` | state (`StateBroadcast`) | mean heuristic score still pending |
| `open:useful-ct` | `st:useful_ct` | state (`StateBroadcast`) | useful finds banked this episode / pool size — the count gradient `st:useful_rate` (a ratio) lacks |

`RelFeats` is a new composable part class in `players.py` —
`fn(state, action, row) -> dict` — for terms `ActionFeats`'s
`(action, row)` signature cannot express.  Every part is registered
in `PARTS`, so spec records (and the player arena's toggle
vocabulary) can name them; `DomainSpec.to_record` round-trips.

## Protocol

Cold retrain per arm: **5 epochs × 36 even pages** (the shipped bake
recipe) of `engine.train_policy` under the baseline's **memoized
verdicts** (`policy_baseline._measure_pools` + patched `lp.evaluate` —
identical oracle draws for every arm, training included).  Held-out
odd pages, budget 17; sampled + greedy frozen arms; AP over the
576-candidate held-out pool (7 useful, random AP 0.0122).

**AP is page-aware.**  `rank_score` (`_cand_feats`) cannot emit
pool-relative features — a lone candidate has no pool — so each
candidate is scored as `w · featurize(fresh_page_state,
evaluate(cand))` and the scores pooled across pages.  For the
incumbent schema this is provably the `rank_score` ordering (all
non-`a:*` terms are identical constants on a fresh page); for richer
schemas it prices the pool context the new features add.

## Results

| arm | extra parts | reward/ep | greedy | use/ep | evals/ep | AP | feats |
|---|---|---|---|---|---|---|---|
| incumbent | — (shipped 5 parts) | 0.2768 | 0.3576 | 0.139 | 15.28 | 0.1332 | 41 |
| inc+score-rank | `open:score-rank` | 0.2598 | 0.3507 | 0.139 | 14.61 | 0.1151 | 42 |
| inc+score-rel | `open:score-rel` | 0.2414 | 0.2823 | 0.139 | 14.61 | 0.0172 | 42 |
| inc+score-x-depth | `open:score-x-depth` | 0.2824 | 0.3553 | 0.139 | 15.28 | 0.1138 | 42 |
| inc+score-x-ops | `open:score-x-ops` | 0.3019 | 0.3555 | 0.167 | 15.28 | 0.1196 | 42 |
| inc+pending-top | `open:pending-top` | 0.2768 | 0.3576 | 0.139 | 15.28 | 0.1335 | 42 |
| inc+pending-mean | `open:pending-mean` | 0.2768 | 0.3576 | 0.139 | 15.28 | 0.1332 | 42 |
| inc+useful-ct | `open:useful-ct` | 0.2768 | 0.3576 | 0.139 | 15.28 | 0.1332 | 42 |
| **all** | all seven | 0.2728 | 0.3590 | 0.167 | 14.44 | **0.2101** | 48 |
| all-score-rank | LOO | 0.2107 | 0.2910 | 0.111 | 14.81 | 0.0206 | 47 |
| all-score-rel | LOO | 0.2768 | 0.3553 | 0.139 | 15.28 | 0.1160 | 47 |
| all-score-x-depth | LOO | 0.2711 | 0.2910 | 0.139 | 14.69 | 0.0235 | 47 |
| all-score-x-ops | LOO | 0.2481 | 0.3590 | 0.139 | 14.33 | 0.2098 | 47 |
| all-pending-top | LOO | 0.2728 | 0.3590 | 0.167 | 14.44 | 0.2101 | 47 |
| all-pending-mean | LOO | 0.2728 | 0.3590 | 0.167 | 14.44 | 0.2101 | 47 |
| all-useful-ct | LOO | 0.2728 | 0.3590 | 0.167 | 14.44 | 0.2101 | 47 |
| **core4** | score-rank, score-rel, x-depth, x-ops | 0.2728 | 0.3590 | 0.167 | 14.44 | **0.2101** | 45 |
| **core3** | score-rank, score-rel, x-depth | 0.2481 | 0.3590 | 0.139 | 14.33 | 0.2095 | 44 |
| — anchors — ||||||||
| baked-prior (old schema, baked weights) | | 0.3271 | 0.3576 | 0.167 | 13.75 | 0.1332 | |
| heuristic drain | | 0.3595 | — | 0.167 | 16.00 | 0.0450 | |
| random | | 0.1339 | — | 0.083 | 8.67 | 0.0123 | |

## Per-feature verdicts

* **`a:score_rank` is the single most valuable feature added — and
  it is non-additive.**  Alone it *hurts* (−0.018 AP vs incumbent);
  leave-one-out from the full set it is the heaviest load
  (−0.19 AP).  The rank signal is only informative next to the other
  extensions — the policy must already trust `a:score` before
  "is this the best pending" pays.  This is exactly the interaction
  the schema leaderboard's subset search could never see: the
  feature is invisible to a single-part probe.
* **`x:score:depth` is the second pillar** (LOO −0.19 AP) — the
  score×depth interaction separates "high-score-and-deep" shapes
  (the paying profile) from high-score shallow ones.  Again
  non-additive: solo −0.019.
* **`a:score_rel`** — LOO −0.094; solo −0.116 (the worst solo arm;
  alone it flattens into a mean-reversion signal with no anchor).
* **`x:score:ops`** — LOO −0.0003 (AP-neutral) but +0.025 sampled
  reward when present (`core3` 0.2481 → `core4` 0.2728); kept.
* **`st:pending_top` / `st:pending_mean` / `st:useful_ct` — measured
  dead weight.**  Solo and LOO deltas are all 0.000–0.0003: the
  pending-pool statistics and the finds-count add nothing the schema
  uses.  They stay registered (a spec record can still name them)
  but are not in the shipped schema.
* **Reward is flat across arms** (0.21–0.30 sampled, ≤0.359 greedy):
  the game prices ordering, and every arm drains ~15 evals — the AP
  column is where the features pay.

## Baked

`core4` — the four candidate-level features — matches `all` on every
metric (AP 0.2101, +0.077 over the 0.1332 anchor, ~7.7σ of the ±0.01
noise band) with the fewest schema additions.  The four parts joined
`OPEN_PARTS`/`OPEN_GAME_FEATURES`/`OPEN_PART_TAGS`; the baked prior
was regenerated with `lab/train/train_open_prior.py` (same shipped
recipe, unmemoized draws — train block means
`[0.333, 0.341, 0.355, 0.289, 0.151, 0.192]`; its own held-out read:
reward 0.285 vs cold 0.143, use/eval 0.0121).  The new weights:
`a:score_rank` 0.080, `a:score_rel` 0.025, `x:score:depth` 0.0048,
`x:score:ops` 0.0093.

## Verdict

**Richer features do raise the ceiling — decisively on the ranking
axis.**  AP 0.210 vs the 0.133 prior anchor is a ~58% relative jump,
and it comes from features that encode what the incumbent schema
structurally could not: *where the candidate sits among the pending
pool* (`a:score_rank`, `a:score_rel`) and *score×shape interactions*
(`x:score:depth`).  The honest caveats: sampled reward does not
improve (0.273 vs 0.277 incumbent — the features sharpen ordering,
not stopping), the pool-stat/`st:*` half of the additions is dead
weight, and `rank_score`/`rank_pool` under-report the shipped
ordering because the ranker's `cand -> dict` contract cannot see
the pool — the page-aware probe is the honest AP.
