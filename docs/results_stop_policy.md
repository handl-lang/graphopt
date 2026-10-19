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
