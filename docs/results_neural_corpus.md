# Neural-arm retest on the enlarged on-distribution pool

**Date:** 2026-10-20 · **Script:** `bench/drivers/policy/neural_corpus.py` ·
**Seed:** 0 throughout · **Runner:**

```bash
.venv/bin/python bench/drivers/policy/neural_corpus.py \
    --json /tmp/nc.json --cache /tmp/neural_corpus_cache.pt
```

(484 s wall on the 11 GB CPU box, excluding the verdicts already
checkpointed in the cache. The per-candidate verdict checkpoint makes
the run crash-resumable; a cold run costs ~2× that for the
held-out/train/tail rows.)

## The question

`docs/results_learned_ablations.md` ablation 4 isolated the neural
ranker's deficit as *data, not capacity*: hidden 32→96 moved AP by
−0.0006, while 576 on-distribution rows beat 7,560 corpus
`rule_samples` rows 0.0412 vs 0.0156. If data is the bottleneck,
growing the on-distribution pool should raise AP. This run grows the
training pool three ways (same featurization, same `lp.evaluate`
oracle, same held-out protocol — odd pages `gen_cases(0, 36, page=16,
start=1, step=2)`, memoized per-candidate-seeded verdicts):

* **tail** — the 8 default-stream candidates past both splits
  (offsets ≥ 1152; the even+odd pages cover 0..1151 of the
  1,160-candidate stream).
* **wide** — the same enumerator over the full grammar vocabulary
  (`GRAMMAR_BINARY_OPS + GRAMMAR_UNARY_OPS` = 14 ops: `pow`, `matmul`,
  `rsqrt`, `sigmoid`, `silu`, `tanh` join the default eight). The
  stream exhausts at 1,673 non-duplicate, non-held-out candidates
  (`--wide 1728` gets all 1,673; 2,833 raw candidates scanned).
* **sites** — near-miss pairs mined off real `nn.Module` exports
  (zoo + `zoo_variants` + 48 intake workloads = 94 workloads, 93
  exported, 1 rejection; 335 candidates at `max_dist=3, max_size=12`).

Every candidate is deduped against the held-out pool (and within each
pool) on the label-free instance key `(repr(lhs), repr(rhs))`.

## Measurement guard (new — needed on this box)

`lp.evaluate` → `verify_law` saturates a fresh e-graph under
`ALL_RULES` with `max_nodes=200_000` and no expansive-rule budgets
(unlike `_cost_delta`, which caps at 4,000 + budgets). The
default-ops stream never trips it; the widened vocabulary does —
`lit-self:1907 matmul(A,A) => add(A,0)` and its orientation twin were
measured OOM-killing this box at ~9.3 GB anon RSS (and an in-process
`MemoryError` catch is *not* survivable: once the address-space limit
bites, even the except-handler's allocations fail — observed cascade).

The driver now evaluates every candidate in a spawned worker under a
6 GiB `RLIMIT_AS` + 180 s timeout. A candidate that kills two fresh
workers (or stalls) is recorded as `None`, dropped from its pool, and
reported as `skipped` — never labeled, never trained on. Every
measured verdict is the protocol's exact verdict (same call, same
per-candidate seed). Skips this run: **2 of 1,673 wide** (the two
`matmul(A,A) ↔ add(A,0)` lit-self spellings), 0 elsewhere.

## Pools

| pool | n | positives (delta>0) | useful | num_true | skipped |
|---|---|---|---|---|---|
| even-pages (train base) | 576 | 271 | 8 | 37 | 0 |
| stream-tail | 8 | 4 | 0 | 0 | 0 |
| wide-grammar (14 ops) | 1,671 | 835 | 4 | 25 | 2 |
| zoo-sites (real programs) | 335 | 123 | 0 | 2 | 0 |
| corpus `rule_samples` | 15,555 | 129 | — | — | 0 |
| held-out (odd pages) | 576 | — | **7** | — | 0 |

(`corpus` = 7,650 `mixture_programs` rows at 30/family + 7,905 rows
over the 93 real exported terms — ablation 4's losing recipe re-fed
with real `nn.Module` programs.)

## Held-out AP — `RuleValueNet`, BCE 300 epochs, hidden 32

| arm | n | AP |
|---|---|---|
| **ondist+wide** (even + tail + wide) | 2,255 | **0.0595** |
| ondist-576 (ablation-4 recipe) | 576 | 0.0412 |
| ondist+tail | 584 | 0.0341 |
| ondist+sites | 911 | 0.0253 |
| ondist+all | 2,590 | 0.0181 |
| ondist+all@h96 | 2,590 | 0.0206 |
| corpus-mixture | 7,650 | 0.0189 |
| extra-only (tail+wide+sites) | 2,014 | 0.0167 |
| corpus+zoo | 15,555 | 0.0166 |

Anchors (this run, same pool): heuristic **0.0450**, trained-linear
`rank_score` **0.1284** (page-aware 0.2101 per
`results_richer_features.md`), random 0.0122.

## Episodes — reward/episode on the same held-out pages

The neural rows play the trained net as a drain-ranker (argmax over
`state.pending`, drain, `stop` — the heuristic arm's behavior, so
`evals/ep` is comparable); the same seeded `train_rule_value` call as
the AP rows, so identical weights.

| arm | reward/ep | useful/ep | evals/ep | useful/eval | found |
|---|---|---|---|---|---|
| random | 0.1339 | 0.083 | 8.67 | 0.0096 | 3 |
| heuristic drain | 0.3595 | 0.167 | 16.00 | 0.0104 | 6 |
| trained-linear (baked) | 0.2758 | 0.139 | 13.06 | 0.0106 | 5 |
| neural-ondist-576 | 0.3089 | 0.167 | 16.00 | 0.0104 | 6 |
| neural-ondist+all | 0.2916 | 0.167 | 16.00 | 0.0104 | 6 |

Both neural arms drain the page and find the same 6 of 7 useful —
every arm misses the 7th (ablation 2 found it unreachable for every
policy). The reward gap to the heuristic (0.309 vs 0.360) is ordering
credit: equal finds, worse order.

## Verdict — partial confirmation, honest limits

* **More same-generator data helps, modestly.** 576→2,255
  on-distribution rows lifts AP 0.0412→0.0595 (+0.018 ≈ 2σ over the
  ±0.01 seed-noise floor) — the first neural arm to beat the
  heuristic's 0.045. The "data, not capacity" diagnosis is confirmed
  in direction.
* **But the lift is sub-linear and the ceiling is still low.** 3.9×
  the data buys +44% AP, and the arm remains 2.2× below the linear
  episode-trained policy (0.0595 vs 0.1284). Data was *a* bottleneck,
  not the whole story — the `a:*`-feature episode policy's inductive
  bias stands.
* **Distribution match is what counts, not just the oracle.**
  `extra-only` — the same 2,014 additions *without* the 576 even-page
  core — scores 0.0167, no better than the corpus arm. And the
  real-program `sites` rows actively hurt: adding them to even+wide
  drops 0.0595→0.0181, and even+sites alone drops below the base
  (0.0253). Near-miss pairs off real terms share the featurizer and
  the oracle but not the proposal distribution — they behave like
  off-distribution data.
* **Real programs don't fix the corpus arm either.** corpus+zoo
  (15,555 rows incl. 7,905 real-export rows) scores 0.0166 ≤
  corpus-mixture 0.0189 — the off-distribution deficit is about the
  row *kind* (program-level `rule_samples`), not program realism.
* **Practical note:** the wide stream is the useful growth axis
  (cheap, dedupe-clean, 4 useful candidates of its own) but it
  exhausts at ~1,670 candidates — this source is now fully consumed.
  The two `lit-self` matmul spellings are unmeasurable on this box
  under `verify_law`'s 200k-node saturation; the skip is honest and
  reported, but if such candidates matter, `evaluate` needs a bounded
  `verify_law` path like `_cost_delta`'s.
  **Update (fixed):** `verify_law` now saturates bounded by default —
  every `EXPANSIVE`-tagged and non-library rule is capped at
  `verifier._EXPANSIVE_BUDGET` (600) new e-nodes and the node ceiling
  is 30,000, so this exact pair terminates at a ~1.5k-e-node fixed
  point in ~3 s and refuses. The worker guard above stays as
  defence-in-depth for any *other* unbounded caller.
