# Delivery prior v8 — featurizer-v2 analysis + retrain

Worktree: `/home/luis/tmp/mlopt-v8` (branch `delivery-v8-features`).
v7 result being beaten: `train_result_v7d.json` — held-out rewards
random 4.647 > shipped 4.475 > trained-greedy 4.429 > playbook 4.401 >
shipped-greedy 4.346 > trained 2.409 > cold 1.971.

## What v1's featurizer could see (and what it learned)

`play.py` v1 exposed 27 features: drain progress (`st:steps`,
`st:n_pending`, `st:n_done`), `st:speedup` (raw argmin ratio), the
eligibility flags (`cuda`/`fusible`/`compile`/`alias`), shape facts
(`n_inputs`/`n_enodes`/`n_params`/`lowprec`), config-kind one-hots,
`a:alias`/`a:fuse`/`a:graph`, `a:stop`, `x:pending`,
`x:stop:speedup`, `x:stop:improve`.

The v7 checkpoint weights are the autopsy: **every `st:*` weight is
~1e-17 — zero.** The shipped prior is a *static* action ranking
(`a:kind:fuse+graph` +1.95, `a:graph` +0.87/+2.41 via `a:graph`,
`a:stop` −1.62 mirrored by `x:pending` +1.62, `x:stop:improve`
−0.18). It cannot condition on the case at all — it plays the same
kind-preference line on every board and mostly drains.

## What was missing (v2 additions, all decision-time honest)

Rebuilt every corpus case's `DeliveryState` from the persisted census
fields (`/home/luis/tmp/delivery_v8/analyze_features.py`,
`features_dump.txt`) — no lifts needed; the census already stores
`eager_med_ms`, `n_enodes`, `n_params`, flags and full drain records.

1. **`st:eager_ms`** — the absolute eager timing, measured at board
   construction (`DeliveryBoard.__init__` benches eager once). v1
   never exposed it. It spans 0.05 ms (hnl kernels) → 250 ms
   (whisper encoders) and is the strongest eager-win predictor:
   all three big encoder cases (`arch-whisper-small-encoder` 245 ms,
   `whisper-small-encoder` 232 ms, `arch-whisper-tiny-full` 39.6 ms)
   are oracle `None`, and `st:big` (eager>30 ms or enodes>300)
   isolates exactly them — no winner in the corpus trips it
   (max winner enodes = 195, whisper-turbo-decoder).
2. **`st:committed`** — the margin-gated speedup `committed()` would
   ship, distinct from raw `st:speedup` (they differ inside the
   WIN_MARGIN noise band).
3. **`st:n_scored` / `st:n_declined`** — referee outcome counts
   (verify-gate/lower/bench declines are observable post-eval).
4. **`x:meas:<kind>` / `x:tried:<kind>` / `x:kfus:<kind>`** — per-kind
   measured speedups computed from each record's OWN paired-round
   eager median (`rec["eager_med_ms"]/rec["med_ms"]`, capped 4.0),
   tried-flags, and kind×fusible interactions. This is the "which
   runner kinds pay on THIS board" signal v1 lacked entirely.
5. **`a:pref`** — position in `deliver_game.CONFIGS` preference order.
6. **Stop-vs-run mirrors** — `x:stop:committed`, `x:stop:eager_ms`,
   `x:stop:big`, `x:stop:n_pending` on stop rows vs `x:act:eager_ms`,
   `x:act:big` on config rows: the linear model can now price
   "bail on huge boards, drain cheap ones".

## Corpus structure the prior must learn

- `fuse` and `compile` **never won** in 22 cases — only
  `fuse+graph` (7), `graph` (5), `compile+graph` (3),
  `fuse+graph-alias` (1), and `None` (6, incl. the no-drain
  kokoro-textenc case).
- All `compile+graph` picks are on **non-fusible** boards; on
  fusible boards `fuse+graph` dominates except thin-margin cases
  (`hnl-linear-tail` graph 0.194 vs fg 0.196; `hnl-select-sort`
  graph 0.044 vs fg 0.100 — graph genuinely 2.3× faster there).
- Held-out picks: dispatch-many-l `fuse+graph` (12.4×),
  select-sort `graph` (~4.4×), where-mix `graph` (~1.7×),
  gemm-bound `None`, arch-whisper-small-encoder `None`.

## Reward-shaping subtlety found in analysis

`committed_speedup` = (argmin-`med_ms` config's OWN paired eager) /
(its med). Each config's paired eager drifts (select-sort: fg eager
0.752 vs graph eager 0.193 — 4× drift on a 0.2 ms workload), so
adding evaluations can *lower* the committed ratio when a
lower-med config carries a luckier-low paired eager. The oracle
line `[pick, "stop"]` is therefore strictly better than drain-all
by `MOVE_COST` per extra move — which is why random (early stops)
beat the v1 trained arms that mostly drained. v2's
`x:stop:committed`/`x:meas:*`/`x:tried:*` features are exactly the
signals needed to learn efficient stop timing.

## v8 run

Driver: `/home/luis/tmp/delivery_v8/train_v8.py` (per-epoch
checkpointed, warm-started from `train_v7c.checkpoint.json` weights —
name-keyed, new features read 0 until trained). 4 imitation epochs
(22/22 moves cloned each) + 8 REINFORCE epochs on the 12 train
cases, then the 7-arm held-out table on the 5 held-out cases.
Needed `SOURCE_ONNX_REPO=/home/luis/GitHub/handl-lang/source-onnx`
for the `arch:` lift (worktree has no source-onnx sibling).

Results (`train_result_v8.json`, all 35 cells completed, no errors):

| arm | v7d | v8 |
|---|---|---|
| random | 4.647 | 4.433 |
| shipped | 4.475 | 4.501 |
| shipped-greedy | 4.346 | 4.563 |
| playbook | 4.401 | 4.439 |
| trained-greedy | 4.429 | **4.314** |
| trained | 2.409 | 4.224 |
| cold | 1.971 | 1.881 |

Trained curve block means: [4.288, 3.995, 2.749, 5.008, 4.325, 3.851].
Imitation: 88/88 moves cloned per epoch.

## Verdict: honest negative — weights.py stays `{}`

v8 did **not** beat random/playbook on held-out (trained-greedy
4.314 vs random 4.433 / playbook 4.439 / shipped-greedy 4.563).
Run-to-run bench noise is ~±0.2 (the shipped arms are the *same*
cold policy in both runs yet moved 4.475→4.501 / 4.346→4.563), so
the v7→v8 trained-greedy delta (4.429→4.314) is inside noise — but
there is no evidence v2 features helped, and the new-feature
weights are small (`x:meas:*`/`x:tried:*`/`st:big`/`st:eager_ms`
all < 0.04; the table is still dominated by kind biases, the
`a:stop`/`x:pending` mirror, `x:kfus:*` interactions and the new
`x:stop:committed` +0.82).

### Why it still fails — the missing signal is not a feature

1. **The oracle pick is only identifiable *after* measuring.** The
   winning config's `med_ms` requires running lower+verify+bench —
   exactly what a move costs. No decision-time feature can name it;
   the best honest policy is ordering + stop timing, and random's
   uniform drain already approximates the argmin-minus-move-cost
   ceiling on a ≤7-config pool.
2. **The None/win boundary among small non-fusible boards is not
   feature-separable.** `hnl-gemm-bound` (enodes=5, eager 0.50 ms →
   oracle None: every config measured *slower* than eager) vs
   `hnl-attn-wide` (enodes=9, eager 0.12 ms → graph 2.4×) have
   near-identical broadcast features and opposite labels — the
   difference is measured med_ms, i.e. label noise w.r.t. any
   pre-measurement featurizer.
3. **`committed_speedup` is drift-noisy.** Per-config paired eager
   medians drift up to ~4× on sub-ms workloads (select-sort:
   0.752 vs 0.193 ms), so the reward itself is a noisy argmin —
   another reason early-stop/random exploration scores competitively.
4. What v2 *did* add that is real: `st:eager_ms`/`st:big` isolate the
   big-encoder eager-win regime, `x:stop:committed` learned a
   sensible +0.82 (stop when the margin-gated speedup is banked),
   and `x:kfus:*` learned fuse-family kinds pay only on fusible
   boards. These are kept — they are honest observable signals and
   the schema is forward-compatible (weights are name-keyed; a
   future corpus can retrain into the same table).

### What would actually move the needle (next iteration)

- **More corpus, not more features**: 12 train cases cannot pin a
  60-feature linear prior; the None-vs-win boundary needs dozens of
  small-board drains.
- **A cheap pre-screen signal**: e.g. one *single*-config probe
  (eager vs one graph capture) whose measured delta predicts the
  rest — changes the game, not just the featurizer.
- **Reward de-noising**: the paired-eager drift suggests scoring
  against the construction-time eager median (or a longer bench)
  would make the argmin label less noisy — a board change, tracked
  separately.
