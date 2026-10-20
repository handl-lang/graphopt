# Learned meta-arm on the `player` domain — trained, and it earns a bake

**Date:** 2026-10-19 · **Script:** `bench/train/train_player_meta_arm.py` ·
**Seed:** 0 · **Runner:**

```bash
timeout 1800 .venv/bin/python bench/train/train_player_meta_arm.py \
    --json /tmp/player_arm.json     # ~23 min (real-oracle memo warm-up dominates)
```

`results_player_on_board.md` left `PLAYER_SPEC` as the untrained
learned arm: the board exists, the `LinearPolicy` can play it, but
every `evaluate` probe is a nested train+eval (~40 s after
memoization, ~10 min cold), so 12 episodes × several probes blows
the box.  The board's own seam dissolves the problem — `score_fn`
(the case field `tests/test_discovery_player_arena.py` stubs with):
train under a *cheap distilled oracle*, verify the emitted child
under the *real* one.

## The training protocol

* **Stub oracle** — `distilled_open`, the schema leaderboard's
  measured three tiers over part subsets: no `open:action` → 0.0989
  (blind-schema plateau); `action` present, `onehot` absent →
  0.2827 (the winner tier — every such subset tied there); else
  0.2301 (the incumbent tier).  32 subsets → 3 published values; it
  shapes *which records the arm builds and probes*, not what the
  emitted child really scores — only the real oracle measures that.
* **Imitation warm start** — 6 epochs of `engine.imitate_policy`
  replaying `GreedyEditor` on the stub board: **864/864**
  demonstrated moves cloned.  Needed: cold REINFORCE alone stalls —
  the paying line is ~5 moves deep in a ~40-action legal set and
  meandering episodes dominate the EMA-baseline gradient (at 30 ep
  the cold arm plateaued at −0.035 with `a:dir:drop` *negative*).
* **REINFORCE** — 150 epochs × 12 stub cases, budget 16, lr 0.1.
  Block means `−0.067 → −0.050`; the trace converges to the
  probe-edit loop `evaluate → toggle → evaluate → … → extract`.
* **Stub iteration mattered.**  A first stub (part-count penalty
  only: `0.10 + 0.18·[action] − 0.012·len(parts)`) trained an arm
  that dropped `open:action` *before* `open:onehot` — under a
  flat-within-`action` landscape both read as "extra part" — hit
  the −0.13 blind cliff, and never re-added.  The tier stub is the
  load-bearing shape: `action` must be kept, `onehot` must go.

## Held-out — the real canonical nested eval

Frozen arms on fresh `PlayerArena` episodes, **no** `score_fn`:
3 epochs × 36 even pages train, odd-page held-out, the memoized
per-candidate verdicts — the same measurement `GreedyEditor`'s
0.2827 was taken under.

| episode | decode | moves | evals | emitted child | score | Δ |
|---|---|---|---|---|---|---|
| real0 | sampled | 16 | 8 | parent (5 parts) | 0.2301 | 0.0 |
| **real0** | **greedy** | 12 | 6 | **`['open:action']`** | **0.2827** | **+0.0526** |
| real1 | sampled | 8 | 4 | `['open:hist','open:action']` | 0.3063 | +0.0060 |
| **real1** | **greedy** | 12 | 6 | **`['open:action']`** | **0.3063** | **+0.0060** |

The learned meta-arm **emits a ≥-as-good child**: on seed 0 the
greedy decode emits literally `GreedyEditor`'s child —
`['open:action']` at real 0.2827 vs parent 0.2301 (+0.0526), in
12 moves and 6 of 8 probes.  The emitted-child *score* is the
real nested eval, not the stub's.  The sampled decode emits
positive-or-zero deltas — never a *worse* child.

**Baked:** `lawdata.PLAYER_ARENA_PLAYER_WEIGHTS` (27 nonzero
weights, rounded to 4 dp), wired as `PLAYER_SPEC.weights` — the
`LearnedPlayer` over spec records now defaults to the trained
prior.  `weights_dict()` → `PLAYER_SPEC.to_record()` →
`spec_from_record` → `weights_dict()` round-trip verified (the
record carries the table by name, like the other baked specs).

## What it learned, honestly

* **The line is legible**: `op:evaluate` +5.22, `a:k` +2.74 —
  probe the record after every edit; `a:dir:drop` +2.54 with
  `a:part:open:action` −0.68 the *least* dropped part and
  `a:part:open:onehot`/`a:part:bias` ordered around it;
  `op:extract` +1.49 closes once `st:best_delta`/`st:probes` say
  a winner exists.  `op:edit_weight`/`set_lr`/`set_weights` all
  negative — the weight/lr axis pays nothing on this board.
* **Sampled is a coin flip**: 2 of 4 sampled episodes emit the
  winner, the rest emit the parent (Δ0 — the `extract` on a board
  whose best-probed is the incumbent is honest, never negative).
  The baked prior's *greedy* decode is the earning arm.
* **Sequencing, not discovery**: the stub's tiers are the
  leaderboard's own published verdicts; the arm learned the
  *mechanics* (which part, when to probe, when to cash out) that
  transfer to the real board.  It cannot mint new parts — the
  edit vocabulary is still the shipped tags.
* **Over-training collapses**: at ~400 epochs the arm decays to
  probe-once-extract-early (Δ0 episodes) — the 150-epoch table is
  the baked one.
* **Wall time**: ~23 min end-to-end, almost all of it the real
  oracle's one-time memo warm-up (first probe ~10 min, subsequent
  probes tens of seconds; the greedy episodes cost 4–6 probes
  each).  The stub-training half is seconds.

## Where it lives

* `bench/train/train_player_meta_arm.py` — the driver: stub oracle,
  `imitate_policy` warm start, REINFORCE loop, real-oracle held-out
  phase (`--skip-real` for the curve only).
* `mlopt_discovery/lawdata/weights.py` —
  `PLAYER_ARENA_PLAYER_WEIGHTS`; `player_arena.PLAYER_SPEC` binds
  it (`weights={}` still reaches the cold start).
* `docs/results_player_on_board.md` — the board's design doc;
  `docs/results_schema_leaderboard.md` — the measured landscape
  the stub distills.
