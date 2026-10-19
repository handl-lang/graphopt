# Player-on-board — the two-tier meta-game

**Date:** 2026-10-19 · **Script:** `python -m mlopt_discovery.player_arena` ·
**Seed:** 0 · **Runner:**

```bash
timeout 1800 .venv/bin/python -m mlopt_discovery.player_arena \
    --objective open --seed 0 --budget 20 --json /tmp/player_headline.json
    # ~11 min wall (one-time oracle memoization dominates; probes then ~40 s each)
```

## What the board is

`player_arena.PlayerArena` is a `Board` whose *state is a
`DomainSpec` record* — the object-level player spec as pure data.
The `PARTS`/`LEGALS`/`RANKS` registries (`players.py`) give every
callable in a spec a name, so `spec.to_record()` reifies to
`{features, legal, parts, weights, rank, lr}` and
`spec_from_record` rebuilds it. Moves rewrite the record:

| move | effect |
|---|---|
| `toggle_feature(part, add)` | drop/restore a feature part within the objective's vocab (requires-safe: `meta:action` needs `meta:state`) |
| `edit_weight(name, delta)` | nudge the weight *prior* the child trains from |
| `set_lr(lr)` | set the child's REINFORCE rate |
| `set_weights(table)` | install a named prior (`cold`/`baked`) |
| `evaluate(k)` | oracle probe: build the spec, train cold at the leaderboard's reduced budget, score held-out. Non-terminal, eval-budgeted |
| `extract()` | terminal: emit the best probed record, pay `delta·(child − parent)` |

The nested evaluation is the schema leaderboard's own protocol:
**open** = 3 epochs × 36 even pages, budget 17; held-out odd pages
under memoized per-candidate verdicts inside a forked torch RNG
(identical draws for parent and every child; the outer episode's
RNG untouched). **meta** = 15 epochs × 40 cases, budget 12; held-out
`gen_cases(seed+7919)`. The scripted meta-player `GreedyEditor` is
greedy sequential part ablation: probe each droppable part
leave-one-out, keep drops that cost ≤ `tol`, revert the rest,
extract the best.

## The headline episode (open objective)

Seed = the incumbent `OPEN_SPEC` record, cold prior — the same
record the leaderboard's incumbent row was measured under.

| step | move | reward | probe result |
|---|---|---|---|
| drop `bias` | toggle | −0.002 | — |
| `evaluate` | probe | −0.012 | 0.2301, Δ+0.0000 → **keep** (dead weight) |
| drop `open:state` | toggle | −0.002 | — |
| `evaluate` | probe | −0.012 | 0.2301, Δ+0.0000 → **keep** |
| drop `open:hist` | toggle | −0.002 | — |
| `evaluate` | probe | −0.012 | 0.2301, Δ+0.0000 → **keep** |
| drop `open:onehot` | toggle | −0.002 | — |
| `evaluate` | probe | −0.012 | **0.2827, Δ+0.0526** → keep |
| drop `open:action` | toggle | −0.002 | — |
| `evaluate` | probe | −0.012 | 0.0989, Δ−0.1313 → **revert** |
| re-add `open:action` | toggle | −0.002 | — |
| `extract` | terminal | +0.0506 | emit best probed record |

```text
parent: score=0.2301  parts=[bias, open:state, open:hist, open:onehot, open:action]
child:  score=0.2827  delta=+0.0526  parts=[open:action]
```

**A meta-episode emitted a player that beats its parent on held-out
object-level boards**: the child is `open:action` alone — the
`a:*`/`a:form:*` candidate-shape block, nothing else — and it wins
the sampled held-out reward 0.2827 vs 0.2301 (+0.0526), at the same
budget, under the same memoized verdicts.

## What it means, honestly

* **The numbers are the leaderboard's, reproduced.** 0.2301
  (incumbent), 0.2827 (`action`-alone winner), 0.0989 (blind
  schema) match `docs/results_schema_leaderboard.md` to 4 decimals —
  expected: same protocol (3 epochs, 36 even-page train / odd-page
  eval), paired seeded draws. The demo *re-derives* the winner
  through the game, rather than re-measuring it — the ablation
  trace is a policy, not a lookup.
* **The child is a different tradeoff, not a dominator.** The
  leaderboard also showed `onehot` buys greedy/AP: dropping it
  *gains* sampled reward but costs ranking quality (AP 0.0701 →
  0.0277). The board scored sampled reward only — a richer score
  (AP, greedy) is a cfg change, not a board change.
* **Bounded regress.** The meta-board's own spec (`PLAYER_SPEC` —
  the featurizer over spec records) stays fixed: this board
  rewrites the object-level player, not its own rules. A third tier
  (rewriting the meta-featurizer) would need a spec-of-specs;
  nothing here claims it. The edit vocabulary is likewise fixed to
  the shipped part tags — a meta-player cannot mint *new* parts,
  only reconfigure shipped ones.
* **Eval-budget is real.** `evaluate` probes are capped
  (`max_evals`, default 8) and priced (−0.012 vs −0.002 edits): the
  oracle call is the episode's scarce resource, and `GreedyEditor`
  spends 5 of 8 on the winning run.
* **Learned meta-arm exists but is untrained.** `PLAYER_SPEC` +
  `legal_actions` give the generic `LinearPolicy` a board to learn
  on (`play --domain player`), but each probe is seconds-to-minutes
  of nested train+eval — a trained meta-player is future work, not
  claimed here. The shipped demo is scripted.
  **Update (same-day):** trained via the board's `score_fn` stub
  seam — the greedy decode now emits the `['open:action']` child
  (0.2827) and the prior is baked as
  `lawdata.PLAYER_ARENA_PLAYER_WEIGHTS`; see
  `docs/results_player_meta_arm.md`.

## Where it lives

* `mlopt_discovery/player_arena.py` — `PlayerArena`, `PlayerState`,
  `legal_actions`, `OBJECTIVES` (`meta`/`open`), `GreedyEditor`,
  `PLAYER_SPEC`, `gen_cases`, CLI.
* `mlopt_discovery/players.py` — `PARTS`/`LEGALS`/`RANKS`
  registries, `DomainSpec.to_record`/`build_from_record`,
  `spec_from_record`; shipped parts register in `meta_player`,
  `open_game`, `play` (`search:*`).
* `play.DOMAINS["player"]` — `python -m mlopt_discovery.play
  --domain player` works (nested evals make episodes minutes-scale;
  use small `--train-cases`/`--eval-cases`).
* `tests/test_discovery_player_arena.py` — 24 tests: record
  round-trips (all shipped specs), board mechanics and reward
  composition via a stub `score_fn`, `GreedyEditor`'s
  keep/revert/extract loop, domain registration.
