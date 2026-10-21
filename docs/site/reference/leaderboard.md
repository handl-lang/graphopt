<!-- provenance: graphopt:docs/leaderboard_players.md@eea02f6 — generated upstream; do not hand-edit -->

# Player leaderboard

Every registered `players/` module on every board it can play, under one paired protocol: identical held-out cases, identical memoized oracle verdicts on `open` (per-candidate seeds replayed to every arm — see `bench/drivers/policy/policy_baseline.py`), identical contraction instances scored by one cost model.  **ratio** is the contraction board's cost/baseline figure; the *notes* column carries the qualitative flags (drain artifacts, premature stops, unsupported boards) that the numbers alone hide.

- date: 2026-10-20T00:00:14+00:00 · git: `2d6e5e8` (dirty) · torch 2.14.0+cu130 on cpu · seed 0
- players: `all-registered`
- meta_cases: `16`
- meta_budget: `12`
- meta_held_out: `gen_cases(seed+7919)`
- open_pages: `36`
- open_page: `16`
- open_budget: `17`
- open_split: `odd pages eval / even pages train (neural)`
- oracle_pairing: `memoized per-candidate-seeded verdicts`
- neural_epochs: `300`
- neural_hidden: `32`
- contraction_scales: `[8, 12]`
- contraction_instances: `4`
- contraction_samples: `32`
- meta: `{'cases': 16, 'budget': 12}`
- open: `{'pool': 576, 'useful': 7, 'random_ap': 0.0122, 'train_pairs': 576}`
- contraction: `{'scales': [8, 12], 'instances': 4}`

## `meta`

| player | kind | reward/ep | evals/ep | useful/eval | rank AP | cost | ratio | wall s | notes |
|---|---|---|---|---|---|---|---|---|---|
| `greedy` | scripted | -0.6000 | 12.00 | — | — | — | — | 0.12 | drains every episode's move budget |
| `linear` | linear | 0.1166 | 1.00 | — | — | 2.50 | — | 0.05 | short lines — extracts early; certified extraction on 16/16 |
| `linear-cold` | linear | -0.4188 | 10.25 | — | — | 2.43 | — | 0.31 | certified extraction on 7/16 |
| `linear-greedy` | linear | 0.1166 | 1.00 | — | — | 2.50 | — | 0.05 | short lines — extracts early; certified extraction on 16/16 |
| `playbook` | scripted | -0.1000 | 2.00 | — | — | 2.94 | — | 0.03 | short lines — extracts early; certified extraction on 16/16 |
| `random` | chance | -0.5324 | 11.12 | — | — | 3.00 | — | 0.10 | certified extraction on 4/16 |

## `open`

| player | kind | reward/ep | evals/ep | useful/eval | rank AP | cost | ratio | wall s | notes |
|---|---|---|---|---|---|---|---|---|---|
| `linear` | linear | 0.2852 | 13.72 | 0.0121 | 0.1284 | — | — | 1.14 | — |
| `linear-cold` | linear | 0.1699 | 9.31 | 0.0119 | 0.0123 | — | — | 1.14 | — |
| `linear-greedy` | linear | 0.3590 | 16.00 | 0.0104 | 0.1284 | — | — | 1.25 | drains every page — no stopping learned |
| `neural` | neural | 0.3089 | 16.00 | 0.0104 | 0.0412 | — | — | 1.33 | drains every page — no stopping learned |
| `playbook` | scripted | 0.3595 | 16.00 | 0.0104 | 0.0450 | — | — | 1.26 | drains every page — no stopping learned |
| `random` | chance | 0.1339 | 8.67 | 0.0096 | 0.0122 | — | — | 0.97 | — |

## `contraction`

| player | kind | reward/ep | evals/ep | useful/eval | rank AP | cost | ratio | wall s | notes |
|---|---|---|---|---|---|---|---|---|---|
| `oe-greedy` | external | — | — | — | — | 28,164.00 | 1.000 | 0.00 | 0 beats / 8 ties / 0 loses vs oe-greedy |
| `oe-rand-greedy` | external | — | — | — | — | 27,591.80 | 0.885 | 0.00 | 4 beats / 3 ties / 1 loses vs oe-greedy |
| `distilled` | distilled | — | — | — | — | 27,775.50 | 0.930 | 0.03 | 5 beats / 1 ties / 2 loses vs oe-greedy |
| `distilled-argmax` | distilled | — | — | — | — | 28,708.80 | 1.125 | 0.00 | 3 beats / 0 ties / 5 loses vs oe-greedy |
| `greedy` | scripted | — | — | — | — | 86,419.50 | 5.199 | 0.00 | 0 beats / 0 ties / 8 loses vs oe-greedy |
| `random` | chance | — | — | — | — | 26,847,963.80 | 2872.670 | 0.00 | 0 beats / 0 ties / 8 loses vs oe-greedy |

