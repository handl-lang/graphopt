# Eval matrix — players × boards × referees

Each cell answers: *the claims this player produces on this board — which survive this referee?*  Episode boards yield extraction claims (input ≡ delivered); `open` yields the candidates the player chose.  **admit** = claims the referee cleared; **value** = the mean `ScoringSpec` blend over measured claims (fraction of the launch bar — lower is better); **refuted** = picks the board's oracle called `useful` that the referee refuses — the strictness ablation as a row.

- date: 2026-10-21T01:04:18+00:00 · git: `adef366` (dirty) · device cpu · seed 0
- meta_cases: `8`
- meta_budget: `10`
- meta_held_out: `gen_cases(seed+7919)`
- open_pages: `4`
- open_page: `8`
- open_budget: `9`
- open_split: `odd pages eval / even pages train (neural)`
- neural_epochs: `300`
- neural_hidden: `32`
- oracle_pairing: `memoized per-candidate-seeded verdicts`

## `meta`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|
| `random` | `naive` | 2 | 1.00 | 0 | — | -0.3545 | 0.07 | — |
| `random` | `oracle` | 2 | 0.00 | 0 | — | -0.3545 | 0.10 | — |
| `random` | `pay` | 2 | 0.00 | 0 | — | -0.3545 | 0.07 | — |
| `random` | `sweep` | 2 | 0.00 | 0 | — | -0.3545 | 0.07 | — |
| `random` | `full` | 2 | 0.00 | 0 | — | -0.3545 | 0.07 | — |
| `playbook` | `naive` | 0 | — | 0 | — | -0.1000 | 0.02 | no claims — the player changed nothing |
| `playbook` | `oracle` | 0 | — | 0 | — | -0.1000 | 0.02 | no claims — the player changed nothing |
| `playbook` | `pay` | 0 | — | 0 | — | -0.1000 | 0.02 | no claims — the player changed nothing |
| `playbook` | `sweep` | 0 | — | 0 | — | -0.1000 | 0.02 | no claims — the player changed nothing |
| `playbook` | `full` | 0 | — | 0 | — | -0.1000 | 0.02 | no claims — the player changed nothing |
| `linear` | `naive` | 4 | 1.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear` | `oracle` | 4 | 0.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear` | `pay` | 4 | 0.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear` | `sweep` | 4 | 0.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear` | `full` | 4 | 0.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear-cold` | `naive` | 2 | 1.00 | 0 | — | -0.3564 | 0.14 | — |
| `linear-cold` | `oracle` | 2 | 0.00 | 0 | — | -0.3564 | 0.14 | — |
| `linear-cold` | `pay` | 2 | 0.00 | 0 | — | -0.3564 | 0.14 | — |
| `linear-cold` | `sweep` | 2 | 0.00 | 0 | — | -0.3564 | 0.14 | — |
| `linear-cold` | `full` | 2 | 0.00 | 0 | — | -0.3564 | 0.14 | — |
| `linear-greedy` | `naive` | 3 | 1.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear-greedy` | `oracle` | 3 | 0.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear-greedy` | `pay` | 3 | 0.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear-greedy` | `sweep` | 3 | 0.00 | 0 | — | 0.1896 | 0.03 | — |
| `linear-greedy` | `full` | 3 | 0.00 | 0 | — | 0.1896 | 0.03 | — |
| `greedy` | `naive` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |
| `greedy` | `oracle` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |
| `greedy` | `pay` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |
| `greedy` | `sweep` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |
| `greedy` | `full` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |

**Pareto** (admit-rate ↑, reward ↑, wall ↓): `linear`/`naive`, `linear-greedy`/`naive`

## `joint`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|
| `random` | `naive` | 3 | 1.00 | 0 | — | -0.4639 | 0.24 | — |
| `random` | `oracle` | 3 | 0.00 | 0 | — | -0.4639 | 0.24 | — |
| `random` | `pay` | 3 | 0.00 | 0 | — | -0.4639 | 0.24 | — |
| `random` | `sweep` | 3 | 0.00 | 0 | — | -0.4639 | 0.24 | — |
| `random` | `full` | 3 | 0.00 | 0 | — | -0.4639 | 0.24 | — |
| `playbook` | `naive` | 5 | 1.00 | 0 | — | -0.1041 | 0.08 | — |
| `playbook` | `oracle` | 5 | 0.00 | 0 | — | -0.1041 | 0.09 | — |
| `playbook` | `pay` | 5 | 0.00 | 0 | — | -0.1041 | 0.08 | — |
| `playbook` | `sweep` | 5 | 0.00 | 0 | — | -0.1041 | 0.08 | — |
| `playbook` | `full` | 5 | 0.00 | 0 | — | -0.1041 | 0.08 | — |
| `linear` | `naive` | 3 | 1.00 | 0 | — | -0.0564 | 0.21 | — |
| `linear` | `oracle` | 3 | 0.00 | 0 | — | -0.0564 | 0.21 | — |
| `linear` | `pay` | 3 | 0.00 | 0 | — | -0.0564 | 0.21 | — |
| `linear` | `sweep` | 3 | 0.00 | 0 | — | -0.0564 | 0.21 | — |
| `linear` | `full` | 3 | 0.00 | 0 | — | -0.0564 | 0.21 | — |
| `linear-cold` | `naive` | 4 | 1.00 | 0 | — | -0.4456 | 1.50 | — |
| `linear-cold` | `oracle` | 4 | 0.00 | 0 | — | -0.4456 | 1.50 | — |
| `linear-cold` | `pay` | 4 | 0.00 | 0 | — | -0.4456 | 1.50 | — |
| `linear-cold` | `sweep` | 4 | 0.00 | 0 | — | -0.4456 | 1.50 | — |
| `linear-cold` | `full` | 4 | 0.00 | 0 | — | -0.4456 | 1.50 | — |
| `linear-greedy` | `naive` | 2 | 1.00 | 0 | — | -0.0500 | 0.18 | — |
| `linear-greedy` | `oracle` | 2 | 0.00 | 0 | — | -0.0500 | 0.18 | — |
| `linear-greedy` | `pay` | 2 | 0.00 | 0 | — | -0.0500 | 0.18 | — |
| `linear-greedy` | `sweep` | 2 | 0.00 | 0 | — | -0.0500 | 0.18 | — |
| `linear-greedy` | `full` | 2 | 0.00 | 0 | — | -0.0500 | 0.18 | — |
| `greedy` | `naive` | 0 | — | 0 | — | -0.5000 | 0.31 | no claims — the player changed nothing |
| `greedy` | `oracle` | 0 | — | 0 | — | -0.5000 | 0.31 | no claims — the player changed nothing |
| `greedy` | `pay` | 0 | — | 0 | — | -0.5000 | 0.31 | no claims — the player changed nothing |
| `greedy` | `sweep` | 0 | — | 0 | — | -0.5000 | 0.31 | no claims — the player changed nothing |
| `greedy` | `full` | 0 | — | 0 | — | -0.5000 | 0.31 | no claims — the player changed nothing |

**Pareto** (admit-rate ↑, reward ↑, wall ↓): `playbook`/`naive`, `linear-greedy`/`naive`

## `torch`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|
| `random` | `naive` | 0 | — | 0 | — | -0.5000 | 0.05 | no claims — the player changed nothing |
| `random` | `oracle` | 0 | — | 0 | — | -0.5000 | 0.05 | no claims — the player changed nothing |
| `random` | `pay` | 0 | — | 0 | — | -0.5000 | 0.05 | no claims — the player changed nothing |
| `random` | `sweep` | 0 | — | 0 | — | -0.5000 | 0.05 | no claims — the player changed nothing |
| `random` | `full` | 0 | — | 0 | — | -0.5000 | 0.05 | no claims — the player changed nothing |
| `playbook` | `naive` | 2 | 1.00 | 0 | — | 0.1495 | 0.02 | — |
| `playbook` | `oracle` | 2 | 1.00 | 0 | — | 0.1495 | 0.02 | — |
| `playbook` | `pay` | 2 | 1.00 | 0 | 0.924 | 0.1495 | 0.02 | — |
| `playbook` | `sweep` | 2 | 1.00 | 0 | 0.924 | 0.1495 | 0.03 | — |
| `playbook` | `full` | 2 | 1.00 | 0 | 0.924 | 0.1495 | 0.03 | — |
| `linear` | `naive` | 1 | 1.00 | 0 | — | 0.0125 | 0.02 | — |
| `linear` | `oracle` | 1 | 0.00 | 0 | — | 0.0125 | 0.02 | — |
| `linear` | `pay` | 1 | 0.00 | 0 | — | 0.0125 | 0.02 | — |
| `linear` | `sweep` | 1 | 0.00 | 0 | — | 0.0125 | 0.02 | — |
| `linear` | `full` | 1 | 0.00 | 0 | — | 0.0125 | 0.02 | — |
| `linear-cold` | `naive` | 1 | 1.00 | 0 | — | -0.4439 | 0.15 | — |
| `linear-cold` | `oracle` | 1 | 1.00 | 0 | — | -0.4439 | 0.15 | — |
| `linear-cold` | `pay` | 1 | 1.00 | 0 | 0.868 | -0.4439 | 0.15 | — |
| `linear-cold` | `sweep` | 1 | 1.00 | 0 | 0.868 | -0.4439 | 0.15 | — |
| `linear-cold` | `full` | 1 | 1.00 | 0 | 0.868 | -0.4439 | 0.16 | — |
| `linear-greedy` | `naive` | 1 | 1.00 | 0 | — | 0.0125 | 0.03 | — |
| `linear-greedy` | `oracle` | 1 | 0.00 | 0 | — | 0.0125 | 0.03 | — |
| `linear-greedy` | `pay` | 1 | 0.00 | 0 | — | 0.0125 | 0.03 | — |
| `linear-greedy` | `sweep` | 1 | 0.00 | 0 | — | 0.0125 | 0.03 | — |
| `linear-greedy` | `full` | 1 | 0.00 | 0 | — | 0.0125 | 0.03 | — |
| `greedy` | `naive` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |
| `greedy` | `oracle` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |
| `greedy` | `pay` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |
| `greedy` | `sweep` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |
| `greedy` | `full` | 0 | — | 0 | — | -0.5000 | 0.06 | no claims — the player changed nothing |

**Pareto** (admit-rate ↑, reward ↑, wall ↓): `playbook`/`naive`, `playbook`/`oracle`, `playbook`/`pay`

## `search`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|
| `random` | `naive` | 0 | — | 0 | — | 0.0000 | 0.02 | no claims — the player changed nothing |
| `random` | `oracle` | 0 | — | 0 | — | 0.0000 | 0.02 | no claims — the player changed nothing |
| `random` | `pay` | 0 | — | 0 | — | 0.0000 | 0.02 | no claims — the player changed nothing |
| `random` | `sweep` | 0 | — | 0 | — | 0.0000 | 0.02 | no claims — the player changed nothing |
| `random` | `full` | 0 | — | 0 | — | 0.0000 | 0.02 | no claims — the player changed nothing |
| `playbook` | `naive` | 3 | 1.00 | 0 | — | 0.5000 | 0.02 | — |
| `playbook` | `oracle` | 3 | 1.00 | 0 | — | 0.5000 | 0.02 | — |
| `playbook` | `pay` | 3 | 1.00 | 0 | 0.384 | 0.5000 | 0.02 | — |
| `playbook` | `sweep` | 3 | 1.00 | 0 | 0.384 | 0.5000 | 0.03 | — |
| `playbook` | `full` | 3 | 1.00 | 0 | 0.384 | 0.5000 | 0.89 | — |
| `linear` | `naive` | 1 | 1.00 | 0 | — | 0.0625 | 0.03 | — |
| `linear` | `oracle` | 1 | 1.00 | 0 | — | 0.0625 | 0.03 | — |
| `linear` | `pay` | 1 | 1.00 | 0 | 0.980 | 0.0625 | 0.03 | — |
| `linear` | `sweep` | 1 | 1.00 | 0 | 0.980 | 0.0625 | 0.03 | — |
| `linear` | `full` | 1 | 1.00 | 0 | 0.980 | 0.0625 | 0.04 | — |
| `linear-cold` | `naive` | 1 | 1.00 | 0 | — | 0.0625 | 0.03 | — |
| `linear-cold` | `oracle` | 1 | 1.00 | 0 | — | 0.0625 | 0.03 | — |
| `linear-cold` | `pay` | 1 | 1.00 | 0 | 0.980 | 0.0625 | 0.03 | — |
| `linear-cold` | `sweep` | 1 | 1.00 | 0 | 0.980 | 0.0625 | 0.03 | — |
| `linear-cold` | `full` | 1 | 1.00 | 0 | 0.980 | 0.0625 | 0.03 | — |
| `linear-greedy` | `naive` | 0 | — | 0 | — | 0.0000 | 0.03 | no claims — the player changed nothing |
| `linear-greedy` | `oracle` | 0 | — | 0 | — | 0.0000 | 0.03 | no claims — the player changed nothing |
| `linear-greedy` | `pay` | 0 | — | 0 | — | 0.0000 | 0.03 | no claims — the player changed nothing |
| `linear-greedy` | `sweep` | 0 | — | 0 | — | 0.0000 | 0.03 | no claims — the player changed nothing |
| `linear-greedy` | `full` | 0 | — | 0 | — | 0.0000 | 0.03 | no claims — the player changed nothing |

**Pareto** (admit-rate ↑, reward ↑, wall ↓): `playbook`/`naive`, `playbook`/`oracle`, `playbook`/`pay`

## `gen`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|
| `random` | `naive` | 2 | 1.00 | 0 | — | -0.3877 | 0.16 | — |
| `random` | `oracle` | 2 | 0.50 | 0 | — | -0.3877 | 0.28 | — |
| `random` | `pay` | 2 | 0.50 | 0 | 0.337 | -0.3877 | 0.26 | — |
| `random` | `sweep` | 2 | 0.50 | 0 | 0.337 | -0.3877 | 1.00 | — |
| `random` | `full` | 2 | 0.00 | 0 | 0.337 | -0.3877 | 1.05 | — |
| `playbook` | `naive` | 3 | 1.00 | 0 | — | 0.0405 | 0.08 | — |
| `playbook` | `oracle` | 3 | 1.00 | 0 | — | 0.0405 | 0.25 | — |
| `playbook` | `pay` | 3 | 1.00 | 0 | 0.558 | 0.0405 | 0.24 | — |
| `playbook` | `sweep` | 3 | 1.00 | 0 | 0.558 | 0.0405 | 1.12 | — |
| `playbook` | `full` | 3 | 0.67 | 0 | 0.558 | 0.0405 | 1.00 | — |
| `linear` | `naive` | 2 | 1.00 | 0 | — | 0.0033 | 0.28 | — |
| `linear` | `oracle` | 2 | 0.00 | 0 | — | 0.0033 | 0.36 | — |
| `linear` | `pay` | 2 | 0.00 | 0 | — | 0.0033 | 0.39 | — |
| `linear` | `sweep` | 2 | 0.00 | 0 | — | 0.0033 | 0.36 | — |
| `linear` | `full` | 2 | 0.00 | 0 | — | 0.0033 | 0.35 | — |
| `linear-cold` | `naive` | 3 | 1.00 | 0 | — | -0.4218 | 2.22 | — |
| `linear-cold` | `oracle` | 3 | 0.33 | 0 | — | -0.4218 | 2.34 | — |
| `linear-cold` | `pay` | 3 | 0.33 | 0 | 0.337 | -0.4218 | 2.52 | — |
| `linear-cold` | `sweep` | 3 | 0.33 | 0 | 0.337 | -0.4218 | 3.23 | — |
| `linear-cold` | `full` | 3 | 0.00 | 0 | 0.337 | -0.4218 | 3.32 | — |
| `linear-greedy` | `naive` | 2 | 1.00 | 0 | — | 0.0095 | 0.28 | — |
| `linear-greedy` | `oracle` | 2 | 0.00 | 0 | — | 0.0095 | 0.36 | — |
| `linear-greedy` | `pay` | 2 | 0.00 | 0 | — | 0.0095 | 0.35 | — |
| `linear-greedy` | `sweep` | 2 | 0.00 | 0 | — | 0.0095 | 0.36 | — |
| `linear-greedy` | `full` | 2 | 0.00 | 0 | — | 0.0095 | 0.39 | — |
| `greedy` | `naive` | 0 | — | 0 | — | -0.5000 | 0.21 | no claims — the player changed nothing |
| `greedy` | `oracle` | 0 | — | 0 | — | -0.5000 | 0.21 | no claims — the player changed nothing |
| `greedy` | `pay` | 0 | — | 0 | — | -0.5000 | 0.21 | no claims — the player changed nothing |
| `greedy` | `sweep` | 0 | — | 0 | — | -0.5000 | 0.21 | no claims — the player changed nothing |
| `greedy` | `full` | 0 | — | 0 | — | -0.5000 | 0.21 | no claims — the player changed nothing |

**Pareto** (admit-rate ↑, reward ↑, wall ↓): `playbook`/`naive`

## `kernel`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|
| `random` | `naive` | 3 | 1.00 | 0 | — | -0.0500 | 0.02 | — |
| `random` | `oracle` | 3 | 0.00 | 0 | — | -0.0500 | 0.02 | — |
| `random` | `pay` | 3 | 0.00 | 0 | — | -0.0500 | 0.02 | — |
| `random` | `sweep` | 3 | 0.00 | 0 | — | -0.0500 | 0.02 | — |
| `random` | `full` | 3 | 0.00 | 0 | — | -0.0500 | 0.02 | — |
| `playbook` | `naive` | 8 | 1.00 | 0 | — | 0.4153 | 0.02 | — |
| `playbook` | `oracle` | 8 | 0.00 | 0 | — | 0.4153 | 0.02 | — |
| `playbook` | `pay` | 8 | 0.00 | 0 | — | 0.4153 | 0.02 | — |
| `playbook` | `sweep` | 8 | 0.00 | 0 | — | 0.4153 | 0.02 | — |
| `playbook` | `full` | 8 | 0.00 | 0 | — | 0.4153 | 0.02 | — |
| `linear` | `naive` | 7 | 1.00 | 0 | — | 0.6211 | 0.02 | — |
| `linear` | `oracle` | 7 | 0.00 | 0 | — | 0.6211 | 0.02 | — |
| `linear` | `pay` | 7 | 0.00 | 0 | — | 0.6211 | 0.02 | — |
| `linear` | `sweep` | 7 | 0.00 | 0 | — | 0.6211 | 0.02 | — |
| `linear` | `full` | 7 | 0.00 | 0 | — | 0.6211 | 0.02 | — |
| `linear-cold` | `naive` | 4 | 1.00 | 0 | — | 0.0844 | 0.03 | — |
| `linear-cold` | `oracle` | 4 | 0.00 | 0 | — | 0.0844 | 0.03 | — |
| `linear-cold` | `pay` | 4 | 0.00 | 0 | — | 0.0844 | 0.03 | — |
| `linear-cold` | `sweep` | 4 | 0.00 | 0 | — | 0.0844 | 0.03 | — |
| `linear-cold` | `full` | 4 | 0.00 | 0 | — | 0.0844 | 0.03 | — |
| `linear-greedy` | `naive` | 8 | 1.00 | 0 | — | 0.7694 | 0.02 | — |
| `linear-greedy` | `oracle` | 8 | 0.00 | 0 | — | 0.7694 | 0.02 | — |
| `linear-greedy` | `pay` | 8 | 0.00 | 0 | — | 0.7694 | 0.02 | — |
| `linear-greedy` | `sweep` | 8 | 0.00 | 0 | — | 0.7694 | 0.02 | — |
| `linear-greedy` | `full` | 8 | 0.00 | 0 | — | 0.7694 | 0.02 | — |

**Pareto** (admit-rate ↑, reward ↑, wall ↓): `linear-greedy`/`naive`

## `open`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|
| `random` | `naive` | 21 | 1.00 | 0 | — | 1.0812 | 0.01 | — |
| `random` | `oracle` | 21 | 0.19 | 0 | — | 1.0812 | 0.33 | — |
| `random` | `pay` | 21 | 0.14 | 0 | 0.741 | 1.0812 | 0.06 | — |
| `random` | `sweep` | 21 | 0.14 | 0 | 0.741 | 1.0812 | 0.65 | — |
| `random` | `full` | 21 | 0.00 | 3 | 0.741 | 1.0812 | 0.03 | 3 pick(s) the oracle called useful refused by full |
| `playbook` | `naive` | 32 | 1.00 | 0 | — | 1.3840 | 0.02 | — |
| `playbook` | `oracle` | 32 | 0.22 | 0 | — | 1.3840 | 0.50 | — |
| `playbook` | `pay` | 32 | 0.16 | 0 | 0.444 | 1.3840 | 0.09 | — |
| `playbook` | `sweep` | 32 | 0.16 | 0 | 0.444 | 1.3840 | 0.71 | — |
| `playbook` | `full` | 32 | 0.06 | 3 | 0.444 | 1.3840 | 0.77 | 3 pick(s) the oracle called useful refused by full |
| `linear` | `naive` | 32 | 1.00 | 0 | — | 1.2152 | 0.02 | — |
| `linear` | `oracle` | 32 | 0.22 | 0 | — | 1.2152 | 0.40 | — |
| `linear` | `pay` | 32 | 0.16 | 0 | 0.444 | 1.2152 | 0.08 | — |
| `linear` | `sweep` | 32 | 0.16 | 0 | 0.444 | 1.2152 | 0.57 | — |
| `linear` | `full` | 32 | 0.06 | 3 | 0.444 | 1.2152 | 0.88 | 3 pick(s) the oracle called useful refused by full |
| `linear-cold` | `naive` | 17 | 1.00 | 0 | — | 0.0415 | 0.02 | — |
| `linear-cold` | `oracle` | 17 | 0.06 | 0 | — | 0.0415 | 0.16 | — |
| `linear-cold` | `pay` | 17 | 0.06 | 0 | 0.000 | 0.0415 | 0.02 | — |
| `linear-cold` | `sweep` | 17 | 0.06 | 0 | 0.000 | 0.0415 | 0.02 | — |
| `linear-cold` | `full` | 17 | 0.06 | 0 | 0.000 | 0.0415 | 0.75 | — |
| `linear-greedy` | `naive` | 32 | 1.00 | 0 | — | 1.5090 | 0.02 | — |
| `linear-greedy` | `oracle` | 32 | 0.22 | 0 | — | 1.5090 | 0.39 | — |
| `linear-greedy` | `pay` | 32 | 0.16 | 0 | 0.444 | 1.5090 | 0.09 | — |
| `linear-greedy` | `sweep` | 32 | 0.16 | 0 | 0.444 | 1.5090 | 0.62 | — |
| `linear-greedy` | `full` | 32 | 0.06 | 3 | 0.444 | 1.5090 | 0.86 | 3 pick(s) the oracle called useful refused by full |
| `neural` | `naive` | 32 | 1.00 | 0 | — | 0.9882 | 0.32 | — |
| `neural` | `oracle` | 32 | 0.22 | 0 | — | 0.9882 | 0.87 | — |
| `neural` | `pay` | 32 | 0.16 | 0 | 0.444 | 0.9882 | 0.39 | — |
| `neural` | `sweep` | 32 | 0.16 | 0 | 0.444 | 0.9882 | 0.91 | — |
| `neural` | `full` | 32 | 0.06 | 3 | 0.444 | 0.9882 | 1.08 | 3 pick(s) the oracle called useful refused by full |

**Pareto** (admit-rate ↑, reward ↑, wall ↓): `random`/`naive`, `linear-greedy`/`naive`

## `contraction`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|

## `player`

| player | referee | claims | admit | refuted | value | reward/ep | wall s | notes |
|---|---|---|---|---|---|---|---|---|

## Coverage — the unrun cells

| board | player | referee | reason |
|---|---|---|---|
| `meta` | `neural` | `naive` | player does not play this board |
| `meta` | `neural` | `oracle` | player does not play this board |
| `meta` | `neural` | `pay` | player does not play this board |
| `meta` | `neural` | `sweep` | player does not play this board |
| `meta` | `neural` | `full` | player does not play this board |
| `meta` | `distilled` | `naive` | player does not play this board |
| `meta` | `distilled` | `oracle` | player does not play this board |
| `meta` | `distilled` | `pay` | player does not play this board |
| `meta` | `distilled` | `sweep` | player does not play this board |
| `meta` | `distilled` | `full` | player does not play this board |
| `meta` | `distilled-argmax` | `naive` | player does not play this board |
| `meta` | `distilled-argmax` | `oracle` | player does not play this board |
| `meta` | `distilled-argmax` | `pay` | player does not play this board |
| `meta` | `distilled-argmax` | `sweep` | player does not play this board |
| `meta` | `distilled-argmax` | `full` | player does not play this board |
| `joint` | `neural` | `naive` | player does not play this board |
| `joint` | `neural` | `oracle` | player does not play this board |
| `joint` | `neural` | `pay` | player does not play this board |
| `joint` | `neural` | `sweep` | player does not play this board |
| `joint` | `neural` | `full` | player does not play this board |
| `joint` | `distilled` | `naive` | player does not play this board |
| `joint` | `distilled` | `oracle` | player does not play this board |
| `joint` | `distilled` | `pay` | player does not play this board |
| `joint` | `distilled` | `sweep` | player does not play this board |
| `joint` | `distilled` | `full` | player does not play this board |
| `joint` | `distilled-argmax` | `naive` | player does not play this board |
| `joint` | `distilled-argmax` | `oracle` | player does not play this board |
| `joint` | `distilled-argmax` | `pay` | player does not play this board |
| `joint` | `distilled-argmax` | `sweep` | player does not play this board |
| `joint` | `distilled-argmax` | `full` | player does not play this board |
| `torch` | `neural` | `naive` | player does not play this board |
| `torch` | `neural` | `oracle` | player does not play this board |
| `torch` | `neural` | `pay` | player does not play this board |
| `torch` | `neural` | `sweep` | player does not play this board |
| `torch` | `neural` | `full` | player does not play this board |
| `torch` | `distilled` | `naive` | player does not play this board |
| `torch` | `distilled` | `oracle` | player does not play this board |
| `torch` | `distilled` | `pay` | player does not play this board |
| `torch` | `distilled` | `sweep` | player does not play this board |
| `torch` | `distilled` | `full` | player does not play this board |
| `torch` | `distilled-argmax` | `naive` | player does not play this board |
| `torch` | `distilled-argmax` | `oracle` | player does not play this board |
| `torch` | `distilled-argmax` | `pay` | player does not play this board |
| `torch` | `distilled-argmax` | `sweep` | player does not play this board |
| `torch` | `distilled-argmax` | `full` | player does not play this board |
| `search` | `neural` | `naive` | player does not play this board |
| `search` | `neural` | `oracle` | player does not play this board |
| `search` | `neural` | `pay` | player does not play this board |
| `search` | `neural` | `sweep` | player does not play this board |
| `search` | `neural` | `full` | player does not play this board |
| `search` | `distilled` | `naive` | player does not play this board |
| `search` | `distilled` | `oracle` | player does not play this board |
| `search` | `distilled` | `pay` | player does not play this board |
| `search` | `distilled` | `sweep` | player does not play this board |
| `search` | `distilled` | `full` | player does not play this board |
| `search` | `distilled-argmax` | `naive` | player does not play this board |
| `search` | `distilled-argmax` | `oracle` | player does not play this board |
| `search` | `distilled-argmax` | `pay` | player does not play this board |
| `search` | `distilled-argmax` | `sweep` | player does not play this board |
| `search` | `distilled-argmax` | `full` | player does not play this board |
| `search` | `greedy` | `naive` | player does not play this board |
| `search` | `greedy` | `oracle` | player does not play this board |
| `search` | `greedy` | `pay` | player does not play this board |
| `search` | `greedy` | `sweep` | player does not play this board |
| `search` | `greedy` | `full` | player does not play this board |
| `gen` | `neural` | `naive` | player does not play this board |
| `gen` | `neural` | `oracle` | player does not play this board |
| `gen` | `neural` | `pay` | player does not play this board |
| `gen` | `neural` | `sweep` | player does not play this board |
| `gen` | `neural` | `full` | player does not play this board |
| `gen` | `distilled` | `naive` | player does not play this board |
| `gen` | `distilled` | `oracle` | player does not play this board |
| `gen` | `distilled` | `pay` | player does not play this board |
| `gen` | `distilled` | `sweep` | player does not play this board |
| `gen` | `distilled` | `full` | player does not play this board |
| `gen` | `distilled-argmax` | `naive` | player does not play this board |
| `gen` | `distilled-argmax` | `oracle` | player does not play this board |
| `gen` | `distilled-argmax` | `pay` | player does not play this board |
| `gen` | `distilled-argmax` | `sweep` | player does not play this board |
| `gen` | `distilled-argmax` | `full` | player does not play this board |
| `kernel` | `neural` | `naive` | player does not play this board |
| `kernel` | `neural` | `oracle` | player does not play this board |
| `kernel` | `neural` | `pay` | player does not play this board |
| `kernel` | `neural` | `sweep` | player does not play this board |
| `kernel` | `neural` | `full` | player does not play this board |
| `kernel` | `distilled` | `naive` | player does not play this board |
| `kernel` | `distilled` | `oracle` | player does not play this board |
| `kernel` | `distilled` | `pay` | player does not play this board |
| `kernel` | `distilled` | `sweep` | player does not play this board |
| `kernel` | `distilled` | `full` | player does not play this board |
| `kernel` | `distilled-argmax` | `naive` | player does not play this board |
| `kernel` | `distilled-argmax` | `oracle` | player does not play this board |
| `kernel` | `distilled-argmax` | `pay` | player does not play this board |
| `kernel` | `distilled-argmax` | `sweep` | player does not play this board |
| `kernel` | `distilled-argmax` | `full` | player does not play this board |
| `kernel` | `greedy` | `naive` | player does not play this board |
| `kernel` | `greedy` | `oracle` | player does not play this board |
| `kernel` | `greedy` | `pay` | player does not play this board |
| `kernel` | `greedy` | `sweep` | player does not play this board |
| `kernel` | `greedy` | `full` | player does not play this board |
| `open` | `distilled` | `naive` | player does not play this board |
| `open` | `distilled` | `oracle` | player does not play this board |
| `open` | `distilled` | `pay` | player does not play this board |
| `open` | `distilled` | `sweep` | player does not play this board |
| `open` | `distilled` | `full` | player does not play this board |
| `open` | `distilled-argmax` | `naive` | player does not play this board |
| `open` | `distilled-argmax` | `oracle` | player does not play this board |
| `open` | `distilled-argmax` | `pay` | player does not play this board |
| `open` | `distilled-argmax` | `sweep` | player does not play this board |
| `open` | `distilled-argmax` | `full` | player does not play this board |
| `open` | `greedy` | `naive` | player does not play this board |
| `open` | `greedy` | `oracle` | player does not play this board |
| `open` | `greedy` | `pay` | player does not play this board |
| `open` | `greedy` | `sweep` | player does not play this board |
| `open` | `greedy` | `full` | player does not play this board |
| `contraction` | `random` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `random` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `random` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `random` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `random` | `full` | no equality claims — the board scores cost, not equalities |
| `contraction` | `playbook` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `playbook` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `playbook` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `playbook` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `playbook` | `full` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear` | `full` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-cold` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-cold` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-cold` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-cold` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-cold` | `full` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-greedy` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-greedy` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-greedy` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-greedy` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `linear-greedy` | `full` | no equality claims — the board scores cost, not equalities |
| `contraction` | `neural` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `neural` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `neural` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `neural` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `neural` | `full` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled` | `full` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled-argmax` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled-argmax` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled-argmax` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled-argmax` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `distilled-argmax` | `full` | no equality claims — the board scores cost, not equalities |
| `contraction` | `greedy` | `naive` | no equality claims — the board scores cost, not equalities |
| `contraction` | `greedy` | `oracle` | no equality claims — the board scores cost, not equalities |
| `contraction` | `greedy` | `pay` | no equality claims — the board scores cost, not equalities |
| `contraction` | `greedy` | `sweep` | no equality claims — the board scores cost, not equalities |
| `contraction` | `greedy` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `random` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `random` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `random` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `random` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `random` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `playbook` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `playbook` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `playbook` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `playbook` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `playbook` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `linear` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `linear` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `linear` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `linear` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `linear` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-cold` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-cold` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-cold` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-cold` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-cold` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-greedy` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-greedy` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-greedy` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-greedy` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `linear-greedy` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `neural` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `neural` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `neural` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `neural` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `neural` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled-argmax` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled-argmax` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled-argmax` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled-argmax` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `distilled-argmax` | `full` | no equality claims — the board scores cost, not equalities |
| `player` | `greedy` | `naive` | no equality claims — the board scores cost, not equalities |
| `player` | `greedy` | `oracle` | no equality claims — the board scores cost, not equalities |
| `player` | `greedy` | `pay` | no equality claims — the board scores cost, not equalities |
| `player` | `greedy` | `sweep` | no equality claims — the board scores cost, not equalities |
| `player` | `greedy` | `full` | no equality claims — the board scores cost, not equalities |

