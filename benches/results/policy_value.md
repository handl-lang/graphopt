# Learned search policy vs heuristics

For held-out matmul chains: the rank (1 = best), cost delta and hit rate of the rule each player picks — random / declaration-order / cost-model greedy / learned — plus the per-decision cost.

- **when**: 2026-10-03T00:04:21+00:00 (git `2211664` dirty)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `train_programs`=24 · `epochs`=120 · `hidden`=32 · `seed`=0 · `n_rules`=51 · `held_out`=8 · `device`=cpu · `quick`=True

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **NEGATIVE** | a learned search policy matches the cost-model greedy on held-out programs | learned mean rank 13.6 vs greedy 7.2 (hit 4/8 vs 6/8) | learned mean rank / greedy mean rank = 1.879 |
| **WIN** | a learned search policy ranks rules better than random and declaration-order | learned mean rank 13.6 vs random/declaration 26.4; hit rate 50% vs 0% | learned mean rank = 13.625 |

## Results

| case | program | d | k | m | random ms | declaration ms | greedy ms | learned ms |
|---|---|---|---|---|---|---|---|---|
| R 3x5x2 | R 3x5x2 | 3 | 5 | 2 | 0.0002941 ± 2.109e-08 | 6.721e-05 ± 0 | 0.002309 ± 0 | 2.281 ± 0 |
| R 4x6x3 | R 4x6x3 | 4 | 6 | 3 | 0.0002844 ± 3.888e-07 | 7.091e-05 ± 0 | 0.002315 ± 0 | 2.328 ± 0 |
| R 2x5x4 | R 2x5x4 | 2 | 5 | 4 | 0.0002835 ± 4.495e-08 | 6.975e-05 ± 0 | 0.002346 ± 0 | 2.33 ± 0 |
| L 5x2x6 | L 5x2x6 | 5 | 2 | 6 | 0.0002837 ± 2.298e-06 | 6.989e-05 ± 0 | 0.002415 ± 0 | 2.305 ± 0 |
| L 3x2x5 | L 3x2x5 | 3 | 2 | 5 | 0.0002902 ± 2.321e-07 | 7.107e-05 ± 0 | 0.002336 ± 0 | 2.302 ± 0 |
| L 4x2x7 | L 4x2x7 | 4 | 2 | 7 | 0.0002784 ± 6.839e-07 | 6.91e-05 ± 0 | 0.002406 ± 0 | 2.282 ± 0 |
| R 5x3x6 | R 5x3x6 | 5 | 3 | 6 | 0.0002823 ± 1.575e-06 | 6.886e-05 ± 0 | 0.002437 ± 0 | 2.421 ± 0 |
| L 4x6x3 | L 4x6x3 | 4 | 6 | 3 | 0.0002832 ± 1.972e-06 | 7.184e-05 ± 0 | 0.002349 ± 0 | 2.461 ± 0 |

## Metrics

<details><summary><b>R 3x5x2</b> — raw metrics</summary>

```json
{
  "program": "R 3x5x2",
  "d": 3,
  "k": 5,
  "m": 2,
  "n_rules": 51,
  "best_rule": "assoc_matmul",
  "best_delta": 54.0,
  "policies": {
    "random": {
      "pick": "assoc_linear_bias",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "declaration": {
      "pick": "comm_add",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "greedy": {
      "pick": "assoc_matmul",
      "rank": 1.0,
      "delta": 54.0,
      "hit": true
    },
    "learned": {
      "pick": "assoc_matmul_rev",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    }
  }
}
```

</details>

<details><summary><b>R 4x6x3</b> — raw metrics</summary>

```json
{
  "program": "R 4x6x3",
  "d": 4,
  "k": 6,
  "m": 3,
  "n_rules": 51,
  "best_rule": "assoc_matmul",
  "best_delta": 96.0,
  "policies": {
    "random": {
      "pick": "silu_expand",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "declaration": {
      "pick": "comm_add",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "greedy": {
      "pick": "assoc_matmul",
      "rank": 1.0,
      "delta": 96.0,
      "hit": true
    },
    "learned": {
      "pick": "assoc_matmul_rev",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    }
  }
}
```

</details>

<details><summary><b>R 2x5x4</b> — raw metrics</summary>

```json
{
  "program": "R 2x5x4",
  "d": 2,
  "k": 5,
  "m": 4,
  "n_rules": 51,
  "best_rule": "assoc_matmul",
  "best_delta": 8.0,
  "policies": {
    "random": {
      "pick": "assoc_mul",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "declaration": {
      "pick": "comm_add",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "greedy": {
      "pick": "assoc_matmul",
      "rank": 1.0,
      "delta": 8.0,
      "hit": true
    },
    "learned": {
      "pick": "assoc_matmul",
      "rank": 1.0,
      "delta": 8.0,
      "hit": true
    }
  }
}
```

</details>

<details><summary><b>L 5x2x6</b> — raw metrics</summary>

```json
{
  "program": "L 5x2x6",
  "d": 5,
  "k": 2,
  "m": 6,
  "n_rules": 51,
  "best_rule": "assoc_matmul_rev",
  "best_delta": 200.0,
  "policies": {
    "random": {
      "pick": "right_distribute_matmul",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "declaration": {
      "pick": "comm_add",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "greedy": {
      "pick": "assoc_matmul_rev",
      "rank": 1.0,
      "delta": 200.0,
      "hit": true
    },
    "learned": {
      "pick": "assoc_matmul_rev",
      "rank": 1.0,
      "delta": 200.0,
      "hit": true
    }
  }
}
```

</details>

<details><summary><b>L 3x2x5</b> — raw metrics</summary>

```json
{
  "program": "L 3x2x5",
  "d": 3,
  "k": 2,
  "m": 5,
  "n_rules": 51,
  "best_rule": "assoc_matmul_rev",
  "best_delta": 54.0,
  "policies": {
    "random": {
      "pick": "right_distribute_matmul",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "declaration": {
      "pick": "comm_add",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "greedy": {
      "pick": "assoc_matmul_rev",
      "rank": 1.0,
      "delta": 54.0,
      "hit": true
    },
    "learned": {
      "pick": "assoc_matmul_rev",
      "rank": 1.0,
      "delta": 54.0,
      "hit": true
    }
  }
}
```

</details>

<details><summary><b>L 4x2x7</b> — raw metrics</summary>

```json
{
  "program": "L 4x2x7",
  "d": 4,
  "k": 2,
  "m": 7,
  "n_rules": 51,
  "best_rule": "assoc_matmul_rev",
  "best_delta": 160.0,
  "policies": {
    "random": {
      "pick": "sdpa_fold_addmul",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "declaration": {
      "pick": "comm_add",
      "rank": 26.5,
      "delta": 0.0,
      "hit": false
    },
    "greedy": {
      "pick": "assoc_matmul_rev",
      "rank": 1.0,
      "delta": 160.0,
      "hit": true
    },
    "learned": {
      "pick": "assoc_matmul_rev",
      "rank": 1.0,
      "delta": 160.0,
      "hit": true
    }
  }
}
```

</details>

<details><summary><b>R 5x3x6</b> — raw metrics</summary>

```json
{
  "program": "R 5x3x6",
  "d": 5,
  "k": 3,
  "m": 6,
  "n_rules": 51,
  "best_rule": "comm_add",
  "best_delta": 0.0,
  "policies": {
    "random": {
      "pick": "sdpa_fold_masked_fill_drop",
      "rank": 26.0,
      "delta": 0.0,
      "hit": false
    },
    "declaration": {
      "pick": "comm_add",
      "rank": 26.0,
      "delta": 0.0,
      "hit": false
    },
    "greedy": {
      "pick": "comm_add",
      "rank": 26.0,
      "delta": 0.0,
      "hit": false
    },
    "learned": {
      "pick": "assoc_matmul_rev",
      "rank": 26.0,
      "delta": 0.0,
      "hit": false
    }
  }
}
```

</details>

<details><summary><b>L 4x6x3</b> — raw metrics</summary>

```json
{
  "program": "L 4x6x3",
  "d": 4,
  "k": 6,
  "m": 3,
  "n_rules": 51,
  "best_rule": "comm_add",
  "best_delta": 0.0,
  "policies": {
    "random": {
      "pick": "weight_distribute_linear",
      "rank": 26.0,
      "delta": 0.0,
      "hit": false
    },
    "declaration": {
      "pick": "comm_add",
      "rank": 26.0,
      "delta": 0.0,
      "hit": false
    },
    "greedy": {
      "pick": "comm_add",
      "rank": 26.0,
      "delta": 0.0,
      "hit": false
    },
    "learned": {
      "pick": "assoc_matmul_rev",
      "rank": 26.0,
      "delta": 0.0,
      "hit": false
    }
  }
}
```

</details>

