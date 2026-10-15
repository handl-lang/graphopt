# Search cost vs exponential space

Saturation stats vs chain depth k: e-graph size against the Catalan program space it represents.

- **when**: 2026-10-14T02:30:34+00:00 (git `c1a9521` clean)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | the e-graph encodes an exponential program space in O(k³) live e-nodes | >10^6 programs encoded in 681 live e-nodes at k=24 (735 ms) | live e-nodes at k = 681 |
| **WIN** | exact saturation holds every Catalan(k−1) bracketing simultaneously | k=11: Catalan(10) = 16,796 programs in 231 e-nodes | programs encoded = 16796 |

## Results

| case | k | budget | sat ms | canon ms | strat ms | e2e ms |
|---|---|---|---|---|---|---|
| exact_k4 | 4 | — | 3.386 ± 0 | — | — | — |
| exact_k5 | 5 | — | 6.555 ± 0 | — | — | — |
| exact_k6 | 6 | — | 14.13 ± 0 | — | — | — |
| exact_k7 | 7 | — | 23.99 ± 0 | — | — | — |
| exact_k8 | 8 | — | 62.14 ± 0 | — | — | — |
| exact_k9 | 9 | — | 294.3 ± 0 | — | — | — |
| exact_k10 | 10 | — | 2324 ± 0 | — | — | — |
| exact_k11 | 11 | — | 1.735e+04 ± 0 | — | — | — |
| bounded_k4 | 4 | 2048 | 4.349 ± 0 | — | — | — |
| strat_k4 | 4 | — | — | 0.07468 ± 0 | 2.543 ± 0 | — |
| e2e_k4 | 4 | — | — | — | — | 58.38 ± 0 |
| bounded_k8 | 8 | 2048 | 70.64 ± 0 | — | — | — |
| strat_k8 | 8 | — | — | 0.1268 ± 0 | 2.28 ± 0 | — |
| e2e_k8 | 8 | — | — | — | — | 145 ± 0 |
| bounded_k12 | 12 | 2048 | 213 ± 0 | — | — | — |
| strat_k12 | 12 | — | — | 0.2697 ± 0 | 2.978 ± 0 | — |
| e2e_k12 | 12 | — | — | — | — | 459.7 ± 0 |
| bounded_k16 | 16 | 2048 | 367.1 ± 0 | — | — | — |
| strat_k16 | 16 | — | — | 0.4218 ± 0 | 7.849 ± 0 | — |
| e2e_k16 | 16 | — | — | — | — | 861.1 ± 0 |
| bounded_k20 | 20 | 2048 | 375.3 ± 0 | — | — | — |
| strat_k20 | 20 | — | — | 0.5987 ± 0 | 8.536 ± 0 | — |
| e2e_k20 | 20 | — | — | — | — | 2012 ± 0 |
| bounded_k24 | 24 | 2048 | 735.4 ± 0 | — | — | — |
| strat_k24 | 24 | — | — | 0.7971 ± 0 | 9.01 ± 0 | — |
| e2e_k24 | 24 | — | — | — | — | 3306 ± 0 |

## Metrics

<details><summary><b>exact_k4</b> — raw metrics</summary>

```json
{
  "policy": "exact",
  "space": 5,
  "space_terms": "Catalan(3)",
  "space_log10": 0.6989700043360189,
  "live_theory": 14,
  "eclasses_theory": 10,
  "enodes": 15,
  "live": 14,
  "eclasses": 10,
  "iters": 3,
  "fires": 5,
  "rule_fires": {
    "assoc_matmul": 5
  },
  "wall_ms": 3.386,
  "extract_ms": 0.29,
  "count_ms": 0.075,
  "encode_dp_count": 5,
  "terms_capped": false,
  "derivations": 5,
  "best_cost": 458752.0,
  "budget_spent": {},
  "budget_suspended": []
}
```

</details>

<details><summary><b>exact_k5</b> — raw metrics</summary>

```json
{
  "policy": "exact",
  "space": 14,
  "space_terms": "Catalan(4)",
  "space_log10": 1.146128035678238,
  "live_theory": 25,
  "eclasses_theory": 15,
  "enodes": 31,
  "live": 25,
  "eclasses": 15,
  "iters": 4,
  "fires": 16,
  "rule_fires": {
    "assoc_matmul": 16
  },
  "wall_ms": 6.555,
  "extract_ms": 0.656,
  "count_ms": 0.126,
  "encode_dp_count": 14,
  "terms_capped": false,
  "derivations": 14,
  "best_cost": 31457280.0,
  "budget_spent": {},
  "budget_suspended": []
}
```

</details>

<details><summary><b>exact_k6</b> — raw metrics</summary>

```json
{
  "policy": "exact",
  "space": 42,
  "space_terms": "Catalan(5)",
  "space_log10": 1.6232492903979006,
  "live_theory": 41,
  "eclasses_theory": 21,
  "enodes": 65,
  "live": 41,
  "eclasses": 21,
  "iters": 5,
  "fires": 44,
  "rule_fires": {
    "assoc_matmul": 43,
    "assoc_matmul_rev": 1
  },
  "wall_ms": 14.132,
  "extract_ms": 1.05,
  "count_ms": 0.222,
  "encode_dp_count": 42,
  "terms_capped": false,
  "derivations": 42,
  "best_cost": 2228224.0,
  "budget_spent": {},
  "budget_suspended": []
}
```

</details>

<details><summary><b>exact_k7</b> — raw metrics</summary>

```json
{
  "policy": "exact",
  "space": 132,
  "space_terms": "Catalan(6)",
  "space_log10": 2.12057393120585,
  "live_theory": 63,
  "eclasses_theory": 28,
  "enodes": 145,
  "live": 63,
  "eclasses": 28,
  "iters": 6,
  "fires": 117,
  "rule_fires": {
    "assoc_matmul": 110,
    "assoc_matmul_rev": 7
  },
  "wall_ms": 23.987,
  "extract_ms": 0.852,
  "count_ms": 0.25,
  "encode_dp_count": 132,
  "terms_capped": false,
  "derivations": 132,
  "best_cost": 26214400.0,
  "budget_spent": {},
  "budget_suspended": []
}
```

</details>

<details><summary><b>exact_k8</b> — raw metrics</summary>

```json
{
  "policy": "exact",
  "space": 429,
  "space_terms": "Catalan(7)",
  "space_log10": 2.632457292184724,
  "live_theory": 92,
  "eclasses_theory": 36,
  "enodes": 353,
  "live": 92,
  "eclasses": 36,
  "iters": 7,
  "fires": 317,
  "rule_fires": {
    "assoc_matmul": 285,
    "assoc_matmul_rev": 32
  },
  "wall_ms": 62.139,
  "extract_ms": 1.264,
  "count_ms": 0.502,
  "encode_dp_count": 429,
  "terms_capped": false,
  "derivations": 429,
  "best_cost": 950272.0,
  "budget_spent": {},
  "budget_suspended": []
}
```

</details>

<details><summary><b>exact_k9</b> — raw metrics</summary>

```json
{
  "policy": "exact",
  "space": 1430,
  "space_terms": "Catalan(8)",
  "space_log10": 3.155336037465062,
  "live_theory": 129,
  "eclasses_theory": 45,
  "enodes": 932,
  "live": 129,
  "eclasses": 45,
  "iters": 8,
  "fires": 887,
  "rule_fires": {
    "assoc_matmul": 766,
    "assoc_matmul_rev": 121
  },
  "wall_ms": 294.263,
  "extract_ms": 1.762,
  "count_ms": 1.498,
  "encode_dp_count": 1430,
  "terms_capped": false,
  "derivations": 1430,
  "best_cost": 62914560.0,
  "budget_spent": {},
  "budget_suspended": []
}
```

</details>

<details><summary><b>exact_k10</b> — raw metrics</summary>

```json
{
  "policy": "exact",
  "space": 4862,
  "space_terms": "Catalan(9)",
  "space_log10": 3.6868149545073168,
  "live_theory": 175,
  "eclasses_theory": 55,
  "enodes": 2604,
  "live": 175,
  "eclasses": 55,
  "iters": 9,
  "fires": 2549,
  "rule_fires": {
    "assoc_matmul": 2136,
    "assoc_matmul_rev": 413
  },
  "wall_ms": 2323.703,
  "extract_ms": 2.266,
  "count_ms": 3.688,
  "encode_dp_count": 4862,
  "terms_capped": false,
  "derivations": 4862,
  "best_cost": 4194304.0,
  "budget_spent": {},
  "budget_suspended": []
}
```

</details>

<details><summary><b>exact_k11</b> — raw metrics</summary>

```json
{
  "policy": "exact",
  "space": 16796,
  "space_terms": "Catalan(10)",
  "space_log10": 4.225205865965902,
  "live_theory": 231,
  "eclasses_theory": 66,
  "enodes": 7520,
  "live": 231,
  "eclasses": 66,
  "iters": 10,
  "fires": 7454,
  "rule_fires": {
    "assoc_matmul": 6117,
    "assoc_matmul_rev": 1337
  },
  "wall_ms": 17347.334,
  "extract_ms": 5.103,
  "count_ms": 17.973,
  "encode_dp_count": 16796,
  "terms_capped": false,
  "derivations": 16796,
  "best_cost": 41943040.0,
  "budget_spent": {},
  "budget_suspended": []
}
```

</details>

<details><summary><b>bounded_k4</b> — raw metrics</summary>

```json
{
  "policy": "bounded",
  "space": 5,
  "space_terms": "Catalan(3)",
  "space_log10": 0.6989700043360189,
  "live_theory": 14,
  "eclasses_theory": 10,
  "enodes": 15,
  "live": 14,
  "eclasses": 10,
  "iters": 3,
  "fires": 5,
  "rule_fires": {
    "assoc_matmul": 5
  },
  "wall_ms": 4.349,
  "extract_ms": 0.382,
  "count_ms": 0.072,
  "encode_dp_count": 5,
  "terms_capped": false,
  "derivations": 5,
  "best_cost": 458752.0,
  "budget_spent": {
    "assoc_matmul": 8,
    "assoc_matmul_rev": 0
  },
  "budget_suspended": []
}
```

</details>

<details><summary><b>strat_k4</b> — raw metrics</summary>

```json
{
  "canon_ms": 0.075,
  "stratified_ms": 2.543,
  "enodes": 7,
  "coherent_dropped": [
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "id_add",
    "id_mul",
    "double_neg",
    "assoc_matmul",
    "assoc_matmul_rev"
  ],
  "balanced_cost": 397312.0
}
```

</details>

<details><summary><b>e2e_k4</b> — raw metrics</summary>

```json
{
  "e2e_ms": 58.383,
  "enodes": 25,
  "iters": 2,
  "fires": 15,
  "verified_rel_diff": 5.387930173128552e-07
}
```

</details>

<details><summary><b>bounded_k8</b> — raw metrics</summary>

```json
{
  "policy": "bounded",
  "space": 429,
  "space_terms": "Catalan(7)",
  "space_log10": 2.632457292184724,
  "live_theory": 92,
  "eclasses_theory": 36,
  "enodes": 353,
  "live": 92,
  "eclasses": 36,
  "iters": 7,
  "fires": 317,
  "rule_fires": {
    "assoc_matmul": 285,
    "assoc_matmul_rev": 32
  },
  "wall_ms": 70.643,
  "extract_ms": 1.006,
  "count_ms": 0.491,
  "encode_dp_count": 429,
  "terms_capped": false,
  "derivations": 429,
  "best_cost": 950272.0,
  "budget_spent": {
    "assoc_matmul": 338,
    "assoc_matmul_rev": 0
  },
  "budget_suspended": []
}
```

</details>

<details><summary><b>strat_k8</b> — raw metrics</summary>

```json
{
  "canon_ms": 0.127,
  "stratified_ms": 2.28,
  "enodes": 15,
  "coherent_dropped": [
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "id_add",
    "id_mul",
    "double_neg",
    "assoc_matmul",
    "assoc_matmul_rev"
  ],
  "balanced_cost": 795648.0
}
```

</details>

<details><summary><b>e2e_k8</b> — raw metrics</summary>

```json
{
  "e2e_ms": 144.993,
  "enodes": 129,
  "iters": 2,
  "fires": 175,
  "verified_rel_diff": 9.13858343665197e-07
}
```

</details>

<details><summary><b>bounded_k12</b> — raw metrics</summary>

```json
{
  "policy": "bounded",
  "space": 58786,
  "space_terms": "Catalan(11)",
  "space_log10": 4.769273910316178,
  "live_theory": 298,
  "eclasses_theory": 78,
  "enodes": 2071,
  "live": 237,
  "eclasses": 78,
  "iters": 4,
  "fires": 1281,
  "rule_fires": {
    "assoc_matmul": 1118,
    "assoc_matmul_rev": 163
  },
  "wall_ms": 212.97,
  "extract_ms": 3.02,
  "count_ms": 19.615,
  "encode_dp_count": 22236,
  "terms_capped": false,
  "derivations": 22236,
  "best_cost": 1441792.0,
  "budget_spent": {
    "assoc_matmul": 2048,
    "assoc_matmul_rev": 0
  },
  "budget_suspended": [
    "assoc_matmul"
  ]
}
```

</details>

<details><summary><b>strat_k12</b> — raw metrics</summary>

```json
{
  "canon_ms": 0.27,
  "stratified_ms": 2.978,
  "enodes": 23,
  "coherent_dropped": [
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "id_add",
    "id_mul",
    "double_neg",
    "assoc_matmul",
    "assoc_matmul_rev"
  ],
  "balanced_cost": 22155264.0
}
```

</details>

<details><summary><b>e2e_k12</b> — raw metrics</summary>

```json
{
  "e2e_ms": 459.736,
  "enodes": 377,
  "iters": 2,
  "fires": 671,
  "verified_rel_diff": 5.758005272582523e-07
}
```

</details>

<details><summary><b>bounded_k16</b> — raw metrics</summary>

```json
{
  "policy": "bounded",
  "space": 9694845,
  "space_terms": "Catalan(15)",
  "space_log10": 6.986540870022996,
  "live_theory": 696,
  "eclasses_theory": 136,
  "enodes": 2591,
  "live": 347,
  "eclasses": 136,
  "iters": 8,
  "fires": 1919,
  "rule_fires": {
    "assoc_matmul": 1075,
    "assoc_matmul_rev": 844
  },
  "wall_ms": 367.097,
  "extract_ms": 4.83,
  "count_ms": 511.971,
  "encode_dp_count": 362684,
  "terms_capped": true,
  "derivations": 414496,
  "best_cost": 1933312.0,
  "budget_spent": {
    "assoc_matmul": 2150,
    "assoc_matmul_rev": 410
  },
  "budget_suspended": [
    "assoc_matmul"
  ]
}
```

</details>

<details><summary><b>strat_k16</b> — raw metrics</summary>

```json
{
  "canon_ms": 0.422,
  "stratified_ms": 7.849,
  "enodes": 31,
  "coherent_dropped": [
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "id_add",
    "id_mul",
    "double_neg",
    "assoc_matmul",
    "assoc_matmul_rev"
  ],
  "balanced_cost": 1592320.0
}
```

</details>

<details><summary><b>e2e_k16</b> — raw metrics</summary>

```json
{
  "e2e_ms": 861.147,
  "enodes": 833,
  "iters": 2,
  "fires": 1695,
  "verified_rel_diff": 8.547226570954081e-07
}
```

</details>

<details><summary><b>bounded_k20</b> — raw metrics</summary>

```json
{
  "policy": "bounded",
  "space": 1767263190,
  "space_terms": "Catalan(19)",
  "space_log10": 9.247301231709962,
  "live_theory": 1350,
  "eclasses_theory": 210,
  "enodes": 2807,
  "live": 526,
  "eclasses": 210,
  "iters": 8,
  "fires": 2237,
  "rule_fires": {
    "assoc_matmul": 1149,
    "assoc_matmul_rev": 1088
  },
  "wall_ms": 375.26,
  "extract_ms": 7.669,
  "count_ms": 1371.674,
  "encode_dp_count": 382320,
  "terms_capped": true,
  "derivations": 10874880,
  "best_cost": 2424832.0,
  "budget_spent": {
    "assoc_matmul": 2298,
    "assoc_matmul_rev": 470
  },
  "budget_suspended": [
    "assoc_matmul"
  ]
}
```

</details>

<details><summary><b>strat_k20</b> — raw metrics</summary>

```json
{
  "canon_ms": 0.599,
  "stratified_ms": 8.536,
  "enodes": 39,
  "coherent_dropped": [
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "id_add",
    "id_mul",
    "double_neg",
    "assoc_matmul",
    "assoc_matmul_rev"
  ],
  "balanced_cost": 43913216.0
}
```

</details>

<details><summary><b>e2e_k20</b> — raw metrics</summary>

```json
{
  "e2e_ms": 2011.645,
  "enodes": 1561,
  "iters": 2,
  "fires": 3150,
  "verified_rel_diff": 7.074161771924992e-07
}
```

</details>

<details><summary><b>bounded_k24</b> — raw metrics</summary>

```json
{
  "policy": "bounded",
  "space": 343059613650,
  "space_terms": "Catalan(23)",
  "space_log10": 11.535369594181725,
  "live_theory": 2324,
  "eclasses_theory": 300,
  "enodes": 2494,
  "live": 681,
  "eclasses": 300,
  "iters": 7,
  "fires": 1849,
  "rule_fires": {
    "assoc_matmul": 1070,
    "assoc_matmul_rev": 779
  },
  "wall_ms": 735.366,
  "extract_ms": 12.119,
  "count_ms": 2042.61,
  "encode_dp_count": 349100,
  "terms_capped": true,
  "derivations": 142991360,
  "best_cost": 2916352.0,
  "budget_spent": {
    "assoc_matmul": 2140,
    "assoc_matmul_rev": 307
  },
  "budget_suspended": [
    "assoc_matmul"
  ]
}
```

</details>

<details><summary><b>strat_k24</b> — raw metrics</summary>

```json
{
  "canon_ms": 0.797,
  "stratified_ms": 9.01,
  "enodes": 47,
  "coherent_dropped": [
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "id_add",
    "id_mul",
    "double_neg",
    "assoc_matmul",
    "assoc_matmul_rev"
  ],
  "balanced_cost": 44311552.0
}
```

</details>

<details><summary><b>e2e_k24</b> — raw metrics</summary>

```json
{
  "e2e_ms": 3306.134,
  "enodes": 2625,
  "iters": 2,
  "fires": 3645,
  "verified_rel_diff": 8.411975613853429e-07
}
```

</details>

