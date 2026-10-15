# Morphism-window composition

Term-flops → wall-time conversion for the morphism laws: which structural rewrites actually pay off at GEMM-bound sizes.

- **when**: 2026-10-14T06:17:47+00:00 (git `2c782dd` dirty)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | morphism windows convert term-level flops headroom into measured wall time | best 12.85× vs eager on chain_x4@4096x128 | morphism / eager = 12.8538 |
| **WIN** | every morphism rewrite verifies equivalent | 14/14 cells verified | verified cells = 14 |

## Results

| case | fixture | rows | dim | eager ms | inductor ms | morphism ms | morphism+inductor ms |
|---|---|---|---|---|---|---|---|
| chain_x2@8x16 | chain_x2 | 8 | 16 | 0.0252 ± 0 | 0.04032 ± 0.003148 | 0.007184 ± 0 | 0.03348 ± 0.001292 |
| chain_x2@1024x64 | chain_x2 | 1024 | 64 | 1.036 ± 0 | 1.033 ± 0.02626 | 0.1642 ± 0 | 0.2045 ± 0.004246 |
| chain_x2@4096x128 | chain_x2 | 4096 | 128 | 15.69 ± 0 | 16.17 ± 0.3616 | 2.578 ± 0 | 2.554 ± 0.1861 |
| chain_x4@8x16 | chain_x4 | 8 | 16 | 0.04644 ± 0.0003147 | 0.05444 ± 0.001505 | 0.009151 ± 0 | 0.03481 ± 0.001355 |
| chain_x4@1024x64 | chain_x4 | 1024 | 64 | 2.08 ± 0 | 2.054 ± 0.06036 | 0.1676 ± 0 | 0.2098 ± 0.006368 |
| chain_x4@4096x128 | chain_x4 | 4096 | 128 | 31.22 ± 0.3155 | 30.2 ± 0.7422 | 2.428 ± 0 | 2.469 ± 0.0496 |
| resid_x3@8x16 | resid_x3 | 8 | 16 | 0.03855 ± 0.0004229 | 0.04928 ± 0.001499 | 0.02748 ± 0.0003971 | 0.04458 ± 0.001212 |
| resid_x3@1024x64 | resid_x3 | 1024 | 64 | 1.651 ± 0 | 1.594 ± 0.02885 | 0.7535 ± 0 | 0.6053 ± 0.008438 |
| resid_x3@4096x128 | resid_x3 | 4096 | 128 | 27.29 ± 0 | 23.1 ± 0.6546 | 13.56 ± 0 | 7.97 ± 0.2469 |
| resid_x4@8x16 | resid_x4 | 8 | 16 | 0.05058 ± 0.0004475 | 0.05725 ± 0.00177 | 0.0369 ± 0.0005991 | 0.04882 ± 0.001611 |
| resid_x4@1024x64 | resid_x4 | 1024 | 64 | 2.242 ± 0 | 2.163 ± 0.03024 | 1.068 ± 0 | 0.8467 ± 0.008938 |
| resid_x4@4096x128 | resid_x4 | 4096 | 128 | 36.29 ± 1.178 | 31.4 ± 0.5507 | 18.3 ± 0 | 11.56 ± 0.2705 |
| kv_x2@8x16 | kv_x2 | 8 | 16 | 0.0611 ± 0.0001049 | 0.07773 ± 0.00266 | 0.0699 ± 0 | 0.09543 ± 0.002988 |
| kv_x2@512x64 | kv_x2 | 512 | 64 | 2.404 ± 0 | 2.332 ± 0.04625 | 2.492 ± 0 | 2.453 ± 0.049 |

## Metrics

<details><summary><b>chain_x2@8x16</b> — raw metrics</summary>

```json
{
  "name": "chain_x2@8x16",
  "params": {
    "fixture": "chain_x2",
    "rows": 8,
    "dim": 16
  },
  "param_bytes_eager": 12288,
  "opt_s": 2.98,
  "fires": {
    "out_in_compose": 1
  },
  "n_rewritten": 2,
  "e2e_rel": 3.8991816500633817e-16,
  "n_grafted": 1,
  "matches": {
    "out_in_compose:blocks.0+blocks.1": {
      "status": "grafted",
      "boundary": "chain",
      "cost_before": 24832.0,
      "cost_after": 4096.0,
      "rel_diff": 3.8991816500633817e-16
    }
  },
  "term_flops_before": 24832.0,
  "term_flops_after": 4096.0,
  "flops_ratio": 6.0625,
  "param_bytes_opt": 2048,
  "morph_verified": true,
  "morph_rel": 3.8991816500633817e-16,
  "rt_flops_eager": 24832.0,
  "rt_flops_morphism": 4096.0,
  "rt_flops_ratio": 6.0625,
  "inductor_status": "compiled in 1.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.1s",
  "morph_ind_verified": true,
  "morph_ind_rel": 3.8991816500633817e-16
}
```

</details>

<details><summary><b>chain_x2@1024x64</b> — raw metrics</summary>

```json
{
  "name": "chain_x2@1024x64",
  "params": {
    "fixture": "chain_x2",
    "rows": 1024,
    "dim": 64
  },
  "param_bytes_eager": 196608,
  "opt_s": 0.23,
  "fires": {
    "out_in_compose": 1
  },
  "n_rewritten": 2,
  "e2e_rel": 8.919553791225219e-16,
  "n_grafted": 1,
  "matches": {
    "out_in_compose:blocks.0+blocks.1": {
      "status": "grafted",
      "boundary": "chain",
      "cost_before": 50462720.0,
      "cost_after": 8388608.0,
      "rel_diff": 8.919553791225219e-16
    }
  },
  "term_flops_before": 50462720.0,
  "term_flops_after": 8388608.0,
  "flops_ratio": 6.015625,
  "param_bytes_opt": 32768,
  "morph_verified": true,
  "morph_rel": 8.919553791225219e-16,
  "rt_flops_eager": 50462720.0,
  "rt_flops_morphism": 8388608.0,
  "rt_flops_ratio": 6.015625,
  "inductor_status": "compiled in 0.1s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.1s",
  "morph_ind_verified": true,
  "morph_ind_rel": 8.919553791225219e-16
}
```

</details>

<details><summary><b>chain_x2@4096x128</b> — raw metrics</summary>

```json
{
  "name": "chain_x2@4096x128",
  "params": {
    "fixture": "chain_x2",
    "rows": 4096,
    "dim": 128
  },
  "param_bytes_eager": 786432,
  "opt_s": 0.29,
  "fires": {
    "out_in_compose": 1
  },
  "n_rewritten": 2,
  "e2e_rel": 1.1546202280224345e-15,
  "n_grafted": 1,
  "matches": {
    "out_in_compose:blocks.0+blocks.1": {
      "status": "grafted",
      "boundary": "chain",
      "cost_before": 806354944.0,
      "cost_after": 134217728.0,
      "rel_diff": 1.1546202280224345e-15
    }
  },
  "term_flops_before": 806354944.0,
  "term_flops_after": 134217728.0,
  "flops_ratio": 6.0078125,
  "param_bytes_opt": 131072,
  "morph_verified": true,
  "morph_rel": 1.1546202280224345e-15,
  "rt_flops_eager": 806354944.0,
  "rt_flops_morphism": 134217728.0,
  "rt_flops_ratio": 6.0078125,
  "inductor_status": "compiled in 0.1s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.1s",
  "morph_ind_verified": true,
  "morph_ind_rel": 1.1546202280224345e-15
}
```

</details>

<details><summary><b>chain_x4@8x16</b> — raw metrics</summary>

```json
{
  "name": "chain_x4@8x16",
  "params": {
    "fixture": "chain_x4",
    "rows": 8,
    "dim": 16
  },
  "param_bytes_eager": 24576,
  "opt_s": 1.22,
  "fires": {
    "window_compose": 1
  },
  "n_rewritten": 4,
  "e2e_rel": 3.5064085798545327e-16,
  "n_grafted": 1,
  "matches": {
    "window_compose:blocks.0+blocks.1+blocks.2+blocks.3": {
      "status": "grafted",
      "boundary": "chain+chain+chain",
      "cost_before": 49664.0,
      "cost_after": 4096.0,
      "rel_diff": 3.5064085798545327e-16
    },
    "out_in_compose:blocks.0+blocks.1": {
      "status": "skipped",
      "reason": "node already rewritten"
    },
    "out_in_compose:blocks.1+blocks.2": {
      "status": "skipped",
      "reason": "node already rewritten"
    },
    "out_in_compose:blocks.2+blocks.3": {
      "status": "skipped",
      "reason": "node already rewritten"
    }
  },
  "term_flops_before": 49664.0,
  "term_flops_after": 4096.0,
  "flops_ratio": 12.125,
  "param_bytes_opt": 2048,
  "morph_verified": true,
  "morph_rel": 3.5064085798545327e-16,
  "rt_flops_eager": 49664.0,
  "rt_flops_morphism": 4096.0,
  "rt_flops_ratio": 12.125,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.1s",
  "morph_ind_verified": true,
  "morph_ind_rel": 3.5064085798545327e-16
}
```

</details>

<details><summary><b>chain_x4@1024x64</b> — raw metrics</summary>

```json
{
  "name": "chain_x4@1024x64",
  "params": {
    "fixture": "chain_x4",
    "rows": 1024,
    "dim": 64
  },
  "param_bytes_eager": 393216,
  "opt_s": 1.21,
  "fires": {
    "window_compose": 1
  },
  "n_rewritten": 4,
  "e2e_rel": 1.2397143289022856e-15,
  "n_grafted": 1,
  "matches": {
    "window_compose:blocks.0+blocks.1+blocks.2+blocks.3": {
      "status": "grafted",
      "boundary": "chain+chain+chain",
      "cost_before": 100925440.0,
      "cost_after": 8388608.0,
      "rel_diff": 1.2397143289022856e-15
    },
    "out_in_compose:blocks.0+blocks.1": {
      "status": "skipped",
      "reason": "node already rewritten"
    },
    "out_in_compose:blocks.1+blocks.2": {
      "status": "skipped",
      "reason": "node already rewritten"
    },
    "out_in_compose:blocks.2+blocks.3": {
      "status": "skipped",
      "reason": "node already rewritten"
    }
  },
  "term_flops_before": 100925440.0,
  "term_flops_after": 8388608.0,
  "flops_ratio": 12.03125,
  "param_bytes_opt": 32768,
  "morph_verified": true,
  "morph_rel": 1.2397143289022856e-15,
  "rt_flops_eager": 100925440.0,
  "rt_flops_morphism": 8388608.0,
  "rt_flops_ratio": 12.03125,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.1s",
  "morph_ind_verified": true,
  "morph_ind_rel": 1.2397143289022856e-15
}
```

</details>

<details><summary><b>chain_x4@4096x128</b> — raw metrics</summary>

```json
{
  "name": "chain_x4@4096x128",
  "params": {
    "fixture": "chain_x4",
    "rows": 4096,
    "dim": 128
  },
  "param_bytes_eager": 1572864,
  "opt_s": 1.34,
  "fires": {
    "window_compose": 1
  },
  "n_rewritten": 4,
  "e2e_rel": 1.583769020636777e-15,
  "n_grafted": 1,
  "matches": {
    "window_compose:blocks.0+blocks.1+blocks.2+blocks.3": {
      "status": "grafted",
      "boundary": "chain+chain+chain",
      "cost_before": 1612709888.0,
      "cost_after": 134217728.0,
      "rel_diff": 1.583769020636777e-15
    },
    "out_in_compose:blocks.0+blocks.1": {
      "status": "skipped",
      "reason": "node already rewritten"
    },
    "out_in_compose:blocks.1+blocks.2": {
      "status": "skipped",
      "reason": "node already rewritten"
    },
    "out_in_compose:blocks.2+blocks.3": {
      "status": "skipped",
      "reason": "node already rewritten"
    }
  },
  "term_flops_before": 1612709888.0,
  "term_flops_after": 134217728.0,
  "flops_ratio": 12.015625,
  "param_bytes_opt": 131072,
  "morph_verified": true,
  "morph_rel": 1.583769020636777e-15,
  "rt_flops_eager": 1612709888.0,
  "rt_flops_morphism": 134217728.0,
  "rt_flops_ratio": 12.015625,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.1s",
  "morph_ind_verified": true,
  "morph_ind_rel": 1.583769020636777e-15
}
```

</details>

<details><summary><b>resid_x3@8x16</b> — raw metrics</summary>

```json
{
  "name": "resid_x3@8x16",
  "params": {
    "fixture": "resid_x3",
    "rows": 8,
    "dim": 16
  },
  "param_bytes_eager": 18432,
  "opt_s": 0.42,
  "fires": {
    "residual_reassoc": 1
  },
  "n_rewritten": 3,
  "e2e_rel": 2.5985308991305546e-16,
  "n_grafted": 1,
  "matches": {
    "residual_reassoc:blocks.0+blocks.1+blocks.2": {
      "status": "grafted",
      "boundary": "residual_wrapped+residual_wrapped",
      "cost_before": 37632.0,
      "cost_after": 12672.0,
      "rel_diff": 2.5985308991305546e-16
    }
  },
  "term_flops_before": 37632.0,
  "term_flops_after": 12672.0,
  "flops_ratio": 2.9696969696969697,
  "param_bytes_opt": 6144,
  "morph_verified": true,
  "morph_rel": 2.5985308991305546e-16,
  "rt_flops_eager": 37632.0,
  "rt_flops_morphism": 13312.0,
  "rt_flops_ratio": 2.826923076923077,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 1.5s",
  "morph_ind_verified": true,
  "morph_ind_rel": 2.5985308991305546e-16
}
```

</details>

<details><summary><b>resid_x3@1024x64</b> — raw metrics</summary>

```json
{
  "name": "resid_x3@1024x64",
  "params": {
    "fixture": "resid_x3",
    "rows": 1024,
    "dim": 64
  },
  "param_bytes_eager": 294912,
  "opt_s": 0.44,
  "fires": {
    "residual_reassoc": 1
  },
  "n_rewritten": 3,
  "e2e_rel": 4.825521243161371e-16,
  "n_grafted": 1,
  "matches": {
    "residual_reassoc:blocks.0+blocks.1+blocks.2": {
      "status": "grafted",
      "boundary": "residual_wrapped+residual_wrapped",
      "cost_before": 75890688.0,
      "cost_after": 25362432.0,
      "rel_diff": 4.825521243161371e-16
    }
  },
  "term_flops_before": 75890688.0,
  "term_flops_after": 25362432.0,
  "flops_ratio": 2.992248062015504,
  "param_bytes_opt": 98304,
  "morph_verified": true,
  "morph_rel": 4.825521243161371e-16,
  "rt_flops_eager": 75890688.0,
  "rt_flops_morphism": 25690112.0,
  "rt_flops_ratio": 2.954081632653061,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.7s",
  "morph_ind_verified": true,
  "morph_ind_rel": 4.825521243161371e-16
}
```

</details>

<details><summary><b>resid_x3@4096x128</b> — raw metrics</summary>

```json
{
  "name": "resid_x3@4096x128",
  "params": {
    "fixture": "resid_x3",
    "rows": 4096,
    "dim": 128
  },
  "param_bytes_eager": 1179648,
  "opt_s": 0.57,
  "fires": {
    "residual_reassoc": 1
  },
  "n_rewritten": 3,
  "e2e_rel": 6.638900357069024e-16,
  "n_grafted": 1,
  "matches": {
    "residual_reassoc:blocks.0+blocks.1+blocks.2": {
      "status": "grafted",
      "boundary": "residual_wrapped+residual_wrapped",
      "cost_before": 1211105280.0,
      "cost_after": 404226048.0,
      "rel_diff": 6.638900357069024e-16
    }
  },
  "term_flops_before": 1211105280.0,
  "term_flops_after": 404226048.0,
  "flops_ratio": 2.9961089494163424,
  "param_bytes_opt": 393216,
  "morph_verified": true,
  "morph_rel": 6.638900357069024e-16,
  "rt_flops_eager": 1211105280.0,
  "rt_flops_morphism": 406847488.0,
  "rt_flops_ratio": 2.9768041237113403,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.7s",
  "morph_ind_verified": true,
  "morph_ind_rel": 6.638900357069024e-16
}
```

</details>

<details><summary><b>resid_x4@8x16</b> — raw metrics</summary>

```json
{
  "name": "resid_x4@8x16",
  "params": {
    "fixture": "resid_x4",
    "rows": 8,
    "dim": 16
  },
  "param_bytes_eager": 24576,
  "opt_s": 0.46,
  "fires": {
    "residual_reassoc": 1
  },
  "n_rewritten": 4,
  "e2e_rel": 4.0525399812299945e-16,
  "n_grafted": 1,
  "matches": {
    "residual_reassoc:blocks.0+blocks.1+blocks.2+blocks.3": {
      "status": "grafted",
      "boundary": "residual_wrapped+residual_wrapped+residual_wrapped",
      "cost_before": 50176.0,
      "cost_after": 17024.0,
      "rel_diff": 4.0525399812299945e-16
    }
  },
  "term_flops_before": 50176.0,
  "term_flops_after": 17024.0,
  "flops_ratio": 2.9473684210526314,
  "param_bytes_opt": 8192,
  "morph_verified": true,
  "morph_rel": 4.0525399812299945e-16,
  "rt_flops_eager": 50176.0,
  "rt_flops_morphism": 17920.0,
  "rt_flops_ratio": 2.8,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.9s",
  "morph_ind_verified": true,
  "morph_ind_rel": 4.0525399812299945e-16
}
```

</details>

<details><summary><b>resid_x4@1024x64</b> — raw metrics</summary>

```json
{
  "name": "resid_x4@1024x64",
  "params": {
    "fixture": "resid_x4",
    "rows": 1024,
    "dim": 64
  },
  "param_bytes_eager": 393216,
  "opt_s": 0.49,
  "fires": {
    "residual_reassoc": 1
  },
  "n_rewritten": 4,
  "e2e_rel": 4.585214510396509e-16,
  "n_grafted": 1,
  "matches": {
    "residual_reassoc:blocks.0+blocks.1+blocks.2+blocks.3": {
      "status": "grafted",
      "boundary": "residual_wrapped+residual_wrapped+residual_wrapped",
      "cost_before": 101187584.0,
      "cost_after": 33882112.0,
      "rel_diff": 4.585214510396509e-16
    }
  },
  "term_flops_before": 101187584.0,
  "term_flops_after": 33882112.0,
  "flops_ratio": 2.986460348162476,
  "param_bytes_opt": 131072,
  "morph_verified": true,
  "morph_rel": 4.585214510396509e-16,
  "rt_flops_eager": 101187584.0,
  "rt_flops_morphism": 34340864.0,
  "rt_flops_ratio": 2.946564885496183,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.9s",
  "morph_ind_verified": true,
  "morph_ind_rel": 4.585214510396509e-16
}
```

</details>

<details><summary><b>resid_x4@4096x128</b> — raw metrics</summary>

```json
{
  "name": "resid_x4@4096x128",
  "params": {
    "fixture": "resid_x4",
    "rows": 4096,
    "dim": 128
  },
  "param_bytes_eager": 1572864,
  "opt_s": 0.66,
  "fires": {
    "residual_reassoc": 1
  },
  "n_rewritten": 4,
  "e2e_rel": 8.060802582569976e-16,
  "n_grafted": 1,
  "matches": {
    "residual_reassoc:blocks.0+blocks.1+blocks.2+blocks.3": {
      "status": "grafted",
      "boundary": "residual_wrapped+residual_wrapped+residual_wrapped",
      "cost_before": 1614807040.0,
      "cost_after": 539492352.0,
      "rel_diff": 8.060802582569976e-16
    }
  },
  "term_flops_before": 1614807040.0,
  "term_flops_after": 539492352.0,
  "flops_ratio": 2.993197278911565,
  "param_bytes_opt": 524288,
  "morph_verified": true,
  "morph_rel": 8.060802582569976e-16,
  "rt_flops_eager": 1614807040.0,
  "rt_flops_morphism": 543162368.0,
  "rt_flops_ratio": 2.972972972972973,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.9s",
  "morph_ind_verified": true,
  "morph_ind_rel": 8.060802582569976e-16
}
```

</details>

<details><summary><b>kv_x2@8x16</b> — raw metrics</summary>

```json
{
  "name": "kv_x2@8x16",
  "params": {
    "fixture": "kv_x2",
    "rows": 8,
    "dim": 16
  },
  "param_bytes_eager": 8192,
  "opt_s": 0.05,
  "fires": {
    "kv_latent_share": 1
  },
  "n_rewritten": 2,
  "e2e_rel": 2.5058835729339943e-14,
  "n_grafted": 1,
  "matches": {
    "kv_latent_share:blocks.0+blocks.1": {
      "status": "grafted",
      "boundary": "family",
      "cost_before": 21120.0,
      "cost_after": 16000.0,
      "rel_diff": 2.5058835729339943e-14,
      "latent_rank": 4,
      "n_kv_sites": 4,
      "kv_flops_before": 8192.0,
      "kv_flops_after": 3072.0,
      "kv_bytes_before": 4096.0,
      "kv_bytes_after": 1536.0
    },
    "kv_latent_share:blocks.0": {
      "status": "skipped",
      "reason": "node already rewritten"
    },
    "kv_latent_share:blocks.1": {
      "status": "skipped",
      "reason": "node already rewritten"
    }
  },
  "term_flops_before": 21120.0,
  "term_flops_after": 16000.0,
  "flops_ratio": 1.32,
  "kv_flops_before": 8192.0,
  "kv_flops_after": 3072.0,
  "kv_flops_ratio": 2.6666666666666665,
  "kv_bytes_before": 4096.0,
  "kv_bytes_after": 1536.0,
  "param_bytes_opt": 5632,
  "morph_verified": true,
  "morph_rel": 2.5058835729339943e-14,
  "rt_flops_eager": 16512.0,
  "rt_flops_morphism": 11648.0,
  "rt_flops_ratio": 1.4175824175824177,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.8s",
  "morph_ind_verified": true,
  "morph_ind_rel": 2.5058835729339943e-14
}
```

</details>

<details><summary><b>kv_x2@512x64</b> — raw metrics</summary>

```json
{
  "name": "kv_x2@512x64",
  "params": {
    "fixture": "kv_x2",
    "rows": 512,
    "dim": 64
  },
  "param_bytes_eager": 131072,
  "opt_s": 0.07,
  "fires": {
    "kv_latent_share": 1
  },
  "n_rewritten": 2,
  "e2e_rel": 9.398685399150736e-15,
  "n_grafted": 1,
  "matches": {
    "kv_latent_share:blocks.0+blocks.1": {
      "status": "grafted",
      "boundary": "family",
      "cost_before": 84049920.0,
      "cost_after": 78807040.0,
      "rel_diff": 9.398685399150736e-15,
      "latent_rank": 16,
      "n_kv_sites": 4,
      "kv_flops_before": 8388608.0,
      "kv_flops_after": 3145728.0,
      "kv_bytes_before": 65536.0,
      "kv_bytes_after": 24576.0
    },
    "kv_latent_share:blocks.0": {
      "status": "skipped",
      "reason": "node already rewritten"
    },
    "kv_latent_share:blocks.1": {
      "status": "skipped",
      "reason": "node already rewritten"
    }
  },
  "term_flops_before": 84049920.0,
  "term_flops_after": 78807040.0,
  "flops_ratio": 1.0665280665280665,
  "kv_flops_before": 8388608.0,
  "kv_flops_after": 3145728.0,
  "kv_flops_ratio": 2.6666666666666665,
  "kv_bytes_before": 65536.0,
  "kv_bytes_after": 24576.0,
  "param_bytes_opt": 90112,
  "morph_verified": true,
  "morph_rel": 9.398685399150736e-15,
  "rt_flops_eager": 16809984.0,
  "rt_flops_morphism": 11632640.0,
  "rt_flops_ratio": 1.4450704225352113,
  "inductor_status": "compiled in 0.2s",
  "inductor_verified": true,
  "inductor_rel": 0.0,
  "morph_ind_status": "compiled in 0.9s",
  "morph_ind_verified": true,
  "morph_ind_rel": 9.398685399150736e-15
}
```

</details>

