# MiniGPT end-to-end

Whole-model optimize + verify: pairing fires per block and the composed module is timed against eager and Inductor.

- **when**: 2026-10-14T06:33:13+00:00 (git `2c782dd` dirty)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | the composed catopt+Inductor module beats plain Inductor end-to-end | best 1.06× vs Inductor on tiny_llama@4x128 | inductor / catopt+inductor = 1.05918 |
| **WIN** | the optimized module verifies against eager | 5/5 cells verified | verified cells = 5 |

## Results

| case | B | T | model | HW | eager ms | inductor ms | catopt ms | catopt+inductor ms | catopt_full ms | catopt_full+ind ms |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny_llama@4x64 | 4 | 64 | tiny_llama | — | 17.25 ± 0 | 15.85 ± 0.3991 | 18.69 ± 0 | 16.81 ± 0.1999 | — | — |
| tiny_llama@4x128 | 4 | 128 | tiny_llama | — | 39.56 ± 1.028 | 35.79 ± 0.9331 | 38.59 ± 1.08 | 33.79 ± 0.7428 | — | — |
| tiny_llama@4x256 | 4 | 256 | tiny_llama | — | 82.34 ± 0 | 81.98 ± 0 | 85.28 ± 0 | 81.84 ± 0 | — | — |
| tiny_llama@8x128 | 8 | 128 | tiny_llama | — | 70.28 ± 0 | 65.52 ± 0.8055 | 78.33 ± 0 | 63.57 ± 0.8868 | — | — |
| conv_attn@4x16 | 4 | 256 | conv_attn | 16 | 5.409 ± 0 | 5.415 ± 0.06156 | 5.538 ± 0.06683 | 5.285 ± 0.08872 | 5.679 ± 0 | 5.338 ± 0.07326 |

## Metrics

<details><summary><b>tiny_llama@4x64</b> — raw metrics</summary>

```json
{
  "name": "tiny_llama@4x64",
  "params": {
    "B": 4,
    "T": 64,
    "model": "tiny_llama"
  },
  "n_tokens": 256,
  "n_params": 3344640,
  "opt_s": 4.01,
  "n_blocks": 6,
  "n_optimized": 6,
  "n_failed": 0,
  "catopt_e2e_rel": 4.899089864408092e-07,
  "blocks": {
    "tok_emb": {
      "status": "optimized",
      "time_s": 2.82,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    },
    "blocks.0": {
      "status": "optimized",
      "time_s": 0.82,
      "rel": 1.1216208202074244e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.1": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 1.1103039175758379e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.2": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 1.1543414523314357e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.3": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 1.1153243080483312e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "head": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    }
  },
  "pairing_groups_total": 8,
  "paired_blocks": 4,
  "nonlocal_lifts_total": 4,
  "fires_total": {
    "sub_to_add": 8,
    "silu_expand": 4,
    "mul_unsq_pad_r": 64,
    "decay_state_lift": 16,
    "decay_state_lift_post": 8,
    "repeat_kv_as_gather": 8,
    "gather_cat_out": 4,
    "index_select_dedup": 1448,
    "gather_cat_in": 352,
    "index_select_id": 1264
  },
  "catopt_rel": 4.899089864408092e-07,
  "catopt_verified": true,
  "catopt_fp64_rel": 8.295695093788756e-16,
  "opt64_s": 1.38,
  "n64_optimized": 6,
  "opt_full_s": 120.0,
  "full_status": "TimeoutError: exceeded 120s budget",
  "inductor_status": "compiled in 1.6s",
  "inductor_rel": 4.899089864408092e-07,
  "inductor_verified": true,
  "catopt_ind_status": "compiled in 0.7s",
  "catopt_ind_rel": 4.453718058552811e-07,
  "catopt_ind_verified": true,
  "tok_s_eager": 14841.2,
  "tok_s_inductor": 16151.6,
  "tok_s_catopt": 13700.4,
  "tok_s_catopt+inductor": 15231.5
}
```

</details>

<details><summary><b>tiny_llama@4x128</b> — raw metrics</summary>

```json
{
  "name": "tiny_llama@4x128",
  "params": {
    "B": 4,
    "T": 128,
    "model": "tiny_llama"
  },
  "n_tokens": 512,
  "n_params": 3344640,
  "opt_s": 1.36,
  "n_blocks": 6,
  "n_optimized": 6,
  "n_failed": 0,
  "catopt_e2e_rel": 5.365598319266025e-07,
  "blocks": {
    "tok_emb": {
      "status": "optimized",
      "time_s": 0.07,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    },
    "blocks.0": {
      "status": "optimized",
      "time_s": 0.89,
      "rel": 1.0072871157980956e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.1": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 1.505602494284003e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.2": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 1.5131275891648176e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.3": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 1.2186151729720695e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "head": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    }
  },
  "pairing_groups_total": 8,
  "paired_blocks": 4,
  "nonlocal_lifts_total": 4,
  "fires_total": {
    "sub_to_add": 8,
    "silu_expand": 4,
    "mul_unsq_pad_r": 64,
    "decay_state_lift": 16,
    "decay_state_lift_post": 8,
    "repeat_kv_as_gather": 8,
    "gather_cat_out": 4,
    "index_select_dedup": 1448,
    "gather_cat_in": 352,
    "index_select_id": 1264
  },
  "catopt_rel": 5.365598319266025e-07,
  "catopt_verified": true,
  "catopt_fp64_rel": 9.559680113133645e-16,
  "opt64_s": 1.59,
  "n64_optimized": 6,
  "opt_full_s": 120.0,
  "full_status": "TimeoutError: exceeded 120s budget",
  "inductor_status": "compiled in 0.4s",
  "inductor_rel": 4.432450785480629e-07,
  "inductor_verified": true,
  "catopt_ind_status": "compiled in 0.8s",
  "catopt_ind_rel": 4.6657376689269785e-07,
  "catopt_ind_verified": true,
  "tok_s_eager": 12943.0,
  "tok_s_inductor": 14304.9,
  "tok_s_catopt": 13266.2,
  "tok_s_catopt+inductor": 15151.4
}
```

</details>

<details><summary><b>tiny_llama@4x256</b> — raw metrics</summary>

```json
{
  "name": "tiny_llama@4x256",
  "params": {
    "B": 4,
    "T": 256,
    "model": "tiny_llama"
  },
  "n_tokens": 1024,
  "n_params": 3344640,
  "opt_s": 1.49,
  "n_blocks": 6,
  "n_optimized": 6,
  "n_failed": 0,
  "catopt_e2e_rel": 4.115913248197901e-07,
  "blocks": {
    "tok_emb": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    },
    "blocks.0": {
      "status": "optimized",
      "time_s": 0.9,
      "rel": 1.2901145614450951e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.1": {
      "status": "optimized",
      "time_s": 0.09,
      "rel": 1.0356248720517957e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.2": {
      "status": "optimized",
      "time_s": 0.09,
      "rel": 1.0431988627173801e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.3": {
      "status": "optimized",
      "time_s": 0.09,
      "rel": 9.841626556488292e-08,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "head": {
      "status": "optimized",
      "time_s": 0.09,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    }
  },
  "pairing_groups_total": 8,
  "paired_blocks": 4,
  "nonlocal_lifts_total": 4,
  "fires_total": {
    "sub_to_add": 8,
    "silu_expand": 4,
    "mul_unsq_pad_r": 64,
    "decay_state_lift": 16,
    "decay_state_lift_post": 8,
    "repeat_kv_as_gather": 8,
    "gather_cat_out": 4,
    "index_select_dedup": 1448,
    "gather_cat_in": 352,
    "index_select_id": 1264
  },
  "catopt_rel": 4.115913248197901e-07,
  "catopt_verified": true,
  "catopt_fp64_rel": 9.370149251370024e-16,
  "opt64_s": 1.88,
  "n64_optimized": 6,
  "opt_full_s": 120.0,
  "full_status": "TimeoutError: exceeded 120s budget",
  "inductor_status": "compiled in 6.7s",
  "inductor_rel": 5.487884330930535e-07,
  "inductor_verified": true,
  "catopt_ind_status": "compiled in 2.7s",
  "catopt_ind_rel": 5.487884330930535e-07,
  "catopt_ind_verified": true,
  "tok_s_eager": 12435.7,
  "tok_s_inductor": 12490.1,
  "tok_s_catopt": 12007.5,
  "tok_s_catopt+inductor": 12511.6
}
```

</details>

<details><summary><b>tiny_llama@8x128</b> — raw metrics</summary>

```json
{
  "name": "tiny_llama@8x128",
  "params": {
    "B": 8,
    "T": 128,
    "model": "tiny_llama"
  },
  "n_tokens": 1024,
  "n_params": 3344640,
  "opt_s": 1.38,
  "n_blocks": 6,
  "n_optimized": 6,
  "n_failed": 0,
  "catopt_e2e_rel": 4.695128633555541e-07,
  "blocks": {
    "tok_emb": {
      "status": "optimized",
      "time_s": 0.06,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    },
    "blocks.0": {
      "status": "optimized",
      "time_s": 0.84,
      "rel": 1.0122399015426426e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.1": {
      "status": "optimized",
      "time_s": 0.09,
      "rel": 1.2700878335201713e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.2": {
      "status": "optimized",
      "time_s": 0.09,
      "rel": 1.0329995827029208e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "blocks.3": {
      "status": "optimized",
      "time_s": 0.09,
      "rel": 1.1418136604814975e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "sub_to_add": 2,
        "silu_expand": 1,
        "mul_unsq_pad_r": 16,
        "decay_state_lift": 4,
        "decay_state_lift_post": 2,
        "repeat_kv_as_gather": 2,
        "gather_cat_out": 1,
        "index_select_dedup": 362,
        "gather_cat_in": 88,
        "index_select_id": 316
      },
      "error": null
    },
    "head": {
      "status": "optimized",
      "time_s": 0.08,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    }
  },
  "pairing_groups_total": 8,
  "paired_blocks": 4,
  "nonlocal_lifts_total": 4,
  "fires_total": {
    "sub_to_add": 8,
    "silu_expand": 4,
    "mul_unsq_pad_r": 64,
    "decay_state_lift": 16,
    "decay_state_lift_post": 8,
    "repeat_kv_as_gather": 8,
    "gather_cat_out": 4,
    "index_select_dedup": 1448,
    "gather_cat_in": 352,
    "index_select_id": 1264
  },
  "catopt_rel": 4.695128633555541e-07,
  "catopt_verified": true,
  "catopt_fp64_rel": 8.745359369862938e-16,
  "opt64_s": 1.76,
  "n64_optimized": 6,
  "opt_full_s": 120.0,
  "full_status": "TimeoutError: exceeded 120s budget",
  "inductor_status": "compiled in 3.3s",
  "inductor_rel": 4.695128633555541e-07,
  "inductor_verified": true,
  "catopt_ind_status": "compiled in 2.6s",
  "catopt_ind_rel": 4.695128633555541e-07,
  "catopt_ind_verified": true,
  "tok_s_eager": 14569.3,
  "tok_s_inductor": 15628.4,
  "tok_s_catopt": 13073.6,
  "tok_s_catopt+inductor": 16107.0
}
```

</details>

<details><summary><b>conv_attn@4x16</b> — raw metrics</summary>

```json
{
  "name": "conv_attn@4x16",
  "params": {
    "B": 4,
    "HW": 16,
    "model": "conv_attn",
    "T": 256
  },
  "n_tokens": 1024,
  "n_params": 122816,
  "opt_s": 0.3,
  "n_blocks": 7,
  "n_optimized": 7,
  "n_failed": 0,
  "catopt_e2e_rel": 1.4610119107366812e-07,
  "blocks": {
    "stem.0": {
      "status": "optimized",
      "time_s": 0.03,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    },
    "stem.1": {
      "status": "optimized",
      "time_s": 0.02,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    },
    "stem.2": {
      "status": "optimized",
      "time_s": 0.03,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    },
    "stem.3": {
      "status": "optimized",
      "time_s": 0.01,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    },
    "blocks.0": {
      "status": "optimized",
      "time_s": 0.11,
      "rel": 3.040681663854512e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "silu_expand": 1
      },
      "error": null
    },
    "blocks.1": {
      "status": "optimized",
      "time_s": 0.04,
      "rel": 1.771506427470302e-07,
      "paired": true,
      "groups": 2,
      "lifts": 1,
      "lowering": "generic",
      "fires": {
        "silu_expand": 1
      },
      "error": null
    },
    "head": {
      "status": "optimized",
      "time_s": 0.03,
      "rel": 0.0,
      "paired": null,
      "groups": null,
      "lifts": null,
      "lowering": "generic",
      "fires": null,
      "error": null
    }
  },
  "pairing_groups_total": 4,
  "paired_blocks": 2,
  "nonlocal_lifts_total": 2,
  "fires_total": {
    "silu_expand": 2
  },
  "catopt_rel": 1.4610119107366812e-07,
  "catopt_verified": true,
  "catopt_fp64_rel": 4.0820199794939e-16,
  "opt64_s": 0.4,
  "n64_optimized": 7,
  "opt_full_s": 0.15,
  "full_pairing_groups": 4,
  "full_paired_extract": true,
  "full_lifts": 1,
  "full_fires": {
    "silu_expand": 2
  },
  "full_rel": 1.4610119107366812e-07,
  "full_verified": true,
  "full_status": "ok",
  "inductor_status": "compiled in 3.5s",
  "inductor_rel": 3.652529776841703e-07,
  "inductor_verified": true,
  "catopt_ind_status": "compiled in 1.3s",
  "catopt_ind_rel": 3.8522774990127334e-07,
  "catopt_ind_verified": true,
  "full_ind_status": "compiled in 0.9s",
  "full_ind_rel": 3.8522774990127334e-07,
  "tok_s_eager": 189318.9,
  "tok_s_inductor": 189109.3,
  "tok_s_catopt": 184919.0,
  "tok_s_catopt+inductor": 193745.5,
  "tok_s_catopt_full": 180314.7,
  "tok_s_catopt_full+ind": 191835.1
}
```

</details>

