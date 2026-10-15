# Certified bounded rewrites on a checkpoint

search(error_budget=B) sweep on stories15M: speedup vs bound vs held-out KL / top-k drift — including the honest negative when the gate declines every candidate.

- **when**: 2026-10-14T06:54:12+00:00 (git `2c782dd` dirty)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | bounded rewrites (error_budget) buy wall-time on a real checkpoint | 4/5 budgets delivered a bounded member; best 1.410× vs eager | budgets delivering = 4 |
| **WIN** | every budget's optimized module stays within the fp32 gate | 5/5 budgets optimized cleanly | clean budgets = 5 |

## Results

| case | budget | seq | eager ms | inductor ms | catopt ms |
|---|---|---|---|---|---|
| budget=none | none | 128 | 41.38 ± 0.7466 | 39.04 ± 0.5267 | 41.14 ± 2.483 |
| budget=1e-4 | 1e-4 | 128 | 46.3 ± 3.444 | 40.16 ± 2.283 | 34.58 ± 0.6992 |
| budget=1e-3 | 1e-3 | 128 | 41.47 ± 0.7874 | 39.3 ± 0 | 33.05 ± 1.084 |
| budget=3e-3 | 3e-3 | 128 | 42.66 ± 0.2719 | 39.91 ± 0.6299 | 31.94 ± 0.1763 |
| budget=1e-2 | 1e-2 | 128 | 42.44 ± 0.1186 | 40.42 ± 0.2004 | 30.1 ± 0.3883 |
| morphism_probe | kv:0.01 | 128 | — | — | — |

## Metrics

<details><summary><b>budget=none</b> — raw metrics</summary>

```json
{
  "name": "budget=none",
  "params": {
    "budget": "none",
    "seq": 128
  },
  "optimize_device": "cpu",
  "opt_s": 13.8,
  "n_optimized": 8,
  "n_blocks": 8,
  "cache": {
    "hits": 5,
    "misses": 3,
    "fallbacks": 0
  },
  "e2e": {
    "max_abs_diff": 0.0,
    "max_rel_diff": 0.0
  },
  "per_block": {
    "emb": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 2.8,
      "bound_total": null,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.0": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.6,
      "bound_total": null,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.1": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": null,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.2": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": null,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.3": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": null,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.4": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": null,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.5": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": null,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "head": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 10.0,
      "bound_total": null,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    }
  },
  "n_special_offers": 0,
  "n_bounded_offers": 0,
  "n_bounds_delivered": 0,
  "bound_total_max": 0.0,
  "fresh_rel": 0.0,
  "fresh_passed": true,
  "proxy": {
    "mean_kl": 0.0,
    "max_kl": 0.0,
    "top1_agree": 1.0,
    "top5_in": 1.0,
    "max_abs_logit": 0.0,
    "n_positions": 640
  },
  "ms_eager": 0.0413812209990283,
  "ms_inductor": 0.039036663500155555,
  "ms_catopt": 0.04113580899866065,
  "speedup_vs_eager": 1.0059658970211487,
  "speedup_vs_inductor": 0.9489703606273201,
  "verdict": "no bounded offer even qualified \u2014 no weight is within budget of a structural special"
}
```

</details>

<details><summary><b>budget=1e-4</b> — raw metrics</summary>

```json
{
  "name": "budget=1e-4",
  "params": {
    "budget": "1e-4",
    "seq": 128
  },
  "optimize_device": "cpu",
  "opt_s": 390.6,
  "n_optimized": 8,
  "n_blocks": 8,
  "cache": {
    "hits": 5,
    "misses": 3,
    "fallbacks": 0
  },
  "e2e": {
    "max_abs_diff": 0.0009882450103759766,
    "max_rel_diff": 5.688349269330689e-05
  },
  "per_block": {
    "emb": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 2.8,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.0": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 3.0,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.1": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.2": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.3": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.4": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.5": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "head": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 5.688349269330689e-05,
      "error": null,
      "time_s": 384.4,
      "bound_total": 2.364814281463623e-05,
      "bounds": [
        {
          "rule": "weight_special#5",
          "bound": 2.364814281463623e-05,
          "norm": "max_abs",
          "output_bound": 0.03869175067302422,
          "site_input_norm": 1636.143310546875,
          "measured_max_rel": 5.688349269330689e-05
        }
      ],
      "specials_offered": [
        {
          "param": "p_weight",
          "kind": "elide_bounded",
          "error_bound": 2.364814281463623e-05
        }
      ],
      "bounds_honored": true
    }
  },
  "n_special_offers": 1,
  "n_bounded_offers": 1,
  "n_bounds_delivered": 1,
  "bound_total_max": 2.364814281463623e-05,
  "fresh_rel": 4.977673479619482e-05,
  "fresh_passed": true,
  "proxy": {
    "mean_kl": -1.077529077520012e-06,
    "max_kl": 4.9594284064369276e-05,
    "top1_agree": 1.0,
    "top5_in": 1.0,
    "max_abs_logit": 0.0015254020690917969,
    "n_positions": 640
  },
  "ms_eager": 0.04630026899940276,
  "ms_inductor": 0.040158808000342106,
  "ms_catopt": 0.03458129250066122,
  "speedup_vs_eager": 1.3388819691605647,
  "speedup_vs_inductor": 1.1612870744948078,
  "verdict": "1 bounded members, bound 2.36e-05, speedup 1.339x, KL -1.08e-06"
}
```

</details>

<details><summary><b>budget=1e-3</b> — raw metrics</summary>

```json
{
  "name": "budget=1e-3",
  "params": {
    "budget": "1e-3",
    "seq": 128
  },
  "optimize_device": "cpu",
  "opt_s": 334.3,
  "n_optimized": 8,
  "n_blocks": 8,
  "cache": {
    "hits": 5,
    "misses": 3,
    "fallbacks": 0
  },
  "e2e": {
    "max_abs_diff": 0.11743414402008057,
    "max_rel_diff": 0.006759522388855351
  },
  "per_block": {
    "emb": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 2.8,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.0": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 3.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.1": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.2": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.3": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.4": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.5": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "head": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.006759522388855351,
      "error": null,
      "time_s": 328.0,
      "bound_total": 0.0009999144822359085,
      "bounds": [
        {
          "rule": "weight_special#5",
          "bound": 0.0009999144822359085,
          "norm": "max_abs",
          "output_bound": 1.6360033912292238,
          "site_input_norm": 1636.143310546875,
          "measured_max_rel": 0.006759522388855351
        }
      ],
      "specials_offered": [
        {
          "param": "p_weight",
          "kind": "elide_bounded",
          "error_bound": 0.0009999144822359085
        }
      ],
      "bounds_honored": true
    }
  },
  "n_special_offers": 1,
  "n_bounded_offers": 1,
  "n_bounds_delivered": 1,
  "bound_total_max": 0.0009999144822359085,
  "fresh_rel": 0.006790873643222583,
  "fresh_passed": false,
  "proxy": {
    "mean_kl": 5.8475548030401114e-06,
    "max_kl": 5.918855458730832e-05,
    "top1_agree": 1.0,
    "top5_in": 1.0,
    "max_abs_logit": 0.1190800666809082,
    "n_positions": 640
  },
  "ms_eager": 0.041472091999821714,
  "ms_inductor": 0.0392984363999858,
  "ms_catopt": 0.03305221399932634,
  "speedup_vs_eager": 1.25474475025083,
  "speedup_vs_inductor": 1.1889804537991546,
  "verdict": "1 bounded members, bound 1.00e-03, speedup 1.255x, KL 5.85e-06"
}
```

</details>

<details><summary><b>budget=3e-3</b> — raw metrics</summary>

```json
{
  "name": "budget=3e-3",
  "params": {
    "budget": "3e-3",
    "seq": 128
  },
  "optimize_device": "cpu",
  "opt_s": 238.1,
  "n_optimized": 8,
  "n_blocks": 8,
  "cache": {
    "hits": 5,
    "misses": 3,
    "fallbacks": 0
  },
  "e2e": {
    "max_abs_diff": 0.3607311248779297,
    "max_rel_diff": 0.020763723662450283
  },
  "per_block": {
    "emb": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 2.7,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.0": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 3.0,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.1": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.2": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.3": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.4": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.5": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "head": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.020763723662450283,
      "error": null,
      "time_s": 231.9,
      "bound_total": 0.0029997825622558594,
      "bounds": [
        {
          "rule": "weight_special#5",
          "bound": 0.0029997825622558594,
          "norm": "max_abs",
          "output_bound": 4.908074172330089,
          "site_input_norm": 1636.143310546875,
          "measured_max_rel": 0.020763723662450283
        }
      ],
      "specials_offered": [
        {
          "param": "p_weight",
          "kind": "elide_bounded",
          "error_bound": 0.0029997825622558594
        }
      ],
      "bounds_honored": true
    }
  },
  "n_special_offers": 1,
  "n_bounded_offers": 1,
  "n_bounds_delivered": 1,
  "bound_total_max": 0.0029997825622558594,
  "fresh_rel": 0.0226738941468024,
  "fresh_passed": false,
  "proxy": {
    "mean_kl": 6.501974712591618e-05,
    "max_kl": 0.00020665681222453713,
    "top1_agree": 1.0,
    "top5_in": 1.0,
    "max_abs_logit": 0.39759373664855957,
    "n_positions": 640
  },
  "ms_eager": 0.04265929800021695,
  "ms_inductor": 0.039906721000079415,
  "ms_catopt": 0.031944913000188535,
  "speedup_vs_eager": 1.3354019151645578,
  "speedup_vs_inductor": 1.249235551207915,
  "verdict": "1 bounded members, bound 3.00e-03, speedup 1.335x, KL 6.50e-05"
}
```

</details>

<details><summary><b>budget=1e-2</b> — raw metrics</summary>

```json
{
  "name": "budget=1e-2",
  "params": {
    "budget": "1e-2",
    "seq": 128
  },
  "optimize_device": "cpu",
  "opt_s": 149.7,
  "n_optimized": 8,
  "n_blocks": 8,
  "cache": {
    "hits": 5,
    "misses": 3,
    "fallbacks": 0
  },
  "e2e": {
    "max_abs_diff": 1.4378125667572021,
    "max_rel_diff": 0.08276065123198757
  },
  "per_block": {
    "emb": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 2.8,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.0": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 3.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.1": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.2": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.3": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.4": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "blocks.5": {
      "status": "optimized",
      "cache": "hit",
      "rel_diff": 0.0,
      "error": null,
      "time_s": 0.1,
      "bound_total": 0.0,
      "bounds": [],
      "specials_offered": [],
      "bounds_honored": null
    },
    "head": {
      "status": "optimized",
      "cache": "miss",
      "rel_diff": 0.08276065123198757,
      "error": null,
      "time_s": 143.4,
      "bound_total": 0.009999886155128479,
      "bounds": [
        {
          "rule": "weight_special#5",
          "bound": 0.009999886155128479,
          "norm": "max_abs",
          "output_bound": 16.36124683894377,
          "site_input_norm": 1636.143310546875,
          "measured_max_rel": 0.08276065123198757
        }
      ],
      "specials_offered": [
        {
          "param": "p_weight",
          "kind": "elide_bounded",
          "error_bound": 0.009999886155128479
        }
      ],
      "bounds_honored": true
    }
  },
  "n_special_offers": 1,
  "n_bounded_offers": 1,
  "n_bounds_delivered": 1,
  "bound_total_max": 0.009999886155128479,
  "fresh_rel": 0.08074912172973428,
  "fresh_passed": false,
  "proxy": {
    "mean_kl": 0.000711094297003001,
    "max_kl": 0.0030438390094786882,
    "top1_agree": 1.0,
    "top5_in": 1.0,
    "max_abs_logit": 2.1795692443847656,
    "n_positions": 640
  },
  "ms_eager": 0.04244444799951452,
  "ms_inductor": 0.040424983999400865,
  "ms_catopt": 0.030104414001471014,
  "speedup_vs_eager": 1.4099077961604078,
  "speedup_vs_inductor": 1.342825806123499,
  "verdict": "1 bounded members, bound 1.00e-02, speedup 1.410x, KL 7.11e-04"
}
```

</details>

<details><summary><b>morphism_probe</b> — raw metrics</summary>

```json
{
  "error_budget_kw": "accepted",
  "law_budget_armed": true,
  "kv_budget": 0.01,
  "opt_s": 9.1,
  "n_blocks": 8,
  "n_lifted": 8,
  "n_rewritten": 0,
  "matches": {
    "kv_latent_share:blocks.0": {
      "status": "declined",
      "reason": "no_improvement",
      "detail": "blocks.0: shared-data kv projections",
      "error": null
    },
    "kv_latent_share:blocks.1": {
      "status": "declined",
      "reason": "no_improvement",
      "detail": "blocks.1: shared-data kv projections",
      "error": null
    },
    "kv_latent_share:blocks.2": {
      "status": "declined",
      "reason": "no_improvement",
      "detail": "blocks.2: shared-data kv projections",
      "error": null
    },
    "kv_latent_share:blocks.3": {
      "status": "declined",
      "reason": "no_improvement",
      "detail": "blocks.3: shared-data kv projections",
      "error": null
    },
    "kv_latent_share:blocks.4": {
      "status": "declined",
      "reason": "no_improvement",
      "detail": "blocks.4: shared-data kv projections",
      "error": null
    },
    "kv_latent_share:blocks.5": {
      "status": "declined",
      "reason": "no_improvement",
      "detail": "blocks.5: shared-data kv projections",
      "error": null
    }
  },
  "end_to_end": {
    "max_abs_diff": 0.0,
    "max_rel_diff": 0.0
  }
}
```

</details>

