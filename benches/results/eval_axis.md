# Evaluation-axis selection

For each family: the root-class Pareto frontier over every pair of built-in axes, and the program a compute- / bandwidth- / launch-bound PredictedCriterion extracts (each pick certified by lower -> verify).

- **when**: 2026-10-03T03:10:46+00:00 (git `ecfbf50` dirty)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `families`=['chain3', 'rect', 'swiglu', 'qkv_sum', 'linattn', 'deep_parallel', 'parallel8'] · `axes`=['flops', 'launch', 'count', 'depth', 'peak', 'params', 'roofline'] · `targets`={'compute': 'synth-compute-bound', 'bandwidth': 'synth-bandwidth-bound', 'launch': 'synth-launch-bound'} · `genuine_gap`=0.01 · `device`=cpu · `quick`=True

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **NEGATIVE** | a bandwidth-bound target extracts a different certified program than the compute / launch targets | 0/7 families: the bandwidth pick is distinct from both the compute and launch picks nowhere — it equals the launch pick in every family.  The three targets disagree among themselves in only 1/7 (swiglu); 2/7 move at all against the default, and every pick verified=True | families where the bandwidth pick is distinct from both the compute and launch picks = 0 |
| **NEGATIVE** | the root-class Pareto frontier exposes a genuine multi-axis trade-off | 0 genuine frontiers over 7 families x 21 axis pairs — the 6 >1 results are 1 exact tie(s) and 5 float-noise pair(s) | genuine frontiers = 0 |
| **NEGATIVE** | the residual target-sensitivity traces to a hardware-genuine trade-off | No — the true roofline value (model.predict(features, target)) ranks the same form first under all three targets in 7/7 families, so the move is the additive marginal decomposition extract_best applies to the non-additive PredictedCriterion, not hardware | families where the true roofline argmin is target-invariant = 7 |

## Results

| case | family | n_root_alts | default ms | compute ms | bandwidth ms | launch ms |
|---|---|---|---|---|---|---|
| chain3 | chain3 | 5 | 18.9 ± 0 | 19.49 ± 0 | 19.56 ± 0 | 20.39 ± 0 |
| rect | rect | 3 | 18 ± 0 | 18.11 ± 0 | 18.13 ± 0 | 18.06 ± 0 |
| swiglu | swiglu | 1 | 23.68 ± 0 | 25.61 ± 0 | 25.05 ± 0 | 25.31 ± 0 |
| qkv_sum | qkv_sum | 3 | 20.67 ± 0 | 21.7 ± 0 | 21.61 ± 0 | 21.33 ± 0 |
| linattn | linattn | 2 | 23.71 ± 0 | 26.82 ± 0 | 27.12 ± 0 | 25.74 ± 0 |
| deep_parallel | deep_parallel | 4 | 25.51 ± 0 | 27.65 ± 0 | 27.35 ± 0 | 27.43 ± 0 |
| parallel8 | parallel8 | 3 | 44.06 ± 0 | 53.46 ± 0 | 52.16 ± 0 | 53.18 ± 0 |

## Metrics

<details><summary><b>chain3</b> — raw metrics</summary>

```json
{
  "family": "chain3",
  "note": "three-deep linear chain",
  "n_root_alts": 5,
  "default": {
    "sig": "(linear x, (matmul W, (matmul W, W)))",
    "flops": 65536.0,
    "read": 18432.0,
    "write": 2048.0,
    "traffic": 20480.0,
    "ops": 3,
    "marginal": {
      "compute": 6.6036e-05,
      "bandwidth": 0.00041009475711999995,
      "launch": 0.005000000327679999
    },
    "true": {
      "compute": 0.001115612,
      "bandwidth": 0.00041109999999999996,
      "launch": 0.015000005570559999
    }
  },
  "picks": {
    "compute": {
      "sig": "(linear x, (matmul W, (matmul W, W)))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.6036e-05,
        "bandwidth": 0.00041009475711999995,
        "launch": 0.005000000327679999
      },
      "true": {
        "compute": 0.001115612,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000005570559999
      },
      "verified": true,
      "max_rel": 3.4304855476806876e-07
    },
    "bandwidth": {
      "sig": "(linear x, (matmul W, (matmul W, W)))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.6036e-05,
        "bandwidth": 0.00041009475711999995,
        "launch": 0.005000000327679999
      },
      "true": {
        "compute": 0.001115612,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000005570559999
      },
      "verified": true,
      "max_rel": 3.4304855476806876e-07
    },
    "launch": {
      "sig": "(linear x, (matmul W, (matmul W, W)))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.6036e-05,
        "bandwidth": 0.00041009475711999995,
        "launch": 0.005000000327679999
      },
      "true": {
        "compute": 0.001115612,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000005570559999
      },
      "verified": true,
      "max_rel": 3.4304855476806876e-07
    }
  },
  "distinct_certified": 1,
  "frontiers": [
    {
      "axes": [
        "flops",
        "depth"
      ],
      "size": 2,
      "kind": "noise",
      "vectors": [
        {
          "flops": 65536.0,
          "depth": 8930.112359550563
        },
        {
          "flops": 131072.0,
          "depth": 8930.112359550561
        }
      ]
    },
    {
      "axes": [
        "launch",
        "depth"
      ],
      "size": 2,
      "kind": "noise",
      "vectors": [
        {
          "launch": 65537.0,
          "depth": 8930.112359550563
        },
        {
          "launch": 131074.0,
          "depth": 8930.112359550561
        }
      ]
    },
    {
      "axes": [
        "count",
        "depth"
      ],
      "size": 2,
      "kind": "noise",
      "vectors": [
        {
          "count": 1.0,
          "depth": 8930.112359550563
        },
        {
          "count": 2.0,
          "depth": 8930.112359550561
        }
      ]
    },
    {
      "axes": [
        "depth",
        "params"
      ],
      "size": 2,
      "kind": "noise",
      "vectors": [
        {
          "depth": 8930.112359550563,
          "params": 4096.0
        },
        {
          "depth": 8930.112359550561,
          "params": 8192.0
        }
      ]
    },
    {
      "axes": [
        "depth",
        "roofline"
      ],
      "size": 2,
      "kind": "noise",
      "vectors": [
        {
          "depth": 8930.112359550563,
          "roofline": 8930.112359550563
        },
        {
          "depth": 8930.112359550561,
          "roofline": 17860.224719101127
        }
      ]
    }
  ]
}
```

</details>

<details><summary><b>rect</b> — raw metrics</summary>

```json
{
  "family": "rect",
  "note": "rectangular two-chain (k > d)",
  "n_root_alts": 3,
  "default": {
    "sig": "(linear x, (matmul W, W))",
    "flops": 65536.0,
    "read": 18432.0,
    "write": 2048.0,
    "traffic": 20480.0,
    "ops": 2,
    "marginal": {
      "compute": 6.603599999999979e-05,
      "bandwidth": 0.00041008951424,
      "launch": 0.0050000003276799996
    },
    "true": {
      "compute": 0.002163688,
      "bandwidth": 0.0004106,
      "launch": 0.01000001081344
    }
  },
  "picks": {
    "compute": {
      "sig": "(linear x, (matmul W, W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 2,
      "marginal": {
        "compute": 6.603599999999979e-05,
        "bandwidth": 0.00041008951424,
        "launch": 0.0050000003276799996
      },
      "true": {
        "compute": 0.002163688,
        "bandwidth": 0.0004106,
        "launch": 0.01000001081344
      },
      "verified": true,
      "max_rel": 3.128532810277264e-07
    },
    "bandwidth": {
      "sig": "(linear x, (matmul W, W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 2,
      "marginal": {
        "compute": 6.603599999999979e-05,
        "bandwidth": 0.00041008951424,
        "launch": 0.0050000003276799996
      },
      "true": {
        "compute": 0.002163688,
        "bandwidth": 0.0004106,
        "launch": 0.01000001081344
      },
      "verified": true,
      "max_rel": 3.128532810277264e-07
    },
    "launch": {
      "sig": "(linear x, (matmul W, W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 2,
      "marginal": {
        "compute": 6.603599999999979e-05,
        "bandwidth": 0.00041008951424,
        "launch": 0.0050000003276799996
      },
      "true": {
        "compute": 0.002163688,
        "bandwidth": 0.0004106,
        "launch": 0.01000001081344
      },
      "verified": true,
      "max_rel": 3.128532810277264e-07
    }
  },
  "distinct_certified": 1,
  "frontiers": []
}
```

</details>

<details><summary><b>swiglu</b> — raw metrics</summary>

```json
{
  "family": "swiglu",
  "note": "SwiGLU MLP (fusable gate/up pair)",
  "n_root_alts": 1,
  "default": {
    "sig": "(linear (mul (silu (split (linear x, (concat W, W, dim=0)), dim=-1, index=0, sizes=(256, 256))), (split (linear x, (concat W, W, dim=0)), dim=-1, index=1, sizes=(256, 256))), W)",
    "flops": 794624.0,
    "read": 231424.0,
    "write": 34816.0,
    "traffic": 266240.0,
    "ops": 7,
    "marginal": {
      "compute": 0.000795076,
      "bandwidth": 0.00483578,
      "launch": 0.025000048332800003
    },
    "true": {
      "compute": 0.000798124,
      "bandwidth": 0.0053283,
      "launch": 0.035000053248000006
    }
  },
  "picks": {
    "compute": {
      "sig": "(linear (mul (mul (split (linear x, (concat W, W, dim=0)), dim=-1, index=0, sizes=(256, 256)), (sigmoid (split (linear x, (concat W, W, dim=0)), dim=-1, index=0, sizes=(256, 256)))), (split (linear x, (concat W, W, dim=0)), dim=-1, index=1, sizes=(256, 256))), W)",
      "flops": 794624.0,
      "read": 247808.0,
      "write": 43008.0,
      "traffic": 290816.0,
      "ops": 8,
      "marginal": {
        "compute": 0.000793028,
        "bandwidth": 0.00483578,
        "launch": 0.025000048332799993
      },
      "true": {
        "compute": 0.000798624,
        "bandwidth": 0.0058203199999999995,
        "launch": 0.0400000581632
      },
      "verified": true,
      "max_rel": 1.6905358586967501e-07
    },
    "bandwidth": {
      "sig": "(linear (mul (silu (split (linear x, (concat W, W, dim=0)), dim=-1, index=0, sizes=(256, 256))), (split (linear x, (concat W, W, dim=0)), dim=-1, index=1, sizes=(256, 256))), W)",
      "flops": 794624.0,
      "read": 231424.0,
      "write": 34816.0,
      "traffic": 266240.0,
      "ops": 7,
      "marginal": {
        "compute": 0.000795076,
        "bandwidth": 0.00483578,
        "launch": 0.025000048332800003
      },
      "true": {
        "compute": 0.000798124,
        "bandwidth": 0.0053283,
        "launch": 0.035000053248000006
      },
      "verified": true,
      "max_rel": 0.0
    },
    "launch": {
      "sig": "(linear (mul (silu (split (linear x, (concat W, W, dim=0)), dim=-1, index=0, sizes=(256, 256))), (split (linear x, (concat W, W, dim=0)), dim=-1, index=1, sizes=(256, 256))), W)",
      "flops": 794624.0,
      "read": 231424.0,
      "write": 34816.0,
      "traffic": 266240.0,
      "ops": 7,
      "marginal": {
        "compute": 0.000795076,
        "bandwidth": 0.00483578,
        "launch": 0.025000048332800003
      },
      "true": {
        "compute": 0.000798124,
        "bandwidth": 0.0053283,
        "launch": 0.035000053248000006
      },
      "verified": true,
      "max_rel": 0.0
    }
  },
  "distinct_certified": 2,
  "frontiers": []
}
```

</details>

<details><summary><b>qkv_sum</b> — raw metrics</summary>

```json
{
  "family": "qkv_sum",
  "note": "sum of three same-input projections",
  "n_root_alts": 3,
  "default": {
    "sig": "(linear x, (add (add W, W), W))",
    "flops": 65536.0,
    "read": 18432.0,
    "write": 2048.0,
    "traffic": 20480.0,
    "ops": 3,
    "marginal": {
      "compute": 6.603599999999999e-05,
      "bandwidth": 0.00041009995903999995,
      "launch": 0.005000004055039999
    },
    "true": {
      "compute": 7.522799999999999e-05,
      "bandwidth": 0.00041109999999999996,
      "launch": 0.015000004095999999
    }
  },
  "picks": {
    "compute": {
      "sig": "(linear x, (add (add W, W), W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.603599999999999e-05,
        "bandwidth": 0.00041009995903999995,
        "launch": 0.005000004055039999
      },
      "true": {
        "compute": 7.522799999999999e-05,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000004095999999
      },
      "verified": true,
      "max_rel": 2.5782359176019615e-07
    },
    "bandwidth": {
      "sig": "(linear x, (add (add W, W), W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.603599999999999e-05,
        "bandwidth": 0.00041009995903999995,
        "launch": 0.005000004055039999
      },
      "true": {
        "compute": 7.522799999999999e-05,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000004095999999
      },
      "verified": true,
      "max_rel": 2.5782359176019615e-07
    },
    "launch": {
      "sig": "(linear x, (add (add W, W), W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.603599999999999e-05,
        "bandwidth": 0.00041009995903999995,
        "launch": 0.005000004055039999
      },
      "true": {
        "compute": 7.522799999999999e-05,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000004095999999
      },
      "verified": true,
      "max_rel": 2.5782359176019615e-07
    }
  },
  "distinct_certified": 1,
  "frontiers": []
}
```

</details>

<details><summary><b>linattn</b> — raw metrics</summary>

```json
{
  "family": "linattn",
  "note": "unnormalised attention (QK^T)V",
  "n_root_alts": 2,
  "default": {
    "sig": "(matmul (matmul (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=0, sizes=(64, 64, 64)), (transpose (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=1, sizes=(64, 64, 64)), dim0=-2, dim1=-1)), (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=2, sizes=(64, 64, 64)))",
    "flops": 8388608.0,
    "read": 344064.0,
    "write": 294912.0,
    "traffic": 638976.0,
    "ops": 9,
    "marginal": {
      "compute": 0.006293956000000001,
      "bandwidth": 0.006228420000000001,
      "launch": 0.0250000622592
    },
    "true": {
      "compute": 0.008393108,
      "bandwidth": 0.01278402,
      "launch": 0.0450001277952
    }
  },
  "picks": {
    "compute": {
      "sig": "(matmul (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=0, sizes=(64, 64, 64)), (matmul (transpose (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=1, sizes=(64, 64, 64)), dim0=-2, dim1=-1), (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=2, sizes=(64, 64, 64))))",
      "flops": 10485760.0,
      "read": 442368.0,
      "write": 393216.0,
      "traffic": 835584.0,
      "ops": 9,
      "marginal": {
        "compute": 0.006293956000000001,
        "bandwidth": 0.006228420000000001,
        "launch": 0.0250000622592
      },
      "true": {
        "compute": 0.01049026,
        "bandwidth": 0.01671618,
        "launch": 0.0450001671168
      },
      "verified": true,
      "max_rel": 4.559944805746715e-07
    },
    "bandwidth": {
      "sig": "(matmul (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=0, sizes=(64, 64, 64)), (matmul (transpose (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=1, sizes=(64, 64, 64)), dim0=-2, dim1=-1), (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=2, sizes=(64, 64, 64))))",
      "flops": 10485760.0,
      "read": 442368.0,
      "write": 393216.0,
      "traffic": 835584.0,
      "ops": 9,
      "marginal": {
        "compute": 0.006293956000000001,
        "bandwidth": 0.006228420000000001,
        "launch": 0.0250000622592
      },
      "true": {
        "compute": 0.01049026,
        "bandwidth": 0.01671618,
        "launch": 0.0450001671168
      },
      "verified": true,
      "max_rel": 4.559944805746715e-07
    },
    "launch": {
      "sig": "(matmul (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=0, sizes=(64, 64, 64)), (matmul (transpose (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=1, sizes=(64, 64, 64)), dim0=-2, dim1=-1), (split (linear x, (concat (concat W, W, dim=0), W, dim=0)), dim=-1, index=2, sizes=(64, 64, 64))))",
      "flops": 10485760.0,
      "read": 442368.0,
      "write": 393216.0,
      "traffic": 835584.0,
      "ops": 9,
      "marginal": {
        "compute": 0.006293956000000001,
        "bandwidth": 0.006228420000000001,
        "launch": 0.0250000622592
      },
      "true": {
        "compute": 0.01049026,
        "bandwidth": 0.01671618,
        "launch": 0.0450001671168
      },
      "verified": true,
      "max_rel": 4.559944805746715e-07
    }
  },
  "distinct_certified": 2,
  "frontiers": [
    {
      "axes": [
        "count",
        "params"
      ],
      "size": 2,
      "kind": "tie",
      "vectors": [
        {
          "count": 5.0,
          "params": 12288.0
        },
        {
          "count": 5.0,
          "params": 12288.0
        }
      ]
    }
  ]
}
```

</details>

<details><summary><b>deep_parallel</b> — raw metrics</summary>

```json
{
  "family": "deep_parallel",
  "note": "(x@W1 + x@W2) @ W3",
  "n_root_alts": 4,
  "default": {
    "sig": "(linear x, (matmul W, (add W, W)))",
    "flops": 65536.0,
    "read": 18432.0,
    "write": 2048.0,
    "traffic": 20480.0,
    "ops": 3,
    "marginal": {
      "compute": 6.603600000000022e-05,
      "bandwidth": 0.00041008943232,
      "launch": 0.005000000327679999
    },
    "true": {
      "compute": 0.0021805720000000004,
      "bandwidth": 0.00041109999999999996,
      "launch": 0.015000010895359999
    }
  },
  "picks": {
    "compute": {
      "sig": "(linear x, (matmul W, (add W, W)))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.603600000000022e-05,
        "bandwidth": 0.00041008943232,
        "launch": 0.005000000327679999
      },
      "true": {
        "compute": 0.0021805720000000004,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000010895359999
      },
      "verified": true,
      "max_rel": 4.844872791693451e-07
    },
    "bandwidth": {
      "sig": "(linear x, (matmul W, (add W, W)))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.603600000000022e-05,
        "bandwidth": 0.00041008943232,
        "launch": 0.005000000327679999
      },
      "true": {
        "compute": 0.0021805720000000004,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000010895359999
      },
      "verified": true,
      "max_rel": 4.844872791693451e-07
    },
    "launch": {
      "sig": "(linear x, (matmul W, (add W, W)))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 3,
      "marginal": {
        "compute": 6.603600000000022e-05,
        "bandwidth": 0.00041008943232,
        "launch": 0.005000000327679999
      },
      "true": {
        "compute": 0.0021805720000000004,
        "bandwidth": 0.00041109999999999996,
        "launch": 0.015000010895359999
      },
      "verified": true,
      "max_rel": 4.844872791693451e-07
    }
  },
  "distinct_certified": 1,
  "frontiers": []
}
```

</details>

<details><summary><b>parallel8</b> — raw metrics</summary>

```json
{
  "family": "parallel8",
  "note": "eight-expert projection sum",
  "n_root_alts": 3,
  "default": {
    "sig": "(linear x, (add (add (add (add (add (add (add W, W), W), W), W), W), W), W))",
    "flops": 65536.0,
    "read": 18432.0,
    "write": 2048.0,
    "traffic": 20480.0,
    "ops": 8,
    "marginal": {
      "compute": 6.6036e-05,
      "bandwidth": 0.00041009985663999996,
      "launch": 0.0050000039526399975
    },
    "true": {
      "compute": 9.8208e-05,
      "bandwidth": 0.00041359999999999997,
      "launch": 0.040000004096
    }
  },
  "picks": {
    "compute": {
      "sig": "(linear x, (add (add (add (add (add (add (add W, W), W), W), W), W), W), W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 8,
      "marginal": {
        "compute": 6.6036e-05,
        "bandwidth": 0.00041009985663999996,
        "launch": 0.0050000039526399975
      },
      "true": {
        "compute": 9.8208e-05,
        "bandwidth": 0.00041359999999999997,
        "launch": 0.040000004096
      },
      "verified": true,
      "max_rel": 2.4166140037484517e-07
    },
    "bandwidth": {
      "sig": "(linear x, (add (add (add (add (add (add (add W, W), W), W), W), W), W), W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 8,
      "marginal": {
        "compute": 6.6036e-05,
        "bandwidth": 0.00041009985663999996,
        "launch": 0.0050000039526399975
      },
      "true": {
        "compute": 9.8208e-05,
        "bandwidth": 0.00041359999999999997,
        "launch": 0.040000004096
      },
      "verified": true,
      "max_rel": 2.4166140037484517e-07
    },
    "launch": {
      "sig": "(linear x, (add (add (add (add (add (add (add W, W), W), W), W), W), W), W))",
      "flops": 65536.0,
      "read": 18432.0,
      "write": 2048.0,
      "traffic": 20480.0,
      "ops": 8,
      "marginal": {
        "compute": 6.6036e-05,
        "bandwidth": 0.00041009985663999996,
        "launch": 0.0050000039526399975
      },
      "true": {
        "compute": 9.8208e-05,
        "bandwidth": 0.00041359999999999997,
        "launch": 0.040000004096
      },
      "verified": true,
      "max_rel": 2.4166140037484517e-07
    }
  },
  "distinct_certified": 1,
  "frontiers": []
}
```

</details>

