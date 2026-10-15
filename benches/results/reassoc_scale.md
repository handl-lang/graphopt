# Weight-chain reassociation vs Inductor

Deep x @ W1 @ … @ Wk chains: the e-graph finds the weights-first form (one runtime GEMM) and the post-grad FX capture proves Inductor cannot reach it.

- **when**: 2026-10-14T06:16:44+00:00 (git `2c782dd` dirty)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | catopt reaches a weight-folded form Inductor's post-grad graph cannot express | 14.92× vs Inductor at (k,d,B·T)=(16,512,4096) (17.40× vs eager) | catopt+inductor / inductor = 14.9243 |
| **WIN** | every timed cell passes the fp32 equivalence gate | 12/12 cells verified | verified cells = 12 |

## Results

| case | k | d | R | eager ms | inductor ms | manual_ref ms | catopt ms | catopt_inductor ms |
|---|---|---|---|---|---|---|---|---|
| k2_d512_R4096 | 2 | 512 | 4096 | 41.91 ± 0.2609 | 37.89 ± 0.5018 | 19.79 ± 0 | 19.69 ± 0 | 20.04 ± 0.3525 |
| k2_d512_R16384 | 2 | 512 | 16384 | 176.2 ± 0 | 175.3 ± 0 | 95.59 ± 0 | 95.34 ± 0 | 95.73 ± 0 |
| k2_d512_R65536 | 2 | 512 | 65536 | 761.8 ± 0 | 766.4 ± 0 | 369.5 ± 0 | 367.2 ± 0 | 353.4 ± 0 |
| k4_d512_R4096 | 4 | 512 | 4096 | 84.71 ± 0 | 82.2 ± 0 | 19.76 ± 0 | 18.66 ± 0 | 19.61 ± 0.2549 |
| k4_d512_R16384 | 4 | 512 | 16384 | 368.2 ± 0 | 326.8 ± 0 | 86.29 ± 0 | 87.85 ± 0 | 86.86 ± 0 |
| k4_d512_R65536 | 4 | 512 | 65536 | 1349 ± 0 | 1289 ± 0 | 339 ± 0 | 338.6 ± 0 | 356 ± 0 |
| k8_d512_R4096 | 8 | 512 | 4096 | 167.9 ± 0 | 157.8 ± 0 | 19.2 ± 0 | 18.3 ± 0 | 19.19 ± 0.2375 |
| k8_d512_R16384 | 8 | 512 | 16384 | 677.7 ± 0 | 641.5 ± 0 | 85.43 ± 0 | 86.6 ± 0 | 85.39 ± 0 |
| k8_d512_R65536 | 8 | 512 | 65536 | 2737 ± 0 | 2508 ± 0 | 342 ± 0 | 339.4 ± 0 | 351.4 ± 0 |
| k16_d512_R4096 | 16 | 512 | 4096 | 333.4 ± 0 | 316 ± 0 | 19.28 ± 0 | 19.16 ± 0 | 21.17 ± 0.3414 |
| k16_d512_R16384 | 16 | 512 | 16384 | 1398 ± 0 | 1289 ± 0 | 87.73 ± 0 | 87.95 ± 0 | 86.74 ± 0 |
| k16_d512_R65536 | 16 | 512 | 65536 | 5934 ± 0 | 5376 ± 0 | 361.3 ± 0 | 362.3 ± 0 | 361.4 ± 0 |

## Metrics

<details><summary><b>k2_d512_R4096</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 3.0994415283203125e-06,
    "max_rel": 6.246465941769099e-07,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 2,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "2 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv1) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 1
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 2.2598042488098145
  },
  "extracted_root": "(matmul x, fused_2)",
  "flops_model": {
    "left_assoc_runtime": 4294967296,
    "weights_first_runtime": 2147483648,
    "weights_first_fold_once": 268435456
  }
}
```

</details>

<details><summary><b>k2_d512_R16384</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 3.337860107421875e-06,
    "max_rel": 6.197272279459442e-07,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 2,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "2 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv1) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 1
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.09697389602661133
  },
  "extracted_root": "(matmul x, fused_2)",
  "flops_model": {
    "left_assoc_runtime": 17179869184,
    "weights_first_runtime": 8589934592,
    "weights_first_fold_once": 268435456
  }
}
```

</details>

<details><summary><b>k2_d512_R65536</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 3.5762786865234375e-06,
    "max_rel": 6.340420204163221e-07,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 2,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "2 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv1) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 1
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.1032712459564209
  },
  "extracted_root": "(matmul x, fused_2)",
  "flops_model": {
    "left_assoc_runtime": 68719476736,
    "weights_first_runtime": 34359738368,
    "weights_first_fold_once": 268435456
  }
}
```

</details>

<details><summary><b>k4_d512_R4096</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 4.172325134277344e-06,
    "max_rel": 7.623715033722088e-07,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 4,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "4 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv3) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 16
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.18848156929016113
  },
  "extracted_root": "(matmul x, fused_6)",
  "flops_model": {
    "left_assoc_runtime": 8589934592,
    "weights_first_runtime": 2147483648,
    "weights_first_fold_once": 805306368
  }
}
```

</details>

<details><summary><b>k4_d512_R16384</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 4.291534423828125e-06,
    "max_rel": 7.71205120253586e-07,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 4,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "4 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv3) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 16
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.19245004653930664
  },
  "extracted_root": "(matmul x, fused_6)",
  "flops_model": {
    "left_assoc_runtime": 34359738368,
    "weights_first_runtime": 8589934592,
    "weights_first_fold_once": 805306368
  }
}
```

</details>

<details><summary><b>k4_d512_R65536</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 4.291534423828125e-06,
    "max_rel": 6.830063541598919e-07,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 4,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "4 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv3) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 16
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.2082061767578125
  },
  "extracted_root": "(matmul x, fused_6)",
  "flops_model": {
    "left_assoc_runtime": 137438953472,
    "weights_first_runtime": 34359738368,
    "weights_first_fold_once": 805306368
  }
}
```

</details>

<details><summary><b>k8_d512_R4096</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 5.4836273193359375e-06,
    "max_rel": 1.04252164773538e-06,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 8,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "8 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None",
      "mm_4: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_3, arg5_1);  mm_3 = arg5_1 = None",
      "mm_5: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_4, arg6_1);  mm_4 = arg6_1 = None",
      "mm_6: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_5, arg7_1);  mm_5 = arg7_1 = None",
      "mm_7: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_6, arg8_1);  mm_6 = arg8_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv7) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 550,
      "assoc_matmul": 140
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.6273095607757568
  },
  "extracted_root": "(matmul x, fused_14)",
  "flops_model": {
    "left_assoc_runtime": 17179869184,
    "weights_first_runtime": 2147483648,
    "weights_first_fold_once": 1879048192
  }
}
```

</details>

<details><summary><b>k8_d512_R16384</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 5.4836273193359375e-06,
    "max_rel": 9.881976546546515e-07,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 8,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "8 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None",
      "mm_4: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_3, arg5_1);  mm_3 = arg5_1 = None",
      "mm_5: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_4, arg6_1);  mm_4 = arg6_1 = None",
      "mm_6: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_5, arg7_1);  mm_5 = arg7_1 = None",
      "mm_7: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_6, arg8_1);  mm_6 = arg8_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv7) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 550,
      "assoc_matmul": 140
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.5824024677276611
  },
  "extracted_root": "(matmul x, fused_14)",
  "flops_model": {
    "left_assoc_runtime": 68719476736,
    "weights_first_runtime": 8589934592,
    "weights_first_fold_once": 1879048192
  }
}
```

</details>

<details><summary><b>k8_d512_R65536</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 5.7220458984375e-06,
    "max_rel": 9.502416266489562e-07,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 8,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "8 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None",
      "mm_4: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_3, arg5_1);  mm_3 = arg5_1 = None",
      "mm_5: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_4, arg6_1);  mm_4 = arg6_1 = None",
      "mm_6: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_5, arg7_1);  mm_5 = arg7_1 = None",
      "mm_7: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_6, arg8_1);  mm_6 = arg8_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv7) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 550,
      "assoc_matmul": 140
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.620513916015625
  },
  "extracted_root": "(matmul x, fused_14)",
  "flops_model": {
    "left_assoc_runtime": 274877906944,
    "weights_first_runtime": 34359738368,
    "weights_first_fold_once": 1879048192
  }
}
```

</details>

<details><summary><b>k16_d512_R4096</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 7.808208465576172e-06,
    "max_rel": 1.308228135812855e-06,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 16,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "16 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None",
      "mm_4: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_3, arg5_1);  mm_3 = arg5_1 = None",
      "mm_5: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_4, arg6_1);  mm_4 = arg6_1 = None",
      "mm_6: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_5, arg7_1);  mm_5 = arg7_1 = None",
      "mm_7: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_6, arg8_1);  mm_6 = arg8_1 = None",
      "mm_8: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_7, arg9_1);  mm_7 = arg9_1 = None",
      "mm_9: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_8, arg10_1);  mm_8 = arg10_1 = None",
      "mm_10: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_9, arg11_1);  mm_9 = arg11_1 = None",
      "mm_11: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_10, arg12_1);  mm_10 = arg12_1 = None",
      "mm_12: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_11, arg13_1);  mm_11 = arg13_1 = None",
      "mm_13: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_12, arg14_1);  mm_12 = arg14_1 = None",
      "mm_14: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_13, arg15_1);  mm_13 = arg15_1 = None",
      "mm_15: \"f32[4096, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_14, arg16_1);  mm_14 = arg16_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv15) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 1079,
      "assoc_matmul": 365
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 0.9929788112640381
  },
  "extracted_root": "(matmul x, fused_30)",
  "flops_model": {
    "left_assoc_runtime": 34359738368,
    "weights_first_runtime": 2147483648,
    "weights_first_fold_once": 4026531840
  }
}
```

</details>

<details><summary><b>k16_d512_R16384</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 8.58306884765625e-06,
    "max_rel": 1.2449432824589566e-06,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 16,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "16 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None",
      "mm_4: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_3, arg5_1);  mm_3 = arg5_1 = None",
      "mm_5: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_4, arg6_1);  mm_4 = arg6_1 = None",
      "mm_6: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_5, arg7_1);  mm_5 = arg7_1 = None",
      "mm_7: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_6, arg8_1);  mm_6 = arg8_1 = None",
      "mm_8: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_7, arg9_1);  mm_7 = arg9_1 = None",
      "mm_9: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_8, arg10_1);  mm_8 = arg10_1 = None",
      "mm_10: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_9, arg11_1);  mm_9 = arg11_1 = None",
      "mm_11: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_10, arg12_1);  mm_10 = arg12_1 = None",
      "mm_12: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_11, arg13_1);  mm_11 = arg13_1 = None",
      "mm_13: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_12, arg14_1);  mm_12 = arg14_1 = None",
      "mm_14: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_13, arg15_1);  mm_13 = arg15_1 = None",
      "mm_15: \"f32[16384, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_14, arg16_1);  mm_14 = arg16_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv15) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 1079,
      "assoc_matmul": 365
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 1.0310983657836914
  },
  "extracted_root": "(matmul x, fused_30)",
  "flops_model": {
    "left_assoc_runtime": 137438953472,
    "weights_first_runtime": 8589934592,
    "weights_first_fold_once": 4026531840
  }
}
```

</details>

<details><summary><b>k16_d512_R65536</b> — raw metrics</summary>

```json
{
  "verify": {
    "max_abs": 8.702278137207031e-06,
    "max_rel": 1.2622341613819978e-06,
    "allclose_fp32": true,
    "rtol": 0.0001,
    "atol": 1e-05
  },
  "inductor_proof": {
    "post_grad_mm": 16,
    "weight_only_mm": 0,
    "left_assoc": true,
    "graph_found": true,
    "note": "16 left-assoc mms on the data spine \u2014 weights-first unreachable",
    "mm_lines": [
      "mm: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None",
      "mm_1: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm, arg2_1);  mm = arg2_1 = None",
      "mm_2: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_1, arg3_1);  mm_1 = arg3_1 = None",
      "mm_3: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_2, arg4_1);  mm_2 = arg4_1 = None",
      "mm_4: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_3, arg5_1);  mm_3 = arg5_1 = None",
      "mm_5: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_4, arg6_1);  mm_4 = arg6_1 = None",
      "mm_6: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_5, arg7_1);  mm_5 = arg7_1 = None",
      "mm_7: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_6, arg8_1);  mm_6 = arg8_1 = None",
      "mm_8: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_7, arg9_1);  mm_7 = arg9_1 = None",
      "mm_9: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_8, arg10_1);  mm_8 = arg10_1 = None",
      "mm_10: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_9, arg11_1);  mm_9 = arg11_1 = None",
      "mm_11: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_10, arg12_1);  mm_10 = arg12_1 = None",
      "mm_12: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_11, arg13_1);  mm_11 = arg13_1 = None",
      "mm_13: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_12, arg14_1);  mm_12 = arg14_1 = None",
      "mm_14: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_13, arg15_1);  mm_13 = arg15_1 = None",
      "mm_15: \"f32[65536, 512][512, 1]cpu\" = torch.ops.aten.mm.default(mm_14, arg16_1);  mm_14 = arg16_1 = None"
    ]
  },
  "catopt_extraction": {
    "weights_first": true,
    "note": "matmul(x, W_q@W_kv1@\u2026@W_kv15) \u2014 weights-first",
    "assoc_rule_fires": {
      "assoc_matmul_rev": 1079,
      "assoc_matmul": 365
    },
    "inductor_backend_mm": 1,
    "pipeline_s": 1.1615052223205566
  },
  "extracted_root": "(matmul x, fused_30)",
  "flops_model": {
    "left_assoc_runtime": 549755813888,
    "weights_first_runtime": 34359738368,
    "weights_first_fold_once": 4026531840
  }
}
```

</details>

