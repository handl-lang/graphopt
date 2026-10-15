# Rewrite-law effect bench

Per registered law: a synthetic term → saturate on that rule alone → extract → lower → verify, timed before/after.

- **when**: 2026-10-14T06:13:49+00:00 (git `2c782dd` clean)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | every registered law fires, is picked by extraction, and lowers to a verified term | 11 verified / 11 picked / 13 fired across 13 cells | verified cells = 11 |

## Results

| case | law | size | before ms | after ms |
|---|---|---|---|---|
| id_add | id_add | 256 | 0.01982 ± 0 | 0.00184 ± 0 |
| double_neg | double_neg | 256 | 0.03495 ± 0 | 0.001845 ± 0 |
| sub_to_add | sub_to_add | 256 | — | — |
| pow_to_square | pow_to_square | 256 | 0.5153 ± 0.007048 | 0.01788 ± 0 |
| silu_expand | silu_expand | 256 | — | — |
| assoc_matmul_rev | assoc_matmul_rev | 256 | 2.152 ± 0 | 0.5395 ± 0.002697 |
| assoc_linear | assoc_linear | 256 | 1.077 ± 0 | 0.5388 ± 0.00218 |
| weight_factor_linear | weight_factor_linear | 256 | 1.111 ± 0 | 0.5396 ± 0.002421 |
| right_factor_linear | right_factor_linear | 256 | 1.111 ± 0 | 0.5729 ± 0.001046 |
| linear_channel_scale | linear_channel_scale | 256 | 0.5632 ± 0.001813 | 0.5414 ± 0.001737 |
| parallel_mul_fuse | parallel_mul_fuse | 256 | 1.119 ± 0 | 1.091 ± 0 |
| sdpa_fold_addmul | sdpa_fold_addmul | 256 | 0.09458 ± 0 | 0.07272 ± 0 |
| affd_lift | affd_lift | 256 | 0.004326 ± 4.942e-05 | 0.02213 ± 0 |

## Metrics

<details><summary><b>id_add</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 29181.7978,
  "cost_out": 0.0,
  "before": "(add x, 0)",
  "after": "x",
  "max_rel": "0.000e+00"
}
```

</details>

<details><summary><b>double_neg</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 58363.5056,
  "cost_out": 0.0,
  "before": "(neg (neg x))",
  "after": "x",
  "max_rel": "0.000e+00"
}
```

</details>

<details><summary><b>sub_to_add</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "no",
  "verified": "same",
  "cost_in": 35072.6292,
  "cost_out": 35072.6292,
  "note": "member found; extraction kept the input term (RHS not cheaper under the pipeline cost model)"
}
```

</details>

<details><summary><b>pow_to_square</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 29181.7978,
  "cost_out": 29181.7528,
  "before": "(pow x, 2)",
  "after": "(square x)",
  "max_rel": "4.610e-08"
}
```

</details>

<details><summary><b>silu_expand</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "no",
  "verified": "same",
  "cost_in": 29181.7528,
  "cost_out": 29181.7528,
  "note": "member found; extraction kept the input term (RHS not cheaper under the pipeline cost model)"
}
```

</details>

<details><summary><b>assoc_matmul_rev</b> — raw metrics</summary>

```json
{
  "fired": 16,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 176974.1824,
  "cost_out": 44243.5456,
  "before": "(matmul (matmul (matmul (matmul x, W1), W2), W3), W4)",
  "after": "(matmul x, (matmul W1, (matmul W2, (matmul W3, W4))))",
  "max_rel": "9.808e-07"
}
```

</details>

<details><summary><b>assoc_linear</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 88487.0912,
  "cost_out": 44243.5456,
  "before": "(linear (linear x, A), B)",
  "after": "(linear x, (matmul B, A))",
  "max_rel": "9.777e-07"
}
```

</details>

<details><summary><b>weight_factor_linear</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 123559.7204,
  "cost_out": 44243.5456,
  "before": "(add (linear x, W), (linear x, W2))",
  "after": "(linear x, (add W, W2))",
  "max_rel": "7.503e-07"
}
```

</details>

<details><summary><b>right_factor_linear</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 123559.7204,
  "cost_out": 79316.1748,
  "before": "(add (linear a, W), (linear b, W))",
  "after": "(linear (add a, b), W)",
  "max_rel": "8.333e-07"
}
```

</details>

<details><summary><b>linear_channel_scale</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 73436.804,
  "cost_out": 44243.5456,
  "before": "(linear (mul x, c), W)",
  "after": "(linear x, (mul W, c))",
  "max_rel": "2.180e-07"
}
```

</details>

<details><summary><b>parallel_mul_fuse</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 123559.7204,
  "cost_out": 123559.7204,
  "before": "(mul (linear x, A), (linear x, B))",
  "after": "(mul (chunk (linear x, (concat A, B, dim=0)), chunks=2, dim=-1, index=0), (chunk (linear x, (concat A, B, dim=0)), chunks=2, dim=-1, index=1))",
  "max_rel": "2.513e-07"
}
```

</details>

<details><summary><b>sdpa_fold_addmul</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 111899.9551,
  "cost_out": 21818.1573,
  "before": "(matmul (softmax (add (mul (matmul q, (transpose k, dim0=-2, dim1=-1)), 0.17677669529663687), m), dim=-1), v)",
  "after": "(sdpa q, k, v, m, scale=0.17677669529663687)",
  "max_rel": "1.039e-07"
}
```

</details>

<details><summary><b>affd_lift</b> — raw metrics</summary>

```json
{
  "fired": 1,
  "rhs_member": "yes",
  "picked": "yes",
  "verified": "pass",
  "cost_in": 34869.0337,
  "cost_out": 26134.5169,
  "before": "(add (mul a, h), x)",
  "after": "(applyd (aff_diag a, x), h)",
  "max_rel": "0.000e+00"
}
```

</details>

