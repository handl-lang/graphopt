# Compositional pairing demo

Per-model lowering autotune (eager / generic / batched / compiled): the fastest verified lowering is reported, including when eager wins.

- **when**: 2026-10-14T06:18:43+00:00 (git `2c782dd` dirty)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | the Autotuned lowering beats eager on at least one model | best 2.69× vs eager on matrix_chain | catopt_best / eager = 2.68929 |
| **WIN** | every model's optimized lowering verifies | 4/4 models verified | verified models = 4 |

## Results

| case | device | eager ms | inductor ms | catopt_best ms |
|---|---|---|---|---|
| linear_recurrence | cpu | 0.02618 ± 0.0002268 | 0.02317 ± 0.000811 | 0.02636 ± 0.0003452 |
| residual_mlp | cpu | 0.03226 ± 0.0007457 | 0.05415 ± 0.001354 | 0.0307 ± 0.0001421 |
| matrix_chain | cpu | 0.01146 ± 0 | 0.0329 ± 0.001064 | 0.00426 ± 3.554e-05 |
| retnet_stack | cpu | 0.2896 ± 0.02393 | 0.06039 ± 0.001769 | 0.309 ± 0.001829 |

## Metrics

<details><summary><b>linear_recurrence</b> — raw metrics</summary>

```json
{
  "pick": "eager",
  "verified": true,
  "lowering": "batched",
  "search_s": 3.84,
  "autotune": "eager:0.028ms generic:0.051ms batched:0.290ms compiled:unknown",
  "inductor_status": "compiled in 2.2s"
}
```

</details>

<details><summary><b>residual_mlp</b> — raw metrics</summary>

```json
{
  "pick": "batched",
  "verified": true,
  "lowering": "generic",
  "search_s": 0.05,
  "autotune": "eager:0.045ms generic:0.044ms batched:0.043ms compiled:unknown",
  "inductor_status": "compiled in 1.4s"
}
```

</details>

<details><summary><b>matrix_chain</b> — raw metrics</summary>

```json
{
  "pick": "batched",
  "verified": true,
  "lowering": "generic",
  "search_s": 0.03,
  "autotune": "eager:0.015ms generic:0.006ms batched:0.006ms compiled:unknown",
  "inductor_status": "compiled in 0.1s"
}
```

</details>

<details><summary><b>retnet_stack</b> — raw metrics</summary>

```json
{
  "pick": "eager",
  "verified": true,
  "lowering": "batched",
  "search_s": 28.3,
  "autotune": "eager:0.293ms generic:0.486ms batched:0.446ms compiled:unknown",
  "inductor_status": "compiled in 2.3s"
}
```

</details>

