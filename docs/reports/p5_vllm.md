# mlopt vs vLLM and through it

Three legs: verified HF export, fixed-window torch decode, and vLLM serving the exported checkpoint — with token agreement against torch eager greedy.

- **when**: 2026-10-27T00:35:09+00:00 (git `400aa70` dirty)
- **device**: `cuda` — NVIDIA GeForce RTX 2050
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | vLLM serves the mlopt-exported HF model token-for-token like torch eager greedy | token agreement 1.000 vs torch eager | token agreement = 1 |
| **WIN** | the optimized module exports to HF without changing the computation | optimize verify rel=1.27e-06; RoPE attention-score max\|Δ\|=1.53e-05 | optimize verify rel = 1.27373e-06 |

## Results

| case | B | ctx | gen | eager ms | mlopt ms | inductor ms | mlopt+inductor ms |
|---|---|---|---|---|---|---|---|
| decode B=1 ctx=64 gen=32 | 1 | 64 | 32 | 86.28 ± 0 | 86.98 ± 0 | 76.53 ± 0 | 80.77 ± 0 |

## Metrics

<details><summary><b>decode B=1 ctx=64 gen=32</b> — raw metrics</summary>

```json
{
  "gen_tokens": 32,
  "ctx": 64,
  "batch": 1,
  "eager_tok_per_s": 370.9,
  "mlopt_tok_per_s": 367.9,
  "inductor_tok_per_s": 418.1,
  "mlopt+inductor_tok_per_s": 396.2,
  "wall_clock": true,
  "note": "vLLM offline LLM.generate",
  "eager_mode": "subprocess /home/luis/vllm-probe/bin/python",
  "eager_init_s": 13.92,
  "eager_prefill_s": 0.0552,
  "eager_gen_s": 0.1065,
  "eager_gen_times": [
    0.4654,
    0.1246,
    0.1065
  ],
  "eager_decode_tok_per_s": 604.6,
  "eager_e2e_tok_per_s": 300.5,
  "eager_tokens": [
    [
      29891,
      29892,
      304,
      278,
      767,
      29889,
      940,
      471,
      1407,
      9796,
      29889,
      940,
      6452,
      287,
      278,
      767,
      322,
      1497,
      29892,
      376,
      25271,
      366,
      29892,
      8889,
      488,
      29892,
      363,
      19912,
      592,
      1284,
      590,
      3271
    ]
  ],
  "eager_enforce_eager": true,
  "eager_tokenizer": "hf-internal-testing/llama-tokenizer",
  "eager_token_agreement_vs_torch_eager": 1.0,
  "mlopt_mode": "subprocess /home/luis/vllm-probe/bin/python",
  "mlopt_init_s": 6.64,
  "mlopt_prefill_s": 0.0461,
  "mlopt_gen_s": 0.0936,
  "mlopt_gen_times": [
    0.107,
    0.0937,
    0.0936
  ],
  "mlopt_decode_tok_per_s": 653.5,
  "mlopt_e2e_tok_per_s": 342.0,
  "mlopt_tokens": [
    [
      29891,
      29892,
      304,
      278,
      767,
      29889,
      940,
      471,
      1407,
      9796,
      29889,
      940,
      6452,
      287,
      278,
      767,
      322,
      1497,
      29892,
      376,
      25271,
      366,
      29892,
      8889,
      488,
      29892,
      363,
      19912,
      592,
      1284,
      590,
      3271
    ]
  ],
  "mlopt_enforce_eager": true,
  "mlopt_tokenizer": "hf-internal-testing/llama-tokenizer",
  "mlopt_token_agreement_vs_torch_eager": 1.0
}
```

</details>

