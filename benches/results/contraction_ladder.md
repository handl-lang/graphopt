# Contraction-ordering ladder

A seeded hard-instance set of 4-8 matrix chains and 3-6 tensor networks, scoring the non-learned contraction-ordering players (one-step oracle, greedy episode, saturation, classic greedy) by cost and rank against the exact DP optimum.

- **when**: 2026-10-03T04:44:29+00:00 (git `3abcb5c` dirty)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `chains`=60 · `nets`=60 · `seed`=0 · `repeats`=3 · `n_rules`=51 · `subset_dp_validated`=40 · `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **NEGATIVE** | a learned search policy can beat greedy on contraction-ordering *quality* | no — the priced rule space already reaches the DP optimum: saturate ties optimal on 60/60 chains; the residual headroom above greedy is search cost, not quality | saturate chains at the DP optimum = 1 |
| **WIN** | full equality saturation reaches the exact contraction-ordering DP optimum on matrix chains | saturate == 2·DP optimum on 60/60 chains (worst ratio 1.000) | saturate chains at optimum / n = 1 |
| **NEGATIVE** | the one-step greedy cost-model oracle is optimal on contraction ordering | one-step oracle at the optimum on 38/60 chains (63%); mean ratio 1.28, worst 4.09x | one-step oracle chains at optimum / n = 0.633 |
| **NEGATIVE** | the multi-step greedy episode is optimal on contraction ordering | greedy episode (h6/p2) at the optimum on 59/60 chains vs one-step 38/60; worst 1.01x | greedy episode chains at optimum / n = 0.983 |
| **NEGATIVE** | classic greedy is optimal on general tensor networks | classic greedy at the optimum on 20/60 networks (33%); mean ratio 1.57, worst 5.5x | classic greedy networks at optimum / n = 0.333 |

## Results

| case | n_mats | dims | n_tensors | tensors | greedy-1step ms | greedy-episode ms | saturate ms | optimal ms | greedy-classic ms |
|---|---|---|---|---|---|---|---|---|---|
| chain-00 | 7 | (27, 3, 17, 33, 32, 26, 20, 31) | — | — | 16.78 ± 0 | 67.45 ± 0 | 15.18 ± 0 | 0.01226 ± 0 | — |
| chain-01 | 6 | (38, 14, 33, 9, 19, 9, 7) | — | — | 14.19 ± 0 | 41.63 ± 0 | 6.226 ± 0 | 0.008118 ± 0 | — |
| chain-02 | 8 | (17, 35, 39, 10, 20, 7, 5, 22, 31) | — | — | 18.34 ± 0 | 53.81 ± 0 | 48.67 ± 0 | 0.01643 ± 0 | — |
| chain-03 | 8 | (7, 23, 28, 21, 40, 14, 36, 31, 29) | — | — | 18.33 ± 0 | 81.04 ± 0 | 48.94 ± 0 | 0.01598 ± 0 | — |
| chain-04 | 8 | (17, 4, 36, 1, 6, 26, 1, 40, 32) | — | — | 18.03 ± 0 | 80.51 ± 0 | 49.33 ± 0 | 0.01633 ± 0 | — |
| chain-05 | 6 | (16, 21, 5, 13, 37, 15, 16) | — | — | 14.46 ± 0 | 56.61 ± 0 | 6.414 ± 0 | 0.008154 ± 0 | — |
| chain-06 | 5 | (35, 29, 6, 6, 21, 33) | — | — | 12.09 ± 0 | 36.02 ± 0 | 3.143 ± 0 | 0.005867 ± 0 | — |
| chain-07 | 7 | (7, 20, 36, 19, 8, 36, 22, 35) | — | — | 16.39 ± 0 | 68 ± 0 | 15.57 ± 0 | 0.01169 ± 0 | — |
| chain-08 | 5 | (39, 36, 38, 19, 29, 6) | — | — | 11.68 ± 0 | 23.19 ± 0 | 3.074 ± 0 | 0.00535 ± 0 | — |
| chain-09 | 8 | (25, 21, 37, 16, 19, 12, 13, 12, 3) | — | — | 17.43 ± 0 | 34.49 ± 0 | 48.43 ± 0 | 0.01651 ± 0 | — |
| chain-10 | 8 | (17, 31, 5, 6, 9, 10, 3, 6, 35) | — | — | 17.02 ± 0 | 51.79 ± 0 | 47.49 ± 0 | 0.01602 ± 0 | — |
| chain-11 | 7 | (34, 18, 34, 16, 14, 38, 27, 38) | — | — | 16.58 ± 0 | 67.6 ± 0 | 15.5 ± 0 | 0.01175 ± 0 | — |
| chain-12 | 6 | (29, 32, 23, 6, 21, 40, 8) | — | — | 14.24 ± 0 | 41.74 ± 0 | 6.188 ± 0 | 0.008124 ± 0 | — |
| chain-13 | 7 | (38, 22, 13, 16, 2, 18, 8, 15) | — | — | 16.73 ± 0 | 48.56 ± 0 | 15.31 ± 0 | 0.01161 ± 0 | — |
| chain-14 | 6 | (11, 22, 28, 4, 7, 10, 15) | — | — | 14.33 ± 0 | 42.24 ± 0 | 6.269 ± 0 | 0.007879 ± 0 | — |
| chain-15 | 4 | (37, 35, 39, 5, 2) | — | — | 10.1 ± 0 | 19.57 ± 0 | 1.477 ± 0 | 0.003046 ± 0 | — |
| chain-16 | 4 | (13, 39, 37, 8, 26) | — | — | 10.38 ± 0 | 30.08 ± 0 | 1.458 ± 0 | 0.00331 ± 0 | — |
| chain-17 | 4 | (24, 8, 3, 39, 2) | — | — | 9.805 ± 0 | 19.4 ± 0 | 1.526 ± 0 | 0.003217 ± 0 | — |
| chain-18 | 5 | (12, 8, 31, 14, 4, 2) | — | — | 12.18 ± 0 | 23.69 ± 0 | 3.104 ± 0 | 0.005474 ± 0 | — |
| chain-19 | 8 | (28, 40, 7, 17, 5, 15, 5, 20, 23) | — | — | 18.1 ± 0 | 81.1 ± 0 | 48.9 ± 0 | 0.01653 ± 0 | — |
| chain-20 | 7 | (12, 4, 33, 30, 3, 39, 7, 26) | — | — | 16.16 ± 0 | 47.28 ± 0 | 14.84 ± 0 | 0.01136 ± 0 | — |
| chain-21 | 5 | (17, 23, 31, 37, 11, 14) | — | — | 11.91 ± 0 | 34.67 ± 0 | 3.047 ± 0 | 0.005222 ± 0 | — |
| chain-22 | 4 | (11, 11, 22, 34, 17) | — | — | 9.973 ± 0 | 29.51 ± 0 | 1.492 ± 0 | 0.0034 ± 0 | — |
| chain-23 | 4 | (39, 29, 12, 1, 31) | — | — | 9.877 ± 0 | 29.55 ± 0 | 1.75 ± 0 | 0.003128 ± 0 | — |
| chain-24 | 7 | (37, 33, 20, 23, 25, 17, 10, 36) | — | — | 16.27 ± 0 | 47.69 ± 0 | 15.06 ± 0 | 0.01168 ± 0 | — |
| chain-25 | 4 | (30, 6, 22, 3, 35) | — | — | 10.08 ± 0 | 29.15 ± 0 | 1.734 ± 0 | 0.003316 ± 0 | — |
| chain-26 | 6 | (9, 16, 31, 23, 40, 19, 23) | — | — | 14.62 ± 0 | 57.28 ± 0 | 6.337 ± 0 | 0.008463 ± 0 | — |
| chain-27 | 8 | (40, 9, 20, 25, 27, 6, 1, 39, 13) | — | — | 18.21 ± 0 | 53.81 ± 0 | 48.54 ± 0 | 0.01673 ± 0 | — |
| chain-28 | 6 | (11, 16, 15, 29, 25, 37, 27) | — | — | 14.67 ± 0 | 56.11 ± 0 | 6.408 ± 0 | 0.008625 ± 0 | — |
| chain-29 | 4 | (26, 37, 27, 3, 11) | — | — | 10.47 ± 0 | 29.76 ± 0 | 1.55 ± 0 | 0.003335 ± 0 | — |
| chain-30 | 7 | (5, 17, 11, 29, 34, 32, 36, 39) | — | — | 16.77 ± 0 | 71.42 ± 0 | 16.69 ± 0 | 0.01275 ± 0 | — |
| chain-31 | 4 | (3, 32, 21, 20, 30) | — | — | 11.12 ± 0 | 43.07 ± 0 | 1.669 ± 0 | 0.003781 ± 0 | — |
| chain-32 | 4 | (27, 13, 36, 6, 9) | — | — | 11.05 ± 0 | 32.46 ± 0 | 1.698 ± 0 | 0.003664 ± 0 | — |
| chain-33 | 4 | (26, 27, 21, 1, 14) | — | — | 10.77 ± 0 | 30.95 ± 0 | 1.848 ± 0 | 0.003783 ± 0 | — |
| chain-34 | 4 | (1, 34, 40, 7, 13) | — | — | 10.35 ± 0 | 40.92 ± 0 | 1.57 ± 0 | 0.003557 ± 0 | — |
| chain-35 | 4 | (39, 13, 20, 18, 12) | — | — | 9.869 ± 0 | 20.12 ± 0 | 1.57 ± 0 | 0.003403 ± 0 | — |
| chain-36 | 4 | (31, 26, 6, 2, 18) | — | — | 9.912 ± 0 | 30.54 ± 0 | 1.758 ± 0 | 0.003236 ± 0 | — |
| chain-37 | 7 | (8, 17, 9, 34, 23, 8, 10, 18) | — | — | 16.72 ± 0 | 68.35 ± 0 | 15.35 ± 0 | 0.01184 ± 0 | — |
| chain-38 | 4 | (3, 3, 14, 17, 36) | — | — | 10.47 ± 0 | 40.36 ± 0 | 1.505 ± 0 | 0.003042 ± 0 | — |
| chain-39 | 6 | (24, 37, 3, 39, 32, 30, 28) | — | — | 14.35 ± 0 | 57.12 ± 0 | 6.438 ± 0 | 0.008318 ± 0 | — |
| chain-40 | 6 | (35, 12, 14, 25, 38, 19, 1) | — | — | 13.76 ± 0 | 27.19 ± 0 | 6.299 ± 0 | 0.008118 ± 0 | — |
| chain-41 | 5 | (10, 18, 22, 22, 24, 6) | — | — | 11.75 ± 0 | 23.7 ± 0 | 3.077 ± 0 | 0.005669 ± 0 | — |
| chain-42 | 6 | (40, 3, 3, 18, 11, 10, 38) | — | — | 14.45 ± 0 | 57.79 ± 0 | 6.234 ± 0 | 0.008157 ± 0 | — |
| chain-43 | 6 | (24, 26, 36, 9, 19, 8, 31) | — | — | 14.03 ± 0 | 42.11 ± 0 | 6.223 ± 0 | 0.008524 ± 0 | — |
| chain-44 | 5 | (4, 20, 12, 34, 5, 20) | — | — | 12.1 ± 0 | 48.08 ± 0 | 3.088 ± 0 | 0.005364 ± 0 | — |
| chain-45 | 7 | (22, 20, 27, 7, 7, 36, 31, 31) | — | — | 16.39 ± 0 | 68.83 ± 0 | 15.65 ± 0 | 0.012 ± 0 | — |
| chain-46 | 6 | (22, 8, 31, 8, 32, 28, 3) | — | — | 14.38 ± 0 | 28.1 ± 0 | 6.543 ± 0 | 0.008513 ± 0 | — |
| chain-47 | 6 | (22, 10, 11, 37, 25, 6, 5) | — | — | 13.95 ± 0 | 27.64 ± 0 | 6.46 ± 0 | 0.008245 ± 0 | — |
| chain-48 | 4 | (13, 15, 4, 25, 1) | — | — | 9.929 ± 0 | 19.59 ± 0 | 1.59 ± 0 | 0.003498 ± 0 | — |
| chain-49 | 4 | (26, 36, 34, 19, 29) | — | — | 10.22 ± 0 | 29.86 ± 0 | 1.571 ± 0 | 0.00327 ± 0 | — |
| chain-50 | 7 | (38, 14, 28, 6, 24, 15, 17, 38) | — | — | 16.89 ± 0 | 67.96 ± 0 | 16.68 ± 0 | 0.01251 ± 0 | — |
| chain-51 | 5 | (28, 13, 23, 8, 5, 2) | — | — | 12.23 ± 0 | 24.63 ± 0 | 3.164 ± 0 | 0.005678 ± 0 | — |
| chain-52 | 8 | (29, 13, 8, 32, 26, 17, 14, 3, 14) | — | — | 20.05 ± 0 | 58.07 ± 0 | 52.16 ± 0 | 0.0171 ± 0 | — |
| chain-53 | 8 | (10, 7, 13, 30, 25, 24, 35, 10, 7) | — | — | 19.23 ± 0 | 35.95 ± 0 | 50.18 ± 0 | 0.01711 ± 0 | — |
| chain-54 | 8 | (32, 10, 37, 26, 28, 34, 32, 21, 32) | — | — | 19.53 ± 0 | 83.7 ± 0 | 49.51 ± 0 | 0.01631 ± 0 | — |
| chain-55 | 7 | (13, 35, 40, 15, 1, 22, 21, 21) | — | — | 15.97 ± 0 | 47.63 ± 0 | 15.62 ± 0 | 0.01189 ± 0 | — |
| chain-56 | 4 | (34, 10, 17, 39, 10) | — | — | 9.995 ± 0 | 19.38 ± 0 | 1.584 ± 0 | 0.003439 ± 0 | — |
| chain-57 | 7 | (38, 19, 31, 5, 6, 34, 3, 5) | — | — | 16.2 ± 0 | 47.99 ± 0 | 15.42 ± 0 | 0.01216 ± 0 | — |
| chain-58 | 5 | (9, 3, 20, 1, 29, 22) | — | — | 12.02 ± 0 | 35.26 ± 0 | 3.077 ± 0 | 0.006255 ± 0 | — |
| chain-59 | 5 | (10, 30, 24, 33, 25, 34) | — | — | 12.64 ± 0 | 62.26 ± 0 | 4.047 ± 0 | 0.007528 ± 0 | — |
| net-00 | — | — | 4 | [(2,), (3,), (3, 5), (1, 6)] | — | — | — | 0.03554 ± 0 | 0.01025 ± 0 |
| net-01 | — | — | 6 | [(0, 1, 4), (0,), (4,), (3,), (1, 2, 3), (4,)] | — | — | — | 0.3064 ± 0 | 0.01933 ± 0 |
| net-02 | — | — | 5 | [(1, 2, 3), (3,), (0, 1, 3), (2,), (0, 2, 3)] | — | — | — | 0.1015 ± 0 | 0.01525 ± 0 |
| net-03 | — | — | 6 | [(3,), (5,), (3, 5), (2,), (2, 5, 6), (3,)] | — | — | — | 0.2896 ± 0 | 0.02169 ± 0 |
| net-04 | — | — | 6 | [(0, 1), (0, 1, 2), (2,), (0,), (2,), (0, 1, 2)] | — | — | — | 0.2677 ± 0 | 0.0228 ± 0 |
| net-05 | — | — | 5 | [(0, 3, 5), (1, 4, 6), (0, 1, 3), (2, 6), (1, 4, 6)] | — | — | — | 0.1339 ± 0 | 0.01583 ± 0 |
| net-06 | — | — | 5 | [(0, 1, 4), (0, 1, 4), (0, 2, 4), (0,), (0, 2)] | — | — | — | 0.09782 ± 0 | 0.0142 ± 0 |
| net-07 | — | — | 5 | [(0,), (0, 2), (0, 1, 2), (0, 1, 2), (0,)] | — | — | — | 0.08643 ± 0 | 0.01339 ± 0 |
| net-08 | — | — | 6 | [(0, 1), (1, 2, 3), (1,), (3,), (0,), (1, 2, 3)] | — | — | — | 0.3083 ± 0 | 0.02117 ± 0 |
| net-09 | — | — | 6 | [(0, 1, 2), (0, 1, 3), (1,), (2,), (0,), (2, 3)] | — | — | — | 0.3134 ± 0 | 0.02172 ± 0 |
| net-10 | — | — | 4 | [(0, 1, 2), (2,), (0, 1, 2), (1,)] | — | — | — | 0.02196 ± 0 | 0.006284 ± 0 |
| net-11 | — | — | 6 | [(3,), (5,), (2, 4), (0, 2), (2, 3, 6), (1,)] | — | — | — | 0.3587 ± 0 | 0.02018 ± 0 |
| net-12 | — | — | 5 | [(1, 2), (0, 1, 2), (2,), (0,), (0,)] | — | — | — | 0.09036 ± 0 | 0.01358 ± 0 |
| net-13 | — | — | 5 | [(1, 2, 4), (0, 2), (1, 3), (4,), (0, 2, 4)] | — | — | — | 0.1107 ± 0 | 0.01532 ± 0 |
| net-14 | — | — | 5 | [(0, 1, 2), (0, 1, 2), (1, 2), (0, 1, 2), (0, 2)] | — | — | — | 0.08664 ± 0 | 0.01338 ± 0 |
| net-15 | — | — | 3 | [(0, 2), (0,), (0, 1, 2)] | — | — | — | 0.00925 ± 0 | 0.004713 ± 0 |
| net-16 | — | — | 4 | [(0, 1, 2), (1, 2, 3), (0, 1, 3), (2,)] | — | — | — | 0.0316 ± 0 | 0.008845 ± 0 |
| net-17 | — | — | 5 | [(2, 3), (0,), (3, 4), (2,), (4,)] | — | — | — | 0.09876 ± 0 | 0.01317 ± 0 |
| net-18 | — | — | 4 | [(1,), (0, 2), (3,), (4,)] | — | — | — | 0.03364 ± 0 | 0.008777 ± 0 |
| net-19 | — | — | 3 | [(2,), (1, 2, 3), (4,)] | — | — | — | 0.009854 ± 0 | 0.004824 ± 0 |
| net-20 | — | — | 4 | [(1,), (1,), (0, 1, 2), (0, 1, 2)] | — | — | — | 0.02697 ± 0 | 0.007884 ± 0 |
| net-21 | — | — | 3 | [(0, 4), (1, 5, 6), (6,)] | — | — | — | 0.0109 ± 0 | 0.005506 ± 0 |
| net-22 | — | — | 4 | [(0, 1, 3), (2,), (0, 1), (0, 1, 3)] | — | — | — | 0.03182 ± 0 | 0.009521 ± 0 |
| net-23 | — | — | 5 | [(4, 6), (0, 3), (1, 2), (0, 5), (0, 4)] | — | — | — | 0.131 ± 0 | 0.01555 ± 0 |
| net-24 | — | — | 5 | [(3, 4), (4,), (1,), (0, 1), (2, 3, 4)] | — | — | — | 0.1158 ± 0 | 0.01519 ± 0 |
| net-25 | — | — | 6 | [(1, 2, 3), (0, 1, 2), (0, 2, 3), (2, 3), (0, 2), (0, 2, 3)] | — | — | — | 0.2513 ± 0 | 0.01873 ± 0 |
| net-26 | — | — | 6 | [(2,), (1, 2), (0, 1, 2), (0, 1, 2), (0, 2, 3), (0, 1)] | — | — | — | 0.2718 ± 0 | 0.0193 ± 0 |
| net-27 | — | — | 3 | [(3,), (1, 3), (1, 3)] | — | — | — | 0.009142 ± 0 | 0.004869 ± 0 |
| net-28 | — | — | 6 | [(0, 1, 2), (0, 2), (2,), (0, 2), (2,), (0, 1, 2)] | — | — | — | 0.2675 ± 0 | 0.02022 ± 0 |
| net-29 | — | — | 4 | [(2, 4), (1, 2, 3), (3, 4), (1,)] | — | — | — | 0.03158 ± 0 | 0.008613 ± 0 |
| net-30 | — | — | 3 | [(0, 1, 2), (0, 1, 2), (0, 1, 2)] | — | — | — | 0.007547 ± 0 | 0.003988 ± 0 |
| net-31 | — | — | 5 | [(1,), (0, 1, 2), (0, 1, 2), (2,), (0, 1)] | — | — | — | 0.07186 ± 0 | 0.01076 ± 0 |
| net-32 | — | — | 4 | [(2, 3, 4), (1, 3, 5), (0, 4), (3, 4, 5)] | — | — | — | 0.03486 ± 0 | 0.008495 ± 0 |
| net-33 | — | — | 4 | [(0, 2, 3), (1, 3, 4), (0,), (0, 1, 2)] | — | — | — | 0.03583 ± 0 | 0.01041 ± 0 |
| net-34 | — | — | 6 | [(1, 3), (0, 2, 3), (4,), (0, 2), (3,), (1,)] | — | — | — | 0.3393 ± 0 | 0.02474 ± 0 |
| net-35 | — | — | 5 | [(1, 4), (1,), (0, 2), (5,), (2, 3, 5)] | — | — | — | 0.1244 ± 0 | 0.01536 ± 0 |
| net-36 | — | — | 5 | [(1, 2, 3), (1, 3), (2, 4), (1, 2, 3), (0, 1, 2)] | — | — | — | 0.1086 ± 0 | 0.01527 ± 0 |
| net-37 | — | — | 4 | [(0, 1, 2), (0, 1, 2), (0, 2), (0,)] | — | — | — | 0.02796 ± 0 | 0.008351 ± 0 |
| net-38 | — | — | 6 | [(1,), (0, 2, 3), (3,), (0, 3), (0,), (0,)] | — | — | — | 0.308 ± 0 | 0.02092 ± 0 |
| net-39 | — | — | 6 | [(0, 2, 4), (0, 1, 3), (1, 3), (1, 3), (1,), (3, 4)] | — | — | — | 0.3414 ± 0 | 0.02269 ± 0 |
| net-40 | — | — | 3 | [(0,), (0,), (1, 2)] | — | — | — | 0.006801 ± 0 | 0.003432 ± 0 |
| net-41 | — | — | 4 | [(0,), (1, 2), (0, 1, 2), (0, 1)] | — | — | — | 0.02242 ± 0 | 0.006393 ± 0 |
| net-42 | — | — | 3 | [(4,), (0,), (0, 6)] | — | — | — | 0.007608 ± 0 | 0.00396 ± 0 |
| net-43 | — | — | 5 | [(1, 2, 3), (0, 1, 2), (0,), (1, 2, 3), (0, 1, 2)] | — | — | — | 0.09579 ± 0 | 0.0139 ± 0 |
| net-44 | — | — | 3 | [(0, 1, 4), (0,), (0,)] | — | — | — | 0.005577 ± 0 | 0.002495 ± 0 |
| net-45 | — | — | 3 | [(2, 4), (2,), (1,)] | — | — | — | 0.006686 ± 0 | 0.003313 ± 0 |
| net-46 | — | — | 3 | [(0, 2), (0, 2), (6,)] | — | — | — | 0.00784 ± 0 | 0.003886 ± 0 |
| net-47 | — | — | 3 | [(3,), (4,), (1, 2, 4)] | — | — | — | 0.01031 ± 0 | 0.005233 ± 0 |
| net-48 | — | — | 4 | [(0, 4, 5), (1, 5), (2,), (2, 4)] | — | — | — | 0.03615 ± 0 | 0.009472 ± 0 |
| net-49 | — | — | 3 | [(1, 2, 4), (2, 4), (0, 2)] | — | — | — | 0.01027 ± 0 | 0.00504 ± 0 |
| net-50 | — | — | 5 | [(3,), (3, 6), (3, 5), (4,), (1,)] | — | — | — | 0.1059 ± 0 | 0.01401 ± 0 |
| net-51 | — | — | 5 | [(1, 2, 3), (2, 4), (0, 1, 4), (2,), (0, 1, 4)] | — | — | — | 0.1096 ± 0 | 0.01505 ± 0 |
| net-52 | — | — | 5 | [(1,), (2,), (0, 1, 2), (2,), (2,)] | — | — | — | 0.08228 ± 0 | 0.01275 ± 0 |
| net-53 | — | — | 5 | [(1, 2, 4), (1, 3, 5), (2, 5), (0, 1, 3), (1, 3)] | — | — | — | 0.1257 ± 0 | 0.01626 ± 0 |
| net-54 | — | — | 6 | [(4, 5, 6), (0, 2), (0, 1), (5, 6), (0, 4, 6), (4, 5)] | — | — | — | 0.4056 ± 0 | 0.02535 ± 0 |
| net-55 | — | — | 6 | [(3, 4, 5), (2, 3), (4,), (1, 4), (1, 2, 6), (0, 3, 4)] | — | — | — | 0.4407 ± 0 | 0.02536 ± 0 |
| net-56 | — | — | 3 | [(0,), (1, 2), (1, 2)] | — | — | — | 0.005059 ± 0 | 0.002587 ± 0 |
| net-57 | — | — | 3 | [(1, 3), (0,), (2,)] | — | — | — | 0.005779 ± 0 | 0.00298 ± 0 |
| net-58 | — | — | 5 | [(2, 3, 5), (1, 3), (3, 5), (0,), (1,)] | — | — | — | 0.07197 ± 0 | 0.009123 ± 0 |
| net-59 | — | — | 6 | [(0, 2), (0,), (0, 2), (0, 1, 2), (2,), (2,)] | — | — | — | 0.1753 ± 0 | 0.01267 ± 0 |

## Metrics

<details><summary><b>chain-00</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 108742.0,
      "ratio": 4.094818496761561,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 26556.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 26556.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 26556.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 26556.0,
  "n_mats": 7,
  "dims": "(27, 3, 17, 33, 32, 26, 20, 31)"
}
```

</details>

<details><summary><b>chain-01</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 21740.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 21740.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 21740.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 21740.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 21740.0,
  "n_mats": 6,
  "dims": "(38, 14, 33, 9, 19, 9, 7)"
}
```

</details>

<details><summary><b>chain-02</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 38990.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 38990.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 38990.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 38990.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 38990.0,
  "n_mats": 8,
  "dims": "(17, 35, 39, 10, 20, 7, 5, 22, 31)"
}
```

</details>

<details><summary><b>chain-03</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 124628.0,
      "ratio": 1.72820811492914,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 72114.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 72114.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 72114.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 72114.0,
  "n_mats": 8,
  "dims": "(7, 23, 28, 21, 40, 14, 36, 31, 29)"
}
```

</details>

<details><summary><b>chain-04</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 4460.0,
      "ratio": 1.0127157129881925,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 4404.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 4404.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 4404.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 4404.0,
  "n_mats": 8,
  "dims": "(17, 4, 36, 1, 6, 26, 1, 40, 32)"
}
```

</details>

<details><summary><b>chain-05</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 24700.0,
      "ratio": 1.322269807280514,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 18680.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 18680.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 18680.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 18680.0,
  "n_mats": 6,
  "dims": "(16, 21, 5, 13, 37, 15, 16)"
}
```

</details>

<details><summary><b>chain-06</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 35868.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 35868.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 35868.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 35868.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 35868.0,
  "n_mats": 5,
  "dims": "(35, 29, 6, 6, 21, 33)"
}
```

</details>

<details><summary><b>chain-07</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 53616.0,
      "ratio": 1.12440231524201,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 47684.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 47684.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 47684.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 47684.0,
  "n_mats": 7,
  "dims": "(7, 20, 36, 19, 8, 36, 22, 35)"
}
```

</details>

<details><summary><b>chain-08</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 48540.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 48540.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 48540.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 48540.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 48540.0,
  "n_mats": 5,
  "dims": "(39, 36, 38, 19, 29, 6)"
}
```

</details>

<details><summary><b>chain-09</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 16428.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 16428.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 16428.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 16428.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 16428.0,
  "n_mats": 8,
  "dims": "(25, 21, 37, 16, 19, 12, 13, 12, 3)"
}
```

</details>

<details><summary><b>chain-10</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 9966.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 9966.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 9966.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 9966.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 9966.0,
  "n_mats": 8,
  "dims": "(17, 31, 5, 6, 9, 10, 3, 6, 35)"
}
```

</details>

<details><summary><b>chain-11</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 143136.0,
      "ratio": 1.0341001040342157,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 138416.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 138416.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 138416.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 138416.0,
  "n_mats": 7,
  "dims": "(34, 18, 34, 16, 14, 38, 27, 38)"
}
```

</details>

<details><summary><b>chain-12</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 36672.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 36672.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 36672.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 36672.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 36672.0,
  "n_mats": 6,
  "dims": "(29, 32, 23, 6, 21, 40, 8)"
}
```

</details>

<details><summary><b>chain-13</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 8656.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 8656.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 8656.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 8656.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 8656.0,
  "n_mats": 7,
  "dims": "(38, 22, 13, 16, 2, 18, 8, 15)"
}
```

</details>

<details><summary><b>chain-14</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 9944.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 9944.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 9944.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 9944.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 9944.0,
  "n_mats": 6,
  "dims": "(11, 22, 28, 4, 7, 10, 15)"
}
```

</details>

<details><summary><b>chain-15</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 11420.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 11420.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 11420.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 11420.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 11420.0,
  "n_mats": 4,
  "dims": "(37, 35, 39, 5, 2)"
}
```

</details>

<details><summary><b>chain-16</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 36608.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 36608.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 36608.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 36608.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 36608.0,
  "n_mats": 4,
  "dims": "(13, 39, 37, 8, 26)"
}
```

</details>

<details><summary><b>chain-17</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 1332.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 1332.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 1332.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 1332.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 1332.0,
  "n_mats": 4,
  "dims": "(24, 8, 3, 39, 2)"
}
```

</details>

<details><summary><b>chain-18</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 3336.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 3336.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 3336.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 3336.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 3336.0,
  "n_mats": 5,
  "dims": "(12, 8, 31, 14, 4, 2)"
}
```

</details>

<details><summary><b>chain-19</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 27830.0,
      "ratio": 1.0182949140139042,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 27330.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 27330.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 27330.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 27330.0,
  "n_mats": 8,
  "dims": "(28, 40, 7, 17, 5, 15, 5, 20, 23)"
}
```

</details>

<details><summary><b>chain-20</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 11622.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 11622.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 11622.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 11622.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 11622.0,
  "n_mats": 7,
  "dims": "(12, 4, 33, 30, 3, 39, 7, 26)"
}
```

</details>

<details><summary><b>chain-21</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 54758.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 54758.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 54758.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 54758.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 54758.0,
  "n_mats": 5,
  "dims": "(17, 23, 31, 37, 11, 14)"
}
```

</details>

<details><summary><b>chain-22</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 33286.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 33286.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 33286.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 33286.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 33286.0,
  "n_mats": 4,
  "dims": "(11, 11, 22, 34, 17)"
}
```

</details>

<details><summary><b>chain-23</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 5376.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 5376.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 5376.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 5376.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 5376.0,
  "n_mats": 4,
  "dims": "(39, 29, 12, 1, 31)"
}
```

</details>

<details><summary><b>chain-24</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 93460.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 93460.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 93460.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 93460.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 93460.0,
  "n_mats": 7,
  "dims": "(37, 33, 20, 23, 25, 17, 10, 36)"
}
```

</details>

<details><summary><b>chain-25</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 8172.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 8172.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 8172.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 8172.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 8172.0,
  "n_mats": 4,
  "dims": "(30, 6, 22, 3, 35)"
}
```

</details>

<details><summary><b>chain-26</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 94024.0,
      "ratio": 1.5705218146589164,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 59868.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 59868.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 59868.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 59868.0,
  "n_mats": 6,
  "dims": "(9, 16, 31, 23, 40, 19, 23)"
}
```

</details>

<details><summary><b>chain-27</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 5808.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 5808.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 5808.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 5808.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 5808.0,
  "n_mats": 8,
  "dims": "(40, 9, 20, 25, 27, 6, 1, 39, 13)"
}
```

</details>

<details><summary><b>chain-28</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 106140.0,
      "ratio": 1.4514276337381031,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 73128.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 73128.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 73128.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 73128.0,
  "n_mats": 6,
  "dims": "(11, 16, 15, 29, 25, 37, 27)"
}
```

</details>

<details><summary><b>chain-29</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 13482.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 13482.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 13482.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 13482.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 13482.0,
  "n_mats": 4,
  "dims": "(26, 37, 27, 3, 11)"
}
```

</details>

<details><summary><b>chain-30</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 203264.0,
      "ratio": 3.957632398753894,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 51360.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 51360.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 51360.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 51360.0,
  "n_mats": 7,
  "dims": "(5, 17, 11, 29, 34, 32, 36, 39)"
}
```

</details>

<details><summary><b>chain-31</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 33012.0,
      "ratio": 3.25177304964539,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 10152.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 10152.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 10152.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 10152.0,
  "n_mats": 4,
  "dims": "(3, 32, 21, 20, 30)"
}
```

</details>

<details><summary><b>chain-32</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 12744.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 12744.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 12744.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 12744.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 12744.0,
  "n_mats": 4,
  "dims": "(27, 13, 36, 6, 9)"
}
```

</details>

<details><summary><b>chain-33</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 3266.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 3266.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 3266.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 3266.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 3266.0,
  "n_mats": 4,
  "dims": "(26, 27, 21, 1, 14)"
}
```

</details>

<details><summary><b>chain-34</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 11040.0,
      "ratio": 3.1889081455805894,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 3462.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 3462.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 3462.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 3462.0,
  "n_mats": 4,
  "dims": "(1, 34, 40, 7, 13)"
}
```

</details>

<details><summary><b>chain-35</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 27048.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 27048.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 27048.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 27048.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 27048.0,
  "n_mats": 4,
  "dims": "(39, 13, 20, 18, 12)"
}
```

</details>

<details><summary><b>chain-36</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 6080.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 6080.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 6080.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 6080.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 6080.0,
  "n_mats": 4,
  "dims": "(31, 26, 6, 2, 18)"
}
```

</details>

<details><summary><b>chain-37</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 27216.0,
      "ratio": 1.0822331795769047,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 25148.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 25148.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 25148.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 25148.0,
  "n_mats": 7,
  "dims": "(8, 17, 9, 34, 23, 8, 10, 18)"
}
```

</details>

<details><summary><b>chain-38</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 5406.0,
      "ratio": 1.0100896860986548,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 5352.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 5352.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 5352.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 5352.0,
  "n_mats": 4,
  "dims": "(3, 3, 14, 17, 36)"
}
```

</details>

<details><summary><b>chain-39</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 75984.0,
      "ratio": 2.748263888888889,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 27648.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 27648.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 27648.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 27648.0,
  "n_mats": 6,
  "dims": "(24, 37, 3, 39, 32, 30, 28)"
}
```

</details>

<details><summary><b>chain-40</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 5220.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 5220.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 5220.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 5220.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 5220.0,
  "n_mats": 6,
  "dims": "(35, 12, 14, 25, 38, 19, 1)"
}
```

</details>

<details><summary><b>chain-41</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 19056.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 19056.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 19056.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 19056.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 19056.0,
  "n_mats": 5,
  "dims": "(10, 18, 22, 22, 24, 6)"
}
```

</details>

<details><summary><b>chain-42</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 16620.0,
      "ratio": 1.2377122430741734,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 13428.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 13428.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 13428.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 13428.0,
  "n_mats": 6,
  "dims": "(40, 3, 3, 18, 11, 10, 38)"
}
```

</details>

<details><summary><b>chain-43</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 44784.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 44784.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 44784.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 44784.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 44784.0,
  "n_mats": 6,
  "dims": "(24, 26, 36, 9, 19, 8, 31)"
}
```

</details>

<details><summary><b>chain-44</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 8080.0,
      "ratio": 1.10989010989011,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 7280.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 7280.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 7280.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 7280.0,
  "n_mats": 5,
  "dims": "(4, 20, 12, 34, 5, 20)"
}
```

</details>

<details><summary><b>chain-45</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 54992.0,
      "ratio": 1.0126321216808456,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 54306.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 54306.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 54306.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 54306.0,
  "n_mats": 7,
  "dims": "(22, 20, 27, 7, 7, 36, 31, 31)"
}
```

</details>

<details><summary><b>chain-46</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 10944.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 10944.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 10944.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 10944.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 10944.0,
  "n_mats": 6,
  "dims": "(22, 8, 31, 8, 32, 28, 3)"
}
```

</details>

<details><summary><b>chain-47</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 18120.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 18120.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 18120.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 18120.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 18120.0,
  "n_mats": 6,
  "dims": "(22, 10, 11, 37, 25, 6, 5)"
}
```

</details>

<details><summary><b>chain-48</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 710.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 710.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 710.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 710.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 710.0,
  "n_mats": 4,
  "dims": "(13, 15, 4, 25, 1)"
}
```

</details>

<details><summary><b>chain-49</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 110732.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 110732.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 110732.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 110732.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 110732.0,
  "n_mats": 4,
  "dims": "(26, 36, 34, 19, 29)"
}
```

</details>

<details><summary><b>chain-50</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 53304.0,
      "ratio": 1.224028658032516,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 43548.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 43548.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 43548.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 43548.0,
  "n_mats": 7,
  "dims": "(38, 14, 28, 6, 24, 15, 17, 38)"
}
```

</details>

<details><summary><b>chain-51</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 3548.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 3548.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 3548.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 3548.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 3548.0,
  "n_mats": 5,
  "dims": "(28, 13, 23, 8, 5, 2)"
}
```

</details>

<details><summary><b>chain-52</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 15930.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 15930.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 15930.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 15930.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 15930.0,
  "n_mats": 8,
  "dims": "(29, 13, 8, 32, 26, 17, 14, 3, 14)"
}
```

</details>

<details><summary><b>chain-53</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 43274.0,
      "ratio": 1.0068403908794787,
      "rank": 3.5
    },
    "greedy-episode": {
      "cost": 43274.0,
      "ratio": 1.0068403908794787,
      "rank": 3.5
    },
    "saturate": {
      "cost": 42980.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 42980.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 42980.0,
  "n_mats": 8,
  "dims": "(10, 7, 13, 30, 25, 24, 35, 10, 7)"
}
```

</details>

<details><summary><b>chain-54</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 206120.0,
      "ratio": 1.690062315513283,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 121960.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 121960.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 121960.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 121960.0,
  "n_mats": 8,
  "dims": "(32, 10, 37, 26, 28, 34, 32, 21, 32)"
}
```

</details>

<details><summary><b>chain-55</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 7262.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 7262.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 7262.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 7262.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 7262.0,
  "n_mats": 7,
  "dims": "(13, 35, 40, 15, 1, 22, 21, 21)"
}
```

</details>

<details><summary><b>chain-56</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 23460.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 23460.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 23460.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 23460.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 23460.0,
  "n_mats": 4,
  "dims": "(34, 10, 17, 39, 10)"
}
```

</details>

<details><summary><b>chain-57</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 11340.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 11340.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 11340.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 11340.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 11340.0,
  "n_mats": 7,
  "dims": "(38, 19, 31, 5, 6, 34, 3, 5)"
}
```

</details>

<details><summary><b>chain-58</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 1846.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "greedy-episode": {
      "cost": 1846.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "saturate": {
      "cost": 1846.0,
      "ratio": 1.0,
      "rank": 2.5
    },
    "optimal": {
      "cost": 1846.0,
      "ratio": 1.0,
      "rank": 2.5
    }
  },
  "optimal": 1846.0,
  "n_mats": 5,
  "dims": "(9, 3, 20, 1, 29, 22)"
}
```

</details>

<details><summary><b>chain-59</b> — raw metrics</summary>

```json
{
  "family": "chain",
  "players": {
    "greedy-1step": {
      "cost": 107600.0,
      "ratio": 1.688107938500157,
      "rank": 4.0
    },
    "greedy-episode": {
      "cost": 63740.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "saturate": {
      "cost": 63740.0,
      "ratio": 1.0,
      "rank": 2.0
    },
    "optimal": {
      "cost": 63740.0,
      "ratio": 1.0,
      "rank": 2.0
    }
  },
  "optimal": 63740.0,
  "n_mats": 5,
  "dims": "(10, 30, 24, 33, 25, 34)"
}
```

</details>

<details><summary><b>net-00</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 588.0,
      "ratio": 1.1620553359683794,
      "rank": 2.0
    },
    "optimal": {
      "cost": 506.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 506.0,
  "n_tensors": 4,
  "tensors": "[(2,), (3,), (3, 5), (1, 6)]"
}
```

</details>

<details><summary><b>net-01</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 6360.0,
      "ratio": 5.36256323777403,
      "rank": 2.0
    },
    "optimal": {
      "cost": 1186.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 1186.0,
  "n_tensors": 6,
  "tensors": "[(0, 1, 4), (0,), (4,), (3,), (1, 2, 3), (4,)]"
}
```

</details>

<details><summary><b>net-02</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 1611.0,
      "ratio": 1.028735632183908,
      "rank": 2.0
    },
    "optimal": {
      "cost": 1566.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 1566.0,
  "n_tensors": 5,
  "tensors": "[(1, 2, 3), (3,), (0, 1, 3), (2,), (0, 2, 3)]"
}
```

</details>

<details><summary><b>net-03</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 799.0,
      "ratio": 1.9023809523809523,
      "rank": 2.0
    },
    "optimal": {
      "cost": 420.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 420.0,
  "n_tensors": 6,
  "tensors": "[(3,), (5,), (3, 5), (2,), (2, 5, 6), (3,)]"
}
```

</details>

<details><summary><b>net-04</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 797.0,
      "ratio": 1.0127064803049555,
      "rank": 2.0
    },
    "optimal": {
      "cost": 787.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 787.0,
  "n_tensors": 6,
  "tensors": "[(0, 1), (0, 1, 2), (2,), (0,), (2,), (0, 1, 2)]"
}
```

</details>

<details><summary><b>net-05</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 3112.0,
      "ratio": 1.0169934640522875,
      "rank": 2.0
    },
    "optimal": {
      "cost": 3060.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 3060.0,
  "n_tensors": 5,
  "tensors": "[(0, 3, 5), (1, 4, 6), (0, 1, 3), (2, 6), (1, 4, 6)]"
}
```

</details>

<details><summary><b>net-06</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 215.0,
      "ratio": 1.0858585858585859,
      "rank": 2.0
    },
    "optimal": {
      "cost": 198.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 198.0,
  "n_tensors": 5,
  "tensors": "[(0, 1, 4), (0, 1, 4), (0, 2, 4), (0,), (0, 2)]"
}
```

</details>

<details><summary><b>net-07</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 426.0,
      "ratio": 1.1064935064935064,
      "rank": 2.0
    },
    "optimal": {
      "cost": 385.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 385.0,
  "n_tensors": 5,
  "tensors": "[(0,), (0, 2), (0, 1, 2), (0, 1, 2), (0,)]"
}
```

</details>

<details><summary><b>net-08</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 680.0,
      "ratio": 1.021021021021021,
      "rank": 2.0
    },
    "optimal": {
      "cost": 666.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 666.0,
  "n_tensors": 6,
  "tensors": "[(0, 1), (1, 2, 3), (1,), (3,), (0,), (1, 2, 3)]"
}
```

</details>

<details><summary><b>net-09</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 648.0,
      "ratio": 1.0156739811912225,
      "rank": 2.0
    },
    "optimal": {
      "cost": 638.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 638.0,
  "n_tensors": 6,
  "tensors": "[(0, 1, 2), (0, 1, 3), (1,), (2,), (0,), (2, 3)]"
}
```

</details>

<details><summary><b>net-10</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 275.0,
      "ratio": 1.2222222222222223,
      "rank": 2.0
    },
    "optimal": {
      "cost": 225.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 225.0,
  "n_tensors": 4,
  "tensors": "[(0, 1, 2), (2,), (0, 1, 2), (1,)]"
}
```

</details>

<details><summary><b>net-11</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 439537.0,
      "ratio": 4.031192105214887,
      "rank": 2.0
    },
    "optimal": {
      "cost": 109034.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 109034.0,
  "n_tensors": 6,
  "tensors": "[(3,), (5,), (2, 4), (0, 2), (2, 3, 6), (1,)]"
}
```

</details>

<details><summary><b>net-12</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 259.0,
      "ratio": 1.1359649122807018,
      "rank": 2.0
    },
    "optimal": {
      "cost": 228.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 228.0,
  "n_tensors": 5,
  "tensors": "[(1, 2), (0, 1, 2), (2,), (0,), (0,)]"
}
```

</details>

<details><summary><b>net-13</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 736.0,
      "ratio": 1.2777777777777777,
      "rank": 2.0
    },
    "optimal": {
      "cost": 576.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 576.0,
  "n_tensors": 5,
  "tensors": "[(1, 2, 4), (0, 2), (1, 3), (4,), (0, 2, 4)]"
}
```

</details>

<details><summary><b>net-14</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 308.0,
      "ratio": 1.0694444444444444,
      "rank": 2.0
    },
    "optimal": {
      "cost": 288.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 288.0,
  "n_tensors": 5,
  "tensors": "[(0, 1, 2), (0, 1, 2), (1, 2), (0, 1, 2), (0, 2)]"
}
```

</details>

<details><summary><b>net-15</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 540.0,
      "ratio": 1.1538461538461537,
      "rank": 2.0
    },
    "optimal": {
      "cost": 468.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 468.0,
  "n_tensors": 3,
  "tensors": "[(0, 2), (0,), (0, 1, 2)]"
}
```

</details>

<details><summary><b>net-16</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 56.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 56.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 56.0,
  "n_tensors": 4,
  "tensors": "[(0, 1, 2), (1, 2, 3), (0, 1, 3), (2,)]"
}
```

</details>

<details><summary><b>net-17</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 266.0,
      "ratio": 1.8732394366197183,
      "rank": 2.0
    },
    "optimal": {
      "cost": 142.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 142.0,
  "n_tensors": 5,
  "tensors": "[(2, 3), (0,), (3, 4), (2,), (4,)]"
}
```

</details>

<details><summary><b>net-18</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 20538.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 20538.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 20538.0,
  "n_tensors": 4,
  "tensors": "[(1,), (0, 2), (3,), (4,)]"
}
```

</details>

<details><summary><b>net-19</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 1870.0,
      "ratio": 3.5416666666666665,
      "rank": 2.0
    },
    "optimal": {
      "cost": 528.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 528.0,
  "n_tensors": 3,
  "tensors": "[(2,), (1, 2, 3), (4,)]"
}
```

</details>

<details><summary><b>net-20</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 334.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 334.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 334.0,
  "n_tensors": 4,
  "tensors": "[(1,), (1,), (0, 1, 2), (0, 1, 2)]"
}
```

</details>

<details><summary><b>net-21</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 2916.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 2916.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 2916.0,
  "n_tensors": 3,
  "tensors": "[(0, 4), (1, 5, 6), (6,)]"
}
```

</details>

<details><summary><b>net-22</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 2808.0,
      "ratio": 1.4385245901639345,
      "rank": 2.0
    },
    "optimal": {
      "cost": 1952.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 1952.0,
  "n_tensors": 4,
  "tensors": "[(0, 1, 3), (2,), (0, 1), (0, 1, 3)]"
}
```

</details>

<details><summary><b>net-23</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 10104.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 10104.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 10104.0,
  "n_tensors": 5,
  "tensors": "[(4, 6), (0, 3), (1, 2), (0, 5), (0, 4)]"
}
```

</details>

<details><summary><b>net-24</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 3285.0,
      "ratio": 3.421875,
      "rank": 2.0
    },
    "optimal": {
      "cost": 960.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 960.0,
  "n_tensors": 5,
  "tensors": "[(3, 4), (4,), (1,), (0, 1), (2, 3, 4)]"
}
```

</details>

<details><summary><b>net-25</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 4356.0,
      "ratio": 2.314558979808714,
      "rank": 2.0
    },
    "optimal": {
      "cost": 1882.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 1882.0,
  "n_tensors": 6,
  "tensors": "[(1, 2, 3), (0, 1, 2), (0, 2, 3), (2, 3), (0, 2), (0, 2, 3)]"
}
```

</details>

<details><summary><b>net-26</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 587.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 587.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 587.0,
  "n_tensors": 6,
  "tensors": "[(2,), (1, 2), (0, 1, 2), (0, 1, 2), (0, 2, 3), (0, 1)]"
}
```

</details>

<details><summary><b>net-27</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 54.0,
      "ratio": 1.8,
      "rank": 2.0
    },
    "optimal": {
      "cost": 30.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 30.0,
  "n_tensors": 3,
  "tensors": "[(3,), (1, 3), (1, 3)]"
}
```

</details>

<details><summary><b>net-28</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 118.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 118.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 118.0,
  "n_tensors": 6,
  "tensors": "[(0, 1, 2), (0, 2), (2,), (0, 2), (2,), (0, 1, 2)]"
}
```

</details>

<details><summary><b>net-29</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 360.0,
      "ratio": 1.0650887573964498,
      "rank": 2.0
    },
    "optimal": {
      "cost": 338.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 338.0,
  "n_tensors": 4,
  "tensors": "[(2, 4), (1, 2, 3), (3, 4), (1,)]"
}
```

</details>

<details><summary><b>net-30</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 1680.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 1680.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 1680.0,
  "n_tensors": 3,
  "tensors": "[(0, 1, 2), (0, 1, 2), (0, 1, 2)]"
}
```

</details>

<details><summary><b>net-31</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 264.0,
      "ratio": 1.306930693069307,
      "rank": 2.0
    },
    "optimal": {
      "cost": 202.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 202.0,
  "n_tensors": 5,
  "tensors": "[(1,), (0, 1, 2), (0, 1, 2), (2,), (0, 1)]"
}
```

</details>

<details><summary><b>net-32</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 22575.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 22575.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 22575.0,
  "n_tensors": 4,
  "tensors": "[(2, 3, 4), (1, 3, 5), (0, 4), (3, 4, 5)]"
}
```

</details>

<details><summary><b>net-33</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 7200.0,
      "ratio": 1.25,
      "rank": 2.0
    },
    "optimal": {
      "cost": 5760.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 5760.0,
  "n_tensors": 4,
  "tensors": "[(0, 2, 3), (1, 3, 4), (0,), (0, 1, 2)]"
}
```

</details>

<details><summary><b>net-34</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 384.0,
      "ratio": 1.0105263157894737,
      "rank": 2.0
    },
    "optimal": {
      "cost": 380.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 380.0,
  "n_tensors": 6,
  "tensors": "[(1, 3), (0, 2, 3), (4,), (0, 2), (3,), (1,)]"
}
```

</details>

<details><summary><b>net-35</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 1188.0,
      "ratio": 2.495798319327731,
      "rank": 2.0
    },
    "optimal": {
      "cost": 476.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 476.0,
  "n_tensors": 5,
  "tensors": "[(1, 4), (1,), (0, 2), (5,), (2, 3, 5)]"
}
```

</details>

<details><summary><b>net-36</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 15448.0,
      "ratio": 5.501424501424501,
      "rank": 2.0
    },
    "optimal": {
      "cost": 2808.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 2808.0,
  "n_tensors": 5,
  "tensors": "[(1, 2, 3), (1, 3), (2, 4), (1, 2, 3), (0, 1, 2)]"
}
```

</details>

<details><summary><b>net-37</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 438.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 438.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 438.0,
  "n_tensors": 4,
  "tensors": "[(0, 1, 2), (0, 1, 2), (0, 2), (0,)]"
}
```

</details>

<details><summary><b>net-38</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 1199.0,
      "ratio": 3.622356495468278,
      "rank": 2.0
    },
    "optimal": {
      "cost": 331.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 331.0,
  "n_tensors": 6,
  "tensors": "[(1,), (0, 2, 3), (3,), (0, 3), (0,), (0,)]"
}
```

</details>

<details><summary><b>net-39</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 972.0,
      "ratio": 1.311740890688259,
      "rank": 2.0
    },
    "optimal": {
      "cost": 741.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 741.0,
  "n_tensors": 6,
  "tensors": "[(0, 2, 4), (0, 1, 3), (1, 3), (1, 3), (1,), (3, 4)]"
}
```

</details>

<details><summary><b>net-40</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 41.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 41.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 41.0,
  "n_tensors": 3,
  "tensors": "[(0,), (0,), (1, 2)]"
}
```

</details>

<details><summary><b>net-41</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 828.0,
      "ratio": 1.0823529411764705,
      "rank": 2.0
    },
    "optimal": {
      "cost": 765.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 765.0,
  "n_tensors": 4,
  "tensors": "[(0,), (1, 2), (0, 1, 2), (0, 1)]"
}
```

</details>

<details><summary><b>net-42</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 68.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 68.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 68.0,
  "n_tensors": 3,
  "tensors": "[(4,), (0,), (0, 6)]"
}
```

</details>

<details><summary><b>net-43</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 920.0,
      "ratio": 1.0098792535675083,
      "rank": 2.0
    },
    "optimal": {
      "cost": 911.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 911.0,
  "n_tensors": 5,
  "tensors": "[(1, 2, 3), (0, 1, 2), (0,), (1, 2, 3), (0, 1, 2)]"
}
```

</details>

<details><summary><b>net-44</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 485.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 485.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 485.0,
  "n_tensors": 3,
  "tensors": "[(0, 1, 4), (0,), (0,)]"
}
```

</details>

<details><summary><b>net-45</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 90.0,
      "ratio": 2.0454545454545454,
      "rank": 2.0
    },
    "optimal": {
      "cost": 44.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 44.0,
  "n_tensors": 3,
  "tensors": "[(2, 4), (2,), (1,)]"
}
```

</details>

<details><summary><b>net-46</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 56.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 56.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 56.0,
  "n_tensors": 3,
  "tensors": "[(0, 2), (0, 2), (6,)]"
}
```

</details>

<details><summary><b>net-47</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 2403.0,
      "ratio": 2.2755681818181817,
      "rank": 2.0
    },
    "optimal": {
      "cost": 1056.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 1056.0,
  "n_tensors": 3,
  "tensors": "[(3,), (4,), (1, 2, 4)]"
}
```

</details>

<details><summary><b>net-48</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 248.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 248.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 248.0,
  "n_tensors": 4,
  "tensors": "[(0, 4, 5), (1, 5), (2,), (2, 4)]"
}
```

</details>

<details><summary><b>net-49</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 204.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 204.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 204.0,
  "n_tensors": 3,
  "tensors": "[(1, 2, 4), (2, 4), (0, 2)]"
}
```

</details>

<details><summary><b>net-50</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 36685.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 36685.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 36685.0,
  "n_tensors": 5,
  "tensors": "[(3,), (3, 6), (3, 5), (4,), (1,)]"
}
```

</details>

<details><summary><b>net-51</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 660.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 660.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 660.0,
  "n_tensors": 5,
  "tensors": "[(1, 2, 3), (2, 4), (0, 1, 4), (2,), (0, 1, 4)]"
}
```

</details>

<details><summary><b>net-52</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 211.0,
      "ratio": 1.0292682926829269,
      "rank": 2.0
    },
    "optimal": {
      "cost": 205.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 205.0,
  "n_tensors": 5,
  "tensors": "[(1,), (2,), (0, 1, 2), (2,), (2,)]"
}
```

</details>

<details><summary><b>net-53</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 9968.0,
      "ratio": 3.5049226441631505,
      "rank": 2.0
    },
    "optimal": {
      "cost": 2844.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 2844.0,
  "n_tensors": 5,
  "tensors": "[(1, 2, 4), (1, 3, 5), (2, 5), (0, 1, 3), (1, 3)]"
}
```

</details>

<details><summary><b>net-54</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 510120.0,
      "ratio": 1.0077240670867822,
      "rank": 2.0
    },
    "optimal": {
      "cost": 506210.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 506210.0,
  "n_tensors": 6,
  "tensors": "[(4, 5, 6), (0, 2), (0, 1), (5, 6), (0, 4, 6), (4, 5)]"
}
```

</details>

<details><summary><b>net-55</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 9322.0,
      "ratio": 1.8878088294856217,
      "rank": 2.0
    },
    "optimal": {
      "cost": 4938.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 4938.0,
  "n_tensors": 6,
  "tensors": "[(3, 4, 5), (2, 3), (4,), (1, 4), (1, 2, 6), (0, 3, 4)]"
}
```

</details>

<details><summary><b>net-56</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 79.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 79.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 79.0,
  "n_tensors": 3,
  "tensors": "[(0,), (1, 2), (1, 2)]"
}
```

</details>

<details><summary><b>net-57</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 1925.0,
      "ratio": 1.0,
      "rank": 1.5
    },
    "optimal": {
      "cost": 1925.0,
      "ratio": 1.0,
      "rank": 1.5
    }
  },
  "optimal": 1925.0,
  "n_tensors": 3,
  "tensors": "[(1, 3), (0,), (2,)]"
}
```

</details>

<details><summary><b>net-58</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 139.0,
      "ratio": 2.0144927536231885,
      "rank": 2.0
    },
    "optimal": {
      "cost": 69.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 69.0,
  "n_tensors": 5,
  "tensors": "[(2, 3, 5), (1, 3), (3, 5), (0,), (1,)]"
}
```

</details>

<details><summary><b>net-59</b> — raw metrics</summary>

```json
{
  "family": "net",
  "players": {
    "greedy-classic": {
      "cost": 1068.0,
      "ratio": 1.0084985835694051,
      "rank": 2.0
    },
    "optimal": {
      "cost": 1059.0,
      "ratio": 1.0,
      "rank": 1.0
    }
  },
  "optimal": 1059.0,
  "n_tensors": 6,
  "tensors": "[(0, 2), (0,), (0, 2), (0, 1, 2), (2,), (2,)]"
}
```

</details>

