# Novelty sieve

Automatic three-gate filter between a minted law and the human-review queue.  **cert** = the law's own rewrite carries a standalone-replayable certificate on its instance (`verify_law` + `verify_certificate`); **derivable** = the widest shipped bank (`default_rules` + `ALL_RULES_WITH_LAYOUT` + `ATTENTION` + scan monoids + the prior-art FUSION/CATEGORICAL slice — `known_bank`), minus the candidate and its alpha-siblings, already proves the instance — derivable laws are theorems of known rules, demoted silently; **beat** = on some `bench/discovery` case the certified extraction (run with the candidate in the rules) lands strictly below the best registered baseline AND the derivation certificate invokes the candidate (the `beat_best` floor plus attribution).  When no registered case exercises the law the gate falls back to the held-out `mlopt_discovery.zoo` registry: every spelled zoo term the LHS strictly matches (`zoo:`-prefixed rows) is measured as a two-arm e-graph site whose floor is the spelled cost AND the bank's own extraction under the same premise bank — a strictly stronger bar than the case arm's registered baselines, since the candidate must pay beyond everything else the library knows.  A candidate is a novelty iff cert ∧ beat ∧ ¬derivable.  `truth` is the fp64 oracle's numeric verdict on the instance — an `unequal` refutes outright.

- date: 2026-10-23T04:53:54+00:00 · git: `9951ca1` (dirty) · torch 2.14.0+cu130 on cpu · bank 242 rules
- verdicts: known=6, unmeasured=11, unpaid=7

| law | prov | instance | truth | cert | derivable | beat | verdict |
|---|---|---|---|---|---|---|---|
| `add_unsq_pad_r` | machine | zoo:ALiBiSlopeAttention | equal | yes | no | no | **unpaid** |
| `chunk_single` | machine | synth | equal | yes | no | no | **unpaid** |
| `conv_bn_fold` | machine | corpus:conv_bn_fold | equal | yes | no | no | **unpaid** |
| `linear_channel_to_row_scale` | machine | synth | equal | yes | yes (linear_channel_scale_rev, linear_row_scale) | — | **known** |
| `mul_reshape_inert_l` | machine | synth | equal | yes | no | no | **unpaid** |
| `mul_unsq_nat` | machine | corpus:mul_unsq_nat | equal | yes | yes (mul_unsq_rev) | — | **known** |
| `mul_unsq_pad_l` | machine | corpus:zoo_time_cond_conv | equal | yes | yes (comm_mul, mul_unsq_pad_r) | — | **known** |
| `mul_unsq_pad_r` | machine | corpus:zoo_time_cond_conv | equal | yes | yes (comm_mul, mul_unsq_pad_l) | — | **known** |
| `sdpa_fold_div_nomask` | machine | registry | equal | yes | no | no | **unpaid** |
| `sdpa_fold_nomask` | machine | registry | equal | yes | no | no | **unpaid** |
| `softsign_fold` | machine | corpus:softsign_fold | equal | yes | yes (softsign_expand) | — | **known** |
| `sub_unsq_pad_l` | machine | zoo:RWKVMixBlock | equal | yes | yes | — | **known** |
| `transpose_noop` | machine | synth | equal | yes | no | no | **unpaid** |
| `cand_mul_factor` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_mul_factor_right` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_neg_factor` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_square_neg` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_exp_factor` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_mul_zero` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_mul_zero_left` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_pow_one` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_sub_self` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_add_inv` | human | synth | equal | yes | no | — | **unmeasured** |
| `cand_div_self` | human | synth | equal | yes | no | — | **unmeasured** |

## Review queue

**Empty.** Every candidate either fails a gate or is a theorem of the bank — nothing currently needs a reviewer's eyes.

## Per-law detail

### `add_unsq_pad_r` — unpaid

- instance (zoo:ALiBiSlopeAttention): `(add (mul (matmul (transpose (reshape (chunk (linear x, p_qkv_weight, p_qkv_bias), chunks=3, dim=-1, index=0), shape=(2, 8, 4, 4)), dim0=1, dim1=2), (transpose (transpose (reshape (chunk (linear x, p_qkv_weight, p_qkv_bias), chunks=3, dim=-1, index=1), shape=(2, 8, 4, 4)), dim0=1, dim1=2), dim0=-1, dim1=-2)), 0.5), (unsqueeze (mul (reshape p_slopes, shape=(4, 1, 1)), (unsqueeze (sub (unsqueeze (arange arg0=8, dtype=float64, pin_memory=False), dim=0), (unsqueeze (arange arg0=8, dtype=float64, pin_memory=False), dim=1)), dim=0)), dim=0))` → `(add (mul (matmul (transpose (reshape (chunk (linear x, p_qkv_weight, p_qkv_bias), chunks=3, dim=-1, index=0), shape=(2, 8, 4, 4)), dim0=1, dim1=2), (transpose (transpose (reshape (chunk (linear x, p_qkv_weight, p_qkv_bias), chunks=3, dim=-1, index=1), shape=(2, 8, 4, 4)), dim0=1, dim1=2), dim0=-1, dim1=-2)), 0.5), (mul (reshape p_slopes, shape=(4, 1, 1)), (unsqueeze (sub (unsqueeze (arange arg0=8, dtype=float64, pin_memory=False), dim=0), (unsqueeze (arange arg0=8, dtype=float64, pin_memory=False), dim=1)), dim=0)))`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: add_unsq_pad_r
- beat: no — 1 case(s) run
  - `zoo_time_cond_conv`: cert yes, fired 0, in-deriv no, cost 226488 → 209088 vs floor 226488 → beat no

### `chunk_single` — unpaid

- instance (synth): `(chunk u, chunks=1, dim=-1, index=0)` → `u`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: chunk_single
- beat: no — 3 case(s) run
  - `glu_fold`: cert yes, fired 0, in-deriv no, cost 98 → 33 vs floor 33 → beat no
  - `zoo_adaln_affd`: cert yes, fired 0, in-deriv no, cost 217830 → 200240 vs floor 217830 → beat no
  - `zoo_cosine_attention`: cert yes, fired 0, in-deriv no, cost 348463 → 348463 vs floor 348463 → beat no

### `conv_bn_fold` — unpaid

- instance (corpus:conv_bn_fold): `(batch_norm (conv2d input, p_0_weight, p_0_bias, stride=(1, 1), padding=(1, 1)), p_1_weight, p_1_bias, b_1_running_mean, b_1_running_var, training=False, momentum=0.1, eps=1e-05, cudnn_enabled=False)` → `(conv2d input, (mul (mul p_0_weight, (reshape (rsqrt (add (reshape b_1_running_var, shape=(1, 6, 1, 1)), 1e-05)), shape=(6, 1, 1, 1))), (reshape (reshape p_1_weight, shape=(1, 6, 1, 1)), shape=(6, 1, 1, 1))), (add (mul (mul (sub p_0_bias, (reshape (reshape b_1_running_mean, shape=(1, 6, 1, 1)), shape=(6,))), (reshape (rsqrt (add (reshape b_1_running_var, shape=(1, 6, 1, 1)), 1e-05)), shape=(6,))), (reshape (reshape p_1_weight, shape=(1, 6, 1, 1)), shape=(6,))), (reshape (reshape p_1_bias, shape=(1, 6, 1, 1)), shape=(6,))), padding=(1, 1), stride=(1, 1))`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: conv_bn_fold
- beat: no — 1 case(s) run
  - `conv_bn_fold`: cert yes, fired 1, in-deriv yes, cost 29186 → 27649 vs floor 27649 → beat no

### `linear_channel_to_row_scale` — known

- instance (synth): `(linear x, (mul W, c))` → `(mul (linear x, W), c)`
- truth: equal · cert: yes
- derivable: yes via linear_channel_scale_rev, linear_row_scale — fixed_point
- siblings excluded: linear_channel_to_row_scale, linear_out_scale_rev
- beat: — — —

### `mul_reshape_inert_l` — unpaid

- instance (synth): `(mul (reshape u, shape=(4,)), v)` → `(mul u, v)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: mul_reshape_inert_l
- beat: no — 0 case(s) run, 3 zoo site(s) run
  - `zoo:ManualGroupNorm`: cert yes, fired 1, in-deriv yes, cost 2597 → 2060 vs floor 2061 → beat no
  - `zoo:RMSNormPerHead`: cert yes, fired 3, in-deriv yes, cost 1350 → 514 vs floor 514 → beat no
  - `zoo:ALiBiSlopeAttention`: cert yes, fired 0, in-deriv no, cost 42503 → 2e+15 vs floor — → beat no

### `mul_unsq_nat` — known

- instance (corpus:mul_unsq_nat): `(mul (unsqueeze a, dim=-1), (unsqueeze b, dim=-1))` → `(unsqueeze (mul a, b), dim=-1)`
- truth: equal · cert: yes
- derivable: yes via mul_unsq_rev — fixed_point
- siblings excluded: mul_unsq_nat
- beat: — — —

### `mul_unsq_pad_l` — known

- instance (corpus:zoo_time_cond_conv): `(mul (unsqueeze t, dim=-1), (unsqueeze (exp (mul (arange arg0=4, dtype=float64, pin_memory=False), (div (neg (log (detach_ (contiguous c_lifted_tensor_0)))), 4))), dim=0))` → `(mul t, (unsqueeze (exp (mul (arange arg0=4, dtype=float64, pin_memory=False), (div (neg (log (detach_ (contiguous c_lifted_tensor_0)))), 4))), dim=0))`
- truth: equal · cert: yes
- derivable: yes via comm_mul, mul_unsq_pad_r — max_iterations
- siblings excluded: mul_unsq_pad_l
- beat: — — —

### `mul_unsq_pad_r` — known

- instance (corpus:zoo_time_cond_conv): `(mul (unsqueeze t, dim=-1), (unsqueeze (exp (mul (arange arg0=4, dtype=float64, pin_memory=False), (div (neg (log (detach_ (contiguous c_lifted_tensor_0)))), 4))), dim=0))` → `(mul (unsqueeze t, dim=-1), (exp (mul (arange arg0=4, dtype=float64, pin_memory=False), (div (neg (log (detach_ (contiguous c_lifted_tensor_0)))), 4))))`
- truth: equal · cert: yes
- derivable: yes via comm_mul, mul_unsq_pad_l — max_iterations
- siblings excluded: mul_unsq_pad_r
- beat: — — —

### `sdpa_fold_div_nomask` — unpaid

- instance (registry): `(matmul (softmax (div (matmul Q, (transpose K, dim0=-2, dim1=-1)), 0.5), dim=-1), V)` → `(sdpa Q, K, V, scale=2.0)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: sdpa_fold_div_nomask
- beat: no — 1 case(s) run
  - `sdpa_nomask`: cert yes, fired 1, in-deriv yes, cost 33415 → 32772 vs floor 32772 → beat no

### `sdpa_fold_nomask` — unpaid

- instance (registry): `(matmul (softmax (matmul Q, (transpose K, dim0=-2, dim1=-1)), dim=-1), V)` → `(sdpa Q, K, V, scale=1.0)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: sdpa_fold_nomask
- beat: no — 1 case(s) run
  - `sdpa_nomask`: cert yes, fired 0, in-deriv no, cost 33415 → 32772 vs floor 32772 → beat no

### `softsign_fold` — known

- instance (corpus:softsign_fold): `(div x, (add (abs x), 1))` → `(softsign x)`
- truth: equal · cert: yes
- derivable: yes via softsign_expand — fixed_point
- siblings excluded: softsign_fold
- beat: — — —

### `sub_unsq_pad_l` — known

- instance (zoo:RWKVMixBlock): `(sub (unsqueeze (arange arg0=8, dtype=float64, pin_memory=False), dim=0), (unsqueeze (arange arg0=8, dtype=float64, pin_memory=False), dim=1))` → `(sub (arange arg0=8, dtype=float64, pin_memory=False), (unsqueeze (arange arg0=8, dtype=float64, pin_memory=False), dim=1))`
- truth: equal · cert: yes
- derivable: yes — e-graph-dependent witness
- siblings excluded: sub_unsq_pad_l
- beat: — — —

### `transpose_noop` — unpaid

- instance (synth): `(transpose u, dim0=0, dim1=-1)` → `u`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: transpose_noop
- beat: no — 8 case(s) run
  - `flash_attention`: cert yes, fired 0, in-deriv no, cost 33544 → 32772 vs floor 32772 → beat no
  - `gqa_absorb_repeat`: cert yes, fired 0, in-deriv no, cost 199688 → 196610 vs floor 196612 → beat no
  - `mha_qkv_fuse`: cert yes, fired 0, in-deriv no, cost 32772 → 32770 vs floor 32770 → beat no
  - `retnet_chunkwise`: cert yes, fired 0, in-deriv no, cost 761863 → 761862 vs floor 491656 → beat no
  - `sdpa_decomposed`: cert yes, fired 0, in-deriv no, cost 32772 → 32772 vs floor 32772 → beat no
  - `sdpa_nomask`: cert yes, fired 0, in-deriv no, cost 33415 → 32772 vs floor 32772 → beat no
  - `zoo_cosine_attention`: cert yes, fired 0, in-deriv no, cost 348463 → 348463 vs floor 348463 → beat no
  - `zoo_gau`: cert yes, fired 0, in-deriv no, cost 182973 → 182967 vs floor 182973 → beat no

### `cand_mul_factor` — unmeasured

- instance (synth): `(add (mul x, y), (mul x, z))` → `(mul x, (add y, z))`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_mul_factor_right` — unmeasured

- instance (synth): `(add (mul y, x), (mul z, x))` → `(mul (add y, z), x)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_neg_factor` — unmeasured

- instance (synth): `(add (neg x), (neg y))` → `(neg (add x, y))`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_square_neg` — unmeasured

- instance (synth): `(square (neg x))` → `(square x)`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_exp_factor` — unmeasured

- instance (synth): `(mul (exp x), (exp y))` → `(exp (add x, y))`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_mul_zero` — unmeasured

- instance (synth): `(mul x, 0)` → `0`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_mul_zero_left` — unmeasured

- instance (synth): `(mul 0, x)` → `0`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_pow_one` — unmeasured

- instance (synth): `(pow x, 1)` → `x`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_sub_self` — unmeasured

- instance (synth): `(sub x, x)` → `0`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_add_inv` — unmeasured

- instance (synth): `(add x, (neg x))` → `0`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

### `cand_div_self` — unmeasured

- instance (synth): `(div x, x)` → `1`
- truth: equal · cert: yes
- derivable: no — fixed_point
- siblings excluded: none
- beat: — — no case or zoo module exercises this law

