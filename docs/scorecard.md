# Discovery scorecard

Can the engine derive *named* optimizations from the literature and real compilers — and can it beat the registered floor?  Each row is one case in `bench/discovery/cases/`; **found** = the expect spec held (signal observed + extraction contract), **cert** = the proof-carrying certificate replays standalone, **beat** = the certified form is *strictly below* the best registered baseline on the case's axis (a tie is a rediscovery, not a find).

- lane: `full` (measured wall + verify)
- date: 2026-10-19T23:41:07+00:00 · git: `712c292` (dirty) · torch 2.14.0+cu130 on cpu

| case | category | found | cert | cost (axis) | before → after | best baseline | beat |
|---|---|---|---|---|---|---|---|
| `blelloch_scan` | speed | yes | yes | launch | 759 → 752 (-0.9%) | eager unrolled 759 | yes |
| `conv_bn_fold` | speed | yes | yes | launch | 29,186 → 27,649 (-5.3%) | hand-folded conv 27,649 | no |
| `flash_attention` | memory | yes | yes | launch | 33,544 → 32,772 (-2.3%) | F.scaled_dot_product_attention 32,772 | no |
| `mul_unsq_nat` | speed | yes | yes | launch | 75 → 50 (-33.3%) | mul-then-unsqueeze 50 | no |
| `retnet_chunkwise` | memory | yes | yes | launch | 761,863 → 761,862 (-0.0%) | paper chunkwise 491,656 | no |
| `swiglu_mlp` | speed | yes | yes | launch | 4,356 → 4,355 (-0.0%) | eager two-GEMM 4,356 | yes |
| `weight_chain_fold` | speed | yes | yes | launch | 4,098 → 2,049 (-50.0%) | pre-folded Linear 2,049 | no |

## Measured (full lane)

| case | verified | max\|Δ\| | mlopt ms | best baseline ms | beat (measured) |
|---|---|---|---|---|---|
| `blelloch_scan` | yes | 8.881784197001252e-16 | 0.207 | 0.022 | no |
| `conv_bn_fold` | yes | 1.6053824936079764e-13 | 0.012 | 0.013 | yes |
| `flash_attention` | yes | 2.220446049250313e-16 | 0.034 | 0.036 | yes |
| `mul_unsq_nat` | yes | 0.0 | 0.005 | 0.003 | no |
| `retnet_chunkwise` | yes | 0.0 | 0.079 | 0.137 | yes |
| `swiglu_mlp` | yes | 0.0 | 0.022 | 0.015 | no |
| `weight_chain_fold` | yes | 2.220446049250313e-16 | 0.005 | 0.006 | yes |

## Per-case detail

### `blelloch_scan` — Unrolled first-order recurrence → parallel scan

Blelloch 1990, "Prefix sums and their applications" — the sequential
recurrence `h_t = a·h_{t-1} + x_t` re-expressed as a parallel prefix
scan.  The spelled fixture unrolls the recurrence over T steps; the
engine's diagonal-affine monoid lift (`decay_state_lift`,
SCAN_DIAG_LAWS) turns it into the `aff_diag`/`applyd` carrier — the
same monoid RetNet's chunked scan runs on.


- signals: law:decay_state_lift, op:applyd
- baselines: eager unrolled (self) 759; cumsum prefix (ref) 1,494 — h_t = P_t · cumsum(x_t/P_t) — the closed-form parallel scan; inductor (compile) — measured-only (no symbolic form)

### `conv_bn_fold` — Conv2d + eval BatchNorm2d fold

The standard inference fold — PyTorch's `torch.nn.utils.fuse_conv_bn_eval`.
In mlopt it is `conv_bn_fold`, the first *verbatim* machine-minted
composite law (depth-4 `syn_*` chain shipped through
`mlopt_core.laws.minted`; project/retros/promoted-laws.md).  The spelled
fixture is the eval-mode `BatchNorm2d(Conv2d(x))` site the mint saw
(tests/test_minted_laws.py).


- signals: law:conv_bn_fold
- baselines: eager (self) 29,186; hand-folded conv (ref) 27,649 — PyTorch fold recipe — one conv, BN absorbed into W'/b'; inductor (compile) — measured-only (no symbolic form)

### `flash_attention` — Hand-spelled softmax attention → fused SDPA

Dao et al. 2022, FlashAttention — attention without materialising the
(T,T) score matrix.  The spelled fixture computes
`softmax(q@kᵀ/√d + causal-mask) @ v` by hand; the `sdpa_fold_*` family
folds it to the fused `sdpa` op and the causal-mask specialization
closes it to `is_causal` (DEFAULT rules).


- signals: law:sdpa_fold_adddiv
- baselines: eager spelled (self) 33,544; F.scaled_dot_product_attention (ref) 32,772 — the fused kernel PyTorch ships; inductor (compile) — measured-only (no symbolic form)

### `mul_unsq_nat` — Two-sided unsqueeze naturality

`mul_unsq_nat` — the first law minted from a corpus op-tuple
(DeltaRuleBlock's `β·k`): `unsq(u,d)·unsq(v,d) → unsq(u·v,d)`, guard
resolved by the adversarial sweep (project/retros/mul-unsq-nat.md;
provenance MACHINE_SHIPPED).  Measured −12.27% on its source workload.


- signals: law:mul_unsq_nat
- baselines: eager spelled (self) 75; mul-then-unsqueeze (ref) 50; inductor (compile) — measured-only (no symbolic form)

### `retnet_chunkwise` — RetNet chunkwise retention carrier

Sun et al. 2023, "Retentive Network" — the chunkwise-recurrent form of
retention trades the (T,T) decay-masked score matrix for a decayed
cross-chunk state.  The spelled fixture is the intake zoo's
`_ChunkedRetention` (`(QKᵀ ⊙ D)V`, decay built in-graph); the engine
lifts it to the `om_decayed` carrier — measured parity with the
paper's own chunked impl, faster under graph replay
(lab/probes/retnet_baseline_probe.py).


- signals: law:om_decayed_lift, op:om_decayed
- baselines: eager quadratic (self) 761,863; paper chunkwise (ref) 491,656 — Sun et al. 2023 chunkwise recurrence, reimplemented; inductor (compile) — measured-only (no symbolic form); FLA chunk_retention (external) unavailable — Triton kernel — scratch dep; measured ~7-11x ahead of the carrier

### `swiglu_mlp` — Two-projection SwiGLU → widened-GEMM fuse

LLaMA's SwiGLU MLP (Shazeer 2020, GLU variants): `silu(x@A.T) ⊙ (x@B.T)`
fuses to one (2d,d) projection plus a split — the pairing pass
(`paired_extract`) or `swiglu_fuse`.  Measured 2.00-2.11x vs
torch.compile on the delivered artifact (docs/results.md).


- signals: stat:paired_extract
- baselines: eager two-GEMM (self) 4,356; fused SwiGLU (ref) 4,485 — one widened Linear + split — the hand-fused impl; inductor (compile) — measured-only (no symbolic form)

### `weight_chain_fold` — Two-layer Linear chain → weights-first fold

`linear(linear(x,A),B) → linear(x, B·A)` — `assoc_linear` reaches the
weights-first form whose param-side matmul folds at compile time.  The
reassoc_scale result (docs/results.md): the e-graph finds a form
Inductor's post-grad graph cannot express (no weight folding without
`freezing`).


- signals: law:assoc_linear
- baselines: eager two-GEMM (self) 4,098; pre-folded Linear (ref) 2,049 — one Linear carrying W2@W1 — what weight folding ships; inductor (compile) — no freeze — Inductor cannot fold the weights

