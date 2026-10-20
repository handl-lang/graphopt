# Discovery scorecard

Can the engine derive *named* optimizations from the literature and real compilers — and can it beat the registered floor?  Each row is one case in `bench/discovery/cases/`; **found** = the expect spec held (signal observed + extraction contract), **cert** = the proof-carrying certificate replays standalone, **beat** = the certified form is *strictly below* the best registered baseline on the case's axis (a tie is a rediscovery, not a find).

- lane: `full` (measured wall + verify)
- date: 2026-10-20T01:24:16+00:00 · git: `646beab` (dirty) · torch 2.14.0+cu130 on cpu

| case | category | found | cert | cost (axis) | before → after | best baseline | beat |
|---|---|---|---|---|---|---|---|
| `biased_chain_fold` | speed | yes | yes | launch | 4,098 → 2,114 (-48.4%) | pre-folded affine 2,049 | no |
| `blelloch_scan` | speed | yes | yes | launch | 759 → 752 (-0.9%) | eager unrolled 759 | yes |
| `conv_bn_fold` | speed | yes | yes | launch | 29,186 → 27,649 (-5.3%) | hand-folded conv 27,649 | no |
| `flash_attention` | memory | yes | yes | launch | 33,544 → 32,772 (-2.3%) | F.scaled_dot_product_attention 32,772 | no |
| `glu_fold` | speed | yes | yes | launch | 98 → 33 (-66.3%) | F.glu 33 | no |
| `mha_qkv_fuse` | speed | yes | yes | launch | 32,772 → 32,770 (-0.0%) | merged QKV 32,770 | no |
| `mul_unsq_nat` | speed | yes | yes | launch | 75 → 50 (-33.3%) | mul-then-unsqueeze 50 | no |
| `retnet_chunkwise` | memory | yes | yes | launch | 761,863 → 761,862 (-0.0%) | paper chunkwise 491,656 | no |
| `rms_norm_bridge` | speed | yes | yes | launch | 218 → 129 (-40.8%) | F.rms_norm 129 | no |
| `rms_norm_fold` | speed | yes | yes | launch | 282 → 129 (-54.3%) | F.rms_norm 129 | no |
| `rms_norm_nogain` | speed | yes | yes | launch | 217 → 65 (-70.0%) | F.rms_norm (no weight) 65 | no |
| `sdpa_nomask` | memory | yes | yes | launch | 33,415 → 32,772 (-1.9%) | F.scaled_dot_product_attention 32,772 | no |
| `select_mul` | speed | yes | yes | launch | 15 → 38 (+153.3%) | eager spelled 15 | no |
| `silu_fold` | speed | yes | yes | launch | 194 → 193 (-0.5%) | F.silu 193 | no |
| `softmax_fold` | speed | yes | yes | launch | 327 → 65 (-80.1%) | torch.softmax 65 | no |
| `softsign_fold` | speed | yes | yes | launch | 387 → 65 (-83.2%) | eager spelled 387 | yes |
| `swiglu_mlp` | speed | yes | yes | launch | 4,356 → 4,355 (-0.0%) | eager two-GEMM 4,356 | yes |
| `weight_chain_fold` | speed | yes | yes | launch | 4,098 → 2,049 (-50.0%) | pre-folded Linear 2,049 | no |
| `weight_merge` | speed | yes | yes | launch | 4,163 → 2,049 (-50.8%) | merged Linear 2,049 | no |

## Measured (full lane)

| case | verified | max\|Δ\| | mlopt ms | best baseline ms | beat (measured) |
|---|---|---|---|---|---|
| `biased_chain_fold` | yes | 2.220446049250313e-16 | 0.009 | 0.007 | no |
| `blelloch_scan` | yes | 8.881784197001252e-16 | 0.226 | 0.024 | no |
| `conv_bn_fold` | yes | 1.6053824936079764e-13 | 0.012 | 0.013 | yes |
| `flash_attention` | yes | 2.220446049250313e-16 | 0.033 | 0.036 | yes |
| `glu_fold` | yes | 0.0 | 0.006 | 0.004 | no |
| `mha_qkv_fuse` | yes | 0.0 | 0.063 | 0.039 | no |
| `mul_unsq_nat` | yes | 0.0 | 0.005 | 0.003 | no |
| `retnet_chunkwise` | yes | 0.0 | 0.081 | 0.140 | yes |
| `rms_norm_bridge` | yes | 6.661338147750939e-15 | 0.015 | 0.012 | no |
| `rms_norm_fold` | yes | 6.927791673660977e-13 | 0.015 | 0.012 | no |
| `rms_norm_nogain` | yes | 3.432809592140984e-13 | 0.013 | 0.010 | no |
| `sdpa_nomask` | yes | 2.220446049250313e-16 | 0.038 | 0.038 | no |
| `select_mul` | yes | 0.0 | 0.006 | 0.004 | no |
| `silu_fold` | yes | 4.440892098500626e-16 | 0.003 | 0.002 | no |
| `softmax_fold` | yes | 5.551115123125783e-17 | 0.004 | 0.002 | no |
| `softsign_fold` | yes | 0.0 | 0.009 | 0.007 | no |
| `swiglu_mlp` | yes | 0.0 | 0.076 | 0.062 | no |
| `weight_chain_fold` | yes | 2.220446049250313e-16 | 0.022 | 0.021 | no |
| `weight_merge` | yes | 7.771561172376096e-16 | 0.021 | 0.021 | yes |

## Per-case detail

### `biased_chain_fold` — Two-layer biased Linear chain → affine compose

`linear(linear(x,A,b1),B,b2) → (BA)x + (B·b1+b2)` — the affine-map
composition `assoc_linear_bias`, the biased sibling of
`weight_chain_fold`'s `assoc_linear`.  The composed member spells the
fused bias `linear(x, BA, B·b1) + b2` so both B-products stay
well-typed for every hidden width and fold at compile time — one
GEMM plus one broadcast add at runtime.


- signals: law:assoc_linear_bias
- baselines: eager two-GEMM (self) 4,098; pre-folded affine (ref) 2,049 — one Linear carrying W=B·A, b=B·b1+b2; inductor (compile) — no freeze — Inductor cannot fold the weights

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

### `glu_fold` — chunk·σ(chunk) manual GLU → fused glu kernel

Dauphin et al. 2017, "Language Modeling with Gated Convolutional
Networks" — `a ⊗ σ(b)` over the two halves of the last dim IS
`F.glu`.  The corpus carried the kernel image but no manual spelling
until `ManualGluMLP` closed the pair (`glu_fold`,
corpus-expansion-r2.md §6); the parity-of-split-axis guard keeps the
odd-extent corner from minting an `glu` that cannot denote.


- signals: law:glu_fold
- baselines: eager spelled (self) 98; F.glu (ref) 33 — the fused kernel; inductor (compile) — measured-only (no symbolic form)

### `mha_qkv_fuse` — Split QKV projections → one widened GEMM

The fused-QKV transform Megatron-LM's `MergedColumnParallelLinear`
performs by hand and TFLite's `fuse_qkv` pass does structurally:
`<q,k,v> : X -> V³` is ONE map into the product space — three
bias-free projections of one input concat into a single (3d,d) GEMM
plus three free split views.  `qkv_fuse`/`qkv_fuse_asym` are
pairing-subsumed in the bank (tags SUBSUMED, kept for the cache);
the shipped reach is the non-local pairing pass — `paired_extract`
is the honest signal that the forced shared-GEMM term won.


- signals: stat:paired_extract
- baselines: eager three-GEMM (self) 32,772; merged QKV (ref) 32,770 — one (3d,d) Linear + chunk — the hand-fused impl; inductor (compile) — measured-only (no symbolic form)

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
(bench/probes/retnet_baseline_probe.py).


- signals: law:om_decayed_lift, op:om_decayed
- baselines: eager quadratic (self) 761,863; paper chunkwise (ref) 491,656 — Sun et al. 2023 chunkwise recurrence, reimplemented; inductor (compile) — measured-only (no symbolic form); FLA chunk_retention (external) unavailable — Triton kernel — scratch dep; measured ~7-11x ahead of the carrier

### `rms_norm_bridge` — Misspelled RMSNorm → fold via canonicalizer bridges

The recorded-miss spellings of `rms-norm-law.md`, folded: the corpus
spells x² as `mul(x, x)` and the reciprocal root as `pow(·, -0.5)`
— `rms_norm_fold` pins `pow(u, 2)` and `rsqrt(·)` literally, so the
fold only reaches this graph through the canonicalizer bridges
`mul_square` → `square_to_pow` and `pow_to_rsqrt`
(tests/test_square_rsqrt_laws.py).  A real corpus fragment reads
exactly this way — normalizer code is written `x*x` + `**-0.5` far
more often than `pow(2)` + `rsqrt`.


- signals: law:rms_norm_fold, law:mul_square, law:pow_to_rsqrt
- baselines: eager spelled (self) 218; F.rms_norm (ref) 129 — the fused kernel PyTorch ships; inductor (compile) — measured-only (no symbolic form)

### `rms_norm_fold` — Manual RMSNorm → fused rms_norm kernel

Zhang & Sennrich 2019 RMSNorm, popularised by LLaMA (Touvron 2023):
`x·rsqrt(mean(x²)+eps)·w` IS `F.rms_norm` spelled by hand.  In mlopt it
is `rms_norm_fold`, the fourth machine-discovered law admitted to the
bank — measured −83% on RMSNorm / −31% on TransformerBlock extracted
cost (project/retros/rms-norm-law.md).  XLA/TFLite and Transformer
Engine ship the fused kernel as a named primitive; Inductor does not
pattern the whole reduce-scale chain back to it.


- signals: law:rms_norm_fold
- baselines: eager spelled (self) 282; F.rms_norm (ref) 129 — the fused kernel PyTorch ships; inductor (compile) — measured-only (no symbolic form)

### `rms_norm_nogain` — Gain-free manual RMSNorm → fused rms_norm

The weight-free twin of the RMSNorm fold — `x·rsqrt(mean(x²)+eps)`
with no channel gain is `F.rms_norm(x, ns, weight=None)`.  It is the
gained fold's inner `mul`, so `rms_norm_fold_nogain` fires inside
every gained site too (the cost model picks the fused member either
way; project/retros/rms-norm-law.md).


- signals: law:rms_norm_fold_nogain
- baselines: eager spelled (self) 217; F.rms_norm (no weight) (ref) 65 — the fused kernel, weight=None; inductor (compile) — measured-only (no symbolic form)

### `sdpa_nomask` — No-mask spelled attention → fused SDPA

`softmax(q@kᵀ/√d) @ v` with no mask — `nn.MultiheadAttention`'s
default path, and the spelling encoder stacks (BERT-style
bidirectional attention) export verbatim.  The add/masked_fill fold
family cannot see it; `sdpa_fold_div_nomask` is its row — minted by
`fold_object` + guarded-compose and promoted after the sweep
(project/retros/promoted-laws.md).  The fold removes the materialised
(T,T) score matrix, same transform as the masked row.


- signals: law:sdpa_fold_div_nomask
- baselines: eager spelled (self) 33,415; F.scaled_dot_product_attention (ref) 32,772 — the fused kernel, explicit scale=1/√d; inductor (compile) — measured-only (no symbolic form)

### `select_mul` — select(a)⊙select(b) → select(a⊙b) commutation

`sel(u) ⊙ sel(v) = sel(u ⊙ v)` — the elementwise product commutes
with the select view (`select_mul`, census-minted on
`DeltaRuleBlock`-class slices; project/retros/select-mul-broadcast.md;
the first machine-discovered law admitted, human-integrated).
Honest row: standalone, the fused form pays for the product over the
full tensor then slices — costlier than multiplying two slices — so
the fold's win needs shared-slice consumers; the scorecard records
the standalone regression, not a hoped-for one.


- signals: law:select_mul
- baselines: eager spelled (self) 15; mul-then-select (ref) 38; inductor (compile) — measured-only (no symbolic form)

### `silu_fold` — x·σ(x) spelled by hand → fused silu kernel

SiLU (Elfwing et al. 2018; Ramachandran et al. 2017 "swish") spelled
`x·σ(x)` folds to the one kernel — `silu_fold`, the definitional
inverse of `silu_expand` and the mediating 3-cell of the
`silu_expand × swiglu_fuse` critical pair
(project/retros/law-coherence-catalogue.md §4).  Inductor's
pointwise-pattern tables carry the same identity
(`mul(x, sigmoid(x)) → silu`).


- signals: law:silu_fold
- baselines: eager spelled (self) 194; F.silu (ref) 193 — the fused kernel; inductor (compile) — measured-only (no symbolic form)

### `softmax_fold` — exp/Σexp manual softmax → fused softmax kernel

`exp(u) / Σ exp(u)` is softmax by definition — the composed-then-
reduced chain the law pipeline's pattern-recognition pass recognised
in the model census (the second machine-discovered law admitted;
project/retros/law-proposer-extensions.md,
project/retros/law-softmax-fold-shipped.md).  Every graph compiler
performs this fold (XLA's softmax rewriter, TFLite's
`IdentifySoftmax`); here it lands as one dispatched op with a
certificate.


- signals: law:softmax_fold
- baselines: eager spelled (self) 327; torch.softmax (ref) 65 — the fused kernel; inductor (compile) — measured-only (no symbolic form)

### `softsign_fold` — x/(|x|+1) spelled by hand → fused softsign kernel

`x / (|x| + 1)` is softsign (Bergstra et al. 2009, the bounded
rational activation) spelled in decomposed div/abs/add form —
`softsign_fold` mints the one fused kernel, discovered by
`fold_object` and promoted after the guarded sweep measured the
region unguarded-clean (project/retros/promoted-laws.md).


- signals: law:softsign_fold
- baselines: eager spelled (self) 387; F.softsign (ref) 387 — the fused kernel; inductor (compile) — measured-only (no symbolic form)

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

### `weight_merge` — linear(x,A)+linear(x,B) → linear(x, A+B)

The LoRA-apply / model-soup merge deployment tooling performs by
hand: two parallel projections of the same input sum into one GEMM
on the summed weight — `weight_factor_linear` (transpose
distributes over +).  The parameter add is compile-time work, so
runtime drops a GEMM: the bias-free analogue of the biased chain's
`assoc_linear_bias`.


- signals: law:weight_factor_linear
- baselines: eager two-GEMM (self) 4,163; merged Linear (ref) 2,049 — one Linear carrying A+B — the merged form; inductor (compile) — measured-only (no symbolic form)

