<!-- provenance: graphopt:docs/scorecard.md@eea02f6 — generated upstream; do not hand-edit -->

# Discovery scorecard

Can the engine derive *named* optimizations from the literature and real compilers — and can it beat the registered floor?  Each row is one case in `bench/discovery/cases/`; **found** = the expect spec held (signal observed + extraction contract), **cert** = the proof-carrying certificate replays standalone, **beat** = the certified form is *strictly below* the best registered baseline on the case's axis (a tie is a rediscovery, not a find).

- lane: `full` (measured wall + verify)
- date: 2026-10-21T00:53:41+00:00 · git: `eea02f6` (dirty) · torch 2.14.0+cu130 on cpu

| case | category | found | cert | cost (axis) | before → after | best baseline | beat |
|---|---|---|---|---|---|---|---|
| `biased_chain_fold` | speed | yes | yes | launch | 4,098 → 2,114 (-48.4%) | pre-folded affine 2,049 | no |
| `blelloch_scan` | speed | yes | yes | launch | 759 → 752 (-0.9%) | eager unrolled 759 | yes |
| `conv_bn_fold` | speed | yes | yes | launch | 29,186 → 27,649 (-5.3%) | hand-folded conv 27,649 | no |
| `cumsum_scan` | memory | yes | yes | exec | 1,115,257 → 1,402,358 (+25.7%) | cumsum prefix 34,940 | no |
| `flash_attention` | memory | yes | yes | launch | 33,544 → 32,772 (-2.3%) | F.scaled_dot_product_attention 32,772 | no |
| `glu_fold` | speed | yes | yes | launch | 98 → 33 (-66.3%) | F.glu 33 | no |
| `gqa_absorb_repeat` | memory | yes | yes | launch | 199,688 → 196,610 (-1.5%) | F.sdpa enable_gqa 196,612 | yes |
| `mha_qkv_fuse` | speed | yes | yes | launch | 32,772 → 32,770 (-0.0%) | merged QKV 32,770 | no |
| `mul_unsq_nat` | speed | yes | yes | launch | 75 → 50 (-33.3%) | mul-then-unsqueeze 50 | no |
| `retnet_chunkwise` | memory | yes | yes | launch | 761,863 → 761,862 (-0.0%) | paper chunkwise 491,656 | no |
| `rms_norm_bridge` | speed | yes | yes | launch | 218 → 129 (-40.8%) | F.rms_norm 129 | no |
| `rms_norm_fold` | speed | yes | yes | launch | 282 → 129 (-54.3%) | F.rms_norm 129 | no |
| `rms_norm_nogain` | speed | yes | yes | launch | 217 → 65 (-70.0%) | F.rms_norm (no weight) 65 | no |
| `sdpa_decomposed` | structure | yes | yes | launch | 32,772 → 32,772 (+0.0%) | eager spelled 32,772 | no |
| `sdpa_nomask` | memory | yes | yes | launch | 33,415 → 32,772 (-1.9%) | F.scaled_dot_product_attention 32,772 | no |
| `select_mul` | speed | yes | yes | launch | 15 → 15 (+0.0%) | eager spelled 15 | no |
| `silu_fold` | speed | yes | yes | launch | 194 → 193 (-0.5%) | F.silu 193 | no |
| `softmax_fold` | speed | yes | yes | launch | 327 → 65 (-80.1%) | torch.softmax 65 | no |
| `softsign_fold` | speed | yes | yes | launch | 387 → 65 (-83.2%) | eager spelled 387 | yes |
| `swiglu_mlp` | speed | yes | yes | launch | 4,356 → 4,355 (-0.0%) | eager two-GEMM 4,356 | yes |
| `weight_chain_fold` | speed | yes | yes | launch | 4,098 → 2,049 (-50.0%) | pre-folded Linear 2,049 | no |
| `weight_merge` | speed | yes | yes | launch | 4,163 → 2,049 (-50.8%) | merged Linear 2,049 | no |
| `zoo_adaln_affd` | structure | yes | yes | exec | 217,830 → 200,240 (-8.1%) | eager 217,830 | yes |
| `zoo_cosine_attention` | memory | yes | yes | exec | 348,463 → 348,440 (-0.0%) | eager spelled 348,463 | yes |
| `zoo_delta_rule` | memory | yes | yes | exec | 975,073 → 818,073 (-16.1%) | eager unrolled 975,073 | yes |
| `zoo_gau` | memory | yes | yes | exec | 182,973 → 165,561 (-9.5%) | eager quadratic 182,973 | yes |
| `zoo_lora_adapter` | speed | yes | yes | exec | 87,045 → 69,650 (-20.0%) | merged LoRA 69,650 | no |
| `zoo_lora_param` | memory | yes | yes | param_bytes | 400 → 400 (+0.0%) | eager low-rank 400 | no |
| `zoo_mos_head` | structure | yes | yes | launch | 637 → 637 (+0.0%) | eager 637 | no |
| `zoo_spline_kan` | speed | yes | yes | exec | 87,058 → 87,055 (-0.0%) | eager spelled 87,058 | yes |
| `zoo_time_cond_conv` | speed | yes | yes | exec | 226,488 → 209,088 (-7.7%) | eager 226,488 | yes |

## Measured (full lane)

| case | verified | max\|Δ\| | mlopt ms | best baseline ms | beat (measured) |
|---|---|---|---|---|---|
| `biased_chain_fold` | yes | 2.220446049250313e-16 | 0.008 | 0.006 | no |
| `blelloch_scan` | yes | 8.881784197001252e-16 | 0.165 | 0.018 | no |
| `conv_bn_fold` | yes | 8.881784197001252e-16 | 0.010 | 0.010 | no |
| `cumsum_scan` | yes | 3.552713678800501e-15 | 0.261 | 0.004 | no |
| `flash_attention` | yes | 2.220446049250313e-16 | 0.029 | 0.029 | yes |
| `glu_fold` | yes | 0.0 | 0.004 | 0.003 | no |
| `gqa_absorb_repeat` | yes | 0.0 | 0.062 | 0.048 | no |
| `mha_qkv_fuse` | yes | 0.0 | 0.048 | 0.030 | no |
| `mul_unsq_nat` | yes | 0.0 | 0.005 | 0.003 | no |
| `retnet_chunkwise` | yes | 0.0 | 0.072 | 0.127 | yes |
| `rms_norm_bridge` | yes | 4.440892098500626e-16 | 0.013 | 0.010 | no |
| `rms_norm_fold` | yes | 0.0 | 0.014 | 0.009 | no |
| `rms_norm_nogain` | yes | 0.0 | 0.011 | 0.008 | no |
| `sdpa_decomposed` | yes | 0.0 | 0.041 | 0.040 | no |
| `sdpa_nomask` | yes | 2.220446049250313e-16 | 0.031 | 0.037 | yes |
| `select_mul` | yes | 0.0 | 0.008 | 0.003 | no |
| `silu_fold` | yes | 4.440892098500626e-16 | 0.003 | 0.002 | no |
| `softmax_fold` | yes | 5.551115123125783e-17 | 0.005 | 0.002 | no |
| `softsign_fold` | yes | 0.0 | 0.007 | 0.006 | no |
| `swiglu_mlp` | yes | 0.0 | 0.020 | 0.015 | no |
| `weight_chain_fold` | yes | 2.220446049250313e-16 | 0.006 | 0.006 | yes |
| `weight_merge` | yes | 7.771561172376096e-16 | 0.006 | 0.006 | yes |
| `zoo_adaln_affd` | yes | 0.0 | 0.071 | 0.054 | no |
| `zoo_cosine_attention` | yes | 1.1102230246251565e-16 | 0.136 | 0.103 | no |
| `zoo_delta_rule` | yes | 0.0 | 0.149 | 0.083 | no |
| `zoo_gau` | yes | 0.0 | 0.051 | 0.048 | no |
| `zoo_lora_adapter` | yes | 2.220446049250313e-16 | 0.016 | 0.015 | no |
| `zoo_lora_param` | yes | 0.0 | 0.021 | 0.016 | no |
| `zoo_mos_head` | yes | 0.0 | 0.044 | 0.034 | no |
| `zoo_spline_kan` | yes | 0.0 | 0.028 | 0.024 | no |
| `zoo_time_cond_conv` | yes | 0.0 | 0.052 | 0.059 | yes |

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

### `cumsum_scan` — Unit-decay recurrence → aff carrier (cumsum association)

The `a = 1` corner of the diagonal-affine monoid — `h_t = h_{t-1} +
x_t` unrolled is a prefix sum, and `affd_lift_unit` /
`affd_lift_unit_step` raise it to `applyd(aff_diag(h, x), 1)` under
`carrier_search`.  Derives-but-loses at benchmark scale, recorded
honestly.  The lift itself is sound and extraction works as designed:
`_carrier_upgrade` selects the fully-carried member because the
delivered-batched model prices it below generic — a win the
uncalibrated constants promise but the wall clock denies (full lane:
the batched delivery runs ~3× slower than the unrolled adds; at
T=32 the level machinery's per-call cost outweighs 31 serial
dispatches — a cost-model fidelity gap, and `delivered_cost_for`
with a measured `TargetProfile` is the correction channel).  The
`exec` axis reports the uniform generic-model price (+25.7 %); the
member ships `batched`, a whole-spine property `dag_cost` cannot
express.  The `cumsum` ref floor (34,940) stays unreachable — no law
produces the `cumsum` op, a coverage gap, not a bad law.


- signals: law:affd_lift_unit, law:affd_lift_unit_step, op:applyd
- baselines: eager unrolled (self) 1,115,257; cumsum prefix (ref) 34,940 — x.cumsum(1)[:, -1] — the practitioner's parallel-scan spelling; inductor (compile) — measured-only (no symbolic form)

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

### `gqa_absorb_repeat` — Materialised repeat_kv → sdpa enable_gqa

The llama2.c GQA spelling: `unsqueeze→expand→reshape` copies each kv
head r times before `F.sdpa` — a materialised copy diagonal.  SDPA's
`enable_gqa` flag implements the same broadcast inside the kernel, and
`gqa_absorb_repeat` pushes the copy into the consumer: the duplicated
heads are never materialised.  (Regression: the law's LHS pinned legacy
`arg4`/`arg5` attr metavars the canonical `ATTR_SCHEMA` boundary never
emits — zero fires until the pattern read `dropout_p`/`is_causal`.)


- signals: law:gqa_absorb_repeat
- baselines: eager spelled (self) 199,688; F.sdpa enable_gqa (ref) 196,612 — the kernel-side head broadcast PyTorch ships; inductor (compile) — measured-only (no symbolic form)

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
(bench/drivers/probes/retnet_baseline_probe.py).


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

### `sdpa_decomposed` — Fused SDPA — decompose, migrate the scale, refold

The fused-op blindness: `F.scaled_dot_product_attention` is opaque to
the e-graph — no law can reach inside the kernel (the measured
0.80–0.88× vs compiled on BiRefNet/pointwise-fusion rows,
docs/results.md).  `sdpa_expand_s` mints the kernel's own spelled
form `softmax(q@kᵀ·s)@v`; inside, the score-scale migration
(`naturality_scalar_left_rev` → `scale_in_{transpose,reshape}` →
`linear_out_scale`) pushes the runtime (T,T) scale into a
compile-time weight mul, and `sdpa_fold_mul_nomask` refolds.  Opt-in:
`default_attention` = DEFAULT + ATTENTION.


- signals: law:sdpa_expand_s, law:naturality_scalar_left_rev, law:linear_out_scale
- baselines: eager spelled (self) 32,772; inductor (compile) — measured-only (no symbolic form)

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
the fold's win needs shared-slice consumers.  With the search priced
on the case's own `launch` axis the law still fires (the derivation
is the signal) but extraction correctly DECLINES the fused member —
the old +153 % row judged an exec-model pick on an axis it was never
asked to optimize.


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

### `zoo_adaln_affd` — Zoo AdaLNBlock — the affd carrier lift, opt-in regime

The held-out zoo's `AdaLNBlock` (Peebles & Xie 2022, DiT) —
`LN(x)·(1+scale) + shift` modulation lifted by `affd_lift` /
`affd_lift_unit` into `applyd(aff_diag(ln(x), shift), 1+scale)` under
the `carrier_search` regime (`SCAN_DIAG_LAWS` is opt-in — DEFAULT
never runs it; the store's `affd_step_lift` twin is admitted but
unshipped).  The lift lands AND pays: −8.1 % on the exec axis, below
the eager floor.  The former +7.9 % was a selection artifact, not the
lift's price — `_carrier_upgrade` moved extraction to a fully-lifted
member whose single-leaf "batched" plan priced below generic in the
executor model, though a 0-level plan is pure gather/stack machinery
(the `human-bar.md` regression).  `build_scan_plan` now declines
spines with no compose levels — nothing to level-schedule — so the
genuinely cheaper lifted member stands: the same verdict the store
arm's −4.07 % hinted at, now reached through the shipped opt-in laws.


- signals: law:affd_lift, op:applyd
- baselines: eager (self) 217,830; inductor (compile) — measured-only (no symbolic form)

### `zoo_cosine_attention` — Zoo CosineAttention — om_lift on the normalized-score map

The held-out zoo's `CosineAttention` (SwinV2-style) — `softmax(τ·q̂k̂ᵀ)`
with L2-normalized projections spelled `linalg_vector_norm`.
`om_lift` (the carrier `OM_LIFT`, metavar-dim guard) raises the
normalized outer-product map onto the `om_apply`/`om_elem` carrier.
Replays the −0.006 % `human-bar.md` cell; honest on both axes —
the executor reads −0.007 %, the launch axis reads a +2.9 %
regression the scorecard records as `beat=no`.


- signals: law:om_lift, op:om_apply
- baselines: eager spelled (self) 348,463; inductor (compile) — measured-only (no symbolic form)

### `zoo_delta_rule` — Zoo DeltaRuleBlock — partial aff-carrier lift on the write

The held-out zoo's `DeltaRuleBlock` (Schlag et al. 2021, DeltaNet) —
the unrolled matrix-state `S ← S + β·k·(v − kᵀS)` write the census
minted `mul_unsq_nat` from.  The full chunked delta rule is honestly
NOT in the library (the zoo's own docstring says so), but the
diagonal-affine monoid still lifts the per-step write: `applyd` /
`aff_diag` pairs absorb all fifteen spelled `add`s while
`mul_unsq_nat` and `select_mul` fire on their source op-tuples —
a partial carrier find measured −15.7 % launch / −16.1 % exec where
the retro table recorded `—` under its pipeline axis.


- signals: law:decay_state_lift_post, law:mul_unsq_nat, law:select_mul, op:applyd
- baselines: eager unrolled (self) 975,073; inductor (compile) — measured-only (no symbolic form); FLA chunk_delta_rule (external) unavailable — Triton WY-repr kernel — scratch dep; the impl the full rule targets

### `zoo_gau` — Zoo GatedAttentionUnit — quadratic gate lifts to om_decayed

The held-out zoo's `GatedAttentionUnit` (Hua et al. 2022, FLASH) —
softmax-free attention whose `relu(u uᵀ)²` self-similarity score is a
quadratic (T,T) site.  `om_decayed_lift` raises the masked-score
matmul onto the decayed-om carrier while the `mul_square` /
`square_to_pow` spelling bridge canonicalises `a*a`; the extraction
keeps one `om_decayed` node (the (T,T) score matrix never
materialises).  Replays the −0.003 % `human-bar.md` cell — −9.5 %
under the executor axis here.


- signals: law:om_decayed_lift, law:mul_square, op:om_decayed
- baselines: eager quadratic (self) 182,973; inductor (compile) — measured-only (no symbolic form); FLASH GAU kernel (external) unavailable — fla.layers.gau — scratch dep; the hand kernel this emulates

### `zoo_lora_adapter` — Zoo LoRAAdapter — the assoc_linear win, replayed

The held-out zoo's own `LoRAAdapter` (Hu et al. 2021): the real
`W x + alpha·(B (A x))` production spelling, imported verbatim from
`mlopt_discovery.zoo` — not a purpose-built respelling.  `assoc_linear`
folds the low-rank chain `up(down(x))` into one dense GEMM whose
`B·A` weight product folds at compile time.  Replays the −19.985 %
verified win of `project/retros/human-bar.md` — under the executor
axis it measures −20.0 % here — the largest single-model yield the
zoo retro recorded.


- signals: law:assoc_linear
- baselines: eager three-GEMM (self) 87,045; merged LoRA (ref) 69,650 — peft.merge_and_unload-style — the branch pre-composed dense; inductor (compile) — no freeze — Inductor cannot fold the adapter weights

### `zoo_lora_param` — Zoo LoRAAdapter on the storage axis — the fold's other side

The same zoo `LoRAAdapter` site priced on `param_bytes` — the memory
twin of `zoo_lora_adapter`.  Folding `up(down(x))` to one (d,d) GEMM
buys the runtime win at a storage price the speed axis hides:
`2·d·r` stored adapter values become `d²` — profitable only when
`r > d/2`.  With the search priced on the case's own axis the
derivation still fires (`law:assoc_linear`) but extraction correctly
DECLINES the merge: the factorized `2·d·r` member is cheaper to
store.  The honest verdict flips from a recorded +32 % regression —
the old row judged an exec-optimal pick on an axis it was never
asked to optimize — to a rediscovery: the fold derives, and the
engine declines it exactly where it does not pay.


- signals: law:assoc_linear
- baselines: eager low-rank (self) 400; merged LoRA (ref) 528 — same storage regression — the hand-merge pays it too; inductor (compile) — measured-only (no symbolic form)

### `zoo_mos_head` — Zoo MoSHead — gather dedup reaches, cost-neutral

The held-out zoo's `MoSHead` (Yang et al. 2018, mixture-of-softmaxes)
— a pure reachability row: `index_select_dedup` (×5) and
`index_select_id` (×4) fire on the einsum-expanded gather/index
structure and the param-sharing lift offers the (3,d,n) component
table a `heads3` share, yet the extraction stays byte-identical
(cost-neutral, as `human-bar.md` recorded: fires, no win).  The row
guards *derivability* — the day the dedup family turns paying on a
real head, this case's cost columns move first.


- signals: law:index_select_dedup, law:index_select_id, stat:param_sharing
- baselines: eager (self) 637; inductor (compile) — measured-only (no symbolic form)

### `zoo_spline_kan` — Zoo SplineKAN — the x·x spelling bridge, canonically

The held-out zoo's `SplineKAN` edge layer (Liu et al. 2024, KAN) —
`stack([x, x*x, sin x])` spells the quadratic basis element as a
self-product, exactly the `x·x` normalizer-code spelling the
canonicalizer bridge exists for.  `mul_square` → `square_to_pow`
rewrite it to `square`; the extraction keeps the canonical op
(`mul` gone).  Replays the −0.003 % `human-bar.md` cell — a small
but real verified win on code nobody targeted.


- signals: law:mul_square, law:square_to_pow
- baselines: eager spelled (self) 87,058; inductor (compile) — measured-only (no symbolic form)

### `zoo_time_cond_conv` — Zoo TimeCondConv — dead broadcast pads stripped

The held-out zoo's `TimeCondConv` — the DDPM diffusion residual block
(Peebles-style sinusoidal time embedding broadcast into a conv block,
`emb[:, :, None, None]`).  The machine-admitted pad strips
`mul_unsq_pad_l` / `mul_unsq_pad_r` (provenance MACHINE_SHIPPED)
remove the dead broadcast pads; `silu_expand` fires alongside.
Replays the −4.996 % `human-bar.md` yield — the only machine-admitted
win in that table — measured −7.7 % on the executor axis here.


- signals: law:mul_unsq_pad_l, law:mul_unsq_pad_r
- baselines: eager (self) 226,488; inductor (compile) — measured-only (no symbolic form)

