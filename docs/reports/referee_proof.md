# Kernel novelty sieve

The codegen port of the novelty sieve — the automatic three-gate filter between a shipped ``KERNEL_RULES`` law and the human-review queue.  **cert** = the law's own rewrite carries a standalone-replayable certificate on its instance (`verify_law` + `verify_certificate`); **derivable** = the known bank — `KERNEL_RULES` U the kernel prior-art bank (`KERNEL_PRIOR_ART_RULES`, the codegen vocabulary) U the shared substrate surface (`default_rules` + `ALL_RULES_WITH_LAYOUT` + `ATTENTION` + scan monoids + the prior-art FUSION/CATEGORICAL slice) — minus the candidate and its alpha-siblings, already proves the instance; **beat** = on some registered `bench.discovery.kernel` case the certified extraction under the full kernel bank lands strictly below the best extraction the rest of the bank reaches on the same term (the floor — itself never worse than spelled) AND the derivation certificate invokes the candidate.  A law is a novelty iff cert ∧ beat ∧ ¬derivable.  `truth` is the fp64 `KernelSink.verify` oracle on the instance — an `unequal` refutes outright.

- date: 2026-10-27 00:02 UTC · git: `d58b66c` · torch 2.14.0+cu130 on cuda · bank 374 rules
- verdicts: known=3

| law | family | prov | instance | truth | cert | derivable | beat | verdict |
|---|---|---|---|---|---|---|---|---|
| `kern_fuse_silu` | fusion | human | corpus:kernel_rms_swiglu | equal | yes | yes (pa_kf_silu_expand) | — | **known** |
| `kern_fuse_softmax` | fusion | human | corpus:kernel_attn_residual | equal | yes | yes (pa_kf_softmax_expand) | — | **known** |
| `kern_fuse_layernorm` | fusion | human | corpus:kernel_ln_gelu_res | equal | yes | yes (pa_kf_layernorm_expand) | — | **known** |

## Review queue

**Empty.** Every candidate either fails a gate or is a theorem of the bank — nothing currently needs a reviewer's eyes.

## Per-law detail

### `kern_fuse_silu` — known

- instance (corpus:kernel_rms_swiglu): `(mul (mul x, (rsqrt (add (mean (pow x, 2), dim=-1, keepdim=True), 1e-06))), (sigmoid (mul x, (rsqrt (add (mean (pow x, 2), dim=-1, keepdim=True), 1e-06)))))` → `(kf_silu (mul x, (rsqrt (add (mean (pow x, 2), dim=-1, keepdim=True), 1e-06))), (mul x, (rsqrt (add (mean (pow x, 2), dim=-1, keepdim=True), 1e-06))))`
- truth: equal · cert: yes
- derivable: yes via pa_kf_silu_expand — fixed_point
- siblings excluded: kern_fuse_silu
- beat: — — —

### `kern_fuse_softmax` — known

- instance (corpus:kernel_attn_residual): `(div (exp (mul (matmul x, w), 0.25)), (sum (exp (mul (matmul x, w), 0.25)), dim=-1, keepdim=True))` → `(kf_softmax (mul (matmul x, w), 0.25), dim=-1)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_softmax_expand — fixed_point
- siblings excluded: kern_fuse_softmax
- beat: — — —

### `kern_fuse_layernorm` — known

- instance (corpus:kernel_ln_gelu_res): `(mul (sub x, (mean x, dim=-1, keepdim=True)), (rsqrt (add (mean (pow (sub x, (mean x, dim=-1, keepdim=True)), 2), dim=-1, keepdim=True), 1e-05)))` → `(kf_layernorm x, dim=-1, eps=1e-05)`
- truth: equal · cert: yes
- derivable: yes via pa_kf_layernorm_expand — fixed_point
- siblings excluded: kern_fuse_layernorm
- beat: — — —

