# Fusion-schema mint — widening the zoo-real candidate families

The `zoo-real` corpus mint (`bench/drivers/overnight.py --corpus
zoo-real`) derives each episode's pool from `pipeline.propose`.  The
v2 funnel (`results_overnight_real_v2.md`, 53 episodes / 87 minted /
0 admitted) showed the pool dominated by `mixed:`/view naturality and
`census:` tuples while the exports carry richer recurring op-classes
the view grammar cannot see.  This report: census the gap, add a
**fusion-schema** proposer arm, and measure the bounded run.

## 1. Census — real fold-class sites vs the minted pool

Per-module `op_tuple_census` + `shape_census` over all 53 zoo exports,
each recurring subterm checked against `pipeline.propose(...,
require_bind=True)`'s minted LHS patterns.  Recurring real op-classes
with **no** minted candidate:

| class | modules | example site |
|---|---|---|
| sdpa spellings | 3 | `mm(softmax(mul/div(mm(Q,K^T),s)),V)` — CosineAttention, DilatedBlockAttention, KVCacheAttention |
| matmul-bias epilogue | 2 | `add(mm(x, w^T), b)` — PerChannelQuantMatmul, GroupedQuantLinear |
| manual norm chains | 2 | `(x-μ)·rsqrt(E[(x-μ)²]+ε)·w+b` — InstanceNormBroadcast, ManualGroupNorm |
| glu / rms fold | 2 | GatedConvUnit, RMSNormPerHead (bound but unminted) |
| gated sigmoid-mul | 5 | `mul(sigmoid(a), b)` — DeepSeekSigMoE, GRUFusedGate, GatedAttentionUnit, HighwayGate, RWKVMixBlock |
| geglu | 1 | `mul(gelu(chunk),chunk)` — GEGLUBlock (no `geglu` op image — no fold schema minted) |

Note: CosineAttention's lhs *matches* the shipped sdpa fold but its
side condition declines every binding (the div-normalized Q/K
spelling) — a real shape that stays honestly unminted under the
viable-instance gate.

## 2. The `fusion-schema` arm (`mlopt_discovery.pipeline`)

A schema mints **only when a real subterm satisfies the whole rule**
(lhs match + `check`/`derive` viability via `_instance_from_match`,
now evaluated on the materialized `as_rule()` so declarative
`cond`/`dspec` gate identically); the recorded `instance` is the
module-bound pair — seed term supplies the concrete attrs/terms the
oracles measure.

- **Shipped fold/fuse laws as recognizers** — every `*_fold*` /
  `*_fuse` rule plus `qkv_fuse_asym` / `gqa_absorb_repeat`
  (`_fusion_laws`): silu/glu/softmax/softsign/rms-norm/conv-bn folds,
  the 15 sdpa folds, swiglu/parallel-mul/qkv fuses.  `cond`/`dspec`
  transport as data (ledger `missing_hooks` stays honest; procedural
  remainders like `conv_bn_fold`'s flag explicitly).
- **`_FUSION_SCHEMAS`** — kernel-fold classes the bank lacks, verified
  true on the real sites before entering the table:
  `fold:addmm_linear` (`add(mm(A,transpose(B)),C) → linear`,
  `axes-last2` guard), `fold:instance_norm`, `fold:group_norm`
  (`num_groups` derives from the reshape target's second dim),
  `fold:silu_comm` (the commuted spelling `silu_fold` misses),
  `fold:rms_div` (the div/sqrt spelling).

`Proposal` gained `cond`/`dspec` fields (`as_rule` forwards them to
`Rewrite`); `propose()` pools the arm between `pattern_recognition`
and `shape_aware` under the same `require_bind` contract.  The
overnight `zoo-real` protocol pins `"fusion"` so pre-arm ledgers
restart loudly.

## 3. Bounded runs

Baseline (v2, full zoo, registry order): 87 minted / **0 admitted**,
families `census-mixed-view`/`layout`/`algebraic-grammar`/
`census-naturality`/`elementwise-algebra`/`slice-naturality`/
`synth-composite`.

`--corpus zoo-real --episodes 8` (first-8 registry slice): 18 minted,
0 admitted — the first eight modules carry no fold-class site, so the
arm minted nothing there (honest negative; fold sites start at index
30).

Targeted bounded pass — the **10 fold-site modules**, same protocol
(`docs/reports/results_overnight_real_fusion.md|.json`):

| metric | view-schema baseline (same modules) | + fusion-schema |
|---|---|---|
| minted | 30 | **40** (+10, all `fusion-schema`) |
| bound real sites | mixed/view only | +8 distinct `fold:*` rules |
| admitted | 0 | **3** (`fold:sdpa_fold_mul_nomask`, `fold:glu_fold`, `fold:rms_norm_fold_nogain` — all `known-art`) |

New-family detail: the three admissions are the shipped folds the
mint now reaches — the sieve correctly classes them `known-art`
(alpha/derivable).  The three *new-class* mints refused honestly:
`fold:addmm_linear` declined at the oracle leg (2eq/0ne real but only
2 accepted envs — below the leg's evidence bar); `fold:instance_norm`
and `fold:group_norm` passed oracle and *paid* (0.077–0.111× count)
but the pay leg's `verify_fail` refused the fused lowering;
`fold:sdpa_fold_{div_nomask,addmul}` likewise fired and paid
(0.21–0.44×) but verify failed on the covered cases.

`1eq` survivors: `fold:sdpa_fold_div_nomask`, `fold:glu_fold`,
`fold:rms_norm_fold_nogain`, `fold:instance_norm`, `fold:group_norm`
each measured exactly one equal real site (the single module that
spells the class); `fold:addmm_linear` measured 2eq.

## 4. Gates

ruff + format clean on touched files; `ty` clean on `pipeline.py`;
`tests/test_discovery_pipeline.py` 92/92 (14 new fusion tests),
`tests/test_overnight.py` 24/24.
