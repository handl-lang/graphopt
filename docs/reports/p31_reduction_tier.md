# p31 — the reduction tier (T-010 lever)

Date: 2026-11-07.  Scope: `mlopt_discovery.genkernel` codegen tiers +
`mlopt_discovery.play` site collectors / claim gates, measured over the
68-workload held-out zoo (`mlopt_discovery.zoo`).

## Question

The T-010 remainder audit (`project/retros/gen-kernels.md`) closed with:
*"Reduction bodies — no tier collects them … the honest next step is a
reduction TIER, not a pricing fix."*  This arm asked the two questions in
order: does the corpus even have paying reduction sites, and if so, does
a minimal tier mint claims that survive the measured-cost referee?

## Census — sites exist (not occurrence-blocked)

Every zoo workload was exported through `TorchSource().to_ir` and walked
for reduction-headed ops.  Reduction vocabulary =
`mlopt_core.opmeta.REDUCE_DIM_OPS` (19 ops: `sum`/`mean`/`max`/`min`/
`prod`/`argmax`/`argmin`/`amax`/`amin`/`median`/`mode`/`std`/`var`/
`nanmean`/`nansum`/`count_nonzero`/`linalg_vector_norm`/`std_mean`/
`var_mean`) plus `softmax` (classified ACTIVATION in opmeta but a
reduce-over-dim op semantically).  `cumsum`/`cumprod`/`topk` were counted
separately — they are scan/rank ops, not reduce-dim, and stay out of the
tier's grammar.

All 68 workloads exported cleanly.  **161 reduction sites zoo-wide**,
across 32 workloads:

| op | sites | workloads (nonzero) |
|---|---|---|
| `sum` | 60 | GCNLayer(4), CapsuleRouting(39), MoSHead, RWKVMixBlock(14), TopKRouterMoE, … |
| `softmax` | 50 | attention variants, SoftSlotMoE, ExpertChoiceMoE, MoE routers, … |
| `mean` | 18 | norms (GroupNorm/InstanceNorm/RMSNormPerHead), CovarianceHead, TernaryWeightLinear, SE blocks |
| `cumsum` | 14 | ChunkedSSD(12), TaylorLinearAttention(2) — *out of tier scope* |
| `topk` | 9 | ExpertChoiceMoE(3), TopKRouterMoE(2), BlockSparseAttention(2), … |
| `amax` | 8 | MoE routers, BlockSparseAttention(4) |
| `linalg_vector_norm` | 2 | CosineAttention |

Reduction ops in the IR are uniformly single-arg with attr dicts
(`{"dim": (-1,), "keepdim": True}` style) — the spec form
`("mean", child_spec, {attrs})` is well-defined.

**But raw site count ≠ fusable claim count.**  The tier only claims a
reduction whose spelled body contains ≥1 elementwise op (the fusion
payload).  Bare `sum(x)`/`softmax(x)` and reductions over
non-elementwise children (`sum(matmul(…))`) decline — measured below.

## What was built

The fourth codegen tier, minimal form — collection + spelled denotation
+ the existing compiled-spelled binding path.  **No reduction Triton
codegen**: `Tier("reduction", …, None)` declares honestly that delivery
is the `torch.compile`d spelled body, which IS the denotation (the
minted op's meaning is the spelled spec verbatim), so the referee
prices exactly what ships.

* `genkernel.reduce_heads()` — `opmeta.REDUCE_DIM_OPS ∪ {softmax}`,
  lazily resolved (opmeta is the single source for the reduce-dim
  vocabulary).
* `genkernel.reduction_spec(spec)` — accepts iff head ∈ reduce heads,
  ≥2 ops total, and every non-head op ∈ `ELEMENTWISE`.  Nested
  reductions, contractions, reshapes and views inside the body decline.
* `genkernel.reduction_handlers(terms, covered)` — mints `red_<i>`
  `{pattern, kernel, args}` entries, `_canon_taken` dedup (attr dicts
  make canon unhashable — equality, not membership).
* `genkernel.TIERS` — `Tier("reduction", reduction_handlers, None)`.
* `play._reduction_sites(term)` / `_reduction_pattern` — collector:
  head + attr dict stay concrete (a claim binds that spelling, not
  every reduction of the op); args spelled by `_fusion_pattern`'s
  boundary rule (non-elementwise children abstract to metavars).
* `play.TIER_SITES["reduction"]` — registry entry; `_mint_tiers` picks
  it up unchanged.
* `play._claim_measurable` — `red_` joins the measurable prefixes:
  spelled eval through `_IR_TO_TORCH` is the honest denotation.

Binding path needed zero new code: `gen_bindings` iterates every handler
and `hybrid_sink`/`gen_sink` seat the compiled/uncompiled spelled eval;
`triton_bindings`/`scan_triton_bindings`/`decayed_triton_bindings` all
decline reduction-headed specs through their own recognizers.

## Measured payoff

`docs/reports/red_tier_probe.json` — per-workload `profitable` sweep
(kernel = delivery-path callable via `_triton_kernel_for`, spelled =
eager `_eval_spec`, reps=20, CUDA-bound values).

**At the default zoo probe sizes (d=16, t=8, b=2):** 22 of 68
workloads mint reduction specs; ~30 distinct `red_*` handlers after
canon dedup; **0 kept**.  Every measured ratio is a *loss* — 1.37×–2.76×
(28 logged samples).  This is the known tiny-shape regime: the
compiled-spelled kernel pays dynamo/inductor overhead the eager
spec-walk never does, exactly the regime that makes the pointwise tier
insist on real Triton codegen.  The gate drops all of them — nothing
unprofitable is offered.

**At d=64, t=64, b=8** (the sizes where kernels are documented to pay)
on the five spec-minting workloads spot-checked:

| workload | handlers | kept | ratios |
|---|---|---|---|
| RWKVMixBlock | 2 | **red_0, red_1** | 0.561, 0.728 |
| AdditiveAttention | 1 | — | 2.16 |
| CapsuleRouting | 4 | — | 2.40–2.81 |
| MoSHead | 2 | — | 2.57–2.67 |
| ManualGroupNorm | 1 | — | 1.43 |

End-to-end on RWKVMixBlock @ (64,64,8): `play._zoo_case` → `gen_probe`
— claims offered `gen_4`, `red_0`, `red_1`; **delivered=True,
verified=True** (max_abs 8.9e-16); **1.206× vs eager baseline**,
**0.892× vs whole-model `torch.compile`** — the incumbent compiler
already fuses these bodies, so the claim's honest ceiling is beating
*eager*, not beating Inductor.

## Verdict

The reduction tier is **implemented and honest, with a narrow real
payoff**.  It is not dead code: claims mint zoo-wide, the spelled
denotation prices correctly through `_claim_measurable`, the gate drops
every loser, and at paying shapes at least one workload delivers a
verified module 1.2× over eager.  Equally honest: it does not beat
`torch.compile` (0.89× on the one paying site measured end-to-end), and
at the default probe sizes nothing survives the referee — the win is
compiled-spelled fusion vs eager, the same mechanism every declined
pointwise claim already has.  The remaining upside would need a real
reduction Triton codegen (a `tl.sum`/`tl.max`-axis kernel over the
spelled elementwise body) to compete with Inductor rather than with
eager — that is a codegen project, not a tier.

Files: `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py`,
`packages/mlopt-discovery/src/mlopt_discovery/play.py`,
`tests/test_discovery_{genkernel,play}.py` (+11 tests),
`docs/reports/red_tier_probe.json` (this report's raw sweep).
