# T-010 increment — the clamp attr grammar reaches the Triton renderer

**Date:** 2026-11-13 · **Scope:** `mlopt_discovery.genkernel` codegen
surface, measured over the 11 clamp-bearing zoo workloads
(`attr_elementwise_probe`, fp64 + fp32, `--probe`).

## Candidate survey and census (before building)

The three candidates from the agenda item, censused against the zoo
corpus first (every workload exported through `TorchSource().to_ir`,
attr-bearing op nodes counted recursively):

| candidate | corpus evidence | verdict |
|---|---|---|
| (a) stride-1-specialised reduction codegen | the noted fp32 blocker (`t010_fp32_attr_levers.md` lever B: "a project, not a patch") | deferred — multi-day emitter rework, payoff unproven |
| (b) next attr grammar | see census below | **chosen** |
| (c) chunked/cumsum seq carriers | `cumsum`: 14 sites, but only **2 workloads** (ChunkedSSD 12, TaylorLinearAttention 2); `p37` already proved no honest fold denotation exists without a chunk-algebra proof | deferred — occurrence-thin and algebraically blocked |

Attr-op census zoo-wide (sites / workloads): the shape/index attrs
(`unsqueeze.dim`, `select`, `transpose`, `reshape`, …) are view
machinery, not fusion payloads.  The *value*-attr ops the elementwise
grammar could plausibly absorb:

- `clamp_min` 20 sites / 6 wls, `clamp` 6 / 5, `clamp_max` 2 / 1 —
  **already in `ELEMENTWISE` and already minted** into `gen_*` specs
  (the attr-elementwise widening shipped earlier), but every Triton
  tier declined them at render time.
- `topk`(k) 9/5, `cumsum`(dim) 14/2, `one_hot`(num_classes) 6/4,
  `tril`(diagonal) 4/2, `linalg_vector_norm`(ord/dim/keepdim) 2/1 —
  none are elementwise bodies; the pointwise grammar cannot absorb
  them, and each is a different tier's problem.
- `layer_norm`/`group_norm`/`batch_norm`-family `eps` attrs —
  normalisation bodies, not elementwise; the `norm`/`postnorm`
  families in `vocab/families.yaml` are a different surface.

**The real gap was not the alphabet — it was the renderer.**  The
clamp attr grammar existed end-to-end in the *spec* layer
(`_fusion_pattern` appends `t.attrs`, `_spec_ops`/`_spec_metas` skip
dicts, `_eval_spec_dev` binds `**attrs`, `attr_site_env` matches
attr-mvars) and the ops sat in `ELEMENTWISE` — but
`_triton_node_h` declined *every* attr-carrying node, so all 28
corpus clamp sites could only deliver through the compiled-spelled
arm and were unpriceable by a generated kernel.

## What shipped

`packages/mlopt-discovery/src/mlopt_discovery/genkernel.py`:

- `_TRITON_ATTR_OPS = {"clamp", "clamp_min", "clamp_max"}` and
  `_triton_clamp` — renders each present bound as
  `tl.maximum(expr, bound)` / `tl.minimum(expr, bound)`.  Bounds
  render through the caller's *const* template (`tl.full` typed by
  `out_ptr.dtype.element_ty`), so fp64 tiles and every tile rank
  (`[BLOCK]` pointwise/scan, `()` reduction, `(BM,BK)`/`(BK,BN)`
  decayed) get the same dtype discipline const leaves already have.
  Absent/`None` bounds apply no clamp on that side (partial-bound
  `torch.clamp` spelling); a clamp with *no* applicable bound, a
  non-numeric bound, or foreign attr keys declines honestly — the
  hoist path (`__c` host operand) still takes it where the caller
  passes `hoists`.
- `_triton_node_h` routes attr-carrying clamp nodes through
  `_triton_clamp` before the generic hoist/decline; every other
  attr-carrying op is unchanged.

Because every tier renders bodies through `_triton_expr`, the one
change lifts **all** of them: the pointwise `_gk`/`_gk_s` kernels,
reduction bodies (`_red_*`), scan step exprs, seqm subexpressions,
and decayed operands.  `vocab/families.yaml` needed no change — the
ops were already in `elementwise_ops`.

## Verification

- **Unit:** `tests/test_discovery_genkernel.py` —
  `test_triton_expr_clamp_attr_grammar` (render shapes, partial
  bounds, const-template typing, all decline arms) and
  `test_triton_bindings_clamp_correct` (fp32+fp64 CUDA
  `assert_close` vs `torch.clamp_min` spelled).  File: 157 passed,
  1 skipped under `scripts/pytest-capped.sh`;
  `tests/test_discovery_play.py` 75 passed.
- **GPU correctness spot-check:** fused `mul(X1, clamp_min(add(X1,X2)))`
  spec — `triton_bindings` kernel matches spelled on contiguous and
  broadcast (64×1 × 1×32) inputs, both dtypes.
- **Honest referee, end-to-end** (`attr_elementwise_probe --probe`,
  11 workloads, zoo 16×8×2):

  | dtype | attr claims minted | Triton-bound | kept by `profitable` |
  |---|---|---|---|
  | fp64 | 22 (11 specs) | 22/22 `gen`-tier `bound` (was `declined` for every attr spec pre-change) | 1 — SinkWindowAttention `gen_0` (0.785× vs spelled) |
  | fp32 | 22 | 22/22 bound | 2 — CapsuleRouting `gen_0` (0.92–0.97×), SinkWindowAttention `gen_0` (0.69×) |

  Every other bound kernel measured 1.3–3.1× *slower* than spelled
  and was dropped — the referee paid only what measured.  Delivered
  module pricing (`gen_probe`, fp64 SinkWindowAttention): verified
  (`max_abs` 0.0), 0.72× vs eager / 0.59× vs compiled — the keep is
  real but does not pay at module level at zoo sizes, the same
  posture every other tier records there.

## Honest boundary

- The increment closes a *codegen coverage* gap, not a performance
  one: at 16×8×2 the clamp bodies are launch-bound and the compiled
  spelled arm is already good.  The payoff corner is wherever a
  clamp sits inside a larger fused body that would otherwise lose
  its Triton kernel entirely (the whole spec used to decline) —
  e.g. reduction bodies like `sum(mul(exp(clamp_min(x,0)), w))`,
  which now render where they previously could not bind at all.
- Non-clamp attr ops (`topk`, `cumsum`, `one_hot`, `tril`, norm
  `eps`) still decline — each needs its own tier/grammar, and the
  census says none are elementwise payloads.
- Worktree note: `play.py`/`meta_arena.py`/`test_discovery_play.py`
  carry unrelated uncommitted WIP (the `_route_scan_chains` delivery
  route) that predates this change; one fp32 probe leg hit a
  transient `NameError` while those files were mid-edit and passed
  clean on rerun.  This increment touched only `genkernel.py` +
  `tests/test_discovery_genkernel.py`.

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py` —
  `_TRITON_ATTR_OPS`, `_triton_clamp`, `_triton_node_h` attr branch.
- `tests/test_discovery_genkernel.py` — two new tests.
- Probe output: `/tmp/attr_probe_t010.json` (fp64),
  `/tmp/attr_probe_t010_fp32.json` (fp32) — session-local; the
  committed comparators are `docs/reports/attr_elementwise_probe_*.json`.
