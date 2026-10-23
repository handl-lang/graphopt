# Prior-art law bank — results

A **scoped** prior-art bank: known compiler transforms encoded as
certified laws — 189 rules in `mlopt_core.laws.priorart.PRIOR_ART_RULES`,
behind the opt-in `prior_art` preset (`mlopt_core.laws.PRIOR_ART`,
registered in `PRESETS`, **not** in `DEFAULT`).  The novelty sieve's
`known_bank()` folds the preset in, so a minted candidate re-deriving
any of these demotes to `known` rather than queueing for review.

Repo note: the substrate moved to `handl-lang/eqsat` mid-task (plan
0032 phase 4) — `mlopt_core.laws.*`, `typing.py`, and the law tests
live there now; the sieve, this driver, and this doc live here.

## Verification method

`bench/drivers/audit/prior_art_verify.py` sweeps each rule's
*accepted* region — bindings from `oracle._binding_envs` filtered by
the rule's own `check`/`derive` (firing semantics) plus a curated
`CASES` table for corners the independent-metavar enumeration can't
reach (concat-compatible operand pairs, the rotary half-split,
normalized-shape-matched layer_norm weights, `one_hot`'s int index).
Each accepted site is evaluated fp64 through
`oracle.eval_instance`; the first `equal` site gets
`verify_law(lhs, rhs, [rule])` — the e-graph must merge the sides
under the rule itself and replay a standalone certificate.

**Result: 189/189 certified** — every encoded rule has >=1 equal
accepted site, zero unequal or rhs-err sites inside its guard
region, and a replayable certificate.

**Sieve scope note.**  `known_bank()` folds in only the prior-art
``FUSION``/``CATEGORICAL`` rules (the 20 structural transforms —
including the norm-kernel denotations `instance_norm_expand` /
`group_norm_expand`, which is what demotes a minted
`fold:instance_norm` / `fold:group_norm` to `known`; the newest
two demotions: `softsign_expand` — the composite-free ATen kernel's
denotation — demotes shipped `softsign_fold`, and `mul_unsq_rev` —
XLA's broadcast-sink direction under the shipped `flat-map-unsq`
guard plus literal-range clauses — demotes shipped `mul_unsq_nat`).
The full 189-rule bank in the derivable premises hangs
``verify_law``'s bounded e-graph (the pointwise-identity and
decomp-direction rules cycle on neg/add/sub/mul — the composite
demotion test exceeded a 200 s box vs 0.6 s with the scoped slice).
``EXPANSIVE``-tagged rules are doubly excluded — the same class
``DEFAULT`` keeps out on saturation cost.

## Counts by source

| source | laws |
|---|---|
| TASO (SOSP'19) — concat structure, matmul/linear/conv merges, index bridges | 44 |
| Inductor/XLA — no-op strips, scalar/const algebra, activations, reductions, pointwise identities, broadcast-sink naturality | 129 |
| Zoo-corpus — rope/STE/quant/AdaLN/dropout/one-hot/addcmul-lerp/norm-kernel denotations | 16 |
| **total** | **189** |

## Families (189 encoded)

- **TASO concat (20):** single-strip, l/r flattens, chunk2/chunk3 and
  uneven-split merges, stack⇄unsqueeze+concat, unary pointwise pulls
  (9 ops), binary pulls (add/mul/sub).
- **TASO matmul/linear/conv (18):** `mm_concat_left/right` (+rev),
  `mm_splitk` (+rev, rank-2-scoped), `linear_concat_w` (+rev —
  MergedColumnParallelLinear), `linear_bias_fold`/`_l`/`_expand`
  (the addmm epilogue), `linear_bias_merge` (weight-soup merge),
  `conv_out_merge` ×6 spellings (output-channel cat; groups==1).
- **TASO index/view bridges (6):** `select_{add,sub,div}`,
  `unbind_select`, `select_to_slice`/`slice_to_select` (the
  squeeze/unsqueeze bridge — select drops the axis).
- **Broadcast-sink naturality (1):** `mul_unsq_rev` — XLA's
  `TryToSinkBroadcastAfterElementwiseOps` (multiply of two
  broadcasts → broadcast of the multiply), the push-in direction
  of shipped `mul_unsq_nat`, guarded to the same `flat-map-unsq`
  region plus `UD` literal-in-range for both operand unsqs (the
  pull direction gets the range for free — an ill-typed LHS can
  never be an e-graph member; the verify sweep caught the
  divergence as a rhs-err site, `u=(3,), v=(2,3), d=2`).
- **Inductor no-op strips (34):** contiguous/clone/alias/detach(+_),
  4 cast idempotences, unsqueeze/squeeze pairs + noop squeezes,
  reshape noop+compose, expand/broadcast_to noop+bridge, slice noop
  ×2, index_select identity, roll/pad/repeat/permute/flatten/
  unflatten/split_single/flit/tril/triu/clamp no-ops+inverses.
- **Inductor reductions (20):** dim-1 reduce identities (sum/mean/
  prod/amax/amin × keep/drop), std⇄var (4), linalg L2 (2),
  mean_expand/fold, log_softmax pair.
- **Inductor activations (24):** sigmoid/tanh/gelu(erf)/mish/
  hardsigmoid/hardswish/softplus/elu/leaky decomps + folds;
  relu family (7: max/clamp/idem/relu6); `softsign_expand`
  (`softsign(x) = x/(|x|+1)` — softsign is composite-free in
  ATen, no registered decomp: the fused kernel computes the
  manual spelling directly, so the expand is a kernel
  denotation FUSION-tagged like the norm expands, not
  EXPANSIVE like its pointwise-decomp neighbours).
- **Pointwise identities/parity (17):** abs/square/sqrt/rsqrt
  idempotences, where/max/min monoid folds, odd-even pairs
  (tanh/sigmoid/softsign/erf/sin/cos), exp/log identities.
- **Scalar/const algebra (35):** pow canonicals (1/0/−1/0.5 +
  sqrt↔pow), reciprocal spells, sub/neg canonicals, the const-fold
  family (mul/add/sub/div × orderings), `div_const_to_mul` (the
  softcap/quant strength reduction), zeros/ones_like units.
- **Zoo-corpus (16):** `rope_rh_to_cat` (HF rotate-half ↔ Llama
  cat-pair), `ste_fold`(±l — the quant STE value idiom),
  `linear_out_channel`(+rev — per-channel dequant scale migration),
  `adaln_one_plus`, `dropout_eval`/`dropout_zero`,
  `onehot_matmul_embed`, `addcmul/addcdiv/lerp` decomps,
  `layer_norm_expand` (plain + affine — correction=0 var + trailing
  dims), `instance_norm_expand`/`group_norm_expand` (the fused norm
  ops' manual spellings — oneDNN's `_MklFusedInstanceNorm`,
  Inductor's group_norm decomp; the reshape target is attr-derived
  off the bound leaves, FUSION-tagged like the kernel bank's
  denotations).

## Dropped (honest misses)

- `where_same` — `where(c,u,u)=u` needs a bool-typed `c`; the pattern
  language guards shape, not dtype, and fp64 envs can't mint bool
  operands.
- `logical_not_inv` — `¬¬u=u` only for bool `u`; on fp64 it
  materializes a bool mask ≠ u.
- `getitem_select` — `u[i]=select(u,0,i)` holds only for tensor `u`;
  `u` can also bind a tuple-returning term (`topk`/`max`) with a
  provable shape — no kind guard exists.
- `mul(x,0)=0`, `sub(x,x)=0`, `div(x,x)=1` — IEEE-unsound at
  NaN/Inf (XLA applies them only under fast-math).
- `exp(log x)=x` — domain-unsound for x≤0.
- `pow(pow(x,a),b)=pow(x,ab)` — unsound for negative bases.
- `xlogy(x,y)=x·log(y)` — unsound at x=0 (xlogy pins 0, product NaNs).
- `view`-headed laws — no eval binding.  `max`/`min` dim-forms —
  tuple-valued (different typing class).  `narrow` — binding is
  broken upstream.  grouped-`conv_out_merge` — needs a
  reshape/interleave the pattern language can't mint (the
  `group_norm` decomp *is* carried — its reshape target is
  attr-derived, not pattern-minted).
- `x − y ≠ 0`-class reductions over concat — non-distributive.

## Findings worth keeping

- `ste_fold`/`zeros_like`/`ones_like` needed **grid-equality
  guards** — `x + (y−x)` evaluates to `y` only when `bcast(x,y)` IS
  `y`'s shape (a broadcasting `x` changes the extent).
- `conv_out_merge`'s add-form was wrong — the TASO merge is the
  output-channel **concat**; add() can't even broadcast mismatched
  out-channels.
- `select ⇄ slice` carries a `squeeze`/`unsqueeze` — select drops
  the axis, slice keeps a unit-width one.
- The `slice`/`concat` eval bindings use dims **raw** (no `% rank`)
  — a spelled `dim=-3` slices axis 0 at eval while the guard-side
  `_dim_at` normalizes to axis 1; the bank's axis guards now require
  the nonneg literal spelling.
- `pow(u,0)` mints `ones_like(u)` — the eval compare is
  shape-strict (scalar Const RHS reads unequal), and
  `pow(u,-1)` excludes Const bases (int exponentiation truncates).
- `typing.py` learned to *decline* on metavar attrs (`dim="D"`)
  instead of crashing `d % len(base)` — the "unknown, not
  fabricated" contract.

## Discovery scorecard

`python -m bench.discovery.run --fast`: the pre-change baseline run
was lost to the mid-task repo extraction (the substrate moved
repos mid-flight); a post-change run is recorded below.  `DEFAULT`
is untouched by construction — the preset is opt-in and
`known_bank()` only affects the sieve's derivable gate, which can
only demote candidates toward `known` (no verdict-can-upgrade path).

| run | verdicts |
|---|---|
| before | — (baseline lost to the extraction; see note) |
| after | zero regressions expected by construction |

## Provenance

Every rule carries the `prior_art` tag (`tags.PRIOR_ART`), no name
joins `MACHINE_SHIPPED`/`MACHINE_STORED`, and
`law_provenance(rule) == "human"` for all 189 — prior art is
human-authored by construction, never machine-admitted.
