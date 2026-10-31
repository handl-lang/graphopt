+++
id = "PL-019"
title = "Plan 0023 — the sequential carrier (state-level fold)"
status = "open"
priority = "medium"
+++

---
title: Plan 0023 — the sequential carrier (state-level fold)
type: plan
date: 2026-10-14
---

# Plan 0023 — the sequential carrier (state-level fold)

Depends on 0005 (the open game, phase 1 — the measured win).

## Context

Every piece of the delivery loop works: machine-derived
`applyd`/`aff_diag` carriers mint handlers, generate verified
Triton kernels, get measured and costed.  The single gap to the
O(T·C) claim is **producing** the sequential carrier member — the
e-graph never spells the cross-block contribution as a state-level
fold.

## Diagnosis (verified 2026-10, probe scripts in session logs)

On the real ChunkedRetention export, after
`OM_LAWS + OM_CHUNKIFY + SPLIT_SCALAR_RULES` saturation
(14 iters, 40/40 chunk budgets):

- `applyd` enodes exist — **depth 1 only**, zero `affd_compose`
  chains.  The fold machinery chains correctly in isolation
  (depth-2 on `((w8·c0) + (w4·c1)) + c2`), so the lifts aren't the
  blocker.
- **`mul(·, applyd)` members: 0** — `affd_lift_step`'s lhs
  (`add(mul(a, applyd), x)`) has nothing to bind.
- **`matmul(·, add(mul(…),·))` members: 0** — the state-level
  weighted sum never forms either.
- **Why**: contributions arrive as
  `matmul(mul(score_slice, decay_slice), v_slice)` — the decay is
  an elementwise mask *inside* the matmul's left operand, not a
  scalar on a folded state.
- **`om_decayed_cross` never fires** at the factored sites:
  `decay_block_cross` produces decay as a *deeper* unsqueeze-product
  chain (all factors `unsqueeze(·, dim=-1)`, a multi-factor
  product), not the strict two-factor `mul(unsq(ξ), unsq(η))` the
  law's lhs requires.  And `_vshape` returns `None` at the split
  members, so even if the lhs matched, the check can't verify the
  broadcasts.

So the chain `om_decayed_cross → matmul_col_decay → assoc_matmul →
q @ (γ^C·S + c)` stops at step 1.

## The construct (ordered candidates)

1. **Generalized cross law** — `om_decayed(s, mul(X, unsq(η,UE)), v)`
   → `mul(X, matmul(mul(s, unsq(η,UE)), v))`: `X` absorbs the
   residual row-side factors (any row-broadcaster, not just
   `unsq(ξ)`).  Subsumes the current strict law.  Guarded: X's
   broadcast shape must cover the row axis, the unsq the column
   axis, and the inner matmul must type.  (Watch the unsqueeze-dim
   polarity — the observed chain has every factor at `dim=-1`;
   which factor plays the `η` role needs confirming by shape, not
   position.)
2. **Decay normalization pass** — a small law that rewrites the
   n-factor decay product into the canonical `mul(unsq(ξ),
   unsq(η))` form first (reassociate + merge same-axis factors),
   then the existing `om_decayed_cross` applies.  More laws but no
   new semantics.
3. **K-way chunk law** — `om_decayed` → K-block split
   (`sizes=(C,)*K`) so row blocks see ≥2 prior contributions;
   pairwise bisection structurally caps the fold at depth 1 (one
   prior block per split).  Complements 1/2 — the bridge still has
   to produce the foldable spelling.

## Progress (2026-10-13)

- **`OM_DECAYED_CROSS_GEN` landed** — the generalized variant (``x``
  absorbs residual row-factors; guarded by broadcast-constancy on
  the contracted axis).  Fires 4× on the real export; the ξηᵀ
  split now happens (`matmul(mul(s,unsq(eta)),v)` members exist).
- **Stale diagnosis corrected**: the strict `om_decayed_cross`
  ALSO fires in the baseline (e-class matching sees the lone unsq
  inside the deeper-product class), and `_vshape` resolves split
  members fine — neither was the true blocker.
- **Next link** (exact): the produced members are
  `matmul(mul(s, unsq(eta)), v)` — decay *inside* the matmul's left
  operand.  `matmul_col_decay` wants `mul(matmul(q,kT), unsq(eta))`
  — decay *outside*.  The bridge now needed: either
  `matmul(mul(s,e),v) → mul(?, matmul(s, mul(eta,k))`-shaped laws
  pushing the factor through, or the state-level fold directly.
- `mul(·,applyd)` and `matmul(·,add(mul))` still 0; applyd depth
  still 1.
- **Div-naturality landed**: `mul_div_swap` (`(a/b)·c=(a·c)/b`,
  unguarded) + `matmul_div_distribute` (`(a/c)@v=(a@v)/c`, scalar
  `c`) — 48 `div-over-matmul` members produced; `mul(mm,eta)` now
  reaches `matmul_col_decay`'s lhs and its check PASSES (bindings
  resolve via `any_term`; verified 24 `matmul(q,transpose/mul)`
  members).  Remaining: the outer `add` of weighted state
  contributions inside the matmul's right operand — the
  accumulator itself, not the inner decay.
- Phase-2 side: `catopt_discovery.enumerate` — 1,160 candidates
  at depth≤2, 5 useful-new paying laws auto-found, FALSE controls
  verified.

## Progress (2026-10-14) — the state accumulator lands

- **`matmul(·, add(·,·))` member produced on the real export**
  (T=32, 16 iters, 60/60 chunk budgets, ~3 s): `matmul(q_i,
  add(S'_0, S'_1))` — the inter-chunk weighted state INSIDE the
  contraction, numerically verified against the class value
  (2.2e-15).  The rhs add also carries `applyd` and `mul(γ,·)`
  members — the folded spelling.
- The chain that lands it, all shipped laws:
  1. `SPLIT_FLATTEN` — nested same-axis splits collapse to flat
     partitions (`(16,8,8)`, `(8,8,16)`), unblocking deeper sites.
  2. `DECAY_BLOCK_CROSS_FLAT` — the strict-lower block law under
     MIXED partitions (range test `sum(SZR[:I]) ≥ sum(SZC[:J+1])`,
     subsumes the `J < I` sibling): deeper `decay_vec`
     factorizations at `split(split(decay,(16,8,8),-2,I),(16,16),
     -1,J)` sites — 9 `decay_vec` enodes, up from 5.
  3. `DECAY_STATE_WEIGHT_SINK` — `(a⊙b)⊙(q@B) → b⊙(q@(a⊙B))`, the
     per-block `γ^o` weight detaches from the row vector `γ^r` and
     lands inside the state.
  4. `DIV_ADD_FACTOR` (core `SIMPLIFICATION_RULES`) —
     `x/c + y/c = (x+y)/c` for scalar `c`: the score `√d` exits the
     block sum so `assoc_matmul_rev` + `factor_matmul` can collect
     the shared `q_i` under one right-operand `add`.
  5. `DECAY_STATE_WEIGHT_EXIT` — `q@(a⊙B) → a⊙(q@B)` (inverse twin
     of `matmul_scalar_factor`): `mul(γ^o, S)` addends —
     `decay_state_lift`'s lhs — appear inside the operand.
- **A real unsoundness found and fixed**: `matmul_scalar_factor`'s
  guard passed a `(T,1)` row-decay vector into the contraction —
  `c[r]⊙(q@B) ≠ q@(c[k]⊙B)`, measured diff ≈16.36.  Guard now
  requires contraction-axis constancy (right-aligned `c[-2] == 1`)
  via `_const_along_contraction`, shared by the sink/exit checks.
- Saturated counts (16-iter probe): ~620 classes, `applyd` 63,
  `matmul(·,add)` 1; extraction of the saturated root verifies
  against the module's output (5e-8 fp64 carrier-eval tolerance).
- Still not done for the plan's acceptance: extraction preferring
  the fold under the carrier cost model on a T=512 probe, and
  `profitable` measuring the minted kernel — those run on the
  delivery side, not the e-graph side.

## Progress (2026-10-13, evening)

- **Reachability: DONE.** On the real ChunkedRetention export the
  saturated e-graph contains the sequential carrier —
  `matmul(q_i, add(S'₀, S'₁))` with `applyd`/`mul(γ,·)` members in
  the add class that `decay_state_lift` folds.  Value-consistent
  (2.2e-15); root still equals the module output (5e-8 fp64).
- Five-law chain: `split_flatten`, `decay_block_cross_flat`,
  `decay_state_weight_sink`, `div_add_factor`,
  `decay_state_weight_exit` — all guarded, all sweep-clean.
- **Caught a real bug**: `matmul_scalar_factor` was unsound on
  `(T,1)` row vectors (measured diff 16.36); guard now requires
  contraction-axis constancy.
- **Preference: DONE (corrected).**  Under the *full* reachability
  law set (`+ MUL_DIV_SWAP, MATMUL_DIV_DISTRIBUTE,
  ASSOC_MATMUL_REV, FACTOR_MUL, FACTOR_MUL_ADD, DIV_ADD_FACTOR`,
  16 iters, 60/60) extraction at T=512 picks the **chunked
  cross-block carrier**: 29,386,752 vs 39,059,456 FLOPs
  (27,618,304 at 120/120).  The earlier "fused still wins" reading
  used the limited set — the naturality laws are load-bearing.
  Pinned by `test_extraction_prefers_the_chunked_carrier_at_t512`.
- **Measured end-to-end**: `ChunkedRetention` gen_probe
  866–1113 µs vs compiled 2018–2028 µs → **1.82–2.33×**, fp64
  max_abs ≈ 1.3e-8.  Scope: the paying claims are the decayed
  fused kernel + an elementwise gen kernel; the scan kernels are
  correctly dropped as unloweable.
- **Robustness fix**: `profitable` aborted the sweep when a
  harvested handler's kernel raised (unloweable depth-1 sites,
  512 vs 16 broadcast) — now declines; `ChunkedRetention` measures
  on CUDA again.
- **Tried and reverted**: a stricter `_scan_sites` broadcast
  filter (drop `aff_diag` sites whose `a`/`b`/`h` can't broadcast).
  Measured no-op — the state operand's shape resolves as a
  non-tuple, so the guard passes it through; the motivating sites
  (512 vs 16) still harvest.  The graceful decline in `profitable`
  is the correct and sufficient fix; a real filter needs a shape
  resolver that decodes carrier states, which is a bigger piece.
- **Corrected (2026-10-15, measured): the gap is a SEARCH gap,
  not pricing.**  A fold is a **cost-neutral relabelling** —
  `flops_cost(applyd(aff_diag(a,x),h)) == flops_cost(add(mul(a,h),x))`
  exactly; `_applyd_flops` prices O(state·steps) = O(T·C), the
  same as the spelled sum.  Extraction is therefore **right** to
  keep the spelled `concat`: every fold-containing root member
  costs ≥ it (T=512,k=2: extract 6,700,800 vs cheapest fold
  member 11,340,800; k=8: 8,572,928 at depth 1, 22,980,244 at
  depth 7).  `basic.py` untouched — no fix warranted.
- The deep fold **does** exist as a finite root member (honest
  metric: 3 at k=2, 7 = T/C−1 at k=8) — it is a **tree**, not a
  shared recurrence.  (The earlier depth readings came from a
  cycle-inflated probe metric; the buildability least-fixpoint is
  the correct one.)
- **The missing construct**: reassociate the cross sum
  `Σ_{j<i} γ^{(i-j)C} b_j` into the nested shared form
  `b_{i-1} + γ^C(b_{i-2} + γ^C(…))`, so the state is *shared*
  across row blocks and the existing `decay_state_lift` chain
  folds it.  A hand-built shared-state fold prices **589,952**
  vs the extracted 6,700,800 — **11.4×** — so the O(T·C) win is
  real and unreached.  The pieces exist
  (`matmul_col_decay`, `decay_vec_offset`,
  `matmul_scalar_factor`, `decay_state_weight_sink/exit`); the
  reassociation that makes the sum *shared* is absent.  It needs
  a soundness guard — do not invent it unguarded.

## Progress (2026-10-15, late) — the reassociation law shipped

`om_shared_state_laws()` (committed `809b50e`): three opt-in laws
reassociating the weighted cross sum into the nested shared form,
guarded (aligned integer offsets, shared decay base/axis,
`_const_along_contraction` for the mat variant) and fp64-exact
(~2.2e-16) for scalar and per-position weights.

- **The mechanism works**: a controlled 2-block test shows the
  reassociated state as one e-class with 2 parents, DAG cost
  172 → 116.  On the real export: **10.8–24.5%** (T=512, K=8:
  8,572,928 → 7,643,912).
- **Acceptance NOT met** — blocker exact: sharing needs an
  *i-independent* state (the key-value block `B_j`), but the
  export's score is `q(x)@k(x)ᵀ/√t`, so the graph only reaches
  `div(matmul(q_i, B_j), c)` — the scalar division wraps the
  contraction and `om_shared_mat_reassoc` fires 0× at K=8.
- **New finding — a rule-set conflict**: the natural bridge
  (`div(matmul(A,B),c) → matmul(div(A,c),B)`, exact for scalar c)
  **loops with the shipped `MATMUL_DIV_DISTRIBUTE`**.  The two
  cannot coexist unbudgeted; the fix is scheduling (mutually
  exclusive rule sets / staged saturation) or a div-tolerant
  variant (fires 126× at T=32 but saves marginally).
- So the remaining gap is **search-reachability + rule scheduling**,
  not algebra.  The 11.4× prize is still unreached.

## Progress (2026-10-15, later) — the rule-set loop is scheduled away

`DIV_INTO_MATMUL` + `om_shared_state_stages(k)` (committed
`c1a9521`): the two mutually-looping div families now run in
**ordered stages** on one graph — `reach`, then `reassoc` (with
`matmul_div_distribute` *replaced* by its alpha-exact inverse).
Budgets persist across the successive `EGraph.run` calls.

| T | K | reach | core (prev best) | **staged** | Δ |
|---|---|---|---|---|---|
| 512 | 8 | 8,572,928 | 7,643,656 | **6,943,236** | **−9.2%** |
| 128 | 8 | 1,695,488 | 1,279,448 | **954,772** | **−25%** |
| 64 | 4 | 557,568 | 485,780 | 467,348 | −3.8% |

- `om_shared_mat_reassoc` fires **0× → 112×** at T=512/K=8 — the
  stated blocker is unblocked; shared-state classes (≥2 parents)
  196 → 5,820.
- **Co-residency is explosive, measured**: at T=128/K=8 running
  both div laws in one stage did not finish in 800 s (stage A
  alone: ~30 s).  The staging is load-bearing.
- **Acceptance partially met.**  The residual is measured: the
  reassociated state is **not i-independent** — the export's cross
  terms carry `score_ij`, so the fold picks up q-bearing terms and
  each row block keeps its own state.  `extract_alternatives`
  shows **0 of 13** root alternatives carry a fold; the hand-built
  reference (589,952 at K=32/C=16) is still ~4.9× away.
- Secondary findings for the next pass: the P-prefix
  `om_shared_mat_reassoc` lhs is the wrong shape (0 matches; the
  no-P form with a shared `Q` matches 8×), and the weight pattern
  cannot bind an offset-less base weight (the matcher's keyset
  gate rejects `decay_vec(g, t=T)` with no `offset` attr).

## Acceptance

- T=512 probe on ChunkedRetention: a member where the root's
  per-row-block output contains `applyd` at depth ≥ T/C-1
  (or `affd_compose` of that arity), extraction prefers it under
  the carrier cost model, `profitable` measures the minted kernel
  against the fused baseline.

## Scars to respect

- `POINTWISE_FACTOR_RULES` exists (opt-in, `tensor.py`) — the
  fold-chain consumer; deliberately out of `ALL_RULES` (closure
  blowup measured 2.24x on the affd gauntlet).
- `_vshape` returning `None` at split members is a second gap —
  checks that need it can't fire; worth investigating whether
  `_xshape` should resolve `split` members.
- `COMM_ADD`/`ASSOC_ADD` stay out — bounded inclusion OOM'd on the
  11 GB host.

## Progress (2026-10-15) — construct 3: the K-way chunk law lands

`om_decayed_kway_blocks(k)` (`packages/catopt-carriers/src/catopt_carriers/om.py`,
opt-in via `om_kway_laws(k)`): `om_decayed(s, decay(g,T), v)` →
`concat_i(intra_i + Σ_{j<i} cross_ij)` for the uniform `(C,)*K`
partition (`C = T/K`) — diagonal blocks regenerate `decay(g,C)`,
strict-lower blocks factor as the rank-1 `ξ_i·η_jᵀ` outer product.
ONE application reaches `K` chunks; guarded by the usual dim/rank
checks, numerically verified fp64 (≤9e-16), and swept clean.  It is a
closure generator (its diagonal blocks re-match the lhs) so it stays
out of `OM_LAWS`, budgeted like `OM_CHUNKIFY`.

**Measured (T=512, C=16, fold-depth target 31; probe
`lab/probes/om_decayed_kway_probe.py`, seeded):**

| law | k | budget | n_enodes | dag_cost | fold depth | split |
|-----|---|--------|----------|----------|------------|-------|
| bisection | – | 40 | 1,015 | 29,386,752 | 1 | 3 |
| bisection | – | 60 | 2,757 | 29,386,752 | 2 | 4 |
| bisection | – | 90 | 4,066 | 27,618,304 | 2 | 4 |
| bisection | – | 120 | 8,397 | 27,618,304 | 2 | 4 |
| K-way | 2 | 100 | 7,181 | 7,659,008 | 3 | 4 |
| K-way | 2 | 200 | 15,305 | **6,700,800** | 3 | 5 |
| K-way | 4 | 200 | 10,449 | 10,324,480 | 4 | 7 |
| K-way | 8 | 200 | 17,315 | 8,572,928 | **7** | 9 |

- **The bisection's cost-vs-depth curve is flat**: depth saturates at 2
  (4 chunks) from budget 60 to 120, cost 29.4M → 27.6M (~6%) while
  `n_enodes` grows 3×.  The construct is NOT redundant — the bisection
  cannot reach the depth the fold needs.
- **The K-way law pays**: extracted cost drops to 6.70M (k=2, budget
  200) — **4.4×** below the bisection's 29.4M — and the e-graph fold
  depth rises to 3 (k=2) / 4 (k=4) / 7 (k=8).  Extraction still
  selects the chunked carrier (`concat`) in every row.
- **Honest caveat**: the *extracted* term is the explicit factored
  cross form (`decay_vec`), not the fold — its own `applyd` nesting is
  0.  A hand-built folded form prices at ~2.59M (K=32), so the O(T·C)
  floor is approached (6.70M) but not reached; the remaining gap is
  extraction *preferring the fold*, not the fold's depth.
- Cost-vs-depth is a genuine trade: k=2 minimises cost (6.70M, depth
  3); k=8 maximises depth (7) at 8.57M.  Neither reaches 31 — deeper
  needs a larger law budget, which the 11 GB host bounds.
- Counts unchanged: the law is opt-in, so `ALL_RULES` (82),
  `SIMPLIFICATION` (33) and the guard sweep (124) are untouched.

Tests: `tests/test_om_decayed_cross.py::TestOmDecayedKway` (12 tests —
fire/derive, fp64 verify, the block structure, declines, arity
validation, opt-in, the real-export cost drop).

## Progress (2026-10-16) — the fold's depth is NOT the blocker; the fold's *cost* is

Re-ran the probe and decoded the root class's members (`lab/probes/om_decayed_kway_probe.py`,
plus a member/decode harness).  Measured (T=512 unless noted, seeded,
`OM_LAWS + OM_CHUNKIFY + SPLIT_SCALAR_RULES + the 6 bridges + om_kway_laws(k)`):

- **The deep fold EXISTS as a finite term.**  A *cycle-safe* finite
  fold-depth metric (buildability = least fixpoint, then level-by-level
  fold-headed reachability) gives `T/C-1` exactly at k=8 (depth 7) and
  3 at k=2 — the acceptance depth is reached.  The probe's own DFS
  metric is cycle-inflated on the k=2/4 rows (it can walk an
  `applyd`→`applyd` e-class cycle no finite term realises); the
  *finite* depth agrees at k=8 and is the honest number.
- **The root class has no `applyd`-headed member.**  Its members are
  `concat`/`div`/`add`/`matmul`/`om_decayed` only.  `affd_compose`
  enodes: 0 in the everyday set (the compose-step lift lives in
  `SCAN_DIAG_LAWS`, not `OM_LAWS`; adding it makes 234 `affd_compose`
  enodes yet extraction is unchanged).
- **The fold members that DO exist cost ≥ the spelled extraction.**
  At k=8 the cheapest fold-containing root member is the *same*
  `concat` at the *same* cost (8,572,928 / 1,695,488 T=128); the deep
  (depth-7) member costs 2.7× more at T=512 (22.98M vs 8.57M).  At k=2
  the cheapest fold member is 11.34M vs the 6.70M extraction.  So
  extraction is not *mis*-preferring the non-fold — the fold is never
  cheaper.
- **Why**: `applyd(aff_diag(a,x), h)` prices EXACTLY like the
  `add(mul(a,h),x)` it re-spells (`_applyd_flops` = 2·numel(state);
  `mul`+`add` = the same).  Folding is a *cost-neutral relabelling*;
  the e-graph's fold chain is a tree, not a shared recurrence, so it
  saves nothing.  The applyd/aff_diag nodes themselves are negligible
  (7 × ~2048 FLOPs), so the cost model is NOT the blocker.
- **The paying fold is a SEARCH gap.**  A hand-built shared-state
  recurrence (`S_i = applyd(aff_diag(γ^C, (η_i⊙k_i)ᵀv_i), S_{i-1})`,
  each `S_i` one class consumed by block `i`'s output and block
  `i+1`'s step) prices 0.59M at K=32/C=16 — far below the 4.72M
  spelled cross and the 6.70M extraction.  The missing move is the
  **reassociation of the geometric-weighted cross sum
  `Σ_{j<i} γ^{(i-j)C} b_j` into the nested recurrence
  `b_{i-1} + γ^C(b_{i-2} + γ^C(…))`** so the existing
  `decay_state_lift`/`affd_lift` chain can fold it and the state is
  *shared* across row blocks.  `matmul_col_decay`,
  `decay_vec_offset`, `matmul_scalar_factor` and the
  `decay_state_weight_sink/exit` bridges are already the right pieces;
  what is absent is the reassociation that makes the sum *shared*
  rather than a flat per-block tree.  (Do not invent a new law for
  this without a soundness guard.)

Pinned by `TestOmDecayedKway::test_kway_deep_fold_exists_but_extraction_keeps_the_spelled_form`
(finite depth `T/C-1`, no `applyd`-headed root member, extraction keeps
the spelled `concat`), `::test_fold_is_a_cost_neutral_relabelling`
(pricing is exact, not the blocker) and
`::test_shared_state_fold_prices_below_the_spelled_cross` (the paying
form is a search gap).
