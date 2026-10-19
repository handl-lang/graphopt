# Gate 3 — the form-level proposer (the fifth move reaches `loop`)

`lab/bridge/form_propose.py`, exercised by
`lab/bridge/combinator_probe.py --gate2 --extracted --form-propose` and
pinned by `tests/test_form_propose.py`.

## Result

**Gate 3 is reached — with one named residual.** On the real extracted
winner (the 7-block mixed fold, `455,752`-FLOP greedy arm) the proposer
reaches a program morphism containing **two `Loop`-headed folds**, one
per chunk-width family, under **one step schema** — and the whole thing
is *e-graph certified*: expanding the morphism (`comb_to_dag`) lands in
the winner's own root e-class (class 25), and every expanded block lands
in its own block e-class.

- chains found: blocks `[1,2,3]` (4-wide family, seed class 4081) and
  `[4,5,6]` (16-wide family, seed class 3549); block 0 is peeled.
- per-block verification: all six covered blocks `True`.
- `unroll(step,3)` per chain → `saturate_comb` with `PROPOSE_LOOP`
  folds each to `Loop(step,3)` (`loop_propose` fired 2×; cost 91 each).
- the assembled program `seq(pre, par(unroll, unroll, id), post)`
  reproduces the winner's concat skeleton and lands in the root class.
- obstruction reported (not hidden): `state-chain-broken:block3->block4`
  — the single-loop arm carries the state across the boundary and the
  per-block checks fail at 4,5,6, because the saturation never merged
  the small-family terminal state with the large-family initial state.
  ~~the two spellings are the same *value*~~ — **corrected 2026-10-16**
  (see below): measured directly they are *different* values,
  `carried = γ^12·M_4` to 1.8e-15, so the pair is not mergeable at
  all. Two loops remains the honest result — for a stronger reason
  than first recorded.

## The move that worked

Use the saturated e-graph **as the anti-unification oracle**, instead of
either pattern-matching the winner's syntax (move 1: `unroll` never
appears) or assuming the algebra (moves 2–4 each failed on the
unfolded geometric sum / re-derived state). Extraction *chose* the
unfolded spelling, but saturation had already **derived** the factored
one: each block's state-read e-class contains a `matmul(q̃, M)` member
whose right operand is the q-free accumulated matrix, and those M-classes
are linked by `add(mul(w,·),B)` update members — the recurrence is in
the graph, only not chosen.

So the step is **synthesised**: `pack(S′, o)` with
`S′ = w·S + (k ⊙ η)ᵀ v`, `o = om_decayed(qkᵀ/c, D, v) + ξ ⊙ ((q/c) @ S)`,
over 9 wires (state + own q/k/v splits + the five decay/scale
constants). Everything family-varying — index, chunk width, constants —
is on wires, so one schema serves both families. Wire bindings are
derived per block and every binding must be *graph-resident*
(`eg._locate`), so a wrong guess fails the check rather than fabricate.

## What this settles

- The combinator presentation **can** function as a search substrate:
  search + this proposer reaches `loop`-headed terms on the real
  extracted winner — twice, under one schema, certified.
- The residual obstruction is precise — and, after re-measurement,
  *stronger* than first recorded. See the next section.
- Lab-only, as required: no shipped IR node, no engine change; all work
  is in `lab/` + the proposal rule already in `lab/bridge/loop_propose.py`.

## Correction — the boundary is a *value* mismatch, not a reach gap

Follow-up measurement (the `compensated_loop_check` arm now in
`form_propose.py`, pinned by
`tests/test_form_propose.py::test_boundary_residual_is_a_value_mismatch`):

- The earlier claim "the two spellings are the same *value*" was
  asserted, not measured — the numeric arm only ever compared the
  per-chain rebuilt concat.  Lowered and compared directly, the
  carried boundary state (the 4-wide chain's terminal, e-class 9763)
  and the large family's initial state (class 3549,
  `matmul(transpose(mul(k_{(16,16,16,16):0}, η_16)), v_{(16,…):0})`)
  differ by exactly **γ^12** (`max|Δ| = 12.4`; `max|carried −
  γ^12·M_4| = 1.8e-15`).  The fold invariant is
  `S_i = γ^{φ_i}·A_i` — the absolute accumulation `A_i = Σ_p γ^{−p}
  k_pv_p` rescaled by the start of the previous *own-partition* chunk;
  across the boundary `φ` jumps 8 → 0.  **No sound rewrite can merge
  the classes** — this is not a budget or rule-coverage gap.
- Budget arm for diligence: `iters=20`, budgets 2–3× (kway 600 /
  bridge 1200 / reassoc 1200 / chunkify 200) → **69,993 enodes**, and
  the boundary is unchanged: 3549-class singleton, no
  `add(mul(w,·),B)` link, `decay_vec(offset=4,t=16)` still absent.
- The uniform step is still correct *at the value level* with two
  reindexed wires — `ξ'_4 = decay_vec(offset=4, t=16)` makes block 4's
  output exact to 2.2e-16, and steering the update with
  `w' = decay_vec(t=1, offset=4)` lands the block-5 state at exactly
  M_5 (1.8e-15).  The compensated arm records verdicts
  `{1:T, 2:T, 3:T, 4:F, 5:F, 6:F}` and names the two concrete
  residuals:
  - `boundary-renorm-unbound:block3->block4` — the producing-step
    steering pair `decay_vec(t=1, offset=-8)` /
    `decay_vec(t=4, sign=-1, offset=-12)` is not in the graph
    (negative-offset decay vectors are never minted);
  - `boundary-xi-nonresident:block4` — `decay_vec(offset=4, t=16)` is
    not in the graph, and even minted it could not help: class 3549 is
    a singleton and no staged rule splits a bare `matmul` along its
    contraction axis (only `om_decayed` has `chunkify_keys`), so the
    compensated read enode `mul(ξ'_4, matmul(q̃_4, carried))` has no
    path into class 93.
  - at block 5 the residual *is* a pure reach gap: the steered state
    equals M_5 in value but lands in a fresh class (the e-graph never
    derived `γ^4·carried + B_4 ≡ M_5`).
- What would close it (measured, not guessed): a "state-reindex" law
  family — a contraction-axis bisection for bare `matmul`, a
  `decay_vec` block-split law producing signed offsets, and scalar
  slide/merge laws on `mul(decay_vec, ·)`, plus saturation budget for
  the derivation.  None is a one-line rule; whether that bundle pays
  for itself is an ADR-0009 question, not a gate-3 oversight.
- Separately observed: blocks 3 and 6 carry 8 / 10 factored state
  candidates that are all *the same value* (verified numerically,
  max|Δ| = 0 between all pairs) in unmerged e-classes — the
  reach-gap phenomenon is real, just not at the boundary.

## Close — the state-reindex law family lands `Loop(step,6)`

The "what would close it" bundle above is now built and measured in
`lab/bridge/state_reindex.py` (lab law family, not in `DEFAULT`):

- `matmul_contract_split(sizes)` — the bare-`matmul` contraction-axis
  bisection as a law *parameter* (the `om_decayed_kway_blocks`
  precedent).  On `M_4`'s singleton 3549-class the measured `(4,4,4,4)`
  partition mints the four 4-chunk contributions — and through the
  naturality chain the boundary's signed-offset vectors.
- `split_mul_aligned` / `split_unsqueeze` / `decay_vec_split_*` —
  the chain that pushes the minted splits through `mul`, `unsqueeze`
  (with `split_transpose`, shipped) down to `decay_vec`, whose
  block-split produces **signed** offsets:
  `dv(g,T,-1,o)[start_i+k] = dv(g,SZ_i,-1,o-start_i)` — mints
  `dv(t=4,-1,-8)`, `dv(t=4,-1,-12)` and `decay_vec_usq_factor` mints
  the scalar `usq(dv(t=1,-8))` steering weight.
- `decay_scalar_gather` (`γ^a·(γ^b·X) = γ^{a+b}·X`), `decay_scalar_id*`
  (`γ^0·X = X`), `state_eta_slide*` (`γ^a ⊙ ((k⊙η_o)ᵀv) =
  (k⊙η_{o+a})ᵀv`), plus `distribute_mul_add` — the slide/merge that
  re-indexes `γ^{-8}·S_in3` into the absolute accumulation the
  bisection spells.

The driver `reindex_close` measures the boundary (the missing steering
eta's `t=4` gives the fine width, the target's contraction gives `T=16`
→ sizes `(4,4,4,4)`), runs one mint pass, then re-walks the compensated
arm with a `post_add` saturation hook so the steered terms merge as
they are minted.  On the real winner: verdicts `{1..6: True}`, the
steered `S'_4` lands in class 3549 exactly (`add(mul(w_{-8},S_in3),
B'_3)` ≡ `add(add(m_0,m_1),m_2)+m_3`), and the emitted **single
`Loop(step,6)`** lowers into the winner's root e-class —
`single_whole_equal: True` (`test_state_reindex_closes_boundary`).

Two implementation lessons, both measured:

- Budget accounting is in *enodes minted*, not fires, and persists
  across `eg.run` calls — the first close silently suspended
  `distribute_mul_add` at its 4000-enode budget before the steered
  merges could fire.  `reindex_budgets(sizes)` now sizes the family
  budget to cover the mint pass plus every post-add round.
- Post-mint the read classes hold extra factored members, so
  `seed_cands[0]` can be a same-shape/wrong-value candidate — the arm
  now probes the seed by verifying the first covered block.

`compensated_loop_check` also records `rep.single_envs` /
`rep.single_seed` so the emitted loop's wire bundles are the
*compensated* ones (steering `w'`/`η'` on the producing step, the
reindexed `ξ'` on the boundary read) — the step schema itself is
unchanged, only the per-iteration wires differ.
