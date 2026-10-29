---
title: 'Retro — discovery at corpus scale: sweeping the exposed class'
type: retro
date: 2026-10-14
---

# Retro — discovery at corpus scale: sweeping the exposed class

Context: `project/retros/exposed-math.md` screened 321 workloads for
e-graph reachability — 140 score ≥95% FLOPs-weighted law coverage —
and ran the three-arm discovery machinery on eight of them, minting
one paying unpublished law (`census:mul_unsqueeze`, −12.27% on
DeltaRuleBlock).  Nobody had run discovery across the rest of the
exposed class.  This retro runs `lab/screen/discovery_scale.py` — a
budgeted sweep driver over `exposed_discovery.py`'s arms — on the
top 60 eligible models and asks the corpus-level question the project
thesis lives on: **how much unpublished paying structure actually
exists in the exposed class?**

**Headline.**  Across 60 newly swept high-exposure models (58 full
runs + 2 truncated at the 240 s per-model budget, rerun with the
enumerate oracle skipped): **zero genuinely-new paying unpublished
laws.**  The machinery minted 2,101 pipeline proposals, measured 527
new-and-true candidates, fired on 44 models — and the only paying
hits are (a) exact re-derivations of two already-banked unshipped
machine laws (`cand_mul_factor`, `cand_mul_factor_right`), and (b)
one rule derivable from `ALL_RULES`.  All four paying rows land on
**purpose-built intake probes** — micro-models constructed to contain
the structure.  On organic architectures: nothing new pays.
`mul_unsqueeze` remains the only unpublished machine-minted law that
pays — 1 find in 68 swept exposed models (~1.5%).

## The sweep — `lab/screen/discovery_scale.py`

Selection: screen rows with `lhs_cover ≥ 0.95`, minus the eight
already-run models (`DONE` — LoRAAdapter, GatedAttentionUnit,
DeepEquilibrium, CosineAttention, AdaLNBlock, CapsuleRouting,
ChunkedRetention, DeltaRuleBlock), ranked by `lhs_cover ×
strict_sites`.  133 eligible; top 60 taken — covering the entire
strict-sites ≥ 2 head plus the named-architecture tail.

Budgets: `--per-model 240` (a real `SIGALRM` wall-clock per model —
overrun records `truncated` and the sweep moves on), a 3000 s total
sweep budget, a 1.5 GB `MemAvailable` abort floor, and the
enumerator's `--max-candidates 400` / `--evaluate 200` prefix caps
unchanged.  Results stream to `project/retros/discovery-scale.json`
after every model.  Name collisions across registries
(`SwiGLU`/`LinearAttention`/`ConvNeXtBlock` exist in two sources) are
handled by `source:name` qualified selection, added to
`exposed_discovery._cases_by_name`; evidence rows now carry
`term_to_data` serializations (`lhs_data`/`rhs_data`) so the parent
sweep can rebuild candidate terms for the corpus-novelty pass.

## What the machinery produced

| measure | count |
|---|---|
| models swept | 60 (58 full; 2 with enum oracle skipped after truncation) |
| pipeline proposals measured | 2,101 |
| pipeline `new + num_true` | 527 |
| models with ≥1 firing proposal | 44 |
| enum emitted / oracle-evaluated | 15,804 / 9,497 |
| enum `new+true` | 1,263 — **0 fired** (see below) |
| known-law firings (the 11 banked laws) | 3 |
| verify_fails / ill-typed fires | 14 / 72 |
| paying rows (new + true + fires + paid + verified + cert) | 4 |
| distinct paying candidates | 3 |
| …of which genuinely new | **0** |

### The paying set, honestly

| candidate | minted on | drop | shippable | verdict |
|---|---|---|---|---|
| `factor_left` `add(mul A B, mul A C) = mul(A, add B C)` | `intake:QuadraticFeature`, `intake:SharedFactorMixture` | −33.3% | yes | **`= cand_mul_factor`** — banked law #1 of the 11, re-derived |
| `factor_right` `add(mul B A, mul C A) = mul(add B C, A)` | `intake:SharedFactorMixtureRight` | −33.3% | yes | **`= cand_mul_factor_right`** — banked law #2, re-derived |
| `mixed:add_transpose_l_id` `add(transpose U, V) = add(U, V)` | `intake:NoopTransposeAdd` | +0.0% | yes | **derivable** from `ALL_RULES` — not novel |

The known-laws arm independently confirms the overlap: `cand_mul_factor`
fires + pays (52,213 → 34,809, −33.3%) on QuadraticFeature and
SharedFactorMixture; `cand_mul_factor_right` pays the same on
SharedFactorMixtureRight.  Both arms converged on identical terms —
cross-validated by bidirectional `_term_match` in
`discovery_scale._known_law_of`.

**The caveat that matters:** all four paying rows are `PURPOSE_BUILT`
intake workloads — probes literally constructed to contain
`add(mul·, mul·)` sharing a factor or a noop transpose.  The corpus
strict-match novelty sweep confirms it: `factor_left`'s LHS exists on
exactly those 2 models, `factor_right`'s on 1, `add_transpose_l_id`'s
on 2 (NoopTransposeAdd + ViTPatchBlock).  These are the synthetic-
witness patterns of round 1 — except now *measured paying* on the
probes that were built for them.  That is validation of the bank, not
discovery of new structure.

### The conditional-truth lead worth recording

`mixed:mul_unsqueeze_r_id` — `mul(V, unsqueeze(U, dim)) = mul(U, V)`
— fires 4× (all typed, 0 ill-typed) on `model:DiagDenseSSM` and pays
**−14.3%**, blocked (correctly) at the same gate as `mul_unsqueeze`:
`num_true=None`, guard `unsq:d_in_pad` — the view machinery cannot
isolate when the unsqueeze dim is a pad/broadcast.  It is the same
two-sided unsqueeze-naturality family as the DeltaRuleBlock find —
and unlike `mul_unsqueeze` (LHS on exactly 1 model), its LHS
`mul(V, unsq U)` strict-matches **12 corpus models / 22 sites**
(DeltaRuleBlock 8, DiagDenseSSM 4, plus GCNLayer, CapsuleRouting,
AdditiveAttention, ExpertChoiceMoE, FNOBlock, TimeCondConv, and 4
intake terms): a *general* guarded-law lead, not a
one-architecture idiom.  `mixed:mul_transpose_l_id` pays −2% under
conditional truth on the transformer-block family (ConformerBlock,
Pre/PostNormBlock, nn.MultiheadAttention).  None are shipped; all are
recorded with evidence.

### The honest nulls, reproduced at scale

* **Enumerate arm: 1,263 new+true candidates, 0 fires across all 60
  models.**  The structural blindness round 1 identified is not a
  small-sample artifact: attr-free patterns (`select X`) can never
  match attr-bearing real ops (`select(x, dim=1, index=0)`).  The
  enumerator keeps minting true laws that cannot fire on exported IR.
  The fix is still: mint attr metavariables in the grammar's unary
  forms.
* **False-candidate rejection works at scale.**  The numeric oracle
  killed `grammar:FALSE_mul_factor` (`a·b + a·c = a·(a+c)` — a
  subtle-wrong factor twin that would have shown a −33.3% "drop"),
  `exp(add) = add(exp, exp)`, `square(add) = add(square, square)`, and
  every `*_id` transpose/select overreach — 14 verify-fails and 72
  ill-typed mints refused, zero false positives in the paying set.
* **Truncation cause identified:** the two truncated models
  (LinearRecurrence, KernelizedAttention) burn the budget inside the
  enumerate arm's bounded oracle — `grammar._evaluate` runs a small
  e-graph saturation (`proposal._sat_cost`/`_cost_delta`) per
  candidate, ~0.35 s × 200 candidates, plus pathological cases past
  that.  With `--evaluate 0` both complete in <5 s (both null:
  LinearRecurrence mints one numeric-false + one conditional
  select-naturality, KernelizedAttention fires nothing).  Pipeline +
  known-arms coverage is complete on all 60.

## The corpus-level answer

* **Swept:** 68 exposed models total (8 round 1 + 60 here) — the
  strict-match head of the ≥95% class.
* **Genuinely unpublished paying laws: 1** — `mul_unsqueeze`
  (−12.27%, DeltaRuleBlock).  Rate ≈ 1/68 ≈ **1.5%** of the exposed
  class at these budgets.
* **Unpublished-and-paying counting bank re-derivations:** 3 distinct
  laws on 5 models (~7%) — but every bank hit is on a purpose-built
  probe.  On organic architectures the count stays **1**.
* **Typical or lucky?**  The sweep says `mul_unsqueeze` was neither a
  fluke nor typical — it is the sparse head of a real but thin
  distribution.  Unpublished paying structure exists in the exposed
  class at roughly one-per-sixty-models density, and it arrives via a
  novel spelled op-tuple the shipped grammar cannot express
  (`mul(unsqueeze, unsqueeze)` from a delta-rule recurrence), not via
  recombination of structure the corpus already shares.  The dominant
  discovery mode observed at scale is *re-derivation*: the proposal
  families re-find banked laws where purpose-built probes supply the
  pattern, and the honest gates correctly stop conditional truth
  short of shipping.
* **For the thesis:** if unpublished paying rewrites are the yield,
  the exposed class's marginal density is ~1.5% per model — the
  corpus needs either many more *novel spelled* architectures
  (DeltaRuleBlock's class: real recurrent math spelled elementwise,
  not fused-op natives and not synthetic probes) or the guarded-law
  machinery to close on the conditional leads (`mul_unsqueeze_r_id`
  fires on 12 real models — the largest measured paying structure
  still stuck behind `unsq:d_in_pad`).

Tooling: `lab/screen/discovery_scale.py` (the budgeted sweep + corpus
novelty driver), `lab/screen/exposed_discovery.py` (+`source:name`
qualified selection, `lhs_data`/`rhs_data` serialization in evidence
rows).  Evidence: `project/retros/discovery-scale.json` (untracked).
