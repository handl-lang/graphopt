# Real-corpus proposers — closing the zoo-real bind gap

The `zoo-real` overnight arm (`bench/drivers/overnight.py --corpus
zoo-real`) mints each episode's candidate pool from the module's own
op-tuple census via `pipeline.propose`.  The first 16-module run
(commit `863825c`, `docs/results_overnight_real.md` @ `911163f`) minted
55 distinct candidates of which **20/55 bound ≥1 real env** (a real
env is one corpus subterm the claim's LHS `_term_match`es — the
guarded-region `envs` count; recount of the committed `.json` stage
details.  The other 35 never touched a real spelling: their oracle
rows read `real: … (0 envs)` and they were judged on the synthesized
window alone — the grammar arm's defect, one level down.

## Diagnosis — where the 35 zero-bind mints came from

| family | zero/total | why it never spells |
|---|---|---|
| algebraic-grammar | 13/15 | fixed algebraic LHS shapes (`div(M,M)`, `pow(M,1)`, `mul(M,0)`, …) the modules never write — the grammar proposes identities, not corpus spellings |
| census-naturality | 3/3 | `f(g(u,V_*), g(v,V_*))` shares one attr-metavariable namespace across both view nodes, so the precondition requires *equal* attrs — `AdditiveAttention` spells `mul(unsq,unsq)` with **different** dims, so `census:mul_unsqueeze` matched zero sites (the `mul_unsq_nat` history repeats: the equality needs a guard, not a shared metavariable) |
| elementwise-algebra / -factor | 9/9 | schema LHSs like `add(mul(A,B),mul(A,C))` need a structurally *shared* operand — absent from these modules |
| select- / slice-naturality | 4/4 | `mul(select(U,d,i),select(V,d,i))` wants equal `dim`/`index` on both sides — never spelled |
| linear-control / layout / annihilator | 6/8 | `matmul`-factor / `reshape(reshape)` / `mul(M,0)` spellings absent or attr-mismatched |

The pattern: every zero-bind candidate carries a *precondition the
corpus's spellings never satisfy* — a repeated metavariable, a shared
attr namespace, or a literal the modules don't write.

## What changed

Two moves, both in `pipeline.propose` (`mlopt_discovery/pipeline.py`),
both keeping the gauntlet untouched — this changes **what** gets
minted, never the bar:

1. **Attr-relaxed same-view family.** `_census_mixed_naturality` no
   longer skips `f(g, g)` view tuples: it emits the two-view family
   `f(g(u,A_*), g(v,B_*)) -> w(f(u,v))` (`wl`/`wr`/`id`) with
   *independent* attr namespaces — the spelling the module actually
   wrote.  The shared-attr `census:f_g` hypothesis still mints where
   it can bind; on this corpus it cannot, and the relaxed family is
   what makes `mul(unsq(d1), unsq(d2))` reachable at all.  Arm
   identity is unchanged — same `"census-mixed-view"` source, same
   five-arm `GENERATOR_ORDER` inventory the meta-game enumerates.
2. **Binding-first gate.** `propose(..., require_bind=True)` keeps
   only proposals whose LHS binds ≥1 `Op` subterm of the census terms
   (`_binding_filter` / `_binds_any` — the same `_term_match` the
   referee's `real_terms` domain measures).  `_RealArena` opts in; a
   module whose spellings mint nothing bindable is the honest empty
   pool.  The mint semantics is pinned in the ledger protocol
   (`"mint": "corpus-bind"`), so a pre-gate ledger resumes loudly
   instead of replaying ungated pools.

## Re-run — 16 zoo-real episodes, same protocol

| | before (`863825c`) | after |
|---|---|---|
| evaluated | 592 | 42 |
| minted | 55 | 29 |
| **bind ≥1 real env** | **20/55 (36%)** | **29/29 (100%)** |
| admitted | 2 (both known-art, both 0 real envs) | 0 |
| wall | 47.4 s | 25.2 s |

Every judged row now reads `real: … (N envs)` with `N ≥ 1`.  The new
same-view mints (`mixed:mul_unsqueeze_unsqueeze_*`,
`mixed:add_unsqueeze_unsqueeze_*`, `mixed:sub_unsqueeze_unsqueeze_*`)
bind the real `mul(unsq,unsq)`/`add(unsq,unsq)`/`sub(unsq,unsq)` sites
the shared-attr variants never reached — and are then honestly
refused (mostly `rhs_err`: the naive wrap does not denote; the true
law needs `mul_unsq_nat`-style premise closure, which is what the
gauntlet is for).

**Honest caveats.**  (a) Admission dropped 2→0: `sub_neg` and
`id_mul_lit` admitted before were synth-domain theorems that never
bound a real site — their loss is the gate working, not a regression.
(b) Pool sizes shrink to 0–7 per module; 4 of 16 episodes mint
nothing (GatedAttentionUnit, LoRAAdapter, SoftSlotMoE, DeepEquilibrium
have no pointwise-over-view tuples at all).  The remaining generative
headroom is real: families for pointwise-over-unary tuples
(`mul(sigmoid, tanh)`) and guarded attr-combine derivations
(`A_dim`/`B_dim` → output dim) are the next proposals the corpus can
already spell.
