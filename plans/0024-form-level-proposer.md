# Plan 0024 — the form-level proposer and vocabulary minting

Depends on: ADR 0005 (the open game), plan 0023 (the carrier),
`catopt_discovery.enumerate` (the instance enumerator).

## Motivation

The enumerator closes the *instance* half of candidate
generation: 27 rewriting forms (comm/assoc/factor/pull/hom/…)
instantiated over the op grammar, deduped and classified against
the library — 1,160 candidates at depth ≤2, five paying laws
auto-found.  That is enumeration **within a fixed form list**.

Two surfaces remain human:

1. **The form list itself.**  `_f_comm`, `_f_factor_left`, … are
   hand-written skeletons.  A shape the list lacks is unreachable
   — the same wall the fold hit, where the machine needed the
   K-way *construct* and a human had to design it.
2. **What to mint.**  The minting machinery is built (composite
   ops, declared atoms + generated kernels, handlers, the cell
   store), but *which* atom to propose is human-supplied.  The
   game can mint anything expressible; nothing proposes it.

Both are the same question one level up: **propose the shape,
not the instance.**

## What exists (don't rebuild)

- `enumerate.py` — the 27 forms + `_form_pairs` round-robin,
  `_filter_reason` pre-filters, `alpha_key` dedup, `_relation`
  classification.
- `proposal.evaluate` — numeric oracle, cost delta, verdicts;
  `rank_pool` — learned/heuristic/random AP.
- `auto_cond_object` — synthesizes the guard a candidate needs
  from its measured domain (the game minted `xy+xz=x(y+z)` iff
  `x≡y`).
- `dspec_synth` — synthesizes a declarative `derive` from a
  hook's measured outputs; refusals name the missing bank op.
- `object_synthesis` — `compose`, `specialize`, `relax_guard`,
  `_mint_guarded`; `cellstore.admit_object` + `run_gauntlet`.
- `open_game` — the enumerated stream as a playable board with a
  trained prior.

## The construct

### A. Form-level enumeration

Generate *skeletons*, not instances.  A form is a small
meta-term: op-classes (binary/unary/literal) at nodes and
metavariables at leaves, with the lhs/rhs pair constrained to a
single algebraic identity.  The 27 shipped forms are a hand
sample of this space; the space itself is generatable.

1. `enumerate_forms(*, max_nodes, arities, classes)` — a lazy
   stream of skeleton pairs over a **meta-grammar** (node =
   op-class + children; leaf = metavar or literal-class), with
   the shipped forms recoverable as members (a test asserts
   each `_f_*` skeleton appears in the stream — the stream is a
   superset, so nothing regresses).
2. Sound-by-construction filters only (arity/class match,
   tautology, uninstantiable metavar); everything else flows to
   the existing `evaluate` + gauntlet — the refusal machinery is
   already honest, so the form level inherits it.
3. The open game's board generalizes: a move may be *a form*
   (whose instances are then evaluated) or an instance — the
   prior learns which forms pay, which is the transferable
   knowledge the instance-level prior cannot express.

### B. Vocabulary minting

Propose an **atom** when a measured need names one:

1. **Need detection.**  Two existing signals: (a)
   `dspec_synth`'s refusals (`_near_misses` naming the missing
   bank op — `pow` today); (b) a candidate class that
   *repeatedly* fails only on an expressibility gap (the same
   subterm shape recurring in saturated graphs across models
   with no law covering it — minable from the census).
2. **Proposal.**  Emit a `declare` move for a declared op + a
   generated kernel (the handler path that already exists),
   with the semantics *verified observationally* against the
   spelled form on real tensors — the ADR-0005 floor
   (substrate-expressibility, not the shipping op list).
3. **Admission.**  Unchanged: numeric oracle → typed pay gate →
   certificate replay → adversarial sweep.  A minted atom that
   does not pay in the cost model is admitted as vocabulary but
   not preferred — the honest outcome.

## Measured residual (2026-10-15) — the need-detection ground truth

`dspec_synth` re-run over every residual procedural derive
(`OM_RULES + TRACE_RULES + XC_RULES + DECODE_RULES +
DECODE_GEOM_RULES`) — **69 rules**, not the earlier 47 (that
count covered `OM_RULES` alone):

| outcome | n | what it means |
|---|---|---|
| converted | 1 | `matmul_t_concat` — `max(rank q, rank k1) − 1`, verified on all 166 envs |
| HIT but unfaithful | 18 | `sdpa_cat_*` — the synth fits only because `SC` is always bound; converting would drop the `**−0.5` default-scale fallback. Kept procedural (honest refusal) |
| decline | 29 | carrier-aware shape logic (`_cat_axis_plan`/`_key_cat_plan`), `%` axis normalisation, constant-only domains |
| no samples | 21 | the hook never declines in a measurable site |

**The residual is not arithmetic depth** — near-misses sharpen
with the new ops (`masked_fill_cat_slice_dim` now agrees 124/129)
but stop short of a fit.  What the bank lacks is *carrier-aware
shape sources* (`_cat_axis_plan`-style planners), `%`/`pow`, and
a shape resolver that reads the value shape rather than the cost
convention.  That is the concrete vocabulary-minting backlog —
each named, each with a measured agreement count.

## Measured outcome (2026-10-15) — part A ran

`catopt_discovery.forms` is built (committed `b0c3e2d`): lazy
skeleton enumeration over the meta-grammar, candidate
instantiation reusing the existing filters/dedup, and the
falsification harness.  All 28 shipped skeletons are recoverable
(superset verified).

**The falsification is NOT triggered — but the reading is narrow.**
Over 12,000 candidates (default 8-op vocab, `max_nodes=3`) the
machine admitted **7 paying laws from forms outside the shipped
27** — the cross-op-agreement family (`sub(X,0)=add(X,0)`,
`div(X,1)=mul(X,1)`, `B0(X,K)=B1(K,X)`, …).  Each is a *composite
re-spelling* of two shipped identities, so the new algebraic
content is thin.

Honest conclusion: **the form list is not the whole grammar, but
the forms it lacks are shallow.**  The form level is a real
capability (it reaches shapes a human list omitted) and it does
not, by itself, produce a deep discovery.  The open question moves
to part B — whether a *minted atom* (vocabulary, not a
re-spelling) can pay where no composition does.

Stream sizes: small vocab (2 bin/2 un) = 7,252,000 skeletons at
`max_nodes=3`; the 8-op default is effectively unbounded — the
stream is lazy and capped by `max_forms`/`max_candidates`.

## Measured outcome (2026-10-15) — part B ran

`vocab.py` (extended) + the bank ops (committed `f1cdb56`).
`scan_needs` over all 68 residual procedural derives produced the
machine's **own ordered backlog** — and it inverts the plan's
assumption:

| op | rules unblocked | Σ agreement gain |
|----|----|----|
| **`mod`** | **6** | 18 |
| `pow` | **0** | — |

`pow` — the gap this plan named — unblocks *no* shipped carrier
rule; its refusal was the synthetic probe's, not a real derive's.
The residual's actual need is `%` axis normalisation (`mod`).
And the ~29 carrier-aware planner declines (`_cat_axis_plan` /
`_key_cat_plan`) are unblocked by **no** arity-2 op — they need a
carrier-aware shape *source*, i.e. a structural gap, not
vocabulary.

**Conclusion for part B:** need detection works and is honest
(proposal ≠ admission; `admitted: False` always), and its value
is the *measured ordering* — the plan's guess about which op was
missing was wrong, and the machinery said so.  The remaining
surface is structural (shape sources), which is the next level up
again.

## Measured outcome (2026-10-15) — the shape source is a zero

The structural gap was addressed (committed `9e7b462`): a
`("vshape", T)` op in both DSL dispatch tables plus a
`register_value_shape` sink seam (core cannot import carriers),
with `xcarrier` registering `_xshape`.

**Rules unblocked: 0 — and the reason is the finding.**  The
synthesis measurement domain (`oracle._binding_envs` via
`_measure_domain`) binds **leaf terms only**, so
`vshape_diff_envs == 0` on all 27 declines and `_xshape ≡
_shape_of` there.  The value-shape gap is real only at
*law-evaluation* time, where `any_term` resolves a metavar to a
carrier member — a domain the synth cannot reach.

So plan 0024's diagnosis was **half-right**: the shape source was
*necessary but not sufficient*.  What actually blocks the
remaining declines is the synth's **expression/guard bank**
(`if`/`guard`/`%`/`dim` composition) and multi-output fitting —
not the shape source.  Eleven of the 27 are constant domains (no
signal at all).

**Revised next target:** deepen the *measurement domain* (bind
op-terms, not just leaves) so a carrier-aware derive is
measurable, and extend the expression bank with guarded
composition.  The DSL capability is now in place and verified
(`_shape_of == (4,)` vs `_xshape == (5,)` on `apply(aff(A,c),h)`).

## Falsification

- **Form level**: over a fixed evaluation budget, does
  form-level enumeration admit laws from forms *outside* the
  shipped 27 at a useful rate?  If every admission is a
  re-instantiation of a shipped form, the level adds nothing and
  the honest conclusion is that the form list is the real
  grammar.
- **Vocabulary**: does need-detection propose an atom that
  admits and pays, without a human naming it?  The `pow` gap is
  the first candidate — the refusal is already measured.

## Metrics to report (not assert)

- yield per oracle eval: form-level vs instance-level;
- admitted laws attributable to a *new* form (the discovery
  count, which must be >0 for the claim);
- vocabulary proposals: proposed / admitted / paid;
- the honest negative: what still needs a human and why (the
  refusal reasons, enumerated).

## Scars to respect

- No comm/assoc in default saturation (measured OOM at bounded
  budgets on the 11 GB host).
- Closure-generating constructs stay opt-in and budgeted
  (`OM_CHUNKIFY`, `om_kway_laws`, `POINTWISE_FACTOR_RULES`).
- `dspec_synth` refuses on thin/degenerate sample sets — the
  form-level proposer inherits that discipline (held-out
  validation, no coincidence fits).
- The measured win is the gate: an admission that does not pay
  is data, not a result.
