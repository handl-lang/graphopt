# Retro — the composed atom-synthesis loop (`tools/atom_synth_loop.py`)

**Date**: 2026-10-14.  **Branch**: `feat/atom-synth` (merged `695cd89`).
**Depends on**: ADR 0007 (semantic-atom minting), `handler_synth`
(layer 2), `tools/atom_law_loop.py` (layer 3).

## What was built

`tools/atom_synth_loop.py` (~1290 lines) — the five-stage composed
loop the ADR-0007 layers only worked separately:

    measured need -> candidate signature -> verified handler
      -> synthesized laws -> certified admission -> pay probe

1. **Need census** (live, not a stale artifact): re-exports the intake
   corpus (254 terms), ranks `invisible_ops` — atoms the runtime sees
   but no law's LHS names.  Measured top needs: `cumsum`
   (5376 flops / 6 sites), `cumprod` (576 / 3), `logcumsumexp`
   (256 / 1); plus `pow` on the vocab lane and the `SelectiveScan`
   closed-form spec.
2. **Signature**: arity/attrs read off real corpus occurrences.
   **199 law-invisible ops have no signature builder and refuse here
   — that list is a deliverable, not a failure.**
3. **Handler**: `handler_synth.mint_handlers` per spelling; ops with
   existing torch bindings report `ambient` (the need is the *law
   set*, not the body).  Minted bodies bind into `_IR_TO_TORCH` so
   the oracle can evaluate the atom.
4. **Laws**: `atom_law_loop._run_atom` verbatim, with
   `_TYPED_FAMILIES` / `_ATTR_DOMAINS` injected per need.
5. **Admission + pay**: handler record + certified law rows +
   `cellstore` fragment, then a bounded default-vs-default+certified
   saturation on real terms.

## What minted end-to-end

- **`scan_fold`** — the one full semantic-atom mint.  The
  `SelectiveScan` corpus site
  (`mul(cumprod(σA), cumsum(B/clamp(cumprod(σA))))`) is spec'd four
  ways; the total-form spelling mints `("fold","A","B","H0",-2)` at
  tried=10, err **1.78e-15** — the selective-scan recurrence
  `h_t = a_t·h_{t-1}+b_t`.  Laws: 5 certified `primitive` —
  `sf.unfold` / `sf.refold` / `sf.b0` (auto_cond-minted guards
  `rank(a)>=2`, `rank(h)<=1`), `sf.split_same`, `sf.scale`.
- **`cumsum` / `cumprod` / `logcumsumexp`** — ambient handlers +
  certified law sets: 5/3/1 `primitive` laws each — `cum.add_same`,
  `cum.sub_same`, `cum.neg`, `cum.scale_same`, `cum.cat`;
  `cp.mul_same`, `cp.div_same`, **`cp.exp`** (the cross-atom
  `cumprod(exp x) = exp(cumsum x)`); `lc.shift_same`.
- **`pow`** — the vocab lane: the `2**rank` refusal -> `pow`
  proposal -> certified unblock (4/5 -> 5/5 agreement).  A separate
  `scan_needs` over *all* shipped rules (~470 s, run once) found
  **zero real blocked derives** — `pow` is a synthetic, not corpus,
  need.

## The stage-level funnel (the real measurement)

- **Signature**: 199 law-invisible ops refuse — the vocabulary-gap
  list.
- **Handler refusals, precisely located**: `cumsum` verbatim -> 0
  candidates (1-meta arity floor).  `cumsum` init-generalized ->
  refused at 512; the `fold_emit` `a≡1` side is only spellable as
  `div(X,X)`, which the *disjoint* a/b partition can never propose,
  and scalar fillers (`1.0`) crash `broadcast_tensors` — dead
  generator space.  `SelectiveScan` emitted spellings -> refused at
  512; the canonical `$acc` emit sits **>200k deep** in the
  round-robin (the descriptor verifies at 6.7e-16 when checked
  directly — the gap is enumeration *ordering*, not expressiveness).
  The `clamp_min` domain boundary is measured: the corpus's
  stabilized spelling diverges from the pure fold semantics at T>=32.
- **Law refusals**: bare linearity candidates -> `rejected-undeclarable`
  (the broadcast-into-scan-axis false region; `auto_cond` can't cover
  <=3 clauses, spelled `shape-compat` covers exactly).
  `cum.add_const` matches the real `cumsum(add(elu k,1.0))` site then
  dies on the `div(x,x)` `0/0` nan-trap.  `sf.corpus_refold` matches
  the corpus site (`real=1`) then dies on shape — the minted atom is
  the fold's *total*, the corpus *emits* per-step.  False twins die
  correctly.
- **Pay**: **0 certified-law fires on every real term.**  The
  corpus's actual cumsum usages (`cumsum(mul(x,x))`,
  `cumsum(elu(k)+1)`, `cumsum(mul(unsq,unsq))`) all live in the
  guarded-out region — broadcast/product compositions, not additive
  splits.  The guards that made the laws certifiable are exactly what
  correctly stops them firing on the real sites.

## Honest read

The loop composes end-to-end mechanically — one atom minted through
all four layers, four needs admitted with certified law sets.  But
**the minted atom does not pay on the real corpus**: `scan_fold`
computes the total, the corpus emits the sequence (the `fold_emit`
sibling is the paying one and it is enumeration-blocked); the
certified cumsum laws are true but fire nowhere real because the
corpus uses cumsum non-linearly.  This is ADR-0007's falsification
criteria working as designed: **truth and pay are measured
separately, and they are disjoint here.**

## Issues encountered

- `_family_gens` hard-gates fold/fold_emit at >=3 metas — a budget
  heuristic, not a semantics bound (documented in the report rows).
- `fold`/`fold_emit` empty-side const fillers (`1.0`/`0.0`) always
  raise in `eval_body` — dead candidate space in the shipped grammar.
- `Provenance(origin="ambient")` rejected (valid origins:
  shipped/minted/etc.) — ambient handlers record `origin="shipped"`.
- `vocab.scan_needs` over all shipped rules ~= 470 s; the driver
  defaults `--scan-rules 0` (the full-scan result: 0 blocked rules).

## Follow-ups this retro produced

- **Enumeration ordering** (`feat/enum-order`): reach the canonical
  `$acc` emit early — the fix that could flip `scan_fold` from
  pay-null to paying.
- Grammar dead space: stop emitting candidates whose const fillers
  cannot evaluate.
