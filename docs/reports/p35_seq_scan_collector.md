# p35 — seq-scan collector: raw recurrence chains mint seq specs, one real site pays

Date: 2026-11-12.  Scope: `mlopt_discovery.play` (new
`_seq_scan_sites` + `_seq_step_cands`/`_seq_spines`/`_seq_deindex`/
`_seq_site`/`_all_scan_sites`, `TIER_SITES["scan"]` now the
composite), `mlopt_discovery.genkernel` (`_site_parts` passes a
site-resolved `"axis"` through to the minted handler),
`tests/test_discovery_play.py` (`TestSeqScanSites`).  Motivation:
p34 revived the seq-scan *pricing* path but recorded the honest
boundary "no collector mints a seq spec" — `_scan_sites` only
harvests post-saturation `applyd` carrier members pinned
`kind="unrolled"`.  This change adds the missing producer.

## Site census (before the collector)

Prior census: seq=0 / unrolled=87 across the 68-workload zoo —
`_scan_sites` output only.  The open question was whether any zoo
workload carries a *legitimate* sequence-axis site the unrolled
collector was flattening or missing.

Raw-IR inspection says the recurrence models mostly do NOT spell a
clean unrolled diagonal chain:

- `RealGatedLRU` — literally an unrolled Python loop over `t`:
  `h = a[:,t]*h + f(a[:,t], i[:,t], u[:,t])` per step, `select`
  dim=1 index=t over `(b,T,d)` operands.  **The seq form's natural
  host.**
- `GatedLinearAttention` — same loop shape but step exprs carry
  `unsqueeze` (outer-product state update); the collector finds the
  spine, `scan_spec` declines the attr-carrying skeleton honestly.
- `MambaSelectiveScan`, `GatedDeltaRule`, `DeltaRuleBlock` —
  `add(mul)` nodes exist but chains break after ≤1 step (state
  threading goes through non-`add(mul)` structure or matrix state).
- `RWKVMixBlock`, `ChunkedSSD`, `ChunkedRetention` — 0 `select`
  nodes / no diagonal chains (chunked, parallel, or matrix-state
  recurrences).

## The collector

`_seq_scan_sites(term)` walks the RAW IR (no saturation — the
unrolled recurrence is already there) and mints
`{"spec", "leaf_env", "axis"}` records that `scan_handlers` turns
into ordinary `scan_<i>` handlers:

- `_seq_step_cands` recognises one step `add(mul(a, h), x)` (either
  add slot, either mul factor — up to 4 readings, `a*h` order
  preferred).
- `_seq_spines` keeps the longest decomposition per node, ties
  capped at 4 so seq-ification arbitrates; a node with no step
  reading is its own base.
- `_seq_deindex` abstracts every `select(child, dim, index)` in the
  per-step `a`/`b` terms to a sequence-operand metavar bound to the
  WHOLE child tensor; `rsub(x,s)` normalises to `sub(s,x)`; any
  bare leaf (a whole-tensor reference the fold cannot slice) or
  string Const declines the term.
- `_seq_site` emits the spec only when: every step's `a`/`b`
  deindex to the IDENTICAL skeleton; all selects share one dim;
  `shape_of` proves equal operand ranks with the step extent ==
  the step count at that dim (the soundness check — the fold's
  broadcast slice then reads exactly what the original `select`
  read); and every select index equals its chronological position.
  Anything unverifiable declines rather than guesses.
- Interior prefix chains are skipped once the maximal chain emits
  (the fold covers them); cap 16 sites/term.

The emitted spec is
`("applyd", ("aff_diag", a_spec, b_spec), "H", {"kind":"seq",
"axis":D})` — seq by attr pin, one leaf, axis resolved.  `leaf_env`
binds each `Q*` operand meta to its full sequence-tensor term and
`H` to the initial-state term; `_site_parts` now merges a site
`"axis"` into the handler entry so `handler_denotation` and the
Triton binding agree on the step axis (overriding the `-2`
default).

## Soundness posture

- The fold denotes the chain BY CONSTRUCTION: the spec is emitted
  only when deindexed skeletons agree step-for-step and select
  indices walk the axis in order — verified empirically below.
- Wrong mul-factor readings self-filter: the incorrect "map" is
  the state term (a `zeros`/attr-op or bare `Var`), which either
  fails deindex or renders nothing `scan_spec` accepts; tied
  alternatives are tried longest-first, first seq-ifiable wins.
- The ambient `applyd` map semantics is never used: `kind="seq"`
  routes `handler_denotation`/`gen_bindings`/`profitable` through
  `seq_scan_denotation` exactly as p34 wired.
- Like the unrolled scan claims, the pattern cannot occur in raw
  IR — it is a minted vocabulary entry priced by the referee, not
  a rewrite the board fires.

## Measured

ZOO CENSUS (16×8×2, fp64 intake; 32×8×4 and 16×16×2 spot-checked —
same result): `_seq_scan_sites` finds 2 candidate spines zoo-wide —
RealGatedLRU (mints) and GatedLinearAttention (declines at
`scan_spec`, `unsqueeze` skeleton).  Every other recurrence model
yields 0.

REAL SITE, REAL PAYOFF — `RealGatedLRU`, fp64 CUDA (RTX 2050),
the unmodified intake→referee path (`_mint_tiers` → `profitable`
with `kernel_for=_triton_kernel_for`,
`denotation_for=_claim_denotation_for`, reps=20):

- minted: 17 `scan_*` handlers total (16 unrolled + **1 seq** —
  `scan_16`, args `(Q1,Q2,Q3,H)`, axis=1);
- measurable: yes (`scan_*` admitted, denotation = the fold);
- bound: Triton seq kernel bound on CUDA;
- **kept**: `PROFITABLE_LOG` window — spelled fold 352.2 µs vs
  Triton fold 102.9 µs → **ratio ≈ 0.29× (~3.4× faster)**.

Fold-vs-chain verification: `handler_denotation` output vs
`eval_term` of the original unrolled chain on the real leaf
values — **max abs diff 0.0** (exact, fp64).

Synthetic controls (`tests/test_discovery_play.py::
TestSeqScanSites`): positive mint + fold==chain + handler
axis/leaf_env; declines on reversed select order, non-chain
`applyd` terms, and whole-tensor (non-select) map factors.

## Honest boundary

One paying site in the zoo.  The collector is general — any
exported unrolled `add(mul(a_t,h),b_t)` chain over `select`
slices with a consistent skeleton mints — but the zoo's other
recurrences are chunked/parallel/matrix-state forms this diagonal
spelling does not cover, and `unsqueeze`-bearing step exprs
decline at `scan_spec` (no elementwise render).  Widening means
more step-expr grammar (view ops in the seq renderer) or matrix-
state carriers — separate levers, not collector bugs.
