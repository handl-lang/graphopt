---
title: 'Retro: seq view-lift + domain-spec deliver field'
type: retro
date: 2026-11-12
---

# Retro: seq view-lift + domain-spec deliver field

Stage-2 leftovers + the seq-scan grammar boundary, one arm.  Report:
`docs/reports/p36_seq_view_lift.md`.

## What it taught

**The seq fold's state space is the operand broadcast — so put the
view in the operand, not the step expr.**  p35 recorded the honest
decline: `unsqueeze` in a step expr has no flat-space meaning
because the fold's accumulator lives in *operand-broadcast minus
step-axis*.  The escape was recognising the commute runs the other
way: `view(select(child))` ≡ `select(view'(child))`.  Lifting the
view onto the whole sequence tensor grows the operand's own shape
until the broadcast *contains* the state dims — GLA's `(b,d,d)`
matrix state stops being a step-expr broadcast trick and becomes
ordinary operand geometry.  No renderer, denotation, kernel or
binding changed; `_seq_view_lift` + a `_seq_deindex` branch was the
whole lever, and `shape_of` already typed the viewed children.

**Two sites, not one.**  The same gap held MambaSelectiveScan —
p35's census counted minted handlers, and both declined sites were
invisible behind `scan_spec`.  After the lift the zoo census is
three paying seq sites (RealGatedLRU ~0.29×, GLA ~0.35–0.39×,
Mamba ~0.375× — all KEPT at fp64 CUDA 16×8×2, fold==chain exact).

**Honest declines are cheap to keep honest.**  Views lift only
directly over a `select`; a view over a compound expr keeps the
node in the skeleton and `scan_spec` declines at mint — tested, not
assumed.  Remaining boundary: reshape/view/expand/permute,
dim-less squeeze, non-diagonal/chunked recurrences.

**The deliver leftover was a field, not a hook.**  `board="torch"`
already delivered; what the spec couldn't express was the *choice*.
`deliver: "auto"|"none"|"torch"` makes it data — and forces the
meta-arena answer into the open: symbolic terms have nothing to
lower or verify against, so `"torch"` is refused at load rather
than failing at `--deliver` time.

## Caveats

- Lifted operands materialise the broadcast frame — `(b,T,d,d)`
  for GLA/Mamba vs `(b,T,d)` spelled.  The win is the fold-vs-loop
  margin (~2.7× at this size), not free lunch; the referee prices
  per site and would drop a loser honestly.
- `test_shipped_primitives` has 5 PRE-EXISTING failures on clean
  HEAD (58 vs 57 prim records — artifact drift unrelated to this
  arm; verified via stash).
