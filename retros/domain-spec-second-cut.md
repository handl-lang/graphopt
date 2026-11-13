---
title: 'Retro: domains-as-data — second cut (torch board, spec priors)'
type: retro
date: 2026-11-12
---

# Retro: domains-as-data — second cut

Follow-up to `domain-spec-stage2.md`.  The first cut's gap list
re-measured: two gaps closed for real, one closed mechanically but
measured flat, one closed cheaply.

## What the second cut taught

**The torch board was closer to data than the gap note feared.**
`_torch_cases` was already a loop over `(name → builder)` pairs —
the only code-shaped part was that the builders were lambdas
*inside* the function.  Lifting them into `play._torch_modules()`
turned "which modules" into `corpus.modules`, and the case tuple
shape `(name, term, rules, decls, supported, payload)` was already
the probe arity every board consumes.  `board="torch"` then fell
out for free: same `MetaArena`, same `count_cost`, sink
`supported_ops` bound, and `play._torch_deliver` as the deliver
hook.  `shapes`/`hardware` stopped being decorative on this board —
`widths`/`batch`/`device`/`dtype` flow into module construction,
the example tensor, and (via `model.to` before `to_ir`) the leaves
and IR params, so the deliver verify stays consistent.  CUDA and
float64 both deliver verified on the smoke path.

**The prior mismatch was real but small.**  Gap 3 assumed the
baked `META_ARENA_PLAYER_WEIGHTS` would mislead a narrow-vocab
domain.  The machinery to fix it is now cheap — `bake_prior`
imitates oracle lines planned on the spec's own boards (~12 s for
200 cases) — but the measurement says the shipped prior already
transfers: spec-baked priors land inside noise of it on pointwise
(+0.173/+0.149 vs +0.171/+0.180 learned/greedy).  The honest read:
inert features (ops/handlers a spec excludes never fire) plus
transferable `op:*`/`a:*` logits cover most of the mismatch; a
per-spec bake earns its keep only where the action *distribution*
diverges sharply.  Keep the bake opt-in, don't ship one by
default.

**Rules-minus is a footgun worth documenting.**  `{"base","minus"}`
subtracts names from an allowlist — but meta-arena cases carry the
*whole* DEFAULT nameset, so subtracting any DEFAULT law empties the
stream.  The profile is honest (it does what it says) but its
useful domain is carving scan extensions / explicit subsets, not
shrinking DEFAULT.  The torch board refuses incompatible profiles
up front instead of emitting zero cases.

## Design notes

* `meta_player.oracle_line`/`plan_oracles`/`imitate_oracle` grew
  `board_of`/`handlers` kwargs defaulting to the ambient `_board` —
  the oracle must simulate the board the player faces, or
  demonstrations name moves the spec board can't legally play.
* `_resolve_player` rebuilds `DomainSpec` over the same
  parts/legal/features with a swapped weight table — no registry
  tags needed because `parts` are objects, not names.
* `corpus.kinds` vs `corpus.modules` are validated *against the
  board* at load time — each errors on the other's board rather
  than silently ignoring the field.

## What remains genuinely code

`gen` (per-case kernel minting + cost gates + hybrid sink),
`open`/`player`/`bilinear`/`search`/`fragment`/`joint`, attr
*values* in the vocab whitelist, statistical footprint sampling,
meta-arena deliver.  The spec format now covers the two board
families whose cases are pure data; everything else needs either a
richer case-description language or stays a code domain — which is
the honest boundary, not a gap to apologize for.
