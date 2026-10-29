---
title: 'Retro — `JOINT_SHAPES`: the enumeration''s correlated corner, reached'
type: retro
date: 2026-10-19
---

# Retro — `JOINT_SHAPES`: the enumeration's correlated corner, reached

**Follows**: `binding-env-cap.md` (its named
residual — "the binding banks have no conv-weight-shaped inhabitants;
the named fix remains a corner-reaching enumeration vocabulary") and
`fixpoint-api-and-conv-sweep.md` (the 60-anchor starvation account).
**Commit**: `819beeb` (the machinery + data), this commit (retro +
GOALS).

## The gap, precisely

`_binding_envs` binds every leaf metavariable **independently** —
`_viewed_bindings` diag-products per-metavar banks, `_pull_bases`
does the same for the free dimension.  A guard whose accepted region
is a *relation across operands* is unreachable no matter the window:

- `conv2d`'s `x[1] == W[1]·groups` *and* `b == (W[0],)` — three
  relations over three metavars;
- `batch_norm`'s four stat vectors all `(x[1],)` — five relations;
- the pointwise sibling's `(1, C, 1, 1)` channel view — a rank-4
  singleton the pad/derived transforms never mint (pad insertions
  only *prepend* axes, never mask a same-rank one).

Measured on the conv+BN composite: `0 accepted / 2000 declined /
2360 envs` — the corner was not *late* in the order, it was outside
the domain.

## The fix — data, not special-casing

`lawdata.JOINT_SHAPES` keys `(op, arity)` → concrete correlated
operand-shape rows (`(x, W, b)` = `(2,4,8,8),(6,4,3,3),(6,)` &c.).
The engine is three generic pieces:

1. **`_joint_plan`** — finds tabulated nodes with metavar args
   (`_joint_nodes`, a `_view_nodes`-style walk keyed `(op, arity)`),
   builds each node's row options (positional binding, `Var`s), and
   merges the product under Var *agreement* — two nodes pinning the
   same metavar must draw identically or the combo dies.  Coverage =
   union of bound names.
2. **Coverage replaces the independent draw** — a covered metavar
   leaves both `viewed` and `free`; the joint combos ride the
   `_viewed_bindings` product as dimension 0, so the coherent corner
   *leads* the enumeration.  Joint-free patterns enumerate
   byte-identically (the dim degenerates to `[{}]`).
3. **`JOINT_VIEW_SHAPES`** — the `(1,C,1,1)` channel-view corner,
   appended via the existing `extra_viewed` channel (the
   `CHAIN_VIEWED_SHAPES` scoping idiom) for joint patterns only.

Plus one attr-kind correction the sweep *measured*: conv2d's
`stride` drew `0` from the generic `int` domain and every accepted
anchor site died `non-positive stride` — `("conv2d","stride")` is now
a `pos-int` override; `dilation`/`groups` are positive by name
default (`as_strided` keeps `stride: int` — its 0 is legal).

## The measured verdict

`lab/probes/conv_fold_premise_pay_probe.py`, gauntlet leg:

| stage | before (`0721cec`) | after |
|---|---|---|
| truth (synth region) | `0 accepted / 2000 declined / 2360 envs` | **`105 accepted / 105 equal / 0 ne / 0 rerr` in 360 envs — PASS** |
| derivable | False → (premise-anchored) True | `True(premise-anchored)` — unchanged |
| novelty | — (unreached) | pass (`new`) |
| typed-pay | — (unreached) | **pass: fires=1 typed=1 ill=0 paid=1** |
| closure | — (unreached) | **FAIL: `enode=6.00x` vs the cap** |

Anchors (`_guarded_evals` @ 360, no escalation): all six measured
variants now read `accepted>0, equal==accepted, unequal=0` —
`scale_sp` 7/7, `scale` 10/10, `shift_sp` 8/8, `scale_spdg` 6/6
(the groups=2 corner also accepts: `x[1]==W[1]·2`), `scale_nobias`
14/14, `shift_sub_sp` 8/8.  `batch_norm_expand` keeps its corner —
the joint rows satisfy the stat-vector contract it already swept.

The truth-side blocker is **closed**.  The composite's refusal moved
to **stage 7 (closure)**: `enode=6.00x` — the fold's RHS expansion
grows the one-term graph past the closure bound.  That is a
*different* gate (cost-side, already named in
`premise-anchored-derivability.md`), not a truth question.

## What stays open

- **Closure bound vs expansion rules**: `enode=6.00x` is the
  composite's honest expansion ratio on the one-case corpus — whether
  `_CLOSURE_LIMIT` should distinguish "fold mints a big RHS member"
  from runaway blow-up is the next named policy question.
- **`matmul`/`linear`/`sdpa` joint rows**: deliberately left out —
  their sweeps measure today, and a covered metavar's domain *is* the
  row set, so an unneeded entry shrinks a working domain (the
  `CONST_DOMAIN` discipline).  Add a row only when a starvation is
  measured.
- **Witness-driven sites**: the retro's alternative route (drive the
  sweep from real fold sites) is now unnecessary for this corner —
  the enumeration reaches it — but remains the honest answer for a
  corner no row vocabulary can name (data-dependent shapes).

## Gate account

`tests/test_discovery_oracle.py` (+6 pins: joint nodes/plan, the
corner leads, joint-free byte-identity, the anchor + BN rescues),
`test_lawdata` (kind table), `test_guard_sweep`, `test_conv_fold_rules`,
`test_batch_norm_expand`, `test_discovery_{gauntlet,evidence,pipeline,
autocond,synthesis,enumerate}` — all green (the `test_every_conditional_
rule_is_swept` 125≠124 failure is pre-existing at `bb6a2c4`).  ruff
check/format + `ty` clean on touched files.
