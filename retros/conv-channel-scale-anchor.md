---
title: Retro — `conv_channel_scale` anchor landed; the fold mints by ITERATED composition
type: retro
date: 2026-10-19
---

# Retro — `conv_channel_scale` anchor landed; the fold mints by ITERATED composition

**Follows**: `conv-bn-fold-synth.md` (named the
missing `conv2d`-headed anchor, gate A).  **Commit**: `945733f`
(`CONV_FOLD_RULES` in `mlopt_core.laws.tensor`, opt-in with
`BATCH_NORM_RULES`).  **Probe**: `lab/probes/conv_bn_fold_synth_probe.py`
— re-run, gates A/B/B3/C/D.

## The anchor — `CONV_FOLD_RULES` (60 rules, all data)

The conv twin of `linear_channel_scale`, plus its shift sibling — the
absorb half of the conv+BN weight-space pair:

```
conv2d(x, W, b) · ŝ  =  conv2d(x, W∘ŝ′, b∘s)      ŝ ~ (1, C, 1, 1)
conv2d(x, W, b) ± ĉ  =  conv2d(x, W, b ± c)       ĉ ~ (1, C, 1, 1)
```

* Per-output-channel scale folds through W's axis 0 AND the bias; the
  per-channel shift folds into the bias outright (conv is linear in
  its filter).  `linear`'s channel scale folds the weight's INPUT
  axis — conv's folds the OUTPUT axis.
* The `sv`/`cv` metavar binds **any term whose shape is the channel
  view** — the fold site carries compounds (`γ̂·r̂`, `ŝ·(−μ̂)+β̂`), never
  a lone reshape.  The `(C,1,1,1)` weight view `WV` and `(C,)` bias
  vector `SV` are dspec-derived reshapes.
* Guards (data): batched rank-4 conv2d only; `x`/`W`/`sv` concrete;
  `sv == (1, W[0], 1, 1)`; `F.conv2d` well-typedness `x[1] == W[1]·G`
  (`dim-eq` when the export elides `groups`, `attr-cmp`×`mul` when
  spelled).
* **Generated** per (attr-prefix × bias-arity × operand-order ×
  direction): conv2d's positional tail is a prefix of
  `(stride, padding, dilation, groups)` and `match_pattern` requires an
  exact attr-key set — 6 shapes × 5 prefixes × 2 arities, the
  `_make_sdpa_fold_rules`/`SPLIT_SCALAR_RULES` precedent.
* Verified fp64 against `F.conv2d` incl. `groups=2`
  (`tests/test_conv_fold_rules.py`, 10 tests).  Opt-in `CONV_FOLD_RULES`
  — held out of `ALL_RULES`/`DEFAULT` with the BN family pending the
  admission floor.

## Probe re-run — the measured account

**Gate A — corpus gap closed.**  conv2d: 0 → 60 lhs + 60 rhs entries.
**Gate B — pairwise `synthesize_rules` still refuses the bare fold**:
`rhs_heads_conv2d` is False in every bank (DEFAULT+BN, ALL+BN,
ALL+BN+CONV, ALL+BN+CONV+PW), `fold_composites` empty.  But the anchor
participates immediately: `syn_batch_norm_expand__
conv_channel_shift_sub_sp_6` mints `batch_norm(conv2d(…)) →
add(mul(γ̂, mul(conv2d(x,W,b−μ̂), r̂)), β̂)` — the running mean absorbed
into the conv bias in ONE pairwise composite.

**Gate B3 — iterated composition mints it.**  `synthesize_rules`
composes pairs over a FIXED `usable` list (derivation depth exactly
2; emitted composites never re-enter the pool).  Feeding each round's
emitted set into the next round's bank — `usable = base + emitted` —
the fold mints in **4 rounds** (emitted 26 → 61 → 242 → 944 on
fuel=30000):

```
syn_…_batch_norm_expand__conv_channel_shift_sub_sp_6
    __conv_channel_scale_sp_48__conv_channel_scale_l_sp_209
    __conv_channel_shift_sp_896
```

i.e. `batch_norm(conv2d(x,W,b),γ,β,μ,σ²) → conv2d(x, W′, b′)` — the
chain expand → shift_sub → scale → scale_l → shift.  Verified
**fp64-exact** on the real seed: max diff 6.8e-13 vs the fused kernel
(env bound from `source_tensors` — the `randn` env NaNs on `rv<0`,
gate B2).  W′ spells `W·rsqrt(σ²+ε)·γ` and b′ `(b−μ̂)·ŝ·γ̂ + β̂`
through the derived `CH`/`WV`/`SV` views.

**Gates C/D unchanged** — `dspec_synth` mints the derives staged
(`scale`/`b′` depth 3, `W′` depth 2 given `scale`); the one-shot
tensor-tier `W′` still refuses on depth ≥4 + `(C,)×(C,i,k,k)`
broadcast.

## What stays open (named, ordered)

1. **Composition depth is the refusal, not material.**  A
   `synthesize_rules` fixpoint knob — feed emitted composites back
   into `usable` until convergence or a round cap (the probe's gate-B3
   loop is the reference; a `rounds=` parameter or a driver in
   `mlopt_discovery`) — or the ADR-0008 anchored mint (antiunify over
   a measured seed↔folded pair) which jumps the depth directly.
2. **The emitted fold's guard content.**  The minted composite
   carries the transported conjunction of all five parents' conds;
   whether it serializes clean through `law_to_data`/`rulecache` and
   what its derived-attr placeholder chain looks like at admission
   time is unmeasured.
3. `CONV_FOLD_RULES` `DEFAULT`/`ALL_RULES` admission — the guarded
   sweep (`bn-admission-sweep.md` posture), now that the fold actually
   composes on a real seed.
4. Staged `dspec_synth` + a channel-broadcast `mul`/`tapp` in the
   enumeration bank — the gate-D sub-gaps, unchanged.

## Caveats measured

* `numeric_check=True` still drops the BN composites — the
  `_tensor_env` `rv<0` → `rsqrt` NaN artifact (gate B2); the iterated
  gate runs `numeric_check=False` and verifies the fold separately on
  the real tensors.
* The `_spdg` `groups` attr-cmp reads `x[1] == W[1]·GR` — the only
  clause in the family that multiplies a bound attr; a non-int `GR`
  declines (tensor-tier `_tcall` wraps the error).
