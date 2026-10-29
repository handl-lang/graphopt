---
title: Retro — `synthesize_rules(rounds=)` landed; `CONV_FOLD_RULES` sweep refuses at truth
type: retro
date: 2026-10-19
---

# Retro — `synthesize_rules(rounds=)` landed; `CONV_FOLD_RULES` sweep refuses at truth

**Follows**: `conv-channel-scale-anchor.md`
(named next step: a fixpoint knob — "depth was the refusal, not
material") and `bn-admission-sweep.md` (the opt-in families' admission
floor).  **Commits**: `18e82dd` (the API + fixpoint tests), `62dce40`
(probe calls the API; the sweep driver), this commit (retro + GOALS).

## Item 1 — the fixpoint is a real API now

`mlopt_core.meta.synthesize_rules` gained three knobs; `rounds=1` is
the unchanged single pass (the body moved verbatim to
`_synthesize_once`):

- `rounds` — each round's emitted stratum feeds back into the bank
  (`usable = base + latest stratum`), so a depth-*k* fold mints within
  *k* strata.  Loop ends on an empty round, `rounds` exhausting, or the
  observer's veto.
- `max_emitted` — caps the aggregate mint; a round stops minting the
  moment the cap hits, so the cap bounds *validation* spend (the
  dominant cost — derivation replay + fp64 eval), not just list length.
- `on_round(rnd, stratum)` — the per-stratum observer; `False` stops
  the fixpoint (caller convergence — the probe's early-stop-on-fold).

The returned aggregate dedups cross-round re-mints by alpha key.
Measured nuance the docstring pins: a re-mint of an *unguarded*
composite is `_subsumed` while its stratum is in-bank (synthetic
f→g→h→k bank strata alternate 2→1→2); guarded composites cannot
subsume, so the probe bank's strata grow 26→61→242→944 — raw strata
1,273 rows, 974 unique after dedup.

**Probe re-verified end-to-end**: `gate_iterated` now calls the API and
reproduces the mint exactly — fold composite
`syn_syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_
channel_scale_sp_48__conv_channel_scale_l_sp_209__conv_channel_shift_
sp_896` in round 4, fires on the real seed, fp64 `allclose` True
(max diff 1.35e-12 vs the fused kernel; the earlier 6.8e-13 figure was
the same run's env draw — both fp64-exact at tol 1e-10).

## Item 2 — the admission sweep: truth-starved, not typed-pay

`lab/probes/conv_fold_admission_probe.py` stores each rule verbatim
(`kind="law"`, `cert=None`) into a scratch store
(`cache_dir()/conv_fold_admission.db`) and runs `run_gauntlet` over
`default_gauntlet_corpus()` (364 terms / 245 probes / 83 base rules,
~7 s to assemble, 0.2–0.5 s per rule).

| family | stage account | verdict |
|---|---|---|
| `CONV_FOLD_RULES` (60) | reconstruct + full-data pass; measure `matches=0`; **truth: 0 accepted guarded sites** (synth 0eq/0ne/0rerr over 2000 declined / 2360 envs — the `_GUARDED_CAP` escalation window included; real 0 accepted, ≤1 declined); `derivable=False` (no conv premises in `ALL_RULES`) | **REFUSED at stage 4 (truth)** — all 60, `unproven (no evaluable instance)`; the 30 `nobias` variants additionally read `numeric=False, view=ill-formed` on the raw unguarded verdict (the minted-bias RHS denotes only where the guard accepts) |
| `BATCH_NORM_RULES` (3) | unchanged: `batch_norm_expand` fires=8 typed=8 ill=0 **paid=0**; the two short arities `fires=0` | **REFUSED at stage 6 (typed-pay)** — identical to the recorded sweep; the fixpoint work did not move it |

**The refusal is honest but earlier than expected.**  The conv family
is fp64-verified *true* in `tests/test_conv_fold_rules.py` (incl.
`groups=2`) — the truth gate refused not on a counterexample but on
**starvation**: the guarded-region sweep's synthesized instance bank
never produces a coherent `conv2d` + channel-view site (rank-4 `x`/`W`
with `x[1] == W[1]·groups` *and* `sv == (1, W[0], 1, 1)` — a corner
outside the enumeration's shape vocabulary, not merely past its
default window), and the real corpus has no spelled
`mul/add/sub(conv2d(…), view)` member — aten exports keep the
per-channel affine inside `batch_norm`/`conv2d`.  `matches=0` so
typed-pay never even measured.

So the family's value is the one measured last cycle: **discovery
enablement** — the anchors are what let the fixpoint mint
conv2d-headed composites at all (gate B3's 944-row stratum is
impossible without them).  `DEFAULT` unchanged.

### What an "enablement-aware admission floor" would require

1. **A truth route that reaches the guarded corner.**  Either extend
   the enumeration bank with the op family's shape vocabulary
   (generators that mint coherent `conv2d`/`(1,C,1,1)` sites — the
   channel-view lattice), or drive the sweep from *witness bindings*:
   real fold sites the composite fires on (the BN-expanded subgraph
   positions gate B found), replayed like `_seed_witnesses` does for
   validation.  Today a guarded corner the bank can't spell is
   indistinguishable from an untrue rule — the floor can't tell
   "unmeasured" from "false".
2. **Premise-of-an-admitted-composite pay accounting.**  Admit an
   anchor when a composite in whose derivation it participates clears
   the whole gauntlet — fires, typed, `paid > 0`, closure-safe.  The
   minted fold is the candidate: if the fused-conv member wins
   extraction on a real conv+BN corpus site, its five-anchor
   derivation earns admission *as a derivation* (the anchors ride as
   premises, not as independently-paying members).  Currently
   unmeasured — that's the next experiment, not an assumption.
3. **Guard transport for stored composites** (see below): whatever
   the floor, a guarded minted composite can't even reach stage 3
   today — its record fails `full-data`.

### Follow-up measured: composite guards vs `law_to_data`/`rulecache`

- `law_to_data`: honest refusal — 64/72 minted rules from rounds 1–2
  are guarded composites; each reports
  `missing_hooks=('check','derive')`, `serializable: False`.  The
  `_compose_guards` closures are procedural by construction, so a
  guarded minted composite stored as an object **fails `full-data`**
  — the evidence store cannot carry a transported guard as data
  today.
- `rulecache`: **partial** — `guard_pats` + parent names serialize
  fine, and `load(key, base + emitted)` round-trips all 72.  But the
  load is all-or-nothing on parent resolution: fixpoint output
  (round-≥2 parents are `syn_*` names not in the caller's bank)
  **misses entirely against the base ruleset** — one unresolvable
  parent name returns `None` for the whole cache, not a partial set.
  A persisted fixpoint needs either the stratum-aware ruleset passed
  back (works today) or parent-transitive records (unbuilt).

## Test / gate account

- New: `tests/test_meta_fixpoint.py` (6 tests — exact strata, the
  `syn_`-parented depth-3 composite firing, cap mid-round and at the
  boundary, veto, empty stratum, `rounds<1`).
- Re-ran: `test_meta`, `test_meta_edges`, `test_synthesis_guards`,
  `test_synthesis_seeds`, `test_rulecache{,_codec}`,
  `test_batch_norm_expand`, `test_conv_fold_rules` (114 pass);
  `test_discovery_dspec_synth` (68 pass with the above).
- `ruff check`/`format`, `ty`, `lint-imports` clean on touched files;
  meta.py coverage on the new code is complete under the slice.
- Pre-existing drift (verified at HEAD `240a345`, not mine): radon
  baseline vs current radon reports +1–3 cc on seven functions across
  `cond.py`/`egraph/core.py`/`meta.py`/`handler_synth.py`; vulture
  flags `tests/test_form_propose.py:142`.  `_synthesize_once` was
  baselined at exactly the moved function's 51 — a rename, not a
  ratchet touch.

## What stays open

- The fold composite itself is unadmitted — a guarded `syn_*` object
  fails `full-data` today (transported guards aren't data); running
  *it* through the gauntlet is blocked on the serialization item
  above, not on truth (it fp64-verifies).
- `synthesize_rules_cached` takes no `rounds`/`max_emitted` — the
  cache key would need them in `params`; not wired (single-round
  callers unaffected).
- `on_round` returns the stratum by value; a caller that wants to
  *filter* feedback (e.g. only guarded composites re-enter) has no
  hook — measured need first.
- BN's typed-pay floor and the enablement floor are the same named
  gap now — one machinery change (premise-of-admitted-composite
  accounting) would re-measure both families.
