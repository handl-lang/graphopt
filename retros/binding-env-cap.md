# Retro — `_MAX_BINDING_ENVS`: the composite gauntlet terminates, honestly

**Date**: 2026-10-19.  **Follows**: `enablement-floor.md` (its named
scaling gap: "`run_gauntlet`/`pl.measure` on guarded minted composites
does not terminate within the box") and `fold-cost-single-source.md`.
**Commit**: this commit (`oracle._MAX_BINDING_ENVS`, the probe's
gauntlet leg, two pins in `test_discovery_oracle.py`).

## The hotspot, corrected

The retro named the 83-rule `_reach`/derivability saturation.  Measured
(`faulthandler` stacks + per-stage timers on the stored conv+BN
composite, one-case corpus):

| stage | measured |
|---|---|
| `verify_law` (derivability, 83 rules) | **0.0 s** — 54 enodes, not derivable |
| `_probe` / `_typed_probe` (lone rule) | 0.0 s each — 29 enodes |
| `_reach` (`_saturate` ×2, 83/84 rules, budgeted) | **0.0 s** — 9→54 enodes, `fixed_point` |
| `sweep_real` | 0.0 s — one real match |
| **`oracle.synthesize`** (`verify_view_candidate`) | **never returned** — the whole hang |

The explosion is not the e-graph; it is the **binding-env
enumeration**.  `synthesize`, `evidence._synth_sites` and
`object_synthesis._oracle_sites` all cap the *sites they yield* — but a
`derive` veto or an uninstantiable pair consumes an env without
producing a site.  The composite is a 4-deep `syn_*` composition whose
composed `derive` declined **2000/2000** enumerated envs at ~8k envs/s
on a domain orders past 10⁹ envs — every cap the consumers held was on
the output side of a faucet that never dripped.  This is the same
failure class `_MAX_VIEWED_BINDINGS` closed one level down; the missing
bound was on the env stream itself.

## The fix

`oracle._binding_envs` now stops after `_MAX_BINDING_ENVS = 250_000`
yielded envs — a property of the enumeration machinery, so every
consumer (`synthesize`, `_synth_sites`, `_oracle_sites`) inherits the
bound unchanged.  Sizing is measured, not guessed: the largest
legitimate sweep observed is `sdpa_fold_nomask`'s at ~180k envs to
natural exhaustion under `synthesize` (its `derive` vetoes ~2 200:1 —
the shipped worst case), so 250k preserves every shipped sweep with
headroom while bounding a veto-everything domain to a fixed corner of
the Cantor order — a partial sweep, the same honesty the instance and
viewed-binding caps already document.  (`rms_norm_fold`'s leaf-only
sweep measures 13.9M envs/134 s to exhaustion — it has no viewish op,
so `synthesize` is unreachable for it through `measure`; the cap now
bounds it too if it is ever swept directly.)

## The measured verdict

`conv_fold_premise_pay_probe.py` now runs `run_gauntlet` end to end on
the one-case corpus: **4.9 s** (was: >9 min, unbounded).  Stages:
reconstruct ✓, full-data ✓ (guard transport), measure ✓, **truth
FAIL** — `numeric=True`, `view=true` (the one evaluable instance
agrees), but the guarded synth region is starved (`0 accepted / 2000
declined / 2360 envs`, escalation included) and `derivable=False`
under `corpus.base_rules` (`ALL_RULES` — the premise closure is not in
the search set).  The refusal is the honest *starvation* verdict the
gate documents: the enumeration bank cannot mint the conv-channel
corner — the same "corner-reaching truth route" already named open —
not a wrong answer and not a hang.

**Pay leg (unchanged, now confirmed under the gauntlet's own measure)**:
`fires=1 changed=True verified=pass`, `dag_cost` fused `17 438.7` vs
spelled `34 874.3` — the FOLD_OPS single-sourcing made the composite a
payer.

## What stays open

- The truth gate's synth-side starvation for stored composites: the
  binding banks have no conv-weight-shaped inhabitants.  The named
  fix remains a corner-reaching enumeration vocabulary (or
  witness-driven sites) — the cap bounds the *work*, it does not move
  the corner.
- A stored composite's premises are outside `corpus.base_rules`, so
  `derivable` reads `False` where the premise-anchored replay
  (`verify_law` under `ALL_RULES + closure`) proves it.  Whether the
  gauntlet corpus should carry the record's premise closure is a
  policy question, not a bound.

Gate account: `tests/test_discovery_oracle.py` (+2 env-cap pins),
`test_law_view_oracle.py`, `test_discovery_{gauntlet,evidence,pipeline,
autocond,dspec_synth,synthesis,arena,meta_arena,enumerate,climb_b}`,
`test_machine_pack.py`, `test_construction_gaps.py`,
`test_conv_fold_rules.py`, `test_batch_norm_expand.py`,
`test_guard_sweep.py` — all green (the one failure,
`test_every_conditional_rule_is_swept` 125≠124, is pre-existing at
`bb6a2c4` — verified with the fix stashed).  ruff/format/ty clean on
touched files.
