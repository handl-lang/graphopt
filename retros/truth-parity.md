# Retro: truth parity — the store stops laundering a refuted derivation

**Plan 0017 / ADR 0004** — follow-up to `project/retros/derivable-gate.md`
(commit c2acfbf) and the asymmetry `project/retros/shippable-audit.md`
pinned.  Date: 2026-10.

## The residual hole

c2acfbf fixed the *gauntlet's* truth gate: `_truth_gate` reads
`num_true is not False and (derivable or num_true is True)` unguarded,
`clean and (derivable or equal >= 1)` guarded — a measured
counterexample outranks a derivation, and `derivable` may waive only
the *absence* of a measured site.

But the store kept classifying verdict rows off the pipeline's own
properties.  `verdict_row` persisted

```python
"verdict": "SHIP" if ev.shippable else f"no:{ev.no_ship_reason}"
```

and `Evidence.shippable` consults `Evidence.truth` — still
`derivable or num_true is True` (pipeline.py, sibling-owned).  So a
candidate the numeric oracle measured **false** on a real instance but
saturation **derived** would persist `verdict="SHIP"` whenever the
pay gates were green.  The store — the artifact the history report,
`--ships`, `--flips` and the cache all read — disagreed with the
gauntlet on exactly the case the derivable-gate change was about:
a proof standing on a measured contradiction.

`grep` confirms the store boundary was the only wave-through in this
module: `_truth_gate` (post-c2acfbf) already refuses; `verdict_row`
was the last place `derivable` could override a measured
`num_true is False` inside evidence.py.

## The fix

`evidence._stored_verdict` — the persisted-verdict classifier —
re-checks the measured site instead of trusting `ev.shippable`:

```python
if ev.num_true is False:
    over = "; derivation overridden" if ev.derivable else ""
    return f"no:false (numeric oracle rejects{over})"
return "SHIP" if ev.shippable else f"no:{ev.no_ship_reason}"
```

- A measured `num_true is False` persists `no:false …` **regardless**
  of `derivable` — the store-side spelling of the gauntlet's
  `num_true is not False` clause.
- `derivable` may still waive an *absent* measurement
  (`num_true is None` → admissible when the other gates pass) —
  starvation is not a counterexample.
- The reason stem reuses the pipeline's own vocabulary
  (`"false (numeric oracle rejects)"`), so a non-derivable
  measured-false row is spelled byte-identically to before; only the
  derivation-override case gains the `; derivation overridden`
  marker — the store's analogue of the gauntlet detail's
  `"derivation overridden by a measured counterexample"`.
- The measured-false check runs *before* `no_ship_reason`, so a
  `num_true=False` record whose view verdict was `conditional` now
  reports the counterexample rather than the weaker "conditional
  truth" — a measured contradiction outranks the view oracle's
  guard hypothesis too.

## What was deliberately NOT collapsed

- **`Evidence.truth` / `Evidence.shippable`** (pipeline.py) still let
  `derivable` override a measured false — the in-memory *candidate*
  bar is sibling-owned and unchanged.  The divergence is now
  *contained*: it cannot reach the store, the reports, or a cache
  hit.  A cache-served row rehydrates through
  `pipeline._evidence_from_row` and recomputes `truth` in memory, so
  the residual lives exactly where it did — flagged here, as the
  derivable-gate retro flagged it there.
- **`gap_gen.Found.truth`** (`base_truth or num_true is True or
  derivable`, gap_gen.py:668-676) is the *same* shape — a third
  instance of the waived-counterexample pattern, sibling-owned,
  flagged here the same way.
- **The ambient-`cert_fail` asymmetry** stays: the gauntlet charges
  only *attributable* cert failures while the pipeline's `shippable`
  demands `cert_fail == 0` — the deliberate looser direction the
  shippable audit documented.  This change aligns only the
  measured-counterexample override; the closure asymmetry is
  untouched.
- **The guarded-region sweep** remains gauntlet-only: `unequal` /
  `rhs_err` / `guard_err` counts are not `Evidence` fields and never
  reach `verdict_row`, so at the store seam `num_true is False` is
  the whole measured-counterexample signal — exactly the unguarded
  branch of `_truth_gate` it now mirrors.

## The delta — measured

Reclassifying every row in the real store (`tools/data/evidence.db`,
132 verdict rows / 66 candidates) under the new rule:

| class | rows | verdict change |
|---|---|---|
| `numeric_true = 0` | 24 | **0** — all already `no:false`, all `derivable = 0` |
| `numeric_true = 0 AND derivable = 1` | 0 | — the hole existed but no row sat in it |
| `SHIP` rows | 0 | — nothing in this store ever shipped |
| total rows whose verdict changes | **0** | |

**Zero reclassifications on the live store.**  The hole was latent,
not populated: every measured-false row on record was non-derivable
and already refused under the correct reason.  The fix is
prospective — the next `derivable ∧ num_true=False` candidate (the
`channel_then_row_scale` class) can no longer persist `SHIP` on the
derivation's strength.  No record's stored `truth` columns
(`numeric_true`, `derivable`) change at all — they were always
honest; only the derived `verdict` label could lie.

## Tests

`tests/test_discovery_evidence.py`, new *Truth parity* section:

- `test_verdict_row_measured_false_outranks_derivation` —
  `num_true=False, derivable=True`, all other gates green:
  `Evidence.shippable is True` (the pinned residual), the stored
  verdict is `no:false (…; derivation overridden)`.
- `test_verdict_row_measured_false_reason_unchanged` —
  non-derivable measured-false keeps the pre-existing spelling.
- `test_verdict_row_derivable_waives_only_absent_measurement` —
  `num_true=None, derivable=True` still stores `SHIP`.
- `test_stored_verdict_agrees_with_the_gauntlet` — the store verdict
  and `_truth_gate`'s unguarded branch refuse the same record.
- `test_stored_verdict_round_trip_refused` — the refused row
  persists and `latest_verdicts` serves it back as `no:`.

## Gates

`pytest tests/test_discovery_evidence.py tests/test_discovery_gauntlet.py
tests/test_discovery_pipeline.py -q`; `ruff check`, `ruff format
--check`, `ty check`, `radon_ratchet` — all clean.  No new
suppressions.  `pipeline.py`, `oracle.py`, `intake.py`, `gap_gen.py`,
`laws/` untouched.

## Limits

- The in-memory `Evidence.shippable`/`emit`/`intake` consumers still
  read the divergent property — a `derivable ∧ num_true=False`
  candidate can still *render* as shippable in a live pipeline run
  until pipeline.py gets the same clause.  The store, the cache and
  the gauntlet no longer believe it.
- Guarded (`cond`-carrying) records flow through `verdict_row` with
  only the unguarded signal available — region counts are
  gauntlet-only measurements, so a guarded object that is
  region-unclean but `num_true is True` still stores `SHIP` until a
  gauntlet run re-measures it.  That was true before and after;
  closing it would mean persisting region counts in the verdict
  schema — a migration, not a clause.
