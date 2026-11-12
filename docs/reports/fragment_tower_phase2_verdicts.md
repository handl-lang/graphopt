# Fragment tower — phase-2 verdict residuals closed

Plan: `project/plans/pl-0029-plan-0033-the-fragment-tower-mlopt-vocabularies-as-language-data.md`
(phase 2 — denotations as Models; the fp64 gate of ADR-0157)
Prior record: `docs/reports/fragment_tower_phase3.md` ("Honest
residuals" — the five `python-shipped` verdicts),
`docs/reports/fragment_tower_phase2.md`
Sibling repo: `handl-lang/host-tensor` — `tensor.select` +
`tensor.linear` shipped in `85c68d3` (T-017 phase 5); **no host-tensor
change was needed for this work** — the ops were already on the
surface, the blocker was entirely graphopt-side.

## Result

`vocab/models.yaml`: **109/109 verdicts `handl-shipped`, 0
`python-shipped`.**  The last five rows — the handler-refused prims
`denotation:synop_{4c5296b3ba, 4c657cb2c0, 976f9836d7, 98a002d64b,
f08a53f9a0}` — promoted through the real verdict path
(`scripts/verify_vocab_models.py` → `modelproj.verify_model` →
`hnl-run`), not schema compliance:

| row | max_err | max_rel | sites | attr_bindings |
|---|---|---|---|---|
| `synop_4c5296b3ba` | 0.0 | 0.0 | 8 | `a97=1` |
| `synop_4c657cb2c0` | 7.1e-15 | 3.2e-13 | 8 | `a155=7, a75=1` |
| `synop_976f9836d7` | 0.0 | 0.0 | 8 | `a101=1` |
| `synop_98a002d64b` | 0.0 | 0.0 | 8 | `a128=1` |
| `synop_f08a53f9a0` | 0.0 | 0.0 | 8 | `a103=1` |

Worst `max_rel` across all 109 shipped verdicts: `3.2e-13`
(`synop_4c657cb2c0`, the `linear`/`select` row — fp64 noise, ~4
orders under the `rtol=1e-9` floor).  Each verdict records its
`attr_bindings` in `models.yaml` — the measured binding the gate
ran under.

## The two gaps, precisely

1. **Domain provenance.**  `_domains()` read
   `handler.record.provenance.domain` off `minted.prim_records()`.
   The five prims' handler stage *refused* at mint time (v2 run,
   `budget=256`), so no `provenance.domain` was ever recorded —
   pre-phase-3 records carry no refusal ledger either.  The
   `--prim-vocab` rerun (`docs/reports/results_overnight_prim_v4.
   json`) minted handlers for the same content-hashed shapes and
   recorded the measured bound-site domain in each
   `handler.record.provenance.domain` — the same bound sites (same
   corpus, same pattern; only the handler budget differed).
   `mlopt_discovery.model_domains.DOMAIN_PROVENANCE` backfills that
   measured data verbatim, with the source cited in the module
   docstring.

2. **Attr-metavariable binding.**  The denotation specs carry attr
   metavariables (`{"dim": "a97"}`, `{"index": "a155"}`) — names,
   not values.  `spec_to_hnl` rendered them as *string literals*
   (`tensor.unsqueeze(v355, "a97")`) and `verify_model` had no way
   to bind them: the torch-side `_eval_spec` would have passed
   `dim="a97"` to torch, and the `.hnl` side had no prelude binding.
   `modelproj` now treats string attr-dict values as attr metavars
   end to end: `spec_attr_mvars` collects them,
   `spec_to_hnl` renders them as `.hnl` names, `hnl_program` binds
   `attr_bindings[name]` as literal prelude lines, and
   `verify_model` binds the reference spec through
   `prim_mint.spec_bind` before drawing — an attr metavar with no
   recorded binding refuses honestly (`unbound attr metavars`
   detail) instead of crashing or guessing.

   The `attr_bindings` are the v4 mint's representative-site freeze
   — `freeze_spec(spec, sites[0], …)` concretized the same spec's
   attr slots (`a97→1`, `a155→7`, `a75→1`, …), the convention every
   shipped minted handler verifies under.  Per-site attr variation
   was never persisted (`prim_domain` strips `$attr:` keys), so the
   gate measures the recorded binding — the same bar the minted
   handlers cleared, now recorded in each verdict.

## Changes

- `packages/mlopt-discovery/src/mlopt_discovery/modelproj.py` —
  `spec_attr_mvars`; `spec_to_hnl`/`_emit_spec`/`_spec_call_to_hnl`/
  `_attr_op_to_hnl` thread the attr-metavar set (`_attr_slot`
  renders mvar→name, else literal — `concat`'s `dim` included);
  `hnl_program` binds `attr_mvars`/`attr_bindings` as `name :: <lit>`
  prelude lines; `verify_model` takes `attr_bindings`, `spec_bind`s
  the ref spec, records `attr_bindings` in the verdict, refuses
  unbound metavars loudly.
- `packages/mlopt-discovery/src/mlopt_discovery/model_domains.py`
  (new) — `DOMAIN_PROVENANCE`: the five ops' measured domain +
  attr freeze, sourced from `results_overnight_prim_v4.json`.
- `scripts/verify_vocab_models.py` — `_domains()` now returns
  `{domain, attr_bindings}` per op (prim `provenance.domain` first,
  `model_domains` backfill via `setdefault`); `collect_verdicts`
  passes `attr_bindings` through; docstring updated.
- `vocab/models.yaml` — regenerated (`scripts/emit_vocab_fragments.
  py`): the five `hnl` projections now spell attr metavars as names
  (`a97`, `a155`, `a75`, …); verdicts rewritten by the verify pass.
  `emit_vocab_fragments.py --check` → `vocabulary fresh`.
- `tests/test_modelproj.py` — +5 tests: attr-mvar collection,
  name-vs-literal rendering, program binding, the both-sides bind
  gate, and the unbound refusal.

## Verified

- `scripts/verify_vocab_models.py` (real `hnl-run`,
  `MLOPT_HNL_RUN=.ci-target/graphopt/debug/hnl-run`, rebuilt after
  host-tensor `85c68d3`): **109 handl-shipped / 0 python-shipped**,
  worst `max_rel` `3.2e-13`.
- `pytest tests/test_modelproj.py tests/test_vocabdir.py` — 54
  passed, 1 skipped (`pytest-capped.sh`).
- `pytest tests/test_vocab_model_parity.py` (e2e, real `hnl-run`) —
  2 passed.
- `ruff check`/`ruff format --check`/`ty check` clean on touched
  files; `lint-imports` contracts kept; `vulture` clean.

## Honest residuals

- **Attr coverage is the representative site.**  Attr metavars
  took other values at other bound sites (that is why they are
  metavars); only `sites[0]`'s freeze survives in any record.  The
  gate therefore proves parity at the recorded binding, not over
  the attr's full range — identical to the shipped minted-handler
  convention.  A future mint that persists per-site `$attr:` values
  (`prim_domain` currently strips them) could widen this honestly.
- **`hnl_program` attr bindings are literals.**  Attr values that
  are themselves computed (`{"hnl": …}` feeds) are not supported —
  no current model needs them.
- The `python-shipped` verdict class is still reachable and honest:
  unbound attr metavars, missing domains, unsupported ops, and
  numeric misses all still refuse with `detail` recorded.
