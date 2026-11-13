# T-029 — the 6 executor-confounded `pay_measured` rows: real, mechanism found, fixed

**Agenda item:** `agenda-paymeasured_confound` (score 0.8286, rank 1) —
"6/14 pay_measured rows are executor-confounded — the marginal measures
routing, not the fold."
**Date:** 2026-11-13 · **Verdict: CONFIRMED, root-caused, and FIXED** —
the confound was a measurement bug; the fix is a pinned-executor arm,
and the re-run resolves all 14 cells (13 lose · 1 tie · **0
confounded · 0 wins**).

## The 6 confounded rows

From `docs/reports/pay_measured.json` (2026-10-25 run), the cells whose
delivered `synop` and `noprim` arms routed to different executors:

| module | size | synop executor | noprim executor | marginal (as delivered) |
|---|---|---|---|---|
| GatedDeltaRule | small | `IRModule` | `BatchedScanModule` | 0.35 |
| GatedDeltaRule | large | `IRModule` | `BatchedScanModule` | 0.20 |
| HighwayGate | small | `IRModule` | `BatchedScanModule` | 1.16 |
| HighwayGate | large | `IRModule` | `BatchedScanModule` | 1.14 |
| GRUFusedGate | small | `IRModule` | `BatchedScanModule` | 1.07 |
| GRUFusedGate | large | `IRModule` | `BatchedScanModule` | 1.07 |

(3 modules × 2 sizes; the other 8 cells already ran both arms on
`IRModule`.)

## Mechanism — which code path routes differently

Executor routing lives in `mlopt_orchestrator.optimize._lower_extracted`
(eqsat `packages/mlopt-orchestrator/src/mlopt_orchestrator/optimize.py:460`):
the extracted term goes to the first carrier `ExecutorSpec` in
`TorchSink.executors` whose `accepts` probe holds, else `sink.lower`
(the generic `IRModule`). The `scan` spec
(`graphopt/packages/mlopt-torch/src/mlopt_torch/adapters.py:212`)
accepts via `mlopt_carriers.scan_lower.is_scan_apply_term`
(`packages/mlopt-carriers/src/mlopt_carriers/scan_lower.py:200`), which
requires an `apply`/`applyd` root whose map argument is a **pure**
`aff`/`aff_diag` compose tree (`_is_aff_tree`, same file:164).

- The `noprim` ruleset (shipped `default_rules()` minus `prim:synop*`)
  extracts a term whose scan spine stays a pure affine tree → routes to
  `BatchedScanModule` (the level-batched carrier executor).
- The `synop` ruleset folds subgraphs into `synop_*`/`synop2_*`/
  `synop3_*` ops. Those ops are not carrier vocabulary, so the map tree
  is no longer pure → `is_scan_apply_term` declines → `IRModule`.

So `synop/noprim` compared **two different executors**, not the fold.
Worse, the direction of the artifact is misleading: on
GatedDeltaRule the delivered `noprim`→`BatchedScanModule` arm was
~3–6× *slower* than the same term on `IRModule` (0.42 vs 0.12 ms
small; 0.92 vs 0.28 ms large in the 2026-11-13 run), which is what
manufactured the only sub-1.0 "marginal" (0.35/0.20) in the table —
the fold appeared to "deliver" by *preventing* a bad carrier routing,
not by saving dispatches. `prim_delivery.md` finding 3 already flagged
this as "a routing bug-shaped result on its own"; it remains open as a
carrier-executor question, separate from this measurement fix.

A second, quieter measurement bug compounded it:
`bench/referees/pay_wall.py::_store_rows` wrote **every** hit's
`marginal` into the minted store as `metric="pay_wall"` — including
the six confounded cells, whose ratios measure executor choice. Fold
*verdicts* already excluded confounded hits, but the stored
measurements did not.

## The fix (implemented)

**Pinned-executor arm.** `prim_delivery_probe.measure_module` now
builds a fifth timed arm, `noprim_ir`: the *same* fold-free extracted
term re-lowered through `sink.lower` — the generic `IRModule` path —
bypassing `_lower_extracted`'s carrier routing
(`bench/drivers/probes/prim_delivery_probe.py:355-371`). One `TorchSink`
instance is shared by both `lower()` calls and the pinned lowering, so
the generic path (op table included) is identical.

**Referee.** `pay_wall.measure` now reports three ratios
(`bench/referees/pay_wall.py:94-129`):

- `marginal = synop/noprim` — as delivered, routing-inclusive (kept for
  transparency);
- `marginal_pinned = synop/noprim_ir` — the fold on a fixed executor;
- `pay` — the verdict basis: `marginal_pinned` when the delivered synop
  arm runs on the same executor class as `noprim_ir` (recorded in
  `executors`), else `marginal` when the delivered arms already share
  an executor, else `None`.

A cell is `confounded` only when the delivered arms route differently
AND no valid pinned comparison exists — with the new arm that set is
empty. `_store_rows` now stores `pay` (never a confounded marginal) and
skips confounded/unmeasured hits; fold aggregation aggregates `pay`.

## Re-run result (`--iters 100 --warmup 30`, 2026-11-13)

`docs/reports/pay_measured.json` regenerated. All 14 cells resolve:

- **13 lose · 1 ties (GatedLinearAttention@large, pay 1.04) · 0
  confounded · 0 delivers.** Worst fold marginal 2.93×.
- The six formerly-confounded cells all resolve to **loses**:
  GatedDeltaRule pay 1.23/1.08 (was "delivers" 0.35/0.20 as-delivered),
  HighwayGate 1.33/1.30, GRUFusedGate 1.14/1.13.
- `executor_confounded` remains `true` on those rows as a flag — the
  *delivered* routing still differs — but `pay`/`verdict` now rest on
  the pinned comparison.

The headline is unchanged and now stronger: the family's pay claim is
false on every measurable cell, and the one apparent counter-example
was an executor artifact.

## Files changed

- `bench/drivers/probes/prim_delivery_probe.py` — `noprim_ir` arm,
  shared `TorchSink`, docstring.
- `bench/referees/pay_wall.py` — `pay`/`marginal_pinned`/
  `executor_confounded` fields, verdict + store + summary on `pay`.
- `tests/test_prim_delivery_probe.py` — arm-set pins updated.
- `docs/reports/pay_measured.{md,json}`, `pay_coverage.md` —
  regenerated/updated (also picked up bank growth 57→58 shipped folds,
  48→49 unmeasured).
- `docs/EVIDENCE.md`, `README.md` — stale cell counts corrected.

## Verification

- `scripts/pytest-capped.sh .venv/bin/python -m pytest
  tests/test_prim_delivery_probe.py -q` → **4 passed**.
- `ruff check` + `ruff format --check` on all touched `.py` → clean.
- Full `pay_wall` re-run completed and wrote `pay_measured.json`.

## Residual notes for the parent agent

- `BatchedScanModule` being ~3.5× slower than `IRModule` on the
  GatedDeltaRule fold-free term is a **separate suspected routing bug**
  (flagged in `prim_delivery.md` finding 3). Not fixed here — it is a
  carrier-executor performance question, not a measurement confound.
- `bench/drivers/probes/prim_kernel_module.py` consumes
  `measure_module` and reports `synop/noprim`; it inherits the extra
  timed arm harmlessly but its own marginal is still the
  routing-inclusive one. If that probe's numbers are ever quoted as
  fold-attribution, it should switch to `noprim_ir` the same way.
- `agenda_miner`'s `paymeasured_confound` signal will now read 0
  confounded rows on the next agenda mine — correct, the item is
  resolved.
- Changes are uncommitted, per instructions.
