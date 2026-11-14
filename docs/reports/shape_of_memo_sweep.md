# `_shape_of` memo sweep — upstream latent-pattern cleanup

Follow-up to `decoder_search_scaling.md`.  That report fixed the
measured decoder-search wall: carrier `check`/`derive` hooks in
`xcarrier.py` / `trace.py` called `mlopt_core.typing._shape_of`
**without a memo** on bound terms that resolve to heavily-shared
`EGraph._min_term` DAGs — an exponential tree walk (~6·10⁸ paths on
the whisper decoder IR).  Commit `746ab75` added module-level
`_SHAPE_MEMO` dicts there.

The same latent pattern existed upstream and in the lowerers —
survived the decoder graph only because those call sites see
shallower terms or run once per lowering.  This sweep threads a
content-keyed `memo` through every remaining unmemoised
`_shape_of`/`shape_of` call in scope.  **Pure caching — no semantic
change**: `typing._shape_of` is deterministic per term and already
accepts `memo`; the memo is keyed on the (interned, content-hashed)
term itself, the same contract as `mlopt_core.laws.base._SHAPE_MEMO`.

## eqsat (`handl-lang/eqsat`)

### `packages/mlopt-core/src/mlopt_core/laws/pairing.py`

The non-local diagram passes (`_pair_shared_input`, `_wshape`,
`_select_tile`, `_uniform_input`, `_gates_uniform`, `_cover_check`)
each did a function-local
`from mlopt_core.typing import _shape_of as _so` and called `_so(t)`
unmemoised — 6 import sites, ~8 call sites.  These resolve
`eg._any_term_cached(...)` terms (interned Op DAGs) once per
candidate weight/input, so a deep shared subgraph is re-walked per
cluster member.

Fix: the local imports now read
`from mlopt_core.laws.base import _shape_of as _so` — the existing
memoised wrapper (`_so(t, _SHAPE_MEMO)`, `base.py:73-80`).  Call
sites unchanged.  This also makes the pre-existing comment at
`pairing.py:180` ("the content-keyed ``_SHAPE_MEMO`` in ``_so``")
actually true.  `laws.base` imports only `egraph.Rewrite`, so the
function-local import keeps the module's lazy-import discipline; no
cycle (`laws/__init__.py` already imports `base` before `pairing`).

Already memoised, untouched: `base.py` (`_SHAPE_MEMO` + wrapper —
the precedent), `cond.py` (own `_MEMO` threaded through `_shape`/
`_vshape`), and `attention.py` / `headshare.py` / `layout.py` /
`priorart.py` (all import `_shape_of` **from `laws.base`** — the
memoised wrapper).  `specials.py`, `tensor.py`, `scan.py`,
`factored.py`, `minted.py`, `ruleset.py`, `store.py`, `serialize.py`
make no `typing._shape_of` calls (`specials` reads tensor `.shape`
attributes directly).

### `packages/mlopt-orchestrator/src/mlopt_orchestrator/regime.py`

`footprint_cost` (the bounded-memory regime's cost fn, registered in
`EXECUTORS`) called `shape_of(term)` unmemoised at two sites while
recursing the term DAG.  Added module-level `_SHAPE_MEMO` and passed
it at both sites (`shape_of(term, _SHAPE_MEMO)`).

### `packages/mlopt-orchestrator/src/mlopt_orchestrator/morphisms/signature.py`

`block_signature` already threads a per-call `memo` through
`_weight_ref`/`_table_refs`/`shape_of(root, memo)`, but `_norm_nodes`
called `shape_of(gain)` unmemoised inside its `_iter_ops` walk.
`_norm_nodes` gained an optional `memo: dict | None = None` parameter
(passed through to `shape_of`); `block_signature` now calls
`_norm_nodes(root, memo)`.  The re-export in `morphisms/__init__.py`
and the one-arg test callers are unaffected (`memo=None` → unmemoised,
same as before).

### `packages/mlopt-orchestrator/src/mlopt_orchestrator/morphisms_kv.py`

`_kv_numbers` called `shape_of(data)` unmemoised once per graft
stats computation.  Added module-level `_SHAPE_MEMO`; now
`shape_of(data, _SHAPE_MEMO)`.

## graphopt (`handl-lang/graphopt`, `packages/mlopt-carriers`)

Each lowerer gained a module-level `_SHAPE_MEMO: dict = {}` (same
comment/pattern as `xcarrier.py:178` / `trace.py:189`) and every
`_shape_of`/`shape_of` call site now passes it.  These run once per
`build_*_plan`/module construction, but the operands are extracted
terms — shared-DAG interned objects — so the exponential applied the
moment a decoder-scale term reached a lowerer.

- **`om_lower.py`** — 7 sites: `_slice_index` (chunk base shape),
  `_sliced_gather` (base shape), `_qk_parts` (k shape),
  `_analyze_elem_group` (k-base shape + per-k shapes),
  `build_om_plan` (per-leaf `om_elem` s/v shapes).  `decayed_lower`
  imports `_slice_index`/`_sliced_gather` from here, so it inherits
  the memoised helpers.
- **`omd_lower.py`** — `_leaf_sig`: `aff_diag`/`aff` leaf a/b shapes.
- **`decayed_lower.py`** — `_uniform_shapes`: intra score/chunk
  shapes + q/inner uniformity checks.
- **`scan_lower.py`** — `_leaf_family_ok`-style uniformity checks
  (`shape_of` on leaf a/b args, 5 sites) and `build_scan_plan`'s
  `b0s` column-state probe.
- **`trace_lift.py`** — `_shape_of_class` already memoised per
  canonical e-class id (`self._shapes`), but each miss called
  `_shape_of(eg.any_term(c))` unmemoised — the any_term resolution
  is itself a shared DAG.  Now `_shape_of(t, _SHAPE_MEMO)`; the
  per-class memo dedupes across plans, `_SHAPE_MEMO` dedupes within
  each class's term walk.

Already memoised, untouched: `xcarrier.py`, `trace.py` (the `746ab75`
fix), `trace_lift._shape_of_class`'s per-class dict (kept — it keys
on e-class id, a different domain).

## Out-of-scope observation

`mlopt_core.cost.backend.CostModel.__call__`
(`packages/mlopt-core/src/mlopt_core/cost/backend.py:39-63`) calls
`_infer_op_shape(term)` and `_shape_of(a)` unmemoised, and its own
`self(arg)` recursion carries no memo at all — the same latent
exponential, in `cost/` rather than `laws/`.  Left untouched (outside
the stated scope; the recursion-without-memo issue is a separate,
larger design question for the cost model).  Worth a follow-up look
if decoder-scale extraction ever routes through `CostModel`.

## Measurement

Synthetic shared-DAG shape query — `add(t, t)` chained 24 deep
(2²⁴ ≈ 1.7·10⁷ tree paths over 25 interned nodes),
`Var("x", (8,))` base:

| call | time |
|---|---|
| `_shape_of(t)` (unmemoised) | **30.6 s** |
| `_shape_of(t, {})` (memoised) | **~0.000 s** |

Identical result `(8,)` — caching only.  At decoder depth the
unmemoised walk is what produced the >20 min wall; the memoised path
is linear in distinct nodes.

## Verification

- eqsat fast lane: `pytest tests/ -q -m "not slow"` → **1656 passed,
  15 skipped** (16.4 s).
- graphopt carriers slice: 641 tests across the om/scan/decayed/
  trace/xcarrier files → **all pass** except
  `test_scan_fused.py::test_fused_dag_shared_leaf_{diag,dense}`,
  which **fail identically on unmodified HEAD** (verified by
  `git checkout` of `scan_lower.py` + re-run) — pre-existing,
  unrelated to this change.
- `ruff check` + `ruff format --check` clean on all touched files in
  both repos; `ty check` clean on all touched files.

No new tests: the change is perf-only caching; every touched call
site already had coverage through the existing carrier/law suites.
