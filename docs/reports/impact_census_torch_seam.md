# impact/census torch-seam — `impact.py` + `census.py` import-clean

**Date:** 2026-11-14 · **Plan:** 0034 phase-0 audit follow-up (the
`discoverykit` extraction) · **Scope:** `impact.py` and `census.py` —
the two modules the proposal-seam report flagged as "still
torch-bound at import".  Follows the `proposal_core` pattern:
lazy torch binds + injected backends, no shim needed (no code
moved — the whole module surface stayed put).

## What was torch-bound

`mlopt_discovery.impact` bound at module level:

- `import torch` (line 48) and `_DEV = torch.device("cpu")`
  (line 87 — a module-level constant evaluated at import);
- `from mlopt_torch.adapters import TorchSink, TorchSource`
  (lines 64–65);
- `class _LSTMSeq(torch.nn.Module)` — a module-level class whose
  base class binds torch at import.

`mlopt_discovery.census` had **no direct torch surface** — its only
non-stdlib/`mlopt_core` import is
`from mlopt_discovery.impact import _bench_cases, model_cases`, so
it inherits impact's binding transitively (the same shape `vocab`
had before the proposal pass).

## What changed (`impact.py`)

- Module-level `import torch` and the `mlopt_torch.adapters` import
  removed; `TorchSink` is now a `TYPE_CHECKING`-only import so the
  `sink: TorchSink` annotations on `_cost_fn` / `_probe` /
  `reach_row` still resolve for `ty` (runtime annotations are lazy
  under `from __future__ import annotations`).
- `_DEV` deleted; `_bench_cases` does `import torch` lazily and
  passes `torch.device("cpu")` to the builders.
- `_LSTMSeq` moved **inside** `_model_cases` — the only consumer —
  so its `torch.nn.Module` base never binds at import.
- `_model_cases` gained `import torch` + the `mlopt_torch.models`
  imports lazily (they were already function-local for the models;
  `import torch` joined them).
- `_synthetic_case` gained a lazy `import torch` (the fp64 feed
  generator).
- New lazy backend factories, mirroring `proposal_core._eval_backend`:
  - `_torch_source()` → `TorchSource()` (lazy `mlopt_torch.adapters`)
  - `_torch_sink()` → `TorchSink()` (lazy `mlopt_torch.adapters`)
- **Injected-backend API** (the `evaluate(numeric=...)` analogue):
  - `model_cases(source=None)` — *source* is anything with
    `to_ir(model, x)`; default is the lazily-bound `TorchSource`.
    (Note: `_model_cases()` itself still builds the torch model
    zoo, so the corpus path needs torch regardless — the injection
    point is the exporter, matching how `TorchSource` was already a
    parameter-shaped collaborator.)
  - `run_impact(*, sink=None)` — *sink* is anything with
    `supported_ops` + `verify(before, after, feed, rtol=...)`; the
    default is the lazily-bound `TorchSink`.  Threaded through the
    existing `_probe`/`reach_row`/`_cost_fn` parameters unchanged.
- `census.py`: **docstring only** — no code change needed; the
  impact import now resolves torch-free.

Everything else stayed: `TermCase`, `Firing`, `new_laws`,
`_candidate_law`, `_params_of`, `_ir_of`, `_iter_subterms`,
`_relax`, `_relaxed_census`, `_rule_budgets`, `_saturate`,
`_cert_ok`, `reach_row`, all renderers, `_dump_json`, `main`, and
the whole `census.py` surface (`CorpusTerm`, `corpus`,
`term_expansion`/`op_expansion`, `shape_key`, `shape_repr`, the
three census passes, `run_census`, `main`).  No public or private
name moved, so no shim was required — every consumer binding
(`impact.TermCase`, `impact._cost_fn`, `impact._iter_subterms`,
`impact._model_cases`, `impact._bench_cases`, `census._iter_subterms`,
`census.shape_key`, …) resolves identically.

## Honest call-time behaviour

Without torch, the torch-touching paths raise `ImportError` at call
time (verified in a `sys.modules['torch'] = None` +
`sys.modules['mlopt_torch'] = None` subprocess):

- `im._bench_cases()` → `ImportError: import of torch halted`
- `im.model_cases()` → `ImportError` (`mlopt_torch.adapters`)
- `im.synthetic_cases()` → `ImportError: import of torch halted`
- `cs.corpus()` / `cs.run_census()` / `cs.main()` → `ImportError`
  via the impact corpus builders

The pure machinery — `new_laws()`, `TermCase`, `_iter_subterms`,
`_relaxed_census`, `_saturate`, `_cert_ok`, `shape_key`,
`op_tuple_census`, `shape_census`, `sharing_census`, the renderers —
runs on caller-supplied terms with no torch at all.

## Import-freedom proof

```
import sys
sys.modules['torch'] = None
sys.modules['mlopt_torch'] = None
import mlopt_discovery.impact   # OK
import mlopt_discovery.census   # OK
```

Pinned by `test_impact_census_modules_import_without_torch` in
`tests/test_discovery_census.py` (same subprocess pattern as
`test_proposal_modules_import_without_torch`).

## Downstream consumers checked

- `intake/__init__.py` / `intake/ledger.py` — import
  `TermCase`/`_ir_of`/`_iter_subterms`/`_bench_cases`/`model_cases`
  from impact; all names unchanged (intake itself stays torch-bound
  by design — it *is* the torch exporter).
- `model_corpus.py`, `coherence2.py`, `vocab.py` — reach-through
  names (`TermCase`, `_ir_of`, `_iter_subterms`, census helpers)
  unchanged; all import cleanly.
- `kernelopt` and the ~20 test files importing
  `TermCase`/`_cost_fn`/`census.*` — untouched surface.

## Test results

- `pytest tests/test_discovery_impact.py tests/test_discovery_census.py -q -m "not slow"`: **42 passed** (incl. the new import-freedom test).
- `ruff check` + `ruff format --check` on all touched files: clean
  (two lazy-import ordering nits auto-fixed).
- `ty check` (repo-wide): 0 errors.
- Consumer import smoke: `model_corpus`, `coherence2`,
  `intake.ledger` import OK.

## Still torch-bound at import (expected, unchanged)

`shape_proposal`/`pipeline`/`enumerate`/`forms`/`grammar`/
`model_corpus`/`workload_gen`/`zoo`/`intake`/`meta_game`/`gap_gen`/
`play`-arena arms — genuinely torch-shaped per plan 0034; they stay
in graphopt.  `impact`/`census` no longer block them *transitively*
through the impact import, but each still binds torch directly.

## Notes for the parent agent

- No shim was needed — unlike `proposal`, nothing moved modules, so
  `impact.py`/`census.py` keep their full surface in place.  If
  `discoverykit` later wants the *pure* census machinery without the
  corpus builders, the split point is `corpus()`/`_bench_cases`/
  `model_cases`/`_model_cases`/`_synthetic_case` (torch) vs. the rest
  (torch-free) — the injected `source`/`sink` params are the seam.
- `run_impact`'s new `sink` kwarg is keyword-only; `main()` calls
  `run_impact()` unchanged.
- The `mini_result` fixture in `test_discovery_impact.py` still
  monkeypatches `_bench_cases`/`model_cases`/`synthetic_cases` and
  calls `run_impact()` — the lazy `TorchSink` default binds inside
  the test env as before.
