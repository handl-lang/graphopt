# Proposal torch-seam split — `proposal_core` + lazy torch binds

**Date:** 2026-11-14 · **Plan:** 0034 phase-0 audit follow-up (the
`discoverykit` extraction blocker) · **Scope:** `proposal.py` first;
`oracle.py` / `vocab.py` where the pattern repeated cleanly.

## The blocker

`mlopt_discovery.proposal` bound `import torch` and
`from kit_torch.learned_policy import LearnedPolicy, train_rule_value`
at **module level**, so every downstream consumer
(`enumerate`/`forms`/`grammar`/`shape_proposal`/`pipeline`/`open_*`/
`prim_mint`/play-arms — ~15 modules, plus `kernelopt`'s
`proposal.Candidate` import) was transitively torch-bound.  The
phase-0 audit flagged `proposal.py:76,93`, `oracle.py:69`,
`vocab.py:77` as the top extraction blocker.

## What moved

**New module `mlopt_discovery.proposal_core`** — the entire
candidate-proposal machinery, import-clean without torch:

- records: `Candidate`, `Outcome`
- structural key / library relation: `_key`, `_library_keys`,
  `_relation`
- leaf/term walks: `_leaves`, `_size`, `_tree_dist`
- generators: `near_miss_candidates`, `composite_candidates`,
  `schema_candidates`, `seed_terms`, `_spec_term`, `_v`
- usefulness: `_sat_cost`, `_rule_budgets`, `_cost_delta`
- ranking: `rank_pool`, `_learned_scores`, `_train_scorer`,
  `_pool_features`, `_heuristic_score`, `_average_precision`,
  `_rank_on`
- evaluation + guarded second chance: `evaluate`, `propose_guarded`,
  `_abstract`, `_probe_rule`, `_guard_sites`, `_eval_guarded`
- reporting/CLI: `yield_table`, `_fmt_*`, `_outcome_row`,
  `_dump_json`, `main`

**`mlopt_discovery.proposal` is now a compatibility shim** (the
repo's established pattern — `engine.py`/`players.py`/`scoring.py`
shims): every name the old module bound — public *and* private,
including the `lp._x` reach-throughs tests and siblings use
(`_numeric_true`, `_allclose`, `_eval_backend`, `_leaves`,
`_spec_term`, `_canon`, `_D`, `_v`, `_cost_delta`, `_rank_on`, …) —
is re-exported from `proposal_core` unchanged, plus a module
`__getattr__` reach-through for anything not explicitly named (e.g.
the live `_EVAL` singleton).  `kernelopt`'s
`from mlopt_discovery.proposal import Candidate` and every
`lp._foo` consumer resolve identically; `python -m
mlopt_discovery.proposal` still works.

## The injected-backend API

- `evaluate(..., numeric=None)` — new keyword: any
  `(lhs, rhs) -> bool | None` callable replaces the default
  `_numeric_true` (the torch fp64 oracle).  Threaded through
  `propose_guarded`'s internal `evaluate` call too.  A torch-free
  domain supplies its own truth oracle and `evaluate` runs without
  torch entirely.
- `rank_pool(..., scorer=None)` — pre-existing injection, now
  actually reachable without torch: an injected `candidate -> float`
  callable skips the `kit_torch.learned_policy` import entirely.
- `_eval_backend()` / `_numeric_true` / `_allclose` /
  `_train_scorer` / `_learned_scores` — torch/`kit_torch` imports
  moved **lazy inside the function** (the `genkernel.py`
  convention), so the module surface is import-clean; calling them
  without torch raises `ImportError` honestly at call time.

## oracle.py / vocab.py — same pattern, same treatment

The pattern repeated cleanly, so both were made import-clean too:

- `oracle.py`: module-level `import torch` removed; lazy `import
  torch` inside `_env_for`, `_as_tensor`, `_bcast_to`, `_feats_id`,
  `_feats_w`, `_features`, `verify_view_candidate` (the only
  torch-touching functions).  `_eval` already delegated to
  `lp._eval_backend` (lazy `kit_torch`).
- `vocab.py`: module-level `import torch`,
  `from mlopt_discovery.census import _iter_subterms`, and
  `from mlopt_discovery.impact import _bench_cases, model_cases`
  removed (census/impact are themselves torch-bound at module
  level); lazy binds inside `_rand`, `corpus_ops`,
  `_value_preserving`, `_is_view`, `_op_output`, `_is_reduction`,
  `classify`, `derive_vocabulary`.

## Import-freedom proof

Subprocess test with `sys.modules['torch'] = None` +
`sys.modules['kit_torch'] = None` (import-system-level block, not
a meta-path finder — verified to actually raise):

```
import mlopt_discovery.proposal      # OK
import mlopt_discovery.proposal_core # OK
import mlopt_discovery.oracle        # OK
import mlopt_discovery.vocab         # OK
```

Pinned by `test_proposal_modules_import_without_torch` in
`tests/test_discovery_proposal.py` (plus
`test_proposal_shim_reexports_core` and
`test_evaluate_numeric_injection`).

**Still torch-bound at import (expected, downstream):**
`shape_proposal`/`pipeline`/`enumerate`/`impact`/`census` — they
import `kit_torch.adapters`/`torch` directly or via `impact`.
Making them torch-free is the *next* seam, not this pass.

## Test results

- `pytest tests/test_discovery_proposal.py tests/test_discovery_bounds.py -q -m "not slow"`: **60 passed, 7 failed** — the 7 are the pre-existing cost-delta failures noted in the task (verified on HEAD~, not chased).  No new failures; the 3 added tests pass.
- `pytest tests/test_discovery_oracle.py tests/test_discovery_vocab.py tests/test_law_view_oracle.py -q -m "not slow"`: 158 passed, **1 failed** — `test_operand_shape_kinds` (`_operand_shape` no longer raises `TypeError` on unresolved attrs).  `_operand_shape` is untouched by this change; the failure is in `mlopt_core.typing._shape_of` behavior — almost certainly the concurrent traversal-agent/mlopt_core work, flagged for follow-up.
- `pytest tests/test_discovery_forms.py tests/test_learned_propose.py tests/test_vocabdir.py -q -m "not slow"`: 97 passed, 1 skipped, **1 failed** — `test_default_seed_bank_is_machine_shipped` (`MACHINE_SHIPPED` vs `_seed_bank` `synop*.expand` name mismatch — law-store state, unrelated to this change).
- `ruff check` + `ruff format --check` on all touched files: clean.
- `ty check`: clean (0 errors).

## Notes for the parent agent

- The working tree carries the **traversal agent's uncommitted
  refactor** (`mlopt_core.traversal` adoption across ~20
  `mlopt_discovery` modules, incl. `_term_expand`/`_term_leaves` in
  `proposal_core`/`oracle`).  This change was built on top of it;
  the two diffs are disjoint (traversal = walk replacements, mine =
  import placement + shim + `numeric` param).
- `proposal._EVAL` via the shim returns the *live* core singleton
  through `__getattr__` (not a stale snapshot).
- `vocab._eval`/`oracle._eval` still lazy-import `_eval_backend`
  via the `mlopt_discovery.proposal` path — resolves identically
  through the shim.
