# discoverykit Phase 1 — the domain-free mining core extracted

**Date:** 2026-11-19 · **Plan:** 0034 (`project/plans/pl-0030`) ·
**Scope:** Phase 1 (repo + package + shims).  Phase 2 (git-tag pin +
coverage-ratchet split) and Phase 3 (second-domain seat) are *not*
done — see "What's left".

## Headline decisions

### Package name: `discoverykit`, not `mlopt_discovery`

The plan floated keeping the `mlopt_discovery` import name inside the
new repo "so shims are trivial".  That is **unworkable for a split
package**: two distributions cannot share one regular package —
whichever `mlopt_discovery/__init__.py` wins `sys.path` order shadows
the other's modules entirely (PEP 420 namespace merging requires *no*
`__init__.py` on either side, and this package has one).  Since the
move set is a strict subset (~30 of ~90 modules; `census`, `vocab`,
`intake`, the torch play arms stay), the honest name is
`discoverykit` with modules mirroring the old layout
(`discoverykit.antiunify`, `discoverykit.lawdata.laws.banks`, …).
Every old `mlopt_discovery.<mod>` path remains as a `sys.modules`
shim, so the rename is invisible to consumers — the same trick the
plan-0032 extraction used, one level down.  The package rename
taxonomy (plan 0038) is unaffected.

### Move set — module-level-closed subset

Moved (29 top-level modules + 3 subpackages):

- proposers: `antiunify`, `combinator`, `form_propose`,
  `index_abstract`, `loop_propose`, `object_synthesis`,
  `handler_synth`, `dspec_synth`
- game core: `meta_arena`, `meta_player`, `meta_prior`,
  `macro_moves`, `bilinear_game`, `cellstore`, `pay_gate`,
  `bounds`, `model_domains`, `domain_spec`, `corpus_cov`
- verify/evidence: `verifier`, `coherence`, `coherence2`,
  `lemma_cert`, `law_order`, `evidence`, `machine_pack`,
  `proposal_core`, `genkernel`, `handl`
- data: `lawdata/` (whole package), `domains/*.json`, `priors/*.json`
- seam: `play/registry.py` → `discoverykit.play.registry`

Selection rule: a module moved only if its **module-level**
`mlopt_discovery` imports closed over the move set (verified by AST
over every import node, function-level included).  Lazy
(function-level) imports of *stay* modules keep the
`mlopt_discovery.<stay>` spelling and resolve through the consumer's
graphopt install — the established lazy-bind seam contract.

### Deferred — and why

- `fragment_game`, `handl_egraph`, `handl_engine` — module-level
  `from mlopt_discovery import vocabdir`; `vocabdir` stays (plan-0033
  single-source YAML + lazy carrier/torch census paths).
- `play/` domain arms (`cli`, `domains`, `gen`, `search`, `sites`,
  `torch_arm`, `fragment`, `meta_joint`, `players`, `probes`,
  `__init__`, `__main__`) — the `DOMAINS` arms are domain-bound by
  construction; only `registry.py` (the frozen kernelopt seam) moved.
  `play.sites`/`play.gen` collectors stay with the arms that use
  them — `_mint_tiers` lives in `play/gen.py`, which stays, so no
  shim was needed for the collector family.
- `proposal.py` — already a compat shim over `proposal_core`; left
  untouched (it re-exports through the `proposal_core` shim).
- everything else on the task's STAYS list (`census`, `impact`,
  `oracle`, `pipeline`, `shape_proposal`, `vocab`, `vocabdir`,
  `propose`, `enumerate`, `forms`, `grammar`, `emit`, `zoo`,
  `intake/*`, `workload_gen`, `model_corpus`, `meta_game`,
  `player_arena`, `arena*`, `open_*`, `prim_mint`,
  `shipped_primitives`, `state_reindex`, `training`, `modelproj`,
  `families`, `family_synth`, `learned_propose`, `gap_gen`,
  `scoring`/`players`/`engine`/`advantage` gamekit shims).

## Shim mechanism

Each moved module is a ~20-line file:

```python
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("discoverykit.<mod>", run_name="__main__")
else:
    from discoverykit import <mod> as _moved
    _sys.modules[__name__] = _moved
```

`sys.modules` swap (not `import *`) means `mlopt_discovery.<mod>`
*is* `discoverykit.<mod>` — attribute access, `is` comparisons and
`__module__` strings all behave as before; the `__main__` guard keeps
`python -m mlopt_discovery.<mod>` working (verified on
`verifier --help`, `proposal --help`, `bounds --help`).

`lawdata/` needed a custom `__init__.py`: it swaps the package and
**pre-registers** `mlopt_discovery.lawdata.laws[.{banks,handlers,
macros,moves,terms}]` and `.game[.features/.weights]` in `sys.modules`
so `import mlopt_discovery.lawdata.laws.banks` resolves to the single
discoverykit/gamekit objects (the plan-0032 one-table-one-home seam,
now across repos).  `tests/test_lawdata.py::TestLawdataSplit` passes
unchanged.

`play/registry.py` shims the same way — `DOMAINS` stays the one live
dict; kernelopt's `DOMAINS["kernel"] = …` mutation is unaffected
(verified: `play.DOMAINS` populates all 9 graphopt domains through
the shim).

## Dependency wiring

- `discoverykit` package deps: `mlopt-core` + `gamekit` +
  `numpy` (git+tag `v0.1.0`, same eqsat/gamekit pins graphopt
  uses); `mlopt-orchestrator` is an **optional extra**
  (`machine_pack`'s composed-defaults probe is the only binder,
  already lazy).
- graphopt: `mlopt-discovery` declares `discoverykit`; root
  `pyproject.toml` `[tool.uv.sources]` has
  `discoverykit = { path = "../discoverykit/packages/discoverykit",
  editable = true }`; `uv.lock` refreshed.
- **Push blocker (Phase 2):** the path dep is machine-local.  Before
  graphopt is pushed anywhere, discoverykit needs a GitHub remote +
  tag and this source must flip to `git = …, tag = "v0.1.0"` —
  otherwise downstream git consumers (kernelopt pins mlopt-discovery
  by git) cannot resolve `discoverykit`.  Local-only per the task
  constraints; nothing was committed or pushed.

## Code changes inside moved modules (minimal, documented)

- Import rewrites only: `mlopt_discovery.<moved>` →
  `discoverykit.<moved>` at every import node (AST-verified); stay
  modules keep `mlopt_discovery.*`.
- `evidence.py` / `verifier.py`: `REPO_ROOT` → new
  `discoverykit.consumer_repo_root()` — resolves the *consumer* repo
  (graphopt's `REPO_ROOT` via `mlopt_discovery`) when installed,
  falling back to discoverykit's own root.  Preserves `code_rev`'s
  bench/corpus hashing and `verifier._law_cases`' `bench` lookup in
  graphopt; honest fallback standalone.
- `verifier._law_cases`: `bench` import now `try/except ImportError
  → {}` — `instance_of` degrades to `generic_instance` when the
  consumer repo is absent (previously `ModuleNotFoundError`).
  No behavior change where `bench` exists.
- `domains/` + `priors/` JSON moved with `domain_spec` (it reads
  them via `Path(__file__).parent`).

## Test results

- **discoverykit suite** (`uv sync --group dev`): **150 passed,
  1 skipped** (torch `importorskip`).  Files: `test_smoke.py` (every
  module imported with torch + `mlopt_discovery` poisoned in
  `sys.modules` — the dynamic form of the import-linter contract),
  `test_lawdata_split.py`, plus the package-bound subset copied from
  graphopt (`test_antiunify`, `test_discovery_handl`,
  `test_index_abstract`, `test_loop_propose`).
- **graphopt gate**: `pytest tests/test_discovery_play.py
  tests/test_discovery_bounds.py tests/test_discovery_model_corpus.py
  -q -m "not slow"` → **168 passed**.
- **Shim-fidelity sweep** (graphopt, `-m "not slow"`):
  `test_lawdata`, `test_domain_spec`, `test_index_abstract`,
  `test_loop_propose`, `test_antiunify`, `test_discovery_verifier`,
  `test_discovery_coherence`, `test_discovery_coherence2`,
  `test_lemma_certificates`, `test_discovery_cellstore`,
  `test_discovery_macro_moves`, `test_discovery_handl`,
  `test_discovery_corpus_cov`, `test_discovery_meta_prior` → all
  green **except 3 pre-existing failures** in
  `test_discovery_coherence.py` (`test_probe_right_factor_linear_
  enumerated`, `test_run_cert_saturation_only_under_tight_fuel`,
  `test_lemma_cert_main_probe_note`) — reproduced identically on a
  clean `HEAD` worktree, unrelated to this change.
  `test_discovery_handler_synth.py::TestProposal::
  test_ew_replay_admits_broadcast_views` stalls >200 s on clean HEAD
  too — pre-existing, not a regression.
- **discoverykit gates**: `ruff check`/`ruff format --check` clean;
  `ty check` 0 errors (120 `import-not-found` *warnings* — the lazy
  torch/`mlopt_torch`/`mlopt_discovery`/`bench` binds, downgraded via
  `[tool.ty.rules]` since ty cannot see function-level intent);
  `vulture` clean; no import-linter contract (grimp cannot
  distinguish lazy binds — `test_smoke.py` enforces the contract
  dynamically instead).

## Known seam debts (for Phase 2/3)

- Moved modules' stay-side lazy imports (`mlopt_discovery.census`,
  `.oracle`, `.pipeline`, `.shape_proposal`, `.vocab`, `.intake`,
  `.prim_mint`, `.zoo`, `.play` domain arms, `.proposal` shim) mean
  discoverykit is **not standalone** for those call paths — by
  design for Phase 1; the second-domain seat (Phase 3) is where the
  honest split of those paths gets decided.
- `meta_arena`'s spec-cap path lazy-imports
  `mlopt_discovery.census.op_expansion` — census stays per the task
  boundary; relocating `op_expansion` (or census itself) is a Phase
  2/3 question.
- `evidence.code_rev` still versions a *single* repo root
  (consumer-preferred).  Post-split the verification code lives in
  two repos — a composite rev (discoverykit + consumer) is the
  honest Phase-2 fix; until then a discoverykit-side edit does not
  by itself invalidate the shared `evidence.db` cache.
- `genkernel`/`handler_synth` moved wholesale: import-clean, but
  their `torch.compile`/Triton codegen is torch-semantic at call
  time (lazy binds decline honestly).  If Phase 3 wants a strictly
  torch-free artifact, the codegen half is the split candidate.
- Coverage floors not yet re-measured per repo (graphopt's
  `fail_under=100` on `mlopt_discovery` now covers shims + stay
  modules; discoverykit's is `fail_under = 0` pending a real
  measurement — Phase 2).

## What's left

- Phase 2: push discoverykit, tag `v0.1.0`, flip graphopt's source
  to git+tag, `uv lock` both repos, re-measure + set coverage floors.
- Phase 3: seat a second domain (kernelopt or lib-tensor) on
  discoverykit — the proof the extraction was real.
- Optional follow-ups: `census`/`impact`/`oracle`/`shape_proposal`/
  `vocab`/`pipeline` re-audit for a second move batch; `code_rev`
  composite; `play.sites` collector extraction (arch-audit §5).
