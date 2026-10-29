---
title: Plan 0008 — Package split
type: plan
date: 2026-10-14
status_note: accepted
---

# Plan 0008 — Package split

Third of three — depends on **0006** (API seam) and **0007** (backend
abstraction). Each lands independently and green.

## Goal

Rename the (now torch-free) orchestrator to **`catopt_orchestrator`**,
reduce the **`catopt` façade** to a thin, torch-free re-export
namespace, retire the `optimize_*` wrappers, and migrate every call
site. After 0007 the orchestrator is already backend-neutral and
torch-free; this plan only moves names.

## Target layout

```
catopt               thin, torch-free façade — re-exports only
                     (no alias table, no torch)
catopt-core          abstractions + engine + profile
                     engine:  ir, egraph, laws, cost, meta, ops,
                              typing, attrs, rulecache
                     ports:   Source, Sink, Capabilities, Composer,
                              Meter, Runner, Criterion, Strategy, Binding
                     results: SearchResult, LowerResult, TimingResult
                     profile: TargetProfile + correction math
catopt-carriers      carrier laws / executors          (unchanged)
catopt-torch         torch mechanisms
                     TorchSource, TorchSink, TorchComposer, TorchMeter,
                     TorchBackend, TorchCompileRunner, calibrate, export,
                     torch_bridge, executors, models, report
catopt-cuda          CUDA mechanisms (created in 0007)
                     CudaGraphRunner
catopt-orchestrator  backend-neutral orchestration + generic impls
                     search, lower, Optimizer
                     Monolithic, Compositional, Autotuned
                     autotune, regime, discover
                     IdentityRunner, ChainedRunner, criteria
```

Dependency direction: `core` ← `carriers` ← `orchestrator`;
`torch` → `core` + `carriers`; `cuda` → `torch`. The orchestrator
imports no torch/cuda (established by 0007).

## The façade becomes thin and torch-free

`catopt/` keeps its namesake but loses everything else: the
`sys.modules` alias table goes, and it re-exports only the
**backend-neutral** public surface — **no torch**. So `import catopt`
works, stays lightweight, and never pulls torch in.

```python
# catopt/__init__.py — re-exports only
from catopt_core import Lifted, SearchResult, LowerResult, Capabilities
from catopt_orchestrator import (
    Optimizer, lift, search, lower, verify,
    Monolithic, Compositional, Autotuned, discover_alternatives,
)
```

Torch stays explicit (`from catopt_torch import TorchBackend`), and
historical alias paths — `catopt.egraph`, `catopt.cost`,
`catopt.optimize`, `catopt.criteria` — no longer resolve.

## Retirement & migration

* Remove `optimize_model` / `optimize_compositional` /
  `optimize_model_autotuned` (the one-line wrappers 0006 left).
* Migrate every call site (`tests/`, `bench/`, `demo.py`):

```
optimize_model(m, x, **kw)           ->  Optimizer(backend=TorchBackend())
                                             .optimize(m, x, **kw)
optimize_compositional(m, x, **kw)   ->  Optimizer(backend=TorchBackend())
                                             .optimize(m, x,
                                               strategy=Compositional(), **kw)
optimize_model_autotuned(m, x, **kw) ->  Optimizer(backend=TorchBackend())
                                             .optimize(m, x,
                                               strategy=Autotuned(**kw))
```

* Import-path migration: `from catopt.optimize import …` →
  `from catopt_orchestrator import …`; `from catopt.adapters import
  TorchSource` → `from catopt_torch import TorchSource`; `from
  catopt.criteria import …` → `from catopt_orchestrator import …`;
  `from catopt.runners import …` → split across
  `catopt_orchestrator` / `catopt_torch` / `catopt_cuda`.
* `discover_alternatives`, `regime_dispatch`, `regime_frontier` keep
  their names (now in `catopt_orchestrator`).
* Rewrite the workspace `pyproject`s, `[tool.importlinter]`, CI and
  the `dev`/`oracle` dependency wiring for the new package name and
  `catopt-cuda`.

## Execution stages

1. Rename the package directory + distribution:
   `catopt_orchestrator` → `catopt_orchestrator`; rewrite intra-package
   imports and the `pyproject` name.
2. Reduce `catopt/` to a thin, torch-free re-export namespace —
   delete the `sys.modules` alias loop and every torch re-export.
3. Remove the `optimize_*` wrappers.
4. Migrate call sites (`tests/`, `bench/`, `demo.py`).
5. Rewrite import-linter contracts + CI; run the full gate suite.

## Tests & gates

* Package-boundary tests: `catopt_core` imports no other catopt
  package; `catopt_orchestrator` imports no torch/cuda.
* Every existing test passes after the import migration.
* Coverage stays 100%; `ty` clean; ruff `D`; radon; vulture; bandit;
  semgrep; import-linter rewritten; `pip-audit` for the renamed/new
  distributions.

## Risks

* **Import churn** — ~443 references across 51 files; mechanical, and
  the suite stays green per stage.
* **Coverage** — moved modules keep 100%; no dead re-exports.
* **Packaging** — the renamed distribution and the new `catopt-cuda`
  need workspace entries, CI install steps, and `--torch-backend cpu`
  handling.
