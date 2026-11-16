# discoverykit seam sweep — torch-blocked import census

**Date:** 2026-11-14 · **Plan:** 0034 (`project/plans/pl-0030`) ·
**Scope:** every module under
`packages/mlopt-discovery/src/mlopt_discovery/` (the `intake/` and
`lawdata/` subpackages included; there is no `priors/` or `domains/`
subpackage — `lawdata/` is the only nested package besides `intake/`).

## Method

Each module was imported in a fresh subprocess with the torch stack
poisoned **before** the import:

```python
import sys
for n in ("torch", "kit_torch", "mlopt_carriers", "mlopt_cuda",
          "mlopt_onnx"):
    sys.modules[n] = None
import importlib
importlib.import_module("mlopt_discovery.<mod>")
```

`sys.modules[name] = None` makes the import system raise
`ModuleNotFoundError`/`ImportError` for the name and its submodules,
so a clean run proves nothing in the module's import closure binds
the torch stack at import time.  85 modules were swept (the package
`__init__`s included).

## Result

**73 / 85 modules import clean.  12 fail — and every failure is a
module the task already classed as torch-domain or carriers-bound.
Zero cheap-seam fixes were needed**: the earlier seam passes
(`proposal_torch_seam.md`, `impact_census_torch_seam.md`) had already
converted every remaining module-level torch bind to the lazy
function-level pattern, and no module outside the excluded set binds
torch (or a torch-bound sibling) at module level.

A module-level `import torch` grep confirms it: outside the excluded
files the only column-0 torch binds are inside string literals
(`emit.py`'s generated-test template, `shipped_primitives.py`'s
docstring example).  Every real bind is function-level, and the
optional-dependency paths degrade honestly (`try/except ImportError`
in `play.py`, `arena.py`, `meta_arena.py`, `player_arena.py`,
`shape_proposal.py`, `vocabdir.py`).

## Classification table

### TORCH-FREE — import-clean (73)

| Module | Notes |
|---|---|
| `mlopt_discovery` (`__init__`) | package root |
| `advantage` | |
| `antiunify` | lazy `evidence`/`oracle` binds inside functions |
| `arena` | lazy `mlopt_carriers.om` laws merge (`_carrier_basis`, ~L1879) |
| `arena_player` | |
| `bilinear_game` | |
| `bounds` | |
| `cellstore` | lazy `evidence`/`lawdata` |
| `census` | seam landed earlier (`impact_census_torch_seam.md`) |
| `coherence` | |
| `coherence2` | lazy `TorchSource` reach probe (~L420) |
| `combinator` | |
| `corpus_cov` | |
| `domain_spec` | |
| `dspec_synth` | |
| `emit` | column-0 `import torch` is inside a generated-file string literal |
| `engine` | shim |
| `enumerate` | |
| `evidence` | |
| `families` | lazy `TorchSource`/`nn` (~L120) |
| `family_synth` | lazy `torch_bridge` (~L327) |
| `form_propose` | |
| `forms` | |
| `fragment_game` | |
| `genkernel` | **import-clean but torch-semantic at call time** — the whole module is torch.compile/Triton/`TorchSink` codegen; ~50 lazy torch binds.  See "call-surface" note below. |
| `grammar` | |
| `handl` | |
| `handl_egraph` | module-level `from mlopt_discovery import vocabdir` — vocabdir itself is import-clean |
| `handl_engine` | guards the optional `mlopt_handl_peer` cdylib, not torch |
| `handler_synth` | import-clean; lazy torch binds for minted-op torch bindings (~L297–1984) — torch-semantic call surface |
| `impact` | seam landed earlier |
| `index_abstract` | |
| `law_order` | lazy torch + `export_to_ir` in `zoo_terms` only (~L261) |
| `lawdata` + `lawdata.laws` + `.banks`/`.handlers`/`.macros`/`.moves`/`.terms` | data tables, no torch |
| `learned_propose` | |
| `lemma_cert` | |
| `loop_propose` | |
| `machine_pack` | |
| `macro_moves` | |
| `meta_arena` | lazy `TorchSink` with `except ImportError` (~L2607) |
| `meta_player` | |
| `meta_prior` | |
| `model_domains` | |
| `modelproj` | lazy torch/`torch_bridge` in the eval paths (~L410–567) |
| `object_synthesis` | lazy `oracle`/`evidence`/`lemma_cert` |
| `open_game` / `open_prior` / `open_self` / `open_verdict` / `open_vlabel` | |
| `oracle` | seam landed earlier |
| `pay_gate` | |
| `pipeline` | lazy `mlopt_discovery.intake` import inside the census path (~L1900) — see TRANSITIVE note |
| `player_arena` | lazy torch with `except ImportError` (~L230–289) |
| `players` | shim |
| `play` | torch arms all lazy (~L258–4219); `except ImportError` honest-decline paths |
| `prim_mint` | |
| `proposal` / `proposal_core` | seam landed earlier (shim + core) |
| `propose` | lazy `mlopt_carriers` rules merge in `_carrier_rules` (~L491) — carriers-bound call path |
| `scoring` | shim |
| `shape_proposal` | lazy `TorchSink` singleton (~L326) |
| `shipped_primitives` | torch denotation registry by design, but binds lazily — `register()`/`_bindings` import `TorchSink`/`torch_bridge` inside functions; import-clean |
| `training` | lazy `eval_term` (~L504) — reverse-mode VJP is term-level, torch-free |
| `verifier` | lazy torch in `_law_cases` build (~L410) |
| `vocab` | seam landed earlier |
| `vocabdir` | **import-clean**; the op-census path imports `mlopt_carriers.om`/`.trace` at call time (~L736, not try-guarded) — see BLOCKED-ON |

### STAYS — torch-shaped, module-level binds (11)

| Module | Why it stays |
|---|---|
| `gap_gen` | module-level `import torch` + `TorchSink` (~L81–87); gap synthesis runs the torch pipeline |
| `meta_game` | module-level `torch`/`nn`/`Categorical` (~L159–167); the meta-arena torch policy |
| `model_corpus` | module-level `torch` + `TorchSink`/`TorchSource` (~L77–79); real-model corpus is torch-domain |
| `workload_gen` | module-level `torch` + `TorchSink` (~L84–94); workload harness is torch-domain |
| `zoo` | module-level `torch`/`nn`/`F` (~L81–83); the model zoo is `nn.Module`s |
| `intake/__init__.py` | re-exports the torch-bound intake surface |
| `intake/ledger.py` | module-level `torch`/`nn`/`TorchSink`/`export_to_ir` (~L23–33) |
| `intake/models.py` | module-level `torch`/`nn`/`F` — workload model definitions |
| `intake/registry.py` | module-level `torch`/`nn`/`F` — intake registry binds `nn.Module` thunks |
| `intake/report.py` | imports clean itself but is part of the intake package — `intake/__init__` fails, so the package does |
| `intake/__main__.py` | `python -m mlopt_discovery.intake` driver; fails via the package `__init__` |

(`intake/report.py` and `intake/__main__.py` have no direct torch
binds — they fail only because `mlopt_discovery.intake.__init__`
binds torch; they stay with the package.)

### BLOCKED-ON (1)

| Module | Blocker |
|---|---|
| `state_reindex` | `from mlopt_carriers.om import SPLIT_TRANSPOSE` + `from mlopt_carriers.xcarrier import _xshape` at module level (~L64–65).  Needs the `SPLIT_TRANSPOSE`/`_xshape` relocation into `mlopt_core` (or an injected carrier seam) before it can import carriers-free.  Not torch — a carriers dep. |

### TRANSITIVE notes (import-clean, torch-bound at call time)

These modules pass the sweep but contain call paths that will raise
`ImportError` in a torch-free env — honest failures, matching the
established lazy-bind contract:

- `pipeline.py` ~L1900: `from mlopt_discovery import intake` inside
  the corpus-census path — the `zoo-real`/intake corpus arm is
  torch-domain by construction.
- `propose.py` ~L491 (`_carrier_rules`), `arena.py` ~L1879,
  `play.py` ~L882/L4092: lazy `mlopt_carriers` binds — the carrier
  law banks are a separate substrate question from torch.
- `vocabdir.py` ~L736: `import mlopt_carriers.om`/`.trace`
  unguarded inside the op-census path — the carrier op census needs
  carriers present at call time even though the module imports
  clean.  (The torch half of that path *is* try-guarded, ~L726.)
- `genkernel`, `handler_synth`, `modelproj`, `families`,
  `family_synth`, `coherence2`, `training`, `play`,
  `player_arena`, `meta_arena`, `shape_proposal`, `verifier`,
  `law_order`, `oracle`, `impact`, `vocab`, `proposal_core`,
  `shipped_primitives`: heavy lazy torch/`kit_torch` call surfaces
  (evaluation oracles, `TorchSink`/`TorchSource`, minted-op torch
  bindings, `torch.compile` codegen).  Import-clean per the seam
  contract — whether they *belong* in discoverykit is a packaging
  call, not an import blocker.  `genkernel` in particular is the
  most torch-semantic of the clean set (its whole purpose is
  torch.compile/Triton kernel generation) and is the likeliest
  candidate to stay behind even though it imports clean.

## Changes made

None to shipped code — the sweep found no cheap-seam cases outside
the excluded set.  This report is the only file added.

## Verification

- Sweep: 73 PASS / 12 FAIL, table above (script: per-module
  subprocess, `sys.modules` poison for `torch`, `kit_torch`,
  `mlopt_carriers`, `mlopt_cuda`, `mlopt_onnx`).
- Existing seam pins still green:
  `tests/test_discovery_proposal.py::test_proposal_modules_import_without_torch`
  and the census equivalent — 2 passed.
- No `mlopt-discovery` files were modified, so no package tests were
  re-run beyond the seam pins; `git status` shows no discovery-side
  dirt from this work.
