# Phase A — install/artifact contract (graphopt slice)

*Implementation report for `docs/reports/production_readiness.md` Phase A.
CPU-only; no version bumps, no commits, no publishes — `0.1.0dev` stays.*

## What landed

### 1. Artifact contract — `packages/mlopt-torch/src/mlopt_torch/export.py`

Every manifest now carries three contract fields alongside the
existing `format`/`torch_version`/`runnable_standalone`/`weights_only`:

- **`artifact_schema`** = `ARTIFACT_SCHEMA` (`"mlopt-artifact/1"`,
  exported in `__all__`) — the schema tag `load_optimized` branches
  on.  A manifest naming an unrecognised schema is refused with
  `ExportError`; manifests *predating* the field (no key) are read as
  schema-1, which is honest — they only ever named the four `format`
  values.
- **`mlopt_version`** — the exporting `mlopt-torch` distribution
  version via `importlib.metadata` (`"unknown"` when the dist isn't
  installed, e.g. a bare source tree).
- **`artifact_kind`** — `"runnable"` (`pt2`/`torchscript`: the artifact
  IS the optimized computation) vs `"weights"` (`safetensors`/
  `state_dict`: parameters only).  The weights formats are
  unmistakably weights-only at three levels: `weights_only` +
  `artifact_kind` in the sidecar manifest, the same fields embedded in
  the safetensors `__metadata__.mlopt_manifest`, and the module
  docstring's "Artifact contract" section.

`.pt2` is documented as *the* runnable artifact; safetensors/state_dict
carry the optimized-layout caveat (fused/derived `state_dict` names —
not a drop-in `load_state_dict` for the original architecture).

### 2. Public-surface decision — FREEZE the narrow set

`docs/public_api.md` gained a **"Torch delivery contracts (→
graphopt)"** section; `tests/test_public_api.py` pins it.  Decision:
**freeze** `TorchBackend(...)`, `export_optimized`/`save_optimized`/
`load_optimized`/`save_optimized_weights`/`ExportError`/`ARTIFACT_SCHEMA`
(`mlopt_torch.export`), the runner constructors (`TorchCompileRunner`,
`CudaGraphRunner`, `IdentityRunner`, `ChainedRunner`), and
`Optimizer.optimize(model, x, *, strategy=None, **kw)`.

Justification: this is the surface a downstream inference/serving
consumer actually binds — the artifact boundary the
".pt2-runs-without-mlopt" claim stands on.  `IRModule`, the executor
table, `adapters`, `autotune.TORCH_BUILDERS`, `calibrate`,
`contraction_policy`, and the model zoo stay **public, unfrozen** —
the discovery loop is still changing that machinery, and freezing it
would calcify internals without protecting any consumer contract.

Mechanics:

- `_PUBLIC_SURFACE` extended with `mlopt_torch`,
  `mlopt_torch.backend`, `mlopt_torch.export`, `mlopt_torch.runners`,
  `mlopt_cuda` rows.
- New `TestFrozenSurface` — the mechanical pin (module imports, name
  resolves, `__all__` seats) that the phase-4 split left covering only
  the substrate rows in `eqsat`; this restores it for **all** domain
  seats (`gamekit.*`, `mlopt_discovery.*`, the new torch/cuda rows —
  ~190 parametrized cases).
- New `TestSignaturePins` cases: `Optimizer.optimize`,
  `TorchBackend`, `export_optimized` (kwarg-only set + defaults),
  `load_optimized`, `save_optimized_weights`, and the three runner
  constructor shapes.  Torch imports are lazy inside test bodies —
  collection stays torch-free.
- `docs/site/reference/frozen-api.md` re-vendored from the updated
  upstream doc.

### 3. Install-path docs

- `docs/site/tutorials/setup.md` — new "No published wheels — yet"
  callout: everything is `0.1.0dev`, nothing is on any index, the
  supported install is clone + `uv sync` with git+ssh pins; "pip
  install mlopt" is explicitly unsupported.
- `docs/site/how-to/optimize-a-model.md` §5 — rewritten around the
  artifact contract: `.pt2` as *the* runnable artifact, the manifest
  field list, a per-`fmt` table, and the GGUF/non-target note.
- `docs/api.md` — the `export_optimized` row now names the kind
  semantics and manifest fields.

### 4. Wheel-build smoke — feasible, verified

`uv build --wheel` in `packages/mlopt-torch` **succeeds**
(`mlopt_torch-0.1.0.dev0-py3-none-any.whl`, setuptools backend,
bundles `artifacts/*.pt`).  Honest gap, now documented: a wheel
install still needs `mlopt-core`/`mlopt-orchestrator`/`gamekit` from
their own repos — there is no self-contained install, and no wheel CI
exists.  Per-package wheels are *buildable*; publishing is out of
scope (and barred by the global no-publish rule anyway).

## Test changes — `tests/test_export.py`

- Manifest assertions for `artifact_schema`/`artifact_kind`/
  `mlopt_version` on the pt2, safetensors (sidecar AND embedded), and
  state_dict round-trips.
- **Clean-interpreter matrix** (`test_pt2_clean_interpreter_matrix`):
  dtype {fp32, fp64} × shape {static, dynamic} = 4 cells, each
  exporting an *optimized* module and running `torch.export.load` in a
  fresh `sys.executable` that asserts no `mlopt*` module loaded.
  Dynamic cells re-run a different batch size; static cells re-run the
  example shape.  (Note: `IRModule.forward` is `*xs`, so the
  `dynamic_shapes` spec nests one pytree level deeper than a plain
  module's — `(({0: Dim},),)`.)
- `test_pt2_clean_interpreter_kwargs` — kwarg-forward artifact runs
  standalone with kwargs.
- `test_load_optimized_rejects_unknown_schema` /
  `test_load_optimized_legacy_manifest_without_schema` — the schema
  gate and its backward-compat edge.

## Verification

- `scripts/pytest-capped.sh .venv/bin/python -m pytest tests/test_export.py -q` — **42 passed** (~2 min).
- `scripts/pytest-capped.sh .venv/bin/python -m pytest tests/test_public_api.py -q` — **240 passed, 1 skipped** (~5 s).
- `ruff check` + `ruff format --check` on `packages scripts examples` — clean.
- `ty check` — 0 errors.

## Follow-ups / honest gaps (not done here)

- **`API_VERSION` bump**: the freeze is additive, which
  `docs/public_api.md`'s own rule maps to a MINOR bump — but
  `API_VERSION` lives in `mlopt_core` (**eqsat, read-only this
  task**).  Left at `0.1.0`; an eqsat-side bump to `0.2.0` (or an
  explicit "graphopt-local freeze doesn't move the substrate
  constant" note) is the orchestrator's call.
- **Orchestrator-side (eqsat)**: nothing required — `Optimizer.optimize`
  already had the pinned shape; the freeze only pins it from this
  repo.
- **Wheel CI / sdist smoke**: `uv build` works per-package but there is
  no pipeline building or installing them; a real "install the wheel
  into a clean venv" test needs the eqsat/gamekit dists resolvable
  first (path or git deps) — Phase A+ work.
- **CUDA matrix cells**: the clean-interpreter matrix is CPU-only by
  design; `requires_cuda` coverage of `.pt2` on GPU is a later phase.
- Pre-existing uncommitted changes in the tree
  (`mlopt_discovery/model_corpus.py`, `play.py`,
  `tests/test_discovery_model_corpus.py`, `docs/reports/per_model_corpus.md`)
  are someone else's WIP — untouched.
