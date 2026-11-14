# Dtype registry — one table for every backend

`docs/reports/arch_audit_graphopt_packages.md` §3 / D5 flagged the
dtype mappings duplicated ~4 places: torch dtype ↔ safetensors tag,
IR name ↔ NumPy ↔ ONNX ↔ onnxruntime, `hardware.dtype` names, and the
`str(dtype).split(".")[-1]` name idiom.  This change unifies them into
`mlopt_dtypes`, a new leaf workspace package.

## Why a new package (not `mlopt_core`)

`mlopt_core` is the natural home — torch-free, stdlib-only — but it
ships from `handl-lang/eqsat` as a **pinned git+tag dep**
(`v0.1.0`); a new module there is a release, not a commit.  And the
registry must be importable by *both* `mlopt_torch` and
`mlopt_onnx`: the import-linter `dependency layers` contract puts
`mlopt_onnx` at the bottom and `mlopt_torch` above it, so onnx cannot
import torch (and must not gain an onnxruntime-free sibling's torch
dependency).  `packages/mlopt-dtypes` is pure stdlib, zero deps, and
sits as the lowest layer — every backend may read it, nothing may
depend back on it.

## The registry

`mlopt_dtypes.DTYPEINFO`-keyed table `DTYPES: Mapping[str, DTypeInfo]`
— 12 canonical dtypes (the union of every shipped vocabulary), keyed
by the **IR-recorded short name** (`str(torch.dtype)` minus the
`torch.` prefix — what `to(dtype="float64")` stores).

`DTypeInfo` fields, all data (no torch/numpy/onnx imports):

| field | meaning | example |
|---|---|---|
| `torch` | `torch.dtype` attribute name | `"bfloat16"` |
| `numpy` | `np.dtype()`-resolvable name, `None` if NumPy lacks it | `"float32"` / `None` for bfloat16, float8 |
| `onnx` | `onnx.TensorProto` enum attribute name, `None` where the ADR-0006 sink policy refuses | `"FLOAT"` / `None` for bfloat16 (deliberate — `TensorProto.BFLOAT16` exists but the sink refuses it) |
| `ort` | onnxruntime input type string | `"tensor(float)"` |
| `st` | safetensors container tag | `"F32"` |
| `hnl` | host-tensor `.hnl` wire tag | `"f64"` — only float64; the wire is f64-only (host-tensor ADR-0083) |
| `eps` | `finfo().eps`, `None` for exact types | `1.1920928955078125e-07` |
| `itemsize` | bytes per element | `4` |

Helpers: `canonical_name(name)` (the `split(".")[-1]` idiom, now one
place), `get(name)` → `DTypeInfo | None` (never raises),
`require(name)` → `DTypeInfo` (`KeyError` listing the registry),
`names()`.

Unregistered names are **not** an error: the torch bridge still
resolves arbitrary `torch.dtype` spellings (complex, quantized, …)
through `getattr` — the registry names the vocabulary, it does not
gate it.

## Consumers migrated

| file | before | after |
|---|---|---|
| `packages/mlopt-torch/src/mlopt_torch/export.py` | literal `_TORCH_TO_ST`/`_ST_TO_TORCH` (12 entries) | derived from `DTYPES` (`torch`+`st` columns) |
| `packages/mlopt-onnx/src/mlopt_onnx/arrays.py` | literal `DTYPE_NAMES`, `_NUMPY_TO_ONNX`, `ONNX_TYPE_NAMES`, `_ORT_TO_NUMPY` | all four derived from `DTYPES`; public names (`DTYPE_NAMES`, `WORKING_DTYPES`, `ONNX_TYPE_NAMES`, `numpy_to_onnx_type`, `ort_type_to_numpy`) unchanged |
| `packages/mlopt-torch/src/mlopt_torch/torch_bridge.py` | five inline `str(dtype).split(".")[-1]` sites + `getattr(torch, …)` in `_creator_dtype` | `canonical_name()` at all five sites; `_creator_dtype` consults the registry then falls back to `getattr` (identical resolution) |
| `packages/mlopt-torch/src/mlopt_torch/report.py` | `_dtype_rtol_floor` used `torch.finfo(dtype).eps` directly | registry `eps` first, `finfo` fallback for unregistered spellings — values identical (`eps` literals were taken from `finfo`) |

## Deliberately not migrated

- **`mlopt_discovery/play.py::_torch_dtype`** (the `hardware.dtype`
  spec table, fp32/fp64/fp16/bfloat16) and **`mlopt_carriers`**
  (`trace.py::_eye_torch`'s `getattr(torch, …)` dtype resolution) —
  both files are **dirty under another agent's in-flight work**; the
  registry's `get()`/`canonical_name()` are drop-in replacements when
  that work lands.
- **`mlopt_onnx/opmap.py::_CASTS`** — maps ATen *cast-op aliases*
  (`"float"`, `"double"`, `"long"`, …) to IR names; that is a
  spelling-alias table, not a dtype-property table, and stays local.
- **`mlopt_torch/torch_bridge.py`'s `"float"` binding** (line ~2276)
  resolves `kw["dtype"]` via raw `getattr(torch, str(...))` without
  the `.`-split — recorded names are already canonical, and adding
  canonicalization would *widen* accepted spellings (drift); left as
  is.
- **`mlopt_torch/calibrate.py`** records `str(dtype).replace("torch.","")`
  as provenance metadata — a reporting string, not a lookup table.

## Tests

`tests/test_dtype_registry.py` (9 tests): registry shape/vocabulary,
`canonical_name`/`get`/`require` spellings, `eps`/`itemsize` vs
`torch.finfo`, the safetensors tables' exact contents + a per-dtype
container round-trip through `_write_safetensors`/`_read_safetensors`,
the ONNX tables' derivation and round-trips (`numpy_to_onnx_type`,
`ort_type_to_numpy`, `ONNX_TYPE_NAMES` inverse), `_creator_dtype`
resolution incl. the unregistered-name fallback, and the verify
rtol-floor values (`8*eps` for bf16, `0.0` for fp32/int).

## Verification

- `pytest tests/test_dtype_registry.py` — 9 passed.
- Focused slice (`test_export`, `test_reports`, `test_onnx_{sink,
  source,liftmap,opmap}`, `test_torch_bridge_edges`,
  `test_dtype_registry`, `-m "not slow"`): **558 passed, 1 failed** —
  `test_onnx_sink.py::test_op_matches_torch[tan]` is **pre-existing
  and unrelated**: `aten::tan` exports as `opaque` because `"tan"`
  is absent from `REGISTRY`/`_IR_TO_TORCH` (filtered out of
  `_ATEN_TO_IR` by the schema-classification commit `248531c`); the
  dtype diff cannot reach that path.
- Smoke lane (`test_law_order`, `test_law_provenance`,
  `test_decode_geom`): 81 passed.
- `ruff check` + `ruff format --check` on touched files: clean.
- `ty check`: 0 errors.  `lint-imports`: contract kept (new bottom
  layer `mlopt_dtypes`).  `vulture`: clean.

## Workspace wiring

- `packages/mlopt-dtypes/pyproject.toml` — new member (`packages/*`
  glob); installed editable by `uv sync`.
- Root `pyproject.toml`: `mlopt-dtypes` added to project deps +
  `[tool.uv.sources]` (`workspace = true`) + import-linter
  `root_packages` and the bottom `layers` entry.
- `mlopt-torch` / `mlopt-onnx` `pyproject.toml`: `"mlopt-dtypes"` dep.
