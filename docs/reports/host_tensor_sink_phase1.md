# HostTensorSink — Phase 1 implementation report

Phase 1 of plan 0036, implementing the split measured in
[`host_tensor_sink_phase0.md`](host_tensor_sink_phase0.md): a real
`mlopt_core.ports.Sink` over the `handl-lang/host-tensor` Rust op
registry for the **137 liftable-now** ops (elementwise / reduce-dim /
linalg / shape / constructors).

## What was built

**Rust binding (host-tensor repo).** `handl_host_tensor` is now an
importable PyO3 extension, gated behind `cargo build --features
python`:

- `Cargo.toml` — `[lib] crate-type = ["rlib", "cdylib"]`, optional
  `pyo3 = "0.26"`, `python = ["dep:pyo3"]`.  `pyo3/extension-module`
  is deliberately *not* enabled: on this toolchain it leaves `Py_*`
  symbols unresolved for `dlopen`; linking `libpython` keeps the .so
  self-contained.
- `src/python.rs` — two functions over the existing `Function`
  contract (`&[serde_json::Value] -> Result<Value>`):
  `ops() -> list[str]` (the 238 `tensor.*` names, declaration order)
  and `call(op_name, args_json) -> result_json`, dispatching through
  `tensor_op_registry()` exactly as a `.hnl` `call_host` would.
  `EvalError` maps to `ValueError`.
- `src/lib.rs` — `#[cfg(feature = "python")] pub mod python;`.

**Python adapter (graphopt repo).** New workspace package
`packages/kit-host-tensor` (`kit_host_tensor`):

- `_binding.py` — extension loader.  Resolution order:
  `import handl_host_tensor` → `$MLOPT_HOST_TENSOR_LIB` → the
  sibling-checkout convention `../host-tensor/target/<profile>/`
  (`$MLOPT_HOST_TENSOR_REPO` / `$MLOPT_HOST_TENSOR_TARGET` override).
  Every failure raises `HostTensorError` with the build command —
  the sink never degrades silently.
- `wire.py` — the `{"__tensor__": {shape, data, dtype}}` f64
  row-major codec, identical to `host-tensor/src/ops/codec.rs` and
  `lib-tensor/src/codec.hnl`.  Since host-tensor `0528649`,
  non-finite elements cross as `{"__nonfinite__": "nan"|"inf"|"-inf"}`
  tag objects in `data` — encode and decode both round-trip exactly;
  a literal `null` element (the pre-tag serde_json spelling) is
  rejected loudly, matching `decode_f64`.
- `bindings.py` — `HOST_BINDINGS`: the IR-op → host positional-arg
  table, exactly the 137 liftable names.  Attr spellings follow the
  torch adapter's convention (canonical names, `argN` fallbacks,
  `Const`-valued attrs unwrapped).  `chunk`/`split`/`unbind` return
  an `_Indexed` marker — their host ports emit a tensor *array* and
  the IR spelling selects via an `index` attr.
- `executor.py` — `HostTensorExecutor.forward(*xs)`: a term
  interpreter (`Var`/`Const`/`Param`/`Op`, `pack` expanded
  internally) that makes one extension call per op node.
- `sink.py` — `HostTensorSink`: `supported_ops` = the 137 names,
  `ops` = `OpTable.core()` + the builders, `executors = {}`,
  `lower` walks the term and declines unsupported ops *at lower
  time*, `verify` delegates to `report.verify_module`.
- `report.py` — `HostTensorVerifyReport` (`max_abs`/`max_rel`/
  `passed`, the same metric as `mlopt_onnx.report` /
  `kit_torch.report`).

Workspace wiring: root `pyproject.toml` gained the dependency, the
`workspace = true` source, the ty `extra-paths` entry, and
`kit_host_tensor` in the import-linter bottom layer beside
`mlopt_onnx`.

## Honest boundaries (unchanged from Phase 0)

- **f64-only wire.** dtype casts, bf16, int dtypes: declined.  The
  `"dtype": "f64"` stamp is true of the payload — `to_wire` upcasts
  to `float64` — but the *source* dtype is not preserved: a
  non-`float64` ndarray input warns at encode rather than casting
  silently.
- **`split` with int `sizes`** declines — torch's split-*size* vs the
  host's equal-count port; a sizes *list* works.
- **`var`/`std` `correction`** outside {0, 1} declines — the host
  driver has no correction slot.
- **`conv2d` `dilation != 1`**, **`expand implicit=True`** decline.
- **`new_zeros`/`new_ones`/`new_full`, `detach_`, `tan`** are *not* in
  `supported_ops` — they are needs-work names in the Phase-0 split,
  not liftable-now (the table is exactly 137).
- **`pack`/`getitem`** are IR-internal: `pack` is expanded by the
  evaluator (multi-output roots work), `getitem` consumers decline.
- **`ops.torch_bindings`** holds arg-list builders, not evaluators —
  same caveat as the ONNX sink's node emitters; search-time const
  folds must not call them for values.
- **No `register()`** — `BackendRegistration.source` is required and
  host-tensor has no `Source` yet; the sink is reachable directly or
  via `Optimizer(sink=HostTensorSink())`.
- Index-valued results (`argmax` etc.) come back as f64 — no int
  dtype on the wire.

## Verification

- `tests/test_host_tensor_sink.py` — **44 tests, all passing**:
  binding enumeration/dispatch, pointwise unary/binary vs NumPy,
  scalar-operand broadcast, `where`/`gt` composition, reductions
  (`dim` int/tuple/`keepdim`), `softmax`, unbiased `var`, `cumsum`,
  `matmul`, `linear` with params, `reshape -1`, `transpose`,
  `stack`/`concat`/`unbind[index]`, constructors (`zeros`/`full`/
  `arange`/`eye`), `numel`/`item`, `Param` leaves + missing-param
  decline, unsupported-op declines (`dropout`/`sort`/`to`/`einsum`/
  `nonzero`), `pack` multi-output, non-finite input refusal, and the
  `verify()` contract (pass / fail / `atol` gate).  Tests skip
  cleanly when the extension isn't built.
- Manual sweep of ~25 additional ops (clamp/lerp/expand/repeat/
  narrow/slice/tril/chunk/split/flatten/unsqueeze/squeeze/
  layer_norm/log_softmax/amax/count_nonzero/argmax/all/pow/gelu/erf/
  sigmoid/hardtanh/matmul+transpose) vs NumPy: all match within
  ~1e-7 (host `erf`/`gelu` differ from `math.erf` at ~1e-7 — inside
  the 1e-4 verify gate).
- `cargo test` in host-tensor: **116 tests pass, 0 failures** (one
  pre-existing warning: `out_has_ell` unused in `aten_kernels.rs`).
- `ruff check` + `ruff format --check`: clean on the new package and
  test file.  `lint-imports`: 1 contract kept.
- `ty check`: **0 diagnostics in `kit_host_tensor`**; the run
  reports 406 pre-existing diagnostics in `mlopt_discovery` that are
  identical on unmodified `HEAD` `pyproject.toml` (verified by
  reverting the config and re-running) — unrelated to this change.

## Files changed

**handl-lang/host-tensor**

- `Cargo.toml` — `[lib]` cdylib+rlib, optional `pyo3`, `python` feature.
- `Cargo.lock` — pyo3 dep closure.
- `src/lib.rs` — `pub mod python` behind the feature.
- `src/python.rs` — new: the PyO3 `ops()`/`call()` binding.

**handl-lang/graphopt**

- `pyproject.toml` — dep, uv source, ty path, import-linter layer.
- `uv.lock` — workspace resolution for `kit-host-tensor`.
- `packages/kit-host-tensor/pyproject.toml` — new package.
- `packages/kit-host-tensor/src/kit_host_tensor/__init__.py`
- `packages/kit-host-tensor/src/kit_host_tensor/_binding.py`
- `packages/kit-host-tensor/src/kit_host_tensor/bindings.py`
- `packages/kit-host-tensor/src/kit_host_tensor/errors.py`
- `packages/kit-host-tensor/src/kit_host_tensor/executor.py`
- `packages/kit-host-tensor/src/kit_host_tensor/report.py`
- `packages/kit-host-tensor/src/kit_host_tensor/sink.py`
- `packages/kit-host-tensor/src/kit_host_tensor/wire.py`
- `tests/test_host_tensor_sink.py` — new.
- `docs/reports/host_tensor_sink_phase1.md` — this report.

Nothing committed or pushed.
