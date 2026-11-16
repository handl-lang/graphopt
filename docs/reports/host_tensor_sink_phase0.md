# Plan 0036 Phase 0 — host-tensor coverage map + codec requirements

Date: 2026-11-19. Scope: analysis only (read-only on
`host-tensor`, `lib-tensor`, `eqsat`); no implementation.

Inputs:

- mlopt IR vocabulary: `mlopt_core.opmeta.REGISTRY`
  (`eqsat/packages/mlopt-core/src/mlopt_core/opmeta.py`) — **231 ops**.
- host-tensor surface: `tensor_op_registry()` in
  `host-tensor/src/ops/mod.rs` — **238 registered `tensor.*` ops**
  (30 handwritten core + ~205 generated ATen ports from
  `tools/aten_port_list.txt` × `aten_schemas.json`, minus seams).
- Parity oracle: `lib-tensor/verify.hnl` — generated verdict program,
  ~204 checks, all `ok` where runnable (see codec findings for the
  skipped `math.*` verdicts).

## 1. Op-coverage diff

Name-level diff of the two vocabularies:

| Set | Count |
|---|---|
| mlopt IR ops | 231 |
| host `tensor.*` ops | 238 |
| Name overlap | **153** |
| IR-only | 78 |
| host-only (not IR ops) | 85 |

Name overlap is *not* liftability: host ops are positional-JSON ATen
ports over an **f64-only** wire; IR ops carry attrs (`dim`, `keepdim`,
`eps`, dtype casts, equation strings) that must be mapped per op, and a
few IR ops are multi-output or in-place. The three sets below are the
plan's liftable-now / needs-work / honest-decline split.

### Liftable-now — 137 ops

Elementwise unary/binary, reduce-dim, linalg/matmul, shape/structural,
constructors — every op where a same-named host impl exists and the
IR attrs map mechanically onto the generated port's positional args
(`dim`/`keepdim` → `dim_IntList` + `keepdim`, etc.). Includes the
Phase-1 first milestone:

- **pointwise**: `abs neg exp exp2 expm1 log log1p log2 log10 sqrt
  rsqrt reciprocal square sin cos tan asin acos atan sinh cosh tanh
  asinh acosh atanh floor ceil round trunc frac sign sigmoid silu mish
  gelu elu celu leaky_relu`, `erf erfc sinc softplus hardtanh
  heaviside nan_to_num`, comparisons `eq ne lt le gt ge`, logic
  `logical_and/or/not/xor`, predicates `isnan isinf isfinite isneginf
  isposinf`, `pow maximum minimum fmod remainder atan2 xlogy lerp
  where clamp clamp_min clamp_max add mul sub div`.
- **reduce-dim**: `sum prod mean max min amax amin var std argmax
  argmin all any count_nonzero nanmean nansum cumsum cumprod
  softmax log_softmax layer_norm rms_norm`. Reduction ops need
  only the mechanical `dim`/`keepdim` attr→arg map.
- **linalg/matmul**: `matmul linear outer einsum?* conv2d` (*einsum
  is needs-work — equation-string attr). `conv2d`/`matmul` quality is
  measured, not assumed, per the plan's hard gate.
- **shape/structural**: `reshape view transpose unsqueeze squeeze
  flatten narrow select slice chunk split stack concat unbind expand
  expand_as repeat contiguous clone detach`, `tril triu flip`,
  `pixel_shuffle pixel_unshuffle`.
- **constructors/scalars**: `zeros zeros_like ones ones_like full
  full_like arange eye`, `numel item`.

(Exact 137-name list: computed set in §5; regenerate with the snippet
there.)

### Needs-work — 70 ops (16 name-overlap + 54 IR-only)

**Overlap but unverified semantics (16):** `sort topk kthvalue median
mode nonzero argsort` (tuple / variable-shape / int-index returns on an
f64 wire), `to` (dtype attr — host `tensor.to` is f64-identity only,
other dtypes error loudly), `batch_norm` (running stats/momentum attrs
→ positional args), `searchsorted gather index_select index_add`
(index-valued results; `index_add` is `write`-tagged in IR),
`einsum` (equation-string attr), `prelu glu` (weighted activations —
arg order must be pinned).

**IR-only, host-feasible (54):** mostly *name/attr mismatches* or
thin composites over existing host ops:

- rename/map: `permute` (host has `moveaxis`/`swapaxes` — compose or
  add a port), `movedim` (= `moveaxis`), `broadcast_to` (= `expand`),
  `broadcast`, `max_pool2d` (host `max_pool`), `conv1d` (generalize
  `conv2d`), `adaptive_avg_pool2d` (host `avg_pool2d` + adaptive
  logic), `tensor_split` (= `split`), `linalg_vector_norm` (= `norm`),
  `std_mean`/`var_mean` (composite of `std`/`var`/`mean`),
  `masked_fill` (`where` + `eq`), `addcmul`/`addcdiv`/`fmax`/`fmin`/
  `isclose`/`bitwise_and`/`bitwise_or` (elementwise — small kernels),
  `relu6`/`hardsigmoid`/`softsign`/`inv` (trivial composites),
  `new_zeros`/`new_ones`/`new_full` (= `zeros`/`full` + shape arg),
  `unflatten`/`unfold`/`roll`/`pad` (`constant_pad_nd`)/`one_hot`/
  `embedding`/`diag_sum`/`cummax`/`cummin`/`logcumsumexp`/
  `group_norm`/`instance_norm`/`index`/`index_put`/`take_along_dim`/
  `scatter*`/`select_scatter`/`slice_scatter` (scatter family is
  `write`-tagged — decide functional-vs-inplace semantics first),
  `upsample_bilinear2d`/`upsample_nearest2d` (new kernels),
  `detach_` (in-place marker — alias `detach`), special functions
  `digamma`/`gammaln`/`erfinv`/`i0` (need `libm`-class impls host
  lacks).

### Honest-decline — 24 ops

- **carrier/domain ops** (no host analogue, out of scope): `om aff
  aff_diag cswap lstm.input sdpa deform_conv2d`.
- **RNG**: `dropout` — the wire has no `Generator`; host seams
  `rand`/`normal`/`empty` already fail loudly for the same reason.
  Deterministic-seed dropout is a codec extension, not a port.
- **dtype casts**: `bfloat16 bool byte char double float half int
  long short type_as` — the codec is `"dtype": "f64"` only; decode
  rejects every other dtype (`codec.rs` `decode_tensor`). Multi-dtype
  is a kernel-first change upstream (`handl-value TensorValue` is
  `Vec<f64>`), per host-tensor README.
- **IR-internal/meta ops**: `alias leaf getitem copy pack` — e-graph
  bookkeeping terms, never lowered to any backend.

`Capabilities.supported_ops` for Phase 1 = the liftable-now set ∪
whatever needs-work entries get verified; extraction prices the rest
`+inf` and the sink declines loudly — matching the plan's explicit
decline-coverage gate.

## 2. Codec findings — how values cross the boundary today

**Wire format** (identical on both sides —
`host-tensor/src/ops/codec.rs`, mirrored by `lib-tensor/src/codec.hnl`,
ADR-0083 §2.1):

```json
{"__tensor__": {"shape": [d, ...], "data": [x, ...], "dtype": "f64"}}
```

Dense row-major `f64` payload; `dtype` optional, absent ⇒ `f64`, any
other value is a `TypeMismatch`. Boundary ops `zeros/ones/from_list`
construct, `to_list/shape` project.

**Execution path today:** there is **no Python↔host callable**. The
only executable path is:

1. `lib-tensor/verify.hnl` is a `.hnl` program run by a tensor-composing
   host — **catopt's `hnl-run --project <repo> verify.hnl`** (catopt is
   *not* in this workspace; it builds `HandlModule::new("tensor",
   TENSOR_HNL).with_ops(TENSOR_OPS, tensor_op_registry())` and resolves
   `from "host-tensor"`/`from "lib-tensor"` through `.handl.toml` deps).
2. Inside that host, `tensor.*` calls dispatch `call_host` →
   `ModuleLookup` → `handl_platform::Function::call(args:
   &[serde_json::Value]) -> Result<Value, EvalError>` — the codec *is*
   already the `Function` boundary.
3. `lib-tensor` ops are pure `.hnl` — they need **zero** host calls and
   run on any engine that evaluates the arrow dialect; `verify.hnl`
   diffs them against the Rust atoms. Caveat found: ~20 `math.*`
   unary verdicts are **skipped**, not failed — `hnl-run`'s CatoptKit
   doesn't register `plugin-lib-stdlib` `math.*` leaves (environment
   gap, not a semantic verdict). The Rust `aten_kernels` implement the
   same math natively (`f64::exp` etc.), so the *sink* path is
   unaffected — this only limits lib-tensor-as-oracle coverage today.

**Codec gaps a `Sink` must handle:**

- **f64-only.** Torch-sourced IR leaves arrive as f32/bf16/fp64/int.
  `HostTensorSink.lower` must upcast leaf values to f64 on encode
  (lossless for f32→f64; bf16→f64 also exact) and the `verify` gate
  must use dtype-aware tolerances vs the f32/f64 oracle — the plan's
  "same tolerances" gate means *no relaxation*, but the comparison
  basis changes (host computes in f64 where torch computes in f32 —
  host is the *more* precise side; keep `rtol/atol` as configured).
- **Non-finite hole.** `serde_json::Value` serializes `NaN`/`±inf` as
  `null` — the codec's documented finite-only hole (host-tensor
  README). Tensors containing non-finite values silently corrupt
  across any serde_json boundary. Phase 1 needs either a tagged
  non-finite encoding in `data` (e.g. string sentinels) or a
  non-JSON arg channel — flag before any verify path that feeds
  `inf`/`nan` (e.g. `nan_to_num` tests, softmax overflow fixtures).
- **Index dtypes.** `nonzero`/`argsort`/`topk`/`searchsorted` return
  index tensors; torch uses int64, the wire is f64. Values compare
  fine; dtype-strict checks must whitelist this.
- **Params.** `Sink.lower(ir, params)` gets `{name: leaf_value}`;
  leaves are `Param`/`Const`/`Var` terms — the binder must encode
  each to the wire record (torch/numpy → nested lists → `data` flat
  row-major).

**Callable-interface verdict: Phase 1 needs a binding layer.**

`handl-host-tensor` is a plain Rust **rlib** — no `crate-type`, no
PyO3, no cdylib. What exists is exactly the right seam for one:

- `tensor_op_registry() -> OpRegistry` of `Function`s whose entire
  contract is `&[serde_json::Value] -> Result<Value, EvalError>` —
  JSON-in/JSON-out, no engine types leak.
- A thin PyO3 extension (new crate or a `pyo3` feature on
  host-tensor) can call `Function::call` directly: Python dicts ↔
  `serde_json::Value` via `serde_json`/`pyo3` conversions. This is a
  ~200-line binding, not a port.
- Fallback with zero new Rust: emit a `.hnl` artifact and shell out
  to `hnl-run` per lowered module — works today but pays process
  startup + full program eval per `forward`, wrong shape for the
  `Executor.forward(*xs)` hot path and for `verify`'s many calls.
- `lib-tensor` itself is *not* the deliverable path — it's the
  transparency/parity spec; the sink lowers to host-tensor's Rust
  atoms.

Recommendation: **PyO3 shim over `tensor_op_registry()`** (host-tensor
side, small), consumed by a `kit-host-tensor` adapter package or a
`mlopt_torch`-adjacent module — keeping `mlopt-core` torch-free per
the hexagon (the binding lives adapter-side, like `TorchSink`).

## 3. `mlopt_core.ports.Sink` contract — what `HostTensorSink` must implement

From `eqsat/.../mlopt_core/ports.py` (protocols are
`@runtime_checkable`, presence-checked):

- `Capabilities.supported_ops -> frozenset[str]` — the extraction
  bound; `backend_cost` prices unsupported ops `+inf`. Set = the
  declared liftable set (explicit, per hard gate).
- `Capabilities.ops -> OpRegistry` — the `OpTable`-shaped registry
  (`torch_bindings`/`shape_rules`/`attr_schemas`/`register`) that
  compile-time const folds dispatch through. Despite the name,
  `torch_bindings` is just `dict[str, Binding]` —
  `Binding = (*args, **attrs) -> Any`; host bindings slot in
  unchanged (name is historical).
- `Sink.lower(ir, params=None) -> Executor` — materialise a runnable.
  `Executor` protocol = `forward(*xs) -> Any` only.
- `Sink.verify(ref, opt, inputs, *, rtol=1e-4, atol=None)` —
  module-level gate returning `VerifyResult` (`max_abs`/`max_rel`/
  `passed`); `TorchSink` delegates to `report.verify_module`. The
  host sink runs `ref` (torch executor or captured oracle outputs)
  vs `opt` (host executor) on identical inputs — **no second verify
  path** per plan Phase 2.
- `Sink.executors -> Mapping[str, ExecutorSpec]` — may be empty; the
  generic `lower` is the fallback. Phase 1 ships `executors = {}`
  (no carrier families); `ExecutorSpec(lower, accepts, engaged,
  carrier)` slots exist if scan/om-style executors are ever ported.

Plus the surrounding machinery that consumes it unchanged:
`Optimizer`/`search` (via `supported_ops`), `Compositional`
(backend-neutral), `BackendRegistry` (plan 0037 — the sink registers
there, not via import-time mutation).

## 4. Phase-1 implementation sketch

1. **Binding (host-tensor repo or new `kit-host-tensor` crate):**
   PyO3 extension exposing `call(op: &str, args: Vec<Value>) ->
   Value` over `tensor_op_registry()`; plus `ops() -> Vec<String>`.
   Alternative if PyO3 is unwelcome: C ABI over the same registry.
2. **Adapter (graphopt side, e.g. `packages/mlopt-host/` or inside
   `mlopt-torch` as `HostTensorSink`):**
   - `HOST_BINDINGS: dict[str, Binding]` — one entry per liftable op:
     encode IR args+attrs → positional JSON args (the bulk of Phase-1
     work is this attr→arg table; `aten_port_list.txt` pins the
     schema per op, and `lib-tensor/src/internal/aten.hnl` already
     encodes the same decoding rules in `.hnl` — a readable spec for
     the mapping).
   - Leaf binder: `Param`/`Const` values → `{"__tensor__": …}` wire
     records (upcast to f64, flatten row-major).
   - `HostTensorExecutor.forward(*xs)`: encode inputs → call op
     chain → decode result wire → return a backend tensor value
     (f64 ndarray or the wire dict — pick one and keep `verify`
     consistent).
   - `HostTensorSink(ops=OpTable…)` implementing the four members
     above; `executors = {}`.
   - `verify`: reuse `mlopt_core.ports.Verifier`/`VerifyResult`
     shape; compare host f64 outputs vs the torch-oracle outputs
     with configured `rtol/atol` (dtype-aware: f32 oracle vs f64
     candidate).
3. **Decline path:** any term containing an op outside
   `supported_ops` fails loudly at lower/eval (never silently
   approximates); the decline list above is the documented boundary.
4. **Measure (Phase 3 preview):** time `forward` vs torch eager on
   elementwise/reduce/linalg subterms; deliverable = `.hnl` artifact
   or the Rust-callable module — honest negative is valid.

## 5. Risks

- **Attr→arg mapping is the real work.** 137 liftable names ≠ 137
  done ops: each needs its IR attrs (`dim` tuple vs int, `keepdim`,
  `eps`, `correction`, index dtypes) mapped onto the generated
  port's decoder (`require_arg_range`/`opt_*` in `aten_generated.rs`).
  `aten_port_list.txt` + `lib-tensor/.../aten.hnl` give the spec but
  someone must write the table.
- **f64-only codec** caps dtype fidelity and has the non-finite
  `null` hole — both must be resolved or explicitly bounded before
  verify runs on real models.
- **No callable path exists** — Phase 1 cannot start lowering until
  the PyO3/C-ABI binding lands (small, but it's a hard dependency,
  and it lives in the host-tensor repo → cross-repo change).
- **catopt/`hnl-run` is not in this workspace** — the only existing
  executable harness is external; don't depend on it for tests.
- **lib-tensor `math.*` gap** means the .hnl oracle currently skips
  ~20 unary verdicts; host-side Rust impls are unaffected, but parity
  *evidence* for those ops is thinner than the "204 verdicts" headline
  (203 clean / 4 stubs per plan note; the skips are environment, not
  semantic).
- **`write`-tagged ops** (`index_add`, scatter family) blur
  functional/in-place lines — IR semantics must be checked per op
  before admitting them to `supported_ops`.
- **matmul/conv quality is measured, not assumed** (plan hard gate) —
  the naive Rust kernels may lose to eager/compile; Phase 3 decides.

## Appendix — reproduction

```python
from mlopt_core.opmeta import REGISTRY
import json
ir = sorted(REGISTRY)                      # 231
host = {l.strip() for l in open("host_ops.txt")}  # 238 tensor.* names
# sets per §1; liftable = overlap - NW_OVERLAP (16 names listed above)
```

Host op list source: `grep -o 'registry.register("tensor\.[a-z0-9_]*"'`
over `host-tensor/src/ops/mod.rs` (238 unique). IR list:
`mlopt_core.opmeta.REGISTRY` keys.
