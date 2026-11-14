# Phase C — serving/autograd surface (graphopt slice)

*Implementation report for `docs/reports/production_readiness.md`
Phase C. Scope delivered: the canonical-artifact decision (documented,
not just asserted), the measured dynamic-shape matrix, and the
measured autograd boundary — all reproducible through
`bench/drivers/probes/phase_c_serving_probe.py` (`--cuda` for the GPU
cells) and pinned by `tests/test_phase_c_serving.py`.*

**Verdict up front.** `.pt2` is confirmed as *the* canonical serving
artifact — nothing measured contradicts it. ONNX is not a post-hoc
conversion of a delivered module; it is a **different sink choice at
optimize time** (`Optimizer(source=TorchSource(), sink=OnnxSink())`),
and it works end-to-end today: verified optimize → `OnnxExecutor.save`
→ fresh `onnxruntime` session reproduces the forward bit-close. It is
static-shape and, in this environment, CPU-provider only. Autograd is
better than the audit feared — delivered modules fine-tune on their
build device/dtype — but `.pt2` is inference + input-grad only, and
one real boundary (`_align_to` rebind) silently hides parameter grads.

## 1. Serving artifact decision

### Decision: `.pt2` is canonical; ONNX is a supported alternative *pipeline output*, not a conversion

| Artifact | Runnable standalone | Dynamic shapes | Autograd | Serving-stack fit | Verdict |
|---|---|---|---|---|---|
| `.pt2` (`fmt="module"`, default) | yes — `torch.export.load`, no mlopt | yes, via `dynamic_shapes` | input-grads only (params baked) | torch-native: `torch.compile`-adjacent, AOTInductor, `torch.export` consumers | **canonical** |
| `torchscript` (`fmt="torchscript"`) | yes | no (traced/static) | traced `IRModule` keeps param autograd; carrier executors untraceable | deprecated API upstream (`torch.jit` FutureWarnings on 2.14) | legacy; trace-only |
| `safetensors` / `state_dict` | no — weights only | n/a | n/a | weight hand-off, not serving | unchanged (already `artifact_kind:"weights"`) |
| ONNX (`OnnxSink`) | yes — `OnnxExecutor.save` → any ORT runtime | **no — static** | no | portable: ORT/TensorRT/TVM class runtimes | supported alternative, not canonical |

Rationale, measured not asserted:

- **`.pt2` is the only artifact that is both verified-at-export and
  torch-native-dynamic.** `export_optimized(..., verify=True)` reloads
  and re-runs the forward before writing the manifest
  (`mlopt-artifact/1`), and `tests/test_export.py`'s clean-interpreter
  matrix already proves CPU fp32/fp64 × static/dynamic × kwargs in a
  mlopt-free process. No other format has that contract.
- **ONNX exists but at a different seam.** `OnnxSink` lowers IR →
  `onnx.ModelProto`; `OnnxExecutor.save(path)` is already the delivery
  step (its docstring says so). The probe confirms the full loop:
  `Optimizer(source=TorchSource(), sink=OnnxSink()).optimize(mod, x)`
  → `verified=True` → `save()` → fresh `onnxruntime.InferenceSession`
  on the *file* reproduces the eager forward (`atol=1e-5`). No new
  code was needed — the "serving-stack integration path" the audit
  flagged as missing is `sink=OnnxSink`, already shipped (ADR 0006).
  What is *not* possible today: ONNX-exporting an already-delivered
  torch module — `IRModule`/carrier executors do not retain their IR,
  and `torch.onnx.export` on a loaded `.pt2` fails in this env with
  `ModuleNotFoundError: onnxscript` (torch 2.14's dynamo exporter wants
  a dep mlopt does not declare). If that path is wanted it needs an
  `onnxscript` dep decision, not code.
- **TorchScript is trace-only and deprecated.** `torch.jit.trace`
  works on `IRModule` (forward matches, and — measured — param autograd
  survives tracing), `torch.jit.script` fails on the `forward(*xs)`
  varargs signature, and carrier executors' `*args` forwards are not
  traceable (the `export.py` docstring already says this). torch 2.14
  emits `FutureWarning: torch.jit.* is deprecated` on both calls.
  Kept as `fmt="torchscript"` for compat; not a serving recommendation.
- **GGUF/Ollama/LM Studio** remain out of scope by design — the
  manifest `notes` already say the honest bridge is converting the
  *original* model via llama.cpp.

**`--serve` wiring: deliberately not added.** The ONNX route needs no
new flag — it is `Optimizer(sink=OnnxSink())` plus
`result.module.save(path)`. A `--serve onnx` CLI would have to
re-plumb the sink choice through `mlopt_discovery.play`'s deliver hook,
which currently hardcodes `TorchSink` for the torch domain
(`play.py:_torch_deliver`). That is a real but small follow-up; the
capability itself is shipped and verified. Documented here rather than
half-wired.

## 2. Dynamic-shape matrix — measured

"Dynamic" means: run a shape different from the optimize-time example
without re-exporting. Measured on the probe's MLP (monolithic) and
MiniGPT (compositional):

| Form | Off-example batch | Mechanism / caveat |
|---|---|---|
| `.pt2` static (default export) | **refused** — `Guard failed: xs_0.size()[0] == 2` | shape-specialized at export; expected |
| `.pt2` + `dynamic_shapes` | **works** — batches 2/5/7 all correct | requires the extra pytree nesting for `*xs` forwards: `dynamic_shapes=(({0: Dim("batch")},),)` (pinned by `test_export.py` and re-measured here). Export itself can still fail if the module baked shape-dependent constants |
| eager delivered module (`IRModule` plan tape) | **per-module** — MLP: batch 5 fine; MiniGPT compositional: batch 3 fails (`shape '[1,8,2,16]' is invalid`) | the plan tape is shape-generic, but `_fold_weight_chains`/const minting bake tensors at build shapes; anything shape-dependent that got folded is frozen. **Not a supported dynamic contract** — treat delivered eager modules as example-shaped unless measured |
| ONNX (`OnnxSink`) | **refused** — ORT `INVALID_ARGUMENT` on batch 5 vs declared 2 | input `ValueInfo` dims come from the IR's static `TensorType` shapes; the IR *can* name `None` extents but `lower_ir` does not emit `dim_param`s today |
| `CudaGraphRunner` | static by construction | `device_graph_key` includes the full shape tuple; a new shape is a new capture, never an alias (Phase B) |
| `torch.compile` twin | unexplored this phase | Inductor handles dynamo in general; not measured against delivered modules |

**Capability reporting.** `mlopt_core.ports.Capabilities` exists but is
search-facing only — `supported_ops`/`ops` bound what the *search* may
select; it says nothing about shapes, dtypes, devices, or autograd.
`mlopt_torch.report` is the verify/stats surface (owned elsewhere this
phase). No serving-capability surface was added: the honest capability
report *is* this matrix plus the probe. If a `ServingCapabilities`
report is wanted later, the natural home is a sibling of
`DevicePolicy` in `mlopt_torch` — declared per artifact format, not
per module instance.

## 3. Autograd — measured boundary

Question: can a delivered module run under `torch.autograd` for
fine-tuning? Answer: **yes on its build device/dtype; with two sharp
edges.** All numbers from the probe / `tests/test_phase_c_serving.py`.

- **Delivered module (CPU fp32, MLP):** `opt(x).square().sum().backward()`
  — input grad matches the eager original *exactly* (max abs diff
  0.0), and all 4 fused parameters collect nonzero grads. The
  weight-chain fold changes the parameterization (fused Params) but
  not differentiability — grads land on the fused params an optimizer
  sees via `module.parameters()`.
- **Compositional delivery (MiniGPT, CPU):** backward succeeds;
  14/14 parameters collect grads.
- **CUDA, module built on CUDA:** same result (params already aligned,
  `_align_to` no-ops).

### Boundary 1 — `.pt2` is not a fine-tuning artifact

`torch.export.load(...).module()` backward *runs* and the **input**
grad matches eager exactly — but every parameter reports `grad=None`:
the exported program bakes params as constants. Usable for
input-space optimization (e.g. prompt/activation tuning), **not** for
training the delivered weights. If a training artifact is ever a
product, it is the live `nn.Module`, not the `.pt2`.

### Boundary 2 — `_align_to` rebind hides parameter grads (the sharp edge)

`IRModule._align_to` (`torch_bridge.py:3527`) moves the live weight
tables (`_param_map`/`_param_values`) to the inputs' device/dtype on
first eval — but only off-CPU. When it actually *moves* tensors it
creates **new** `torch.nn.Parameter` objects stored only in those
dicts; `module.parameters()` keeps yielding the registered
`_parameters` entries, which are now stale.

Measured (CUDA available): optimize on CPU → call with CUDA inputs
*without* `.to()` → forward+backward gives correct input grads, the
live tables hold 4/4 param grads, `module.parameters()` reports 0/4 —
an optimizer stepping `module.parameters()` would update tensors the
forward never reads. **The fix is caller-side and cheap:** `.to(device)`
*before* serving (Parameter `.to()` mutates `.data` in place, keeping
`_param_map`/`_parameters` identity — verified: after `.cuda()` the
tables and registered params are the same objects and grads land
correctly), or build/optimize on the target device. This is a
documented boundary, not a bug fix — the rebind exists for the
verify/park paths and changing it touches `torch_bridge.py` (owned
elsewhere this phase).

### Boundary 3 — dtype is build-fixed

`_align_to` early-returns on CPU (`like.device.type == "cpu"`), so a
CPU-delivered module serves exactly the dtype it was built at — an
fp64 input against fp32 params fails inside `F.linear`, same as stock
torch. `opt.to(dtype)` works (in-place `.data` swap, identity kept).

### What autograd does NOT cover (honest remainder)

- No training loop, optimizer, or LR surface — this is "the module is
  differentiable", not a training API.
- No gradcheck/grad-parity *suite* — parity was measured on one MLP
  and one MiniGPT, fp32 CPU + the CUDA cells above.
- Carrier executors (`BatchedScanModule` etc.) got one backward
  measurement through MiniGPT; the level-batched paths are untested
  for second-order or exotic dtypes.
- `torch.compile`d delivered modules under autograd: unmeasured.
- Second-order gradients: unmeasured (plan-tape eval is plain torch
  ops, so nothing obvious blocks them, but nothing verifies them).

## 4. What remains for full Phase C

1. **Serving manifest for ONNX** — `OnnxExecutor.save` writes a bare
   `.onnx`; there is no `mlopt-artifact/1`-style sidecar recording
   opset/dtype/verify results the way `export_optimized` does. A
   `save_optimized_onnx`-style wrapper (or a documented convention)
   would close the contract asymmetry.
2. **Dynamic ONNX** — `lower_ir` would need `dim_param` emission for
   `None` extents plus shape-generic builders; the IR already
   represents unknown dims.
3. **`torch.onnx.export` from `.pt2`** — blocked on an `onnxscript`
   dependency decision (torch 2.14 dynamo exporter), or accept the
   OnnxSink route as the only ONNX path.
4. **Autograd hardening** — a grad-parity test matrix (dtypes ×
   devices × strategies × carrier executors), a decision on whether
   `_align_to`'s rebind should re-register Parameters (would fix
   Boundary 2 at the source), and an explicit "inference artifact vs
   trainable module" statement in the export docs (`.pt2` =
   inference; the live module = the only training surface).
5. **vLLM/TGI-class integration** — still nothing committed; the
   honest near-term story is `.pt2` for torch-native serving and
   `OnnxSink` for ORT-class runtimes. A real LLM-serving adapter is
   its own project.

## Files

- `bench/drivers/probes/phase_c_serving_probe.py` — the measurement
  driver (artifact paths, dynamic-shape matrix, autograd cells;
  `--cuda` for the GPU cells). Prints a JSON report.
- `tests/test_phase_c_serving.py` — pins the load-bearing claims:
  delivered-module autograd parity, `.pt2` input-grad-only, and the
  `_align_to` rebind boundary (`requires_cuda`).
- This document.
