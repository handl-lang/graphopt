# Production readiness — graphopt optimizer

*Audit by read-only agent; saved verbatim by parent.*

## Verdict

The optimizer has a credible production-shaped core:

- a backend-neutral `Optimizer` / `search` / `lower` / `Strategy` / `Runner` pipeline;
- a Torch adapter built on `torch.export`;
- per-block `Compositional` optimization;
- measured `Autotuned` delivery;
- a `.pt2` export path that is genuinely intended to run without mlopt;
- an existing clean-interpreter test proving the `.pt2` claim on CPU.

It is **not yet production-packaged end to end**:

- the repository is still a `0.1.0dev` uv workspace split across multiple repos;
- the frozen public API covers substrate/orchestrator/game contracts, but not the Torch adapter/export surface;
- `.pt2` is the only broadly useful standalone runnable artifact today;
- safetensors/state-dict artifacts are weights-only and layout-specific;
- device handling is operand-driven and effectively single-device per lowered module;
- CUDA graphs are single-capture-context and have no device-mesh/distributed story;
- the joint forward+backward work is a real research result, but it is not a general autograd/training API;
- there is no committed serving-stack integration such as vLLM/TGI/ORT deployment.

The highest-leverage next step is **Phase A: make the install/artifact contract explicit and tested**. The core `.pt2` claim already exists and has a clean-interpreter test; production readiness needs the same rigor around installability, artifact schema/versioning, dynamic shapes, dtype/device matrices, and documentation.

## Key findings (condensed — full detail in agent transcript)

### What exists

- `Optimizer.optimize(model, x)` → `search`/`lower` through port-based `Source`/`Sink`/`Executor`/`Runner`/`Strategy`/`Meter`/`Verifier` contracts (`mlopt_core.ports`).
- Torch lifting: `nn.Module` → `torch.export.export` → `ExportedProgram` graph → mlopt IR + `param_values` (`mlopt_torch/torch_bridge.py:export_to_ir`).
- `IRModule` plan-tape lowering; params/buffers preserved; `_align_to` migrates tensors toward computation device/dtype.
- Strategies: `Monolithic`, `Compositional` (per-block hooks + graft + e2e verify/rollback), `Autotuned` (measured winner).
- Runners: `IdentityRunner`, `ChainedRunner`, `TorchCompileRunner`, `CudaGraphRunner` (device-coupled, single capture context).
- Export: `fmt="module"/"pt2"` → standalone `ExportedProgram`; `"torchscript"` (less capable); `"safetensors"`/`"state_dict"` weights-only. Sibling manifest records format/torch version/`runnable_standalone`/`weights_only`.
- Tests already prove `.pt2` clean-interpreter round-trip, dynamic shapes, kwargs (`tests/test_export.py`).
- Device: operand-driven, no literal `cuda:0`; but no multi-device policy — `_align_to` is one-device-at-a-time, `TorchMeter` syncs globally, `CudaGraphRunner` has no per-device cache.
- Autograd: joint fwd+bwd domain real (+13% vs `torch.autograd`, verified) but partial VJP table, not integrated into `Optimizer.optimize`, not a training API.
- Packaging: all `0.1.0dev`, git+ssh subdir deps, no wheel CI; frozen API covers substrate/orchestrator/game but NOT `mlopt_torch`/`export_optimized`/runners.

### Gaps → phases

- **Phase A (packaging/artifact contract, M+S–M+S):** real version policy + wheel/install smoke; manifest `artifact_schema`/`mlopt_version`; `.pt2` documented as THE runnable artifact; clean-interpreter matrix (CPU fp32/fp64, dynamic shapes, kwargs); freeze-or-label-unstable the Torch/export API.
- **Phase B (multi-device, M–L):** `DevicePolicy` layer; per-device CUDA-graph cache keyed by device+signature; explicit refusal of DDP/FSDP/TP wrappers; per-rank independent replay first, collectives never claimed.
- **Phase C (serving/autograd, M–L/L):** decide canonical serving artifact (`.pt2` torch-native vs ONNX portable vs vLLM adapter — none committed today); expand `lawdata.REVERSE` VJP coverage + gradcheck parity + decide the training-artifact product shape.

### Highest-leverage next step

**Phase A — install/artifact contract.** Validates the strongest existing claim (`.pt2` runs without mlopt) in a clean env; CPU-testable, no GPU needed; every later phase depends on a stable artifact boundary.

First deliverable: clean-interpreter artifact matrix test + manifest schema/version + documented install path + frozen-vs-unstable Torch API decision.
