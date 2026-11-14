# Device policy — Phase B first honest slice

*Implementation notes for `docs/reports/production_readiness.md` Phase B
(multi-device). Scope delivered: homogeneous single-device per process —
normalize/validate placement, per-device CUDA-graph cache, explicit
distributed refusal. NOT delivered: collectives, sharded params,
per-rank replay, true multi-GPU serving.*

## Hardware limit (honest)

This host has **one** GPU (`nvidia-smi`: RTX 2050, `cuda:0` only).
Two-device placement could not be exercised end-to-end. The
multi-device logic is covered by:

- `meta`-device tensors for the placement census (`cpu` vs `meta`
  exercises every branch of `DevicePolicy.inspect`/`resolve` without
  hardware);
- `device_graph_key` unit tests pinning the `(device_index,
  signature)` key shape — the index is the outer key, so `cuda:0` and
  `cuda:1` graphs can never alias by construction;
- real CUDA-graph capture/replay tests on `cuda:0`
  (`tests/test_runners.py`, `requires_cuda`).

A second physical GPU is still needed to prove cross-device replay —
that remains an open verification item, not a claimed one.

## What landed

### `mlopt_torch/device.py` (new)

- **`DevicePolicy`** — `inspect(module, inputs) -> DeviceReport`
  censuses every tensor device across `parameters()`/`buffers()`/
  `_param_map`/`_param_values` plus input leaves; `resolve(...) ->
  torch.device` answers the one canonical device:
  - no tensors → `cpu`; one distinct device → that device;
  - mixed devices → `DevicePlacementError` naming the devices and the
    honest fix (`.to(device)` before *or* after optimise — never mix),
    or `cpu` under `cpu_fallback=True` (names the target only; moving
    tensors stays the caller's job — the policy is side-effect free).
- **`distributed_wrapper_name` / `DevicePolicy.check_module`** —
  refusal, not approximation: torch `DistributedDataParallel` /
  `FullyShardedDataParallel` by `isinstance`, third-party wrappers by
  group-handle attributes (`process_group`, `device_mesh`, `tp_group`,
  …) and class-name fragments (`TensorParallel`, `PipelineParallel`,
  …). Modules binding collective ops (`_torch_bindings` keys matching
  `is_collective_op`) are refused too. Raises
  `UnsupportedDistributedError`.
- **`is_collective_op` / `COLLECTIVE_OP_NAMES`** — deliberately
  conservative: `all_reduce`/`all_gather`/`reduce_scatter`/`all_to_all`/
  `barrier`/`send`/`recv` + `dist_`/`c10d_`/`nccl_` prefixes.
  `broadcast`/`gather`/`scatter`/`reduce` are **excluded** — they are
  real `_IR_TO_TORCH` tensor ops, not collectives (verified against
  the live binding table).
- **`device_graph_key(xs)`** — `(device_index, ((shape…), dtype)…)`
  or `None` unless every positional arg is a CUDA tensor on the SAME
  device. `None` = honest "do not capture" (CPU, mixed-device,
  non-tensor args).
- **`storage_fingerprint(module)`** — untyped-storage identity tuple
  over `_module_tensors`. This is the CUDA-graph staleness check:
  `Module.to()` swaps `param.data` and `IRModule._align_to` /
  `_park_params` rebind `_param_map`/`_param_values` to fresh tensors
  *without* changing Parameter-object identity — only storage identity
  catches every real move. Holding the storages also keeps captured
  memory alive (no free-then-realloc ABA).
- **`cuda_device_indices(*groups)`** — sorted distinct CUDA device
  indices; the meter's per-device sync set.

Names are re-exported lazily through `mlopt_torch.__init__`'s
`_DELEGATED` table.

### `mlopt_torch/meter.py`

`TorchMeter.time` now syncs **per device**: `cuda_device_indices`
collects the devices the inputs occupy and each gets its own
`torch.cuda.synchronize(idx)` — the same wait the global call gave,
without blocking on GPUs the measurement never touched. `TimingResult.
device` still reports the first tensor's device (informational).

### `mlopt_cuda/runners.py`

- `CudaGraphRunner(multi_device=True)` routes the generic path
  through **`_DeviceGraphModule`** — a lazy per-`(device,
  signature)` capture cache. One `_GraphReplayModule` per key, never
  shared across devices; capture runs inside
  `torch.cuda.device(idx)` on first matching call. Non-CUDA /
  mixed-device / non-tensor inputs take the eager path (same refusal
  as the single-capture path). Executor modules keep their own
  `capture_cuda_graph` — the flag does not reach them.
- **Staleness guard**: each entry snapshots `storage_fingerprint`;
  before every replay the fingerprint is recomputed and any mismatch
  drops the entry and recaptures against current storage. This is
  load-bearing: CUDA graphs hold raw pointers and do not keep tensors
  alive, and `_align_to`/`_park_params`/`Module.to()` all rebind
  storage.
- **Honest stats**: capture is lazy, so `stats["cuda_graph"]=True` +
  `stats["cuda_graph_per_device"]=True` means *armed*, not yet
  captured. Per-key capture failures land in
  `wrapped.capture_failures` (classified via
  `mlopt_core.failures.classify`) and that key serves eager forever —
  no capture storm.
- **Refusal**: `apply` runs `DevicePolicy.check_module` before every
  early-out — DDP/FSDP/TP wrappers and collective-binding modules
  raise `UnsupportedDistributedError` on CPU *and* CUDA inputs.
- `mlopt-cuda` now declares `mlopt-torch` (the import-linter layer
  contract already placed `mlopt_cuda` above `mlopt_torch`; the dep
  was previously undeclared).

## Tests

- `tests/test_device_policy.py` (new, CPU-safe): collective
  vocabulary incl. the tensor-op exclusions, `iter_tensors` depth,
  wrapper detection (attr markers, class fragments, **real DDP over
  gloo world_size=1**), `check_module` refusals, the full
  inspect/resolve matrix on cpu/meta, `storage_fingerprint`
  stability + `_param_map`-rebind detection, `cuda_device_indices`,
  `device_graph_key`.
- `tests/test_runners.py` (e2e file, extended): refusal on CPU input,
  multi_device CPU noop, `_DeviceGraphModule` eager/fingerprint paths,
  and `requires_cuda` end-to-end capture — per-signature keys (two
  shapes → two graphs, never shared), replay correctness vs eager,
  stale-graph drop + recapture on param rebind, capture-failure →
  eager-forever.

## What remains (Phase B exit criteria, per the report)

- **Collectives**: never claimed. Refused explicitly; a real
  implementation needs per-rank graph instances + collective-aware
  capture — out of scope.
- **Sharded parameters** (FSDP/TP): refused explicitly. Needs
  per-rank param tables — the `_param_map`/`_param_values` seam is
  the natural hook but is unbuilt.
- **Per-rank replay**: not implemented. The honest first step would
  be N independent `_DeviceGraphModule` instances (one per rank's
  device), each capturing its own replica — no communication, just
  replicated delivery. `device_graph_key`'s device index already
  namespaces the graphs correctly for this.
- **True multi-device serving**: one module instance still holds ONE
  parameter set; `_align_to` migrates it per call (params ping-pong
  across devices — correct but slow, and each switch invalidates +
  recaptures graphs). Real multi-device wants per-device parameter
  copies or per-device module replicas — the fingerprint machinery
  already detects and survives the migration case.
- **`IRModule._parameters` vs `_param_map` divergence** (pre-existing,
  surfaced by `_park_params`): `_align_to`/`_park_params` rewrite the
  live tables but never the registered `_parameters`, so
  `module.parameters()`/`state_dict()`/`Module.to()` see the stale
  pre-align clones. `storage_fingerprint` reads all three tables so
  the graph guard is correct regardless, but the divergence itself is
  a separate correctness question for the torch_bridge owner.
- **Executor path**: batched executors' `capture_cuda_graph` remains
  single-device single-capture; giving them the per-device cache is
  follow-up work in `mlopt_carriers`.
- **`.to(device)` ergonomics**: `DevicePolicy.resolve` names the
  target but does not move tensors; wiring `resolve` into
  `Optimizer.optimize`/`TorchBackend` (e.g. auto-`.to()` the example
  inputs or refuse early) is a pipeline-integration decision left to
  the orchestrator owners.
