# Phase B — device policy: what's proven vs hardware-pending

*Continuation of `docs/reports/device_policy.md` (the homogeneous
single-device slice, commit `95e9156`). This report covers the
extension: pipeline wiring, the sharded-parameter refusal surface,
and the per-device placement data layer. Companion substrate change
lives in `handl-lang/eqsat` (`DevicePolicy` port +
`Backend.device_policy` + `search`/`lower`/`Optimizer` kwargs).*

## Hardware limit (unchanged, honest)

This host has **one** GPU (RTX 2050, `cuda:0` only) and **no
multi-GPU**. Everything below that touches two physical devices is
data-structure + CPU/`meta`-tensor coverage only — the refusal and
census logic is proven; cross-device replay is **not claimed**.

## 1. `DevicePolicy.resolve` wiring — DONE (two vintages)

Before this change nothing in the optimizer path called
`DevicePolicy.resolve`: `CudaGraphRunner.apply` ran `check_module`
(refusal only), and `resolve`/`inspect` were library calls no entry
point consumed.

**Substrate (eqsat, backend-neutral).** `mlopt_core.ports` gains a
`DevicePolicy` protocol (`resolve(module, inputs) -> Any`,
`check_module(module) -> None`); the substrate stores no device type
— the resolved value is `str()`-rendered into stats. Wiring:

- `search(..., device_policy=...)` calls `resolve(model, x)` **once,
  before export** — a refusal fails before the e-graph is built —
  and records `stats["device"]`.
- `lower(..., device_policy=...)` runs the policy's `check_module`
  (duck-typed — a resolve-only policy stays valid) on the **delivered**
  module after `runner.apply`, then `resolve` →
  `stats["delivery_device"]`.
- `Backend` gains an optional `device_policy` field;
  `Optimizer` resolves it (explicit arg > backend) and forwards it
  through `search`/`lower`/`optimize`/`discover`. `Monolithic`
  picks it up automatically via the signature-derived kwarg split.
- `discover_alternatives` forwards it too.

**Domain (graphopt).** `TorchBackend(device_policy=...)` wires it
twice so the answer is consistent on either pin vintage:

- `Backend.device_policy` is set when the pinned `Backend` carries
  the field (activates automatically once the eqsat pin is bumped);
- `source` is wrapped in `DeviceCheckedSource`, a `Source` decorator
  that runs `policy.resolve(model, x)` inside `to_ir` — **this path
  works on the currently pinned eqsat (v0.1.0) today**, verified
  end-to-end (`Optimizer(backend=TorchBackend(device_policy=...))
  .optimize(...)` resolves once before export).

For the delivery side on the pinned substrate,
`DevicePolicyRunner` (a `Runner`) runs `check_module` + `resolve` on
the delivered module and records `stats["device_checked"]` /
`stats["device"]` — compose via `ChainedRunner`.

## 2. Sharded/distributed policy — refusal surface DONE

`mlopt_torch/device_placement.py` (new module; `device.py`
untouched):

- **`is_dtensor` / `is_sharded_param`** — DTensor detection by
  `isinstance` (`torch.distributed.tensor.DTensor`) plus the
  structural markers (`device_mesh` + `placements`) a DTensor cannot
  function without; FSDP `FlatParameter` by class name +
  `_local_shard`/`_full_param_padded`/`_mp_shard` markers.
- **`sharded_params_in` / `check_sharded_params`** — reports shard
  leaves *and* module-level FSDP markers (`_fsdp_state` /
  `_fsdp_wrapped_module`). The module arm is load-bearing: a
  world_size=1 FSDP module runs `NO_SHARD` and its leaves stay plain
  `Parameter`s — only the marker refuses it (proven with a real FSDP
  module over gloo, `tests/test_device_placement.py`).
- **`ShardedParamError`** — subclass of `UnsupportedDistributedError`
  with the actionable message: materialise (`DTensor.to_local()` /
  `full_tensor()`, FSDP `summon_full_params()`) or optimise per-rank.
- **`ShardedAwareDevicePolicy(DevicePolicy)`** — `check_module`
  runs the shard check *before* the base wrapper/collective checks
  so the more specific error wins. `CudaGraphRunner` now instantiates
  this policy — the CUDA-graph delivery path refuses shards too.

**Proven on CPU:** fake marker objects, a real `DTensor` (gloo
world_size=1 + `init_device_mesh("cpu", (1,))`), a real
`FullyShardedDataParallel` module. **Not proven:** world_size>1
sharding, CUDA FSDP, actual shard values — the refusal fires before
any of that matters.

## 3. Multi-GPU placement data structures — data layer DONE

`DeviceSlot` (canonical `type[:index]` + `torch.device.index`),
`PlacementPlan` (ordered dedup'd slots, `homogeneous`,
`for_device`/`__contains__`), `plan_placement(*devices,
homogeneous_only=...)` (empty → implicit `cpu`, matching
`DevicePolicy.resolve`; mixed + `homogeneous_only` →
`DevicePlacementError`), and `group_by_device(tensors)` — the
per-device map a multi-device delivery would iterate. All exercised
on `cpu`/`meta`/string devices; `"cuda"` without an index keeps
`index=None` honestly rather than guessing the current device.

`device_graph_key`'s `(device_index, signature)` outer key already
namespaces per-device graphs correctly for a future per-rank replay
(each rank's replica gets its own graph; no communication needed for
replicated delivery — per `device_policy.md`'s "what remains").

## 4. What remains hardware/pin-pending

- **Real multi-GPU validation** — needs a second physical GPU (or a
  CUDA MIG/vGPU slice): two-device `PlacementPlan` → per-device
  parameter copies → per-device `_DeviceGraphModule` instances.
  The structures and keys are in place; the replay path is not.
- **Per-rank replay** — N independent captures, one per rank's
  device. `CudaGraphRunner(multi_device=True)` already keys graphs
  per device; sharding the *parameters* per rank is unbuilt.
- **Collectives** — still refused, never claimed.
- **eqsat pin bump** — `Backend.device_policy` /
  `Optimizer.device_policy` activate when graphopt's
  `[tool.uv.sources]` tag moves past the commit carrying them;
  until then `TorchBackend`'s source-wrap is the live path (the
  `__dataclass_fields__` guard makes the transition automatic).

## Tests

- eqsat `tests/test_device_policy_hook.py` (10): protocol
  conformance, search/lower/optimize/discover forwarding, stats keys,
  refusal propagation, resolve-only duck-typing, no-policy identity.
- graphopt `tests/test_device_placement.py` (20): shard detection
  (fakes + real DTensor + real FSDP over gloo), `ShardedParamError`
  taxonomy, policy subclass ordering, placement structures on
  cpu/meta, `DeviceCheckedSource`/`DevicePolicyRunner`, and the
  `TorchBackend(device_policy=...)` wrap (both pin vintages).
