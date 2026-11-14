"""DevicePolicy — honest single-device placement for mlopt-torch.

Phase B of ``docs/reports/production_readiness.md`` asks for a device
layer on top of the operand-driven status quo.  This module is the
first honest slice: **homogeneous single-device per process**.  It does
not shard parameters, does not replay per-rank, and does not claim
collectives — it *normalises and validates* placement, names the mixed
device case explicitly, and refuses distributed wrappers outright.

What it provides:

* :class:`DevicePolicy` — inspect a module + example inputs, report
  the device census (:class:`DeviceReport`), resolve the one canonical
  device the delivery may run on, or raise
  :class:`DevicePlacementError` when tensors span devices.
* :func:`distributed_wrapper_name` — detect DDP / FSDP / tensor-
  parallel style wrappers (torch built-ins by ``isinstance``, third-
  party wrappers by class-name and group-handle markers) so callers
  can refuse loudly instead of silently optimising a wrapped module.
* :func:`is_collective_op` / :func:`collective_ops_in` — name the IR
  op vocabulary that implies cross-rank communication.  Deliberately
  conservative: ``broadcast``/``gather``/``scatter``/``reduce`` are
  *real* tensor ops in ``_IR_TO_TORCH`` and are NOT flagged.
* :func:`device_graph_key` / :func:`cuda_device_indices` — the cache
  keys and sync sets the CUDA runner/meter consume.

Placement rules (``DevicePolicy.resolve``):

* no tensor operands → ``cpu``;
* exactly one distinct device → that device (``cpu`` included);
* mixed devices → ``cpu_fallback=True`` resolves ``cpu`` (the caller
  still has to move the tensors — the policy only names the target);
  otherwise :class:`DevicePlacementError`.  The message states the
  honest fix: place the module *or* the inputs on one device before
  optimising — ``.to(device)`` before ``optimize`` and after are both
  valid, mixing is not.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "COLLECTIVE_OP_NAMES",
    "DevicePlacementError",
    "DevicePolicy",
    "DeviceReport",
    "UnsupportedDistributedError",
    "collective_ops_in",
    "cuda_device_indices",
    "device_graph_key",
    "distributed_wrapper_name",
    "is_collective_op",
    "iter_tensors",
    "storage_fingerprint",
]


class DevicePlacementError(ValueError):
    """Raised when operands span devices the policy cannot honour."""


class UnsupportedDistributedError(DevicePlacementError):
    """Raised for DDP/FSDP/TP-wrapped modules or collective ops.

    Phase B's honest slice refuses rather than approximates: a
    distributed wrapper's parameters are replicated or sharded across
    ranks, and replaying its forward on one rank produces silently
    wrong results.  The error names what was detected.
    """


# IR op names that unambiguously imply cross-rank collectives.  The
# single-word tensor ops ``broadcast``/``gather``/``scatter``/``reduce``
# are EXCLUDED on purpose — they are ordinary ``_IR_TO_TORCH`` bindings,
# not ``torch.distributed`` collectives; flagging them would refuse
# legitimate single-device graphs.
COLLECTIVE_OP_NAMES = frozenset(
    {
        "all_reduce",
        "allreduce",
        "all_gather",
        "allgather",
        "all_to_all",
        "alltoall",
        "all_to_all_single",
        "reduce_scatter",
        "reducescatter",
        "barrier",
        "send",
        "recv",
    }
)

# Prefixes that mark backend collectives regardless of spelling
# (``dist.all_reduce`` lowers differently per backend; ``nccl_*`` /
# ``c10d_*`` are the process-group entry points).
_COLLECTIVE_PREFIXES = ("dist_", "c10d_", "nccl_")


def is_collective_op(name: str) -> bool:
    """Return True when IR op *name* implies a cross-rank collective."""
    n = name.lower()
    return n in COLLECTIVE_OP_NAMES or n.startswith(
        _COLLECTIVE_PREFIXES
    )


def collective_ops_in(names: Iterable[str]) -> list[str]:
    """Return the collective ops in *names*, sorted."""
    return sorted(n for n in names if is_collective_op(n))


def iter_tensors(obj: Any, *, depth: int = 1) -> Iterable[torch.Tensor]:
    """Yield tensor leaves of *obj* — itself, or one level of tuple/list.

    ``depth`` bounds the nesting walk so a deeply nested pytree does
    not turn device inspection into a traversal of the whole input.
    """
    if isinstance(obj, torch.Tensor):
        yield obj
    elif depth > 0 and isinstance(obj, (tuple, list)):
        for item in obj:
            yield from iter_tensors(item, depth=depth - 1)


def _module_tensors(module: Any) -> list[torch.Tensor]:
    """Tensor leaves of a module's params/buffers/param_values.

    Covers ``nn.Module.parameters()``/``buffers()`` (registered
    tensors), ``IRModule._param_map`` — the table ``_eval`` actually
    reads, which ``_align_to`` rebinds independently of the registered
    ``_parameters`` — plus ``IRModule._param_values`` entries that
    never registered as Parameters (the exporter can bake tensors
    there that ``_align_to`` still migrates).  Duplicates are fine:
    the fingerprint only compares element-wise.
    """
    out: list[torch.Tensor] = []
    for getter in ("parameters", "buffers"):
        fn = getattr(module, getter, None)
        if callable(fn):
            out.extend(fn())
    for table_name in ("_param_map", "_param_values"):
        table = getattr(module, table_name, None)
        if isinstance(table, dict):
            out.extend(
                v for v in table.values() if isinstance(v, torch.Tensor)
            )
    return out


def _torch_class(name: str) -> type | None:
    """Resolve a torch distributed wrapper class by attribute path."""
    obj: Any = torch
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, type) else None


_DISTRIBUTED_CLASS_PATHS = (
    "nn.parallel.DistributedDataParallel",
    "distributed.fsdp.FullyShardedDataParallel",
)

# Attribute markers distributed wrappers carry — a process group or
# device mesh handle never appears on an ordinary module.
_DISTRIBUTED_ATTR_MARKERS = (
    "process_group",
    "device_mesh",
    "tp_group",
    "tensor_model_parallel_group",
    "mp_group",
)

# Class-name fragments for third-party wrappers (Megatron-style TP,
# fairscale, composer) that have no torch builtin to isinstance.
_DISTRIBUTED_NAME_FRAGMENTS = (
    "DistributedDataParallel",
    "FullyShardedDataParallel",
    "TensorParallel",
    "TPParallel",
    "PipelineParallel",
)


def distributed_wrapper_name(module: Any) -> str | None:
    """Name the distributed wrapper *module* is, or ``None``.

    Detection is deliberately layered: torch's own wrappers by
    ``isinstance`` (exact), then group-handle attributes any wrapper
    must carry to function, then class-name fragments for wrappers
    with no common base.  A false positive is possible on a module
    that happens to expose ``process_group`` — the honest failure is
    a refusal the user can override by unwrapping.
    """
    for path in _DISTRIBUTED_CLASS_PATHS:
        cls = _torch_class(path)
        if cls is not None and isinstance(module, cls):
            return path.rsplit(".", 1)[-1]
    for attr in _DISTRIBUTED_ATTR_MARKERS:
        if getattr(module, attr, None) is not None:
            return f"module exposing '{attr}'"
    cls_name = type(module).__name__
    for frag in _DISTRIBUTED_NAME_FRAGMENTS:
        if frag in cls_name:
            return cls_name
    return None


@dataclass(frozen=True)
class DeviceReport:
    """The device census of one (module, inputs) pair.

    ``device`` is the single canonical device when ``homogeneous``
    (``"cpu"`` when there are no tensor operands at all); it is ``""``
    for a mixed placement, where ``devices`` lists every distinct
    ``str(device)`` found, sorted.
    """

    device: str
    homogeneous: bool
    devices: tuple[str, ...]
    n_tensors: int
    policy: str  # "homogeneous" | "mixed" | "empty"


class DevicePolicy:
    """Normalise/validate device placement — homogeneous slice only.

    ``cpu_fallback=True`` makes :meth:`resolve` answer ``cpu`` for a
    mixed placement instead of raising.  It names the target only —
    actually moving tensors stays the caller's job (``module.to`` /
    ``tensor.to``), keeping the policy side-effect free.
    """

    def __init__(self, *, cpu_fallback: bool = False) -> None:
        """``cpu_fallback``: mixed placement resolves ``cpu``."""
        self.cpu_fallback = cpu_fallback

    # -- refusal -----------------------------------------------------

    def check_module(self, module: Any) -> None:
        """Refuse distributed wrappers and collective-op modules.

        Raises :class:`UnsupportedDistributedError` — a clear error,
        never a silent wrong delivery.  Modules without torch
        distributed machinery pass untouched.
        """
        wrapper = distributed_wrapper_name(module)
        if wrapper is not None:
            raise UnsupportedDistributedError(
                f"refusing to optimise {wrapper!r}: distributed "
                "wrappers (DDP/FSDP/TP) shard or replicate parameters "
                "across ranks; Phase B supports homogeneous "
                "single-device modules only — unwrap the module or "
                "optimise each rank's replica independently"
            )
        bindings = getattr(module, "_torch_bindings", None)
        if isinstance(bindings, dict):
            bad = collective_ops_in(bindings)
            if bad:
                raise UnsupportedDistributedError(
                    f"refusing to optimise module binding "
                    f"collective op(s) {bad}: cross-rank collectives "
                    "are outside the single-device slice"
                )

    # -- census ------------------------------------------------------

    def inspect(
        self, module: Any = None, inputs: Any = None
    ) -> DeviceReport:
        """Census the tensor devices of *module* + *inputs*.

        Distributed wrappers raise via :meth:`check_module` first —
        a wrapped module's device census is meaningless because its
        parameters are not where the math runs.
        """
        if module is not None:
            self.check_module(module)
        mod_ts = _module_tensors(module) if module is not None else []
        in_ts = list(iter_tensors(inputs)) if inputs is not None else []
        devices = {
            str(t.device) for t in itertools.chain(mod_ts, in_ts)
        }
        n_tensors = len(mod_ts) + len(in_ts)
        devices_sorted = tuple(sorted(devices))
        if not devices_sorted:
            return DeviceReport("", True, (), n_tensors, "empty")
        if len(devices_sorted) == 1:
            return DeviceReport(
                devices_sorted[0],
                True,
                devices_sorted,
                n_tensors,
                "homogeneous",
            )
        return DeviceReport(
            "", False, devices_sorted, n_tensors, "mixed"
        )

    # -- resolution --------------------------------------------------

    def resolve(
        self, module: Any = None, inputs: Any = None
    ) -> torch.device:
        """Resolve the one device this delivery may run on.

        Homogeneous placements resolve to their device; empty
        placements resolve ``cpu``; mixed placements resolve ``cpu``
        under ``cpu_fallback`` and raise
        :class:`DevicePlacementError` otherwise.  The error names the
        devices found and the honest fix — ``.to(device)`` the module
        or the inputs *before* optimising (or after, on the delivered
        module), never mix.
        """
        rep = self.inspect(module, inputs)
        if rep.homogeneous:
            return torch.device(rep.device or "cpu")
        if self.cpu_fallback:
            return torch.device("cpu")
        raise DevicePlacementError(
            f"mixed-device operands {list(rep.devices)}: this slice "
            "optimises one homogeneous device per module — place the "
            "module and inputs on a single device with .to(device) "
            "before optimising (or .to(device) the delivered module "
            "afterwards); per-device parameter copies and per-rank "
            "replay are not implemented"
        )


def storage_fingerprint(module: Any) -> tuple:
    """Untyped-storage identity tuple over a module's tensors.

    The CUDA-graph safety check (Phase B): a captured graph holds raw
    pointers to the storages live at capture time.  ``Module.to()``
    swaps ``param.data`` in place and ``IRModule._align_to`` rebinds
    ``_param_map``/``_param_values`` to fresh tensors — neither changes
    the *Parameter object's* identity, so an ``is`` check on parameters
    misses both.  The untyped storage does change on every real move,
    and holding the storage objects in the snapshot keeps the captured
    memory alive (no free-then-realloc ABA).  Tensors without an
    untyped storage (meta, sparse, nested) fall back to the tensor
    object itself — identity still catches rebinding, just not
    in-place storage swaps those types cannot express anyway.
    """
    out: list[Any] = []
    for t in _module_tensors(module):
        try:
            out.append(t.untyped_storage())
        except Exception:  # pragma: no cover — exotic tensor types
            out.append(t)
    return tuple(out)


def cuda_device_indices(*groups: Any) -> tuple[int, ...]:
    """Sorted distinct CUDA device indices over tensor leaves.

    The meter's sync set: ``torch.cuda.synchronize(device)`` per
    index scopes the wait to the devices actually touched instead of
    the global ``torch.cuda.synchronize()``.  An index-less ``cuda``
    device resolves to the current device.
    """
    idxs: set[int] = set()
    for g in groups:
        items = g if isinstance(g, (tuple, list)) else (g,)
        for t in iter_tensors(items):
            if t.is_cuda:
                i = t.device.index
                idxs.add(
                    i if i is not None else torch.cuda.current_device()
                )
    return tuple(sorted(idxs))


def device_graph_key(xs: tuple) -> tuple[int, tuple] | None:
    """Return the ``(device_index, signature)`` CUDA-graph cache key.

    ``None`` — the honest "do not capture" answer — unless *every*
    positional argument is a CUDA tensor on the SAME device (mixed-
    device inputs cannot share one graph's buffers, and a non-tensor
    arg cannot be refreshed per call).  The signature bakes
    ``(shape, dtype)`` per argument; the device index is the outer
    key so graphs are never shared across devices.
    """
    if not xs or not all(
        isinstance(t, torch.Tensor) and t.is_cuda for t in xs
    ):
        return None
    if len({t.device for t in xs}) != 1:
        return None
    idx = xs[0].device.index
    if idx is None:  # pragma: no cover — real tensors carry an index
        idx = torch.cuda.current_device()
    sig = tuple(
        (tuple(int(d) for d in t.shape), str(t.dtype)) for t in xs
    )
    return (idx, sig)
