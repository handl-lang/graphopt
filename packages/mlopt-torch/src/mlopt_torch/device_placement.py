"""Phase B extension — sharded-param refusal + per-device placement.

``device.py`` landed the homogeneous single-device slice: census,
``resolve``, distributed-wrapper refusal, per-device CUDA-graph keys.
This module extends it honestly in three directions, none of which
claims what the hardware cannot yet prove:

* **Sharded-parameter refusal** — :func:`sharded_params_in` /
  :func:`check_sharded_params` name DTensor and FSDP ``FlatParameter``
  leaves (and FSDP module markers) so
  :class:`ShardedAwareDevicePolicy` refuses them with an actionable
  message instead of silently optimising a slice of a larger tensor.
  ``DevicePolicy.check_module`` alone does NOT catch these: an FSDP
  module's ``parameters()`` can still yield plain-looking tensors
  (``NO_SHARD``, unsharded params), and a DTensor leaf passes every
  device census — ``str(device)`` is honest but the *values* are
  shards.

* **Per-device placement structures** — :class:`DeviceSlot` /
  :class:`PlacementPlan` / :func:`plan_placement` /
  :func:`group_by_device` are the data layer a future multi-device
  delivery needs: canonical ``type:index`` device keys, dedup, and
  the homogeneous-only refusal.  They are pure data — CPU/``meta``
  tensors exercise every branch; real multi-GPU replay still needs
  hardware (see ``docs/reports/phase_b_device_policy.md``).

* **Pipeline wiring that works on the pinned substrate** —
  :class:`DeviceCheckedSource` (a ``Source`` decorator running
  ``policy.resolve`` before export) and :class:`DevicePolicyRunner`
  (a ``Runner`` running ``check_module`` + ``resolve`` on the
  delivered module) plug ``DevicePolicy`` into the orchestrator
  through ports that already exist.  When the eqsat pin carries the
  first-class ``device_policy`` kwarg (``Backend.device_policy`` /
  ``Optimizer.device_policy``), :func:`mlopt_torch.backend.TorchBackend`
  sets it too — both paths resolve the same policy, so the answer is
  consistent either way.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from mlopt_torch.device import (
    DevicePlacementError,
    DevicePolicy,
    UnsupportedDistributedError,
    _module_tensors,
)

__all__ = [
    "DeviceCheckedSource",
    "DevicePolicyRunner",
    "DeviceSlot",
    "PlacementPlan",
    "ShardedAwareDevicePolicy",
    "ShardedParamError",
    "check_sharded_params",
    "group_by_device",
    "is_dtensor",
    "is_sharded_param",
    "plan_placement",
    "sharded_params_in",
]


class ShardedParamError(UnsupportedDistributedError):
    """Raised when a module's parameters are DTensor/FSDP shards.

    A shard is a slice of a larger tensor: the module's forward on
    one rank computes a *partial* result, and the equality-saturation
    search — which reasons about whole-tensor values — cannot see
    the sharding at all.  The honest Phase B answer is refusal with
    the materialisation path named, never a silent wrong delivery.
    """


# ------------------------------------------------------------------
#  Sharded-parameter detection
# ------------------------------------------------------------------


def _dtensor_cls() -> type | None:
    """Resolve ``torch.distributed.tensor.DTensor`` when importable."""
    obj: Any = torch
    for part in ("distributed", "tensor", "DTensor"):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, type) else None


def _is_dtensor_obj(t: Any) -> bool:
    """DTensor check — isinstance plus structural markers.

    ``isinstance`` fires when torch exposes the class; the marker arm
    (``device_mesh`` + ``placements`` — attributes a DTensor cannot
    function without) catches duck-typed/test doubles and older torch
    spellings.
    """
    cls = _dtensor_cls()
    if cls is not None and isinstance(t, cls):
        return True
    return (
        getattr(t, "device_mesh", None) is not None
        and getattr(t, "placements", None) is not None
    )


def _is_flat_param(t: Any) -> bool:
    """FSDP ``FlatParameter`` — the sharded-param wrapper.

    Detected by class name (``FlatParameter`` is the stable public
    spelling across FSDP1/FSDP2) plus the ``_local_shard`` /
    ``_full_param_padded`` attributes only a flat param carries.
    """
    if "FlatParameter" in type(t).__name__:
        return True
    return getattr(t, "_local_shard", None) is not None or (
        getattr(t, "_full_param_padded", None) is not None
        and getattr(t, "_mp_shard", None) is not None
    )


def is_dtensor(t: Any) -> bool:
    """Return True when *t* is a DTensor (or carries its markers)."""
    return _is_dtensor_obj(t)


def is_sharded_param(t: Any) -> bool:
    """Return True when *t* is a sharded parameter leaf.

    Covers DTensor leaves and FSDP ``FlatParameter``s.  Plain
    ``torch.nn.Parameter`` / ``torch.Tensor`` leaves — including an
    FSDP module's *unsharded* ``NO_SHARD`` params — return False;
    module-level FSDP markers are :func:`check_sharded_params`'s
    separate arm.
    """
    return _is_dtensor_obj(t) or _is_flat_param(t)


def _describe_shard(t: Any, index: int) -> str:
    """One-line description of a sharded leaf for the refusal."""
    kind = "DTensor" if _is_dtensor_obj(t) else "FlatParameter"
    placements = getattr(t, "placements", None)
    if placements is not None:
        return (
            f"tensor[{index}] ({kind}, placements={list(placements)!r})"
        )
    return f"tensor[{index}] ({kind})"


def sharded_params_in(module: Any) -> list[str]:
    """Describe the sharded parameter leaves of *module*.

    Returns a list of ``tensor[i] (kind, ...)`` strings — empty when
    every leaf is a plain replicated tensor.  Module-level FSDP
    markers (``_fsdp_state`` / ``_fsdp_wrapped_module``) are reported
    as ``module (FSDP state)`` entries even when no leaf looks sharded
    — an ``NO_SHARD`` FSDP module still routes through FSDP's
    unsharded-param machinery the search never sees.
    """
    out: list[str] = []
    for attr in ("_fsdp_state", "_fsdp_wrapped_module"):
        if getattr(module, attr, None) is not None:
            out.append(f"module ({attr})")
            break
    for i, t in enumerate(_module_tensors(module)):
        if is_sharded_param(t):
            out.append(_describe_shard(t, i))
    return out


def check_sharded_params(module: Any) -> None:
    """Raise :class:`ShardedParamError` on sharded parameters.

    The message names what was detected and the honest paths out:
    materialise full parameters (``DTensor.to_local()`` /
    ``full_tensor()``, FSDP ``summon_full_params()``) or optimise
    each rank's shard independently — Phase B does neither, it
    refuses.
    """
    found = sharded_params_in(module)
    if not found:
        return
    raise ShardedParamError(
        f"refusing to optimise module with sharded parameters "
        f"({'; '.join(found)}): DTensor/FSDP leaves are slices of a "
        "larger tensor — a single-rank forward computes a partial "
        "result and the search cannot see the sharding. Phase B "
        "supports replicated homogeneous parameters only; "
        "materialise full parameters first (DTensor.to_local() / "
        "full_tensor(), FSDP summon_full_params()) or optimise each "
        "rank's shard independently"
    )


class ShardedAwareDevicePolicy(DevicePolicy):
    """``DevicePolicy`` + sharded-parameter refusal.

    ``check_module`` runs :func:`check_sharded_params` BEFORE the
    base wrapper/collective checks: an FSDP module's shard leaves
    give the more actionable error (which params, which kind), and
    an ``NO_SHARD`` FSDP module is still refused via its
    ``_fsdp_state`` marker even though every leaf looks plain.
    """

    def check_module(self, module: Any) -> None:
        """Refuse sharded params, then defer to the base checks."""
        check_sharded_params(module)
        super().check_module(module)


# ------------------------------------------------------------------
#  Per-device placement structures (data layer only)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceSlot:
    """One canonical device slot in a :class:`PlacementPlan`.

    ``device`` is the normalised ``str(torch.device)`` — ``"cpu"``,
    ``"meta"``, ``"cuda:0"``; ``index`` is ``torch.device.index``
    (``None`` for index-less devices like ``cpu``/``meta`` and a bare
    ``cuda``).  Two slots are equal iff both fields are.
    """

    device: str
    index: int | None = None

    @classmethod
    def of(cls, device: Any) -> DeviceSlot:
        """Normalise a ``torch.device``/str/``DeviceSlot`` into a slot."""
        if isinstance(device, DeviceSlot):
            return device
        dev = (
            device
            if isinstance(device, torch.device)
            else torch.device(str(device))
        )
        return cls(str(dev), dev.index)


@dataclass(frozen=True)
class PlacementPlan:
    """An ordered, deduplicated set of :class:`DeviceSlot`s.

    ``slots`` preserves first-seen order — the plan is data, not a
    scheduler: it names WHICH devices a delivery spans so callers can
    refuse (``homogeneous_only`` at plan time) or group tensors per
    device (:func:`group_by_device`).  ``homogeneous`` is the Phase B
    admission predicate.
    """

    slots: tuple[DeviceSlot, ...] = field(default_factory=tuple)

    @property
    def homogeneous(self) -> bool:
        """True when the plan spans at most one device."""
        return len(self.slots) <= 1

    @property
    def devices(self) -> tuple[str, ...]:
        """The distinct device strings, in plan order."""
        return tuple(s.device for s in self.slots)

    def for_device(self, device: Any) -> DeviceSlot | None:
        """Return the slot for *device*, or ``None`` when unplanned."""
        want = DeviceSlot.of(device)
        for s in self.slots:
            if s == want:
                return s
        return None

    def __contains__(self, device: Any) -> bool:
        """Return whether *device* normalises to a planned slot."""
        return self.for_device(device) is not None


def plan_placement(
    *devices: Any, homogeneous_only: bool = False
) -> PlacementPlan:
    """Build a :class:`PlacementPlan` over *devices*.

    Accepts ``torch.device`` objects and strings (``"cpu"``,
    ``"cuda:0"``, ``"meta"``); duplicates collapse to first-seen order.
    No devices plans the implicit ``cpu`` slot — matching
    ``DevicePolicy.resolve``'s empty-placement answer.

    ``homogeneous_only=True`` raises :class:`DevicePlacementError`
    naming every device when the plan would span more than one — the
    data-layer twin of ``DevicePolicy.resolve``'s mixed-device
    refusal, usable before any tensor exists (e.g. planning a
    multi-GPU layout).
    """
    slots: list[DeviceSlot] = []
    seen: set[DeviceSlot] = set()
    for d in devices:
        slot = DeviceSlot.of(d)
        if slot not in seen:
            seen.add(slot)
            slots.append(slot)
    if not slots:
        slots = [DeviceSlot("cpu")]
    plan = PlacementPlan(tuple(slots))
    if homogeneous_only and not plan.homogeneous:
        raise DevicePlacementError(
            f"mixed-device placement {list(plan.devices)}: this "
            "slice plans one homogeneous device per module — place "
            "the module and inputs on a single device, or use a "
            "per-device placement plan explicitly"
        )
    return plan


def group_by_device(
    tensors: Iterable[Any],
) -> dict[str, list[Any]]:
    """Group tensor leaves by ``str(t.device)`` — the per-device map.

    The data structure a multi-device delivery iterates: each value
    is the list of tensors living on that device string (``"cpu"``,
    ``"cuda:0"``, …).  Non-tensor leaves and tensors without a
    ``device`` attribute land under ``"<none>"`` — honest, not
    silently dropped.  Empty input returns ``{}``.
    """
    out: dict[str, list[Any]] = {}
    for t in tensors:
        dev = getattr(t, "device", None)
        key = str(dev) if dev is not None else "<none>"
        out.setdefault(key, []).append(t)
    return out


# ------------------------------------------------------------------
#  Pipeline wiring through existing ports (pinned-substrate path)
# ------------------------------------------------------------------


class DeviceCheckedSource:
    """A ``Source`` decorator running ``policy.resolve`` before export.

    Structural ``Source`` conformance: ``to_ir`` resolves placement on
    ``(model, example_inputs)`` — a refusal raises before the wrapped
    source exports anything — then delegates.  The resolved value is
    kept on ``last_device`` for callers that want it; ``__getattr__``
    forwards every other attribute (``supported_ops``, ``ops``, …) to
    the wrapped source so the decorator is transparent.
    """

    def __init__(self, source: Any, policy: DevicePolicy) -> None:
        """Wrap *source*; *policy* is the ``DevicePolicy``-shaped value."""
        self._source = source
        self._policy = policy
        self.last_device: Any = None

    def to_ir(self, model: Any, example_inputs: Any) -> Any:
        """Resolve placement, then export through the wrapped source."""
        self.last_device = self._policy.resolve(model, example_inputs)
        return self._source.to_ir(model, example_inputs)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped source."""
        return getattr(self._source, name)


class DevicePolicyRunner:
    """A ``Runner`` running ``policy.check_module`` + ``resolve``.

    The delivery-side twin of :class:`DeviceCheckedSource`: after the
    lowering produces the module, ``apply`` refuses what the policy
    refuses (distributed wrappers, collective bindings, sharded
    params) and records the resolved device into ``stats`` —
    ``stats["device_checked"] = True`` and
    ``stats["delivery_device"] = str(resolved)`` (the same key the
    substrate's ``lower(device_policy=...)`` writes, so the two
    wiring paths agree).  Compose it with the real delivery runner
    through ``ChainedRunner`` (or pass it alone to keep the identity
    delivery but gain the check).
    """

    name = "device_policy"

    def __init__(self, policy: DevicePolicy) -> None:
        """Store the ``DevicePolicy``-shaped value."""
        self._policy = policy

    def apply(
        self,
        module: Any,
        example_input: Any,
        stats: dict[str, Any],
    ) -> Any:
        """Check + resolve *module*, record into *stats*, return it."""
        check = getattr(self._policy, "check_module", None)
        if callable(check):
            check(module)
        stats["device_checked"] = True
        stats["delivery_device"] = str(
            self._policy.resolve(module, example_input)
        )
        return module
