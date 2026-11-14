"""Phase B extension — sharded-param refusal + placement structures.

``mlopt_torch/device_placement.py``: the honest decline surface for
DTensor/FSDP-sharded parameters (``ShardedParamError`` — a subclass of
``UnsupportedDistributedError``), the pure-data per-device placement
layer (``DeviceSlot``/``PlacementPlan``/``plan_placement``/
``group_by_device``), and the port-level pipeline wiring
(``DeviceCheckedSource`` / ``DevicePolicyRunner`` / the
``TorchBackend(device_policy=...)`` seam) that works on the pinned
substrate today.

Real DTensor/FSDP coverage uses gloo ``world_size=1`` process groups
on CPU — the same pattern ``test_device_policy.py`` uses for real
DDP.  Sharded *values* (world_size>1, CUDA) remain hardware-pending;
what is proven here is the refusal surface and the data structures.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from mlopt_torch.device import (
    DevicePlacementError,
    DevicePolicy,
    UnsupportedDistributedError,
)
from mlopt_torch.device_placement import (
    DeviceCheckedSource,
    DevicePolicyRunner,
    DeviceSlot,
    ShardedAwareDevicePolicy,
    ShardedParamError,
    check_sharded_params,
    group_by_device,
    is_dtensor,
    is_sharded_param,
    plan_placement,
    sharded_params_in,
)

# ------------------------------------------------------------------
#  Sharded-parameter detection — fakes first (no process group)
# ------------------------------------------------------------------


def _fake_dtensor():
    """A DTensor stand-in: carries the two markers a DTensor cannot
    function without (``device_mesh`` + ``placements``)."""
    return SimpleNamespace(
        device=torch.device("cpu"),
        device_mesh=object(),
        placements=["Shard(dim=0)"],
    )


def _fake_flat_param():
    return SimpleNamespace(
        device=torch.device("cpu"),
        _local_shard=torch.randn(2),
    )


def test_is_dtensor_markers():
    assert is_dtensor(_fake_dtensor())
    assert not is_dtensor(torch.randn(2))
    # device_mesh alone is not enough — placements is required too
    assert not is_dtensor(
        SimpleNamespace(device_mesh=object(), placements=None)
    )


def test_is_sharded_param_kinds():
    assert is_sharded_param(_fake_dtensor())
    assert is_sharded_param(_fake_flat_param())

    # class-name arm
    class MyFlatParameter(torch.nn.Parameter):
        pass

    assert is_sharded_param(MyFlatParameter(torch.randn(2)))
    # plain tensors are not shards — including FSDP NO_SHARD params
    assert not is_sharded_param(torch.randn(2))
    assert not is_sharded_param(nn.Parameter(torch.randn(2)))


def test_sharded_params_in_finds_marked_leaves():
    mod = SimpleNamespace(
        parameters=lambda: [torch.randn(2), _fake_dtensor()]
    )
    found = sharded_params_in(mod)
    assert len(found) == 1
    assert "DTensor" in found[0] and "Shard" in found[0]


def test_sharded_params_in_module_fsdp_marker():
    """An ``NO_SHARD`` FSDP module's leaves look plain — the module
    marker is what refuses it."""
    mod = SimpleNamespace(
        _fsdp_wrapped_module=object(),
        parameters=lambda: [torch.randn(2)],
    )
    found = sharded_params_in(mod)
    assert found == ["module (_fsdp_wrapped_module)"]


def test_sharded_params_in_clean_module_empty():
    mod = nn.Linear(4, 4)
    assert sharded_params_in(mod) == []
    check_sharded_params(mod)  # no raise


def test_check_sharded_params_raises_actionable():
    mod = SimpleNamespace(parameters=lambda: [_fake_dtensor()])
    with pytest.raises(ShardedParamError) as ei:
        check_sharded_params(mod)
    msg = str(ei.value)
    assert "sharded" in msg
    assert "to_local" in msg or "summon_full_params" in msg
    # it IS a distributed refusal — callers catching
    # UnsupportedDistributedError still catch it
    assert isinstance(ei.value, UnsupportedDistributedError)


def test_sharded_aware_policy_refuses_shards_first():
    mod = SimpleNamespace(parameters=lambda: [_fake_dtensor()])
    with pytest.raises(ShardedParamError):
        ShardedAwareDevicePolicy().check_module(mod)
    # the base policy alone does NOT see the shard — that is the gap
    # this subclass closes
    DevicePolicy().check_module(mod)  # no raise: census is honest


def test_sharded_aware_policy_still_refuses_wrappers():
    class FakeWrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.process_group = object()

    with pytest.raises(UnsupportedDistributedError):
        ShardedAwareDevicePolicy().check_module(FakeWrapped())


# ------------------------------------------------------------------
#  Real DTensor / FSDP over gloo world_size=1 (CPU)
# ------------------------------------------------------------------


def _gloo_pg(tmp_path):
    if torch.distributed.is_initialized():
        pytest.skip("process group already initialized")
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'store'}",
        rank=0,
        world_size=1,
    )


def test_real_dtensor_is_detected(tmp_path):
    _gloo_pg(tmp_path)
    try:
        from torch.distributed._tensor import DTensor, Shard
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (1,))
        dt = DTensor.from_local(torch.randn(4, 4), mesh, [Shard(0)])
        assert is_dtensor(dt)
        assert is_sharded_param(dt)

        class DTensorModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter(
                    "w", nn.Parameter(dt, requires_grad=False)
                )

        mod = DTensorModule()
        found = sharded_params_in(mod)
        assert len(found) == 1 and "DTensor" in found[0]
        with pytest.raises(ShardedParamError):
            ShardedAwareDevicePolicy().check_module(mod)
    finally:
        torch.distributed.destroy_process_group()


def test_real_fsdp_module_refused_via_marker(tmp_path):
    """world_size=1 FSDP runs NO_SHARD — leaves stay plain Parameters,
    so the refusal comes from the module marker, exactly the case the
    leaf census cannot see."""
    _gloo_pg(tmp_path)
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )

        fsdp = FSDP(nn.Linear(4, 4))
        # sanity: leaves really are plain Parameters under NO_SHARD
        assert all(
            type(p).__name__ == "Parameter" for p in fsdp.parameters()
        )
        found = sharded_params_in(fsdp)
        assert any("_fsdp" in f for f in found)
        with pytest.raises(ShardedParamError, match="sharded"):
            ShardedAwareDevicePolicy().check_module(fsdp)
    finally:
        torch.distributed.destroy_process_group()


# ------------------------------------------------------------------
#  PlacementPlan / DeviceSlot / group_by_device — the data layer
# ------------------------------------------------------------------


def test_device_slot_normalisation():
    assert DeviceSlot.of("cpu") == DeviceSlot("cpu", None)
    assert DeviceSlot.of(torch.device("meta")) == DeviceSlot(
        "meta", None
    )
    # torch.device("cuda") without index keeps index=None — honest,
    # the current device is runtime state the plan does not guess
    assert DeviceSlot.of("cuda") == DeviceSlot("cuda", None)
    s = DeviceSlot.of(DeviceSlot("cuda:0", 0))
    assert s == DeviceSlot("cuda:0", 0)


def test_plan_placement_empty_is_cpu():
    plan = plan_placement()
    assert plan.slots == (DeviceSlot("cpu"),)
    assert plan.homogeneous
    assert plan.devices == ("cpu",)


def test_plan_placement_dedup_and_order():
    plan = plan_placement("cpu", "meta", "cpu", torch.device("meta"))
    assert plan.devices == ("cpu", "meta")
    assert not plan.homogeneous
    assert plan.for_device("cpu") == DeviceSlot("cpu")
    assert plan.for_device(torch.device("meta")) is not None
    assert "cpu" in plan and "xpu:0" not in plan


def test_plan_placement_homogeneous_only():
    plan = plan_placement("cpu", homogeneous_only=True)
    assert plan.homogeneous
    with pytest.raises(DevicePlacementError, match=r"cpu.*meta"):
        plan_placement("cpu", "meta", homogeneous_only=True)


def test_group_by_device():
    cpu_t = torch.randn(2)
    meta_t = torch.empty(2, device="meta")
    groups = group_by_device([cpu_t, meta_t, cpu_t, 42])
    assert groups["cpu"] == [cpu_t, cpu_t]
    assert groups["meta"] == [meta_t]
    assert groups["<none>"] == [42]
    assert group_by_device([]) == {}


# ------------------------------------------------------------------
#  Port-level wiring — works on the pinned substrate today
# ------------------------------------------------------------------


class _StubSource:
    """Minimal Source stand-in — records the call, returns a marker."""

    def __init__(self) -> None:
        self.calls: list = []

    def to_ir(self, model, example_inputs):
        self.calls.append((model, example_inputs))
        return ("IR", {"w": 1.0})


class _RecordingPolicy(DevicePolicy):
    def __init__(self) -> None:
        self.resolved: list = []
        self.checked: list = []

    def resolve(self, module=None, inputs=None):
        self.resolved.append((module, inputs))
        return torch.device("cpu")

    def check_module(self, module):
        self.checked.append(module)


def test_device_checked_source_resolves_then_delegates():
    src = _StubSource()
    pol = _RecordingPolicy()
    wrapped = DeviceCheckedSource(src, pol)
    model, x = object(), (torch.randn(2),)
    out = wrapped.to_ir(model, x)
    assert out == ("IR", {"w": 1.0})
    assert src.calls == [(model, x)]
    assert pol.resolved == [(model, x)]
    assert wrapped.last_device == torch.device("cpu")
    # attribute passthrough keeps the decorator transparent
    assert wrapped.calls is src.calls


def test_device_checked_source_refusal_blocks_export():
    class Refusing(_RecordingPolicy):
        def resolve(self, module=None, inputs=None):
            raise DevicePlacementError("mixed devices")

    src = _StubSource()
    wrapped = DeviceCheckedSource(src, Refusing())
    with pytest.raises(DevicePlacementError):
        wrapped.to_ir(object(), None)
    assert src.calls == []  # export never ran


def test_device_policy_runner_checks_and_records():
    pol = _RecordingPolicy()
    runner = DevicePolicyRunner(pol)
    mod = nn.Linear(4, 4)
    stats: dict = {}
    out = runner.apply(mod, torch.randn(2, 4), stats)
    assert out is mod
    assert pol.checked == [mod]
    assert stats["device_checked"] is True
    assert stats["delivery_device"] == "cpu"


def test_device_policy_runner_propagates_refusal():
    class Refusing(_RecordingPolicy):
        def check_module(self, module):
            raise UnsupportedDistributedError("ddp")

    with pytest.raises(UnsupportedDistributedError):
        DevicePolicyRunner(Refusing()).apply(nn.Linear(4, 4), None, {})


def test_torch_backend_device_policy_wraps_source():
    from mlopt_core.pipeline import Backend
    from mlopt_torch.backend import TorchBackend

    pol = _RecordingPolicy()
    backend = TorchBackend(device_policy=pol)
    assert isinstance(backend.source, DeviceCheckedSource)
    assert backend.source._policy is pol
    # When the pinned Backend carries the Phase B field the policy is
    # set on it too; on the older pin the field simply does not exist
    # and only the source-wrap path is live — assert whichever is true.
    if "device_policy" in getattr(Backend, "__dataclass_fields__", {}):
        assert backend.device_policy is pol
    else:
        assert not hasattr(backend, "device_policy")
    # None leaves the historical path: plain TorchSource, no wrapper
    plain = TorchBackend()
    assert not isinstance(plain.source, DeviceCheckedSource)
