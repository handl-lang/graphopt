"""DevicePolicy — homogeneous single-device placement (Phase B).

``docs/reports/production_readiness.md`` Phase B, first honest slice:
normalize/validate device placement, refuse distributed wrappers and
collective ops explicitly, and provide the cache keys the CUDA runner
and the meter's per-device sync consume.  Mixed-device coverage uses
``meta`` tensors so the policy logic is exercised without needing a
second physical GPU (this host has exactly one — see
``docs/reports/device_policy.md``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mlopt_torch.device import (
    COLLECTIVE_OP_NAMES,
    DevicePlacementError,
    DevicePolicy,
    UnsupportedDistributedError,
    collective_ops_in,
    cuda_device_indices,
    device_graph_key,
    distributed_wrapper_name,
    is_collective_op,
    iter_tensors,
    storage_fingerprint,
)


# ------------------------------------------------------------------
#  Collective-op vocabulary
# ------------------------------------------------------------------


def test_is_collective_op_names():
    for name in (
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "barrier",
        "send",
        "recv",
    ):
        assert is_collective_op(name)
        assert name in COLLECTIVE_OP_NAMES


def test_is_collective_op_prefixes():
    assert is_collective_op("dist_all_reduce")
    assert is_collective_op("c10d_broadcast")
    assert is_collective_op("nccl_all_reduce")


def test_is_collective_op_excludes_tensor_ops():
    """``broadcast``/``gather``/``scatter``/``reduce`` are real
    ``_IR_TO_TORCH`` bindings — flagging them would refuse legitimate
    single-device graphs.  Only unambiguous collective spellings and
    backend prefixes count."""
    for name in ("broadcast", "gather", "scatter", "reduce", "matmul"):
        assert not is_collective_op(name)


def test_collective_ops_in():
    names = ["matmul", "all_reduce", "add", "nccl_send", "gather"]
    assert collective_ops_in(names) == ["all_reduce", "nccl_send"]
    assert collective_ops_in(["matmul", "add"]) == []


# ------------------------------------------------------------------
#  iter_tensors
# ------------------------------------------------------------------


def test_iter_tensors_shapes():
    t = torch.randn(2)
    assert list(iter_tensors(t)) == [t]
    got = list(iter_tensors((t, 3, t)))
    assert len(got) == 2 and all(g is t for g in got)
    # depth bounds the walk: a list nested inside the tuple is beyond
    # the one-level horizon and is not descended into
    assert len(list(iter_tensors((t, 3, [t])))) == 1
    assert list(iter_tensors("nope")) == []
    assert list(iter_tensors((), depth=0)) == []


# ------------------------------------------------------------------
#  distributed_wrapper_name / check_module — explicit refusal
# ------------------------------------------------------------------


def test_distributed_wrapper_name_plain_module():
    assert distributed_wrapper_name(nn.Linear(4, 4)) is None
    assert distributed_wrapper_name(object()) is None


def test_distributed_wrapper_name_group_handle_marker():
    """A module exposing a process-group handle is refused by marker —
    no torch distributed init needed."""

    class FakeWrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.process_group = object()

    name = distributed_wrapper_name(FakeWrapped())
    assert name is not None and "process_group" in name


def test_distributed_wrapper_name_class_fragment():
    class MyTensorParallelLinear(nn.Module):
        pass

    assert (
        distributed_wrapper_name(MyTensorParallelLinear())
        == "MyTensorParallelLinear"
    )


def test_check_module_raises_on_wrapper():
    class FakeDDP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device_mesh = object()

    with pytest.raises(UnsupportedDistributedError, match="device_mesh"):
        DevicePolicy().check_module(FakeDDP())


def test_check_module_raises_on_collective_bindings():
    mod = nn.Linear(4, 4)
    mod._torch_bindings = {"all_reduce": lambda x: x}  # type: ignore[attr-defined]
    with pytest.raises(UnsupportedDistributedError, match="all_reduce"):
        DevicePolicy().check_module(mod)


def test_check_module_allows_tensor_op_bindings():
    """``gather``/``broadcast`` bindings are tensor ops, not
    collectives — the refusal must not fire on them."""
    mod = nn.Linear(4, 4)
    mod._torch_bindings = {  # type: ignore[attr-defined]
        "gather": lambda x: x,
        "broadcast": lambda x: x,
        "matmul": lambda x: x,
    }
    DevicePolicy().check_module(mod)  # no raise


def test_distributed_wrapper_name_real_ddp(tmp_path):
    """A real ``DistributedDataParallel`` (gloo, world_size=1) is
    caught by the isinstance path."""
    if torch.distributed.is_initialized():
        pytest.skip("process group already initialized")
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'store'}",
        rank=0,
        world_size=1,
    )
    try:
        from torch.nn.parallel import DistributedDataParallel

        ddp = DistributedDataParallel(nn.Linear(4, 4))
        assert distributed_wrapper_name(ddp) == "DistributedDataParallel"
        with pytest.raises(UnsupportedDistributedError):
            DevicePolicy().check_module(ddp)
    finally:
        torch.distributed.destroy_process_group()


# ------------------------------------------------------------------
#  DevicePolicy.inspect / resolve
# ------------------------------------------------------------------


def test_inspect_empty_is_cpu():
    rep = DevicePolicy().inspect()
    assert rep.policy == "empty"
    assert rep.homogeneous
    assert rep.device == ""
    assert DevicePolicy().resolve() == torch.device("cpu")


def test_inspect_homogeneous_cpu():
    mod = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    rep = DevicePolicy().inspect(mod, x)
    assert rep.policy == "homogeneous"
    assert rep.device == "cpu"
    assert rep.devices == ("cpu",)
    assert rep.homogeneous
    assert DevicePolicy().resolve(mod, x) == torch.device("cpu")


def test_inspect_homogeneous_meta():
    """``meta`` tensors exercise non-CPU device strings without a
    second GPU."""
    x = torch.empty(2, 4, device="meta")
    rep = DevicePolicy().inspect(None, x)
    assert rep.device == "meta"
    assert rep.homogeneous
    assert DevicePolicy().resolve(inputs=x) == torch.device("meta")


def test_inspect_mixed_meta_and_cpu():
    """The mixed-device census: module on cpu, input on meta."""
    mod = nn.Linear(4, 4)
    x = torch.empty(2, 4, device="meta")
    rep = DevicePolicy().inspect(mod, x)
    assert rep.policy == "mixed"
    assert not rep.homogeneous
    assert rep.devices == ("cpu", "meta")


def test_resolve_mixed_raises_with_guidance():
    mod = nn.Linear(4, 4)
    x = torch.empty(2, 4, device="meta")
    with pytest.raises(DevicePlacementError) as ei:
        DevicePolicy().resolve(mod, x)
    msg = str(ei.value)
    assert "cpu" in msg and "meta" in msg
    assert ".to(device)" in msg


def test_resolve_mixed_cpu_fallback():
    mod = nn.Linear(4, 4)
    x = torch.empty(2, 4, device="meta")
    pol = DevicePolicy(cpu_fallback=True)
    rep = pol.inspect(mod, x)
    assert rep.policy == "mixed"  # census is honest either way
    assert pol.resolve(mod, x) == torch.device("cpu")


def test_inspect_counts_param_values_table():
    """``IRModule._param_values`` tensors join the census even when
    they never registered as Parameters."""
    from types import SimpleNamespace

    mod = SimpleNamespace(
        _param_values={"w": torch.empty(2, device="meta")}
    )
    rep = DevicePolicy().inspect(mod, None)
    assert rep.devices == ("meta",)
    assert rep.n_tensors == 1


def test_inspect_non_module_object():
    """A bare object has no tensors — empty census, cpu resolve."""
    rep = DevicePolicy().inspect(object(), 42)
    assert rep.policy == "empty"
    assert DevicePolicy().resolve(object(), 42) == torch.device("cpu")


# ------------------------------------------------------------------
#  storage_fingerprint — the CUDA-graph staleness check
# ------------------------------------------------------------------


def test_storage_fingerprint_stable_and_detects_move():
    lin = nn.Linear(4, 4)
    fp = storage_fingerprint(lin)
    assert storage_fingerprint(lin) == fp
    lin = lin.to(torch.float64)  # storage swap, same Parameter objects
    assert storage_fingerprint(lin) != fp


def test_storage_fingerprint_covers_param_map_rebind():
    """``IRModule._align_to`` rebinds ``_param_map`` without touching
    registered ``_parameters`` — the fingerprint must see it."""
    lin = nn.Linear(4, 4)
    lin._param_map = {"w": lin.weight}  # type: ignore[attr-defined]
    fp = storage_fingerprint(lin)
    lin._param_map["w"] = torch.nn.Parameter(  # type: ignore[attr-defined]
        torch.randn(4, 4)
    )
    assert storage_fingerprint(lin) != fp


# ------------------------------------------------------------------
#  cuda_device_indices / device_graph_key
# ------------------------------------------------------------------


def test_cuda_device_indices_cpu_and_meta_empty():
    assert cuda_device_indices(torch.randn(2), (torch.randn(2),)) == ()
    assert (
        cuda_device_indices(torch.empty(1, device="meta")) == ()
    )
    assert cuda_device_indices() == ()


@pytest.mark.requires_cuda
def test_cuda_device_indices_cuda():
    x = torch.randn(2, device="cuda")
    y = torch.randn(2, device="cuda")
    assert cuda_device_indices((x, y)) == (0,)
    assert cuda_device_indices(x, [y]) == (0,)


def test_device_graph_key_refuses_non_cuda():
    x = torch.randn(2, 4)
    assert device_graph_key((x,)) is None
    assert device_graph_key((x, torch.randn(2))) is None
    assert device_graph_key(()) is None
    assert device_graph_key(("not-a-tensor",)) is None
    # meta tensors are not cuda — no capture key
    assert device_graph_key((torch.empty(1, device="meta"),)) is None


def test_device_graph_key_refuses_mixed_devices():
    """A cpu tensor mixed into cuda inputs cannot share one graph's
    buffers — the key declines (single-GPU host: cpu+meta stands in
    for the multi-device case; the device-set check is device-agnostic)."""
    # two 'devices' that are both non-cuda already fail the all-cuda
    # gate; the same-device rule is exercised on cuda below.
    assert (
        device_graph_key(
            (torch.empty(1, device="meta"), torch.empty(1))
        )
        is None
    )


@pytest.mark.requires_cuda
def test_device_graph_key_cuda_signature():
    x = torch.randn(2, 4, device="cuda")
    y = torch.randn(2, 4, device="cuda", dtype=torch.float64)
    kx = device_graph_key((x,))
    ky = device_graph_key((y,))
    assert kx is not None and ky is not None
    assert kx[0] == 0 and ky[0] == 0
    # dtype is part of the signature — different dtype, different key
    assert kx != ky
    # shape is part of the signature
    kx2 = device_graph_key((torch.randn(3, 4, device="cuda"),))
    assert kx2 is not None and kx2 != kx
    # arity is part of the signature
    kxy = device_graph_key((x, x))
    assert kxy is not None and kxy != kx
    # mixed cpu+cuda inputs decline
    assert device_graph_key((x, torch.randn(2, 4))) is None
    # non-tensor arg declines
    assert device_graph_key((x, 3)) is None
