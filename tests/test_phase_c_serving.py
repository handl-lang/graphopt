"""Phase C serving/autograd boundary pins (CPU-safe + one CUDA cell).

``docs/reports/phase_c_serving.md`` documents the measured serving
surface; ``bench/drivers/probes/phase_c_serving_probe.py`` is the
reproducible measurement.  These tests pin the claims that are cheap
to re-check and load-bearing for the honest boundary:

* the delivered ``IRModule`` IS differentiable end-to-end on CPU —
  input grads match the eager original exactly and every fused
  parameter collects a grad;
* the ``.pt2`` artifact is NOT a fine-tuning artifact: input grads
  flow through the exported program and match eager, but the
  parameters are baked constants — ``p.grad`` stays ``None``;
* the ``_align_to`` rebind (a module built on one device/dtype
  serving inputs on another, without ``.to()``) moves the live weight
  tables onto NEW ``Parameter`` objects that ``module.parameters()``
  never sees — grads land where no optimizer can reach them.
"""

import pytest
import torch
import torch.nn as nn
from mlopt_orchestrator import Optimizer
from mlopt_torch.backend import TorchBackend
from mlopt_torch.export import export_optimized, load_optimized


def _mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)
    ).eval()


def _optimize(mod: nn.Module, x: torch.Tensor):
    module, _stats = Optimizer(backend=TorchBackend()).optimize(
        mod, x, verify=False, verbose=False
    )
    return module


def test_delivered_module_trains_on_cpu():
    """Forward+backward through the delivered module: input grad is
    bit-close to the eager original and every parameter gets a grad."""
    x = torch.randn(2, 8)
    mod = _mlp()
    opt = _optimize(mod, x)

    xin = x.clone().requires_grad_(True)
    opt(xin).square().sum().backward()

    ref = _mlp()
    ref.load_state_dict(mod.state_dict())
    xe = x.clone().requires_grad_(True)
    ref(xe).square().sum().backward()

    assert xin.grad is not None and xe.grad is not None
    assert torch.allclose(xin.grad, xe.grad, atol=1e-6)
    grads = [p.grad for p in opt.parameters()]
    assert all(g is not None for g in grads)
    assert all(g.abs().sum() > 0 for g in grads)


def test_pt2_is_input_grad_only_not_finetuning(tmp_path):
    """The .pt2 artifact: input grads flow and match eager, but the
    baked parameters collect no grad — it is not a training artifact."""
    x = torch.randn(2, 8)
    mod = _mlp()
    opt = _optimize(mod, x)
    ref = _mlp()
    ref.load_state_dict(mod.state_dict())

    out = tmp_path / "opt.pt2"
    export_optimized(None, opt, out, example_input=x)
    loaded = load_optimized(out)

    xi = x.clone().requires_grad_(True)
    loaded(xi).square().sum().backward()
    xe = x.clone().requires_grad_(True)
    ref(xe).square().sum().backward()

    assert xi.grad is not None and xe.grad is not None
    assert torch.allclose(xi.grad, xe.grad, atol=1e-6)
    # The parameters exist but are constants of the exported graph.
    assert all(p.grad is None for p in loaded.parameters())


@pytest.mark.requires_cuda
def test_align_to_rebind_hides_param_grads():
    """CPU-built module served CUDA inputs without ``.to()``:
    ``_align_to`` rebinds the live weight tables to fresh Parameters
    that are NOT the registered ``_parameters`` — input grads stay
    correct but ``module.parameters()`` reports ``grad=None`` while
    the live tables hold the real grads.  The documented boundary:
    fine-tune on the build device/dtype, or ``.to()`` first."""
    if not torch.cuda.is_available():
        pytest.skip("requires_cuda: no CUDA device")
    x = torch.randn(2, 8)
    opt = _optimize(_mlp(), x)  # built on CPU

    xin = x.cuda().clone().requires_grad_(True)
    opt(xin).square().sum().backward()

    assert xin.grad is not None
    # Registered params: stale CPU tensors, no grads.
    assert all(p.grad is None for p in opt.parameters())
    # Live tables: the aligned Parameters DID collect grads.
    live = [v for v in opt._param_map.values()]
    assert all(v.grad is not None for v in live)
    # And they are genuinely different objects from the registered set.
    assert not all(
        id(opt._param_map[k]) == id(opt._parameters[k])
        for k in opt._param_map
        if k in opt._parameters
    )
