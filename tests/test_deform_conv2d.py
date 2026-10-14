"""``deform_conv2d`` lowering — the torchvision binding (BiRefNet gap).

The BiRefNet bounded probe (retro ``birefnet-bounded-lane.md``) found
``deform_conv2d`` exporting into the IR with **no torch lowering
binding**: a silent coverage gap behind the earlier "no missing
bindings" claim (the op was absent from the sink's ``supported_ops``,
so extraction never picked it, and nothing refused at the boundary
either).  These tests pin the closed gap:

* the aten op maps to the IR op and the sink binds it;
* the binding lowers through torchvision's kernel and is numerically
  exact (masked and unmasked, grouped);
* absent torchvision the boundary refuses **loudly**, naming the
  dependency — never a silent ``argN``-shaped no-op.
"""

import sys

import pytest
import torch
import torch.nn as nn

from catopt_core.attrs import ATTR_SCHEMA
from catopt_torch.torch_bridge import (
    _ATEN_TO_IR,
    _IR_TO_TORCH,
    _torchvision_deform_conv2d,
    export_to_ir,
)


def test_deform_conv2d_is_a_registered_bound_op():
    """The identity alias + binding close the coverage gap.

    The alias pulls the op under the ``test_contracts`` binding-coverage
    pin (``_ATEN_TO_IR`` values must be bound), and the schema names the
    scalar tail the export emits.
    """
    assert _ATEN_TO_IR["deform_conv2d"] == "deform_conv2d"
    assert callable(_IR_TO_TORCH["deform_conv2d"])
    assert ATTR_SCHEMA["deform_conv2d"][11] == "groups"
    assert ATTR_SCHEMA["deform_conv2d"][12] == "offset_groups"
    assert ATTR_SCHEMA["deform_conv2d"][13] == "use_mask"


def test_deform_conv2d_binding_is_in_supported_ops():
    """The sink reports the op it can now lower (the gap's symptom)."""
    from catopt_torch.adapters import TorchSink

    assert "deform_conv2d" in TorchSink().supported_ops


def test_deform_conv2d_lowering_requires_torchvision(monkeypatch):
    """Absent torchvision → a loud error naming the dependency.

    The op ships in torchvision's C++ extension; a boundary that cannot
    lower it must say so, not fail with a bare ``AttributeError``.
    ``sys.modules[name] = None`` makes ``import name`` raise
    ``ImportError``.
    """
    monkeypatch.setitem(sys.modules, "torchvision", None)
    with pytest.raises(RuntimeError, match="torchvision"):
        _torchvision_deform_conv2d()


@pytest.mark.parametrize("masked", [False, True])
def test_deform_conv2d_round_trip(masked):
    """export → lower through the sink → numerically exact.

    ``masked`` exercises the v2 (modulated) path — a real ``mask``
    operand and ``use_mask=True`` — against the v1 path where
    torchvision's wrapper materialises a zero mask at export.
    """
    pytest.importorskip("torchvision")
    from torchvision.ops import DeformConv2d

    from catopt_torch.adapters import TorchSink

    torch.manual_seed(0)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = DeformConv2d(6, 4, 3, padding=1, groups=2)

        def forward(self, x, offset, mask=None):
            if mask is None:
                return self.d(x, offset)
            return self.d(x, offset, mask)

    model = Wrapper().eval()
    x = torch.randn(2, 6, 8, 8)
    offset = torch.randn(2, 2 * 2 * 3 * 3, 8, 8)
    mask = torch.rand(2, 2 * 3 * 3, 8, 8)
    args = (x, offset) if not masked else (x, offset, mask)

    ir, tensors = export_to_ir(model, args)
    assert ir.root.op == "deform_conv2d"
    # The export named every scalar tail position through the schema —
    # no bare ``argN`` survives on a schema'd op.
    assert not any(k.startswith("arg") for k in ir.root.attrs)

    sink = TorchSink()
    lowered = sink.lower(ir, tensors)
    report = sink.verify(model, lowered, args)
    assert report.passed
    assert report.max_rel < 1e-6
