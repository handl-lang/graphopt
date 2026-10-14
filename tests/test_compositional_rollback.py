"""Compositional e2e rollback — a failing graft must never ship.

The BiRefNet bounded probe (retro ``birefnet-bounded-lane.md``) found
the compositional e2e check *recording* a broken graft without rolling
it back.  With ``mul_scl_ipt='cat'`` the backbone ``bb`` runs twice per
forward at two resolutions: a shape-specialized graft passes the
per-block gate (which only ever sees the FIRST captured call) and then
silently breaks the second call — and the broken module was returned
anyway.  These tests pin the fail-closed contract: a verified-failing
graft reverts to the un-grafted composition.
"""

import torch
import torch.nn as nn

from catopt_orchestrator import Compositional, Optimizer
from catopt_torch.adapters import TorchSink, TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer


class _BakedMask(nn.Module):
    """A block whose forward depends on a runtime attr the export bakes.

    ``n`` is a Python int the parent mutates per call; ``torch.export``
    specializes it into the lowered graph, so the grafted block only
    ever reproduces the FIRST call.  This is the BiRefNet ``self.H`` /
    ``self.W`` trap in miniature.
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(16, 16)
        self.n = 4

    def forward(self, x):
        y = self.w(x)
        mask = (torch.arange(16) >= self.n).to(y.dtype)
        return y * mask


class _TwiceAtTwoSettings(nn.Module):
    """Calls one block twice per forward at two ``n`` settings."""

    def __init__(self):
        super().__init__()
        self.b = _BakedMask()

    def forward(self, x):
        self.b.n = 4
        a = self.b(x)
        self.b.n = 8
        b2 = self.b(x)
        return a.sum() + b2.sum()


def _select_baked_mask(parent, name, child):
    """Block predicate selecting the whole ``_BakedMask`` block."""
    return isinstance(child, _BakedMask)


def test_compositional_rolls_back_broken_graft():
    """A graft that breaks the second call is reverted, not shipped."""
    torch.manual_seed(0)
    model = _TwiceAtTwoSettings().eval()
    x = torch.randn(2, 16)
    ref = model(x)

    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=Compositional(block_pred=_select_baked_mask),
        max_iterations=2,
        verbose=False,
    )

    # The per-block gate passed (it only sees the first call) …
    assert stats["blocks"]["b"]["status"] == "optimized"
    # … but the whole-model check caught the broken second call …
    assert stats["end_to_end"]["max_rel_diff"] > 0.1
    assert stats["end_to_end"]["rolled_back"] is True
    # … and the graft was reverted: nothing shipped, and the returned
    # model is the un-grafted composition (a fresh clone, numerically
    # the original).
    assert stats["n_optimized"] == 0
    assert opt is not model
    assert (opt(x) - ref).abs().max().item() == 0.0


class _E2EExplodes(TorchSink):
    """A sink whose whole-model verify raises; per-block verifies pass."""

    def __init__(self, model):
        super().__init__()
        self._model = model

    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
        if ref is self._model:
            raise RuntimeError("forced e2e failure")
        return super().verify(ref, opt, inputs, rtol=rtol, atol=atol)


def test_compositional_rolls_back_when_e2e_verify_raises():
    """An e2e verify that *raises* rolls back too — not just records."""
    torch.manual_seed(0)
    model = _TwiceAtTwoSettings().eval()
    x = torch.randn(2, 16)
    ref = model(x)

    opt, stats = Optimizer(
        source=TorchSource(),
        sink=_E2EExplodes(model),
        composer=TorchComposer(),
    ).optimize(
        model, x, strategy=Compositional(), max_iterations=1,
        verbose=False,
    )

    assert "error" in stats["end_to_end"]
    assert stats["end_to_end"]["rolled_back"] is True
    assert stats["n_optimized"] == 0
    assert opt is not model
    assert (opt(x) - ref).abs().max().item() == 0.0
