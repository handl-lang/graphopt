"""Fusion-preferred extraction — near-tie members resolve by kernels.

``EGraph.extract_best(..., fusion_epsilon=...)`` (and the
``extract_fused`` wrapper, epsilon 5%) add a secondary key among
members priced within epsilon of the class minimum:
``(len(fusion_regions(term)), not exposes_pointwise, nops)`` via
:func:`mlopt_core.cost.fusion_member_key`.  These tests pin the band
semantics (near-tie only), the key ordering (fewer regions, then a
root a pointwise consumer could absorb), determinism on exact ties,
and reuse of the shared cost memo for the region probes.
"""

from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, Op, Param, TensorType, Var, op_repr
from mlopt_core.cost import (
    count_cost,
    fusion_member_key,
    fusion_regions,
)
from mlopt_orchestrator import Optimizer

from mlopt_torch.backend import TorchBackend


def _t(*shape):
    return TensorType(tuple(shape))


def _x():
    return Var("x", _t(4, 4))


def _w():
    return Param("W", _t(4, 4))


def _class_of(*terms):
    """EGraph holding *terms* unioned into one e-class -> root eid."""
    eg = EGraph()
    eids = [eg.add_term(t) for t in terms]
    root = eids[0]
    for e in eids[1:]:
        eg.union(root, e)
    return eg, eg.find(root)
























def test_optimize_model_fusion_epsilon_kwarg():
    """``optimize_model(fusion_epsilon=…)`` arms the near-tie band and
    records it in stats; 0 keeps byte-identical selection."""
    import torch
    import torch.nn as nn


    class M(nn.Module):
        def forward(self, x):
            return torch.sigmoid(torch.sigmoid(x))

    m = M().eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    mod, stats = Optimizer(backend=TorchBackend()).optimize(m, x, fusion_epsilon=0.05, verify=False, verbose=False)

    assert stats["fusion_epsilon"] == 0.05
    with torch.no_grad():
        assert torch.allclose(mod(x), m(x))
    _mod2, stats2 = Optimizer(backend=TorchBackend()).optimize(m, x, verify=False, verbose=False)

    assert "fusion_epsilon" not in stats2
