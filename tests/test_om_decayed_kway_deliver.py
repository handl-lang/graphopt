"""The K-way chunked carrier, delivered and timed end to end.

Plan 0023 construct 3 is a *symbolic* win: on the real
``_ChunkedRetention`` export the K-way law (``om_kway_laws(k)``) cuts
the extracted carrier's cost and extraction *does* pick the chunked
``concat`` form.  These tests close the loop to a measured one:

* The extracted ``concat`` carrier lowers through the **ambient**
  ``TorchSink`` — its ops (``om_decayed`` / ``decay`` / ``decay_vec``
  / ``concat`` / ``split``) all carry torch bindings, so no minted
  handler is needed — and verifies against the source module in fp64.
* Past a crossover size the lowered module beats the eager baseline
  *and* ``torch.compile`` of it: the FLOP win survives as *time*.

The crossover is real and is reported honestly: at T=512 the carrier
is many small ops (splits, concats, ``decay_vec`` materialisations,
per-block matmuls) and the fused quadratic is too small to hide that
launch overhead, so the carrier *loses* there.  The timing test
therefore uses T=1024 (measured ~2.2x over eager, ~2.8x over
compiled); the companion probe
``lab/probes/om_decayed_kway_deliver_probe.py`` prints the full T x k
sweep, both regimes included.
"""

from __future__ import annotations

import pytest
pytestmark = pytest.mark.e2e  # boundary-purpose file: crosses an external engine boundary
import torch
from mlopt_carriers.om import OM_CHUNKIFY, OM_LAWS, om_kway_laws
from mlopt_core.cost import flops_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Op
from mlopt_core.laws.tensor import (
    ASSOC_MATMUL_REV,
    DIV_ADD_FACTOR,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    MATMUL_DIV_DISTRIBUTE,
    MUL_DIV_SWAP,
    SPLIT_SCALAR_RULES,
)
from mlopt_discovery.play import _time_module
from mlopt_discovery.zoo import _ChunkedRetention
from mlopt_torch.adapters import TorchSink, TorchSource

#: The naturality bridges the fold's last hops need (plan 0023) —
#: load-bearing for the carrier preference.
_BRIDGES = [
    MUL_DIV_SWAP,
    MATMUL_DIV_DISTRIBUTE,
    ASSOC_MATMUL_REV,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    DIV_ADD_FACTOR,
]


def _ops_of(term: object) -> set:
    """Every op name reachable in *term*."""
    acc: set = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            acc.add(t.op)
            stack.extend(t.args)
    return acc


def _extract(t: int, k: int) -> tuple:
    """Saturate the real export; return ``(model, x, ir, leaves, best)``."""
    torch.manual_seed(0)
    rules = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + _BRIDGES
        + om_kway_laws(k)
    )
    budgets = {
        "om_decayed_chunkify_rows": 60,
        "om_decayed_chunkify_keys": 60,
        f"om_decayed_kway_blocks_{k}": 200,
    }
    model = _ChunkedRetention(16).double()
    x = torch.randn(2, t, 16, dtype=torch.float64)
    ir, leaves = TorchSource().to_ir(model, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(
        rules,
        root,
        max_iterations=16,
        max_nodes=200_000,
        patience=8,
        rule_budgets=budgets,
    )
    best = eg.extract_best(eg.find(root), flops_cost)
    return model, x, ir, leaves, best


def _lower(ir: IR, best: Op, leaves: dict) -> object:
    """Rebuild the extracted term as an IR and lower it (ambient sink)."""
    ex_ir = IR(
        root=best,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )
    return TorchSink().lower(ex_ir, params=leaves)


class TestKwayCarrierDelivery:
    """The extracted ``concat`` carrier reaches a runnable module."""

    def test_t512_extraction_lowers_through_ambient_sink(self):
        """T=512 headline: the chunked carrier lowers and verifies.

        The extracted root is ``concat``; every op it uses is in the
        ambient ``TorchSink`` table, so it lowers without a minted
        handler, and the lowered module reproduces the source in
        fp64.  (At this size it is *slower* than the fused quadratic
        — see :mod:`tools.om_decayed_kway_deliver_probe` — but the
        delivery path itself is sound.)
        """
        pytest.importorskip("mlopt_torch")
        model, x, ir, leaves, best = _extract(512, 2)
        assert best.op == "concat"
        sink = TorchSink()
        assert _ops_of(best) <= sink.supported_ops
        module = _lower(ir, best, leaves)
        rep = sink.verify(model, module, (x,))
        assert rep.passed
        assert rep.max_abs < 1e-6

    @pytest.mark.requires_cuda
    def test_carrier_beats_eager_past_the_crossover(self):
        """T=2048: the lowered carrier beats eager and compiled.

        Past the crossover the many-small-op carrier wins: the fused
        quadratic's ``(T,T)`` contraction dominates.  Measured
        ~5.0-5.9x over eager and ~6.2-7.4x over ``torch.compile`` at
        T=2048 — the assertion is the direction, and the size is
        chosen where the margin is wide enough to survive a loaded
        host (at T=1024 the margin is ~2.2x, which a contended box
        can invert).
        """
        pytest.importorskip("mlopt_torch")
        t, k = 2048, 4
        model, x, ir, leaves, best = _extract(t, k)
        assert best.op == "concat"
        sink = TorchSink()
        assert _ops_of(best) <= sink.supported_ops
        device = "cuda"
        base = model.to(device)
        xin = x.to(device)
        module = _lower(ir, best, leaves).to(device)
        rep = sink.verify(base, module, (xin,))
        assert rep.passed
        assert rep.max_abs < 1e-6
        cfg = torch._dynamo.config
        cfg.recompile_limit = max(int(cfg.recompile_limit), 256)
        t_base = _time_module(base, xin, 100)
        t_comp = _time_module(torch.compile(base), xin, 100)
        t_car = _time_module(module, xin, 100)
        assert t_car < t_base
        assert t_car < t_comp
