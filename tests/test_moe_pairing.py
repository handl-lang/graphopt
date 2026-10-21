"""Routed-MoE pairing regression tests.

Pins the e2e_models2 ``moe_stack`` finding: stacked expert parameters
``W (E, out, in)`` export per expert as ``select(W, dim=0, index=e)``,
so the pairing pass's fused weight was spelled
``concat(select(W,0), …, select(W,E-1))`` — a RUNTIME concat over the
view/extraction family that ``IRModule._fold_weight_chains``
deliberately declines to fold (dedup re-materialisation).  The honest
cost model billed that concat every call, forced extraction lost by
~1.7M on one block, and all 40 offered groups were declined
(``paired_extract=0``).

The pass now recognises tiles that jointly cover a stacked base and
spells the fused piece ``reshape(base)`` — a free view that folds to
one fused ``Param`` at lowering — while ``cost._folds_to_param``
prices the pure-view family the way the lowerer actually folds it.
Per-member ``select`` tiles that only PARTIALLY cover a base keep the
runtime concat (and the decline is recorded in stats).

The residual after pairing — the per-expert down projections
``Σ_e g_e ⊙ linear(h_e, W[e])`` over DISTINCT gated inputs — is not a
pairing pattern (no shared domain) but it IS a grouped GEMM:
``batch_tiled_expert_sums`` offers each expert-sum class the batched
member ``reshape(sum(mul(bmm(stack h, W.mT), stack g), 0))`` (the
stack flattens to rank-3 because ``torch.matmul`` right-aligns batch
dims).  ``transpose(W)`` folds to a fused param at lowering, and
extraction picks the member only when honestly cheaper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Param, TensorType, Var

from mlopt_orchestrator import Optimizer

from mlopt_torch.backend import TorchBackend


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _ops(term, seen=None):
    """All distinct Op nodes in a term DAG."""
    if seen is None:
        seen = set()
    if isinstance(term, Op) and term not in seen:
        seen.add(term)
        for a in term.args:
            _ops(a, seen)
    return seen


class RoutedMoEFFN(nn.Module):
    """Mixtral-lite routed MoE FFN in dense-eval form.

    rms → softmax router → top-k scatter mask → renormalised gates;
    ALL experts run (stacked ``(E, out, in)`` Parameters indexed
    ``w[e]`` — the ``select``-tile pattern) and gate-sum.  The gate
    multiply is data-dependent, so the expert sum must NOT fold; the
    same-input up/gate projections are the pairing targets.
    """

    def __init__(self, dim=64, dff=128, n_experts=4, top_k=2):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.rms = nn.Parameter(torch.ones(dim))
        self.router = nn.Linear(dim, n_experts, bias=False)
        g1, g2 = dim**-0.5, dff**-0.5
        self.w1 = nn.Parameter(torch.randn(n_experts, dff, dim) * g1)
        self.w3 = nn.Parameter(torch.randn(n_experts, dff, dim) * g1)
        self.w2 = nn.Parameter(torch.randn(n_experts, dim, dff) * g2)

    def forward(self, x):
        n = F.rms_norm(x, (x.shape[-1],), self.rms, 1e-5)
        w = F.softmax(self.router(n), dim=-1)
        tv, ti = w.topk(self.k, dim=-1)
        g = torch.zeros_like(w).scatter(-1, ti, tv)
        g = g / g.sum(-1, keepdim=True).clamp_min(1e-9)
        out = None
        for e in range(self.E):
            he = F.silu(F.linear(n, self.w1[e])) * F.linear(
                n, self.w3[e]
            )
            ye = F.linear(he, self.w2[e]) * g[..., e].unsqueeze(-1)
            out = ye if out is None else out + ye
        return x + out


class PartialTile(nn.Module):
    """Two linears sharing one input, weighted by ``select`` slices 0
    and 2 of a ``(4, o, d)`` stacked parameter — NOT a full tile, so
    the fused weight keeps the runtime concat and cost decides."""

    def __init__(self, dim=64, out=64, stack=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(stack, out, dim) * dim**-0.5)

    def forward(self, x):
        return F.linear(x, self.W[0]) + F.linear(x, self.W[2])


class SumExperts(nn.Module):
    """Every expert of a ``(E, o, d)`` stack applied to x and summed:
    ``weight_factor_matmul``'s ``linear(x, ΣWₑ)`` is strictly cheaper
    than the E-wide fused GEMM — pairing must decline honestly."""

    def __init__(self, dim=64, out=64, stack=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(stack, out, dim) * dim**-0.5)

    def forward(self, x):
        acc = None
        for e in range(self.W.shape[0]):
            y = F.linear(x, self.W[e])
            acc = y if acc is None else acc + y
        return acc








def test_moe_stacked_experts_pair_and_verify():
    """Routed MoE FFN: the w1/w3/router projections on the shared
    norm output pair into ONE fused GEMM (weights fold through
    ``reshape`` of the stacked params), and the gated expert sum
    batches into ONE grouped GEMM — ``sum_0(bmm(stack h, W2.mT)
    ⊙ stack g)`` — whose transposed weight folds to a fused Param.
    No serial per-expert ``linear`` survives.  The extracted module
    verifies against eager."""
    torch.manual_seed(0)
    m = RoutedMoEFFN().eval()
    x = torch.randn(2, 16, 64)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(m, x, max_iterations=12, verify=False, verbose=False)

    assert stats.get("pairing_groups", 0) >= 1
    assert stats.get("paired_extract") is True
    with torch.no_grad():
        ref = m(x.clone())
        diff = (ref - opt(x.clone())).abs().max().item()
    assert diff < 1e-4

    nodes = _ops(opt._root)
    # One fused GEMM replaces the 9 same-input projections
    # (4*w1 + 4*w3 + router) and one grouped GEMM replaces the 4
    # select-weighted w2 linears — none survive serially.
    sel_lins = [
        t
        for t in nodes
        if t.op == "linear"
        and getattr(t.args[1], "op", None) == "select"
    ]
    assert not sel_lins
    # The grouped GEMM: matmul(reshape(stack(h_e)), w2.mT).
    mms = [t for t in nodes if t.op == "matmul"]
    assert len(mms) == 1
    assert mms[0].args[0].op == "reshape"
    assert mms[0].args[0].args[0].op == "stack"
    # The E per-expert gate muls collapse to ONE broadcast multiply;
    # the E activation muls (silu ⊙ gate-projection) are unchanged.
    assert sum(1 for t in nodes if t.op == "mul") == m.E + 1

    names = {n for n, _ in opt.named_parameters()}
    # All expert weights fold: up/gate/router into the shared GEMM,
    # the w2 stack through its transpose into a second fused param.
    assert "p_w1" not in names
    assert "p_w3" not in names
    assert "p_router_weight" not in names
    assert "p_w2" not in names
    assert sum(1 for n in names if n.startswith("fused_")) >= 2


def test_partial_tile_decline_is_recorded():
    """When forcing the shared GEMM honestly loses, the stats record
    the decline and the cost delta instead of silence."""
    torch.manual_seed(0)
    m = PartialTile().eval()
    x = torch.randn(4, 16, 64)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(m, x, verify=False, verbose=False)

    assert stats.get("pairing_groups") == 1
    assert stats.get("paired_extract") is False
    assert stats.get("paired_delta") > 0
    with torch.no_grad():
        diff = (m(x) - opt(x)).abs().max().item()
    assert diff < 1e-4


def test_weight_factor_beats_pairing_decline():
    """A plain expert SUM admits ``linear(x, ΣWₑ)`` — strictly cheaper
    than the E-wide paired GEMM, so extraction declines the group and
    the verdict lands in stats."""
    torch.manual_seed(0)
    m = SumExperts().eval()
    x = torch.randn(4, 16, 64)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(m, x, verify=False, verbose=False)

    assert stats.get("pairing_groups", 0) >= 1
    assert stats.get("paired_extract") is False
    assert stats.get("paired_delta") > 0
    with torch.no_grad():
        diff = (m(x) - opt(x)).abs().max().item()
    assert diff < 1e-4


# ---------------------------------------------------------------------------
#  Expert-sum batching — unit-level offers and decline paths.
# ---------------------------------------------------------------------------


def _expert_sum_eg(gated=True, e=4, b=2, i=16, o=8, cover=None):
    """Build ``Σ_e [g_e ⊙] linear(h_e, select(W,0,e))`` in an e-graph.

    ``cover`` overrides the tile indices actually summed (default: all
    ``e`` tiles).  Returns ``(eg, root_eid, base_param)``.
    """
    eg = EGraph()
    base = _p("W", e, o, i)
    leaves = []
    for idx in range(e) if cover is None else cover:
        h = eg.add_term(_v(f"h{idx}", b, i))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        lin = eg.add_enode("linear", (h, we), {})
        if gated:
            g = eg.add_term(_v(f"g{idx}", b, 1))
            lin = eg.add_enode("mul", (lin, g), {})
        leaves.append(lin)
    acc = leaves[0]
    for lin in leaves[1:]:
        acc = eg.add_enode("add", (acc, lin), {})
    return eg, acc, base


def _root_members(eg, root, op):
    return [n for n in eg._classes[eg.find(root)].nodes if n.op == op]


























