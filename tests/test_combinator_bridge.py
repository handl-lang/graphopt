"""Unit tests for the ADR-0008 combinator-bridge prototype.

The bridge lives in ``lab/bridge/combinator.py`` (a falsification
prototype, not shipped source): sharing-compilation round-trips
and the spelled ``loop`` RetNet fold are the properties under test.
"""

import pytest
import torch
from mlopt_core.ir import Const, Op, Param, TensorType, Var

from lab.bridge import combinator as cb


def _v(name, *shape):
    return Var(name, TensorType(shape))


def _p(name, *shape):
    return Param(name, TensorType(shape))


X = _v("x", 2, 4)
W = _p("w", 4, 4)
B = _p("b", 4)


@pytest.mark.parametrize(
    "term",
    [
        X,  # bare leaf program
        Op.make("mul", X, X),  # same wire twice into one node
        Op.make("relu", Op.make("linear", X, W, B)),  # plain chain
        Op.make(
            "mul", Op.make("add", X, W), Op.make("add", X, W)
        ),  # diamond sharing (Δ on a computed wire)
        Op.make("div", Op.make("matmul", X, W), Const(2.0)),
        Op.make(
            "pack",
            Op.make("add", X, W),
            Op.make("mul", X, W),
        ),  # multi-output root — pack is the bundle itself
        Op.make(
            "getitem",
            Op.make("pack", Op.make("add", X, W), W),
            i=0,
            validate=False,
        ),  # mid-graph pack consumed by a projection
    ],
)
def test_roundtrip(term):
    """``dag → combinator → dag`` is structurally exact."""
    morph = cb.dag_to_comb(term)
    rt = cb.comb_to_dag(morph)
    assert rt == term


def test_roundtrip_preserves_sharing():
    """A shared subterm comes back as the SAME hash-consed object."""
    shared = Op.make("add", X, W)
    root = Op.make("mul", shared, Op.make("matmul", shared, W))
    rt = cb.comb_to_dag(cb.dag_to_comb(root))
    assert rt == root
    assert rt.args[0] is rt.args[1].args[0]


def test_fanout_is_dup():
    """Every Δ node corresponds to a real fan-out site."""
    shared = Op.make("add", X, W)
    root = Op.make("mul", shared, Op.make("matmul", shared, W))
    st = cb.morph_stats(cb.dag_to_comb(root))
    # x, w leaves fan out; `shared` fans out to mul and matmul
    assert st.dup >= 1
    assert st.dup_copies >= 2


def test_retnet_step_arity():
    """The spelled step is ``S ⊗ (q,k,v) → S' ⊗ o``."""
    step = cb.retnet_step(Const(0.9), Const(0.9**4), 8.0, 4)
    assert cb.arity(step) == (4, 2)


def test_retnet_fold_expands_and_chains():
    """The loop expansion produces the nested state recurrence."""
    t, k, d, b = 16, 2, 4, 1
    x = _v("x", b, t, d)
    s0 = _v("s0", d, d)
    params = {
        "q_weight": _p("wq", d, d),
        "q_bias": _p("bq", d),
        "k_weight": _p("wk", d, d),
        "k_bias": _p("bk", d),
        "v_weight": _p("wv", d, d),
        "v_bias": _p("bv", d),
    }
    prog = cb.retnet_fold_program(
        x, s0, params, Const(0.9), Const(0.9 ** (t // k)), t, k, d
    )
    assert cb.arity(prog) == (0, 1)
    term = cb.comb_to_dag(prog)
    # the expanded DAG itself re-projects through the bridge
    assert cb.comb_to_dag(cb.dag_to_comb(term)) == term


def test_retnet_fold_verifies_against_model():
    """End-to-end: the spelled loop IS chunked retention."""
    import mlopt_carriers  # noqa: F401 — carrier op bindings
    from mlopt_core.ir import IR
    from mlopt_discovery.zoo import _ChunkedRetention
    from mlopt_torch.adapters import TorchSink

    torch.manual_seed(0)
    t, k, d, b = 16, 2, 8, 1
    model = _ChunkedRetention(d).double()
    x = _v("x", b, t, d)
    s0 = _v("s0", d, d)
    params = {
        "q_weight": _p("wq", d, d),
        "q_bias": _p("bq", d),
        "k_weight": _p("wk", d, d),
        "k_bias": _p("bk", d),
        "v_weight": _p("wv", d, d),
        "v_bias": _p("bv", d),
    }
    prog = cb.retnet_fold_program(
        x, s0, params, Const(0.9), Const(0.9 ** (t // k)), t, k, d
    )
    term = cb.comb_to_dag(prog)
    ir = IR(
        root=term,
        inputs=[x, s0],
        params={p.name: p for p in params.values()},
    )
    mod = TorchSink().lower(
        ir,
        params={
            "wq": model.q.weight.detach(),
            "bq": model.q.bias.detach(),
            "wk": model.k.weight.detach(),
            "bk": model.k.bias.detach(),
            "wv": model.v.weight.detach(),
            "bv": model.v.bias.detach(),
        },
    )
    xt = torch.randn(b, t, d, dtype=torch.float64)
    s0t = torch.zeros(d, d, dtype=torch.float64)
    with torch.no_grad():
        ref = model(xt)
        out = mod(xt, s0t)
    # the eta=gamma^{-c} rank-1 factor grows geometrically in C —
    # fp64 lands ~1e-7 (the documented conditioning bound for the
    # chunked decomposition)
    assert torch.allclose(ref, out, atol=1e-6)
