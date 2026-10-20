"""Unit tests for the ADR-0008 combinator bridge.

The bridge lives in ``mlopt_discovery.combinator`` (promoted out of
``bench/bridge/`` at the 99% coverage floor): sharing-compilation
round-trips and the spelled ``loop`` RetNet fold are the properties
under test.
"""

import pytest
import torch
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_discovery import combinator as cb


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


def test_arity_rejects_empty_seq():
    """``seq()`` has no footprint — the arity guard fires."""
    with pytest.raises(ValueError, match="empty seq"):
        cb.arity(cb.Seq(()))


def test_arity_rejects_non_morphism():
    """``arity`` refuses anything outside the morphism grammar."""
    with pytest.raises(TypeError, match="not a morphism"):
        cb.arity(42)


def test_comb_to_dag_rejects_non_morphism():
    """``_apply`` refuses a term outside the morphism grammar."""
    with pytest.raises(TypeError, match="not a morphism"):
        cb.comb_to_dag(42)


def test_comb_to_dag_rejects_underfed_arr():
    """An ``arr`` whose arity exceeds the wire bundle starves."""
    with pytest.raises(ValueError, match="needs 2 wires"):
        cb.comb_to_dag(cb.Arr("mul", 2))


def test_comb_to_dag_rejects_empty_program():
    """A program that consumes every wire has no output to return."""
    with pytest.raises(ValueError, match="no output wires"):
        cb.comb_to_dag(cb.Drop())


def test_dag_to_comb_refuses_non_term_arg():
    """An op arg the basis cannot present is a named gap, not a term."""
    bad = Op.make("f", 5, validate=False)
    with pytest.raises(TypeError, match="no leaf/generator"):
        cb.dag_to_comb(bad)


def test_par_smart_constructor_edges():
    """``_par`` flattens nested ``par`` and drops ``id(0)`` wires."""
    # a nested Par flattens into the parent product
    assert cb._par(cb.Par((cb.Id(1),)), cb.Id(1)) == cb.Id(2)
    # Id(0) — the unit object — contributes no wires
    assert cb._par(cb.Id(0), cb.Id(2)) == cb.Id(2)
    # an empty (or all-unit) product is the unit map itself
    assert cb._par() == cb.Id(0)
    assert cb._par(cb.Id(0)) == cb.Id(0)


def test_morph_stats_census():
    """The census counts drop/loop/structural and ignores non-terms."""
    assert cb.morph_stats(cb.Drop()).drop == 1
    st = cb.morph_stats(cb.Loop(cb.Id(2), 3))
    assert st.loop == 1
    assert st.structural == 1  # the Id step body
    # a non-morphism still counts as a visited node, nothing more
    st = cb.morph_stats(42)
    assert st.nodes == 1
    assert st.arr == st.structural == 0


def test_dag_size_counts_distinct_subterms():
    """Shared children count once; leaves and ops both count."""
    shared = Op.make("add", X, W)
    root = Op.make("mul", shared, Op.make("matmul", shared, W))
    assert cb.dag_size(root) == 5  # mul, matmul, add, x, w
    assert cb.dag_size(X) == 1


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
    # census: exactly one loop, one drop (the discarded final state)
    st = cb.morph_stats(prog)
    assert st.loop == 1
    assert st.drop == 1
    assert st.dup > 0


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
