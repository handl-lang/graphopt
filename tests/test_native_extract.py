"""Native extraction + ``dag_cost`` parity — Rust walk vs reference.

``NativeEngine.extract_best`` drives the per-class DP in Rust while
``cost_fn`` stays a Python callable.  These tests pin byte-identical
results against the reference ``EGraph`` extractor — member order
(including commutativity spellings), the ``(total, nops)`` tie-break,
the param-only discount, ``overrides``/``bans``/``_cache_out``, the
``fusion_epsilon`` band — plus ``mlopt_native.dag_cost`` against
``mlopt_core.cost.dag_cost`` on shared-DAG and fold-heavy terms, and
the contraction primitives against ``contraction_policy``'s Python.

Self-skips when the wheel is absent (``maturin develop --release`` in
``packages/mlopt-native``).
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from mlopt_core.cost import (
    count_cost,
    dag_cost,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost,
)
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, Op, Param, TensorType, Var, op_repr
from mlopt_core.laws.tensor import (
    ASSOC_ADD,
    COMM_ADD,
    COMM_MUL,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
)

pytest.importorskip("mlopt_native")
from mlopt_native import NativeEngine
from mlopt_native import contraction as _ncon
from mlopt_native import dag_cost as native_dag_cost

_SYMMETRY = [COMM_ADD, ASSOC_ADD, COMM_MUL, DISTRIBUTE_MUL, FACTOR_MUL]


def _pair(term: Any, rules: list, iters: int = 8) -> tuple:
    """Saturate the same term under both engines; return both roots."""
    py = EGraph()
    pr = py.add_term(term)
    py.run(rules, pr, max_iterations=iters, max_nodes=200_000)
    nat = NativeEngine()
    nr = nat.add_term(term)
    nat.run(rules, nr, max_iterations=iters, max_nodes=200_000)
    return py, pr, nat, nr


def _assert_same_extraction(py, pr, nat, nr, cost_fn=flops_cost, **kw):
    pt = py.extract_best(pr, cost_fn, **kw)
    nt = nat.extract_best(nr, cost_fn, **kw)
    assert (pt is None) == (nt is None)
    if pt is not None:
        assert op_repr(nt) == op_repr(pt)
    return pt, nt


def test_native_extract_commutative_class() -> None:
    """add(x,y) and add(y,x) in one class: the canonical member wins."""
    x = Var("x", TensorType((4,)))
    y = Var("y", TensorType((4,)))
    t = Op.make("add", x, y)
    py, pr, nat, nr = _pair(t, [COMM_ADD])
    _pt, nt = _assert_same_extraction(py, pr, nat, nr, count_cost)
    # The leaf-free cheaper member is a Var — a Var member exists.
    assert nt is not None


def test_native_extract_tie_break_prefers_fewer_ops() -> None:
    """``add(W, 0)`` ~ ``W`` at equal cost: fewest ops wins."""
    x = Var("x", TensorType((4,)))
    w = Param("W", TensorType((4, 4)))
    t = Op.make("matmul", x, Op.make("add", w, Const(0)))
    py, pr, nat, nr = _pair(t, _SYMMETRY)
    _assert_same_extraction(py, pr, nat, nr)


def test_native_extract_param_only_discount() -> None:
    """Weight-chain folds price at zero: extraction prefers them."""
    w1 = Param("w1", TensorType((8, 8)))
    w2 = Param("w2", TensorType((8, 8)))
    x = Var("x", TensorType((4, 8)))
    t = Op.make("matmul", Op.make("matmul", x, w1), w2)
    py, pr, nat, nr = _pair(t, [*_SYMMETRY, ASSOC_ADD])
    _assert_same_extraction(py, pr, nat, nr)


def test_native_extract_overrides_and_bans() -> None:
    """``overrides=`` forces a member; ``bans=`` excludes it back."""
    x = Var("x", TensorType((4,)))
    y = Var("y", TensorType((4,)))
    t = Op.make("add", x, y)
    py, pr, nat, nr = _pair(t, [COMM_ADD])
    canon = nat.find(nr)
    assert canon == py.find(pr)
    # Pick a non-leaf member and force it through ``overrides``.
    node = next(n for n in py._classes[canon].nodes if n.op != "leaf")
    if node is None:
        return
    _assert_same_extraction(
        py, pr, nat, nr, count_cost, overrides={canon: node}
    )
    # Banning every member leaves nothing extractable.
    all_nodes = set(py._classes[canon].nodes)
    pt = py.extract_best(pr, count_cost, bans={canon: all_nodes})
    nt = nat.extract_best(nr, count_cost, bans={canon: all_nodes})
    assert pt is None and nt is None


def test_native_extract_cache_out_shape() -> None:
    """``_cache_out`` entries mirror the Python tuple layout."""
    x = Var("x", TensorType((4,)))
    y = Var("y", TensorType((4,)))
    t = Op.make("mul", Op.make("add", x, y), x)
    py, pr, nat, nr = _pair(t, _SYMMETRY)
    pcache: dict = {}
    ncache: dict = {}
    _assert_same_extraction(
        py, pr, nat, nr, flops_cost, _cache_out=pcache
    )
    nat.extract_best(nr, flops_cost, _cache_out=ncache)
    assert set(pcache) == set(ncache)
    for k in pcache:
        pe, ne = pcache[k], ncache[k]
        # (total, term, used, param_only, nops, cfn(term), eo, rc)
        assert pe[0] == ne[0]
        assert (pe[1] is None) == (ne[1] is None)
        if pe[1] is not None:
            assert op_repr(pe[1]) == op_repr(ne[1])
        assert pe[3:] == ne[3:]


def test_native_extract_fusion_band() -> None:
    """``fusion_epsilon`` armed: identical member pick under the band."""
    x = Var("x", TensorType((4, 8)))
    y = Var("y", TensorType((4, 8)))
    t = Op.make("add", Op.make("mul", x, y), Op.make("mul", y, x))
    py, pr, nat, nr = _pair(t, _SYMMETRY)
    for eps in (0.0, 0.05, 0.5):
        _assert_same_extraction(
            py, pr, nat, nr, flops_cost, fusion_epsilon=eps
        )
        _assert_same_extraction(
            py, pr, nat, nr, count_cost, fusion_epsilon=eps
        )


def test_native_extract_cost_models() -> None:
    """flops / launch-aware / count / param-bytes all agree."""
    w = Param("w", TensorType((4, 4)))
    x = Var("x", TensorType((4, 4)))
    t = Op.make("matmul", Op.make("add", x, x), Op.make("add", w, w))
    py, pr, nat, nr = _pair(t, _SYMMETRY)
    for cfn in (flops_cost, launch_aware_cost, count_cost):
        _assert_same_extraction(py, pr, nat, nr, cfn)
    _assert_same_extraction(py, pr, nat, nr, param_bytes_cost)


def test_native_extract_repeated_is_stable() -> None:
    """A second extract over the same graph re-derives the same term."""
    x = Var("x", TensorType((4,)))
    y = Var("y", TensorType((4,)))
    t = Op.make("mul", Op.make("add", x, y), Op.make("add", y, x))
    _, _, nat, nr = _pair(t, _SYMMETRY)
    t1 = nat.extract_best(nr, flops_cost)
    t2 = nat.extract_best(nr, flops_cost)
    assert op_repr(t1) == op_repr(t2)


# ---------------------------------------------------------------------------
#  dag_cost
# ---------------------------------------------------------------------------


def _sample_terms() -> list:
    x = Var("x", TensorType((4, 8)))
    w = Param("w", TensorType((8, 8)))
    w2 = Param("w2", TensorType((8, 8)))
    shared = Op.make("matmul", x, w)
    return [
        x,
        w,
        Op.make("add", shared, shared),  # shared subtree bills once
        Op.make("matmul", Op.make("add", w, w2), x),  # param-only fold
        Op.make("add", Op.make("matmul", w, w2), x),  # fold + var mix
        Op.make("mul", Op.make("gelu", x), Const(2)),
    ]


@pytest.mark.parametrize(
    "cost_fn", [flops_cost, launch_aware_cost, count_cost]
)
def test_native_dag_cost_matches(cost_fn: Any) -> None:
    """``mlopt_native.dag_cost`` == ``mlopt_core.cost.dag_cost``."""
    for term in _sample_terms():
        assert native_dag_cost(term, cost_fn) == pytest.approx(
            dag_cost(term, cost_fn)
        )


def test_native_dag_cost_param_storage() -> None:
    """``charges_param_only`` models bill the folded subtrees."""
    for term in _sample_terms():
        assert native_dag_cost(term, param_bytes_cost) == pytest.approx(
            dag_cost(term, param_bytes_cost)
        )


def test_native_dag_cost_shared_memo() -> None:
    """A caller-supplied memo survives with identical contents."""
    term = _sample_terms()[2]
    pm: dict = {}
    nm: dict = {}
    assert native_dag_cost(term, flops_cost, memo=nm) == pytest.approx(
        dag_cost(term, flops_cost, memo=pm)
    )
    # Both runs priced through their memos — equal key sets.
    assert set(pm) == set(nm)


# ---------------------------------------------------------------------------
#  Contraction primitives
# ---------------------------------------------------------------------------


def _ref_greedy(
    tensors: Any,
    sizes: dict[int, int],
    rng: random.Random | None = None,
    top_k: int = 1,
) -> float:
    """The pre-port Python ``greedy`` verbatim."""

    def pc(x, y, s):
        c = 1.0
        for i in x | y:
            c *= s[i]
        return c

    def mt(ts, a, b):
        m = ts[a] ^ ts[b]
        return [t for k, t in enumerate(ts) if k not in (a, b)] + [m]

    ts = [frozenset(t) for t in tensors]
    total = 0.0
    while len(ts) > 1:
        cands = sorted(
            (pc(ts[a], ts[b], sizes), a, b)
            for a in range(len(ts))
            for b in range(a + 1, len(ts))
        )
        if rng is None:
            c, a, b = cands[0]
        else:
            c, a, b = cands[rng.randrange(min(top_k, len(cands)))]
        total += c
        ts = mt(ts, a, b)
    return total


def test_native_contraction_parity() -> None:
    """greedy/cost_of_order identical on seeded bond networks."""
    from mlopt_torch.contraction_policy import random_bond_network

    for n in (3, 8, 16, 24, 40):
        tensors, sizes = random_bond_network(n, seed=n * 31)
        assert _ncon.greedy(tensors, sizes) == _ref_greedy(
            tensors, sizes
        )
        # Randomised arm: identical draws -> identical cost.
        r1, r2 = random.Random(9), random.Random(9)
        assert _ncon.greedy(
            tensors, sizes, rng=r1, top_k=4
        ) == _ref_greedy(tensors, sizes, rng=r2, top_k=4)
        order, total = _ncon.greedy_order(tensors, sizes)
        assert _ncon.cost_of_order(tensors, sizes, order) == total


def test_native_contraction_shipped_fns() -> None:
    """``contraction_policy`` delegates to the native core when built."""
    from mlopt_torch import contraction_policy as cp

    tensors, sizes = cp.random_bond_network(12, seed=5)
    assert cp.greedy(tensors, sizes) == _ref_greedy(tensors, sizes)
    ts = [frozenset(t) for t in tensors]
    assert cp.pair_cost(ts[0], ts[1], sizes) == _ncon.pair_cost(
        ts[0], ts[1], sizes
    )
    assert cp.merge_tensors(ts, 0, 1) == _ncon.merge_tensors(ts, 0, 1)
