"""Coverage wave-2 tests for ``mlopt_core.laws`` — the rewrite side
conditions exercised in BOTH directions, and the non-local pairing
passes over adversarial e-graph shapes.

The check/derive hooks in ``laws.tensor`` take ``bound`` dicts mapping
metavariables to terms and ``"$attr:NAME"`` keys to the matched node's
concrete attribute values — these tests build real bindings and prove
each guard accepts exactly the shapes it claims and vetoes the rest.
``laws.pairing``'s passes run over whole e-graphs, so their tests build
e-graphs with the degenerate structures the guards exist for (cyclic
e-classes, concat-valued weights, var-containing weights, uneven and
unknown weight shapes)."""

import torch

from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_core.laws.pairing import (
    share_duplicate_param_slices,
    share_duplicate_params,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _ill_typed():
    """A term whose shape is provably invalid (not a tuple)."""
    return Op.make("add", _v("ia", 2), _v("ib", 3))


# ---------------------------------------------------------------------------
#  laws/base — scalar / row / channel scale predicates on non-tuple shapes
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  laws/tensor — _check_linear_bias_compose shape guards
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  laws/tensor — _derive_split_sizes (asymmetric fused QKV)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  laws/tensor — repeat_kv chain + GQA absorption guards
# ---------------------------------------------------------------------------


def _repeat_bound(r=4, *, ud=3, es=None, rs=None, k_shape=(1, 4, 2, 3)):
    """A well-formed unsqueeze→expand→reshape binding: repeat r along
    the inserted axis of ``k`` — exactly repeat_interleave semantics."""
    k = _p("k", *k_shape)
    us = (*k_shape[:ud], 1, *k_shape[ud:])
    if es is None:
        es = (*us[:ud], r, *us[ud + 1 :])
    if rs is None:
        rs = (*us[: ud - 1], us[ud - 1] * r, *us[ud + 1 :])
    return {"k": k, "$attr:UDk": ud, "$attr:ESk": es, "$attr:RSk": rs}












# ---------------------------------------------------------------------------
#  laws/tensor — sdpa fold guards (score transpose / softmax dim / fill)
# ---------------------------------------------------------------------------


def _sdpa_bound(**kw):
    b = {
        "Q": _p("Q", 1, 4, 8, 6),
        "K": _p("K", 1, 4, 8, 6),
        "V": _p("V", 1, 4, 8, 6),
        "M": _p("M", 8, 8),
        "MK": _p("MK", 8, 8),
        "S": Const(0.125),
        "F": Const(float("-inf")),
        "$attr:TD1": -2,
        "$attr:TD2": -1,
        "$attr:SD": -1,
    }
    b.update(kw)
    return b








# ---------------------------------------------------------------------------
#  laws/pairing — small helpers
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  laws/pairing — _pair_shared_input guard paths
# ---------------------------------------------------------------------------


def _cyclic_eclass(eg):
    """An e-class whose every member is self-referential.

    Built by unioning a computed class into a leaf class, then
    dropping the acyclic member — the degenerate shape the pairing
    pass's var-reachability memo must cut rather than loop on."""
    leaf = eg.add_term(_p(f"cyc{len(eg._classes)}", 2))
    f = eg.add_enode("neg", (leaf,))
    eg.union(f, leaf)
    cid = eg.find(f)
    for n in list(eg._classes[cid].nodes):
        if n.op == "leaf":
            eg._classes[cid].nodes.discard(n)
    return cid














# ---------------------------------------------------------------------------
#  laws/pairing — share_duplicate_params edges
# ---------------------------------------------------------------------------


def test_share_duplicate_params_edges():
    torch.manual_seed(0)
    a = torch.randn(4, 4, dtype=torch.float64)
    src = {
        "p1": a.clone(),
        "p2": a.clone(),  # exact tie with p1
        "p3": torch.randn(4, 4, dtype=torch.float64),
        "odd": "not-a-tensor",  # non-tensor entry is skipped
    }
    eg = EGraph()
    for n in ("p1", "p2", "p3"):
        eg.add_term(Param(n, _T(4, 4)))
    groups = share_duplicate_params(eg, src)
    assert sorted(map(sorted, groups)) == [["p1", "p2"]]
    # the merge was witnessed with a named pointwise rule
    assert any(r.startswith("share#") for r in eg._rule_objs)

    # Re-running over the already-merged classes is a no-op —
    # the same groups are reported and nothing breaks.
    groups2 = share_duplicate_params(eg, src)
    assert sorted(map(sorted, groups2)) == [["p1", "p2"]]

    # witness=False merges without a synthesised rule.
    eg2 = EGraph()
    for n in ("p1", "p2"):
        eg2.add_term(Param(n, _T(4, 4)))
    groups3 = share_duplicate_params(
        eg2, {"p1": a.clone(), "p2": a.clone()}, witness=False
    )
    assert groups3 == [["p1", "p2"]]
    assert not any(r.startswith("share#") for r in eg2._rule_objs)


def _dup_head_setup():
    torch.manual_seed(0)
    h, d, i = 4, 3, 5
    blk = torch.randn(d, i, dtype=torch.float64)
    other = torch.randn(d, i, dtype=torch.float64)
    W = torch.cat([blk, other, blk, blk], dim=0)  # heads 0,2,3 tied
    return h, d, i, W


def test_share_duplicate_param_slices_name_collision():
    """A colliding dedup name gets suffixed — the planted tensor is
    never clobbered and the offer carries the renamed param."""
    h, d, i, W = _dup_head_setup()
    source = {"W": W}
    # a source tensor absent from the graph is skipped outright
    source["ghost"] = W.clone()
    # pre-register a DIFFERENT tensor under the name the pass wants —
    # it must rename to W__heads4_1 rather than clobber it.
    source["W__heads4"] = torch.zeros(2, d, i, dtype=torch.float64)

    eg = EGraph()
    eg.add_term(Param("W", _T(h * d, i)))
    offers = share_duplicate_param_slices(eg, source)
    assert len(offers) == 1
    off = offers[0]
    assert off["dedup_param"] == "W__heads4_1"
    assert tuple(source[off["dedup_param"]].shape) == (2, d, i)
    assert torch.equal(source["W__heads4"], torch.zeros(2, d, i))


def test_share_duplicate_param_slices_rerun_reuses_name():
    """A second run over the same source sees the registered dedup
    tensor already value-equal → registration is skipped and the
    offer reuses the same name."""
    h, d, i, W = _dup_head_setup()
    source = {"W": W}
    eg = EGraph()
    eg.add_term(Param("W", _T(h * d, i)))
    offers = share_duplicate_param_slices(eg, source)
    assert len(offers) == 1
    assert offers[0]["dedup_param"] == "W__heads4"
    offers2 = share_duplicate_param_slices(eg, source)
    assert len(offers2) == 1
    assert offers2[0]["dedup_param"] == "W__heads4"


def test_share_duplicate_param_slices_witness_off():
    """witness=False unions without a synthesised rule."""
    h, d, i, W = _dup_head_setup()
    eg = EGraph()
    eg.add_term(Param("W2", _T(h * d, i)))
    offers = share_duplicate_param_slices(
        eg, {"W2": W.clone()}, witness=False
    )
    assert len(offers) == 1
    assert not any(r.startswith("share_slices#") for r in eg._rule_objs)
