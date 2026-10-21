"""Substrate test functions whose assertions cross the domain seam.

Plan 0032 phase 4 carved these out of the substrate test files that
moved to ``handl-lang/eqsat`` (``test_cond_laws.py``,
``test_derive_laws.py``, ``test_minted_laws.py``, ``test_rules.py``,
``test_perf_oracles.py``, ``test_core_torch_free.py``,
``test_cost.py``, ``test_native_extract.py``,
``test_backend_neutral.py``, ``test_opmeta.py``): each exercises
substrate machinery *through* a domain package that stays here
(``mlopt_discovery``, ``mlopt_carriers``, ``mlopt_torch``, ``torch``,
or the ambient carrier registry), so they run against the pinned
eqsat git dep from this repo.  The private helpers below are
verbatim copies of the ones the source files carried.
"""

from __future__ import annotations

import json
import random
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

import mlopt_core.meta as meta
import mlopt_core.ops as ops_mod
from mlopt_core import opmeta as om
from mlopt_core.cost import param_bytes_cost, param_bytes_cost_for
from mlopt_core.ir import Op, Param, TensorType, Var, op_repr
from mlopt_core.ops import OpTable
from mlopt_core.pipeline import Backend
from mlopt_orchestrator import Optimizer
from mlopt_core.laws import (
    BATCH_NORM_RULES,
    CONV_BN_FOLD,
    CONV_FOLD_RULES,
    minted,
)
from mlopt_core.laws.cond import derive_from_data, derive_to_data
from mlopt_core.laws.tensor import DISTRIBUTE_MUL, FACTOR_MUL, GQA_ABSORB


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _T(*shape: int) -> TensorType:
    return TensorType(tuple(shape))


def _spelled(cout: int = 6, bn_c: int | None = None) -> Op:
    """The eval-mode ``BatchNorm2d(Conv2d(x))`` spelling the mint saw."""
    bc = cout if bn_c is None else bn_c
    return Op.make(
        "batch_norm",
        Op.make(
            "conv2d",
            Var("x", _T(1, 4, 8, 8)),
            Param("W", _T(cout, 4, 3, 3)),
            Param("b", _T(cout)),
            stride=(1, 1),
            padding=(1, 1),
        ),
        Param("g", _T(bc)),
        Param("be", _T(bc)),
        Param("mu", _T(bc)),
        Param("var", _T(bc)),
        training=False,
        momentum=0.1,
        eps=1e-05,
        cudnn_enabled=False,
    )


def _forced_member_term(eg, eid, op_name):
    """Extract the offered (non-leaf) member of a class by forcing the
    override to the enode with the given op — the same coordinated-
    extraction mechanism the pipeline uses for pairing offers."""
    from mlopt_core.cost import count_cost

    cls = eg.get_class(eid)
    node = next(n for n in cls.nodes if n.op == op_name)
    return eg.extract_best(
        eid, count_cost, overrides={eg.find(eid): node}
    )


# --- from eqsat tests/test_cond_laws.py -------------------------------


def test_mm_addends_rank_ge2_requires_contraction_axis():
    """The rank>=2 branch must align the CONTRACTION axis (-2), not
    merely the rank: ``a=(3,5)``, ``b=(1,5)`` broadcasts the
    contraction axis (1 -> 3) yet ``matmul(W, a)``/``matmul(W, b)``
    cannot contract with one ``W`` — a measured rhs-err the old
    leading-axis-padding argument accepted.  The minimal closure
    ``dim-eq(a, -2, b, -2)`` declines it and keeps every sound
    rank>=2 mixed-rank binding."""
    from mlopt_core.egraph.terms import _term_instantiate
    from mlopt_discovery import oracle as vo

    bad = {"W": _p("W", 4, 3), "a": _v("a", 3, 5), "b": _v("b", 1, 5)}
    assert not DISTRIBUTE_MUL.check(bad)
    assert not FACTOR_MUL.check(bad)
    # the same binding is a measured rhs-err, not an equal site.
    lhs = _term_instantiate(DISTRIBUTE_MUL.lhs, bad)
    rhs = _term_instantiate(DISTRIBUTE_MUL.rhs, bad)
    assert vo.eval_instance(lhs, rhs)[0] == "rhs-err"
    # a sound rank>=2 mixed-rank pair (shared -2) still clears.
    good = {
        "W": _p("W", 5, 4),
        "a": _v("a", 4, 3),
        "b": _v("b", 2, 4, 3),
    }
    assert DISTRIBUTE_MUL.check(good)
    lhs = _term_instantiate(DISTRIBUTE_MUL.lhs, good)
    rhs = _term_instantiate(DISTRIBUTE_MUL.rhs, good)
    assert vo.eval_instance(lhs, rhs)[0] == "equal"


def test_gqa_absorb_repeat_requires_rank4_operands():
    """``sdpa`` + ``enable_gqa`` repeats along the head axis at ``-3``;
    the pattern's ``transpose(1, 2)`` only lands the heads there when
    the leaves are rank-4 ``(b, t, h, d)``.  A rank-3 spelling —
    ``q=(2,6,4)``, ``k=v=(2,3,4)`` with the consistent
    ``unsq(-2) → expand (2,3,2,4) → reshape (2,6,4)`` chain —
    satisfies every repeat clause but its minted ``enable_gqa`` RHS
    cannot contract: a measured ``rhs-err``
    (``Expected size for first two dimensions of batch2 tensor to be:
    [2, 6] but got: [2, 3]``).  The ``rank == 4`` clauses decline it.
    """
    from mlopt_core.egraph.terms import _term_instantiate
    from mlopt_discovery import oracle as vo

    # the measured counterexample: all repeat clauses pass, rank-3
    # leaves make the enable_gqa RHS ill-typed.
    bad = {
        "q": _v("q", 2, 6, 4),
        "k": _v("k", 2, 3, 4),
        "v": _v("v", 2, 3, 4),
        "$attr:UDk": -2,
        "$attr:ESk": (2, 3, 2, 4),
        "$attr:RSk": (2, 6, 4),
        "$attr:UDv": -2,
        "$attr:ESv": (2, 3, 2, 4),
        "$attr:RSv": (2, 6, 4),
        "$attr:D": 1e-5,
        "$attr:C": False,
    }
    assert not GQA_ABSORB.check(bad)
    # …and it was a real rhs-err, not a vacuous decline: the
    # instantiated RHS genuinely cannot denote.
    lhs = _term_instantiate(GQA_ABSORB.lhs, bad)
    rhs = _term_instantiate(GQA_ABSORB.rhs, bad)
    assert vo.eval_instance(lhs, rhs)[0] == "rhs-err"
    # the rank-4 (b, t, h, d) corner — the binding the chained
    # enumerator mints — still clears and evaluates equal.
    good = {
        "q": _v("q", 2, 3, 4, 4),
        "k": _v("k", 2, 3, 2, 4),
        "v": _v("v", 2, 3, 2, 4),
        "$attr:UDk": -2,
        "$attr:ESk": (2, 3, 2, 2, 4),
        "$attr:RSk": (2, 3, 4, 4),
        "$attr:UDv": -2,
        "$attr:ESv": (2, 3, 2, 2, 4),
        "$attr:RSv": (2, 3, 4, 4),
        "$attr:D": 1e-5,
        "$attr:C": False,
    }
    assert GQA_ABSORB.check(good)
    lhs = _term_instantiate(GQA_ABSORB.lhs, good)
    rhs = _term_instantiate(GQA_ABSORB.rhs, good)
    assert vo.eval_instance(lhs, rhs)[0] == "equal"


# --- from eqsat tests/test_derive_laws.py -----------------------------


def test_migrated_carrier_rules_carry_dspec():
    """Every migrated carrier derive is now declarative data.

    ``missing_hooks`` must not report ``"derive"`` — the folded hook
    is the spec-only closure — and the spec must survive the
    law-record JSON hop minting identical attrs.
    """
    from mlopt_carriers import decode_geom, decode_laws, om, trace
    from mlopt_core.laws.serialize import (
        law_from_data,
        law_to_data,
        missing_hooks,
    )

    migrated = [
        trace.TR_VANISH_SPLIT,
        trace.TR_VANISH_MERGE,
        trace.TR_SUPERPOSE_REV,
        trace.TR_SLIDE,
        trace.TR_SLIDE_REV,
        trace.TR_EXPAND,
        om.SPLIT_TRANSPOSE,
        om.DECAY_BLOCK_DIAG,
        om.DECAY_BLOCK_CROSS,
        decode_laws.INDEX_SELECT_DEDUP,
        decode_geom.CAT_NARROW_MERGE,
    ]
    for rule in migrated:
        assert rule.dspec is not None, rule.name
        assert "derive" not in missing_hooks(rule), rule.name
        blob = json.dumps(derive_to_data(rule.dspec))
        assert (
            derive_from_data(json.loads(blob)) == rule.dspec
        ), rule.name
        rebuilt = law_from_data(
            json.loads(json.dumps(law_to_data(rule)))
        )
        assert rebuilt.dspec == rule.dspec, rule.name


def test_migrated_carrier_derives_mint_identically():
    """The specs produce the same attr maps the Python hooks did."""
    from mlopt_carriers import decode_geom, decode_laws, om, trace

    v = lambda n, *s: Var(n, TensorType(tuple(s)))  # noqa: E731
    # tr_vanish_split: usize (u,v) → two int dims
    assert trace.TR_VANISH_SPLIT.derive(
        {"f": v("f", 8, 8), "$attr:UV": (3, 2)}
    ) == {"$attr:DU": 3, "$attr:DV": 2}
    assert trace.TR_VANISH_SPLIT.derive({"$attr:UV": (3, 2, 1)}) is None
    # tr_slide / tr_slide_rev: the widened block axis
    g = v("g", 5, 7)
    assert trace.TR_SLIDE.derive({"g": g, "$attr:DU": 3}) == {
        "$attr:DY": 2
    }
    assert trace.TR_SLIDE.derive({"g": v("g2", 3, 7), "$attr:DU": 3}) is None
    assert trace.TR_SLIDE.derive({"g": v("g3", None, 7), "$attr:DU": 3}) is None
    assert trace.TR_SLIDE_REV.derive(
        {"g": v("g4", 7, 5), "$attr:DU": 3}
    ) == {"$attr:DX": 2}
    # tr_expand: (du, f_i − du) per axis
    assert trace.TR_EXPAND.derive({"f": v("f", 6, 8), "$attr:DU": 4}) == {
        "$attr:RS": (4, 2),
        "$attr:CS": (4, 4),
    }
    assert trace.TR_EXPAND.derive(
        {"f": v("f", 6, 8), "$attr:DU": 0}
    ) is None
    # tr_vanish_merge / tr_superpose_rev: the fused product size
    b = {"$attr:DU": 4, "$attr:DV": 2}
    assert trace.TR_VANISH_MERGE.derive(b) == {"$attr:UV": (4, 2)}
    assert trace.TR_SUPERPOSE_REV.derive(b) == {"$attr:UV": (4, 2)}
    # split_transpose: the pre-transpose axis pick
    dims = {"$attr:SD": -2, "$attr:D0": -2, "$attr:D1": -1}
    assert om.SPLIT_TRANSPOSE.derive(dims) == {"$attr:DP": -1}
    assert om.SPLIT_TRANSPOSE.derive(
        {**dims, "$attr:SD": -1}
    ) == {"$attr:DP": -2}
    assert om.SPLIT_TRANSPOSE.derive({**dims, "$attr:SD": 0}) == {
        "$attr:DP": 0
    }
    assert (
        om.SPLIT_TRANSPOSE.derive({**dims, "$attr:SD": "x"}) is None
    )
    # decay_block_diag: TC = SZ[I] on the (i,i) block
    ok = {"$attr:SZ": (8, 8), "$attr:I": 1, "$attr:T": 16}
    assert om.DECAY_BLOCK_DIAG.derive(ok) == {"$attr:TC": 8}
    assert om.DECAY_BLOCK_DIAG.derive({**ok, "$attr:I": 5}) is None
    assert om.DECAY_BLOCK_DIAG.derive({**ok, "$attr:SZ": (8, "x")}) is None
    assert om.DECAY_BLOCK_DIAG.derive({**ok, "$attr:T": 12}) is None
    # decay_block_cross: extents + the strict-lower offset
    ok2 = {"$attr:SZ": (4, 5), "$attr:T": 9, "$attr:I": 1, "$attr:J": 0}
    assert om.DECAY_BLOCK_CROSS.derive(ok2) == {
        "$attr:TI": 5,
        "$attr:TJ": 4,
        "$attr:OD": 4,
    }
    assert om.DECAY_BLOCK_CROSS.derive({**ok2, "$attr:J": 1}) is None
    assert om.DECAY_BLOCK_CROSS.derive({**ok2, "$attr:T": 8}) is None
    # index_select_dedup: uniques + position map
    assert decode_laws.INDEX_SELECT_DEDUP.derive(
        {"$attr:I": (7, 3, 7, 0, 3)}
    ) == {"$attr:U": (7, 3, 0), "$attr:VI": (0, 1, 0, 2, 1)}
    # cat_narrow_merge: the merged extent
    assert decode_geom.CAT_NARROW_MERGE.derive(
        {"$attr:L1": 2, "$attr:L2": 4}
    ) == {"$attr:LT": 6}
    assert (
        decode_geom.CAT_NARROW_MERGE.derive({"$attr:L1": "2"}) is None
    )


# --- from eqsat tests/test_minted_laws.py -----------------------------


def test_shipped_object_restores_through_the_store():
    """store → admit rebuilds the transported hooks; fires identically."""
    from mlopt_core.laws.serialize import alpha_key
    from mlopt_discovery import evidence as ev

    lib = {r.name: r for r in [*BATCH_NORM_RULES, *CONV_FOLD_RULES]}
    conn = ev.connect(":memory:")
    try:
        # the full premise closure as store rows — leaf parents plus
        # each materialized ``syn_*`` intermediate
        for nm in (
            "batch_norm_expand",
            "conv_channel_shift_sub_sp",
            "conv_channel_scale_sp",
            "conv_channel_scale_l_sp",
            "conv_channel_shift_sp",
        ):
            ev.store_object(conn, lib[nm], kind="law", cert=None)
        for nm in minted._RECORDS:
            ev.store_object(
                conn,
                minted.materialize(nm, lib),
                kind="abstraction",
                cert=None,
            )
        key = repr(alpha_key(CONV_BN_FOLD.lhs, CONV_BN_FOLD.rhs))
        got = ev.admit_object(conn, key)
        assert got is not None
        rebuilt, rec = got
        assert sorted(rec["transported"]) == ["check", "derive"]
        term = _spelled()
        assert op_repr(
            meta.apply_rewrite_at(rebuilt, term, ())
        ) == op_repr(meta.apply_rewrite_at(CONV_BN_FOLD, term, ()))
    finally:
        conn.close()


# --- from eqsat tests/test_rules.py -----------------------------------


def test_share_duplicate_param_slices_offers_dedup_member():
    """Heads 0,2 bitwise-equal / 1,3 distinct -> the W class gains a
    member that stores only the unique head-blocks and evaluates
    fp64-exact through ir_to_torch_module."""
    import torch

    from mlopt_core.egraph import EGraph
    from mlopt_core.ir import IR
    from mlopt_core.laws import share_duplicate_param_slices
    from mlopt_torch.torch_bridge import ir_to_torch_module

    torch.manual_seed(0)
    h, d, i = 4, 3, 5
    o = h * d
    a = torch.randn(d, i, dtype=torch.float64)
    b = torch.randn(d, i, dtype=torch.float64)
    c = torch.randn(d, i, dtype=torch.float64)
    W = torch.cat([a, b, a, c], dim=0)  # heads 0 and 2 tied
    source = {"W": W}

    eg = EGraph()
    w_eid = eg.add_term(Param("W", TensorType((o, i))))
    offers = share_duplicate_param_slices(eg, source)

    assert len(offers) == 1
    off = offers[0]
    assert off["param"] == "W"
    assert off["heads"] == h
    assert off["head_dim"] == d
    assert off["unique"] == 3  # {a, b, c}
    assert off["index_map"] == (0, 1, 0, 2)
    assert off["stored_after"] == 3 * d * i
    assert off["stored_after"] < off["stored_before"] == o * i

    # The deduplicated tensor was registered for lowering.
    dedup = source[off["dedup_param"]]
    assert tuple(dedup.shape) == (3, d, i)
    assert torch.equal(dedup[0], a) and torch.equal(dedup[1], b)
    assert torch.equal(dedup[2], c)

    # W's e-class now holds the reshape(index_select(D)) member.
    member = _forced_member_term(eg, w_eid, "reshape")
    assert member.op == "reshape"
    assert member.args[0].op == "index_select"
    assert member.args[0].args[0].name == off["dedup_param"]

    # The offered member stores fewer values than the original param.
    ir = IR(root=member, inputs=[], input_names=set(), params={})
    mod = ir_to_torch_module(ir, param_values=source)
    stored = sum(p.numel() for p in mod.parameters())
    assert stored == 3 * d * i < o * i

    # ... and evaluates bitwise-equal to W (float64, exact gather).
    out = mod(torch.zeros(1))  # no Var leaves: arg unused
    assert out.dtype == torch.float64
    assert torch.equal(out, W)

    # The merge carried a replayable exact witness.
    edge = eg.merge_log[-1]
    assert edge.rule.startswith("share_slices#")
    wit = eg._rule_objs[edge.rule]
    assert wit.error_bound is None
    assert wit.lhs == Param("W", TensorType((o, i)))
    assert wit.rhs == member

    # Under the storage cost model the member wins extraction on its
    # own — 45 stored values < 60 — no coordinated override needed.
    from mlopt_core.cost import param_bytes_cost_for

    best = eg.extract_best(w_eid, param_bytes_cost_for(source))
    assert best == member


def test_share_duplicate_param_slices_no_offer_when_all_distinct():
    """All-distinct head slices -> no member offered, no dedup tensor
    registered, the class keeps only its leaf."""
    import torch

    from mlopt_core.egraph import EGraph
    from mlopt_core.laws import share_duplicate_param_slices

    torch.manual_seed(1)
    W = torch.randn(
        12, 5, dtype=torch.float64
    )  # 4 heads x 3 rows, all distinct
    source = {"W": W}

    eg = EGraph()
    w_eid = eg.add_term(Param("W", TensorType((12, 5))))
    offers = share_duplicate_param_slices(eg, source)

    assert offers == []
    assert {n.op for n in eg.get_class(w_eid).nodes} == {"leaf"}
    assert list(source) == ["W"]  # nothing registered


def test_share_duplicate_param_slices_guards():
    """Only 2-D params that actually appear in the e-graph are
    considered; a dedup form must strictly shrink storage."""
    import torch

    from mlopt_core.egraph import EGraph
    from mlopt_core.laws import share_duplicate_param_slices

    torch.manual_seed(2)
    # A 1-D "weight" (bias-like) with a duplicated half — not 2-D, skip.
    v = torch.randn(4, dtype=torch.float64)
    v1 = torch.cat([v, v])
    # A 2-D param whose duplicated halves don't fit head_counts: o=2,
    # the only divisor in range is h=2 with d=1 — two equal rows DO
    # qualify, so use o=1*2... simplest non-shrinking case: W whose
    # only equal-block factorisation is h=o (all rows unique under it).
    W2 = torch.randn(6, 4, dtype=torch.float64)
    W2[3:] = W2[:3]  # halves equal: h=2 works
    # W3 is 2-D with duplicated rows but never appears in the e-graph.
    W3 = torch.cat([v.reshape(2, 2), v.reshape(2, 2)], dim=0)
    source = {"bias": v1, "W": W2, "absent": W3}

    eg = EGraph()
    eg.add_term(Param("bias", TensorType((8,))))
    _w_eid = eg.add_term(Param("W", TensorType((6, 4))))
    offers = share_duplicate_param_slices(eg, source)

    # Only W qualifies: h=2 splits it into two equal (3,4) halves ->
    # one unique block, index_map (0, 0).
    assert len(offers) == 1
    off = offers[0]
    assert off["param"] == "W" and off["heads"] == 2
    assert off["unique"] == 1 and off["index_map"] == (0, 0)
    assert off["stored_after"] == 3 * 4 < 6 * 4
    # 'bias' (1-D) and 'absent' (not in graph) produced no dedup params.
    assert [
        n
        for n in source
        if n != "W" and n != off["dedup_param"] and n != "bias"
    ] == ["absent"]
    assert off["dedup_param"] in source


# --- from eqsat tests/test_perf_oracles.py ----------------------------


def test_search_plumbs_stop_and_patience():
    """``search(..., stop="improving", patience=..)`` reaches eg.run."""
    import torch
    import torch.nn as nn
    from mlopt_orchestrator import search
    from mlopt_torch.adapters import TorchSource

    torch.manual_seed(0)
    m = nn.Linear(8, 8, bias=False).double()
    x = torch.randn(2, 8, dtype=torch.float64)
    res = search(
        m,
        x,
        source=TorchSource(),
        stop="improving",
        patience=1,
        max_iterations=10,
    )
    assert res.stats["stop"] in ("improving", "fixed_point")
    assert "improved" in res.stats


# --- from eqsat tests/test_core_torch_free.py -------------------------


def test_register_core_bindings_is_idempotent():
    """Re-registering the same table object does not duplicate it."""
    from mlopt_torch.torch_bridge import _CORE_TORCH_BINDINGS

    before = len(ops_mod._CORE_BINDING_TABLES)
    ops_mod.register_core_bindings(_CORE_TORCH_BINDINGS)
    assert len(ops_mod._CORE_BINDING_TABLES) == before
    # ... and the core table is unaffected.
    assert "matmul" in OpTable.core().torch_bindings


def test_concrete_eval_backend_is_registered():
    """Importing the torch adapter registers the backend; core reads it
    only through the injected object."""
    import mlopt_torch.meta_eval  # noqa: F401

    assert meta._concrete_eval is not None
    assert hasattr(meta._concrete_eval, "eval_term")


def test_pairing_exact_equal_is_value_agnostic():
    """The sharing passes' comparator is duck-typed: shape mismatch is
    never equal, and equal-shape values compare elementwise (NaN != NaN,
    -0.0 == +0.0 — ``torch.equal`` semantics, no torch import)."""
    import torch
    from mlopt_core.laws.pairing import _exact_equal, _is_tensor

    a = torch.zeros(2, 3)
    assert _exact_equal(a, a.clone())
    # different shapes never compare equal
    assert not _exact_equal(a, torch.zeros(3, 2))
    assert not _exact_equal(torch.zeros(2), torch.zeros(3))
    # torch.equal parity: NaN != NaN, -0.0 == +0.0
    nan = torch.tensor([float("nan")])
    assert not _exact_equal(nan, nan.clone())
    assert _exact_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    # the duck-typed tensor check rejects non-tensors
    assert _is_tensor(a)
    assert not _is_tensor("not-a-tensor")


def test_full_without_ambient_returns_private_table(monkeypatch):
    """With no backend ambient dict registered ``full()`` returns a
    private composed table rather than raising — and the ambiently
    installed carriers still compose in (needs mlopt_carriers, so it
    runs here, not in eqsat)."""
    monkeypatch.setattr(ops_mod, "_AMBIENT_BINDINGS", None)
    t = OpTable.full()
    # A plain dict (not the ambient subclass) — nothing to seat on.
    assert type(t.torch_bindings) is dict
    # Carrier ops still compose in (carriers are installed here).
    assert "omd_elem" in t.torch_bindings


# --- from eqsat tests/test_cost.py ------------------------------------


def test_param_bytes_source_tensors():
    """source_tensors is authoritative for numel; TensorType is fallback."""
    import torch

    x = Var("x", TensorType((4, 8)))
    W = Param(
        "W", TensorType((None, None))
    )  # shape unknown at type level
    t = Op.make("linear", x, W)
    src = {"W": torch.zeros(8, 16)}
    assert param_bytes_cost(t, src) == 8 * 16
    # names absent from source_tensors fall back to the TensorType
    U = Param("U", TensorType((3, 5)))
    t2 = Op.make(
        "add", Op.make("linear", x, W), Op.make("linear", x, U)
    )
    assert param_bytes_cost(t2, src) == 8 * 16 + 3 * 5
    # the bound-closure form prices identically
    assert param_bytes_cost_for(src)(t2) == 8 * 16 + 3 * 5


# --- from eqsat tests/test_native_extract.py --------------------------
# These need BOTH mlopt_native (eqsat, maturin-only) and mlopt_torch
# (graphopt): dormant unless the optional engine is built into this env.


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
    _ncon = pytest.importorskip("mlopt_native.contraction")
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
    _ncon = pytest.importorskip("mlopt_native.contraction")
    from mlopt_torch import contraction_policy as cp

    tensors, sizes = cp.random_bond_network(12, seed=5)
    assert cp.greedy(tensors, sizes) == _ref_greedy(tensors, sizes)
    ts = [frozenset(t) for t in tensors]
    assert cp.pair_cost(ts[0], ts[1], sizes) == _ncon.pair_cost(
        ts[0], ts[1], sizes
    )
    assert cp.merge_tensors(ts, 0, 1) == _ncon.merge_tensors(ts, 0, 1)


# --- from eqsat tests/test_backend_neutral.py -------------------------


def test_torch_package_delegations():
    import mlopt_torch
    from mlopt_torch.autotune import TORCH_BUILDERS
    from mlopt_torch.backend import TorchBackend
    from mlopt_torch.runners import TorchCompileRunner

    assert mlopt_torch.TorchBackend is TorchBackend
    assert mlopt_torch.TorchCompileRunner is TorchCompileRunner
    assert mlopt_torch.TORCH_BUILDERS is TORCH_BUILDERS
    for gone in (
        "optimize_model",
        "optimize_compositional",
        "optimize_model_autotuned",
    ):
        with pytest.raises(AttributeError):
            getattr(mlopt_torch, gone)
    with pytest.raises(AttributeError):
        _ = mlopt_torch.bogus_name


def test_torch_sink_executors_degrade_without_carriers(monkeypatch):
    """Blocked carrier imports leave the trace+generic routing table."""
    import sys as _sys

    from mlopt_torch.adapters import TorchSink

    monkeypatch.setitem(
        _sys.modules, "mlopt_carriers.scan_lower", None
    )
    monkeypatch.setitem(_sys.modules, "mlopt_carriers.om_lower", None)
    monkeypatch.setitem(_sys.modules, "mlopt_carriers.omd_lower", None)
    monkeypatch.setitem(
        _sys.modules, "mlopt_carriers.om_streaming", None
    )
    sink = TorchSink()
    sink._executors = None  # rebuild under the blocked names
    table = sink.executors
    assert set(table) == {"trace", "generic"}


def test_build_egraph_accepts_explicit_source():
    import torch
    import torch.nn as nn
    from mlopt_torch.adapters import TorchSource
    from mlopt_torch.regime import build_egraph

    m = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    eg, _root, ir, _src, _stats = build_egraph(
        m, x, source=TorchSource(), rules=[], xc=False
    )
    assert eg is not None and ir is not None


def test_torch_builder_requires_ir():
    from mlopt_orchestrator.autotune import (
        AutotuneContext,
        CandidateUnavailableError,
    )
    from mlopt_torch.autotune import TORCH_BUILDERS

    ctx = AutotuneContext(
        model=None,
        example_input=None,
        ir=None,
        term=None,
        param_values={},
        sink=None,
        delivered=None,
        lowering="generic",
    )
    for name in ("torch_compile", "torch_compile_generic"):
        with pytest.raises(CandidateUnavailableError):
            TORCH_BUILDERS[name](ctx)


def test_torch_backend_value():
    from mlopt_torch.backend import TorchBackend

    b = TorchBackend()
    assert isinstance(b, Backend)
    from mlopt_torch.adapters import TorchSink, TorchSource

    assert isinstance(b.source, TorchSource)
    assert isinstance(b.sink, TorchSink)
    with pytest.raises(FrozenInstanceError):
        b.meter = None  # type: ignore[misc]


def test_torch_backend_optimizes_end_to_end():
    import torch
    import torch.nn as nn
    from mlopt_torch.backend import TorchBackend

    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.w(x))

    m = M()
    x = torch.randn(4, 8)
    lr = Optimizer(backend=TorchBackend()).optimize(
        m, x, max_iterations=2, verbose=False
    )
    with torch.no_grad():
        assert torch.allclose(lr.module(x), m(x), atol=1e-5)


# --- from eqsat tests/test_opmeta.py ----------------------------------


def test_view_sets_agree_across_modules():
    """``pipeline._VIEW_OPS`` and ``signature._VIEW_OPS`` are the same
    *source*, and the generator's alphabet is a subset of it — the
    measured inconsistency the registry removes."""
    from mlopt_discovery import pipeline as pl
    from mlopt_orchestrator.morphisms import signature as sig

    assert set(pl._VIEW_OPS) <= set(sig._VIEW_OPS)
    assert set(pl._VIEW_OPS) == set(om.GENERATOR_VIEW_OPS)
    assert set(sig._VIEW_OPS) == set(om.RELAYOUT_OPS)


def test_consumers_read_the_registry_object():
    """Each consumer set *is* the registry projection — identity, not
    a copy, so a twin cannot drift."""
    from mlopt_core import typing as ty
    from mlopt_core.cost import basic as cost_basic
    from mlopt_core.laws import layout
    from mlopt_discovery import oracle as orc
    from mlopt_discovery import pipeline as pl
    from mlopt_orchestrator.morphisms import signature as sig

    assert cost_basic._VIEW_OPS is om.COST_VIEW_OPS
    assert ty._COMMUTATIVE_BROADCAST is om.COMMUTATIVE_OPS
    assert sig._VIEW_OPS is om.RELAYOUT_OPS
    assert sig._POINTWISE_OPS is om.POINTWISE_OPS
    assert sig._ACT_OPS is om.ACTIVATION_OPS
    assert sig._MUT_OPS is om.WRITE_OPS
    assert sig._TABLE_OPS is om.TABLE_OPS
    assert orc._VIEWISH is om.VIEWISH_OPS
    assert orc._REDUCTION_DIM_OPS is om.REDUCE_DIM_OPS
    assert set(layout._POINTWISE_UNARY) == set(om.POINTWISE_UNARY_OPS)
    assert set(layout._POINTWISE_BINARY) == set(om.POINTWISE_BINARY_OPS)
    assert set(pl._POINTWISE) == set(om.POINTWISE_BINARY_OPS)


def test_fold_alphabet_is_single_sourced():
    """The lowerer's fold gate, the cost fold predicate and the
    extraction param-only discount all read ``FOLD_OPS`` — the one
    table — so a subtree the lowering folds is exactly the subtree
    the cost model prices at zero marginal.  The conv+BN weight-fold
    drift (``rsqrt`` folded at lowering but absent from the cost
    side's 13-op subset) is what this pins."""
    from mlopt_core.cost import params as cost_params
    from mlopt_core.egraph import extract as ex
    from mlopt_torch import torch_bridge as tb

    assert ex._FOLDABLE_OPS is om.FOLD_OPS
    assert tb.FOLD_OPS is om.FOLD_OPS
    # matmul/concat keep their tighter real-operand rule at the
    # predicate; the shallow-eval residue is the rest of the set.
    assert cost_params._FOLD_EVAL_OPS | {"matmul", "concat"} == (
        om.FOLD_OPS
    )
    # The declined slice/extraction family stays runtime.
    assert not (
        om.FOLD_OPS
        & {
            "select",
            "index_select",
            "slice",
            "narrow",
            "unbind",
            "getitem",
            "tensor_split",
            "expand",
            "broadcast_to",
            "repeat",
            "expand_as",
            "unfold",
            "stack",
            "index",
            "gather",
        }
    )


def test_aten_alias_map_mints_only_registered_ops():
    """Every IR op the exporter can mint has registry metadata.

    The alias map legitimately lives in the adapter (core is
    torch-free); this is the check that keeps it honest.
    """
    from mlopt_torch.torch_bridge import (
        _ATEN_TO_IR,
        _IR_TO_TORCH_EXTRA,
    )

    merged = {**_ATEN_TO_IR, **_IR_TO_TORCH_EXTRA}
    assert om.validate_aten_map(merged) == []
