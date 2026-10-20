"""The shipped minted-composite class — ``conv_bn_fold``, verbatim.

``mlopt_core.laws.minted`` embeds the stored-closure records the
conv+BN eval-mode fold minted as (the ``syn_*`` chain of
``bench/drivers/probes/conv_fold_premise_pay_probe.py``) and rebuilds the
transported guard at import.  These tests pin that the *shipped*
object is the measured one — same guarded region, same fired instance,
same extraction delta — and that the materializer refuses to degrade
into a weaker admit.
"""

from __future__ import annotations

import pytest
from mlopt_core import meta
from mlopt_core.cost import count_cost, dag_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Param, TensorType, Var, op_repr
from mlopt_core.laws import (
    ALL_RULES,
    BATCH_NORM_RULES,
    CONV_BN_FOLD,
    CONV_FOLD_RULES,
    DEFAULT,
    FUSION,
    MINTED_RULES,
    law_provenance,
    minted,
)
from mlopt_core.laws.serialize import alpha_key, missing_hooks


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


# ---------------------------------------------------------------------------
#  membership + provenance
# ---------------------------------------------------------------------------


def test_conv_bn_fold_ships_in_default_and_classifies_machine():
    assert CONV_BN_FOLD in ALL_RULES
    assert MINTED_RULES == [CONV_BN_FOLD]
    assert CONV_BN_FOLD in DEFAULT.rules
    assert CONV_BN_FOLD in FUSION.rules
    assert law_provenance(CONV_BN_FOLD) == "machine"
    # ``axiom`` by annotation — the shipped-names ``derivation`` field
    # cannot name the unshipped ``syn_*`` intermediates the true
    # derivation runs through; the honest lineage is ``parents``:
    assert CONV_BN_FOLD.derivation == ()
    assert CONV_BN_FOLD.parents == (
        "syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6"
        "__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209",
        "conv_channel_shift_sp",
    )
    assert CONV_BN_FOLD.guard_pats is not None
    # The transported class: procedural hooks flagged honestly, the
    # guard *spec* carried as data (guard_pats + parents).
    assert missing_hooks(CONV_BN_FOLD) == ("check", "derive")


def test_shipped_pattern_is_the_embedded_record():
    """The shipped 2-cell is byte-identical to the stored mint."""
    root = minted._RECORDS[
        "syn_syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6"
        "__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209"
        "__conv_channel_shift_sp_896"
    ]
    from mlopt_core.laws.serialize import object_from_data

    bare = object_from_data(root)
    assert alpha_key(CONV_BN_FOLD.lhs, CONV_BN_FOLD.rhs) == alpha_key(
        bare.lhs, bare.rhs
    )


# ---------------------------------------------------------------------------
#  firing — the measured region, torch-free
# ---------------------------------------------------------------------------


def test_conv_bn_fold_fires_on_spelled_eval_site():
    out = meta.apply_rewrite_at(CONV_BN_FOLD, _spelled(), ())
    assert out is not None
    assert out.op == "conv2d"
    rep = op_repr(out)
    assert "batch_norm" not in rep
    # the folded scale chain materialized through the derive
    assert "rsqrt" in rep
    assert "shape=(6, 1, 1, 1)" in rep  # WV — the weight-side view
    assert "shape=(1, 6, 1, 1)" in rep  # CH — the channel view


def test_conv_bn_fold_declines_off_region():
    # a channel count disagreeing with the weight's cout — the
    # transported guard's derive chain declines
    assert (
        meta.apply_rewrite_at(CONV_BN_FOLD, _spelled(bn_c=3), ())
        is None
    )
    # a non-default eps spelling is outside the minted region (the
    # LHS carries the seed's literal attrs)
    off = Op.make(
        "batch_norm",
        Op.make(
            "conv2d",
            Var("x", _T(1, 4, 8, 8)),
            Param("W", _T(6, 4, 3, 3)),
            Param("b", _T(6)),
            stride=(1, 1),
            padding=(1, 1),
        ),
        Param("g", _T(6)),
        Param("be", _T(6)),
        Param("mu", _T(6)),
        Param("var", _T(6)),
        training=False,
        momentum=0.1,
        eps=1e-03,
        cudnn_enabled=False,
    )
    assert meta.apply_rewrite_at(CONV_BN_FOLD, off, ()) is None


def test_conv_bn_fold_extraction_pays():
    term = _spelled()
    eg = EGraph()
    root = eg.add_term(term)
    eg.run([CONV_BN_FOLD], root, max_iterations=4, max_nodes=20_000)
    best = eg.extract_best(root, count_cost)
    assert best is not None and best != term
    assert dag_cost(best, count_cost) < dag_cost(term, count_cost)


# ---------------------------------------------------------------------------
#  the materializer contract — loud, never a silent weaker admit
# ---------------------------------------------------------------------------


def test_materialize_refuses_unresolvable_parent():
    # a syn record whose leaf parents resolve to nothing must raise —
    # shipping a guard-less rebuild would be the weakened admit
    with pytest.raises(ValueError, match="parent"):
        minted.materialize(
            "syn_batch_norm_expand__conv_channel_shift_sub_sp_6", {}
        )
    with pytest.raises(ValueError, match="parent"):
        minted.materialize("no_such_mint", {})


def test_materialize_refuses_partial_lib():
    # only one of the two leaf parents resolvable — still a raise
    lib = {r.name: r for r in BATCH_NORM_RULES}
    with pytest.raises(ValueError, match="parent"):
        minted.materialize(
            "syn_batch_norm_expand__conv_channel_shift_sub_sp_6", lib
        )


def test_materialize_intermediate_rebuilds_with_guard():
    lib = {
        r.name: r for r in [*BATCH_NORM_RULES, *CONV_FOLD_RULES]
    }
    inner = minted.materialize(
        "syn_batch_norm_expand__conv_channel_shift_sub_sp_6", lib
    )
    assert inner.name == (
        "syn_batch_norm_expand__conv_channel_shift_sub_sp_6"
    )
    assert inner.check is not None
    assert inner.parents == ("batch_norm_expand", "conv_channel_shift_sub_sp")
    assert meta.provenance(inner) == inner.parents


# ---------------------------------------------------------------------------
#  store round-trip — the shipped object re-encodes its guard spec
# ---------------------------------------------------------------------------


def test_shipped_object_restores_through_the_store():
    """store → admit rebuilds the transported hooks; fires identically."""
    from mlopt_core.laws.serialize import alpha_key
    from mlopt_discovery import evidence as ev

    lib = {
        r.name: r for r in [*BATCH_NORM_RULES, *CONV_FOLD_RULES]
    }
    conn = ev.connect(":memory:")
    try:
        # the full premise closure as store rows — leaf parents plus
        # each materialized ``syn_*`` intermediate
        for nm in ("batch_norm_expand", "conv_channel_shift_sub_sp",
                   "conv_channel_scale_sp", "conv_channel_scale_l_sp",
                   "conv_channel_shift_sp"):
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
