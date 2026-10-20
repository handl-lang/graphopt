"""Tests for the ADR-0008 gate-3 state-reindex law family.

``mlopt_discovery.state_reindex`` is the named close of the gate-3
residual: the ``matmul`` contraction-axis bisection, the split-
naturality chain down to signed-offset ``decay_vec`` block-splits, and
the scalar slide/merge laws — minting the compensated arm's missing
steering wires so ONE ``Loop(step, n)`` covers the whole scan.

Controls: the law family's check/derive guards classify bindings
exactly; the driver's boundary measurement, early exits and verdict
paths are exercised on fabricated reports; the synthetic single-family
fold closes non-slowly; the real winner's two-family close is the
``slow`` certificate.
"""

import pytest
from mlopt_core.cost import count_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_discovery import form_propose as fp
from mlopt_discovery import state_reindex as sr
from mlopt_discovery.combinator import Loop, comb_to_dag

from tests.test_form_propose import CB, _fold_term, _lin

C = Const(8.0)


def _v(name, shape):
    return Var(name, TensorType(shape))


def _usq_dv(**a):
    return Op.make("unsqueeze", Op.make("decay_vec", CB, **a), dim=-1)


# ---------------------------------------------------------------------------
#  matmul_contract_split — the contraction-axis bisection
# ---------------------------------------------------------------------------


def test_matmul_contract_split_validates_sizes():
    """The partition must tile: >=2 positive ints."""
    with pytest.raises(ValueError):
        sr.matmul_contract_split((4,))
    with pytest.raises(ValueError):
        sr.matmul_contract_split((4, 0, 4))
    r = sr.matmul_contract_split((4, 4, 4, 4))
    assert r.name == "matmul_contract_split_4_4_4_4"


def test_check_contract_split():
    """Contraction extents must equal the partition total; batch dims
    must broadcast."""
    sizes = (4, 4)
    good = {
        "a": _v("a", (2, 8)),
        "b": _v("b", (8, 4)),
    }
    assert sr._check_contract_split(sizes, good)
    # rank < 2 or no shape → decline
    assert not sr._check_contract_split(
        sizes, {"a": _v("a", (8,)), "b": _v("b", (8, 4))}
    )
    assert not sr._check_contract_split(
        sizes, {"a": Const(1.0), "b": _v("b", (8, 4))}
    )
    # contraction extent mismatch
    assert not sr._check_contract_split(
        sizes, {"a": _v("a", (2, 6)), "b": _v("b", (8, 4))}
    )
    # batch dims that do not broadcast → decline
    assert not sr._check_contract_split(
        sizes, {"a": _v("a", (2, 2, 8)), "b": _v("b", (3, 8, 4))}
    )


# ---------------------------------------------------------------------------
#  split_mul_aligned / split_unsqueeze / decay_vec splits
# ---------------------------------------------------------------------------


def test_check_split_mul_aligned():
    """Both operands must cover the split axis at ``sum(SZ)``."""
    base = {
        "$attr:SZ": (4, 4),
        "$attr:D": -1,
        "$attr:I": 0,
        "a": _v("a", (2, 8)),
        "b": _v("b", (2, 8)),
    }
    assert sr._check_split_mul_aligned(base)
    # attr metavars missing / wrong shape → decline
    assert not sr._check_split_mul_aligned(
        {"$attr:SZ": 4, "$attr:D": -1, "$attr:I": 0}
    )
    assert not sr._check_split_mul_aligned(
        {
            "$attr:SZ": (4, 4),
            "$attr:D": -1,
            "$attr:I": 5,
        }
    )
    # operands without resolvable shapes → decline
    assert not sr._check_split_mul_aligned(
        {
            "$attr:SZ": (4, 4),
            "$attr:D": -1,
            "$attr:I": 0,
            "a": "x",
            "b": _v("b", (2, 8)),
        }
    )
    # an operand that does not cover the axis → decline
    assert not sr._check_split_mul_aligned(
        {
            "$attr:SZ": (4, 4),
            "$attr:D": -1,
            "$attr:I": 0,
            "a": _v("a", (2, 8)),
            "b": _v("b", (2, 3)),
        }
    )


def test_derive_split_unsq():
    """``DP`` is the split axis's pre-image once the inserted axis is
    skipped; mismatched extents decline."""
    good = {
        "$attr:SZ": (4, 4),
        "$attr:D": -1,
        "$attr:UD": -1,
        "$attr:I": 1,
        "a": _v("a", (2, 8)),
    }
    # D on the last axis, UD inserting a new last axis — but then
    # d_out == u_out → decline; use UD=-2 (a different position).
    good["$attr:UD"] = -2
    got = sr._derive_split_unsq(good)
    assert got == {"$attr:DP": -1}
    # invalid attr metavars → None
    assert sr._derive_split_unsq({"$attr:SZ": (4,)}) is None
    # no resolvable shape → None
    assert (
        sr._derive_split_unsq(
            {
                "$attr:SZ": (4, 4),
                "$attr:D": -1,
                "$attr:UD": -2,
                "$attr:I": 0,
                "a": "x",
            }
        )
        is None
    )
    # split axis == the inserted axis → None
    assert (
        sr._derive_split_unsq(
            {
                "$attr:SZ": (4, 4),
                "$attr:D": -1,
                "$attr:UD": -1,
                "$attr:I": 0,
                "a": _v("a", (2, 8)),
            }
        )
        is None
    )
    # the pre-image extent doesn't match the partition → None
    assert (
        sr._derive_split_unsq(
            {
                "$attr:SZ": (4, 4),
                "$attr:D": -1,
                "$attr:UD": -2,
                "$attr:I": 0,
                "a": _v("a", (2, 6)),
            }
        )
        is None
    )


def test_dv_split_attrs_and_derives():
    """``t`` must equal ``sum(SZ)`` and the split axis must be the last
    (vector) axis; the derives mint signed offsets."""
    good = {
        "$attr:SZ": (4, 4, 4, 4),
        "$attr:T": 16,
        "$attr:I": 3,
        "$attr:D": -1,
        "$attr:O": 0,
        "g": CB,
    }
    assert sr._check_dv_split(good)
    d = sr._derive_dv_split(-1)(good)
    assert d == {"$attr:TI": 4, "$attr:OD": -12}
    d = sr._derive_dv_split(1)(good)
    assert d == {"$attr:TI": 4, "$attr:OD": 12}
    # t != sum(SZ) → decline both check and derive
    bad_t = dict(good, **{"$attr:T": 8})
    assert not sr._check_dv_split(bad_t)
    assert sr._derive_dv_split(-1)(bad_t) is None
    # attr metavars missing → decline
    assert not sr._check_dv_split({"$attr:SZ": (4,)})
    assert sr._derive_dv_split(-1)({"$attr:SZ": (4,)}) is None
    # split axis off the vector axis → decline
    off_axis = dict(good, **{"$attr:D": 0, "g": _v("g", (16,))})
    assert not sr._check_dv_split(off_axis)


def test_scalar_law_guards():
    """The offset-sum derive and the small check predicates."""
    assert sr._derive_od_sum({"$attr:O1": 2, "$attr:O2": 3}) == {
        "$attr:OD": 5
    }
    assert sr._derive_od_sum({"$attr:O1": 2, "$attr:O2": "x"}) is None
    assert not sr._check_od_sum({"$attr:O1": 2})
    assert sr._check_offset_zero({"$attr:O": 0})
    assert not sr._check_offset_zero({"$attr:O": 3})
    assert sr._is_decay_scalar({})
    assert sr._check_eta_slide({"$attr:O1": 1, "$attr:O2": -2})
    assert sr._check_usq_factor({"$attr:O": -8})
    assert not sr._check_usq_factor({"$attr:O": 0})
    assert not sr._check_usq_factor({"$attr:O": "x"})


def test_state_reindex_laws_and_budgets():
    """The family is the bisection plus the naturality/slide chain."""
    rules = sr.state_reindex_laws((4, 4, 4, 4))
    names = {r.name for r in rules}
    assert "matmul_contract_split_4_4_4_4" in names
    assert "split_mul_aligned" in names
    assert "state_eta_slide" in names
    budgets = sr.reindex_budgets((4, 4, 4, 4))
    assert budgets["matmul_contract_split_4_4_4_4"] == 8000
    assert budgets["distribute_mul_add"] == 60000


# ---------------------------------------------------------------------------
#  _boundary_sizes — the driver's measurement
# ---------------------------------------------------------------------------


def _rep_with_boundary(**kw):
    rep = fp.FormReport()
    rep.boundary = kw
    return rep


def test_boundary_sizes_underivable():
    """No steering hint and no shaped target → ``None``."""
    rep = _rep_with_boundary()
    assert sr._boundary_sizes(rep) is None
    # a non-matmul target term gives no extent either
    rep = _rep_with_boundary(target_term=Const(1.0))
    assert sr._boundary_sizes(rep) is None
    # a matmul whose right operand isn't a split → no extent
    rep = _rep_with_boundary(
        target_term=Op.make("matmul", _v("a", (4, 4)), _v("b", (4, 4)))
    )
    assert sr._boundary_sizes(rep) is None
    # a split with a malformed sizes/index → no extent
    bad = Op.make(
        "split", _v("b", (16, 4)), sizes=(4, 4), index=7, dim=-2
    )
    rep = _rep_with_boundary(
        target_term=Op.make("matmul", _v("a", (4, 16)), bad)
    )
    assert sr._boundary_sizes(rep) is None


def test_boundary_sizes_from_steering_hint_and_span():
    """``w`` comes from the missing steering eta; the extent from the
    target's contraction split — else the producing block's span."""
    ce = Op.make(
        "unsqueeze",
        Op.make("decay_vec", CB, t=4, sign=-1, offset=-12),
        dim=-1,
    )
    target = Op.make(
        "matmul",
        _v("a", (4, 16)),
        Op.make(
            "split", _v("b", (16, 4)), sizes=(16,), index=0, dim=-2
        ),
    )
    rep = _rep_with_boundary(
        missing_steer={3: {"ce": ce}}, target_term=target
    )
    assert sr._boundary_sizes(rep) == (4, 4, 4, 4)
    # no steering hint → ``w`` falls back to the producing block's span
    rep = _rep_with_boundary(at=1, target_term=target)
    rep.bindings = {
        1: {
            fp.XQ: Op.make(
                "split",
                _lin("p_q_weight"),
                dim=-2,
                index=0,
                sizes=(4, 4),
            )
        }
    }
    assert sr._boundary_sizes(rep) == (4, 4, 4, 4)
    # neither hint → underivable
    rep = _rep_with_boundary(at=1)
    rep.bindings = {1: {}}
    assert sr._boundary_sizes(rep) is None


# ---------------------------------------------------------------------------
#  reindex_close — driver early exits and the verdicts gate
# ---------------------------------------------------------------------------


def test_reindex_close_early_exits():
    """No covered blocks, no step morphism, underivable sizes."""
    eg = EGraph()
    rep = fp.FormReport()
    sr.reindex_close(eg, rep, Const(0.0), count_cost)
    assert rep.reindex["obstruction"] == "no covered blocks"

    rep = fp.FormReport()
    rep.chains = [fp.ChainResult(blocks=[1, 2], ok=True)]
    sr.reindex_close(eg, rep, Const(0.0), count_cost)
    assert rep.reindex["obstruction"] == "no step morphism"

    rep.step_morph = object()
    rep.step = fp.step_template()
    sr.reindex_close(eg, rep, Const(0.0), count_cost)
    assert rep.reindex["obstruction"] == "boundary sizes underivable"


def test_reindex_close_verdicts_gate():
    """When the compensated arm cannot verify, nothing is emitted —
    and ``closed`` stays ``False`` (explicit sizes bypass derive)."""
    term = _fold_term(k=3)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    # a report whose covered blocks verify but whose per-iteration
    # bindings are empty — the steered step cannot land in the class.
    rep2 = fp.FormReport()
    rep2.chains = [fp.ChainResult(blocks=[1, 2], ok=True)]
    rep2.step = rep.step
    rep2.step_morph = rep.step_morph
    rep2.bindings = {1: {}, 2: {}}
    rep2.n_blocks = rep.n_blocks
    sr.reindex_close(
        eg, rep2, term, count_cost, root_eid=root_eid, sizes=(4, 4)
    )
    assert rep2.reindex["closed"] is False
    assert rep2.reindex["sizes"] == (4, 4)
    assert rep2.single_prog is None


def test_reindex_close_synthetic_fold():
    """The single-family synthetic fold closes with one Loop — the
    non-slow certificate that mint+verify+emit all compose."""
    term = _fold_term(k=3)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    sr.reindex_close(
        eg, rep, term, count_cost, root_eid=root_eid, sizes=(4, 4)
    )
    assert rep.reindex["closed"] is True
    assert rep.reindex["verdicts"] == {1: True, 2: True}
    assert isinstance(rep.single_loop, Loop)
    assert rep.single_loop.n == 2
    assert rep.single_whole_equal
    assert eg.find(
        eg.add_term(comb_to_dag(rep.single_prog))
    ) == eg.find(root_eid)


# ---------------------------------------------------------------------------
#  The gate-3 certificate on the real winner (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_state_reindex_closes_boundary():
    """The named close: ONE Loop(step,6) in the winner's e-class.

    The state-reindex law family (``mlopt_discovery.state_reindex``)
    supplies the three missing pieces the residual named: the bare
    ``matmul`` contraction bisection ``(4,4,4,4)``, signed-offset
    ``decay_vec`` block-splitting (mints ``dv(t=4,-1,-4i)`` and the
    scalar steering weights), and the scalar slide/merge laws
    (``gamma^a·(gamma^b·X) = gamma^{a+b}·X``, ``gamma^0·X = X``,
    ``gamma^a ⊙ ((k⊙eta_o)ᵀv) = (k⊙eta_{o+a})ᵀv``).  After the mint
    pass the compensated arm's steered state lands in M_4's e-class,
    all six covered blocks verify, and the emitted single-loop program
    lowers into the winner's root class.
    """
    import mlopt_carriers  # noqa: F401
    from mlopt_carriers.om import (
        om_shared_state_budgets,
        om_shared_state_stages,
    )
    from mlopt_core.cost import flops_cost

    from bench.probes import om_shared_state_probe as osp

    t, k = 64, 4
    reach, reassoc = om_shared_state_stages(k)
    budgets = om_shared_state_budgets(k)
    eg = EGraph()
    root = eg.add_term(osp._export(t))
    osp._saturate(eg, root, reach, 10, budgets)
    osp._saturate(eg, root, reassoc, 10, budgets)
    root_eid = eg.find(root)
    best = eg.extract_best(root_eid, flops_cost)
    rep = fp.form_propose(eg, best, flops_cost, root_eid=root_eid)

    sr.reindex_close(eg, rep, best, flops_cost, root_eid=root_eid)

    assert rep.reindex["closed"] is True
    assert rep.reindex["sizes"] == (4, 4, 4, 4)
    assert rep.single_loop_compensated == {i: True for i in range(1, 7)}
    assert isinstance(rep.single_loop, Loop)
    assert rep.single_loop.n == 6
    assert rep.single_whole_equal
    # the e-class certificate: the single-loop program's DAG is the
    # winner's own class member.
    assert eg.find(
        eg.add_term(comb_to_dag(rep.single_prog))
    ) == eg.find(root_eid)
    # Torch-free spot check is not needed here — the e-class equality
    # is the certificate — but the program must still eval-lower:
    assert isinstance(rep.single_prog_stats.nodes, int)
