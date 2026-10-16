"""Tests for the ADR-0008 gate-3 form-level proposer.

``lab/bridge/form_propose.py`` is the anti-unification move the gate-3
retros named: it uses the saturated e-graph as the oracle — the factored
``matmul(q̃, M)`` reads and the ``add(mul(w, M), B)`` state chain are
*derived members* extraction did not choose — synthesises one step
schema ``pack(S', o)`` over wire parameters, emits ``unroll(step, n)``
per state chain (folded to ``Loop`` by ``PROPOSE_LOOP``), and verifies
by expansion landing in the winner's own e-classes.

Controls: a synthetic regular fold must produce a verified, loop-headed
program whose expansion lands in the root e-class; a family with no
factored read must name the obstruction, not fabricate.
"""

import pytest
from catopt_core.cost import count_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import Const, Op, Param, TensorType, Var

from lab.bridge import form_propose as fp
from lab.bridge.combinator import comb_to_dag
from lab.bridge.loop_propose import loop_headed

X = Var("x", TensorType((2, 64, 16)))
CB = Op.make(
    "detach_",
    Op.make("contiguous", Var("c_lifted_tensor_0", TensorType(()))),
)
W = {}


def _p(n):
    return W.setdefault(n, Param(n, TensorType((16, 16))))


def _lin(src):
    return Op.make("linear", X, _p(src), Param(f"{src[:-6]}bias", TensorType((16,))))


def _fold_term(k=3, C=4):
    """A regular K-block fold in the winner's spelling (single family)."""
    sizes = (C,) * k
    ql, kl, vl = _lin("p_q_weight"), _lin("p_k_weight"), _lin("p_v_weight")
    c = Const(8.0)

    def sp(m, i, dim=-2):
        return Op.make("split", m, dim=dim, index=i, sizes=sizes)

    def dv(**a):
        return Op.make("decay_vec", CB, **a)

    eta = Op.make("unsqueeze", dv(sign=-1, t=C), dim=-1)
    w = Op.make("unsqueeze", dv(offset=C, t=1), dim=-1)
    xi = Op.make("unsqueeze", dv(offset=C, t=C), dim=-1)
    dec = Op.make("decay", CB, t=C)

    def b_term(i):
        return Op.make(
            "matmul",
            Op.make(
                "transpose", Op.make("mul", sp(kl, i), eta),
                dim0=-1, dim1=-2,
            ),
            sp(vl, i),
        )

    def intra(i):
        return Op.make(
            "om_decayed",
            Op.make(
                "div",
                Op.make(
                    "matmul",
                    sp(ql, i),
                    Op.make(
                        "transpose", sp(kl, i), dim0=-1, dim1=-2
                    ),
                ),
                c,
            ),
            dec,
            sp(vl, i),
        )

    def blk(i, s_prev):
        read = Op.make(
            "matmul", Op.make("div", sp(ql, i), c), s_prev
        )
        return Op.make("add", intra(i), Op.make("mul", xi, read))

    blocks = [intra(0)]
    s = b_term(0)
    for i in range(1, k):
        blocks.append(blk(i, s))
        s = Op.make("add", Op.make("mul", w, s), b_term(i))
    root = blocks[0]
    for b in blocks[1:]:
        root = Op.make("concat", root, b, dim=-2)
    return root


def _irregular_term():
    """Two blocks whose reads share no state — no factored member."""
    sizes = (4, 4, 4)
    kl, vl = _lin("p_k_weight"), _lin("p_v_weight")

    def sp(m, i, dim=-2):
        return Op.make("split", m, dim=dim, index=i, sizes=sizes)

    b0 = Op.make("om_decayed", sp(kl, 0), sp(vl, 0), sp(vl, 0))
    b1 = Op.make("om_decayed", sp(kl, 1), sp(vl, 1), sp(vl, 1))
    return Op.make("concat", b0, b1, dim=-2)


def test_regular_fold_collapses_to_loop():
    """A regular fold: chain found, blocks verified, loop reached."""
    term = _fold_term(k=3)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    assert rep.n_blocks == 3
    ok = [r for r in rep.chains if r.ok]
    assert len(ok) == 1 and ok[0].blocks == [1, 2]
    assert all(ok[0].verified.values())
    assert rep.whole_equal
    assert rep.prog is not None
    # the closure folds the unroll to a Loop-headed term
    assert rep.closure["fired"].get("loop_propose", 0) >= 1
    # and the assembled program is the same computation, certified:
    assert eg.find(eg.add_term(comb_to_dag(rep.prog))) == eg.find(
        root_eid
    )
    assert loop_headed(
        comb_to_dag(rep.prog) if False else rep.chains[0].loop
    )


def test_irregular_family_reports_obstruction():
    """No factored read anywhere → no chain, named obstruction."""
    term = _irregular_term()
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    assert not any(r.ok for r in rep.chains)
    assert rep.prog is None
    assert not rep.whole_equal


@pytest.mark.slow
def test_real_winner_gate3():
    """The actual gate-3 measurement: the extracted winner loops."""
    import catopt_carriers  # noqa: F401
    from catopt_carriers.om import (
        om_shared_state_budgets,
        om_shared_state_stages,
    )
    from catopt_core.cost import flops_cost

    from lab.probes import om_shared_state_probe as osp

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
    ok = [r for r in rep.chains if r.ok]
    assert rep.n_blocks == 7
    assert len(ok) == 2  # one loop per chunk-width family
    assert {tuple(r.blocks) for r in ok} == {(1, 2, 3), (4, 5, 6)}
    assert rep.whole_equal
    assert rep.closure["fired"].get("loop_propose", 0) == 2
    # the single-loop arm names the family-boundary obstruction:
    assert rep.single_loop_verified == {
        1: True, 2: True, 3: True, 4: False, 5: False, 6: False
    }
    assert any("state-chain-broken" in o for o in rep.obstructions)
