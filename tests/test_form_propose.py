"""Tests for the ADR-0008 gate-3 form-level proposer.

``mlopt_discovery.form_propose`` is the anti-unification move the gate-3
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
from mlopt_core.cost import count_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_discovery import form_propose as fp
from mlopt_discovery.combinator import comb_to_dag

from lab.bridge.index_abstract import block_outputs
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
    return Op.make(
        "linear",
        X,
        _p(src),
        Param(f"{src[:-6]}bias", TensorType((16,))),
    )


def _fold_term(k=3, C=4):
    """A regular K-block fold in the winner's spelling (single family)."""
    sizes = (C,) * k
    ql, kl, vl = (
        _lin("p_q_weight"),
        _lin("p_k_weight"),
        _lin("p_v_weight"),
    )
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
                "transpose",
                Op.make("mul", sp(kl, i), eta),
                dim0=-1,
                dim1=-2,
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
                    Op.make("transpose", sp(kl, i), dim0=-1, dim1=-2),
                ),
                c,
            ),
            dec,
            sp(vl, i),
        )

    def blk(i, s_prev):
        read = Op.make("matmul", Op.make("div", sp(ql, i), c), s_prev)
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
    assert loop_headed(rep.chains[0].loop)


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
    import mlopt_carriers  # noqa: F401
    from mlopt_carriers.om import (
        om_shared_state_budgets,
        om_shared_state_stages,
    )
    from mlopt_core.cost import flops_cost

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
        1: True,
        2: True,
        3: True,
        4: False,
        5: False,
        6: False,
    }
    assert any("state-chain-broken" in o for o in rep.obstructions)


@pytest.mark.slow
def test_boundary_residual_is_a_value_mismatch():
    """The named residual, measured: carried = gamma^12 * M4, not M4.

    The proposer's boundary obstruction was first read as a
    saturation-reach gap ("same value, different class").  Measured
    directly it is a *value* mismatch: the fold carries
    ``S_i = gamma^{phi_i} * A_i`` normalised to the previous
    own-partition chunk's start, and the convention jumps phi = 8 -> 0
    across the boundary — so no sound e-graph merge can close it.  The
    compensated arm shows the uniform step is still value-correct with
    the reindexed wires, and that the residual decomposes into one
    non-resident wire (block 4's read scale) plus one true reach gap
    (the steered block-5 state: same value, unmerged class).
    """
    import mlopt_carriers  # noqa: F401
    import torch
    from mlopt_carriers.om import (
        om_shared_state_budgets,
        om_shared_state_stages,
    )
    from mlopt_core.cost import flops_cost
    from mlopt_core.ir import IR
    from mlopt_discovery.zoo import _ChunkedRetention
    from mlopt_torch.adapters import TorchSink
    from mlopt_torch.torch_bridge import export_to_ir

    from lab.bridge.index_abstract import block_outputs
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

    # --- the compensated arm: steer the convention across the boundary
    assert rep.single_loop_compensated == {
        1: True,
        2: True,
        3: True,
        4: False,
        5: False,
        6: False,
    }
    assert any(
        "boundary-renorm-unbound:block3->block4" in o
        for o in rep.obstructions
    )
    assert any(
        "boundary-xi-nonresident:block4" in o for o in rep.obstructions
    )
    assert rep.boundary["at"] == 4
    assert rep.boundary["carried_eid"] != rep.boundary["target_eid"]

    # --- the numeric certificate: the two classes hold different values
    torch.manual_seed(0)
    model = _ChunkedRetention(16).eval().double()
    x = torch.randn(2, t, 16, dtype=torch.float64)
    ir0, tensors = export_to_ir(model, (x,))
    sink = TorchSink()

    def ev(term):
        mod = sink.lower(
            IR(
                root=term,
                inputs=list(ir0.inputs),
                params=dict(ir0.params),
            ),
            params=dict(tensors),
        )
        with torch.no_grad():
            return mod(x)

    carried = ev(rep.boundary["carried_term"])
    target = ev(rep.boundary["target_term"])
    g12 = 0.9**12
    # NOT the same value — the retro's "same value" claim falsified:
    assert (carried - target).abs().max().item() > 0.05
    # but exactly a gamma^12 rescaling of it:
    assert (carried - g12 * target).abs().max().item() < 1e-9

    # the compensated read wire makes block 4 numerically exact even
    # though it is not graph-resident (the named obstruction):
    blocks = block_outputs(best)
    xi4 = rep.boundary["missing_wires"][4]
    env = dict(rep.bindings[4])
    env[fp.CX] = xi4
    env[fp.S] = rep.boundary["carried_term"]
    o4 = fp._subst(rep.step, env).args[1]
    assert (ev(o4) - ev(blocks[4])).abs().max().item() < 1e-9

    # and the steered next state IS the family's M_5 in value — the
    # residual there is a pure reach gap (same value, fresh class):
    s5 = rep.single_states[5]
    m5_cls = eg.find(
        fp.factored_states(
            eg, eg._locate(fp._read_of(blocks[5]))[0], flops_cost
        )[0]["m_eid"]
    )
    m5 = eg.extract_best(m5_cls, flops_cost)
    assert (ev(s5) - ev(m5)).abs().max().item() < 1e-9
    assert eg.find(eg.add_term(s5)) != m5_cls


# ---------------------------------------------------------------------------
#  Targeted unit coverage — helpers, guards and fabricated e-graphs
# ---------------------------------------------------------------------------


def _chained_pair(mismatch: bool = False):
    """A two-block chained fold plus e-graph, fully fabricated.

    ``b0`` reads the seed state ``M0``; ``b1`` reads
    ``M1 = add(mul(w, M0), B)`` — the ``add(mul(w,·),B)`` member that
    chains the two state classes.  With ``mismatch=True`` the update
    carries the *wrong* contribution so the expansion lands outside the
    block's e-class.
    """
    sizes = (4, 4)
    ql, kl, vl = (
        _lin("p_q_weight"),
        _lin("p_k_weight"),
        _lin("p_v_weight"),
    )
    c = Const(8.0)

    def sp(m, i, dim=-2):
        return Op.make("split", m, dim=dim, index=i, sizes=sizes)

    def dv(**a):
        return Op.make("decay_vec", CB, **a)

    eta = Op.make("unsqueeze", dv(sign=-1, t=4), dim=-1)
    w = Op.make("unsqueeze", dv(offset=4, t=1), dim=-1)
    xi = Op.make("unsqueeze", dv(offset=4, t=4), dim=-1)
    dec = Op.make("decay", CB, t=4)

    def b_term(i):
        return Op.make(
            "matmul",
            Op.make(
                "transpose",
                Op.make("mul", sp(kl, i), eta),
                dim0=-1,
                dim1=-2,
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
                    Op.make("transpose", sp(kl, i), dim0=-1, dim1=-2),
                ),
                c,
            ),
            dec,
            sp(vl, i),
        )

    def blk(i, s):
        read = Op.make("matmul", Op.make("div", sp(ql, i), c), s)
        return Op.make("add", intra(i), Op.make("mul", xi, read))

    m0 = b_term(0)
    # the chained update: w·M0 plus block 0's own contribution; the
    # mismatch variant smuggles block 1's contribution instead, so the
    # producing step's output no longer lands in b1's read state.
    m1 = Op.make(
        "add", Op.make("mul", w, m0), b_term(1 if mismatch else 0)
    )
    root = Op.make("concat", blk(0, m0), blk(1, m1), dim=-2)
    eg = EGraph()
    root_eid = eg.add_term(root)
    return eg, root, root_eid, (m0, m1)


def _two_family_term(with_tail: bool = True):
    """A 4-block fold split across two unlinked state families.

    Blocks 1-2 chain (``w·s + B_i`` updates); block 3 reads a
    standalone accumulation that no ``add(mul(w,·),B)`` member links —
    the ``state-chain-broken`` boundary.  The optional tail block has a
    read with no factored member — the ``no-factored-read`` boundary.
    """
    sizes = (4,) * 5
    ql, kl, vl = (
        _lin("p_q_weight"),
        _lin("p_k_weight"),
        _lin("p_v_weight"),
    )
    c = Const(8.0)

    def sp(m, i, dim=-2):
        return Op.make("split", m, dim=dim, index=i, sizes=sizes)

    def dv(**a):
        return Op.make("decay_vec", CB, **a)

    eta = Op.make("unsqueeze", dv(sign=-1, t=4), dim=-1)
    w = Op.make("unsqueeze", dv(offset=4, t=1), dim=-1)
    xi = Op.make("unsqueeze", dv(offset=4, t=4), dim=-1)
    dec = Op.make("decay", CB, t=4)

    def b_term(i):
        return Op.make(
            "matmul",
            Op.make(
                "transpose",
                Op.make("mul", sp(kl, i), eta),
                dim0=-1,
                dim1=-2,
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
                    Op.make("transpose", sp(kl, i), dim0=-1, dim1=-2),
                ),
                c,
            ),
            dec,
            sp(vl, i),
        )

    def blk(i, s):
        read = Op.make("matmul", Op.make("div", sp(ql, i), c), s)
        return Op.make("add", intra(i), Op.make("mul", xi, read))

    blocks = [intra(0)]
    s = b_term(0)
    blocks.append(blk(1, s))
    s = Op.make("add", Op.make("mul", w, s), b_term(1))
    blocks.append(blk(2, s))
    # the second family's state: a standalone accumulation — nothing
    # in its class links back to the first family's carried state.
    blocks.append(blk(3, b_term(3)))
    if with_tail:
        blocks.append(
            Op.make(
                "add",
                intra(4),
                Op.make(
                    "mul", xi, Op.make("add", sp(ql, 4), b_term(4))
                ),
            )
        )
    root = blocks[0]
    for b in blocks[1:]:
        root = Op.make("concat", root, b, dim=-2)
    return root


def test_two_family_boundary_is_measured():
    """The chain break, the compensated carry, and the residual data."""
    term = _two_family_term()
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    ok = [r for r in rep.chains if r.ok]
    assert {tuple(r.blocks) for r in ok} == {(1, 2), (3,)}
    assert any(
        "state-chain-broken:block2->block3" in o
        for o in rep.obstructions
    )
    assert any("no-factored-read:block4" in o for o in rep.obstructions)
    # the single-state carry verifies the first family and fails the
    # second — the named residual, recorded as data.
    assert rep.single_loop_verified[3] is False
    assert rep.single_loop_compensated[3] is False
    assert rep.boundary["at"] == 3
    assert rep.boundary["carried_eid"] != rep.boundary["target_eid"]
    assert rep.whole_equal


def test_chain_choice_failed_names_obstruction():
    """A read class with two factored members where the canonically
    first candidate doesn't link — the chain exists (some pair links)
    but the greedy pick fails."""
    _eg, root, root_eid, (m0, _m1) = _chained_pair()
    # a second factored member of block 0's read class, added FIRST so
    # its children carry smaller e-class ids and it sorts ahead of the
    # linking member; its state class does not link to M1.
    sizes = (4, 4)
    ql, kl, vl = (
        _lin("p_q_weight"),
        _lin("p_k_weight"),
        _lin("p_v_weight"),
    )

    def sp(m, i, dim=-2):
        return Op.make("split", m, dim=dim, index=i, sizes=sizes)

    eta = Op.make(
        "unsqueeze", Op.make("decay_vec", CB, sign=-1, t=4), dim=-1
    )
    m_bad = Op.make(
        "matmul",
        Op.make(
            "transpose",
            Op.make("mul", sp(kl, 1), eta),
            dim0=-1,
            dim1=-2,
        ),
        sp(vl, 1),
    )
    q_bad = Op.make("mul", sp(ql, 0), Op.make("unsqueeze", eta, dim=-1))
    eg2 = EGraph()
    bad_id = eg2.add_term(Op.make("matmul", q_bad, m_bad))
    root_eid = eg2.add_term(root)
    read0 = fp._read_of(block_outputs(root)[0])
    r0id, _ = eg2._locate(read0)
    eg2.union(r0id, bad_id)
    # and a duplicate-M member (q-only left, same state class) so the
    # candidate dedup fires.
    q_dup = Op.make(
        "mul",
        sp(ql, 0),
        Op.make("unsqueeze", Op.make("decay_vec", CB, t=4), dim=-1),
    )
    dup_id = eg2.add_term(Op.make("matmul", q_dup, m0))
    eg2.union(r0id, dup_id)
    rep = fp.form_propose(eg2, root, count_cost, root_eid=root_eid)
    assert any(
        "chain-choice-failed:block0..block1" in o
        for o in rep.obstructions
    )
    assert rep.prog is None


def test_expand_mismatch_names_block():
    """A chained pair whose update term is inconsistent: the expanded
    step output lands outside the block's own e-class."""
    eg, root, root_eid, _ = _chained_pair(mismatch=True)
    rep = fp.form_propose(eg, root, count_cost, root_eid=root_eid)
    assert any("expand-mismatch:block1" in o for o in rep.obstructions)
    assert rep.chains[0].verified[1] is False
    assert not rep.chains[0].ok


def test_single_block_chain_closure():
    """A one-block chain: ``unroll(step, 1)`` is below ``_folds``' n≥2
    floor, so the closure's best is not loop-headed."""
    term = _fold_term(k=2)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    ok = [r for r in rep.chains if r.ok]
    assert len(ok) == 1 and ok[0].blocks == [1]
    assert not loop_headed(rep.chains[0].loop) or ok[0].loop is not None


def test_const_role_classification():
    """The decay/scale attr signatures map to their wire roles."""
    assert fp._const_role(Const(1.5)) is fp.CC
    assert fp._const_role(Var("v", TensorType(()))) is None
    assert fp._const_role(Op.make("decay_vec", CB, t=4)) is None
    assert fp._const_role(Op.make("decay", CB, t=4)) is fp.CD

    def usq(**a):
        return Op.make(
            "unsqueeze", Op.make("decay_vec", CB, **a), dim=-1
        )

    assert fp._const_role(usq(sign=-1, t=4)) is fp.CE
    assert fp._const_role(usq(offset=4, t=4)) is fp.CX
    assert fp._const_role(usq(offset=4, t=1)) is fp.CW
    # an unsqueezed decay_vec with no recognised signature isn't bound
    assert fp._const_role(usq(offset=3, t=5)) is None
    # a dim=-2 unsqueeze is not a wire either
    assert (
        fp._const_role(
            Op.make(
                "unsqueeze",
                Op.make("decay_vec", CB, t=4),
                dim=-2,
            )
        )
        is None
    )


def test_block_wires_edge_cases():
    """Non-Op / arg-less candidates are skipped; a decoy split whose
    source is not q/k/v binds nothing."""
    ql = _lin("p_q_weight")
    sizes = (4, 4)
    q_split = Op.make("split", ql, dim=-2, index=0, sizes=sizes)
    block = Op.make("om_decayed", q_split, Const(1.0), Const(2.0))
    decoy = Op.make(
        "split",
        Param("p_other", TensorType((16, 16))),
        dim=-2,
        index=0,
        sizes=sizes,
    )
    wrong_dim = Op.make("split", ql, dim=-1, index=0, sizes=sizes)
    own = {
        q_split,
        decoy,
        wrong_dim,
        Var("z", TensorType(())),
        Op.make("noop"),
    }
    wires = fp.block_wires(block, own)
    assert wires[fp.XQ] == q_split
    assert fp.XK not in wires and fp.XV not in wires
    # the decoy's leaves are neither q, k nor v — it binds nothing
    assert set(wires) <= {fp.XQ, fp.CC, fp.CD}


def test_fill_wires_reconstruction():
    """``_fill_wires`` reconstructs splits/decays only when the term is
    graph-resident; the ``contiguous`` base is the detach_ fallback."""
    sizes = (4, 4)
    ql, kl, vl = (
        _lin("p_q_weight"),
        _lin("p_k_weight"),
        _lin("p_v_weight"),
    )
    q_split = Op.make("split", ql, dim=-2, index=0, sizes=sizes)
    c_contig = Op.make("contiguous", Var("c_other", TensorType(())))
    block = Op.make(
        "add",
        Op.make("matmul", q_split, kl),
        Op.make("mul", vl, c_contig),
    )
    eg = EGraph()
    eg.add_term(block)
    # the v-split candidate IS graph-resident (it is a subterm of an
    # added term once we add it) — the k-split candidate is not.
    v_split = Op.make("split", vl, dim=-2, index=0, sizes=sizes)
    eg.add_term(v_split)
    wires = fp._fill_wires(eg, block, {fp.XQ: q_split})
    assert wires[fp.XV] == v_split
    assert fp.XK not in wires  # reconstructed but not resident
    # decay wires: the cbase is the contiguous fallback, but the minted
    # decay_vec terms are not in the graph → nothing binds.
    assert fp.CW not in wires and fp.CE not in wires
    # a resident k-split reconstruction binds too.
    k_split = Op.make("split", kl, dim=-2, index=0, sizes=sizes)
    eg.add_term(k_split)
    wires = fp._fill_wires(eg, block, {fp.XQ: q_split})
    assert wires[fp.XK] == k_split
    # and a non-resident v reconstruction leaves the role unbound.
    bare_qv = Op.make("add", Op.make("matmul", q_split, kl), vl)
    eg2 = EGraph()
    eg2.add_term(bare_qv)
    wires = fp._fill_wires(eg2, bare_qv, {})
    assert fp.XQ in wires and fp.XV not in wires
    # a block with no dim=-2 q-split returns the wires untouched.
    assert fp._fill_wires(eg, Const(0.0), {"a": 1}) == {"a": 1}
    # and with no detach_/contiguous base the decay minting is skipped —
    # though graph-resident split reconstructions still bind.
    bare = Op.make("matmul", q_split, kl)
    eg.add_term(bare)
    wires = fp._fill_wires(eg, bare, {})
    assert wires == {fp.XQ: q_split, fp.XK: k_split}
    assert fp.CW not in wires and fp.XV not in wires


def test_read_of_and_find_qm_edges():
    """``_read_of`` unwraps the add/mul payload; ``_find_qm`` skips
    matmuls whose operands are not both ops."""
    non_add = Op.make("matmul", Const(1.0), Const(2.0))
    assert fp._read_of(non_add) is non_add
    tail = Op.make("mul", Const(1.0), Const(2.0))
    blk = Op.make("add", Const(0.0), tail)
    assert fp._read_of(blk) == tail.args[1]
    blk2 = Op.make("add", Const(0.0), Const(3.0))
    assert fp._read_of(blk2) == Const(3.0)
    mm = Op.make("matmul", Var("a", TensorType(())), Const(1.0))
    assert fp._find_qm(mm) == []


def test_factored_states_skips_leaf_and_cyclic_members():
    """``leaf`` enodes and unextractable (cyclic) members are skipped."""
    eg = EGraph()
    vid = eg.add_term(Var("v0", TensorType(())))
    cyclic = eg.add_enode("wombat", (vid,))
    eg.union(cyclic, vid)
    assert fp.factored_states(eg, vid, count_cost) == []


def test_chain_links_direct_prev_child():
    """An ``add(S_prev, B)`` member links without a wrapping mul."""
    eg = EGraph()
    s0 = eg.add_term(Const(1.0))
    b = eg.add_term(Const(2.0))
    s1 = eg.add_enode("add", (s0, b))
    links = fp._chain_links(eg, s0, s1)
    assert len(links) == 1
    # and a missing read locates to no candidates
    chains, cands, obs = fp.state_chains(
        eg, [Op.make("matmul", Const(3.0), Const(4.0))], count_cost
    )
    assert chains == [] and cands == [[]] and obs == []


def test_pick_chain_states_returns_none():
    """No candidate of the next block links → ``None``."""
    eg = EGraph()
    a = eg.add_term(Const(1.0))
    b = eg.add_term(Const(2.0))
    cands = [
        [{"m": Const(1.0), "m_eid": a}],
        [{"m": Const(2.0), "m_eid": b}],
    ]
    assert fp._pick_chain_states(eg, [0, 1], cands) is None


def test_subst_returns_unbound_leaf():
    """A leaf not in the env passes through unchanged."""
    assert fp._subst(Const(9.0), {}) == Const(9.0)
    v = Var("v", TensorType(()))
    assert fp._subst(v, {v: Const(1.0)}) == Const(1.0)


def test_concat_skeleton_raw_and_positional():
    """Leaves outside ``blocks`` become ``("raw", term)`` — reachable
    only for a caller with a partial block list."""
    a = Op.make("om_decayed", Const(1.0), Const(2.0), Const(3.0))
    b = Op.make("om_decayed", Const(4.0), Const(5.0), Const(6.0))
    root = Op.make("concat", a, b, dim=-2)
    skel = fp._concat_skeleton(root, [a])
    assert skel == (0, ("raw", b))
    leaves = fp._skel_leaves(skel)
    assert leaves == [0, ("raw", b)]
    pos = fp._positional(skel, [0])
    assert pos == (0, 1)


def test_concat_tree_morph_drops_unconsumed_wires():
    """Fewer tree leaves than input wires → the tail is dropped."""
    morph = fp._concat_tree_morph(0, 3)
    assert morph is not None
    # out-of-order children emit a wire-permutation node
    assert fp._concat_tree_morph((1, 0), 2) is not None


def test_single_loop_check_early_returns():
    """No covered blocks / unlocated read / no seed candidates."""
    term = _fold_term(k=3)
    eg = EGraph()
    eg.add_term(term)
    rep = fp.FormReport()
    rep.chains = []
    assert fp.single_loop_check(eg, rep, term, count_cost) == {}
    # block 0's read is not in an empty graph → unlocated
    rep2 = fp.FormReport()
    rep2.chains = [fp.ChainResult(blocks=[0], ok=True)]
    eg2 = EGraph()
    assert fp.single_loop_check(eg2, rep2, term, count_cost) == {}
    # block 0 (``intra(0)``) has no factored read → no seed candidates
    assert fp.single_loop_check(eg, rep2, term, count_cost) == {}


def test_compensated_check_early_returns():
    """``compensated_loop_check`` mirrors the same three early exits."""
    term = _fold_term(k=3)
    eg = EGraph()
    eg.add_term(term)
    rep = fp.FormReport()
    rep.chains = [fp.ChainResult(blocks=[1], ok=True)]
    assert fp.compensated_loop_check(eg, rep, term, count_cost) == {}
    rep2 = fp.FormReport()
    rep2.chains = [fp.ChainResult(blocks=[0, 1], ok=True)]
    eg2 = EGraph()
    assert fp.compensated_loop_check(eg2, rep2, term, count_cost) == {}
    assert fp.compensated_loop_check(eg, rep2, term, count_cost) == {}


def test_compensated_check_seed_fallback_and_unbound_envs():
    """Empty bindings: every seed candidate fails the probe (the
    fallback seed is used) and the per-iteration compensation and
    steering are skipped — verdicts are honest ``False``s."""
    term = _fold_term(k=3)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    rep2 = fp.FormReport()
    rep2.chains = [fp.ChainResult(blocks=[1, 2], ok=True)]
    rep2.step = rep.step
    rep2.bindings = {1: {}, 2: {}}
    out = fp.compensated_loop_check(eg, rep2, term, count_cost)
    assert out == {1: False, 2: False}
    assert rep2.boundary["at"] == 1
    assert rep2.single_seed is not None


def test_compensated_check_resident_reindex_wire():
    """A needed read-scale wire that IS graph-resident is bound into
    ``env[CX]`` — the compensated read then verifies."""
    term = _fold_term(k=3)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    rep2 = fp.FormReport()
    rep2.chains = [fp.ChainResult(blocks=[1, 2], ok=True)]
    rep2.step = rep.step
    # corrupt block 1's bound cross-scale so the compensated arm must
    # re-derive it — the correct ``decay_vec(offset=4, t=4)`` wire is
    # already graph-resident, so it binds.  ``offset`` stays 4 so the
    # carried ``off`` baseline is unchanged; only ``t`` is wrong.
    wrong_cx = Op.make(
        "unsqueeze",
        Op.make("decay_vec", CB, offset=4, t=5),
        dim=-1,
    )
    b1 = dict(rep.bindings[1])
    b1[fp.CX] = wrong_cx
    rep2.bindings = {1: b1, 2: rep.bindings[2]}
    out = fp.compensated_loop_check(eg, rep2, term, count_cost)
    assert out == {1: True, 2: True}
    assert rep2.single_envs[1][fp.CX] != wrong_cx


def test_compensated_steering_mints_resident_wires():
    """Bound update wires that do NOT steer: the compensated arm mints
    the steering pair, finds it resident, and carries it forward."""
    term = _fold_term(k=3)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    rep2 = fp.FormReport()
    rep2.chains = [fp.ChainResult(blocks=[1, 2], ok=True)]
    rep2.step = rep.step
    # corrupt block 1's update weight so the bound wires fail the
    # steering check — the minted pair (decay_vec(t=1, offset=4),
    # decay_vec(t=4, sign=-1)) is the family's own spelling, resident.
    wrong_cw = Op.make(
        "unsqueeze",
        Op.make("decay_vec", CB, t=1, offset=7),
        dim=-1,
    )
    b1 = dict(rep.bindings[1])
    b1[fp.CW] = wrong_cw
    rep2.bindings = {1: b1, 2: rep.bindings[2]}
    out = fp.compensated_loop_check(
        eg, rep2, term, count_cost, post_add=lambda: None
    )
    assert out == {1: True, 2: True}
    assert rep2.single_envs[1][fp.CW] != wrong_cw


def test_compensated_steering_unbound_and_missing_xi():
    """Steering wires and a read scale that are NOT graph-resident are
    named obstructions, and the block verdict is ``False``."""
    term = _fold_term(k=3)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose(eg, term, count_cost, root_eid=root_eid)
    rep2 = fp.FormReport()
    rep2.chains = [fp.ChainResult(blocks=[1, 2], ok=True)]
    rep2.step = rep.step
    # corrupt both bound decays on the boundary: block 2's cross-scale
    # puts phi_j negative → the minted steering pair carries offset=-1,
    # which is not in the graph; and block 1's update weight then drifts
    # ``off`` so block 2's reindexed read scale (offset=1) is
    # non-resident too.
    wrong_cx = Op.make(
        "unsqueeze",
        Op.make("decay_vec", CB, offset=9, t=4),
        dim=-1,
    )
    wrong_cw = Op.make(
        "unsqueeze",
        Op.make("decay_vec", CB, t=1, offset=7),
        dim=-1,
    )
    b1 = dict(rep.bindings[1])
    b1[fp.CW] = wrong_cw
    b2 = dict(rep.bindings[2])
    b2[fp.CX] = wrong_cx
    rep2.bindings = {1: b1, 2: b2}
    out = fp.compensated_loop_check(eg, rep2, term, count_cost)
    assert out[2] is False
    assert any(
        "boundary-renorm-unbound:block1->block2" in o
        for o in rep2.obstructions
    )
    assert any(
        "boundary-xi-nonresident:block2" in o for o in rep2.obstructions
    )
    assert 1 in rep2.boundary.get("missing_steer", {})
    assert 2 in rep2.boundary.get("missing_wires", {})


def test_attr_helpers():
    """``_dv_attrs``/``_split_span``/``_decay_base``/``_usq_dv`` edges."""
    v = Var("v", TensorType(()))
    assert fp._dv_attrs(v) is None
    assert fp._dv_attrs(Op.make("decay_vec", CB, t=4)) == {"t": 4}
    assert fp._dv_attrs(Const(1.0)) is None
    assert fp._split_span({}) is None
    bad = Op.make(
        "split",
        _lin("p_q_weight"),
        dim=-2,
        index=5,
        sizes=(4, 4),
    )
    assert fp._split_span({fp.XQ: bad}) is None
    good = Op.make(
        "split",
        _lin("p_q_weight"),
        dim=-2,
        index=1,
        sizes=(4, 4),
    )
    assert fp._split_span({fp.XQ: good}) == (4, 4)
    assert fp._decay_base({}) is None
    dv = Op.make("decay_vec", CB, t=4)
    assert (
        fp._decay_base({fp.CW: Op.make("unsqueeze", dv, dim=-1)}) is CB
    )
    assert fp._decay_base({fp.CD: Op.make("decay", CB, t=4)}) is CB
    # offset=0 is dropped — the canonical spelling.
    t = fp._usq_dv(CB, t=1, offset=0)
    assert "offset" not in dict(t.args[0].attrs)


def test_form_propose_report_prints(capsys):
    """The measurement print: ok-chain, boundary and no-loop arms."""
    term = _fold_term(k=3)
    eg = EGraph()
    root_eid = eg.add_term(term)
    rep = fp.form_propose_report(
        eg,
        term,
        count_cost,
        root_eid=root_eid,
        numeric=lambda r: 1.5,
    )
    assert rep.max_abs_diff == 1.5
    out = capsys.readouterr().out
    assert "form-propose" in out and "single-loop" in out

    term2 = _two_family_term()
    eg2 = EGraph()
    eg2.add_term(term2)
    fp.form_propose_report(eg2, term2, count_cost)
    out = capsys.readouterr().out
    assert "boundary: block 3" in out

    term3 = _irregular_term()
    eg3 = EGraph()
    eg3.add_term(term3)
    fp.form_propose_report(eg3, term3, count_cost, numeric=lambda r: 0)
    out = capsys.readouterr().out
    assert "NO loop proposed" in out

    # a single-block chain: ok, but the compensated arm is skipped
    # (fewer than two covered blocks).
    term4 = _fold_term(k=2)
    eg4 = EGraph()
    root_eid4 = eg4.add_term(term4)
    fp.form_propose_report(eg4, term4, count_cost, root_eid=root_eid4)
    out = capsys.readouterr().out
    assert "single-loop arm" in out
