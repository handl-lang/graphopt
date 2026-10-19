"""Tests for the ADR-0008 gate-3 parametric-abstraction move.

``lab/bridge/index_abstract.py`` supplies the *abstraction over the step*
the gate-3 retro named as the missing move: it binds the per-block index
(and the recurrence state) as **parameters**, turning ``K`` distinct block
subterms into **one shared step** — the ``unroll(step, K)`` spelling the
``loop``-proposing rule then folds.  The properties under test: the move
recognises a *regular* fold, builds a step that round-trips, and reports
an *irregular* one with the first structural difference (the gate-3
negative).
"""

import pytest
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_discovery import combinator as cb

from lab.bridge import index_abstract as ia
from lab.bridge import loop_propose as lp


def _v(name, *shape):
    return Var(name, TensorType(shape))


def _fold(k=4, sizes=None, extra_mul=None):
    """A clean ``K``-block fold: ``o_i = b_i·S_{i-1} + c``, ``S_i = a·S_{i-1} + b_i``.

    The state is read *through* the block's own input (``b_i·S_{i-1}``) —
    the shape a real attention step has (the state is read via a query) —
    so the block's own inputs shield the state from the "largest term that
    carries no own split" test.

    ``extra_mul`` — when given, an extra factor applied to *block 1 only*,
    which makes the family irregular (a difference the abstraction must
    report rather than paper over).
    """
    sizes = sizes or (16,) * k
    x = _v("x", 2, 16 * k, 16)
    a, c = Const(0.9), Const(2.0)
    state = _v("s0", 2, 16, 16)
    outs = []
    for i in range(k):
        b = Op.make("split", x, sizes=sizes, dim=-2, index=i)
        o = Op.make("add", Op.make("mul", b, state), c)
        if extra_mul is not None and i == 1:
            o = Op.make("mul", extra_mul, o)
        outs.append(o)
        state = Op.make("add", Op.make("mul", a, state), b)
    root = outs[0]
    for o in outs[1:]:
        root = Op.make("concat", root, o, dim=-2)
    return root


def test_block_outputs_flattens_the_concat_chain():
    """The ``K`` block outputs are the leaves of the ``concat`` chain."""
    blocks = ia.block_outputs(_fold(4))
    assert len(blocks) == 4
    assert [b.op for b in blocks] == ["add", "add", "add", "add"]


def test_block_outputs_of_a_non_concat_is_one_block():
    """A non-``concat`` root is a single (degenerate) block."""
    assert len(ia.block_outputs(Op.make("neg", _v("z", 2)))) == 1


def test_index_family_groups_the_splits():
    """The per-block splits group by ``(sizes, dim)`` → indices."""
    fams = ia.index_family(_fold(4))
    assert list(fams.values()) == [[0, 1, 2, 3]]


def test_regular_fold_collapses_to_one_step():
    """A regular fold: the abstraction binds the index and the state."""
    rep = ia.abstract_index(_fold(4))
    assert rep.verdict == "regular"
    assert rep.agree
    assert rep.diff == ""
    assert rep.n_blocks == 4
    assert rep.n_inputs == 1
    assert rep.roundtrip


def test_regular_fold_builds_an_unroll_that_folds_to_a_loop():
    """The shared step spells as ``unroll(step, K)``; the closure loops it."""
    rep = ia.abstract_index(_fold(4))
    assert cb.arity(rep.step) == (2, 2)  # S ⊗ X → S' ⊗ o
    res = lp.saturate_comb(rep.morph, [lp.PROPOSE_LOOP, lp.EXPAND_LOOP])
    assert res["loop_reachable"]
    assert lp.loop_headed(res["best"])
    assert res["cost"] < cb.morph_stats(rep.morph).nodes


def test_irregular_fold_is_reported_with_its_cause():
    """A block that differs structurally is named, not silently folded."""
    rep = ia.abstract_index(_fold(4, extra_mul=Const(3.0)))
    assert rep.verdict == "irregular"
    assert not rep.agree
    assert rep.diff  # the first difference, not a bare "no"
    assert rep.morph is None  # nothing to fold


def test_single_block_is_not_a_fold():
    """One block is not a recurrence — the move declines."""
    rep = ia.abstract_index(Op.make("neg", _v("z", 2)))
    assert not rep.agree
    assert rep.diff == "fewer than two blocks"


def test_agreeing_templates_that_do_not_reproduce_are_irregular():
    """A fold whose state read is linear is *declined*, not mis-folded.

    ``o_i = c·S_{i-1} + b_i`` reads the state without an own-dependent
    node, so the block-local transform ``c·S_{i-1}`` is the largest
    own-split-free subterm — the templates still agree, but the abstracted
    state is not the recurrence state, so the fold does not reproduce the
    term.  The verdict must be ``irregular``, not a false ``regular``.
    """
    x = _v("x", 2, 64, 16)
    a, c = Const(0.9), Const(2.0)
    state = _v("s0", 2, 16, 16)
    outs = []
    for i in range(4):
        b = Op.make("split", x, sizes=(16,) * 4, dim=-2, index=i)
        outs.append(Op.make("add", Op.make("mul", c, state), b))
        state = Op.make("add", Op.make("mul", a, state), b)
    root = outs[0]
    for o in outs[1:]:
        root = Op.make("concat", root, o, dim=-2)
    rep = ia.abstract_index(root)
    assert rep.agree
    assert not rep.roundtrip
    assert rep.verdict == "irregular"


@pytest.mark.parametrize("k", [2, 3, 5])
def test_regular_fold_round_trips_for_several_k(k):
    """The fold reproduces the original blocks for a range of ``K``."""
    rep = ia.abstract_index(_fold(k))
    assert rep.agree
    assert rep.roundtrip


def test_state_inputs_finds_the_recurrence_state():
    """``state_in[i]`` is the subterm the block reads but does not build."""
    blocks = ia.block_outputs(_fold(4))
    own = ia._block_splits(blocks)
    state_in = ia._state_inputs(blocks, own)
    assert state_in[0] is None  # the boundary
    # blocks 1..3 read the previous block's state (an ``add``)
    assert [s.op for s in state_in[1:]] == ["add", "add", "add"]


def test_compile_with_inputs_consumes_the_inputs():
    """A ``pack(S', o)`` template compiles to ``S ⊗ X → S' ⊗ o``."""
    tmpl = Op.make(
        "pack",
        Op.make("add", ia._state_var(), ia._input_var(0)),
        Op.make("mul", Const(2.0), ia._state_var()),
    )
    step = ia._compile_with_inputs(
        tmpl, [ia._state_var(), ia._input_var(0)]
    )
    assert cb.arity(step) == (2, 2)


# ---------------------------------------------------------------------------
#  The block width as a parameter (gate 3, cause 1)
# ---------------------------------------------------------------------------


def _fold_widths(widths, surviving=None, structural=False):
    """A ``K``-block fold whose blocks differ in their block **width**.

    ``widths`` — the per-block width.  ``surviving`` — when given, a
    ``decay_vec`` carrying the width on ``t`` that *survives* into the
    template (the width does not reach a wire).  ``structural`` — when
    true, block 2 gets an extra factor, a *non*-parametric difference the
    move must not paper over.
    """
    x = _v("x", 2, sum(widths), 16)
    state = _v("s0", 2, 16, 16)
    outs = []
    for i, w in enumerate(widths):
        b = Op.make("split", x, sizes=(w,), dim=-2, index=i)
        tail = (
            Op.make("decay_vec", Const(0.9), t=w, sign=-1)
            if surviving
            else Const(2.0)
        )
        o = Op.make("add", Op.make("mul", b, state), tail)
        if structural and i == 2:
            o = Op.make("mul", Const(3.0), o)
        outs.append(o)
        state = Op.make("add", Op.make("mul", Const(0.9), state), b)
    root = outs[0]
    for o in outs[1:]:
        root = Op.make("concat", root, o, dim=-2)
    return root


def test_own_split_width_is_already_a_wire():
    """A block's *own* split width needs no new parameter: it is the wire.

    The own split is replaced wholesale, so its ``sizes`` never reaches
    the template — a fold whose blocks differ only in their own width is
    *already* exactly regular.  This is the half of "abstract sizes" that
    the move always had; it is recorded so the other half is not mistaken
    for the whole.
    """
    rep = ia.abstract_index(_fold_widths((4, 16, 16, 4)))
    assert rep.agree
    assert rep.verdict == "regular"
    assert rep.roundtrip


def test_width_on_a_surviving_node_is_recognised_modulo_attrs():
    """The width-parametric recognition the gate-3 retro asked for.

    A width that survives into the template (here a ``decay_vec``'s
    ``t``) makes the templates differ *exactly* but leaves them the same
    step up to the block width and index: ``agree_mod`` is True and the
    param-attr diff is empty.
    """
    rep = ia.abstract_index(
        _fold_widths((4, 16, 16, 4), surviving=True)
    )
    assert not rep.agree
    assert rep.agree_mod
    assert rep.diff_mod == ""
    assert rep.diff  # the exact difference is still named
    # the widths the negative is up against are reported
    assert sorted(rep.param_families) == [
        (("sign", "-1"), ("t", "16")),
        (("sign", "-1"), ("t", "4")),
    ]


def test_mod_agreement_never_promotes_to_a_pass():
    """Parametric agreement explains a negative; it is never a verdict.

    The width-parametric fold still has no fold built (``morph`` is
    ``None``) and stays ``irregular``: the fixed-shape IR cannot carry a
    width parameter, so the exact collapse — and the round-trip — are
    still required.
    """
    rep = ia.abstract_index(
        _fold_widths((4, 16, 16, 4), surviving=True)
    )
    assert rep.verdict == "irregular"
    assert rep.morph is None
    assert not rep.roundtrip


def test_a_structural_difference_is_not_a_width_parameter():
    """A genuine structural difference survives the param-attr erasure."""
    rep = ia.abstract_index(
        _fold_widths((4, 16, 16, 4), structural=True)
    )
    assert not rep.agree
    assert not rep.agree_mod
    assert rep.diff_mod  # the non-parametric difference is named


def test_templates_agree_mod_implies_templates_agree_is_weaker():
    """Mod agreement is strictly weaker: exact agreement implies it."""
    exact = ia.abstract_index(_fold_widths((16, 16, 16, 16)))
    assert exact.agree and exact.agree_mod
    mod = ia.abstract_index(
        _fold_widths((4, 16, 16, 4), surviving=True)
    )
    assert mod.agree_mod and not mod.agree
