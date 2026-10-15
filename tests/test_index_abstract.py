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
from catopt_core.ir import Const, Op, TensorType, Var

from lab.bridge import combinator as cb
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
