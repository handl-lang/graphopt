"""Tests for the ADR-0008 gate-3 loop-proposing rule.

``mlopt_discovery.loop_propose`` supplies the rewrite
``unroll(step, n) -> Loop(step, n)`` over the combinator presentation and
the smallest search that can carry it.  The properties under test:
``unroll`` and ``Loop`` denote the same morphism, the rule fires on the
presentation's spelled scan, and it does **not** fire on a flat DAG
compilation (the gate-3 negative).
"""

import pytest
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_discovery import combinator as cb

from mlopt_discovery import loop_propose as lp


def _v(name, *shape):
    return Var(name, TensorType(shape))


def _step():
    """The RetNet shared-state step, ``S ⊗ (q,k,v) → S' ⊗ o``."""
    return cb.retnet_step(Const(0.9), Const(0.9**4), 8.0, 4)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_unroll_denotes_the_loop(n):
    """``unroll(step, n)`` and ``Loop(step, n)`` are the same morphism."""
    step = _step()
    unrolled = lp.unroll(step, n)
    loop = lp._to_loop(step, n)
    assert cb.arity(unrolled) == cb.arity(loop)
    wires = [_v(f"w{i}", 2, 4, 4) for i in range(1 + 3 * n)]
    assert cb._apply(unrolled, list(wires)) == cb._apply(
        loop, list(wires)
    )


def test_unroll_arity_and_guards():
    """The arity is ``S ⊗ X^n → S ⊗ Y^n``; bad inputs raise."""
    step = _step()
    assert cb.arity(lp.unroll(step, 3)) == (1 + 3 * 3, 1 + 3 * 1)
    with pytest.raises(ValueError):
        lp.unroll(step, 0)
    with pytest.raises(ValueError):
        lp.unroll(cb.Arr("neg", 1), 2)  # no state wire


def test_folds_skips_the_degenerate_n1():
    """A bare step is ``unroll(step, 1)`` — not a fold, so not matched."""
    assert lp._folds(_step()) == []


def test_folds_finds_the_spelled_scan():
    """The presentation's spelled scan is matched with its true ``n``."""
    step = _step()
    spelled = lp.unroll(step, 4)
    ns = {n for _s, _st, n in lp._folds(spelled)}
    assert 4 in ns
    assert 1 not in ns


def test_loop_headed_detects_loops():
    """``loop_headed`` reports a ``Loop`` anywhere in the term."""
    assert not lp.loop_headed(_step())
    assert lp.loop_headed(cb.Loop(_step(), 3, nx=3, ny=1))


def test_propose_loop_fires_on_the_spelled_scan():
    """The search folds the spelled scan to a ``Loop``-headed winner."""
    step = _step()
    spelled = lp.unroll(step, 4)
    res = lp.saturate_comb(spelled, [lp.PROPOSE_LOOP, lp.EXPAND_LOOP])
    assert res["fired"]["loop_propose"] >= 1
    assert res["loop_reachable"]
    assert lp.loop_headed(res["best"])
    assert res["cost"] < cb.morph_stats(spelled).nodes


def test_propose_loop_does_not_fire_on_a_flat_dag():
    """Gate 3's negative: a flat DAG compilation has no spelled scan.

    A bridge-compiled DAG is a ``seq`` of ``arr`` steps, not the
    presentation's regular ``perm``/``step`` scan — so the proposing
    rule finds nothing and the search stays un-``loop``-ed.
    """
    x = _v("x", 2, 4)
    w = Param("w", TensorType((4, 4)))
    flat = cb.dag_to_comb(Op.make("mul", Op.make("add", x, w), w))
    assert lp._folds(flat) == []
    res = lp.saturate_comb(flat, [lp.PROPOSE_LOOP, lp.EXPAND_LOOP])
    assert res["fired"] == {"loop_propose": 0, "loop_expand": 0}
    assert not res["loop_reachable"]
    assert not lp.loop_headed(res["best"])


def test_replace_rebuilds_inside_composites():
    """``_replace`` recurses into ``Seq``/``Par``/``Loop`` and swaps a
    matched subterm wholesale; non-matches come back unchanged."""
    target = cb.Id(1)
    new = cb.Arr("neg", 1)
    assert lp._replace(target, target, new) is new
    assert lp._replace(cb.Id(3), target, new) == cb.Id(3)
    assert lp._replace(cb.Par((target, cb.Id(2))), target, new) == cb.Par(
        (new, cb.Id(2))
    )
    assert lp._replace(cb.Seq((target,)), target, new) == cb.Seq((new,))
    assert lp._replace(cb.Loop(target, 2, 1, 1), target, new) == cb.Loop(
        new, 2, 1, 1
    )


def test_in_arity_declines_non_morphisms():
    """A term that is not a morphism reports ``-1``, not a raise."""
    assert lp._in_arity(object()) == -1
    assert lp._in_arity(cb.Id(2)) == 2


def test_apply_skips_a_noop_build():
    """A rule whose ``build`` returns the matched term rewrites
    nothing — the ``new == matched`` skip keeps it out of results."""
    rule = lp.CombRule(
        "noop",
        find=lambda m, _n: [(m, m, 2)],
        build=lambda step, _n: step,
    )
    assert rule.apply(cb.Id(1)) == []


def test_saturate_comb_expansion_never_improves_cost():
    """EXPAND alone: the unroll is reachable but strictly costlier —
    the seed stays best, and the new term is not ``Loop``-headed."""
    step = _step()
    loop = lp._to_loop(step, 2)
    res = lp.saturate_comb(loop, [lp.EXPAND_LOOP])
    assert res["fired"]["loop_expand"] >= 1
    assert res["best"] is loop
    assert res["cost"] == cb.morph_stats(loop).nodes
    # loop_reachable only counts *rewritten* terms — the expansion
    # carries no Loop, so the verdict stays False on this seed
    assert not res["loop_reachable"]


def test_propose_loop_report_spelled_and_flat(capsys):
    """The gate-3 printer: spelled-scan hits plus the control, and
    the flat-DAG negative."""
    step = _step()
    lp.propose_loop_report(lp.unroll(step, 3), lp.unroll(step, 2))
    out = capsys.readouterr().out
    assert "match: unroll(step," in out
    assert "control (authored spelled scan)" in out

    x = _v("x", 2, 4)
    w = Param("w", TensorType((4, 4)))
    flat = cb.dag_to_comb(Op.make("mul", Op.make("add", x, w), w))
    lp.propose_loop_report(flat)
    out = capsys.readouterr().out
    assert "found=0" in out
    assert "not the presentation's spelled scan" in out
