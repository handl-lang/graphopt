"""Coverage-gap tests for mlopt_core.meta.

The behavioural suite (test_meta / test_synthesis_*) exercises the
happy paths of ``canonicalize``/``stratified_run``/``synthesize_rules``.
This file drives the internals the happy paths never reach:

* canonicalize on *pattern-shaped* terms — metavar leaves sorted by
  ``op_repr``, attrs preserved through the rebuild, memo reuse;
* ``stratified_run`` with no extraction objective (the ``extract``
  short-circuit) and ``canonicalize_output=False``;
* ``_iter_module_rules`` over list/tuple attributes with dedup;
* ``_common_prefix``'s deliberate non-strict zip;
* every veto branch of ``match_pattern``/``apply_rewrite_at``/
  ``_fire_guarded``/``_compose_guards``/``_reexpress_*``;
* ``_leaf_generalize`` keep/names/fresh paths, ``_eval_term`` /
  ``_eval_allclose`` / ``_term_is_ground`` leaves;
* ``_validate_candidate`` well-formedness + witness + replay +
  numeric acceptance and rejection;
* ``synthesize_rules`` fuel-bounded early exits, the placeholder-attr
  binding resolution (``conc1``), seed-side guard vetoes, and the
  un-reexpressible-binding rejection channel.

Defensive branches believed unreachable by construction (flagged for
``pragma: no cover`` rather than contorted tests) are listed in
``test_defensive_branch_inventory`` at the bottom.
"""

import types

import pytest
import torch

from mlopt_core import meta
from mlopt_core.egraph import EGraph, Rewrite
from mlopt_core.ir import Const, Op, Param, TensorType, Var


def _T(*shape):
    return TensorType(tuple(shape) if shape else (4, 4))


def _opcount(t, memo=None):
    """Additively-counts Op nodes — a valid ``(term, memo)`` cost fn."""
    if memo is None:
        memo = {}
    if t in memo:
        return memo[t]
    out = (
        1.0 + sum(_opcount(a, memo) for a in t.args)
        if isinstance(t, Op)
        else 0.0
    )
    memo[t] = out
    return out


# ---------------------------------------------------------------------------
# canonicalize — pattern-shaped terms, attrs, memoisation
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# stratified_run — extraction short-circuits
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# module_rules / _iter_module_rules / positions
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# _has_attr_metavars / _synthesizable
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# match_pattern / apply_rewrite_at / _fire_guarded
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# re-expression helpers
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# _compose_guards — every _eval branch
# ---------------------------------------------------------------------------


def _guard_pair(
    r1_check=None,
    r1_derive=None,
    r2_check=None,
    r2_derive=None,
    pat1=None,
    pat2=None,
):
    r1 = Rewrite(
        "g1",
        Op.make("n1", "x", "y"),
        Op.make("n1o", "x", "y"),
        check=r1_check,
        derive=r1_derive,
    )
    r2 = Rewrite(
        "g2",
        Op.make("n2", "u"),
        Op.make("n2o", "u"),
        check=r2_check,
        derive=r2_derive,
    )
    p1 = pat1 if pat1 is not None else {"x": "x", "y": "y"}
    p2 = pat2 if pat2 is not None else {"u": "u"}
    return r1, r2, p1, p2










# ---------------------------------------------------------------------------
# concrete-matched leaves / leaf generalisation / alpha key
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def test_eval_term_leaf_and_op_paths():
    x = Var("x", _T())
    env = {x: torch.ones(4, 4, dtype=torch.float64)}
    assert float(meta._eval_term(Const(2.0), env)) == 2.0
    assert meta._eval_term(x, env) is env[x]
    t = meta._eval_term(Op.make("add", x, x), env)
    assert float(t.max()) == 2.0
    # ops without a torch binding raise KeyError, odd leaves TypeError
    with pytest.raises(KeyError):
        meta._eval_term(Op.make("no_such_op", x), env)
    with pytest.raises(TypeError):
        meta._eval_term(3.5, env)
    with pytest.raises(TypeError):
        meta._eval_term("metavar", env)


def test_eval_allclose_and_term_is_ground():
    a = torch.ones(2, 2, dtype=torch.float64)
    assert meta._eval_allclose(a, a.clone())
    assert meta._eval_allclose((a, a), (a.clone(), a.clone()))
    # length-mismatched tuples and non-tensor leaves are not equal
    assert not meta._eval_allclose((a,), (a, a))
    assert not meta._eval_allclose(a, "nope")
    assert not meta._eval_allclose("x", "y")
    # broadcastable-but-rank-mismatched operands are NOT equal —
    # the oracle must not call (4,) == (1,4) or (4,4) == () true
    v = torch.ones(4, dtype=torch.float64)
    assert not meta._eval_allclose(v, v.reshape(1, 4))
    assert not meta._eval_allclose(
        a, torch.tensor(1.0, dtype=torch.float64)
    )
    x = Var("x", _T())
    assert meta._term_is_ground(Op.make("add", x, Const(1.0)))
    assert not meta._term_is_ground("mv")
    assert not meta._term_is_ground(Op.make("add", "mv", x))
    assert not meta._term_is_ground(
        Op.make("transpose", x, dim0="D", dim1=-1)
    )


# ---------------------------------------------------------------------------
# _replays / instantiation pool / tensor env
# ---------------------------------------------------------------------------






def test_tensor_env_skips_unshaped_leaves():
    v = Var("v", TensorType((2, 3)))
    open_dim = Var("w", TensorType((None, 4)))
    env = meta._tensor_env({"a": v, "b": open_dim, "c": Const(1.0)})
    assert env[v].shape == (2, 3)
    assert env[v].dtype == torch.float64
    # non-integer dims and non-Var leaves contribute nothing
    assert open_dim not in env and len(env) == 1


# ---------------------------------------------------------------------------
# _validate_candidate / _subsumed — acceptance & rejection
# ---------------------------------------------------------------------------








def test_validate_candidate_rejects_unreplayable_and_unsound():
    x, y = Var("x", _T()), Var("y", _T())
    cand = Rewrite("c", Op.make("add", "x", "y"), Op.make("mul", "x", "y"))
    # claimed parents cannot produce mul at all → fatal replay failure
    wrong1 = Rewrite("w1", Op.make("add", "a", "b"), Op.make("add", "b", "a"))
    wrong2 = Rewrite("w2", Op.make("add", "a", "b"), "a")
    assert not meta._validate_candidate(
        cand, wrong1, wrong2, numeric=False, witness={"x": x, "y": y}
    )
    # a derivation that replays but is semantically WRONG is rejected
    # by the numeric check, never trusted
    bad1 = Rewrite("b1", Op.make("add", "a", "b"), Op.make("mul", "a", "b"))
    id2 = Rewrite("b2", Op.make("mul", "a", "b"), Op.make("mul", "a", "b"))
    assert not meta._validate_candidate(
        cand, bad1, id2, numeric=True, witness={"x": x, "y": y}
    )




# ---------------------------------------------------------------------------
# synthesize_rules — fuel bounds, seed-path internals, guard_pats
# ---------------------------------------------------------------------------












def test_synthesized_rules_replay_and_eval_sound():
    """Real acceptance path: derived rules replay on a fresh instance
    and evaluate fp64-identical on random tensors."""
    x, y = Var("x", _T()), Var("y", _T())
    comm = Rewrite("cm", Op.make("add", "a", "b"), Op.make("add", "b", "a"))
    negid = Rewrite(
        "nn", Op.make("neg", Op.make("neg", "a")), "a"
    )
    seed = Op.make("add", Op.make("neg", Op.make("neg", x)), y)
    out = meta.synthesize_rules(
        [comm, negid], [seed], fuel=500, numeric_check=True
    )
    assert isinstance(out, list)
    torch.manual_seed(0)
    env = {v: torch.randn(4, 4, dtype=torch.float64) for v in (x, y)}
    checked = 0
    for d in out:
        assert not meta._is_tautology(d.lhs, d.rhs)
        subst = meta.match_pattern(d.lhs, seed, {})
        if subst is None:
            continue
        t0 = meta.instantiate_pattern(d.lhs, subst)
        applied = meta.apply_rewrite_at(d, t0, ())
        assert applied is not None
        assert meta._eval_allclose(
            meta._eval_term(t0, env), meta._eval_term(applied, env)
        )
        checked += 1
    assert checked >= 1  # at least one derived rule fired on the seed




# ---------------------------------------------------------------------------
# defensive branches believed unreachable — pragma candidates
# ---------------------------------------------------------------------------


