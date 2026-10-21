"""decay_vec + the cross-block factorization — rank-1 decay blocks.

``decay_vec(g, t=T, offset=o, sign=s)[k] = g^(s·k + o)`` — the xi/eta
vectors of the RetNet inter-chunk factorization.  ``decay_block_cross``
rewrites a strict-lower ``split(split(decay))`` block as the outer
product ``mul(unsq(xi,-1), unsq(eta,-2))``; ``om_decayed_cross`` then
distributes a decayed product over a rank-1 decay into row/column
scalings — the carried-state algebra the chunked recurrence needs.
"""

import pytest

torch = pytest.importorskip("torch")

import mlopt_torch.torch_bridge as tb  # noqa: E402
from mlopt_carriers.om import (  # noqa: E402
    DECAY_BLOCK_CROSS,
    DECAY_BLOCK_CROSS_FLAT,
    DECAY_STATE_WEIGHT_EXIT,
    DECAY_STATE_WEIGHT_SINK,
    DECAY_VEC_OFFSET,
    MATMUL_COL_DECAY,
    MATMUL_SCALAR_FACTOR,
    OM_CHUNKIFY,
    OM_DECAYED_CHUNKIFY_KEYS,
    OM_DECAYED_CHUNKIFY_ROWS,
    OM_DECAYED_CROSS,
    OM_DECAYED_CROSS_GEN,
    OM_LAWS,
    SHARED_STATE_MAT_REASSOC,
    SPLIT_COMMUTE,
    SPLIT_FLATTEN,
    SPLIT_MATMUL_COLS,
    SPLIT_MATMUL_ROWS,
    SPLIT_TRANSPOSE,
    UNSQ_MUL,
    _decay_vec_torch,
    om_decayed_kway_blocks,
    om_div_bridge_laws,
    om_kway_laws,
    om_shared_state_budgets,
    om_shared_state_laws,
    om_shared_state_stages,
)
from mlopt_core.egraph import EGraph  # noqa: E402
from mlopt_core.ir import (  # noqa: E402
    Op,
    TensorType,
    Var,
    op_repr,
)
from mlopt_core.laws.scan import SCAN_DIAG_LAWS  # noqa: E402
from mlopt_core.meta import (  # noqa: E402
    instantiate_pattern,
    match_pattern,
)
from mlopt_core.typing import shape_of  # noqa: E402
from mlopt_torch.torch_bridge import eval_term  # noqa: E402

G = torch.tensor(0.9, dtype=torch.float64)


def _v(name, *shape):
    return Var(name, TensorType(shape))


class TestDecayVecBinding:
    def test_unit_offset_sign(self):
        out = _decay_vec_torch(G, t=4)
        assert torch.allclose(
            out, G ** torch.arange(4, dtype=torch.float64)
        )

    def test_offset_shifts_exponent(self):
        out = _decay_vec_torch(G, t=3, offset=5)
        assert torch.allclose(
            out, G ** (torch.arange(3, dtype=torch.float64) + 5)
        )

    def test_negative_sign_inverts(self):
        out = _decay_vec_torch(G, t=3, sign=-1)
        expect = G ** (torch.arange(3, dtype=torch.float64) * -1)
        assert torch.allclose(out, expect)

    def test_per_head_leading_dims(self):
        gh = torch.tensor([0.9, 0.8], dtype=torch.float64)
        out = _decay_vec_torch(gh, t=3)
        assert out.shape == (2, 3)
        assert torch.allclose(
            out[1], torch.tensor([1.0, 0.8, 0.64], dtype=torch.float64)
        )

    def test_python_scalar_normalized(self):
        out = _decay_vec_torch(0.9, t=3)
        assert out.shape == (3,) and out[0] == 1.0

    def test_shape_rule(self):
        t = Op.make("decay_vec", _v("g"), t=7)
        assert shape_of(t) == (7,)
        th = Op.make("decay_vec", _v("g", 4), t=7)
        assert shape_of(th) == (4, 7)

    def test_shape_rule_declines(self):
        t = Op.make("decay_vec", _v("g"))
        assert shape_of(t) is None  # no t attr → unknown
        bad = Op.make("decay_vec", _v("g"), t=-1)
        assert shape_of(bad) == "__invalid_shape__"


class TestDecayBlockCross:
    T = 9

    def _split_term(self):
        return Op.make(
            "split",
            Op.make(
                "split",
                Op.make("decay", _v("g"), t=self.T),
                sizes=(4, 5),
                dim=-2,
                index=1,
            ),
            sizes=(4, 5),
            dim=-1,
            index=0,
        )

    def test_fires_and_derives(self):
        subst = match_pattern(
            DECAY_BLOCK_CROSS.lhs, self._split_term(), {}
        )
        assert subst is not None
        assert DECAY_BLOCK_CROSS.check(subst)
        der = DECAY_BLOCK_CROSS.derive(subst)
        assert der == {"$attr:TI": 5, "$attr:TJ": 4, "$attr:OD": 4}

    def test_rhs_spelling(self):
        subst = match_pattern(
            DECAY_BLOCK_CROSS.lhs, self._split_term(), {}
        )
        subst.update(DECAY_BLOCK_CROSS.derive(subst))
        out = instantiate_pattern(DECAY_BLOCK_CROSS.rhs, subst)
        r = op_repr(out)
        assert "decay_vec" in r and "sign=-1" in r and "offset=4" in r

    def test_numerically_equals_sliced_block(self):
        """The factored block IS the split block's content."""
        pos = torch.arange(self.T, dtype=torch.float64)
        D = torch.tril(
            G ** (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0)
        )
        block = D[4:9, 0:4]  # rows of chunk 1 x cols of chunk 0
        env = {"g": G}
        rhs_term = instantiate_pattern(
            DECAY_BLOCK_CROSS.rhs,
            {
                "g": _v("g"),
                "$attr:TI": 5,
                "$attr:TJ": 4,
                "$attr:OD": 4,
            },
        )
        factored = eval_term(
            rhs_term, var_env=env, bindings=tb._IR_TO_TORCH
        )
        assert torch.allclose(block, factored, atol=1e-12)

    def test_diagonal_declines(self):
        """i == j is the diagonal — decay_block_diag's block, not this."""
        term = Op.make(
            "split",
            Op.make(
                "split",
                Op.make("decay", _v("g"), t=self.T),
                sizes=(4, 5),
                dim=-2,
                index=1,
            ),
            sizes=(4, 5),
            dim=-1,
            index=1,  # J == I → diagonal
        )
        subst = match_pattern(DECAY_BLOCK_CROSS.lhs, term, {})
        assert subst is None or not DECAY_BLOCK_CROSS.check(subst)

    def test_upper_declines(self):
        """i < j is the provably-zero block — not this law."""
        term = Op.make(
            "split",
            Op.make(
                "split",
                Op.make("decay", _v("g"), t=self.T),
                sizes=(4, 5),
                dim=-2,
                index=0,
            ),
            sizes=(4, 5),
            dim=-1,
            index=1,  # J > I → strict upper
        )
        subst = match_pattern(DECAY_BLOCK_CROSS.lhs, term, {})
        assert subst is None or not DECAY_BLOCK_CROSS.check(subst)


class TestCrossDeclines:
    """Every guard branch declines honestly."""

    def _term(self, index_i=1, index_j=0, dr=-2, dc=-1):
        return Op.make(
            "split",
            Op.make(
                "split",
                Op.make("decay", _v("g"), t=9),
                sizes=(4, 5),
                dim=dr,
                index=index_i,
            ),
            sizes=(4, 5),
            dim=dc,
            index=index_j,
        )

    def test_no_attrs_declines(self):
        assert DECAY_BLOCK_CROSS.derive({"g": _v("g")}) is None
        assert not DECAY_BLOCK_CROSS.check({"g": _v("g")})

    def test_unshapeable_g_needs_raw_axes(self):
        subst = {
            "$attr:SZ": (4, 5),
            "$attr:T": 9,
            "$attr:I": 1,
            "$attr:J": 0,
            "$attr:DR": -2,
            "$attr:DC": -1,
        }
        # g unbound → shape unresolvable → raw -2/-1 axis check
        assert DECAY_BLOCK_CROSS.check(subst)
        assert not DECAY_BLOCK_CROSS.check({**subst, "$attr:DR": 0})

    def test_non_int_dims_decline(self):
        subst = {
            "$attr:SZ": (4, 5),
            "$attr:T": 9,
            "$attr:I": 1,
            "$attr:J": 0,
            "$attr:DR": "x",
            "$attr:DC": -1,
            "g": _v("g"),
        }
        assert not DECAY_BLOCK_CROSS.check(subst)

    def test_wrong_axes_decline(self):
        subst = {
            "$attr:SZ": (4, 5),
            "$attr:T": 9,
            "$attr:I": 1,
            "$attr:J": 0,
            "$attr:DR": -1,  # swapped axes
            "$attr:DC": -2,
            "g": _v("g", 3),
        }
        assert not DECAY_BLOCK_CROSS.check(subst)

    def test_unknown_g_shape_needs_raw_axes(self):
        subst = {
            "$attr:SZ": (4, 5),
            "$attr:T": 9,
            "$attr:I": 1,
            "$attr:J": 0,
            "$attr:DR": -2,
            "$attr:DC": -1,
            "g": _v("g"),  # scalar-shape → rank unknown
        }
        assert DECAY_BLOCK_CROSS.check(subst)

    def test_cross_unresolved_shapes_decline(self):
        bound = {
            "s": _v("s"),
            "xi": _v("xi"),
            "eta": _v("eta"),
            "v": _v("v"),
        }
        # unbound metas → unshapeable → decline
        assert not OM_DECAYED_CROSS.check({"s": _v("s", 3, 4)})
        # eta scalar-shape → len(es)==0 → decline (not IndexError)
        assert not OM_DECAYED_CROSS.check(bound)
        bound["eta"] = _v("eta", 4)
        bound["xi"] = _v("xi", 3)
        bound["s"] = _v("s", 3, 4)
        bound["v"] = _v("v", 4, 6)
        assert OM_DECAYED_CROSS.check(bound)

    def test_cross_unbroadcastable_declines(self):
        bound = {
            "s": _v("s", 3, 4),
            "xi": _v("xi", 4),
            "eta": _v("eta", 7),  # 7 ≠ s's col dim 4 → unbroadcastable
            "v": _v("v", 4, 6),
        }
        assert not OM_DECAYED_CROSS.check(bound)

    def test_cross_bad_contraction_declines(self):
        bound = {
            "s": _v("s", 3, 4),
            "xi": _v("xi", 3),
            "eta": _v("eta", 4),
            "v": _v("v", 5, 6),  # inner dim 5 ≠ s's cols 4
        }
        assert not OM_DECAYED_CROSS.check(bound)

    def test_cross_bad_xi_broadcast_declines(self):
        bound = {
            "s": _v("s", 3, 4),
            "xi": _v("xi", 7),  # 7 ≠ product's row dim 3
            "eta": _v("eta", 4),
            "v": _v("v", 4, 6),
        }
        assert not OM_DECAYED_CROSS.check(bound)


class TestDecayBlockCrossFlat:
    """The mixed-partition strict-lower block law.

    Recursive bisection partitions the two axes differently (the
    ``(16,8,8)``-rows against ``(16,16)``-cols split nests are the
    observed shape), so the strict-lower test is range arithmetic:
    ``sum(SZR[:I]) >= sum(SZC[:J+1])``.  On equal partitions that is
    exactly ``J < I`` — this law subsumes the strict sibling's
    sites it cannot see under a shared ``SZ`` metavar.
    """

    T = 32

    def _term(self, szr=(16, 8, 8), i=1, szc=(16, 16), j=0,
              dr=-2, dc=-1):
        return Op.make(
            "split",
            Op.make(
                "split",
                Op.make("decay", _v("g"), t=self.T),
                sizes=szr,
                dim=dr,
                index=i,
            ),
            sizes=szc,
            dim=dc,
            index=j,
        )

    def test_fires_and_derives_mixed(self):
        """Rows 16:24 x cols 0:16 — TI=8, TJ=16, OD=16."""
        subst = match_pattern(
            DECAY_BLOCK_CROSS_FLAT.lhs, self._term(), {}
        )
        assert subst is not None
        assert DECAY_BLOCK_CROSS_FLAT.check(subst)
        der = DECAY_BLOCK_CROSS_FLAT.derive(subst)
        assert der == {"$attr:TI": 8, "$attr:TJ": 16, "$attr:OD": 16}

    def test_equal_partitions_subsumed(self):
        """The strict sibling's site also matches the flat lhs."""
        subst = match_pattern(
            DECAY_BLOCK_CROSS_FLAT.lhs,
            self._term(szr=(16, 16), i=1, szc=(16, 16), j=0),
            {},
        )
        assert subst is not None
        assert DECAY_BLOCK_CROSS_FLAT.check(subst)
        der = DECAY_BLOCK_CROSS_FLAT.derive(subst)
        assert der == {"$attr:TI": 16, "$attr:TJ": 16, "$attr:OD": 16}

    def test_straddle_declines(self):
        """Rows 8:16 x cols 0:16 contains the diagonal — not rank-1."""
        subst = match_pattern(
            DECAY_BLOCK_CROSS_FLAT.lhs,
            self._term(szr=(8, 8, 16), i=1, szc=(16, 16), j=0),
            {},
        )
        assert subst is not None
        assert not DECAY_BLOCK_CROSS_FLAT.check(subst)

    def test_upper_declines(self):
        """Rows 0:8 x cols 16:32 is provably zero — not this law."""
        subst = match_pattern(
            DECAY_BLOCK_CROSS_FLAT.lhs,
            self._term(szr=(8, 8, 16), i=0, szc=(16, 16), j=1),
            {},
        )
        assert subst is not None
        assert not DECAY_BLOCK_CROSS_FLAT.check(subst)

    def test_wrong_axes_decline(self):
        subst = match_pattern(
            DECAY_BLOCK_CROSS_FLAT.lhs,
            self._term(dr=-1, dc=-2),
            {},
        )
        assert subst is not None
        assert not DECAY_BLOCK_CROSS_FLAT.check(subst)

    def test_non_int_dims_decline(self):
        """Non-int DR/DC attrs veto before the axis alignment check."""
        bound = {
            "g": _v("g", 8),
            "$attr:T": 8,
            "$attr:SZR": (4, 2, 2),
            "$attr:SZC": (4, 4),
            "$attr:DR": "x",
            "$attr:DC": -1,
            "$attr:I": 1,
            "$attr:J": 0,
        }
        assert not DECAY_BLOCK_CROSS_FLAT.check(bound)

    def test_unshaped_decay_falls_back_to_axis_defaults(self):
        """An unshapeable ``g`` can't read its rank — the check falls
        back to the canonical ``-2``/``-1`` split axes."""
        bound = {
            # no "g" — _vshape declines to a shape
            "$attr:T": 8,
            "$attr:SZR": (4, 2, 2),
            "$attr:SZC": (4, 4),
            "$attr:I": 1,
            "$attr:J": 0,
        }
        assert DECAY_BLOCK_CROSS_FLAT.check(
            {**bound, "$attr:DR": -2, "$attr:DC": -1}
        )
        assert not DECAY_BLOCK_CROSS_FLAT.check(
            {**bound, "$attr:DR": -2, "$attr:DC": 0}
        )

    def test_derive_none_without_partition(self):
        """No resolvable partitions → nothing for derive to mint."""
        assert DECAY_BLOCK_CROSS_FLAT.derive({}) is None

    def test_numerically_equals_sliced_block(self):
        """The mixed-partition factored block IS the slice."""
        pos = torch.arange(self.T, dtype=torch.float64)
        D = torch.tril(
            G ** (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0)
        )
        block = D[16:24, 0:16]  # rows 16:24 x cols 0:16
        rhs_term = instantiate_pattern(
            DECAY_BLOCK_CROSS_FLAT.rhs,
            {
                "g": _v("g"),
                "$attr:TI": 8,
                "$attr:TJ": 16,
                "$attr:OD": 16,
            },
        )
        factored = eval_term(
            rhs_term, var_env={"g": G}, bindings=tb._IR_TO_TORCH
        )
        assert torch.allclose(block, factored, atol=1e-12)


class TestSplitFlatten:
    """``split(split(t, SZ, D, I), SZ2, D, J)`` → the flat block.

    Recursive bisection leaves nested same-axis splits; flattening
    joins the partition tuples so a flat inner split can stand in
    for the nested pair — the member the block laws match.
    """

    def _term(self, sz=(16, 16), i=1, sz2=(8, 8), j=1, d=-2, d2=-2):
        return Op.make(
            "split",
            Op.make(
                "split", _v("t", 4, 32, 8), sizes=sz, dim=d, index=i
            ),
            sizes=sz2,
            dim=d2,
            index=j,
        )

    def test_fires_and_derives(self):
        subst = match_pattern(SPLIT_FLATTEN.lhs, self._term(), {})
        assert subst is not None
        assert SPLIT_FLATTEN.check(subst)
        der = SPLIT_FLATTEN.derive(subst)
        assert der == {"$attr:SZF": (16, 8, 8), "$attr:IF": 2}

    def test_rhs_is_the_flat_split(self):
        subst = match_pattern(SPLIT_FLATTEN.lhs, self._term(), {})
        subst.update(SPLIT_FLATTEN.derive(subst))
        out = instantiate_pattern(SPLIT_FLATTEN.rhs, subst)
        r = op_repr(out)
        assert "sizes=(16, 8, 8)" in r and "index=2" in r

    def test_exact_slice(self):
        """The flat member IS the nested slice."""
        subst = match_pattern(SPLIT_FLATTEN.lhs, self._term(), {})
        subst.update(SPLIT_FLATTEN.derive(subst))
        out = instantiate_pattern(SPLIT_FLATTEN.rhs, subst)
        env = {"t": torch.randn(4, 32, 8, dtype=torch.float64)}
        a = eval_term(
            self._term(), var_env=env, bindings=tb._IR_TO_TORCH
        )
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.equal(a, b)

    def test_bad_partition_declines(self):
        """SZ2 not a partition of SZ[I] → no flat form."""
        subst = match_pattern(
            SPLIT_FLATTEN.lhs, self._term(sz2=(6, 9)), {}
        )
        assert subst is not None
        assert not SPLIT_FLATTEN.check(subst)
        # and the failed check leaves derive nothing to mint
        assert SPLIT_FLATTEN.derive(subst) is None

    def test_derive_none_without_attrs(self):
        assert SPLIT_FLATTEN.derive({}) is None

    def test_mismatched_dims_decline(self):
        subst = match_pattern(
            SPLIT_FLATTEN.lhs, self._term(d2=-1), {}
        )
        assert subst is None or not SPLIT_FLATTEN.check(subst)

    def test_out_of_range_declines(self):
        subst = match_pattern(
            SPLIT_FLATTEN.lhs, self._term(j=3), {}
        )
        assert subst is not None
        assert not SPLIT_FLATTEN.check(subst)


class TestStateWeightLaws:
    """The contraction-constant scalar bridges around the state.

    ``(a⊙b)⊙(q@B) = b⊙(q@(a⊙B))`` (sink) and its inverse
    ``q@(a⊙B) = a⊙(q@B)`` (exit) — exact only when ``a`` is
    constant along ``B``'s contracted axis, the same condition
    ``matmul_scalar_factor`` enforces.
    """

    def test_sink_fires_scalar(self):
        lhs = Op.make(
            "mul",
            Op.make("mul", _v("a"), _v("b", 5, 1)),
            Op.make("matmul", _v("q", 5, 3), _v("B", 3, 6)),
        )
        subst = match_pattern(DECAY_STATE_WEIGHT_SINK.lhs, lhs, {})
        assert subst is not None
        assert DECAY_STATE_WEIGHT_SINK.check(subst)
        out = instantiate_pattern(DECAY_STATE_WEIGHT_SINK.rhs, subst)
        env = {
            "a": torch.tensor(0.81, dtype=torch.float64),
            "b": torch.randn(5, 1, dtype=torch.float64),
            "q": torch.randn(5, 3, dtype=torch.float64),
            "B": torch.randn(3, 6, dtype=torch.float64),
        }
        a_ = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b_ = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a_, b_)

    def test_sink_declines_row_varying(self):
        """``a`` varying along B's contracted rows is not exact."""
        bound = {
            "a": _v("a", 3, 1),
            "b": _v("b", 5, 1),
            "q": _v("q", 5, 3),
            "B": _v("B", 3, 6),
        }
        assert not DECAY_STATE_WEIGHT_SINK.check(bound)

    def test_sink_declines_unbroadcastable_residual(self):
        bound = {
            "a": _v("a"),
            "b": _v("b", 7),
            "q": _v("q", 5, 3),
            "B": _v("B", 3, 6),
        }
        assert not DECAY_STATE_WEIGHT_SINK.check(bound)

    def test_sink_declines_unshapeable_factor(self):
        """An unbound (shapeless) factor can't prove contraction
        constancy — the guard declines rather than guess."""
        bound = {
            # no "a" — _vshape declines to a shape
            "b": _v("b", 5, 1),
            "q": _v("q", 5, 3),
            "B": _v("B", 3, 6),
        }
        assert not DECAY_STATE_WEIGHT_SINK.check(bound)

    def test_exit_fires_scalar(self):
        lhs = Op.make(
            "matmul",
            _v("q", 5, 3),
            Op.make("mul", _v("a"), _v("B", 3, 6)),
        )
        subst = match_pattern(DECAY_STATE_WEIGHT_EXIT.lhs, lhs, {})
        assert subst is not None
        assert DECAY_STATE_WEIGHT_EXIT.check(subst)
        out = instantiate_pattern(DECAY_STATE_WEIGHT_EXIT.rhs, subst)
        env = {
            "a": torch.tensor(0.81, dtype=torch.float64),
            "q": torch.randn(5, 3, dtype=torch.float64),
            "B": torch.randn(3, 6, dtype=torch.float64),
        }
        a_ = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b_ = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a_, b_)

    def test_exit_declines_row_varying(self):
        bound = {
            "a": _v("a", 3, 1),
            "q": _v("q", 5, 3),
            "B": _v("B", 3, 6),
        }
        assert not DECAY_STATE_WEIGHT_EXIT.check(bound)

    def test_exit_declines_bad_contraction(self):
        bound = {
            "a": _v("a"),
            "q": _v("q", 5, 5),
            "B": _v("B", 3, 6),
        }
        assert not DECAY_STATE_WEIGHT_EXIT.check(bound)

    def test_exit_declines_unshapeable_factor(self):
        bound = {
            # no "a" — _vshape declines to a shape
            "q": _v("q", 5, 3),
            "B": _v("B", 3, 6),
        }
        assert not DECAY_STATE_WEIGHT_EXIT.check(bound)


class TestOmDecayedCross:
    def _lhs(self):
        return Op.make(
            "om_decayed",
            _v("s", 5, 4),
            Op.make(
                "mul",
                Op.make("unsqueeze", _v("xi", 5), dim=-1),
                Op.make("unsqueeze", _v("eta", 4), dim=-2),
            ),
            _v("v", 4, 6),
        )

    def test_fires(self):
        subst = match_pattern(OM_DECAYED_CROSS.lhs, self._lhs(), {})
        assert subst is not None and OM_DECAYED_CROSS.check(subst)

    def test_exact_distribution(self):
        """(s ⊙ xietaᵀ) @ v = xi_row ⊙ ((s ⊙ eta_col) @ v) — exact."""
        subst = match_pattern(OM_DECAYED_CROSS.lhs, self._lhs(), {})
        out = instantiate_pattern(OM_DECAYED_CROSS.rhs, subst)
        xi = torch.tensor(
            [0.9**6, 0.9**7, 0.9**8, 0.9**9, 0.9**10],
            dtype=torch.float64,
        )
        eta = torch.tensor(
            [1.0, 0.9**-1, 0.9**-2, 0.9**-3], dtype=torch.float64
        )
        s = torch.randn(5, 4, dtype=torch.float64)
        v = torch.randn(4, 6, dtype=torch.float64)
        env = {"s": s, "xi": xi, "eta": eta, "v": v}
        a = eval_term(
            self._lhs(), var_env=env, bindings=tb._IR_TO_TORCH
        )
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b, atol=1e-12)

    def test_scalar_operands_decline(self):
        term = Op.make(
            "om_decayed",
            _v("s"),
            Op.make(
                "mul",
                Op.make("unsqueeze", _v("xi"), dim=-1),
                Op.make("unsqueeze", _v("eta"), dim=-2),
            ),
            _v("v"),
        )
        subst = match_pattern(OM_DECAYED_CROSS.lhs, term, {})
        assert subst is None or not OM_DECAYED_CROSS.check(subst)


class TestOmDecayedCrossGen:
    """The generalized cross law — ``x`` absorbs residual row factors.

    On the real ChunkedRetention export the block decay arrives as a
    deeper unsqueeze-product chain (every residual factor a row
    broadcaster), not the strict two-factor spelling — the metavar
    absorbs the whole row-side product and the xi/eta split still
    happens.
    """

    def _lhs(self, x=None, ue=-2):
        if x is None:
            x = Op.make(
                "mul",
                Op.make("unsqueeze", _v("a", 5), dim=-1),
                Op.make("unsqueeze", _v("b", 5), dim=-1),
            )
        return Op.make(
            "om_decayed",
            _v("s", 5, 4),
            Op.make(
                "mul",
                x,
                Op.make("unsqueeze", _v("eta", 4), dim=ue),
            ),
            _v("v", 4, 6),
        )

    def test_fires_on_deep_row_chain(self):
        subst = match_pattern(OM_DECAYED_CROSS_GEN.lhs, self._lhs(), {})
        assert subst is not None and OM_DECAYED_CROSS_GEN.check(subst)

    def test_exact_distribution(self):
        """(s ⊙ x·etaᵀ) @ v = x ⊙ ((s ⊙ etaᵀ) @ v) — exact at fp64."""
        lhs = self._lhs()
        subst = match_pattern(OM_DECAYED_CROSS_GEN.lhs, lhs, {})
        out = instantiate_pattern(OM_DECAYED_CROSS_GEN.rhs, subst)
        env = {
            "s": torch.randn(5, 4, dtype=torch.float64),
            "a": torch.tensor(
                [0.9**k for k in range(6, 11)], dtype=torch.float64
            ),
            "b": torch.tensor(
                [0.9**k for k in range(5)], dtype=torch.float64
            ),
            "eta": torch.tensor(
                [1.0, 0.9**-1, 0.9**-2, 0.9**-3],
                dtype=torch.float64,
            ),
            "v": torch.randn(4, 6, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b, atol=1e-12)

    def test_subsumes_strict_spelling(self):
        """x = unsq(xi, -1) is the strict law's site, verbatim."""
        lhs = self._lhs(x=Op.make("unsqueeze", _v("xi", 5), dim=-1))
        subst = match_pattern(OM_DECAYED_CROSS_GEN.lhs, lhs, {})
        assert subst is not None
        assert OM_DECAYED_CROSS_GEN.check(subst)
        out = instantiate_pattern(OM_DECAYED_CROSS_GEN.rhs, subst)
        env = {
            "s": torch.randn(5, 4, dtype=torch.float64),
            "xi": torch.tensor(
                [0.9**k for k in range(6, 11)], dtype=torch.float64
            ),
            "eta": torch.tensor(
                [1.0, 0.9**-1, 0.9**-2, 0.9**-3],
                dtype=torch.float64,
            ),
            "v": torch.randn(4, 6, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b, atol=1e-12)

    def test_row_factor_eta_stays_inside(self):
        """UE = -1: a row-side eta stays inside the contraction."""
        lhs = Op.make(
            "om_decayed",
            _v("s", 5, 4),
            Op.make(
                "mul",
                _v("x", 5, 1),
                Op.make("unsqueeze", _v("eta", 5), dim=-1),
            ),
            _v("v", 4, 6),
        )
        subst = match_pattern(OM_DECAYED_CROSS_GEN.lhs, lhs, {})
        assert subst is not None
        assert OM_DECAYED_CROSS_GEN.check(subst)
        out = instantiate_pattern(OM_DECAYED_CROSS_GEN.rhs, subst)
        env = {
            "s": torch.randn(5, 4, dtype=torch.float64),
            "x": torch.randn(5, 1, dtype=torch.float64),
            "eta": torch.tensor(
                [0.9**k for k in range(5)], dtype=torch.float64
            ),
            "v": torch.randn(4, 6, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b, atol=1e-12)

    def test_batched_fires(self):
        lhs = Op.make(
            "om_decayed",
            _v("s", 2, 5, 4),
            Op.make(
                "mul",
                _v("x", 2, 5, 1),
                Op.make("unsqueeze", _v("eta", 4), dim=-2),
            ),
            _v("v", 2, 4, 6),
        )
        subst = match_pattern(OM_DECAYED_CROSS_GEN.lhs, lhs, {})
        assert subst is not None and OM_DECAYED_CROSS_GEN.check(subst)


class TestOmDecayedCrossGenDeclines:
    """Every guard branch of the generalized law declines honestly."""

    def test_unbound_metas_decline(self):
        assert not OM_DECAYED_CROSS_GEN.check({})
        assert not OM_DECAYED_CROSS_GEN.check(
            {"s": _v("s", 5, 4), "x": _v("x", 5, 1)}
        )

    def test_column_varying_x_declines(self):
        """x varying along columns CANNOT factor out — the crux."""
        assert not OM_DECAYED_CROSS_GEN.check(
            {
                "s": _v("s", 5, 4),
                "x": _v("x", 5, 4),  # last dim 4: not row-constant
                "eta": _v("eta", 4),
                "v": _v("v", 4, 6),
                "$attr:UE": -2,
            }
        )
        # a bare length-4 vector varies along the contracted axis —
        # a column-role factor that must stay inside the matmul
        assert not OM_DECAYED_CROSS_GEN.check(
            {
                "s": _v("s", 5, 4),
                "x": _v("x", 4),
                "eta": _v("eta", 4),
                "v": _v("v", 4, 6),
                "$attr:UE": -2,
            }
        )

    def test_scalar_x_fires(self):
        """A scalar x is constant along every axis — distributes."""
        assert OM_DECAYED_CROSS_GEN.check(
            {
                "s": _v("s", 5, 4),
                "x": _v("x"),
                "eta": _v("eta", 4),
                "v": _v("v", 4, 6),
                "$attr:UE": -2,
            }
        )

    def test_unbroadcastable_eta_declines(self):
        assert not OM_DECAYED_CROSS_GEN.check(
            {
                "s": _v("s", 5, 4),
                "x": _v("x", 5, 1),
                "eta": _v("eta", 7),  # 7 ≠ s's col dim 4
                "v": _v("v", 4, 6),
                "$attr:UE": -2,
            }
        )

    def test_unbroadcastable_x_declines(self):
        assert not OM_DECAYED_CROSS_GEN.check(
            {
                "s": _v("s", 5, 4),
                "x": _v("x", 7, 1),  # 7 ≠ product's row dim 5
                "eta": _v("eta", 4),
                "v": _v("v", 4, 6),
                "$attr:UE": -2,
            }
        )

    def test_bad_contraction_declines(self):
        assert not OM_DECAYED_CROSS_GEN.check(
            {
                "s": _v("s", 5, 4),
                "x": _v("x", 5, 1),
                "eta": _v("eta", 4),
                "v": _v("v", 5, 6),  # inner dim 5 ≠ s's cols 4
                "$attr:UE": -2,
            }
        )

    def test_scalar_s_declines(self):
        assert not OM_DECAYED_CROSS_GEN.check(
            {
                "s": _v("s"),
                "x": _v("x"),
                "eta": _v("eta"),
                "v": _v("v"),
                "$attr:UE": -2,
            }
        )

    def test_non_int_ue_declines(self):
        assert not OM_DECAYED_CROSS_GEN.check(
            {
                "s": _v("s", 5, 4),
                "x": _v("x", 5, 1),
                "eta": _v("eta", 4),
                "v": _v("v", 4, 6),
                "$attr:UE": "x",
            }
        )


class TestComposition:
    """The factored cross appears inside a saturated chunked site."""

    def test_decay_vec_enodes_in_saturated_graph(self):
        s1 = _v("s1", 4, 9)
        s2 = _v("s2", 5, 9)
        v1 = _v("v1", 4, 4)
        v2 = _v("v2", 5, 4)
        site = Op.make(
            "om_decayed",
            Op.make("concat", s1, s2, dim=-2),
            Op.make("decay", _v("g"), t=9),
            Op.make("concat", v1, v2, dim=-2),
        )
        eg = EGraph()
        root = eg.add_term(site)
        eg.run(
            list(OM_LAWS) + list(OM_CHUNKIFY),
            root,
            max_iterations=8,
            rule_budgets={
                "om_decayed_chunkify_rows": 4,
                "om_decayed_chunkify_keys": 4,
            },
        )
        dv = [
            n
            for cls in eg._classes.values()
            for n in cls.nodes
            if n.op == "decay_vec"
        ]
        # the cross block's two factors — xi (offset>0) and eta (sign=-1)
        assert any(dict(n.attrs).get("sign") == -1 for n in dv), (
            "the eta factor (g^{-c}) should appear"
        )
        assert any(dict(n.attrs).get("offset", 0) > 0 for n in dv), (
            "the xi factor (g^{o+r}) should appear"
        )


class TestStateFoldLaws:
    """The four axioms that expose the inter-chunk fold."""

    def test_decay_vec_offset_splits(self):
        t = Op.make("decay_vec", _v("g"), t=5, offset=4)
        subst = match_pattern(DECAY_VEC_OFFSET.lhs, t, {})
        assert subst is not None and DECAY_VEC_OFFSET.check(subst)
        out = instantiate_pattern(DECAY_VEC_OFFSET.rhs, subst)
        env = {"g": G}
        a = eval_term(t, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)

    def test_decay_vec_offset_zero_declines(self):
        t = Op.make("decay_vec", _v("g"), t=5, offset=0)
        subst = match_pattern(DECAY_VEC_OFFSET.lhs, t, {})
        assert subst is None or not DECAY_VEC_OFFSET.check(subst)

    def test_matmul_col_decay_fires(self):
        lhs = Op.make(
            "mul",
            Op.make(
                "matmul",
                _v("q", 5, 3),
                Op.make(
                    "transpose",
                    _v("k", 4, 3),
                    dim0=-2,
                    dim1=-1,
                ),
            ),
            Op.make("unsqueeze", _v("eta", 4), dim=-2),
        )
        subst = match_pattern(MATMUL_COL_DECAY.lhs, lhs, {})
        assert subst is not None and MATMUL_COL_DECAY.check(subst)
        out = instantiate_pattern(MATMUL_COL_DECAY.rhs, subst)
        q = torch.randn(5, 3, dtype=torch.float64)
        k = torch.randn(4, 3, dtype=torch.float64)
        eta = torch.tensor(
            [1.0, 0.9**-1, 0.9**-2, 0.9**-3],
            dtype=torch.float64,
        )
        env = {"q": q, "k": k, "eta": eta}
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)

    def test_matmul_col_decay_declines(self):
        # eta 7-long vs k's 4 rows → unbroadcastable
        bound = {
            "q": _v("q", 5, 3),
            "k": _v("k", 4, 3),
            "eta": _v("eta", 7),
        }
        assert not MATMUL_COL_DECAY.check(bound)
        # k rank-1 → no matrix axes to scale
        assert not MATMUL_COL_DECAY.check(
            {"q": _v("q", 5, 3), "k": _v("k", 4), "eta": _v("eta", 4)}
        )
        # the inner contraction must type — q's K dim vs k's K dim
        assert not MATMUL_COL_DECAY.check(
            {
                "q": _v("q", 5, 5),
                "k": _v("k", 4, 3),
                "eta": _v("eta", 4),
            }
        )
        # the scaled product must broadcast — eta vs q's col span
        assert not MATMUL_COL_DECAY.check(
            {
                "q": _v("q", 5, 3),
                "k": _v("k", 4, 3),
                "eta": _v("eta", 2),
            }
        )
        # unbound metas → unshapeable → decline
        assert not MATMUL_COL_DECAY.check({})

    def test_matmul_scalar_factor_fires(self):
        lhs = Op.make(
            "mul",
            _v("c"),
            Op.make("matmul", _v("q", 5, 3), _v("b", 3, 6)),
        )
        subst = match_pattern(MATMUL_SCALAR_FACTOR.lhs, lhs, {})
        assert subst is not None and MATMUL_SCALAR_FACTOR.check(subst)
        out = instantiate_pattern(MATMUL_SCALAR_FACTOR.rhs, subst)
        env = {
            "c": torch.tensor(0.7, dtype=torch.float64),
            "q": torch.randn(5, 3, dtype=torch.float64),
            "b": torch.randn(3, 6, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)

    def test_matmul_scalar_factor_declines(self):
        # unbroadcastable scalar
        assert not MATMUL_SCALAR_FACTOR.check(
            {
                "c": _v("c", 7),
                "q": _v("q", 5, 3),
                "b": _v("b", 3, 6),
            }
        )
        # right operand rank-1 → not a matrix contraction
        assert not MATMUL_SCALAR_FACTOR.check(
            {"c": _v("c"), "q": _v("q", 5, 3), "b": _v("b", 6)}
        )
        # contraction itself must type
        assert not MATMUL_SCALAR_FACTOR.check(
            {"c": _v("c"), "q": _v("q", 5, 5), "b": _v("b", 3, 6)}
        )
        # c broadcasts to b but not to the taller product rows
        assert not MATMUL_SCALAR_FACTOR.check(
            {
                "c": _v("c", 3, 6),
                "q": _v("q", 5, 3),
                "b": _v("b", 3, 6),
            }
        )
        # c TALLER than b: broadcasts against both b and the product,
        # but a rank-3 scale over a rank-2 contraction is not a
        # scalar factor — the len(broadcast) > len(b) arm declines.
        assert not MATMUL_SCALAR_FACTOR.check(
            {
                "c": _v("c", 1, 5, 1),
                "q": _v("q", 5, 5),
                "b": _v("b", 5, 5),
            }
        )
        # c varying along b's contracted rows — the unsound corner:
        # c[r]⊙(q@B) ≠ q@(c[k]⊙B).  Square operands broadcast fine;
        # the contraction-constancy arm is what declines.
        assert not MATMUL_SCALAR_FACTOR.check(
            {
                "c": _v("c", 5, 1),
                "q": _v("q", 5, 5),
                "b": _v("b", 5, 5),
            }
        )
        assert not MATMUL_SCALAR_FACTOR.check({})

    def test_matmul_scalar_factor_row_varying_is_wrong(self):
        """The declined corner really is wrong — not a missed fire."""
        lhs = Op.make(
            "mul",
            _v("c", 5, 1),
            Op.make("matmul", _v("q", 5, 5), _v("b", 5, 5)),
        )
        out = Op.make(
            "matmul",
            _v("q", 5, 5),
            Op.make("mul", _v("c", 5, 1), _v("b", 5, 5)),
        )
        env = {
            "c": torch.randn(5, 1, dtype=torch.float64),
            "q": torch.randn(5, 5, dtype=torch.float64),
            "b": torch.randn(5, 5, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert not torch.allclose(a, b)
        # per-column c IS exact — the boundary the guard draws
        env["c"] = torch.randn(1, 5, dtype=torch.float64)
        lhs2 = Op.make(
            "mul",
            _v("c", 1, 5),
            Op.make("matmul", _v("q", 5, 5), _v("b", 5, 5)),
        )
        a2 = eval_term(lhs2, var_env=env, bindings=tb._IR_TO_TORCH)
        b2 = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a2, b2)

    def test_unsq_mul_fires(self):
        lhs = Op.make(
            "unsqueeze",
            Op.make("mul", _v("a", 4), _v("c")),
            dim=-1,
        )
        subst = match_pattern(UNSQ_MUL.lhs, lhs, {})
        assert subst is not None and UNSQ_MUL.check(subst)
        out = instantiate_pattern(UNSQ_MUL.rhs, subst)
        env = {
            "a": torch.randn(4, dtype=torch.float64),
            "c": torch.tensor(0.7, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)

    def test_unsq_mul_declines(self):
        # dim outside the (rank+1) window for a 1-D operand
        lhs = Op.make(
            "unsqueeze",
            Op.make("mul", _v("a", 4), _v("b", 4)),
            dim=3,
        )
        subst = match_pattern(UNSQ_MUL.lhs, lhs, {})
        assert subst is None or not UNSQ_MUL.check(subst)
        # non-int dim declines
        assert not UNSQ_MUL.check({"$attr:UD": "x"})
        # scalar-shape operands still pass the window at -1
        assert UNSQ_MUL.check(
            {"$attr:UD": -1, "a": _v("a"), "b": _v("b")}
        )
        # a provably-ill-typed operand declines
        bad = Op.make("decay_vec", _v("g"), t=-1)
        assert not UNSQ_MUL.check(
            {"$attr:UD": -1, "a": bad, "b": _v("b")}
        )
        # an unbound operand is unshapeable → skips the window
        assert UNSQ_MUL.check({"$attr:UD": -1, "a": _v("a", 4)})

    def test_decay_state_like(self):
        from mlopt_carriers.om import _decay_state_like

        # matrix state + scalar weight → lifts
        assert _decay_state_like({"h": _v("h", 3, 6), "a": _v("a")})
        # vector state → declines (the fold carries matrices)
        assert not _decay_state_like({"h": _v("h", 4), "a": _v("a")})
        # unshapeable weight → allowed (conservative)
        assert _decay_state_like({"h": _v("h", 3, 6), "a": None})
        # ill-typed weight → declines
        bad = Op.make("decay_vec", _v("g"), t=-1)
        assert not _decay_state_like({"h": _v("h", 3, 6), "a": bad})
        # non-broadcastable weight → declines
        assert not _decay_state_like(
            {"h": _v("h", 3, 6), "a": _v("a", 7)}
        )


class TestScoreSplit:
    """The score-block decomposition laws — the chain's entry point."""

    def _site(self):
        s = _v("s", 8, 8)
        d = _v("d", 8, 8)
        v = _v("v", 8, 4)
        return Op.make("om_decayed", s, d, v)

    def test_chunkify_rows_fires(self):
        subst = match_pattern(
            OM_DECAYED_CHUNKIFY_ROWS.lhs, self._site(), {}
        )
        assert subst is not None
        assert OM_DECAYED_CHUNKIFY_ROWS.check(subst)
        der = OM_DECAYED_CHUNKIFY_ROWS.derive(subst)
        assert der == {"$attr:SZ": (4, 4)}
        out = instantiate_pattern(
            OM_DECAYED_CHUNKIFY_ROWS.rhs, {**subst, **der}
        )
        assert "split" in op_repr(out) and "concat" in op_repr(out)

    def test_chunkify_rows_verifies(self):
        """Row-bisected blocks ≡ the fused decayed product."""
        T, DV = 8, 4
        s = torch.randn(T, T, dtype=torch.float64)
        d = torch.tril(
            torch.randn(T, T, dtype=torch.float64).abs() + 0.1
        )
        v = torch.randn(T, DV, dtype=torch.float64)
        env = {"s": s, "d": d, "v": v}
        fused = eval_term(
            Op.make("om_decayed", _v("s"), _v("d"), _v("v")),
            var_env=env,
            bindings=tb._IR_TO_TORCH,
        )
        subst = match_pattern(
            OM_DECAYED_CHUNKIFY_ROWS.lhs,
            Op.make(
                "om_decayed",
                _v("s", 8, 8),
                _v("d", 8, 8),
                _v("v", 8, 4),
            ),
            {},
        )
        subst.update(OM_DECAYED_CHUNKIFY_ROWS.derive(subst))
        rhs = instantiate_pattern(OM_DECAYED_CHUNKIFY_ROWS.rhs, subst)
        chunked = eval_term(rhs, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(fused, chunked)

    def test_chunkify_rows_declines(self):
        # s not ≥2-D / T<2 / mismatched decay extent
        assert not OM_DECAYED_CHUNKIFY_ROWS.check(
            {"s": _v("s", 8), "d": _v("d", 8, 8)}
        )
        assert not OM_DECAYED_CHUNKIFY_ROWS.check(
            {"s": _v("s", 1, 8), "d": _v("d", 1, 8)}
        )
        # rank-1 decay — no row axis to slice
        assert not OM_DECAYED_CHUNKIFY_ROWS.check(
            {"s": _v("s", 8, 8), "d": _v("d", 8)}
        )
        assert not OM_DECAYED_CHUNKIFY_ROWS.check(
            {"s": _v("s", 8, 8), "d": _v("d", 7, 7)}
        )

    def test_chunkify_keys_fires(self):
        subst = match_pattern(
            OM_DECAYED_CHUNKIFY_KEYS.lhs, self._site(), {}
        )
        assert subst is not None
        assert OM_DECAYED_CHUNKIFY_KEYS.check(subst)
        assert OM_DECAYED_CHUNKIFY_KEYS.derive(subst) == {
            "$attr:SZ": (4, 4)
        }

    def test_chunkify_keys_verifies(self):
        """Key-bisected blocks sum to the fused decayed product."""
        T, DV = 8, 4
        s = torch.randn(T, T, dtype=torch.float64)
        d = torch.tril(
            torch.randn(T, T, dtype=torch.float64).abs() + 0.1
        )
        v = torch.randn(T, DV, dtype=torch.float64)
        env = {"s": s, "d": d, "v": v}
        fused = eval_term(
            Op.make("om_decayed", _v("s"), _v("d"), _v("v")),
            var_env=env,
            bindings=tb._IR_TO_TORCH,
        )
        subst = match_pattern(
            OM_DECAYED_CHUNKIFY_KEYS.lhs,
            Op.make(
                "om_decayed",
                _v("s", 8, 8),
                _v("d", 8, 8),
                _v("v", 8, 4),
            ),
            {},
        )
        subst.update(OM_DECAYED_CHUNKIFY_KEYS.derive(subst))
        rhs = instantiate_pattern(OM_DECAYED_CHUNKIFY_KEYS.rhs, subst)
        chunked = eval_term(rhs, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(fused, chunked)

    def test_chunkify_keys_declines(self):
        # key-extent mismatch — decay or v doesn't cover the axis
        assert not OM_DECAYED_CHUNKIFY_KEYS.check(
            {
                "s": _v("s", 8, 8),
                "d": _v("d", 8, 7),
                "v": _v("v", 8, 4),
            }
        )
        assert not OM_DECAYED_CHUNKIFY_KEYS.check(
            {
                "s": _v("s", 8, 8),
                "d": _v("d", 8, 8),
                "v": _v("v", 7, 4),
            }
        )
        # unshapeable → decline
        assert not OM_DECAYED_CHUNKIFY_KEYS.check({})
        # s rank < 2 → no key axis to bisect
        assert not OM_DECAYED_CHUNKIFY_KEYS.check(
            {"s": _v("s", 8), "d": _v("d", 8, 8), "v": _v("v", 8, 4)}
        )
        # single-key extent — nothing to bisect
        assert not OM_DECAYED_CHUNKIFY_KEYS.check(
            {"s": _v("s", 8, 1), "d": _v("d", 8, 1), "v": _v("v", 1, 4)}
        )

    def test_chunkify_keys_derives(self):
        subst = match_pattern(
            OM_DECAYED_CHUNKIFY_KEYS.lhs, self._site(), {}
        )
        assert OM_DECAYED_CHUNKIFY_KEYS.derive(subst) == {
            "$attr:SZ": (4, 4)
        }
        # rank-1 score → no key extent → derive declines
        assert (
            OM_DECAYED_CHUNKIFY_KEYS.derive({"s": _v("s", 8)}) is None
        )
        # single-key extent — nothing to bisect
        assert (
            OM_DECAYED_CHUNKIFY_KEYS.derive({"s": _v("s", 8, 1)})
            is None
        )

    def test_split_matmul_rows_fires(self):
        lhs = Op.make(
            "split",
            Op.make("matmul", _v("q", 8, 4), _v("k", 4, 8)),
            sizes=(4, 4),
            dim=-2,
            index=1,
        )
        subst = match_pattern(SPLIT_MATMUL_ROWS.lhs, lhs, {})
        assert subst is not None and SPLIT_MATMUL_ROWS.check(subst)
        out = instantiate_pattern(SPLIT_MATMUL_ROWS.rhs, subst)
        q = torch.randn(8, 4, dtype=torch.float64)
        kt = torch.randn(4, 8, dtype=torch.float64)
        env = {"q": q, "k": kt}
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)

    def test_split_matmul_cols_fires(self):
        lhs = Op.make(
            "split",
            Op.make("matmul", _v("q", 8, 4), _v("k", 4, 8)),
            sizes=(4, 4),
            dim=-1,
            index=1,
        )
        subst = match_pattern(SPLIT_MATMUL_COLS.lhs, lhs, {})
        assert subst is not None and SPLIT_MATMUL_COLS.check(subst)
        out = instantiate_pattern(SPLIT_MATMUL_COLS.rhs, subst)
        q = torch.randn(8, 4, dtype=torch.float64)
        kt = torch.randn(4, 8, dtype=torch.float64)
        env = {"q": q, "k": kt}
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)

    def test_split_matmul_declines(self):
        # index out of range for the partition
        assert not SPLIT_MATMUL_ROWS.check(
            {"$attr:SZ": (4, 4), "$attr:I": 2}
        )
        # non-positive size
        assert not SPLIT_MATMUL_ROWS.check(
            {"$attr:SZ": (4, 0), "$attr:I": 0}
        )
        assert not SPLIT_MATMUL_ROWS.check({})

    def test_split_transpose_fires(self):
        # split kᵀ's col axis = split k's row axis, transposed
        lhs = Op.make(
            "split",
            Op.make("transpose", _v("k", 8, 4), dim0=-2, dim1=-1),
            sizes=(4, 4),
            dim=-1,
            index=1,
        )
        subst = match_pattern(SPLIT_TRANSPOSE.lhs, lhs, {})
        assert subst is not None and SPLIT_TRANSPOSE.check(subst)
        der = SPLIT_TRANSPOSE.derive(subst)
        # splitting the transposed col (-1) reads k's row axis (-2)
        assert der == {"$attr:DP": -2}
        out = instantiate_pattern(SPLIT_TRANSPOSE.rhs, {**subst, **der})
        k = torch.randn(8, 4, dtype=torch.float64)
        env = {"k": k}
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)

    def test_split_transpose_declines(self):
        # non-int split dim → no derived preimage
        assert not SPLIT_TRANSPOSE.check(
            {
                "$attr:SZ": (4, 4),
                "$attr:I": 0,
                "$attr:SD": "x",
                "$attr:D0": -2,
                "$attr:D1": -1,
            }
        )
        assert not SPLIT_TRANSPOSE.check({})

    def test_split_scalar_pointwise(self):
        """split_D(mul(x,c)) = mul(split_D(x),c) — c scalar."""
        from mlopt_core.ir import Const
        from mlopt_core.laws.tensor import SPLIT_SCALAR_RULES

        r = next(
            r for r in SPLIT_SCALAR_RULES if r.name == "split_scalar_div_r"
        )
        lhs = Op.make(
            "split",
            Op.make("div", _v("m", 8, 8), Const(2.83)),
            sizes=(4, 4),
            dim=-2,
            index=1,
        )
        subst = match_pattern(r.lhs, lhs, {})
        assert subst is not None and r.check(subst)
        out = instantiate_pattern(r.rhs, subst)
        m = torch.randn(8, 8, dtype=torch.float64)
        a = eval_term(lhs, var_env={"m": m}, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env={"m": m}, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)
        # left variant — scalar on the left of a non-commutative op
        rl = next(
            r for r in SPLIT_SCALAR_RULES if r.name == "split_scalar_sub_l"
        )
        lhs2 = Op.make(
            "split",
            Op.make("sub", Const(1.5), _v("m", 8, 8)),
            sizes=(4, 4),
            dim=-1,
            index=0,
        )
        subst2 = match_pattern(rl.lhs, lhs2, {})
        assert subst2 is not None and rl.check(subst2)
        out2 = instantiate_pattern(rl.rhs, subst2)
        a2 = eval_term(lhs2, var_env={"m": m}, bindings=tb._IR_TO_TORCH)
        b2 = eval_term(out2, var_env={"m": m}, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a2, b2)
        # non-scalar operand declines — a matrix can't accompany a split
        lhs3 = Op.make(
            "split",
            Op.make("mul", _v("m", 8, 8), _v("c", 8, 8)),
            sizes=(4, 4),
            dim=-2,
            index=0,
        )
        rm = next(
            r for r in SPLIT_SCALAR_RULES if r.name == "split_scalar_mul_r"
        )
        subst3 = match_pattern(rm.lhs, lhs3, {})
        assert subst3 is not None and not rm.check(subst3)

    def test_split_commute_fires(self):
        # split_{-1}(split_{-2}(t, SA, i), SB, j) ↔ the swapped order
        lhs = Op.make(
            "split",
            Op.make("split", _v("t", 8, 8), sizes=(4, 4), dim=-2, index=1),
            sizes=(4, 4),
            dim=-1,
            index=0,
        )
        subst = match_pattern(SPLIT_COMMUTE.lhs, lhs, {})
        assert subst is not None and SPLIT_COMMUTE.check(subst)
        out = instantiate_pattern(SPLIT_COMMUTE.rhs, subst)
        t = torch.randn(8, 8, dtype=torch.float64)
        env = {"t": t}
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b)

    def test_split_commute_declines(self):
        b = {
            "$attr:SA": (4, 4),
            "$attr:SB": (4, 4),
            "$attr:I": 0,
            "$attr:J": 0,
        }
        # same axis → not a disjoint-axis swap
        assert not SPLIT_COMMUTE.check(
            {**b, "$attr:DA": -1, "$attr:DB": -1}
        )
        # rank-2 tensor: -2 and 0 are the SAME axis
        assert not SPLIT_COMMUTE.check(
            {**b, "t": _v("t", 8, 8), "$attr:DA": -2, "$attr:DB": 0}
        )
        # unshapeable t, same raw dims
        assert not SPLIT_COMMUTE.check(
            {**b, "$attr:DA": -1, "$attr:DB": -1}
        )
        # non-int dims decline
        assert not SPLIT_COMMUTE.check(
            {**b, "$attr:DA": "x", "$attr:DB": -1}
        )
        # bad split sizes / out-of-range index
        assert not SPLIT_COMMUTE.check(
            {**b, "t": _v("t", 8, 8), "$attr:DA": -2, "$attr:DB": -1,
             "$attr:SA": (4, 0)}
        )
        assert not SPLIT_COMMUTE.check(
            {**b, "t": _v("t", 8, 8), "$attr:DA": -2, "$attr:DB": -1,
             "$attr:I": 5}
        )

    def test_split_commute_unshaped_distinct_dims(self):
        """An unshapeable ``t`` with distinct raw dims skips the
        mod-rank equality test — the per-split attrs decide."""
        assert SPLIT_COMMUTE.check(
            {
                # no "t" — _vshape declines to a shape
                "$attr:DA": -1,
                "$attr:DB": 0,
                "$attr:SA": (4, 4),
                "$attr:I": 0,
                "$attr:SB": (4, 4),
                "$attr:J": 0,
            }
        )


class TestRealExportFold:
    """The whole chain composes on a REAL torch.export graph.

    ChunkedRetention exports ``matmul(mul(div(q·kᵀ, √d), tril(...)), v)``
    — the decay spelled out, the score fused.  The machine folds the
    spelled decay (``decay_lift`` on the ``detach_``/``contiguous``-wra-
    pped leaf), chunkifies rows and keys, splits the score into per-
    block ``matmul(q_i, k_jᵀ)``, factors the strict-lower decay block
    (``decay_vec``), and lifts the weighted state into ``applyd`` —
    none of it by name, all of it from the law set.
    """

    def test_real_export_reaches_applyd(self):
        pytest.importorskip("mlopt_torch")
        from mlopt_torch.adapters import TorchSource
        from mlopt_discovery.zoo import _ChunkedRetention

        m = _ChunkedRetention(16)
        x = torch.randn(2, 8, 16)
        term = TorchSource().to_ir(m, x)[0].root
        eg = EGraph()
        root = eg.add_term(term)
        from mlopt_core.laws.tensor import SPLIT_SCALAR_RULES

        eg.run(
            list(OM_LAWS) + list(OM_CHUNKIFY) + list(SPLIT_SCALAR_RULES),
            root,
            max_iterations=12,
            max_nodes=100_000,
            rule_budgets={
                "om_decayed_chunkify_rows": 30,
                "om_decayed_chunkify_keys": 30,
            },
            patience=6,
        )
        ops = [
            n.op
            for cls in eg._classes.values()
            for n in cls.nodes
        ]
        # the spelled tril-decay folded into the generator
        assert "decay" in ops
        # the rank-1 cross factorisation landed
        assert "decay_vec" in ops
        # the weighted state folded into the carrier
        assert "applyd" in ops

    def test_real_export_reaches_state_fold(self):
        """The sequential-carrier form: ``matmul(·, add(·,·))``.

        With the state-weight bridges shipped — flattened splits,
        the mixed-partition cross law, the scalar sink/exit, and the
        shared-denominator naturality — the saturated graph spells a
        block contribution as ``matmul(q_i, add(S', S''))``: the
        inter-chunk accumulator sitting INSIDE the contraction, the
        O(T·C) fold the plan is named for.
        """
        pytest.importorskip("mlopt_torch")
        from mlopt_torch.adapters import TorchSource
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_core.laws.tensor import (
            ASSOC_MATMUL_REV,
            DIV_ADD_FACTOR,
            FACTOR_MUL,
            FACTOR_MUL_ADD,
            MATMUL_DIV_DISTRIBUTE,
            MUL_DIV_SWAP,
            SPLIT_SCALAR_RULES,
        )

        m = _ChunkedRetention(16)
        x = torch.randn(2, 32, 16)
        term = TorchSource().to_ir(m, x)[0].root
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(
            list(OM_LAWS) + list(OM_CHUNKIFY) + list(SPLIT_SCALAR_RULES)
            + [
                MUL_DIV_SWAP,
                MATMUL_DIV_DISTRIBUTE,
                ASSOC_MATMUL_REV,
                FACTOR_MUL,
                FACTOR_MUL_ADD,
                DIV_ADD_FACTOR,
            ],
            root,
            max_iterations=16,
            max_nodes=150_000,
            patience=8,
            rule_budgets={
                "om_decayed_chunkify_rows": 60,
                "om_decayed_chunkify_keys": 60,
            },
        )
        cls = eg._classes
        matmul_rhs_adds = [
            n
            for c in cls.values()
            for n in c.nodes
            if n.op == "matmul"
            and len(n.children) == 2
            and any(
                x.op == "add"
                for x in cls[eg.find(n.children[1])].nodes
            )
        ]
        assert matmul_rhs_adds, (
            "no matmul(·, add) member — the accumulator never "
            "reached the right operand"
        )
        # the sum inside carries the weighted-state spelling: an
        # applyd fold or a mul(γ, S) member on an addend
        rhs = eg.find(matmul_rhs_adds[0].children[1])
        rhs_ops = {x.op for x in cls[rhs].nodes}
        assert "applyd" in rhs_ops or "mul" in rhs_ops

    def test_extraction_prefers_the_chunked_carrier_at_t512(self):
        """T=512 ≫ the fused quadratic: extraction picks the carrier.

        Plan 0023's headline, costed honestly.  On the same real
        export, saturated with the state-weight bridges, the root
        class holds BOTH the fused ``om_decayed`` (the ``(T,T)``
        quadratic) and the chunked cross-block carrier — per-chunk
        intra ``om_decayed`` plus the ``q_i @ (k_jᵀ v_j)`` state
        readout.  Under ``flops_cost`` the chunked member prices
        lower because it does less work, not because a carrier
        discount was invented, so extraction selects it.  The fused
        member must still price higher (the honest direction).
        """
        pytest.importorskip("mlopt_torch")
        from mlopt_core.cost import dag_cost, flops_cost
        from mlopt_core.laws.tensor import (
            ASSOC_MATMUL_REV,
            DIV_ADD_FACTOR,
            FACTOR_MUL,
            FACTOR_MUL_ADD,
            MATMUL_DIV_DISTRIBUTE,
            MUL_DIV_SWAP,
            SPLIT_SCALAR_RULES,
        )
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_torch.adapters import TorchSource

        m = _ChunkedRetention(16).double()
        x = torch.randn(2, 512, 16, dtype=torch.float64)
        term = TorchSource().to_ir(m, x)[0].root
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(
            list(OM_LAWS) + list(OM_CHUNKIFY) + list(SPLIT_SCALAR_RULES)
            + [
                MUL_DIV_SWAP,
                MATMUL_DIV_DISTRIBUTE,
                ASSOC_MATMUL_REV,
                FACTOR_MUL,
                FACTOR_MUL_ADD,
                DIV_ADD_FACTOR,
            ],
            root,
            max_iterations=16,
            max_nodes=150_000,
            patience=8,
            rule_budgets={
                "om_decayed_chunkify_rows": 60,
                "om_decayed_chunkify_keys": 60,
            },
        )
        rc = eg.find(root)
        best = eg.extract_best(rc, flops_cost)
        assert best.op == "concat", (
            f"extraction picked {best.op!r}, not the chunked carrier"
        )
        fused = [
            n for n in eg._classes[rc].nodes if n.op == "om_decayed"
        ]
        assert fused, "the fused om_decayed member vanished"
        fused_term = eg.extract_best(
            rc, flops_cost, overrides={rc: fused[0]}
        )
        assert dag_cost(best, flops_cost) < dag_cost(
            fused_term, flops_cost
        ), "the carrier must price lower honestly, not by discount"


class TestMachineFold:
    """The machine derives the recurrence from spelled cross terms.

    Feed the e-graph the spelled per-block cross contributions —
    ``g^o ⊙ (xi_r ⊙ ((q·kᵀ ⊙ etaᵀ) @ v))`` — and it composes the whole
    chain itself: unsq_mul frees the scalar, matmul_col_decay lands
    eta on the keys, factor_matmul collects the shared q₂, and the
    decay-state lift folds the weighted sum onto ``applyd``.
    """

    def test_spelled_cross_folds_to_applyd(self):
        """The weighted state sum itself folds onto the carrier.

        The earlier chain — spelled cross → eta on keys → matmul(q,B)
        → g^o scalars pulled out — was demonstrated in the saturated
        e-graph; the fold this test pins is the last hop, the
        ``add(mul(g^o, B), mul(g^o', B'))`` sum the chain lands on.
        """
        g = _v("g")
        b0 = _v("b0", 3, 6)
        b1 = _v("b1", 3, 6)
        w8 = Op.make("decay_vec", g, t=1, offset=8)
        w4 = Op.make("decay_vec", g, t=1, offset=4)
        term = Op.make(
            "add",
            Op.make("mul", w8, b0),
            Op.make("mul", w4, b1),
        )
        eg = EGraph()
        root = eg.add_term(term)
        laws = list(OM_LAWS) + list(SCAN_DIAG_LAWS)
        eg.run(laws, root, max_iterations=6)
        app = [
            n
            for cls in eg._classes.values()
            for n in cls.nodes
            if n.op in ("applyd", "aff_diag", "affd_compose")
        ]
        assert app, "the machine should fold the sum onto applyd"

    def test_folded_form_verifies(self):
        """The applyd spelling the machine produced equals the sum."""
        g = torch.tensor(0.9, dtype=torch.float64)
        b0 = torch.randn(3, 6, dtype=torch.float64)
        b1 = torch.randn(3, 6, dtype=torch.float64)
        w8 = Op.make("decay_vec", _v("g"), t=1, offset=8)
        w4 = Op.make("decay_vec", _v("g"), t=1, offset=4)
        term = Op.make(
            "add",
            Op.make("mul", w8, _v("b0")),
            Op.make("mul", w4, _v("b1")),
        )
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(
            list(OM_LAWS) + list(SCAN_DIAG_LAWS),
            root,
            max_iterations=8,
        )
        env = {"g": g, "b0": b0, "b1": b1}
        spelled = eval_term(term, var_env=env, bindings=tb._IR_TO_TORCH)
        folded = Op.make(
            "applyd",
            Op.make(
                "aff_diag",
                Op.make("decay_vec", _v("g"), t=1, offset=8),
                Op.make(
                    "mul",
                    Op.make("decay_vec", _v("g"), t=1, offset=4),
                    _v("b1"),
                ),
            ),
            _v("b0"),
        )
        out = eval_term(folded, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(spelled, out, atol=1e-12)

    def test_full_chunked_form_verifies(self):
        """Spelled fused eval ≡ the factored chunked form, fp64.

        The causal four-block split: block (0,1) is zero (strict
        upper), block (1,0) is the factored cross, diagonals are
        local ``om_decayed`` — the RetNet inner/outer-chunk split.
        """
        T1, T2, DV = 4, 5, 4
        s = torch.randn(T1 + T2, T1 + T2, dtype=torch.float64)
        v = torch.randn(T1 + T2, DV, dtype=torch.float64)
        pos = torch.arange(T1 + T2, dtype=torch.float64)
        D = torch.tril(
            G ** (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0)
        )
        fused = (s * D) @ v
        # row block 0 — only cols < T1 survive the causal mask
        row0 = (s[:T1, :T1] * D[:T1, :T1]) @ v[:T1]
        # row block 1 — intra (cols ≥ T1) + factored cross (cols < T1)
        xi = G ** (torch.arange(T2, dtype=torch.float64) + T1)
        eta = G ** (-torch.arange(T1, dtype=torch.float64))
        row1 = (s[T1:, T1:] * D[T1:, T1:]) @ v[T1:] + xi.unsqueeze(
            -1
        ) * ((s[T1:, :T1] * eta.unsqueeze(-2)) @ v[:T1])
        chunked = torch.cat([row0, row1], dim=0)
        assert torch.allclose(fused, chunked, atol=1e-10)


class TestOmDecayedKway:
    """The K-way causal block law — the fold's material in one step.

    ``om_decayed(s, decay(g,T), v)`` -> the full causal block
    decomposition for the uniform ``(C,)*K`` partition (``C = T/K``):
    diagonal blocks regenerate ``decay(g,C)``, strict-lower blocks
    factor as the rank-1 ``xi_i * eta_j^T`` outer product.  ONE
    application reaches ``K`` chunks, so a row block sees ``K-1``
    prior contributions at once — the pairwise bisection only ever
    hands the inter-chunk fold one prior block per row (plan 0023
    construct 3).
    """

    def _site(self, T):
        return Op.make(
            "om_decayed",
            _v("s", T, T),
            Op.make("decay", _v("g"), t=T),
            _v("v", T, 4),
        )

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_kway_fires_and_derives(self, k):
        law = om_decayed_kway_blocks(k)
        T = 16
        subst = match_pattern(law.lhs, self._site(T), {})
        assert subst is not None
        assert law.check(subst)
        der = law.derive(subst)
        c = T // k
        assert der["$attr:C"] == c
        assert der["$attr:SZ"] == (c,) * k
        # strict-lower block (i,j) sits (i-j)C rows below the diagonal
        for i in range(k):
            for j in range(i):
                assert der[f"$attr:OFF_{i}_{j}"] == (i - j) * c

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_kway_verifies(self, k):
        """The K-way rhs ≡ the fused decayed product, fp64."""
        T, DV = 16, 4
        law = om_decayed_kway_blocks(k)
        lhs = self._site(T)
        subst = match_pattern(law.lhs, lhs, {})
        rhs = instantiate_pattern(
            law.rhs, {**subst, **law.derive(subst)}
        )
        torch.manual_seed(0)
        s = torch.randn(T, T, dtype=torch.float64)
        v = torch.randn(T, DV, dtype=torch.float64)
        env = {"s": s, "v": v, "g": G}
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(rhs, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b, atol=1e-10)

    def test_kway_rhs_is_the_block_decomposition(self):
        """K diagonal blocks, K(K-1)/2 factored crosses, one concat."""
        k = 4
        rhs = om_decayed_kway_blocks(k).rhs
        ops = op_repr(rhs)
        assert ops.count("decay_vec") == k * (k - 1)  # 2 per cross
        assert ops.count("concat") == k - 1  # nested binary cat

    def test_kway_declines(self):
        law = om_decayed_kway_blocks(4)
        ok = {"s": _v("s", 16, 16), "v": _v("v", 16, 4)}
        # T not divisible by k
        assert not law.check({**ok, "$attr:T": 12})
        # T < k
        assert not law.check({**ok, "$attr:T": 2})
        # non-int / missing T
        assert not law.check({**ok, "$attr:T": "x"})
        assert not law.check(ok)
        # non-square score plane
        assert not law.check(
            {"$attr:T": 16, "s": _v("s", 16, 8), "v": _v("v", 16, 4)}
        )
        # value rows disagree with the decay extent
        assert not law.check(
            {"$attr:T": 16, "s": _v("s", 16, 16), "v": _v("v", 8, 4)}
        )
        # rank-1 operands — no block axes to split
        assert not law.check(
            {"$attr:T": 16, "s": _v("s", 16), "v": _v("v", 16, 4)}
        )
        # unshapeable operand
        assert not law.check({"$attr:T": 16, "s": None, "v": None})
        # the derive declines wherever the check does
        assert law.derive({"$attr:T": 12, **ok}) is None

    def test_kway_arity_validated(self):
        """A bad arity fails at construction, not at saturation."""
        with pytest.raises(ValueError):
            om_decayed_kway_blocks(1)
        with pytest.raises(ValueError):
            om_decayed_kway_blocks("x")

    def test_kway_laws_is_opt_in(self):
        """The K-way law is NOT in the everyday carrier set."""
        laws = om_kway_laws(4)
        assert len(laws) == 1
        assert laws[0].name == "om_decayed_kway_blocks_4"
        assert laws[0].check is not None
        assert laws[0].derive is not None
        names = {r.name for r in OM_LAWS}
        assert "om_decayed_kway_blocks_4" not in names

    def test_kway_reaches_more_chunks_than_bisection(self):
        """ONE application ⇒ K chunks; the bisection needs a split/level.

        On the fused site the bisection's own law yields two blocks; the
        K-way law yields K — the structural claim the fold rides on.
        """
        T = 16
        site = self._site(T)
        kway = om_decayed_kway_blocks(4)
        subst = match_pattern(kway.lhs, site, {})
        rhs = instantiate_pattern(
            kway.rhs, {**subst, **kway.derive(subst)}
        )
        # every leaf om_decayed sits over a (T/4, T/4) score block
        sizes = {
            dict(n.attrs).get("sizes")
            for n in _walk_ops(rhs)
            if n.op == "split"
        }
        assert sizes == {(T // 4,) * 4}

    def test_kway_drops_the_extracted_cost_on_the_real_export(self):
        """The K-way material lowers the carrier's priced cost.

        On the real ChunkedRetention export the pairwise bisection
        saturates the carrier at ~2.25M FLOPs (T=128); handing the
        fold laws the K-way block decomposition in ONE application
        drops the extracted cost to ~1.21M — and extraction still
        selects the chunked carrier, not the fused quadratic.
        """
        pytest.importorskip("mlopt_torch")
        from mlopt_core.cost import dag_cost, flops_cost
        from mlopt_core.laws.tensor import (
            ASSOC_MATMUL_REV,
            DIV_ADD_FACTOR,
            FACTOR_MUL,
            FACTOR_MUL_ADD,
            MATMUL_DIV_DISTRIBUTE,
            MUL_DIV_SWAP,
            SPLIT_SCALAR_RULES,
        )
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_torch.adapters import TorchSource

        base = (
            list(OM_LAWS)
            + list(OM_CHUNKIFY)
            + list(SPLIT_SCALAR_RULES)
            + [
                MUL_DIV_SWAP,
                MATMUL_DIV_DISTRIBUTE,
                ASSOC_MATMUL_REV,
                FACTOR_MUL,
                FACTOR_MUL_ADD,
                DIV_ADD_FACTOR,
            ]
        )
        chunk_budget = {
            "om_decayed_chunkify_rows": 60,
            "om_decayed_chunkify_keys": 60,
        }

        def saturate(rules, budgets):
            m = _ChunkedRetention(16).double()
            x = torch.randn(2, 128, 16, dtype=torch.float64)
            term = TorchSource().to_ir(m, x)[0].root
            eg = EGraph()
            root = eg.add_term(term)
            eg.run(
                rules,
                root,
                max_iterations=16,
                max_nodes=150_000,
                patience=8,
                rule_budgets=budgets,
            )
            rc = eg.find(root)
            return eg.extract_best(rc, flops_cost)

        base_best = saturate(base, chunk_budget)
        kw_best = saturate(
            base + om_kway_laws(2),
            {**chunk_budget, "om_decayed_kway_blocks_2": 60},
        )
        assert base_best.op == "concat"
        assert kw_best.op == "concat"
        assert dag_cost(kw_best, flops_cost) < dag_cost(
            base_best, flops_cost
        )

    @pytest.mark.slow

    def test_kway_deep_fold_exists_but_extraction_keeps_the_spelled_form(
        self,
    ):
        """The deep fold reaches the target depth, yet extraction keeps
        the non-fold — a fold is a *cost-neutral relabelling*.

        At ``k=8`` (``C=16``, ``T=128``) the saturated root class holds
        a finite member whose ``applyd`` nesting is the acceptance
        target ``T/C-1 == 7``: the deep fold DOES exist.  But the root
        class has no ``applyd``-headed member, a fold prices exactly
        like the ``mul``/``add`` it re-spells (see
        ``test_fold_is_a_cost_neutral_relabelling``), and extraction
        (correctly) keeps the ``concat`` spelled form at nesting 0.
        The *paying* fold needs the cross-sum reassociated onto a
        SHARED state recurrence, which the law set never spells — a
        search gap, not a cost gap.
        """
        pytest.importorskip("mlopt_torch")
        from mlopt_core.cost import flops_cost
        from mlopt_core.egraph import EGraph
        from mlopt_core.laws.tensor import (
            ASSOC_MATMUL_REV,
            DIV_ADD_FACTOR,
            FACTOR_MUL,
            FACTOR_MUL_ADD,
            MATMUL_DIV_DISTRIBUTE,
            MUL_DIV_SWAP,
            SPLIT_SCALAR_RULES,
        )
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_torch.adapters import TorchSource

        T, K = 128, 8
        rules = (
            list(OM_LAWS)
            + list(OM_CHUNKIFY)
            + list(SPLIT_SCALAR_RULES)
            + [
                MUL_DIV_SWAP,
                MATMUL_DIV_DISTRIBUTE,
                ASSOC_MATMUL_REV,
                FACTOR_MUL,
                FACTOR_MUL_ADD,
                DIV_ADD_FACTOR,
            ]
            + om_kway_laws(K)
        )
        budgets = {
            "om_decayed_chunkify_rows": 60,
            "om_decayed_chunkify_keys": 60,
            f"om_decayed_kway_blocks_{K}": 200,
        }
        torch.manual_seed(0)
        m = _ChunkedRetention(16).double()
        x = torch.randn(2, T, 16, dtype=torch.float64)
        term = TorchSource().to_ir(m, x)[0].root
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(
            rules,
            root,
            max_iterations=16,
            max_nodes=150_000,
            patience=8,
            rule_budgets=budgets,
        )
        rc = eg.find(root)
        root_ops = {n.op for n in eg._classes[rc].nodes}
        # No member of the root class is headed by the fold carrier.
        assert "applyd" not in root_ops
        assert "affd_compose" not in root_ops
        best = eg.extract_best(rc, flops_cost)
        assert best.op == "concat"
        assert _fold_nesting(best) == 0
        # …yet the deep fold exists, at the acceptance depth T/C-1, and
        # extraction leaves it on the table.
        deep = _finite_fold_depth(eg, rc)
        assert deep == T // (T // K) - 1
        assert deep > _fold_nesting(best)

    def test_fold_is_a_cost_neutral_relabelling(self):
        """``applyd(aff_diag(a, x), h)`` prices EXACTLY like the
        ``add(mul(a, h), x)`` it re-spells.

        Folding alone saves no FLOPs — the O(T·C) win comes from the
        reassociation onto a shared state, not the carrier spelling —
        so extraction has no cost reason to prefer a fold member.
        """
        from mlopt_core.cost import flops_cost

        a, x, h = _v("a"), _v("x", 32, 16), _v("h", 32, 16)
        folded = Op.make("applyd", Op.make("aff_diag", a, x), h)
        spelled = Op.make("add", Op.make("mul", a, h), x)
        assert flops_cost(folded) == flops_cost(spelled)

    def test_shared_state_fold_prices_below_the_spelled_cross(self):
        """The *paying* fold is O(T·C); the e-graph's relabelling is not.

        A hand-built shared-state recurrence — each ``S_i`` one
        e-class consumed by block ``i``'s output and block ``i+1``'s
        step — prices far below the spelled cross sum.  (Written when
        this was still a search gap; the staged pair now spells and
        selects it — see ``TestSharedStateExtraction``.)
        """
        from mlopt_core.cost import dag_cost, flops_cost

        T, C, K, d = 128, 16, 8, 16
        shared = _shared_state_fold(T, C, K, d)
        spelled = _spelled_cross(T, C, K, d)
        assert dag_cost(shared, flops_cost) < dag_cost(
            spelled, flops_cost
        )


class TestSharedStateReassoc:
    """The shared-state reassociation — the sequential carrier's move.

    A fold is a *cost-neutral relabelling* (``applyd(aff_diag(a,x),h)``
    prices exactly like ``add(mul(a,h),x)``), so the O(T·C) win needs
    the cross sum reassociated onto a *shared* recurrence
    ``T_k = gamma^C⊙T_{k-1} + b_{k-1}`` rather than the flat left-nested
    tree.  ``om_shared_state_laws()`` supplies the geometric move
    ``W1⊙X + W2⊙Y = W2 ⊙ (gamma^{O1-O2}⊙X + Y)`` plus its block-nested
    and contraction-factoring variants; it is opt-in (a closure
    generator over the cross-sum space) and stays out of ``OM_LAWS``.
    """

    @staticmethod
    def _w(off, t=1):
        return Op.make(
            "unsqueeze",
            Op.make("decay_vec", _v("g"), t=t, offset=off),
            dim=-1,
        )

    def _flat_lhs(self, t=1):
        return Op.make(
            "add",
            Op.make("mul", self._w(2, t), _v("X", 4, 4)),
            Op.make("mul", self._w(1, t), _v("Y", 4, 4)),
        )

    def test_flat_fires_and_derives(self):
        law = om_shared_state_laws()[0]
        subst = match_pattern(law.lhs, self._flat_lhs(), {})
        assert subst is not None
        assert law.check(subst)
        assert law.derive(subst) == {"$attr:OD": 1}
        subst.update(law.derive(subst))
        out = instantiate_pattern(law.rhs, subst)
        r = op_repr(out)
        assert "offset=1" in r and "add" in r

    @pytest.mark.parametrize("t", [1, 4])
    def test_flat_is_exact_fp64(self, t):
        """``W1⊙X + W2⊙Y = W2 ⊙ (gamma^{O1-O2}⊙X + Y)`` — exact."""
        law = om_shared_state_laws()[0]
        lhs = self._flat_lhs(t)
        subst = match_pattern(law.lhs, lhs, {})
        assert subst is not None and law.check(subst)
        subst.update(law.derive(subst))
        out = instantiate_pattern(law.rhs, subst)
        env = {
            "g": G,
            "X": torch.randn(4, 4, dtype=torch.float64),
            "Y": torch.randn(4, 4, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b, atol=1e-12)

    def test_mat_factoring_is_exact_fp64(self):
        """``W1⊙(Q@A) + W2⊙(Q@B) = W2⊙(Q@(gamma^{step}⊙A + B))``."""
        law = om_shared_state_laws()[2]
        q = _v("Q", 3, 4)
        lhs = Op.make(
            "add",
            Op.make(
                "add",
                _v("P", 3, 4),
                Op.make(
                    "mul",
                    self._w(2),
                    Op.make("matmul", q, _v("A", 4, 4)),
                ),
            ),
            Op.make(
                "mul",
                self._w(1),
                Op.make("matmul", q, _v("B", 4, 4)),
            ),
        )
        subst = match_pattern(law.lhs, lhs, {})
        assert subst is not None and law.check(subst)
        subst.update(law.derive(subst))
        out = instantiate_pattern(law.rhs, subst)
        env = {
            "g": G,
            "P": torch.randn(3, 4, dtype=torch.float64),
            "Q": torch.randn(3, 4, dtype=torch.float64),
            "A": torch.randn(4, 4, dtype=torch.float64),
            "B": torch.randn(4, 4, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b, atol=1e-12)

    def test_step_guard_declines(self):
        """Zero / negative / non-aligned / non-int steps all veto."""
        law = om_shared_state_laws()[0]
        # OD <= 0
        assert law.derive({"$attr:O1": 4, "$attr:O2": 4}) is None
        assert law.derive({"$attr:O1": 2, "$attr:O2": 4}) is None
        # the step must divide both offsets (chunk alignment)
        assert law.derive({"$attr:O1": 5, "$attr:O2": 2}) is None
        # non-int offsets
        assert law.derive({"$attr:O1": "x", "$attr:O2": 2}) is None
        # a genuine geometric pair passes
        assert law.derive({"$attr:O1": 8, "$attr:O2": 4}) == {
            "$attr:OD": 4
        }

    def test_declines_unshapeable_operands(self):
        """The check needs resolvable value shapes on both summands."""
        law = om_shared_state_laws()[0]
        bound = {"$attr:O1": 2, "$attr:O2": 1, "X": _v("X", 4, 4)}
        assert not law.check(bound)  # Y unbound
        bound["Y"] = _v("Y", 4, 4)
        assert law.check(bound)

    def test_declines_unbroadcastable_summands(self):
        law = om_shared_state_laws()[0]
        bound = {
            "$attr:O1": 2,
            "$attr:O2": 1,
            "X": _v("X", 4, 4),
            "Y": _v("Y", 3, 4),  # rows disagree → no broadcast
        }
        assert not law.check(bound)

    def test_mat_declines_unshapeable_operands(self):
        """``om_shared_mat_reassoc`` needs resolvable value shapes on
        all three contraction operands before typing the pair."""
        base = {
            "$attr:O1": 2,
            "$attr:O2": 1,
            "Q": _v("Q", 3, 4),
            "A": _v("A", 4, 4),
            "B": _v("B", 4, 4),
        }
        assert SHARED_STATE_MAT_REASSOC.check(base)
        for key in ("Q", "A", "B"):
            bound = {k: v for k, v in base.items() if k != key}
            assert not SHARED_STATE_MAT_REASSOC.check(bound), key

    def test_mat_declines_rank1_contractand(self):
        """A vector A/B isn't a matrix state contraction."""
        bound = {
            "$attr:O1": 2,
            "$attr:O2": 1,
            "Q": _v("Q", 3, 4),
            "A": _v("A", 4),   # rank < 2
            "B": _v("B", 4, 4),
        }
        assert not SHARED_STATE_MAT_REASSOC.check(bound)
        bound["A"], bound["B"] = _v("A", 4, 4), _v("B", 4)
        assert not SHARED_STATE_MAT_REASSOC.check(bound)

    def test_laws_are_opt_in(self):
        laws = om_shared_state_laws()
        names = [r.name for r in laws]
        assert names == [
            "om_shared_reassoc",
            "om_shared_reassoc_prefix",
            "om_shared_mat_reassoc",
        ]
        everyday = {r.name for r in OM_LAWS}
        assert not (set(names) & everyday)
        for law in laws:
            assert law.check is not None and law.derive is not None

    def test_prefix_reassociates_the_block_nesting(self):
        """``add(add(P, W1⊙X), W2⊙Y)`` → the nested pair, P intact."""
        law = om_shared_state_laws()[1]
        lhs = Op.make(
            "add",
            Op.make("add", _v("P", 4, 4), Op.make("mul", self._w(2), _v("X", 4, 4))),
            Op.make("mul", self._w(1), _v("Y", 4, 4)),
        )
        subst = match_pattern(law.lhs, lhs, {})
        assert subst is not None and law.check(subst)
        subst.update(law.derive(subst))
        out = instantiate_pattern(law.rhs, subst)
        assert out.op == "add"
        assert op_repr(out.args[0]) == "P"
        assert out.args[1].op == "mul"

    def test_shared_state_is_shared_across_row_blocks(self):
        """The reassociated state is ONE e-class used by two blocks.

        Two row blocks whose cross sums share the prefix ``B0, B1``: the
        flat spelling re-derives the weighted contributions per block,
        the reassociated one consumes a single state class — so the DAG
        cost drops and the state has ≥2 parents.
        """
        law = om_shared_state_laws()[1]
        from mlopt_core.cost import dag_cost, flops_cost

        w = self._w
        B0, B1, B2 = _v("B0", 4, 4), _v("B1", 4, 4), _v("B2", 4, 4)
        X, Y = _v("X", 4, 4), _v("Y", 4, 4)
        blk2 = Op.make(
            "add",
            Op.make("add", X, Op.make("mul", w(2), B0)),
            Op.make("mul", w(1), B1),
        )
        blk3 = Op.make(
            "add",
            Op.make(
                "add",
                Op.make("add", Y, Op.make("mul", w(3), B0)),
                Op.make("mul", w(2), B1),
            ),
            Op.make("mul", w(1), B2),
        )
        root_term = Op.make("concat", blk2, blk3, dim=-2)
        eg = EGraph()
        root = eg.add_term(root_term)
        before = dag_cost(
            eg.extract_best(eg.find(root), flops_cost), flops_cost
        )
        eg.run([law], root, max_iterations=10, max_nodes=20_000)
        after = dag_cost(
            eg.extract_best(eg.find(root), flops_cost), flops_cost
        )
        assert after < before
        # the state add(mul(gamma^C, B0), B1) is one class with two parents
        state = Op.make("add", Op.make("mul", w(1), B0), B1)
        sc = eg.find(eg.add_term(state))
        parents = {
            eg.find(cid)
            for cid, cl in eg._classes.items()
            for n in cl.nodes
            if any(eg.find(c) == sc for c in n.children)
        }
        assert len(parents) >= 2

    @pytest.mark.slow

    def test_reassoc_drops_the_extracted_cost_on_the_real_export(self):
        """The law fires on the chunked export and lowers its cost.

        Measured (T=64, K=4, C=16, budget 200): the reassociation fires
        across the block-nested cross sums and extraction picks a
        strictly cheaper member — the shared-state spelling is reachable
        and priced.  (The state is not yet fully shared across ALL row
        blocks — the score's scalar division wraps the contraction, so
        the q-factored ``matmul(Q, ·)`` member the ``_mat`` variant
        wants is not exposed; see the plan's remaining gap.)
        """
        pytest.importorskip("mlopt_torch")
        from mlopt_core.cost import dag_cost, flops_cost
        from mlopt_core.laws.tensor import (
            ASSOC_MATMUL_REV,
            DIV_ADD_FACTOR,
            FACTOR_MUL,
            FACTOR_MUL_ADD,
            MATMUL_DIV_DISTRIBUTE,
            MUL_DIV_SWAP,
            SPLIT_SCALAR_RULES,
        )
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_torch.adapters import TorchSource

        T, K = 64, 4
        base = (
            list(OM_LAWS)
            + list(OM_CHUNKIFY)
            + list(SPLIT_SCALAR_RULES)
            + [
                MUL_DIV_SWAP,
                MATMUL_DIV_DISTRIBUTE,
                ASSOC_MATMUL_REV,
                FACTOR_MUL,
                FACTOR_MUL_ADD,
                DIV_ADD_FACTOR,
            ]
            + om_kway_laws(K)
        )
        budgets = {
            "om_decayed_chunkify_rows": 60,
            "om_decayed_chunkify_keys": 60,
            f"om_decayed_kway_blocks_{K}": 200,
        }

        def saturate(rules, rule_budgets):
            torch.manual_seed(0)
            m = _ChunkedRetention(16).double()
            x = torch.randn(2, T, 16, dtype=torch.float64)
            term = TorchSource().to_ir(m, x)[0].root
            eg = EGraph()
            root = eg.add_term(term)
            eg.run(
                rules,
                root,
                max_iterations=16,
                max_nodes=150_000,
                patience=8,
                rule_budgets=rule_budgets,
            )
            return dag_cost(
                eg.extract_best(eg.find(root), flops_cost), flops_cost
            )

        shared = om_shared_state_laws()
        base_best = saturate(base, budgets)
        law_best = saturate(
            base + shared,
            {**budgets, **{r.name: 400 for r in shared}},
        )
        assert law_best < base_best


class TestDivBridgeStaging:
    """The div bridge and the scheduled pair it belongs to.

    ``om_shared_mat_reassoc`` fires 0× on the real export because the
    score is ``q(x)@k(x)ᵀ/√t``: the scalar denominator wraps every cross
    contraction (``div(matmul(q_i, B_j), c)``), so the reassociation
    never sees ``mul(W, matmul(Q, B))``.  The bridge that lifts the
    denominator inside — ``div(A@B, c) = (A/c)@B`` — is the *exact
    inverse* of the shipped ``matmul_div_distribute``, so the two are a
    two-cycle and can never be co-resident.  ``om_shared_state_stages``
    splits them across two saturation stages.
    """

    def _div_of_matmul(self, cshape=()):
        return Op.make(
            "div",
            Op.make("matmul", _v("A", 4, 5), _v("B", 5, 3)),
            _v("c", *cshape),
        )

    def test_bridge_is_the_exact_inverse_of_matmul_div_distribute(self):
        """Same equality, opposite spelling — that is why they cycle."""
        from mlopt_core.laws.tensor import MATMUL_DIV_DISTRIBUTE

        law = om_div_bridge_laws()[0]
        assert law.name == "div_into_matmul"
        assert _op_skeleton(MATMUL_DIV_DISTRIBUTE.rhs) == _op_skeleton(
            law.lhs
        )
        assert _op_skeleton(MATMUL_DIV_DISTRIBUTE.lhs) == _op_skeleton(
            law.rhs
        )

    def test_fires_and_spells_the_distributed_form(self):
        law = om_div_bridge_laws()[0]
        subst = match_pattern(law.lhs, self._div_of_matmul(), {})
        assert subst is not None
        assert law.check(subst)
        out = instantiate_pattern(law.rhs, subst)
        assert out.op == "matmul"
        assert out.args[0].op == "div"

    def test_is_exact_fp64(self):
        """``div(A@B, c) = (A/c)@B`` — exact to fp64 round-off."""
        law = om_div_bridge_laws()[0]
        lhs = self._div_of_matmul()
        subst = match_pattern(law.lhs, lhs, {})
        assert subst is not None and law.check(subst)
        out = instantiate_pattern(law.rhs, subst)
        env = {
            "A": torch.randn(4, 5, dtype=torch.float64),
            "B": torch.randn(5, 3, dtype=torch.float64),
            "c": torch.tensor(2.5, dtype=torch.float64),
        }
        a = eval_term(lhs, var_env=env, bindings=tb._IR_TO_TORCH)
        b = eval_term(out, var_env=env, bindings=tb._IR_TO_TORCH)
        assert torch.allclose(a, b, atol=1e-14)

    def test_declines_a_non_scalar_denominator(self):
        """Only a scalar denominator may move inside the contraction."""
        law = om_div_bridge_laws()[0]
        subst = match_pattern(law.lhs, self._div_of_matmul((4, 3)), {})
        assert subst is None or not law.check(subst)

    def test_bridge_is_opt_in(self):
        laws = om_div_bridge_laws()
        assert [r.name for r in laws] == ["div_into_matmul"]
        assert "div_into_matmul" not in {r.name for r in OM_LAWS}
        assert "div_into_matmul" not in {r.name for r in OM_CHUNKIFY}

    def test_stages_split_the_two_div_laws(self):
        """Stage A has ``matmul_div_distribute``; stage B its inverse."""
        reach, reassoc = om_shared_state_stages(4)
        reach_names = {r.name for r in reach}
        reassoc_names = {r.name for r in reassoc}
        assert "matmul_div_distribute" in reach_names
        assert "matmul_div_distribute" not in reassoc_names
        assert "div_into_matmul" in reassoc_names
        assert "div_into_matmul" not in reach_names
        # the reassociation family is a stage-B concern only
        assert not any(n.startswith("om_shared_") for n in reach_names)
        assert "om_shared_mat_reassoc" in reassoc_names
        # stage B is stage A plus the bridge and the reassociation family
        assert reach_names <= reassoc_names | {"matmul_div_distribute"}

    def test_budgets_cover_the_closure_generators(self):
        budgets = om_shared_state_budgets(8, kway=200)
        assert budgets["om_decayed_kway_blocks_8"] == 200
        assert budgets["div_into_matmul"] > 0
        assert budgets["om_shared_mat_reassoc"] > 0
        assert budgets["om_decayed_chunkify_rows"] > 0

    @pytest.mark.slow

    def test_staged_saturation_unlocks_the_mat_reassoc(self):
        """The scheduling works: the bridge revives the mat variant.

        Co-resident the two div laws explode the closure (measured: the
        T=128/K=8 saturation does not finish inside 800 s); scheduled,
        stage B completes and ``om_shared_mat_reassoc`` fires — 0× in
        the all-at-once arm — while extraction picks a cheaper member.
        """
        pytest.importorskip("mlopt_torch")
        from mlopt_core.cost import dag_cost, flops_cost
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_torch.adapters import TorchSource

        T, K = 64, 4
        reach, reassoc = om_shared_state_stages(K)
        budgets = om_shared_state_budgets(K, kway=100)

        def build():
            torch.manual_seed(0)
            m = _ChunkedRetention(16).double()
            x = torch.randn(2, T, 16, dtype=torch.float64)
            eg = EGraph()
            return eg, eg.add_term(TorchSource().to_ir(m, x)[0].root)

        eg, root = build()
        eg.run(
            reach,
            root,
            max_iterations=16,
            max_nodes=150_000,
            patience=8,
            rule_budgets=budgets,
        )
        reach_cost = dag_cost(
            eg.extract_best(eg.find(root), flops_cost), flops_cost
        )
        assert eg.rule_fires.get("om_shared_mat_reassoc", 0) == 0

        eg.run(
            reassoc,
            root,
            max_iterations=16,
            max_nodes=150_000,
            patience=8,
            rule_budgets=budgets,
        )
        staged_cost = dag_cost(
            eg.extract_best(eg.find(root), flops_cost), flops_cost
        )
        # the bridge fired and the mat reassociation came alive
        assert eg.rule_fires["div_into_matmul"] > 0
        assert eg.rule_fires["om_shared_mat_reassoc"] > 0
        assert staged_cost < reach_cost
        # …and the shared state is a real class with ≥2 parents
        assert _shared_state_classes(eg, root) > 0


class TestSharedStateExtraction:
    """The staged extract *is* the shared-state carrier.

    The earlier read — "the reassociated state is not i-independent,
    the extract carries no fold" — came from counting ``applyd`` /
    ``aff_diag`` members in the extracted term.  That metric is
    structurally blind: ``applyd(aff_diag(a,x),h)`` is a COST-NEUTRAL
    respelling of ``add(mul(a,h),x)``, so extraction never prefers the
    carrier spelling even when it has selected exactly that structure.
    The real signature is the DAG shape of the extract: every
    cross-bearing row block reads ``ξ ⊙ (q_i @ S_{i-1})`` where
    ``S_i = g^C·S_{i-1} + B_i`` is ONE subterm — block ``i+1``'s state
    literally nests block ``i``'s.  These tests assert that signature
    (and the fp64 correctness of the delivered program).
    """

    @staticmethod
    def _leaves(t, acc=None):
        acc = set() if acc is None else acc
        if not isinstance(t, Op):
            acc.add(repr(t))
            return acc
        for a in t.args:
            TestSharedStateExtraction._leaves(a, acc)
        return acc

    @classmethod
    def _fold_reads(cls, term):
        """The ``q_i @ S`` read matmuls: q-only left, q-free kv right."""
        kv_leaves = {"p_k_weight", "p_k_bias", "p_v_weight", "p_v_bias"}
        q_leaves = {"p_q_weight", "p_q_bias"}
        kv_core = {"p_k_weight", "p_v_weight"}
        reads = []
        seen: set[int] = set()
        stack = [term]
        while stack:
            t = stack.pop()
            if not isinstance(t, Op) or id(t) in seen:
                continue
            seen.add(id(t))
            if t.op == "matmul" and len(t.args) == 2:
                left, right = t.args
                if isinstance(left, Op) and isinstance(right, Op):
                    ll = cls._leaves(left)
                    rl = cls._leaves(right)
                    # right arg is a state: carries BOTH the k and v
                    # paths and no q — the score blocks (k-only) and
                    # the projections do not qualify.
                    if (
                        not (ll & kv_leaves)
                        and kv_core <= rl
                        and not (rl & q_leaves)
                    ):
                        reads.append(t)
            stack.extend(t.args)
        return reads

    @staticmethod
    def _subtree_ids(t):
        ids: set[int] = set()
        stack = [t]
        while stack:
            x = stack.pop()
            if not isinstance(x, Op) or id(x) in ids:
                continue
            ids.add(id(x))
            stack.extend(x.args)
        return ids

    @pytest.mark.slow

    def test_extract_carries_the_chained_shared_state(self):
        """Every cross block reads a shared state; states chain.

        At T=64/K=4 the extract contains the fold reads
        ``q_i @ S_{i-1}`` (left subtree touches only the q projection;
        right subtree is q-free and carries the k/v paths), and the
        state object of an earlier block is nested inside a later
        block's — the ``S_i = g^C·S_{i-1} + B_i`` chain, shared.
        """
        pytest.importorskip("mlopt_torch")
        from mlopt_core.cost import dag_cost, flops_cost
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_torch.adapters import TorchSource

        T, K = 64, 4
        reach, reassoc = om_shared_state_stages(K)
        budgets = om_shared_state_budgets(K, kway=200)
        torch.manual_seed(0)
        m = _ChunkedRetention(16).double()
        x = torch.randn(2, T, 16, dtype=torch.float64)
        term = TorchSource().to_ir(m, x)[0].root
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(
            reach,
            root,
            max_iterations=16,
            max_nodes=150_000,
            patience=8,
            rule_budgets=budgets,
        )
        reach_best = dag_cost(
            eg.extract_best(eg.find(root), flops_cost), flops_cost
        )
        eg.run(
            reassoc,
            root,
            max_iterations=16,
            max_nodes=150_000,
            patience=8,
            rule_budgets=budgets,
        )
        best = eg.extract_best(eg.find(root), flops_cost)
        staged_cost = dag_cost(best, flops_cost)
        assert staged_cost < reach_best

        reads = self._fold_reads(best)
        assert len(reads) >= K - 2
        states = [r.args[1] for r in reads]
        ids = [self._subtree_ids(s) for s in states]
        chained = any(
            id(sj) in si_ids
            for i, si_ids in enumerate(ids)
            for j, sj in enumerate(states)
            if i != j
        )
        assert chained, (
            "no row-block state nests another — the extracted cross "
            "sum is a per-pair spelling, not the shared recurrence"
        )

    @pytest.mark.slow

    def test_extracted_carrier_verifies_fp64(self):
        """The extracted shared-state program lowers and verifies."""
        pytest.importorskip("mlopt_torch")
        from mlopt_core.cost import flops_cost
        from mlopt_core.ir import IR
        from mlopt_discovery.zoo import _ChunkedRetention
        from mlopt_torch.adapters import TorchSink, TorchSource

        T, K = 64, 4
        reach, reassoc = om_shared_state_stages(K)
        budgets = om_shared_state_budgets(K, kway=200)
        torch.manual_seed(0)
        model = _ChunkedRetention(16).double()
        x = torch.randn(2, T, 16, dtype=torch.float64)
        ir, leaves = TorchSource().to_ir(model, x)
        eg = EGraph()
        root = eg.add_term(ir.root)
        for stage in (reach, reassoc):
            eg.run(
                stage,
                root,
                max_iterations=16,
                max_nodes=150_000,
                patience=8,
                rule_budgets=budgets,
            )
        best = eg.extract_best(eg.find(root), flops_cost)
        ops = set()
        stack = [best]
        while stack:
            t = stack.pop()
            if isinstance(t, Op):
                ops.add(t.op)
                stack.extend(t.args)
        sink = TorchSink()
        assert not (ops - sink.supported_ops)
        ex_ir = IR(
            root=best,
            inputs=list(ir.inputs),
            input_names=set(ir.input_names),
            params=dict(ir.params),
        )
        module = sink.lower(ex_ir, params=leaves)
        rep = sink.verify(model, module, (x,))
        assert rep.passed and rep.max_abs < 1e-9


def _op_skeleton(term):
    """Return *term*'s op tree with every leaf collapsed to ``"@"``.

    Two patterns with the same skeleton are the same *shape* of
    expression — alpha-equivalent up to metavariable names.
    """
    if not isinstance(term, Op):
        return "@"
    return (term.op, tuple(_op_skeleton(a) for a in term.args))


def _shared_state_classes(eg, root) -> int:
    """Carrier-state e-classes reachable from *root* with ≥2 parents."""
    seen: set[int] = set()
    stack = [eg.find(root)]
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        for n in eg.get_class(cid).nodes:
            stack.extend(eg.find(c) for c in n.children)
    out = 0
    for cid in seen:
        if not any(
            n.op in ("applyd", "aff_diag")
            for n in eg.get_class(cid).nodes
        ):
            continue
        parents = {eg.find(p) for p in eg._parents.get(cid, ())}
        if len(parents) >= 2:
            out += 1
    return out


def _fold_nesting(term):
    """Max ``applyd``/``affd_compose`` nesting of a decoded term."""
    memo: dict[int, int] = {}

    def depth(t, stack):
        if not isinstance(t, Op):
            return 0
        key = id(t)
        if key in memo:
            return memo[key]
        if key in stack:
            return 0
        stack = stack | {key}
        child = max((depth(a, stack) for a in t.args), default=0)
        best = child + (t.op in ("applyd", "affd_compose"))
        memo[key] = best
        return best

    return depth(term, frozenset())


def _finite_fold_depth(eg, root):
    """Deepest *finite* fold nesting over the root class's members.

    Buildability is the least fixpoint (a class is buildable when some
    member's children are all buildable); the depth is then the
    level-by-level reachability of a fold-headed member.  Unlike a
    naive e-graph walk this cannot be inflated by cycles — a
    rewrite-introduced ``applyd`` chain can revisit a class, which no
    finite term ever does.
    """
    fold_ops = ("applyd", "affd_compose")
    build: set = set()
    changed = True
    while changed:
        changed = False
        for cid, cls in eg._classes.items():
            cid = eg.find(cid)
            if cid in build:
                continue
            for n in cls.nodes:
                if n.op == "leaf" or all(
                    eg.find(c) in build for c in n.children
                ):
                    build.add(cid)
                    changed = True
                    break
    reach: list[set] = [set(build)]
    d = 1
    while d <= 64:
        prev = reach[d - 1]
        cur: set = set()
        for cid, cls in eg._classes.items():
            cid = eg.find(cid)
            for n in cls.nodes:
                if n.op not in fold_ops:
                    continue
                if not all(eg.find(c) in build for c in n.children):
                    continue
                if any(eg.find(c) in prev for c in n.children):
                    cur.add(cid)
                    break
        if not cur:
            break
        # Non-fold members propagate the same level.
        closing = True
        while closing:
            closing = False
            for cid, cls in eg._classes.items():
                cid = eg.find(cid)
                if cid in cur:
                    continue
                for n in cls.nodes:
                    if n.op == "leaf" or n.op in fold_ops:
                        continue
                    if not all(eg.find(c) in build for c in n.children):
                        continue
                    if any(eg.find(c) in cur for c in n.children):
                        cur.add(cid)
                        closing = True
                        break
        reach.append(cur)
        d += 1
    rc = eg.find(root)
    return max(
        (lvl for lvl, s in enumerate(reach) if rc in s), default=0
    )


def _block(x, i, dim, sz):
    """Block ``i`` of a uniform ``sz`` split on axis ``dim``."""
    return Op.make("split", x, sizes=sz, dim=dim, index=i)


def _shared_state_fold(T, C, K, d):
    """The paying fold — a shared per-chunk state recurrence.

    ``out_i = intra_i + xi0 ⊙ (q_i @ S_{i-1})`` with ``S_i =
    applyd(aff_diag(g^C, (eta_i ⊙ k_i)ᵀ v_i), S_{i-1})``: the state
    ``S_{i-1}`` is ONE term consumed by block ``i``'s output and block
    ``i+1``'s step, so the whole fold is O(T·C).  Under the staged pair
    (``om_shared_state_stages``) the e-graph DOES produce and select
    this spelling — see ``TestSharedStateExtraction`` — in its
    ``add(mul(g,S),B)`` form (``applyd`` is a cost-neutral respelling
    the extractor never prefers).
    """
    sz = (C,) * K
    q, k, v, s = (
        _v("q", T, d),
        _v("k", T, d),
        _v("v", T, d),
        _v("s", T, T),
    )
    g, a = _v("g"), _v("a")  # a := g^C (a compile-time scalar)
    eta = Op.make("decay_vec", g, t=C, sign=-1)
    xi0 = Op.make("decay_vec", g, t=C)
    S = _v("S0", d, d)
    blocks = []
    for i in range(K):
        qi, ki, vi = (
            _block(q, i, -2, sz),
            _block(k, i, -2, sz),
            _block(v, i, -2, sz),
        )
        intra = Op.make(
            "om_decayed",
            Op.make(
                "split",
                _block(s, i, -2, sz),
                sizes=sz,
                dim=-1,
                index=i,
            ),
            Op.make("decay", g, t=C),
            vi,
        )
        s_prev = S
        b = Op.make(
            "matmul",
            Op.make(
                "transpose",
                Op.make("mul", ki, Op.make("unsqueeze", eta, dim=-1)),
                dim0=-2,
                dim1=-1,
            ),
            vi,
        )
        S = Op.make("applyd", Op.make("aff_diag", a, b), s_prev)
        cross = Op.make(
            "mul",
            Op.make("unsqueeze", xi0, dim=-1),
            Op.make("matmul", qi, s_prev),
        )
        blocks.append(Op.make("add", intra, cross))
    out = blocks[-1]
    for blk in reversed(blocks[:-1]):
        out = Op.make("concat", blk, out, dim=-2)
    return out


def _spelled_cross(T, C, K, d):
    """The explicit factored cross sum extraction actually selects."""
    sz = (C,) * K
    v, s = _v("v", T, d), _v("s", T, T)
    g = _v("g")
    eta = Op.make("decay_vec", g, t=C, sign=-1)
    blocks = []
    for i in range(K):
        vi = _block(v, i, -2, sz)
        si = Op.make(
            "split",
            _block(s, i, -2, sz),
            sizes=sz,
            dim=-1,
            index=i,
        )
        acc = Op.make("om_decayed", si, Op.make("decay", g, t=C), vi)
        for j in range(i):
            xi = Op.make("decay_vec", g, t=C, offset=(i - j) * C)
            sj = Op.make(
                "split",
                _block(s, i, -2, sz),
                sizes=sz,
                dim=-1,
                index=j,
            )
            cross = Op.make(
                "mul",
                Op.make("unsqueeze", xi, dim=-1),
                Op.make(
                    "matmul",
                    Op.make(
                        "mul", sj, Op.make("unsqueeze", eta, dim=-2)
                    ),
                    _block(v, j, -2, sz),
                ),
            )
            acc = Op.make("add", acc, cross)
        blocks.append(acc)
    out = blocks[-1]
    for blk in reversed(blocks[:-1]):
        out = Op.make("concat", blk, out, dim=-2)
    return out


def _walk_ops(term, out=None, seen=None):
    """Yield every ``Op`` node of a term DAG once (cycle-safe)."""
    if out is None:
        out, seen = [], set()
    if not isinstance(term, Op) or id(term) in seen:
        return out
    seen.add(id(term))
    out.append(term)
    for a in term.args:
        _walk_ops(a, out, seen)
    return out
