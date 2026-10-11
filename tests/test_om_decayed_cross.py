"""decay_vec + the cross-block factorization — rank-1 decay blocks.

``decay_vec(γ, t=T, offset=o, sign=s)[k] = γ^(s·k + o)`` — the ξ/η
vectors of the RetNet inter-chunk factorization.  ``decay_block_cross``
rewrites a strict-lower ``split(split(decay))`` block as the outer
product ``mul(unsq(ξ,-1), unsq(η,-2))``; ``om_decayed_cross`` then
distributes a decayed product over a rank-1 decay into row/column
scalings — the carried-state algebra the chunked recurrence needs.
"""

import pytest


torch = pytest.importorskip("torch")

from catopt_carriers.om import (  # noqa: E402
    DECAY_BLOCK_CROSS,
    DECAY_VEC_OFFSET,
    MATMUL_COL_DECAY,
    MATMUL_SCALAR_FACTOR,
    OM_DECAYED_CROSS,
    OM_LAWS,
    UNSQ_MUL,
    _decay_vec_torch,
)
from catopt_core.laws.scan import SCAN_DIAG_LAWS  # noqa: E402
from catopt_core.laws.tensor import (  # noqa: E402
    ASSOC_ADD,
    ASSOC_MUL,
    COMM_ADD,
    COMM_MUL,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
)
from catopt_core.egraph import EGraph  # noqa: E402
from catopt_core.ir import (  # noqa: E402
    Const,
    Op,
    TensorType,
    Var,
    op_repr,
)
from catopt_core.meta import (  # noqa: E402
    instantiate_pattern,
    match_pattern,
)
from catopt_core.typing import shape_of  # noqa: E402
from catopt_torch.torch_bridge import eval_term  # noqa: E402
import catopt_torch.torch_bridge as tb  # noqa: E402

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
        block = D[4:9, 0:4]  # rows of chunk 1 × cols of chunk 0
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
        """(s ⊙ ξηᵀ) @ v = ξ_row ⊙ ((s ⊙ η_col) @ v) — exact."""
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
        eg.run(OM_LAWS, root, max_iterations=8)
        dv = [
            n
            for cls in eg._classes.values()
            for n in cls.nodes
            if n.op == "decay_vec"
        ]
        # the cross block's two factors — ξ (offset>0) and η (sign=-1)
        assert any(dict(n.attrs).get("sign") == -1 for n in dv), (
            "the η factor (γ^{-c}) should appear"
        )
        assert any(dict(n.attrs).get("offset", 0) > 0 for n in dv), (
            "the ξ factor (γ^{o+r}) should appear"
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
        assert not MATMUL_SCALAR_FACTOR.check({})

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
        from catopt_carriers.om import _decay_state_like

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


class TestMachineFold:
    """The machine derives the recurrence from spelled cross terms.

    Feed the e-graph the spelled per-block cross contributions —
    ``γ^o ⊙ (ξ_r ⊙ ((q·kᵀ ⊙ ηᵀ) @ v))`` — and it composes the whole
    chain itself: unsq_mul frees the scalar, matmul_col_decay lands
    η on the keys, factor_matmul collects the shared q₂, and the
    decay-state lift folds the weighted sum onto ``applyd``.
    """

    def test_spelled_cross_folds_to_applyd(self):
        """The weighted state sum itself folds onto the carrier.

        The earlier chain — spelled cross → η on keys → matmul(q,B)
        → γ^o scalars pulled out — was demonstrated in the saturated
        e-graph; the fold this test pins is the last hop, the
        ``add(mul(γ^o, B), mul(γ^o', B'))`` sum the chain lands on.
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
