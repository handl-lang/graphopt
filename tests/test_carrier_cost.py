"""Carrier-member costing — honest flops for the applyd family.

``applyd(aff_diag(a, x), h)`` evaluates ``a⊙h + x`` — elementwise over
the broadcast of all three factors, priced at ``2·numel(broadcast)``
(one mul + one add).  ``affd_compose`` is the pointwise pair
composition (``3·numel`` of the map's convention shape); the carrier
*packaging* ops (``aff``/``aff_diag``/``om``/``omd``) and the
``affd_*``/``aff_*`` pair-slot projections carry no flops at all — the
members' subtrees already bill the work.

The chunked-vs-fused comparison pins the asymptotic story the carrier
exists for: a fused ``om_decayed`` on a ``(T,T)`` decay is ``O(T²·d)``;
the chunked carrier member — per-chunk intra ``om_decayed`` plus the
``applyd`` state fold — is ``O(T·C·d) + O(T·d²)`` and must price lower
once ``T`` clears ~``2·C`` — and lose at small ``T``, which is the
honest direction (no carrier discount is invented).
"""

import pytest
from catopt_core.cost.basic import (
    _INVALID_COST,
    _flops_of,
    flops_cost,
    launch_aware_cost,
)
from catopt_core.cost.params import dag_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, TensorType, Var
from catopt_core.typing import _shape_of


def _v(name, *shape):
    return Var(name, TensorType(shape))


# ---------------------------------------------------------------------------
#  applyd — the elementwise state step
# ---------------------------------------------------------------------------


class TestApplydCost:
    def test_elementwise_over_the_state(self):
        """a broadcasts onto h: work is 2·numel(h) — mul + add."""
        t = Op.make(
            "applyd",
            Op.make("aff_diag", _v("a", 1, 8), _v("x", 4, 8)),
            _v("h", 4, 8),
        )
        assert _flops_of(t) == pytest.approx(2 * 4 * 8)

    def test_prices_the_broadcast_not_just_h(self):
        """a (2,4) ⊙ h (4,) materialises (2,4) — the convention shape
        (h's) under-reports; the flops bill the true broadcast."""
        t = Op.make(
            "applyd",
            Op.make("aff_diag", _v("a", 2, 4), _v("x", 2, 4)),
            _v("h", 4),
        )
        assert _flops_of(t) == pytest.approx(2 * 2 * 4)

    def test_contribution_joins_the_broadcast(self):
        """x (3,4) against a⊙h (4,) → the add runs over (3,4)."""
        t = Op.make(
            "applyd",
            Op.make("aff_diag", _v("a", 4), _v("x", 3, 4)),
            _v("h", 4),
        )
        assert _flops_of(t) == pytest.approx(2 * 3 * 4)

    def test_ill_typed_contribution_poisons(self):
        """x that cannot add to a⊙h is provably ill-typed — the
        carrier member must never win an extraction."""
        t = Op.make(
            "applyd",
            Op.make("aff_diag", _v("a", 4), _v("x", 5)),
            _v("h", 4),
        )
        assert _flops_of(t) >= _INVALID_COST
        assert flops_cost(t) >= _INVALID_COST

    def test_incompatible_map_state_broadcast_poisons(self):
        """An opaque map whose scale can't broadcast onto h."""
        t = Op.make("applyd", _v("f", 2, 4), _v("h", 3))
        assert _flops_of(t) >= _INVALID_COST

    def test_map_shape_unknown_falls_back(self):
        """Unshapeable operands price flat on the reported shape."""
        f = _v("f")  # scalar-shape leaf — unknown rank
        h = _v("h")
        t = Op.make("applyd", f, h)
        assert _flops_of(t) == pytest.approx(2 * 1)

    def test_compose_map_uses_convention_shape(self):
        """applyd over an affd_compose chain: the map's convention
        shape IS the scale part — broadcast against h only."""
        leaves = [
            Op.make("aff_diag", _v(f"a{i}", 8), _v(f"b{i}", 8))
            for i in range(3)
        ]
        tree = Op.make(
            "affd_compose",
            leaves[0],
            Op.make("affd_compose", *leaves[1:]),
        )
        t = Op.make("applyd", tree, _v("h", 8))
        assert _flops_of(t) == pytest.approx(2 * 8)

    def test_carrier_parity_with_spelled(self):
        """a⊙h+x as applyd ≡ the spelled mul+add in raw FLOPs, and
        wins one kernel launch (the packaging emits none)."""
        a, x, h = _v("a", 4, 8), _v("x", 4, 8), _v("h", 4, 8)
        carrier = Op.make("applyd", Op.make("aff_diag", a, x), h)
        spelled = Op.make("add", Op.make("mul", a, h), x)
        assert flops_cost(carrier) == pytest.approx(flops_cost(spelled))
        assert launch_aware_cost(carrier) < launch_aware_cost(spelled)


# ---------------------------------------------------------------------------
#  Packaging / projections — zero-FLOP plumbing
# ---------------------------------------------------------------------------


class TestCarrierPlumbing:
    def test_packaging_is_free(self):
        """aff/aff_diag/om/omd assemble tuples — no arithmetic."""
        f, m, l_, a, b = (
            _v("f", 4),
            _v("m", 4, 8),
            _v("l", 4, 8),
            _v("a", 4, 8),
            _v("b", 4, 8),
        )
        assert _flops_of(Op.make("aff", _v("A", 8, 8), f)) == 0.0
        assert _flops_of(Op.make("aff_diag", f, f)) == 0.0
        assert _flops_of(Op.make("om", m, l_, a)) == 0.0
        assert _flops_of(Op.make("omd", m, l_, a, b)) == 0.0

    def test_projections_are_free(self):
        """affd_a/affd_b/aff_A/aff_b read tuple slots — f[0]/f[1]."""
        f = _v("f", 8)
        for op in ("affd_a", "affd_b", "aff_A", "aff_b"):
            assert _flops_of(Op.make(op, f)) == 0.0, op

    def test_affd_compose_is_three_passes(self):
        """(a2,b2)∘(a1,b1) = (a2⊙a1, a2⊙b1 + b2) — mul+mul+add."""
        f1 = _v("f1", 8)
        assert _flops_of(
            Op.make("affd_compose", f1, f1)
        ) == pytest.approx(3 * 8)


# ---------------------------------------------------------------------------
#  om_decayed — the fused decayed product
# ---------------------------------------------------------------------------


class TestOmDecayedCost:
    def test_fused_is_quadratic_in_T(self):
        """(s ⊙ d) @ v: one pass over the (T,T) plane + the GEMM."""
        t = Op.make(
            "om_decayed", _v("s", 8, 8), _v("d", 8, 8), _v("v", 8, 4)
        )
        assert _flops_of(t) == pytest.approx(8 * 8 + 2 * (8 * 4) * 8)


# ---------------------------------------------------------------------------
#  The point: chunked carrier vs fused at scale
# ---------------------------------------------------------------------------


def _fused_decayed(t: int, d: int) -> Op:
    """``om_decayed(q·kᵀ, decay(g,T), v)`` — the fused (T,d) output."""
    q, k, v, g = _v("q", t, d), _v("k", t, d), _v("v", t, d), _v("g")
    s = Op.make("matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1))
    return Op.make("om_decayed", s, Op.make("decay", g, t=t), v)


def _chunked_decayed(t: int, c: int, d: int) -> Op:
    """The factored chunked form, ``(T,d)``, on the applyd carrier.

    Per chunk ``i`` of size ``C``: the intra block
    ``om_decayed(s_i, decay(g,C), v_i)`` plus the cross readout
    ``ξ_i ⊙ (q_i @ B_i)`` against the carried ``(d,d)`` state
    ``B_{i+1} = applyd(aff_diag(g^C, (k_i⊙η)ᵀ·v_i), B_i)`` — the
    applyd/aff_diag members the decay-state lift produces.
    """
    q, k, v, g = _v("q", t, d), _v("k", t, d), _v("v", t, d), _v("g")
    n = t // c
    sizes = (c,) * n

    def block(base, i):
        return Op.make("split", base, sizes=sizes, dim=-2, index=i)

    state = _v("h0", d, d)
    rows = []
    for i in range(n):
        qi, ki, vi = block(q, i), block(k, i), block(v, i)
        si = Op.make(
            "matmul", qi, Op.make("transpose", ki, dim0=-2, dim1=-1)
        )
        intra = Op.make("om_decayed", si, Op.make("decay", g, t=c), vi)
        # cross readout: xi_i ⊙ (q_i @ B_i)
        xi = Op.make("decay_vec", g, t=c, offset=1)
        cross = Op.make(
            "mul",
            Op.make("unsqueeze", xi, dim=-1),
            Op.make("matmul", qi, state),
        )
        rows.append(Op.make("add", intra, cross))
        # state fold: B_{i+1} = g^C ⊙ B_i + (k_i⊙η)ᵀ @ v_i
        a_i = Op.make("decay_vec", g, t=1, offset=c)
        eta = Op.make("decay_vec", g, t=c, sign=-1, offset=c - 1)
        kdec = Op.make("mul", ki, Op.make("unsqueeze", eta, dim=-1))
        x_i = Op.make(
            "matmul",
            Op.make("transpose", kdec, dim0=-2, dim1=-1),
            vi,
        )
        state = Op.make("applyd", Op.make("aff_diag", a_i, x_i), state)
    return Op.make("concat", *rows, dim=-2)


class TestChunkedVsFused:
    T, C, D = 512, 16, 64

    def test_same_output_shape(self):
        fused = _fused_decayed(self.T, self.D)
        chunked = _chunked_decayed(self.T, self.C, self.D)
        assert (
            _shape_of(fused) == _shape_of(chunked) == (self.T, self.D)
        )

    def test_fused_formulas(self):
        """The honest asymptotics: T²·d fused vs T·(C+d)·d chunked."""
        fused = _fused_decayed(self.T, self.D)
        chunked = _chunked_decayed(self.T, self.C, self.D)
        assert dag_cost(fused, flops_cost) > 6 * dag_cost(
            chunked, flops_cost
        )

    def test_carrier_member_wins_at_scale(self):
        """T=512 ≫ 2·C: the chunked applyd member prices lower."""
        fused = _fused_decayed(self.T, self.D)
        chunked = _chunked_decayed(self.T, self.C, self.D)
        assert dag_cost(chunked, flops_cost) < dag_cost(
            fused, flops_cost
        )
        assert dag_cost(chunked, launch_aware_cost) < dag_cost(
            fused, launch_aware_cost
        )

    def test_carrier_member_loses_at_small_T(self):
        """T=32 ≈ 2·C: honestly, the fused element wins — the carrier
        path is not discounted into a fake victory."""
        fused = _fused_decayed(32, self.D)
        chunked = _chunked_decayed(32, self.C, self.D)
        assert dag_cost(fused, flops_cost) < dag_cost(
            chunked, flops_cost
        )

    def test_extraction_prefers_the_carrier(self):
        """Unioned into one class, extract_best picks the chunked
        member at T=512 — the machinery reaching for the carrier
        when it actually wins."""
        fused = _fused_decayed(self.T, self.D)
        chunked = _chunked_decayed(self.T, self.C, self.D)
        eg = EGraph()
        root = eg.add_term(fused)
        eg.union(root, eg.add_term(chunked))
        best = eg.extract_best(root, flops_cost)
        assert best is not None and best.op == "concat"
        # and at small T the fused member must come back
        fused_s = _fused_decayed(32, self.D)
        chunked_s = _chunked_decayed(32, self.C, self.D)
        eg2 = EGraph()
        root2 = eg2.add_term(fused_s)
        eg2.union(root2, eg2.add_term(chunked_s))
        best2 = eg2.extract_best(root2, flops_cost)
        assert best2 is not None and best2.op == "om_decayed"
