"""Coverage tests for mlopt_carriers.decayed_lower.

The pairwise-batched chunked-``om_decayed`` executor: the plan
parser (concat arms, summand peels, state-half decomposition,
uniformity gates), the batched module's runtime paths (batched
scores, shared vs distinct decay masks, gated/ungated intra rows,
parsed vs eval-mode state halves, missing row gates, const/scale
caching, the eval fallback), and the end-to-end equivalence with the
generic ``IRModule`` on a hand-built chunked term.  All CPU; the
CUDA-graph arm is marked ``requires_cuda``.
"""
# ruff: noqa: E731 — test-idiom lambdas

import pytest
import torch
from mlopt_carriers.decayed_lower import (
    ChunkedDecayModule,
    _classify_body,
    _decay_row,
    _dedup,
    _flat_same,
    _parse_arm,
    _peel_summand,
    _score_pair,
    _state_half,
    _uniform,
    build_decayed_plan,
    is_decayed_chunked_term,
    to_chunked_decayed_module,
)
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var
from mlopt_torch.adapters import TorchSink
from mlopt_torch.torch_bridge import IRModule


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


# ---------------------------------------------------------------------------
#  Term builder — the chunked decayed-scan pattern, spelled out
# ---------------------------------------------------------------------------

B, C, D, K = 2, 4, 4, 3


class _Builder:
    """A K=3 chunked decayed term over a ``(B, K*C, D)`` input."""

    def __init__(self) -> None:
        self.x = _v("x", B, K * C, D)
        self.wq, self.wk, self.wv = (
            _p("wq", D, D),
            _p("wk", D, D),
            _p("wv", D, D),
        )
        self.bq, self.bk, self.bv = (
            _p("bq", D),
            _p("bk", D),
            _p("bv", D),
        )
        self.g = _p("g")
        self.scale = Const(2.0)
        self.q = Op.make("linear", self.x, self.wq, self.bq)
        self.k = Op.make("linear", self.x, self.wk, self.bk)
        self.v = Op.make("linear", self.x, self.wv, self.bv)

    def qi(self, i):
        return Op.make("split", self.q, dim=-2, index=i, sizes=[C] * K)

    def ki(self, i):
        return Op.make("split", self.k, dim=-2, index=i, sizes=[C] * K)

    def kti(self, i):
        return Op.make(
            "split",
            Op.make("transpose", self.k, dim0=-1, dim1=-2),
            dim=-1,
            index=i,
            sizes=[C] * K,
        )

    def vi(self, i):
        return Op.make("split", self.v, dim=-2, index=i, sizes=[C] * K)

    def dv(self, sign=1, offset=0):
        return Op.make(
            "decay_vec", self.g, t=C, sign=sign, offset=offset
        )

    def row(self, offset):
        return Op.make("unsqueeze", self.dv(offset=offset), dim=-1)

    def col(self):
        return Op.make("unsqueeze", self.dv(sign=-1), dim=-1)

    def decay(self):
        return Op.make("decay", self.g, t=C)

    def cross(self, i, j, *, gate="row", state="parsed"):
        """A decay-gated pair product summand."""
        if state == "parsed":
            a = Op.make(
                "transpose",
                Op.make("mul", self.ki(j), self.col()),
                dim0=-1,
                dim1=-2,
            )
        elif state == "no_mul":
            a = Op.make("transpose", self.ki(j), dim0=-1, dim1=-2)
        else:  # "add" — not a transpose at all
            a = Op.make("add", self.ki(j), self.ki(j))
        inner = Op.make("matmul", a, self.vi(j))
        body = Op.make("matmul", self.qi(i), inner)
        if gate == "row":
            return Op.make(
                "mul",
                self.row((i - j) * C),
                Op.make("div", body, self.scale),
            )
        if gate == "row_rev":  # mul operands swapped
            return Op.make(
                "mul",
                Op.make("div", body, self.scale),
                self.row((i - j) * C),
            )
        if gate == "row_in":  # gate inside the div
            return Op.make(
                "div",
                Op.make("mul", self.row((i - j) * C), body),
                self.scale,
            )
        return Op.make("div", body, self.scale)  # gateless

    def arm(self, i, *, gate="row", score=None, decay=None, row=False):
        """Chunk i's arm: one intra element + i cross summands."""
        sc = (
            score
            if score is not None
            else Op.make("matmul", self.qi(i), self.kti(i))
        )
        dec = decay if decay is not None else self.decay()
        intra = Op.make("om_decayed", sc, dec, self.vi(i))
        t = Op.make("div", intra, self.scale)
        if row:
            t = Op.make("mul", self.row(C), t)
        for j in range(i):
            t = Op.make("add", t, self.cross(i, j, gate=gate))
        return t

    def root(self, **kw):
        return Op.make(
            "concat",
            *[self.arm(i, **kw) for i in range(K)],
            dim=-2,
        )

    def ir(self, root):
        params = {
            p.name: p
            for p in (
                self.wq,
                self.wk,
                self.wv,
                self.bq,
                self.bk,
                self.bv,
                self.g,
            )
        }
        return IR(
            root=root,
            inputs=[self.x],
            input_names={"x"},
            params=params,
        )

    def values(self, dtype=torch.float64):
        torch.manual_seed(0)
        return {
            "wq": torch.randn(D, D, dtype=dtype),
            "wk": torch.randn(D, D, dtype=dtype),
            "wv": torch.randn(D, D, dtype=dtype),
            "bq": torch.randn(D, dtype=dtype),
            "bk": torch.randn(D, dtype=dtype),
            "bv": torch.randn(D, dtype=dtype),
            "g": torch.tensor(0.9, dtype=dtype),
        }, torch.randn(B, K * C, D, dtype=dtype)


def _assert_matches(term_root: Op, extra_ir=None, extra_vals=None):
    """Batched module output == generic IRModule output."""
    b = _Builder()
    ir = b.ir(term_root) if extra_ir is None else extra_ir
    vals, x = b.values()
    vals.update(extra_vals or {})
    fused = to_chunked_decayed_module(ir, param_values=vals)
    generic = IRModule(ir, param_values=vals)
    with torch.no_grad():
        out_f, out_g = fused(x), generic(x)
    torch.testing.assert_close(out_f, out_g, rtol=1e-9, atol=1e-9)
    return fused


# ---------------------------------------------------------------------------
#  Small plumbing units
# ---------------------------------------------------------------------------


def test_flat_same_concat_and_add():
    x = _v("x", 4)
    a = Op.make("add", x, x)
    b = Op.make("add", a, x)
    assert len(_flat_same(b, "add")) == 3
    assert _flat_same(x, "add") == [x]
    c = Op.make("concat", a, b, dim=0)
    assert len(_flat_same(c, "concat", 0)) == 2
    # wrong dim → not flattened
    assert _flat_same(c, "concat", 1) == [c]


def test_decay_row_matches_only_unsqueezed_decay_vec():
    g = _p("g")
    dv = Op.make("decay_vec", g, t=4)
    u = Op.make("unsqueeze", dv, dim=-1)
    assert _decay_row(u) is u
    assert _decay_row(dv) is None  # missing unsqueeze
    other = Op.make("unsqueeze", _v("z", 4), dim=-1)
    assert _decay_row(other) is None
    assert _decay_row(_v("z", 4)) is None


def test_peel_summand_forms_and_declines():
    g = _p("g")
    s = Op.make("matmul", _v("s", 2, 4, 4), _v("k", 2, 4, 4))
    u = Op.make("unsqueeze", Op.make("decay_vec", g, t=4), dim=-1)
    # div(mul(u, s), c): gate inside the div
    t = Op.make("div", Op.make("mul", u, s), Const(2.0))
    assert _peel_summand(t) == (u, s, Const(2.0))
    # mul(s, u): swapped operand order
    t2 = Op.make("mul", s, u)
    assert _peel_summand(t2) == (u, s, None)
    # second gate → decline
    t3 = Op.make("mul", u, Op.make("mul", u, s))
    assert _peel_summand(t3) is None
    # div with an Op denominator is not a scale: left as body
    t4 = Op.make("div", s, u)
    row, body, sc = _peel_summand(t4)
    assert row is None and body is t4 and sc is None
    # mul with no gate → body is the mul itself
    t5 = Op.make("mul", s, s)
    row, body, sc = _peel_summand(t5)
    assert row is None and body is t5 and sc is None
    # non-Op → None
    assert _peel_summand(Const(1.0)) is None


def test_classify_body_forms():
    s, d, v, q = (_v(n, 2, 4, 4) for n in ("s", "d", "v", "q"))
    omd = Op.make("om_decayed", s, d, v)
    assert _classify_body(omd) == ("intra", (s, d, v))
    a, vv = _v("a", 2, 4, 4), _v("vv", 2, 4, 4)
    inner = Op.make("matmul", a, vv)
    cross = Op.make("matmul", q, inner)
    assert _classify_body(cross) == ("cross", (q, inner))
    assert _classify_body(Op.make("matmul", q, v)) is None
    assert _classify_body(Op.make("add", q, v)) is None


def test_state_half_parsed_and_eval_modes():
    g = _p("g")
    kj, vj = _v("kj", 2, 4, 4), _v("vj", 2, 4, 4)
    u = Op.make("unsqueeze", Op.make("decay_vec", g, t=4), dim=-1)
    # parsed: transpose(mul(k, u))
    a = Op.make("transpose", Op.make("mul", kj, u), dim0=-1, dim1=-2)
    inner = Op.make("matmul", a, vj)
    assert _state_half(inner) == (kj, u, vj, -1, -2)
    # mul on the other side
    a2 = Op.make("transpose", Op.make("mul", u, kj), dim0=-1, dim1=-2)
    assert _state_half(Op.make("matmul", a2, vj)) == (kj, u, vj, -1, -2)
    # eval modes
    assert _state_half(Op.make("matmul", kj, vj)) == (
        kj,
        None,
        vj,
        None,
        None,
    )
    a3 = Op.make("transpose", Op.make("add", kj, kj), dim0=-1, dim1=-2)
    assert _state_half(Op.make("matmul", a3, vj)) == (
        a3,
        None,
        vj,
        None,
        None,
    )
    a4 = Op.make("transpose", Op.make("mul", kj, kj), dim0=-1, dim1=-2)
    assert _state_half(Op.make("matmul", a4, vj)) == (
        a4,
        None,
        vj,
        None,
        None,
    )


def test_uniform_dedup_score_pair():
    assert _uniform([(2, 3), (2, 3)]) == (2, 3)
    assert _uniform([(2, 3), (2, 4)]) is None
    assert _uniform([None, (2, 3)]) is None
    assert _uniform([(2, None), (2, None)]) is None
    a, b = _v("a", 2), _v("b", 2)
    terms, idx = _dedup([a, b, a])
    assert terms == [a, b] and idx[b] == 1
    m = Op.make("matmul", a, b)
    assert _score_pair(m) == (a, b)
    assert _score_pair(Op.make("add", a, b)) is None


# ---------------------------------------------------------------------------
#  Plan acceptance / decline
# ---------------------------------------------------------------------------


def test_accepts_clean_pattern():
    b = _Builder()
    root = b.root()
    assert is_decayed_chunked_term(root)
    plan = build_decayed_plan(root)
    assert plan["n"] == K and plan["dim"] == -2
    assert len(plan["pairs"]) == K * (K - 1) // 2


def test_declines_non_concat_and_bad_dim():
    b = _Builder()
    assert not is_decayed_chunked_term(b.q)
    assert build_decayed_plan(_v("x", 4)) is None
    no_dim = Op.make("concat", b.arm(0), b.arm(1), validate=False)
    assert build_decayed_plan(no_dim) is None


def test_declines_arm_without_intra():
    b = _Builder()
    only_cross = b.cross(1, 0)
    root = Op.make("concat", b.arm(0), only_cross, dim=-2)
    assert build_decayed_plan(root) is None


def test_declines_double_intra_and_bad_summand():
    b = _Builder()
    intra = Op.make(
        "om_decayed",
        Op.make("matmul", b.qi(1), b.kti(1)),
        b.decay(),
        b.vi(1),
    )
    two = Op.make("add", intra, intra)
    root = Op.make("concat", b.arm(0), two, dim=-2)
    assert build_decayed_plan(root) is None
    # a non-Op summand
    weird = Op.make("add", b.arm(0), Const(1.0))
    assert _parse_arm(weird) is None
    # a mul with no gate is not a known body
    mm = Op.make(
        "matmul", b.qi(1), Op.make("matmul", b.kti(1), b.vi(1))
    )
    bad = Op.make("add", b.arm(0), Op.make("mul", mm, mm))
    assert _parse_arm(bad) is None
    # a body that is neither om_decayed nor pair-product
    bad2 = Op.make("add", b.arm(0), Op.make("neg", mm))
    assert _parse_arm(bad2) is None


def test_declines_nonuniform_shapes():
    b = _Builder()
    # v of a different width in one arm → non-uniform chunk shape
    v_wide = _v("v_wide", B, C, D + 1)
    arm1 = Op.make(
        "div",
        Op.make(
            "om_decayed",
            Op.make("matmul", b.qi(1), b.kti(1)),
            b.decay(),
            v_wide,
        ),
        b.scale,
    )
    root = Op.make("concat", b.arm(0), arm1, b.arm(2), dim=-2)
    assert build_decayed_plan(root) is None


def test_declines_nonuniform_cross_shapes():
    b = _Builder()
    # a second q with a different width → non-uniform q shapes
    q_wide = _v("q_wide", B, C, D + 1)
    a = Op.make(
        "transpose", Op.make("mul", b.ki(0), b.col()), dim0=-1, dim1=-2
    )
    inner = Op.make("matmul", a, b.vi(0))
    body = Op.make("matmul", q_wide, inner)
    extra = Op.make("mul", b.row(C), Op.make("div", body, b.scale))
    arm2 = Op.make("add", b.arm(2), extra)
    root = Op.make("concat", b.arm(0), b.arm(1), arm2, dim=-2)
    assert build_decayed_plan(root) is None


def test_declines_nonuniform_inner_shapes():
    b = _Builder()
    # inner matmul of a different width → non-uniform state shapes
    w = _v("w", B, C, D)
    v_wide = _v("v_wide", B, C, D + 1)
    inner = Op.make("matmul", w, v_wide)
    body = Op.make("matmul", b.qi(2), inner)
    extra = Op.make("mul", b.row(C), Op.make("div", body, b.scale))
    arm2 = Op.make("add", b.arm(2), extra)
    root = Op.make("concat", b.arm(0), b.arm(1), arm2, dim=-2)
    assert build_decayed_plan(root) is None


# ---------------------------------------------------------------------------
#  Batched execution — end-to-end equivalence
# ---------------------------------------------------------------------------


def test_batched_matches_generic_clean():
    b = _Builder()
    fused = _assert_matches(b.root())
    assert fused.is_batched
    assert fused.n_blocks == K


def test_batched_gate_variants():
    for gate in ("row_rev", "row_in", "none"):
        b = _Builder()
        _assert_matches(b.root(gate=gate))


def test_batched_distinct_decays():
    b = _Builder()
    g2 = _p("g2")

    def arm(i):
        dec = Op.make("decay", g2 if i else b.g, t=C)
        t = Op.make(
            "div",
            Op.make(
                "om_decayed",
                Op.make("matmul", b.qi(i), b.kti(i)),
                dec,
                b.vi(i),
            ),
            b.scale,
        )
        for j in range(i):
            t = Op.make("add", t, b.cross(i, j))
        return t

    root = Op.make("concat", *[arm(i) for i in range(K)], dim=-2)
    ir = b.ir(root)
    ir.params["g2"] = g2
    fused = _assert_matches(
        root,
        extra_ir=ir,
        extra_vals={"g2": torch.tensor(0.8, dtype=torch.float64)},
    )
    assert fused.is_batched


def test_batched_intra_row_gate():
    b = _Builder()
    fused = _assert_matches(b.root(row=True))
    assert fused.is_batched


def test_batched_non_matmul_scores():
    b = _Builder()
    # a score that is not a bare matmul → the generic-eval stack path
    arm0 = Op.make(
        "div",
        Op.make(
            "om_decayed",
            Op.make(
                "add",
                Op.make("matmul", b.qi(0), b.kti(0)),
                Op.make("matmul", b.qi(0), b.kti(0)),
            ),
            b.decay(),
            b.vi(0),
        ),
        b.scale,
    )
    root = Op.make("concat", arm0, b.arm(1), b.arm(2), dim=-2)
    fused = _assert_matches(root)
    assert fused.is_batched
    assert fused._plan["score_ab"] is None


def test_batched_eval_mode_states():
    # inner lhs a plain transpose — the eval-mode ``u is None`` path
    b = _Builder()
    arm1 = Op.make(
        "add",
        b.arm(1, gate="none"),
        b.cross(1, 0, state="no_mul"),
    )
    arm2 = Op.make(
        "add",
        Op.make(
            "add",
            b.arm(2, gate="none"),
            b.cross(2, 0, state="no_mul"),
        ),
        b.cross(2, 1, state="no_mul"),
    )
    root = Op.make("concat", b.arm(0), arm1, arm2, dim=-2)
    fused = _assert_matches(root)
    assert fused.is_batched


def test_batched_mixed_states():
    # parsed and eval-mode inners in the same plan → the mixed path
    b = _Builder()
    arm2 = Op.make(
        "add",
        Op.make(
            "add",
            b.arm(2, gate="none"),
            b.cross(2, 0, state="parsed"),
        ),
        b.cross(2, 1, state="add"),
    )
    root = Op.make("concat", b.arm(0), b.arm(1), arm2, dim=-2)
    fused = _assert_matches(root)
    assert fused.is_batched


def test_batched_intra_only_arms():
    # every arm intra-only → no cross pairs at all
    b = _Builder()
    arms = [
        Op.make(
            "div",
            Op.make(
                "om_decayed",
                Op.make("matmul", b.qi(i), b.kti(i)),
                b.decay(),
                b.vi(i),
            ),
            b.scale,
        )
        for i in range(K)
    ]
    fused = _assert_matches(Op.make("concat", *arms, dim=-2))
    assert fused.is_batched
    assert fused._plan["pairs"] == []
    assert (
        fused._forward_cross(fused._plan, None, None, None, None)
        is None
    )


def test_fallback_on_non_pattern():
    b = _Builder()
    ir = b.ir(b.q)  # a bare linear — not the pattern
    vals, x = b.values()
    mod = to_chunked_decayed_module(ir, param_values=vals)
    assert not mod.is_batched
    assert mod.n_blocks == 0
    generic = IRModule(ir, param_values=vals)
    with torch.no_grad():
        torch.testing.assert_close(mod(x), generic(x), rtol=0, atol=0)


def test_scalar_leaf_variants():
    b = _Builder()
    vals, x = b.values()
    ir = b.ir(b.root())
    mod = to_chunked_decayed_module(ir, param_values=vals)
    # Const / raw float / Param scale paths
    assert mod._scalar(Const(2.0), {}, x) == 2.0
    assert mod._scalar(2.5, {}, x) == 2.5
    assert mod._scalar(b.g, {}, x) == 0.9  # the Param-eval path


def test_executor_routing_claims_the_term():
    b = _Builder()
    ir = b.ir(b.root())
    sink = TorchSink()
    spec = next(
        s
        for s in sink.executors.values()
        if s.carrier is not None and s.accepts(ir.root)
    )
    assert spec.name == "om_decayed_chunked"
    vals, _ = b.values()
    mod = spec.lower(ir, vals)
    assert spec.engaged(mod)


def test_const_cache_reuses_and_rekeys():
    b = _Builder()
    vals, x = b.values()
    mod = to_chunked_decayed_module(b.ir(b.root()), param_values=vals)
    with torch.no_grad():
        mod(x)
        n = len(mod._const_cache)
        mod(x)
        assert len(mod._const_cache) == n  # steady state reuses
        mod_f = mod.float()
        mod_f(x.float())
        assert len(mod._const_cache) > n  # new dtype → new entries


def test_state_eval_mode_inner_matmul():
    # inner lhs is not a transpose at all → eval-mode via ev
    b = _Builder()
    a = Op.make("add", b.ki(0), b.ki(0))
    inner = Op.make("matmul", a, b.vi(0))
    body = Op.make("matmul", b.qi(1), inner)
    summand = Op.make("mul", b.row(C), Op.make("div", body, b.scale))
    arm1 = Op.make("add", b.arm(1, gate="none"), summand)
    root = Op.make("concat", b.arm(0), arm1, b.arm(2), dim=-2)
    fused = _assert_matches(root)
    assert fused.is_batched


@pytest.mark.requires_cuda
def test_cuda_graph_replay():
    b = _Builder()
    vals, x = b.values()
    mod = to_chunked_decayed_module(b.ir(b.root()), param_values=vals)
    mod = mod.to("cuda")
    xc = x.to("cuda")
    with torch.no_grad():
        ref = mod(xc)
    mod.capture_cuda_graph(xc)
    assert mod.is_graph_captured
    with torch.no_grad():
        out = mod(xc)
        assert out is mod._graph_out
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    # a different-shaped input falls back to eager
    x2 = torch.randn(B, K * C, D, dtype=x.dtype, device="cuda")
    with torch.no_grad():
        mod(x2)
    mod.drop_cuda_graph()
    assert not mod.is_graph_captured


class TestPlanDetails:
    """Targeted plan/forward branches."""

    def test_parse_arm_add_flattening(self):
        b = _Builder()
        arm = b.arm(2)
        parsed = _parse_arm(arm)
        assert parsed is not None and len(parsed["cross"]) == 2

    def test_chunked_decay_module_init_plans(self):
        b = _Builder()
        vals, _ = b.values()
        mod = ChunkedDecayModule(b.ir(b.root()), param_values=vals)
        assert mod._plan is not None

    def _mod(self):
        b = _Builder()
        vals, _ = b.values()
        return to_chunked_decayed_module(
            b.ir(b.root()), param_values=vals
        )

    def test_eval_sliced_select_and_chunk(self):
        mod = self._mod()
        base = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        ev = lambda t: base
        sel = mod._eval_sliced((None, "select", 0, 3, 1), ev)
        assert sel.shape == (3, 4) and torch.equal(sel, base)
        b4 = torch.arange(24, dtype=torch.float64).reshape(2, 12)
        ev2 = lambda t: b4
        ch = mod._eval_sliced((None, "chunk", 1, 3, 4), ev2)
        assert ch.shape == (3, 2, 4)

    def test_stack_terms_fallbacks(self):
        mod = self._mod()
        # non-slice terms → per-term eval + stack
        t1, t2 = _v("t1", 4), _v("t2", 4)
        vals = {t1: torch.ones(4), t2: torch.zeros(4)}
        out = mod._stack_terms([t1, t2], lambda t: vals[t])
        assert out.shape == (2, 4)
        # non-contiguous slices of one base → stack fallback too
        base = _v("base", 2, 12)
        s0 = Op.make("split", base, dim=-1, index=0, sizes=[4, 4, 4])
        s2 = Op.make("split", base, dim=-1, index=2, sizes=[4, 4, 4])
        bfull = torch.arange(24, dtype=torch.float64).reshape(2, 12)
        out2 = mod._stack_terms(
            [s0, s2],
            lambda t: bfull.narrow(-1, t.attrs["index"] * 4, 4),
        )
        assert out2.shape == (2, 2, 4)
        # slices of different bases → not one range → stack fallback
        other = _v("other", 2, 12)
        o0 = Op.make(
            "split", other, dim=-1, index=0, sizes=[4, 4, 4]
        )
        vals2 = {s0: torch.ones(2, 4), o0: torch.zeros(2, 4)}
        out3 = mod._stack_terms([s0, o0], lambda t: vals2[t])
        assert out3.shape == (2, 2, 4)
        # a select (not a chunk) → the contiguous-range decline path
        sel = Op.make("select", base, dim=0, index=1)
        vals3 = {s0: torch.ones(2, 4), sel: torch.ones(2, 4)}
        out4 = mod._stack_terms([s0, sel], lambda t: vals3[t])
        assert out4.shape == (2, 2, 4)

    def test_pad_helpers(self):
        mod = self._mod()
        t = torch.ones(4, 4)
        assert mod._pad_row(t, 4).shape == (1, 1, 4, 4)
        assert mod._pad_row(t, 2).shape == (4, 4)
        st = torch.ones(3, 4, 4)
        assert mod._pad_stacked(st, 4).shape == (3, 1, 4, 4)
        assert mod._pad_stacked(st, 3).shape == (3, 4, 4)
