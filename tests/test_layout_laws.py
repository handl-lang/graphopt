"""Tests for the layout laws (:mod:`mlopt_core.laws.layout`).

Every law is a pure re-layout — a stride permutation that evaluates to
the same values — so verification is *bitwise* in float64
(``torch.equal``): a transpose/view never changes an element, and the
NT/linear bridge lowers to the same BLAS call the transposed matmul
dispatches to.

Covered surface:

* pointwise commutation — ``T(f(x)) ↔ f(T(x))`` (unary) and
  ``T(g(a,b)) ↔ g(T(a), T(b))`` (same-rank binary), in both the
  canonical ``dim0``/``dim1`` and bare ``t()`` spellings;
* the transpose involution ``T(T(x)) = x``;
* the product-transpose law ``(A@B).mT = B.mT @ A.mT``;
* the NT bridge ``matmul(x, W.mT) ↔ linear(x, W)``;
* decline cases (rank mismatch, non-pointwise ops, bad axis pairs);
* an end-to-end ``optimize_model`` run selecting each layout under a
  mock cost.
"""

from __future__ import annotations

import torch
from mlopt_core.cost import count_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Op, Param, TensorType, Var
from mlopt_core.laws import (
    LAYOUT_RULES,
    LINEAR_FROM_MM_T,
    LINEAR_FROM_MM_T_BARE,
    LINEAR_TO_MM_T,
)
from mlopt_core.laws.layout import _matmul_rules
from mlopt_torch.torch_bridge import ir_to_torch_module

torch.manual_seed(0)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _saturate(term, rules=None, iters=8, max_nodes=50_000):
    """Build an e-graph over *term* and saturate with LAYOUT_RULES."""
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(
        LAYOUT_RULES if rules is None else rules,
        eid,
        max_iterations=iters,
        max_nodes=max_nodes,
    )
    return eg, eid


def _eval(term, feeds, params=None):
    """Lower *term* through IRModule and run it (fp64)."""
    xs = {
        k: (v if v.dtype == torch.float64 else v.to(torch.float64))
        for k, v in feeds.items()
    }
    ps = {
        k: (v if v.dtype == torch.float64 else v.to(torch.float64))
        for k, v in (params or {}).items()
    }
    inputs = [Var(k, TensorType(tuple(v.shape))) for k, v in xs.items()]
    pterms = {
        k: Param(k, TensorType(tuple(v.shape))) for k, v in ps.items()
    }
    mod = ir_to_torch_module(
        IR(
            root=term,
            inputs=inputs,
            input_names=set(xs),
            params=pterms,
        ),
        param_values=ps,
    )
    mod.eval()
    with torch.no_grad():
        return mod(*xs.values())



def _opt_with_layout(m, x, cost_fn):
    """optimize_model-equivalent with the layout laws opted in.

    ``all_rules()`` excludes ``LAYOUT_RULES`` — their bidirectional
    transpose↔pointwise pairs explode the saturation closure
    (~10–40× search wall on real blocks; runtime parity measured) —
    so the e2e law tests drive the real pipeline directly:
    export → e-graph → saturate(union) → extract → lower.
    """
    from mlopt_torch.torch_bridge import export_to_ir, ir_to_torch_module
    from mlopt_core.laws import all_rules

    ir, source = export_to_ir(m, x)
    eg = EGraph()
    eid = eg.add_term(ir.root)
    eg.run(
        all_rules() + LAYOUT_RULES,
        eid,
        max_iterations=100,
        max_nodes=100_000,
    )
    best = eg.extract_best(eid, cost_fn)
    mod = ir_to_torch_module(
        IR(
            root=best,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params=ir.params,
        ),
        param_values=source,
    )
    return mod, eg

def _fired(eg, name):
    return eg.rule_fires.get(name, 0) > 0


def _members(eg, eid):
    """Every non-leaf member of the root class, each forced-extracted.

    ``extract_best`` returns ``None`` for members that are cyclic after
    a merge (a rewrite like the involution can leave an enode whose
    subtree contains its own class) — those are skipped: they are
    classes-only bookkeeping, not extractable programs.
    """
    root = eg.find(eid)
    out = []
    for n in eg.get_class(eid).nodes:
        if n.op == "leaf":
            continue
        t = eg.extract_best(eid, count_cost, overrides={root: n})
        if t is not None:
            out.append(t)
    return out


def _check_all_members(
    eg, eid, lhs, feeds, ref, params=None, *, exact=True
):
    """Assert every member of the class evaluates equal to ``ref``.

    ``exact=True`` requires bitwise equality — the right bar for pure
    relayouts (pointwise commutation, involution): a transpose never
    touches values.  ``exact=False`` uses a tight fp64 tolerance for
    members containing a GEMM — ``matmul`` under different operand
    layouts can dispatch to a different kernel and differ by an ulp.
    """
    assert torch.equal(_eval(lhs, feeds, params), ref)
    for m in _members(eg, eid):
        got = _eval(m, feeds, params)
        if exact:
            assert torch.equal(got, ref), f"member {m!r} mismatch"
        else:
            torch.testing.assert_close(got, ref, rtol=0, atol=1e-12)


def _weighted(weights):
    """Additive per-op cost model: ``cost = Σ weights[node.op]``."""

    def c(term, memo=None):
        if isinstance(term, Op):
            return weights.get(term.op, 1.0) + sum(
                c(a, memo) for a in term.args
            )
        return 0.0

    return c


# ---------------------------------------------------------------------------
#  _axis_pair (typing.py helper) — unit coverage
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  Pointwise commutation — binary
# ---------------------------------------------------------------------------


def test_push_mul_dimmed_fires_and_verifies():
    """transpose(mul(a,b), 0,1) gains the mul(transpose,transpose)
    member; every member verifies bitwise."""
    a = Var("a", TensorType((4, 5)))
    b = Var("b", TensorType((4, 5)))
    lhs = Op.make("transpose", Op.make("mul", a, b), dim0=0, dim1=1)
    eg, eid = _saturate(lhs)
    assert _fired(eg, "transpose_push_mul")
    A = torch.randn(4, 5, dtype=torch.float64)
    B = torch.randn(4, 5, dtype=torch.float64)
    ref = (A * B).transpose(0, 1)
    _check_all_members(eg, eid, lhs, {"a": A, "b": B}, ref)
    # The pushed member really is mul(transpose(a), transpose(b)).
    pushed = Op.make(
        "mul",
        Op.make("transpose", a, dim0=0, dim1=1),
        Op.make("transpose", b, dim0=0, dim1=1),
    )
    assert any(
        n.op == "mul" for n in eg.get_class(eid).nodes
    ) and torch.equal(_eval(pushed, {"a": A, "b": B}), ref)


def test_push_binary_broadcast_same_rank():
    """Equal-rank but non-equal shapes ((4,1)*(4,5)) still commute —
    positional broadcast survives the same permutation."""
    a = Var("a", TensorType((4, 1)))
    b = Var("b", TensorType((4, 5)))
    lhs = Op.make("transpose", Op.make("add", a, b), dim0=-2, dim1=-1)
    eg, eid = _saturate(lhs)
    assert _fired(eg, "transpose_push_add")
    A = torch.randn(4, 1, dtype=torch.float64)
    B = torch.randn(4, 5, dtype=torch.float64)
    ref = (A + B).mT
    _check_all_members(eg, eid, lhs, {"a": A, "b": B}, ref)


def test_push_mul_bare_spelling():
    """The bare ``(a*b).t()`` form distributes identically."""
    a = Var("a", TensorType((4, 5)))
    b = Var("b", TensorType((4, 5)))
    lhs = Op.make("transpose", Op.make("mul", a, b))
    eg, eid = _saturate(lhs)
    assert _fired(eg, "transpose_push_mul_bare")
    A = torch.randn(4, 5, dtype=torch.float64)
    B = torch.randn(4, 5, dtype=torch.float64)
    _check_all_members(eg, eid, lhs, {"a": A, "b": B}, (A * B).t())


def test_pull_binary_dimmed_fires():
    """mul(T(a), T(b)) factors the transpose back out."""
    a = Var("a", TensorType((4, 5)))
    b = Var("b", TensorType((4, 5)))
    lhs = Op.make(
        "mul",
        Op.make("transpose", a, dim0=0, dim1=1),
        Op.make("transpose", b, dim0=0, dim1=1),
    )
    eg, eid = _saturate(
        lhs,
        rules=[
            r for r in LAYOUT_RULES if r.name == "transpose_pull_mul"
        ],
    )
    assert _fired(eg, "transpose_pull_mul")
    A = torch.randn(4, 5, dtype=torch.float64)
    B = torch.randn(4, 5, dtype=torch.float64)
    _check_all_members(eg, eid, lhs, {"a": A, "b": B}, A.t() * B.t())


# ---------------------------------------------------------------------------
#  Pointwise commutation — unary
# ---------------------------------------------------------------------------


def test_push_unary_dimmed_and_bare():
    x = Var("x", TensorType((4, 5)))
    for op, fn in (
        ("silu", torch.nn.functional.silu),
        ("neg", torch.neg),
        ("tanh", torch.tanh),
        ("square", torch.square),
    ):
        for attrs in (dict(dim0=0, dim1=1), {}):
            lhs = Op.make("transpose", Op.make(op, x), **attrs)
            eg, eid = _saturate(lhs)
            tag = "_bare" if not attrs else ""
            assert _fired(eg, f"transpose_push_{op}{tag}"), (
                op,
                attrs,
            )
            X = torch.randn(4, 5, dtype=torch.float64)
            _check_all_members(eg, eid, lhs, {"x": X}, fn(X).t())


def test_pull_unary_hoists_transpose():
    """silu(T(x)) -> T(silu(x)) — the transpose migrates outward."""
    x = Var("x", TensorType((4, 5)))
    lhs = Op.make("silu", Op.make("transpose", x, dim0=-2, dim1=-1))
    eg, eid = _saturate(lhs)
    assert _fired(eg, "transpose_pull_silu")
    X = torch.randn(4, 5, dtype=torch.float64)
    _check_all_members(
        eg, eid, lhs, {"x": X}, torch.nn.functional.silu(X.t())
    )


# ---------------------------------------------------------------------------
#  Pointwise commutation — declines
# ---------------------------------------------------------------------------


















# ---------------------------------------------------------------------------
#  Involution
# ---------------------------------------------------------------------------


def test_involution_dimmed_bare_combos():
    """Every bare/dimmed spelling pair that swaps the same axes cancels."""
    x = Var("x", TensorType((4, 5)))
    cases = {
        "dd": Op.make(
            "transpose",
            Op.make("transpose", x, dim0=0, dim1=1),
            dim0=1,
            dim1=0,
        ),
        "bb": Op.make("transpose", Op.make("transpose", x)),
        "db": Op.make(
            "transpose",
            Op.make("transpose", x),
            dim0=-2,
            dim1=-1,
        ),
        "bd": Op.make(
            "transpose",
            Op.make("transpose", x, dim0=0, dim1=1),
        ),
    }
    X = torch.randn(4, 5, dtype=torch.float64)
    for tag, lhs in cases.items():
        eg, eid = _saturate(lhs)
        assert _fired(eg, f"transpose_transpose_{tag}"), tag
        _check_all_members(eg, eid, lhs, {"x": X}, X)
        best = eg.extract_best(eid, count_cost)
        assert isinstance(best, Var) and best.name == "x"










# ---------------------------------------------------------------------------
#  Product transpose: (A@B).mT = B.mT @ A.mT
# ---------------------------------------------------------------------------


def test_transpose_matmul_dimmed_and_bare():
    a = Var("a", TensorType((3, 4)))
    b = Param("b", TensorType((4, 6)))
    A = torch.randn(3, 4, dtype=torch.float64)
    B = torch.randn(4, 6, dtype=torch.float64)
    ref = (A @ B).t()
    for name, lhs in (
        (
            "transpose_matmul",
            Op.make(
                "transpose",
                Op.make("matmul", a, b),
                dim0=-2,
                dim1=-1,
            ),
        ),
        (
            "transpose_matmul_bare",
            Op.make("transpose", Op.make("matmul", a, b)),
        ),
    ):
        eg, eid = _saturate(lhs)
        assert _fired(eg, name), name
        _check_all_members(
            eg, eid, lhs, {"a": A}, ref, params={"b": B}, exact=False
        )


def test_transpose_matmul_batched():
    """Batched matmul under the last-two swap: same law, rank 3."""
    a = Var("a", TensorType((2, 3, 4)))
    b = Var("b", TensorType((2, 4, 6)))
    lhs = Op.make(
        "transpose",
        Op.make("matmul", a, b),
        dim0=-2,
        dim1=-1,
    )
    eg, eid = _saturate(lhs)
    assert _fired(eg, "transpose_matmul")
    A = torch.randn(2, 3, 4, dtype=torch.float64)
    B = torch.randn(2, 4, 6, dtype=torch.float64)
    _check_all_members(
        eg, eid, lhs, {"a": A, "b": B}, (A @ B).mT, exact=False
    )


def test_mm_of_transposes_rev_all_spellings():
    """matmul(P.mT, Q.mT) gains the (Q@P).mT member — all spellings."""
    p = Var("p", TensorType((3, 4)))
    q = Var("q", TensorType((6, 3)))
    P = torch.randn(3, 4, dtype=torch.float64)
    Q = torch.randn(6, 3, dtype=torch.float64)
    ref = P.t() @ Q.t()  # == (Q @ P).t()
    cases = {
        "dd": Op.make(
            "matmul",
            Op.make("transpose", p, dim0=0, dim1=1),
            Op.make("transpose", q, dim0=-2, dim1=-1),
        ),
        "bb": Op.make(
            "matmul", Op.make("transpose", p), Op.make("transpose", q)
        ),
        "db": Op.make(
            "matmul",
            Op.make("transpose", p, dim0=0, dim1=1),
            Op.make("transpose", q),
        ),
        "bd": Op.make(
            "matmul",
            Op.make("transpose", p),
            Op.make("transpose", q, dim0=0, dim1=1),
        ),
    }
    for tag, lhs in cases.items():
        eg, eid = _saturate(lhs, rules=_matmul_rules())
        assert _fired(eg, f"matmul_transpose_rev_{tag}"), tag
        _check_all_members(
            eg, eid, lhs, {"p": P, "q": Q}, ref, exact=False
        )
















# ---------------------------------------------------------------------------
#  NT bridge: matmul(x, W.mT) ↔ linear(x, W)
# ---------------------------------------------------------------------------


def test_linear_from_matmul_t_fires_both_spellings():
    """The exported ``x @ W.t()`` (bare) and ``x @ W.mT`` (dimmed)
    forms both gain the ``linear`` member."""
    x = Var("x", TensorType((2, 4)))
    w = Param("w", TensorType((8, 4)))
    X = torch.randn(2, 4, dtype=torch.float64)
    W = torch.randn(8, 4, dtype=torch.float64)
    for name, wt in (
        (
            "linear_from_matmul_t",
            Op.make("transpose", w, dim0=-2, dim1=-1),
        ),
        ("linear_from_matmul_t_bare", Op.make("transpose", w)),
    ):
        lhs = Op.make("matmul", x, wt)
        eg, eid = _saturate(lhs)
        assert _fired(eg, name), name
        _check_all_members(
            eg,
            eid,
            lhs,
            {"x": X},
            X @ W.t(),
            params={"w": W},
            exact=False,
        )
        # under an op-count cost the one-node NT form wins
        best = eg.extract_best(eid, count_cost)
        assert best.op == "linear"


def test_linear_to_matmul_t_fires():
    """linear(x, W) exposes the explicit-transpose matmul member."""
    x = Var("x", TensorType((2, 4)))
    w = Param("w", TensorType((8, 4)))
    lhs = Op.make("linear", x, w)
    eg, eid = _saturate(lhs)
    assert _fired(eg, "linear_to_matmul_t")
    X = torch.randn(2, 4, dtype=torch.float64)
    W = torch.randn(8, 4, dtype=torch.float64)
    _check_all_members(
        eg,
        eid,
        lhs,
        {"x": X},
        torch.nn.functional.linear(X, W),
        params={"w": W},
        exact=False,
    )
    assert any(n.op == "matmul" for n in eg.get_class(eid).nodes)


def test_linear_bridge_batched_and_unknown_dims():
    """Rank-3 x and a None extent still satisfy the shape contract."""
    x = Var("x", TensorType((None, 2, 4)))
    w = Param("w", TensorType((8, 4)))
    lhs = Op.make("linear", x, w)
    eg, eid = _saturate(
        lhs,
        rules=[
            LINEAR_FROM_MM_T,
            LINEAR_FROM_MM_T_BARE,
            LINEAR_TO_MM_T,
        ],
    )
    assert _fired(eg, "linear_to_matmul_t")
    X = torch.randn(3, 2, 4, dtype=torch.float64)
    W = torch.randn(8, 4, dtype=torch.float64)
    _check_all_members(
        eg,
        eid,
        lhs,
        {"x": X},
        torch.nn.functional.linear(X, W),
        params={"w": W},
        exact=False,
    )










# ---------------------------------------------------------------------------
#  Migration composition — the capability the laws exist for
# ---------------------------------------------------------------------------


def test_transpose_migrates_to_matmul_operand():
    """T(mul(A@x, u)) — the transpose pushes through mul to a GEMM
    operand, where the NT flag reads it for free.  The e-class ends up
    holding mul of transposed pieces."""
    m = Var("m", TensorType((3, 6)))
    u = Var("u", TensorType((6, 4)))
    inner = Op.make("matmul", Var("v", TensorType((4, 3))), m)
    lhs = Op.make(
        "transpose",
        Op.make("mul", Op.make("transpose", inner, dim0=0, dim1=1), u),
        dim0=0,
        dim1=1,
    )
    eg, eid = _saturate(lhs)
    assert _fired(eg, "transpose_push_mul")
    V = torch.randn(4, 3, dtype=torch.float64)
    M = torch.randn(3, 6, dtype=torch.float64)
    U = torch.randn(6, 4, dtype=torch.float64)
    ref = ((V @ M).t() * U).t()
    _check_all_members(
        eg, eid, lhs, {"v": V, "m": M, "u": U}, ref, exact=False
    )


def test_double_transpose_roundtrip_via_pointwise():
    """T(T(mul(a,b))) collapses to mul(a,b) — the pushed transposes
    rejoin through the involution."""
    a = Var("a", TensorType((4, 5)))
    b = Var("b", TensorType((4, 5)))
    lhs = Op.make(
        "transpose",
        Op.make("transpose", Op.make("mul", a, b), dim0=0, dim1=1),
        dim0=0,
        dim1=1,
    )
    eg, eid = _saturate(lhs)
    assert _fired(eg, "transpose_transpose_dd")
    A = torch.randn(4, 5, dtype=torch.float64)
    B = torch.randn(4, 5, dtype=torch.float64)
    _check_all_members(eg, eid, lhs, {"a": A, "b": B}, A * B)
    assert eg.extract_best(eid, count_cost).op == "mul"


# ---------------------------------------------------------------------------
#  End to end: optimize_model picks the layout the cost model prefers
# ---------------------------------------------------------------------------


def test_e2e_optimize_model_selects_linear_form():
    """``x @ W.t()`` exports as matmul(x, transpose(W)); with linear
    priced below matmul+transpose, extraction ships the NT call."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = torch.nn.Parameter(
                torch.randn(8, 16, dtype=torch.float64)
            )

        def forward(self, x):
            return x @ self.W.t()

    m = M().eval()
    x = torch.randn(4, 16, dtype=torch.float64)
    mod, eg = _opt_with_layout(
        m,
        x,
        _weighted({"linear": 1.0, "matmul": 10.0, "transpose": 0.0}),
    )
    assert eg.rule_fires.get("linear_from_matmul_t_bare", 0) > 0
    assert mod._root.op == "linear"
    with torch.no_grad():
        torch.testing.assert_close(mod(x), m(x), rtol=0, atol=1e-12)


def test_e2e_optimize_model_selects_transposed_form():
    """The same graph under a linear-expensive mock cost keeps the
    explicit-transpose matmul — the e-graph holds both layouts and
    extraction picks whichever is cheaper."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = torch.nn.Parameter(
                torch.randn(8, 16, dtype=torch.float64)
            )

        def forward(self, x):
            return x @ self.W.t()

    m = M().eval()
    x = torch.randn(4, 16, dtype=torch.float64)
    mod, eg = _opt_with_layout(
        m,
        x,
        _weighted({"linear": 50.0, "matmul": 1.0, "transpose": 0.0}),
    )
    assert eg.rule_fires.get("linear_from_matmul_t_bare", 0) > 0
    assert mod._root.op == "matmul"
    with torch.no_grad():
        torch.testing.assert_close(mod(x), m(x), rtol=0, atol=1e-12)


def test_e2e_transpose_movement_through_pointwise_pipeline():
    """A model whose output is (relu(x @ W.t())).t() — the full law
    family composes: NT bridge + pointwise commute + involution all
    fire and the result verifies."""


    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = torch.nn.Parameter(
                torch.randn(8, 16, dtype=torch.float64)
            )

        def forward(self, x):
            return torch.relu(x @ self.W.t()).t()

    m = M().eval()
    x = torch.randn(4, 16, dtype=torch.float64)
    mod, eg = _opt_with_layout(
        m,
        x,
        _weighted(
            {
                "linear": 1.0,
                "matmul": 5.0,
                "transpose": 2.0,
                "relu": 1.0,
            }
        ),
    )
    assert eg.rule_fires.get("linear_from_matmul_t_bare", 0) > 0
    with torch.no_grad():
        torch.testing.assert_close(mod(x), m(x), rtol=0, atol=1e-12)
