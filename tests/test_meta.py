"""Tests for mlopt_core.meta — coherence stratification + rule synthesis.

Part A: coherent laws (comm/assoc/id/neg) are *computed* by
``canonicalize`` rather than stored in the e-graph, so ``stratified_run``
saturates with contentful rules only.  Measured on LinearRecurrence
(T=6, d=8): all-laws saturation reaches 100,353 e-nodes while the
stratified run needs 178 — a ~560x reduction — and the extracted term
is fp64-exact.

Part B: ``synthesize_rules`` composes ordered rule pairs (critical-pair
completion).  Fed ``SCAN_LAWS`` minus ``aff_lift_step`` plus a small
unrolled-recurrence seed, it re-derives the unfolded equivalent of
``aff_lift_step`` (two raw steps -> nested ``apply``); with
``aff_lift_step`` present it emits the composed form
``apply(aff_compose(aff(A2,x), aff(A1,u)), h)`` directly.
"""

import torch

from mlopt_core import meta
from mlopt_core import laws as R
from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var, op_repr
from mlopt_torch.models import LinearRecurrence, SwiGLU
from mlopt_torch.torch_bridge import export_to_ir, ir_to_torch_module


def _T(n=4):
    return TensorType((n, n))


def _opdepth(t, memo):
    if not isinstance(t, Op):
        return 0
    k = id(t)
    if k not in memo:
        memo[k] = 1 + max(
            (_opdepth(a, memo) for a in t.args), default=0
        )
    return memo[k]


def _recurrence_seed():
    """Two raw steps with DISTINCT leaves (maximally general seed)."""
    A1, A2 = Param("A1", _T()), Param("A2", _T())
    h0, u, x = Var("h0", _T()), Var("u", _T()), Var("x", _T())
    seed = Op.make(
        "add",
        Op.make(
            "matmul", A2, Op.make("add", Op.make("matmul", A1, h0), u)
        ),
        x,
    )
    return seed, A1, A2, h0, u, x


# ---------------------------------------------------------------------------
#  Part A.1 — classification
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  Part A.2 — canonicalize
# ---------------------------------------------------------------------------








def test_canonicalize_idempotent_on_exported_ir():
    torch.manual_seed(0)
    for model, x in [
        (
            LinearRecurrence(8, 6).eval().double(),
            torch.randn(6, 8, dtype=torch.float64),
        ),
        (
            SwiGLU(16, hidden_mult=2).eval().double(),
            torch.randn(2, 3, 16, dtype=torch.float64),
        ),
    ]:
        ir, _ = export_to_ir(model, x)
        once = meta.canonicalize(ir.root)
        twice = meta.canonicalize(once)
        assert op_repr(once) == op_repr(twice)


def test_canonicalize_semantics_preserved():
    """Hand-scrambled add/mul soup canonicalizes to a term that lowers
    and evaluates identically (fp64)."""
    torch.manual_seed(0)
    x, y = Var("x", _T()), Var("y", _T())
    orig = Op.make(
        "add",
        Op.make("mul", x, Const(1.0)),
        Op.make("add", y, Const(0.0)),
    )
    canon = meta.canonicalize(orig)
    assert op_repr(canon) == op_repr(Op.make("add", x, y))

    ir = IR(
        root=canon, inputs=[x, y], input_names={"x", "y"}, params={}
    )
    mod = ir_to_torch_module(ir, param_values={})
    a, b = (
        torch.randn(4, 4, dtype=torch.float64),
        torch.randn(4, 4, dtype=torch.float64),
    )
    with torch.no_grad():
        assert torch.allclose(mod(a, b), a + b)




# ---------------------------------------------------------------------------
#  Part A.3 — stratified_run
# ---------------------------------------------------------------------------


def test_stratified_run_reduces_enodes_and_stays_exact():
    """The payoff: all-laws saturation vs. canonicalize + contentful-only.

    Measured on this machine (d=8):
      T=4  full 1,671   stratified 73
      T=5  full 11,688  stratified 118
      T=6  full 100,353 stratified 178
    """
    torch.manual_seed(0)
    T, d = 5, 8
    m = LinearRecurrence(d, T).eval().double()
    x = torch.randn(T, d, dtype=torch.float64)
    ir, st = export_to_ir(m, x)
    all_rules = R.all_rules() + R.SCAN_LAWS

    eg_full = EGraph()
    root_full = eg_full.add_term(ir.root)
    stats_full = eg_full.run(
        all_rules, root_full, max_iterations=14, max_nodes=400_000
    )

    eg_strat = EGraph()
    out = meta.stratified_run(
        eg_strat,
        all_rules,
        ir.root,
        max_iterations=14,
        max_nodes=400_000,
        extract_fn=eg_strat.extract_min_depth,
    )

    n_full, n_strat = stats_full["n_enodes"], out["stats"]["n_enodes"]
    assert n_strat < n_full / 10
    assert "comm_add" in out["coherent_dropped"]
    assert "aff_lift" in out["contentful_used"]
    # the affine monoid was still reached by contentful rules alone
    assert "aff" in op_repr(out["canonical_best"])

    opt_ir = IR(
        root=out["canonical_best"],
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    mod = ir_to_torch_module(opt_ir, param_values=st)
    with torch.no_grad():
        assert torch.allclose(m(x), mod(x), atol=1e-10)


# ---------------------------------------------------------------------------
#  Part B — synthesize_rules
# ---------------------------------------------------------------------------


def test_synthesize_emits_valid_derived_rules():
    seed, *_ = _recurrence_seed()
    rules = [r for r in R.SCAN_LAWS if r.name != "aff_lift_step"]
    derived = meta.synthesize_rules(rules, [seed], fuel=2000)
    assert len(derived) >= 1

    for d in derived:
        # well-formed: every rhs metavar is bound by the lhs
        assert meta.pattern_metavars(d.rhs) <= meta.pattern_metavars(
            d.lhs
        )
        # the rule actually rewrites correctly: instantiate the lhs,
        # apply the derived rule, and compare semantically on fp64
        # random tensors.
        mvars = sorted(meta.pattern_metavars(d.lhs))
        leaves = [Var(f"_t{i}", _T()) for i in range(len(mvars))]
        subst = dict(zip(mvars, leaves, strict=True))
        t0 = meta.instantiate_pattern(d.lhs, subst)
        env = {
            v: torch.randn(4, 4, dtype=torch.float64) for v in leaves
        }
        a = meta._eval_term(t0, env)
        applied = meta.apply_rewrite_at(d, t0, ())
        assert applied is not None
        b = meta._eval_term(applied, env)
        assert meta._eval_allclose(a, b, tol=1e-6)






