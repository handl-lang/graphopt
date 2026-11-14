"""``mlopt_torch.region_fuse`` — census, eligibility, rewrite, screening.

Pins the delivery-time region fuser behind
``docs/reports/kokoro_region_fuse.md``: the elementwise-component
census and its honest skip counters, the single-exit eligibility
rules, canonicalisation/dedup, the ``fused_region`` term rewrite, the
per-kernel spelled-vs-compiled screen on real tensors, the
whole-module output gate, and the ``RegionFusionRunner`` stats
ledger.  Spelled mode is the bitwise control arm; compiled mode is
exercised on tiny CPU regions (inductor compile is seconds-scale, so
compiled tests stay minimal and marked where slow).
"""

import mlopt_carriers.xcarrier  # noqa: F401 — carrier torch bindings
import torch
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var
from mlopt_torch import region_fuse as rf
from mlopt_torch.region_fuse import (
    ELEMENTWISE_OPS,
    RegionFuseConfig,
    collect_regions,
    fuse_module,
    region_census,
    rewrite_term,
)
from mlopt_torch.runners import RegionFusionRunner
from mlopt_torch.torch_bridge import ir_to_torch_module

torch.manual_seed(0)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _mod(root, inputs=(), params=None, pv=None):
    ir = IR(
        root=root,
        inputs=list(inputs),
        input_names={v.name for v in inputs},
        params=params or {},
    )
    return ir_to_torch_module(ir, param_values=pv or {})


def _region_term():
    """add(sin(exp(mul(x,x))), tanh(mul(y,y))) — one 6-op region."""
    x = _v("x", 4, 4)
    y = _v("y", 4, 4)
    a = Op.make("mul", x, x)
    b = Op.make("exp", a)
    c = Op.make("sin", b)
    d = Op.make("mul", y, y)
    e = Op.make("tanh", d)
    return Op.make("add", c, e), (x, y)


def _spelled_module():
    root, inputs = _region_term()
    mod = _mod(root, inputs)
    return mod, (torch.randn(4, 4), torch.randn(4, 4))


# ---------------------------------------------------------------------------
#  Census
# ---------------------------------------------------------------------------


def test_elementwise_vocab_size():
    assert len(ELEMENTWISE_OPS) == 28
    assert "fused_region" not in ELEMENTWISE_OPS


def test_census_single_region():
    root, _ = _region_term()
    regions, n = region_census(root)
    assert n == 6
    assert len(regions) == 1
    _r, members, leaves = regions[0]
    assert {m.op for m in members} == {
        "add",
        "mul",
        "exp",
        "sin",
        "tanh",
    }
    assert leaves == [] or all(not isinstance(lv, Op) for lv in leaves)


def test_census_empty_and_non_elem():
    x = _v("x", 2)
    regions, n = region_census(x)
    assert regions == [] and n == 0
    # matmul is not elementwise — no regions even with elem args
    m = Op.make("matmul", x, x)
    regions, n = region_census(m)
    assert regions == [] and n == 0


def test_census_multi_root_fallback_and_shared_dag():
    """Two elem roots sharing one elem child → one component, 2 roots.

    ``s`` is consumed by both ``add`` and ``mul`` (both elementwise,
    neither has an elementwise parent) — the union-find merges them
    into ONE component whose root pick takes the ``len(roots) != 1``
    fallback (largest ``_subtree_ops``).
    """
    x = _v("x", 2)
    s = Op.make("square", x)
    a = Op.make("add", s, x)
    m = Op.make("mul", s, x)
    root = Op.make("concat", a, m, dim=0)
    regions, n = region_census(root)
    assert n == 3  # square, add, mul (concat is not elementwise)
    # members: square, add, mul → one component, roots {add, mul}
    assert len(regions) == 1
    r, members, _lv = regions[0]
    assert r.op in {"add", "mul"}
    assert len(members) == 3


def test_census_diamond_unions_find_and_root_paths():
    """Diamond sharing exercises ``find`` path-compression + ``union``
    same-root no-op: ``u``/``v`` are each args of BOTH ``add`` and
    ``mul`` — the second/third/fourth unions hit non-root finds and a
    ``ra == rb`` pair."""
    x = _v("x", 2)
    u = Op.make("square", x)
    v = Op.make("neg", x)
    a = Op.make("add", u, v)
    m = Op.make("mul", u, v)
    root = Op.make("concat", a, m, dim=0)
    regions, n = region_census(root)
    assert n == 4
    assert len(regions) == 1
    assert len(regions[0][1]) == 4


def test_subtree_ops_dedupes_shared():
    x = _v("x", 2)
    s = Op.make("square", x)
    a = Op.make("add", s, s)  # s shared — counted once
    assert rf._subtree_ops(a) == 2
    b = Op.make("matmul", a, x)  # non-op leaf inside
    assert rf._subtree_ops(b) == 2


# ---------------------------------------------------------------------------
#  Eligibility — collect_regions
# ---------------------------------------------------------------------------


def test_collect_fuses_single_region():
    root, _ = _region_term()
    plan = collect_regions(root)
    assert plan.stats["fused"] == 1
    assert plan.stats["ops_fused"] == 6
    assert len(plan.sites) == 1
    assert len(plan.exemplars) == 1
    site = next(iter(plan.sites.values()))
    kid, leaves = site
    assert kid == 0
    assert [type(lv).__name__ for lv in leaves] == ["Var", "Var"]


def test_collect_small_regions_skipped():
    x = _v("x", 2)
    root = Op.make("neg", Op.make("abs", x))  # 2 ops — min_ops=3 skips
    plan = collect_regions(root, min_ops=3)
    assert plan.stats["small"] == 1
    assert not plan.sites
    # default min_ops=2 fuses it
    plan2 = collect_regions(root)
    assert plan2.stats["fused"] == 1


def test_collect_multi_root_skipped():
    x = _v("x", 2)
    s = Op.make("square", x)
    a = Op.make("add", s, x)
    m = Op.make("mul", s, x)
    root = Op.make("concat", a, m, dim=0)
    plan = collect_regions(root)
    # one component {square,add,mul}, roots {add,mul} → multi_root
    assert plan.stats["multi_root"] == 1
    assert not plan.sites


def test_collect_multi_exit_skipped():
    """A non-root member consumed OUTSIDE the region → multi_exit.

    ``inner`` feeds ``top`` (elementwise, in-region) AND ``leak`` (a
    non-elementwise op): ``top`` is the unique root but ``inner`` has
    an outside consumer, so the region is not single-exit.
    """
    x = _v("x", 2, 2)
    inner = Op.make("square", x)
    top = Op.make("add", inner, x)
    leak = Op.make("reshape", inner, shape=(4,))
    root = Op.make("concat", top, leak, dim=0)
    plan = collect_regions(root)
    assert plan.stats.get("multi_exit") == 1
    assert not plan.sites


def test_collect_dedup_shared_shape():
    """Two structurally identical regions → ONE kernel."""
    x = _v("x", 2)
    y = _v("y", 2)
    # region A: add(mul(x,x), sin(x)); region B: same shape on y
    a = Op.make("add", Op.make("mul", x, x), Op.make("sin", x))
    b = Op.make("add", Op.make("mul", y, y), Op.make("sin", y))
    root = Op.make("concat", a, b, dim=0)
    plan = collect_regions(root)
    assert plan.stats["fused"] == 2
    assert plan.stats["ops_fused"] == 6
    assert len(plan.exemplars) == 1  # dedup by internal structure
    kids = {kid for kid, _ in plan.sites.values()}
    assert kids == {0}


def test_collect_dedup_const_values_share_kernel():
    """Regions differing only in a Const VALUE share one kernel."""
    x = _v("x", 2)
    a = Op.make("mul", x, Const(2))
    b = Op.make("sin", a)
    c = Op.make("mul", x, Const(3))
    d = Op.make("sin", c)
    root = Op.make("concat", b, d, dim=0)
    plan = collect_regions(root)
    assert plan.stats["fused"] == 2
    assert len(plan.exemplars) == 1
    # leaves include the Consts as positional args
    _kid, leaves = next(iter(plan.sites.values()))
    assert any(isinstance(lv, Const) for lv in leaves)


def test_collect_unreachable_via_canon_none(monkeypatch):
    root, _ = _region_term()
    monkeypatch.setattr(rf, "_canon_region", lambda *a: None)
    plan = collect_regions(root)
    assert plan.stats["unreachable"] == 1
    assert not plan.sites


# ---------------------------------------------------------------------------
#  _canon_region direct edges
# ---------------------------------------------------------------------------


def test_canon_region_unreachable_members():
    x = _v("x", 2)
    a = Op.make("square", x)
    b = Op.make("neg", x)
    # members list claims connectivity that does not exist
    assert rf._canon_region(a, [a, b]) is None


def test_canon_region_leaf_order_and_specs():
    x = _v("x", 2)
    y = _v("y", 2)
    p = Param("p", _T(2))
    c = Const(0.5)
    m1 = Op.make("mul", x, y)
    m2 = Op.make("add", m1, p)
    m3 = Op.make("mul", m2, c)
    root = m3
    members = [m1, m2, m3]
    canon = rf._canon_region(root, members)
    assert canon is not None
    order, specs, leaves = canon
    assert order == [m1, m2, m3]
    # leaves in first-appearance order: x, y, p, c
    assert leaves == [x, y, p, c]
    assert specs[0] == [("x", 0), ("x", 1)]
    assert specs[1] == [("m", 0), ("x", 2)]
    assert specs[2] == [("m", 1), ("x", 3)]


def test_canon_region_dfs_early_returns():
    """``dfs`` early-return: shared member (``seen``) and non-member
    Op arg (``not in member_ids``)."""
    x = _v("x", 2)
    s = Op.make("square", x)
    t = Op.make("neg", x)
    ma = Op.make("mul", s, t)
    mb = Op.make("sub", s, t)
    top = Op.make("mul", ma, mb)
    canon = rf._canon_region(top, [s, t, ma, mb, top])
    assert canon is not None
    order, _specs, _leaves = canon
    assert len(order) == 5  # s reached twice via ma/mb — seen-hit
    # non-member Op arg: w is an Op leaf of member m
    w = Op.make("matmul", x, x)
    m = Op.make("mul", w, x)
    top2 = Op.make("add", m, s)
    canon2 = rf._canon_region(top2, [m, s, top2])
    assert canon2 is not None
    _o2, specs2, leaves2 = canon2
    assert w in leaves2  # Op boundary leaf is positional
    assert ("x", leaves2.index(w)) in specs2[0]


def test_region_key_attrs_and_ops():
    x = _v("x", 2)
    a = Op.make("clamp", x, min=0.0, max=1.0)
    b = Op.make("square", a)
    order = [a, b]
    specs = [[("x", 0)], [("m", 0)]]
    k1 = rf._region_key(order, specs)
    k2 = rf._region_key(order, specs)
    assert k1 == k2
    # different attrs → different key
    a2 = Op.make("clamp", x, min=0.0, max=2.0)
    order2 = [a2, b]
    assert rf._region_key(order2, specs) != k1


# ---------------------------------------------------------------------------
#  _make_body / _make_dispatch / _FusedBindings
# ---------------------------------------------------------------------------


def test_make_body_unrolled_eval():
    x = _v("x", 2)
    a = Op.make("square", x)
    b = Op.make("neg", a)
    order = [a, b]
    specs = [[("x", 0)], [("m", 0)]]
    binds = {"square": lambda t: t * t, "neg": lambda t: -t}
    f1 = rf._make_body(order, specs, binds)
    f2 = rf._make_body(order, specs, binds)
    assert f1 is not f2
    t = torch.randn(3)
    assert torch.equal(f1(t), -(t * t))
    # attrs reach the binding
    c = Op.make("clamp", x, min=0.0, max=1.0)
    order2 = [c]
    specs2 = [[("x", 0)]]
    f3 = rf._make_body(
        order2, specs2, {"clamp": lambda t, **kw: t.clamp(**kw)}
    )
    tin = torch.randn(3) * 3
    assert torch.equal(f3(tin), tin.clamp(min=0.0, max=1.0))


def test_ensure_recompile_limit_raises_and_records():
    rep: dict = {}
    rf._ensure_recompile_limit(64, rep)
    info = rep["recompile_limit"]
    assert info["needed"] == 64
    assert info["after"] >= 64
    # a smaller need leaves the (now-raised) global alone but still records
    rep2: dict = {}
    rf._ensure_recompile_limit(1, rep2)
    assert rep2["recompile_limit"]["after"] >= 64


def test_ensure_recompile_limit_error_recorded(monkeypatch):
    class _Bad:
        @property
        def recompile_limit(self):
            raise RuntimeError("no dynamo")

    monkeypatch.setattr(
        torch,
        "_dynamo",
        type("D", (), {"config": _Bad()})(),
        raising=False,
    )
    rep: dict = {}
    rf._ensure_recompile_limit(8, rep)
    info = rep["recompile_limit"]
    assert "error" in info and "RuntimeError" in info["error"]


def test_make_dispatch_and_fused_bindings():
    reg = {0: lambda *xs: xs[0] * 2, 1: lambda *xs: xs[0] + 1}
    disp = rf._make_dispatch(reg)
    assert disp(3.0, fused_k=0) == 6.0
    assert disp(3.0, fused_k=1) == 4.0

    base = {"add": lambda a, b: a + b}
    fb = rf._FusedBindings(base, disp)
    assert isinstance(fb, dict)
    assert fb["fused_region"] is disp
    assert fb["add"] is base["add"]
    assert fb.get("fused_region") is disp
    assert fb.get("missing") is None
    assert fb.get("missing", 7) == 7
    assert "fused_region" in fb and "add" in fb and "nope" not in fb
    assert set(fb) == {"add", "fused_region"}
    assert len(fb) == 2
    assert fb  # __bool__ — eval_term's `bindings or {}` guard
    assert fb.keys() == ["add", "fused_region"]
    assert fb["add"](1, 2) == 3
    assert [k for k, _ in fb.items()] == ["add", "fused_region"]
    assert len(fb.values()) == 2


# ---------------------------------------------------------------------------
#  rewrite_term
# ---------------------------------------------------------------------------


def test_rewrite_term_basic_and_sharing():
    root, _ = _region_term()
    plan = collect_regions(root)
    new_root = rewrite_term(root, plan.sites)
    # root 'add' was the region root → now a fused_region op
    assert isinstance(new_root, Op)
    assert new_root.op == "fused_region"
    assert new_root.attrs["fused_k"] == 0
    # leaves are the two Vars, positional
    assert [type(a).__name__ for a in new_root.args] == ["Var", "Var"]


def test_rewrite_term_nested_and_unchanged():
    """Region under a non-elem parent; untouched sibling stays interned."""
    x = _v("x", 2)
    y = _v("y", 2)
    reg = Op.make("add", Op.make("square", x), x)
    other = Op.make("neg", y)
    root = Op.make("matmul", reg, other)
    plan = collect_regions(root)
    assert plan.stats["fused"] == 1
    new_root = rewrite_term(root, plan.sites)
    assert new_root.op == "matmul"
    # the untouched 'other' arg keeps object identity
    assert new_root.args[1] is other
    assert new_root.args[0].op == "fused_region"


def test_rewrite_term_attr_named_validate_fallback():
    """An attr literally named ``validate`` forces the Op() fallback.

    The odd op must have a CHANGED arg for the rebuild path to run —
    it consumes the region, so its arg becomes ``fused_region`` and
    ``Op.make(..., validate=False, **{"validate": 1})`` raises
    TypeError → the direct ``Op(...)`` mint.
    """
    x = _v("x", 2)
    y = _v("y", 2)
    reg = Op.make("add", Op.make("square", x), x)
    weird = Op("odd_op", (reg, y), {"validate": 1, "dim": 0})
    root = Op.make("concat", weird, y, dim=0)
    plan = collect_regions(root)
    assert plan.stats["fused"] == 1
    new_root = rewrite_term(root, plan.sites)
    rebuilt = new_root.args[0]
    assert rebuilt.op == "odd_op"
    assert rebuilt.attrs["validate"] == 1
    assert rebuilt.args[0].op == "fused_region"


def test_rewrite_term_memo_shared_root():
    """A region root shared by two parents rewrites once (memo hit)."""
    x = _v("x", 2)
    reg = Op.make("add", Op.make("square", x), x)
    root = Op.make("concat", reg, reg, dim=0)
    plan = collect_regions(root)
    assert plan.stats["fused"] == 1
    new_root = rewrite_term(root, plan.sites)
    assert new_root.args[0] is new_root.args[1]
    assert new_root.args[0].op == "fused_region"


def test_rewrite_term_non_op_and_no_sites():
    x = _v("x", 2)
    assert rewrite_term(x, {}) is x
    root, _ = _region_term()
    # empty sites → identical object back (all args unchanged)
    assert rewrite_term(root, {}) is root


# ---------------------------------------------------------------------------
#  fuse_module — spelled mode (bitwise control) + gates
# ---------------------------------------------------------------------------


def test_fuse_module_not_ir_module():
    rep = fuse_module(
        object(), torch.randn(2), config=RegionFuseConfig()
    )
    assert rep["applied"] is False
    assert rep["reason"] == "not_ir_module"


def test_fuse_module_no_regions():
    x = _v("x", 2)
    mod = _mod(Op.make("neg", x), [x])
    rep = fuse_module(mod, torch.randn(2), config=RegionFuseConfig())
    assert rep["applied"] is False
    assert rep["reason"] == "no_eligible_regions"
    assert rep["census"]["small"] == 1


def test_fuse_module_spelled_bitwise():
    mod, inp = _spelled_module()
    before = mod(*inp)
    rep = fuse_module(mod, inp, config=RegionFuseConfig(compiled=False))
    assert rep["applied"] is True
    assert rep["mode"] == "spelled"
    assert rep["fused_vs_unfused"]["max_abs"] == 0.0
    after = mod(*inp)
    assert torch.equal(before, after)
    # bindings view resolves fused_region
    assert "fused_region" in mod._torch_bindings


def test_fuse_module_output_gate_declines(monkeypatch):
    """A fused-vs-unfused miss beyond output_rtol reverts the module."""
    mod, inp = _spelled_module()
    root_before = mod._root
    plan_before = mod._root_plan
    bindings_before = mod._torch_bindings
    # Force a mismatch: compare against a doctored 'before' by
    # shrinking output_rtol to impossible — spelled is bitwise so the
    # gate passes; instead monkeypatch verify_equiv to fail once.
    calls = []
    real = rf.verify_equiv

    def fake(ref, out, rtol=1e-4, atol=None):
        rep = real(ref, out, rtol=rtol, atol=atol)
        calls.append(rep.passed)
        if len(calls) == 1:  # the module gate (only call in spelled)
            return type(rep)(
                max_abs=rep.max_abs,
                max_rel=1.0,
                passed=False,
                rtol=rep.rtol,
            )
        return rep

    monkeypatch.setattr(rf, "verify_equiv", fake)
    rep = fuse_module(mod, inp, config=RegionFuseConfig(compiled=False))
    assert rep["applied"] is False
    assert rep["reason"] == "output_gate"
    assert rep["fused_vs_unfused"]["passed"] is False
    # module fully reverted
    assert mod._root is root_before
    assert mod._root_plan is plan_before
    assert mod._torch_bindings is bindings_before
    assert mod(*inp) is not None


def test_fuse_module_exception_declines(monkeypatch):
    """An unexpected error mid-fuse reverts and reports honestly."""
    mod, inp = _spelled_module()
    bindings_before = mod._torch_bindings
    monkeypatch.setattr(
        rf,
        "rewrite_term",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    rep = fuse_module(mod, inp, config=RegionFuseConfig(compiled=False))
    assert rep["applied"] is False
    assert "RuntimeError" in rep["reason"]
    assert mod._torch_bindings is bindings_before


def test_fuse_module_exception_after_swap_restores(monkeypatch):
    """An error AFTER the bindings swap still reverts the module."""
    mod, inp = _spelled_module()
    root_before = mod._root
    plan_before = mod._root_plan
    bindings_before = mod._torch_bindings

    def boom(*a, **k):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(rf, "verify_equiv", boom)
    rep = fuse_module(mod, inp, config=RegionFuseConfig(compiled=False))
    assert rep["applied"] is False
    assert "RuntimeError" in rep["reason"]
    assert mod._root is root_before
    assert mod._root_plan is plan_before
    assert mod._torch_bindings is bindings_before


def test_fuse_module_forward_failure_declines():
    """Warm-up forward raising → decline before any mutation."""
    x = _v("x", 2)
    mod = _mod(Op.make("add", Op.make("square", x), x), [x])
    rep = fuse_module(mod, "not-a-tensor", config=RegionFuseConfig())
    assert rep["applied"] is False
    assert rep["reason"] is not None


def test_fuse_module_example_input_tuple_and_scalar():
    """_example_env: tuple vs bare input; empty tuple path."""
    mod, inp = _spelled_module()
    xs, env, _x = rf._example_env(mod, inp)
    assert xs == inp and env["x"] is inp[0] and env["y"] is inp[1]
    xs2, env2, _x2 = rf._example_env(mod, inp[0])
    assert xs2 == (inp[0],) and env2["y"] is inp[0]  # fallback to x
    xs3, env3, x3 = rf._example_env(mod, ())
    assert xs3 == () and x3 is None and env3["self"] is None


# ---------------------------------------------------------------------------
#  Screening internals
# ---------------------------------------------------------------------------


def test_screen_kernel_pass_and_fail():
    mod, inp = _spelled_module()
    xs, env, x = rf._example_env(mod, inp)
    with torch.no_grad():
        mod(*xs)  # align
    plan = collect_regions(mod._root)
    kid = next(iter(plan.exemplars))
    order, specs = plan.exemplars[kid]
    spelled = rf._make_body(order, specs, mod._torch_bindings)
    # identity "compiled" → passes screen
    chosen, rep = rf._screen_kernel(
        mod,
        env,
        x,
        plan.first_leaves[kid],
        spelled,
        spelled,
        RegionFuseConfig(),
    )
    assert chosen is spelled and rep["passed"] is True
    # a wrong "compiled" → fails screen → spelled chosen
    wrong = lambda *a: a[0] + 1.0  # noqa: E731
    chosen2, rep2 = rf._screen_kernel(
        mod,
        env,
        x,
        plan.first_leaves[kid],
        spelled,
        wrong,
        RegionFuseConfig(),
    )
    assert chosen2 is spelled and rep2["chosen"] == "spelled"


def test_screen_kernel_exception_declines():
    mod, inp = _spelled_module()
    xs, env, x = rf._example_env(mod, inp)
    with torch.no_grad():
        mod(*xs)
    plan = collect_regions(mod._root)
    kid = next(iter(plan.exemplars))
    order, specs = plan.exemplars[kid]
    spelled = rf._make_body(order, specs, mod._torch_bindings)

    def boom(*a):
        raise RuntimeError("inductor died")

    chosen, rep = rf._screen_kernel(
        mod,
        env,
        x,
        plan.first_leaves[kid],
        spelled,
        boom,
        RegionFuseConfig(),
    )
    assert chosen is spelled
    assert rep["screened"] is False
    assert "RuntimeError" in rep["reason"]


def test_fuse_module_spelled_kernel_reports():
    mod, inp = _spelled_module()
    rep = fuse_module(mod, inp, config=RegionFuseConfig(compiled=False))
    k = rep["kernels"]["0"]
    assert k["chosen"] == "spelled"
    assert k["reason"] == "compiled_disabled"
    assert rep["distinct_kernels"] == 1
    assert rep["regions_fused"] == 1
    assert rep["ops_fused"] == 6


# ---------------------------------------------------------------------------
#  Compiled path — one tiny region (inductor compile ~seconds)
# ---------------------------------------------------------------------------


def test_fuse_module_compiled_screened():
    """Compiled body adopted when it screens within tolerance."""
    x = _v("x", 8, 8)
    # mild region: add(square(x), neg(x)) — inductor-friendly
    root = Op.make("add", Op.make("square", x), Op.make("neg", x))
    mod = _mod(root, [x])
    inp = (torch.randn(8, 8),)
    rep = fuse_module(mod, inp)  # compiled default
    # Either adopted (screened pass) or honestly declined to spelled —
    # both are correct; assert the ledger is honest either way.
    assert rep["applied"] is True
    assert rep["mode"] == "compiled"
    k = rep["kernels"]["0"]
    assert k["screened"] is True
    assert k["chosen"] in {"compiled", "spelled"}
    assert rep["fused_vs_unfused"]["passed"] is True
    # output still matches the unfused module within gate
    ref_mod = _mod(root, [x])
    with torch.no_grad():
        a = mod(*inp)
        b = ref_mod(*inp)
    assert (a - b).abs().max().item() <= 1e-4 + 1e-8


def test_fuse_module_compiled_decline_reverts():
    """output_rtol=0 forces the module gate to fail → full revert."""
    x = _v("x", 8, 8)
    root = Op.make("add", Op.make("square", x), Op.make("neg", x))
    mod = _mod(root, [x])
    inp = (torch.randn(8, 8),)
    bindings_before = mod._torch_bindings
    rep = fuse_module(
        mod,
        inp,
        config=RegionFuseConfig(compiled=False, output_rtol=-1.0),
    )
    assert rep["applied"] is False
    assert rep["reason"] == "output_gate"
    assert mod._torch_bindings is bindings_before


# ---------------------------------------------------------------------------
#  RegionFusionRunner
# ---------------------------------------------------------------------------


def test_runner_apply_records_stats():
    mod, inp = _spelled_module()
    runner = RegionFusionRunner(compiled=False)
    stats = {}
    out = runner.apply(mod, inp, stats)
    assert out is mod
    assert stats["region_fuse"]["applied"] is True
    assert stats["region_fuse"]["mode"] == "spelled"


def test_runner_apply_non_module_declines():
    runner = RegionFusionRunner(compiled=False)
    stats = {}
    sentinel = object()
    assert runner.apply(sentinel, None, stats) is sentinel
    assert stats["region_fuse"]["reason"] == "not_ir_module"


def test_runner_config_and_overrides():
    r1 = RegionFusionRunner()
    assert r1.config.min_ops == 2 and r1.config.compiled is True
    r2 = RegionFusionRunner(min_ops=5, output_rtol=1e-3)
    assert r2.config.min_ops == 5
    assert r2.config.output_rtol == 1e-3
    cfg = RegionFuseConfig(min_ops=3)
    r3 = RegionFusionRunner(cfg)
    assert r3.config is cfg
    assert r3.name == "region_fuse"


def test_runner_through_lower_path_protocol():
    """Runner conforms to the ports.Runner protocol shape."""
    from mlopt_core.ports import Runner

    assert isinstance(RegionFusionRunner(), Runner)
