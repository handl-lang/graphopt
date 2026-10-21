"""The kernelopt spike — laws, cost, sink, and the kernel board."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mlopt_core.egraph import EGraph, verify_certificate
from mlopt_core.ir import IR, Const, Op, TensorType, Var
from mlopt_core.ports import Sink
from mlopt_kernel import (
    KERNEL_RULES,
    KernelBoard,
    KernelExecutor,
    KernelSink,
    kernel_cost,
)
from mlopt_kernel.board import STOP

X = Var("x", TensorType((4, 8)))
Y = Var("y", TensorType((4, 8)))


def _rule(name):
    """Look a bank rule up by name — indices churn as the bank grows."""
    return next(r for r in KERNEL_RULES if r.name == name)


def _silu_term():
    return Op.make("mul", X, Op.make("sigmoid", X))


def _softmax_term(keepdim=True):
    return Op.make(
        "div",
        Op.make("exp", X),
        Op.make("sum", Op.make("exp", X), dim=-1, keepdim=keepdim),
    )


def _layernorm_term(eps=Const(1e-5), power=Const(2)):
    c = Op.make("sub", X, Op.make("mean", X, dim=-1, keepdim=True))
    return Op.make(
        "mul",
        c,
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "mean",
                    Op.make("pow", c, power),
                    dim=-1,
                    keepdim=True,
                ),
                eps,
            ),
        ),
    )


def _rmsnorm_term(eps=Const(1e-6), power=Const(2), keepdim=True):
    return Op.make(
        "mul",
        X,
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "mean",
                    Op.make("pow", X, power),
                    dim=-1,
                    keepdim=keepdim,
                ),
                eps,
            ),
        ),
    )


class TestLaws:
    def test_bank_shape(self):
        """Sixteen codegen laws; the fused ops take leaf children."""
        assert [r.name for r in KERNEL_RULES] == [
            "kern_fuse_silu",
            "kern_fuse_softsign",
            "kern_fuse_addmul",
            "kern_fuse_swiglu",
            "kern_fuse_silu_sigmoid",
            "kern_fuse_addmul_tanh",
            "kern_fuse_softmax",
            "kern_fuse_softmax_mm",
            "kern_fuse_layernorm",
            "kern_fuse_rmsnorm",
            "kern_fuse_mm_bias",
            "kern_fuse_mm_scale",
            "kern_fuse_mm_scale_bias",
            "kern_fuse_mm_gelu",
            "kern_square_strength",
            "kern_div_recip",
        ]

    def test_fuse_fires_and_extracts(self):
        """``mul(a, sigmoid(b))`` gains an extractable ``kf_silu``."""
        eg = EGraph()
        root = eg.add_term(_silu_term())
        assert eg.apply_rule(_rule("kern_fuse_silu"), root)
        best = eg.extract_best(root, kernel_cost)
        assert best is not None and best.op == "kf_silu"

    def test_softsign_and_fma_fire(self):
        """The softsign and FMA sites fold to their fused ops."""
        eg = EGraph()
        root = eg.add_term(
            Op.make(
                "div", X, Op.make("add", Op.make("abs", X), Const(1))
            )
        )
        assert eg.apply_rule(_rule("kern_fuse_softsign"), root)
        assert eg.extract_best(root, kernel_cost).op == "kf_softsign"
        eg2 = EGraph()
        root2 = eg2.add_term(
            Op.make("add", Op.make("mul", X, Y), Op.make("tanh", Y))
        )
        assert eg2.apply_rule(_rule("kern_fuse_addmul"), root2)
        assert eg2.extract_best(root2, kernel_cost).op == "kf_addmul"

    def test_strength_reductions(self):
        """pow(x,2) → x·x and x/c → x·(1/c) both fire and verify."""
        eg = EGraph()
        root = eg.add_term(Op.make("pow", X, Const(2)))
        assert eg.apply_rule(_rule("kern_square_strength"), root)
        assert eg.extract_best(root, kernel_cost) == Op.make(
            "mul", X, X
        )
        eg2 = EGraph()
        t = Op.make("div", X, Const(4.0))
        root2 = eg2.add_term(t)
        assert eg2.apply_rule(_rule("kern_div_recip"), root2)
        best = eg2.extract_best(root2, kernel_cost)
        assert best == Op.make("mul", X, Const(0.25))
        cert = eg2.certificate(t, best)
        verify_certificate(t, cert)

    def test_recip_declines_nonconst(self):
        """The recip fold needs a numeric Const divisor."""
        eg = EGraph()
        root = eg.add_term(Op.make("div", X, Y))
        assert not eg.apply_rule(_rule("kern_div_recip"), root)

    def test_swiglu_beats_shallow_silu(self):
        """``(x·σ(x))·y`` extracts the one-launch kf_swiglu."""
        eg = EGraph()
        t = Op.make("mul", Op.make("mul", X, Op.make("sigmoid", X)), Y)
        root = eg.add_term(t)
        eg.apply_rule(_rule("kern_fuse_silu"), root)
        eg.apply_rule(_rule("kern_fuse_swiglu"), root)
        best = eg.extract_best(root, kernel_cost)
        assert best is not None and best.op == "kf_swiglu"

    def test_silu_sigmoid_beats_swiglu_residue(self):
        """``(x·σ(x))·σ(y)`` extracts the whole-gate kf_silu_sigmoid."""
        eg = EGraph()
        t = Op.make(
            "mul",
            Op.make("mul", X, Op.make("sigmoid", X)),
            Op.make("sigmoid", Y),
        )
        root = eg.add_term(t)
        eg.apply_rule(_rule("kern_fuse_silu"), root)
        eg.apply_rule(_rule("kern_fuse_swiglu"), root)
        eg.apply_rule(_rule("kern_fuse_silu_sigmoid"), root)
        best = eg.extract_best(root, kernel_cost)
        # kf_swiglu(x, σ(y)) keeps a spelled σ — the residue fold wins.
        assert best is not None and best.op == "kf_silu_sigmoid"
        assert kernel_cost(best) == 1.0

    def test_addmul_tanh_absorbs_survivor(self):
        """``a·b + tanh(c)`` extracts the whole-site kf_addmul_tanh."""
        eg = EGraph()
        t = Op.make(
            "add", Op.make("mul", X, Y), Op.make("tanh", Y)
        )
        root = eg.add_term(t)
        eg.apply_rule(_rule("kern_fuse_addmul"), root)
        eg.apply_rule(_rule("kern_fuse_addmul_tanh"), root)
        best = eg.extract_best(root, kernel_cost)
        assert best is not None and best.op == "kf_addmul_tanh"
        assert best == Op.make("kf_addmul_tanh", X, Y, Y)

    def test_softmax_fires_and_derives_dim(self):
        """``exp(u)/Σexp(u)`` folds to ``kf_softmax`` with the dim."""
        eg = EGraph()
        root = eg.add_term(_softmax_term())
        assert eg.apply_rule(_rule("kern_fuse_softmax"), root)
        best = eg.extract_best(root, kernel_cost)
        assert best == Op.make("kf_softmax", X, dim=-1)

    def test_softmax_declines_dropped_dim(self):
        """``keepdim=False`` broadcasts wrongly — the site declines."""
        eg = EGraph()
        root = eg.add_term(_softmax_term(keepdim=False))
        assert not eg.apply_rule(_rule("kern_fuse_softmax"), root)

    def test_softmax_mm_fires_and_beats_chain(self):
        """``softmax(u)·v`` folds to kf_softmax_mm — attention chain."""
        v = Var("v", TensorType((4, 4)))
        u = Op.make("matmul", X, Var("w", TensorType((8, 4))))
        t = Op.make(
            "matmul",
            Op.make(
                "div",
                Op.make("exp", u),
                Op.make(
                    "sum", Op.make("exp", u), dim=-1, keepdim=True
                ),
            ),
            v,
        )
        eg = EGraph()
        root = eg.add_term(t)
        assert eg.apply_rule(_rule("kern_fuse_softmax"), root)
        assert eg.apply_rule(_rule("kern_fuse_softmax_mm"), root)
        best = eg.extract_best(root, kernel_cost)
        # the deeper fold wraps the spelled qk-mm — two launches.
        assert best is not None and best.op == "kf_softmax_mm"
        assert best.attrs["dim"] == -1
        cert = eg.certificate(t, best)
        verify_certificate(t, cert)

    def test_softmax_mm_declines_dropped_dim(self):
        """The chain fold inherits the softmax site's keepdim gate."""
        v = Var("v", TensorType((4, 4)))
        u = Op.make("exp", X)
        t = Op.make(
            "matmul",
            Op.make(
                "div",
                u,
                Op.make("sum", u, dim=-1, keepdim=False),
            ),
            v,
        )
        eg = EGraph()
        root = eg.add_term(t)
        assert not eg.apply_rule(_rule("kern_fuse_softmax_mm"), root)

    def test_layernorm_fires_and_derives_eps(self):
        """The mean/var/rsqrt block folds to ``kf_layernorm``."""
        eg = EGraph()
        t = _layernorm_term()
        root = eg.add_term(t)
        assert eg.apply_rule(_rule("kern_fuse_layernorm"), root)
        best = eg.extract_best(root, kernel_cost)
        assert best == Op.make("kf_layernorm", X, dim=-1, eps=1e-05)
        cert = eg.certificate(t, best)
        verify_certificate(t, cert)

    def test_layernorm_declines_bad_site(self):
        """A non-numeric eps or a non-square power vetoes the fold."""
        eg = EGraph()
        root = eg.add_term(_layernorm_term(eps=X))
        assert not eg.apply_rule(_rule("kern_fuse_layernorm"), root)
        eg2 = EGraph()
        root2 = eg2.add_term(_layernorm_term(power=Const(3)))
        assert not eg2.apply_rule(_rule("kern_fuse_layernorm"), root2)

    def test_rmsnorm_fires_and_derives_eps(self):
        """``x·rsqrt(mean(x²)+eps)`` folds to ``kf_rmsnorm``."""
        eg = EGraph()
        t = _rmsnorm_term()
        root = eg.add_term(t)
        assert eg.apply_rule(_rule("kern_fuse_rmsnorm"), root)
        best = eg.extract_best(root, kernel_cost)
        assert best == Op.make("kf_rmsnorm", X, dim=-1, eps=1e-06)
        cert = eg.certificate(t, best)
        verify_certificate(t, cert)

    def test_rmsnorm_declines_bad_site(self):
        """The same front guards as layernorm veto the rmsnorm fold."""
        eg = EGraph()
        root = eg.add_term(_rmsnorm_term(eps=X))
        assert not eg.apply_rule(_rule("kern_fuse_rmsnorm"), root)
        eg2 = EGraph()
        root2 = eg2.add_term(_rmsnorm_term(power=Const(3)))
        assert not eg2.apply_rule(_rule("kern_fuse_rmsnorm"), root2)
        eg3 = EGraph()
        root3 = eg3.add_term(_rmsnorm_term(keepdim=False))
        assert not eg3.apply_rule(_rule("kern_fuse_rmsnorm"), root3)

    def test_mm_epilogues_fire(self):
        """mm+b and mm·s+b fold to their one-launch epilogue ops."""
        w = Var("w", TensorType((8, 8)))
        b = Var("b", TensorType((8,)))
        eg = EGraph()
        root = eg.add_term(Op.make("add", Op.make("matmul", X, w), b))
        assert eg.apply_rule(_rule("kern_fuse_mm_bias"), root)
        assert (
            eg.extract_best(root, kernel_cost).op == "kf_mm_bias"
        )
        eg2 = EGraph()
        t = Op.make(
            "add",
            Op.make("mul", Op.make("matmul", X, w), Const(0.5)),
            b,
        )
        root2 = eg2.add_term(t)
        # both the addmul and the scale-bias epilogue match; the
        # deeper fusion wins on cost.
        eg2.apply_rule(_rule("kern_fuse_addmul"), root2)
        eg2.apply_rule(_rule("kern_fuse_mm_scale_bias"), root2)
        assert (
            eg2.extract_best(root2, kernel_cost).op == "kf_mm_sb"
        )

    def test_mm_scale_and_gelu_fire(self):
        """mm·s and gelu(mm) fold to their one-launch epilogue ops."""
        w = Var("w", TensorType((8, 8)))
        eg = EGraph()
        t = Op.make("mul", Op.make("matmul", X, w), Const(0.125))
        root = eg.add_term(t)
        assert eg.apply_rule(_rule("kern_fuse_mm_scale"), root)
        best = eg.extract_best(root, kernel_cost)
        assert best == Op.make("kf_mm_scale", X, w, Const(0.125))
        eg2 = EGraph()
        t2 = Op.make("gelu", Op.make("matmul", X, w))
        root2 = eg2.add_term(t2)
        assert eg2.apply_rule(_rule("kern_fuse_mm_gelu"), root2)
        best2 = eg2.extract_best(root2, kernel_cost)
        assert best2 == Op.make("kf_mm_gelu", X, w)
        cert = eg2.certificate(t2, best2)
        verify_certificate(t2, cert)

    def test_addmul_reaches_reduced_site(self):
        """Strength reductions expose an addmul site — gen-2 chain."""
        eg = EGraph()
        t = Op.make(
            "add",
            Op.make("pow", X, Const(2)),
            Op.make("div", Y, Const(4.0)),
        )
        root = eg.add_term(t)
        eg.apply_rule(_rule("kern_square_strength"), root)
        eg.apply_rule(_rule("kern_div_recip"), root)
        eg.apply_rule(_rule("kern_fuse_addmul"), root)
        best = eg.extract_best(root, kernel_cost)
        # kf_addmul(x, x, y·0.25) — the reduced pair, one fused add.
        assert best is not None and best.op == "kf_addmul"
        assert kernel_cost(best) == 2.0


class TestCost:
    def test_fused_is_one_launch(self):
        """A ``kf_*`` op costs one launch vs the spelled subtree."""
        spelled = _silu_term()
        fused = Op.make("kf_silu", X, X)
        assert kernel_cost(fused) < kernel_cost(spelled)

    def test_weights_order(self):
        """pow > mul under the op-weight axis."""
        assert kernel_cost(Op.make("pow", X, Const(2))) > kernel_cost(
            Op.make("mul", X, X)
        )
        assert kernel_cost(X) == 0.0


class TestSink:
    def test_port_conformance(self):
        """KernelSink satisfies the Sink port."""
        sink = KernelSink()
        assert isinstance(sink, Sink)
        assert "kf_silu" in sink.supported_ops
        assert sink.executors == {}
        assert sink.ops.torch_bindings["add"] is not None

    def test_lower_verify_equal(self):
        """Spelled vs fused lowerings verify numerically equal."""
        sink = KernelSink()
        x = torch.randn(4, 8, dtype=torch.float64)
        spelled = IR(root=_silu_term(), inputs=[X], input_names={"x"})
        fused = IR(
            root=Op.make("kf_silu", X, X),
            inputs=[X],
            input_names={"x"},
        )
        ref = sink.lower(spelled)
        opt = sink.lower(fused)
        rep = sink.verify(ref, opt, (x,))
        assert rep.passed and rep.max_abs == 0.0

    @pytest.mark.parametrize(
        "spelled,fused",
        [
            (
                _softmax_term(),
                Op.make("kf_softmax", X, dim=-1),
            ),
            (
                _layernorm_term(),
                Op.make("kf_layernorm", X, dim=-1, eps=1e-05),
            ),
            (
                _rmsnorm_term(),
                Op.make("kf_rmsnorm", X, dim=-1, eps=1e-06),
            ),
            (
                Op.make(
                    "mul",
                    Op.make("mul", X, Op.make("sigmoid", X)),
                    Y,
                ),
                Op.make("kf_swiglu", X, Y),
            ),
            (
                Op.make(
                    "mul",
                    Op.make("mul", X, Op.make("sigmoid", X)),
                    Op.make("sigmoid", Y),
                ),
                Op.make("kf_silu_sigmoid", X, Y),
            ),
            (
                Op.make(
                    "add",
                    Op.make("mul", X, Y),
                    Op.make("tanh", Y),
                ),
                Op.make("kf_addmul_tanh", X, Y, Y),
            ),
        ],
        ids=[
            "softmax",
            "layernorm",
            "rmsnorm",
            "swiglu",
            "silu_sigmoid",
            "addmul_tanh",
        ],
    )
    def test_new_fused_ops_verify(self, spelled, fused):
        """Each new kf_* composite is numerically equal to its site."""
        sink = KernelSink()
        x = torch.randn(4, 8, dtype=torch.float64)
        y = torch.randn(4, 8, dtype=torch.float64)
        inputs = [v for v in (X, Y) if repr(v) in repr(spelled)]
        ir_s = IR(
            root=spelled,
            inputs=inputs,
            input_names={v.name for v in inputs},
        )
        ir_f = IR(
            root=fused, inputs=inputs, input_names=ir_s.input_names
        )
        rep = sink.verify(
            sink.lower(ir_s),
            sink.lower(ir_f),
            (x, y)[: len(inputs)],
        )
        assert rep.passed and rep.max_abs < 1e-9

    def test_chain_fused_ops_verify(self):
        """softmax_mm / mm_scale / mm_gelu composites verify equal."""
        sink = KernelSink()
        w = Var("w", TensorType((8, 4)))
        v = Var("v", TensorType((4, 4)))
        qk = Op.make("matmul", X, w)
        spelled = Op.make(
            "matmul",
            Op.make(
                "div",
                Op.make("exp", qk),
                Op.make(
                    "sum", Op.make("exp", qk), dim=-1, keepdim=True
                ),
            ),
            v,
        )
        fused = Op.make("kf_softmax_mm", qk, v, dim=-1)
        x = torch.randn(4, 8, dtype=torch.float64)
        wv = torch.randn(8, 4, dtype=torch.float64)
        vv = torch.randn(4, 4, dtype=torch.float64)
        inputs = [X, w, v]
        ir_s = IR(
            root=spelled,
            inputs=inputs,
            input_names={t.name for t in inputs},
        )
        ir_f = IR(
            root=fused, inputs=inputs, input_names=ir_s.input_names
        )
        rep = sink.verify(
            sink.lower(ir_s), sink.lower(ir_f), (x, wv, vv)
        )
        assert rep.passed and rep.max_abs < 1e-9
        # the mm epilogue twins — scale and gelu
        for s_term, f_term in (
            (
                Op.make("mul", qk, Const(0.125)),
                Op.make("kf_mm_scale", X, w, Const(0.125)),
            ),
            (
                Op.make("gelu", qk),
                Op.make("kf_mm_gelu", X, w),
            ),
        ):
            ins = [X, w]
            i_s = IR(
                root=s_term,
                inputs=ins,
                input_names={t.name for t in ins},
            )
            i_f = IR(
                root=f_term,
                inputs=ins,
                input_names=i_s.input_names,
            )
            rep2 = sink.verify(
                sink.lower(i_s), sink.lower(i_f), (x, wv)
            )
            assert rep2.passed and rep2.max_abs < 1e-9

    def test_mm_epilogues_verify(self):
        """The mm epilogue composites verify against spelled mm+b."""
        sink = KernelSink()
        w = Var("w", TensorType((8, 8)))
        b = Var("b", TensorType((8,)))
        x = torch.randn(4, 8, dtype=torch.float64)
        wv = torch.randn(8, 8, dtype=torch.float64)
        bv = torch.randn(8, dtype=torch.float64)
        inputs = [X, w, b]
        ir_s = IR(
            root=Op.make("add", Op.make("matmul", X, w), b),
            inputs=inputs,
            input_names={"x", "w", "b"},
        )
        ir_f = IR(
            root=Op.make("kf_mm_bias", X, w, b),
            inputs=inputs,
            input_names=ir_s.input_names,
        )
        rep = sink.verify(
            sink.lower(ir_s), sink.lower(ir_f), (x, wv, bv)
        )
        assert rep.passed and rep.max_abs == 0.0

    def test_compile_fused_off_runs_eager(self):
        """``compile_fused=False`` evaluates the same program."""
        sink = KernelSink(compile_fused=False)
        x = torch.randn(4, 8, dtype=torch.float64)
        fused = IR(
            root=Op.make("kf_silu", X, X),
            inputs=[X],
            input_names={"x"},
        )
        out = sink.lower(fused)(x)
        assert torch.allclose(out, x * torch.sigmoid(x))

    def test_executor_is_callable(self):
        """The executor satisfies the callable Executor surface."""
        e = KernelExecutor(
            IR(root=X, inputs=[X], input_names={"x"}), {}
        )
        x = torch.randn(4, 8, dtype=torch.float64)
        assert torch.equal(e(x), x) and torch.equal(e.forward(x), x)

    def test_unsupported_op_is_honest(self):
        """An op outside the table raises, never silently drops."""
        sink = KernelSink()
        ir = IR(
            root=Op.make("conv2d", X),
            inputs=[X],
            input_names={"x"},
        )
        with pytest.raises(KeyError, match="cannot lower"):
            sink.lower(ir)(torch.randn(4, 8))

    def test_verify_rejects_unequal(self):
        """verify fails on a genuinely different program."""
        sink = KernelSink(compile_fused=False)
        x = torch.randn(4, 8, dtype=torch.float64)
        a = IR(root=X, inputs=[X], input_names={"x"})
        b = IR(
            root=Op.make("mul", X, Const(3.0)),
            inputs=[X],
            input_names={"x"},
        )
        rep = sink.verify(sink.lower(a), sink.lower(b), (x,))
        assert not rep.passed


class TestBoard:
    def test_episode_pays_and_certifies(self):
        """A kernel episode: fuse move pays, terminal certifies."""
        b = KernelBoard(_silu_term())
        st = b.observe()
        assert "kern_fuse_silu" in st.matches
        assert st.ops == frozenset({"mul", "sigmoid"})
        _s, rep = b.step("kern_fuse_silu")
        assert rep.reward > 0
        _s, rep = b.step(STOP)
        assert rep.terminal and rep.certificate_ok
        assert b.extract_best().op == "kf_silu"

    def test_verify_law_referee(self):
        """The injected gauntlet derives the scheduled extraction."""
        from mlopt_discovery.verifier import verify_law

        b = KernelBoard(_silu_term(), verifier=verify_law)
        b.observe()
        b.step("kern_fuse_silu")
        _s, rep = b.step(STOP)
        assert rep.certificate_ok is True

    def test_step_cost_charges_moves(self):
        """Firing a rule pays step_cost; stop is free."""
        b = KernelBoard(_silu_term(), step_cost=0.05)
        b.observe()
        _s, rep = b.step("kern_div_recip")  # no site — pure charge
        assert rep.reward == pytest.approx(-0.05)
        _s, rep = b.step(STOP)
        assert rep.reward == 0.0 and rep.terminal


class TestPlayDomain:
    def test_registered_and_playable(self):
        """The kernel domain registers; scripted plays an episode."""
        from gamekit import engine

        from mlopt_discovery import play

        d = play.DOMAINS["kernel"]
        case = d.cases(0, 4)[0]
        traj = engine.run_episode(d.board_of(case), d.scripted(), 8)
        assert traj.reports

    def test_linear_plays_with_shipped_prior(self):
        """The registered linear arm plays kernel under KERNEL_SPEC."""
        from gamekit import engine
        from gamekit.players import registry

        from mlopt_discovery import play

        mk = registry.make("linear", "kernel", seed=0, ctx={})
        d = play.DOMAINS["kernel"]
        traj = engine.run_episode(
            d.board_of(d.cases(1, 1)[0]), mk(0), 8
        )
        assert traj.reports

    def test_deliver_verifies(self):
        """--deliver lowers the extraction and verifies it."""
        from gamekit import engine

        from mlopt_discovery import play

        d = play.DOMAINS["kernel"]
        case = d.cases(0, 8)[0]
        board = d.board_of(case)
        traj = engine.run_episode(board, d.scripted(), 8)
        res = d.deliver(board, case, traj)
        assert res["delivered"] and res["verified"]
