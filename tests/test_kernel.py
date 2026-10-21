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


class TestLaws:
    def test_bank_shape(self):
        """Ten codegen laws; the fused ops take leaf children."""
        assert [r.name for r in KERNEL_RULES] == [
            "kern_fuse_silu",
            "kern_fuse_softsign",
            "kern_fuse_addmul",
            "kern_fuse_swiglu",
            "kern_fuse_softmax",
            "kern_fuse_layernorm",
            "kern_fuse_mm_bias",
            "kern_fuse_mm_scale_bias",
            "kern_square_strength",
            "kern_div_recip",
        ]

    def test_fuse_fires_and_extracts(self):
        """``mul(a, sigmoid(b))`` gains an extractable ``kf_silu``."""
        eg = EGraph()
        root = eg.add_term(_silu_term())
        assert eg.apply_rule(KERNEL_RULES[0], root)
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
        assert eg.apply_rule(KERNEL_RULES[1], root)
        assert eg.extract_best(root, kernel_cost).op == "kf_softsign"
        eg2 = EGraph()
        root2 = eg2.add_term(
            Op.make("add", Op.make("mul", X, Y), Op.make("tanh", Y))
        )
        assert eg2.apply_rule(KERNEL_RULES[2], root2)
        assert eg2.extract_best(root2, kernel_cost).op == "kf_addmul"

    def test_strength_reductions(self):
        """pow(x,2) → x·x and x/c → x·(1/c) both fire and verify."""
        eg = EGraph()
        root = eg.add_term(Op.make("pow", X, Const(2)))
        assert eg.apply_rule(KERNEL_RULES[8], root)
        assert eg.extract_best(root, kernel_cost) == Op.make(
            "mul", X, X
        )
        eg2 = EGraph()
        t = Op.make("div", X, Const(4.0))
        root2 = eg2.add_term(t)
        assert eg2.apply_rule(KERNEL_RULES[9], root2)
        best = eg2.extract_best(root2, kernel_cost)
        assert best == Op.make("mul", X, Const(0.25))
        cert = eg2.certificate(t, best)
        verify_certificate(t, cert)

    def test_recip_declines_nonconst(self):
        """The recip fold needs a numeric Const divisor."""
        eg = EGraph()
        root = eg.add_term(Op.make("div", X, Y))
        assert not eg.apply_rule(KERNEL_RULES[9], root)

    def test_swiglu_beats_shallow_silu(self):
        """``(x·σ(x))·y`` extracts the one-launch kf_swiglu."""
        eg = EGraph()
        t = Op.make("mul", Op.make("mul", X, Op.make("sigmoid", X)), Y)
        root = eg.add_term(t)
        eg.apply_rule(KERNEL_RULES[0], root)  # silu fires inside too
        eg.apply_rule(KERNEL_RULES[3], root)
        best = eg.extract_best(root, kernel_cost)
        assert best is not None and best.op == "kf_swiglu"

    def test_softmax_fires_and_derives_dim(self):
        """``exp(u)/Σexp(u)`` folds to ``kf_softmax`` with the dim."""
        eg = EGraph()
        root = eg.add_term(_softmax_term())
        assert eg.apply_rule(KERNEL_RULES[4], root)
        best = eg.extract_best(root, kernel_cost)
        assert best == Op.make("kf_softmax", X, dim=-1)

    def test_softmax_declines_dropped_dim(self):
        """``keepdim=False`` broadcasts wrongly — the site declines."""
        eg = EGraph()
        root = eg.add_term(_softmax_term(keepdim=False))
        assert not eg.apply_rule(KERNEL_RULES[4], root)

    def test_layernorm_fires_and_derives_eps(self):
        """The mean/var/rsqrt block folds to ``kf_layernorm``."""
        eg = EGraph()
        t = _layernorm_term()
        root = eg.add_term(t)
        assert eg.apply_rule(KERNEL_RULES[5], root)
        best = eg.extract_best(root, kernel_cost)
        assert best == Op.make("kf_layernorm", X, dim=-1, eps=1e-05)
        cert = eg.certificate(t, best)
        verify_certificate(t, cert)

    def test_layernorm_declines_bad_site(self):
        """A non-numeric eps or a non-square power vetoes the fold."""
        eg = EGraph()
        root = eg.add_term(_layernorm_term(eps=X))
        assert not eg.apply_rule(KERNEL_RULES[5], root)
        eg2 = EGraph()
        root2 = eg2.add_term(_layernorm_term(power=Const(3)))
        assert not eg2.apply_rule(KERNEL_RULES[5], root2)

    def test_mm_epilogues_fire(self):
        """mm+b and mm·s+b fold to their one-launch epilogue ops."""
        w = Var("w", TensorType((8, 8)))
        b = Var("b", TensorType((8,)))
        eg = EGraph()
        root = eg.add_term(Op.make("add", Op.make("matmul", X, w), b))
        assert eg.apply_rule(KERNEL_RULES[6], root)
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
        eg2.apply_rule(KERNEL_RULES[2], root2)
        eg2.apply_rule(KERNEL_RULES[7], root2)
        assert (
            eg2.extract_best(root2, kernel_cost).op == "kf_mm_sb"
        )


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
                Op.make(
                    "mul",
                    Op.make("mul", X, Op.make("sigmoid", X)),
                    Y,
                ),
                Op.make("kf_swiglu", X, Y),
            ),
        ],
        ids=["softmax", "layernorm", "swiglu"],
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
