"""mlopt → `.hnl` → CatoptKit: terms emitted as programs, evaluated.

The frontier the kit existed for: a mlopt *term* — the same object the
e-graph rewrites — emitted as a `.hnl` program and evaluated through
the assembled runtime by ``lab/kit/term_hnl.py``, over the compiled
``hnl-run`` subprocess door (``packages/mlopt-kit/src/bin/hnl-run.rs``).

What this proves:

* a **real artifact** — ``TorchSource``'s ``torch.export`` IR of an
  eval-mode ``Conv2d → BatchNorm2d`` — emits and evaluates to the
  torch-fp64 answer;
* a **law's two forms** — ``conv_channel_scale``'s
  ``mul(conv2d(x,W,b), s-view) -> conv2d(x, W*s', b*s)`` via
  ``meta.apply_rewrite_at`` — both evaluate, value-equal, through the
  runtime, and ``dag_cost`` credits the fold (the absorbed mul/reshape
  are param-only subtrees — compile-time work);
* an **optimization decision costed through the runtime** — the
  conv+BN fold's deployed form ``conv2d(x, W2, b2)`` vs the spelled
  ``batch_norm(conv2d(x,W,b), …)``, where ``W2``/``b2`` are themselves
  mlopt subterms materialized through the kit (``s = gamma*rsqrt(var+eps)``
  rides the ``{"hnl"}`` marshaling channel — the surface has no
  elementwise ``rsqrt``, the named gap), and per-call cost is measured
  through ``hnl-run --repeat``, not estimated;
* the bound is **honest** — ops outside host-tensor's 18-op surface
  refuse by name.

Marked ``e2e``: the suite crosses an external engine boundary (the
handl tree-walk runtime, via subprocess).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from mlopt_core import meta
from mlopt_core.cost.basic import flops_cost
from mlopt_core.cost.params import dag_cost
from mlopt_core.ir import Op, Param, TensorType, Var
from mlopt_core.laws import CONV_FOLD_RULES

from lab.kit import term_hnl as kh

pytestmark = pytest.mark.e2e

_REPO = Path(__file__).resolve().parents[1]
_BIN = Path(
    os.environ.get(
        "MLOPT_HNL_RUN", _REPO / "target" / "debug" / "hnl-run"
    )
)


def _build_bin() -> Path:
    """The compiled ``hnl-run``, building it if absent."""
    if _BIN.exists():
        return _BIN
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo not on PATH — hnl-run unavailable")
    env = dict(os.environ, CARGO_PROFILE_DEV_DEBUG="0")
    out = subprocess.run(
        [
            cargo,
            "build",
            "-p",
            "mlopt-kit",
            "--bin",
            "hnl-run",
            "--locked",
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if out.returncode != 0 or not _BIN.exists():
        pytest.skip(f"hnl-run build failed: {out.stderr[-400:]}")
    return _BIN


@pytest.fixture(scope="module")
def kit() -> kh.KitRunner:
    """A runner over the compiled binary."""
    return kh.KitRunner(_build_bin())


def _f(t: torch.Tensor) -> dict:
    """A torch tensor as a ``{"shape", "data"}`` feed entry."""
    return {
        "shape": list(t.shape),
        "data": t.detach().reshape(-1).tolist(),
    }


def _assert_close(got: dict, ref: torch.Tensor, tol: float) -> None:
    """A wire tensor vs a torch reference, elementwise."""
    assert got["shape"] == list(ref.shape)
    got_t = torch.tensor(got["data"], dtype=torch.float64)
    diff = (got_t - ref.reshape(-1)).abs().max().item()
    assert diff <= tol, f"max abs diff {diff} > {tol}"


# ---------------------------------------------------------------------------
#  1. The real artifact — a torch.export IR end to end
# ---------------------------------------------------------------------------


class _ConvBN(nn.Module):
    """``BatchNorm2d(Conv2d(x))`` in eval mode — the fold site."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 6, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv then batch norm."""
        return self.bn(self.conv(x))


def test_torchsource_conv_bn_evaluates_through_the_kit(
    kit: kh.KitRunner,
) -> None:
    """``torch.export`` IR → `.hnl` → kit → the torch-fp64 answer.

    This is the pipeline's own artifact — ``aten.conv2d`` and
    ``aten.batch_norm``'s full attr tail (``training=False, momentum,
    eps, cudnn_enabled``) — not a hand-spelled term.
    """
    from mlopt_torch.adapters import TorchSource

    torch.manual_seed(0)
    mod = _ConvBN().eval().double()
    x = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    ir, tensors = TorchSource().to_ir(mod, x)

    feed = {name: _f(t) for name, t in tensors.items()}
    feed["x"] = _f(x)
    out = kit.tensor(ir.root, feed)

    _assert_close(out, mod(x).detach(), tol=1e-12)


def test_emit_renders_the_canonical_call(kit: kh.KitRunner) -> None:
    """The emitted program is real `.hnl` — attr marshaling included.

    ``eps=1e-05`` must land as ``0.00001``: the handl lexer does not
    read scientific notation (``1e-05`` lexes as ``1`` ``e`` ``-05``).
    """
    from mlopt_torch.adapters import TorchSource

    mod = _ConvBN().eval().double()
    x = torch.zeros(1, 4, 8, 8, dtype=torch.float64)
    ir, _ = TorchSource().to_ir(mod, x)
    assert kh.emit(ir.root) == (
        "tensor.batch_norm(tensor.conv2d(x, p_conv_weight, "
        "p_conv_bias, [1, 1], [1, 1]), p_bn_weight, p_bn_bias, "
        "b_bn_running_mean, b_bn_running_var, 0.00001)"
    )
    # A program binds every leaf — an incomplete feed refuses by name.
    with pytest.raises(ValueError, match="p_conv_weight"):
        kh.emit_program(ir.root, {})


# ---------------------------------------------------------------------------
#  2. A law's two forms, both evaluated through the runtime
# ---------------------------------------------------------------------------


def _conv_fold_site() -> tuple[Op, dict[str, Var | Param]]:
    """``mul(conv2d(x, W, b), ŝ)`` — the conv_channel_scale LHS."""
    T = TensorType
    leaves = {
        "x": Var("x", T((1, 2, 4, 4))),
        "w": Param("w", T((2, 2, 3, 3))),
        "b": Param("b", T((2,))),
        "sv": Param("sv", T((1, 2, 1, 1))),
    }
    lhs = Op.make(
        "mul",
        Op.make("conv2d", leaves["x"], leaves["w"], leaves["b"]),
        leaves["sv"],
    )
    return lhs, leaves


def test_conv_channel_scale_both_forms_evaluate_equal(
    kit: kh.KitRunner,
) -> None:
    """The minted rewrite's two forms, value-equal through the kit."""
    lhs, _ = _conv_fold_site()
    rule = next(
        r for r in CONV_FOLD_RULES if r.name == "conv_channel_scale"
    )
    rhs = meta.apply_rewrite_at(rule, lhs, ())
    assert rhs is not None, "the guarded law must fire on the site"
    assert kh.emit(rhs) == (
        "tensor.conv2d(x, tensor.mul(w, "
        "tensor.reshape(sv, [2, 1, 1, 1])), "
        "tensor.mul(b, tensor.reshape(sv, [2])))"
    )

    # Dyadic fixtures — the two forms are *exactly* f64-equal, not
    # tolerance-close (the kit's own conv_bn tests' convention).
    feed = {
        "x": _f(
            torch.arange(32, dtype=torch.float64).reshape(1, 2, 4, 4)
            / 8
        ),
        "w": _f(
            torch.arange(36, dtype=torch.float64).reshape(2, 2, 3, 3)
            / 16
            - 1
        ),
        "b": _f(torch.tensor([0.5, -0.25])),
        "sv": _f(torch.tensor([[[[2.0]], [[0.5]]]])),
    }
    a = kit.tensor(lhs, feed)
    c = kit.tensor(rhs, feed)
    assert a == c, "the law's two forms must agree through the kit"

    # The term-level cost credits the fold: the absorbed mul/reshape
    # are param-only subtrees — compile-time work — so `dag_cost`
    # prices the folded form as the bare conv.
    spelled = dag_cost(lhs, flops_cost)
    folded = dag_cost(rhs, flops_cost)
    assert folded < spelled


# ---------------------------------------------------------------------------
#  3. The conv+BN fold decision, costed through the actual runtime
# ---------------------------------------------------------------------------

#: `s[c] = gamma[c]*rsqrt(var[c]+eps)` per channel — the one piece of the fold
#: the op surface cannot spell (no elementwise rsqrt), so it rides the
#: `{"hnl"}` marshaling channel: the (C,) stats materialize to lists and
#: scalar `**` finishes each channel — `hnl/conv_bn.hnl`'s own escape.
_CHANNEL_SCALES = (
    "(gs, vs, eps) ~> while { gs: gs, vs: vs, acc: [] } {\n"
    "  (gs, vs, acc) ~> match gs {\n"
    "    []: { out: acc },\n"
    "    [g, ...gt]: match vs {\n"
    "      []: { out: acc },\n"
    "      [v, ...vt]: { state: { gs: gt,\n"
    "                            vs: vt,\n"
    "                            acc: [...acc, g / ((v + eps) ** 0.5)] } }\n"
    "    }\n"
    "  }\n"
    "}"
)


def test_conv_bn_fold_decision_costed_through_the_runtime(
    kit: kh.KitRunner,
) -> None:
    """Folded-vs-spelled, evaluated and priced by the kit itself.

    The deployed form is ``conv2d(x, W2, b2)`` — a mlopt term whose
    ``W2``/``b2`` leaves are *materialized through the kit* from mlopt
    subterms (``mul(w, reshape(s, (C,1,1,1)))`` /
    ``add(mul(b,s), sub(be, mul(s,m)))`` — every op on the surface).
    Per-call cost is `hnl-run --repeat` wall — the measured decision,
    not the estimated one.
    """
    T = TensorType
    n, c_in, c_out, hw, eps = 1, 4, 8, 24, 1e-5
    x = Var("x", T((n, c_in, hw, hw)))
    w = Param("w", T((c_out, c_in, 1, 1)))
    b = Param("b", T((c_out,)))
    g = Param("g", T((c_out,)))
    be = Param("be", T((c_out,)))
    m = Param("m", T((c_out,)))
    v = Param("v", T((c_out,)))
    s = Param("s", T((c_out,)))
    w2 = Param("w2", T((c_out, c_in, 1, 1)))
    b2 = Param("b2", T((c_out,)))

    unfolded = Op.make(
        "batch_norm",
        Op.make("conv2d", x, w, b),
        g,
        be,
        m,
        v,
        training=False,
        momentum=0.1,
        eps=eps,
        cudnn_enabled=False,
    )
    # The deployed term — the fold absorbed; W2/b2 are materialized
    # weights (leaves), the compile-time half of the decision.
    deployed = Op.make("conv2d", x, w2, b2)
    # …computed by mlopt subterms — the fold's RHS shape, every op
    # on the surface.
    w2_term = Op.make(
        "mul", w, Op.make("reshape", s, shape=(c_out, 1, 1, 1))
    )
    b2_term = Op.make(
        "add",
        Op.make("mul", b, s),
        Op.make("sub", be, Op.make("mul", s, m)),
    )

    torch.manual_seed(2)
    feed = {
        "x": _f(torch.randn(n, c_in, hw, hw, dtype=torch.float64)),
        "w": _f(torch.randn(c_out, c_in, 1, 1, dtype=torch.float64)),
        "b": _f(torch.randn(c_out, dtype=torch.float64)),
        "g": _f(torch.randn(c_out, dtype=torch.float64).abs() + 0.5),
        "be": _f(torch.randn(c_out, dtype=torch.float64)),
        "m": _f(torch.randn(c_out, dtype=torch.float64)),
        "v": _f(torch.rand(c_out, dtype=torch.float64) + 0.25),
        # The surface gap, marshaled: s = gamma*rsqrt(var+eps) in .hnl.
        "cs": {"hnl": _CHANNEL_SCALES},
        "s": {
            "hnl": (
                "tensor.from_list(cs(tensor.to_list(g), "
                "tensor.to_list(v), 0.00001))"
            )
        },
    }

    # Fold the weights through the kit — the compile-time step.
    folded_w = kit.run(kh.emit_program(w2_term, feed))["__tensor__"]
    folded_b = kit.run(kh.emit_program(b2_term, feed))["__tensor__"]

    a = kit.tensor(unfolded, feed)
    c = kit.tensor(
        deployed, {"x": feed["x"], "w2": folded_w, "b2": folded_b}
    )
    assert (
        torch.tensor(c["data"]) - torch.tensor(a["data"])
    ).abs().max().item() < 1e-12, (
        "the deployed fold must reproduce batch_norm(conv2d) "
        "through the kit"
    )

    # The measured decision: per-call wall over --repeat, in-process.
    reps = 30

    def per_call(term: Op, f: dict) -> float:
        src = kh.emit_program(term, f)
        t1 = time.perf_counter()
        kit.run(src)
        t1 = time.perf_counter() - t1
        tn = time.perf_counter()
        kit.run(src, repeat=reps)
        tn = time.perf_counter() - tn
        return (tn - t1) / (reps - 1)

    deploy_feed = {"x": feed["x"], "w2": folded_w, "b2": folded_b}
    t_unfolded = per_call(unfolded, feed)
    t_deployed = per_call(deployed, deploy_feed)
    assert t_deployed < t_unfolded, (
        f"deployed fold ({t_deployed:.4f}s/call) should beat "
        f"spelled conv+BN ({t_unfolded:.4f}s/call)"
    )
    # And the term-level model agrees — one fused conv vs conv+BN.
    assert dag_cost(deployed, flops_cost) < dag_cost(
        unfolded, flops_cost
    )


# ---------------------------------------------------------------------------
#  4. Generality and the honest bound
# ---------------------------------------------------------------------------


def test_generic_terms_emit_and_evaluate(kit: kh.KitRunner) -> None:
    """A non-conv term — matmul/add/sub/reshape/transpose/concat."""
    T = TensorType
    a = Var("a", T((2, 3)))
    w = Param("w", T((3, 4)))
    bias = Param("bias", T((4,)))
    t = Op.make(
        "sub",
        Op.make("matmul", a, w),
        Op.make(
            "concat",
            Op.make("reshape", bias, shape=(1, 4)),
            Op.make(
                "transpose",
                Op.make("reshape", bias, shape=(4, 1)),
                dim0=0,
                dim1=1,
            ),
            dim=0,
        ),
    )
    ta = torch.randn(2, 3, dtype=torch.float64)
    tw = torch.randn(3, 4, dtype=torch.float64)
    tb = torch.randn(4, dtype=torch.float64)
    out = kit.tensor(t, {"a": _f(ta), "w": _f(tw), "bias": _f(tb)})
    ref = ta @ tw - torch.cat([tb[None, :], tb[:, None].T], dim=0)
    _assert_close(out, ref, tol=1e-12)


def test_multioutput_pack_emits_a_list(kit: kh.KitRunner) -> None:
    """``pack(t0, t1)`` — the multi-output root — lands as a list."""
    T = TensorType
    a = Var("a", T((2,)))
    t = Op.make("pack", Op.make("add", a, a), Op.make("mul", a, a))
    out = kit.eval(t, {"a": _f(torch.tensor([3.0, 4.0]))})
    assert [d["__tensor__"]["data"] for d in out] == [
        [6.0, 8.0],
        [9.0, 16.0],
    ]


def test_unsupported_surface_is_named_not_silent() -> None:
    """Ops outside the 18-op surface refuse by name."""
    T = TensorType
    x = Var("x", T((2, 2)))
    with pytest.raises(kh.UnsupportedOpError, match="sigmoid"):
        kh.emit(Op.make("sigmoid", x, validate=False))
    with pytest.raises(kh.UnsupportedOpError, match="rsqrt"):
        kh.emit(Op.make("add", x, Op.make("rsqrt", x, validate=False)))
    # Eval-mode only: a training-mode batch_norm refuses on `training`.
    bn = Op.make(
        "batch_norm",
        x,
        Param("g", T((2,))),
        Param("b", T((2,))),
        Param("m", T((2,))),
        Param("v", T((2,))),
        training=True,
        momentum=0.1,
        eps=1e-5,
        cudnn_enabled=False,
    )
    with pytest.raises(kh.UnsupportedOpError, match="training"):
        kh.emit(bn)
    # And the fold's spelled RHS names its own gap: `rsqrt`.
    w = Param("w", T((2, 2)))
    g = Param("g", T((2,)))
    v = Param("v", T((2,)))
    spelled_s = Op.make(
        "mul",
        g,
        Op.make("rsqrt", Op.make("add", v, 0.00001, validate=False)),
    )
    with pytest.raises(kh.UnsupportedOpError, match="rsqrt"):
        kh.emit(Op.make("mul", w, spelled_s))


def test_kit_eval_failure_surfaces_the_engine_error(
    kit: kh.KitRunner,
) -> None:
    """A program the kit refuses fails loudly, carrying stderr."""
    with pytest.raises(kh.KitEvalError):
        kit.run("tensor.matmul(tensor.from_list([1]), unbound_name)")
