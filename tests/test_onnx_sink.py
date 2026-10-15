"""The ONNX sink — ADR 0006 stage 1, verified through onnxruntime.

Two things are under test here, and the second is the experiment:

1. that :class:`catopt_onnx.OnnxSink` lowers catopt IR to a runnable
   ONNX artifact whose numbers match the source program, and
2. that it does so as a *port implementation* — the pipeline drives it
   through ``catopt_core.ports`` exactly as it drives the torch sink,
   with no branch anywhere for "which backend".

The op-coverage table is deliberately broad: every entry is a small
torch program exported to IR, lowered by the sink, and compared
against torch's own result.  Ops outside the mapping are exercised by
:func:`test_unsupported_op_is_refused` instead — the ADR's *refuse,
never drop* rule.

``onnx``/``onnxruntime`` are declared dependencies of the
``catopt-onnx`` package; the module skips cleanly when they are
absent from the environment so the rest of the suite stays green.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
pytestmark = pytest.mark.e2e  # boundary-purpose file: crosses an external engine boundary
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx
from catopt_core.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
)
from catopt_core.ports import (
    Capabilities,
    Executor,
    Sink,
    VerifyResult,
)
from catopt_onnx import (
    ONNX_BINDINGS,
    OnnxExecutor,
    OnnxLoweringError,
    OnnxSink,
    UnsupportedOnnxOpError,
    verify_equiv,
)
from catopt_orchestrator.optimize import Optimizer
from catopt_torch.adapters import TorchSource
from catopt_torch.torch_bridge import export_to_ir

DT = torch.float32

#: Every op name the sink claims, as a set for cheap membership tests.
SUPPORTED = frozenset(ONNX_BINDINGS)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


class _Program(nn.Module):
    """A one-input program with its free constants as parameters."""

    def __init__(self, fn, consts=()):
        """Wrap ``fn`` and register each constant as a parameter."""
        super().__init__()
        self.fn = fn
        for i, c in enumerate(consts):
            self.register_parameter(f"c{i}", nn.Parameter(c))
        self._n = len(consts)

    def forward(self, x):
        """Apply the wrapped function to ``x`` and the constants."""
        consts = [getattr(self, f"c{i}") for i in range(self._n)]
        return self.fn(x, *consts)


def _lower(fn, x, consts=(), *, sink=None, input_dtypes=None):
    """Export ``fn`` to IR and lower it; return ``(module, ir, leaves)``."""
    model = _Program(fn, consts).eval()
    ir, leaves = export_to_ir(model, (x,))
    sink = sink or OnnxSink(dtype="float32", input_dtypes=input_dtypes)
    return model, sink.lower(ir, leaves), ir, leaves


def _max_abs(onnx_out, torch_out) -> float:
    """Largest absolute difference between an ONNX and a torch result."""
    left = np.asarray(onnx_out)
    right = (
        torch_out.numpy()
        if isinstance(torch_out, torch.Tensor)
        else np.asarray(torch_out)
    )
    if left.shape != right.shape:
        raise AssertionError(
            f"shape mismatch: onnx {left.shape} vs torch {right.shape}"
        )
    if left.dtype == np.bool_:
        return 0.0 if np.array_equal(left, right) else 1.0
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _round_trip(name, fn, x, consts=(), *, tol=1e-5, **kw):
    """Lower ``fn`` and assert its ONNX result matches torch's."""
    model, executor, _, _ = _lower(fn, x, consts, **kw)
    with torch.no_grad():
        reference = model(x)
    got = executor.forward(x)
    diff = _max_abs(got, reference)
    assert diff <= tol, f"{name}: max_abs={diff:.3e} > {tol:.1e}"


def _ops_of(term) -> set[str]:
    """Return every op name in a term."""
    found: set[str] = set()
    stack = [term]
    while stack:
        node = stack.pop()
        if isinstance(node, Op):
            found.add(node.op)
            stack.extend(node.args)
    return found


# ---------------------------------------------------------------------------
#  the port surface
# ---------------------------------------------------------------------------


def test_sink_conforms_to_the_sink_port():
    """A runtime check sees a real ``Sink`` (and ``Capabilities``)."""
    sink = OnnxSink()
    assert isinstance(sink, Sink)
    assert isinstance(sink, Capabilities)


def test_supported_ops_is_the_op_table_key_set():
    """``supported_ops`` names exactly the mapped ops."""
    sink = OnnxSink()
    assert sink.supported_ops == SUPPORTED
    assert "matmul" in sink.supported_ops
    assert "sdpa" not in sink.supported_ops


def test_ops_registry_carries_the_onnx_builders():
    """The ``OpRegistry`` view exposes the ONNX lowering entries."""
    sink = OnnxSink()
    assert set(sink.ops.torch_bindings) == SUPPORTED
    assert sink.ops.attr_schemas is not None


def test_executor_table_is_empty():
    """No carrier families — routing always falls through to ``lower``."""
    assert dict(OnnxSink().executors) == {}


_ONNX_ONLY_SCRIPT = r"""
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "catopt_torch",
                                  "catopt_carriers", "catopt_cuda"):
            raise ModuleNotFoundError(f"blocked import: {name}")
        return None


sys.meta_path.insert(0, _Blocker())

import importlib
import pkgutil

import catopt_onnx

# Every submodule of the ONNX backend must import with torch blocked —
# the boundary claim is that it depends on catopt_core alone.
for _info in pkgutil.walk_packages(
    catopt_onnx.__path__, "catopt_onnx."
):
    importlib.import_module(_info.name)

assert "torch" not in sys.modules, sorted(sys.modules)
assert "catopt_torch" not in sys.modules
assert "catopt_carriers" not in sys.modules
assert "catopt_orchestrator" not in sys.modules

from catopt_core.ports import Sink
assert isinstance(catopt_onnx.OnnxSink(), Sink)
print("onnx-imported-torch-free")
"""


def test_the_onnx_backend_imports_without_torch():
    """``catopt_onnx`` never pulls the torch backend family.

    This is the port-boundary claim in its strongest form: the second
    backend is a *peer* of the torch one, reachable through
    ``catopt_core.ports`` and importing nothing from it.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _ONNX_ONLY_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"torch-free ONNX import failed:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "onnx-imported-torch-free" in proc.stdout


def test_executor_conforms_to_the_executor_port():
    """The lowered runnable is an ``Executor`` (``forward``)."""
    x = torch.randn(2, 3, dtype=DT)
    _, executor, _, _ = _lower(lambda t: t + 1.0, x)
    assert isinstance(executor, OnnxExecutor)
    assert isinstance(executor, Executor)


def test_executor_refuses_the_wrong_arity():
    """A call with the wrong number of inputs is a clear error."""
    x = torch.randn(2, 3, dtype=DT)
    _, executor, _, _ = _lower(lambda t: t + 1.0, x)
    with pytest.raises(ValueError, match="expected 1 input"):
        executor.forward(x, x)


# ---------------------------------------------------------------------------
#  the artifact
# ---------------------------------------------------------------------------


def test_lowered_model_is_a_checked_onnx_artifact():
    """The artifact passes ONNX's checker and pins its opset."""
    x = torch.randn(2, 3, dtype=DT)
    _, executor, _, _ = _lower(lambda t: t * 2.0, x)
    model = executor.model
    assert model.ir_version == OnnxSink().ir_version
    assert model.opset_import[0].version == 18
    assert model.producer_name == "catopt-onnx"
    onnx.checker.check_model(model)


def test_save_writes_a_reloadable_artifact(tmp_path):
    """``save`` is the delivery step: the file reopens and runs."""
    x = torch.randn(2, 3, dtype=DT)
    _, executor, _, _ = _lower(lambda t: t + 1.0, x)
    path = executor.save(tmp_path / "program.onnx")
    reloaded = onnx.load(str(path))
    onnx.checker.check_model(reloaded)
    again = OnnxExecutor(reloaded)
    assert np.allclose(
        np.asarray(again.forward(x)), np.asarray(executor.forward(x))
    )


def test_shared_subterms_emit_one_node_each():
    """Hash-consed subterms become one node referenced twice."""
    x = torch.randn(2, 3, dtype=DT)

    def fn(t):
        branch = t * 2.0
        return branch + branch

    _, executor, _, _ = _lower(fn, x)
    kinds = [n.op_type for n in executor.model.graph.node]
    assert kinds.count("Mul") == 1
    assert kinds.count("Add") == 1


def test_a_program_that_is_its_input_still_has_a_distinct_output():
    """An identity program emits an ``Identity`` so names stay unique."""
    x = torch.randn(2, 3, dtype=DT)
    _, executor, _, _ = _lower(lambda t: t.contiguous(), x)
    assert executor.output_names != executor.input_names
    assert np.allclose(np.asarray(executor.forward(x)), x.numpy())


# ---------------------------------------------------------------------------
#  multi-output — the ``pack`` root, lowered to ONNX's output list
# ---------------------------------------------------------------------------


class _TwoHeads(nn.Module):
    """A module returning a *tuple* — the ``pack`` root."""

    def __init__(self, width=6):
        """Build two independent heads."""
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, 2)

    def forward(self, x):
        """Return ``(relu(x @ W₁ᵀ), softmax(x @ W₂ᵀ))``."""
        return (
            torch.relu(self.first(x)),
            torch.softmax(self.second(x), dim=-1),
        )


def test_pack_is_in_the_op_surface():
    """``pack`` is lowerable — extraction must price it."""
    assert "pack" in OnnxSink().supported_ops


def test_a_nested_pack_is_refused():
    """A pack that is a *value* has no ONNX spelling."""
    var = Var("x", TensorType((2, 2)))
    inner = Op.make("pack", var, var, validate=False)
    ir = IR(root=Op.make("add", inner, var, validate=False))
    with pytest.raises(UnsupportedOnnxOpError, match="nested pack"):
        OnnxSink().lower(ir, {})


def test_multi_output_lowering_keeps_every_output():
    """Both of a tuple-returning module's outputs survive the lowering.

    This is the case ADR 0006's prerequisite closed on the torch side
    (7 outputs, 1 delivered).  ONNX is natively multi-output, so the
    ``pack`` root becomes the graph's output list and nothing is
    truncated.
    """
    torch.manual_seed(0)
    model = _TwoHeads().eval()
    x = torch.randn(3, 6)
    ir, leaves = export_to_ir(model, (x,))
    assert ir.root.op == "pack"
    assert len(ir.roots) == 2
    executor = OnnxSink().lower(ir, leaves)
    assert len(executor.output_names) == 2
    with torch.no_grad():
        reference = model(x)
    got = executor.forward(x)
    assert isinstance(got, tuple) and len(got) == 2
    for leaf, expected in zip(got, reference, strict=True):
        assert np.allclose(
            np.asarray(leaf), expected.numpy(), atol=1e-6
        )


def test_multi_output_round_trips_through_the_pipeline():
    """A tuple-returning model optimizes and verifies through ONNX."""
    torch.manual_seed(0)
    model = _TwoHeads().eval()
    x = torch.randn(3, 6)
    optimizer = Optimizer(source=TorchSource(), sink=OnnxSink())
    result = optimizer.optimize(model, x, max_iterations=4)
    assert result.verified is not None
    assert result.verified.passed
    with torch.no_grad():
        reference = model(x)
    got = result.module.forward(x)
    assert len(got) == len(reference)
    for leaf, expected in zip(got, reference, strict=True):
        assert np.allclose(
            np.asarray(leaf), expected.numpy(), atol=1e-6
        )


# ---------------------------------------------------------------------------
#  op coverage — every mapped op against torch
# ---------------------------------------------------------------------------

_X = torch.randn(4, 6, dtype=DT)
_IX = torch.tensor([[0, 1, 2, 3, 4, 5]] * 4)
_IDX = torch.tensor([0, 2])


OP_CASES: list[tuple] = [
    ("add", lambda t: t + 1.0),
    ("sub", lambda t: t - 1.0),
    ("rsub", lambda t: 1.0 - t),
    ("mul", lambda t: t * 2.0),
    ("div", lambda t: t / 2.0),
    ("div_floor", lambda t: torch.div(t, 2.0, rounding_mode="floor")),
    ("div_trunc", lambda t: torch.div(t, 2.0, rounding_mode="trunc")),
    ("pow", lambda t: t**2),
    ("neg", lambda t: -t),
    ("abs", lambda t: t.abs()),
    ("sign", lambda t: t.sign()),
    ("floor", lambda t: t.floor()),
    ("ceil", lambda t: t.ceil()),
    ("round", lambda t: t.round()),
    ("trunc", lambda t: t.trunc()),
    ("frac", lambda t: t.frac()),
    ("sqrt", lambda t: t.abs().sqrt()),
    ("rsqrt", lambda t: t.abs().add(1).rsqrt()),
    ("square", lambda t: t.square()),
    ("reciprocal", lambda t: t.reciprocal()),
    ("exp", lambda t: t.exp()),
    ("exp2", lambda t: t.exp2()),
    ("expm1", lambda t: t.expm1()),
    ("log", lambda t: t.abs().add(1).log()),
    ("log2", lambda t: t.abs().add(1).log2()),
    ("log10", lambda t: t.abs().add(1).log10()),
    ("log1p", lambda t: t.abs().log1p()),
    ("sin", lambda t: t.sin()),
    ("cos", lambda t: t.cos()),
    ("tan", lambda t: t.tan()),
    ("asin", lambda t: t.clamp(-1, 1).asin()),
    ("acos", lambda t: t.clamp(-1, 1).acos()),
    ("atan", lambda t: t.atan()),
    ("sinh", lambda t: t.sinh()),
    ("cosh", lambda t: t.cosh()),
    ("tanh", lambda t: t.tanh()),
    ("asinh", lambda t: t.asinh()),
    ("acosh", lambda t: t.abs().add(2).acosh()),
    ("atanh", lambda t: t.clamp(-0.5, 0.5).atanh()),
    ("erf", lambda t: t.erf()),
    ("erfc", lambda t: t.erfc()),
    ("maximum", lambda t: torch.maximum(t, torch.zeros_like(t))),
    ("minimum", lambda t: torch.minimum(t, torch.zeros_like(t))),
    ("fmod", lambda t: torch.fmod(t, 2.0)),
    ("remainder", lambda t: torch.remainder(t, 2.0)),
    ("xlogy", lambda t: torch.xlogy(t, t.abs().add(1))),
    ("heaviside", lambda t: torch.heaviside(t, t.new_tensor(0.5))),
    ("lerp", lambda t: torch.lerp(t, t * 2, 0.25)),
    ("addcmul", lambda t: torch.addcmul(t, t, t, value=0.5)),
    ("addcdiv", lambda t: torch.addcdiv(t, t, t.abs().add(1))),
    ("clamp", lambda t: torch.clamp(t, -0.5, 0.5)),
    ("clamp_min", lambda t: torch.clamp_min(t, 0.1)),
    ("clamp_max", lambda t: torch.clamp_max(t, 0.1)),
    ("masked_fill", lambda t: t.masked_fill(t > 0, -1.0)),
    ("where", lambda t: torch.where(t > 0, t, torch.zeros_like(t))),
    ("zeros_like", lambda t: torch.zeros_like(t)),
    ("ones_like", lambda t: torch.ones_like(t)),
    ("full_like", lambda t: torch.full_like(t, 3.0)),
    ("zeros", lambda t: t + torch.zeros(4, 6)),
    ("ones", lambda t: t + torch.ones(4, 6)),
    ("full", lambda t: t + torch.full((4, 6), 2.0)),
    ("new_zeros", lambda t: t + t.new_zeros((4, 6))),
    ("new_ones", lambda t: t + t.new_ones((4, 6))),
    ("new_full", lambda t: t + t.new_full((4, 6), 3.0)),
    ("eq", lambda t: t == 1.0),
    ("ne", lambda t: t != 1.0),
    ("lt", lambda t: t < 0),
    ("le", lambda t: t <= 0),
    ("gt", lambda t: t > 0),
    ("ge", lambda t: t >= 0),
    ("isnan", lambda t: torch.isnan(t)),
    ("isinf", lambda t: torch.isinf(t)),
    ("isfinite", lambda t: torch.isfinite(t)),
    ("logical_not", lambda t: torch.logical_not(t > 0)),
    ("logical_and", lambda t: (t > 0) & (t < 1)),
    ("bitwise_or", lambda t: (t > 0) | (t < 0)),
    ("matmul", lambda t: t @ t.t()),
    (
        "linear",
        lambda t, w, b: F.linear(t, w, b),
        (torch.randn(3, 6, dtype=DT), torch.randn(3, dtype=DT)),
    ),
    (
        "conv2d",
        lambda t, w: F.conv2d(t.reshape(2, 2, 3, 2), w, padding=1),
        (torch.randn(3, 2, 3, 3, dtype=DT),),
    ),
    (
        "conv1d",
        lambda t, w: F.conv1d(t.reshape(2, 2, 6), w, padding=1),
        (torch.randn(3, 2, 3, dtype=DT),),
    ),
    ("sum", lambda t: t.sum()),
    ("sum_dim", lambda t: t.sum(dim=-1, keepdim=True)),
    ("mean", lambda t: t.mean(dim=0)),
    ("amax", lambda t: torch.amax(t, dim=-1)),
    ("amin", lambda t: torch.amin(t, dim=-1)),
    ("prod", lambda t: t.prod(dim=-1)),
    ("argmax", lambda t: torch.argmax(t, dim=-1)),
    ("argmin", lambda t: torch.argmin(t, dim=-1)),
    ("argmax_flat", lambda t: torch.argmax(t)),
    ("max_all", lambda t: torch.max(t)),
    ("min_all", lambda t: torch.min(t)),
    ("max_dim", lambda t: torch.max(t, dim=-1)[0]),
    ("max_index", lambda t: torch.max(t, dim=-1)[1]),
    ("min_dim", lambda t: torch.min(t, dim=-1)[0]),
    ("cumsum", lambda t: torch.cumsum(t, dim=1)),
    ("diag_sum", lambda t: torch.trace(t @ t.t())),
    (
        "linalg_vector_norm",
        lambda t: torch.linalg.vector_norm(t, ord=2, dim=-1),
    ),
    ("count_nonzero", lambda t: torch.count_nonzero(t > 0)),
    ("any", lambda t: (t > 0).any()),
    ("all", lambda t: (t > 0).all(dim=-1)),
    ("softmax", lambda t: F.softmax(t, dim=-1)),
    ("log_softmax", lambda t: F.log_softmax(t, dim=-1)),
    ("relu", lambda t: F.relu(t)),
    ("sigmoid", lambda t: torch.sigmoid(t)),
    ("gelu", lambda t: F.gelu(t)),
    ("silu", lambda t: F.silu(t)),
    ("mish", lambda t: F.mish(t)),
    ("elu", lambda t: F.elu(t)),
    ("leaky_relu", lambda t: F.leaky_relu(t, 0.1)),
    ("softplus", lambda t: F.softplus(t)),
    ("softsign", lambda t: F.softsign(t)),
    ("hardsigmoid", lambda t: F.hardsigmoid(t)),
    ("hardswish", lambda t: F.hardswish(t)),
    ("selu", lambda t: F.selu(t)),
    ("relu6", lambda t: F.relu6(t)),
    ("hardtanh", lambda t: F.hardtanh(t, -1, 1)),
    ("glu", lambda t: F.glu(t, dim=-1)),
    (
        "prelu",
        lambda t, w: F.prelu(t, w),
        (torch.randn(6, dtype=DT),),
    ),
    ("dropout", lambda t: F.dropout(t, 0.5, training=False)),
    ("transpose", lambda t: t.transpose(0, 1)),
    ("t", lambda t: t.t()),
    ("permute", lambda t: t.permute(1, 0)),
    ("movedim", lambda t: torch.movedim(t, 0, 1)),
    ("reshape", lambda t: t.reshape(6, 4)),
    ("flatten", lambda t: t.flatten(0, -1)),
    ("flatten_partial", lambda t: t.reshape(2, 2, 6).flatten(1, -1)),
    ("unsqueeze", lambda t: t.unsqueeze(0)),
    ("squeeze", lambda t: t.unsqueeze(0).squeeze(0)),
    ("expand", lambda t: t.expand(2, 4, 6)),
    ("broadcast_to", lambda t: torch.broadcast_to(t, (2, 4, 6))),
    ("repeat", lambda t: t.repeat(2, 1)),
    ("tile", lambda t: torch.tile(t, (2, 1))),
    ("slice", lambda t: t[..., 1:3]),
    ("narrow", lambda t: t.narrow(1, 1, 2)),
    ("select", lambda t: torch.select(t, 1, 2)),
    ("flip", lambda t: torch.flip(t, [1])),
    ("pad", lambda t: F.pad(t, (1, 2))),
    ("concat", lambda t: torch.cat([t, t], dim=-1)),
    ("stack", lambda t: torch.stack([t, t], dim=0)),
    ("hstack", lambda t: torch.hstack([t, t])),
    ("split_int", lambda t: torch.split(t, 2, dim=-1)[1]),
    ("split_list", lambda t: torch.split(t, [2, 4], dim=-1)[1]),
    ("chunk", lambda t: torch.chunk(t, 4, dim=-1)[2]),
    ("index_select", lambda t: torch.index_select(t, 1, _IDX)),
    ("gather", lambda t: t.gather(1, _IX)),
    ("take_along_dim", lambda t: torch.take_along_dim(t, _IX, dim=1)),
    ("contiguous", lambda t: t.contiguous()),
    ("clone", lambda t: t.clone()),
    ("to_float", lambda t: t.float()),
    ("to_long", lambda t: t.long()),
    ("to_bool", lambda t: t.bool()),
    ("triu", lambda t: torch.triu(t, diagonal=1)),
    ("tril", lambda t: torch.tril(t, diagonal=-1)),
    ("arange", lambda t: torch.arange(0, 6, 2)),
    ("arange_one_bound", lambda t: t + torch.arange(6, dtype=DT)),
    ("arange_two_bounds", lambda t: t + torch.arange(1, 7, dtype=DT)),
    ("arange_step", lambda t: t + torch.arange(0, 12, 2, dtype=DT)),
    (
        "batch_norm",
        lambda t, w, b, m, v: F.batch_norm(t, w, b, m, v, False),
        (
            torch.ones(6, dtype=DT),
            torch.zeros(6, dtype=DT),
            torch.zeros(6, dtype=DT),
            torch.ones(6, dtype=DT),
        ),
    ),
    (
        "layer_norm",
        lambda t, w, b: F.layer_norm(t, (6,), w, b),
        (torch.randn(6, dtype=DT), torch.randn(6, dtype=DT)),
    ),
    (
        "rms_norm",
        lambda t, w: F.rms_norm(t, (6,), w),
        (torch.randn(6, dtype=DT),),
    ),
    (
        "instance_norm",
        lambda t, w, b: F.instance_norm(t.reshape(1, 1, 4, 6), w, b),
        (torch.ones(1, dtype=DT), torch.zeros(1, dtype=DT)),
    ),
    (
        "expand_as",
        lambda t, o: t.expand_as(o),
        (torch.randn(2, 4, 6, dtype=DT),),
    ),
    ("nonzero", lambda t: torch.nonzero(t > 0)),
]


@pytest.mark.parametrize(
    "case",
    OP_CASES,
    ids=[c[0] for c in OP_CASES],
)
def test_op_matches_torch(case):
    """Every mapped op reproduces torch's numbers through ONNX."""
    name, fn = case[0], case[1]
    consts = case[2] if len(case) > 2 else ()
    _round_trip(name, fn, _X, consts)


def test_embedding_with_a_declared_index_input():
    """An integer *input* is declared through ``input_dtypes``."""
    weight = torch.randn(5, 7, dtype=DT)
    _round_trip(
        "embedding",
        lambda t, w: F.embedding(t, w),
        torch.tensor([[0, 2, 1, 3]] * 4),
        (weight,),
        input_dtypes={"x": "int64"},
    )


def test_one_hot():
    """``one_hot`` produces ATen's int64 encoding."""
    _round_trip(
        "one_hot",
        lambda t: F.one_hot(t, 6),
        torch.tensor([[0, 2, 1, 3]]),
        input_dtypes={"x": "int64"},
    )


# ---------------------------------------------------------------------------
#  dtype policy
# ---------------------------------------------------------------------------


def test_float64_lowering_is_exact_on_the_arithmetic_core():
    """In fp64 the delivered artifact is exact to rounding."""
    x = torch.randn(4, 4, dtype=torch.float64)

    def fn(t):
        return torch.softmax(t @ t.t() + t, dim=-1) * 0.5

    model, executor, _, _ = _lower(
        fn, x, sink=OnnxSink(dtype="float64")
    )
    with torch.no_grad():
        reference = model(x)
    assert _max_abs(executor.forward(x), reference) < 1e-14


def test_float32_default_is_the_interchange_precision():
    """The default working dtype rounds fp64 leaves to fp32."""
    x = torch.randn(4, 4, dtype=torch.float64)

    def fn(t):
        return torch.softmax(t @ t.t() + t, dim=-1) * 0.5

    model, executor, _, _ = _lower(fn, x)
    with torch.no_grad():
        reference = model(x)
    diff = _max_abs(executor.forward(x), reference)
    assert 0.0 < diff < 1e-5


def test_unknown_working_dtype_is_refused():
    """A dtype the sink does not know is a loud error."""
    with pytest.raises(ValueError, match="unknown working dtype"):
        OnnxSink(dtype="float128")


def test_unknown_input_dtype_name_is_refused():
    """An unknown ``input_dtypes`` element type is a loud error."""
    with pytest.raises(ValueError, match="unknown input dtype"):
        OnnxSink(input_dtypes={"x": "complex128"})


# ---------------------------------------------------------------------------
#  refuse, never drop
# ---------------------------------------------------------------------------


def _mint(op: str, *args, var_shape=(2, 2), **attrs) -> IR:
    """Build a one-input IR rooted at ``Op(op, ...)``."""
    var = Var("x", TensorType(tuple(var_shape)))
    return IR(root=Op.make(op, var, *args, validate=False, **attrs))


def test_unsupported_op_is_refused():
    """An unmapped op raises instead of vanishing from the graph."""
    ir = _mint("sdpa")
    with pytest.raises(UnsupportedOnnxOpError) as caught:
        OnnxSink().lower(ir, {})
    assert caught.value.op == "sdpa"
    assert "refused" in str(caught.value)


def test_unsupported_spelling_is_refused():
    """A mapped op in an inexpressible spelling is refused too."""
    ir = _mint("div", Const(2.0), rounding_mode="bogus")
    with pytest.raises(UnsupportedOnnxOpError) as caught:
        OnnxSink().lower(ir, {})
    assert caught.value.op == "div"
    assert "rounding_mode" in caught.value.detail


def test_reshape_with_a_zero_extent_is_refused():
    """A 0 extent means different things to ATen and to ONNX."""
    ir = _mint("reshape", shape=(0, 4))
    with pytest.raises(UnsupportedOnnxOpError, match="0 extent"):
        OnnxSink().lower(ir, {})


def test_rank_dependent_op_without_a_known_rank_is_refused():
    """``transpose`` needs a static rank; an unknown one is refused.

    ``new_zeros`` is one of the mapped ops core's shape inference
    cannot type, so its result has no rank — and the sink declines
    rather than guessing a permutation.
    """
    var = Var("x", TensorType((2, 2)))
    unknown = Op.make("new_zeros", var, size=(2, 2), validate=False)
    ir = IR(root=Op.make("transpose", unknown, dim0=0, dim1=1))
    with pytest.raises(UnsupportedOnnxOpError, match="rank"):
        OnnxSink().lower(ir, {})


def test_missing_parameter_is_refused():
    """An unbound leaf is refused, not silently dropped."""
    param = Param("w", TensorType((2, 2)))
    ir = IR(root=Op.make("matmul", Var("x", TensorType((2, 2))), param))
    with pytest.raises(OnnxLoweringError, match="no value supplied"):
        OnnxSink().lower(ir, {})


def test_getitem_over_a_single_output_op_is_refused():
    """``getitem`` is only the multi-output pick, never identity."""
    ir = _mint("getitem", index=0)
    with pytest.raises(UnsupportedOnnxOpError, match="getitem"):
        OnnxSink().lower(ir, {})


def test_every_declined_op_reports_a_reason():
    """The declines that matter name their reason."""
    sink = OnnxSink()
    for op in (
        "sdpa",
        "max_pool2d",
        "upsample_bilinear2d",
        "randn",
        "cumprod",
        "unbind",
        "sort",
        "topk",
        "scatter",
        "fmax",
    ):
        assert op not in sink.supported_ops


DECLINE_CASES = [
    (
        "split_index_out_of_range",
        "split",
        {"sizes": (2, 2), "index": 9},
        (2, 2),
    ),
    (
        "chunk_index_out_of_range",
        "chunk",
        {"chunks": 2, "index": 9},
        (2, 2),
    ),
    (
        "pad_non_constant",
        "pad",
        {"pad": (1, 1), "mode": "reflect"},
        (2, 2),
    ),
    ("clamp_without_bounds", "clamp", {}, (2, 2)),
    (
        "flatten_not_to_the_end",
        "flatten",
        {"start_dim": 0, "end_dim": 0},
        (2, 2),
    ),
    ("glu_with_an_odd_extent", "glu", {"dim": -1}, (2, 3)),
    ("softplus_with_a_beta", "softplus", {"beta": 2.0}, (2, 2)),
    ("elu_with_a_scale", "elu", {"alpha": 1.0, "scale": 2.0}, (2, 2)),
    ("gelu_approximate", "gelu", {"approximate": "tanh"}, (2, 2)),
    ("dropout_in_training", "dropout", {"train": True}, (2, 2)),
    (
        "batch_norm_in_training",
        "batch_norm",
        {"training": True},
        (2, 2),
    ),
    (
        "instance_norm_running_stats",
        "instance_norm",
        {"use_input_stats": False},
        (2, 2),
    ),
    ("to_an_unknown_dtype", "to", {"dtype": "complex64"}, (2, 2)),
    ("linalg_norm_ord_three", "linalg_vector_norm", {"ord": 3}, (2, 2)),
    ("linalg_norm_ord_zero", "linalg_vector_norm", {"ord": 0}, (2, 2)),
    (
        "transpose_axis_out_of_range",
        "transpose",
        {"dim0": 0, "dim1": 7},
        (2, 2),
    ),
    (
        "pad_more_pairs_than_axes",
        "pad",
        {"pad": (1, 1, 1, 1, 1, 1)},
        (2, 2),
    ),
    ("split_with_a_bad_size_list", "split", {"sizes": (0, 4)}, (2, 2)),
    ("one_hot_without_depth", "one_hot", {"num_classes": -1}, (2, 2)),
]


@pytest.mark.parametrize(
    "name,op,attrs,var_shape",
    DECLINE_CASES,
    ids=[c[0] for c in DECLINE_CASES],
)
def test_a_declined_spelling_is_refused(name, op, attrs, var_shape):
    """A mapped op in an inexpressible spelling is refused loudly."""
    with pytest.raises(UnsupportedOnnxOpError):
        OnnxSink().lower(_mint(op, var_shape=var_shape, **attrs), {})


# ---------------------------------------------------------------------------
#  the equivalence gate
# ---------------------------------------------------------------------------


def test_verify_equiv_matches_the_torch_metric():
    """``max_rel`` is the historical rel-diff, restated."""
    reference = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    report = verify_equiv(reference, reference + 1e-6, rtol=1e-4)
    assert report.passed
    assert report.max_abs == pytest.approx(1e-6, rel=1e-9)
    assert report.max_rel == pytest.approx(1e-6 / 3.0, rel=1e-9)


def test_verify_equiv_honours_atol():
    """An explicit ``atol`` adds a second, absolute gate."""
    reference = np.zeros(4, dtype=np.float64)
    report = verify_equiv(
        reference, reference + 1e-3, rtol=1.0, atol=1e-4
    )
    assert not report.passed


def test_verify_module_runs_both_sides():
    """``Sink.verify`` compares two lowered runnables in ONNX."""
    x = torch.randn(4, 6, dtype=DT)
    sink = OnnxSink()
    _, executor, ir, leaves = _lower(lambda t: t * 2.0, x, sink=sink)
    reference = sink.lower(ir, leaves)
    report = sink.verify(reference, executor, (x,))
    assert report.passed
    assert report.max_abs == 0.0
    # The report is a ``VerifyResult`` without core naming its type.
    assert isinstance(report, VerifyResult)


def test_verify_module_accepts_a_bare_input():
    """A single input (not a tuple) is wrapped, as the pipeline does."""
    x = torch.randn(4, 6, dtype=DT)
    sink = OnnxSink()
    _, executor, ir, leaves = _lower(lambda t: t + 1.0, x, sink=sink)
    assert sink.verify(sink.lower(ir, leaves), executor, x).passed


def test_verify_module_rejects_a_mismatched_output_arity():
    """A leaf-count mismatch is an error, not a silent pass."""
    with pytest.raises(ValueError, match="arity mismatch"):
        verify_equiv(np.zeros((4, 6)), (np.zeros((4, 6)), np.zeros(1)))


# ---------------------------------------------------------------------------
#  the experiment: the pipeline drives the ONNX sink unchanged
# ---------------------------------------------------------------------------


class _Stack(nn.Module):
    """A stack of ``nn.Linear`` with no nonlinearity between them."""

    def __init__(self, width=8, depth=3):
        """Build ``depth`` square linear layers."""
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(width, width) for _ in range(depth)
        )

    def forward(self, x):
        """Apply every layer in order."""
        for layer in self.layers:
            x = layer(x)
        return x


class _Head(nn.Module):
    """Linear → ReLU → linear → softmax."""

    def __init__(self, width=8):
        """Build the head's two layers."""
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, 4)

    def forward(self, x):
        """Run the head."""
        return torch.softmax(
            self.second(torch.relu(self.first(x))), dim=-1
        )


def test_optimizer_runs_the_onnx_sink_end_to_end():
    """``Optimizer`` + ``OnnxSink``: search, lower, verify — all ONNX.

    The claim ADR 0006 tests: the orchestrator needs no change to
    drive a second backend.  The search runs with the ONNX sink as
    its ``Capabilities``, so extraction prices only ONNX-expressible
    forms; the delivery is verified *in onnxruntime*.
    """
    model = _Head().eval()
    x = torch.randn(3, 8)
    optimizer = Optimizer(source=TorchSource(), sink=OnnxSink())
    result = optimizer.optimize(model, x, max_iterations=4)
    assert result.verified is not None
    assert result.verified.passed
    assert result.stats["lowering"] == "generic"
    assert result.stats["runner"] == "identity"


def test_the_search_only_selects_ops_the_sink_can_lower():
    """``backend_cost`` bounds extraction by ``supported_ops``."""
    model = _Stack().eval()
    x = torch.randn(3, 8)
    sink = OnnxSink()
    optimizer = Optimizer(source=TorchSource(), sink=sink)
    result = optimizer.search(model, x, max_iterations=4)
    assert _ops_of(result.term) <= sink.supported_ops
    # The saturation really fired: the delivered term is not the
    # exported one.
    assert _ops_of(result.term) != _ops_of(result.ir.root)


def test_delivered_artifact_matches_the_source_module():
    """The optimized ONNX program equals the *original* torch model.

    The end-to-end statement the ADR asks for: an optimized program
    delivered through ONNX reproduces the source module's numbers,
    measured through onnxruntime.
    """
    torch.manual_seed(0)
    model = _Head().eval()
    x = torch.randn(3, 8)
    optimizer = Optimizer(source=TorchSource(), sink=OnnxSink())
    result = optimizer.optimize(model, x, max_iterations=4)
    with torch.no_grad():
        reference = model(x)
    delivered = np.asarray(result.module.forward(x))
    assert delivered.shape == tuple(reference.shape)
    diff = float(np.max(np.abs(delivered - reference.numpy())))
    assert diff < 1e-5, f"max_abs={diff:.3e}"


def test_the_delivered_artifact_can_be_saved_and_rerun(tmp_path):
    """An optimized program leaves as a standalone ``.onnx`` file."""
    model = _Stack().eval()
    x = torch.randn(3, 8)
    optimizer = Optimizer(source=TorchSource(), sink=OnnxSink())
    result = optimizer.optimize(model, x, max_iterations=4)
    path = result.module.save(tmp_path / "optimized.onnx")
    reopened = OnnxExecutor(onnx.load(str(path)))
    assert np.allclose(
        np.asarray(reopened.forward(x)),
        np.asarray(result.module.forward(x)),
    )
