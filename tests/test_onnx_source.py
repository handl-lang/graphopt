"""The ONNX source — ADR 0006 stage 2, verified through onnxruntime.

Two things are under test here, and the second is the experiment:

1. that :class:`mlopt_onnx.OnnxSource` lifts an ONNX graph to mlopt
   IR whose re-lowered numbers match the original model, and
2. that it does so as a *port implementation* — the pipeline drives
   it through ``mlopt_core.ports.Source`` exactly as it drives the
   torch source, so an ONNX model that was never torch gets searched,
   lowered and verified end to end.

The round-trip tests run the original model in onnxruntime, lift it
to IR, lower it back through :class:`mlopt_onnx.OnnxSink`, and
compare — a refusal anywhere in that chain is a loud failure, which
is the ADR's rule.  Ops and spellings outside the lifting table are
exercised by the refusal tests instead.
"""

from __future__ import annotations

import numpy as np
import pytest
pytestmark = pytest.mark.e2e  # boundary-purpose file: crosses an external engine boundary
import torch
import torch.nn as nn

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx
from mlopt_core.ir import Op
from mlopt_core.ports import Source
from mlopt_onnx import (
    ONNX_LIFTS,
    OnnxExecutor,
    OnnxImportError,
    OnnxSink,
    OnnxSource,
    UnsupportedOnnxOpError,
)
from mlopt_orchestrator.optimize import Optimizer
from mlopt_torch.torch_bridge import export_to_ir
from onnx import TensorProto, helper, numpy_helper

DT = np.float32


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _model(nodes, inputs, outputs, initializer=(), *, opset=18):
    """Build a checked ``ModelProto`` at ``opset``."""
    graph = helper.make_graph(
        nodes, "g", inputs, outputs, initializer=list(initializer)
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    return model


def _vi(name, elem, shape):
    """A typed tensor ``ValueInfo`` (``None`` dims stay dynamic)."""
    return helper.make_tensor_value_info(name, elem, shape)


def _f32(name, shape):
    """A float32 tensor ``ValueInfo``."""
    return _vi(name, TensorProto.FLOAT, shape)


def _init(name, array):
    """An initializer ``TensorProto``."""
    return numpy_helper.from_array(np.asarray(array), name)


def _node(op, ins, outs, **attrs):
    """One ONNX node."""
    return helper.make_node(op, ins, outs, **attrs)


def _lift(model):
    """Lift ``model``; return ``(source, ir, leaves)``."""
    source = OnnxSource()
    ir, leaves = source.to_ir(model)
    return source, ir, leaves


def _round_trip(model, xs, *, tol=1e-6, name=""):
    """Run ``model`` and its lift->lower round trip; compare outputs.

    ``xs`` are positional inputs for both executors.  Returns the
    largest absolute difference over every output leaf.
    """
    source, ir, leaves = _lift(model)
    sink = OnnxSink(input_dtypes=source.input_dtypes)
    executor = sink.lower(ir, leaves)
    reference = OnnxExecutor(model).forward(*xs)
    got = executor.forward(*xs)
    refs = reference if isinstance(reference, tuple) else (reference,)
    outs = got if isinstance(got, tuple) else (got,)
    assert len(refs) == len(outs), f"{name}: output arity differs"
    worst = 0.0
    for expected, leaf in zip(refs, outs, strict=True):
        expected, leaf = np.asarray(expected), np.asarray(leaf)
        assert expected.shape == leaf.shape, (
            f"{name}: shape {leaf.shape} != {expected.shape}"
        )
        if leaf.size:
            worst = max(worst, float(np.max(np.abs(expected - leaf))))
    return worst


def _ops_of(term) -> set:
    """Return every op name in a term."""
    found: set = set()
    stack = [term]
    while stack:
        node = stack.pop()
        if isinstance(node, Op):
            found.add(node.op)
            stack.extend(node.args)
    return found


_X = np.random.default_rng(0).standard_normal((4, 6)).astype(DT)


# ---------------------------------------------------------------------------
#  the port surface
# ---------------------------------------------------------------------------


def test_source_conforms_to_the_source_port():
    """A runtime check sees a real ``Source``."""
    assert isinstance(OnnxSource(), Source)


def test_the_lift_table_is_a_coherent_subset():
    """The imported op count is explicit, not discovered."""
    assert len(ONNX_LIFTS) >= 80
    assert "MatMul" in ONNX_LIFTS
    assert "If" not in ONNX_LIFTS
    assert "SequenceConstruct" not in ONNX_LIFTS


def test_provenance_keeps_graph_value_names():
    """``Var``/``Param`` names are the graph's own value names."""
    model = _model(
        [_node("MatMul", ["x", "weight"], ["y"])],
        [_f32("x", (2, 3))],
        [_f32("y", (2, 4))],
        [_init("weight", np.ones((3, 4), dtype=DT))],
    )
    _, ir, leaves = _lift(model)
    assert [v.name for v in ir.inputs] == ["x"]
    assert list(ir.params) == ["weight"]
    assert list(leaves) == ["weight"]


def test_input_dtypes_follow_the_graph():
    """``input_dtypes`` records declared element types, per input."""
    model = _model(
        [_node("Gather", ["w", "idx"], ["y"], axis=0)],
        [
            _f32("w", (5, 4)),
            _vi("idx", TensorProto.INT64, (3,)),
        ],
        [_f32("y", (3, 4))],
    )
    source, _, _ = _lift(model)
    assert source.input_dtypes == {"w": "float32", "idx": "int64"}


def test_lift_result_carries_the_same_side_channel():
    """``source.lift`` exposes ``input_dtypes`` on the record itself."""
    model = _model(
        [_node("Relu", ["x"], ["y"])],
        [_f32("x", (2, 2))],
        [_f32("y", (2, 2))],
    )
    result = OnnxSource().lift(model)
    assert result.input_dtypes == {"x": "float32"}
    assert result.leaves == {}


def test_example_inputs_is_ignored():
    """The port's second argument has no role for ONNX."""
    model = _model(
        [_node("Relu", ["x"], ["y"])],
        [_f32("x", (2, 2))],
        [_f32("y", (2, 2))],
    )
    source = OnnxSource()
    assert (
        source.to_ir(model, object())[0].root
        == source.to_ir(model)[0].root
    )


def test_a_path_input_loads_the_model(tmp_path):
    """``to_ir`` accepts a ``.onnx`` path as well as a ModelProto."""
    model = _model(
        [_node("Relu", ["x"], ["y"])],
        [_f32("x", (2, 2))],
        [_f32("y", (2, 2))],
    )
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    ir, _ = OnnxSource().to_ir(str(path))
    assert ir.root.op == "relu"


# ---------------------------------------------------------------------------
#  round trips — ONNX -> IR -> ONNX, verified in onnxruntime
# ---------------------------------------------------------------------------


def test_an_elementwise_chain_round_trips():
    """Arithmetic and activations lift and re-lower exactly."""
    model = _model(
        [
            _node("Add", ["x", "x"], ["a"]),
            _node("Mul", ["a", "x"], ["b"]),
            _node("Sigmoid", ["b"], ["c"]),
            _node("Sub", ["c", "x"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    assert _round_trip(model, [_X], name="elementwise") == 0.0


def test_a_matmul_graph_round_trips():
    """``MatMul`` + ``Add`` over an initializer — the linear idiom."""
    w = np.random.default_rng(1).standard_normal((6, 3)).astype(DT)
    b = np.random.default_rng(2).standard_normal((3,)).astype(DT)
    model = _model(
        [
            _node("MatMul", ["x", "w"], ["mm"]),
            _node("Add", ["mm", "b"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 3))],
        [_init("w", w), _init("b", b)],
    )
    assert _round_trip(model, [_X], name="matmul") == 0.0


def test_a_multi_output_graph_round_trips():
    """Two outputs become a ``pack`` root and both survive."""
    model = _model(
        [
            _node("Relu", ["x"], ["a"]),
            _node("Sigmoid", ["x"], ["b"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("a", (4, 6)), _f32("b", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "pack"
    assert len(ir.roots) == 2
    assert _round_trip(model, [_X], name="multi") == 0.0


def test_constants_and_attributes_round_trip():
    """Initializers, ``Constant`` nodes and node attrs all carry."""
    w = np.random.default_rng(3).standard_normal((6, 4)).astype(DT)
    model = _model(
        [
            _node("Constant", [], ["two"], value=_init("two_v", 2.0)),
            _node("Pow", ["x", "two"], ["p"]),
            _node("Clip", ["p", "lo", "hi"], ["c"]),
            _node("LeakyRelu", ["c"], ["l"], alpha=0.3),
            _node("Transpose", ["l"], ["t"], perm=[1, 0]),
            _node("Add", ["t", "w"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (6, 4))],
        [
            _init("w", w),
            _init("lo", np.asarray([-0.5], dtype=DT)),
            _init("hi", np.asarray([0.5], dtype=DT)),
        ],
    )
    _, ir, _ = _lift(model)
    assert "pow" in _ops_of(ir.root)
    assert "clamp" in _ops_of(ir.root)
    assert _round_trip(model, [_X], name="consts+attrs") < 1e-6


def test_reductions_and_layout_round_trip():
    """Axes-as-input folds, reshapes, squeezes and concats."""
    shape = np.asarray([2, 12], dtype=np.int64)
    axes = np.asarray([1], dtype=np.int64)
    model = _model(
        [
            _node("Reshape", ["x", "sh"], ["r"]),
            _node("Flatten", ["r"], ["f"], axis=1),
            _node("ReduceSum", ["f", "ax"], ["s"], keepdims=1),
            _node("Unsqueeze", ["s", "ax"], ["u"]),
            _node("Concat", ["u", "u"], ["c"], axis=1),
            _node("Squeeze", ["c"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (2, 2))],
        [_init("sh", shape), _init("ax", axes)],
    )
    diff = _round_trip(model, [_X], name="reduce+layout")
    assert diff < 1e-6


def test_a_split_graph_round_trips():
    """``Split`` mints a per-output ``split`` projection."""
    model = _model(
        [
            _node("Split", ["x", "sz"], ["o1", "o2", "o3"], axis=1),
            _node("Add", ["o1", "o3"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 2))],
        [_init("sz", np.asarray([2, 2, 2], dtype=np.int64))],
    )
    _, ir, _ = _lift(model)
    assert "split" in _ops_of(ir.root)
    assert _round_trip(model, [_X], name="split") == 0.0


def test_hardsigmoid_with_the_onnx_default_round_trips():
    """The ADR's alpha=0.2 trap — composed, not silently re-defaulted.

    ONNX ``HardSigmoid``'s default alpha is NOT ATen's 1/6; the
    lifter must mint the exact ``clamp(alpha*x + beta, 0, 1)``
    rather than pretending it saw ATen's op.
    """
    model = _model(
        [_node("HardSigmoid", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert "hardsigmoid" not in _ops_of(ir.root)
    assert "clamp" in _ops_of(ir.root)
    assert _round_trip(model, [_X], name="hardsigmoid") < 1e-6


def test_gemm_at_the_linear_spelling_round_trips():
    """``Gemm`` with transB=1 and unit scales mints ``linear``."""
    w = np.random.default_rng(4).standard_normal((3, 6)).astype(DT)
    b = np.random.default_rng(5).standard_normal((3,)).astype(DT)
    model = _model(
        [_node("Gemm", ["x", "w", "b"], ["y"], transB=1)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 3))],
        [_init("w", w), _init("b", b)],
    )
    _, ir, _ = _lift(model)
    assert "linear" in _ops_of(ir.root)
    assert _round_trip(model, [_X], name="gemm") == 0.0


def test_slice_and_pad_attr_folds_round_trip():
    """``Slice`` bounds and ``Pad`` widths fold from initializers."""
    model = _model(
        [
            _node(
                "Slice",
                ["x", "st", "en", "ax", "sp"],
                ["s"],
            ),
            _node("Pad", ["s", "pw"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 4))],
        [
            _init("st", np.asarray([1], dtype=np.int64)),
            _init("en", np.asarray([5], dtype=np.int64)),
            _init("ax", np.asarray([1], dtype=np.int64)),
            _init("sp", np.asarray([2], dtype=np.int64)),
            _init("pw", np.asarray([0, 0, 0, 2], dtype=np.int64)),
        ],
    )
    _, ir, _ = _lift(model)
    assert "slice" in _ops_of(ir.root)
    assert "pad" in _ops_of(ir.root)
    assert _round_trip(model, [_X], name="slice+pad") < 1e-6


def test_an_identity_program_round_trips():
    """``Identity`` mints ``clone`` and re-lowers to ``Identity``."""
    model = _model(
        [_node("Identity", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "clone"
    assert _round_trip(model, [_X], name="identity") == 0.0


def test_an_int64_input_round_trips_via_input_dtypes():
    """An index input survives — through the side channel, not the IR.

    This is the port gap ADR 0006 measured: ``Var.typ`` carries no
    element type, so the source's ``input_dtypes`` is what keeps
    ``Gather``'s index an index.
    """
    w = np.random.default_rng(6).standard_normal((5, 4)).astype(DT)
    model = _model(
        [_node("Gather", ["w", "idx"], ["y"], axis=0)],
        [_vi("idx", TensorProto.INT64, (3,))],
        [_f32("y", (3, 4))],
        [_init("w", w)],
    )
    idx = np.asarray([0, 2, 1], dtype=np.int64)
    assert _round_trip(model, [idx], name="gather") == 0.0


def test_dynamic_dims_lift_as_unknown_extents():
    """``dim_param`` dims are representable — they lift as ``None``."""
    model = _model(
        [_node("Relu", ["x"], ["y"])],
        [_vi("x", TensorProto.FLOAT, ("batch", 6))],
        [_vi("y", TensorProto.FLOAT, ("batch", 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.inputs[0].typ.shape == (None, 6)
    assert _round_trip(model, [_X], name="dyn-dim") == 0.0


# ---------------------------------------------------------------------------
#  the pipeline — the optimizer drives the ONNX source unchanged
# ---------------------------------------------------------------------------


class _Head(nn.Module):
    """Linear → ReLU → linear → softmax, the sink test's program."""

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


def _head_model():
    """Export the torch head to IR, then to an ONNX model."""
    torch.manual_seed(0)
    model = _Head().eval()
    x = torch.randn(3, 8)
    ir, leaves = export_to_ir(model, (x,))
    return OnnxSink().build_model(ir, leaves), model, x.numpy()


def test_a_torch_exported_model_round_trips():
    """torch -> IR -> ONNX -> IR -> ONNX, verified against torch."""
    proto, model, x = _head_model()
    source, ir, leaves = _lift(proto)
    executor = OnnxSink(input_dtypes=source.input_dtypes).lower(
        ir, leaves
    )
    with torch.no_grad():
        reference = model(torch.from_numpy(x))
    got = np.asarray(executor.forward(x))
    assert got.shape == tuple(reference.shape)
    assert float(np.max(np.abs(got - reference.numpy()))) < 1e-5


def test_the_optimizer_runs_source_and_sink_end_to_end():
    """``Optimizer(source=OnnxSource())`` — no torch in the loop.

    The direction-2 claim: an ONNX model enters the pipeline through
    the same ``Source`` port, saturates, extracts and verifies in
    onnxruntime — the source needs no special-casing anywhere.
    """
    proto, model, x = _head_model()
    source = OnnxSource()
    optimizer = Optimizer(
        source=source,
        sink=OnnxSink(input_dtypes={"x": "float32"}),
    )
    result = optimizer.optimize(proto, x, max_iterations=4)
    assert result.verified is not None
    assert result.verified.passed
    with torch.no_grad():
        reference = model(torch.from_numpy(x))
    got = np.asarray(result.module.forward(x))
    assert float(np.max(np.abs(got - reference.numpy()))) < 1e-5


def test_a_shipped_law_fires_on_the_lifted_ir():
    """``Mul(x, Sigmoid(x))`` — ``silu_fold`` rewrites it to ``silu``.

    The optimizer-path requirement: the lifted term is real IR — the
    shipped law library reads it, rewrites it, and extraction picks
    the fused form.
    """
    model = _model(
        [
            _node("Sigmoid", ["x"], ["s"]),
            _node("Mul", ["x", "s"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    source = OnnxSource()
    optimizer = Optimizer(source=source, sink=OnnxSink())
    result = optimizer.search(model, _X, max_iterations=4)
    assert result.stats["rule_fires"].get("silu_fold", 0) > 0
    assert result.term == Op.make(
        "silu", source.to_ir(model)[0].inputs[0]
    )


def test_softmax_spelled_manually_still_folds():
    """``Div(Exp(x), ReduceSum(Exp(x)))`` — ``softmax_fold`` sees it."""
    axes = np.asarray([-1], dtype=np.int64)
    model = _model(
        [
            _node("Exp", ["x"], ["e"]),
            _node("ReduceSum", ["e", "ax"], ["s"], keepdims=1),
            _node("Div", ["e", "s"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [_init("ax", axes)],
    )
    source = OnnxSource()
    optimizer = Optimizer(source=source, sink=OnnxSink())
    result = optimizer.search(model, _X, max_iterations=4)
    fires = {k: v for k, v in result.stats["rule_fires"].items() if v}
    assert fires, "no rules fired on the manual softmax"
    assert "softmax" in _ops_of(result.term)


# ---------------------------------------------------------------------------
#  refuse, never drop
# ---------------------------------------------------------------------------


def test_an_unsupported_op_is_refused():
    """An unmapped ``op_type`` raises instead of vanishing."""
    model = _model(
        [_node("ConvTranspose", ["x", "w"], ["y"])],
        [_f32("x", (1, 2, 4, 4))],
        [_f32("y", (1, 2, 6, 6))],
        [_init("w", np.ones((2, 2, 3, 3), dtype=DT))],
    )
    with pytest.raises(UnsupportedOnnxOpError) as caught:
        _lift(model)
    assert caught.value.op == "ConvTranspose"


def test_a_computed_attr_fold_input_is_refused():
    """``ReduceSum`` axes from a *computed* tensor — no IR spelling.

    ONNX carries the axes as a tensor input; the IR spells them as
    an attr.  A non-constant tensor is a refusal, not a silently
    wrong program — the ADR's ambiguous-attribute case.
    """
    model = _model(
        [
            _node("ReduceSum", ["x", "ax"], ["y"], keepdims=1),
        ],
        [_f32("x", (4, 6)), _vi("ax", TensorProto.INT64, (1,))],
        [_f32("y", (4, 1))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="not a constant"):
        _lift(model)


def test_a_computed_reshape_shape_is_refused():
    """``Reshape``'s shape input must be constant."""
    model = _model(
        [
            _node("Shape", ["x"], ["sh"]),
            _node("Reshape", ["x", "sh"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError):
        _lift(model)


def test_an_input_with_no_tensor_type_is_refused():
    """A sequence-typed input cannot mint a ``Var`` — refuse."""
    seq = helper.make_tensor_sequence_value_info(
        "xs", TensorProto.FLOAT, None
    )
    model = helper.make_model(
        helper.make_graph([], "g", [seq], [_f32("y", (1,))]),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    with pytest.raises(OnnxImportError, match="no tensor type"):
        _lift(model)


def test_a_non_tensor_output_is_refused():
    """A sequence-typed graph output is refused, not dropped."""
    seq = helper.make_tensor_sequence_value_info(
        "ys", TensorProto.FLOAT, None
    )
    model = helper.make_model(
        helper.make_graph(
            [_node("Relu", ["x"], ["y"])],
            "g",
            [_f32("x", (2, 2))],
            [seq],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    with pytest.raises(OnnxImportError, match="not a tensor"):
        _lift(model)


def test_an_opset_outside_the_window_is_refused():
    """Opset drift is a loud refusal, not a guess."""
    model = _model(
        [_node("Relu", ["x"], ["y"])],
        [_f32("x", (2, 2))],
        [_f32("y", (2, 2))],
        opset=7,
    )
    with pytest.raises(OnnxImportError, match="opset"):
        _lift(model)


def test_a_graph_attribute_is_refused_when_read():
    """``If`` carries a graph attribute — unmappable by construction."""
    body = helper.make_graph(
        [_node("Relu", ["c"], ["d"])],
        "body",
        [_f32("c", (2, 2))],
        [_f32("d", (2, 2))],
    )
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, ())
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["y"],
                    then_branch=body,
                    else_branch=body,
                )
            ],
            "g",
            [cond, _f32("x", (2, 2))],
            [_f32("y", (2, 2))],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    with pytest.raises(UnsupportedOnnxOpError):
        _lift(model)


def test_a_training_mode_batch_norm_is_refused():
    """Training-mode stats updates are a state change, not a function."""
    c = np.ones(6, dtype=DT)
    z = np.zeros(6, dtype=DT)
    model = _model(
        [
            _node(
                "BatchNormalization",
                ["x", "s", "b", "m", "v"],
                ["y"],
                training_mode=1,
            )
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [
            _init("s", c),
            _init("b", z),
            _init("m", z),
            _init("v", c),
        ],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="training"):
        _lift(model)


def test_every_declined_family_reports_a_reason():
    """The declines that matter name their reason."""
    cases = {
        "asymmetric conv pads": _model(
            [
                _node(
                    "Conv",
                    ["x", "w"],
                    ["y"],
                    pads=[1, 1, 0, 2],
                )
            ],
            [_f32("x", (1, 2, 5, 5))],
            [_f32("y", (1, 3, 6, 5))],
            [_init("w", np.ones((3, 2, 3, 3), dtype=DT))],
        ),
        "computed clip bound": _model(
            [
                _node("Neg", ["lo"], ["nlo"]),
                _node("Clip", ["x", "nlo", "hi"], ["y"]),
            ],
            [_f32("x", (4, 6)), _f32("lo", (1,))],
            [_f32("y", (4, 6))],
            [_init("hi", np.asarray([0.5], dtype=DT))],
        ),
        "select_last_index": _model(
            [
                _node(
                    "ArgMax",
                    ["x"],
                    ["y"],
                    axis=1,
                    select_last_index=1,
                )
            ],
            [_f32("x", (4, 6))],
            [_vi("y", TensorProto.INT64, (4, 1))],
        ),
    }
    for name, model in cases.items():
        with pytest.raises(UnsupportedOnnxOpError):
            _lift(model)
