"""The ONNX -> IR op table — per-op lifting and its refusals.

``test_onnx_source.py`` exercises the port end to end; this file
exercises :mod:`catopt_onnx.liftmap` itself: every mapped op's minted
spelling, the attr folds (constant inputs becoming IR attrs), the
composed lifts (``Gemm``'s non-``linear`` spellings, the flag-driven
activations) and each refusal the table declares.  Where the minted
IR is one the sink lowers, the model is round-tripped through
onnxruntime; where the lift is deliberately wider than the lowering
table (``topk``, ``group_norm``, ``isposinf`` …), the test asserts
the minted term and the sink's own refusal.

``GraphLifter`` is driven directly where the seam is a *partial*
shape table — the state ``_collect_value_info`` itself produces when
inference leaves an intermediate unshaped (a documented input of the
class, not a fake).
"""

from __future__ import annotations

import numpy as np
import pytest
pytestmark = pytest.mark.e2e  # boundary-purpose file: crosses an external engine boundary

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from catopt_core.ir import Op
from catopt_onnx import (
    GraphLifter,
    OnnxImportError,
    OnnxSink,
    OnnxSource,
    UnsupportedOnnxOpError,
)
from onnx import TensorProto, helper, numpy_helper

from tests.test_onnx_source import (
    _X,
    _f32,
    _init,
    _lift,
    _model,
    _node,
    _ops_of,
    _round_trip,
    _vi,
)

DT = np.float32

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _raw_model(nodes, inputs, outputs, initializer=(), *, opset=18):
    """A ``ModelProto`` the checker is not asked to bless.

    ONNX's checker enforces arity/shape rules the *lifter* does not
    need — the refusal tests here build deliberately ill-formed
    graphs (an extra output, an empty output name) that a real
    exporter could still emit and the lifter must still answer.
    """
    graph = helper.make_graph(
        nodes, "g", inputs, outputs, initializer=list(initializer)
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )


def _lift_raw(nodes, inputs, outputs, initializer=(), *, opset=18):
    """Build an unchecked model and lift it."""
    return _lift(
        _raw_model(nodes, inputs, outputs, initializer, opset=opset)
    )


def _lifter(nodes, inputs, outputs, initializer=(), value_info=None):
    """Drive ``GraphLifter`` directly with an explicit shape table.

    ``value_info=None`` gives every shape/dtype query a miss except
    the initializers — the state the lifter sees when ONNX's own
    inference could not type an intermediate (the ``value_info``
    parameter exists exactly so a caller may pass a partial table).
    """
    graph = helper.make_graph(
        nodes, "g", inputs, outputs, initializer=list(initializer)
    )
    initializers = {
        init.name: numpy_helper.to_array(init) for init in initializer
    }
    return GraphLifter(
        graph, initializers=initializers, value_info=value_info or {}
    )


def _attrs_of(term, op):
    """Return the attr dict of the first ``op`` node in ``term``."""
    stack = [term]
    while stack:
        node = stack.pop()
        if isinstance(node, Op):
            if node.op == op:
                return node.attrs
            stack.extend(node.args)
    raise AssertionError(f"no {op!r} in term")


def _i64(name, shape):
    """An int64 tensor ``ValueInfo``."""
    return _vi(name, TensorProto.INT64, shape)


def _bool(name, shape):
    """A bool tensor ``ValueInfo``."""
    return _vi(name, TensorProto.BOOL, shape)


# ---------------------------------------------------------------------------
#  elementwise / scalar-shaped
# ---------------------------------------------------------------------------


def test_mod_lifts_fmod_and_remainder():
    """``Mod``'s ``fmod`` flag selects the two remainder ops."""
    f_model = _model(
        [_node("Mod", ["x", "d"], ["y"], fmod=1)],
        [_f32("x", (4, 6)), _f32("d", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(f_model)
    assert ir.root.op == "fmod"
    xf = np.abs(_X) + 0.5
    df = np.full_like(_X, 2.0)
    assert _round_trip(f_model, [xf, df], name="mod-fmod") < 1e-6

    i_model = _model(
        [_node("Mod", ["x", "d"], ["y"], fmod=0)],
        [_i64("x", (4, 6)), _i64("d", (4, 6))],
        [_i64("y", (4, 6))],
    )
    _, ir, _ = _lift(i_model)
    assert ir.root.op == "remainder"
    x = np.asarray([[7, -7, 5, 4, 9, 2]] * 4, dtype=np.int64)
    d = np.asarray([[3, 4, -3, 5, 2, 6]] * 4, dtype=np.int64)
    assert _round_trip(i_model, [x, d], name="mod-int") == 0.0


def test_mean_folds_variadic_adds_over_the_count():
    """``Mean(a, b, c)`` mints ``(a + b + c) / 3``."""
    model = _model(
        [_node("Mean", ["x", "x", "x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "div"
    assert _round_trip(model, [_X], name="mean") < 1e-6


def test_variadic_ops_fold_left():
    """``Sum``/``Max``/``Min`` fold to binary ``add``/``maximum``/``minimum``."""
    for op, ir_op in (
        ("Sum", "add"),
        ("Max", "maximum"),
        ("Min", "minimum"),
    ):
        model = _model(
            [_node(op, ["x", "x", "x"], ["y"])],
            [_f32("x", (4, 6))],
            [_f32("y", (4, 6))],
        )
        _, ir, _ = _lift(model)
        assert ir.root.op == ir_op
        assert _round_trip(model, [_X], name=op) < 1e-6


def test_a_single_operand_fold_is_refused():
    """``Sum``/`Mean` with one operand cannot fold."""
    model = _model(
        [_node("Sum", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match=">= 2"):
        _lift(model)

    model = _model(
        [_node("Mean", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match=">= 2"):
        _lift(model)


def test_clip_partial_bounds_pick_the_clamp_variant():
    """lo-only -> ``clamp_min``, hi-only -> ``clamp_max``, none -> ``clone``."""
    lo = _init("lo", np.asarray([-0.5], dtype=DT))
    hi = _init("hi", np.asarray([0.5], dtype=DT))
    cases = [
        (["x", "lo"], "clamp_min", [lo]),
        (["x", "", "hi"], "clamp_max", [hi]),
        (["x"], "clone", []),
    ]
    for ins, expected, inits in cases:
        model = _model(
            [_node("Clip", ins, ["y"])],
            [_f32("x", (4, 6))],
            [_f32("y", (4, 6))],
            inits,
        )
        _, ir, _ = _lift(model)
        assert ir.root.op == expected
        assert _round_trip(model, [_X], name=expected) < 1e-6


def test_thresholded_relu_mints_the_where_composition():
    """``ThresholdedRelu`` is ``where(x > alpha, x, 0)`` spelled out."""
    model = _model(
        [_node("ThresholdedRelu", ["x"], ["y"], alpha=0.5)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "where"
    assert "gt" in _ops_of(ir.root)
    assert _round_trip(model, [_X], name="thresholdedrelu") < 1e-6


def test_elu_carries_its_alpha():
    """``Elu``'s alpha attr rides the minted ``elu``."""
    model = _model(
        [_node("Elu", ["x"], ["y"], alpha=0.7)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "elu"
    assert _attrs_of(ir.root, "elu")["alpha"] == pytest.approx(0.7)
    assert _round_trip(model, [_X], name="elu") < 1e-6


def test_celu_lifts_past_the_sink():
    """``Celu`` mints ``celu`` — a lift the lowering table declines."""
    model = _model(
        [_node("Celu", ["x"], ["y"], alpha=2.0)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, leaves = _lift(model)
    assert ir.root.op == "celu"
    with pytest.raises(UnsupportedOnnxOpError):
        OnnxSink().lower(ir, leaves)


def test_selu_at_atens_constants_mints_selu():
    """Default-parameter ``Selu`` is exactly ATen's ``selu``."""
    model = _model(
        [_node("Selu", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "selu"
    assert _round_trip(model, [_X], name="selu") < 1e-6


def test_selu_off_atens_constants_composes():
    """A parameterised ``Selu`` mints ``gamma * elu(x, alpha)``."""
    model = _model(
        [_node("Selu", ["x"], ["y"], alpha=1.5, gamma=2.0)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "mul"
    assert "elu" in _ops_of(ir.root)
    assert "selu" not in _ops_of(ir.root)
    assert _round_trip(model, [_X], name="selu-composed") < 1e-5


def test_hardsigmoid_at_atens_spelling_mints_the_fused_op():
    """``alpha=1/6, beta=0.5`` is ATen's hardsigmoid exactly."""
    model = _model(
        [
            _node(
                "HardSigmoid",
                ["x"],
                ["y"],
                alpha=1.0 / 6.0,
                beta=0.5,
            )
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "hardsigmoid"
    assert _round_trip(model, [_X], name="hardsigmoid") < 1e-6


def test_gelu_carries_the_approximate_spelling():
    """``Gelu`` (opset 20) round-trips both of its spellings."""
    model = _model(
        [_node("Gelu", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        opset=20,
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "gelu"
    assert _round_trip(model, [_X], name="gelu") < 1e-6


def test_gelu_tanh_lifts_but_the_sink_declines():
    """``approximate="tanh"`` carries through; the sink refuses it."""
    model = _model(
        [_node("Gelu", ["x"], ["y"], approximate="tanh")],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        opset=20,
    )
    _, ir, leaves = _lift(model)
    assert ir.root.op == "gelu"
    with pytest.raises(UnsupportedOnnxOpError):
        OnnxSink().lower(ir, leaves)


def test_isinf_detect_flags_select_the_predicate():
    """The flags pick ``isinf`` / ``isposinf`` / ``isneginf``."""
    cases = [
        ({}, "isinf"),
        ({"detect_negative": 0}, "isposinf"),
        ({"detect_positive": 0}, "isneginf"),
    ]
    for attrs, expected in cases:
        model = _model(
            [_node("IsInf", ["x"], ["y"], **attrs)],
            [_f32("x", (4, 6))],
            [_bool("y", (4, 6))],
            opset=20,
        )
        _, ir, _ = _lift(model)
        assert ir.root.op == expected


def test_isinf_detecting_nothing_is_refused():
    """``detect_positive=0, detect_negative=0`` is degenerate."""
    model = _model(
        [
            _node(
                "IsInf",
                ["x"],
                ["y"],
                detect_positive=0,
                detect_negative=0,
            )
        ],
        [_f32("x", (4, 6))],
        [_bool("y", (4, 6))],
        opset=20,
    )
    with pytest.raises(UnsupportedOnnxOpError, match="constant-false"):
        _lift(model)


def test_cast_records_the_target_dtype():
    """``Cast to=X`` mints ``to(dtype=...)`` with the type's name."""
    model = _model(
        [_node("Cast", ["x"], ["y"], to=TensorProto.INT64)],
        [_f32("x", (4, 6))],
        [_i64("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "to"
    assert _attrs_of(ir.root, "to")["dtype"] == "int64"
    assert _round_trip(model, [_X], name="cast") == 0.0


def test_cast_to_an_unnamed_type_is_refused():
    """A ``Cast`` target with no IR dtype name is refused."""
    model = _model(
        [_node("Cast", ["x"], ["y"], to=TensorProto.STRING)],
        [_f32("x", (4, 6))],
        [_vi("y", TensorProto.STRING, (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="no dtype name"):
        _lift(model)


def test_castlike_reads_the_exemplars_dtype():
    """``CastLike`` folds the second input's declared dtype."""
    model = _model(
        [_node("CastLike", ["x", "like"], ["y"])],
        [_f32("x", (4, 6)), _i64("like", (1,))],
        [_i64("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "to"
    assert _attrs_of(ir.root, "to")["dtype"] == "int64"


def test_castlike_with_an_unknown_exemplar_is_refused():
    """An untyped exemplar cannot name the cast's target."""
    lifter = _lifter(
        [_node("CastLike", ["x", "like"], ["y"])],
        [_f32("x", (4, 6)), _f32("like", (1,))],
        [_f32("y", (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="not statically"):
        lifter.lift()


def test_castlike_on_an_unmapped_initializer_dtype_is_refused():
    """A string exemplar exists — its dtype just has no IR name."""
    model = _raw_model(
        [_node("CastLike", ["x", "like"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [numpy_helper.from_array(np.asarray(["a"]), "like")],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="typed exemplar"):
        _lift(model)


# ---------------------------------------------------------------------------
#  product structure / layout
# ---------------------------------------------------------------------------


def test_transpose_without_a_perm_reverses():
    """A perm-less ``Transpose`` mints the rank-reversing ``permute``."""
    model = _model(
        [_node("Transpose", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (6, 4))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "permute"
    assert _round_trip(model, [_X], name="transpose-rev") == 0.0


def test_transpose_with_a_general_perm_mints_permute():
    """A permutation moving more than two axes is ``permute``."""
    model = _model(
        [_node("Transpose", ["x"], ["y"], perm=[1, 2, 0])],
        [_f32("x", (2, 4, 6))],
        [_f32("y", (4, 6, 2))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "permute"
    x = np.random.default_rng(0).standard_normal((2, 4, 6)).astype(DT)
    assert _round_trip(model, [x], name="permute") == 0.0


def test_reshape_zero_extent_copies_the_input_dim():
    """``shape=[0, -1]`` resolves through the input's static shape."""
    model = _model(
        [_node("Reshape", ["x", "sh"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [_init("sh", np.asarray([0, -1], dtype=np.int64))],
    )
    _, ir, _ = _lift(model)
    assert _attrs_of(ir.root, "reshape")["shape"] == (4, -1)
    assert _round_trip(model, [_X], name="reshape-0") == 0.0


def test_reshape_allowzero_is_refused():
    """``allowzero=1`` reads 0 as an empty axis — not ATen's."""
    model = _model(
        [_node("Reshape", ["x", "sh"], ["y"], allowzero=1)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 0, 6))],
        [_init("sh", np.asarray([4, 0, 6], dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="allowzero"):
        _lift(model)


def test_reshape_zero_extent_on_an_unknown_dim_is_refused():
    """A 0 extent needs the input dim it copies to be static."""
    model = _model(
        [_node("Reshape", ["x", "sh"], ["y"])],
        [_vi("x", TensorProto.FLOAT, (None, 6))],
        [_vi("y", TensorProto.FLOAT, (None, None))],
        [_init("sh", np.asarray([0, -1], dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="not statically"):
        _lift(model)


def test_unsqueeze_axis_out_of_range_is_refused():
    """An insert axis past the output rank is a loud refusal."""
    model = _model(
        [_node("Unsqueeze", ["x", "ax"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6, 1))],
        [_init("ax", np.asarray([5], dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="out of range"):
        _lift(model)


def test_unsqueeze_without_a_known_output_rank_is_refused():
    """Multi-axis inserts need the output rank statically known."""
    lifter = _lifter(
        [_node("Unsqueeze", ["x", "ax"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 1, 6))],
        [_init("ax", np.asarray([1], dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="output rank"):
        lifter.lift()


def test_squeeze_with_explicit_axes_decomposes():
    """An axes list mints one ``squeeze`` per axis, highest first."""
    model = _model(
        [_node("Squeeze", ["x", "ax"], ["y"])],
        [_f32("x", (1, 4, 1))],
        [_f32("y", (4,))],
        [_init("ax", np.asarray([0, 2], dtype=np.int64))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "squeeze"
    x = np.random.default_rng(0).standard_normal((1, 4, 1)).astype(DT)
    assert _round_trip(model, [x], name="squeeze-axes") == 0.0


def test_squeeze_axes_without_a_known_input_rank_is_refused():
    """The per-axis decomposition needs the input's rank."""
    lifter = _lifter(
        [_node("Squeeze", ["x", "ax"], ["y"])],
        [_f32("x", (1, 4))],
        [_f32("y", (4,))],
        [_init("ax", np.asarray([0], dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="input rank"):
        lifter.lift()


def test_expand_and_tile_fold_their_constant_shapes():
    """``Expand``/`Tile` mint ``expand``/`repeat` over the const input."""
    expand = _model(
        [_node("Expand", ["x", "sh"], ["y"])],
        [_f32("x", (4, 1))],
        [_f32("y", (4, 6))],
        [_init("sh", np.asarray([4, 6], dtype=np.int64))],
    )
    _, ir, _ = _lift(expand)
    assert ir.root.op == "expand"
    x = np.random.default_rng(0).standard_normal((4, 1)).astype(DT)
    assert _round_trip(expand, [x], name="expand") == 0.0

    tile = _model(
        [_node("Tile", ["x", "rc"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 12))],
        [_init("rc", np.asarray([1, 2], dtype=np.int64))],
    )
    _, ir, _ = _lift(tile)
    assert ir.root.op == "repeat"
    assert _round_trip(tile, [_X], name="tile") == 0.0


def test_pad_with_a_constant_value_round_trips():
    """The optional ``value`` input folds into the ``pad`` attr."""
    model = _model(
        [_node("Pad", ["x", "pw", "pv"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 8))],
        [
            _init("pw", np.asarray([0, 0, 0, 2], dtype=np.int64)),
            _init("pv", np.asarray(0.5, dtype=DT)),
        ],
    )
    _, ir, _ = _lift(model)
    assert _attrs_of(ir.root, "pad")["value"] == pytest.approx(0.5)
    assert _round_trip(model, [_X], name="pad-value") < 1e-6


def test_pad_modes_map_onto_the_aten_vocabulary():
    """``edge`` -> ``replicate``, ``wrap`` -> ``circular``."""
    model = _model(
        [_node("Pad", ["x", "pw"], ["y"], mode="edge")],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 8))],
        [_init("pw", np.asarray([0, 0, 0, 2], dtype=np.int64))],
    )
    _, ir, leaves = _lift(model)
    assert _attrs_of(ir.root, "pad")["mode"] == "replicate"
    # The sink only lowers constant-mode pads — the lift is wider.
    with pytest.raises(UnsupportedOnnxOpError):
        OnnxSink().lower(ir, leaves)


def test_slice_open_end_omits_the_end_attr():
    """``end=INT64_MAX, step>0`` is ATen's open bound — no ``end``."""
    model = _model(
        [_node("Slice", ["x", "st", "en"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 5))],
        [
            _init("st", np.asarray([1], dtype=np.int64)),
            _init("en", np.asarray([_INT64_MAX], dtype=np.int64)),
        ],
    )
    _, ir, _ = _lift(model)
    attrs = _attrs_of(ir.root, "slice")
    assert "end" not in attrs
    assert _round_trip(model, [_X], name="slice-open") == 0.0


def test_slice_full_reversal_mints_flip():
    """``start=-1, end=INT64_MIN, step=-1`` is ``flip``, not a slice."""
    model = _model(
        [_node("Slice", ["x", "st", "en", "ax", "sp"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [
            _init("st", np.asarray([-1], dtype=np.int64)),
            _init("en", np.asarray([_INT64_MIN], dtype=np.int64)),
            _init("ax", np.asarray([1], dtype=np.int64)),
            _init("sp", np.asarray([-1], dtype=np.int64)),
        ],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "flip"
    assert _round_trip(model, [_X], name="slice-flip") == 0.0


def test_slice_with_ragged_bounds_is_refused():
    """``starts``/`ends`/`axes`/`steps` must agree in length."""
    model = _model(
        [_node("Slice", ["x", "st", "en"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [
            _init("st", np.asarray([1, 0], dtype=np.int64)),
            _init("en", np.asarray([5], dtype=np.int64)),
        ],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="disagree"):
        _lift(model)


def test_split_equal_parts_fold_from_the_input_shape():
    """No ``split`` input — ``num_outputs`` and the extent decide."""
    model = _model(
        [_node("Split", ["x"], ["o1", "o2", "o3"], axis=1)],
        [_f32("x", (4, 6))],
        [_f32("o1", (4, 2)), _f32("o2", (4, 2)), _f32("o3", (4, 2))],
    )
    _, ir, _ = _lift(model)
    assert "split" in _ops_of(ir.root)
    assert ir.root.op == "pack"


def test_split_equal_parts_needs_the_extent():
    """An equal split over an unknown extent is a refusal."""
    lifter = _lifter(
        [_node("Split", ["x"], ["o1", "o2"], axis=1)],
        [_f32("x", (4, 6))],
        [_f32("o1", (4, 3)), _f32("o2", (4, 3))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="axis extent"):
        lifter.lift()


def test_split_equal_parts_must_divide_evenly():
    """``num_outputs`` that does not divide the extent is refused."""
    model = _model(
        [_node("Split", ["x"], ["o1", "o2", "o3", "o4"], axis=1)],
        [_f32("x", (4, 6))],
        [_f32(n, (4, 2)) for n in ("o1", "o2", "o3", "o4")],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="evenly"):
        _lift(model)


def test_split_skips_an_empty_output_name():
    """ONNX's optional-output holes bind nothing."""
    _, ir, _ = _lift_raw(
        [
            _node("Split", ["x", "sz"], ["o1", "", "o3"], axis=1),
            _node("Add", ["o1", "o3"], ["y"]),
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 2))],
        [_init("sz", np.asarray([2, 2, 2], dtype=np.int64))],
    )
    assert "split" in _ops_of(ir.root)


def test_topk_binds_values_and_the_getitem_indices():
    """``TopK`` mints ``topk``; the indices output rides ``getitem``."""
    model = _model(
        [_node("TopK", ["x", "k"], ["v", "i"], axis=1)],
        [_f32("x", (4, 6))],
        [_f32("v", (4, 2))],
        [_init("k", np.asarray(2, dtype=np.int64))],
    )
    _, ir, leaves = _lift(model)
    assert ir.root.op == "topk"
    assert _attrs_of(ir.root, "topk")["k"] == 2
    # ``topk`` is a deliberate lift-only op — the sink declines it.
    with pytest.raises(UnsupportedOnnxOpError):
        OnnxSink().lower(ir, leaves)


def test_topk_skips_an_empty_output_name():
    """An unused optional output binds nothing."""
    _, ir, _ = _lift_raw(
        [_node("TopK", ["x", "k"], ["v", ""], axis=1)],
        [_f32("x", (4, 6))],
        [_f32("v", (4, 2))],
        [_init("k", np.asarray(2, dtype=np.int64))],
    )
    assert ir.root.op == "topk"


# ---------------------------------------------------------------------------
#  indexing / selects
# ---------------------------------------------------------------------------


def test_gather_scalar_index_paths():
    """Literal scalar -> ``select``; live scalar -> select+reshape."""
    w = np.random.default_rng(0).standard_normal((5, 4)).astype(DT)
    # A scalar *initializer* index folds to ``select``.
    model = _model(
        [_node("Gather", ["w", "idx"], ["y"], axis=0)],
        [],
        [_f32("y", (4,))],
        [_init("w", w), _init("idx", np.asarray(2, dtype=np.int64))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "select"

    # A scalar *input* index decomposes through the flat select.
    # ONNX's spec (and its reference evaluator) says a scalar index
    # drops the axis — ``(4,)``; onnxruntime's CPU EP deviates and
    # keeps a ``(1, 4)``, so the oracle here is NumPy's ``w[2]``,
    # which is the semantics Gather names.
    model = _model(
        [_node("Gather", ["w", "idx"], ["y"], axis=0)],
        [_i64("idx", ())],
        [_f32("y", (4,))],
        [_init("w", w)],
    )
    source, ir, leaves = _lift(model)
    ops = _ops_of(ir.root)
    assert {"reshape", "index_select"} <= ops
    executor = OnnxSink(input_dtypes=source.input_dtypes).lower(
        ir, leaves
    )
    got = np.asarray(executor.forward(np.asarray(2, dtype=np.int64)))
    np.testing.assert_array_equal(got, w[2])


def test_gather_with_a_multi_rank_index_reshapes():
    """ONNX's any-rank index flattens, selects, then restores."""
    w = np.random.default_rng(0).standard_normal((5, 4)).astype(DT)
    model = _model(
        [_node("Gather", ["w", "idx"], ["y"], axis=0)],
        [_i64("idx", (2, 2))],
        [_f32("y", (2, 2, 4))],
        [_init("w", w)],
    )
    _, ir, _ = _lift(model)
    assert "index_select" in _ops_of(ir.root)
    idx = np.asarray([[0, 2], [1, 3]], dtype=np.int64)
    assert _round_trip(model, [idx], name="gather-2d") == 0.0


def test_gather_off_axis_mints_index_select():
    """A rank-1 index on a non-zero axis is plain ``index_select``."""
    model = _model(
        [_node("Gather", ["x", "idx"], ["y"], axis=1)],
        [_f32("x", (4, 6)), _i64("idx", (3,))],
        [_f32("y", (4, 3))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "index_select"
    idx = np.asarray([0, 2, 5], dtype=np.int64)
    assert _round_trip(model, [_X, idx], name="gather-axis1") == 0.0


def test_gather_decompose_needs_the_output_shape():
    """A non-vector index restores its rank — statically or not at all."""
    lifter = _lifter(
        [_node("Gather", ["w", "idx"], ["y"], axis=0)],
        [_i64("idx", (2, 2))],
        [_f32("y", (2, 2, 4))],
        [_init("w", np.ones((5, 4), dtype=DT))],
        value_info={"idx": ((2, 2), "int64")},
    )
    with pytest.raises(
        UnsupportedOnnxOpError, match="statically known"
    ):
        lifter.lift()


def test_gather_elements_mints_gather():
    """``GatherElements`` is the IR's same-rank ``gather``."""
    model = _model(
        [_node("GatherElements", ["x", "idx"], ["y"], axis=1)],
        [_f32("x", (4, 6)), _i64("idx", (4, 2))],
        [_f32("y", (4, 2))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "gather"
    idx = np.asarray([[0, 1], [2, 3], [4, 5], [0, 2]], dtype=np.int64)
    assert _round_trip(model, [_X, idx], name="gatherelements") == 0.0


def test_nonzero_mints_the_transposed_form():
    """ONNX's ``[rank, N]`` is ATen's ``[N, rank]`` transposed back."""
    model = _model(
        [_node("NonZero", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_i64("y", (2, None))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "permute"
    assert "nonzero" in _ops_of(ir.root)
    z = _X.copy()
    z[z < 0] = 0.0
    assert _round_trip(model, [z], name="nonzero") == 0.0


def test_cumsum_folds_the_axis_constant():
    """``CumSum``'s axis input becomes the ``dim`` attr."""
    model = _model(
        [_node("CumSum", ["x", "ax"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [_init("ax", np.asarray(1, dtype=np.int64))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "cumsum"
    assert _round_trip(model, [_X], name="cumsum") < 1e-6


@pytest.mark.parametrize("attr", [{"exclusive": 1}, {"reverse": 1}])
def test_cumsum_exclusive_or_reverse_is_refused(attr):
    """A different scan is refused, not approximated."""
    model = _model(
        [_node("CumSum", ["x", "ax"], ["y"], **attr)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [_init("ax", np.asarray(1, dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="scan"):
        _lift(model)


def test_trilu_upper_and_lower():
    """``Trilu``'s ``upper`` flag picks ``triu``/``tril``; ``k`` folds."""
    for upper, expected in ((1, "triu"), (0, "tril")):
        model = _model(
            [_node("Trilu", ["x", "k"], ["y"], upper=upper)],
            [_f32("x", (4, 4))],
            [_f32("y", (4, 4))],
            [_init("k", np.asarray(1, dtype=np.int64))],
        )
        _, ir, _ = _lift(model)
        assert ir.root.op == expected
        x = np.random.default_rng(0).standard_normal((4, 4)).astype(DT)
        assert _round_trip(model, [x], name=expected) == 0.0


def test_range_folds_into_arange():
    """``Range``'s three scalar bounds become ``arange``'s args."""
    model = _model(
        [_node("Range", ["s", "e", "d"], ["y"])],
        [],
        [_f32("y", (3,))],
        [
            _init("s", np.asarray(0, dtype=DT)),
            _init("e", np.asarray(6, dtype=DT)),
            _init("d", np.asarray(2, dtype=DT)),
        ],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "arange"
    assert _round_trip(model, [], name="range") == 0.0


def test_range_with_an_untypeable_output_drops_the_dtype():
    """An ``arange`` with no nameable output dtype mints without one.

    The mint carries no ``dtype`` when the declared element type has
    no IR name — and the graph-output check then refuses the string
    output honestly, so the model is still refused, just later.
    """
    model = _raw_model(
        [_node("Range", ["s", "e", "d"], ["y"])],
        [],
        [_vi("y", TensorProto.STRING, (3,))],
        [
            _init("s", np.asarray(0, dtype=np.int64)),
            _init("e", np.asarray(6, dtype=np.int64)),
            _init("d", np.asarray(2, dtype=np.int64)),
        ],
    )
    with pytest.raises(OnnxImportError, match="not a tensor"):
        _lift(model)


def test_constant_of_shape_defaults_to_zero():
    """No ``value`` attr is ``full(shape, 0.0)`` at float32."""
    model = _model(
        [_node("ConstantOfShape", ["sh"], ["y"])],
        [],
        [_f32("y", (2, 3))],
        [_init("sh", np.asarray([2, 3], dtype=np.int64))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "full"
    assert _round_trip(model, [], name="cofshape") == 0.0


def test_constant_of_shape_folds_the_value_tensor():
    """A single-element ``value`` fills at its own dtype."""
    model = _model(
        [
            _node(
                "ConstantOfShape",
                ["sh"],
                ["y"],
                value=_init("v", np.asarray([2.5], dtype=DT)),
            )
        ],
        [],
        [_f32("y", (2, 3))],
        [_init("sh", np.asarray([2, 3], dtype=np.int64))],
    )
    _, ir, _ = _lift(model)
    assert _attrs_of(ir.root, "full")["fill_value"] == pytest.approx(
        2.5
    )
    assert _round_trip(model, [], name="cofshape-v") == 0.0


def test_constant_of_shape_refuses_the_bad_values():
    """Multi-element or unnameable ``value`` tensors are refused."""
    model = _model(
        [
            _node(
                "ConstantOfShape",
                ["sh"],
                ["y"],
                value=_init("v", np.asarray([1.0, 2.0], dtype=DT)),
            )
        ],
        [],
        [_f32("y", (2, 3))],
        [_init("sh", np.asarray([2, 3], dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="single element"):
        _lift(model)

    model = _raw_model(
        [
            _node(
                "ConstantOfShape",
                ["sh"],
                ["y"],
                value=numpy_helper.from_array(
                    np.asarray([complex(1, 2)], dtype=np.complex64),
                    "v",
                ),
            )
        ],
        [],
        [_f32("y", (2, 3))],
        [_init("sh", np.asarray([2, 3], dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="dtype"):
        _lift(model)


def test_one_hot_at_the_trailing_axis():
    """``OneHot`` with a ``[0, 1]`` pair mints ``one_hot``."""
    model = _model(
        [_node("OneHot", ["x", "dp", "vs"], ["y"], axis=-1)],
        [_i64("x", (4,))],
        [_i64("y", (4, 6))],
        [
            _init("dp", np.asarray(6, dtype=np.int64)),
            _init("vs", np.asarray([0, 1], dtype=np.int64)),
        ],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "one_hot"
    idx = np.asarray([0, 2, 1, 5], dtype=np.int64)
    assert _round_trip(model, [idx], name="onehot") == 0.0


def test_one_hot_refuses_the_wrong_spellings():
    """A non-``[0, 1]`` pair or an off-axis encode is refused."""
    model = _model(
        [_node("OneHot", ["x", "dp", "vs"], ["y"], axis=-1)],
        [_i64("x", (4,))],
        [_i64("y", (4, 6))],
        [
            _init("dp", np.asarray(6, dtype=np.int64)),
            _init("vs", np.asarray([5, 7], dtype=np.int64)),
        ],
    )
    with pytest.raises(UnsupportedOnnxOpError, match=r"\[0, 1\]"):
        _lift(model)

    model = _model(
        [_node("OneHot", ["x", "dp", "vs"], ["y"], axis=0)],
        [_i64("x", (4,))],
        [_i64("y", (6, 4))],
        [
            _init("dp", np.asarray(6, dtype=np.int64)),
            _init("vs", np.asarray([0, 1], dtype=np.int64)),
        ],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="last axis"):
        _lift(model)


def test_where_mints_the_ternary_select():
    """``Where`` carries its three operands verbatim."""
    model = _model(
        [_node("Where", ["c", "x", "z"], ["y"])],
        [_bool("c", (4, 6)), _f32("x", (4, 6)), _f32("z", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "where"
    c = _X > 0
    z = np.zeros_like(_X)
    assert _round_trip(model, [c, _X, z], name="where") == 0.0


def test_dropout_in_eval_is_the_identity_path():
    """A single-output ``Dropout`` mints eval-mode ``dropout``."""
    model = _model(
        [_node("Dropout", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "dropout"
    assert _round_trip(model, [_X], name="dropout") == 0.0


def test_dropout_skips_an_empty_mask_output():
    """An unnamed mask slot binds nothing — an empty name is a hole."""
    _, ir, _ = _lift_raw(
        [_node("Dropout", ["x"], ["y", ""])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    assert ir.root.op == "dropout"


def test_dropout_with_a_consumed_mask_is_refused():
    """The mask output has no IR spelling — eval mode makes none."""
    model = _model(
        [_node("Dropout", ["x"], ["y", "mask"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="mask"):
        _lift(model)


# ---------------------------------------------------------------------------
#  linear algebra / conv / pooling
# ---------------------------------------------------------------------------


def test_gemm_transposed_and_scaled_decomposes():
    """Off the ``linear`` spelling, ``Gemm`` composes honestly."""
    w = np.random.default_rng(1).standard_normal((6, 3)).astype(DT)
    c = np.random.default_rng(2).standard_normal((3,)).astype(DT)
    model = _model(
        [
            _node(
                "Gemm",
                ["x", "w", "c"],
                ["y"],
                transA=0,
                transB=0,
                alpha=2.0,
                beta=0.5,
            )
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 3))],
        [_init("w", w), _init("c", c)],
    )
    _, ir, _ = _lift(model)
    ops = _ops_of(ir.root)
    assert {"matmul", "mul", "add"} <= ops
    assert "linear" not in ops
    assert _round_trip(model, [_X], name="gemm-scaled") < 1e-5


def test_gemm_with_unit_beta_adds_unscaled():
    """``beta=1`` adds ``C`` without the scaling ``mul``."""
    w = np.random.default_rng(1).standard_normal((6, 3)).astype(DT)
    c = np.random.default_rng(2).standard_normal((3,)).astype(DT)
    model = _model(
        [_node("Gemm", ["x", "w", "c"], ["y"], alpha=2.0, beta=1.0)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 3))],
        [_init("w", w), _init("c", c)],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "add"
    assert _round_trip(model, [_X], name="gemm-beta1") < 1e-5


def test_gemm_transposes_its_operands():
    """``transA``/`transB` mint operand ``permute``s."""
    w = np.random.default_rng(1).standard_normal((3, 6)).astype(DT)
    model = _model(
        [_node("Gemm", ["a", "b"], ["y"], transA=1, transB=1)],
        [_f32("a", (6, 4))],
        [_f32("y", (4, 3))],
        [_init("b", w)],
    )
    _, ir, _ = _lift(model)
    ops = _ops_of(ir.root)
    assert "permute" in ops and "matmul" in ops


def test_einsum_carries_the_equation():
    """``Einsum`` mints ``einsum`` — a lift the sink declines."""
    w = np.random.default_rng(1).standard_normal((6, 3)).astype(DT)
    model = _model(
        [_node("Einsum", ["x", "w"], ["y"], equation="ij,jk->ik")],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 3))],
        [_init("w", w)],
    )
    _, ir, leaves = _lift(model)
    assert ir.root.op == "einsum"
    assert _attrs_of(ir.root, "einsum")["equation"] == "ij,jk->ik"
    with pytest.raises(UnsupportedOnnxOpError):
        OnnxSink().lower(ir, leaves)


def test_einsum_without_an_equation_is_refused():
    """The equation is the op's whole meaning — absent is refused."""
    lifter = _lifter(
        [_node("Einsum", ["x", "x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="equation"):
        lifter.lift()


def test_conv_2d_round_trips():
    """A standard ``Conv`` lifts to ``conv2d`` and re-lowers."""
    w = (
        np.random.default_rng(1)
        .standard_normal((3, 2, 3, 3))
        .astype(DT)
    )
    model = _model(
        [
            _node(
                "Conv",
                ["x", "w"],
                ["y"],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
                dilations=[1, 1],
            )
        ],
        [_f32("x", (1, 2, 8, 8))],
        [_f32("y", (1, 3, 4, 4))],
        [_init("w", w)],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "conv2d"
    x = (
        np.random.default_rng(0)
        .standard_normal((1, 2, 8, 8))
        .astype(DT)
    )
    assert _round_trip(model, [x], name="conv2d") < 1e-5


def test_conv_1d_round_trips():
    """A rank-3 weight mints ``conv1d``."""
    w = np.random.default_rng(1).standard_normal((3, 2, 3)).astype(DT)
    model = _model(
        [_node("Conv", ["x", "w"], ["y"], pads=[1, 1])],
        [_f32("x", (1, 2, 8))],
        [_f32("y", (1, 3, 8))],
        [_init("w", w)],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "conv1d"


def test_conv_auto_pad_is_refused():
    """``auto_pad`` computes padding the IR cannot spell."""
    model = _model(
        [_node("Conv", ["x", "w"], ["y"], auto_pad="SAME_UPPER")],
        [_f32("x", (1, 2, 8, 8))],
        [_f32("y", (1, 3, 8, 8))],
        [_init("w", np.ones((3, 2, 3, 3), dtype=DT))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="auto_pad"):
        _lift(model)


def test_conv_refuses_the_mismatched_windows():
    """``strides``/`dilations` must match the spatial rank."""
    model = _model(
        [_node("Conv", ["x", "w"], ["y"], strides=[1, 1, 1])],
        [_f32("x", (1, 2, 8, 8))],
        [_f32("y", (1, 3, 8, 8))],
        [_init("w", np.ones((3, 2, 3, 3), dtype=DT))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="rank"):
        _lift(model)


def test_conv_refuses_the_unsupported_ranks():
    """3-D convolution and an unshaped weight are both refusals."""
    model = _model(
        [_node("Conv", ["x", "w"], ["y"])],
        [_f32("x", (1, 2, 4, 4, 4))],
        [_f32("y", (1, 3, 2, 2, 2))],
        [_init("w", np.ones((3, 2, 3, 3, 3), dtype=DT))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="convolution"):
        _lift(model)

    lifter = _lifter(
        [_node("Conv", ["x", "w"], ["y"])],
        [_f32("x", (1, 2, 8, 8)), _f32("w", (3, 2, 3, 3))],
        [_f32("y", (1, 3, 8, 8))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="weight"):
        lifter.lift()


def test_max_pool_lifts_values_only():
    """``MaxPool`` mints ``max_pool2d`` — wider than the sink."""
    model = _model(
        [_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2])],
        [_f32("x", (1, 2, 8, 8))],
        [_f32("y", (1, 2, 7, 7))],
    )
    _, ir, leaves = _lift(model)
    assert ir.root.op == "max_pool2d"
    assert _attrs_of(ir.root, "max_pool2d")["kernel_size"] == (2, 2)
    with pytest.raises(UnsupportedOnnxOpError):
        OnnxSink().lower(ir, leaves)


def test_max_pool_refuses_the_extra_surfaces():
    """Indices outputs, column-major order and non-2-D pools refuse."""
    # The Indices output present (even unused) is a refusal.
    model = _raw_model(
        [_node("MaxPool", ["x"], ["y", "idx"], kernel_shape=[2, 2])],
        [_f32("x", (1, 2, 8, 8))],
        [_f32("y", (1, 2, 7, 7))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="Indices"):
        _lift(model)

    model = _model(
        [
            _node(
                "MaxPool",
                ["x"],
                ["y"],
                kernel_shape=[2, 2],
                storage_order=1,
            )
        ],
        [_f32("x", (1, 2, 8, 8))],
        [_f32("y", (1, 2, 7, 7))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="storage_order"):
        _lift(model)

    model = _model(
        [_node("MaxPool", ["x"], ["y"], kernel_shape=[3])],
        [_f32("x", (1, 2, 8))],
        [_f32("y", (1, 2, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="max_pool2d"):
        _lift(model)


def test_global_pools_reduce_the_spatial_axes():
    """``GlobalAveragePool``/`GlobalMaxPool` are keepdim reductions."""
    for op, ir_op in (
        ("GlobalAveragePool", "mean"),
        ("GlobalMaxPool", "amax"),
    ):
        model = _model(
            [_node(op, ["x"], ["y"])],
            [_f32("x", (1, 2, 8, 8))],
            [_f32("y", (1, 2, 1, 1))],
        )
        _, ir, _ = _lift(model)
        assert ir.root.op == ir_op
        assert _attrs_of(ir.root, ir_op)["dim"] == (2, 3)
        x = np.random.default_rng(0).standard_normal((1, 2, 8, 8))
        assert _round_trip(model, [x.astype(DT)], name=op) < 1e-6


def test_global_pool_needs_a_known_rank():
    """Global pooling over an unshaped input is a refusal."""
    lifter = _lifter(
        [_node("GlobalAveragePool", ["x"], ["y"])],
        [_f32("x", (1, 2, 8, 8))],
        [_f32("y", (1, 2, 1, 1))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="rank"):
        lifter.lift()


# ---------------------------------------------------------------------------
#  normalization
# ---------------------------------------------------------------------------


def test_batch_norm_in_eval_round_trips():
    """Inference ``BatchNormalization`` mints ``batch_norm``."""
    c = np.ones(6, dtype=DT)
    z = np.zeros(6, dtype=DT)
    model = _model(
        [
            _node(
                "BatchNormalization",
                ["x", "s", "b", "m", "v"],
                ["y"],
                epsilon=1e-4,
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
    _, ir, _ = _lift(model)
    assert ir.root.op == "batch_norm"
    assert _attrs_of(ir.root, "batch_norm")["training"] is False
    assert _round_trip(model, [_X], name="batchnorm") < 1e-4


def test_batch_norm_refuses_the_extra_surfaces():
    """Running-stat outputs and missing stats are refusals."""
    c = np.ones(6, dtype=DT)
    z = np.zeros(6, dtype=DT)
    inits = [
        _init("s", c),
        _init("b", z),
        _init("m", z),
        _init("v", c),
    ]
    model = _raw_model(
        [
            _node(
                "BatchNormalization",
                ["x", "s", "b", "m", "v"],
                ["y", "m2", "v2"],
            )
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        inits,
    )
    with pytest.raises(UnsupportedOnnxOpError, match="statistic"):
        _lift(model)

    model = _raw_model(
        [_node("BatchNormalization", ["x", "s", "b"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        inits[:2],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="running stats"):
        _lift(model)


def test_layer_norm_round_trips():
    """``LayerNormalization`` folds ``normalized_shape`` off the scale."""
    s = np.random.default_rng(1).standard_normal((6,)).astype(DT)
    b = np.random.default_rng(2).standard_normal((6,)).astype(DT)
    model = _model(
        [
            _node(
                "LayerNormalization",
                ["x", "s", "b"],
                ["y"],
                axis=-1,
                epsilon=1e-4,
            )
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [_init("s", s), _init("b", b)],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "layer_norm"
    assert _attrs_of(ir.root, "layer_norm")["dim"] == (6,)
    assert _round_trip(model, [_X], name="layernorm") < 1e-4


def test_layer_norm_refuses_the_extra_surfaces():
    """Mean/InvStdDev outputs and off-axis scales are refusals."""
    s = np.ones(6, dtype=DT)
    b = np.zeros(6, dtype=DT)
    model = _raw_model(
        [
            _node(
                "LayerNormalization",
                ["x", "s", "b"],
                ["y", "mu", "sd"],
            )
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [_init("s", s), _init("b", b)],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="Mean"):
        _lift(model)

    model = _model(
        [
            _node(
                "LayerNormalization",
                ["x", "s", "b"],
                ["y"],
                axis=0,
            )
        ],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [_init("s", s), _init("b", b)],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="axis"):
        _lift(model)

    lifter = _lifter(
        [_node("LayerNormalization", ["x", "s", "b"], ["y"], axis=-1)],
        [_f32("x", (4, 6)), _f32("s", (6,)), _f32("b", (6,))],
        [_f32("y", (4, 6))],
    )
    with pytest.raises(
        UnsupportedOnnxOpError, match="normalized_shape"
    ):
        lifter.lift()


def test_group_normalization_mints_group_norm():
    """``GroupNormalization`` (opset 21) is a lift-only op."""
    model = _model(
        [
            _node(
                "GroupNormalization",
                ["x", "s", "b"],
                ["y"],
                num_groups=2,
                epsilon=1e-4,
            )
        ],
        [_f32("x", (1, 4, 8))],
        [_f32("y", (1, 4, 8))],
        [
            _init("s", np.ones((4,), dtype=DT)),
            _init("b", np.zeros((4,), dtype=DT)),
        ],
        opset=21,
    )
    _, ir, leaves = _lift(model)
    assert ir.root.op == "group_norm"
    assert _attrs_of(ir.root, "group_norm")["num_groups"] == 2
    with pytest.raises(UnsupportedOnnxOpError):
        OnnxSink().lower(ir, leaves)


def test_group_normalization_refuses_a_bad_group_count():
    """``num_groups<=0`` is refused, not clamped."""
    model = _model(
        [
            _node(
                "GroupNormalization",
                ["x", "s", "b"],
                ["y"],
                num_groups=0,
            )
        ],
        [_f32("x", (1, 4, 8))],
        [_f32("y", (1, 4, 8))],
        [
            _init("s", np.ones((4,), dtype=DT)),
            _init("b", np.zeros((4,), dtype=DT)),
        ],
        opset=21,
    )
    with pytest.raises(UnsupportedOnnxOpError, match="num_groups"):
        _lift(model)


def test_instance_normalization_round_trips():
    """``InstanceNormalization`` mints ``use_input_stats`` instance norm."""
    model = _model(
        [
            _node(
                "InstanceNormalization",
                ["x", "s", "b"],
                ["y"],
                epsilon=1e-4,
            )
        ],
        [_f32("x", (1, 2, 4, 4))],
        [_f32("y", (1, 2, 4, 4))],
        [
            _init("s", np.ones((2,), dtype=DT)),
            _init("b", np.zeros((2,), dtype=DT)),
        ],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "instance_norm"
    assert (
        _attrs_of(ir.root, "instance_norm")["use_input_stats"] is True
    )
    x = np.random.default_rng(0).standard_normal((1, 2, 4, 4))
    assert (
        _round_trip(model, [x.astype(DT)], name="instancenorm") < 1e-4
    )


# ---------------------------------------------------------------------------
#  reductions and arg-reductions
# ---------------------------------------------------------------------------


def test_reduce_axes_from_the_attribute_spelling():
    """The opset-<18 ``axes`` attribute still reads as ``dim``."""
    model = _raw_model(
        [_node("ReduceSum", ["x"], ["y"], axes=[1], keepdims=0)],
        [_f32("x", (4, 6))],
        [_f32("y", (4,))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "sum"
    assert _attrs_of(ir.root, "sum")["dim"] == (1,)


def test_reduce_without_axes_reduces_everything():
    """No axes input and no ``axes`` attr means a full reduction."""
    model = _model(
        [_node("ReduceSum", ["x"], ["y"], keepdims=0)],
        [_f32("x", (4, 6))],
        [_f32("y", ())],
    )
    _, ir, _ = _lift(model)
    assert "dim" not in _attrs_of(ir.root, "sum")
    assert _round_trip(model, [_X], name="reduce-all") < 1e-6


def test_reduce_noop_with_empty_axes_is_identity():
    """``noop_with_empty_axes=1`` mints ``clone``, honestly."""
    model = _model(
        [_node("ReduceSum", ["x", ""], ["y"], noop_with_empty_axes=1)],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "clone"
    assert _round_trip(model, [_X], name="reduce-noop") == 0.0


def test_vector_norms_lift_l1_and_l2():
    """``ReduceL1``/`ReduceL2` are ``linalg_vector_norm`` at ord 1/2."""
    for op, order in (("ReduceL1", 1), ("ReduceL2", 2)):
        model = _model(
            [_node(op, ["x", "ax"], ["y"], keepdims=0)],
            [_f32("x", (4, 6))],
            [_f32("y", (4,))],
            [_init("ax", np.asarray([1], dtype=np.int64))],
        )
        _, ir, _ = _lift(model)
        assert ir.root.op == "linalg_vector_norm"
        assert _attrs_of(ir.root, "linalg_vector_norm")["ord"] == order
        assert _round_trip(model, [_X], name=op) < 1e-6


def test_vector_norm_without_axes_reduces_all():
    """``ReduceL2`` with no axes mints a dim-less vector norm."""
    model = _model(
        [_node("ReduceL2", ["x"], ["y"], keepdims=0)],
        [_f32("x", (4, 6))],
        [_f32("y", ())],
    )
    _, ir, _ = _lift(model)
    assert "dim" not in _attrs_of(ir.root, "linalg_vector_norm")


def test_arg_reductions_carry_axis_and_keepdims():
    """``ArgMax``/`ArgMin` mint ``argmax``/`argmin`` at the ``axis``."""
    for op, ir_op in (("ArgMax", "argmax"), ("ArgMin", "argmin")):
        model = _model(
            [_node(op, ["x"], ["y"], axis=1, keepdims=0)],
            [_f32("x", (4, 6))],
            [_i64("y", (4,))],
        )
        _, ir, _ = _lift(model)
        assert ir.root.op == ir_op
        attrs = _attrs_of(ir.root, ir_op)
        assert attrs["dim"] == 1 and attrs["keepdim"] is False
        assert _round_trip(model, [_X], name=op) == 0.0


# ---------------------------------------------------------------------------
#  the lifter engine — environment, constants, attrs
# ---------------------------------------------------------------------------


def test_a_single_output_op_with_extra_outputs_is_refused():
    """``_one`` refuses a binding that is not exactly one output."""
    model = _raw_model(
        [_node("Relu", ["x"], ["a", "b"])],
        [_f32("x", (4, 6))],
        [_f32("a", (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="1 output"):
        _lift(model)


def test_constant_nodes_register_every_value_kind():
    """Scalar constants mint ``Const``; shaped ones mint ``Param``.

    ONNX has no weak typing, so the int and float spellings feed
    chains of their own element type — an ``Add`` of float + int64
    is ill-typed in ONNX itself, not just in the IR.
    """
    model = _model(
        [
            _node("Constant", [], ["two"], value_int=2),
            _node("Constant", [], ["half"], value_float=0.5),
            _node("Constant", [], ["w"], value_floats=[1.0, 2.0]),
            _node("Add", ["ix", "two"], ["iy"]),
            _node("Mul", ["x", "half"], ["a"]),
            _node("Add", ["a", "w"], ["y"]),
        ],
        [_f32("x", (2,)), _i64("ix", (2,))],
        [_f32("y", (2,)), _i64("iy", (2,))],
    )
    _, ir, leaves = _lift(model)
    assert "add" in _ops_of(ir.root)
    assert "w" in leaves  # the shaped constant materialised as Param
    x = np.ones(2, dtype=DT)
    ix = np.asarray([1, 2], dtype=np.int64)
    assert _round_trip(model, [x, ix], name="consts") < 1e-6


def test_a_constant_with_no_value_spelling_registers_nothing():
    """A ``sparse_value``-only ``Constant`` is not a producer."""
    sparse = helper.make_sparse_tensor(
        numpy_helper.from_array(np.asarray([1.0], dtype=DT), "vals"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), "ix"),
        [2],
    )
    model = _raw_model(
        [
            helper.make_node(
                "Constant", [], ["c"], sparse_value=sparse
            ),
            _node("Add", ["x", "c"], ["y"]),
        ],
        [_f32("x", (2,))],
        [_f32("y", (2,))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="no producer"):
        _lift(model)


def test_an_unmappable_attribute_is_refused_when_read():
    """A graph-typed attribute is unmappable *at the read site*."""
    body = helper.make_graph(
        [_node("Relu", ["c"], ["d"])],
        "body",
        [_f32("c", (2,))],
        [_f32("d", (2,))],
    )
    model = _raw_model(
        [
            helper.make_node(
                "ConstantOfShape", ["sh"], ["y"], value=body
            )
        ],
        [_f32("y", (2,))],
        [_f32("y", (2,))],
        [_init("sh", np.asarray([2], dtype=np.int64))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="cannot spell"):
        _lift(model)


def test_attr_decodes_the_string_and_list_kinds():
    """``attr`` decodes STRINGS lists; scalars decode at their kind."""
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"], names=["a", "b"])],
        "g",
        [_f32("x", (2,))],
        [_f32("y", (2,))],
    )
    lifter = GraphLifter(graph)
    assert lifter.attr(graph.node[0], "names") == ["a", "b"]
    assert lifter.attr(graph.node[0], "missing", 7) == 7


def test_a_non_default_domain_node_is_refused():
    """Vendor-domain nodes refuse — even when the name matches."""
    model = _raw_model(
        [helper.make_node("Relu", ["x"], ["y"], domain="ai.onnx.ml")],
        [_f32("x", (2, 2))],
        [_f32("y", (2, 2))],
    )
    with pytest.raises(UnsupportedOnnxOpError):
        _lift(model)


def test_an_input_shadowed_by_an_initializer_is_a_param():
    """A graph input that is also an initializer lifts as ``Param``."""
    model = _model(
        [_node("Add", ["x", "w"], ["y"])],
        [_f32("x", (2,)), _f32("w", (2,))],
        [_f32("y", (2,))],
        [_init("w", np.asarray([1.0, 2.0], dtype=DT))],
    )
    _, ir, leaves = _lift(model)
    assert [v.name for v in ir.inputs] == ["x"]
    assert list(ir.params) == ["w"]
    assert leaves["w"].tolist() == [1.0, 2.0]


def test_an_unresolved_producer_is_refused():
    """An input no node, initializer or constant defines is refused."""
    model = _raw_model(
        [_node("Add", ["x", "ghost"], ["y"])],
        [_f32("x", (2,))],
        [_f32("y", (2,))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="no producer"):
        _lift(model)


def test_an_absent_attr_fold_input_is_refused():
    """``Slice`` without its bounds input is refused, not guessed."""
    model = _raw_model(
        [_node("Slice", ["x"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="absent"):
        _lift(model)


def test_a_non_scalar_bound_is_refused():
    """``Clip``'s bounds must fold to a single element."""
    model = _model(
        [_node("Clip", ["x", "lo"], ["y"])],
        [_f32("x", (4, 6))],
        [_f32("y", (4, 6))],
        [_init("lo", np.asarray([-0.5, 0.1], dtype=DT))],
    )
    with pytest.raises(UnsupportedOnnxOpError, match="single-element"):
        _lift(model)


def test_a_string_typed_input_is_refused():
    """An element type with no IR name cannot mint a ``Var``."""
    model = _raw_model(
        [_node("Relu", ["x"], ["y"])],
        [_vi("x", TensorProto.STRING, (2,))],
        [_vi("y", TensorProto.STRING, (2,))],
    )
    with pytest.raises(OnnxImportError, match="no tensor type"):
        _lift(model)


def test_outputs_may_resolve_to_leaf_terms():
    """A graph output that is an initializer or a constant lifts."""
    # An initializer can BE the graph output — a Param leaf, not a Var.
    model = _model(
        [_node("Relu", ["x"], ["a"])],
        [_f32("x", (2,))],
        [_f32("a", (2,)), _f32("w", (2,))],
        [_init("w", np.asarray([1.0, 2.0], dtype=DT))],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "pack"
    assert "w" in ir.params

    # So can a ``Constant`` node's output — a Const/Param leaf.
    model = _model(
        [
            _node("Constant", [], ["c"], value_int=3),
            _node("Relu", ["x"], ["a"]),
        ],
        [_f32("x", (2,))],
        [_f32("a", (2,)), _i64("c", ())],
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "pack"


def test_a_twice_named_constant_output_materialises_once():
    """A graph outputting one constant twice reuses its ``Param``.

    The output path resolves leaf outputs without the ``_env``
    memo, so the second resolution hits ``_const_leaf``'s own
    param cache — the idempotence the ``params`` dict exists for.
    """
    model = _raw_model(
        [_node("Constant", [], ["c"], value_floats=[1.0, 2.0])],
        [],
        [_f32("c", (2,)), _f32("c", (2,))],
    )
    _, ir, leaves = _lift(model)
    assert ir.root.op == "pack"
    assert list(leaves) == ["c"]


def test_an_empty_bound_name_is_never_installed(monkeypatch):
    """A lifter binding ``""`` has it filtered — the env stays clean.

    The ``if name`` guard in ``_lift_node`` exists for table
    entries that bind optional outputs; a lifter that yields an
    empty name (ONNX's optional-output hole spelled as a key) must
    not poison the environment.
    """
    from catopt_onnx import ONNX_LIFTS

    def weird(g, node):
        """Bind a term to the empty output name."""
        return {"": g.mint("neg", g.term(node, 0))}

    monkeypatch.setitem(ONNX_LIFTS, "Zod", weird)
    lifter = _lifter(
        [_node("Zod", ["x"], [""])],
        [_f32("x", (2,))],
        [_f32("y", (2,))],
    )
    with pytest.raises(OnnxImportError, match="no producer"):
        lifter.lift()
    assert "" not in lifter._env


def test_an_output_with_no_producer_is_refused():
    """A graph output nothing defines is an ill-formed graph."""
    model = _raw_model(
        [_node("Relu", ["x"], ["a"])],
        [_f32("x", (2,))],
        [_f32("y", (2,))],
    )
    with pytest.raises(OnnxImportError, match="no producer"):
        _lift(model)


def test_inference_failure_falls_back_to_declared_types():
    """When ONNX's inference raises, declared types still carry.

    An un-imported-domain node makes ``infer_shapes`` raise; the
    lifter falls back to the model's declared ``ValueInfo``s and the
    refusal it then reaches is the *op* refusal, not a shape panic.
    """
    model = _raw_model(
        [
            helper.make_node("Relu", ["x"], ["a"], domain="ai.onnx.ml"),
            _node("Neg", ["a"], ["y"]),
        ],
        [_f32("x", (2,))],
        [_f32("y", (2,))],
    )
    with pytest.raises(UnsupportedOnnxOpError):
        _lift(model)


def test_an_unmapped_initializer_dtype_skips_the_table():
    """A dtype with no IR name leaves the initializer out of the table.

    The ``Split`` lifter still needs the operand's shape, so the
    query falls through ``value_info`` to the raw initializer — the
    documented second shape channel.
    """
    strings = np.asarray(["a", "b", "c", "d", "e", "f"])
    model = _raw_model(
        [
            _node("Split", ["w"], ["o1", "o2"], axis=0),
            _node("Identity", ["o1"], ["y"]),
        ],
        [],
        [_f32("y", (3,))],
        [numpy_helper.from_array(strings, "w")],
    )
    _, ir, _ = _lift(model)
    assert "split" in _ops_of(ir.root)


def test_non_default_domain_opsets_are_skipped():
    """``_model_opset`` reads the default domain's entry only."""
    graph = helper.make_graph(
        [_node("Relu", ["x"], ["y"])],
        "g",
        [_f32("x", (2,))],
        [_f32("y", (2,))],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("ai.onnx.ml", 2),
            helper.make_opsetid("", 18),
        ],
        ir_version=10,
    )
    _, ir, _ = _lift(model)
    assert ir.root.op == "relu"


def test_a_model_with_no_default_opset_is_refused():
    """No default-domain ``opset_import`` means opset 0 — refused."""
    graph = helper.make_graph(
        [_node("Relu", ["x"], ["y"])],
        "g",
        [_f32("x", (2,))],
        [_f32("y", (2,))],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("ai.onnx.ml", 2)],
        ir_version=10,
    )
    with pytest.raises(OnnxImportError, match="opset"):
        _lift(model)


def test_a_non_model_input_is_refused():
    """``to_ir`` accepts a ModelProto or a path — nothing else."""
    with pytest.raises(OnnxImportError, match="ModelProto"):
        OnnxSource().to_ir(42)
