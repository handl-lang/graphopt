"""The IR -> ONNX op table — spelling variants and their refusals.

``test_onnx_sink.py`` exercises the sink through torch exports — the
spellings a real ``nn.Module`` produces.  This file exercises
:mod:`catopt_onnx.opmap` on the spellings torch *doesn't* produce:
the alternative attr names the canonical schema accepts (``dims``
for ``dim``, ``size`` for ``shape``), the attr-vs-operand fold
positions, and each refusal the builders declare, plus the small
``lower``/``arrays``/``report`` edges the end-to-end tests never
reach.
"""

from __future__ import annotations

import numpy as np
import pytest
pytestmark = pytest.mark.e2e  # boundary-purpose file: crosses an external engine boundary

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from catopt_onnx import (
    OnnxExecutor,
    OnnxLoweringError,
    UnsupportedOnnxOpError,
    lower_ir,
    to_numpy,
    verify_module,
)

DT = np.float32


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _var(name="x", shape=(4, 6)):
    """A typed input leaf."""
    return Var(name, TensorType(tuple(shape)))


def _one_op(op, *args, var_shape=(4, 6), **attrs):
    """Build a one-input IR rooted at ``Op(op, var, *args, **attrs)``."""
    return IR(
        root=Op.make(
            op, _var(shape=var_shape), *args, validate=False, **attrs
        )
    )


def _lowered_nodes(ir, params=None):
    """Lower ``ir`` and return the emitted ``op_type`` list."""
    model = lower_ir(ir, params)
    return [n.op_type for n in model.graph.node]


def _run(ir, params=None, xs=(), input_dtypes=None):
    """Lower and execute ``ir`` on ``xs``."""
    executor = OnnxExecutor(
        lower_ir(ir, params, input_dtypes=input_dtypes)
    )
    return executor.forward(*xs)


_X = np.random.default_rng(0).standard_normal((4, 6)).astype(DT)


# ---------------------------------------------------------------------------
#  the small edges — lower.py, arrays.py, report.py
# ---------------------------------------------------------------------------


def test_a_bad_working_dtype_is_refused_at_the_builder():
    """``lower_ir`` rejects a dtype outside ``WORKING_DTYPES``."""
    ir = _one_op("relu")
    with pytest.raises(OnnxLoweringError, match="working dtype"):
        lower_ir(ir, dtype="float16")


def test_a_generated_name_never_collides_with_an_input():
    """An input named ``v1`` forces the counter past the collision."""
    var = _var(name="v1")
    ir = IR(root=Op.make("add", var, var))
    model = lower_ir(ir)
    names = [n.name for n in model.graph.input]
    assert names == ["v1"]
    # The single node emitted under a fresh, non-colliding name.
    assert model.graph.node[0].output == ["v2"]


def test_a_const_operand_refusal_names_the_reason():
    """A scalar that must become an attribute can't be a tensor."""
    ir = _one_op("masked_fill", _var("m", (4, 6)), _var("v", (4, 6)))
    with pytest.raises(UnsupportedOnnxOpError, match="constant"):
        lower_ir(ir)


def test_a_bare_const_root_emits_a_scalar_initializer():
    """``Const`` as the whole program lowers, then trips the checker.

    The emit path produces the constant fine; ONNX's own checker is
    what declines an untyped scalar output — the sink's refusal
    story ends at the artifact boundary, not before it.
    """
    import onnx

    with pytest.raises(onnx.checker.ValidationError):
        lower_ir(IR(root=Const(2.5)))


def test_a_non_term_leaf_is_refused():
    """A leaf that is no IR term raises ``OnnxLoweringError``."""
    ir = _one_op("add", "junk")
    with pytest.raises(OnnxLoweringError, match="cannot lower leaf"):
        lower_ir(ir)


def test_to_numpy_accepts_a_plain_sequence():
    """The duck-typed converter handles list-likes, not just arrays."""
    out = to_numpy([1.0, 2.0, 3.0])
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0]
    out = to_numpy(5)
    assert out.shape == ()


def test_verify_module_accepts_bare_callables():
    """``verify_module`` duck-types: no ``forward`` means call it."""
    report = verify_module(
        lambda a: a * 2.0,
        lambda a: a * 2.0,
        np.ones(4, dtype=DT),
    )
    assert report.passed and report.max_abs == 0.0


# ---------------------------------------------------------------------------
#  fills — shaped, like, new
# ---------------------------------------------------------------------------


def test_linear_without_a_bias_omits_the_add():
    """``linear(x, W)`` is ``MatMul(x, Wᵀ)`` — no ``Add`` node."""
    w = Param("w", TensorType((3, 6)))
    ir = IR(root=Op.make("linear", _var(), w, validate=False))
    nodes = _lowered_nodes(ir, {"w": np.ones((3, 6), dtype=DT)})
    assert "Add" not in nodes
    got = np.asarray(
        _run(ir, {"w": np.ones((3, 6), dtype=DT)}, xs=(_X,))
    )
    np.testing.assert_allclose(
        got, _X @ np.ones((3, 6), dtype=DT).T, atol=1e-6
    )


def test_full_like_with_the_fill_attr():
    """``full_like`` reads ``fill_value`` before the operand fold."""
    ir = _one_op("full_like", fill_value=3.0)
    assert "ConstantOfShape" in _lowered_nodes(ir)


def test_shaped_fills_accept_the_dim_spelling():
    """``zeros``/`ones`/`full` fall back to ``dim`` for the shape."""
    ir = _one_op("zeros", dim=(4, 6))
    assert "ConstantOfShape" in _lowered_nodes(ir)


def test_a_shaped_fill_without_a_shape_is_refused():
    """A dynamic allocation is not a constant."""
    ir = _one_op("zeros")
    with pytest.raises(UnsupportedOnnxOpError, match="shape"):
        lower_ir(ir)


def test_full_with_a_const_operand_fills():
    """``full`` takes its fill from a ``Const`` operand too."""
    ir = _one_op("full", Const(1.5), shape=(4, 6))
    assert "ConstantOfShape" in _lowered_nodes(ir)


def test_full_without_a_value_is_refused():
    """``full`` with neither attr nor operand fill is a refusal."""
    ir = IR(root=Op.make("full", shape=(4, 6), validate=False))
    with pytest.raises(UnsupportedOnnxOpError, match="fill value"):
        lower_ir(ir)


def test_new_fill_spellings():
    """``new_full`` fills by attr; absent fill or shape is refused."""
    ir = _one_op("new_full", shape=(4, 6), fill_value=2.0)
    assert "ConstantOfShape" in _lowered_nodes(ir)

    ir = IR(root=Op.make("new_full", shape=(4, 6), validate=False))
    with pytest.raises(UnsupportedOnnxOpError, match="fill value"):
        lower_ir(ir)
    with pytest.raises(UnsupportedOnnxOpError, match="shape"):
        lower_ir(_one_op("new_zeros"))


def test_addcmul_value_as_an_operand():
    """``addcmul`` reads ``value`` off a trailing ``Const`` operand."""
    a = _var("a", (4, 6))
    ir = _one_op("addcmul", a, a, Const(0.5))
    assert "Mul" in _lowered_nodes(ir)


# ---------------------------------------------------------------------------
#  layout — permutations, reshapes, pads
# ---------------------------------------------------------------------------


def test_permute_accepts_the_dims_spelling():
    """``dims`` is the alternative name for the permutation."""
    ir = _one_op("permute", dims=(1, 0))
    nodes = _lowered_nodes(ir)
    assert "Transpose" in nodes


def test_movedim_refuses_mismatched_axes():
    """``source``/`destination` must agree in length."""
    ir = _one_op("movedim", source=(0, 1), destination=1)
    with pytest.raises(UnsupportedOnnxOpError, match="lengths"):
        lower_ir(ir)


def test_reshape_without_a_shape_is_refused():
    ir = _one_op("reshape")
    with pytest.raises(UnsupportedOnnxOpError, match="shape"):
        lower_ir(ir)


def test_expand_spellings_and_refusal():
    """``expand`` falls back to ``dim``; absent shape is refused."""
    ir = _one_op("expand", dim=(4, 6))
    assert "Expand" in _lowered_nodes(ir)
    with pytest.raises(UnsupportedOnnxOpError, match="shape"):
        lower_ir(_one_op("expand"))


def test_repeat_refusals():
    """No ``shape`` — or more counts than axes — is refused."""
    with pytest.raises(UnsupportedOnnxOpError, match="shape"):
        lower_ir(_one_op("repeat"))
    with pytest.raises(UnsupportedOnnxOpError, match="repeat counts"):
        lower_ir(_one_op("repeat", shape=(1, 2, 3)))


def test_flip_spellings_and_refusal():
    """``dims`` works for ``flip``; no axes at all is refused."""
    ir = _one_op("flip", dims=(1,), var_shape=(4, 6))
    assert "Slice" in _lowered_nodes(ir)
    with pytest.raises(UnsupportedOnnxOpError, match="no axes"):
        lower_ir(_one_op("flip"))


def test_pad_refuses_the_missing_and_ragged_widths():
    """No ``pad`` attr, or an odd count of widths, is refused."""
    with pytest.raises(UnsupportedOnnxOpError, match="pad"):
        lower_ir(_one_op("pad"))
    with pytest.raises(UnsupportedOnnxOpError, match="pairs"):
        lower_ir(_one_op("pad", pad=(1, 2, 3)))


def test_pad_with_a_fill_value_matches_torch():
    """The optional ``value`` lowers to ``Pad``'s third input."""
    import torch
    import torch.nn.functional as F

    ir = _one_op("pad", pad=(1, 2), value=0.5)
    got = np.asarray(_run(ir, xs=(_X,)))
    expected = F.pad(torch.from_numpy(_X), (1, 2), value=0.5)
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
#  product structure — split/chunk extent edges
# ---------------------------------------------------------------------------


def test_split_int_size_needs_a_known_extent():
    """An int ``sizes`` fold divides a *static* extent or refuses."""
    dynamic = _var(shape=(4, None))
    ir = IR(
        root=Op.make("split", dynamic, sizes=2, index=0, validate=False)
    )
    with pytest.raises(UnsupportedOnnxOpError, match="extent"):
        lower_ir(ir)


def test_chunk_needs_a_positive_count_and_extent():
    """``chunks=0`` — or an unknowable extent — is refused."""
    with pytest.raises(UnsupportedOnnxOpError, match="chunk"):
        lower_ir(_one_op("chunk", chunks=0))
    dynamic = _var(shape=(4, None))
    ir = IR(root=Op.make("chunk", dynamic, chunks=2, validate=False))
    with pytest.raises(UnsupportedOnnxOpError, match="chunk"):
        lower_ir(ir)


# ---------------------------------------------------------------------------
#  reductions
# ---------------------------------------------------------------------------


def test_vector_norm_without_a_dim_reduces_all():
    """``linalg_vector_norm`` with no ``dim`` emits a bare Reduce."""
    ir = _one_op("linalg_vector_norm", ord=2, keepdim=False)
    nodes = _lowered_nodes(ir)
    assert "ReduceL2" in nodes


def test_count_nonzero_with_a_dim():
    """The ``dim`` spelling reduces along one axis."""
    ir = _one_op("count_nonzero", dim=1)
    assert "ReduceSum" in _lowered_nodes(ir)


# ---------------------------------------------------------------------------
#  normalization
# ---------------------------------------------------------------------------


def test_layer_norm_refusals():
    """No scale, no ``dim``, or a shape longer than the rank."""
    with pytest.raises(UnsupportedOnnxOpError, match="scale"):
        lower_ir(_one_op("layer_norm"))
    with pytest.raises(UnsupportedOnnxOpError, match="normalized"):
        lower_ir(_one_op("layer_norm", _var("w", (6,))))
    with pytest.raises(UnsupportedOnnxOpError, match="longer"):
        lower_ir(_one_op("layer_norm", _var("w", (2,)), dim=(2, 2, 2)))


def test_rms_norm_with_an_explicit_eps_and_no_weight():
    """The attr ``eps`` and the weight-free form both lower."""
    w = _var("w", (6,))
    ir = _one_op("rms_norm", w, dim=(6,), eps=1e-4)
    nodes = _lowered_nodes(ir)
    assert {"Mul", "ReduceMean", "Sqrt", "Div"} <= set(nodes)

    ir = _one_op("rms_norm", dim=(6,))
    nodes = _lowered_nodes(ir)
    assert "Mul" in nodes

    # … and the numbers match the formula the op names.
    x = _X
    got = np.asarray(_run(ir, xs=(x,)))
    expected = x / np.sqrt(
        (x**2).mean(-1, keepdims=True) + np.finfo(DT).eps
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_batch_norm_needs_its_stats():
    """Fewer than five operands is a refusal."""
    ir = _one_op("batch_norm", _var("s", (6,)), _var("b", (6,)))
    with pytest.raises(UnsupportedOnnxOpError, match="stats"):
        lower_ir(ir)


def test_instance_norm_needs_scale_and_bias():
    """Fewer than three operands is a refusal."""
    ir = _one_op(
        "instance_norm",
        _var("s", (6,)),
        use_input_stats=True,
        var_shape=(1, 1, 4, 6),
    )
    with pytest.raises(UnsupportedOnnxOpError, match="scale and bias"):
        lower_ir(ir)


# ---------------------------------------------------------------------------
#  conv / gather / arange / casts
# ---------------------------------------------------------------------------


def test_spatial_args_must_match_the_rank():
    """A three-entry stride on a 2-D conv is refused."""
    w = Param("w", TensorType((3, 2, 3, 3)))
    ir = IR(
        root=Op.make(
            "conv2d",
            _var(shape=(1, 2, 8, 8)),
            w,
            stride=(1, 1, 1),
            validate=False,
        )
    )
    with pytest.raises(UnsupportedOnnxOpError, match="entries"):
        lower_ir(ir, {"w": np.ones((3, 2, 3, 3), dtype=DT)})


def test_conv_needs_a_weight_and_a_known_weight_rank():
    """One operand, or an unshaped weight, is a refusal."""
    ir = _one_op("conv2d")
    with pytest.raises(UnsupportedOnnxOpError, match="weight"):
        lower_ir(ir)

    unknown = Op.make("new_zeros", _var(), size=(3, 3), validate=False)
    ir = IR(
        root=Op.make(
            "conv2d", _var(shape=(1, 2, 8, 8)), unknown, validate=False
        )
    )
    with pytest.raises(UnsupportedOnnxOpError, match="weight"):
        lower_ir(ir)


def test_index_select_by_attr_and_its_refusal():
    """The ``index`` attr spelling lowers; absent both is refused."""
    ir = _one_op("index_select", index=(0, 2), dim=1)
    assert "Gather" in _lowered_nodes(ir)
    with pytest.raises(UnsupportedOnnxOpError, match="index"):
        lower_ir(_one_op("index_select", dim=1))


def test_arange_bound_spellings_and_refusals():
    """``argN`` bounds fold; none — or too many — is refused."""
    with pytest.raises(UnsupportedOnnxOpError, match="bounds"):
        lower_ir(IR(root=Op.make("arange", validate=False)))
    ir = IR(
        root=Op.make(
            "arange", arg0=0, arg1=5, arg2=1, arg3=9, validate=False
        )
    )
    with pytest.raises(UnsupportedOnnxOpError, match="at most 3"):
        lower_ir(ir)


def test_arange_float_bounds_use_the_working_dtype():
    """Non-int bounds mean the emitted ``Range`` is float."""
    ir = IR(
        root=Op.make(
            "arange", arg0=0.0, arg1=6.0, arg2=2.0, validate=False
        )
    )
    got = np.asarray(_run(ir))
    np.testing.assert_allclose(got, np.arange(0.0, 6.0, 2.0))


@pytest.mark.parametrize(
    "op,dtype",
    [("float", np.float32), ("long", np.int64), ("bool", np.bool_)],
)
def test_named_casts_emit_cast_nodes(op, dtype):
    """``float``/`long`/`bool` and kin lower to ``Cast``."""
    ir = _one_op(op)
    got = np.asarray(_run(ir, xs=(_X,)))
    assert got.dtype == dtype
