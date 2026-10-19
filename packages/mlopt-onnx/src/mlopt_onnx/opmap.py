"""The IR -> ONNX op mapping — ADR 0006, direction 1.

One entry per IR op name; each value is a *builder*,
``(builder, term, operand_names) -> value_name``, which appends the
ONNX node(s) implementing that op and returns the name of the value
the term evaluates to.

The mapping is deliberately a **coherent subset, not the whole op
set**.  It covers elementwise arithmetic and transcendental
functions, ``matmul`` / ``linear`` / ``conv``, the product structure
(``concat`` / ``stack`` / ``split`` / ``chunk``), the layout ops
(``transpose`` / ``permute`` / ``reshape`` / ``unsqueeze`` /
``squeeze`` / ``flatten`` / ``pad`` / ``slice`` / ``narrow`` /
``flip``), the reductions, ``softmax`` / ``log_softmax``, the
activations, the normalizations, and the dtype casts.  Everything
else is *refused*: an op with no entry raises
:class:`~mlopt_onnx.errors.UnsupportedOnnxOpError`, and so does a
mapped op in a spelling ONNX cannot express.

Why a subset is the honest answer rather than a shortfall: the
sink's ``supported_ops`` is the bound
``mlopt_core.cost.backend_cost`` prices extraction against, so the
search only ever selects forms this table can lower.  A narrower
table is a narrower *search*, not a broken delivery.  Notable
declines, each for a stated reason rather than by omission:

* ``sdpa`` — no ONNX op; the fused kernel would have to be spelled
  out from primitives, and getting it *equal* is exactly the
  long-tail work ADR 0006 warns about.
* ``max_pool2d`` / ``upsample_*2d`` — the ADR's measured traps
  (``ceil_mode``/``align_corners`` arriving positionally) live here.
* ``randn`` / ``rand`` / ``empty`` — random and uninitialised
  allocation is not a function; a backend that re-randomises on
  every run cannot verify.  (``zeros``/``ones``/``full`` and the
  ``*_like`` family *are* mapped: a constant tensor is a function,
  and ONNX ``ConstantOfShape`` expresses it exactly.)
* ``scatter*`` / ``index_put`` / ``index`` — ONNX's ``ScatterND``
  has different duplicate-index semantics from ATen's.
* ``sort`` / ``topk`` / ``var_mean`` / ``std_mean`` / ``unbind`` —
  multi-output ops whose second output no ONNX op produces
  (``max``/``min`` are the exception: ``ReduceMax``+``ArgMax``).
* ``cumprod`` — ONNX's default domain has ``CumSum`` but no
  ``CumProd`` at all (verified against the installed schemas).
* ``fmax`` / ``fmin`` — NaN-ignoring semantics ONNX ``Max``/``Min``
  do not promise.
* ``randn`` / ``rand`` / ``empty`` — random and uninitialised
  allocation is not a function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from mlopt_core.ir import Op
from onnx import numpy_helper

from mlopt_onnx.arrays import DTYPE_NAMES, numpy_to_onnx_type
from mlopt_onnx.errors import UnsupportedOnnxOpError

if TYPE_CHECKING:
    from mlopt_onnx.lower import GraphBuilder

__all__ = ["ONNX_BINDINGS", "OnnxBuilder"]

#: An op builder: ``(builder, term, operand_names) -> value_name``.
OnnxBuilder = Callable[["GraphBuilder", Op, list[str]], str]

#: ``int64`` bounds, the conventional open-ended ``Slice`` sentinels.
_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)

#: Ops that map onto exactly one ONNX node of the same arity and
#: whose result is the first operand's dtype.
_UNARY: dict[str, str] = {
    "neg": "Neg",
    "abs": "Abs",
    "sign": "Sign",
    "floor": "Floor",
    "ceil": "Ceil",
    "round": "Round",
    "sqrt": "Sqrt",
    "exp": "Exp",
    "log": "Log",
    "reciprocal": "Reciprocal",
    "sin": "Sin",
    "cos": "Cos",
    "tan": "Tan",
    "asin": "Asin",
    "acos": "Acos",
    "atan": "Atan",
    "sinh": "Sinh",
    "cosh": "Cosh",
    "tanh": "Tanh",
    "erf": "Erf",
    "relu": "Relu",
    "relu_": "Relu",
    "sigmoid": "Sigmoid",
    "softsign": "Softsign",
    "selu": "Selu",
    # Memory-layout / aliasing ops are the identity map.
    "contiguous": "Identity",
    "clone": "Identity",
    "alias": "Identity",
    "detach": "Identity",
    "detach_": "Identity",
    "_assert_tensor_metadata": "Identity",
    "broadcast": "Identity",
}

#: Predicate ops — one ONNX node each, with a boolean result.
_UNARY_BOOL: dict[str, str] = {
    "isnan": "IsNaN",
    "isinf": "IsInf",
    "logical_not": "Not",
}

#: Two-operand elementwise ops: ONNX's NumPy-style broadcasting is
#: the same broadcast the IR's typing assumes.
_BINARY: dict[str, str] = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "pow": "Pow",
    "maximum": "Max",
    "minimum": "Min",
    "bitwise_and": "And",
    "bitwise_or": "Or",
    "logical_and": "And",
    "logical_or": "Or",
    "logical_xor": "Xor",
}

#: Comparisons — one ONNX node each, with a boolean result.
_COMPARISON: dict[str, str] = {
    "eq": "Equal",
    "lt": "Less",
    "le": "LessOrEqual",
    "gt": "Greater",
    "ge": "GreaterOrEqual",
}

#: The dtype-cast ops, keyed by IR op name -> NumPy dtype name.  The
#: IR records a dtype *name* (``"float64"``) rather than a torch
#: dtype object, which is what makes this table possible at all.
_CASTS: dict[str, str] = {
    "float": "float32",
    "double": "float64",
    "half": "float16",
    "bool": "bool",
    "int": "int32",
    "long": "int64",
    "short": "int16",
    "char": "int8",
    "byte": "uint8",
}

_BOOL = np.dtype(np.bool_)
_INT64 = np.dtype(np.int64)

#: Reductions that take their axes as an ONNX *input* (opset 18).
_REDUCERS: dict[str, str] = {
    "sum": "ReduceSum",
    "mean": "ReduceMean",
    "amax": "ReduceMax",
    "amin": "ReduceMin",
    "prod": "ReduceProd",
}


# ---------------------------------------------------------------------------
#  small shared helpers
# ---------------------------------------------------------------------------


def _as_int_list(value: Any) -> list[int]:
    """Normalise an int-or-sequence attribute to a list of ints."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _norm_dim(dim: Any, rank: int, op: str = "axis") -> int:
    """Resolve a possibly-negative axis against a known rank."""
    d = int(dim)
    d = d + rank if d < 0 else d
    if not 0 <= d < rank:
        raise UnsupportedOnnxOpError(
            op, f"axis {dim} is out of range for rank {rank}"
        )
    return d


def _norm_index(index: int, count: int, op: str = "index") -> int:
    """Resolve a possibly-negative index against a known count."""
    i = index + count if index < 0 else index
    if not 0 <= i < count:
        raise UnsupportedOnnxOpError(
            op, f"index {index} is out of range for {count} parts"
        )
    return i


def _simple(op_type: str) -> OnnxBuilder:
    """Build the one-node lowering for ``op_type``."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit ``op_type`` over the term's operands."""
        return b.node(op_type, ins)[0]

    return build


def _boolean(op_type: str) -> OnnxBuilder:
    """Build a one-node lowering whose result is boolean."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit ``op_type`` and type the result as a predicate."""
        return b.node(op_type, ins, dtype=_BOOL)[0]

    return build


def _reducer(op_type: str) -> OnnxBuilder:
    """Build a reduction over an optional ``dim``/``keepdim`` pair."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit the reduction, with axes as an input when given."""
        dim = b.attr(term, "dim", None)
        keepdim = int(bool(b.attr(term, "keepdim", False)))
        if dim is None:
            return b.node(op_type, ins, keepdims=keepdim)[0]
        axes = _as_int_list(dim)
        return b.node(
            op_type, [ins[0], b.int64(axes)], keepdims=keepdim
        )[0]

    return build


# ---------------------------------------------------------------------------
#  individual lowerings
# ---------------------------------------------------------------------------


def _pack(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``pack`` — the multi-output root, refused as a *value*.

    ONNX is natively multi-output: the pairing lowers to the graph's
    output list (see :meth:`mlopt_onnx.lower.GraphBuilder.build`),
    not to a node, and ONNX has no tuple value.  The entry exists so
    ``supported_ops`` admits ``pack`` — extraction must be able to
    price a multi-output program — and so a *nested* pack, which has
    no ONNX spelling at all, is refused loudly rather than dropped.
    """
    raise UnsupportedOnnxOpError(
        term.op,
        "pack is the multi-output root: it lowers to the graph's "
        "output list, not to a node, so a nested pack has no ONNX "
        "equivalent",
    )


def _matmul(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``matmul`` — ONNX ``MatMul`` (batched and vector cases alike)."""
    return b.node("MatMul", ins)[0]


def _linear(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``linear`` — ``MatMul(x, Wᵀ)`` plus an optional bias."""
    rank = b.required_rank(term, 1)
    perm = list(range(rank))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    weight_t = b.node("Transpose", [ins[1]], perm=perm)[0]
    out = b.node("MatMul", [ins[0], weight_t])[0]
    if len(ins) > 2:
        out = b.node("Add", [out, ins[2]])[0]
    return out


def _dropout(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``dropout`` — the identity map in eval mode, refused in train."""
    if bool(b.attr(term, "train", False)):
        raise UnsupportedOnnxOpError(
            term.op,
            "training-mode dropout is random — a graph that "
            "re-randomises cannot be verified",
        )
    return b.node("Identity", ins)[0]


def _mod(fmod_flag: bool) -> OnnxBuilder:
    """Build ``fmod`` — ONNX ``Mod`` with ATen's sign convention."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit the modulo."""
        return b.node("Mod", ins, fmod=int(fmod_flag))[0]

    return build


def _remainder(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``remainder`` — floor-mod, spelled out.

    ONNX ``Mod(fmod=0)`` *is* floor-mod, and onnxruntime accepts it
    for integer inputs (``fmod=0`` is the integer path the runtime
    requires).  For floating types it refuses ("fmod attribute must
    be true for floating point types"), so floats compose
    ``x - floor(x / y) * y`` instead — a spelling ORT has no integer
    ``Floor`` kernel for either, which is why the composition is
    float-only.
    """
    x, y = ins[0], ins[1]
    if b.dtype_of(x).kind in "iu":
        return b.node("Mod", ins, fmod=0)[0]
    floored = b.node("Floor", b.node("Div", [x, y]))[0]
    return b.node("Sub", [x, b.node("Mul", [floored, y])[0]])[0]


def _fill_like(value: Any) -> OnnxBuilder:
    """Build ``zeros_like``/``ones_like`` — ``ConstantOfShape``."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit a same-shape, same-dtype filled tensor."""
        return _constant_of_shape(b, ins[0], value, b.dtype_of(ins[0]))

    return build


def _full_like(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``full_like(x, v)`` — ``ConstantOfShape`` with ``v``."""
    fill = b.attr(term, "fill_value", None)
    if fill is None:
        fill = b.const_operand(term, 1)
    return _constant_of_shape(b, ins[0], fill, b.dtype_of(ins[0]))


def _shaped_fill(value: Any, has_value: bool) -> OnnxBuilder:
    """Build ``zeros``/``ones``/``full`` — ``ConstantOfShape``."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit a constant tensor of the recorded shape and dtype."""
        shape = b.attr(term, "shape", None)
        if shape is None:
            shape = b.attr(term, "dim", None)
        if shape is None:
            raise UnsupportedOnnxOpError(
                term.op,
                "no ``shape`` attr — a dynamic allocation "
                "is not a constant",
            )
        fill = value
        if has_value:
            fill = b.attr(term, "fill_value", None)
            if fill is None:
                fill = (
                    b.const_operand(term, len(term.args) - 1)
                    if term.args
                    else None
                )
            if fill is None:
                raise UnsupportedOnnxOpError(term.op, "no fill value")
        name = b.attr(term, "dtype", None)
        dtype = DTYPE_NAMES.get(name, b.work)
        return _constant_of_shape_values(
            b, _as_int_list(shape), fill, dtype
        )

    return build


def _new_fill(value: Any, has_value: bool) -> OnnxBuilder:
    """Build ``new_zeros``/``new_ones``/``new_full``."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit a constant tensor sized by the recorded shape."""
        shape = b.attr(term, "shape", None)
        if shape is None:
            shape = b.attr(term, "size", None)
        if shape is None:
            raise UnsupportedOnnxOpError(
                term.op, "no ``shape``/``size`` attr"
            )
        fill = value
        if has_value:
            fill = b.attr(term, "fill_value", None)
            if fill is None:
                fill = (
                    b.const_operand(term, len(term.args) - 1)
                    if term.args
                    else None
                )
            if fill is None:
                raise UnsupportedOnnxOpError(term.op, "no fill value")
        return _constant_of_shape_values(
            b, _as_int_list(shape), fill, b.dtype_of(ins[0])
        )

    return build


def _constant_of_shape(
    b: GraphBuilder, source: str, value: Any, dtype: np.dtype
) -> str:
    """``ConstantOfShape(Shape(source), value)``."""
    shape = b.node("Shape", [source], dtype=_INT64)[0]
    return _constant_of_shape_values(b, shape, value, dtype)


def _constant_of_shape_values(
    b: GraphBuilder, shape: Any, value: Any, dtype: np.dtype
) -> str:
    """Emit ``ConstantOfShape`` over a static or computed shape."""
    shape_name = (
        shape if isinstance(shape, str) else b.int64(list(shape))
    )
    filler = numpy_helper.from_array(np.asarray([value], dtype=dtype))
    return b.node(
        "ConstantOfShape", [shape_name], value=filler, dtype=dtype
    )[0]


def _nonzero(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``nonzero`` — ``NonZero`` transposed to ATen's ``[N, rank]``."""
    return b.node(
        "Transpose",
        [b.node("NonZero", ins, dtype=_INT64)[0]],
        perm=[1, 0],
        dtype=_INT64,
    )[0]


def _rsub(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``rsub(x, s) = s - x`` — ``Sub`` with the operands swapped."""
    return b.node("Sub", [ins[1], ins[0]])[0]


def _div(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``div`` — ``Div``, or a floor/trunc rounding composition."""
    mode = b.attr(term, "rounding_mode", None)
    if mode in (None, "none"):
        return b.node("Div", ins)[0]
    if mode == "floor":
        return b.node("Floor", b.node("Div", ins))[0]
    if mode == "trunc":
        return _trunc_quotient(b, b.node("Div", ins)[0])
    raise UnsupportedOnnxOpError(
        term.op, f"rounding_mode={mode!r} has no ONNX equivalent"
    )


def _trunc_quotient(b: GraphBuilder, value: str) -> str:
    """``trunc(v)`` as ``sign(v) * floor(|v|)`` — ONNX has no Trunc."""
    return b.node(
        "Mul",
        [
            b.node("Sign", [value])[0],
            b.node("Floor", [b.node("Abs", [value])[0]])[0],
        ],
    )[0]


def _trunc(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``trunc`` — the sign/floor composition."""
    return _trunc_quotient(b, ins[0])


def _frac(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``frac(x) = x - trunc(x)``."""
    return b.node("Sub", [ins[0], _trunc_quotient(b, ins[0])])[0]


def _rsqrt(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``rsqrt`` — ``Reciprocal(Sqrt(x))``."""
    return b.node("Reciprocal", b.node("Sqrt", ins))[0]


def _square(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``square`` — ``x * x`` (ONNX has no Square)."""
    return b.node("Mul", [ins[0], ins[0]])[0]


def _exp2(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``exp2(x) = 2 ** x``."""
    return b.node("Pow", [b.scalar(2.0), ins[0]])[0]


def _expm1(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``expm1(x) = exp(x) - 1``."""
    return b.node("Sub", [b.node("Exp", ins)[0], b.scalar(1.0)])[0]


def _log1p(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``log1p(x) = log(1 + x)``."""
    return b.node("Log", [b.node("Add", [ins[0], b.scalar(1.0)])[0]])[0]


def _log_base(base: float) -> OnnxBuilder:
    """Build ``log(x) / log(base)`` for a fixed base."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit the base-change composition."""
        return b.node(
            "Div",
            [b.node("Log", ins)[0], b.scalar(float(np.log(base)))],
        )[0]

    return build


def _asinh(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``asinh(x) = log(x + sqrt(x*x + 1))``."""
    x = ins[0]
    inner = b.node("Add", [b.node("Mul", [x, x])[0], b.scalar(1.0)])[0]
    return b.node(
        "Log", [b.node("Add", [x, b.node("Sqrt", [inner])[0]])[0]]
    )[0]


def _acosh(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``acosh(x) = log(x + sqrt(x*x - 1))``."""
    x = ins[0]
    inner = b.node("Sub", [b.node("Mul", [x, x])[0], b.scalar(1.0)])[0]
    return b.node(
        "Log", [b.node("Add", [x, b.node("Sqrt", [inner])[0]])[0]]
    )[0]


def _atanh(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``atanh(x) = 0.5 * log((1 + x) / (1 - x))``."""
    x = ins[0]
    one = b.scalar(1.0)
    ratio = b.node(
        "Div",
        [
            b.node("Add", [one, x])[0],
            b.node("Sub", [one, x])[0],
        ],
    )[0]
    return b.node("Mul", [b.scalar(0.5), b.node("Log", [ratio])[0]])[0]


def _erfc(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``erfc(x) = 1 - erf(x)``."""
    return b.node("Sub", [b.scalar(1.0), b.node("Erf", ins)[0]])[0]


def _ne(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``ne`` — ONNX has no NotEqual; ``Not(Equal(x, y))``."""
    return b.node("Not", b.node("Equal", ins, dtype=_BOOL))[0]


def _isfinite(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``isfinite`` — ``not (isinf or isnan)``."""
    bad = b.node(
        "Or",
        [b.node("IsInf", ins)[0], b.node("IsNaN", ins)[0]],
        dtype=_BOOL,
    )[0]
    return b.node("Not", [bad], dtype=_BOOL)[0]


def _heaviside(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``heaviside(x, h)`` — nested ``Where`` on the sign of ``x``."""
    x, h = ins[0], ins[1]
    pos = b.node("Greater", [x, b.scalar(0.0)], dtype=_BOOL)[0]
    neg = b.node("Less", [x, b.scalar(0.0)], dtype=_BOOL)[0]
    inner = b.node("Where", [neg, b.constant(0.0, b.dtype_of(h)), h])[0]
    return b.node(
        "Where", [pos, b.constant(1.0, b.dtype_of(h)), inner]
    )[0]


def _xlogy(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``xlogy(x, y) = x * log(y)``."""
    return b.node("Mul", [ins[0], b.node("Log", [ins[1]])[0]])[0]


def _lerp(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``lerp(t, e, w) = t + w * (e - t)``."""
    t, e, w = ins[0], ins[1], ins[2]
    return b.node(
        "Add", [t, b.node("Mul", [w, b.node("Sub", [e, t])[0]])[0]]
    )[0]


def _addcmul(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``addcmul(t, a, b, value=1) = t + value * a * b``."""
    scale = b.constant(_scale_of(b, term), b.dtype_of(ins[0]))
    product = b.node("Mul", [ins[1], ins[2]])[0]
    return b.node("Add", [ins[0], b.node("Mul", [scale, product])[0]])[
        0
    ]


def _addcdiv(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``addcdiv(t, a, b, value=1) = t + value * a / b``."""
    scale = b.constant(_scale_of(b, term), b.dtype_of(ins[0]))
    quotient = b.node("Div", [ins[1], ins[2]])[0]
    return b.node("Add", [ins[0], b.node("Mul", [scale, quotient])[0]])[
        0
    ]


def _scale_of(b: GraphBuilder, term: Op) -> Any:
    """Read ``addcmul``/``addcdiv``'s ``value`` — attr or operand."""
    value = b.attr(term, "value", None)
    if value is not None:
        return value
    if len(term.args) > 3:
        return b.const_operand(term, 3)
    return 1.0


def _masked_fill(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``masked_fill(x, mask, v) = where(mask, v, x)``."""
    value = b.const_operand(term, 2)
    return b.node(
        "Where",
        [
            ins[1],
            b.constant(value, b.dtype_of(ins[0])),
            ins[0],
        ],
    )[0]


def _where(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``where`` — ``Where``, typed by its value operands."""
    return b.node("Where", ins, dtype=b.dtype_of(ins[1]))[0]


def _clamp(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``clamp``/``clamp_min``/``clamp_max`` — ONNX ``Clip``."""
    lo = b.attr(term, "min", None)
    hi = b.attr(term, "max", None)
    return _clip(b, term, ins[0], lo, hi)


def _clip(
    b: GraphBuilder,
    term: Op,
    x: str,
    lo: Any,
    hi: Any,
) -> str:
    """Emit ``Clip(x, lo, hi)`` with ONNX's empty-input placeholders."""
    if lo is None and hi is None:
        raise UnsupportedOnnxOpError(
            term.op, "neither bound is given — Clip would be identity"
        )
    dtype = b.dtype_of(x)
    inputs = [x]
    if lo is not None:
        inputs.append(b.constant(lo, dtype))
    if hi is not None:
        if lo is None:
            inputs.append("")
        inputs.append(b.constant(hi, dtype))
    return b.node("Clip", inputs)[0]


def _hardtanh(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``hardtanh`` — ``Clip`` with ATen's ``(-1, 1)`` defaults."""
    return _clip(
        b,
        term,
        ins[0],
        b.attr(term, "min", -1.0),
        b.attr(term, "max", 1.0),
    )


def _relu6(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``relu6`` — ``Clip(x, 0, 6)``."""
    return _clip(b, term, ins[0], 0.0, 6.0)


def _elu(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``elu`` — ONNX ``Elu``, which has only the alpha parameter."""
    alpha = float(b.attr(term, "alpha", 1.0))
    scale = float(b.attr(term, "scale", 1.0))
    input_scale = float(b.attr(term, "input_scale", 1.0))
    if scale != 1.0 or input_scale != 1.0:
        raise UnsupportedOnnxOpError(
            term.op,
            f"scale={scale}, input_scale={input_scale}: ONNX Elu "
            f"applies alpha only",
        )
    return b.node("Elu", ins, alpha=alpha)[0]


def _leaky_relu(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``leaky_relu`` — ONNX ``LeakyRelu``; slope must be static."""
    slope = b.attr(term, "negative_slope", None)
    if slope is None:
        slope = b.const_operand(term, 1) if len(term.args) > 1 else 0.01
    return b.node("LeakyRelu", [ins[0]], alpha=float(slope))[0]


def _softplus(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``softplus`` — ONNX ``Softplus`` is ``log(exp(x) + 1)``."""
    beta = float(b.attr(term, "beta", 1.0))
    threshold = float(b.attr(term, "threshold", 20.0))
    if beta != 1.0 or threshold != 20.0:
        raise UnsupportedOnnxOpError(
            term.op,
            f"beta={beta}, threshold={threshold}: ONNX Softplus has "
            f"neither parameter",
        )
    return b.node("Softplus", ins)[0]


def _gelu(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``gelu`` — the exact ``erf`` form, spelled out."""
    approximate = b.attr(term, "approximate", None)
    if approximate not in (None, "none"):
        raise UnsupportedOnnxOpError(
            term.op, f"approximate={approximate!r} is not exact"
        )
    x = ins[0]
    half = b.node("Mul", [x, b.scalar(0.5)])[0]
    arg = b.node("Mul", [x, b.scalar(float(1.0 / np.sqrt(2.0)))])[0]
    erf = b.node("Erf", [arg])[0]
    return b.node(
        "Mul", [half, b.node("Add", [erf, b.scalar(1.0)])[0]]
    )[0]


def _silu(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``silu(x) = x * sigmoid(x)``."""
    return b.node("Mul", [ins[0], b.node("Sigmoid", ins)[0]])[0]


def _mish(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``mish(x) = x * tanh(softplus(x))``."""
    return b.node(
        "Mul",
        [ins[0], b.node("Tanh", b.node("Softplus", ins))[0]],
    )[0]


def _glu(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``glu`` — split the axis in half, gate one half."""
    dim = int(b.attr(term, "dim", -1))
    rank = b.required_rank(term)
    shape = b.operand_shape(term)
    axis = _norm_dim(dim, rank, term.op)
    extent = shape[axis] if shape is not None else None
    if extent is None or extent % 2:
        raise UnsupportedOnnxOpError(
            term.op,
            "glu needs an even, statically known extent on its "
            "split axis",
        )
    half = extent // 2
    value, gate = b.node(
        "Split",
        [ins[0], b.int64([half, half])],
        n_outputs=2,
        axis=dim,
    )
    return b.node("Mul", [value, b.node("Sigmoid", [gate])[0]])[0]


def _hardsigmoid(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``hardsigmoid`` — ``x/6 + 0.5`` clamped.

    ONNX ``HardSigmoid`` *defaults* to ``alpha=0.2``, but ATen's
    hardsigmoid is ``relu6(x + 3) / 6`` — ``alpha = 1/6``.  Taking
    the ONNX default here would be a silent 20% slope error, which is
    exactly the class of trap ADR 0006's attribute-spelling warning
    is about.
    """
    return b.node("HardSigmoid", ins, alpha=1.0 / 6.0, beta=0.5)[0]


def _hardswish(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``hardswish`` — ONNX ``HardSwish`` (which uses ``1/6``)."""
    return b.node("HardSwish", ins)[0]


def _prelu(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``prelu`` — ONNX ``PRelu`` takes the slope as an input."""
    return b.node("PRelu", ins)[0]


# -- layout ------------------------------------------------------------


def _transpose(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``transpose(x, dim0, dim1)`` — a two-axis swap."""
    rank = b.required_rank(term)
    d0 = _norm_dim(b.attr(term, "dim0", -2), rank, term.op)
    d1 = _norm_dim(b.attr(term, "dim1", -1), rank, term.op)
    perm = list(range(rank))
    perm[d0], perm[d1] = perm[d1], perm[d0]
    return b.node("Transpose", ins, perm=perm)[0]


def _permute(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``permute`` — ``Transpose`` with the explicit permutation."""
    rank = b.required_rank(term)
    dims = b.attr(term, "dim", None)
    if dims is None:
        dims = b.attr(term, "dims", None)
    perm = (
        list(reversed(range(rank)))
        if dims is None
        else [_norm_dim(d, rank, term.op) for d in _as_int_list(dims)]
    )
    return b.node("Transpose", ins, perm=perm)[0]


def _movedim(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``movedim`` — ``Transpose`` with the axes re-inserted."""
    rank = b.required_rank(term)
    source = [
        _norm_dim(d, rank, term.op)
        for d in _as_int_list(b.attr(term, "source", 0))
    ]
    dest = [
        _norm_dim(d, rank, term.op)
        for d in _as_int_list(b.attr(term, "destination", -1))
    ]
    if len(source) != len(dest):
        raise UnsupportedOnnxOpError(
            term.op, "source and destination lengths differ"
        )
    perm = [d for d in range(rank) if d not in source]
    for src, dst in sorted(zip(source, dest, strict=True)):
        perm.insert(dst, src)
    return b.node("Transpose", ins, perm=perm)[0]


def _reshape(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``reshape`` — ``Reshape`` with a static target shape."""
    shape = b.attr(term, "shape", None)
    if shape is None:
        raise UnsupportedOnnxOpError(term.op, "no ``shape`` attr")
    dims = _as_int_list(shape)
    if any(d == 0 for d in dims):
        raise UnsupportedOnnxOpError(
            term.op,
            "a 0 extent is ambiguous — ONNX Reshape reads 0 as "
            "'copy the input dim' where ATen reads it as an empty "
            "axis",
        )
    return b.node("Reshape", [ins[0], b.int64(dims)])[0]


def _flatten(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``flatten`` — ``Flatten`` when the collapse runs to the end.

    ONNX ``Flatten`` is always two-dimensional, so a full flatten
    (``start_dim=0``) is ``Reshape(x, [-1])`` and a partial one is
    ``Flatten(axis=start_dim)`` — which is exactly ATen's
    ``flatten(start_dim, -1)``.
    """
    start = int(b.attr(term, "start_dim", 0))
    end = int(b.attr(term, "end_dim", -1))
    rank = b.required_rank(term)
    if _norm_dim(end, rank, term.op) != rank - 1:
        raise UnsupportedOnnxOpError(
            term.op,
            f"end_dim={end} is not the last axis; ONNX Flatten "
            f"always collapses to the last axis",
        )
    if _norm_dim(start, rank, term.op) == 0:
        return b.node("Reshape", [ins[0], b.int64([-1])])[0]
    return b.node("Flatten", ins, axis=start)[0]


def _unsqueeze(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``unsqueeze`` — ``Unsqueeze`` with the axis as an input."""
    dim = int(b.attr(term, "dim", -1))
    return b.node("Unsqueeze", [ins[0], b.int64([dim])])[0]


def _squeeze(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``squeeze`` — one axis, or every extent-1 axis."""
    dim = b.attr(term, "dim", None)
    if dim is None:
        return b.node("Squeeze", ins)[0]
    return b.node("Squeeze", [ins[0], b.int64([int(dim)])])[0]


def _expand(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``expand`` / ``broadcast_to`` — ``Expand`` to a static shape."""
    shape = b.attr(term, "shape", None)
    if shape is None:
        shape = b.attr(term, "dim", None)
    if shape is None:
        raise UnsupportedOnnxOpError(term.op, "no target shape")
    return b.node("Expand", [ins[0], b.int64(_as_int_list(shape))])[0]


def _expand_as(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``expand_as`` — ``Expand`` to the other operand's shape."""
    return b.node(
        "Expand",
        [ins[0], b.node("Shape", [ins[1]], dtype=_INT64)[0]],
    )[0]


def _repeat(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``repeat``/``tile`` — ``Tile`` with the repeat counts.

    ATen's ``repeat`` left-pads a short count list with ones (a
    trailing-axis repeat); ONNX ``Tile`` wants one count per axis.
    """
    shape = b.attr(term, "shape", None)
    if shape is None:
        raise UnsupportedOnnxOpError(term.op, "no ``shape`` attr")
    rank = b.required_rank(term)
    counts = _as_int_list(shape)
    if len(counts) > rank:
        raise UnsupportedOnnxOpError(
            term.op, f"{len(counts)} repeat counts for rank {rank}"
        )
    return b.node(
        "Tile", [ins[0], b.int64([1] * (rank - len(counts)) + counts)]
    )[0]


def _slice(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``slice`` — one-axis ``Slice`` with open bounds allowed."""
    dim = int(b.attr(term, "dim", 0))
    start = b.attr(term, "start", None)
    end = b.attr(term, "end", None)
    step = b.attr(term, "step", None)
    return b.node(
        "Slice",
        [
            ins[0],
            b.int64([0 if start is None else int(start)]),
            b.int64([_INT64_MAX if end is None else int(end)]),
            b.int64([dim]),
            b.int64([1 if step is None else int(step)]),
        ],
    )[0]


def _narrow(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``narrow(t, dim, start, length)`` — ``Slice``."""
    dim = int(b.attr(term, "dim", 0))
    start = int(b.attr(term, "start", 0))
    length = int(b.attr(term, "length", 1))
    return b.node(
        "Slice",
        [
            ins[0],
            b.int64([start]),
            b.int64([start + length]),
            b.int64([dim]),
        ],
    )[0]


def _select(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``select(t, dim, index)`` — ``Gather`` then ``Squeeze``."""
    dim = int(b.attr(term, "dim", 0))
    index = int(b.attr(term, "index", 0))
    gathered = b.node("Gather", [ins[0], b.int64([index])], axis=dim)[0]
    return b.node("Squeeze", [gathered, b.int64([dim])])[0]


def _flip(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``flip`` — ``Slice`` with a -1 step on each axis."""
    dims = b.attr(term, "dim", None)
    if dims is None:
        dims = b.attr(term, "dims", None)
    if not dims:
        raise UnsupportedOnnxOpError(term.op, "no axes to flip")
    rank = b.required_rank(term)
    axes = [_norm_dim(d, rank, term.op) for d in _as_int_list(dims)]
    n = len(axes)
    return b.node(
        "Slice",
        [
            ins[0],
            b.int64([-1] * n),
            b.int64([_INT64_MIN] * n),
            b.int64(axes),
            b.int64([-1] * n),
        ],
    )[0]


def _pad(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``pad`` — constant-mode ``Pad`` with ATen's reversed widths."""
    mode = b.attr(term, "mode", "constant")
    if mode != "constant":
        raise UnsupportedOnnxOpError(
            term.op, f"mode={mode!r} is not constant padding"
        )
    widths = b.attr(term, "pad", None)
    if widths is None:
        raise UnsupportedOnnxOpError(term.op, "no ``pad`` attr")
    rank = b.required_rank(term)
    widths = _as_int_list(widths)
    if len(widths) % 2:
        raise UnsupportedOnnxOpError(
            term.op, "pad widths must come in (before, after) pairs"
        )
    begins = [0] * rank
    ends = [0] * rank
    for i in range(len(widths) // 2):
        axis = rank - 1 - i
        if axis < 0:
            raise UnsupportedOnnxOpError(
                term.op, "more padding pairs than axes"
            )
        begins[axis] = widths[2 * i]
        ends[axis] = widths[2 * i + 1]
    pads = b.int64(begins + ends)
    value = b.attr(term, "value", None)
    if value is None:
        return b.node("Pad", [ins[0], pads])[0]
    return b.node(
        "Pad", [ins[0], pads, b.constant(value, b.dtype_of(ins[0]))]
    )[0]


# -- product structure -------------------------------------------------


def _concat(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``concat`` — ONNX ``Concat`` along the named axis."""
    return b.node("Concat", ins, axis=int(b.attr(term, "dim", 0)))[0]


def _stack(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``stack`` — ``Unsqueeze`` every operand then ``Concat``."""
    axis = int(b.attr(term, "dim", 0))
    lifted = [b.node("Unsqueeze", [x, b.int64([axis])])[0] for x in ins]
    return b.node("Concat", lifted, axis=axis)[0]


def _split(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``split`` — ``Split``, selecting the ``index``-th part."""
    sizes = b.attr(term, "sizes", ())
    dim = int(b.attr(term, "dim", -1))
    index = int(b.attr(term, "index", 0))
    parts = _split_parts(b, term, sizes, dim)
    outs = b.node(
        "Split",
        [ins[0], b.int64(parts)],
        n_outputs=len(parts),
        axis=dim,
    )
    return outs[_norm_index(index, len(parts), term.op)]


def _split_parts(
    b: GraphBuilder, term: Op, sizes: Any, dim: int
) -> list[int]:
    """Resolve a ``split`` size spelling to explicit part sizes."""
    if isinstance(sizes, int):
        rank = b.required_rank(term)
        shape = b.operand_shape(term)
        extent = shape[_norm_dim(dim, rank, term.op)] if shape else None
        if extent is None:
            raise UnsupportedOnnxOpError(
                term.op,
                "an int split size needs a statically known extent",
            )
        parts: list[int] = []
        remaining = extent
        while remaining > 0:
            parts.append(min(sizes, remaining))
            remaining -= sizes
        return parts
    parts = _as_int_list(sizes)
    if not parts or any(p <= 0 for p in parts):
        raise UnsupportedOnnxOpError(
            term.op, f"sizes={sizes!r} is not a positive part list"
        )
    return parts


def _chunk(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``chunk`` — ATen's ceil-sized parts, then select by index."""
    chunks = int(b.attr(term, "chunks", 2))
    dim = int(b.attr(term, "dim", -1))
    index = int(b.attr(term, "index", 0))
    rank = b.required_rank(term)
    shape = b.operand_shape(term)
    extent = shape[_norm_dim(dim, rank, term.op)] if shape else None
    if extent is None or chunks <= 0:
        raise UnsupportedOnnxOpError(
            term.op,
            "chunk needs a statically known extent and a positive "
            "chunk count",
        )
    size = -(-extent // chunks)
    parts: list[int] = []
    remaining = extent
    while remaining > 0:
        parts.append(min(size, remaining))
        remaining -= size
    outs = b.node(
        "Split",
        [ins[0], b.int64(parts)],
        n_outputs=len(parts),
        axis=dim,
    )
    return outs[_norm_index(index, len(parts), term.op)]


def _hstack(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``hstack`` — ``Concat`` along axis 1, or axis 0 for vectors."""
    rank = b.required_rank(term)
    return b.node("Concat", ins, axis=0 if rank <= 1 else 1)[0]


def _getitem(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``getitem`` — pick an extra output of a multi-output op.

    Only that reading is mapped: ``getitem`` over a single-output
    term is general tensor indexing (``t[0]`` takes a *slice*, not
    the whole value), and silently treating it as identity would be
    the drop the ADR forbids.
    """
    names = b.multi_outputs(term.args[0])
    if names is None:
        raise UnsupportedOnnxOpError(
            term.op,
            "getitem is mapped only onto the extra outputs of a "
            "multi-output op (max/min); general indexing is declined",
        )
    return names[
        _norm_index(int(b.attr(term, "index", 0)), len(names), term.op)
    ]


# -- reductions --------------------------------------------------------


def _extreme_values(
    op_type: str,
    index_op: str,
) -> OnnxBuilder:
    """Build ``max``/``min`` — values plus indices when ``dim`` is set."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit the reduction and, with a dim, its index sibling."""
        dim = b.attr(term, "dim", None)
        keepdim = int(bool(b.attr(term, "keepdim", False)))
        if dim is None:
            return b.node(op_type, ins, keepdims=0)[0]
        axis = _norm_dim(dim, b.required_rank(term), term.op)
        values = b.node(
            op_type, [ins[0], b.int64([axis])], keepdims=keepdim
        )[0]
        indices = b.node(
            index_op, ins, axis=axis, keepdims=keepdim, dtype=_INT64
        )[0]
        b.declare_multi(term, [values, indices])
        return values

    return build


def _arg_reducer(op_type: str) -> OnnxBuilder:
    """Build ``argmax``/``argmin``, flattening when ``dim`` is absent."""

    def build(b: GraphBuilder, term_: Op, ins: list[str]) -> str:
        """Emit the arg reduction."""
        dim = b.attr(term_, "dim", None)
        if dim is None:
            flat = b.node("Reshape", [ins[0], b.int64([-1])])[0]
            return b.node(
                op_type, [flat], axis=0, keepdims=0, dtype=_INT64
            )[0]
        keepdim = int(bool(b.attr(term_, "keepdim", False)))
        return b.node(
            op_type,
            ins,
            axis=int(dim),
            keepdims=keepdim,
            dtype=_INT64,
        )[0]

    return build


def _cumulative(op_type: str) -> OnnxBuilder:
    """Build ``cumsum``/``cumprod`` — the axis is a scalar input."""

    def build(b: GraphBuilder, term_: Op, ins: list[str]) -> str:
        """Emit the scan along the named axis."""
        axis = int(b.attr(term_, "dim", 0))
        return b.node(op_type, [ins[0], b.int64_scalar(axis)])[0]

    return build


def _vector_norm(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``linalg_vector_norm`` — ``ReduceL1``/``ReduceL2`` for 1 and 2."""
    order = b.attr(term, "ord", 2)
    if order not in (1, 2):
        raise UnsupportedOnnxOpError(
            term.op, f"ord={order!r} is not 1 or 2"
        )
    op_type = "ReduceL1" if order == 1 else "ReduceL2"
    dim = b.attr(term, "dim", None)
    keepdim = int(bool(b.attr(term, "keepdim", False)))
    if dim is None:
        return b.node(op_type, ins, keepdims=keepdim)[0]
    return b.node(
        op_type, [ins[0], b.int64(_as_int_list(dim))], keepdims=keepdim
    )[0]


def _diag_sum(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``diag_sum`` — ``Einsum`` with ``ii->``."""
    return b.node("Einsum", ins, equation="ii->")[0]


def _count_nonzero(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``count_nonzero`` — sum the predicate, cast back to int64."""
    dtype = b.dtype_of(ins[0])
    zero = b.constant(0, dtype)
    nonzero = b.node(
        "Not", b.node("Equal", [ins[0], zero], dtype=_BOOL)
    )[0]
    counted = b.node(
        "Cast", [nonzero], to=_onnx_elem_type(_INT64), dtype=_INT64
    )[0]
    dim = b.attr(term, "dim", None)
    if dim is None:
        reduced = b.node("ReduceSum", [counted], keepdims=0)[0]
    else:
        reduced = b.node(
            "ReduceSum",
            [counted, b.int64(_as_int_list(dim))],
            keepdims=0,
        )[0]
    return b.node(
        "Cast", [reduced], to=_onnx_elem_type(_INT64), dtype=_INT64
    )[0]


def _logical_reduce(op_type: str) -> OnnxBuilder:
    """Build ``any``/``all`` — an int cast, reduce, cast back."""

    def build(b: GraphBuilder, term_: Op, ins: list[str]) -> str:
        """Emit the boolean reduction."""
        as_int = b.node(
            "Cast", [ins[0]], to=_onnx_elem_type(_INT64), dtype=_INT64
        )[0]
        dim = b.attr(term_, "dim", None)
        keepdim = int(bool(b.attr(term_, "keepdim", False)))
        if dim is None:
            reduced = b.node(op_type, [as_int], keepdims=keepdim)[0]
        else:
            reduced = b.node(
                op_type,
                [as_int, b.int64(_as_int_list(dim))],
                keepdims=keepdim,
            )[0]
        return b.node(
            "Cast", [reduced], to=_onnx_elem_type(_BOOL), dtype=_BOOL
        )[0]

    return build


def _onnx_elem_type(dtype: np.dtype) -> int:
    """Map a NumPy dtype to its ONNX element type."""
    return numpy_to_onnx_type(np.empty(0, dtype=dtype))


# -- normalization -----------------------------------------------------


def _layer_norm(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``layer_norm`` — ONNX ``LayerNormalization`` (opset 17)."""
    if len(ins) < 2:
        raise UnsupportedOnnxOpError(
            term.op, "ONNX LayerNormalization requires a scale"
        )
    shape = b.attr(term, "dim", None)
    if shape is None:
        raise UnsupportedOnnxOpError(term.op, "no normalized_shape")
    rank = b.required_rank(term)
    axis = rank - len(_as_int_list(shape))
    if axis < 0:
        raise UnsupportedOnnxOpError(
            term.op, "normalized_shape is longer than the rank"
        )
    return b.node(
        "LayerNormalization",
        ins[:3],
        axis=axis,
        epsilon=float(b.attr(term, "eps", 1e-5)),
    )[0]


def _rms_norm(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``rms_norm`` — composed (ONNX RMSNormalization needs opset 23).

    ATen's default ``eps`` is ``finfo(dtype).eps``, not a constant,
    so the sink reads the same rule off its own working dtype when
    the term carries no ``eps``.
    """
    rank = b.required_rank(term)
    shape = b.attr(term, "dim", None)
    width = len(_as_int_list(shape)) if shape is not None else rank
    axes = list(range(rank - width, rank))
    x = ins[0]
    eps = b.attr(term, "eps", None)
    if eps is None:
        eps = float(np.finfo(b.dtype_of(x)).eps)
    squared = b.node("Mul", [x, x])[0]
    mean = b.node("ReduceMean", [squared, b.int64(axes)], keepdims=1)[0]
    denom = b.node(
        "Sqrt",
        [b.node("Add", [mean, b.constant(eps, b.dtype_of(x))])[0]],
    )[0]
    normalized = b.node("Div", [x, denom])[0]
    if len(ins) > 1:
        return b.node("Mul", [normalized, ins[1]])[0]
    return normalized


def _batch_norm(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``batch_norm`` — inference-mode ``BatchNormalization``."""
    if bool(b.attr(term, "training", False)):
        raise UnsupportedOnnxOpError(
            term.op,
            "training-mode batch_norm updates running statistics — "
            "a graph state change, not a function",
        )
    if len(ins) < 5:
        raise UnsupportedOnnxOpError(
            term.op, "batch_norm needs weight, bias and running stats"
        )
    return b.node(
        "BatchNormalization",
        ins[:5],
        epsilon=float(b.attr(term, "eps", 1e-5)),
    )[0]


def _instance_norm(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``instance_norm`` — ``InstanceNormalization`` from input stats."""
    if not bool(b.attr(term, "use_input_stats", True)):
        raise UnsupportedOnnxOpError(
            term.op,
            "use_input_stats=False normalizes with running stats — "
            "ONNX InstanceNormalization has none",
        )
    if len(ins) < 3:
        raise UnsupportedOnnxOpError(
            term.op, "instance_norm needs scale and bias"
        )
    return b.node(
        "InstanceNormalization",
        ins[:3],
        epsilon=float(b.attr(term, "eps", 1e-5)),
    )[0]


def _spatial(value: Any, count: int, op: str, name: str) -> list[int]:
    """Normalise a conv window argument to one entry per spatial dim.

    ONNX ``Conv`` requires ``strides``/``pads``/``dilations`` to have
    exactly one value per spatial axis; ATen accepts a bare int.
    """
    values = (
        [int(value)] * count
        if isinstance(value, int)
        else _as_int_list(value)
    )
    if len(values) != count:
        raise UnsupportedOnnxOpError(
            op,
            f"{name}={value!r} has {len(values)} entries for "
            f"{count} spatial dimensions",
        )
    return values


def _conv(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``conv1d``/``conv2d`` — ONNX ``Conv``."""
    if len(ins) < 2:
        raise UnsupportedOnnxOpError(term.op, "conv needs a weight")
    weight_shape = b.operand_shape(term, 1)
    if weight_shape is None or len(weight_shape) < 3:
        raise UnsupportedOnnxOpError(
            term.op, "the conv weight's rank must be statically known"
        )
    spatial = len(weight_shape) - 2
    strides = _spatial(
        b.attr(term, "stride", 1), spatial, term.op, "stride"
    )
    dilations = _spatial(
        b.attr(term, "dilation", 1), spatial, term.op, "dilation"
    )
    pads = _spatial(
        b.attr(term, "padding", 0), spatial, term.op, "padding"
    )
    return b.node(
        "Conv",
        ins[:3],
        strides=strides,
        dilations=dilations,
        pads=pads + pads,
        group=int(b.attr(term, "groups", 1)),
    )[0]


# -- indexing / gathering ----------------------------------------------


def _gather(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``index_select`` / ``embedding`` — ``Gather`` on one axis."""
    dim = int(b.attr(term, "dim", 0))
    if len(ins) > 1:
        return b.node("Gather", [ins[0], ins[1]], axis=dim)[0]
    indices = b.attr(term, "index", None)
    if indices is None:
        raise UnsupportedOnnxOpError(term.op, "no index operand")
    return b.node(
        "Gather", [ins[0], b.int64(_as_int_list(indices))], axis=dim
    )[0]


def _embedding(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``embedding(W, idx)`` — ``Gather`` along axis 0."""
    return b.node("Gather", ins, axis=0)[0]


def _gather_elements(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``gather``/``take_along_dim`` — ONNX ``GatherElements``."""
    return b.node(
        "GatherElements", ins, axis=int(b.attr(term, "dim", 0))
    )[0]


def _one_hot(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``one_hot`` — ``OneHot`` with an int64 ``[0, 1]`` value pair."""
    depth = b.attr(term, "num_classes", -1)
    if not isinstance(depth, int) or depth <= 0:
        raise UnsupportedOnnxOpError(
            term.op, f"num_classes={depth!r} is not a positive int"
        )
    values = b.initializer(np.asarray([0, 1], dtype=np.int64))
    return b.node(
        "OneHot", [ins[0], b.int64_scalar(depth), values], axis=-1
    )[0]


def _trilu(upper: bool) -> OnnxBuilder:
    """Build ``triu``/``tril`` — ONNX ``Trilu``."""

    def build(b: GraphBuilder, term_: Op, ins: list[str]) -> str:
        """Emit the triangular mask."""
        k = int(b.attr(term_, "diagonal", 0))
        return b.node(
            "Trilu",
            [ins[0], b.int64_scalar(k)],
            upper=int(upper),
        )[0]

    return build


def _arange(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``arange`` — ``Range``; the bounds arrive as ``argN`` attrs.

    ATen's ``arange`` is variadic and *positional*: one bound is the
    end, two are start and end, three add the step.  The IR keeps the
    call's argument order, so the arity decides the reading.
    """
    bounds = [
        term.attrs[f"arg{i}"]
        for i in range(len(term.attrs))
        if f"arg{i}" in term.attrs
    ]
    if not bounds:
        raise UnsupportedOnnxOpError(
            term.op, "no bounds — a dynamic arange is not a constant"
        )
    if len(bounds) > 3:
        raise UnsupportedOnnxOpError(
            term.op, f"arange takes at most 3 bounds, got {len(bounds)}"
        )
    start, end, step = 0, bounds[0], 1
    if len(bounds) >= 2:
        start, end = bounds[0], bounds[1]
    if len(bounds) == 3:
        step = bounds[2]
    name = b.attr(term, "dtype", None)
    if name in DTYPE_NAMES:
        dtype = DTYPE_NAMES[name]
    elif all(isinstance(v, int) for v in (start, end, step)):
        dtype = _INT64
    else:
        dtype = b.work
    return b.node(
        "Range",
        [
            b.constant(start, dtype),
            b.constant(end, dtype),
            b.constant(step, dtype),
        ],
    )[0]


def _to(b: GraphBuilder, term: Op, ins: list[str]) -> str:
    """``to(dtype=...)`` — ``Cast`` to the recorded dtype name."""
    name = b.attr(term, "dtype", None)
    if name not in DTYPE_NAMES:
        raise UnsupportedOnnxOpError(
            term.op,
            f"dtype={name!r} is not a recorded dtype name the sink "
            f"can cast to",
        )
    target = DTYPE_NAMES[name]
    return b.node(
        "Cast", ins, to=_onnx_elem_type(target), dtype=target
    )[0]


def _named_cast(op_name: str) -> OnnxBuilder:
    """Build a named cast (``float``/``double``/``long`` …)."""
    target = DTYPE_NAMES[_CASTS[op_name]]

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit the cast node."""
        return b.node(
            "Cast", ins, to=_onnx_elem_type(target), dtype=target
        )[0]

    return build


def _softmax(op_type: str) -> OnnxBuilder:
    """Build ``softmax``/``log_softmax`` over their ``dim`` attr."""

    def build(b: GraphBuilder, term: Op, ins: list[str]) -> str:
        """Emit the axis-wise normalization."""
        axis = int(b.attr(term, "dim", -1))
        return b.node(op_type, ins, axis=axis)[0]

    return build


# ---------------------------------------------------------------------------
#  the table
# ---------------------------------------------------------------------------

#: The complete IR -> ONNX mapping.  Keys are IR op names; the sink's
#: ``supported_ops`` is exactly this key set.
ONNX_BINDINGS: dict[str, OnnxBuilder] = {
    **{name: _simple(op) for name, op in _UNARY.items()},
    **{name: _boolean(op) for name, op in _UNARY_BOOL.items()},
    **{name: _simple(op) for name, op in _BINARY.items()},
    **{name: _boolean(op) for name, op in _COMPARISON.items()},
    **{name: _reducer(op) for name, op in _REDUCERS.items()},
    **{name: _named_cast(name) for name in _CASTS},
    "pack": _pack,
    "matmul": _matmul,
    "linear": _linear,
    "dropout": _dropout,
    "fmod": _mod(True),
    "remainder": _remainder,
    "zeros_like": _fill_like(0),
    "ones_like": _fill_like(1),
    "full_like": _full_like,
    "zeros": _shaped_fill(0, False),
    "ones": _shaped_fill(1, False),
    "full": _shaped_fill(0, True),
    "new_zeros": _new_fill(0, False),
    "new_ones": _new_fill(1, False),
    "new_full": _new_fill(0, True),
    "nonzero": _nonzero,
    "rsub": _rsub,
    "div": _div,
    "trunc": _trunc,
    "frac": _frac,
    "rsqrt": _rsqrt,
    "square": _square,
    "exp2": _exp2,
    "expm1": _expm1,
    "log1p": _log1p,
    "log2": _log_base(2.0),
    "log10": _log_base(10.0),
    "asinh": _asinh,
    "acosh": _acosh,
    "atanh": _atanh,
    "erfc": _erfc,
    "ne": _ne,
    "isfinite": _isfinite,
    "heaviside": _heaviside,
    "xlogy": _xlogy,
    "lerp": _lerp,
    "addcmul": _addcmul,
    "addcdiv": _addcdiv,
    "masked_fill": _masked_fill,
    "where": _where,
    "clamp": _clamp,
    "clamp_min": _clamp,
    "clamp_max": _clamp,
    "hardtanh": _hardtanh,
    "relu6": _relu6,
    "elu": _elu,
    "leaky_relu": _leaky_relu,
    "softplus": _softplus,
    "gelu": _gelu,
    "silu": _silu,
    "mish": _mish,
    "hardsigmoid": _hardsigmoid,
    "hardswish": _hardswish,
    "glu": _glu,
    "prelu": _prelu,
    "softmax": _softmax("Softmax"),
    "log_softmax": _softmax("LogSoftmax"),
    "transpose": _transpose,
    "permute": _permute,
    "movedim": _movedim,
    "reshape": _reshape,
    "flatten": _flatten,
    "unsqueeze": _unsqueeze,
    "squeeze": _squeeze,
    "expand": _expand,
    "expand_as": _expand_as,
    "broadcast_to": _expand,
    "repeat": _repeat,
    "slice": _slice,
    "narrow": _narrow,
    "select": _select,
    "flip": _flip,
    "pad": _pad,
    "concat": _concat,
    "stack": _stack,
    "hstack": _hstack,
    "split": _split,
    "chunk": _chunk,
    "getitem": _getitem,
    "max": _extreme_values("ReduceMax", "ArgMax"),
    "min": _extreme_values("ReduceMin", "ArgMin"),
    "argmax": _arg_reducer("ArgMax"),
    "argmin": _arg_reducer("ArgMin"),
    "cumsum": _cumulative("CumSum"),
    "linalg_vector_norm": _vector_norm,
    "diag_sum": _diag_sum,
    "count_nonzero": _count_nonzero,
    "any": _logical_reduce("ReduceMax"),
    "all": _logical_reduce("ReduceMin"),
    "layer_norm": _layer_norm,
    "rms_norm": _rms_norm,
    "batch_norm": _batch_norm,
    "instance_norm": _instance_norm,
    "conv1d": _conv,
    "conv2d": _conv,
    "index_select": _gather,
    "embedding": _embedding,
    "gather": _gather_elements,
    "take_along_dim": _gather_elements,
    "one_hot": _one_hot,
    "triu": _trilu(True),
    "tril": _trilu(False),
    "arange": _arange,
    "to": _to,
}
