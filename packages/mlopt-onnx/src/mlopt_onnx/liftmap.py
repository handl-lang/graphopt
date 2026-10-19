"""The ONNX -> IR op lifting — ADR 0006, direction 2.

One entry per ONNX ``op_type``; each value is a *lifter*,
``(lifter, node) -> {output_name: term}``, which mints the IR term
implementing that node and binds its graph outputs.  The table is
the inverse vocabulary of :mod:`mlopt_onnx.opmap` — where the sink
spells an IR op *as* ONNX nodes, the lifter reads those nodes back
into the op, using the same canonical attr spellings
(``mlopt_core.attrs.ATTR_SCHEMA``) every core reader consumes.

The mapping is deliberately a **coherent subset, not the whole op
set**.  It covers the ops that invert cleanly into the IR's op
vocabulary: elementwise arithmetic and transcendental functions,
``MatMul``/``Gemm``/``Conv``, the product structure (``Concat``/
``Split``/``TopK``), the layout ops (``Transpose``/``Reshape``/
``Unsqueeze``/``Squeeze``/``Flatten``/``Pad``/``Slice``/``Expand``/
``Tile``), the reductions, ``Softmax``/``LogSoftmax``, the
activations, the normalizations, and the dtype casts.  Everything
else is *refused* — an op with no entry raises
:class:`~mlopt_onnx.errors.UnsupportedOnnxOpError`, and so does a
mapped op in a spelling the IR cannot express.

Notable declines, each for a stated reason rather than by omission:

* ``If`` / ``Loop`` / ``Scan`` — control flow: an IR term is a
  morphism, not a program with data-dependent control.  Their graph
  attributes are unmappable by construction.
* ``Sequence*`` / ``Optional*`` / ``Map*`` — non-tensor values; the
  IR's product structure carries tensors only.
* ``Shape`` / ``Size`` — metadata extraction; there is no IR op for
  "the shape of".
* ``RandomUniform`` / ``RandomNormal`` / ``Multinomial`` /
  ``Bernoulli`` — randomness is not a function; a source that
  re-randomises cannot verify.
* ``QuantizeLinear`` / ``DequantizeLinear`` / ``QLinear*`` /
  ``MatMulInteger`` — quantization carries scale/zero-point state
  the IR ops do not name.
* ``Scatter*`` / ``GatherND`` — the same duplicate-index semantics
  trap the sink declined.
* ``Resize`` / ``Upsample`` / ``GridSample`` / ``RoiAlign`` /
  ``NonMaxSuppression`` — the coordinate-convention tail ADR 0006
  measured on the torch side.
* ``RNN`` / ``LSTM`` / ``GRU`` — sequential state, not a morphism.
* ``Attention`` / fused ops from opsets past 18 — the pinned table
  does not claim them.
* ``AveragePool`` (``count_include_pad`` semantics), ``LpPool``,
  ``ConvTranspose``, ``Hardmax``, ``Det``, ``DFT``/window functions,
  ``ai.onnx.ml`` featurizers — no IR op with matching semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
from mlopt_core.ir import Const

from mlopt_onnx.arrays import DTYPE_NAMES, ONNX_TYPE_NAMES

if TYPE_CHECKING:
    from mlopt_onnx.lift import GraphLifter

__all__ = ["ONNX_LIFTS", "OnnxLifter"]

#: An op lifter: ``(lifter, node) -> {output_name: term}``.
OnnxLifter = Callable[["GraphLifter", "onnx.NodeProto"], dict[str, Any]]

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)


# ---------------------------------------------------------------------------
#  small shared helpers
# ---------------------------------------------------------------------------


def _outs(node: onnx.NodeProto) -> list[str]:
    """Return ``node``'s non-empty output names."""
    return [o for o in node.output if o]


def _one(g: GraphLifter, node: onnx.NodeProto, term: Any) -> dict:
    """Bind a single-output term to ``node``'s output."""
    outs = _outs(node)
    if len(outs) != 1:
        g.refuse(
            node,
            f"expected 1 output, got {len(node.output)} "
            f"({list(node.output)})",
        )
    return {outs[0]: term}


def _axes(g: GraphLifter, node: onnx.NodeProto, i: int = 1) -> Any:
    """Read an axes list — input at ``i``, then the ``axes`` attr.

    Opset 18 moved the reduction ops' axes from an attribute to an
    input; both spellings appear inside the supported window.
    ``None`` means "reduce every axis".
    """
    if i < len(node.input) and node.input[i]:
        arr = g.const(node, i)
        return tuple(int(v) for v in arr.reshape(-1))
    value = g.attr(node, "axes", None)
    if value is None:
        return None
    return tuple(int(v) for v in value)


def _ints(g: GraphLifter, node: onnx.NodeProto, i: int) -> list[int]:
    """Read a constant input as a flat list of ints."""
    return [int(v) for v in g.const(node, i).reshape(-1)]


def _simple(ir_op: str) -> OnnxLifter:
    """Build a lifter for an op carried verbatim by its operands."""

    def lift(g: GraphLifter, node: onnx.NodeProto) -> dict:
        """Mint ``ir_op`` over the node's operands."""
        return _one(g, node, g.mint(ir_op, *g.args(node)))

    return lift


def _fold(ir_op: str) -> OnnxLifter:
    """Build a lifter folding a variadic elementwise op left."""

    def lift(g: GraphLifter, node: onnx.NodeProto) -> dict:
        """Fold ``ir_op`` over the node's operands, left to right."""
        args = g.args(node)
        if len(args) < 2:
            g.refuse(node, f"{node.op_type} needs >= 2 operands")
        acc = args[0]
        for operand in args[1:]:
            acc = g.mint(ir_op, acc, operand)
        return _one(g, node, acc)

    return lift


def _reducer(ir_op: str) -> OnnxLifter:
    """Build a lifter for a reduce op (axes + keepdims)."""

    def lift(g: GraphLifter, node: onnx.NodeProto) -> dict:
        """Mint the reduction with its folded ``dim``/``keepdim``."""
        axes = _axes(g, node)
        if axes is None and int(
            g.attr(node, "noop_with_empty_axes", 0)
        ):
            # An empty axes input with noop semantics is the
            # identity — clone spells it honestly.
            return _one(g, node, g.mint("clone", g.term(node, 0)))
        attrs: dict[str, Any] = {
            "keepdim": bool(g.attr(node, "keepdims", 1))
        }
        if axes is not None:
            attrs["dim"] = axes
        return _one(g, node, g.mint(ir_op, g.term(node, 0), **attrs))

    return lift


def _vector_norm(order: int) -> OnnxLifter:
    """Build a lifter for ``ReduceL1``/``ReduceL2``."""

    def lift(g: GraphLifter, node: onnx.NodeProto) -> dict:
        """Mint ``linalg_vector_norm`` with the fixed ``ord``."""
        axes = _axes(g, node)
        attrs: dict[str, Any] = {
            "ord": order,
            "keepdim": bool(g.attr(node, "keepdims", 1)),
        }
        if axes is not None:
            attrs["dim"] = axes
        return _one(
            g,
            node,
            g.mint("linalg_vector_norm", g.term(node, 0), **attrs),
        )

    return lift


def _arg_reducer(ir_op: str) -> OnnxLifter:
    """Build a lifter for ``ArgMax``/``ArgMin``."""

    def lift(g: GraphLifter, node: onnx.NodeProto) -> dict:
        """Mint the arg-reduction along its ``axis`` attr."""
        if int(g.attr(node, "select_last_index", 0)):
            g.refuse(
                node,
                "select_last_index=1 picks the LAST extremum; ATen's "
                "arg reductions pick the first — different answers "
                "on ties",
            )
        axis = int(g.attr(node, "axis", 0))
        keepdim = bool(g.attr(node, "keepdims", 1))
        return _one(
            g,
            node,
            g.mint(
                ir_op,
                g.term(node, 0),
                dim=axis,
                keepdim=keepdim,
            ),
        )

    return lift


def _const_of(value: Any) -> Const:
    """Mint a ``Const`` leaf (int spellings stay ints)."""
    return Const(value)


# ---------------------------------------------------------------------------
#  elementwise / scalar-shaped
# ---------------------------------------------------------------------------


def _identity(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Identity`` — the memory op ``clone`` (identity semantics)."""
    return _one(g, node, g.mint("clone", g.term(node, 0)))


def _mod(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Mod`` — ``fmod`` when fmod=1, floor-mod ``remainder`` at 0."""
    fmod = int(g.attr(node, "fmod", 0))
    return _one(
        g, node, g.mint("fmod" if fmod else "remainder", *g.args(node))
    )


def _mean(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Mean`` — the variadic mean, folded as ``sum / n``."""
    args = g.args(node)
    if len(args) < 2:
        g.refuse(node, "Mean needs >= 2 operands")
    acc = args[0]
    for operand in args[1:]:
        acc = g.mint("add", acc, operand)
    return _one(
        g, node, g.mint("div", acc, _const_of(float(len(args))))
    )


def _clip(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Clip`` — ``clamp`` family, bounds folded from const inputs."""
    x = g.term(node, 0)
    lo = (
        g.const_scalar(node, 1)
        if len(node.input) > 1 and node.input[1]
        else None
    )
    hi = (
        g.const_scalar(node, 2)
        if len(node.input) > 2 and node.input[2]
        else None
    )
    if lo is not None and hi is not None:
        return _one(g, node, g.mint("clamp", x, min=lo, max=hi))
    if lo is not None:
        return _one(g, node, g.mint("clamp_min", x, min=lo))
    if hi is not None:
        return _one(g, node, g.mint("clamp_max", x, max=hi))
    # A boundless Clip is the identity — spell it with clone.
    return _one(g, node, g.mint("clone", x))


def _thresholded_relu(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``ThresholdedRelu`` — ``where(x > alpha, x, 0)`` spelled out."""
    alpha = float(g.attr(node, "alpha", 1.0))
    x = g.term(node, 0)
    return _one(
        g,
        node,
        g.mint(
            "where",
            g.mint("gt", x, _const_of(alpha)),
            x,
            _const_of(0.0),
        ),
    )


def _leaky_relu(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``LeakyRelu`` — ``negative_slope`` is ONNX's ``alpha``."""
    alpha = float(g.attr(node, "alpha", 0.01))
    return _one(
        g,
        node,
        g.mint("leaky_relu", g.term(node, 0), negative_slope=alpha),
    )


def _elu(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Elu`` — ONNX carries only the alpha parameter."""
    alpha = float(g.attr(node, "alpha", 1.0))
    return _one(g, node, g.mint("elu", g.term(node, 0), alpha=alpha))


def _celu(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Celu`` — ONNX carries only the alpha parameter."""
    alpha = float(g.attr(node, "alpha", 1.0))
    return _one(g, node, g.mint("celu", g.term(node, 0), alpha=alpha))


_SELU_ALPHA = 1.6732631921768188
_SELU_SCALE = 1.0507010221481323


def _selu(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Selu`` — ``selu`` at ATen's constants, else the exact composition.

    ONNX ``Selu`` parameterises what ATen fixes; a non-default
    spelling is ``gamma * elu(x, alpha)``, minted as such rather
    than silently re-defaulted (the HardSigmoid trap's twin).
    """
    alpha = float(g.attr(node, "alpha", _SELU_ALPHA))
    gamma = float(g.attr(node, "gamma", _SELU_SCALE))
    x = g.term(node, 0)
    if (
        abs(alpha - _SELU_ALPHA) < 1e-5
        and abs(gamma - _SELU_SCALE) < 1e-5
    ):
        return _one(g, node, g.mint("selu", x))
    return _one(
        g,
        node,
        g.mint("mul", g.mint("elu", x, alpha=alpha), _const_of(gamma)),
    )


def _hard_sigmoid(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``HardSigmoid`` — ``hardsigmoid`` at ATen's params, else exact.

    The ADR's measured trap, read the other way: ONNX's *default*
    ``alpha=0.2`` is NOT ATen's ``1/6``.  At the ATen spelling the
    fused op lifts; every other parameterisation mints the exact
    ``clamp(alpha*x + beta, 0, 1)`` composition.
    """
    alpha = float(g.attr(node, "alpha", 0.2))
    beta = float(g.attr(node, "beta", 0.5))
    x = g.term(node, 0)
    if abs(alpha - 1.0 / 6.0) < 1e-6 and abs(beta - 0.5) < 1e-9:
        return _one(g, node, g.mint("hardsigmoid", x))
    return _one(
        g,
        node,
        g.mint(
            "clamp",
            g.mint(
                "add",
                g.mint("mul", x, _const_of(alpha)),
                _const_of(beta),
            ),
            min=0.0,
            max=1.0,
        ),
    )


def _gelu(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Gelu`` (opset 20) — ``approximate`` carries through verbatim."""
    approximate = g.attr(node, "approximate", None)
    attrs = {} if approximate is None else {"approximate": approximate}
    return _one(g, node, g.mint("gelu", g.term(node, 0), **attrs))


def _softmax(ir_op: str) -> OnnxLifter:
    """Build a lifter for ``Softmax``/``LogSoftmax`` (axis -> dim)."""

    def lift(g: GraphLifter, node: onnx.NodeProto) -> dict:
        """Mint the axis-wise normalization."""
        axis = int(g.attr(node, "axis", -1))
        return _one(g, node, g.mint(ir_op, g.term(node, 0), dim=axis))

    return lift


def _isinf(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``IsInf`` — the detect flags select the predicate."""
    pos = bool(g.attr(node, "detect_positive", 1))
    neg = bool(g.attr(node, "detect_negative", 1))
    x = g.term(node, 0)
    if pos and neg:
        return _one(g, node, g.mint("isinf", x))
    if pos:
        return _one(g, node, g.mint("isposinf", x))
    if neg:
        return _one(g, node, g.mint("isneginf", x))
    g.refuse(
        node,
        "detect_positive=0, detect_negative=0 is the "
        "constant-false predicate — degenerate, not liftable",
    )


def _cast(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Cast`` — ``to(dtype=...)`` with the recorded dtype name."""
    to = int(g.attr(node, "to", 0))
    name = ONNX_TYPE_NAMES.get(to)
    if name is None:
        g.refuse(
            node,
            f"Cast to element type {to} — no dtype name the IR "
            "records (string/bfloat16/etc. are refused)",
        )
    return _one(g, node, g.mint("to", g.term(node, 0), dtype=name))


def _cast_like(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``CastLike`` — ``to`` at the second input's declared dtype."""
    name = g.dtype_name(node.input[1])
    if name is None:
        g.refuse(
            node,
            "the cast target's dtype is not statically known — "
            "CastLike needs a typed exemplar",
        )
    return _one(g, node, g.mint("to", g.term(node, 0), dtype=name))


# ---------------------------------------------------------------------------
#  product structure / layout
# ---------------------------------------------------------------------------


def _concat(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Concat`` — variadic cat along the ``axis`` attr."""
    axis = int(g.attr(node, "axis", 0))
    return _one(g, node, g.mint("concat", *g.args(node), dim=axis))


def _transpose(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Transpose`` — a 2-swap mints ``transpose``, else ``permute``."""
    perm = g.attr(node, "perm", None)
    x = g.term(node, 0)
    if perm is None:
        return _one(g, node, g.mint("permute", x))
    perm = [int(v) for v in perm]
    moved = [i for i, p in enumerate(perm) if i != p]
    if len(moved) == 2:
        return _one(
            g,
            node,
            g.mint("transpose", x, dim0=moved[0], dim1=moved[1]),
        )
    return _one(g, node, g.mint("permute", x, dim=tuple(perm)))


def _reshape(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Reshape`` — the target shape folds from a constant input."""
    shape = _ints(g, node, 1)
    if 0 in shape:
        if int(g.attr(node, "allowzero", 0)):
            g.refuse(
                node,
                "allowzero=1: a literal 0 extent is not ATen's "
                "reshape (which reads 0 as 'copy the input dim')",
            )
        base = g.shape(node.input[0])
        resolved = []
        for i, d in enumerate(shape):
            if d != 0:
                resolved.append(d)
                continue
            if base is None or i >= len(base) or base[i] is None:
                g.refuse(
                    node,
                    "a 0 extent reads 'copy the input dim' but the "
                    f"input's dim {i} is not statically known",
                )
            resolved.append(base[i])
        shape = resolved
    return _one(
        g,
        node,
        g.mint("reshape", g.term(node, 0), shape=tuple(shape)),
    )


def _flatten(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Flatten`` — always 2-D; ``start_dim=axis, end_dim=-1``."""
    axis = int(g.attr(node, "axis", 1))
    return _one(
        g,
        node,
        g.mint("flatten", g.term(node, 0), start_dim=axis, end_dim=-1),
    )


def _unsqueeze(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Unsqueeze`` — decompose multi-axis inserts into single dims.

    ONNX axes index into the OUTPUT tensor's rank; inserting them in
    sorted order with single ``unsqueeze`` mints reconstructs the
    same result.
    """
    axes = sorted(_ints(g, node, 1))
    out_shape = g.shape(node.output[0])
    if out_shape is None:
        g.refuse(node, "the output rank is not statically known")
    n = len(out_shape)
    term = g.term(node, 0)
    for axis in axes:
        d = axis + n if axis < 0 else axis
        if not 0 <= d < n:
            g.refuse(node, f"axis {axis} is out of range for rank {n}")
        term = g.mint("unsqueeze", term, dim=d)
    return _one(g, node, term)


def _squeeze(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Squeeze`` — a const axes list, or the drop-all-ones form."""
    x = g.term(node, 0)
    if len(node.input) < 2 or not node.input[1]:
        return _one(g, node, g.mint("squeeze", x))
    axes = _ints(g, node, 1)
    base = g.shape(node.input[0])
    if base is None:
        g.refuse(node, "the input rank is not statically known")
    n = len(base)
    # Squeeze the highest axes first so lower positions stay valid.
    term = x
    for axis in sorted(
        (a + n if a < 0 else a for a in axes), reverse=True
    ):
        term = g.mint("squeeze", term, dim=axis)
    return _one(g, node, term)


def _expand(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Expand`` — the target shape folds from a constant input."""
    shape = tuple(_ints(g, node, 1))
    return _one(g, node, g.mint("expand", g.term(node, 0), shape=shape))


def _tile(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Tile`` — the repeat counts fold from a constant input."""
    counts = tuple(_ints(g, node, 1))
    return _one(
        g, node, g.mint("repeat", g.term(node, 0), shape=counts)
    )


def _pad(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Pad`` — ONNX's [begins..|ends..] becomes ATen's reversed pairs.

    Modes map onto ``F.pad``'s vocabulary: ``edge`` -> ``replicate``,
    ``wrap`` -> ``circular``.  A non-constant ``value`` is refused
    like every attr-fold position.
    """
    pads = _ints(g, node, 1)
    n = len(pads) // 2
    begins, ends = pads[:n], pads[n:]
    aten: list[int] = []
    for i in range(n):
        aten.extend((begins[n - 1 - i], ends[n - 1 - i]))
    mode = g.attr(node, "mode", "constant")
    mode = {"edge": "replicate", "wrap": "circular"}.get(mode, mode)
    attrs: dict[str, Any] = {"pad": tuple(aten), "mode": mode}
    if len(node.input) > 2 and node.input[2]:
        attrs["value"] = g.const_scalar(node, 2)
    return _one(g, node, g.mint("pad", g.term(node, 0), **attrs))


def _is_reversal(
    starts: list[int], ends: list[int], steps: list[int]
) -> bool:
    """Whether every axis slice is the whole-axis ``flip`` spelling."""
    return all(
        s == -1 and e == _INT64_MIN and st == -1
        for s, e, st in zip(starts, ends, steps, strict=True)
    )


def _slice_axes(
    g: GraphLifter,
    x: Any,
    axes: list[int],
    starts: list[int],
    ends: list[int],
    steps: list[int],
) -> Any:
    """Mint one IR ``slice`` per ONNX axis, in order."""
    term = x
    for axis, start, end, step in zip(
        axes, starts, ends, steps, strict=True
    ):
        attrs: dict[str, Any] = {
            "dim": axis,
            "start": start,
            "step": step,
        }
        # An open bound has no ATen spelling either — both sides
        # omit it (INT64_MAX is the sink's own open-end sentinel).
        if not (end == _INT64_MAX and step > 0):
            attrs["end"] = end
        term = g.mint("slice", term, **attrs)
    return term


def _slice(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Slice`` — per-axis decomposition, or ``flip`` for full reversals.

    A -1 step with ``start=-1, end=INT64_MIN`` is the whole-axis
    reversal the sink spells for ``flip`` — lifted back to ``flip``
    rather than a boundary-value slice.  Other spellings decompose
    into one IR ``slice`` per ONNX axis.
    """
    starts = _ints(g, node, 1)
    ends = _ints(g, node, 2)
    axes = (
        _ints(g, node, 3)
        if len(node.input) > 3 and node.input[3]
        else list(range(len(starts)))
    )
    steps = (
        _ints(g, node, 4)
        if len(node.input) > 4 and node.input[4]
        else [1] * len(starts)
    )
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        g.refuse(node, "starts/ends/axes/steps lengths disagree")
    x = g.term(node, 0)
    if _is_reversal(starts, ends, steps):
        return _one(g, node, g.mint("flip", x, dim=tuple(axes)))
    return _one(g, node, _slice_axes(g, x, axes, starts, ends, steps))


def _split(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Split`` — each consumed output mints a ``split`` projection.

    The IR's ``split`` is per-section (``sizes`` + ``index``); ONNX's
    multi-output node maps one projection per live output name.
    """
    x = g.term(node, 0)
    axis = int(g.attr(node, "axis", 0))
    n_out = len(_outs(node))
    if len(node.input) > 1 and node.input[1]:
        sizes: Any = tuple(_ints(g, node, 1))
    else:
        num = int(g.attr(node, "num_outputs", n_out))
        base = g.shape(node.input[0])
        extent = base[axis % len(base)] if base is not None else None
        if extent is None:
            g.refuse(
                node,
                "an equal split needs the axis extent statically "
                "known (no ``split`` sizes input)",
            )
        if extent % num:
            g.refuse(
                node,
                f"equal split of extent {extent} into {num} parts "
                "does not divide evenly",
            )
        sizes = tuple([extent // num] * num)
    bound = {}
    for index, name in enumerate(node.output):
        if name:
            bound[name] = g.mint(
                "split", x, sizes=sizes, dim=axis, index=index
            )
    return bound


def _topk(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``TopK`` — values output is the term; indices ride ``getitem``."""
    x = g.term(node, 0)
    k = g.const_scalar(node, 1)
    axis = int(g.attr(node, "axis", -1))
    largest = int(g.attr(node, "largest", 1))
    sorted_ = int(g.attr(node, "sorted", 1))
    term = g.mint(
        "topk", x, k=k, dim=axis, largest=largest, sorted=sorted_
    )
    bound = {}
    for index, name in enumerate(node.output):
        if name:
            bound[name] = (
                term
                if index == 0
                else g.mint("getitem", term, index=index)
            )
    return bound


def _gather(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Gather`` — ``embedding``/``index_select``/composed by rank.

    ONNX indices may be any rank; ATen's ``index_select`` takes a
    1-D index.  Rank-1 maps directly (``embedding`` when it is a
    rank-2 table on axis 0 — the more meaningful IR op), a scalar
    index reads as ``select`` when literal (else the flat-select +
    reshape composition), and a higher rank flattens then reshapes
    to the node's inferred output shape.
    """
    data = g.term(node, 0)
    idx = g.term(node, 1)
    axis = int(g.attr(node, "axis", 0))
    idx_shape = g.shape(node.input[1])
    data_shape = g.shape(node.input[0])
    scalar = g.leaf_scalar(node.input[1])
    if idx_shape == () and isinstance(scalar, (int, np.integer)):
        return _one(
            g,
            node,
            g.mint("select", data, dim=axis, index=int(scalar)),
        )
    if idx_shape == () or (
        idx_shape is not None and len(idx_shape) > 1
    ):
        # Decompose through the flat select, then restore the rank.
        out_shape = g.shape(node.output[0])
        if out_shape is None:
            g.refuse(
                node,
                "a scalar or multi-rank index needs the output "
                "shape statically known for the reshape",
            )
        flat = g.mint("reshape", idx, shape=(-1,))
        selected = g.mint("index_select", data, flat, dim=axis)
        return _one(
            g,
            node,
            g.mint("reshape", selected, shape=tuple(out_shape)),
        )
    if axis == 0 and data_shape is not None and len(data_shape) == 2:
        return _one(g, node, g.mint("embedding", data, idx))
    return _one(g, node, g.mint("index_select", data, idx, dim=axis))


def _gather_elements(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``GatherElements`` — the IR's ``gather`` (same-rank index)."""
    axis = int(g.attr(node, "axis", 0))
    return _one(g, node, g.mint("gather", *g.args(node), dim=axis))


def _nonzero(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``NonZero`` — ONNX's ``[rank, N]`` is ATen's transposed."""
    x = g.term(node, 0)
    return _one(
        g,
        node,
        g.mint("permute", g.mint("nonzero", x), dim=(1, 0)),
    )


def _cumsum(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``CumSum`` — the axis folds from a constant scalar input."""
    if int(g.attr(node, "exclusive", 0)) or int(
        g.attr(node, "reverse", 0)
    ):
        g.refuse(
            node,
            "exclusive/reverse CumSum is a different scan — no IR "
            "spelling without a shift composition the table does "
            "not claim",
        )
    axis = int(g.const_scalar(node, 1))
    return _one(g, node, g.mint("cumsum", g.term(node, 0), dim=axis))


def _trilu(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Trilu`` — ``triu``/``tril`` by the ``upper`` attr."""
    upper = int(g.attr(node, "upper", 1))
    k = (
        int(g.const_scalar(node, 1))
        if len(node.input) > 1 and node.input[1]
        else 0
    )
    return _one(
        g,
        node,
        g.mint(
            "triu" if upper else "tril",
            g.term(node, 0),
            diagonal=k,
        ),
    )


def _arange(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Range`` — the three bounds fold into ``arange``'s argN attrs."""
    attrs: dict[str, Any] = {
        "arg0": g.const_scalar(node, 0),
        "arg1": g.const_scalar(node, 1),
        "arg2": g.const_scalar(node, 2),
    }
    dtype = g.dtype_name(node.output[0])
    if dtype is not None:
        attrs["dtype"] = dtype
    return _one(g, node, g.mint("arange", **attrs))


def _constant_of_shape(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``ConstantOfShape`` — ``full`` over a constant shape input."""
    shape = tuple(_ints(g, node, 0))
    value = g.attr(node, "value", None)
    if value is None:
        fill, name = 0.0, "float32"
    else:
        arr = np.asarray(value)
        if arr.size != 1:
            g.refuse(
                node,
                "ConstantOfShape's value must be a single element",
            )
        fill = arr.reshape(()).item()
        name = arr.dtype.name
        if name not in DTYPE_NAMES:
            g.refuse(
                node,
                f"ConstantOfShape value dtype {arr.dtype} has no "
                "IR dtype name",
            )
    return _one(
        g,
        node,
        g.mint("full", shape=shape, fill_value=fill, dtype=name),
    )


def _one_hot(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``OneHot`` — last-axis ``one_hot`` with a ``[0, 1]`` value pair."""
    depth = g.const_scalar(node, 1)
    values = g.const(node, 2)
    if values.size != 2 or list(values.reshape(-1)) != [0, 1]:
        g.refuse(
            node,
            "one_hot's values are not [0, 1] — ATen emits a fixed "
            "0/1 encoding",
        )
    axis = int(g.attr(node, "axis", -1))
    base = g.shape(node.input[0])
    if base is not None and axis not in (-1, len(base)):
        g.refuse(
            node,
            f"axis={axis} is not the last axis — ATen's one_hot "
            "always encodes on a new trailing axis",
        )
    return _one(
        g,
        node,
        g.mint("one_hot", g.term(node, 0), num_classes=int(depth)),
    )


def _where(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Where`` — the ternary select."""
    return _one(g, node, g.mint("where", *g.args(node)))


def _dropout(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Dropout`` — eval-mode identity; a consumed mask refuses."""
    x = g.term(node, 0)
    bound = {}
    for index, name in enumerate(node.output):
        if not name:
            continue
        if index == 0:
            bound[name] = g.mint("dropout", x, train=False)
        else:
            g.refuse(
                node,
                "the mask output has no IR spelling — eval-mode "
                "dropout produces no mask",
            )
    return bound


# ---------------------------------------------------------------------------
#  linear algebra / conv / pooling
# ---------------------------------------------------------------------------


def _gemm(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Gemm`` — ``linear`` at the ATen spelling, composed otherwise.

    ``alpha * A.B' + beta * C`` with ``transB=1, alpha=beta=1`` is
    exactly ``F.linear``; other spellings decompose through
    ``matmul``/``mul``/``add`` with the transposes minted.
    """
    a_t = g.term(node, 0)
    b_t = g.term(node, 1)
    c_t = g.term(node, 2)
    alpha = float(g.attr(node, "alpha", 1.0))
    beta = float(g.attr(node, "beta", 1.0))
    trans_a = int(g.attr(node, "transA", 0))
    trans_b = int(g.attr(node, "transB", 0))
    if (
        not trans_a
        and trans_b
        and alpha == 1.0
        and beta == 1.0
        and c_t is not None
    ):
        return _one(g, node, g.mint("linear", a_t, b_t, c_t))
    if trans_a:
        a_t = g.mint("permute", a_t)
    if trans_b:
        b_t = g.mint("permute", b_t)
    term = g.mint("matmul", a_t, b_t)
    if alpha != 1.0:
        term = g.mint("mul", term, _const_of(alpha))
    if c_t is not None:
        if beta != 1.0:
            c_t = g.mint("mul", c_t, _const_of(beta))
        term = g.mint("add", term, c_t)
    return _one(g, node, term)


def _matmul(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``MatMul`` — batched and vector cases alike."""
    return _one(g, node, g.mint("matmul", *g.args(node)))


def _einsum(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Einsum`` — the equation attr rides through verbatim."""
    equation = g.attr(node, "equation", None)
    if equation is None:
        g.refuse(node, "no equation attribute")
    return _one(
        g, node, g.mint("einsum", *g.args(node), equation=equation)
    )


def _spatial_window(
    g: GraphLifter,
    node: onnx.NodeProto,
    spatial: int,
    stride_default: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Read ``(padding, strides, dilations)`` for a spatial op.

    Shared by ``Conv`` and ``MaxPool``: ONNX ``pads`` is
    ``[b1..bn, e1..en]`` and ATen's symmetric ``padding`` carries one
    value per spatial dim, so asymmetric padding is refused — as is
    ``auto_pad``, which computes pads the IR cannot spell.
    """
    if g.attr(node, "auto_pad", "NOTSET") not in (None, "NOTSET"):
        g.refuse(
            node,
            "auto_pad computes padding the IR cannot spell — "
            "export explicit pads",
        )
    pads = [int(v) for v in g.attr(node, "pads", [0] * 2 * spatial)]
    begins, ends = pads[:spatial], pads[spatial:]
    if begins != ends:
        g.refuse(
            node,
            f"asymmetric padding {pads} — ATen takes one value per "
            "spatial dim",
        )
    strides = [int(v) for v in g.attr(node, "strides", stride_default)]
    dilations = [
        int(v) for v in g.attr(node, "dilations", [1] * spatial)
    ]
    if len(strides) != spatial or len(dilations) != spatial:
        g.refuse(node, "strides/dilations do not match the rank")
    return begins, strides, dilations


def _conv(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``Conv`` — ``conv1d``/``conv2d`` by weight rank."""
    weight_shape = g.shape(node.input[1])
    if weight_shape is None or len(weight_shape) < 3:
        g.refuse(node, "the conv weight's rank is not statically known")
    spatial = len(weight_shape) - 2
    if spatial not in (1, 2):
        g.refuse(
            node,
            f"{spatial}-D convolution has no IR op (conv1d/conv2d "
            "only)",
        )
    pads, strides, dilations = _spatial_window(
        g, node, spatial, [1] * spatial
    )
    group = int(g.attr(node, "group", 1))
    op = "conv1d" if spatial == 1 else "conv2d"
    return _one(
        g,
        node,
        g.mint(
            op,
            *g.args(node),
            stride=tuple(strides),
            padding=tuple(pads),
            dilation=tuple(dilations),
            groups=group,
        ),
    )


def _max_pool(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``MaxPool`` — ``max_pool2d`` over two spatial dims, values only."""
    outs = _outs(node)
    if len(outs) != 1 or outs[0] != node.output[0]:
        g.refuse(
            node,
            "the Indices output has no IR spelling — ATen's "
            "max_pool2d-with-indices is not the mapped op",
        )
    if int(g.attr(node, "storage_order", 0)):
        g.refuse(node, "storage_order=1 (column-major) is refused")
    kernel = [int(v) for v in g.attr(node, "kernel_shape", ())]
    if len(kernel) != 2:
        g.refuse(
            node,
            f"{len(kernel)}-D max pool has no IR op (max_pool2d only)",
        )
    pads, strides, dilations = _spatial_window(g, node, 2, kernel)
    ceil_mode = int(g.attr(node, "ceil_mode", 0))
    return {
        outs[0]: g.mint(
            "max_pool2d",
            g.term(node, 0),
            kernel_size=tuple(kernel),
            stride=tuple(strides),
            padding=tuple(pads),
            dilation=tuple(dilations),
            ceil_mode=ceil_mode,
        )
    }


def _global_pool(ir_op: str) -> OnnxLifter:
    """Build a lifter for ``GlobalAveragePool``/``GlobalMaxPool``."""

    def lift(g: GraphLifter, node: onnx.NodeProto) -> dict:
        """Reduce over the spatial axes (2..rank-1), keep dims."""
        base = g.shape(node.input[0])
        if base is None or len(base) < 3:
            g.refuse(
                node,
                "global pooling needs a statically known rank >= 3",
            )
        dims = tuple(range(2, len(base)))
        return _one(
            g,
            node,
            g.mint(ir_op, g.term(node, 0), dim=dims, keepdim=True),
        )

    return lift


# ---------------------------------------------------------------------------
#  normalization
# ---------------------------------------------------------------------------


def _batch_norm(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``BatchNormalization`` — inference mode only, five operands."""
    if int(g.attr(node, "training_mode", 0)):
        g.refuse(
            node,
            "training-mode batch_norm updates running statistics — "
            "a graph state change, not a function",
        )
    if len(_outs(node)) != 1:
        g.refuse(
            node,
            "the running-statistic outputs have no IR spelling in "
            "eval mode",
        )
    args = g.args(node)
    if len(args) < 5:
        g.refuse(
            node, "batch_norm needs weight, bias and running stats"
        )
    eps = float(g.attr(node, "epsilon", 1e-5))
    momentum = float(g.attr(node, "momentum", 0.9))
    return _one(
        g,
        node,
        g.mint(
            "batch_norm",
            *args,
            eps=eps,
            momentum=momentum,
            training=False,
        ),
    )


def _layer_norm(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``LayerNormalization`` — normalized_shape from the scale's shape."""
    if len(_outs(node)) != 1:
        g.refuse(node, "the Mean/InvStdDev outputs have no IR spelling")
    scale_shape = g.shape(node.input[1])
    if scale_shape is None:
        g.refuse(
            node,
            "normalized_shape is not statically known (the scale "
            "operand is unshaped)",
        )
    axis = int(g.attr(node, "axis", -1))
    base = g.shape(node.input[0])
    if base is not None and axis not in (
        len(base) - len(scale_shape),
        -len(scale_shape),
    ):
        g.refuse(
            node,
            f"axis={axis} does not bound the scale's shape "
            f"{scale_shape} on a rank-{len(base)} input",
        )
    eps = float(g.attr(node, "epsilon", 1e-5))
    return _one(
        g,
        node,
        g.mint(
            "layer_norm",
            *g.args(node),
            dim=tuple(scale_shape),
            eps=eps,
        ),
    )


def _group_norm(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``GroupNormalization`` (opset 21) — groups ride an attr."""
    num_groups = int(g.attr(node, "num_groups", 0))
    if num_groups <= 0:
        g.refuse(node, f"num_groups={num_groups} is not positive")
    eps = float(g.attr(node, "epsilon", 1e-5))
    return _one(
        g,
        node,
        g.mint(
            "group_norm",
            *g.args(node),
            num_groups=num_groups,
            eps=eps,
        ),
    )


def _instance_norm(g: GraphLifter, node: onnx.NodeProto) -> dict:
    """``InstanceNormalization`` — input stats by construction."""
    eps = float(g.attr(node, "epsilon", 1e-5))
    return _one(
        g,
        node,
        g.mint(
            "instance_norm",
            *g.args(node),
            use_input_stats=True,
            eps=eps,
        ),
    )


# ---------------------------------------------------------------------------
#  the table
# ---------------------------------------------------------------------------

#: Ops that map onto exactly one IR op of the same arity.
_UNARY: dict[str, str] = {
    "Neg": "neg",
    "Abs": "abs",
    "Sign": "sign",
    "Floor": "floor",
    "Ceil": "ceil",
    "Round": "round",
    "Sqrt": "sqrt",
    "Exp": "exp",
    "Log": "log",
    "Reciprocal": "reciprocal",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Asin": "asin",
    "Acos": "acos",
    "Atan": "atan",
    "Sinh": "sinh",
    "Cosh": "cosh",
    "Tanh": "tanh",
    "Erf": "erf",
    "Relu": "relu",
    "Sigmoid": "sigmoid",
    "Softsign": "softsign",
    "Mish": "mish",
    "Softplus": "softplus",
    "HardSwish": "hardswish",
    "IsNaN": "isnan",
    "Not": "logical_not",
    "Asinh": "asinh",
    "Acosh": "acosh",
    "Atanh": "atanh",
}

#: Two-operand elementwise ops — ONNX's NumPy-style broadcasting is
#: the same broadcast the IR's typing assumes.
_BINARY: dict[str, str] = {
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "Div": "div",
    "Pow": "pow",
    "Equal": "eq",
    "Less": "lt",
    "LessOrEqual": "le",
    "Greater": "gt",
    "GreaterOrEqual": "ge",
    "And": "logical_and",
    "Or": "logical_or",
    "Xor": "logical_xor",
    "BitwiseAnd": "bitwise_and",
    "BitwiseOr": "bitwise_or",
}

#: Variadic elementwise ops folded left.
_VARIADIC: dict[str, str] = {
    "Sum": "add",
    "Max": "maximum",
    "Min": "minimum",
}

#: Reductions — axes folded from input or attr.
_REDUCERS: dict[str, str] = {
    "ReduceSum": "sum",
    "ReduceMean": "mean",
    "ReduceProd": "prod",
    "ReduceMax": "amax",
    "ReduceMin": "amin",
}


#: The complete ONNX -> IR lifting table.  Keys are ONNX
#: ``op_type``s in the default domain; a node with a non-empty
#: ``domain`` or an unlisted type refuses through the lifter's
#: missing-entry path.
ONNX_LIFTS: dict[str, OnnxLifter] = {
    **{name: _simple(op) for name, op in _UNARY.items()},
    **{name: _simple(op) for name, op in _BINARY.items()},
    **{name: _fold(op) for name, op in _VARIADIC.items()},
    **{name: _reducer(op) for name, op in _REDUCERS.items()},
    "Identity": _identity,
    "Mod": _mod,
    "Mean": _mean,
    "Clip": _clip,
    "ThresholdedRelu": _thresholded_relu,
    "LeakyRelu": _leaky_relu,
    "Elu": _elu,
    "Celu": _celu,
    "Selu": _selu,
    "HardSigmoid": _hard_sigmoid,
    "Gelu": _gelu,
    "Softmax": _softmax("softmax"),
    "LogSoftmax": _softmax("log_softmax"),
    "IsInf": _isinf,
    "Cast": _cast,
    "CastLike": _cast_like,
    "Concat": _concat,
    "Transpose": _transpose,
    "Reshape": _reshape,
    "Flatten": _flatten,
    "Unsqueeze": _unsqueeze,
    "Squeeze": _squeeze,
    "Expand": _expand,
    "Tile": _tile,
    "Pad": _pad,
    "Slice": _slice,
    "Split": _split,
    "TopK": _topk,
    "Gather": _gather,
    "GatherElements": _gather_elements,
    "NonZero": _nonzero,
    "CumSum": _cumsum,
    "Trilu": _trilu,
    "Range": _arange,
    "ConstantOfShape": _constant_of_shape,
    "OneHot": _one_hot,
    "Where": _where,
    "Dropout": _dropout,
    "Gemm": _gemm,
    "MatMul": _matmul,
    "Einsum": _einsum,
    "Conv": _conv,
    "MaxPool": _max_pool,
    "GlobalAveragePool": _global_pool("mean"),
    "GlobalMaxPool": _global_pool("amax"),
    "ArgMax": _arg_reducer("argmax"),
    "ArgMin": _arg_reducer("argmin"),
    "ReduceL1": _vector_norm(1),
    "ReduceL2": _vector_norm(2),
    "BatchNormalization": _batch_norm,
    "LayerNormalization": _layer_norm,
    "GroupNormalization": _group_norm,
    "InstanceNormalization": _instance_norm,
}
