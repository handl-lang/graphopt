"""IR -> ONNX graph lowering — ADR 0006, direction 1.

:class:`GraphBuilder` walks one :class:`mlopt_core.ir.IR` term and
emits an ``onnx.ModelProto``.  It is the whole of the sink's
*lowering* half; :mod:`mlopt_onnx.opmap` holds the op table it
dispatches through, and :mod:`mlopt_onnx.sink` wraps both behind the
``mlopt_core.ports.Sink`` contract.

Design notes that are also the honest caveats:

* **Refuse, never drop.**  An op with no entry in the mapping raises
  :class:`~mlopt_onnx.errors.UnsupportedOnnxOpError`.  So does a
  *mapped* op in a spelling the ONNX op cannot express (a missing
  static rank for a ``Transpose`` permutation, an unknown
  ``rounding_mode``, a dynamic ``leaky_relu`` slope that would have
  to become a static attribute).
* **Terms are memoised by identity.**  IR terms are hash-consed, so a
  shared subterm emits one node and is referenced twice — the ONNX
  graph is a DAG, like the IR.
* **Ranks come from the core's typing.**  ``transpose``/``permute``/
  ``pad``/``chunk`` need a static rank or shape; the builder asks
  ``mlopt_core.typing.shape_of`` (the same backend-neutral shape
  inference the cost model uses) and declines when the answer is
  unknown rather than guessing.
* **Output types are inferred, not declared.**  The graph is built
  with output *names* only and then run through ONNX's own
  ``shape_inference``, so a boolean or int64 result is typed by the
  graph rather than by a guess at the builder.
* **Input dtypes are declared, not derived.**  The IR types inputs by
  *shape* (``Var.typ``), not by element type, so the sink takes an
  ``input_dtypes`` map.  A program whose input is an index tensor
  must say so; a program whose input is a float tensor needs nothing.
  This is the one place the port's ``(IR, leaves)`` surface does not
  carry what a non-torch backend needs — recorded in the ADR 0006
  stage-1 report, worked around on the sink side rather than by
  widening core.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import onnx
from mlopt_core.attrs import ATTR_SCHEMA, attr_of
from mlopt_core.ir import IR, Const, Op, Param, Var
from mlopt_core.typing import shape_of
from onnx import helper, numpy_helper

from mlopt_onnx.arrays import (
    DTYPE_NAMES,
    WORKING_DTYPES,
    numpy_to_onnx_type,
    to_numpy,
)
from mlopt_onnx.errors import (
    OnnxLoweringError,
    UnsupportedOnnxOpError,
)
from mlopt_onnx.opmap import ONNX_BINDINGS

__all__ = ["GraphBuilder", "lower_ir"]

#: ``(op, canonical attr name) -> positional index`` — the reverse of
#: ``mlopt_core.attrs.ATTR_SCHEMA``.  The builder reads the canonical
#: spelling and falls back to the positional one, exactly as every
#: core reader does.
_ATTR_POS: dict[tuple[str, str], int] = {
    (op, canonical): pos
    for op, schema in ATTR_SCHEMA.items()
    for pos, canonical in schema.items()
}


class GraphBuilder:
    """Builds one ONNX graph from one IR term.

    Parameters
    ----------
    ir : IR
        The program.  ``ir.inputs`` gives the graph inputs (in order);
        ``ir.params`` names the leaves the initializers come from.
    params : dict[str, Any]
        The concrete leaf values, keyed by param name.
    dtype : str
        The floating working dtype — ``"float32"`` or ``"float64"``.
    input_dtypes : Mapping[str, str] | None
        Per-input element types, keyed by ``Var`` name.  The IR
        carries *shapes* for inputs, not dtypes, so a program whose
        input is an index tensor must declare it here (see
        :meth:`GraphBuilder._declare_input`).
    opset : int
        The ONNX opset the mapping targets.
    ir_version : int
        The ONNX IR version stamped on the model.  Defaults to a
        version every current onnxruntime accepts (ONNX's own latest
        ``IR_VERSION`` outruns released runtimes).
    name : str
        The graph name.

    """

    def __init__(
        self,
        ir: IR,
        params: dict[str, Any] | None = None,
        *,
        dtype: str = "float32",
        input_dtypes: Mapping[str, str] | None = None,
        opset: int = 18,
        ir_version: int = 10,
        name: str = "mlopt",
    ) -> None:
        """Initialise the builder's tables and working dtype."""
        if dtype not in WORKING_DTYPES:
            raise OnnxLoweringError(
                f"unknown working dtype {dtype!r}; expected one of "
                f"{sorted(WORKING_DTYPES)}"
            )
        self.ir = ir
        self.params = params or {}
        self.work = WORKING_DTYPES[dtype]
        self.dtype_name = dtype
        self.input_dtypes = dict(input_dtypes or {})
        self.opset = opset
        self.ir_version = ir_version
        self.name = name
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.graph_inputs: list[onnx.ValueInfoProto] = []
        self._values: dict[Any, str] = {}
        self._multi: dict[Any, list[str]] = {}
        self._taken: set[str] = set()
        self._input_names: set[str] = set()
        self._dtypes: dict[str, np.dtype] = {}
        self._counter = 0
        self._shape_memo: dict[Any, Any] = {}

    # -- name / value plumbing ----------------------------------------

    def fresh(self, prefix: str = "v") -> str:
        """Return an unused value name."""
        while True:
            self._counter += 1
            name = f"{prefix}{self._counter}"
            if name not in self._taken:
                self._taken.add(name)
                return name

    def dtype_of(self, name: str) -> np.dtype:
        """Return the tracked dtype of a value, or the working dtype."""
        return self._dtypes.get(name, self.work)

    def node(
        self,
        op_type: str,
        inputs: Any,
        n_outputs: int = 1,
        dtype: np.dtype | None = None,
        **attrs: Any,
    ) -> list[str]:
        """Append one ONNX node; return its output value names.

        ``dtype`` overrides the result type; by default it is the
        first input's tracked dtype (elementwise ops preserve their
        operand's type), falling back to the working dtype.
        """
        outs = [self.fresh() for _ in range(n_outputs)]
        self.nodes.append(
            helper.make_node(
                op_type, list(inputs), outs, name=outs[0], **attrs
            )
        )
        if dtype is None:
            dtype = next(
                (self._dtypes[i] for i in inputs if i in self._dtypes),
                self.work,
            )
        for out in outs:
            self._dtypes[out] = dtype
        return outs

    def initializer(self, array: np.ndarray) -> str:
        """Append a constant tensor; return its value name."""
        name = self.fresh("c")
        arr = np.asarray(array)
        self.initializers.append(numpy_helper.from_array(arr, name))
        self._dtypes[name] = arr.dtype
        return name

    def constant(
        self, value: Any, dtype: np.dtype | None = None
    ) -> str:
        """Add a 0-d constant of ``dtype``; return its value name.

        ``Const`` leaves carry a Python number, not a dtype.  The
        builder therefore emits a scalar in the dtype of the tensor it
        is combined with (``dtype=None`` falls back to the working
        dtype) — ONNX elementwise ops require both operands to share
        one element type, and an ``int64`` tensor compared against a
        float scalar is a type error, not a promotion.
        """
        return self.initializer(
            np.asarray(
                value, dtype=dtype if dtype is not None else self.work
            )
        )

    def scalar(self, value: Any) -> str:
        """Add a 0-d constant in the working dtype; return its name."""
        return self.constant(value, self.work)

    def int64(self, values: Any) -> str:
        """Add an int64 constant (axes / shapes); return its name."""
        return self.initializer(
            np.asarray(list(values), dtype=np.int64)
        )

    def int64_scalar(self, value: int) -> str:
        """Add a 0-d int64 constant; return its name."""
        return self.initializer(np.asarray(value, dtype=np.int64))

    # -- attr / shape access ------------------------------------------

    def attr(self, term: Op, name: str, default: Any = None) -> Any:
        """Read a canonical attr, falling back to its ``argN`` spelling."""
        pos = _ATTR_POS.get((term.op, name))
        names = (name,) if pos is None else (name, f"arg{pos}")
        return attr_of(term, *names, default=default)

    def operand_shape(self, term: Op, i: int = 0) -> tuple | None:
        """Return the believed shape of operand ``i``, or ``None``."""
        result = shape_of(term.args[i], self._shape_memo)
        return result if isinstance(result, tuple) else None

    def rank(self, term: Op, i: int = 0) -> int | None:
        """Return the rank of operand ``i`` when it is statically known."""
        shape = self.operand_shape(term, i)
        return None if shape is None else len(shape)

    def required_rank(self, term: Op, i: int = 0) -> int:
        """Return operand ``i``'s rank or refuse the lowering."""
        rank = self.rank(term, i)
        if rank is None:
            raise UnsupportedOnnxOpError(
                term.op,
                "the ONNX lowering needs a statically known rank for "
                f"operand {i} (shape inference returned unknown)",
            )
        return rank

    def const_operand(self, term: Op, i: int) -> Any:
        """Return operand ``i``'s Python value when it is a constant.

        Several IR ops carry a scalar as a real *operand* rather than
        an attribute (``leaky_relu``'s slope, ``clamp``'s bounds on
        some paths).  An ONNX attribute must be static, so a
        non-constant operand is a refusal, not a runtime value.
        """
        arg = term.args[i]
        if isinstance(arg, Const):
            return arg.value
        raise UnsupportedOnnxOpError(
            term.op,
            f"operand {i} must be a constant to become an ONNX "
            f"attribute, got {type(arg).__name__}",
        )

    def multi_outputs(self, term: Any) -> list[str] | None:
        """Return the value names a multi-output op produced, if any."""
        return self._multi.get(term)

    def declare_multi(self, term: Any, names: list[str]) -> None:
        """Record the extra outputs of a multi-output op."""
        self._multi[term] = names

    # -- leaves --------------------------------------------------------

    def _declare_input(self, var: Var) -> str:
        """Declare one graph input; return its value name.

        The IR types tensors by *shape*, not dtype, so the graph
        input's element type comes from the sink's declared
        ``input_dtypes`` map when it names this input and from the
        working dtype otherwise.  A program whose *input* is an index
        tensor must say so — there is no channel in the IR to carry
        it (see the module docstring).
        """
        shape = list(var.typ.shape)
        dtype = DTYPE_NAMES[
            self.input_dtypes.get(var.name, self.dtype_name)
        ]
        self.graph_inputs.append(
            helper.make_tensor_value_info(
                var.name,
                numpy_to_onnx_type(np.empty(0, dtype=dtype)),
                shape,
            )
        )
        self._taken.add(var.name)
        self._input_names.add(var.name)
        self._dtypes[var.name] = dtype
        return var.name

    def _declare_param(self, param: Param) -> str:
        """Materialise one parameter as an initializer."""
        if param.name not in self.params:
            raise OnnxLoweringError(
                f"no value supplied for parameter {param.name!r} "
                f"(shape {param.typ.shape}) — the sink refuses to "
                f"lower a program with an unbound leaf"
            )
        arr = to_numpy(self.params[param.name])
        if arr.dtype.kind == "f":
            arr = arr.astype(self.work, copy=False)
        name = (
            param.name
            if param.name not in self._taken
            else self.fresh(f"{param.name}_")
        )
        self._taken.add(name)
        self.initializers.append(
            numpy_helper.from_array(np.ascontiguousarray(arr), name)
        )
        self._dtypes[name] = arr.dtype
        return name

    # -- the walk ------------------------------------------------------

    def _emit_args(self, term: Op) -> list[str]:
        """Emit ``term``'s operands, giving scalars a sibling's dtype.

        Non-constant operands are emitted first so each ``Const``
        scalar can adopt the element type of the tensor it is
        combined with — ONNX has no weak typing, so the choice is
        part of the lowering rather than something the runtime
        promotes away.
        """
        names: list[str | None] = [None] * len(term.args)
        for i, arg in enumerate(term.args):
            if not isinstance(arg, Const):
                names[i] = self.emit(arg)
        ref = self._scalar_dtype(names)
        for i, arg in enumerate(term.args):
            if isinstance(arg, Const):
                names[i] = self.constant(arg.value, ref)
        return [n for n in names if n is not None]

    def _scalar_dtype(self, names: list[str | None]) -> np.dtype | None:
        """Return the dtype a ``Const`` operand should adopt.

        A scalar adopts a sibling's dtype — but a *boolean* sibling
        is a predicate, never the constant's own type (``where``'s
        condition, ``masked_fill``'s mask).  Prefer the first
        non-bool tracked dtype; an all-bool term falls back to the
        first tracked one so ``logical_and(mask, Const)`` still
        emits a bool scalar.
        """
        ref = next(
            (
                self._dtypes[n]
                for n in names
                if n is not None
                and self._dtypes.get(n, np.dtype(np.bool_)).kind != "b"
            ),
            None,
        )
        if ref is not None:
            return ref
        return next(
            (
                self._dtypes[n]
                for n in names
                if n is not None and n in self._dtypes
            ),
            None,
        )

    def emit(self, term: Any) -> str:
        """Emit ``term`` (and its children); return its value name.

        Terms are hash-consed, so a shared subterm — the residual
        ``x`` feeding two branches, a weight reused by two matmuls —
        emits exactly one node or initializer and is referenced at
        each use site.  The ONNX graph is a DAG, like the IR.
        """
        cached = self._values.get(term)
        if cached is not None:
            return cached
        if isinstance(term, Var):
            name = self._declare_input(term)
        elif isinstance(term, Param):
            name = self._declare_param(term)
        elif isinstance(term, Const):
            name = self.constant(term.value, self.work)
        elif isinstance(term, Op):
            build = ONNX_BINDINGS.get(term.op)
            if build is None:
                raise UnsupportedOnnxOpError(
                    term.op, "no ONNX mapping for this op"
                )
            name = build(self, term, self._emit_args(term))
        else:
            raise OnnxLoweringError(
                f"cannot lower leaf {term!r} — expected an IR term"
            )
        self._values[term] = name
        return name

    def build(self) -> onnx.ModelProto:
        """Emit the whole graph and return the checked model.

        The graph's outputs are ``ir.roots`` — one entry for a
        single-output program, every entry of the ``pack`` pairing for
        a multi-output one.  ONNX is natively multi-output, so the
        pairing itself is *not* a node: it is the output list, which
        is why the sink inherits none of the single-output truncation
        the ADR 0006 prerequisite closed.
        """
        for var in self.ir.inputs:
            self._values[var] = self._declare_input(var)
        out_names = [self.emit(term) for term in self.ir.roots]
        # A program that *is* its input (an export that collapsed
        # ``x.contiguous()`` to ``x``) would name a graph output after
        # a graph input; ONNX reads that as the output value_info
        # redefining the input.  Route those through an Identity so
        # every output is a distinct value.
        out_names = [
            (
                self.node("Identity", [name])[0]
                if name in self._input_names
                else name
            )
            for name in out_names
        ]
        outputs = []
        for name in out_names:
            info = onnx.ValueInfoProto()
            info.name = name
            outputs.append(info)
        graph = helper.make_graph(
            self.nodes,
            self.name,
            self.graph_inputs,
            outputs,
            initializer=self.initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", self.opset)],
            ir_version=self.ir_version,
            producer_name="mlopt-onnx",
        )
        model = onnx.shape_inference.infer_shapes(
            model, strict_mode=False
        )
        onnx.checker.check_model(model)
        return model


def lower_ir(
    ir: IR,
    params: dict[str, Any] | None = None,
    *,
    dtype: str = "float32",
    input_dtypes: Mapping[str, str] | None = None,
    opset: int = 18,
    ir_version: int = 10,
    name: str = "mlopt",
) -> onnx.ModelProto:
    """Lower ``ir`` to an ONNX model — the sink's ``lower`` half."""
    builder = GraphBuilder(
        ir,
        params,
        dtype=dtype,
        input_dtypes=input_dtypes,
        opset=opset,
        ir_version=ir_version,
        name=name,
    )
    return builder.build()
