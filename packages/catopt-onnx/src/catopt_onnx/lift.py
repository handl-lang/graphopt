"""ONNX graph -> IR lifting — ADR 0006, direction 2.

:class:`GraphLifter` walks one ``onnx.GraphProto`` and mints the
equivalent :class:`catopt_core.ir.IR` term.  It is the whole of the
source's *lifting* half; :mod:`catopt_onnx.liftmap` holds the op
table it dispatches through, and :mod:`catopt_onnx.source` wraps both
behind the ``catopt_core.ports.Source`` contract.

Design notes that are also the honest caveats:

* **Refuse, never drop.**  An ``op_type`` with no entry in the
  lifting table raises
  :class:`~catopt_onnx.errors.UnsupportedOnnxOpError` — the same
  error the sink raises for the same reason, in the other direction.
  So does a *mapped* op whose spelling the IR cannot express: a
  ``Slice`` whose bounds arrive as a computed (non-constant) tensor,
  a graph output that is not a tensor, a ``Split`` with data-driven
  sizes.  A silently narrower import would be a different program
  wearing the original's name — the ADR's rule applies both ways.
* **ONNX attrs become IR attrs; constant *inputs* become attrs
  too.**  ONNX moved its op parameters from attributes to tensor
  inputs (opset 13+): ``Slice`` bounds, ``ReduceSum`` axes, ``Clip``
  limits, ``Reshape`` shapes are all inputs now.  The IR spells those
  same parameters as attrs — ``catopt_core.attrs.ATTR_SCHEMA``'s
  canonical names — so the lifter *folds* constant inputs into
  attrs.  A parameter input that is not an initializer or
  ``Constant`` node is a refusal: there is no honest IR spelling for
  a data-driven axis list.
* **Shapes come from ONNX's own inference.**  ``Var.typ`` is
  shape-typed; the lifter runs ``onnx.shape_inference`` once and
  types every leaf from it (initializers type themselves — their
  values are concrete).  A dim ONNX cannot pin lands as the IR's
  ``None`` (unknown) extent — representable — while a value with no
  tensor type at all is a refusal, never a fabricated shape.
* **Element types ride a side channel.**  ``Var.typ`` carries shape
  but NOT element type — the same port gap the sink works around
  with its ``input_dtypes`` map.  The source's equivalent is
  :attr:`LiftResult.input_dtypes`: the element types ONNX *does*
  declare on graph inputs, so a caller wiring
  ``OnnxSink(input_dtypes=source.input_dtypes)`` keeps an ``int64``
  index input an index input instead of a float tensor.  This is
  the documented source-side workaround, not a widening of core.
* **Provenance is a leaf name.**  ``Var``/``Param`` names are the
  graph's own value names — ``IR.inputs``/``ir.params`` read as the
  model reads.  Op-node names have no IR channel (``Op`` has no name
  field); they are carried in refusal messages, not on terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import onnx
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from onnx import numpy_helper

from catopt_onnx.arrays import DTYPE_NAMES, ONNX_TYPE_NAMES
from catopt_onnx.errors import OnnxImportError, UnsupportedOnnxOpError

if TYPE_CHECKING:
    from catopt_onnx.liftmap import OnnxLifter

__all__ = ["GraphLifter", "LiftResult", "lift_model"]

#: The opset window the lifting table is written against.  Below 13
#: the parameter-as-input convention the lifter relies on does not
#: hold (``Slice`` bounds were attributes; ``Softmax`` had different
#: axis semantics); above 21 is past what the table has been measured
#: against.  A model outside the window is refused, not guessed at.
MIN_OPSET = 13
MAX_OPSET = 21

#: NumPy dtype -> the name the IR records (``to(dtype=...)``) — the
#: inverse of :data:`catopt_onnx.arrays.DTYPE_NAMES`.
_NUMPY_TO_NAME: dict[np.dtype, str] = {
    dtype: name for name, dtype in DTYPE_NAMES.items()
}

#: Sentinel for attribute values the lifter refuses outright: graph,
#: sparse-tensor and tensor-list attributes (``If``/``Loop``/``Scan``
#: bodies) have no IR spelling.
_UNMAPPABLE = object()


class LiftResult:
    """The import record — what :func:`lift_model` returns.

    Attributes
    ----------
    ir : IR
        The lifted program.
    leaves : dict[str, np.ndarray]
        The concrete leaf values (initializers and tensor
        ``Constant`` nodes), keyed by ``Param`` name — the same
        ``(IR, leaves)`` pair the torch source returns as
        ``source_tensors``.
    input_dtypes : dict[str, str]
        The element types ONNX declares on the graph's inputs, keyed
        by ``Var`` name — the source-side counterpart of the sink's
        ``input_dtypes`` workaround for ``Var.typ`` carrying no
        element type.

    """

    def __init__(
        self,
        ir: IR,
        leaves: dict[str, np.ndarray],
        input_dtypes: dict[str, str],
    ) -> None:
        """Store the lifted program and its side-channel metadata."""
        self.ir = ir
        self.leaves = leaves
        self.input_dtypes = input_dtypes


def _attr_value(attr: onnx.AttributeProto) -> Any:
    """Decode one ONNX attribute to a Python value (or ``_UNMAPPABLE``).

    Scalars and scalar lists become Python ``int``/``float``/``str``;
    a tensor attribute (``ConstantOfShape``'s ``value``) becomes a
    NumPy array.  Graph, sparse-tensor and list-of-tensor attributes
    have no IR spelling — the lifter refuses them only when read.
    """
    kind = attr.type
    if kind == onnx.AttributeProto.INT:
        return int(attr.i)
    if kind == onnx.AttributeProto.FLOAT:
        return float(attr.f)
    if kind == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if kind == onnx.AttributeProto.INTS:
        return [int(v) for v in attr.ints]
    if kind == onnx.AttributeProto.FLOATS:
        return [float(v) for v in attr.floats]
    if kind == onnx.AttributeProto.STRINGS:
        return [
            s.decode("utf-8", errors="replace") for s in attr.strings
        ]
    if kind == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    return _UNMAPPABLE


def _tensor_shape(
    value_info: onnx.ValueInfoProto,
) -> tuple[tuple[int | None, ...], str] | None:
    """Read ``(shape, dtype_name)`` off a tensor ValueInfo, or ``None``.

    ``None`` covers every non-tensor or unshaped type (sequences,
    maps, optionals, a tensor with no element type): the callers turn
    it into a refusal where a tensor was required.  Dynamic dims
    (``dim_param``, or dims ONNX left unwritten) become the IR's
    ``None`` extent — representable, unlike a missing shape.
    """
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
        return None
    dtype = ONNX_TYPE_NAMES.get(tensor_type.elem_type)
    if dtype is None:
        return None
    shape = tensor_type.shape
    dims = tuple(
        int(d.dim_value) if d.HasField("dim_value") else None
        for d in shape.dim
    )
    return dims, dtype


class GraphLifter:
    """Lifts one ONNX graph to an IR program.

    Parameters
    ----------
    graph : onnx.GraphProto
        The graph body.  ONNX requires nodes in topological order;
        the lifter resolves every value through its environment, so
        an unresolved producer is a refusal, not a hole in the graph.
    initializers : Mapping[str, np.ndarray]
        Every initializer in scope, by name (``graph.initializer``,
        decoded).  An initializer that is *also* a graph input is a
        parameter with a default — the lifter types it as ``Param``,
        like the torch source's lifted buffers.
    value_info : Mapping[str, tuple]
        The ``name -> (shape, dtype_name)`` table built from the
        graph's declared and inferred ``ValueInfo``s.  Intermediates
        the lifter queries (``Gather``'s output shape, a ``Conv``
        weight's rank) read through it; a query that misses is a
        refusal only where a static answer is required.

    """

    def __init__(
        self,
        graph: onnx.GraphProto,
        *,
        initializers: Mapping[str, np.ndarray] | None = None,
        value_info: Mapping[str, tuple] | None = None,
    ) -> None:
        """Initialise the environment over the graph's values."""
        self.graph = graph
        self.initializers = dict(initializers or {})
        self.value_info = dict(value_info or {})
        self.inputs: list[Var] = []
        self.params: dict[str, Param] = {}
        self.leaves: dict[str, np.ndarray] = {}
        self.input_dtypes: dict[str, str] = {}
        self._env: dict[str, Any] = {}
        self._input_names: set[str] = set()
        self._constants: dict[str, np.ndarray] = {}

    # -- attr / shape access ------------------------------------------

    def attr(
        self, node: onnx.NodeProto, name: str, default: Any = None
    ) -> Any:
        """Read one attribute; refuse the unmappable kinds on access.

        A graph or tensor-list attribute is only a refusal when the
        lifter actually needs it — ``Constant`` carries several
        alternative value attributes of which only one matters.
        """
        for a in node.attribute:
            if a.name == name:
                v = _attr_value(a)
                if v is _UNMAPPABLE:
                    raise UnsupportedOnnxOpError(
                        node.op_type,
                        f"attribute {name!r} of node "
                        f"{node.name or node.output[0]!r} is a kind "
                        "the IR cannot spell (graph / sparse / "
                        "tensor-list)",
                    )
                return v
        return default

    def shape(self, name: str) -> tuple | None:
        """Return the declared/inferred shape of a value, or ``None``."""
        info = self.value_info.get(name)
        if info is not None:
            return info[0]
        arr = self.initializers.get(name)
        return tuple(arr.shape) if arr is not None else None

    def dtype_name(self, name: str) -> str | None:
        """Return the declared/inferred element type name, or ``None``."""
        info = self.value_info.get(name)
        if info is not None:
            return info[1]
        arr = self.initializers.get(name)
        if arr is None:
            arr = self._constants.get(name)
        return (
            _NUMPY_TO_NAME.get(arr.dtype) if arr is not None else None
        )

    # -- value resolution ----------------------------------------------

    def _param(self, name: str) -> Param:
        """Materialise an initializer as a ``Param`` leaf (idempotent)."""
        param = self.params.get(name)
        if param is not None:
            return param
        arr = self.initializers[name]
        param = Param(name, TensorType(tuple(arr.shape)))
        self.params[name] = param
        self.leaves[name] = arr
        return param

    def _const_leaf(self, name: str) -> Any:
        """Materialise a ``Constant`` node's output as an IR leaf.

        A 0-d numeric constant mints as :class:`Const` — the torch
        source's own spelling for scalar operands (``pow(x, 2)`` is a
        ``Const`` operand, not a ``Param``).  A shaped tensor constant
        becomes a ``Param`` leaf, like an initializer.
        """
        arr = self._constants[name]
        if arr.ndim == 0 and arr.dtype.kind in "fiu":
            return Const(
                int(arr) if arr.dtype.kind in "iu" else float(arr)
            )
        param = self.params.get(name)
        if param is None:
            param = Param(name, TensorType(tuple(arr.shape)))
            self.params[name] = param
            self.leaves[name] = arr
        return param

    def term(self, node: onnx.NodeProto, i: int) -> Any:
        """Resolve input ``i`` of ``node`` to a term; refuse the gaps.

        An empty optional input (``""``) is ``None``; an unresolved
        name — neither input, initializer, constant nor produced —
        is a refusal, not a hole in the graph.
        """
        name = node.input[i] if i < len(node.input) else ""
        if name == "":
            return None
        term = self._env.get(name)
        if term is not None:
            return term
        if name in self.initializers:
            term = self._param(name)
        elif name in self._constants:
            term = self._const_leaf(name)
        else:
            raise UnsupportedOnnxOpError(
                node.op_type,
                f"input {i} of node {node.name or name!r} is "
                f"{name!r}, which no producer defines — the graph is "
                "ill-formed or the value comes from an outer scope "
                "the lifter cannot see",
            )
        self._env[name] = term
        return term

    def args(self, node: onnx.NodeProto) -> list[Any]:
        """Resolve every non-empty input of ``node``."""
        return [
            t
            for t in (
                self.term(node, i) for i in range(len(node.input))
            )
            if t is not None
        ]

    def const(self, node: onnx.NodeProto, i: int) -> np.ndarray:
        """Resolve input ``i`` to a concrete array, or refuse.

        The attr-fold positions — ``Slice`` bounds, axes lists, clip
        bounds, shapes — require an initializer or ``Constant`` node:
        ONNX carries them as tensors but the IR carries them as
        attrs, and a computed (non-constant) parameter input has no
        honest spelling.
        """
        name = node.input[i] if i < len(node.input) else ""
        if name == "":
            raise UnsupportedOnnxOpError(
                node.op_type,
                f"input {i} is absent — the parameter the IR needs "
                "as an attribute has no value",
            )
        arr = self.initializers.get(name)
        if arr is None:
            arr = self._constants.get(name)
        if arr is None:
            raise UnsupportedOnnxOpError(
                node.op_type,
                f"input {i} ({name!r}) is not a constant — the IR "
                "spells this parameter as an attribute, so a computed "
                "value cannot be lifted",
            )
        return arr

    def const_scalar(self, node: onnx.NodeProto, i: int) -> Any:
        """Return input ``i`` as a Python scalar, or refuse."""
        arr = self.const(node, i)
        if arr.size != 1:
            raise UnsupportedOnnxOpError(
                node.op_type,
                f"input {i} must be a single-element constant, got "
                f"shape {tuple(arr.shape)}",
            )
        return arr.reshape(()).item()

    def leaf_scalar(self, name: str) -> Any:
        """Return the scalar payload of a constant *value name*, or ``None``.

        Used where a *tensor* operand may still be a constant (a
        scalar ``Gather`` index reads as ATen's ``select`` when it is
        literal).  ``None`` means the value is not constant or is not
        a single element — never an error, unlike :meth:`const`.
        """
        arr = self.initializers.get(name)
        if arr is None:
            arr = self._constants.get(name)
        if arr is None or arr.size != 1:
            return None
        return arr.reshape(()).item()

    # -- minting ---------------------------------------------------------

    def mint(self, op: str, *args: Any, **attrs: Any) -> Op:
        """Mint an IR term under the ``ATTR_SCHEMA`` contract."""
        return Op.make(op, *args, **attrs)

    def refuse(self, node: onnx.NodeProto, detail: str) -> NoReturn:
        """Decline ``node`` with its op type and a specific reason."""
        raise UnsupportedOnnxOpError(
            node.op_type,
            f"node {node.name or node.output[0]!r}: {detail}",
        )

    # -- the walk --------------------------------------------------------

    def _declare_inputs(self) -> None:
        """Type every graph input — ``Var`` or defaulted ``Param``."""
        for vi in self.graph.input:
            if vi.name in self.initializers:
                # A graph input shadowed by an initializer is a
                # parameter with a default, not a program input.
                self._param(vi.name)
                continue
            typed = _tensor_shape(vi)
            if typed is None:
                raise OnnxImportError(
                    f"graph input {vi.name!r} has no tensor type — "
                    "the IR types inputs by shape, so an unshaped or "
                    "non-tensor input cannot be lifted"
                )
            shape, dtype = typed
            var = Var(vi.name, TensorType(tuple(shape)))
            self.inputs.append(var)
            self._input_names.add(vi.name)
            self._env[vi.name] = var
            self.input_dtypes[vi.name] = dtype

    def _scan_constants(self) -> None:
        """Index ``Constant`` node outputs for lazy leaf materialisation.

        Only the value spellings that decode to a NumPy array are
        registered; an unmappable ``Constant`` (a string or sparse
        tensor) registers nothing and is refused when a consumer
        resolves its output, like any other undefined producer.
        """
        for node in self.graph.node:
            if node.op_type != "Constant" or not node.output:
                continue
            for name in (
                "value",
                "value_float",
                "value_int",
                "value_floats",
                "value_ints",
            ):
                value = self.attr(node, name, _UNMAPPABLE)
                if value is _UNMAPPABLE:
                    continue
                arr = (
                    value
                    if isinstance(value, np.ndarray)
                    else np.asarray(value)
                )
                self._constants[node.output[0]] = arr
                break

    def _lift_node(self, node: onnx.NodeProto) -> None:
        """Lift one node through the table; bind its outputs."""
        if node.op_type == "Constant":
            # Constants were pre-scanned; their term materialises on
            # first use.  A Constant that is never consumed mints
            # nothing — it was never part of the program.
            return
        lifter = _lifters().get(node.op_type)
        if lifter is None:
            raise UnsupportedOnnxOpError(
                node.op_type,
                f"node {node.name or node.output[0]!r}: no ONNX -> "
                "IR lifting for this op — the model is refused "
                "rather than silently narrowed (ADR 0006)",
            )
        bound = lifter(self, node)
        for name, term in bound.items():
            if name:
                self._env[name] = term

    def lift(self) -> LiftResult:
        """Walk the graph and return the import record."""
        self._declare_inputs()
        self._scan_constants()
        for node in self.graph.node:
            self._lift_node(node)
        outputs: list[Any] = []
        for vi in self.graph.output:
            typed = _tensor_shape(vi)
            if typed is None:
                raise OnnxImportError(
                    f"graph output {vi.name!r} is not a tensor (or "
                    "has no element type) — the IR's product "
                    "structure carries tensor outputs only"
                )
            term = self._env.get(vi.name)
            if term is None:
                if vi.name in self.initializers:
                    term = self._param(vi.name)
                elif vi.name in self._constants:
                    term = self._const_leaf(vi.name)
                else:
                    raise OnnxImportError(
                        f"graph output {vi.name!r} has no producer — "
                        "the graph is ill-formed"
                    )
            outputs.append(term)
        root = (
            outputs[0]
            if len(outputs) == 1
            else Op.make("pack", *outputs)
        )
        return LiftResult(
            IR(
                root=root,
                inputs=self.inputs,
                input_names=self._input_names,
                params=self.params,
            ),
            self.leaves,
            self.input_dtypes,
        )


def _lifters() -> Mapping[str, OnnxLifter]:
    """Return the ONNX -> IR op table (imported lazily, like the sink's)."""
    from catopt_onnx.liftmap import ONNX_LIFTS

    return ONNX_LIFTS


def _collect_value_info(model: onnx.ModelProto) -> dict[str, tuple]:
    """Run ONNX shape inference; index every typed value by name.

    Declared inputs/outputs carry their own types; inference fills
    the intermediates (and confirms the declared ones).  The
    initializers' concrete shapes come from their arrays directly.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=False
        )
    except Exception:
        inferred = model
    graph = inferred.graph
    table: dict[str, tuple] = {}
    for vi in (
        list(graph.input) + list(graph.value_info) + list(graph.output)
    ):
        typed = _tensor_shape(vi)
        if typed is not None:
            table[vi.name] = typed
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        name = _NUMPY_TO_NAME.get(arr.dtype)
        if name is not None:
            table[init.name] = (tuple(arr.shape), name)
    return table


def _model_opset(model: onnx.ModelProto) -> int:
    """Return the default-domain opset the model declares."""
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    return 0


def lift_model(model: onnx.ModelProto | str) -> LiftResult:
    """Lift an ONNX model (or a ``.onnx`` path) to a ``LiftResult``.

    The opset window is pinned: below 13 the parameter-as-input
    convention the attr folds rely on does not hold; above 21 the
    table has not been measured.  Non-default-domain nodes
    (``ai.onnx.ml``, vendor extensions) are refused by the same
    mechanism that refuses unmapped op types.
    """
    if isinstance(model, str):
        model = onnx.load(model)
    if not isinstance(model, onnx.ModelProto):
        raise OnnxImportError(
            f"expected onnx.ModelProto (or a path to one), got "
            f"{type(model).__name__}"
        )
    opset = _model_opset(model)
    if not MIN_OPSET <= opset <= MAX_OPSET:
        raise OnnxImportError(
            f"model declares opset {opset}; the lifting table covers "
            f"{MIN_OPSET}..{MAX_OPSET} — opset drift is refused, not "
            "guessed (ADR 0006)"
        )
    graph = model.graph
    initializers = {
        init.name: numpy_helper.to_array(init)
        for init in graph.initializer
    }
    value_info = _collect_value_info(model)
    lifter = GraphLifter(
        graph,
        initializers=initializers,
        value_info=value_info,
    )
    return lifter.lift()
