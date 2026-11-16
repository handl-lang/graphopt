"""ONNX codec-table drift guards — ADR-0178 WP0a.

``kit_onnx.opmap`` (IR → ONNX, ~169 entries) and
``kit_onnx.liftmap`` (ONNX → IR, ~107 entries) were the last
unguarded codec tables (``host/project/reports/
p3_codec_consolidation_spike.md`` §1b).  This file guards them the
way ``test_surface_table_drift.py`` guards the hnl/modelproj/torch
tables — behavioral probes against the real emitters, not parallel
name tables:

* **Covered-set pins.**  ``ONNX_BINDINGS`` / ``ONNX_LIFTS`` key sets
  are pinned literally — a new mapping is a loud diff, and the
  ``_PROBE`` table below must grow with it (the completeness assert
  fails otherwise).
* **IR-op reality.**  Every IR op name either table claims
  (``ONNX_BINDINGS`` keys; the names ``ONNX_LIFTS`` lifters mint —
  recovered by AST-scanning the module's ``g.mint("...")`` call
  sites plus the four declarative tables) must resolve to real IR
  vocabulary: ``opmeta.REGISTRY`` | ``HOST_BINDINGS`` |
  ``_IR_TO_TORCH`` | ``ATEN_TO_IR`` targets (set union).  Names with no
  ``tensor.*`` host port must sit in the pinned ``_NON_HOST_OPS``
  exception list — adapter/ONNX-only vocabulary
  (``relu_``/``hstack``/``pack``/the dtype-cast spellings/…), the
  same posture as ``_TORCH_ONLY_ATTR_OPS`` in
  ``test_surface_table_drift.py``.
* **ONNX-schema reality + arity.**  Every ``ONNX_LIFTS`` key is a
  real ONNX ``op_type`` (``onnx.defs`` schema in the default domain
  within the lift opset window).  Every ``ONNX_BINDINGS`` builder is
  *run* against a canonical probe term through the real
  ``GraphBuilder``; each emitted node's ``op_type`` must have an
  opset-18 schema (the sink's pinned opset) and its input count must
  sit inside the schema's ``min_input``/``max_input`` — the same
  arity discipline ``_check_emitted`` applies to ``tensor.*``
  signatures.  The declarative lift tables get the mirror-image
  check: ``_UNARY`` ops are exactly 1-input, ``_BINARY`` exactly
  2-input, ``_VARIADIC`` genuinely variadic, ``_REDUCERS`` never
  more than data+axes inputs across the whole opset window — so a
  malformed node cannot mint an IR op of the wrong arity silently.

Why no IR-arity ↔ host-port arity comparison (the ``_sig_arity``
half of the sibling guard): IR ops carry dims/axes/shapes as *attrs*
while ``tensor.*`` ports take them as *positional* arguments, so the
counts are not comparable without per-op lowering knowledge — the
ONNX-schema bounds above are the honest arity contract for these
tables.  Host-port *resolution* (existence + emittable status) is
still asserted.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from kit_host_tensor.surface import host_ops
from mlopt_core.ir import IR, Const, Op, TensorType, Var
from onnx import defs

from tests.test_surface_table_drift import _EMITTABLE

pytest.importorskip("onnx")

_OPSET = 18  # OnnxSink's pinned opset (sink.py default)
_OPSET_WINDOW = range(13, 22)  # lift.py MIN_OPSET..MAX_OPSET

_T = TensorType((2, 3))


# ---------------------------------------------------------------------------
#  shared oracles
# ---------------------------------------------------------------------------


def _ir_op_oracle() -> frozenset[str]:
    """Every name the stack treats as a real IR op.

    ``opmeta.REGISTRY`` is core's declared-op table; ``HOST_BINDINGS``
    / ``_IR_TO_TORCH`` are the two sinks' lowering tables; ``ATEN_TO_IR``
    targets are the generated aten→IR classification — together they
    are the IR vocabulary an ONNX map may legitimately name.
    """
    from kit_host_tensor.bindings import HOST_BINDINGS
    from mlopt_core.opmeta import REGISTRY
    from mlopt_torch.aten_classify import ATEN_TO_IR
    from mlopt_torch.torch_bindings import _IR_TO_TORCH

    return (
        frozenset(REGISTRY)
        | frozenset(HOST_BINDINGS)
        | frozenset(_IR_TO_TORCH)
        | frozenset(ATEN_TO_IR.values())
    )


#: IR-op names the ONNX tables use that have NO emittable ``tensor.*``
#: host port — adapter/ONNX-only vocabulary.  Pinned so a *new* name
#: must either land a host port or join this list deliberately (same
#: posture as ``_TORCH_ONLY_ATTR_OPS``).
_NON_HOST_OPS = frozenset(
    {
        "_assert_tensor_metadata",
        "addcdiv",
        "addcmul",
        "alias",
        "bitwise_and",
        "bitwise_or",
        "bool",
        "broadcast",
        "broadcast_to",
        "byte",
        "char",
        "conv1d",
        "detach_",
        "diag_sum",
        "double",
        "dropout",
        "embedding",
        "float",
        "getitem",
        "group_norm",
        "half",
        "hardsigmoid",
        "hardswish",
        "hstack",
        "instance_norm",
        "int",
        "linalg_vector_norm",
        "long",
        "masked_fill",
        "max_pool2d",
        "movedim",
        "new_full",
        "new_ones",
        "new_zeros",
        "one_hot",
        "pack",
        "pad",
        "permute",
        "relu6",
        "relu_",
        "rsub",
        "short",
        "softsign",
        "take_along_dim",
    }
)


# ---------------------------------------------------------------------------
#  opmap — IR -> ONNX (ONNX_BINDINGS, 169 entries)
# ---------------------------------------------------------------------------

_ONNX_BINDINGS_OPS = frozenset(
    {
        "_assert_tensor_metadata",
        "abs",
        "acos",
        "acosh",
        "add",
        "addcdiv",
        "addcmul",
        "alias",
        "all",
        "amax",
        "amin",
        "any",
        "arange",
        "argmax",
        "argmin",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "batch_norm",
        "bitwise_and",
        "bitwise_or",
        "bool",
        "broadcast",
        "broadcast_to",
        "byte",
        "ceil",
        "char",
        "chunk",
        "clamp",
        "clamp_max",
        "clamp_min",
        "clone",
        "concat",
        "contiguous",
        "conv1d",
        "conv2d",
        "cos",
        "cosh",
        "count_nonzero",
        "cumsum",
        "detach",
        "detach_",
        "diag_sum",
        "div",
        "double",
        "dropout",
        "elu",
        "embedding",
        "eq",
        "erf",
        "erfc",
        "exp",
        "exp2",
        "expand",
        "expand_as",
        "expm1",
        "flatten",
        "flip",
        "float",
        "floor",
        "fmod",
        "frac",
        "full",
        "full_like",
        "gather",
        "ge",
        "gelu",
        "getitem",
        "glu",
        "gt",
        "half",
        "hardsigmoid",
        "hardswish",
        "hardtanh",
        "heaviside",
        "hstack",
        "index_select",
        "instance_norm",
        "int",
        "isfinite",
        "isinf",
        "isnan",
        "layer_norm",
        "le",
        "leaky_relu",
        "lerp",
        "linalg_vector_norm",
        "linear",
        "log",
        "log10",
        "log1p",
        "log2",
        "log_softmax",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "long",
        "lt",
        "masked_fill",
        "matmul",
        "max",
        "maximum",
        "mean",
        "min",
        "minimum",
        "mish",
        "movedim",
        "mul",
        "narrow",
        "ne",
        "neg",
        "new_full",
        "new_ones",
        "new_zeros",
        "nonzero",
        "one_hot",
        "ones",
        "ones_like",
        "pack",
        "pad",
        "permute",
        "pow",
        "prelu",
        "prod",
        "reciprocal",
        "relu",
        "relu6",
        "relu_",
        "remainder",
        "repeat",
        "reshape",
        "rms_norm",
        "round",
        "rsqrt",
        "rsub",
        "select",
        "selu",
        "short",
        "sigmoid",
        "sign",
        "silu",
        "sin",
        "sinh",
        "slice",
        "softmax",
        "softplus",
        "softsign",
        "split",
        "sqrt",
        "square",
        "squeeze",
        "stack",
        "sub",
        "sum",
        "take_along_dim",
        "tan",
        "tanh",
        "to",
        "transpose",
        "tril",
        "triu",
        "trunc",
        "unsqueeze",
        "where",
        "xlogy",
        "zeros",
        "zeros_like",
    }
)


# ---------------------------------------------------------------------------
#  opmap probe table — canonical terms per op
# ---------------------------------------------------------------------------
#
#  Arg spec letters: ``v`` = Var((2,3)), ``V``-with-shape via tuple,
#  ``c`` = Const scalar.  Attrs may be a dict or a list of dicts
#  (multiple probes — e.g. both spellings a builder accepts).
#  ``pack`` never emits a node by design (multi-output root);
#  ``getitem`` needs a multi-output producer emitted first.

_PROBE_ARGS: dict[str, tuple[Any, ...]] = {}
_PROBE_ATTRS: dict[str, Any] = {}


def _p(name: str, args: tuple[Any, ...], attrs: Any = None) -> None:
    """Register a probe spec: *args* letters, *attrs* dict(s)."""
    _PROBE_ARGS[name] = args
    if attrs is None:
        _PROBE_ATTRS[name] = [{}]
    elif isinstance(attrs, list):
        _PROBE_ATTRS[name] = attrs
    else:
        _PROBE_ATTRS[name] = [attrs]


# -- bespoke lowerings -------------------------------------------------------

_p("matmul", ("v", "v"))
_p("linear", ("v", "v", "v"))
_p("dropout", ("v",))
for _op in ("fmod", "remainder", "rsub", "ne", "xlogy", "heaviside"):
    _p(_op, ("v", "v"))
_p(
    "div",
    ("v", "v"),
    [{}, {"rounding_mode": "floor"}, {"rounding_mode": "trunc"}],
)
_p("zeros_like", ("v",))
_p("ones_like", ("v",))
_p("full_like", ("v", "c"))
for _op in ("zeros", "ones"):
    _p(_op, (), {"shape": [2, 3]})
_p("full", (), {"shape": [2, 3], "fill_value": 1.5})
for _op in ("new_zeros", "new_ones"):
    _p(_op, ("v",), {"shape": [2, 3]})
_p("new_full", ("v",), {"shape": [2, 3], "fill_value": 2.0})
_p("nonzero", ("v",))
for _op in (
    "trunc",
    "frac",
    "rsqrt",
    "square",
    "exp2",
    "expm1",
    "log1p",
    "log2",
    "log10",
    "asinh",
    "acosh",
    "atanh",
    "erfc",
    "isfinite",
    "hardtanh",
    "relu6",
    "elu",
    "leaky_relu",
    "softplus",
    "gelu",
    "silu",
    "mish",
    "hardsigmoid",
    "hardswish",
    "transpose",
    "movedim",
    "flatten",
    "unsqueeze",
    "squeeze",
    "slice",
    "narrow",
    "select",
    "chunk",
    "argmax",
    "argmin",
    "cumsum",
    "diag_sum",
    "count_nonzero",
    "any",
    "all",
    "rms_norm",
    "triu",
    "tril",
):
    _p(_op, ("v",))
_p("glu", ((2, 4),))
_p("argmax", ("v",), [{}, {"dim": 0, "keepdim": False}])
_p("argmin", ("v",), [{}, {"dim": 0, "keepdim": False}])
_p("prelu", ("v", "v"))
for _op in ("softmax", "log_softmax"):
    _p(_op, ("v",), {"dim": 0})
_p("permute", ("v",), {"dim": [1, 0]})
_p("reshape", ("v",), {"shape": [6]})
_p("expand", ("v",), {"shape": [2, 3]})
_p("expand_as", ("v", "v"))
_p("broadcast_to", ("v",), {"shape": [4, 3]})
_p("repeat", ("v",), {"shape": [2, 1]})
_p("flip", ("v",), {"dims": [0]})
_p("pad", ("v",), {"pad": [1, 1]})
_p("concat", ("v", "v"), {"dim": 0})
for _op in ("stack", "hstack"):
    _p(_op, ("v", "v"))
_p(
    "split",
    ("v",),
    {"sizes": [1, 2], "dim": -1, "index": 0},
)
for _op in ("max", "min"):
    _p(_op, ("v",), {"dim": 0, "keepdim": True})
_p("lerp", ("v", "v", "v"))
_p("where", ("v", "v", "v"))
_p("masked_fill", ("v", "v", "c"))
for _op in ("addcmul", "addcdiv"):
    _p(_op, ("v", "v", "v", "c"))
_p("clamp", ("v",), {"min": 0.0, "max": 1.0})
_p("clamp_min", ("v",), {"min": 0.0})
_p("clamp_max", ("v",), {"max": 1.0})
_p("layer_norm", ("v", "v", "v"), {"dim": [3]})
_p("batch_norm", ("v", "v", "v", "v", "v"))
_p("instance_norm", ("v", "v", "v"))
_p("conv1d", ("v", (4, 3, 3)))
_p("conv2d", ("v", (4, 3, 3, 3)))
for _op in ("index_select", "embedding", "gather", "take_along_dim"):
    _p(_op, ("v", "v"))
_p("one_hot", ("v",), {"num_classes": 4})
_p("linalg_vector_norm", ("v",), [{}, {"ord": 1}])
_p("arange", (), {"arg0": 5})
_p("to", ("v",), {"dtype": "float64"})
_p("pack", ())
_p("getitem", ("v",), {"index": 1})


def _declarative_specs() -> dict[
    str, tuple[tuple[Any, ...], list[dict]]
]:
    """Probe specs for the table-driven entries (no bespoke code).

    ``_UNARY``/``_UNARY_BOOL`` → 1 operand; ``_BINARY``/``_COMPARISON``
    → 2; ``_REDUCERS`` → 1 plus a ``dim`` variant exercising the
    axes-as-input spelling; ``_CASTS`` → 1.
    """
    from kit_onnx.opmap import (
        _BINARY,
        _CASTS,
        _COMPARISON,
        _REDUCERS,
        _UNARY,
        _UNARY_BOOL,
    )

    specs: dict[str, tuple[tuple[Any, ...], list[dict]]] = {}
    for name in set(_UNARY) | set(_UNARY_BOOL) | set(_CASTS):
        specs[name] = (("v",), [{}])
    for name in set(_BINARY) | set(_COMPARISON):
        specs[name] = (("v", "v"), [{}])
    for name in set(_REDUCERS):
        specs[name] = (("v",), [{}, {"dim": 0, "keepdim": True}])
    return specs


#: Builders that cannot emit under a bare probe — ``pack`` is the
#: multi-output root (declines by design, ``opmap._pack``) and
#: ``getitem`` only lowers as a projection off a multi-output op.
_NEVER_NODE_EMITS = frozenset({"pack", "getitem"})


def _probe_term(name: str, args: tuple[Any, ...], attrs: dict) -> Op:
    """Build the canonical probe term for *name*.

    Arg specs: ``"v"`` → ``Var((2,3))``, ``"c"`` → ``Const`` scalar,
    a shape tuple → ``Var`` of that shape.
    """
    terms: list[Any] = []
    for spec in args:
        if spec == "v":
            terms.append(Var(f"v{len(terms)}", _T))
        elif spec == "c":
            terms.append(Const(1.0))
        else:
            terms.append(Var(f"v{len(terms)}", TensorType(tuple(spec))))
    return Op.make(name, *terms, validate=False, **attrs)


def _probe_ir() -> IR:
    """A throwaway IR to hang the ``GraphBuilder`` on."""
    return IR(root=Op.make("relu", Var("x", _T)))


def _emitted_nodes(name: str) -> list[tuple[str, int]]:
    """Run *name*'s real builder; return ``(op_type, n_inputs)`` pairs.

    ``getitem`` is exercised against a ``max`` producer emitted in
    the same builder first — the only spelling it lowers.
    """
    from kit_onnx.lower import GraphBuilder
    from kit_onnx.opmap import ONNX_BINDINGS

    if name in _PROBE_ARGS:
        args, attr_variants = _PROBE_ARGS[name], _PROBE_ATTRS[name]
    else:
        args, attr_variants = _declarative_specs()[name]

    emitted: list[tuple[str, int]] = []
    for attrs in attr_variants:
        gb = GraphBuilder(_probe_ir())
        if name == "getitem":
            # getitem lowers only as a projection off a multi-output
            # op — emit the producer first so ``declare_multi`` has
            # run, exactly as ``lower_ir`` would have done, and reuse
            # the same term object (``_multi`` is term-keyed).
            producer = Op.make(
                "max", Var("p", _T), dim=0, validate=False
            )
            ONNX_BINDINGS["max"](gb, producer, ["pin0"])
            term = Op.make("getitem", producer, index=1, validate=False)
            ONNX_BINDINGS["getitem"](gb, term, ["pin0"])
        else:
            term = _probe_term(name, args, attrs)
            ins = [f"in{i}" for i in range(len(term.args))]
            ONNX_BINDINGS[name](gb, term, ins)
        emitted.extend((n.op_type, len(n.input)) for n in gb.nodes)
    return emitted


class TestOnnxBindingsTable:
    def test_covered_op_set_is_pinned(self) -> None:
        """``supported_ops`` is exactly this set — additions are loud."""
        from kit_onnx.opmap import ONNX_BINDINGS

        assert frozenset(ONNX_BINDINGS) == _ONNX_BINDINGS_OPS

    def test_probe_table_covers_every_binding(self) -> None:
        """Every binding has a probe spec (or a named exception)."""
        from kit_onnx.opmap import (
            _BINARY,
            _CASTS,
            _COMPARISON,
            _REDUCERS,
            _UNARY,
            _UNARY_BOOL,
        )

        declared = (
            set(_UNARY)
            | set(_UNARY_BOOL)
            | set(_BINARY)
            | set(_COMPARISON)
            | set(_REDUCERS)
            | set(_CASTS)
        )
        probed = declared | set(_PROBE_ARGS) | _NEVER_NODE_EMITS
        from kit_onnx.opmap import ONNX_BINDINGS

        assert set(ONNX_BINDINGS) == probed, (
            "ONNX_BINDINGS gained/lost ops without a probe spec: "
            f"{sorted(set(ONNX_BINDINGS) ^ probed)}"
        )

    def test_every_key_is_a_real_ir_op(self) -> None:
        """Forward guard — each claimed IR op exists in the vocabulary."""
        from kit_onnx.opmap import ONNX_BINDINGS

        oracle = _ir_op_oracle()
        missing = sorted(set(ONNX_BINDINGS) - oracle)
        assert not missing, (
            "ONNX_BINDINGS names ops that are not real IR vocabulary: "
            f"{missing}"
        )

    def test_keys_resolve_to_host_ports_or_declared_exceptions(
        self,
    ) -> None:
        """Each key names an emittable ``tensor.*`` port, or is pinned.

        ONNX is a different backend — a missing host port is not a
        lowering bug — but the *claim* must be deliberate: ops the
        host-tensor surface cannot spell join ``_NON_HOST_OPS`` by
        name, so a typo'd or renamed op fails here.
        """
        from kit_onnx.opmap import ONNX_BINDINGS

        ops = host_ops()
        assert ops, "host_surface.json loaded empty — dep broke"
        missing = sorted(
            op
            for op in ONNX_BINDINGS
            if f"tensor.{op}" not in ops
            or ops[f"tensor.{op}"]["status"] not in _EMITTABLE
        )
        unexpected = set(missing) - _NON_HOST_OPS
        assert not unexpected, (
            "ONNX_BINDINGS names ops with no emittable tensor.* "
            "port and no declared exception: "
            f"{sorted(unexpected)}"
        )

    def test_emitted_onnx_ops_and_arity(self) -> None:
        """Every emitted node is a real ONNX op within schema arity.

        The builders are run for real — a retargeted or misspelled
        ``op_type`` fails ``defs.get_schema``, and an input count
        outside ``min_input``/``max_input`` fails the arity assert.
        """
        from kit_onnx.errors import UnsupportedOnnxOpError
        from kit_onnx.opmap import ONNX_BINDINGS

        for name in sorted(ONNX_BINDINGS):
            if name in _NEVER_NODE_EMITS:
                if name == "pack":
                    # pack is the multi-output root — it refuses as a
                    # value by design; assert the refusal is the
                    # documented one, not an accidental crash.
                    with pytest.raises(UnsupportedOnnxOpError):
                        _emitted_nodes(name)
                else:  # getitem — covered by its dedicated probe below
                    nodes = _emitted_nodes(name)
                    assert nodes, "getitem probe emitted no nodes"
                continue
            emitted = _emitted_nodes(name)
            assert emitted, f"{name}: probe emitted no ONNX nodes"
            for op_type, n_inputs in emitted:
                try:
                    schema = defs.get_schema(op_type, _OPSET, "")
                except defs.SchemaError:
                    schema = None
                assert schema is not None, (
                    f"{name}: emits {op_type} — no opset-{_OPSET} "
                    "schema in the default domain"
                )
                assert (
                    schema.min_input <= n_inputs <= schema.max_input
                ), (
                    f"{name}: {op_type} node carries {n_inputs} "
                    f"inputs; schema wants "
                    f"{schema.min_input}..{schema.max_input}"
                )


def _declarative_names() -> set[str]:
    from kit_onnx.opmap import (
        _BINARY,
        _CASTS,
        _COMPARISON,
        _REDUCERS,
        _UNARY,
        _UNARY_BOOL,
    )

    return (
        set(_UNARY)
        | set(_UNARY_BOOL)
        | set(_BINARY)
        | set(_COMPARISON)
        | set(_REDUCERS)
        | set(_CASTS)
    )


# ---------------------------------------------------------------------------
#  liftmap — ONNX -> IR (ONNX_LIFTS, 107 entries)
# ---------------------------------------------------------------------------

_ONNX_LIFTS_OPS = frozenset(
    {
        "Abs",
        "Acos",
        "Acosh",
        "Add",
        "And",
        "ArgMax",
        "ArgMin",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BatchNormalization",
        "BitwiseAnd",
        "BitwiseOr",
        "Cast",
        "CastLike",
        "Ceil",
        "Celu",
        "Clip",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Cosh",
        "CumSum",
        "Div",
        "Dropout",
        "Einsum",
        "Elu",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Flatten",
        "Floor",
        "Gather",
        "GatherElements",
        "Gelu",
        "Gemm",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "Greater",
        "GreaterOrEqual",
        "GroupNormalization",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "InstanceNormalization",
        "IsInf",
        "IsNaN",
        "LayerNormalization",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "Log",
        "LogSoftmax",
        "MatMul",
        "Max",
        "MaxPool",
        "Mean",
        "Min",
        "Mish",
        "Mod",
        "Mul",
        "Neg",
        "NonZero",
        "Not",
        "OneHot",
        "Or",
        "Pad",
        "Pow",
        "Range",
        "Reciprocal",
        "ReduceL1",
        "ReduceL2",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Round",
        "Selu",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Slice",
        "Softmax",
        "Softplus",
        "Softsign",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Sum",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
        "Tile",
        "TopK",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
        "Xor",
    }
)


def _minted_ir_ops() -> frozenset[str]:
    """Every IR op name ``liftmap`` can mint.

    Recovered from the module's own source: string literals passed as
    the first arg of ``g.mint(...)`` calls, plus the IR targets of the
    four declarative tables (``_simple``/``_fold``/``_reducer``/
    ``_vector_norm``/``_arg_reducer``/``_global_pool``/``_trilu``/
    ``_softmax`` close over their argument).  An AST scan — not a
    parallel table — so a new ``g.mint("typo", ...)`` is caught by
    the oracle check below.
    """
    import kit_onnx.liftmap as lm

    path = Path(lm.__file__)
    tree = ast.parse(path.read_text())
    minted: set[str] = set()

    # The lifter factories close over their IR-op argument
    # (``_simple(ir_op)`` & friends) — the minted name arrives as a
    # call argument, not a ``g.mint`` literal, so collect string
    # first-args of the factory calls too.
    _FACTORIES = {
        "_simple",
        "_fold",
        "_reducer",
        "_arg_reducer",
        "_softmax",
        "_global_pool",
    }

    class _MintVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else ""
            )
            if (
                (name == "mint" or name in _FACTORIES)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                minted.add(node.args[0].value)
            self.generic_visit(node)

    _MintVisitor().visit(tree)
    for table in (lm._UNARY, lm._BINARY, lm._VARIADIC, lm._REDUCERS):
        minted.update(table.values())
    return frozenset(minted)


def _schema_in_window(op_type: str) -> bool:
    """Whether *op_type* has a default-domain schema in the lift window."""
    for v in _OPSET_WINDOW:
        try:
            defs.get_schema(op_type, v, "")
            return True
        except defs.SchemaError:
            continue
    return False


def _schema_at(op_type: str, opset: int) -> Any:
    return defs.get_schema(op_type, opset, "")


class TestOnnxLiftsTable:
    def test_covered_op_set_is_pinned(self) -> None:
        """``ONNX_LIFTS`` keys are exactly this set — additions loud."""
        from kit_onnx.liftmap import ONNX_LIFTS

        assert frozenset(ONNX_LIFTS) == _ONNX_LIFTS_OPS

    def test_every_key_is_a_real_onnx_op(self) -> None:
        """Forward guard — each lifted ``op_type`` has a real schema."""
        from kit_onnx.liftmap import ONNX_LIFTS

        missing = sorted(
            k for k in ONNX_LIFTS if not _schema_in_window(k)
        )
        assert not missing, (
            "ONNX_LIFTS names op_types with no ONNX schema in the "
            f"default domain within opsets {_OPSET_WINDOW.start}.."
            f"{_OPSET_WINDOW.stop - 1}: {missing}"
        )

    def test_minted_ir_ops_are_real(self) -> None:
        """Every IR op the lifters mint resolves to real vocabulary."""
        oracle = _ir_op_oracle()
        missing = sorted(_minted_ir_ops() - oracle)
        assert not missing, (
            "liftmap mints IR ops that are not real vocabulary: "
            f"{missing}"
        )

    def test_minted_ir_ops_host_port_or_declared(self) -> None:
        """Minted names with no host port sit in ``_NON_HOST_OPS``."""
        ops = host_ops()
        assert ops, "host_surface.json loaded empty — dep broke"
        missing = sorted(
            op
            for op in _minted_ir_ops()
            if f"tensor.{op}" not in ops
            or ops[f"tensor.{op}"]["status"] not in _EMITTABLE
        )
        unexpected = set(missing) - _NON_HOST_OPS
        assert not unexpected, (
            "liftmap mints ops with no emittable tensor.* port and "
            f"no declared exception: {sorted(unexpected)}"
        )

    def test_declarative_arity_compat(self) -> None:
        """The table-driven lifts cannot mint a wrong-arity IR op.

        ``_simple`` mints the IR op over *all* of the node's inputs,
        so the ONNX schema's input bounds ARE the minted arity: a
        unary ONNX op must be strictly 1-input (a stray second input
        would mint ``neg(x, y)``), a binary strictly 2, a variadic
        fold genuinely variadic, and a reducer never more than
        data+axes.  Checked at opset 18 (the sink's pin) for the
        exact-arity tables and across the whole lift window for the
        reducers (axes moved attribute→input at opset 18).
        """
        from kit_onnx.liftmap import (
            _BINARY,
            _REDUCERS,
            _UNARY,
            _VARIADIC,
        )

        for op_type in _UNARY:
            s = _schema_at(op_type, _OPSET)
            assert (s.min_input, s.max_input) == (1, 1), (
                f"{op_type}: _UNARY lift mints a 1-arg IR op but the "
                f"schema allows {s.min_input}..{s.max_input} inputs"
            )
        for op_type in _BINARY:
            s = _schema_at(op_type, _OPSET)
            assert (s.min_input, s.max_input) == (2, 2), (
                f"{op_type}: _BINARY lift mints a 2-arg IR op but "
                f"the schema allows {s.min_input}..{s.max_input} "
                "inputs"
            )
        for op_type in _VARIADIC:
            s = _schema_at(op_type, _OPSET)
            assert s.max_input > 2, (
                f"{op_type}: _VARIADIC lift folds N inputs but the "
                f"schema caps at {s.max_input}"
            )
        for op_type in _REDUCERS:
            # Across the window the node may carry data only
            # (axes attr, opset <18) or data+axes (opset >=18); the
            # lifter reads both.  Never more.
            for v in _OPSET_WINDOW:
                try:
                    s = _schema_at(op_type, v)
                except defs.SchemaError:
                    continue
                assert s.min_input <= 1 and s.max_input <= 2, (
                    f"{op_type}@opset{v}: reducer schema allows "
                    f"{s.min_input}..{s.max_input} inputs — the "
                    "lifter consumes data + optional axes only"
                )
