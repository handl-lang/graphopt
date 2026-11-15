"""``HOST_BINDINGS`` — the IR-op → host ``tensor.*`` positional-arg table.

Each entry is a ``Binding``-shaped callable ``fn(*operand_values,
**attrs) -> list`` returning the *positional JSON args* for the
same-named ``tensor.*`` host op (the ATen-port arg order pinned by
``host-tensor/tools/aten_port_list.txt`` + the generated decoders in
``src/ops/aten_generated.rs`` / the handwritten impls in ``src/ops/``).
The executor JSON-encodes the list and calls the extension.

Operand values reaching a builder are already evaluated: ndarrays for
tensor terms, Python numbers for ``Const`` leaves, tuples for
multi-output terms.  Attr spellings follow the torch adapter's
convention — canonical names first, ``argN`` positional-attr
fallbacks where the op's ``ATTR_SCHEMA``/hand-minted terms produce
them (see ``mlopt_torch.torch_bindings.attr_of``).

Two result shapes exist:

* a plain ``list`` of JSON args — the executor calls the op and
  decodes the result;
* an :class:`_Indexed` marker — for ``chunk``/``split``/``unbind``,
  whose host port returns a JSON *array* of wire tensors while the IR
  spelling carries the selection as an ``index`` attr (the torch
  binding's ``...[index]``).  The executor resolves the marker.

Ops whose host port exists but whose IR semantics do *not* line up
(``split`` with an int ``sizes`` — torch's split-*size* vs the host's
split-*count*; ``var``/``std`` ``correction`` outside {0, 1}) raise
:class:`UnsupportedHostOpError` loudly inside the builder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mlopt_host_tensor.errors import UnsupportedHostOpError
from mlopt_host_tensor.wire import (
    scalar,
    scalar_or_tensor,
    tensor_arg,
)

__all__ = ["HOST_BINDINGS"]

_MISSING = object()


def _attr(kw: dict[str, Any], *names: str, default: Any = None) -> Any:
    """First present attr among ``names`` (canonical → ``argN``).

    Attr values that are ``Const`` leaves unwrap to their scalar —
    hand-minted terms spell ``min=Const(0)``.
    """
    from mlopt_core.ir import Const

    for name in names:
        if name in kw:
            v = kw[name]
            return v.value if isinstance(v, Const) else v
    return default


def _unsupported(op: str, why: str) -> UnsupportedHostOpError:
    return UnsupportedHostOpError(f"host-tensor: op {op!r} {why}")


def _num(v: Any) -> float | int:
    """Attr/operand → JSON number (bool → int, ndarray size-1 → item)."""
    if isinstance(v, np.ndarray):
        return scalar(v)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    raise _unsupported(
        "attr", f"expected a number, got {type(v).__name__}"
    )


def _int(v: Any) -> int:
    return int(_num(v))


def _dim_list(v: Any) -> int | list[int] | None:
    """Normalise a dim attr/operand to the host's int-or-int-list form."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [int(_num(d)) for d in v]
    return _int(v)


def _shape_list(v: Any) -> list[int]:
    """Normalise a shape attr/operand to a list of ints."""
    if isinstance(v, (list, tuple)):
        return [int(_num(d)) for d in v]
    if isinstance(v, np.ndarray):
        return [int(d) for d in v.reshape(-1)]
    raise _unsupported("shape", f"expected an int sequence, got {v!r}")


def _resolve_neg1(shape: list[int], numel: int, op: str) -> list[int]:
    """Resolve a single ``-1`` wildcard against the operand's numel."""
    if -1 not in shape:
        return shape
    if shape.count(-1) > 1:
        raise _unsupported(
            op, f"shape {shape} has multiple -1 wildcards"
        )
    known = 1
    for d in shape:
        if d != -1:
            known *= d
    if known == 0 or numel % known:
        raise _unsupported(
            op, f"cannot resolve -1 in shape {shape} for numel {numel}"
        )
    return [numel // known if d == -1 else d for d in shape]


def _shape_arg(vals: tuple, kw: dict, op: str) -> list[int]:
    """Read the target shape for creator-like ops.

    Canonical attr spellings ``shape``/``dim`` first (the torch
    binding's own order), then Const operands as the size list
    (hand-minted ``zeros(2, 3)``).
    """
    v = _attr(kw, "shape", "dim", default=_MISSING)
    if v is not _MISSING:
        return _shape_list(v)
    if vals:
        return [int(_num(d)) for d in vals]
    raise _unsupported(
        op, "missing shape (attr 'shape'/'dim' or operands)"
    )


def _dim_keep(vals: tuple, kw: dict, op: str) -> tuple[Any, bool]:
    """Shared ``(dim, keepdim)`` extraction for the reduce family.

    Extra positional operands fill ``(dim, keepdim)`` (the torch
    binding's ``_dim_args`` convention); otherwise the ``dim``/``axis``
    and ``keepdim`` attrs apply.
    """
    extra = vals[1:]
    if extra:
        dim = extra[0]
        keep = extra[1] if len(extra) > 1 else kw.get("keepdim", False)
    else:
        dim = _attr(kw, "dim", "axis")
        keep = kw.get("keepdim", False)
    return dim, bool(_num(keep) if keep is not None else False)


# ---------------------------------------------------------------------------
# Non-trivial builders
# ---------------------------------------------------------------------------


def _u1(_op: str) -> Callable[..., list]:
    """Plain unary: ``[tensor]``."""
    return lambda t, *a, **kw: [tensor_arg(t)]


def _u2(_op: str) -> Callable[..., list]:
    """Plain binary, both tensor slots (scalars wrap to 0-d)."""
    return lambda a, b, *r, **kw: [tensor_arg(a), tensor_arg(b)]


def _bin_scalar_rhs(_op: str) -> Callable[..., list]:
    """Binary with a ``Scalar | Tensor`` rhs (comparisons, fmod…)."""
    return lambda a, b, *r, **kw: [
        tensor_arg(a),
        scalar_or_tensor(b),
    ]


def _reduce(_op: str) -> Callable[..., list]:
    """``(self, dim?, keepdim?)`` — sum/prod/mean/amax/amin/nan*."""

    def build(t: Any, *vals: Any, **kw: Any) -> list:
        dim, keep = _dim_keep((t, *vals), kw, _op)
        return [tensor_arg(t), _dim_list(dim), keep]

    return build


def _clamp_args(t: Any, a: tuple, kw: dict) -> list:
    """``clamp(self, min?, max?)`` — bounds are Scalar?/Tensor?."""
    lo = _attr(kw, "min")
    hi = _attr(kw, "max")
    argpos = kw.get("argpos")
    if argpos is None:
        # Positional operands fill min then max; attrs win.
        if len(a) > 2:
            raise _unsupported("clamp", f"got {len(a)} bound operands")
        if lo is None and a:
            lo = a[0]
        if hi is None and len(a) > 1:
            hi = a[1]
    else:
        if len(a) != len(argpos):
            raise _unsupported(
                "clamp", "argpos/tensor-bound arity mismatch"
            )
        for pos, v in zip(argpos, a, strict=True):
            if pos == 1:
                lo = v
            elif pos == 2:
                hi = v
            else:
                raise _unsupported("clamp", f"bad argpos entry {pos}")

    def bound(v: Any) -> Any:
        return None if v is None else scalar_or_tensor(v)

    return [tensor_arg(t), bound(lo), bound(hi)]


def _max_min_args(t: Any, a: tuple, kw: dict, want_max: bool) -> list:
    """``max``/``min`` reduction and elementwise spellings.

    All-reduce, dim-reduce (tuple out), or the elementwise
    ``(self, other)`` spelling per the torch binding.
    """
    op = "max" if want_max else "min"
    if a:
        if len(a) != 1 or kw:
            raise _unsupported(
                op,
                "elementwise spelling takes exactly one operand and "
                "no attrs (use 'maximum'/'minimum')",
            )
        return [tensor_arg(t), tensor_arg(a[0])]
    dim = _attr(kw, "dim", "axis")
    if dim is None:
        return [tensor_arg(t)]
    keep = bool(_attr(kw, "keepdim", default=False))
    return [tensor_arg(t), _dim_list(dim), keep]


def _arg_reduce(t: Any, a: tuple, kw: dict, want_max: bool) -> list:
    """``argmax``/``argmin(self, dim?, keepdim?)`` — dim is one int."""
    op = "argmax" if want_max else "argmin"
    dim, keep = _dim_keep((t, *a), kw, op)
    dj = _dim_list(dim)
    if isinstance(dj, list):
        raise _unsupported(op, "dim must be a single int, not a tuple")
    return [tensor_arg(t), dj, keep]


def _all_any(t: Any, a: tuple, kw: dict, want_all: bool) -> list:
    """``all``/``any(self, dims?, keepdim?)`` — int or int-list dims."""
    op = "all" if want_all else "any"
    dim, keep = _dim_keep((t, *a), kw, op)
    return [tensor_arg(t), _dim_list(dim), keep]


def _var_std(t: Any, a: tuple, kw: dict, want_std: bool) -> list:
    """``var``/``std`` — the host driver has NO ``correction`` slot.

    ``correction`` counts Bessel corrections; torch's ``unbiased``
    is ``correction == 1``.  Only 0/1 map — the host ``k_var_std``
    applies exactly the n/(n-c) correction.
    """
    op = "std" if want_std else "var"
    dim, keep = _dim_keep((t, *a), kw, op)
    corr = _num(_attr(kw, "correction", "arg2", default=1))
    if corr not in (0, 1):
        raise _unsupported(
            op, f"correction={corr} — host driver supports only 0 or 1"
        )
    # k_var_std arg layout: (self, dims|bool, unbiased, keepdim); a
    # null dim slot still reads args[2]/args[3] as unbiased/keepdim.
    return [tensor_arg(t), _dim_list(dim), corr == 1, keep]


def _layer_norm(t: Any, *a: Any, **kw: Any) -> list:
    """``layer_norm(x, w?, b?)`` — normalized_shape is the ``dim`` attr."""
    shape = _attr(kw, "dim", "normalized_shape", "arg1")
    if shape is None:
        raise _unsupported(
            "layer_norm", "missing normalized_shape attr"
        )
    w = a[1] if len(a) > 1 else _attr(kw, "weight")
    b = a[2] if len(a) > 2 else _attr(kw, "bias")
    eps = _num(_attr(kw, "eps", "arg4", default=1e-5))
    return [
        tensor_arg(t),
        _shape_list(shape),
        None if w is None else tensor_arg(w),
        None if b is None else tensor_arg(b),
        eps,
    ]


def _rms_norm(t: Any, *a: Any, **kw: Any) -> list:
    """``rms_norm(x, w?)`` — no bias in ATen; eps default 1e-6."""
    shape = _attr(kw, "dim", "normalized_shape", "arg1")
    if shape is None:
        raise _unsupported("rms_norm", "missing normalized_shape attr")
    w = a[1] if len(a) > 1 else _attr(kw, "weight")
    eps = _attr(kw, "eps", "arg3", default=None)
    return [
        tensor_arg(t),
        _shape_list(shape),
        None if w is None else tensor_arg(w),
        None if eps is None else _num(eps),
    ]


def _linear(x: Any, w: Any, *a: Any, **kw: Any) -> list:
    """``linear(x, w, b?)`` — synthesize a zero bias when absent."""
    bias = a[0] if a else _attr(kw, "bias")
    if bias is None:
        w_arr = np.asarray(w, dtype=np.float64)
        if w_arr.ndim != 2:
            raise _unsupported(
                "linear", f"weight rank {w_arr.ndim} != 2"
            )
        bias = np.zeros(w_arr.shape[0], dtype=np.float64)
    return [tensor_arg(x), tensor_arg(w), tensor_arg(bias)]


def _conv2d(x: Any, w: Any, *a: Any, **kw: Any) -> list:
    """``conv2d`` — NCHW; ``dilation`` must be 1 (the host has no slot)."""
    dilation = _attr(kw, "dilation", "arg5", default=1)
    if _dim_list(dilation) not in (1, [1, 1]):
        raise _unsupported("conv2d", f"dilation={dilation} != 1")
    bias = a[0] if a else _attr(kw, "bias")
    stride = _attr(kw, "stride", "arg3", default=1)
    padding = _attr(kw, "padding", "arg4", default=0)
    groups = _int(_attr(kw, "groups", "arg6", default=1))
    stride_j = (
        _shape_list(stride)
        if isinstance(stride, (list, tuple))
        else _int(stride)
    )
    padding_j = (
        _shape_list(padding)
        if isinstance(padding, (list, tuple))
        else _int(padding)
    )
    return [
        tensor_arg(x),
        tensor_arg(w),
        None if bias is None else tensor_arg(bias),
        stride_j,
        padding_j,
        groups,
    ]


def _reshape_view(t: Any, *a: Any, **kw: Any) -> list:
    """``reshape``/``view`` — resolve ``-1`` against the operand numel."""
    arr = np.asarray(t, dtype=np.float64)
    shape = _resolve_neg1(
        _shape_arg(a, kw, "reshape"), int(arr.size), "reshape"
    )
    return [tensor_arg(t), shape]


def _squeeze(t: Any, *a: Any, **kw: Any) -> list:
    dim = _attr(kw, "dim", "arg1", default=_MISSING)
    if dim is _MISSING:
        return [tensor_arg(t)]
    return [tensor_arg(t), _dim_list(dim)]


def _slice(t: Any, *a: Any, **kw: Any) -> list:
    start = _attr(kw, "start", "arg2", default=None)
    end = _attr(kw, "end", "arg3", default=None)
    return [
        tensor_arg(t),
        _int(_attr(kw, "dim", default=0)),
        None if start is None else _int(start),
        None if end is None else _int(end),
        _int(_attr(kw, "step", "arg4", default=1)),
    ]


class _Indexed:
    """Marker: builder result the executor resolves via a host call."""

    __slots__ = ("args", "index", "op")

    def __init__(self, op: str, args: list, index: int) -> None:
        self.op = op
        self.args = args
        self.index = index


def _chunk(t: Any, *a: Any, **kw: Any) -> _Indexed:
    """``chunk`` — host returns a tensor ARRAY; pick ``index``."""
    chunks = _int(
        _attr(kw, "chunks", "arg1", default=a[1] if len(a) > 1 else 2)
    )
    dim = _int(
        _attr(kw, "dim", "arg2", default=a[2] if len(a) > 2 else -1)
    )
    index = _int(
        _attr(kw, "index", "arg3", default=a[3] if len(a) > 3 else 0)
    )
    return _Indexed("chunk", [tensor_arg(t), chunks, dim], index)


def _split(t: Any, *a: Any, **kw: Any) -> _Indexed:
    """``split`` — sizes must be a LIST.

    An int ``sizes`` is torch's split-*size* semantics while the host
    port implements equal-count split — an honest decline, not a
    silent wrong answer.
    """
    sizes = _attr(
        kw, "sizes", "arg1", default=a[1] if len(a) > 1 else None
    )
    if sizes is None:
        raise _unsupported("split", "missing sizes")
    if isinstance(sizes, (int, float)) and not isinstance(sizes, bool):
        raise _unsupported(
            "split",
            "int sizes is torch's split-size semantics; the host port "
            "implements equal-count split — use a sizes list",
        )
    dim = _int(
        _attr(kw, "dim", "arg2", default=a[2] if len(a) > 2 else -1)
    )
    index = _int(
        _attr(kw, "index", "arg3", default=a[3] if len(a) > 3 else 0)
    )
    return _Indexed(
        "split", [tensor_arg(t), _shape_list(sizes), dim], index
    )


def _unbind(t: Any, *a: Any, **kw: Any) -> _Indexed:
    dim = _int(
        _attr(kw, "dim", "arg1", default=a[1] if len(a) > 1 else 0)
    )
    index = _int(
        _attr(kw, "index", "arg2", default=a[2] if len(a) > 2 else 0)
    )
    return _Indexed("unbind", [tensor_arg(t), dim], index)


def _stack(*ts: Any, **kw: Any) -> list:
    dim = _int(_attr(kw, "dim", "arg1", default=0))
    if not ts:
        raise _unsupported("stack", "no tensor operands")
    return [[tensor_arg(t) for t in ts], dim]


def _concat(*ts: Any, **kw: Any) -> list:
    dim = _int(_attr(kw, "dim", "arg1", default=0))
    if not ts:
        raise _unsupported("concat", "no tensor operands")
    return [[tensor_arg(t) for t in ts], dim]


def _expand(t: Any, *a: Any, **kw: Any) -> list:
    v = _attr(kw, "shape", "dim", default=_MISSING)
    if v is not _MISSING:
        size = _shape_list(v)
    elif a[1:]:
        size = [int(_num(d)) for d in a[1:]]
    else:
        raise _unsupported("expand", "missing target shape")
    if _attr(kw, "implicit", default=False):
        raise _unsupported("expand", "implicit=True is a host seam")
    return [tensor_arg(t), size]


def _repeat(t: Any, *a: Any, **kw: Any) -> list:
    v = _attr(kw, "shape", "dim", default=_MISSING)
    if v is not _MISSING:
        reps = _shape_list(v)
    elif a[1:]:
        reps = [int(_num(d)) for d in a[1:]]
    else:
        raise _unsupported("repeat", "missing repeats")
    return [tensor_arg(t), reps]


def _full(*a: Any, **kw: Any) -> list:
    """``full(shape…, fill?)`` — shape attr or operands.

    Fill is a Const operand or the ``fill_value`` attr (torch binding
    order).
    """
    fill = a[0] if a else _attr(kw, "fill_value", default=0)
    shape = _attr(kw, "shape", "dim", default=_MISSING)
    if shape is _MISSING:
        shape = a[1] if len(a) > 1 else _MISSING
    if shape is _MISSING:
        raise _unsupported("full", "missing shape")
    return [_shape_list(shape), _num(fill)]


def _arange(*a: Any, **kw: Any) -> list:
    """``arange`` - 1-3 Const operands (end / start,end / +step)."""
    vals = list(a)
    if not vals:
        vals = [
            v
            for i in (1, 2, 3)
            if (v := _attr(kw, f"arg{i}", default=None)) is not None
        ]
    if not vals or len(vals) > 3:
        raise _unsupported("arange", f"got {len(vals)} bound operands")
    return [_num(v) for v in vals]


def _eye(*a: Any, **kw: Any) -> list:
    n = _attr(kw, "dim", "n", "arg0", "arg1", default=None)
    m = _attr(kw, "m", "arg1", "arg2", default=None)
    if n is None:
        raise _unsupported("eye", "missing n (attr 'dim'/'n')")
    out = [_int(n)]
    if m is not None:
        out.append(_int(m))
    return out


def _where(c: Any, *a: Any, **kw: Any) -> list:
    """``where(cond, x, y)`` - x/y may ride ``arg1``/``arg2`` attrs.

    The torch binding's slot-fill convention.
    """
    slots = [_attr(kw, "arg1"), _attr(kw, "arg2")]
    it = iter(a)
    filled = [s if s is not None else next(it, None) for s in slots]
    x, y = filled
    if x is None or y is None:
        raise _unsupported(
            "where",
            "self/other operands missing (children or argN attrs)",
        )
    if next(it, None) is not None:
        raise _unsupported("where", "surplus value operands")
    return [tensor_arg(c), scalar_or_tensor(x), scalar_or_tensor(y)]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

HOST_BINDINGS: dict[str, Callable[..., Any]] = {
    # --- pointwise unary -------------------------------------------------
    "abs": _u1("abs"),
    "neg": _u1("neg"),
    "exp": _u1("exp"),
    "exp2": _u1("exp2"),
    "expm1": _u1("expm1"),
    "log": _u1("log"),
    "log1p": _u1("log1p"),
    "log2": _u1("log2"),
    "log10": _u1("log10"),
    "sqrt": _u1("sqrt"),
    "rsqrt": _u1("rsqrt"),
    "reciprocal": _u1("reciprocal"),
    "square": _u1("square"),
    "sin": _u1("sin"),
    "cos": _u1("cos"),
    "asin": _u1("asin"),
    "acos": _u1("acos"),
    "atan": _u1("atan"),
    "sinh": _u1("sinh"),
    "cosh": _u1("cosh"),
    "tanh": _u1("tanh"),
    "asinh": _u1("asinh"),
    "acosh": _u1("acosh"),
    "atanh": _u1("atanh"),
    "floor": _u1("floor"),
    "ceil": _u1("ceil"),
    "round": _u1("round"),
    "trunc": _u1("trunc"),
    "frac": _u1("frac"),
    "sign": _u1("sign"),
    "sigmoid": _u1("sigmoid"),
    "silu": _u1("silu"),
    "mish": _u1("mish"),
    "relu": _u1("relu"),
    "erf": _u1("erf"),
    "erfc": _u1("erfc"),
    "sinc": _u1("sinc"),
    "tan": _u1("tan"),
    "isnan": _u1("isnan"),
    "isinf": _u1("isinf"),
    "isfinite": _u1("isfinite"),
    "isneginf": _u1("isneginf"),
    "isposinf": _u1("isposinf"),
    "logical_not": _u1("logical_not"),
    # activations with scalar attrs
    "gelu": lambda t, *a, **kw: [
        tensor_arg(t),
        str(_attr(kw, "approximate", "arg1", default="none")),
    ],
    "elu": lambda t, *a, **kw: [
        tensor_arg(t),
        _num(_attr(kw, "alpha", default=1.0)),
        _num(_attr(kw, "scale", default=1.0)),
        _num(_attr(kw, "input_scale", default=1.0)),
    ],
    "celu": lambda t, *a, **kw: [
        tensor_arg(t),
        _num(_attr(kw, "alpha", "arg1", default=1.0)),
    ],
    "leaky_relu": lambda t, *a, **kw: [
        tensor_arg(t),
        _num(
            a[0]
            if a
            else _attr(kw, "negative_slope", "arg1", default=0.01)
        ),
    ],
    "softplus": lambda t, *a, **kw: [
        tensor_arg(t),
        _num(_attr(kw, "beta", default=1)),
        _num(_attr(kw, "threshold", default=20)),
    ],
    "hardtanh": lambda t, *a, **kw: [
        tensor_arg(t),
        _num(_attr(kw, "min", default=-1.0)),
        _num(_attr(kw, "max", default=1.0)),
    ],
    "nan_to_num": lambda t, *a, **kw: [
        tensor_arg(t),
        _num(_attr(kw, "nan", "arg1", default=0.0)),
        (
            None
            if _attr(kw, "posinf", "arg2", default=None) is None
            else _num(_attr(kw, "posinf", "arg2"))
        ),
        (
            None
            if _attr(kw, "neginf", "arg3", default=None) is None
            else _num(_attr(kw, "neginf", "arg3"))
        ),
    ],
    # --- pointwise binary -------------------------------------------------
    "add": _u2("add"),
    "mul": _u2("mul"),
    "sub": _u2("sub"),
    "div": _u2("div"),
    "maximum": _u2("maximum"),
    "minimum": _u2("minimum"),
    "atan2": _u2("atan2"),
    "logical_and": _u2("logical_and"),
    "logical_or": _u2("logical_or"),
    "logical_xor": _u2("logical_xor"),
    "heaviside": _u2("heaviside"),
    "pow": lambda a, b, *r, **kw: [tensor_arg(a), scalar_or_tensor(b)],
    "fmod": _bin_scalar_rhs("fmod"),
    "remainder": lambda a, b, *r, **kw: [
        scalar_or_tensor(a),
        scalar_or_tensor(b),
    ],
    "xlogy": lambda a, b, *r, **kw: [
        scalar_or_tensor(a),
        scalar_or_tensor(b),
    ],
    "lerp": lambda a, e, w, *r, **kw: [
        tensor_arg(a),
        tensor_arg(e),
        scalar_or_tensor(w),
    ],
    "eq": _bin_scalar_rhs("eq"),
    "ne": _bin_scalar_rhs("ne"),
    "lt": _bin_scalar_rhs("lt"),
    "le": _bin_scalar_rhs("le"),
    "gt": _bin_scalar_rhs("gt"),
    "ge": _bin_scalar_rhs("ge"),
    "clamp": lambda t, *a, **kw: _clamp_args(t, a, kw),
    "clamp_min": lambda t, *a, **kw: [
        tensor_arg(t),
        scalar_or_tensor(a[0] if a else _attr(kw, "min", default=None)),
    ],
    "clamp_max": lambda t, *a, **kw: [
        tensor_arg(t),
        scalar_or_tensor(a[0] if a else _attr(kw, "max", default=None)),
    ],
    # --- reductions -------------------------------------------------------
    "sum": _reduce("sum"),
    "prod": _reduce("prod"),
    "mean": _reduce("mean"),
    "amax": _reduce("amax"),
    "amin": _reduce("amin"),
    "nanmean": _reduce("nanmean"),
    "nansum": _reduce("nansum"),
    "max": lambda t, *a, **kw: _max_min_args(t, a, kw, True),
    "min": lambda t, *a, **kw: _max_min_args(t, a, kw, False),
    "argmax": lambda t, *a, **kw: _arg_reduce(t, a, kw, True),
    "argmin": lambda t, *a, **kw: _arg_reduce(t, a, kw, False),
    "all": lambda t, *a, **kw: _all_any(t, a, kw, True),
    "any": lambda t, *a, **kw: _all_any(t, a, kw, False),
    "count_nonzero": lambda t, *a, **kw: [
        tensor_arg(t),
        _dim_list(_dim_keep((t, *a), kw, "count_nonzero")[0]),
    ],
    "var": lambda t, *a, **kw: _var_std(t, a, kw, False),
    "std": lambda t, *a, **kw: _var_std(t, a, kw, True),
    "cumsum": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(
            _attr(kw, "dim", "arg1", default=a[1] if len(a) > 1 else 0)
        ),
    ],
    "cumprod": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(
            _attr(kw, "dim", "arg1", default=a[1] if len(a) > 1 else 0)
        ),
    ],
    "softmax": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "dim", default=a[1] if len(a) > 1 else -1)),
    ],
    "log_softmax": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "dim", default=a[1] if len(a) > 1 else -1)),
    ],
    "layer_norm": _layer_norm,
    "rms_norm": _rms_norm,
    # --- linalg -------------------------------------------------------------
    "matmul": _u2("matmul"),
    "outer": _u2("outer"),
    "linear": _linear,
    "conv2d": _conv2d,
    # --- shape ---------------------------------------------------------------
    "reshape": _reshape_view,
    "view": _reshape_view,
    "transpose": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "dim0", "arg1", default=-2)),
        _int(_attr(kw, "dim1", "arg2", default=-1)),
    ],
    "unsqueeze": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(
            _attr(kw, "dim", "arg1", default=a[1] if len(a) > 1 else -1)
        ),
    ],
    "squeeze": _squeeze,
    "flatten": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "start_dim", "arg1", default=0)),
        _int(_attr(kw, "end_dim", "arg2", default=-1)),
    ],
    "narrow": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "dim", default=0)),
        _int(_attr(kw, "start", default=0)),
        _int(_attr(kw, "length", default=1)),
    ],
    "select": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "dim", default=0)),
        _int(_attr(kw, "index", default=0)),
    ],
    "slice": _slice,
    "chunk": _chunk,
    "split": _split,
    "stack": _stack,
    "concat": _concat,
    "unbind": _unbind,
    "expand": _expand,
    "expand_as": _u2("expand_as"),
    "repeat": _repeat,
    "contiguous": _u1("contiguous"),
    "clone": _u1("clone"),
    "detach": _u1("detach"),
    "tril": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "diagonal", "arg1", default=0)),
    ],
    "triu": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "diagonal", "arg1", default=0)),
    ],
    "flip": lambda t, *a, **kw: [
        tensor_arg(t),
        _shape_list(
            _attr(kw, "dim", "dims", default=a[1] if len(a) > 1 else ())
        ),
    ],
    "pixel_shuffle": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "upscale_factor", "arg1", default=1)),
    ],
    "pixel_unshuffle": lambda t, *a, **kw: [
        tensor_arg(t),
        _int(_attr(kw, "downscale_factor", "arg1", default=1)),
    ],
    # --- constructors / metadata ---------------------------------------------
    "zeros": lambda *a, **kw: [_shape_arg(a, kw, "zeros")],
    "ones": lambda *a, **kw: [_shape_arg(a, kw, "ones")],
    "full": _full,
    "full_like": lambda t, *a, **kw: [
        tensor_arg(t),
        _num(
            a[1] if len(a) > 1 else _attr(kw, "fill_value", default=0)
        ),
    ],
    "zeros_like": _u1("zeros_like"),
    "ones_like": _u1("ones_like"),
    "arange": _arange,
    "eye": _eye,
    "numel": _u1("numel"),
    "item": _u1("item"),
    "where": _where,
}
