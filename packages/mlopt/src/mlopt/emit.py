"""mlopt terms → ``.hnl`` programs over ``tensor.*`` — the emit bridge.

Graduated from ``graphopt:bench/drivers/kit/term_hnl.py`` (the
measurement driver that proved the bridge): a mlopt term — the same
object the e-graph rewrites — rendered as a ``.hnl`` expression, and
bound into a runnable program or an exported function.

* :func:`emit` renders any term inside the covered op surface as a
  ``.hnl`` expression over ``tensor.*``.  Coverage is a per-op builder
  table (``_OPS``), not per-term code — 203 ops measured against the
  *current* ~239-op ``host-tensor`` ``TENSOR_OPS`` surface (each
  builder's call shape was verified against a live ``hnl-run``):
  pointwise unary/binary, reduce-dim, linalg, shape/structural,
  constructors, pair-returning reductions (``sort``/``topk``/
  ``mode``/``kthvalue``/``median(dim)`` consumed through
  ``getitem``), gather/scatter, pooling, and the distance/norm
  family.  Carrier ops are re-spelled first —
  :func:`mlopt.carrier_expand.expand_carriers` applies the two
  definitional unfolds (``om_apply(om_elem(s,v))`` →
  ``matmul(softmax(s,-1), v)``, ``om_decayed(s,d,v)`` →
  ``matmul(mul(s,d), v)`` — the shipped ``OM_UNLIFT`` /
  ``OM_DECAYED_UNLIFT`` laws).  ``to`` casts emit the spellings the
  host ``k_to`` kernel honours — the f64 identity plus the real
  narrowing casts (``float32``/``int64``/…/``bool``) it landed in
  host-tensor ecc57d9; anything outside that table (f16/bf16/
  float8, unknown names) still declines at the dtype gate, as does
  every *other* op's non-f64 ``dtype`` attr (the compute-precision
  contract is unchanged).
  Anything still outside the surface — the remaining carrier shapes
  (``om_compose`` trees, ``omd_*``, ``aff*``/``apply*``, ``trace``/
  ``scan``, decode ops), ops the host port declines — raises
  :class:`UnsupportedOpError` naming the op: the honest bound.
  ``tests/test_emit_surface.py`` drift-guards the
  table against ``kit_host_tensor``'s ``host_surface.json`` — every
  builder's emitted ``tensor.*`` call is probed against the canonical
  239-op artifact (existence, emittable status, arg bounds), and
  uncovered canonical ops must decline by name.  The mechanical
  ``_u1``/``_u2`` half of the table is *generated* from that artifact
  at import (``_canonicalize_mechanical`` / ``_MECH_SPECS`` — ADR-0178
  invariant 4); the semantic builders stay hand-written because the
  surface does not record IR-side binding conventions.
* :func:`emit_program` binds every ``Var``/``Param`` leaf in a
  ``do { name :: … }`` prelude — tensor data as
  ``tensor.reshape(tensor.from_list([flat]), [shape])``, or a caller
  ``{"hnl": "<expr>"}`` marshaling channel for computed inputs —
  then binds every multiply-referenced ``Op`` subterm once as
  ``__sN ::`` (hash-consed emit: the term is a DAG, the artifact
  carries the DAG, not the tree expansion — see
  :func:`_emit_shared`).
* :func:`emit_function` wraps the same prelude in an exported
  ``.hnl`` function — ``Param`` leaves bind to data, ``Var`` leaves
  become the function's parameters — the artifact ``mlopt export``
  writes.
* Opt-in **sidecar mode** (``sidecar=`` on either emitter): large
  ``Param`` leaves emit ``tensor.load("<basename>", "<leaf>")`` and
  their values land in a ``.safetensors`` file written next to the
  artifact — the out-of-band weight channel from
  ``host:project/reports/tensor_load_spike.md``.  :class:`KitRunner`
  exports ``TENSOR_LOAD_DIR`` so stdin eval resolves the sidecar;
  the default path never writes files.
* :class:`KitRunner` shells a compiled ``hnl-run`` binary (located via
  ``$MLOPT_HNL_RUN``, sibling-repo target dirs, or ``PATH``) and
  decodes the ``{"__tensor__": …}`` wire form.  Imported lazily — the
  module stays torch-free and binary-free at import time.  Emitted
  sources carry ``from "host-tensor" import tensor`` (self-contained
  file artifacts); stdin eval strips it and uses ambient ``tensor.*``
  dispatch — ``hnl-run -`` cannot resolve project imports into a
  value (measured: project mode returns ``null`` on snippets).

Differences from the measurement driver, all forced by the wider
surface:

* **Scalar operands.**  IR scalar-operand ops (``pow``, the
  comparisons, ``clamp``…) accept bare ``.hnl`` number literals; the
  tensor-required spellings (``add``/``mul``/``sub``/``div``,
  ``maximum``/``minimum``/``atan2``/``heaviside``/``lerp``) need a
  rank-0 tensor, emitted as
  ``tensor.reshape(tensor.from_list([v]), [])`` — the same 0-d wrap
  ``HOST_BINDINGS.tensor_arg`` performs on the JSON wire.
* **``-1`` shapes.**  The handwritten ``tensor.reshape``/``view``
  ports take a fully-spelled shape (no wildcard).  The emitter
  resolves ``-1`` through ``mlopt_core.typing.shape_of`` when the
  input shape is concrete, and declines honestly when it is not.
* **Bias-less ``linear``.**  ``tensor.linear`` requires its bias
  operand; the emitter synthesizes
  ``tensor.zeros([tensor.shape(<w>)[0]])`` — ``.hnl`` list indexing
  makes the out-features extent expressible without values.
* **Multi-output ops.**  ``chunk``/``split``/``unbind`` return tensor
  arrays on the host; the IR spelling carries an ``index`` attr, so
  the emitter wraps the call in ``(...)[index]`` — verified against
  ``hnl-run``.

The lexer wart is preserved: the handl lexer does not read scientific
notation (``1e-05`` lexes as ``1`` ``e`` ``-05``), so :func:`_num`
expands every float through :class:`~decimal.Decimal` — ``1e-05`` →
``0.00001`` — and a mlopt ``eps`` can never silently change value
across the boundary.  Non-finite literals decline: the tensor wire
carries f64 data only.
"""

from __future__ import annotations

import array
import json
import os
import shutil
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from mlopt_core.term import Const, Op, Param, Var
from mlopt_core.typing import INVALID as _INVALID_SHAPE
from mlopt_core.typing import shape_of

from mlopt.carrier_expand import expand_carriers

__all__ = [
    "KitEvalError",
    "KitRunner",
    "UnsupportedOpError",
    "emit",
    "emit_function",
    "emit_program",
    "leaf_kinds",
    "leaf_names",
    "write_safetensors",
]


class UnsupportedOpError(ValueError):
    """The term reaches outside the covered ``tensor.*`` op surface.

    Carries the op name (or the value) — the bridge's honest bound.  A
    spelled conv+BN fold dies on whatever the surface cannot honour; a
    ``batch_norm`` term dies on ``training``/running-stats attrs the
    eval-mode port does not take positionally.
    """


class KitEvalError(RuntimeError):
    """``hnl-run`` exited non-zero — the kit refused the program."""


# ---------------------------------------------------------------------------
#  Literals and leaves
# ---------------------------------------------------------------------------


def _num(v: float | int) -> str:
    """Render a number as a handl literal — no scientific notation.

    The lexer does not read ``1e-05`` (it lexes as ``1`` ``e`` ``-05``);
    ``Decimal`` expansion keeps every f64 exact.
    """
    if isinstance(v, bool):  # bool precedes int in the numeric tower
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise UnsupportedOpError(
            f"non-finite literal {v!r} — the tensor wire carries f64 data only"
        )
    return format(Decimal(repr(f)), "f")


def _lit(v: Any) -> str:
    """Render an attr value / scalar as a ``.hnl`` literal."""
    if v is None:
        return "null"
    if isinstance(v, Const):
        return _lit(v.value)
    if isinstance(v, (bool, int, float)):
        return _num(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    raise UnsupportedOpError(f"unemittable attr value: {v!r}")


def _leaf_key(t: Any) -> tuple[str, str] | None:
    """``(name, kind)`` for a ``Var``/``Param`` leaf, else ``None``."""
    if isinstance(t, Var):
        return (t.name, "var")
    if isinstance(t, Param):
        return (t.name, "param")
    return None


def leaf_kinds(term: Any) -> list[tuple[str, str]]:
    """Every ``Var``/``Param`` ``(name, kind)`` in *term*, first-use order.

    The term is a hash-consed DAG, so the walk visits each unique
    ``Op`` node once — O(unique edges), not O(tree occurrences) (the
    whisper-encoder term tree-expands to ~4M occurrences over ~258
    unique subterms).  Iterative — term depth is unbounded.  Leaf
    order matches the tree expansion's first-use order: a shared
    node's leaves are all recorded at its first (earliest) occurrence,
    and later occurrences contribute nothing new.
    """
    seen: set[str] = set()
    order: list[tuple[str, str]] = []
    visited: set[Op] = set()
    stack: list[Any] = [term]
    while stack:
        t = stack.pop()
        k = _leaf_key(t)
        if k is not None:
            if k[0] not in seen:
                seen.add(k[0])
                order.append(k)
            continue
        if isinstance(t, Op) and t not in visited:
            visited.add(t)
            stack.extend(reversed(t.args))
    return order


def leaf_names(term: Any) -> list[str]:
    """Every ``Var``/``Param`` name in *term*, first-use order."""
    return [n for n, _ in leaf_kinds(term)]


def _sanitize(name: str, taken: set[str]) -> str:
    """Return a safe ``.hnl`` binding name for a mlopt leaf.

    Export names are dotted (``conv.weight``) — dots are field access
    in handl.  And ``tensor`` itself is reserved: binding it would
    shadow the op namespace.
    """
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not out or out[0].isdigit() or out == "tensor":
        out = "v_" + out
    while out in taken:
        out += "_"
    taken.add(out)
    return out


# ---------------------------------------------------------------------------
#  Attr reading — canonical names first, ``argN`` fallbacks (the torch
#  adapter's convention; ``Const``-valued attrs unwrap to their scalar)
# ---------------------------------------------------------------------------

_MISSING = object()


def _attr(kw: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in kw:
            v = kw[name]
            return v.value if isinstance(v, Const) else v
    return default


def _attr_lit(kw: Mapping[str, Any], *names: str, default: Any = None) -> str:
    """Read an attr and render it as a ``.hnl`` literal (``null`` default)."""
    v = _attr(kw, *names, default=default)
    return _lit(v)


def _int_attr(kw: Mapping[str, Any], *names: str, default: Any = None) -> int:
    v = _attr(kw, *names, default=default)
    if v is None:
        return 0 if default is None else int(default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise UnsupportedOpError(f"attr {names[0]}={v!r} is not an int")
    return int(v)


# ---------------------------------------------------------------------------
#  Operand emission helpers
# ---------------------------------------------------------------------------


def _is_operand(t: Any) -> bool:
    """Whether a positional arg is a term operand (vs a scalar tail)."""
    return isinstance(t, (Op, Var, Param, Const))


def _rank0(v: Any) -> str:
    """Emit a scalar as a rank-0 ``tensor.*`` value (the 0-d Const wrap)."""
    return f"tensor.reshape(tensor.from_list([{_lit(v)}]), [])"


def _tens(t: Any, go) -> str:
    """Emit a tensor-required slot: terms recurse, scalars wrap rank-0."""
    if isinstance(t, Const):
        return _rank0(t.value)
    if isinstance(t, (bool, int, float)) or t is None:
        return _rank0(t)
    return go(t)


def _scalar_or_tensor(t: Any, go) -> str:
    """Emit a ``Scalar | Tensor`` slot: numbers stay literals."""
    if isinstance(t, Const):
        return _lit(t.value)
    if isinstance(t, (bool, int, float)) or t is None:
        return _lit(t)
    return go(t)


def _maybe_term(v: Any, go) -> str:
    """Emit an attr-held value that may be a term (layer_norm weights)."""
    if v is None:
        return "null"
    if isinstance(v, (Op, Var, Param)):
        return go(v)
    return _lit(v)


def _shape_list_emit(v: Any, go) -> str:
    """Render a shape spec as a ``.hnl`` int list."""
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    if isinstance(v, Const):
        return _shape_list_emit(v.value, go)
    if isinstance(v, (Op, Var, Param)):
        # A computed shape (rare): emit the term — host decodes lists.
        return go(v)
    raise UnsupportedOpError(f"shape spec {v!r} is not an int list")


def _dim_list(v: Any) -> Any:
    """Normalise a dim attr to int-or-int-list (``None`` stays ``None``)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [_lit_dim(d) for d in v]
    return _lit_dim(v)


def _lit_dim(d: Any) -> Any:
    if isinstance(d, Const):
        d = d.value
    if isinstance(d, bool) or not isinstance(d, (int, float)):
        raise UnsupportedOpError(f"dim {d!r} is not an int")
    return int(d)


def _resolve_neg1(shape: list[Any], in_term: Any, op: str) -> list[Any]:
    """Resolve a single ``-1`` wildcard against the input's shape.

    ``shape_of`` gives best-effort shapes; ``None`` dims or an unknown
    input shape decline honestly — the host ``reshape``/``view`` ports
    take a fully-spelled shape.
    """
    if -1 not in shape:
        return shape
    if shape.count(-1) > 1:
        raise UnsupportedOpError(f"{op}: shape {shape} has multiple -1 wildcards")
    s: Any = shape_of(in_term)
    if s is None or s is _INVALID_SHAPE or any(d is None for d in s):
        raise UnsupportedOpError(
            f"{op}: -1 wildcard needs a concrete input shape; shape_of({in_term!r}) is unknown"
        )
    numel = 1
    for d in s:
        numel *= d
    known = 1
    for d in shape:
        if d != -1:
            known *= d
    if known == 0 or numel % known:
        raise UnsupportedOpError(f"{op}: cannot resolve -1 in shape {shape} for numel {numel}")
    return [numel // known if d == -1 else d for d in shape]


# ---------------------------------------------------------------------------
#  Shared builder fragments — ports of kit_host_tensor.bindings helpers
# ---------------------------------------------------------------------------


def _shape_arg(args: list, kw: Mapping, op: str, go) -> str:
    """Target shape for creator-like ops: attr, else operand list."""
    v = _attr(kw, "shape", "dim", default=_MISSING)
    if v is not _MISSING:
        return _shape_list_emit(v, go)
    for name in ("arg0", "arg1", "arg2", "arg3"):
        if name in kw:
            return _shape_list_emit(kw[name], go)
    if args:
        return "[" + ", ".join(_scalar_or_tensor(a, go) for a in args) + "]"
    raise UnsupportedOpError(f"{op}: missing shape (attr 'shape'/'dim' or operands)")


def _dim_keep(args: list, kw: Mapping, op: str, go) -> tuple[str, str]:
    """Shared ``(dim, keepdim)`` extraction for the reduce family."""
    extra = args[1:]
    if extra:
        dim_e = _scalar_or_tensor(extra[0], go)
        keep = extra[1] if len(extra) > 1 else _attr(kw, "keepdim", default=False)
    else:
        dim = _attr(kw, "dim", "axis", "arg1", default=None)
        dim_e = _lit(_dim_list(dim))
        keep = _attr(kw, "keepdim", "arg2", default=False)
    return dim_e, _lit(bool(keep))


def _reduce(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        dim_e, keep_e = _dim_keep(args, kw, op, go)
        return f"tensor.{op}({_tens(args[0], go)}, {dim_e}, {keep_e})"

    return build


def _arg_reduce(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        _, keep_e = _dim_keep(args, kw, op, go)
        # argmax/argmin take a SINGLE int dim (or null), not a list.
        dim_raw = _attr(kw, "dim", "axis", "arg1", default=None)
        if args[1:]:
            dim_raw = args[1]
        if dim_raw is None:
            dim_e = "null"
        elif isinstance(dim_raw, (list, tuple)):
            raise UnsupportedOpError(f"{op}: dim must be a single int, not a tuple")
        elif isinstance(dim_raw, (Op, Var, Param)):
            dim_e = go(dim_raw)
        else:
            dim_e = _lit(dim_raw)
        return f"tensor.{op}({_tens(args[0], go)}, {dim_e}, {keep_e})"

    return build


def _clamp_args(args: list, kw: Mapping, go) -> list[str]:
    """``clamp(self, min?, max?)`` — bounds are Scalar?/Tensor?."""
    lo = _attr(kw, "min", default=_MISSING)
    hi = _attr(kw, "max", default=_MISSING)
    rest = args[1:]
    if lo is _MISSING and rest:
        lo = rest[0]
    if hi is _MISSING and len(rest) > 1:
        hi = rest[1]
    if len(rest) > 2:
        raise UnsupportedOpError("clamp: surplus bound operands")

    def bound(v: Any) -> str:
        if v is _MISSING or v is None:
            return "null"
        if isinstance(v, (Op, Var, Param)):
            return go(v)
        return _scalar_or_tensor(v, go)

    return [_tens(args[0], go), bound(lo), bound(hi)]


def _max_min(op: str, want_max: bool):
    def build(args: list, kw: Mapping, go) -> str:
        rest = args[1:]
        if rest:
            if len(rest) != 1 or kw:
                raise UnsupportedOpError(
                    f"{op}: elementwise spelling takes exactly one "
                    "operand and no attrs (use 'maximum'/'minimum')"
                )
            return f"tensor.{op}({_tens(args[0], go)}, {_tens(rest[0], go)})"
        dim = _attr(kw, "dim", "axis", "arg1", default=None)
        if dim is None:
            return f"tensor.{op}({_tens(args[0], go)})"
        keep = bool(_attr(kw, "keepdim", default=False))
        return f"tensor.{op}({_tens(args[0], go)}, {_lit(_dim_list(dim))}, {_lit(keep)})"

    return build


def _all_any(op: str, want_all: bool):
    def build(args: list, kw: Mapping, go) -> str:
        dim_e, keep_e = _dim_keep(args, kw, op, go)
        return f"tensor.{op}({_tens(args[0], go)}, {dim_e}, {keep_e})"

    return build


def _var_std(op: str, want_std: bool):
    def build(args: list, kw: Mapping, go) -> str:
        # The host driver has NO correction slot: only 0/1 map
        # (torch's unbiased == correction == 1).
        dim_e, keep_e = _dim_keep(args, kw, op, go)
        corr = _attr(kw, "correction", "arg2", default=1)
        if isinstance(corr, Const):
            corr = corr.value
        if corr not in (0, 1):
            raise UnsupportedOpError(
                f"{op}: correction={corr!r} — the tensor surface supports only 0 or 1"
            )
        # k_var_std layout: (self, dims|bool, unbiased, keepdim)
        return f"tensor.{op}({_tens(args[0], go)}, {dim_e}, {_lit(corr == 1)}, {keep_e})"

    return build


def _layer_norm(args: list, kw: Mapping, go) -> str:
    shape = _attr(kw, "dim", "normalized_shape", "arg1", default=None)
    if shape is None:
        raise UnsupportedOpError("layer_norm: missing normalized_shape")
    w = args[1] if len(args) > 1 else _attr(kw, "weight", default=None)
    b = args[2] if len(args) > 2 else _attr(kw, "bias", default=None)
    eps = _attr(kw, "eps", "arg4", default=1e-5)
    return (
        f"tensor.layer_norm({_tens(args[0], go)}, "
        f"{_shape_list_emit(shape, go)}, {_maybe_term(w, go)}, "
        f"{_maybe_term(b, go)}, {_lit(eps)})"
    )


def _rms_norm(args: list, kw: Mapping, go) -> str:
    shape = _attr(kw, "dim", "normalized_shape", "arg1", default=None)
    if shape is None:
        raise UnsupportedOpError("rms_norm: missing normalized_shape")
    w = args[1] if len(args) > 1 else _attr(kw, "weight", default=None)
    eps = _attr(kw, "eps", "arg3", default=None)
    eps_e = "null" if eps is None else _lit(eps)
    return (
        f"tensor.rms_norm({_tens(args[0], go)}, "
        f"{_shape_list_emit(shape, go)}, {_maybe_term(w, go)}, {eps_e})"
    )


def _linear(args: list, kw: Mapping, go) -> str:
    """``linear(x, w, b?)`` — synthesize a zero bias when absent."""
    bias = args[2] if len(args) > 2 else _attr(kw, "bias", default=None)
    if bias is None:
        # tensor.linear requires the bias slot; the out-features extent
        # is shape(w)[0] — expressible in .hnl via list indexing.
        bias_e = f"tensor.zeros([tensor.shape({_tens(args[1], go)})[0]])"
    else:
        bias_e = _tens(bias, go)
    return f"tensor.linear({_tens(args[0], go)}, {_tens(args[1], go)}, {bias_e})"


def _conv2d(args: list, kw: Mapping, go) -> str:
    """``conv2d`` — NCHW; ``dilation`` must be 1 (no host slot)."""
    dilation = _attr(kw, "dilation", "arg5", default=1)
    if isinstance(dilation, (list, tuple)):
        ok = list(dilation) in ([1, 1], (1, 1)) or dilation == 1
    else:
        ok = dilation == 1
    if not ok:
        raise UnsupportedOpError(f"conv2d: dilation={dilation!r} != 1")
    bias = args[2] if len(args) > 2 else _attr(kw, "bias", default=None)
    stride = _attr(kw, "stride", "arg3", default=1)
    padding = _attr(kw, "padding", "arg4", default=0)
    groups = _attr(kw, "groups", "arg6", default=1)
    bias_e = "null" if bias is None else _tens(bias, go)
    return (
        f"tensor.conv2d({_tens(args[0], go)}, {_tens(args[1], go)}, "
        f"{bias_e}, {_lit(stride)}, {_lit(padding)}, {_lit(groups)})"
    )


def _conv1d_dim(v: Any, name: str, op: str) -> int:
    """Normalize a conv1d geometry attr to a scalar int.

    torch's ``_single`` convention stores stride/padding/dilation as an
    int or a 1-tuple (``(2,)``); anything wider has no conv2d-mapped
    form and declines by name.
    """
    if isinstance(v, Const):
        v = v.value
    if isinstance(v, (list, tuple)):
        if len(v) != 1:
            raise UnsupportedOpError(f"{op}: {name}={v!r} — expected int or 1-tuple")
        v = v[0].value if isinstance(v[0], Const) else v[0]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise UnsupportedOpError(f"{op}: {name}={v!r} is not an int")
    if isinstance(v, float) and not v.is_integer():
        raise UnsupportedOpError(f"{op}: {name}={v!r} is not an integer")
    return int(v)


def _conv1d(args: list, kw: Mapping, go) -> str:
    """``conv1d`` — NCL → unsqueeze to NCHW×1 → ``conv2d`` → squeeze.

    The host surface has no ``tensor.conv1d``; the 1-D convolution is
    the degenerate 2-D one (kernel/stride/padding live on the W axis,
    H is size-1 with unit stride/padding).  ``dilation`` must be 1 —
    ``tensor.conv2d`` has no dilation slot, same bound as ``conv2d``.
    """
    dilation = _attr(kw, "dilation", "arg5", default=1)
    if isinstance(dilation, Const):
        dilation = dilation.value
    if isinstance(dilation, (list, tuple)):
        dilation = _conv1d_dim(dilation, "dilation", "conv1d")
    if dilation != 1:
        raise UnsupportedOpError(f"conv1d: dilation={dilation!r} != 1")
    bias = args[2] if len(args) > 2 else _attr(kw, "bias", default=None)
    stride = _conv1d_dim(_attr(kw, "stride", "arg3", default=1), "stride", "conv1d")
    padding = _conv1d_dim(_attr(kw, "padding", "arg4", default=0), "padding", "conv1d")
    groups = _attr(kw, "groups", "arg6", default=1)
    groups = groups.value if isinstance(groups, Const) else groups
    bias_e = "null" if bias is None else _tens(bias, go)
    return (
        f"tensor.squeeze("
        f"tensor.conv2d("
        f"tensor.unsqueeze({_tens(args[0], go)}, 2), "
        f"tensor.unsqueeze({_tens(args[1], go)}, 2), "
        f"{bias_e}, [1, {stride}], [0, {padding}], {_lit(groups)}"
        f"), 2)"
    )


def _permute(args: list, kw: Mapping, go) -> str:
    """``permute`` — no ``tensor.permute``; decompose into ``swapaxes``.

    A permutation is a product of transpositions: walk the target
    order left-to-right, swapping each position into place.  The dims
    must be a literal permutation of ``range(rank)`` — computed dims
    or a non-bijective list decline honestly.
    """
    dims = _attr(kw, "dim", "dims", "order", default=_MISSING)
    if dims is _MISSING:
        # Positional dims — the torch binding spells ``permute(x, *dims)``.
        # ``_normalize`` folds trailing scalar args into ``arg{pos}``
        # attrs while operand args (a Const list, a computed term) stay
        # positional; walking original positions 1.. reunites both
        # spellings.  A lone list-valued dim stays the list spelling
        # (the attr convention's positional twin).
        pos: list[Any] = []
        i = 1
        while i < len(args) or f"arg{i}" in kw:
            if i < len(args):
                pos.append(args[i])
            else:
                pos.append(kw[f"arg{i}"])
            i += 1
        if pos:
            if len(pos) == 1:
                sole = pos[0].value if isinstance(pos[0], Const) else pos[0]
                dims = sole if isinstance(sole, (list, tuple)) else pos
            else:
                dims = pos
    if dims is _MISSING or dims is None:
        # permute() with no dims reverses all axes — needs the rank.
        s = shape_of(args[0])
        if s is None or s is _INVALID_SHAPE or not s or any(d is None for d in s):
            raise UnsupportedOpError("permute: missing dims and the input rank is unknown")
        dims = list(reversed(range(len(s))))
    vv = dims.value if isinstance(dims, Const) else dims
    if isinstance(vv, (Op, Var, Param)):
        raise UnsupportedOpError(
            f"permute: computed dims {vv!r} — the swapaxes chain needs a literal permutation"
        )
    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
        if isinstance(vv, float) and not vv.is_integer():
            raise UnsupportedOpError(f"permute: dim {vv!r} is not an integer")
        vv = [int(vv)]
    if not isinstance(vv, (list, tuple)):
        raise UnsupportedOpError(f"permute: dims {dims!r} is not an int list")
    p: list[int] = []
    for d in vv:
        d = d.value if isinstance(d, Const) else d
        if isinstance(d, (Op, Var, Param)):
            raise UnsupportedOpError(
                f"permute: computed dims {vv!r} — the swapaxes chain needs a literal permutation"
            )
        if isinstance(d, bool) or not isinstance(d, (int, float)):
            raise UnsupportedOpError(f"permute: dim {d!r} is not an int")
        if isinstance(d, float) and not d.is_integer():
            raise UnsupportedOpError(f"permute: dim {d!r} is not an integer")
        p.append(int(d))
    n = len(p)
    if n == 0 or sorted(d % n for d in p) != list(range(n)):
        raise UnsupportedOpError(f"permute: dims {p!r} is not a permutation of range({n})")
    target = [d % n for d in p]
    cur = list(range(n))
    swaps: list[tuple[int, int]] = []
    for i in range(n):
        if cur[i] == target[i]:
            continue
        j = cur.index(target[i], i + 1)
        swaps.append((i, j))
        cur[i], cur[j] = cur[j], cur[i]
    expr = _tens(args[0], go)
    for i, j in swaps:
        expr = f"tensor.swapaxes({expr}, {i}, {j})"
    return expr


def _reshape_view(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        shape = _attr(kw, "shape", "dim", "arg1", "arg2", default=None)
        if shape is None and len(args) > 1:
            shape = args[1]
        if shape is None:
            raise UnsupportedOpError(f"{op}: missing target shape")
        dims = shape.value if isinstance(shape, Const) else shape
        if isinstance(dims, (Op, Var, Param)):
            raise UnsupportedOpError(
                f"{op}: computed shape {dims!r} — the host port takes a literal int list"
            )
        if isinstance(dims, (int, float)) and not isinstance(dims, bool):
            dims = [int(dims)]
        if isinstance(dims, (list, tuple)):
            dims = [_lit_dim(d) for d in dims]
            if -1 in dims:
                dims = _resolve_neg1(dims, args[0], op)
        else:
            raise UnsupportedOpError(f"{op}: shape {dims!r} is not an int list")
        return f"tensor.{op}({_tens(args[0], go)}, {_lit(dims)})"

    return build


def _transpose(args: list, kw: Mapping, go) -> str:
    d0 = _attr(kw, "dim0", "arg1", default=-2)
    d1 = _attr(kw, "dim1", "arg2", default=-1)
    return f"tensor.transpose({_tens(args[0], go)}, {_lit(d0)}, {_lit(d1)})"


def _squeeze(args: list, kw: Mapping, go) -> str:
    dim = _attr(kw, "dim", "arg1", default=_MISSING)
    if dim is _MISSING and len(args) > 1:
        dim = args[1]
    if dim is _MISSING:
        return f"tensor.squeeze({_tens(args[0], go)})"
    if isinstance(dim, (list, tuple)):
        dim_e = _lit([_lit_dim(d) for d in dim])
    elif isinstance(dim, Const):
        dim_e = _lit(dim.value)
    else:
        dim_e = _lit(dim)
    return f"tensor.squeeze({_tens(args[0], go)}, {dim_e})"


def _slice(args: list, kw: Mapping, go) -> str:
    start = _attr(kw, "start", "arg2", default=None)
    end = _attr(kw, "end", "arg3", default=None)
    dim = _attr(kw, "dim", "arg1", default=0)
    step = _attr(kw, "step", "arg4", default=1)
    se = "null" if start is None else _lit(start)
    ee = "null" if end is None else _lit(end)
    return f"tensor.slice({_tens(args[0], go)}, {_lit(dim)}, {se}, {ee}, {_lit(step)})"


def _chunk_split_unbind(op: str):
    # The IR spelling carries the selection as an ``index`` attr —
    # chunk/split spell it arg3, unbind arg2 (dim is arg1 there).
    _INDEX_ATTRS = {
        "chunk": ("index", "arg3"),
        "split": ("index", "arg3"),
        "unbind": ("index", "arg2"),
    }

    def build(args: list, kw: Mapping, go) -> str:
        index = _attr(kw, *_INDEX_ATTRS[op], default=0)
        index_e = _lit(index)
        if op == "chunk":
            chunks = _attr(kw, "chunks", "arg1", default=None)
            if chunks is None:
                raise UnsupportedOpError("chunk: missing chunks")
            dim = _attr(kw, "dim", "arg2", default=-1)
            inner = f"tensor.chunk({_tens(args[0], go)}, {_lit(chunks)}, {_lit(dim)})"
        elif op == "split":
            sizes = _attr(kw, "sizes", "arg1", default=None)
            if sizes is None:
                raise UnsupportedOpError("split: missing sizes")
            if isinstance(sizes, (int, float)) and not isinstance(sizes, bool):
                raise UnsupportedOpError(
                    "split: int sizes is torch's split-size semantics; "
                    "the host port implements equal-count split — "
                    "use a sizes list"
                )
            dim = _attr(kw, "dim", "arg2", default=-1)
            inner = (
                f"tensor.split({_tens(args[0], go)}, {_shape_list_emit(sizes, go)}, {_lit(dim)})"
            )
        else:  # unbind
            dim = _attr(kw, "dim", "arg1", default=0)
            inner = f"tensor.unbind({_tens(args[0], go)}, {_lit(dim)})"
        return f"({inner})[{index_e}]"

    return build


def _stack_concat(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        if not args:
            raise UnsupportedOpError(f"{op}: no tensor operands")
        dim = _attr(kw, "dim", "arg1", default=0)
        inner = ", ".join(_tens(a, go) for a in args)
        return f"tensor.{op}([{inner}], {_lit(dim)})"

    return build


def _positional_dims(kw: Mapping, start: int = 1) -> list[Any] | None:
    """Collect ``arg{start}…`` attr values in position order.

    Ops without an ``ATTR_SCHEMA`` entry (``expand``/``repeat`` size
    lists) spell their dims positionally; ``_normalize`` lands them
    under ``argN``.
    """
    vals: list[tuple[int, Any]] = []
    i = start
    while f"arg{i}" in kw:
        vals.append((i, kw[f"arg{i}"]))
        i += 1
    if not vals:
        return None
    return [v for _, v in vals]


def _expand(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        implicit = _attr(kw, "implicit", default=False)
        if implicit:
            raise UnsupportedOpError("expand: implicit=True is a host seam")
        v = _attr(kw, "shape", "dim", default=_MISSING)
        if v is not _MISSING:
            size = _shape_list_emit(v, go)
        elif len(args) > 1:
            size = "[" + ", ".join(_scalar_or_tensor(a, go) for a in args[1:]) + "]"
        elif (dims := _positional_dims(kw)) is not None:
            size = "[" + ", ".join(_lit(d) for d in dims) + "]"
        else:
            raise UnsupportedOpError("expand: missing target shape")
        return f"tensor.{op}({_tens(args[0], go)}, {size})"

    return build


def _repeat(args: list, kw: Mapping, go) -> str:
    v = _attr(kw, "shape", "dim", default=_MISSING)
    if v is not _MISSING:
        reps = _shape_list_emit(v, go)
    elif len(args) > 1:
        reps = "[" + ", ".join(_scalar_or_tensor(a, go) for a in args[1:]) + "]"
    elif (dims := _positional_dims(kw)) is not None:
        reps = "[" + ", ".join(_lit(d) for d in dims) + "]"
    else:
        raise UnsupportedOpError("repeat: missing repeats")
    return f"tensor.repeat({_tens(args[0], go)}, {reps})"


def _full(args: list, kw: Mapping, go) -> str:
    fill = args[0] if args else _attr(kw, "fill_value", "arg0", default=0)
    shape = _attr(kw, "shape", "dim", "arg1", "arg2", default=_MISSING)
    if shape is _MISSING and len(args) > 1:
        shape = args[1]
    if shape is _MISSING:
        raise UnsupportedOpError("full: missing shape")
    return f"tensor.full({_shape_list_emit(shape, go)}, {_scalar_or_tensor(fill, go)})"


def _arange(args: list, kw: Mapping, go) -> str:
    vals = list(args)
    if not vals:
        vals = [v for i in (1, 2, 3) if (v := _attr(kw, f"arg{i}", default=None)) is not None]
    if not vals or len(vals) > 3:
        raise UnsupportedOpError(f"arange: got {len(vals)} operands")
    return f"tensor.arange({', '.join(_scalar_or_tensor(v, go) for v in vals)})"


def _eye(args: list, kw: Mapping, go) -> str:
    n = _attr(kw, "dim", "n", "arg0", "arg1", default=_MISSING)
    m = _attr(kw, "m", "arg1", "arg2", default=_MISSING)
    if n is _MISSING:
        if not args:
            raise UnsupportedOpError("eye: missing n")
        n, m = args[0], (args[1] if len(args) > 1 else _MISSING)
    out = [_scalar_or_tensor(n, go)]
    if m is not _MISSING:
        out.append(_scalar_or_tensor(m, go))
    return f"tensor.eye({', '.join(out)})"


def _to(args: list, kw: Mapping, go) -> str:
    """``to`` — dtype/device cast; the host ``k_to`` honours real casts.

    ``k_to`` accepts a dtype string from :data:`_TO_DTYPES` (the f64
    identity plus the ``float32``/integer/``bool`` narrowings it
    implements honestly) or a tensor exemplar (``aten::to.other`` —
    a same-dtype copy → clone).  The emit-side ``dtype`` gate has
    already canonicalised an accepted ``dtype`` attr back into
    ``kw["dtype"]``; positional cast specs land as ``arg1`` (raw
    scalar tail) or a string ``Const`` exemplar operand — outside the
    gate's view — so every spelling is checked against the same
    table here.  A tensor exemplar and a dtype spec together have no
    aten spelling and decline by name.
    """
    if len(args) > 2:
        raise UnsupportedOpError(f"to: {len(args)} operands — no aten spelling carries that")
    exemplar = args[1] if len(args) > 1 else None
    spec_vals: list[Any] = []
    dtype_attr = _attr(kw, "dtype", default=None)
    if dtype_attr is not None:
        spec_vals.append(dtype_attr)
    arg1 = _attr(kw, "arg1", default=None)
    if arg1 is not None:
        spec_vals.append(arg1)
    if isinstance(exemplar, Const) and isinstance(exemplar.value, str):
        spec_vals.append(exemplar)
        exemplar = None
    canon: set[str] = set()
    for s in spec_vals:
        c = _to_dtype_canonical(s)
        if c is None:
            raise UnsupportedOpError(
                f"to: dtype={s!r} — outside the k_to table "
                "(float64/float32/int64/int32/int16/int8/uint8/bool "
                "+ aliases); f16/bf16/float8 and unknown names have "
                "no honest artifact"
            )
        canon.add(c)
    if len(canon) > 1:
        raise UnsupportedOpError(
            f"to: conflicting cast specs {sorted(canon)} — one aten::to carries one dtype"
        )
    if exemplar is not None:
        if canon:
            raise UnsupportedOpError(
                "to: tensor exemplar and dtype spec both present — no aten spelling carries both"
            )
        return f"tensor.to({_tens(args[0], go)}, {_tens(exemplar, go)})"
    target = next(iter(canon), "float64")
    return f'tensor.to({_tens(args[0], go)}, "{target}")'


def _where(args: list, kw: Mapping, go) -> str:
    """``where(cond, x, y)`` — x/y may ride ``arg1``/``arg2`` attrs."""
    slots = [_attr(kw, "arg1", default=None), _attr(kw, "arg2", default=None)]
    it = iter(args[1:])
    filled = [s if s is not None else next(it, None) for s in slots]
    x, y = filled
    if x is None or y is None:
        raise UnsupportedOpError("where: value operands missing (children or argN attrs)")
    if next(it, None) is not None:
        raise UnsupportedOpError("where: surplus value operands")
    return (
        f"tensor.where({_tens(args[0], go)}, "
        f"{_scalar_or_tensor(x, go)}, {_scalar_or_tensor(y, go)})"
    )


def _flip(args: list, kw: Mapping, go) -> str:
    dim = _attr(kw, "dim", "dims", default=_MISSING)
    if dim is _MISSING and len(args) > 1:
        dim = args[1]
    dims = _shape_list_emit(dim if dim is not _MISSING else (), go)
    return f"tensor.flip({_tens(args[0], go)}, {dims})"


# ---------------------------------------------------------------------------
#  Pair-returning ops + getitem — sort/topk/mode/kthvalue/median(dim)
#
#  The IR term for these IS the (values, indices) pair; the host port
#  returns a ``{"values": …, "indices": …}`` record, and a ``getitem``
#  consumer picks ``.values`` (index 0) / ``.indices`` (index 1) — the
#  same fold the torch binding's namedtuple pick performs.
# ---------------------------------------------------------------------------

_PAIR_FIELDS: dict[str, tuple[str, str]] = {
    "sort": ("values", "indices"),
    "argsort_pair": ("values", "indices"),  # unused — argsort is tensor-valued
    "topk": ("values", "indices"),
    "mode": ("values", "indices"),
    "kthvalue": ("values", "indices"),
    "median": ("values", "indices"),
}


def _pair_index(kw: Mapping) -> int:
    v = _attr(kw, "index", "arg1", default=0)
    if isinstance(v, Const):
        v = v.value
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise UnsupportedOpError(f"getitem: pair index {v!r} is not an int")
    i = int(v)
    if i not in (0, 1):
        raise UnsupportedOpError(f"getitem: pair index {i} — expected 0 (values) or 1 (indices)")
    return i


def _getitem(args: list, kw: Mapping, go) -> str:
    """``getitem(t, index)`` — pair-field pick or first-axis select."""
    if not args:
        raise UnsupportedOpError("getitem: no operand")
    child = args[0]
    idx = _pair_index(kw)
    if isinstance(child, Op):
        fields = _PAIR_FIELDS.get(child.op)
        if child.op == "median":
            # tensor.median with NO dim returns a bare scalar on the
            # host while torch/IR spell the (values, indices) pair —
            # only the dim variant has an honest pair form.
            cattrs = dict(child.attrs)
            if _attr(cattrs, "dim", "arg1", default=None) is None:
                raise UnsupportedOpError(
                    "getitem: whole-tensor median has no host pair form "
                    "(tensor.median returns a scalar; torch returns a pair)"
                )
        if fields is not None:
            return f"({go(child)}).{fields[idx]}"
        if child.op == "pack":
            return f"({go(child)})[{_lit(idx)}]"
        if child.op in ("chunk", "split", "unbind"):
            raise UnsupportedOpError(
                f"getitem over {child.op}: the IR spells the pick as the "
                "op's own index attr, not a getitem wrapper"
            )
    # Tensor getitem t[i] — first-axis select (negative index wraps,
    # matching torch and the host select port).
    return f"tensor.select({go(child)}, 0, {_lit(idx)})"


def _sortlike(op: str):
    """``sort``/``topk``/``mode``/``kthvalue`` pair builders."""

    def build(args: list, kw: Mapping, go) -> str:
        t = _tens(args[0], go)
        if op == "topk":
            k = _attr(kw, "k", "arg1", default=_MISSING)
            if k is _MISSING and len(args) > 1:
                k = args[1]
            if k is _MISSING:
                raise UnsupportedOpError("topk: missing k")
            dim = _attr(kw, "dim", "arg2", default=-1)
            largest = _attr(kw, "largest", "arg3", default=True)
            sorted_ = _attr(kw, "sorted", "arg4", default=True)
            return (
                f"tensor.topk({t}, {_scalar_or_tensor(k, go)}, "
                f"{_lit(dim)}, {_lit(bool(largest))}, {_lit(bool(sorted_))})"
            )
        if op == "kthvalue":
            k = _attr(kw, "k", "arg1", default=_MISSING)
            if k is _MISSING and len(args) > 1:
                k = args[1]
            if k is _MISSING:
                raise UnsupportedOpError("kthvalue: missing k")
            dim = _attr(kw, "dim", "arg2", default=-1)
            keep = _attr(kw, "keepdim", "arg3", default=False)
            return (
                f"tensor.kthvalue({t}, {_scalar_or_tensor(k, go)}, {_lit(dim)}, {_lit(bool(keep))})"
            )
        if op == "mode":
            dim = _attr(kw, "dim", "arg1", default=-1)
            keep = _attr(kw, "keepdim", "arg2", default=False)
            return f"tensor.mode({t}, {_lit(dim)}, {_lit(bool(keep))})"
        dim = _attr(kw, "dim", "arg1", default=-1)
        desc = _attr(kw, "descending", "arg2", default=False)
        return f"tensor.{op}({t}, {_lit(dim)}, {_lit(bool(desc))})"

    return build


def _median(args: list, kw: Mapping, go) -> str:
    dim = _attr(kw, "dim", "arg1", default=None)
    if dim is None and len(args) > 1:
        dim = args[1]
    if dim is None:
        raise UnsupportedOpError(
            "median: whole-tensor median has no host pair form — "
            "the dim variant returns {values, indices} like torch"
        )
    keep = _attr(kw, "keepdim", "arg2", default=False)
    return f"tensor.median({_tens(args[0], go)}, {_lit(dim)}, {_lit(bool(keep))})"


def _logsumexp(args: list, kw: Mapping, go) -> str:
    dim = _attr(kw, "dim", "arg1", default=_MISSING)
    if dim is _MISSING and len(args) > 1:
        dim = args[1]
    if dim is _MISSING or dim is None:
        raise UnsupportedOpError("logsumexp: dim is required (no whole-tensor host form)")
    keep = _attr(kw, "keepdim", "arg2", default=False)
    # dim arrives unwrapped (int or int-list); the host takes either,
    # a lone int wraps as a one-element list for a uniform emit.
    dims = _shape_list_emit(dim, go) if isinstance(dim, (list, tuple)) else _lit([_lit_dim(dim)])
    return f"tensor.logsumexp({_tens(args[0], go)}, {dims}, {_lit(bool(keep))})"


def _gather(args: list, kw: Mapping, go) -> str:
    if len(args) < 2:
        raise UnsupportedOpError("gather: missing index operand")
    dim = _attr(kw, "dim", "arg1", default=0)
    return f"tensor.gather({_tens(args[0], go)}, {_lit(dim)}, {_tens(args[1], go)})"


def _index_select(args: list, kw: Mapping, go) -> str:
    dim = _attr(kw, "dim", "arg1", default=0)
    idx = args[1] if len(args) > 1 else _attr(kw, "index", "arg2", default=None)
    if idx is None:
        raise UnsupportedOpError("index_select: missing index (operand or attr)")
    if isinstance(idx, (list, tuple)) and not isinstance(idx, Const):
        flat = [int(d) for d in idx]
        idx_e = f"tensor.from_list({_lit(flat)})"
    elif isinstance(idx, Const) and isinstance(idx.value, (list, tuple)):
        idx_e = f"tensor.from_list({_lit([int(d) for d in idx.value])})"
    else:
        idx_e = _tens(idx, go)
    return f"tensor.index_select({_tens(args[0], go)}, {_lit(dim)}, {idx_e})"


def _embedding(args: list, kw: Mapping, go) -> str:
    """``embedding(W, idx)`` — the row gather spelled as ``index_select``.

    The host surface has no ``tensor.embedding``; ``index_select(W, 0,
    idx)`` is the same gather for the spelling every shipped term
    carries — a rank-1 index operand (the arch/lifted terms flatten
    the ids and the parent ``reshape`` restores ``idx.shape + (d,)``,
    exactly what ``index_select`` returns for a rank-1 index).  A
    rank>1 index would decline honestly at eval — ``index_select``
    requires a rank-1 index — rather than silently mis-shape.  The
    scalar tail (``padding_idx``/``scale_grad_by_freq``/…) is
    gradient/storage metadata with no forward effect on real
    checkpoints, except ``padding_idx`` itself: the zero-fill is only
    forward-equal while the table row is already zeros, so a
    non-``None`` value declines by name.
    """
    if len(args) < 2:
        raise UnsupportedOpError("embedding: needs (weight, indices)")
    if _attr(kw, "padding_idx", "arg2", default=None) is not None:
        raise UnsupportedOpError("embedding: padding_idx has no honest index_select spelling")
    # Same operand positions as index_select(self, dim=0, index):
    # weight → self, indices → index; the dim attr defaults to 0.
    return _index_select(args, kw, go)


def _index_add(args: list, kw: Mapping, go) -> str:
    if len(args) < 3:
        raise UnsupportedOpError("index_add: needs (self, index, source)")
    dim = _attr(kw, "dim", "arg1", default=0)
    alpha = _attr(kw, "alpha", "arg4", default=1)
    return (
        f"tensor.index_add({_tens(args[0], go)}, {_lit(dim)}, "
        f"{_tens(args[1], go)}, {_tens(args[2], go)}, "
        f"{_scalar_or_tensor(alpha, go)})"
    )


def _searchsorted(args: list, kw: Mapping, go) -> str:
    if len(args) < 2:
        raise UnsupportedOpError("searchsorted: needs (sorted_sequence, values)")
    if _attr(kw, "sorter", "arg4", "arg5", default=None) is not None:
        raise UnsupportedOpError("searchsorted: sorter= has no honest host form")
    o32 = _attr(kw, "out_int32", "arg2", default=False)
    right = _attr(kw, "right", "arg3", default=False)
    return (
        f"tensor.searchsorted({_tens(args[0], go)}, {_tens(args[1], go)}, "
        f"{_lit(bool(o32))}, {_lit(bool(right))})"
    )


def _einsum(args: list, kw: Mapping, go) -> str:
    eq = _attr(kw, "equation", "arg0", default=_MISSING)
    if eq is _MISSING:
        raise UnsupportedOpError("einsum: missing equation")
    if isinstance(eq, Const):
        eq = eq.value
    if not isinstance(eq, str):
        raise UnsupportedOpError(f"einsum: equation {eq!r} is not a string")
    if not args:
        raise UnsupportedOpError("einsum: no tensor operands")
    operands = ", ".join(_tens(a, go) for a in args)
    # The host port takes the operand tensors as ONE list argument.
    return f"tensor.einsum({_lit(eq)}, [{operands}])"


def _diag(args: list, kw: Mapping, go) -> str:
    diagonal = _attr(kw, "diagonal", "arg1", default=0)
    return f"tensor.diag({_tens(args[0], go)}, {_lit(diagonal)})"


def _diagonal(args: list, kw: Mapping, go) -> str:
    offset = _attr(kw, "offset", "arg1", default=0)
    dim1 = _attr(kw, "dim1", "arg2", default=0)
    dim2 = _attr(kw, "dim2", "arg3", default=1)
    return f"tensor.diagonal({_tens(args[0], go)}, {_lit(offset)}, {_lit(dim1)}, {_lit(dim2)})"


def _movedim(args: list, kw: Mapping, go) -> str:
    src = _attr(kw, "source", "arg1", default=_MISSING)
    dst = _attr(kw, "destination", "arg2", default=_MISSING)
    if src is _MISSING and len(args) > 1:
        src = args[1]
    if dst is _MISSING and len(args) > 2:
        dst = args[2]
    if src is _MISSING or dst is _MISSING:
        raise UnsupportedOpError("movedim: missing source/destination")

    def dims(v: Any) -> str:
        vv = v.value if isinstance(v, Const) else v
        if isinstance(vv, (int, float)) and not isinstance(vv, bool):
            return _lit([int(vv)])
        if isinstance(vv, (list, tuple)):
            return _lit([int(d) for d in vv])
        raise UnsupportedOpError(f"movedim: dims {v!r} is not int-or-int-list")

    # The host moveaxis port takes LIST dims (int args mis-decode).
    return f"tensor.moveaxis({_tens(args[0], go)}, {dims(src)}, {dims(dst)})"


def _rot90(args: list, kw: Mapping, go) -> str:
    k = _attr(kw, "k", "arg1", default=1)
    dims = _attr(kw, "dims", "arg2", default=[0, 1])
    return f"tensor.rot90({_tens(args[0], go)}, {_lit(k)}, {_shape_list_emit(dims, go)})"


def _pad(args: list, kw: Mapping, go) -> str:
    pad = _attr(kw, "pad", "arg1", default=_MISSING)
    if pad is _MISSING and len(args) > 1:
        pad = args[1]  # args[0] is the padded tensor
    if pad is _MISSING:
        raise UnsupportedOpError("pad: missing pad widths")
    mode = _attr(kw, "mode", "arg2", default="constant")
    if mode not in (None, "constant"):
        raise UnsupportedOpError(f"pad: mode={mode!r} — host port is constant-pad only")
    value = _attr(kw, "value", "arg3", default=None)
    pad_e = f"tensor.constant_pad_nd({_tens(args[0], go)}, {_shape_list_emit(pad, go)}"
    # constant_pad_nd takes 2 or 3 args — a null fill is the host's
    # implicit zero, so only emit the third slot for a real value.
    if value is None:
        return pad_e + ")"
    return f"{pad_e}, {_scalar_or_tensor(value, go)})"


def _avg_pool2d(args: list, kw: Mapping, go) -> str:
    def pair(name: str, i: int, default: Any = None) -> Any:
        v = _attr(kw, name, f"arg{i}", default=_MISSING)
        return default if v is _MISSING else v

    kernel = pair("kernel_size", 1)
    stride = pair("stride", 2, [])
    padding = pair("padding", 3, 0)
    dilation = pair("dilation", 4, 1)
    ceil = pair("ceil_mode", 5, False)
    cip = pair("count_include_pad", 6, True)
    div = pair("divisor_override", 7, None)
    if kernel is None:
        raise UnsupportedOpError("avg_pool2d: missing kernel_size")
    dil = dilation if isinstance(dilation, (list, tuple)) else [dilation, dilation]
    if list(dil) not in ([1, 1], [1]):
        raise UnsupportedOpError(f"avg_pool2d: dilation={dilation!r} != 1")
    return (
        f"tensor.avg_pool2d({_tens(args[0], go)}, {_lit(kernel)}, {_lit(stride)}, "
        f"{_lit(padding)}, {_lit(bool(ceil))}, {_lit(bool(cip))}, "
        f"{'null' if div is None else _lit(div)})"
    )


def _max_pool2d(args: list, kw: Mapping, go) -> str:
    def pair(name: str, i: int, default: Any = None) -> Any:
        v = _attr(kw, name, f"arg{i}", default=_MISSING)
        return default if v is _MISSING else v

    kernel = pair("kernel_size", 1)
    stride = pair("stride", 2, [])
    padding = pair("padding", 3, 0)
    dilation = pair("dilation", 4, 1)
    ceil = pair("ceil_mode", 5, False)
    if kernel is None:
        raise UnsupportedOpError("max_pool2d: missing kernel_size")
    pad_list = padding if isinstance(padding, (list, tuple)) else [padding, padding]
    dil = dilation if isinstance(dilation, (list, tuple)) else [dilation, dilation]
    if list(pad_list) != [0, 0] or list(dil) != [1, 1] or ceil:
        raise UnsupportedOpError(
            "max_pool2d: host port covers kernel/stride only — "
            f"padding={padding!r} dilation={dilation!r} ceil_mode={ceil!r}"
        )
    stride_e = "" if not stride else f", {_lit(stride)}"
    return f"tensor.max_pool({_tens(args[0], go)}, {_lit(kernel)}{stride_e})"


def _cdist(args: list, kw: Mapping, go) -> str:
    if len(args) < 2:
        raise UnsupportedOpError("cdist: needs (x1, x2)")
    p = _attr(kw, "p", "arg2", default=2.0)
    return f"tensor.cdist({_tens(args[0], go)}, {_tens(args[1], go)}, {_lit(p)})"


def _renorm(args: list, kw: Mapping, go) -> str:
    # aten.renorm(self, p, dim, maxnorm) — all-scalar tail folds to argN.
    p = _attr(kw, "p", "arg1", default=_MISSING)
    dim = _attr(kw, "dim", "arg2", default=_MISSING)
    maxnorm = _attr(kw, "maxnorm", "arg3", default=_MISSING)
    if p is _MISSING and len(args) > 1:
        p = args[1]
    if dim is _MISSING and len(args) > 2:
        dim = args[2]
    if maxnorm is _MISSING and len(args) > 3:
        maxnorm = args[3]
    if _MISSING in (p, dim, maxnorm):
        raise UnsupportedOpError("renorm: needs p, dim, maxnorm")
    return (
        f"tensor.renorm({_tens(args[0], go)}, {_scalar_or_tensor(p, go)}, "
        f"{_scalar_or_tensor(dim, go)}, {_scalar_or_tensor(maxnorm, go)})"
    )


def _beta_alpha(kw: Mapping, base: int) -> tuple[str, str]:
    beta = _attr(kw, "beta", f"arg{base}", default=1)
    alpha = _attr(kw, "alpha", f"arg{base + 1}", default=1)
    return _lit(beta), _lit(alpha)


def _addmm(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        if len(args) < 3:
            raise UnsupportedOpError(f"{op}: needs (self, mat1, mat2)")
        beta, alpha = _beta_alpha(kw, 3)
        return (
            f"tensor.{op}({_tens(args[0], go)}, {_tens(args[1], go)}, "
            f"{_tens(args[2], go)}, {beta}, {alpha})"
        )

    return build


def _addmv_addr(op: str, n_mats: int):
    def build(args: list, kw: Mapping, go) -> str:
        if len(args) < 1 + n_mats:
            raise UnsupportedOpError(f"{op}: missing operands")
        beta, alpha = _beta_alpha(kw, 1 + n_mats)
        mats = ", ".join(_tens(a, go) for a in args[: 1 + n_mats])
        return f"tensor.{op}({mats}, {beta}, {alpha})"

    return build


def _linspace_logspace(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        names = ("start", "end", "steps") if op == "linspace" else ("start", "end", "steps", "base")
        vals = []
        for i, name in enumerate(names):
            v = _attr(kw, name, f"arg{i}", default=_MISSING)
            if v is _MISSING and i < len(args):
                v = args[i]
            if v is _MISSING:
                if op == "logspace" and name == "base":
                    v = 10.0
                elif op == "linspace" and name == "steps":
                    v = 100
                else:
                    raise UnsupportedOpError(f"{op}: missing {name}")
            vals.append(_scalar_or_tensor(v, go))
        return f"tensor.{op}({', '.join(vals)})"

    return build


def _bincount(args: list, kw: Mapping, go) -> str:
    if not args:
        raise UnsupportedOpError("bincount: no operand")
    weights = args[1] if len(args) > 1 else _attr(kw, "weights", "arg1", default=None)
    minlength = _attr(kw, "minlength", "arg2", default=0)
    w_e = "null" if weights is None else _tens(weights, go)
    return f"tensor.bincount({_tens(args[0], go)}, {w_e}, {_lit(minlength)})"


def _quantile(args: list, kw: Mapping, go) -> str:
    q = args[1] if len(args) > 1 else _attr(kw, "q", "arg1", default=_MISSING)
    if q is _MISSING:
        raise UnsupportedOpError("quantile: missing q")
    dim = _attr(kw, "dim", "arg2", default=None)
    keep = _attr(kw, "keepdim", "arg3", default=False)
    interp = _attr(kw, "interpolation", "arg4", default="linear")
    dim_e = "null" if dim is None else _scalar_or_tensor(dim, go)
    return (
        f"tensor.quantile({_tens(args[0], go)}, {_scalar_or_tensor(q, go)}, "
        f"{dim_e}, {_lit(bool(keep))}, {_lit(interp)})"
    )


def _tensordot(args: list, kw: Mapping, go) -> str:
    if len(args) < 2:
        raise UnsupportedOpError("tensordot: needs (a, b)")
    ds = _attr(kw, "dims_self", "arg2", default=_MISSING)
    do = _attr(kw, "dims_other", "arg3", default=_MISSING)
    if ds is _MISSING or do is _MISSING:
        raise UnsupportedOpError("tensordot: needs dims_self and dims_other")
    return (
        f"tensor.tensordot({_tens(args[0], go)}, {_tens(args[1], go)}, "
        f"{_shape_list_emit(ds, go)}, {_shape_list_emit(do, go)})"
    )


def _dist(args: list, kw: Mapping, go) -> str:
    if len(args) < 2:
        raise UnsupportedOpError("dist: needs (a, b)")
    p = _attr(kw, "p", "arg2", default=2.0)
    return f"tensor.dist({_tens(args[0], go)}, {_tens(args[1], go)}, {_lit(p)})"


def _allclose(args: list, kw: Mapping, go) -> str:
    if len(args) < 2:
        raise UnsupportedOpError("allclose: needs (a, b)")
    rtol = _attr(kw, "rtol", "arg2", default=1e-5)
    atol = _attr(kw, "atol", "arg3", default=1e-8)
    eqnan = _attr(kw, "equal_nan", "arg4", default=False)
    return (
        f"tensor.allclose({_tens(args[0], go)}, {_tens(args[1], go)}, "
        f"{_scalar_or_tensor(rtol, go)}, {_scalar_or_tensor(atol, go)}, "
        f"{_lit(bool(eqnan))})"
    )


def _pdist(args: list, kw: Mapping, go) -> str:
    p = _attr(kw, "p", "arg1", default=2.0)
    return f"tensor.pdist({_tens(args[0], go)}, {_lit(p)})"


def _cross(args: list, kw: Mapping, go) -> str:
    if len(args) < 2:
        raise UnsupportedOpError("cross: needs (a, b)")
    dim = _attr(kw, "dim", "arg2", default=None)
    dim_e = "null" if dim is None else _scalar_or_tensor(dim, go)
    return f"tensor.cross({_tens(args[0], go)}, {_tens(args[1], go)}, {dim_e})"


def _norm(args: list, kw: Mapping, go) -> str:
    """``norm(x, p?)`` — the host port is ``aten::norm.Scalar``:
    ``(self, p)`` over the whole tensor, no dim/keepdim slots.

    The previous spelling emitted ``(x, p, dim, keepdim)`` — four
    args against a ``max_args=2`` port (drift-guard finding,
    ``tests/test_emit_surface.py``).  Dim'd norms have no honest host
    form and decline by name.
    """
    dim = _attr(kw, "dim", "arg2", default=None)
    if dim is not None:
        raise UnsupportedOpError(
            "norm: dim= has no host form — tensor.norm is aten::norm.Scalar "
            "(whole-tensor); dim'd norms need a linalg port the surface lacks"
        )
    keep = _attr(kw, "keepdim", "arg3", default=False)
    if keep:
        raise UnsupportedOpError("norm: keepdim= has no host slot (aten::norm.Scalar)")
    rest = args[1:]
    if len(rest) > 1:
        raise UnsupportedOpError("norm: surplus operands")
    p = rest[0] if rest else _attr(kw, "p", "arg1", default=2)
    pv = p.value if isinstance(p, Const) else p
    if isinstance(pv, (Op, Var, Param, bool)) or not isinstance(pv, (int, float)):
        raise UnsupportedOpError(f"norm: p={pv!r} — the host Scalar p slot takes a number")
    return f"tensor.norm({_tens(args[0], go)}, {_scalar_or_tensor(p, go)})"


def _nonzero(args: list, kw: Mapping, go) -> str:
    as_tuple = _attr(kw, "as_tuple", default=False)
    if as_tuple:
        raise UnsupportedOpError(
            "nonzero: as_tuple=True has no host form — tensor.nonzero returns the index tensor"
        )
    return f"tensor.nonzero({_tens(args[0], go)})"


def _batch_broadcast(ba: tuple, bb: tuple) -> tuple | None:
    """Torch's right-aligned batch-dim broadcast; ``None`` if incompatible.

    Missing leading dims count as 1; a dim pair broadcasts only when
    one side is 1 or they agree — the same rule ``matmul_dims`` in
    host-tensor applies to its single batch dim.
    """
    n = max(len(ba), len(bb))
    out: list[int] = []
    for i in range(n):
        da = ba[i - (n - len(ba))] if i >= n - len(ba) else 1
        db = bb[i - (n - len(bb))] if i >= n - len(bb) else 1
        if da != db and da != 1 and db != 1:
            return None
        out.append(max(da, db))
    return tuple(out)


def _matmul(args: list, kw: Mapping, go) -> str:
    """``matmul`` — rank>3 operands flatten their batch dims to rank-3.

    The host port is rank-2/3 only: ``[B,m,k]×[B,k,n]`` with the single
    batch dim broadcasting on 1.  Torch's rank-N spelling broadcasts
    *all* leading dims (whisper attention's ``[B,H,T,D]`` scores
    matmuls); when ``shape_of`` proves a rank>3 operand we merge its
    leading dims into the host's one batch dim and reshape the rank-3
    product back to torch's broadcast output shape.  That flatten is
    honest only when every operand's batch *product* already equals
    the broadcast product (or is all-ones, which the host batch
    broadcast covers) — a real expand has no spelling and declines by
    name.  Ranks ``shape_of`` cannot prove pass through unchanged:
    the host still enforces its own rank bound at eval.
    """
    a, b = args[0], args[1]
    ea, eb = _tens(a, go), _tens(b, go)
    passthrough = f"tensor.matmul({ea}, {eb})"
    sa, sb = shape_of(a), shape_of(b)
    if (
        not isinstance(sa, tuple)
        or not isinstance(sb, tuple)
        or len(sa) < 2
        or len(sb) < 2
        or (len(sa) <= 3 and len(sb) <= 3)
    ):
        return passthrough
    if any(d is None for d in (*sa, *sb)):
        raise UnsupportedOpError(
            "matmul: rank>3 operand carries unknown dims — the batch flatten needs concrete extents"
        )
    sa_i, sb_i = [int(d) for d in sa], [int(d) for d in sb]
    if sa_i[-1] != sb_i[-2]:
        # Inner dims disagree — torch-invalid geometry; the plain
        # spelling lets the host refuse loudly rather than reshape
        # around a broken term.
        return passthrough
    ba, bb = tuple(sa_i[:-2]), tuple(sb_i[:-2])
    bstar = _batch_broadcast(ba, bb)
    if bstar is None:
        # Torch-invalid operand geometry — emit the plain spelling and
        # let the host refuse loudly rather than guess a reshape.
        return passthrough
    flat = 1
    for d in bstar:
        flat *= d

    def _prod(ds: tuple) -> int:
        p = 1
        for d in ds:
            p *= d
        return p

    pa, pb = _prod(ba), _prod(bb)
    if pa not in (1, flat) or pb not in (1, flat):
        raise UnsupportedOpError(
            f"matmul: batch dims {ba}x{bb} broadcast to {bstar} — "
            "flattening needs an expand the host surface lacks"
        )
    fa = flat if pa == flat else 1
    fb = flat if pb == flat else 1
    if len(sa_i) > 3:
        ea = f"tensor.reshape({ea}, [{fa}, {sa_i[-2]}, {sa_i[-1]}])"
    if len(sb_i) > 3:
        eb = f"tensor.reshape({eb}, [{fb}, {sb_i[-2]}, {sb_i[-1]}])"
    out = [*bstar, sa_i[-2], sb_i[-1]]
    return f"tensor.reshape(tensor.matmul({ea}, {eb}), [{', '.join(str(d) for d in out)}])"


# ---------------------------------------------------------------------------
#  The op table — IR op → emitted ``tensor.*`` expression.
#
#  Coverage is the measured host-tensor port surface (203 ops — each
#  builder's arity/order verified against ``hnl-run``; pair-returning
#  reductions expose the host's ``{values, indices}`` record to a
#  ``getitem`` consumer).  Every builder returns the full ``.hnl`` RHS
#  expression; ``tensor.<op>(…)`` is spelled here, ``(tensor.chunk(…))[i]``
#  style where the host returns a tensor array.
#
#  The ``_u1``/``_u2`` positional-passthrough entries are *candidates*:
#  ``_canonicalize_mechanical`` (below the table) swaps the ones
#  ``host_surface.json`` fully determines for generated builders —
#  ~77 of the 203 today; the rest of the table is hand-written because
#  the artifact does not record the IR-side binding conventions.
# ---------------------------------------------------------------------------


def _full_like_fill(args: list, kw: Mapping) -> Any:
    return args[1] if len(args) > 1 else _attr(kw, "fill_value", "arg1", default=0)


#: IR name → positional arity for the ``_u1``/``_u2`` mechanical
#: candidates (see ``_canonicalize_mechanical``).  The factories record
#: themselves here at ``_OPS``-table build time; the artifact decides
#: which candidates become generated builders.
_MECH_TAGS: dict[str, int] = {}


def _u1(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        return f"tensor.{op}({_tens(args[0], go)})"

    _MECH_TAGS[op] = 1
    return build


def _u2(op: str):
    def build(args: list, kw: Mapping, go) -> str:
        return f"tensor.{op}({_tens(args[0], go)}, {_tens(args[1], go)})"

    _MECH_TAGS[op] = 2
    return build


def _bin_scalar_rhs(op: str):
    """Binary op whose rhs slot accepts a bare scalar (``Scalar|Tensor``).

    Verified against ``hnl-run``: ``pow``/``fmod``/``remainder``/
    ``xlogy``/the comparisons take number literals; ``maximum``/
    ``minimum``/``atan2``/``heaviside``/``lerp``/the elementwise four
    require a Tensor rhs (scalars wrap rank-0 via ``_tens``).
    """

    def build(args: list, kw: Mapping, go) -> str:
        return f"tensor.{op}({_tens(args[0], go)}, {_scalar_or_tensor(args[1], go)})"

    return build


_OPS: dict[str, Any] = {
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
    "isnan": _u1("isnan"),
    "isinf": _u1("isinf"),
    "isfinite": _u1("isfinite"),
    "isneginf": _u1("isneginf"),
    "isposinf": _u1("isposinf"),
    "logical_not": _u1("logical_not"),
    "contiguous": _u1("contiguous"),
    "clone": _u1("clone"),
    "detach": _u1("detach"),
    "numel": _u1("numel"),
    "item": _u1("item"),
    "zeros_like": _u1("zeros_like"),
    "ones_like": _u1("ones_like"),
    "expand_as": _u2("expand_as"),
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
    "fmod": _bin_scalar_rhs("fmod"),
    "remainder": _bin_scalar_rhs("remainder"),
    "xlogy": _bin_scalar_rhs("xlogy"),
    "pow": _bin_scalar_rhs("pow"),
    "eq": _bin_scalar_rhs("eq"),
    "ne": _bin_scalar_rhs("ne"),
    "lt": _bin_scalar_rhs("lt"),
    "le": _bin_scalar_rhs("le"),
    "gt": _bin_scalar_rhs("gt"),
    "ge": _bin_scalar_rhs("ge"),
    # --- activations with scalar attrs ------------------------------------
    "gelu": lambda a, k, g: (
        f"tensor.gelu({_tens(a[0], g)}, {_attr_lit(k, 'approximate', 'arg1', default='none')})"
    ),
    "elu": lambda a, k, g: (
        f"tensor.elu({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'alpha', default=1.0))}, "
        f"{_lit(_attr(k, 'scale', default=1.0))}, "
        f"{_lit(_attr(k, 'input_scale', default=1.0))})"
    ),
    "celu": lambda a, k, g: (
        f"tensor.celu({_tens(a[0], g)}, {_lit(_attr(k, 'alpha', 'arg1', default=1.0))})"
    ),
    "leaky_relu": lambda a, k, g: (
        f"tensor.leaky_relu({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'negative_slope', 'arg1', default=0.01))})"
    ),
    "softplus": lambda a, k, g: (
        f"tensor.softplus({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'beta', default=1))}, "
        f"{_lit(_attr(k, 'threshold', default=20))})"
    ),
    "hardtanh": lambda a, k, g: (
        f"tensor.hardtanh({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'min', default=-1.0))}, "
        f"{_lit(_attr(k, 'max', default=1.0))})"
    ),
    "nan_to_num": lambda a, k, g: (
        f"tensor.nan_to_num({_tens(a[0], g)}, "
        f"{_attr_lit(k, 'nan', 'arg1', default=0.0)}, "
        f"{_attr_lit(k, 'posinf', 'arg2', default=None)}, "
        f"{_attr_lit(k, 'neginf', 'arg3', default=None)})"
    ),
    "heaviside": _u2("heaviside"),
    "clamp": lambda a, k, g: f"tensor.clamp({', '.join(_clamp_args(a, k, g))})",
    "clamp_min": lambda a, k, g: (
        f"tensor.clamp_min({_tens(a[0], g)}, "
        f"{_scalar_or_tensor(a[1] if len(a) > 1 else _attr(k, 'min', 'arg1', default=None), g)})"
    ),
    "clamp_max": lambda a, k, g: (
        f"tensor.clamp_max({_tens(a[0], g)}, "
        f"{_scalar_or_tensor(a[1] if len(a) > 1 else _attr(k, 'max', 'arg1', default=None), g)})"
    ),
    "lerp": lambda a, k, g: f"tensor.lerp({_tens(a[0], g)}, {_tens(a[1], g)}, {_tens(a[2], g)})",
    "where": _where,
    # --- reduce-dim --------------------------------------------------------
    "sum": _reduce("sum"),
    "prod": _reduce("prod"),
    "mean": _reduce("mean"),
    "amax": _reduce("amax"),
    "amin": _reduce("amin"),
    "nanmean": _reduce("nanmean"),
    "nansum": _reduce("nansum"),
    "max": _max_min("max", True),
    "min": _max_min("min", False),
    "argmax": _arg_reduce("argmax"),
    "argmin": _arg_reduce("argmin"),
    "all": _all_any("all", True),
    "any": _all_any("any", False),
    "count_nonzero": lambda a, k, g: (
        f"tensor.count_nonzero({_tens(a[0], g)}, "
        f"{_lit(_dim_list(_attr(k, 'dim', 'axis', 'arg1', default=None)))})"
    ),
    "var": _var_std("var", False),
    "std": _var_std("std", True),
    "cumsum": lambda a, k, g: (
        f"tensor.cumsum({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'dim', 'arg1', default=a[1] if len(a) > 1 else 0))})"
    ),
    "cumprod": lambda a, k, g: (
        f"tensor.cumprod({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'dim', 'arg1', default=a[1] if len(a) > 1 else 0))})"
    ),
    "softmax": lambda a, k, g: (
        f"tensor.softmax({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'dim', 'arg1', default=a[1] if len(a) > 1 else -1))})"
    ),
    "log_softmax": lambda a, k, g: (
        f"tensor.log_softmax({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'dim', 'arg1', default=a[1] if len(a) > 1 else -1))})"
    ),
    "layer_norm": _layer_norm,
    "rms_norm": _rms_norm,
    # --- linalg -------------------------------------------------------------
    "matmul": _matmul,
    "outer": _u2("outer"),
    "linear": _linear,
    "conv2d": _conv2d,
    "conv1d": _conv1d,
    # --- shape ---------------------------------------------------------------
    "reshape": _reshape_view("reshape"),
    "view": _reshape_view("view"),
    "transpose": _transpose,
    "permute": _permute,
    "unsqueeze": lambda a, k, g: (
        f"tensor.unsqueeze({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'dim', 'arg1', default=a[1] if len(a) > 1 else -1))})"
    ),
    "squeeze": _squeeze,
    "flatten": lambda a, k, g: (
        f"tensor.flatten({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'start_dim', 'arg1', default=0))}, "
        f"{_lit(_attr(k, 'end_dim', 'arg2', default=-1))})"
    ),
    "narrow": lambda a, k, g: (
        f"tensor.narrow({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'dim', default=0))}, "
        f"{_lit(_attr(k, 'start', default=0))}, "
        f"{_lit(_attr(k, 'length', default=1))})"
    ),
    "select": lambda a, k, g: (
        f"tensor.select({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'dim', default=0))}, "
        f"{_lit(_attr(k, 'index', default=0))})"
    ),
    "slice": _slice,
    "chunk": _chunk_split_unbind("chunk"),
    "split": _chunk_split_unbind("split"),
    "unbind": _chunk_split_unbind("unbind"),
    "stack": _stack_concat("stack"),
    "concat": _stack_concat("concat"),
    "expand": _expand("expand"),
    "repeat": _repeat,
    "tril": lambda a, k, g: (
        f"tensor.tril({_tens(a[0], g)}, {_lit(_attr(k, 'diagonal', 'arg1', default=0))})"
    ),
    "triu": lambda a, k, g: (
        f"tensor.triu({_tens(a[0], g)}, {_lit(_attr(k, 'diagonal', 'arg1', default=0))})"
    ),
    "flip": _flip,
    "pixel_shuffle": lambda a, k, g: (
        f"tensor.pixel_shuffle({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'upscale_factor', 'arg1', default=1))})"
    ),
    "pixel_unshuffle": lambda a, k, g: (
        f"tensor.pixel_unshuffle({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'downscale_factor', 'arg1', default=1))})"
    ),
    # --- constructors / metadata ---------------------------------------------
    "zeros": lambda a, k, g: f"tensor.zeros({_shape_arg(a, k, 'zeros', g)})",
    "ones": lambda a, k, g: f"tensor.ones({_shape_arg(a, k, 'ones', g)})",
    "full": _full,
    "full_like": lambda a, k, g: (
        f"tensor.full_like({_tens(a[0], g)}, {_scalar_or_tensor(_full_like_fill(a, k), g)})"
    ),
    "arange": _arange,
    "eye": _eye,
    "to": _to,
    # --- pair-returning + getitem -----------------------------------------
    # sort/topk/mode/kthvalue/median(dim) evaluate to the host's
    # {values, indices} record; a ``getitem`` consumer picks a field.
    "sort": _sortlike("sort"),
    "topk": _sortlike("topk"),
    "mode": _sortlike("mode"),
    "kthvalue": _sortlike("kthvalue"),
    "median": _median,
    "argsort": lambda a, k, g: (
        f"tensor.argsort({_tens(a[0], g)}, "
        f"{_lit(_attr(k, 'dim', 'arg1', default=-1))}, "
        f"{_lit(bool(_attr(k, 'descending', 'arg2', default=False)))})"
    ),
    "getitem": _getitem,
    # --- gather/scatter family ---------------------------------------------
    "embedding": _embedding,
    "gather": _gather,
    "index_select": _index_select,
    "index_add": _index_add,
    "searchsorted": _searchsorted,
    "take": _u2("take"),
    # --- activations / pointwise extras -------------------------------------
    "selu": _u1("selu"),
    "signbit": _u1("signbit"),
    "deg2rad": _u1("deg2rad"),
    "rad2deg": _u1("rad2deg"),
    "msort": _u1("msort"),
    "argwhere": _u1("argwhere"),
    "nonzero": _nonzero,
    "is_nonzero": _u1("is_nonzero"),
    "glu": lambda a, k, g: (
        f"tensor.glu({_tens(a[0], g)}, {_lit(_attr(k, 'dim', 'arg1', default=-1))})"
    ),
    "softshrink": lambda a, k, g: (
        f"tensor.softshrink({_tens(a[0], g)}, "
        f"{_scalar_or_tensor(_attr(k, 'lambd', 'arg1', default=0.5), g)})"
    ),
    "hardshrink": lambda a, k, g: (
        f"tensor.hardshrink({_tens(a[0], g)}, "
        f"{_scalar_or_tensor(_attr(k, 'lambd', 'arg1', default=0.5), g)})"
    ),
    "logit": lambda a, k, g: (
        f"tensor.logit({_tens(a[0], g)}, "
        f"{_scalar_or_tensor(_attr(k, 'eps', 'arg1', default=1e-6), g)})"
    ),
    "floor_divide": _bin_scalar_rhs("floor_divide"),
    "hypot": _u2("hypot"),
    "logaddexp": _u2("logaddexp"),
    "logaddexp2": _u2("logaddexp2"),
    "nextafter": _u2("nextafter"),
    "copysign": _u2("copysign"),
    "prelu": _u2("prelu"),
    "inner": _u2("inner"),
    "kron": _u2("kron"),
    # --- reductions / linalg extras -----------------------------------------
    "logsumexp": _logsumexp,
    "einsum": _einsum,
    "diag": _diag,
    "diagonal": _diagonal,
    "diag_sum": lambda a, k, g: f"tensor.trace({_tens(a[0], g)})",
    "norm": _norm,
    "renorm": _renorm,
    "normalize": lambda a, k, g: (
        f"tensor.normalize({_tens(a[0], g)}, "
        f"{_scalar_or_tensor(_attr(k, 'p', 'arg1', default=2.0), g)}, "
        f"{_scalar_or_tensor(_attr(k, 'dim', 'arg2', default=1), g)}, "
        f"{_scalar_or_tensor(_attr(k, 'eps', 'arg3', default=1e-12), g)})"
    ),
    "addmm": _addmm("addmm"),
    "addbmm": _addmm("addbmm"),
    "baddbmm": _addmm("baddbmm"),
    "addmv": _addmv_addr("addmv", 2),
    "addr": _addmv_addr("addr", 2),
    "cdist": _cdist,
    "dist": _dist,
    "pdist": _pdist,
    "cross": _cross,
    "corrcoef": _u1("corrcoef"),
    "tensordot": _tensordot,
    "allclose": _allclose,
    # --- shape extras ---------------------------------------------------------
    "movedim": _movedim,
    "rot90": _rot90,
    "pad": _pad,
    # --- pooling ---------------------------------------------------------------
    "avg_pool2d": _avg_pool2d,
    "max_pool2d": _max_pool2d,
    # --- constructors extras ---------------------------------------------------
    "linspace": _linspace_logspace("linspace"),
    "logspace": _linspace_logspace("logspace"),
    "bincount": _bincount,
    "quantile": _quantile,
}


# ---------------------------------------------------------------------------
#  Class-(a) canonicalization — mechanical builders resolved against
#  ``host_surface.json`` (ADR-0178 invariant 4; dissolution plan Step C).
#
#  ``_u1``/``_u2`` mark *positional-passthrough candidates*; whether a
#  candidate becomes an artifact-resolved builder is decided by the
#  canonical surface, not by this file: ``tensor.<name>`` must exist,
#  carry an emittable status, and its ``signature`` must decode to
#  exactly the tagged arity of bare ``Tensor`` slots — the only call
#  shape a name-passthrough builder can honour.  Resolved specs land
#  in :data:`_MECH_SPECS` (``ir_name -> (tensor_op, arity)``) and the
#  ``_OPS`` entry is swapped for the generated builder.
#
#  Everything the artifact cannot express stays hand-written: the
#  attr-carrying builders (IR attr names, ``argN`` fallbacks, operand/
#  attr precedence and defaults are IR-side conventions the surface
#  does not record), renamed targets (``diag_sum`` → ``tensor.trace``,
#  ``embedding`` → ``tensor.index_select``, ``permute`` → ``swapaxes``…),
#  pair-record/``getitem``/``index``-attr semantics, synthesized
#  operands (``linear``'s zero bias), ``-1`` shape resolution, the
#  ``to`` dtype gate — and the ``Value...``-signature candidates
#  (``zeros_like``/``ones_like``/``copysign`` plus every
#  ``_bin_scalar_rhs`` op), whose ``Scalar | Tensor`` rhs slot the
#  artifact's ``Tensor other`` schema cannot distinguish from a
#  tensor-only slot.  Those keep their factory builders; the drift
#  guard in ``tests/test_emit_surface.py`` pins the split.
# ---------------------------------------------------------------------------


def _host_surface() -> dict[str, dict[str, Any]]:
    """``host_surface.json`` ops by name — read without importing the package.

    ``kit_host_tensor/__init__`` pulls numpy + the PyO3 executor;
    emit stays import-light, so the artifact is read straight off the
    installed package directory (``find_spec`` locates it without
    executing the package).  A missing/unreadable artifact yields ``{}``
    — candidates then keep their factory builders and the drift guard
    fails loudly rather than emit silently degrading coverage.
    """
    import importlib.util

    spec = importlib.util.find_spec("kit_host_tensor")
    if spec is None or not spec.submodule_search_locations:
        return {}
    try:
        path = Path(next(iter(spec.submodule_search_locations))) / "host_surface.json"
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        o["op"]: o
        for o in data.get("ops", [])
        if isinstance(o, dict) and isinstance(o.get("op"), str)
    }


def _tensor_slots(signature: str) -> int | None:
    """Positional arity when *signature* is all bare ``Tensor`` slots.

    ``Tensor -> Tensor`` → 1, ``(Tensor * Tensor) -> Tensor`` → 2.
    ``None`` for anything else — ``Value…`` variadics, ``Int``/``List``/
    composite slots, empty signatures — i.e. every shape a
    name-passthrough builder cannot honour without IR-side binding
    knowledge the artifact does not carry.  Only the LHS (argument
    side) is inspected; the result type is irrelevant to emission.
    """
    lhs = signature.split("->", 1)[0].strip()
    if lhs.startswith("(") and lhs.endswith(")"):
        lhs = lhs[1:-1].strip()
    parts = [p.strip() for p in lhs.split("*")] if lhs else []
    if parts and all(p == "Tensor" for p in parts):
        return len(parts)
    return None


def _surface_build(target: str, arity: int):
    """Builder generated from a resolved surface spec.

    Emits ``tensor.<name>(<_tens(arg0)>, …)`` for the spec's arity —
    identical semantics to the ``_u1``/``_u2`` factories it replaces
    (surplus operands ignored, attrs unread, ``IndexError`` on missing
    operands surfacing as the emitter's named decline).
    """

    def build(args: list, kw: Mapping, go) -> str:
        return f"{target}(" + ", ".join(_tens(args[i], go) for i in range(arity)) + ")"

    return build


def _canonicalize_mechanical() -> dict[str, tuple[str, int]]:
    """Resolve ``_u1``/``_u2`` candidates against ``host_surface.json``.

    Returns ``{ir_name: (tensor_op, arity)}`` for every candidate the
    artifact fully determines, swapping the ``_OPS`` entry for the
    generated builder.  Candidates the artifact cannot vouch for keep
    their factory builder (the drift guard catches any real drift).
    """
    surface = _host_surface()
    specs: dict[str, tuple[str, int]] = {}
    for name, arity in _MECH_TAGS.items():
        entry = surface.get(f"tensor.{name}")
        if entry is None or entry.get("status") not in ("Ported", "Handwritten"):
            continue
        if _tensor_slots(str(entry.get("signature", ""))) != arity:
            continue
        target = f"tensor.{name}"
        specs[name] = (target, arity)
        _OPS[name] = _surface_build(target, arity)
    return specs


_MECH_SPECS: dict[str, tuple[str, int]] = _canonicalize_mechanical()


#: Attrs that are inert on the f64 eval surface — dropped before the
#: builder runs (the ``drop`` half of the old ``_OpSpec`` contract).
_DROP_ATTRS = frozenset(
    {
        "cudnn_enabled",
        "memory_format",
        "layout",
        "device",
        "pin_memory",
        "requires_grad",
    }
)

#: ``dtype`` spellings the f64 wire can honour HONESTLY: only the
#: float64 names — the host evaluates every tensor as f64, so a
#: ``float32``/``float16``/``bfloat16`` attr would claim a precision
#: the artifact does not deliver.  Those spellings decline instead of
#: silently upcasting — UNLESS the op is a constant-fill constructor
#: whose fill literal round-trips exactly through the named dtype
#: (see :func:`_dtype_narrowing_honest`): the f64 value then equals
#: the dtype's value bit-for-bit and dropping the attr is honest.
_FLOAT_DTYPES = frozenset({"float64", "f64", "double"})

#: ``tensor.to`` cast spellings the host ``k_to`` kernel honours on
#: the f64-dense wire — **keep in sync** with the match arms +
#: ``int_cast_range`` in host-tensor ``src/ops/aten_kernels.rs``
#: (commit ecc57d9): ``float64``-family identity, ``float32``
#: round-through-``f32``, ``bool`` nonzero→``1.0``, and the
#: truncating integer family (``int64``/``int32``/``int16``/``int8``/
#: ``uint8`` + ``iN``/``long``/``u8`` aliases) with named
#: range/finite declines.  ``torch.*``-prefixed spellings normalise
#: on the last ``.`` segment exactly like ``k_to``.  Each accepted
#: spelling maps to the canonical name the artifact emits;
#: ``float16``/``bfloat16``/``float8_*`` and unknown names are absent
#: — they stay declined (the host still errors on them).
_TO_DTYPES = {
    "float64": "float64",
    "double": "float64",
    "f64": "float64",
    "float32": "float32",
    "f32": "float32",
    "bool": "bool",
    "int64": "int64",
    "i64": "int64",
    "long": "int64",
    "int32": "int32",
    "i32": "int32",
    "int16": "int16",
    "i16": "int16",
    "int8": "int8",
    "i8": "int8",
    "uint8": "uint8",
    "u8": "uint8",
}


def _to_dtype_canonical(v: Any) -> str | None:
    """Canonical ``tensor.to`` dtype spelling for a cast-spec value.

    Looks *v* up in :data:`_TO_DTYPES` after the same normalisation
    ``k_to`` applies (lowercase, ``torch.*`` prefix stripped on the
    last ``.`` segment).  Returns ``None`` for non-string specs and
    for names the host kernel would refuse.
    """
    if isinstance(v, Const):
        v = v.value
    if not isinstance(v, str):
        return None
    return _TO_DTYPES.get(v.lower().rsplit(".", 1)[-1])


def _dtype_fill_operand(op: str, args: list, attrs: Mapping) -> Any:
    """The literal fill a dtype-narrowing ``full``/``full_like`` carries.

    Mirrors the operand reads in :func:`_full` / ``_full_like_fill`` —
    returns ``_MISSING`` for any other op (dtype narrowing is only
    meaningful where the attr scoped a *constant* value).
    """
    if op == "full":
        return args[0] if args else _attr(attrs, "fill_value", "arg0", default=_MISSING)
    if op == "full_like":
        return args[1] if len(args) > 1 else _attr(attrs, "fill_value", "arg1", default=_MISSING)
    return _MISSING


def _dtype_roundtrips(dtype_name: str, value: Any) -> bool:
    """True iff literal ``value`` is exactly representable in ``dtype_name``.

    Stdlib-only: ``struct``'s ``e``/``f`` formats and ``array``'s
    ``f`` typecode give the IEEE half/single round-trip; bfloat16 is
    the f32 image with its low 16 mantissa bits zero.  Non-finite/
    NaN payloads and non-numeric fills decline (``nan != nan`` fails
    the compare — honest).
    """
    if isinstance(value, Const):
        value = value.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    v = float(value)
    try:
        if dtype_name in ("float16", "f16", "half"):
            return struct.unpack("<e", struct.pack("<e", v))[0] == v
        if dtype_name in ("float32", "f32"):
            return array.array("f", [v])[0] == v
        if dtype_name in ("bfloat16", "bf16"):
            f32 = array.array("f", [v])[0]
            if f32 != v:
                return False
            bits = int.from_bytes(array.array("f", [f32]).tobytes(), "little")
            return bits & 0xFFFF == 0
    except (OverflowError, ValueError):
        return False
    return False


def _dtype_narrowing_honest(op: str, args: list, attrs: Mapping, dtype_name: str) -> bool:
    """Whether a non-f64 ``dtype`` attr may be dropped without lying.

    Only constant-fill constructors qualify, and only when the emitted
    fill literal is exactly representable in the named dtype — the
    artifact's f64 value then carries the identical number the dtype
    would have held.  Every other op/dtype/value still declines.
    """
    fill = _dtype_fill_operand(op, args, attrs)
    if fill is _MISSING:
        return False
    return _dtype_roundtrips(dtype_name, fill)


# ---------------------------------------------------------------------------
#  The emitter
# ---------------------------------------------------------------------------


def _normalize(t: Op) -> tuple[list[Any], dict[str, Any]]:
    """Split *t*'s args into operand terms and named scalar attrs.

    Trailing non-operand args (raw scalars/lists a hand-minted term
    spells positionally) fold into ``arg{position}`` attrs — the same
    fallback names the torch bindings read.
    """
    args = list(t.args)
    attrs = dict(t.attrs)
    scalars: list[tuple[int, Any]] = []
    while args and not _is_operand(args[-1]):
        scalars.insert(0, (len(args) - 1, args.pop()))
    for pos, v in scalars:
        attrs.setdefault(f"arg{pos}", v)
    return args, attrs


# ---------------------------------------------------------------------------
#  Hash-consed (shared-subterm) emission
# ---------------------------------------------------------------------------
#
# mlopt terms are hash-consed DAGs (``Op.make`` interns by structure),
# but a plain recursive render re-emits every *occurrence* — a
# sharing-heavy extraction tree-expands exponentially (the
# whisper-encoder artifact measured 258 unique subterms vs ~4M tree
# occurrences; see host's hnl_run_scale_spike.md).  ``.hnl`` ``do {}``
# blocks already evaluate ``name :: expr`` bindings in order, so the
# program/function emitters bind every multiply-referenced ``Op``
# node once as ``__sN :: <expr>`` and render use sites as the name.
#
# Both passes below walk the *DAG* — each unique node once — so the
# whole share analysis is O(unique edges), not O(tree occurrences).

#: Minimum parent references before an ``Op`` subterm earns a
#: ``__sN ::`` binding.  A binding costs its RHS once plus ~12 B of
#: ``  __sN :: `` overhead and ~7 B per name reference; inlining costs
#: the full call spelling per occurrence.  Every ``tensor.*`` call
#: spelling is longer than a bare name, so refs >= 2 is profitable
#: except for sub-~15-char calls shared exactly twice (~10 B loss —
#: accepted: the binding also collapses eval from per-occurrence
#: recomputation to a single dispatch, which dominates at scale).
_SHARED_MIN_REFS = 2


def _op_subterms(v: Any):
    """Yield the ``Op`` nodes directly reachable inside *v*.

    Descends through list/tuple/dict wrappers (shape specs and attr
    payloads can carry terms); an ``Op`` yields *itself* — its own
    children are reached when the walk pops it as a node.  ``Var`` /
    ``Param`` / ``Const`` leaves terminate the walk (they are not
    bindable and have no children).
    """
    if isinstance(v, Op):
        yield v
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _op_subterms(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _op_subterms(x)


def _child_ops(t: Op):
    """Every ``Op`` node *t* references — operand args and attr-held
    terms (``layer_norm`` weight/bias slots, computed shape specs)."""
    for a in t.args:
        yield from _op_subterms(a)
    for v in t.attrs.values():
        yield from _op_subterms(v)


def _count_refs(root: Any) -> dict[Op, int]:
    """Parent-reference count per ``Op`` node, over the DAG.

    Each unique node is visited once; every child slot contributes
    one ref (``add(t, t)`` counts *t* twice — both use sites render
    the name).  Iterative — term depth is unbounded.
    """
    counts: dict[Op, int] = {}
    seen: set[Op] = set()
    stack = [root]
    while stack:
        t = stack.pop()
        if not isinstance(t, Op) or t in seen:
            continue
        seen.add(t)
        for c in _child_ops(t):
            counts[c] = counts.get(c, 0) + 1
            stack.append(c)
    return counts


def _bind_order(root: Any, bind: set[Op]) -> list[Op]:
    """*bind* members in dependency order — children before parents.

    Post-order DFS over the DAG (iterative); non-bound nodes are
    traversed through but not emitted, so a shared node buried under
    unshared ancestors still lands before its first use.
    """
    order: list[Op] = []
    pushed: set[Op] = set()
    stack: list[tuple[Any, bool]] = [(root, False)]
    while stack:
        t, expanded = stack.pop()
        if not isinstance(t, Op):
            continue
        if expanded:
            if t in bind:
                order.append(t)
            continue
        if t in pushed:
            continue
        pushed.add(t)
        stack.append((t, True))
        for c in _child_ops(t):
            stack.append((c, False))
    return order


def _renderer(
    bound: Mapping[str, str] | None,
    shared: dict[Op, str] | None,
) -> tuple[Any, Any, set[str]]:
    """Build ``(go, render, taken)`` for one emit pass.

    ``go`` is the share-aware recursive renderer the ``_OPS`` builders
    call for child terms; ``render`` is the same walk minus the
    shared-name shortcut — the entry point for binding RHSs and the
    body, whose roots must render in full even when they carry a
    shared name.  ``taken`` is the used-name set (leaf bindings plus
    ``tensor``) shared-name generation must avoid.
    """
    names: dict[str, str] = dict(bound or {})
    taken: set[str] = set(names.values()) | {"tensor"}

    def render(t: Any) -> str:
        key = _leaf_key(t)
        if key is not None:
            leaf, kind = key
            if leaf not in names:
                names[leaf] = _sanitize(leaf, taken)
            return names[leaf]
        if isinstance(t, Const):
            return _num(t.value) if isinstance(t.value, (int, float)) else _lit(t.value)
        if isinstance(t, (bool, int, float)) or t is None:
            return _lit(t)
        if isinstance(t, (tuple, list)):
            return _lit(list(t))
        if isinstance(t, str):
            raise UnsupportedOpError(f"raw metavar {t!r} in a term — patterns are not programs")
        if not isinstance(t, Op):
            raise UnsupportedOpError(f"unemittable term: {t!r}")
        if t.op == "pack":
            # pack(t0, …) — the multi-output root — lands as a `.hnl` list.
            return "[" + ", ".join(go(a) for a in t.args) + "]"
        build = _OPS.get(t.op)
        if build is None:
            raise UnsupportedOpError(
                f"op {t.op!r} is not on the tensor.* emit surface "
                f"({len(_OPS)} ops covered — the liftable-now set; "
                "see module docstring for the needs-work/decline split)"
            )
        args, attrs = _normalize(t)
        for name in _DROP_ATTRS:
            attrs.pop(name, None)
        dtype = attrs.pop("dtype", None)
        if dtype is not None:
            dv = dtype.value if isinstance(dtype, Const) else dtype
            dname = str(dv).lower()
            if t.op == "to":
                # ``tensor.to`` is the one op whose dtype attr IS the
                # computation: the host ``k_to`` performs the real
                # narrowing cast, so an accepted spelling is handed to
                # the builder (canonicalised) rather than dropped.
                # Everything else falls through to the compute-
                # precision contract below.
                canon = _to_dtype_canonical(dv)
                if canon is not None:
                    attrs["dtype"] = canon
                elif dname not in _FLOAT_DTYPES:
                    raise UnsupportedOpError(
                        f"{t.op}: dtype={dv!r} — outside the k_to table "
                        "(float64/float32/int64/int32/int16/int8/uint8/"
                        "bool + aliases); the host refuses f16/bf16/"
                        "float8 and unknown names"
                    )
            elif dname not in _FLOAT_DTYPES and not _dtype_narrowing_honest(
                t.op, args, attrs, dname
            ):
                raise UnsupportedOpError(
                    f"{t.op}: dtype={dv!r} — the host evaluates f64 only; "
                    "non-f64 dtype attrs have no honest artifact"
                )
        try:
            return build(args, attrs, go)
        except IndexError:
            raise UnsupportedOpError(
                f"{t.op}: missing operands — the surface signature "
                "needs more args than the term carries"
            ) from None

    def go(t: Any) -> str:
        if shared is not None and isinstance(t, Op) and t in shared:
            return shared[t]
        return render(t)

    return go, render, taken


def emit(term: Any, bound: Mapping[str, str] | None = None) -> str:
    """Render a mlopt term as a ``.hnl`` expression over ``tensor.*``.

    ``Var``/``Param`` leaves emit sanitized binding names (use
    :func:`emit_program`/:func:`emit_function` to bind them — they
    pass their *bound* map in so prelude names and use sites agree);
    ``Const`` and scalar attr values emit literals.  Scalar operands
    on tensor-required slots wrap as rank-0 tensors (see module
    docstring).  Raises :class:`UnsupportedOpError` naming the first
    op or value the surface cannot honour.

    This is the *unshared* spelling — every occurrence of a
    hash-consed subterm re-renders inline.  :func:`emit_program` and
    :func:`emit_function` go through :func:`_emit_shared`, which
    binds multiply-referenced subterms as ``__sN ::`` let-bindings;
    ``emit`` stays the bare-expression door (tests and expression
    probes want exactly one expression, no ``do`` block).

    Carrier ops are re-spelled first (:func:`mlopt.carrier_expand.
    expand_carriers`): the definitional unfolds
    (``om_apply(om_elem(s,v))`` → ``softmax(s) @ v``,
    ``om_decayed(s,d,v)`` → ``(s ⊙ d) @ v``) land in tensor-land;
    every other carrier shape still declines by name below.
    """
    term = expand_carriers(term)
    _go, render, _taken = _renderer(bound, None)
    return render(term)


def _emit_shared(term: Any, bound: Mapping[str, str]) -> tuple[list[str], str]:
    """Hash-consed emit — ``(binding_lines, body_expr)`` for a ``do`` block.

    Every ``Op`` node with ``>= _SHARED_MIN_REFS`` parent references
    binds once as ``  __sN :: <expr>`` in dependency order (children
    before parents, so a binding's RHS only names earlier bindings);
    every use site — including inside other binding RHSs — renders
    the name.  ``bound`` is the leaf-name map from :func:`_prelude`;
    ``__sN`` names are drawn avoiding it (and ``tensor``).

    The root itself is never bound (it has no parents); a ``pack``
    root or any other node reachable only once renders inline as
    before.
    """
    term = expand_carriers(term)
    shared: dict[Op, str] = {}
    _go, render, taken = _renderer(bound, shared)
    counts = _count_refs(term)
    bind = {t for t, c in counts.items() if c >= _SHARED_MIN_REFS}
    lines: list[str] = []
    i = 0
    for t in _bind_order(term, bind):
        while f"__s{i}" in taken:
            i += 1
        name = f"__s{i}"
        i += 1
        shared[t] = name
        taken.add(name)
        lines.append(f"  {name} :: {render(t)}")
    return lines, render(term)


# ---------------------------------------------------------------------------
#  Feed marshaling — leaf values → prelude bindings
# ---------------------------------------------------------------------------


def _feed_expr(name: str, f: Any) -> str:
    """Render one feed entry as a ``.hnl`` tensor expression."""
    if isinstance(f, Mapping) and "hnl" in f:
        return str(f["hnl"])
    if isinstance(f, Mapping):
        if "shape" not in f or "data" not in f:
            raise ValueError(f"feed[{name!r}]: mapping needs 'shape'+'data' or 'hnl'")
        shape = _lit(f["shape"])
        data = _lit(list(f["data"]))
        return f"tensor.reshape(tensor.from_list({data}), {shape})"
    arr = _as_array(f)
    if arr is not None:
        shape = _lit([int(d) for d in arr[0]])
        data = _lit([float(x) for x in arr[1]])
        return f"tensor.reshape(tensor.from_list({data}), {shape})"
    raise ValueError(
        f"feed[{name!r}]: expected {{'shape','data'}}, {{'hnl'}}, "
        f"or an array — got {type(f).__name__}"
    )


def _as_array(f: Any, dtype: str = "F64") -> tuple[Any, Any] | None:
    """Best-effort (shape, flat-data) for torch/numpy/list inputs.

    *dtype* is the safetensors spelling (``"F64"``/``"F32"``) — the
    sidecar path passes its own so f32-native weights stay f32 views
    (zero-copy) instead of being widened to f64 and pinned twice.
    """
    detach = getattr(f, "detach", None)
    if callable(detach):
        try:
            f = detach().cpu().numpy()
        except Exception:  # noqa: BLE001 — fall through to numpy
            return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        np_dt = np.float32 if dtype == "F32" else np.float64
        a = np.asarray(f, dtype=np_dt)
    except Exception:  # noqa: BLE001
        return None
    return (tuple(int(d) for d in a.shape), a.reshape(-1))


# ---------------------------------------------------------------------------
#  Safetensors sidecar — the out-of-band weight channel (``tensor.load``)
# ---------------------------------------------------------------------------
#
# ``tensor.from_list`` literals are the only weight channel *inside* the
# ``.hnl`` text — a whisper-class export is ~80% decimal literal text
# (host:project/reports/tensor_load_spike.md).  host-tensor's
# ``tensor.load(path[, name])`` reads a safetensors sidecar instead:
# 8-byte LE header length + JSON header + raw LE payload.  Emit-side,
# ``emit_program``/``emit_function`` take an opt-in ``sidecar=`` path;
# ``Param`` leaves at/above ``sidecar_min_elems`` bind as
# ``tensor.load("<basename>", "<leaf>")`` and their values land in one
# ``<slug>.safetensors`` written next to the artifact.  The rendered
# path is the file's *basename* — resolution is ``$TENSOR_LOAD_DIR``
# (which :class:`KitRunner` exports) else the process cwd, so the
# ``.hnl`` + ``.safetensors`` pair stays relocatable.  ``Var`` leaves
# are runtime inputs and never sidecar; ``{"hnl"}`` feed entries are
# computed expressions and stay inline.  Sidecar mode is strictly
# opt-in — the hot verify paths never write files.

#: Default element-count gate for sidecar emission: smaller ``Param``
#: leaves stay ``tensor.from_list`` literals (a tiny tensor costs less
#: as text than a file lookup, and small artifacts stay self-contained).
_SIDECAR_MIN_ELEMS = 4096

_SAFETENSORS_DTYPES = ("F64", "F32")


def _sidecar_payload(flat: Any, dtype: str) -> bytes:
    """Little-endian raw payload for one tensor's flat values."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None:
        a = np.ascontiguousarray(
            np.asarray(flat, dtype=np.float64 if dtype == "F64" else np.float32)
        )
        if sys.byteorder == "big":
            a = a.byteswap()
        return a.tobytes()
    # numpy-free fallback — same LE contract via array.array.
    arr = array.array("d" if dtype == "F64" else "f", (float(x) for x in flat))
    if sys.byteorder == "big":
        arr.byteswap()
    return arr.tobytes()


_SAFETENSORS_ITEMSIZE = {"F64": 8, "F32": 4}


def write_safetensors(
    path: str | Path,
    tensors: Mapping[str, tuple[Sequence[int], Any]],
    *,
    dtype: str = "F64",
) -> Path:
    """Write ``{name: (shape, flat values)}`` as a safetensors file.

    The container ``tensor.load`` reads: ``u64le header_len`` + JSON
    header ``{name: {dtype, shape, data_offsets}}`` + raw little-endian
    payload (offsets are payload-relative).  ``dtype`` is ``"F64"``
    (identity — the internal repr is f64, NaN/±inf ride the binary
    channel natively) or ``"F32"`` (rounds on write; ``tensor.load``
    widens back — the deliberate size/precision trade for f32-native
    weights).  Returns the written path.

    Streams: the header's ``data_offsets`` are computable up front
    (``prod(shape) × itemsize``), so payloads write straight to the
    file one tensor at a time — never a second whole-file ``bytearray``
    + ``bytes()`` copy, which pinned ~3× the payload in RSS on
    llama-scale exports.
    """
    if dtype not in _SAFETENSORS_DTYPES:
        raise ValueError(f"safetensors dtype must be one of {_SAFETENSORS_DTYPES}, got {dtype!r}")
    itemsize = _SAFETENSORS_ITEMSIZE[dtype]
    header: dict[str, Any] = {}
    offset = 0
    for name, (shape, _flat) in tensors.items():
        dims = [int(d) for d in shape]
        nbytes = itemsize
        for d in dims:
            nbytes *= d
        header[name] = {
            "dtype": dtype,
            "shape": dims,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    hbytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(len(hbytes).to_bytes(8, "little") + hbytes)
        for name, (shape, flat) in tensors.items():
            blob = _sidecar_payload(flat, dtype)
            want = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
            if len(blob) != want:
                raise ValueError(
                    f"safetensors tensor {name!r}: flat data is "
                    f"{len(blob)} bytes, shape {shape} wants {want} "
                    f"— refusing to write a corrupt sidecar"
                )
            fh.write(blob)
    return path


def _leaf_array(f: Any, dtype: str = "F64") -> tuple[tuple[int, ...], Any] | None:
    """``(shape, flat-data)`` for a feed value, else ``None``.

    ``{"shape","data"}`` mappings read directly; array-likes go through
    :func:`_as_array` (torch/numpy/list) at the sidecar's *dtype* so an
    F32 sidecar keeps f32-native values as zero-copy views.  ``{"hnl"}``
    computed expressions and unconvertible values return ``None`` —
    they stay inline literals.
    """
    if isinstance(f, Mapping):
        if "hnl" in f or "shape" not in f or "data" not in f:
            return None
        try:
            shape = tuple(int(d) for d in f["shape"])
        except (TypeError, ValueError):
            return None
        return shape, list(f["data"])
    return _as_array(f, dtype=dtype)


class _Sidecar:
    """Sidecar emit state: ``tensor.load`` bindings + safetensors entries.

    One instance per emit call.  :meth:`expr` decides per ``Param``
    leaf whether it sidecars (data-bearing, at/above ``min_elems``) or
    stays an inline ``from_list`` literal; eligible leaves record their
    ``(shape, flat)`` under the leaf name — the safetensors key
    ``tensor.load``'s second arg names.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dtype: str = "F64",
        min_elems: int = _SIDECAR_MIN_ELEMS,
    ) -> None:
        self.path = Path(path)
        self.basename = self.path.name
        self.dtype = dtype
        self.min_elems = min_elems
        self.entries: dict[str, tuple[tuple[int, ...], Any]] = {}

    def expr(self, leaf: str, feed: Any) -> str | None:
        """``tensor.load("<basename>", "<leaf>")`` or ``None`` (stay inline)."""
        shape_flat = _leaf_array(feed, dtype=self.dtype)
        if shape_flat is None:
            return None
        shape, flat = shape_flat
        numel = 1
        for d in shape:
            numel *= d
        if numel < self.min_elems:
            return None
        self.entries[leaf] = (shape, flat)
        return f"tensor.load({json.dumps(self.basename)}, {json.dumps(leaf)})"

    def write(self) -> Path | None:
        """Write the collected entries; ``None`` when nothing sidecarred."""
        if not self.entries:
            return None
        return write_safetensors(self.path, self.entries, dtype=self.dtype)


def _prelude(
    term: Any,
    feed: Mapping[str, Any] | None,
    *,
    require_kinds: set[str] | None = None,
    kinds: list[tuple[str, str]] | None = None,
    sidecar: _Sidecar | None = None,
) -> tuple[list[str], dict[str, str], set[str]]:
    """Shared ``emit_program``/``emit_function`` prelude builder.

    Returns ``(binding_lines, bound_map, var_names)`` — every feed
    entry binds in dict order (staged ``{"hnl"}`` expressions may
    reference earlier bindings), leaf names sanitize into *bound_map*.
    *kinds* lets a caller that already ran :func:`leaf_kinds` reuse the
    result instead of walking the DAG again.  *sidecar*, when given,
    diverts eligible ``Param`` leaves to ``tensor.load`` bindings and
    records their values for the ``.safetensors`` write.
    """
    feed = feed or {}
    if kinds is None:
        kinds = leaf_kinds(term)
    names = [n for n, _ in kinds]
    kind_of = dict(kinds)
    missing = [
        n for n in names if n not in feed and (require_kinds is None or kind_of[n] in require_kinds)
    ]
    if missing:
        raise ValueError(f"feed is missing leaves {missing} — every bound Var/Param must bind")
    taken: set[str] = {"tensor"}
    bound: dict[str, str] = {}
    for name, _f in feed.items():
        bound[name] = _sanitize(name, taken)
    for n in names:
        if n not in bound:
            bound[n] = _sanitize(n, taken)
    lines = []
    for name, f in feed.items():
        expr = None
        if sidecar is not None and kind_of.get(name) == "param":
            expr = sidecar.expr(name, f)
        lines.append(f"  {bound[name]} :: {expr if expr is not None else _feed_expr(name, f)}")
    return lines, bound, {n for n, k in kinds if k == "var"}


def emit_program(
    term: Any,
    feed: Mapping[str, Any] | None = None,
    *,
    sidecar: str | Path | None = None,
    sidecar_dtype: str = "F64",
    sidecar_min_elems: int = _SIDECAR_MIN_ELEMS,
) -> str:
    """Render *term* as a runnable ``.hnl`` program — a ``do`` block.

    Every ``Var``/``Param`` leaf binds in the prelude from *feed*:

    * ``{"shape": [d…], "data": [f…]}`` →
      ``tensor.reshape(tensor.from_list([f…]), [d…])``;
    * ``{"hnl": "<expr>"}`` → the expression verbatim — the marshaling
      channel for values that are *computed* inputs;
    * a nested list / array-like → ``tensor.from_list``-compatible
      ``{"shape","data"}`` extraction (torch/numpy tensors accepted).

    *feed* binds **in dict order**, and may carry entries beyond the
    term's leaves — a ``{"hnl"}`` expression references any *earlier*
    binding by its sanitized name, so a prelude can compute in stages.

    *sidecar* (opt-in, default ``None``) is a ``.safetensors`` output
    path: ``Param`` leaves with at least *sidecar_min_elems* elements
    bind as ``tensor.load("<basename>", "<leaf>")`` instead of
    ``from_list`` literals, and their values are written to the file
    in *sidecar_dtype* (``"F64"``/``"F32"`` — see
    :func:`write_safetensors`).  The file is written only when at
    least one leaf sidecars; nothing is written on the default path,
    so hot verify callers never touch the filesystem.  ``Var`` leaves
    and ``{"hnl"}`` entries always stay inline.
    """
    sc = (
        _Sidecar(sidecar, dtype=sidecar_dtype, min_elems=sidecar_min_elems)
        if sidecar is not None
        else None
    )
    lines, bound, _ = _prelude(term, feed, sidecar=sc)
    shared_lines, body = _emit_shared(term, bound)
    out = [
        # Self-contained artifact header — resolves `tensor` through
        # the project's host-tensor dependency (mlopt-kit's
        # .handl.toml dev-links the sibling checkout).  Bare
        # `tensor.*` would only work under ambient dispatch.
        'from "host-tensor" import tensor',
        "do {",
    ]
    out.extend(lines)
    out.extend(shared_lines)
    out.append(f"  {body}")
    out.append("}")
    src = "\n".join(out)
    if sc is not None:
        sc.write()
    return src


def emit_function(
    term: Any,
    param_feed: Mapping[str, Any] | None = None,
    *,
    name: str = "optimized",
    sidecar: str | Path | None = None,
    sidecar_dtype: str = "F64",
    sidecar_min_elems: int = _SIDECAR_MIN_ELEMS,
) -> str:
    """Render *term* as an exported ``.hnl`` function — the export artifact.

    ``Param`` leaves (and any extra staged ``{"hnl"}`` feed entries)
    bind to data in the ``do`` prelude; ``Var`` leaves become the
    function's parameters — the honest shape for a program whose
    inputs are not known at export time::

        export optimized := (x, pos) ~> do {
          w :: tensor.reshape(tensor.from_list([…]), […])
          tensor.matmul(x, w)
        }

    *param_feed* must cover every ``Param`` leaf (missing names raise
    ``ValueError``); ``Var`` leaves must NOT appear in it — they are
    the parameters.  Feed order is binding order, so staged ``{"hnl"}``
    expressions work exactly as in :func:`emit_program`.  Shared
    ``Op`` subterms (``>= _SHARED_MIN_REFS`` parents) bind as
    ``__sN ::`` lines between the leaf prelude and the body, same as
    :func:`emit_program`.

    *sidecar*/*sidecar_dtype*/*sidecar_min_elems* behave exactly as in
    :func:`emit_program` — the ``mlopt export --sidecar`` path: large
    ``Param`` leaves emit ``tensor.load`` and their values land in the
    ``.safetensors`` file (written only when at least one leaf
    sidecars).
    """
    feed = dict(param_feed or {})
    kinds = leaf_kinds(term)
    kind_map = dict(kinds)
    vars_in_feed = [n for n, k in kind_map.items() if k == "var" and n in feed]
    if vars_in_feed:
        raise ValueError(
            f"Var leaves {vars_in_feed} are function parameters — "
            "they cannot also bind in param_feed"
        )
    sc = (
        _Sidecar(sidecar, dtype=sidecar_dtype, min_elems=sidecar_min_elems)
        if sidecar is not None
        else None
    )
    # Vars are NOT fed: prelude requires only Param leaves.
    lines, bound, _var_set = _prelude(term, feed, require_kinds={"param"}, kinds=kinds, sidecar=sc)
    var_names = [n for n, k in kinds if k == "var"]
    params = ", ".join(bound[n] for n in var_names)
    out = [
        f"--- mlopt export — {name}: optimized term over tensor.*",
        'from "host-tensor" import tensor',
        f"export {name} := ({params}) ~> do {{",
    ]
    out.extend(lines)
    shared_lines, body = _emit_shared(term, bound)
    out.extend(shared_lines)
    out.append(f"  {body}")
    out.append("}")
    src = "\n".join(out)
    if sc is not None:
        sc.write()
    return src


# ---------------------------------------------------------------------------
#  The runner — the hnl-run subprocess door (lazy: no binary needed to emit)
# ---------------------------------------------------------------------------

#: The repo root that contains ``packages/mlopt`` (this checkout — the
#: merged repo; its on-disk dir name is not load-bearing).
_REPO = Path(__file__).resolve().parents[4]


def _hnl_run_candidates() -> list[Path]:
    """Candidate ``hnl-run`` locations, in preference order."""
    cands: list[Path] = []
    env = os.environ.get("MLOPT_HNL_RUN")
    if env:
        cands.append(Path(env))
    sibling = _REPO.parent
    for base in (
        _REPO / "target",
        sibling / ".ci-target" / _REPO.name,
        sibling / ".ci-target" / "graphopt",
    ):
        cands += [base / "debug" / "hnl-run", base / "release" / "hnl-run"]
    which = shutil.which("hnl-run")
    if which:
        cands.append(Path(which))
    return cands


def find_hnl_run() -> Path | None:
    """Locate a built ``hnl-run`` binary, or ``None`` (honest skip)."""
    for c in _hnl_run_candidates():
        if c.exists():
            return c
    return None


def _ambient_src(src: str) -> str:
    """Drop ``from "host-tensor" import tensor`` lines for stdin eval.

    ``hnl-run -`` has no ambient package scope: WITH ``--project`` the
    import resolves but the snippet's value comes back ``null``
    (measured — the project path is for module files, not stdin
    snippets); WITHOUT it bare ``tensor.*`` dispatches ambiently.
    Emitted sources keep the header for file artifacts; the runner
    strips it so eval exercises the same op surface.
    """
    return "\n".join(
        ln for ln in src.splitlines() if ln.strip() != 'from "host-tensor" import tensor'
    )


_HNL_RUN_TIMEOUT_ENV = "MLOPT_HNL_RUN_TIMEOUT"
_HNL_RUN_TIMEOUT_DEFAULT = 600.0


def _hnl_run_timeout() -> float:
    """Per-invocation ``hnl-run`` timeout in seconds.

    ``$MLOPT_HNL_RUN_TIMEOUT`` (float seconds) overrides the 600 s
    default — whole-graph verify programs on large models can
    legitimately run longer.  Read at call time so the env var is the
    single knob; an unparsable or non-positive value is a named
    ``ValueError``, never a silent default.
    """
    raw = os.environ.get(_HNL_RUN_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return _HNL_RUN_TIMEOUT_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{_HNL_RUN_TIMEOUT_ENV} must be a float (seconds), got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{_HNL_RUN_TIMEOUT_ENV} must be positive, got {raw!r}")
    return value


class KitRunner:
    """Run ``.hnl`` programs through a compiled ``hnl-run`` binary.

    Resolution: ``$MLOPT_HNL_RUN``, then sibling-repo cargo target
    dirs (``graphopt/target``, ``.ci-target/graphopt``), then ``PATH``.
    ``--project`` is opt-in (``project=``): stdin snippets evaluate
    through ambient ``tensor.*`` dispatch — the emitted
    ``from "host-tensor" import tensor`` header is stripped for eval
    (see :func:`_ambient_src` for why project mode returns ``null``
    on snippets).
    Each call is one subprocess — the same evaluation surface the
    hand-written ``hnl/`` modules go through.  Per-call timeout is
    ``$MLOPT_HNL_RUN_TIMEOUT`` seconds (default 600).
    """

    def __init__(
        self,
        binary: str | Path | None = None,
        project: str | Path | None = None,
        tensor_load_dir: str | Path | None = None,
    ) -> None:
        """Resolve the binary; raise if it is not built."""
        if binary is not None:
            self.binary = Path(binary)
            if not self.binary.exists():
                raise FileNotFoundError(f"hnl-run not found: {self.binary}")
        else:
            found = find_hnl_run()
            if found is None:
                raise FileNotFoundError(
                    "hnl-run not found — set $MLOPT_HNL_RUN or build "
                    "mlopt-kit's hnl-run (graphopt: cargo build -p "
                    "mlopt-kit --bin hnl-run)"
                )
            self.binary = found
        self.project = Path(project) if project is not None else None
        # ``tensor.load`` resolves relative sidecar paths against
        # $TENSOR_LOAD_DIR (host-tensor eec98bb) — stdin eval has no
        # artifact directory, so callers running sidecar-emitted
        # programs pass the ``.safetensors`` directory here.
        self.tensor_load_dir = Path(tensor_load_dir) if tensor_load_dir is not None else None

    def run(self, src: str, repeat: int = 1) -> Any:
        """Evaluate *src*; return the decoded JSON boundary value."""
        src = _ambient_src(src)
        cmd = [str(self.binary)]
        if self.project is not None:
            cmd += ["--project", str(self.project)]
        if repeat > 1:
            cmd += ["--repeat", str(repeat)]
        cmd.append("-")
        env = None
        if self.tensor_load_dir is not None:
            env = dict(os.environ)
            env["TENSOR_LOAD_DIR"] = str(self.tensor_load_dir)
        timeout = _hnl_run_timeout()
        try:
            out = subprocess.run(
                cmd,
                input=src,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            # Name the knob: the caller (e.g. `export --verify-run`)
            # prints this verbatim, and the fix is the env var.
            raise KitEvalError(
                f"hnl-run exceeded {timeout:g}s — raise {_HNL_RUN_TIMEOUT_ENV} "
                "(seconds) or shrink the program"
            ) from e
        if out.returncode != 0:
            raise KitEvalError(f"hnl-run exited {out.returncode}: {out.stderr.strip()}")
        return json.loads(out.stdout)

    def eval(
        self,
        term: Any,
        feed: Mapping[str, Any],
        repeat: int = 1,
    ) -> Any:
        """Emit *term* over *feed* and evaluate it; return the JSON."""
        return self.run(emit_program(term, feed), repeat=repeat)

    def tensor(
        self,
        term: Any,
        feed: Mapping[str, Any],
        repeat: int = 1,
    ) -> dict:
        """Like :meth:`eval`, decoding the ``__tensor__`` wire form."""
        out = self.eval(term, feed, repeat=repeat)
        t = out.get("__tensor__") if isinstance(out, dict) else None
        if t is None:
            raise KitEvalError(f"expected a tensor result, got {out!r}")
        return t
