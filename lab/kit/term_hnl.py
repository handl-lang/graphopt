"""catopt terms → `.hnl` programs → CatoptKit — the emit/eval bridge.

Why this exists: the e-graph's terms and `packages/catopt-kit`'s runtime
are the same program space.  A catopt term is an op tree over
:class:`~catopt_core.ir.Op`/:class:`Var`/:class:`Param`/:class:`Const`;
a `.hnl` program over `host-tensor`'s 18-op surface is the same tree
with its leaves bound.  This module is the honest bridge between them:

* :func:`emit` renders any term as a `.hnl` expression over
  ``tensor.*`` — a per-op **data table** (``_SPECS``), not per-term
  code, so any term inside the surface emits.
* :func:`emit_program` binds the term's leaves (``Var``/``Param``) in a
  ``do { name :: … }`` prelude — tensor data as
  ``tensor.reshape(tensor.from_list([flat]), [shape])``, or a caller
  ``.hnl`` expression (the marshaling channel a weight-space fold
  needs: the deployed term keeps ``W2``/``b2`` as leaves and the
  prelude computes them).
* :class:`KitRunner` shells the compiled ``hnl-run`` binary — the same
  door the hand-written ``hnl/`` modules go through — and decodes the
  ``{"__tensor__": …}`` wire form.

The op surface is the bound, stated honestly: host-tensor carries the
elementwise four + ``matmul``/``conv2d``/``batch_norm``/``max_pool`` +
the shape-manipulation three + constructors/projections.  A term
naming anything else — ``rsqrt``/``sqrt``/``exp``/``sigmoid``/``neg``,
``sum``/``mean``, ``softmax`` — raises :class:`UnsupportedOpError`
naming the op: that is precisely the missing surface for the conv+BN
weight-space fold's *spelled* RHS (``s = gamma*rsqrt(var+eps)`` — see
``tests/test_kit_bridge.py``).  Signature mismatches the surface can
honour map through ``accepts``/``drop``/``arity`` (``conv2d``'s
dilation has no slot and must be the default; ``batch_norm``'s
``momentum``/``cudnn_enabled`` are inert in eval mode; its export
arities synthesize the identity affine the op requires).

One wart worth naming in data, not prose: the handl lexer does not
read scientific notation — ``1e-05`` lexes as the tokens
``1`` ``e`` ``-05`` (measured: it parses *silently* as ``1`` in
statement position, and fails inside a list).  ``_num`` therefore
expands every float through :class:`~decimal.Decimal` — ``1e-05`` →
``0.00001`` — so a catopt ``eps`` can never silently change value
across the boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from catopt_core.ir import Const, Op, Param, Var

__all__ = [
    "KitEvalError",
    "KitRunner",
    "UnsupportedOpError",
    "emit",
    "emit_program",
    "leaf_names",
]

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_BIN = _REPO / "target" / "debug" / "hnl-run"


class UnsupportedOpError(ValueError):
    """The term reaches outside the `tensor.*` op surface.

    Carries the op name — the bridge's honest bound.  A spelled conv+BN
    fold dies on ``rsqrt``; a training-mode ``batch_norm`` dies on
    ``training`` (the surface is eval-mode only).
    """


class KitEvalError(RuntimeError):
    """`hnl-run` exited non-zero — the kit refused the program."""


# ---------------------------------------------------------------------------
#  The op table — catopt spelling → tensor.* call, as data
# ---------------------------------------------------------------------------

#: Sentinel for "term arg at index i if present, else ``null``".
_OPT = "opt"
#: Sentinel for "every positional arg, emitted as one `[a0, …]` list".
_ARGS = "args"


@dataclass(frozen=True)
class _OpSpec:
    """How one catopt op lands on the ``tensor.*`` surface.

    ``slots`` lists the emitted call arguments in order; each entry is

    * ``int i`` — term arg ``i`` (required);
    * ``("opt", i)`` — term arg ``i`` if present, else ``null``;
    * ``str name`` — the attr's value as a literal (``null`` if absent);
    * ``("zeros", i)`` / ``("ones", i)`` — a synthesized
      ``tensor.zeros(tensor.shape(<arg i>))`` operand;
    * ``("lit", v)`` — the literal ``v``;
    * ``"args"`` — every positional arg, as one ``[a0, …]`` list
      (``concat``/``pack``).

    Trailing ``null`` slots are trimmed — the ops' optional tails are
    positional, so a trailing default need not be spelled.

    ``positional`` maps args that are not tensor operands — the aten
    scalar tail (``conv2d``'s arg3 is ``stride``) — onto attr names
    before the slots run.  ``arity`` picks the slot tuple by
    ``len(args)`` where operand count is semantic (``batch_norm``'s
    export arities).  ``accepts`` pins attrs to the values the surface
    can honour; ``drop`` ignores the semantically inert ones.
    """

    target: str
    slots: tuple[Any, ...] = ()
    positional: tuple[str, ...] = ()
    arity: Mapping[int, tuple[Any, ...]] = field(default_factory=dict)
    accepts: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    drop: frozenset[str] = frozenset()


#: catopt op → surface spec.  Coverage is host-tensor's 18 ops minus the
#: marshaling-only four (``zeros``/``ones``/``from_list``/``to_list``,
#: which the *emitter* uses, and ``shape``/``derive``/
#: ``batch_norm_backward``, which have no catopt counterpart) — every
#: op absent here fails loudly by name.
_SPECS: dict[str, _OpSpec] = {
    "add": _OpSpec("add", (0, 1)),
    "mul": _OpSpec("mul", (0, 1)),
    "sub": _OpSpec("sub", (0, 1)),
    "div": _OpSpec("div", (0, 1)),
    "matmul": _OpSpec("matmul", (0, 1)),
    # conv2d(x, w, b?, stride?, padding?, dilation?, groups?) — the
    # surface drops dilation: `(x, w, b|null, s|null, p|null, g|null)`.
    "conv2d": _OpSpec(
        "conv2d",
        (0, 1, (_OPT, 2), "stride", "padding", "groups"),
        positional=("stride", "padding", "dilation", "groups"),
        accepts={"dilation": (1, (1, 1), [1, 1])},
    ),
    # batch_norm(x, w, b, rm, rv, *, training, momentum, eps,
    # cudnn_enabled) — eval-mode only, so `training` must be False and
    # momentum/cudnn are inert.  The op requires the full rank-1 affine,
    # so the nobias/plain export arities synthesize beta=0 / gamma=1,beta=0 from
    # the mean's shape — semantically exact marshaling.
    "batch_norm": _OpSpec(
        "batch_norm",
        arity={
            5: (0, 1, 2, 3, 4, "eps"),
            4: (0, 1, ("zeros", 2), 2, 3, "eps"),
            3: (0, ("ones", 1), ("zeros", 1), 1, 2, "eps"),
        },
        accepts={"training": (False,)},
        drop=frozenset({"momentum", "cudnn_enabled"}),
    ),
    "max_pool2d": _OpSpec(
        "max_pool",
        (0, "kernel_size", "stride"),
        positional=(
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "ceil_mode",
        ),
        accepts={
            "padding": (0, (0, 0), [0, 0]),
            "dilation": (1, (1, 1), [1, 1]),
            "ceil_mode": (False,),
        },
    ),
    "reshape": _OpSpec("reshape", (0, "shape"), positional=("shape",)),
    "transpose": _OpSpec(
        "transpose", (0, "dim0", "dim1"), positional=("dim0", "dim1")
    ),
    "concat": _OpSpec("concat", (_ARGS, "dim"), positional=("dim",)),
    # pack(t0, …) — the multi-output root — lands as a `.hnl` list.
    "pack": _OpSpec("", (_ARGS,)),
}


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
            f"non-finite literal {v!r} — the tensor wire carries "
            "f64 data only"
        )
    return format(Decimal(repr(f)), "f")


def _lit(v: Any) -> str:
    """Render an attr value / scalar arg as a `.hnl` literal."""
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return _num(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    raise UnsupportedOpError(f"unemittable attr value: {v!r}")


def _leaf_key(t: Any) -> str | None:
    """Return the feed dict key for a leaf, or ``None``."""
    if isinstance(t, (Var, Param)):
        return t.name
    return None


def leaf_names(term: Any) -> list[str]:
    """Every ``Var``/``Param`` name in *term*, first-use order."""
    seen: set[str] = set()
    order: list[str] = []

    def go(t: Any) -> None:
        k = _leaf_key(t)
        if k is not None:
            if k not in seen:
                seen.add(k)
                order.append(k)
            return
        if isinstance(t, Op):
            for a in t.args:
                go(a)

    go(term)
    return order


def _sanitize(name: str, taken: set[str]) -> str:
    """Return a safe `.hnl` binding name for a catopt leaf.

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
#  The emitter
# ---------------------------------------------------------------------------


def _is_operand(t: Any) -> bool:
    """Whether a positional arg is a tensor operand (vs a scalar tail)."""
    return isinstance(t, (Op, Var, Param))


def _normalize(
    t: Op, spec: _OpSpec
) -> tuple[list[Any], dict[str, Any]]:
    """Split *t*'s args into operands and the named scalar tail.

    A positional arg that is not an operand-typed term folds into the
    spec's ``positional`` names — ``conv2d(x, w, b, [1,1], [1,1])``
    spells ``stride``/``padding`` positionally.
    """
    args = list(t.args)
    attrs = dict(t.attrs)
    scalars: list[Any] = []
    while args and not _is_operand(args[-1]):
        scalars.insert(0, args.pop())
    for i, v in enumerate(scalars):
        if i >= len(spec.positional):
            raise UnsupportedOpError(
                f"{t.op}: positional scalar arg {i + len(args)} has no "
                f"name — the surface signature is unknown"
            )
        name = spec.positional[i]
        if name in attrs and attrs[name] != v:
            raise UnsupportedOpError(
                f"{t.op}: {name} spelled both positionally ({v!r}) and "
                f"as an attr ({attrs[name]!r})"
            )
        attrs.setdefault(name, v)
    return args, attrs


def _slot(
    slot: Any, args: list[Any], attrs: dict[str, Any], emit
) -> str:
    """Emit one call-argument slot."""
    if isinstance(slot, int):
        if slot >= len(args):
            raise UnsupportedOpError(
                f"missing operand at arg{slot} — "
                f"{len(args)} args emitted"
            )
        return emit(args[slot])
    if isinstance(slot, str):
        return _lit(attrs.get(slot))
    kind, v = slot
    if kind == _OPT:
        return emit(args[v]) if v < len(args) else "null"
    if kind == "lit":
        return _lit(v)
    if kind in ("zeros", "ones"):
        inner = emit(args[v]) if v < len(args) else "null"
        return f"tensor.{kind}(tensor.shape({inner}))"
    raise UnsupportedOpError(f"bad emit slot {slot!r}")


def emit(term: Any, bound: Mapping[str, str] | None = None) -> str:
    """Render a catopt term as a `.hnl` expression over ``tensor.*``.

    ``Var``/``Param`` leaves emit sanitized binding names (use
    :func:`emit_program` to bind them — it passes its *bound* map in so
    the prelude names and the use sites agree); ``Const`` and scalar
    attr values emit literals.  Raises :class:`UnsupportedOpError`
    naming the first op or value the surface cannot honour.
    """
    names: dict[str, str] = dict(bound or {})
    taken: set[str] = set(names.values()) | {"tensor"}

    def go(t: Any) -> str:
        key = _leaf_key(t)
        if key is not None:
            if key not in names:
                names[key] = _sanitize(key, taken)
            return names[key]
        if isinstance(t, Const):
            return _num(t.value)
        if isinstance(t, (bool, int, float)) or t is None:
            return _lit(t)
        if isinstance(t, (tuple, list)):
            return _lit(list(t))
        if isinstance(t, str):
            raise UnsupportedOpError(
                f"raw metavar {t!r} in a term — patterns are not "
                "programs"
            )
        if not isinstance(t, Op):
            raise UnsupportedOpError(f"unemittable term: {t!r}")
        spec = _SPECS.get(t.op)
        if spec is None:
            raise UnsupportedOpError(
                f"op {t.op!r} is not on the tensor.* surface "
                f"(18 ops; the elementwise unaries rsqrt/sqrt/neg/exp "
                "and reductions sum/mean are the named gap)"
            )
        args, attrs = _normalize(t, spec)
        for name in spec.drop:
            attrs.pop(name, None)
        for name, ok in spec.accepts.items():
            if name in attrs and attrs[name] not in ok:
                raise UnsupportedOpError(
                    f"{t.op}: attr {name}={attrs[name]!r} has no slot "
                    f"on the tensor surface (accepts {ok})"
                )
            attrs.pop(name, None)
        if spec.arity:
            slots = spec.arity.get(len(args))
            if slots is None:
                raise UnsupportedOpError(
                    f"{t.op}: arity {len(args)} has no surface "
                    "signature"
                )
        else:
            slots = spec.slots
        if _ARGS in slots:
            inner = ", ".join(go(a) for a in args)
            parts = [f"[{inner}]"]
            parts += [
                _slot(s, args, attrs, go) for s in slots if s != _ARGS
            ]
        else:
            parts = [_slot(s, args, attrs, go) for s in slots]
        while parts and parts[-1] == "null":
            parts.pop()
        leftover = {
            k
            for k in attrs
            if k not in spec.positional
            and k not in spec.drop
            and k not in spec.accepts
            and k not in {s for s in slots if isinstance(s, str)}
        }
        if leftover:
            raise UnsupportedOpError(
                f"{t.op}: attrs {sorted(leftover)} have no slot on "
                "the tensor surface"
            )
        if not spec.target:
            return parts[0]
        return f"tensor.{spec.target}({', '.join(parts)})"

    return go(term)


def emit_program(
    term: Any,
    feed: Mapping[str, Any] | None = None,
) -> str:
    """Render *term* as a runnable `.hnl` program — a `do` block.

    Every ``Var``/``Param`` leaf binds in the prelude from *feed*:

    * ``{"shape": [d…], "data": [f…]}`` →
      ``tensor.reshape(tensor.from_list([f…]), [d…])``;
    * ``{"hnl": "<expr>"}`` → the expression verbatim — the marshaling
      channel for values that are *computed* inputs (a weight-space
      fold's ``s`` escape, computed in `.hnl` because the op surface
      has no elementwise ``rsqrt``);
    * a nested list → ``tensor.from_list`` (shape inferred).

    *feed* binds **in dict order**, and may carry entries beyond the
    term's leaves — a ``{"hnl"}`` expression references any *earlier*
    binding by its sanitized name, so a prelude can compute in stages
    (bind the stats, then ``s``, then the folded weight term reads it
    through the emitted expression).
    """
    feed = feed or {}
    names = leaf_names(term)
    missing = [n for n in names if n not in feed]
    if missing:
        raise ValueError(
            f"feed is missing leaves {missing} — every Var/Param "
            "must bind"
        )
    taken: set[str] = {"tensor"}
    bound = {n: _sanitize(n, taken) for n in feed}
    lines = ["do {"]
    for name, f in feed.items():
        if isinstance(f, Mapping) and "hnl" in f:
            expr = f["hnl"]
        elif isinstance(f, Mapping):
            shape = _lit(f["shape"])
            data = _lit(list(f["data"]))
            expr = f"tensor.reshape(tensor.from_list({data}), {shape})"
        else:
            expr = f"tensor.from_list({_lit(f)})"
        lines.append(f"  {bound[name]} :: {expr}")
    lines.append(f"  {emit(term, bound)}")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  The runner — the hnl-run subprocess door
# ---------------------------------------------------------------------------


class KitRunner:
    """Run `.hnl` programs through the compiled ``hnl-run`` binary.

    The binary is ``target/debug/hnl-run`` (built by
    ``cargo build -p catopt-kit --bin hnl-run``; override with the
    *binary* arg or ``$CATOPT_HNL_RUN``).  Each call is one subprocess —
    the same evaluation surface ``cargo test -p catopt-kit`` exercises.
    """

    def __init__(
        self,
        binary: str | Path | None = None,
        project: str | Path | None = None,
    ) -> None:
        """Resolve the binary; raise if it is not built."""
        self.binary = Path(
            binary or os.environ.get("CATOPT_HNL_RUN", _DEFAULT_BIN)
        )
        self.project = Path(project) if project is not None else None
        if not self.binary.exists():
            raise FileNotFoundError(
                f"hnl-run not built: {self.binary} — "
                "cargo build -p catopt-kit --bin hnl-run "
                "(source ~/.cache/devin/handl_deps.env first)"
            )

    def run(self, src: str, repeat: int = 1) -> Any:
        """Evaluate *src*; return the decoded JSON boundary value."""
        cmd = [str(self.binary)]
        if self.project is not None:
            cmd += ["--project", str(self.project)]
        if repeat > 1:
            cmd += ["--repeat", str(repeat)]
        cmd.append("-")
        out = subprocess.run(
            cmd,
            input=src,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if out.returncode != 0:
            raise KitEvalError(
                f"hnl-run exited {out.returncode}: {out.stderr.strip()}"
            )
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
