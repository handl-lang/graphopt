"""``.hnl`` → mlopt term — the reverse of :mod:`.hnl_surface`.

Where :func:`mlopt_discovery.hnl_surface.emit` renders a term as a
``.hnl`` expression over ``tensor.*``, :func:`hnl_to_term` parses
``.hnl`` *source* back into an :class:`mlopt_core.ir` term — the
bridge that lets handl-native models enter mlopt search.  This is a
**scoped** importer, stated honestly:

* it parses the *expression grammar* the emitters produce plus the
  ``do { name :: expr … }`` program form — calls, literals, lists,
  dotted ``tensor.*`` callees.  Full module files (``import`` lines,
  ``cap``/``op`` declarations) are stripped by :func:`_strip_directives`
  before parsing; anything else is a syntax error, not a guess;
* every ``tensor.<op>`` call must name an op in :data:`_HANDLERS` —
  the covered set is exactly the mlopt IR ops whose ``tensor.*``
  spelling is pinned by ``mlopt_host_tensor.bindings.HOST_BINDINGS``
  (plus ``batch_norm``/``max_pool2d``/``pack``, which the host surface
  carries but the executor table does not).  Anything else —
  ``tensor.from_list`` in term position, ``tensor.chunk``'s Array
  result, an op the IR does not know — raises
  :class:`UnsupportedOpError` naming it; ``tensor.t`` is the one
  derived spelling accepted (``transpose(-2,-1)``, exact for the
  ≤2-D case torch defines it on);
* free identifiers become :class:`Var` leaves with a wildcard
  ``TensorType(None)`` shape (override per name through *leaves*);
  ``do``-block bindings inline: ``name :: <expr>`` substitutes the
  imported subterm, while the marshaling spellings
  ``tensor.from_list([…])`` /
  ``tensor.reshape(tensor.from_list([…]), […])`` /
  ``tensor.zeros([…])`` / ``tensor.ones([…])`` mint a
  :class:`Param` leaf carrying the shape — the term-level reading of
  a constant tensor binding.

Round-trip posture: ``emit → hnl_to_term → emit`` is the identity on
the *surface projection* — attrs the surface erases (``batch_norm``'s
``momentum``/``cudnn_enabled``, ``conv2d``'s ``dilation=1``) cannot be
recovered, and synthesized ``tensor.zeros(tensor.shape(…))`` operands
have no term spelling, so importers decline those programs rather than
mint a wrong term.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, cast

from mlopt_core.ir import Const, Op, Param, TensorType, Var

from mlopt_discovery.hnl_surface import UnsupportedOpError

#: ``TensorType.shape=None`` is the documented wholly-unknown wildcard
#: (``mlopt_core.ir.TensorType``), but the dataclass annotation only
#: admits ``tuple[int | None, ...]`` — cast keeps ty honest about the
#: runtime contract without touching the eqsat-owned signature.
_WILDCARD_SHAPE = cast("tuple[int | None, ...]", None)

#: Accepted *leaves* value spellings for :func:`hnl_to_term`: a
#: :class:`TensorType` (wildcard ``shape=None`` included), a shape
#: tuple/list of ints (``None`` entries allowed), or ``None`` for the
#: wildcard.  Anything else is a type error at the boundary.
LeafShape = TensorType | Sequence[int | None] | None

__all__ = [
    "HnlImportError",
    "LeafShape",
    "covered_ops",
    "hnl_to_term",
]


class HnlImportError(ValueError):
    """The source is not in the importer's expression grammar.

    Distinct from :class:`UnsupportedOpError`: this is a *syntax*
    failure (unbalanced parens, a stray token), while an unsupported
    op is a well-formed program outside the covered ``tensor.*``
    surface.
    """


# ---------------------------------------------------------------------------
#  Lexer
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<punct>::|\.|[()\[\]{},])
    """,
    re.VERBOSE,
)


class _Tok:
    __slots__ = ("kind", "val")

    def __init__(self, kind: str, val: Any) -> None:
        self.kind = kind
        self.val = val

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Tok({self.kind}, {self.val!r})"


def _lex(src: str) -> list[_Tok]:
    """Tokenize *src*; raise :class:`HnlImportError` on bad input."""
    out: list[_Tok] = []
    i = 0
    n = len(src)
    while i < n:
        m = _TOKEN.match(src, i)
        if m is None:
            raise HnlImportError(
                f"unexpected character {src[i]!r} at offset {i}"
            )
        i = m.end()
        kind = m.lastgroup
        val = m.group()
        if kind == "ws":
            continue
        if kind == "num":
            has_frac = "." in val or "e" in val.lower()
            out.append(
                _Tok("num", float(val) if has_frac else int(val))
            )
        elif kind == "str":
            try:
                out.append(_Tok("str", json.loads(val)))
            except json.JSONDecodeError as e:
                raise HnlImportError(
                    f"bad string literal {val!r}: {e}"
                ) from e
        elif kind == "ident":
            out.append(_Tok("ident", val))
        else:
            out.append(_Tok(val, val))
    out.append(_Tok("eof", None))
    return out


# ---------------------------------------------------------------------------
#  Parser — AST nodes are tuples:
#    ("lit", value)          number / string / bool / None
#    ("name", ident)         a bound or free identifier
#    ("list", [nodes])       a list literal
#    ("call", [path], args)  path is the dotted callee, e.g. ["tensor","add"]
#    ("do", [binds], body)   binds are (name, expr) pairs
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, toks: list[_Tok]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self, ahead: int = 0) -> _Tok:
        return self.toks[min(self.i + ahead, len(self.toks) - 1)]

    def next(self) -> _Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, val: str) -> None:
        t = self.next()
        if t.val != val:
            raise HnlImportError(f"expected {val!r}, got {t.val!r}")

    def at_ident(self, word: str) -> bool:
        t = self.peek()
        return t.kind == "ident" and t.val == word

    def parse(self) -> Any:
        node = self.expr()
        if self.peek().kind != "eof":
            raise HnlImportError(
                f"trailing input after expression: {self.peek().val!r}"
            )
        return node

    def expr(self) -> Any:
        if self.at_ident("do"):
            return self.do_block()
        return self.postfix()

    def do_block(self) -> Any:
        self.next()  # 'do'
        self.expect("{")
        binds: list[tuple[str, Any]] = []
        while True:
            t = self.peek()
            if t.val == "}":
                self.next()
                break
            # binding: IDENT '::' expr — two-token lookahead
            if t.kind == "ident" and self.peek(1).val == "::":
                name = self.next().val
                self.next()  # '::'
                binds.append((name, self.expr()))
                continue
            # otherwise this is the body expression; it must be last
            body = self.expr()
            self.expect("}")
            return ("do", binds, body)
        raise HnlImportError("empty do block — no body expression")

    def postfix(self) -> Any:
        node = self.primary()
        path: list[str] | None = None
        if node[0] == "name":
            path = [node[1]]
            while self.peek().val == ".":
                self.next()
                t = self.next()
                if t.kind != "ident":
                    raise HnlImportError(
                        f"expected field name after '.', got {t.val!r}"
                    )
                path.append(t.val)
        if self.peek().val == "(":
            if path is None:
                raise HnlImportError(
                    "call target must be a (dotted) name"
                )
            self.next()
            args = self.arg_list()
            return ("call", path, args)
        if path is not None and len(path) > 1:
            # a bare dotted reference (tensor.add without a call) —
            # keep it a name so import declines it honestly.
            return ("name", ".".join(path))
        return node

    def arg_list(self) -> list[Any]:
        args: list[Any] = []
        if self.peek().val == ")":
            self.next()
            return args
        while True:
            args.append(self.expr())
            t = self.next()
            if t.val == ")":
                return args
            if t.val != ",":
                raise HnlImportError(
                    f"expected ',' or ')' in call args, got {t.val!r}"
                )

    def primary(self) -> Any:
        t = self.next()
        if t.kind == "num":
            return ("lit", t.val)
        if t.kind == "str":
            return ("lit", t.val)
        if t.kind == "ident":
            if t.val == "true":
                return ("lit", True)
            if t.val == "false":
                return ("lit", False)
            if t.val == "null":
                return ("lit", None)
            return ("name", t.val)
        if t.val == "(":
            node = self.expr()
            self.expect(")")
            return node
        if t.val == "[":
            items: list[Any] = []
            if self.peek().val == "]":
                self.next()
                return ("list", items)
            while True:
                items.append(self.expr())
                t2 = self.next()
                if t2.val == "]":
                    return ("list", items)
                if t2.val != ",":
                    raise HnlImportError(
                        f"expected ',' or ']' in list, got {t2.val!r}"
                    )
        raise HnlImportError(f"unexpected token {t.val!r}")


# ---------------------------------------------------------------------------
#  Module-level directive stripping
# ---------------------------------------------------------------------------

_DIRECTIVE = re.compile(
    r"^\s*(?:---|import\s|from\s|#|cap\s|op\s|handle\s)"
)


def _strip_directives(src: str) -> str:
    """Drop .hnl module directives, keeping expression lines.

    ``---`` doc comments, ``import``/``from`` module lines, ``#``
    comments and ``cap``/``op``/``handle`` declarations are interface
    text, not expressions — ``tensor.*`` resolves ambiently in the
    programs this importer targets.  Any other content must parse.
    """
    return "\n".join(
        line for line in src.splitlines() if not _DIRECTIVE.match(line)
    )


# ---------------------------------------------------------------------------
#  The importer
# ---------------------------------------------------------------------------

_NULL = object()  # the .hnl `null` literal, distinct from Python None
_NOARG = ("lit", None)  # AST spelling of an absent optional arg


class _Importer:
    """Stateful importer: ``do``-block env + leaf-shape overrides."""

    def __init__(self, leaves: dict[str, TensorType]) -> None:
        self.env: dict[str, Any] = {}
        self.leaves = leaves

    # -- literal / leaf helpers (no importer state) ----------------------

    @staticmethod
    def _lit(node: Any, what: str = "literal") -> Any:
        """Return a data literal (number/string/bool/None/list)."""
        if node[0] == "lit":
            return node[1]
        if node[0] == "list":
            return [_Importer._lit(x, what) for x in node[1]]
        raise UnsupportedOpError(
            f"expected a {what}, got {node[0]!r} node"
        )

    @classmethod
    def _num(cls, node: Any) -> int | float:
        v = cls._lit(node, "number")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise UnsupportedOpError(f"expected a number, got {v!r}")
        return v

    @classmethod
    def _bool(cls, node: Any) -> bool:
        v = cls._lit(node, "bool")
        if not isinstance(v, bool):
            raise UnsupportedOpError(f"expected true/false, got {v!r}")
        return v

    @classmethod
    def _dim(cls, node: Any) -> Any:
        """Return a dim attr value: int or int list."""
        v = cls._lit(node, "dim")
        if isinstance(v, list):
            return _int_list(v, "dim")
        if isinstance(v, bool) or not isinstance(v, int):
            raise UnsupportedOpError(
                f"expected an int or int list dim, got {v!r}"
            )
        return v

    # -- operand / term positions -----------------------------------------

    def _leaf(self, name: str) -> Any:
        if "." in name:
            raise UnsupportedOpError(
                f"dotted name {name!r} is a callee, not a leaf"
            )
        if name == "tensor":
            raise UnsupportedOpError(
                "'tensor' is the op namespace, not a leaf"
            )
        if name in self.env:
            return self.env[name]
        return Var(
            name, self.leaves.get(name, TensorType(_WILDCARD_SHAPE))
        )

    def _operand(self, node: Any) -> Any:
        """Import a node in operand position → term, Const, or ``_NULL``."""
        if node[0] == "lit":
            v = node[1]
            if v is None:
                return _NULL
            if isinstance(v, bool):
                return Const(int(v))
            if isinstance(v, (int, float)):
                return Const(v)
            raise UnsupportedOpError(
                f"string literal {v!r} is not a tensor operand"
            )
        if node[0] == "name":
            return self._leaf(node[1])
        if node[0] == "call":
            return self._call(node)
        raise UnsupportedOpError(
            "list literal is not a tensor operand here — it is data "
            "(shape/dims) or a concat/stack operand list"
        )

    def _term(self, node: Any) -> Any:
        """Import a node that must be a term (operand position)."""
        v = self._operand(node)
        if v is _NULL or v is None:
            raise UnsupportedOpError(
                "null in a required operand position"
            )
        return v

    def _call(self, node: Any) -> Any:
        _, path, args = node
        if len(path) != 2 or path[0] != "tensor":
            raise UnsupportedOpError(
                f"call {'.'.join(path)} is not a tensor.* op — only "
                "the tensor surface imports"
            )
        name = path[1]
        handler = _HANDLERS.get(name)
        if handler is None:
            raise UnsupportedOpError(
                f"tensor.{name} is not on the covered import surface "
                f"({len(_HANDLERS)} ops)"
            )
        return handler(self, args)

    def _oplist(self, node: Any) -> list[Any]:
        """Return a list literal of term operands."""
        if node[0] != "list":
            raise UnsupportedOpError("expected a [t0, …] operand list")
        return [self._term(x) for x in node[1]]

    def _present(self, node: Any) -> bool:
        """Whether an optional arg slot carries a value (not ``null``)."""
        return node != _NOARG

    # -- do-block ----------------------------------------------------------

    def _bind(self, name: str, expr: Any) -> None:
        """Bind one ``name :: expr`` — Param for marshaling, else inline."""
        param = self._marshaled_param(name, expr)
        if param is not None:
            self.env[name] = param
            return
        self.env[name] = self._import(expr)

    def _marshaled_param(self, name: str, expr: Any) -> Param | None:
        """Recognize the constant-tensor binding spellings → ``Param``.

        ``tensor.from_list([…])``, ``tensor.reshape(
        tensor.from_list([…]), […])``, ``tensor.zeros([…])`` and
        ``tensor.ones([…])`` all denote a *fixed* tensor — a ``Param``
        leaf whose shape the literal carries (the values themselves
        stay on the feed side, exactly like a weight leaf).
        """
        if expr[0] != "call":
            return None
        _, path, args = expr
        if len(path) != 2 or path[0] != "tensor":
            return None
        op = path[1]
        if op == "from_list" and len(args) == 1:
            # nested number arrays — the shape is the nesting itself
            shape = (
                _list_shape(args[0]) if args[0][0] == "list" else None
            )
            if shape is not None:
                return Param(name, TensorType(tuple(shape)))
            return None
        if op in ("zeros", "ones") and len(args) == 1:
            # the arg IS the shape list, not nested data
            shape = _shape_arg(args[0])
            if shape is not None:
                return Param(name, TensorType(tuple(shape)))
            return None
        if (
            op == "reshape"
            and len(args) == 2
            and args[0][0] == "call"
            and args[0][1] == ["tensor", "from_list"]
        ):
            shape = _shape_arg(args[1])
            if shape is not None:
                return Param(name, TensorType(tuple(shape)))
        return None

    def _import(self, node: Any) -> Any:
        """Import any AST node to a term (do-body / binding position)."""
        if node[0] == "do":
            _, binds, body = node
            for name, expr in binds:
                self._bind(name, expr)
            return self._import(body)
        if node[0] == "list":
            # A bare top-level list is the pack spelling.
            return Op.make("pack", *[self._term(x) for x in node[1]])
        return self._term(node)


def _list_shape(node: Any) -> list[int] | None:
    """Shape of a nested numeric list literal, or ``None`` if not data."""

    def dims(n: Any) -> list[int] | None:
        if n[0] == "lit":
            v = n[1]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            return []
        if n[0] != "list" or not n[1]:
            return None if n[0] != "list" else [0]
        subs: list[list[int]] = []
        for x in n[1]:
            d = dims(x)
            if d is None:
                return None
            subs.append(d)
        first = subs[0]
        if any(s != first for s in subs):
            return None  # ragged — from_list rejects it too
        return [len(n[1]), *first]

    return dims(node)


def _int_list(v: list[Any], what: str) -> list[int]:
    """Validate a literal list is all ints (bool/float/string reject)."""
    out: list[int] = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int):
            raise UnsupportedOpError(
                f"{what} entries must be ints, got {x!r}"
            )
        out.append(x)
    return out


def _shape_arg(node: Any) -> list[int] | None:
    """Return a shape list literal as ``list[int]``; else ``None``."""
    try:
        v = _Importer._lit(node, "shape")
    except UnsupportedOpError:
        return None
    if not isinstance(v, list):
        return None
    try:
        return _int_list(v, "shape")
    except UnsupportedOpError:
        return None


# ---------------------------------------------------------------------------
#  Op handlers — tensor.* name → mlopt term.  Each takes
#  ``(imp, args)`` where *args* are raw AST arg nodes.
# ---------------------------------------------------------------------------


def _arity(args: list[Any], lo: int, hi: int, op: str) -> None:
    if not lo <= len(args) <= hi:
        raise UnsupportedOpError(
            f"tensor.{op}: arity {len(args)} outside {lo}..{hi}"
        )


def _mint(imp: _Importer, op: str, ops: list[Any], attrs: dict) -> Any:
    """``Op.make`` over already-imported operand terms.

    Schema violations (e.g. an attr the op does not declare) surface
    as :class:`UnsupportedOpError` — a term mlopt cannot mint is a
    term the surface cannot honour.
    """
    try:
        return Op.make(op, *ops, **attrs)
    except ValueError as e:  # pragma: no cover - defensive: arity and
        # shape are pre-validated; this is the last line if a future
        # handler under-specifies a schema'd attr.
        raise UnsupportedOpError(f"tensor.{op}: {e}") from e


def _h_unary(op: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 1, 1, op)
        return Op.make(op, imp._term(args[0]))

    return go


def _h_nary(op: str, n: int):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, n, n, op)
        return Op.make(op, *[imp._term(a) for a in args])

    return go


def _h_reduce(op: str, *, keepdim: bool = True):
    """``(t, dim?, keepdim?)`` → attrs ``dim``/``keepdim``."""

    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 1, 3 if keepdim else 2, op)
        t = imp._term(args[0])
        attrs: dict[str, Any] = {}
        if len(args) > 1 and imp._present(args[1]):
            attrs["dim"] = imp._dim(args[1])
        if keepdim and len(args) > 2 and imp._present(args[2]):
            attrs["keepdim"] = imp._bool(args[2])
        return _mint(imp, op, [t], attrs)

    return go


def _h_var_std(op: str):
    """``tensor.var/std(t, dim?, unbiased?, keepdim?)``.

    The host port's third slot is the *unbiased* bool — the generated
    decoder has no ``correction`` slot (``mlopt_host_tensor.bindings``
    documents ``correction ∈ {0,1}`` mapping onto it): ``true`` →
    ``correction=1``, ``false`` → ``correction=0``.
    """

    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 1, 4, op)
        t = imp._term(args[0])
        attrs: dict[str, Any] = {}
        if len(args) > 1 and imp._present(args[1]):
            attrs["dim"] = imp._dim(args[1])
        if len(args) > 2 and imp._present(args[2]):
            attrs["correction"] = 1 if imp._bool(args[2]) else 0
        if len(args) > 3 and imp._present(args[3]):
            attrs["keepdim"] = imp._bool(args[3])
        return _mint(imp, op, [t], attrs)

    return go


def _h_conv2d(imp: _Importer, args: list[Any]) -> Any:
    """``(x, w, b?, stride?, padding?, groups?)`` — dilation has no slot."""
    _arity(args, 2, 6, "conv2d")
    ops = [imp._term(args[0]), imp._term(args[1])]
    if len(args) > 2 and imp._present(args[2]):
        ops.append(imp._term(args[2]))
    attrs: dict[str, Any] = {}
    for i, name in ((3, "stride"), (4, "padding"), (5, "groups")):
        if len(args) > i and imp._present(args[i]):
            attrs[name] = imp._lit(args[i], name)
    return _mint(imp, "conv2d", ops, attrs)


def _h_batch_norm(imp: _Importer, args: list[Any]) -> Any:
    """``(x, gamma, beta, mean, var, eps?)`` — eval-mode only."""
    _arity(args, 5, 6, "batch_norm")
    ops = [imp._term(a) for a in args[:5]]
    attrs: dict[str, Any] = {"training": False}
    if len(args) > 5 and imp._present(args[5]):
        attrs["eps"] = imp._num(args[5])
    return _mint(imp, "batch_norm", ops, attrs)


def _h_max_pool(imp: _Importer, args: list[Any]) -> Any:
    """``(input, kernel, stride?)`` → ``max_pool2d`` attrs."""
    _arity(args, 2, 3, "max_pool")
    attrs: dict[str, Any] = {
        "kernel_size": imp._lit(args[1], "kernel_size")
    }
    if len(args) > 2 and imp._present(args[2]):
        attrs["stride"] = imp._lit(args[2], "stride")
    return _mint(imp, "max_pool2d", [imp._term(args[0])], attrs)


def _h_reshape(op: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 2, 2, op)
        shape = imp._lit(args[1], "shape")
        if not isinstance(shape, list):
            raise UnsupportedOpError(
                f"tensor.{op}: shape arg must be a list literal"
            )
        return _mint(imp, op, [imp._term(args[0])], {"shape": shape})

    return go


def _h_transpose(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 3, 3, "transpose")
    return _mint(
        imp,
        "transpose",
        [imp._term(args[0])],
        {"dim0": imp._num(args[1]), "dim1": imp._num(args[2])},
    )


def _h_concat(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 2, 2, "concat")
    ts = imp._oplist(args[0])
    if not ts:
        raise UnsupportedOpError("tensor.concat: empty operand list")
    return _mint(imp, "concat", ts, {"dim": imp._num(args[1])})


def _h_stack(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 1, 2, "stack")
    ts = imp._oplist(args[0])
    if not ts:
        raise UnsupportedOpError("tensor.stack: empty operand list")
    attrs: dict[str, Any] = {}
    if len(args) > 1 and imp._present(args[1]):
        attrs["dim"] = imp._num(args[1])
    return _mint(imp, "stack", ts, attrs)


def _h_unsqueeze(op: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 2, 2, op)
        return _mint(
            imp, op, [imp._term(args[0])], {"dim": imp._num(args[1])}
        )

    return go


def _h_select(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 3, 3, "select")
    return _mint(
        imp,
        "select",
        [imp._term(args[0])],
        {"dim": imp._num(args[1]), "index": imp._num(args[2])},
    )


def _h_squeeze(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 1, 2, "squeeze")
    attrs: dict[str, Any] = {}
    if len(args) > 1 and imp._present(args[1]):
        attrs["dim"] = imp._dim(args[1])
    return _mint(imp, "squeeze", [imp._term(args[0])], attrs)


def _h_flatten(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 1, 3, "flatten")
    attrs: dict[str, Any] = {}
    if len(args) > 1 and imp._present(args[1]):
        attrs["start_dim"] = imp._num(args[1])
    if len(args) > 2 and imp._present(args[2]):
        attrs["end_dim"] = imp._num(args[2])
    return _mint(imp, "flatten", [imp._term(args[0])], attrs)


def _h_narrow(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 4, 4, "narrow")
    return _mint(
        imp,
        "narrow",
        [imp._term(args[0])],
        {
            "dim": imp._num(args[1]),
            "start": imp._num(args[2]),
            "length": imp._num(args[3]),
        },
    )


def _h_slice(imp: _Importer, args: list[Any]) -> Any:
    """``(t, dim?, start?, end?, step?)`` — attrs for present slots."""
    _arity(args, 1, 5, "slice")
    names = ("dim", "start", "end", "step")
    attrs: dict[str, Any] = {}
    for i, name in enumerate(names, start=1):
        if len(args) > i and imp._present(args[i]):
            attrs[name] = imp._num(args[i])
    if attrs and "dim" not in attrs:
        # ATTR_REQUIRED['slice'] = {'dim'} — the host default is 0.
        attrs["dim"] = 0
    return _mint(imp, "slice", [imp._term(args[0])], attrs)


def _h_cumsum(op: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 2, 2, op)
        return _mint(
            imp, op, [imp._term(args[0])], {"dim": imp._num(args[1])}
        )

    return go


def _h_softmax(op: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 2, 2, op)
        return _mint(
            imp, op, [imp._term(args[0])], {"dim": imp._num(args[1])}
        )

    return go


def _h_layer_norm(op: str, *, has_bias: bool):
    """``layer_norm(x, shape, w?, b?, eps?)`` / ``rms_norm(x, shape, w?, eps?)``."""

    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 2, 5 if has_bias else 4, op)
        x = imp._term(args[0])
        shape = imp._lit(args[1], "normalized_shape")
        if not isinstance(shape, list):
            raise UnsupportedOpError(
                f"tensor.{op}: normalized_shape must be a list"
            )
        ops: list[Any] = [x]
        attrs: dict[str, Any] = {"dim": [int(d) for d in shape]}
        rest = args[2:]

        def slot(i: int) -> Any:
            return rest[i] if len(rest) > i else _NOARG

        w, b = slot(0), slot(1) if has_bias else _NOARG
        eps = slot(2 if has_bias else 1)
        if imp._present(w):
            ops.append(imp._term(w))
        elif has_bias and imp._present(b):
            raise UnsupportedOpError(
                f"tensor.{op}: bias without weight has no term spelling"
            )
        if has_bias and imp._present(b):
            ops.append(imp._term(b))
        if imp._present(eps):
            attrs["eps"] = imp._num(eps)
        return _mint(imp, op, ops, attrs)

    return go


def _h_clamp(imp: _Importer, args: list[Any]) -> Any:
    """``(t, min?, max?)`` — numeric bounds → attrs, term bounds → args."""
    _arity(args, 1, 3, "clamp")
    t = imp._term(args[0])
    attrs: dict[str, Any] = {}
    extra: list[Any] = []
    for i, name in ((1, "min"), (2, "max")):
        if len(args) <= i or not imp._present(args[i]):
            continue
        node = args[i]
        if node[0] == "lit":
            attrs[name] = imp._num(node)
        elif node[0] in ("name", "call"):
            extra.append(imp._term(node))
        else:
            raise UnsupportedOpError(
                f"tensor.clamp: {name} bound must be a number or term"
            )
    if extra and attrs:
        raise UnsupportedOpError(
            "tensor.clamp: mixed literal/term bounds have no "
            "unambiguous term spelling"
        )
    return _mint(imp, "clamp", [t, *extra], attrs)


def _h_clamp_side(op: str, attr: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 2, 2, op)
        t = imp._term(args[0])
        node = args[1]
        if node[0] == "lit":
            return _mint(imp, op, [t], {attr: imp._num(node)})
        return _mint(imp, op, [t, imp._term(node)], {})

    return go


def _h_where(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 3, 3, "where")
    return _mint(
        imp,
        "where",
        [imp._term(args[0]), imp._term(args[1]), imp._term(args[2])],
        {},
    )


def _h_nan_to_num(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 1, 4, "nan_to_num")
    attrs: dict[str, Any] = {}
    for i, name in ((1, "nan"), (2, "posinf"), (3, "neginf")):
        if len(args) > i and imp._present(args[i]):
            attrs[name] = imp._num(args[i])
    return _mint(imp, "nan_to_num", [imp._term(args[0])], attrs)


def _h_act_attrs(op: str, names: tuple[str, ...]):
    """Unary activation with a positional scalar tail → attrs."""

    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 1, 1 + len(names), op)
        attrs: dict[str, Any] = {}
        for i, name in enumerate(names, start=1):
            if len(args) > i and imp._present(args[i]):
                attrs[name] = imp._num(args[i])
        return _mint(imp, op, [imp._term(args[0])], attrs)

    return go


def _h_gelu(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 1, 2, "gelu")
    attrs: dict[str, Any] = {}
    if len(args) > 1 and imp._present(args[1]):
        v = imp._lit(args[1], "approximate")
        if not isinstance(v, str):
            raise UnsupportedOpError(
                "tensor.gelu: approximate must be a string"
            )
        attrs["approximate"] = v
    return _mint(imp, "gelu", [imp._term(args[0])], attrs)


def _h_expand(op: str):
    """``(t, *shape)`` or ``(t, shape-list)`` → ``shape`` attr."""

    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 2, 16, op)
        t = imp._term(args[0])
        rest = args[1:]
        if len(rest) == 1 and rest[0][0] == "list":
            shape = imp._lit(rest[0], "shape")
        else:
            shape = [imp._num(a) for a in rest]
        return _mint(imp, op, [t], {"shape": shape})

    return go


def _h_repeat(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 2, 16, "repeat")
    t = imp._term(args[0])
    rest = args[1:]
    if len(rest) == 1 and rest[0][0] == "list":
        shape = imp._lit(rest[0], "repeats")
    else:
        shape = [imp._num(a) for a in rest]
    return _mint(imp, "repeat", [t], {"shape": shape})


def _h_flip(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 2, 2, "flip")
    dims = imp._lit(args[1], "dims")
    if not isinstance(dims, list):
        raise UnsupportedOpError("tensor.flip: dims must be a list")
    return _mint(imp, "flip", [imp._term(args[0])], {"dim": dims})


def _h_tril_triu(op: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 1, 2, op)
        attrs: dict[str, Any] = {}
        if len(args) > 1 and imp._present(args[1]):
            attrs["diagonal"] = imp._num(args[1])
        return _mint(imp, op, [imp._term(args[0])], attrs)

    return go


def _h_pixel(op: str, attr: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 2, 2, op)
        return _mint(
            imp, op, [imp._term(args[0])], {attr: imp._num(args[1])}
        )

    return go


def _h_lerp(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 3, 3, "lerp")
    return _mint(
        imp,
        "lerp",
        [imp._term(args[0]), imp._term(args[1]), imp._term(args[2])],
        {},
    )


def _h_full(imp: _Importer, args: list[Any]) -> Any:
    """``(shape-list, fill?)`` — the host order ``bindings._full`` pins."""
    _arity(args, 1, 2, "full")
    if args[0][0] != "list":
        raise UnsupportedOpError(
            "tensor.full: first arg must be a shape list literal"
        )
    shape = _shape_arg(args[0])
    if shape is None:
        raise UnsupportedOpError(
            "tensor.full: first arg must be an int shape list"
        )
    attrs: dict[str, Any] = {"shape": shape}
    if len(args) > 1 and imp._present(args[1]):
        attrs["fill_value"] = imp._num(args[1])
    return _mint(imp, "full", [], attrs)


def _h_full_like(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 1, 2, "full_like")
    attrs: dict[str, Any] = {}
    if len(args) > 1 and imp._present(args[1]):
        attrs["fill_value"] = imp._num(args[1])
    return _mint(imp, "full_like", [imp._term(args[0])], attrs)


def _h_zeros_ones(op: str):
    def go(imp: _Importer, args: list[Any]) -> Any:
        _arity(args, 1, 1, op)
        shape = _shape_arg(args[0])
        if shape is None:
            raise UnsupportedOpError(
                f"tensor.{op}: shape must be an int list literal"
            )
        return _mint(imp, op, [], {"shape": shape})

    return go


def _h_arange(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 1, 3, "arange")
    return Op.make("arange", *[Const(imp._num(a)) for a in args])


def _h_eye(imp: _Importer, args: list[Any]) -> Any:
    _arity(args, 1, 2, "eye")
    attrs: dict[str, Any] = {"dim": imp._num(args[0])}
    if len(args) > 1 and imp._present(args[1]):
        attrs["m"] = imp._num(args[1])
    return _mint(imp, "eye", [], attrs)


def _h_t(imp: _Importer, args: list[Any]) -> Any:
    """``tensor.t`` — the ≤2-D transpose spelling → ``transpose(-2,-1)``."""
    _arity(args, 1, 1, "t")
    return _mint(
        imp,
        "transpose",
        [imp._term(args[0])],
        {"dim0": -2, "dim1": -1},
    )


# ---------------------------------------------------------------------------
#  The handler table
# ---------------------------------------------------------------------------

#: Attr-free unary ops — ``tensor.op(t)`` → ``Op(op, t)``.  The set is
#: ``mlopt_host_tensor.bindings``' ``_u1`` builders (the IR ops whose
#: host port takes exactly one tensor and no attrs); ``numel``/``item``
#: return scalars but are still one-operand IR ops.
_UNARY_OPS = frozenset(
    [
        "abs",
        "neg",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "square",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "floor",
        "ceil",
        "round",
        "trunc",
        "frac",
        "sign",
        "sigmoid",
        "silu",
        "mish",
        "relu",
        "erf",
        "erfc",
        "sinc",
        "isnan",
        "isinf",
        "isfinite",
        "isneginf",
        "isposinf",
        "logical_not",
        "contiguous",
        "clone",
        "detach",
        "zeros_like",
        "ones_like",
        "numel",
        "item",
    ]
)

#: Attr-free binary ops — ``tensor.op(a, b)`` → ``Op(op, a, b)``.
#: ``mlopt_host_tensor.bindings``' ``_u2``/``_bin_scalar_rhs`` set plus
#: the elementwise max/min family (``maximum``/``minimum`` — ``max``/
#: ``min`` themselves are reductions and decline, see below).
_BINARY_OPS = frozenset(
    [
        "add",
        "sub",
        "mul",
        "div",
        "matmul",
        "maximum",
        "minimum",
        "atan2",
        "logical_and",
        "logical_or",
        "logical_xor",
        "heaviside",
        "pow",
        "fmod",
        "remainder",
        "xlogy",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "outer",
        "expand_as",
    ]
)


def _build_handlers() -> dict[str, Any]:
    h: dict[str, Any] = {}
    for op in sorted(_UNARY_OPS):
        h[op] = _h_unary(op)
    for op in sorted(_BINARY_OPS):
        h[op] = _h_nary(op, 2)
    # reductions — (t, dim?, keepdim?)
    for op in (
        "sum",
        "prod",
        "mean",
        "amax",
        "amin",
        "nanmean",
        "nansum",
        "argmax",
        "argmin",
        "all",
        "any",
    ):
        h[op] = _h_reduce(op)
    h["count_nonzero"] = _h_reduce("count_nonzero", keepdim=False)
    h["var"] = _h_var_std("var")
    h["std"] = _h_var_std("std")
    # conv / norm / pool / linalg
    h["conv2d"] = _h_conv2d
    h["batch_norm"] = _h_batch_norm
    h["max_pool"] = _h_max_pool
    h["linear"] = _h_nary("linear", 3)
    # shape manipulation
    h["reshape"] = _h_reshape("reshape")
    h["view"] = _h_reshape("view")
    h["transpose"] = _h_transpose
    h["concat"] = _h_concat
    h["stack"] = _h_stack
    h["unsqueeze"] = _h_unsqueeze("unsqueeze")
    h["select"] = _h_select
    h["squeeze"] = _h_squeeze
    h["flatten"] = _h_flatten
    h["narrow"] = _h_narrow
    h["slice"] = _h_slice
    h["cumsum"] = _h_cumsum("cumsum")
    h["cumprod"] = _h_cumsum("cumprod")
    h["softmax"] = _h_softmax("softmax")
    h["log_softmax"] = _h_softmax("log_softmax")
    h["layer_norm"] = _h_layer_norm("layer_norm", has_bias=True)
    h["rms_norm"] = _h_layer_norm("rms_norm", has_bias=False)
    h["expand"] = _h_expand("expand")
    h["repeat"] = _h_repeat
    h["flip"] = _h_flip
    h["tril"] = _h_tril_triu("tril")
    h["triu"] = _h_tril_triu("triu")
    h["pixel_shuffle"] = _h_pixel("pixel_shuffle", "upscale_factor")
    h["pixel_unshuffle"] = _h_pixel(
        "pixel_unshuffle", "downscale_factor"
    )
    # pointwise with scalar tails / attrs
    h["where"] = _h_where
    h["clamp"] = _h_clamp
    h["clamp_min"] = _h_clamp_side("clamp_min", "min")
    h["clamp_max"] = _h_clamp_side("clamp_max", "max")
    h["nan_to_num"] = _h_nan_to_num
    h["elu"] = _h_act_attrs("elu", ("alpha", "scale", "input_scale"))
    h["celu"] = _h_act_attrs("celu", ("alpha",))
    h["leaky_relu"] = _h_act_attrs("leaky_relu", ("negative_slope",))
    h["softplus"] = _h_act_attrs("softplus", ("beta", "threshold"))
    h["hardtanh"] = _h_act_attrs("hardtanh", ("min", "max"))
    h["gelu"] = _h_gelu
    h["lerp"] = _h_lerp
    # creators
    h["full"] = _h_full
    h["full_like"] = _h_full_like
    h["zeros"] = _h_zeros_ones("zeros")
    h["ones"] = _h_zeros_ones("ones")
    h["arange"] = _h_arange
    h["eye"] = _h_eye
    h["t"] = _h_t
    return h


_HANDLERS: dict[str, Any] = _build_handlers()


def covered_ops() -> frozenset[str]:
    """Return the ``tensor.*`` op names :func:`hnl_to_term` accepts."""
    return frozenset(_HANDLERS)


# ---------------------------------------------------------------------------
#  Public entry
# ---------------------------------------------------------------------------


def hnl_to_term(
    src: str,
    *,
    leaves: dict[str, LeafShape] | None = None,
) -> Any:
    """Parse ``.hnl`` *src* into an mlopt term over the ``tensor.*`` surface.

    *src* is an expression (``tensor.add(x, y)``), a ``do { … }``
    program, or a module fragment — directive lines (``---`` comments,
    ``import``/``from``/``cap``/``op``/``handle``) are stripped first.
    *leaves* optionally maps free identifier names to shapes —
    ``TensorType``, a shape sequence of ints (``None`` entries
    allowed), or ``None`` for the wildcard;
    unlisted names get ``TensorType(None)``.

    Raises :class:`HnlImportError` on syntax the grammar does not
    cover and :class:`UnsupportedOpError` naming the first
    ``tensor.*`` op (or value) outside the covered surface.
    """
    text = _strip_directives(src)
    toks = _lex(text)
    node = _Parser(toks).parse()
    leaf_types: dict[str, TensorType] = {}
    for name, shape in (leaves or {}).items():
        if isinstance(shape, TensorType):
            leaf_types[name] = shape
        elif shape is None:
            leaf_types[name] = TensorType(_WILDCARD_SHAPE)
        else:
            leaf_types[name] = TensorType(tuple(shape))
    imp = _Importer(leaf_types)
    return imp._import(node)
