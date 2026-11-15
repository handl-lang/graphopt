"""mlopt term ↔ serde-shaped handl ``Term`` JSON — the wire codec.

This is the JSON half of the ``.hnl`` codec pair (ADR-0172): where
:mod:`.hnl_surface` / :mod:`.hnl_import` move terms through *surface
syntax*, this module moves them through the **Term-JSON** envelope —
the externally-tagged serde shape ``hnl-engine`` parses
(``{"Op": …}`` / ``{"Arr": …}`` / ``{"Const": …}`` / ``{"Opaque": …}``),
promoted from the ``term_ir_spike.py`` proof of concept
(``docs/reports/term_ir_spike.md``) now that ADR-0173 has decided the
two envelope conventions:

* **Non-finite floats** spell ``{"__nonfinite__": "nan" | "inf" |
  "-inf"}`` — host-tensor's wire-codec tag, adopted verbatim
  (``serde_json::Value`` cannot hold NaN/±inf at all, so the tag is a
  codec-profile convention, never a bare literal).  It applies
  wherever a domain scalar lands in a ``Value`` position: ``Const``
  values and the ``__mlopt_attrs__`` / ``attrs`` payload channels,
  recursively (tuple/list spellings included).
* **DAG sharing** rides ``{"Opaque": {"__tag__": "dag.program",
  "nodes": [...], "root": n}}`` — the ADR-0173 derived form.  The
  kernel carries the record opaquely; this codec interprets it: a
  term whose subtrees are shared serializes as a node table (every
  distinct node once, ``Op.args`` as *indices* into ``nodes``,
  children preceding parents) instead of exponential tree-expansion,
  and decode rebuilds the sharing by memoized construction.  Surface
  ``let`` syntax is deliberately absent — the record is an envelope,
  not language.

The attrs channel is the spike's ``lit.__mlopt_attrs__`` record, with
the ADR-0173 ``attrs`` field accepted on decode as the forward
spelling (the Rust ``Term::Op.attrs`` slot lands separately; when it
ships, ``attrs`` becomes the canonical channel and ``lit`` the
legacy one — decode accepts exactly one, never both ambiguous).

Honest declines, never silent wrong answers:

* ``term_to_json(..., dag=False)`` refuses a shared term rather than
  tree-expanding it (the exponential blow-up the record exists to
  prevent);
* ``term_from_json(..., allow_dag=False)`` refuses the
  ``dag.program`` record for consumers that cannot represent sharing;
* any other ``Opaque`` tag, a malformed node table, or an
  unserializable leaf raises :class:`HnlCodecError` / ``TypeError``.
"""

from __future__ import annotations

import math
from typing import Any, cast

from mlopt_core.ir import (
    Const,
    Op,
    Param,
    TensorType,
    Var,
    attr_from_data,
    attr_to_data,
)

__all__ = [
    "HnlCodecError",
    "term_from_json",
    "term_to_json",
]


class HnlCodecError(ValueError):
    """The JSON is not a term this codec can faithfully represent."""


# ---------------------------------------------------------------------------
#  Wire conventions — ADR-0173 envelope tags
# ---------------------------------------------------------------------------

#: Non-finite scalar tag — the exact host-tensor wire spelling.
_NONFINITE_TAG = "__nonfinite__"
_NONFINITE_VALUES = ("nan", "inf", "-inf")

#: The dag record's ``__tag__`` value, carried inside ``Term::Opaque``.
_DAG_TAG = "dag.program"

#: serde defaults pinned by the spike: ``Term::Op.dim`` defaults to
#: ``Const(1)`` and serializes; ``source`` is the ``tensor`` module
#: boundary (the ``tensor.*`` host surface's registry key).
_DIM1 = {"Const": {"value": 1}}
_TENSOR = {"Foreign": "tensor"}


# ---------------------------------------------------------------------------
#  ``__nonfinite__`` — scalar ↔ wire value
# ---------------------------------------------------------------------------


def _wire_value(v: Any) -> Any:
    """Encode *v* (post-``attr_to_data`` shape) as JSON-safe data.

    Non-finite floats become the ``__nonfinite__`` tagged object;
    everything else passes through, recursing into lists and dicts
    (the ``__tuple__``/``__list__`` attr wrappers included).
    """
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        if math.isnan(v):
            return {_NONFINITE_TAG: "nan"}
        return {_NONFINITE_TAG: "inf" if v > 0 else "-inf"}
    if isinstance(v, (list, tuple)):
        return [_wire_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _wire_value(x) for k, x in v.items()}
    raise TypeError(f"unserializable wire value: {v!r}")


def _unwire_value(v: Any) -> Any:
    """Decode :func:`_wire_value` output — tags back to floats.

    A dict that is *exactly* ``{"__nonfinite__": s}`` decodes to the
    float; every other dict keeps its shape with values decoded
    recursively, so the ``__tuple__``/``__list__`` wrappers survive for
    :func:`attr_from_data`.
    """
    if isinstance(v, dict):
        if set(v) == {_NONFINITE_TAG}:
            s = v[_NONFINITE_TAG]
            if s not in _NONFINITE_VALUES:
                raise HnlCodecError(
                    f"bad {_NONFINITE_TAG} payload {s!r} — expected "
                    "'nan', 'inf' or '-inf'"
                )
            if s == "nan":
                return float("nan")
            return float("inf") if s == "inf" else float("-inf")
        return {k: _unwire_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_unwire_value(x) for x in v]
    return v


# ---------------------------------------------------------------------------
#  Shared helpers — leaf records and the attrs channel
# ---------------------------------------------------------------------------


def _shape_data(typ: TensorType) -> Any:
    return None if typ.shape is None else list(typ.shape)


def _leaf_record(kind: str, name: str, typ: TensorType) -> dict:
    return {
        "__mlopt_leaf__": kind,
        "name": name,
        "shape": _shape_data(typ),
    }


def _attrs_payload(attrs: dict[str, Any]) -> dict:
    """Return the ``lit.__mlopt_attrs__`` record, wire-encoded."""
    return {
        "__mlopt_attrs__": {
            k: _wire_value(attr_to_data(v)) for k, v in attrs.items()
        }
    }


def _attrs_from_node(node: dict) -> dict[str, Any]:
    """Decode an ``Op`` payload's attrs — one channel, never ambiguous.

    Accepts the spike's ``lit.__mlopt_attrs__`` record and the
    ADR-0173 ``attrs`` field (``Option<Value>`` — for mlopt terms the
    fragment schema is the same ``attr_to_data`` encoding).  Both
    present is a decline: two spellings of one fact must not be
    silently merged.
    """
    lit = node.get("lit")
    field = node.get("attrs")
    if lit is not None and field is not None:
        raise HnlCodecError(
            "Op carries both lit.__mlopt_attrs__ and attrs — "
            "ambiguous attrs channel"
        )
    if field is not None:
        if not isinstance(field, dict):
            raise HnlCodecError(
                f"Op.attrs must be an object, got {field!r}"
            )
        raw = field
    elif lit is not None:
        if not isinstance(lit, dict) or "__mlopt_attrs__" not in lit:
            raise HnlCodecError(
                f"Op.lit is not an attrs record: {lit!r}"
            )
        raw = lit["__mlopt_attrs__"]
        if not isinstance(raw, dict):
            raise HnlCodecError(
                f"__mlopt_attrs__ must be an object, got {raw!r}"
            )
    else:
        return {}
    try:
        return {
            k: attr_from_data(_unwire_value(v)) for k, v in raw.items()
        }
    except (TypeError, ValueError) as e:
        raise HnlCodecError(f"bad attrs payload: {e}") from e


# ---------------------------------------------------------------------------
#  Encode — mlopt term → Term JSON
# ---------------------------------------------------------------------------


def _leaf_json(t: Var | Param) -> dict:
    kind = "var" if isinstance(t, Var) else "param"
    return {
        "Arr": {
            "dim": _DIM1,
            "value": _leaf_record(kind, t.name, t.typ),
        }
    }


def _const_json(t: Const) -> dict:
    return {"Const": {"value": _wire_value(t.value)}}


def _enc(t: Any) -> dict:
    """Tree-encode *t* — valid only where no subtree is shared."""
    if isinstance(t, Op):
        node: dict[str, Any] = {
            "dim": _DIM1,
            "name": t.op,
            "args": [_enc(a) for a in t.args],
            "source": _TENSOR,
        }
        if t.attrs:
            node["lit"] = _attrs_payload(t.attrs)
        return {"Op": node}
    if isinstance(t, (Var, Param)):
        return _leaf_json(t)
    if isinstance(t, Const):
        return _const_json(t)
    raise TypeError(f"unencodable term: {t!r}")


def _has_sharing(root: Any) -> bool:
    """Whether any node is referenced by more than one parent.

    Counts *child occurrences* over the DAG — mlopt's ``Op.make``
    interning means structurally equal subtrees are the same object,
    so ``id`` counting is exactly semantic sharing.  Recursion into a
    node happens once (``seen``), so the walk is linear in distinct
    nodes even on exponential tree-expansions.
    """
    counts: dict[int, int] = {}
    seen: set[int] = set()

    def walk(t: Any) -> None:
        if not isinstance(t, Op) or id(t) in seen:
            return
        seen.add(id(t))
        for a in t.args:
            counts[id(a)] = counts.get(id(a), 0) + 1
            walk(a)

    walk(root)
    # children of the root's *siblings* are counted; a node used twice
    # anywhere (including twice under one parent) has count > 1.
    return any(c > 1 for c in counts.values())


def _enc_dag(root: Any) -> dict:
    """Encode *root* as the ``dag.program`` node-table record.

    Every distinct node becomes one ``nodes`` entry in post-order
    (children before parents); an ``Op``'s ``args`` are *indices* into
    the table, so a shared subtree is emitted once and referenced —
    the record is linear in distinct nodes where tree encoding is
    exponential.  Leaves (``Arr``/``Const``) are complete records.
    """
    index: dict[int, int] = {}
    nodes: list[dict] = []

    def emit(t: Any) -> int:
        key = id(t)
        if key in index:
            return index[key]
        if isinstance(t, Op):
            for a in t.args:
                emit(a)
            node: dict[str, Any] = {
                "dim": _DIM1,
                "name": t.op,
                "args": [index[id(a)] for a in t.args],
                "source": _TENSOR,
            }
            if t.attrs:
                node["lit"] = _attrs_payload(t.attrs)
            record: dict[str, Any] = {"Op": node}
        elif isinstance(t, (Var, Param)):
            record = _leaf_json(t)
        elif isinstance(t, Const):
            record = _const_json(t)
        else:
            raise TypeError(f"unencodable term: {t!r}")
        index[key] = len(nodes)
        nodes.append(record)
        return index[key]

    root_index = emit(root)
    return {
        "Opaque": {
            "__tag__": _DAG_TAG,
            "nodes": nodes,
            "root": root_index,
        }
    }


def term_to_json(term: Any, *, dag: bool | None = None) -> dict:
    """Encode *term* as serde-shaped handl ``Term`` JSON.

    ``dag=None`` (default) picks the envelope honestly: a term with
    shared subtrees serializes as the ``dag.program`` record, an
    unshared one as the plain tree.  ``dag=True`` forces the record;
    ``dag=False`` refuses a shared term with :class:`HnlCodecError`
    rather than tree-expanding it — the consumer that cannot
    represent sharing gets an honest decline, not an exponential
    surprise.
    """
    shared = _has_sharing(term)
    if dag is False and shared:
        raise HnlCodecError(
            "term has shared subtrees — dag=False cannot represent "
            "sharing (use the dag.program record or dag=None)"
        )
    if dag is True or (dag is None and shared):
        return _enc_dag(term)
    return _enc(term)


# ---------------------------------------------------------------------------
#  Decode — Term JSON → mlopt term
# ---------------------------------------------------------------------------


def _dec_leaf(j: dict) -> Any:
    """Decode an ``Arr``/``Const`` node (no children to resolve)."""
    if "Arr" in j:
        rec = j["Arr"]["value"]
        shape = rec["shape"]
        # ``TensorType.shape=None`` is the documented wholly-unknown
        # wildcard, but the annotation only admits the tuple — the
        # same cast ``hnl_import`` uses for the runtime contract.
        typ = TensorType(
            cast(
                "tuple[int | None, ...]",
                tuple(shape) if shape is not None else None,
            )
        )
        if rec["__mlopt_leaf__"] == "var":
            return Var(rec["name"], typ)
        return Param(rec["name"], typ)
    if "Const" in j:
        return Const(_unwire_value(j["Const"]["value"]))
    raise HnlCodecError(f"bad leaf encoding: {list(j)[:3]}")


def _dec(j: Any) -> Any:
    """Decode a tree-position Term JSON node."""
    if not isinstance(j, dict) or len(j) < 1:
        raise HnlCodecError(f"bad term encoding: {j!r}")
    if "Opaque" in j:
        raise HnlCodecError(
            "nested Opaque node — only a top-level dag.program record "
            "is supported"
        )
    if "Op" in j:
        node = j["Op"]
        return Op.make(
            node["name"],
            *[_dec(a) for a in node["args"]],
            **_attrs_from_node(node),
        )
    return _dec_leaf(j)


def _dec_dag(rec: dict) -> Any:
    """Decode the ``dag.program`` record, rebuilding shared nodes once.

    ``nodes`` is a post-order table — every ``Op.args`` entry is an
    *index* of an earlier node, so children are always built before
    their parents and each distinct node is constructed exactly once:
    the decoded term's sharing is the record's sharing.
    """
    nodes = rec.get("nodes")
    root = rec.get("root")
    if not isinstance(nodes, list):
        raise HnlCodecError("dag.program: 'nodes' must be a list")
    if (
        not isinstance(root, int)
        or isinstance(root, bool)
        or not 0 <= root < len(nodes)
    ):
        raise HnlCodecError(
            f"dag.program: bad root index {root!r} ({len(nodes)} nodes)"
        )
    built: list[Any] = []
    for i, n in enumerate(nodes):
        if not isinstance(n, dict) or len(n) != 1:
            raise HnlCodecError(
                f"dag.program: node {i} is not a tagged record: {n!r}"
            )
        if "Op" in n:
            node = n["Op"]
            args = node.get("args")
            if not isinstance(args, list):
                raise HnlCodecError(
                    f"dag.program: node {i} args must be an index list"
                )
            kids = []
            for a in args:
                if (
                    not isinstance(a, int)
                    or isinstance(a, bool)
                    or not 0 <= a < i
                ):
                    raise HnlCodecError(
                        f"dag.program: node {i} arg {a!r} is not an "
                        f"earlier node index (0..{i - 1})"
                    )
                kids.append(built[a])
            built.append(
                Op.make(node["name"], *kids, **_attrs_from_node(node))
            )
        elif "Arr" in n or "Const" in n:
            built.append(_dec_leaf(n))
        else:
            raise HnlCodecError(
                f"dag.program: node {i} has unknown tag {list(n)[:3]}"
            )
    return built[root]


def term_from_json(data: Any, *, allow_dag: bool = True) -> Any:
    """Decode Term JSON back to an mlopt term — :func:`term_to_json`'s inverse.

    Understands both envelopes: the plain tree and the
    ``dag.program`` record (decoded with sharing intact).
    ``allow_dag=False`` refuses the record for consumers that cannot
    represent sharing.  Any other ``Opaque`` tag declines — mlopt
    terms have no opaque slot, so carrying one would be a silent
    wrong answer.
    """
    if not isinstance(data, dict) or len(data) < 1:
        raise HnlCodecError(f"bad term encoding: {data!r}")
    if "Opaque" in data:
        payload = data["Opaque"]
        if (
            isinstance(payload, dict)
            and payload.get("__tag__") == _DAG_TAG
        ):
            if not allow_dag:
                raise HnlCodecError(
                    "dag.program record — this consumer cannot "
                    "represent sharing (allow_dag=False)"
                )
            return _dec_dag(payload)
        tag = (
            payload.get("__tag__")
            if isinstance(payload, dict)
            else None
        )
        raise HnlCodecError(
            f"Opaque term (tag {tag!r}) has no mlopt term spelling"
        )
    return _dec(data)
