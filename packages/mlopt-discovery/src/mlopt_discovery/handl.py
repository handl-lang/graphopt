"""Handl-fragment schema bridge — mlopt cells as language data.

Phase 3a of the cell-store plan: emit mlopt's assembled
:class:`~mlopt_core.cells.CellStore` in the *shape* of a handl
language fragment — the ``ast.yaml`` / ``transforms.yaml`` data a
``merge_fragments`` call assembles into ``lang.ast`` /
``lang.transforms`` — so both kernels can read the same vocabulary.

The mapping is honest about which layer carries what:

- ``role="op"``     → ``AstShapeDef`` (``node_kind`` + child roles +
                      ``payload_schema``); the op's shape rule rides
                      in ``mlopt.opdef`` — handl has no shape DSL yet.
- ``role="law"``/``"object"`` → ``TransformDef``: the lhs pattern's
                      root kind becomes ``node_kind``, inner op
                      positions become ``when: !KindAt`` checks,
                      metavars bind to child *paths* — first
                      occurrence pins the path, repeats re-use it —
                      and the rhs rebuilds as ``!Declarative``
                      ``!Create``/``!KeepPath`` children.  Guards
                      (``cond``/``dspec``) and ``check``/``derive``
                      hooks live under ``mlopt.law`` verbatim —
                      structural data maps, semantic data survives.
- a **root-eliminating** law (rhs a bare metavar, e.g.
                      ``double_neg: neg(neg(a)) → a``) maps to
                      handl's ``!Replace { path }`` rule form (ADR-0127):
                      the rewritten node *is* the existing descendant at
                      ``path``, so no node is created and the law is
                      expressed faithfully.  (Before ADR-0127 no faithful
                      form existed — ``TransformRule::Declarative`` always
                      *creates* a node — and the bridge emitted an explicit
                      ``!Native`` refusal; that is now only the fallback for
                      a bare-metavar rhs with *no* bound path, see
                      :data:`_ELIMINATE_ROOT_NATIVE`.)
- ``role="handle"`` → a ``!Native``-style binding record
                      (``mlopt:<kernel>``) — handl's own resolver
                      seam; the mlopt kernel name is the tag.

Every emitted node carries a ``mlopt`` extension field with the
cell's own serialized body — the bridge is *lossless by
construction*: the handl-shaped fields are a structural view, the
``mlopt`` field is authoritative, and
:func:`cell_from_handl` reconstructs the cell from it.

The schema here is a *data* contract, not a runtime dependency —
nothing imports handl; the output is plain YAML-able dicts.
"""

from typing import Any

from mlopt_core.cells import Cell, CellStore
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op, Var
from mlopt_core.laws.serialize import law_to_data, object_to_data
from mlopt_core.opdata import opdef_to_data

__all__ = [
    "cell_to_handl",
    "handle_to_handl",
    "law_to_handl",
    "opdef_to_handl",
    "store_to_handl",
]


def _op_parts(term: Any) -> tuple | None:
    """Split a pattern node into ``(kind, arg-tuple)`` — Op or spec.

    Law patterns are ``Op`` trees; handler patterns are spec
    tuples ``("matmul", …)`` — both carry the same head+args shape;
    ``None`` on leaves.
    """
    if isinstance(term, Op):
        return term.op, tuple(term.args)
    if (
        isinstance(term, (tuple, list))
        and term
        and isinstance(term[0], str)
    ):
        return term[0], tuple(
            e for e in term[1:] if not isinstance(e, dict)
        )
    return None


def _metavar_paths(term: Any, paths: dict, at: tuple = ()) -> None:
    """Collect metavariable leaf positions inside a pattern term.

    A metavar leaf is a bare ``str``; its *path* is the tuple of
    child indices from the pattern root — the position a handl
    ``!KeepPath`` steps reproduces.  First occurrence wins.
    """
    if isinstance(term, str):
        paths.setdefault(term, at)
        return
    parts = _op_parts(term)
    if parts:
        for i, a in enumerate(parts[1]):
            _metavar_paths(a, paths, (*at, i))


def _when_clauses(term: Any) -> list:
    """Emit ``KindAt``/``ChildCount`` conditions at depth one.

    Each op child of the pattern root becomes a ``!KindAt``; the
    root's arity is its ``!ChildCount``.  Deeper positions exceed
    what handl's single-index ``KindAt`` vocabulary can express —
    the full lhs survives in ``mlopt.law`` regardless.
    """
    parts = _op_parts(term)
    if parts is None:
        return []
    out: list = []
    for i, a in enumerate(parts[1]):
        sub = _op_parts(a)
        if sub:
            out.append({"KindAt": {"index": i, "kind": sub[0]}})
    out.append({"ChildCount": len(parts[1])})
    return out


def _when_condition(term: Any) -> dict | None:
    """Fold the clause list into handl's single ``TransformCondition``.

    handl's ``TransformDef.when`` is ``Option<TransformCondition>`` — one
    condition — and a conjunction is spelled ``And([...])``.  So the
    clauses ``_when_clauses`` produces fold here: ``None`` when there are
    none, the clause itself when there is one, ``{"And": [...]}``
    otherwise.  Emitting the bare list (the earlier shape) does not
    deserialize.
    """
    clauses = _when_clauses(term)
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"And": clauses}


#: Node kinds the bridge mints for right-hand-side leaves.  handl's
#: ``RewriteStep`` has no literal-leaf step, so a leaf becomes a ``Create``
#: of a mlopt-namespaced kind carrying its data in the payload; the shape
#: rows below declare them, so the fragment loads.
_CONST_KIND = "mlopt:const"
_VAR_KIND = "mlopt:var"
_DATA_KIND = "mlopt:data"

#: The ``Native`` rule name the bridge emits as a **fallback refusal** for a
#: root-eliminating law whose rhs is a bare metavar with *no* bound path —
#: i.e. a metavar that does not occur in the lhs (no shipped law has one).
#: A root-eliminating law whose rhs metavar *is* bound in the lhs — the 5 in
#: the shipped bank (``id_add``, ``id_mul``, ``double_neg``,
#: ``transpose_noop``, ``chunk_single``) — now maps faithfully to handl's
#: ``!Replace { path }`` rule form (handl ADR-0127), so it no longer lands
#: here.  ``TransformRule::Declarative`` always *creates* a node
#: (``target_kind: String``), so before ``Replace`` existed such a law had no
#: faithful form at all.  Emitting ``target_kind: None`` is not a valid
#: refusal either: serde_yaml coerces the YAML ``null`` scalar into the
#: *string* ``"null"``, so handl's engine silently fabricated a malformed
#: node of kind ``"null"`` (measured — the bridge used to do exactly this).
#: For the residual case the bridge therefore refuses *explicitly* with a
#: ``!Native`` rule naming the missing operation: handl's engine rejects it
#: loudly ("requires native rewrite … but no native registry is wired")
#: rather than fabricating a node.  The law's meaning still round-trips
#: verbatim through ``mlopt.law``.
#: Plan 0030 step 3 / ``project/retros/handl-differential-rewrite-check.md``.
_ELIMINATE_ROOT_NATIVE = "mlopt:eliminate_root"

#: The ``AstShapeDef`` rows declaring those kinds (emitted with the ops).
_LEAF_SHAPES = (
    {
        "id": "ast.mlopt.const",
        "node_kind": _CONST_KIND,
        "children": [],
        "payload_schema": {
            "type": "object",
            "properties": {"value": {}},
        },
        "mlopt": {"leaf": "const"},
    },
    {
        "id": "ast.mlopt.var",
        "node_kind": _VAR_KIND,
        "children": [],
        "payload_schema": {
            "type": "object",
            "properties": {"name": {}},
        },
        "mlopt": {"leaf": "var"},
    },
    {
        "id": "ast.mlopt.data",
        "node_kind": _DATA_KIND,
        "children": [],
        "payload_schema": {
            "type": "object",
            "properties": {"repr": {}},
        },
        "mlopt": {"leaf": "data"},
    },
)


def _leaf_step(kind: str, payload: dict) -> dict:
    """Render a leaf as a ``Create`` of a declared mlopt-namespaced kind."""
    return {
        "Create": {"kind": kind, "payload": payload, "children": []}
    }


def _children_spec(term: Any, paths: dict) -> dict:
    """Render a rhs pattern as handl ``children`` steps.

    Metavars → ``!KeepPath`` (the bound position from the lhs); op nodes
    → ``!Create`` (new structure); a **leaf** → a step handl's
    ``RewriteStep`` vocabulary actually has:

    * an unbound metavar → ``!Fresh`` (allocate a fresh variable — what
      an unbound pattern variable on the right-hand side means);
    * a ``Const`` → ``!Create`` of kind ``mlopt:const`` with the value in
      the node's ``payload``;
    * a ``Var`` → ``!Create`` of kind ``mlopt:var`` with the name in the
      payload.

    The two ``mlopt:``-namespaced kinds are declared in the fragment's
    ``ast`` rows, so handl can resolve them.  (An earlier version emitted
    ``{"mlopt:leaf": …}`` — an *extension*, not a step — which handl's
    serde rejects with *"expected a YAML tag"*: the full bank failed to
    load because of it.)
    """
    if isinstance(term, str):
        p = paths.get(term)
        return (
            {"KeepPath": list(p)} if p is not None else {"Fresh": term}
        )
    parts = _op_parts(term)
    if parts:
        return {
            "Create": {
                "kind": parts[0],
                "children": [
                    _children_spec(a, paths) for a in parts[1]
                ],
            }
        }
    if isinstance(term, Const):
        return _leaf_step(_CONST_KIND, {"value": term.value})
    if isinstance(term, Var):
        return _leaf_step(_VAR_KIND, {"name": term.name})
    return _leaf_step(
        _DATA_KIND, {"repr": repr(term)}
    )  # pragma: no cover


def _rule_body(rule: Rewrite, paths: dict) -> dict:
    """Render a law's ``TransformRule`` — ``!Declarative`` or ``!Replace``.

    An rhs with an **op root** maps to ``!Declarative``: create
    ``target_kind`` (the root op) from the child steps.  An rhs that is a
    **bare metavar bound in the lhs** — a *root-eliminating* law returning a
    subterm — maps to ``!Replace { path }`` (handl ADR-0127): the rewritten
    node *is* the existing descendant at the metavar's bound position, so
    ``double_neg: neg(neg(a)) → a`` becomes ``!Replace { path: [0, 0] }``.
    That is the faithful form — handl's ``!Declarative`` always *creates* a
    node, so it cannot express a law that drops its root.

    A bare-metavar rhs with **no non-empty** bound path — a metavar absent
    from the lhs, or bound only at the lhs root (an empty path, which would
    name the matched node itself) — has no descendant to name, so it falls
    back to the explicit ``!Native`` refusal; see
    :data:`_ELIMINATE_ROOT_NATIVE`.
    """
    parts = _op_parts(rule.rhs)
    if parts is None:
        if isinstance(rule.rhs, str) and paths.get(rule.rhs):
            return {"Replace": {"path": list(paths[rule.rhs])}}
        return {"Native": _ELIMINATE_ROOT_NATIVE}
    return {
        "Declarative": {
            "target_kind": parts[0],
            "children": [_children_spec(a, paths) for a in parts[1]],
        }
    }


def law_to_handl(rule: Rewrite, name: str) -> dict:
    """Map a law/object cell body to a ``TransformDef`` fragment row.

    The structural fields (``node_kind``/``when``/``rule``) are the
    handl-native view — a rewrite over the lhs's root kind whose
    children rebuild the rhs via bound paths.  ``mlopt.law`` keeps
    the full serialized law — the authoritative record — so guards,
    derivations and provenance round-trip verbatim.
    """
    paths: dict = {}
    _metavar_paths(rule.lhs, paths)
    lhs_parts = _op_parts(rule.lhs)
    row: dict = {
        "id": f"transform.mlopt.{name}",
        "pass": "mlopt",
        "node_kind": lhs_parts[0] if lhs_parts else None,
        "rule": _rule_body(rule, paths),
        "mlopt": {
            "law": law_to_data(rule),
            "metavars": {m: list(p) for m, p in paths.items()},
        },
    }
    # ``when`` is a single ``TransformCondition`` in handl's schema — a
    # conjunction is spelled ``And([...])`` — so the clause list folds
    # (and the field is omitted entirely when there are no clauses).
    condition = _when_condition(rule.lhs)
    if condition is not None:
        row["when"] = condition
    return row


def opdef_to_handl(d: Any) -> dict:
    """Map an op cell body to an ``AstShapeDef`` fragment row."""
    children = [{"role": f"arg{i}"} for i in range(d.arity or 0)]
    return {
        "id": f"ast.mlopt.{d.name}",
        "node_kind": d.name,
        "children": children,
        "payload_schema": {
            "type": "object",
            "properties": {
                "attrs": {"type": "object"},
                "children": {
                    "type": "array",
                    "minItems": d.arity or 0,
                    "maxItems": d.arity or 0,
                },
            },
            "required": ["children"],
        },
        "mlopt": {"opdef": opdef_to_data(d)},
    }


def handle_to_handl(name: str, h: dict) -> dict:
    """Map a handle cell to handl's ``!Native`` binding record.

    The ``native`` tag spells the resolver seam — ``mlopt:<kernel>``
    names the generated/shipped kernel the same way handl spells a
    native resolver; ``pattern``/``args`` keep the binding spec.
    """
    return {
        "id": f"native.mlopt.{name}",
        "native": f"mlopt:{h.get('kernel', name)}",
        "pattern": h.get("pattern"),
        "args": list(h.get("args", ())),
        "mlopt": {"handler": dict(h)},
    }


def object_to_handl(cell: Cell) -> dict:
    """Map an object cell — a ``TransformDef`` with object provenance."""
    row = law_to_handl(cell.body, cell.name)
    row["mlopt"]["object"] = object_to_data(cell.body, kind=cell.kind)
    return row


def cell_to_handl(cell: Cell) -> dict:
    """Map one cell to its handl-fragment row (any role)."""
    if cell.role == "op":
        return opdef_to_handl(cell.body)
    if cell.role == "object":
        return object_to_handl(cell)
    if cell.role == "law":
        return law_to_handl(cell.body, cell.name)
    if cell.role == "handle":
        return handle_to_handl(cell.name, cell.body)
    raise ValueError(  # pragma: no cover — role is validated
        f"no handl mapping for cell role {cell.role!r}"
    )


def store_to_handl(store: CellStore) -> dict:
    """Assemble a store into handl's fragment *shape*.

    Returns the mergeable fragment: ``ast`` rows (op cells, plus the
    leaf kinds the bridge mints — see :data:`_LEAF_SHAPES`),
    ``transforms`` rows (law + object cells), ``natives`` rows
    (handle cells) — plus a ``mlopt`` provenance block listing the
    fragment sources the store was merged from.
    """
    ast: list = list(_LEAF_SHAPES)
    transforms: list = []
    natives: list = []
    for c in store:
        row = cell_to_handl(c)
        if c.role == "op":
            ast.append(row)
        elif c.role in ("law", "object"):
            transforms.append(row)
        elif c.role == "handle":  # pragma: no branch — the four
            natives.append(row)  # roles exhaust the vocabulary
    # handl keys a rule by `pass:node_kind:priority`, so two laws with the
    # same root kind collide unless they carry distinct priorities.  The
    # value here is a *load-time disambiguator*, not a semantic ordering:
    # mlopt's saturation is unordered, so only handl's per-kind
    # application order is affected (and that order is not part of the
    # law's meaning).
    _disambiguate_priorities(transforms)
    return {
        "ast": ast,
        "transforms": transforms,
        "natives": natives,
        "mlopt": {
            "cells": len(list(store)),
            "provenance": sorted({c.provenance.origin for c in store}),
        },
    }


def _disambiguate_priorities(rows: list) -> None:
    """Give same-kind transform rows distinct ``priority`` values.

    handl's identity key is ``pass:node_kind:priority``; a bank with
    several laws per node kind (mlopt has many) collides without this.
    """
    seen: dict = {}
    for row in rows:
        key = (row.get("pass"), row.get("node_kind"))
        seen[key] = seen.get(key, 0) + 1
        row["priority"] = seen[key]


def cell_from_handl(row: dict) -> Cell | None:
    """Reconstruct a cell from a handl row's ``mlopt`` extension.

    The inverse of :func:`cell_to_handl` where the extension data is
    present — ``None`` on a row without mlopt data (a pure-handl
    node the bridge did not emit).
    """
    from mlopt_core.cells import Provenance
    from mlopt_core.laws.serialize import law_from_data
    from mlopt_core.opdata import opdef_from_data

    ext = row.get("mlopt") or {}
    if "law" in ext:
        return Cell(
            role="law",
            name=row["id"].rsplit(".", 1)[-1],
            body=law_from_data(ext["law"]),
            provenance=Provenance(origin="imported"),
        )
    if "opdef" in ext:
        d = opdef_from_data(ext["opdef"])
        return Cell(
            role="op",
            name=d.name,
            body=d,
            provenance=Provenance(origin="imported"),
        )
    if "handler" in ext:
        return Cell(
            role="handle",
            name=row["id"].rsplit(".", 1)[-1],
            body=ext["handler"],
            provenance=Provenance(origin="imported"),
        )
    return None
