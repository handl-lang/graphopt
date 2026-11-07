"""The canonical fragment vocabulary — plan 0033 phase 0.

One fragment directory (``vocab/`` at the repo root) is the single
source of truth for the discovery vocabulary.  Today the same facts
live as Python objects in five places — the ``opmeta``/``ir`` op
registries, the ``law_to_data`` law bank, ``pipeline._FUSION_SCHEMAS``,
the ``handler_synth`` grammar family tables and
``shipped_primitives``/``mlopt_core.laws.minted``'s embedded prim
records — plus the handler tables (``lawdata.HANDLERS``,
``shipped_primitives.handler_entries()``, the minted handler records
embedded in the prim records).

The directory carries two file classes:

* **canonical** — ``ops.yaml`` / ``laws.yaml`` / ``handlers.yaml`` /
  ``schemas.yaml`` / ``prims.yaml`` / ``families.yaml`` /
  ``models.yaml``.  Each is a
  JSON document (JSON is a YAML subset — handl's ``serde_yaml`` and
  Python's ``json`` both read it) pinned by a ``schema:`` version
  field.  These are *data*: op rows, ``law_to_data`` records,
  handler entries, schema records, verbatim prim records, the
  family tables, and the phase-2 Models (denotations + minted
  handlers projected to ``.hnl`` over ``host-tensor`` —
  :mod:`mlopt_discovery.modelproj`; ``models.yaml``'s ``verdicts``
  map is the fp64 gate's provenance record, written by
  ``scripts/verify_vocab_models.py``).  :func:`scripts/emit_vocab_fragments.py` (the
  ``emit_vocab_fragments`` entry point :func:`emit_all` backs)
  regenerates them from their current owners; the upsert API below
  (:func:`upsert_op` / :func:`upsert_law` / :func:`upsert_handler` /
  :func:`upsert_schema` / :func:`emit_prim`) is the mint-side seam —
  a minted op or handler lands as a fragment row, not a Python-side
  registry entry (workstream 0b).
* **projections** — the handl-facing files ``language_from_dir``
  reads: the seven empty shell files plus ``ast.yaml`` (op shapes)
  and ``transforms.yaml`` (law rules).  They are *generated* from the
  canonical files by :func:`sync_projections` — never hand-edited —
  so ``mlopt-handl-peer``'s ``transform_ids(dir)`` and
  ``pipeline.propose``'s ``vocab_dir=`` consume the same directory.
  Handl's ``AstShapeDef``/``TransformDef`` deserializers ignore the
  ``mlopt`` extension fields, which carry the authoritative mlopt
  record (the same lossless-by-construction bridge
  :mod:`mlopt_discovery.handl` established).

Loading
-------

:func:`load_vocab` reads the seven canonical files back into objects —
``laws``/``schemas`` as :class:`~mlopt_core.egraph.Rewrite` via
``law_from_data``, handlers as spec-decoded entries, prims verbatim,
ops/families as data.  Records whose procedural ``check``/``derive``
hooks cannot serialize (``missing_hooks`` non-empty — today only
``conv_bn_fold``) re-bind those hooks *by name* from the shipped
``ALL_RULES`` bank: the record still pins pattern/cond/dspec from the
 dir, and :attr:`Vocab.hooked` names every law whose hooks came from
code rather than data — the same honesty ``missing_hooks`` reports.

Consumers
---------

* ``pipeline.propose(..., vocab_dir=...)`` — ``None`` auto-resolves
  :func:`default_vocab_dir` (the repo ``vocab/`` when present, an
  ``MLOPT_VOCAB_DIR`` override honoured first); ``False`` forces the
  historical Python tables.  The fusion-schema arm reads
  ``schemas.yaml`` + the fold/fuse subset of ``laws.yaml``.
* ``mlopt-handl-peer`` — ``transform_ids(vocab/)`` /
  ``apply_rule(vocab/, ...)`` read the projections unchanged.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from mlopt_core.cells import spec_from_data, spec_to_data
from mlopt_core.egraph import Rewrite
from mlopt_core.laws.serialize import law_from_data, law_to_data

__all__ = [
    "CANONICAL_FILES",
    "PROJECTION_FILES",
    "SCHEMA_VERSION",
    "SHELL_FILES",
    "Vocab",
    "check_vocab",
    "default_vocab_dir",
    "emit_all",
    "emit_atom",
    "emit_family_rows",
    "emit_prim",
    "emit_prims",
    "emit_yaml_rows",
    "invalidate_vocab",
    "is_fusion_law_name",
    "load_vocab",
    "sync_projections",
    "upsert_family",
    "upsert_handler",
    "upsert_law",
    "upsert_model",
    "upsert_model_verdicts",
    "upsert_op",
    "upsert_refusal",
    "upsert_schema",
    "upsert_schema_premise",
    "vocab_extra_families",
]

#: The fragment-schema pin every canonical file carries.  A bump is a
#: breaking change for both engines — it must ride the ``uv.lock``
#: pin loudly (plan 0033 hard gates).
SCHEMA_VERSION = "mlopt-vocab/1"

#: The seven canonical files, in emit order.  ``models.yaml`` is the
#: phase-2 addition (plan 0033): denotations and minted handlers
#: projected to ``.hnl`` Models over ``host-tensor`` — see
#: :mod:`mlopt_discovery.modelproj`.
CANONICAL_FILES: tuple[str, ...] = (
    "ops.yaml",
    "laws.yaml",
    "handlers.yaml",
    "schemas.yaml",
    "prims.yaml",
    "families.yaml",
    "models.yaml",
)

#: Runtime-minted op names — ``prim_mint``'s ``{prefix}_{sha256[:10]}``
#: (``{prefix}{d}_`` for deeper tiers) and ``antiunify.mint_atoms``'
#: ``{prefix}_{i}``.  ``collect_ops`` treats matching ``_OP_REGISTRY``
#: entries as machine-added, never owner rows (see its docstring).
_MINTED_OP_NAME = re.compile(r"^.+_(?:[0-9a-f]{10}|\d+)$")

#: The handl-facing projection files — the seven ``language_from_dir``
#: requires (empty ``[]`` stubs: mlopt's syntax is the IR, not text)
#: plus the two it reads content from.
SHELL_FILES: tuple[str, ...] = (
    "tokens.yaml",
    "grammar.yaml",
    "eval.yaml",
    "types.yaml",
    "caps.yaml",
    "options.yaml",
    "lexer_rules.yaml",
)
PROJECTION_FILES: tuple[str, ...] = (
    *SHELL_FILES,
    "ast.yaml",
    "transforms.yaml",
)


# ---------------------------------------------------------------------------
#  YAML emission — enum variants as tags (shared with dump_fragments)
# ---------------------------------------------------------------------------
#
# JSON is valid YAML, but serde_yaml spells enum variants as YAML *tags*
# (``!Declarative``, ``!Create``, ``!KeepPath``…) where JSON spells them
# as maps — ``language_from_dir`` reads YAML, so the projection files
# are written with tags.  This is the emitter ``scripts/dump_fragments.py``
# grew; it lives here so the vocab projections and the spike fixtures
# share one code path.

#: Enum variants handl spells as YAML tags.  A single-key mapping whose
#: key is one of these is a variant, not a field.
_TAGS = frozenset(
    {
        "Declarative",
        "Replace",
        "Native",
        "Keep",
        "KeepRole",
        "KeepPath",
        "KeepRest",
        "Splice",
        "Create",
        "Apply",
        "Fresh",
        "KindAt",
        "AnyChildKind",
        "PayloadEq",
        "ChildPayloadEq",
        "ChildCount",
        "And",
        "Or",
        "Not",
    }
)


def _scalar(value: object) -> str:
    """Render a scalar as YAML (JSON scalars are valid YAML)."""
    return json.dumps(value)


def _is_scalar(value: object) -> bool:
    """Return whether *value* renders inline as a YAML scalar."""
    return isinstance(value, (str, int, float)) or value is None


def _emit_tag(key: str, value: object, indent: int) -> str:
    """Render one ``!Variant`` tag node."""
    pad = " " * indent
    # a tag on a scalar/flow value reads better inline
    if _is_scalar(value):
        return f"{pad}!{key} {_scalar(value)}"
    return f"{pad}!{key}\n{_emit(value, indent)}"


def _emit_dict_items(node: dict, indent: int) -> list[str]:
    """Render a mapping's ``key:`` lines (recursing on composite values)."""
    pad = " " * indent
    out = []
    for key, value in node.items():
        rendered = _emit(value, indent + 2)
        if _is_scalar(value):
            out.append(f"{pad}{key}: {_scalar(value)}")
        elif isinstance(value, dict) and not value:
            out.append(f"{pad}{key}: {{}}")
        elif isinstance(value, list) and not value:
            out.append(f"{pad}{key}: []")
        else:
            out.append(f"{pad}{key}:")
            out.append(rendered)
    return out


def _emit_list_items(node: list, indent: int) -> list[str]:
    """Render a sequence's ``- item`` lines."""
    pad = " " * indent
    out = []
    for item in node:
        rendered = _emit(item, indent + 2)
        out.append(f"{pad}- {rendered[indent + 2 :]}")
    return out


def _emit(node: object, indent: int) -> str:
    """Render *node* as YAML, spelling enum variants as tags."""
    pad = " " * indent
    if isinstance(node, dict):
        if len(node) == 1:
            ((key, value),) = node.items()
            if key in _TAGS:
                return _emit_tag(key, value, indent)
        if not node:
            return f"{pad}{{}}"
        return "\n".join(_emit_dict_items(node, indent))
    if isinstance(node, list):
        if not node:
            return f"{pad}[]"
        return "\n".join(_emit_list_items(node, indent))
    return f"{pad}{_scalar(node)}"


def emit_yaml_rows(path: str | Path, rows: list) -> None:
    """Write *rows* as one YAML sequence (the fragment file's shape)."""
    Path(path).write_text(_emit(rows, 0) + "\n")


# ---------------------------------------------------------------------------
#  Canonical file IO
# ---------------------------------------------------------------------------


def _read_canonical(directory: Path, name: str) -> dict:
    """Read one canonical file; refuse a foreign schema loudly."""
    doc = json.loads((directory / name).read_text())
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{directory / name}: bad vocab schema pin "
            f"{doc.get('schema') if isinstance(doc, dict) else doc!r} "
            f"(want {SCHEMA_VERSION!r})"
        )
    return doc


def _write_canonical(directory: Path, name: str, doc: dict) -> None:
    body = {k: v for k, v in doc.items() if k != "schema"}
    (directory / name).write_text(
        json.dumps({"schema": SCHEMA_VERSION, **body}, indent=2) + "\n"
    )


def _canonical(directory: Path) -> dict[str, dict]:
    """Read all seven canonical files (schema-checked)."""
    return {
        name: _read_canonical(directory, name)
        for name in CANONICAL_FILES
    }


def _section(doc: dict, key: str) -> list:
    """Return a canonical doc's row list, creating it when absent."""
    return doc.setdefault(key, [])


def _upsert(rows: list, key: str, row: dict) -> None:
    """Replace the row whose *key* field equals ``row[key]`` in place."""
    want = row[key]
    for i, cur in enumerate(rows):
        if cur.get(key) == want:
            rows[i] = row
            return
    rows.append(row)


def _signature(directory: Path) -> tuple:
    """Cheap freshness signature — the six files' (mtime, size) pairs."""
    out = []
    for name in CANONICAL_FILES:
        try:
            st = (directory / name).stat()
        except OSError:
            out.append((name, None, None))
        else:
            out.append((name, st.st_mtime_ns, st.st_size))
    return tuple(out)


# ---------------------------------------------------------------------------
#  Loading — the Python-side projection of the directory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vocab:
    """The loaded fragment vocabulary — projections, not sources.

    ``laws``/``schemas`` are :class:`Rewrite` objects rebuilt through
    ``law_from_data``; ``ops``/``prims``/``families`` stay data (the
    consumers read fields, not behaviour); ``handlers`` are the
    spec-decoded interpretation entries.
    """

    directory: Path
    ops: tuple[dict, ...] = ()
    laws: tuple[Rewrite, ...] = ()
    schemas: tuple[Rewrite, ...] = ()
    fusion_premises: dict[str, tuple[Rewrite, ...]] = field(
        default_factory=dict
    )
    handlers: tuple[dict, ...] = ()
    prim_handlers: tuple[dict, ...] = ()
    minted_handlers: tuple[dict, ...] = ()
    prims: dict[str, Any] = field(default_factory=dict)
    families: dict[str, Any] = field(default_factory=dict)
    #: ``models.yaml`` rows — denotation/handler Models projected to
    #: ``.hnl`` over ``host-tensor`` (phase 2).
    models: tuple[dict, ...] = ()
    #: ``models.yaml``'s ``verdicts`` map — the fp64 gate's provenance
    #: record (``handl-shipped`` / ``python-shipped`` per model id),
    #: written by ``scripts/verify_vocab_models.py`` and preserved
    #: across re-emission.
    model_verdicts: dict[str, Any] = field(default_factory=dict)
    #: Law names whose procedural ``check``/``derive`` hooks re-bound
    #: from the live ``ALL_RULES`` bank (the record's
    #: ``missing_hooks`` was non-empty — pattern/cond/dspec still come
    #: from the directory).
    hooked: tuple[str, ...] = ()

    def op_names(self) -> frozenset[str]:
        """Return every declared op name."""
        return frozenset(r["name"] for r in self.ops)

    def op_row(self, name: str) -> dict | None:
        """Return the op row for *name*, or ``None``."""
        for r in self.ops:
            if r["name"] == name:
                return r
        return None

    def laws_by_name(self) -> dict[str, Rewrite]:
        """Index the law bank by rule name."""
        return {r.name: r for r in self.laws}

    def fusion_laws(self) -> tuple[Rewrite, ...]:
        """Return the fold/fuse subset of the bank — ``pipeline``'s schema arm."""
        return tuple(r for r in self.laws if is_fusion_law_name(r.name))

    def handler_table(self) -> dict[str, dict]:
        """Merge the interpretation entries into one tag→entry table.

        The shape ``lawdata.HANDLERS`` / ``genkernel.gen_handlers``
        consumers read: ``{pattern, kernel, args}`` plus
        ``attr_mvars`` where the entry declares them.  Patterns are
        spec tuples again (``spec_from_data`` decode).
        """
        out: dict[str, dict] = {}
        for h in (*self.handlers, *self.prim_handlers):
            tag = cast(str, h.get("tag") or h.get("op"))
            out[tag] = {
                "pattern": h["pattern"],
                "kernel": h["kernel"],
                "args": h["args"],
                **(
                    {"attr_mvars": h["attr_mvars"]}
                    if "attr_mvars" in h
                    else {}
                ),
            }
        return out


def is_fusion_law_name(name: str) -> bool:
    """Return whether *name* is a shipped fold/fuse law.

    The same test ``pipeline._fusion_laws`` applies, factored here so
    the vocab projection and the live bank agree on membership by
    construction.
    """
    return (
        name.endswith(("_fold", "_fuse"))
        or "_fold_" in name
        or name in ("qkv_fuse_asym", "gqa_absorb_repeat")
    )


def _faithful_rule(record: dict) -> tuple[Rewrite, bool]:
    """Rebuild a law record; re-bind procedural hooks by name when needed.

    Returns ``(rule, hooked)`` — *hooked* is True when the record's
    ``missing_hooks`` was non-empty and the same-named shipped rule
    supplied its ``check``/``derive`` (code cannot live in data; the
    record's pattern/cond/dspec still came from the directory, and
    the name is reported through :attr:`Vocab.hooked`).
    """
    rule = law_from_data(record)
    if not record.get("missing_hooks"):
        return rule, False
    from mlopt_core.laws import ALL_RULES

    for shipped in ALL_RULES:
        if shipped.name == record.get("name"):
            return shipped, True
    return rule, False  # no live counterpart — data-only, honestly


def _decode_handlers(doc: dict) -> tuple[tuple, tuple, tuple]:
    """Decode ``handlers.yaml``'s three sections."""
    handlers = tuple(
        {
            "tag": h["tag"],
            "source": h.get("source", ""),
            "pattern": spec_from_data(h["pattern"]),
            "kernel": h["kernel"],
            "args": tuple(h.get("args", ())),
        }
        for h in doc.get("handlers", ())
    )
    prim = tuple(
        {
            "op": h["op"],
            "source": h.get("source", "prim"),
            "pattern": spec_from_data(h["pattern"]),
            "kernel": h["kernel"],
            "args": tuple(h.get("args", ())),
            **(
                {"attr_mvars": tuple(h["attr_mvars"])}
                if "attr_mvars" in h
                else {}
            ),
        }
        for h in doc.get("prim", ())
    )
    minted = tuple(doc.get("minted", ()))
    return handlers, prim, minted


def _load_uncached(directory: Path) -> Vocab:
    docs = _canonical(directory)

    laws: list[Rewrite] = []
    hooked: list[str] = []
    for record in docs["laws.yaml"].get("laws", ()):
        rule, was_hooked = _faithful_rule(record)
        laws.append(rule)
        if was_hooked:
            hooked.append(rule.name)

    schemas = tuple(
        _faithful_rule(r)[0]
        for r in docs["schemas.yaml"].get("schemas", ())
    )
    premises = {
        name: tuple(_faithful_rule(r)[0] for r in records)
        for name, records in docs["schemas.yaml"]
        .get("premises", {})
        .items()
    }
    handlers, prim_handlers, minted = _decode_handlers(
        docs["handlers.yaml"]
    )
    return Vocab(
        directory=directory,
        ops=tuple(docs["ops.yaml"].get("ops", ())),
        laws=tuple(laws),
        schemas=schemas,
        fusion_premises=premises,
        handlers=handlers,
        prim_handlers=prim_handlers,
        minted_handlers=minted,
        prims=dict(docs["prims.yaml"].get("prims", {})),
        families=dict(docs["families.yaml"].get("families", {})),
        models=tuple(docs["models.yaml"].get("models", ())),
        model_verdicts=dict(docs["models.yaml"].get("verdicts", {})),
        hooked=tuple(hooked),
    )


@lru_cache(maxsize=16)
def _cached_load(key: str, signature: tuple) -> Vocab:
    """Return the cached :func:`_load_uncached`, keyed on path + signature."""
    return _load_uncached(Path(key))


def load_vocab(directory: str | Path, *, fresh: bool = False) -> Vocab:
    """Load the canonical fragment vocabulary at *directory*.

    Refuses a missing file or a foreign ``schema:`` pin loudly — a
    silent schema drift is a release blocker (plan 0033 hard gates).
    Results are cached on the files' ``(mtime, size)`` signature;
    pass ``fresh=True`` or call :func:`invalidate_vocab` after
    writing (the upsert API does this for you).
    """
    directory = Path(directory)
    if fresh:
        _cached_load.cache_clear()
        return _load_uncached(directory)
    return _cached_load(str(directory), _signature(directory))


def invalidate_vocab(directory: str | Path) -> None:
    """Drop the cached load for *directory* (the upserts call this)."""
    _cached_load.cache_clear()


def default_vocab_dir() -> Path | None:
    """Return the ambient fragment directory, or ``None``.

    ``MLOPT_VOCAB_DIR`` wins when set; otherwise the nearest ancestor
    of this module carrying ``vocab/ops.yaml`` — the repo checkout in
    development, absent in an installed wheel (consumers then fall
    back to the Python tables, the pre-phase-0 behaviour).
    """
    env = os.environ.get("MLOPT_VOCAB_DIR")
    if env:
        cand = Path(env)
        return cand if (cand / "ops.yaml").is_file() else None
    for parent in Path(__file__).resolve().parents:
        cand = parent / "vocab"
        if (cand / "ops.yaml").is_file() and (
            cand / "laws.yaml"
        ).is_file():
            return cand
    return None


# ---------------------------------------------------------------------------
#  Projections — the handl-facing files, generated never edited
# ---------------------------------------------------------------------------


def _ast_shape_row(op: dict) -> dict:
    """Render an op row as an ``AstShapeDef`` fragment row.

    Same shape ``handl.opdef_to_handl`` emits for declared ops —
    ``arg{i}`` child roles, the attrs/children payload schema — with
    the full vocab row riding in ``mlopt.op`` (handl ignores the
    extension; the Python side reads it back from ``ops.yaml``).
    """
    arity = op.get("arity") or 0
    return {
        "id": f"ast.mlopt.{op['name']}",
        "node_kind": op["name"],
        "children": [{"role": f"arg{i}"} for i in range(arity)],
        "payload_schema": {
            "type": "object",
            "properties": {
                "attrs": {"type": "object"},
                "children": {
                    "type": "array",
                    "minItems": arity,
                    "maxItems": arity,
                },
            },
            "required": ["children"],
        },
        "mlopt": {"op": op},
    }


def _projection_rows(directory: Path) -> dict[str, list]:
    """Compute the projection rows from the canonical files (no write)."""
    from mlopt_discovery.handl import (
        _LEAF_SHAPES,
        _disambiguate_priorities,
        law_to_handl,
    )

    vocab = load_vocab(directory, fresh=True)
    # ``arity: None`` ops (variadic — ``linear``, ``batch_norm`` …)
    # cannot declare a fixed child-role list, and handl's transform
    # validator rejects ``KeepPath``/``KindAt`` indices into a
    # zero-role shape.  The honest projection omits their
    # ``AstShapeDef`` — the same posture the spike fixtures took
    # (they declare no op shapes at all); the op itself still rides
    # in ``ops.yaml`` and the ``mlopt.op`` extension of every row
    # that *is* emitted.
    ast_rows = [
        _ast_shape_row(o)
        for o in vocab.ops
        if o.get("arity") is not None
    ]
    ast_rows = list(_LEAF_SHAPES) + ast_rows
    transforms = [law_to_handl(r, r.name) for r in vocab.laws]
    _disambiguate_priorities(transforms)
    return {"ast.yaml": ast_rows, "transforms.yaml": transforms}


def sync_projections(directory: str | Path) -> None:
    """Regenerate ``ast.yaml`` / ``transforms.yaml`` from canonical files.

    ``ast.yaml`` carries the bridge's leaf kinds plus one
    ``AstShapeDef`` per op row whose arity is known (``arity: None``
    variadic ops are omitted — see :func:`_projection_rows`);
    ``transforms.yaml`` carries one
    ``TransformDef`` per law (``pipeline``'s fusion *schemas* stay
    mlopt-side — they are proposal recognizers, not admitted bank
    rules, and handl's engine must never fire them as rewrites).
    The seven shell files are asserted present (written ``[]`` by
    :func:`emit_all`) but never touched here.
    """
    directory = Path(directory)
    for name, rows in _projection_rows(directory).items():
        emit_yaml_rows(directory / name, rows)
    invalidate_vocab(directory)


# ---------------------------------------------------------------------------
#  Emit — the collectors (one per registry owner)
# ---------------------------------------------------------------------------


def _op_denotation(name: str, torch_ops: set, prim_ops: set) -> str:
    """Return what evaluates *name* — prim fold, torch kernel, or unbound."""
    if name in prim_ops:
        return "prim"
    if name in torch_ops:
        return "torch"
    return "unbound"


def _op_row(name: str, meta: Any, gen: Any, denotation: str) -> dict:
    """Build one ``ops.yaml`` row from its ``opmeta``/``GenDef`` halves."""
    from mlopt_core import opmeta

    return {
        "name": name,
        "arity": (
            meta.arity
            if meta is not None
            else (gen.n_in if gen is not None else None)
        ),
        "tags": sorted(meta.tags) if meta is not None else [],
        "attrs": (
            [[int(i), n] for i, n in sorted(opmeta.attrs(name).items())]
            if meta is not None
            else []
        ),
        "required": (
            sorted(opmeta.required(name)) if meta is not None else []
        ),
        "gen": _gen_data(gen),
        "denotation": denotation,
    }


def _gen_data(gen: Any) -> dict | None:
    """Serialize a registered ``GenDef``'s categorical fields."""
    if gen is None:
        return None
    return {
        "n_in": gen.n_in,
        "n_out": gen.n_out,
        "commutative": gen.commutative,
        "associative": gen.associative,
        "identity": gen.identity,
        "law": gen.law,
    }


def collect_ops() -> list[dict]:
    """Emit the op rows — ``opmeta`` registry union ``ir`` GenDefs.

    One row per op name: arity/tags/attr-schema/required from
    ``opmeta``, the categorical ``GenDef`` fields when the op is a
    *shipped* registered generator (the minted ``synop_*`` ops —
    they declare through ``mlopt_core.ir.op_def`` and ship in
    ``mlopt_core.laws.minted.SYNOP_SHIPPED``), and a ``denotation``
    ref naming what evaluates the op.

    Runtime-registered generators — a live mint's fresh ``synop_*``
    or ``defn_*`` op, a test's ``ksynop_*`` — are deliberately NOT
    owner rows: ``_OP_REGISTRY`` is process-global, so counting them
    would make ``check_vocab`` drift on every directory the moment
    any mint ran in the process.  They enter ``ops.yaml`` only as
    machine-added fragments through :func:`emit_prim` /
    :func:`upsert_op` — the same posture ``prims.yaml``'s non-shipped
    records take.
    """
    from mlopt_core import opmeta
    from mlopt_core.ir import _OP_REGISTRY
    from mlopt_core.laws import minted

    # Torch-bound names = the ambient ``_IR_TO_TORCH`` (populated by
    # importing the bridge) *plus* the carrier ``TORCH_BINDINGS``
    # exports — ``OpTable.full()`` seeds those into the ambient dict.
    # Reading either alone made the census order-dependent.
    try:
        import mlopt_torch.torch_bridge as tb
        from mlopt_core.ops import OpTable

        OpTable.full()
        torch_ops = set(tb._IR_TO_TORCH)
    except Exception:  # pragma: no cover — torch always present here
        torch_ops = set()
    # The carrier modules declare their ops via ``op_def`` at import
    # and ``mlopt_carriers`` keeps them lazy — without these imports
    # the op census depends on which tests happened to run first.
    import mlopt_carriers.om
    import mlopt_carriers.trace  # noqa: F401

    prim_ops = set(minted.SYNOP_SHIPPED)

    # Runtime-registered generators — a live mint's fresh ``synop_*``
    # (``{prefix}_{sha256[:10]}``) or definitional atom (``{prefix}_{i}``)
    # — are NOT owner rows: ``_OP_REGISTRY`` is process-global, so
    # counting them would make ``check_vocab`` drift on every
    # directory the moment any mint ran in the process.  They enter
    # ``ops.yaml`` only as machine-added fragments through
    # :func:`emit_prim` / :func:`emit_atom` / :func:`upsert_op`.
    # Carrier/hand ops (``attnbias``, ``trace``, …) keep their rows —
    # no shipped op name ends ``_<hex10>`` or ``_<digits>`` outside
    # the minted shapes (``synop_5147826430`` is shipped, so the
    # ``prim_ops`` check runs first).  ``antiunify.mint_atom`` names
    # are caller-chosen (``defn_0``, a probe's ``test_at0``) — the
    # mint records them in ``antiunify._MINTED_NAMES`` so the census
    # recognizes atoms the name-shape filter cannot.
    try:
        from mlopt_discovery.antiunify import minted_names

        atom_ops = set(minted_names())
    except Exception:  # pragma: no cover — same package, always present
        atom_ops = set()
    gens = {
        n: g
        for n, g in _OP_REGISTRY.items()
        if n in prim_ops
        or (n not in atom_ops and not _MINTED_OP_NAME.match(n))
    }
    return [
        _op_row(
            name,
            opmeta.REGISTRY.get(name),
            gens.get(name),
            _op_denotation(name, torch_ops, prim_ops),
        )
        for name in sorted(set(opmeta.REGISTRY) | set(gens))
    ]


def collect_laws() -> list[dict]:
    """Emit the law bank — ``law_to_data`` rows for ``ALL_RULES``."""
    from mlopt_core.laws import ALL_RULES

    return [law_to_data(r) for r in ALL_RULES]


def collect_schemas() -> dict:
    """Emit the fusion-schema table + premise closures.

    ``pipeline._FUSION_SCHEMAS`` (the measured-but-unshipped
    kernel-fold classes) as ``law_to_data`` rows, plus
    ``_FUSION_PREMISES`` — the declared premise closure each schema
    name carries — so a vocab-loaded schema re-attaches the same
    axioms ``_schema_proposal`` does.
    """
    from mlopt_discovery import pipeline

    return {
        "schemas": [law_to_data(r) for r in pipeline._FUSION_SCHEMAS],
        "premises": {
            name: [law_to_data(p) for p in premises]
            for name, premises in pipeline._FUSION_PREMISES.items()
        },
    }


def collect_handlers() -> dict:
    """Emit the handler vocabulary — three provenance classes.

    ``handlers`` — the shipped interpretation table
    (``lawdata.HANDLERS``); ``prim`` — the minted-primitive handler
    entries (``shipped_primitives.handler_entries()``, kernel = the
    shipped op); ``minted`` — the grammar-minter's verified handler
    records embedded in the prim records (verbatim
    ``handler_record`` shapes — ``handler_from_record`` rebuilds
    them).
    """
    from mlopt_core.laws import minted

    from mlopt_discovery import lawdata, shipped_primitives

    handlers = [
        {
            "tag": tag,
            "source": "lawdata",
            "pattern": spec_to_data(h["pattern"]),
            "kernel": h["kernel"],
            "args": list(h["args"]),
        }
        for tag, h in lawdata.HANDLERS.items()
    ]
    prim = [
        {
            "op": name,
            "source": "prim",
            "pattern": spec_to_data(e["pattern"]),
            "kernel": e["kernel"],
            "args": list(e["args"]),
            "attr_mvars": list(e.get("attr_mvars", ())),
        }
        for name, e in shipped_primitives.handler_entries().items()
    ]
    minted_rows = []
    for name, rec in minted.prim_records().items():
        h = rec.get("handler") or {}
        record = h.get("record")
        if record:
            minted_rows.append(
                {
                    "op": name,
                    "family": h.get("family"),
                    "max_err": h.get("max_err"),
                    "record": record,
                }
            )
    return {"handlers": handlers, "prim": prim, "minted": minted_rows}


def collect_prims() -> dict:
    """Emit the shipped-primitive records verbatim."""
    from mlopt_core.laws import minted

    return dict(minted.prim_records())


def collect_models() -> dict:
    """Emit the phase-2 Models — denotations + minted handlers as ``.hnl``.

    One row per ``synop_*`` denotation (``kind: "denotation"`` — the
    op's meaning is its definitional premise's RHS spec, projected by
    :func:`modelproj.spec_to_hnl`) and per minted handler record
    (``kind: "handler"`` — the ``handler_body`` descriptor projected
    by :func:`modelproj.handler_body_to_hnl`).  Rows the
    ``tensor.*`` surface cannot honour carry ``hnl: null`` +
    ``unsupported`` naming the gap — the honest-skip record; the fp64
    gate (:func:`modelproj.verify_model`, driven by
    ``scripts/verify_vocab_models.py``) writes ``verdicts`` separately
    and they survive re-emission.

    The shipped ``lawdata.HANDLERS`` interpretation entries are
    deliberately absent: their kernels are torch primitives
    (``silu``/``square``/``swiglu``), not specs — there is no spelled
    body to project, and re-spelling the pattern would denote the
    pattern, not the kernel.
    """
    from mlopt_core.laws import minted

    from mlopt_discovery import shipped_primitives

    rows: list[dict] = []
    for name in sorted(minted.SYNOP_SHIPPED):
        spec = shipped_primitives.spec(name)
        rows.append(_denotation_model_row(name, spec))
    for name, rec in minted.prim_records().items():
        handler = (rec.get("handler") or {}).get("record")
        if not handler:
            continue  # refused mints carry no body to project
        rows.append(_handler_model_row(name, rec))
    return {"models": rows}


def _denotation_model_row(name: str, spec: Any) -> dict:
    """Build one ``denotation:`` model row for *name*'s spec.

    Shared by :func:`collect_models` (shipped prims) and
    :func:`emit_prim` (mint-emitted prims) so an emitted row is
    byte-identical to what the collector would produce were the op
    ever admitted — no drift when it ships.
    """
    from mlopt_core.cells import spec_to_data

    from mlopt_discovery import modelproj

    mvars = modelproj.spec_mvars(spec)
    try:
        hnl = modelproj.spec_to_hnl(spec, mvars=mvars)
        unsupported = None
    except modelproj.ModelUnsupportedError as exc:
        hnl, unsupported = None, str(exc)
    return {
        "id": f"denotation:{name}",
        "op": name,
        "kind": "denotation",
        "mvars": sorted(mvars),
        "spec": spec_to_data(spec),
        "hnl": hnl,
        "unsupported": unsupported,
    }


def _handler_model_row(name: str, rec: dict) -> dict:
    """Build one ``handler:`` model row from a prim record's handler."""
    from mlopt_discovery import modelproj

    handler = ((rec.get("handler") or {}).get("record")) or {}
    body = handler.get("handler_body")
    mvars = modelproj.body_mvars(body)
    try:
        hnl = modelproj.handler_body_to_hnl(body, mvars=mvars)
        unsupported = None
    except modelproj.ModelUnsupportedError as exc:
        hnl, unsupported = None, str(exc)
    return {
        "id": f"handler:{name}",
        "op": name,
        "kind": "handler",
        "family": (rec.get("handler") or {}).get("family"),
        "mvars": sorted(mvars),
        # ``body`` stays in ``body_to_data`` form — the same
        # encoding ``handler_from_record`` decodes.
        "body": body,
        "hnl": hnl,
        "unsupported": unsupported,
    }


def collect_families() -> dict:
    """Emit the handler-grammar family tables as data.

    ``handler_synth``'s family inventory and per-family vocabularies
    (the constants a minted body is verified under), ``family_synth``'s
    descriptor grammar (leaf forms, composition orders, budget) and
    ``genkernel.ELEMENTWISE`` — the ``ew`` family's op alphabet.
    """
    from mlopt_discovery import family_synth, genkernel, handler_synth

    return {
        "names": list(handler_synth._FAMILIES),
        "grammar_version": handler_synth._GRAMMAR_VERSION,
        "fold_axes": list(handler_synth.FOLD_AXES),
        "fold_carries": list(handler_synth.FOLD_CARRIES),
        "reduce_ops": list(handler_synth.REDUCE_OPS),
        "reduce_axes": list(handler_synth.REDUCE_AXES),
        "norm_ops": list(handler_synth.NORM_OPS),
        "post_unary": list(handler_synth.POST_UNARY),
        "broadcast_ops": sorted(handler_synth.BROADCAST_OPS),
        "elementwise_ops": sorted(genkernel.ELEMENTWISE),
        "leaf_forms": family_synth.LEAF_FORMS,
        "orders": list(family_synth.ORDERS),
        "budget": family_synth._BUDGET,
    }


_COLLECTORS = {
    "ops.yaml": lambda: {"ops": collect_ops()},
    "laws.yaml": lambda: {"laws": collect_laws()},
    "handlers.yaml": collect_handlers,
    "schemas.yaml": collect_schemas,
    "prims.yaml": lambda: {"prims": collect_prims()},
    "families.yaml": lambda: {"families": collect_families()},
    "models.yaml": collect_models,
}


# ---------------------------------------------------------------------------
#  Emit + upsert — the mint-side fragment API (workstream 0b)
# ---------------------------------------------------------------------------


def _merge_doc(name: str, existing: dict, fresh: dict) -> dict:
    """Merge a fresh owner emit into an existing canonical doc.

    Owner rows upsert by key — a registry change propagates — while
    rows the owners do not know (hand- or machine-added fragments,
    the mint's emissions) survive regeneration.  That is what makes
    the directory a *source* rather than a build artifact: emit
    refreshes the owned rows, never wipes the minted ones.
    """
    if name in ("ops.yaml", "laws.yaml", "schemas.yaml"):
        section = name[: -len(".yaml")]
        rows = existing.setdefault(section, [])
        for row in fresh.get(section, []):
            _upsert(rows, "name", row)
        if name == "schemas.yaml":
            existing.setdefault("premises", {}).update(
                fresh.get("premises", {})
            )
        return existing
    if name == "handlers.yaml":
        for section, key in (
            ("handlers", "tag"),
            ("prim", "op"),
            ("minted", "op"),
        ):
            rows = existing.setdefault(section, [])
            for row in fresh.get(section, []):
                _upsert(rows, key, row)
        return existing
    if name == "prims.yaml":
        existing.setdefault("prims", {}).update(fresh.get("prims", {}))
        return existing
    if name == "families.yaml":
        # Top-level family tables refresh wholesale; extra keys (a
        # minted-family ledger, say) are preserved.
        existing.setdefault("families", {}).update(
            fresh.get("families", {})
        )
        return existing
    if name == "models.yaml":
        # Owner rows upsert by ``id``; the ``verdicts`` map is
        # machine-written provenance (the fp64 gate's record) and is
        # never touched by emission — stale verdicts are detectable
        # because each carries the ``hnl`` text it measured.
        rows = existing.setdefault("models", [])
        for row in fresh.get("models", []):
            _upsert(rows, "id", row)
        return existing
    raise ValueError(f"no merge rule for {name}")  # pragma: no cover


def emit_all(directory: str | Path) -> None:
    """Refresh the vocabulary directory from its owners; write it.

    Each canonical file is *merged*, not replaced: the collector's
    rows upsert by key into the existing doc, so fragments added by
    hand or by the mint (:func:`upsert_op` / :func:`emit_prim` &c.)
    survive re-emission — the directory is the source of truth and
    the Python registries are projections refreshed through it.
    The seven shell stubs and the two projections are then written.
    This is what ``scripts/emit_vocab_fragments.py`` runs.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_FILES:
        fresh = _COLLECTORS[name]()
        path = directory / name
        if path.is_file():
            doc = _merge_doc(
                name, _read_canonical(directory, name), fresh
            )
        else:
            doc = fresh
        _write_canonical(directory, name, doc)
    for name in SHELL_FILES:
        (directory / name).write_text("[]\n")
    sync_projections(directory)


def _read_docs_or_drift(directory: Path) -> tuple[dict, list[str]]:
    """Read the seven canonical files; collect missing/malformed drift."""
    docs: dict[str, dict] = {}
    drifted: list[str] = []
    for name in CANONICAL_FILES:
        path = directory / name
        if not path.is_file():
            drifted.append(f"{name}: missing")
            continue
        try:
            docs[name] = _read_canonical(directory, name)
        except (ValueError, json.JSONDecodeError) as exc:
            drifted.append(f"{name}: {exc}")
    return docs, drifted


def _expect_owner_rows() -> dict[tuple[str, str], dict]:
    """Map ``(file, section) → {key value: collector row}`` for owners."""
    expected: dict[tuple[str, str], dict] = {}

    def _expect(
        file: str, section: str, key: object, row: object
    ) -> None:
        expected.setdefault((file, section), {})[key] = row

    for name in ("ops.yaml", "laws.yaml"):
        section = name[: -len(".yaml")]
        for row in _COLLECTORS[name]().get(section, []):
            _expect(name, section, row["name"], row)
    schemas = _COLLECTORS["schemas.yaml"]()
    for row in schemas.get("schemas", []):
        _expect("schemas.yaml", "schemas", row["name"], row)
    premises = cast(dict, schemas.get("premises", {}))
    for prem_name, records in premises.items():
        _expect("schemas.yaml", "premises", prem_name, records)
    h = _COLLECTORS["handlers.yaml"]()
    for section, key in (
        ("handlers", "tag"),
        ("prim", "op"),
        ("minted", "op"),
    ):
        for row in h.get(section, []):
            _expect("handlers.yaml", section, row[key], row)
    prims = cast(dict, _COLLECTORS["prims.yaml"]().get("prims", {}))
    for prim_name, record in prims.items():
        _expect("prims.yaml", "prims", prim_name, record)
    families = cast(
        dict, _COLLECTORS["families.yaml"]().get("families", {})
    )
    for fam_key, fam_val in families.items():
        _expect("families.yaml", "families", fam_key, fam_val)
    for row in _COLLECTORS["models.yaml"]().get("models", []):
        _expect("models.yaml", "models", row["id"], row)
    return expected


def _index_canonical(
    docs: dict[str, dict], expected: dict[tuple[str, str], dict]
) -> dict[tuple[str, str], dict]:
    """Index the canonical side the same way ``_expect_owner_rows`` did.

    Dict containers key directly; row lists key on the section's
    merge field.
    """
    key_field = {
        ("ops.yaml", "ops"): "name",
        ("laws.yaml", "laws"): "name",
        ("schemas.yaml", "schemas"): "name",
        ("schemas.yaml", "premises"): None,
        ("handlers.yaml", "handlers"): "tag",
        ("handlers.yaml", "prim"): "op",
        ("handlers.yaml", "minted"): "op",
        ("prims.yaml", "prims"): None,
        ("families.yaml", "families"): None,
        ("models.yaml", "models"): "id",
    }
    actual: dict[tuple[str, str], dict] = {}
    for (file, section), _exp in expected.items():
        container = docs[file].get(section)
        field = key_field.get((file, section))
        if isinstance(container, dict):
            actual[(file, section)] = container
        else:
            actual[(file, section)] = {
                r[field]: r
                for r in container or ()
                if isinstance(r, dict)
            }
    return actual


def _owner_row_drift(
    expected: dict[tuple[str, str], dict],
    actual: dict[tuple[str, str], dict],
) -> list[str]:
    """Report owner rows missing or edited in the canonical files."""
    drifted: list[str] = []
    for (file, section), rows in expected.items():
        table = actual.get((file, section), {})
        for key, row in rows.items():
            # Canonical files are JSON — tuples/frozensets on the
            # collector side come back as lists; compare post-codec.
            if table.get(key) != json.loads(json.dumps(row)):
                drifted.append(f"{file}:{section}:{key}")
    return drifted


def _projection_drift(directory: Path) -> list[str]:
    """Report projection files that differ from a fresh sync."""
    drifted: list[str] = []
    for name, rows in _projection_rows(directory).items():
        expected_text = _emit(rows, 0) + "\n"
        path = directory / name
        if not path.is_file() or path.read_text() != expected_text:
            drifted.append(name)
    return drifted


def check_vocab(directory: str | Path) -> list[str]:
    """Return the drift list for *directory* — empty means fresh.

    Three checks, all structural: every canonical file carries the
    ``schema:`` pin; every owner-emitted row is present *and
    identical* in its canonical file (a hand-edit to an owned row is
    drift — the next emit would revert it, so it is reported now);
    and the two projections match what :func:`sync_projections`
    would write from the canonical files.
    """
    directory = Path(directory)
    docs, drifted = _read_docs_or_drift(directory)
    if drifted:
        return drifted
    expected = _expect_owner_rows()
    return _owner_row_drift(
        expected, _index_canonical(docs, expected)
    ) + _projection_drift(directory)


def _update_canonical(directory: Path, name: str, mutate) -> None:
    """Read → mutate → write one canonical file (no projection sync)."""
    doc = _read_canonical(directory, name)
    mutate(doc)
    _write_canonical(directory, name, doc)


def _rewrite_canonical(directory: Path, name: str, mutate) -> None:
    """Read → mutate → write one canonical file, then resync."""
    _update_canonical(directory, name, mutate)
    sync_projections(directory)


def upsert_op(directory: str | Path, row: dict) -> None:
    """Insert/replace an op row (key ``name``) in ``ops.yaml``."""
    directory = Path(directory)

    def mut(doc: dict) -> None:
        _upsert(_section(doc, "ops"), "name", row)

    _rewrite_canonical(directory, "ops.yaml", mut)


def upsert_law(directory: str | Path, record: dict) -> None:
    """Insert/replace a ``law_to_data`` record (key ``name``)."""
    directory = Path(directory)

    def mut(doc: dict) -> None:
        _upsert(_section(doc, "laws"), "name", record)

    _rewrite_canonical(directory, "laws.yaml", mut)


def upsert_schema(directory: str | Path, record: dict) -> None:
    """Insert/replace a schema record (key ``name``) in ``schemas.yaml``."""
    directory = Path(directory)

    def mut(doc: dict) -> None:
        _upsert(_section(doc, "schemas"), "name", record)

    _rewrite_canonical(directory, "schemas.yaml", mut)


def upsert_handler(directory: str | Path, row: dict) -> None:
    """Insert/replace a handler row in ``handlers.yaml``.

    *row* carries ``tag`` (interpretation-table entries) or ``op``
    (prim/minted entries) plus ``source`` selecting the section —
    ``"lawdata"``/``"shipped"`` → ``handlers``, ``"prim"`` → ``prim``,
    anything else → ``minted``.
    """
    directory = Path(directory)
    source = row.get("source", "")
    section = (
        "handlers"
        if source in ("lawdata", "shipped")
        else "prim"
        if source == "prim"
        else "minted"
    )
    key = "tag" if section == "handlers" else "op"

    def mut(doc: dict) -> None:
        _upsert(_section(doc, section), key, row)

    _rewrite_canonical(directory, "handlers.yaml", mut)


def upsert_prim(directory: str | Path, name: str, record: dict) -> None:
    """Insert/replace a prim record (key = the ``synop*`` name)."""
    directory = Path(directory)

    def mut(doc: dict) -> None:
        # ``prims`` is a name→record map, not a row list.
        doc.setdefault("prims", {})[name] = record

    _rewrite_canonical(directory, "prims.yaml", mut)


def upsert_model_verdicts(
    directory: str | Path, verdicts: dict[str, dict]
) -> None:
    """Write fp64-gate verdicts into ``models.yaml``'s ``verdicts`` map.

    *verdicts* maps model ``id`` → the :func:`modelproj.verify_model`
    record.  This is the provenance write the plan's gate asks for —
    ``handl-shipped`` only where every bound site held the shipped
    fp64 floor; the map survives re-emission (owner rows upsert,
    verdicts are never collector output).
    """
    directory = Path(directory)

    def mut(doc: dict) -> None:
        doc.setdefault("verdicts", {}).update(verdicts)

    _rewrite_canonical(directory, "models.yaml", mut)


def upsert_model(directory: str | Path, row: dict) -> None:
    """Insert/replace a ``models.yaml`` row (key ``id``).

    The mint-side half of the model table: :func:`emit_prim` writes
    ``denotation:``/``handler:`` rows through the batched path, and
    callers adding a Model by hand use this — owner rows still
    upsert on re-emission, machine rows survive.
    """
    directory = Path(directory)

    def mut(doc: dict) -> None:
        _upsert(_section(doc, "models"), "id", row)

    _rewrite_canonical(directory, "models.yaml", mut)


def upsert_schema_premise(
    directory: str | Path, name: str, records: list
) -> None:
    """Set ``schemas.yaml``'s premise closure for schema *name*.

    The ``premises`` map is the cert leg's anchored-axiom channel —
    ``pipeline._schema_proposal`` attaches ``premises[name]`` to the
    minted claim.  *records* are ``law_to_data`` records (the same
    shape ``collect_schemas`` emits for ``_FUSION_PREMISES``).
    """
    directory = Path(directory)

    def mut(doc: dict) -> None:
        doc.setdefault("premises", {})[name] = list(records)

    _rewrite_canonical(directory, "schemas.yaml", mut)


def upsert_family(directory: str | Path, descriptor: dict) -> None:
    """Upsert a minted handler-family descriptor into ``families.yaml``.

    *descriptor* is a ``family_synth.mint_families`` survivor —
    ``{name, gate, leaves, broadcast, requires, orders, provenance}``
    — stored under ``families.minted_families`` (a machine-added
    section the collectors do not own, so it survives re-emission).
    Re-upserting the same ``name`` unions the ``covers`` op list and
    keeps the latest descriptor + provenance.
    """
    directory = Path(directory)
    name = descriptor["name"]

    def mut(doc: dict) -> None:
        rows = doc.setdefault("families", {}).setdefault(
            "minted_families", []
        )
        covers = set(descriptor.get("covers") or ())
        for i, row in enumerate(rows):
            if row.get("name") == name:
                merged = dict(descriptor)
                merged["covers"] = sorted(
                    covers | set(row.get("covers") or ())
                )
                rows[i] = merged
                return
        rows.append({**descriptor, "covers": sorted(covers)})

    _rewrite_canonical(directory, "families.yaml", mut)


def upsert_refusal(directory: str | Path, row: dict) -> None:
    """Upsert a refused-spec ledger row into ``families.yaml``.

    Phase 3's refused-spec analysis: a shape the handler grammar
    refused still emits its measured wall as data — *row* carries
    ``key`` (dedup), ``spec`` (``spec_to_data`` form), ``verdict``,
    ``tried``/``best_err``/``note`` and optionally ``op``/``domain``/
    ``family_minted``.  Stored under ``families.refused_specs`` —
    machine-added, survives re-emission.
    """
    directory = Path(directory)

    def mut(doc: dict) -> None:
        rows = doc.setdefault("families", {}).setdefault(
            "refused_specs", []
        )
        _upsert(rows, "key", row)

    _rewrite_canonical(directory, "families.yaml", mut)


def vocab_extra_families(vocab: Vocab) -> tuple[dict, ...]:
    """Return the directory's minted family descriptors as data.

    The consume half of the refused-spec loop closure: rows
    ``upsert_family`` wrote come back as the ``extra_families``
    argument ``handler_synth.mint_handlers`` /
    ``family_synth.mint_families`` already accept — a family minted
    for one refused spec is grammar vocabulary for the next.
    ``covers`` bookkeeping stays behind; the row decodes through
    ``handler_synth.body_from_data`` so tuple-shaped leaf rules and
    orders come back exactly as ``mint_families`` returned them —
    ``extra_families`` consumers see live descriptors, not YAML.
    """
    from mlopt_discovery.handler_synth import body_from_data

    out = []
    for row in vocab.families.get("minted_families") or ():
        desc = body_from_data(
            {k: v for k, v in row.items() if k != "covers"}
        )
        out.append(desc)
    return tuple(out)


def emit_family_rows(
    directory: str | Path,
    *,
    families: Iterable[dict] = (),
    refusals: Iterable[dict] = (),
) -> None:
    """Batch-write family descriptors and refused-spec rows, syncing once.

    The pool-level counterpart of :func:`upsert_family` /
    :func:`upsert_refusal`: ``prim_mint.prim_pool`` collects every
    minted descriptor and refusal across the pool, then lands them
    in a single ``families.yaml`` rewrite + projection sync instead
    of paying one sync per row.
    """
    directory = Path(directory)
    fams = list(families)
    refs = list(refusals)
    if not fams and not refs:
        return

    def mut(doc: dict) -> None:
        fams_doc = doc.setdefault("families", {})
        if fams:
            rows = fams_doc.setdefault("minted_families", [])
            for desc in fams:
                name = desc["name"]
                covers = set(desc.get("covers") or ())
                for i, row in enumerate(rows):
                    if row.get("name") == name:
                        merged = dict(desc)
                        merged["covers"] = sorted(
                            covers | set(row.get("covers") or ())
                        )
                        rows[i] = merged
                        break
                else:
                    rows.append({**desc, "covers": sorted(covers)})
        if refs:
            rows = fams_doc.setdefault("refused_specs", [])
            for row in refs:
                _upsert(rows, "key", row)

    _rewrite_canonical(directory, "families.yaml", mut)


# ---------------------------------------------------------------------------
#  Record → fragment rows — the emit_prim internals (phase 3)
# ---------------------------------------------------------------------------


def _emit_op_row(name: str, record: dict) -> dict:
    """Build the ``ops.yaml`` row a minted prim owns.

    Collector-consistent where it counts: arity/tags/attrs/gen read
    the live ``opmeta``/``_OP_REGISTRY`` (a runtime-minted op IS
    ``op_def``'d, so its ``GenDef`` fields land exactly as a shipped
    synop's would), while ``denotation`` pins ``"prim"`` — the
    record's meaning is the spelled subgraph regardless of which
    executor table currently binds it.  Were the op ever admitted to
    ``SYNOP_SHIPPED`` the collector emits the identical row, so the
    fragment never drifts on promotion.
    """
    from mlopt_core import opmeta
    from mlopt_core.ir import _OP_REGISTRY

    row = _op_row(
        name,
        opmeta.REGISTRY.get(name),
        _OP_REGISTRY.get(name),
        "prim",
    )
    if row["arity"] is None:
        # Unregistered record (a synthetic/test mint): fall back to
        # the fold call's positional arity, the pre-phase-3 rule.
        fold = record.get("fold") or {}
        rhs = fold.get("rhs") or {}
        row["arity"] = (
            len(rhs.get("args", ())) if isinstance(rhs, dict) else 0
        )
    return row


def _record_denotation_spec(record: dict) -> Any:
    """Return the prim record's denotation as a genkernel spec.

    The definitional premise's RHS is the denotation spelling —
    ``record["expand"]`` for depth-2+ mints (the skeleton re-expanded
    to manual ops), the fold LHS for depth-1 — rendered by
    ``prim_mint.pat_spec`` with metavar names kept, the same spec
    ``shipped_primitives.spec`` produces for shipped ops.
    """
    from mlopt_core.laws.serialize import object_from_data

    from mlopt_discovery.prim_mint import pat_spec

    fold = object_from_data(record["fold"])
    if "expand" in record:
        return pat_spec(object_from_data(record["expand"]).rhs)
    return pat_spec(fold.lhs)


def _prim_expand_record(name: str, record: dict) -> dict:
    """Return the record's ``.expand`` premise as a law record.

    Depth-2+ records embed it verbatim; a depth-1 record's premise
    is derivable — ``call -> fold.lhs`` — and is rebuilt through
    :func:`mlopt_discovery.prim_mint.prim_fold_of`, the same
    materializer ``minted.prim_fold`` mirrors for shipped records.
    """
    if "expand" in record:
        return record["expand"]
    from mlopt_core.laws.serialize import law_to_data

    from mlopt_discovery.prim_mint import prim_fold_of

    return law_to_data(prim_fold_of(name, record)[1])


def _prim_schema_record(name: str, record: dict) -> dict:
    """Build the ``fold:<name>`` schema row for a minted prim.

    The fold claim spelled as a fusion-schema recognizer — the
    corpus-derived ``fold:*``-class replacement the plan's phase 3
    asks mined shapes to become.  The record is the fold's own
    ``law_to_data`` fields renamed into the schema namespace, so
    ``pipeline._fusion_schemas`` proposes it through the same
    ``_instance_from_match`` path every shipped schema takes; the
    ``.expand`` premise rides ``schemas.yaml``'s ``premises`` map
    under the schema name.
    """
    schema = dict(record["fold"])
    schema["name"] = f"fold:{name}"
    schema["law"] = (
        f"{name} as a fold-class schema — the mined shape's "
        "corpus-derived recognizer (plan 0033 phase 3)"
    )
    return schema


def _emit_prim_rows(directory: Path, name: str, record: dict) -> None:
    """Write every fragment row one minted prim owns (no sync).

    Phase 3's full emission — the mint's output becomes canonical
    vocabulary, not just a sink-table entry:

    * ``ops.yaml`` — the op row (:func:`_emit_op_row`);
    * ``laws.yaml`` — the ``prim:<name>`` fold record (the claim
      itself, visible to ``load_vocab`` consumers and the handl
      projections — the ``.expand`` premise deliberately stays OUT
      of the law bank: expanding a synop inside saturation
      re-adds the unfolded spelling the fold collapses);
    * ``handlers.yaml`` — the minted handler record, when the mint
      verified one;
    * ``prims.yaml`` — the verbatim mint record;
    * ``models.yaml`` — the ``denotation:`` row (always) and the
      ``handler:`` row (when a body verified), built by the same
      helpers :func:`collect_models` uses so promotion never drifts;
    * ``schemas.yaml`` — the ``fold:<name>`` recognizer row plus its
      ``.expand`` premise closure.
    """
    fold = record.get("fold")
    expand_rec = None
    if fold:
        # Materialize the fold first: ``prim_fold_of`` registers the
        # op's ``op_def`` (so ``_emit_op_row`` sees its ``GenDef``)
        # and validates the embedded ``.expand`` premise's LHS
        # against the fold's own call — a mismatched record fails
        # here, before any fragment lands.
        from mlopt_discovery.prim_mint import prim_fold_of

        prim_fold_of(name, record)
        expand_rec = _prim_expand_record(name, record)
    _update_canonical(
        directory,
        "ops.yaml",
        lambda doc: _upsert(
            _section(doc, "ops"), "name", _emit_op_row(name, record)
        ),
    )
    if fold:
        _update_canonical(
            directory,
            "laws.yaml",
            lambda doc: _upsert(_section(doc, "laws"), "name", fold),
        )
    handler = (record.get("handler") or {}).get("record")
    if handler:
        # Collector-consistent shape (``collect_handlers``'s
        # ``minted`` rows): op/family/max_err/record, no ``source``.
        hentry = record.get("handler") or {}
        _update_canonical(
            directory,
            "handlers.yaml",
            lambda doc: _upsert(
                _section(doc, "minted"),
                "op",
                {
                    "op": name,
                    "family": hentry.get("family"),
                    "max_err": hentry.get("max_err"),
                    "record": handler,
                },
            ),
        )
    _update_canonical(
        directory,
        "prims.yaml",
        lambda doc: doc.setdefault("prims", {}).__setitem__(
            name, record
        ),
    )
    if fold:
        _update_canonical(
            directory,
            "models.yaml",
            lambda doc: _upsert(
                _section(doc, "models"),
                "id",
                _denotation_model_row(
                    name, _record_denotation_spec(record)
                ),
            ),
        )
        if handler:
            _update_canonical(
                directory,
                "models.yaml",
                lambda doc: _upsert(
                    _section(doc, "models"),
                    "id",
                    _handler_model_row(name, record),
                ),
            )
        schema_name = f"fold:{name}"
        _update_canonical(
            directory,
            "schemas.yaml",
            lambda doc: (
                _upsert(
                    _section(doc, "schemas"),
                    "name",
                    _prim_schema_record(name, record),
                ),
                doc.setdefault("premises", {}).__setitem__(
                    schema_name,
                    [expand_rec],
                ),
            ),
        )


def emit_prim(directory: str | Path, name: str, record: dict) -> None:
    """Land a minted primitive as fragments — workstream 0b's seam.

    Phase 3 extends the phase-0 seam: one call now writes the op row,
    the ``prim:`` fold law, the minted handler record, the verbatim
    prim record, the ``denotation:``/``handler:`` model rows and the
    ``fold:`` schema + premise — then resyncs the projections once.
    See :func:`_emit_prim_rows` for the per-file contract.
    """
    emit_prims(directory, [(name, record)])


def emit_prims(
    directory: str | Path, items: Iterable[tuple[str, dict]]
) -> None:
    """Emit a batch of minted primitives, syncing projections once.

    The pool-level emit :func:`prim_mint.prim_pool` calls when
    ``vocab_dir`` is set — N prims cost one ``sync_projections``
    instead of N.
    """
    directory = Path(directory)
    for name, record in items:
        _emit_prim_rows(directory, name, record)
    sync_projections(directory)


def emit_atom(directory: str | Path, atom: Any) -> None:
    """Land an anti-unified definitional atom as schema fragments.

    Plan 0033 phase 3's anti-unification arm: a
    :class:`~mlopt_discovery.antiunify.MintedAtom` — a mined
    ``AtomCandidate`` admitted as ``{prefix}_{i}`` — writes the
    ``fold:<atom>`` schema recognizer row (its ``intro`` claim:
    ``expansion -> atom(leaf_mvars)``) with the ``expand`` premise
    in ``schemas.yaml``'s closure, the ``intro`` law record in
    ``laws.yaml`` (visible to ``load_vocab`` and the handl
    projections), and the op row in ``ops.yaml``.  Duck-typed on
    ``.name``/``.intro``/``.expand`` so any atom-shaped record
    works.
    """
    from mlopt_core.laws.serialize import law_to_data

    directory = Path(directory)
    name = atom.name
    intro_rec = law_to_data(atom.intro)
    _update_canonical(
        directory,
        "ops.yaml",
        lambda doc: _upsert(
            _section(doc, "ops"),
            "name",
            _emit_op_row(name, {"fold": intro_rec}),
        ),
    )
    _update_canonical(
        directory,
        "laws.yaml",
        lambda doc: _upsert(_section(doc, "laws"), "name", intro_rec),
    )
    schema = dict(intro_rec)
    schema["name"] = f"fold:{name}"
    schema["law"] = (
        f"{name} as a fold-class schema — an anti-unified definitional "
        "atom's corpus-derived recognizer (plan 0033 phase 3)"
    )
    schema_name = f"fold:{name}"

    def mut(doc: dict) -> None:
        _upsert(_section(doc, "schemas"), "name", schema)
        doc.setdefault("premises", {})[schema_name] = [
            law_to_data(atom.expand)
        ]

    _update_canonical(directory, "schemas.yaml", mut)
    sync_projections(directory)
