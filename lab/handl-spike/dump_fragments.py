"""Dump mlopt's vocabulary as a handl fragment directory.

The bridge (`mlopt_discovery.handl`) already emits handl-shaped rows;
this tool writes them into the *ten-file layout* `language_from_dir`
reads (`handl-kit-data-prelude`), so a handl host can assemble a
`Language` from mlopt's vocabulary:

    tokens.yaml · grammar.yaml · eval.yaml · types.yaml · ast.yaml
    caps.yaml · options.yaml · lexer_rules.yaml
    transforms.yaml (optional) · lints.yaml / lsp.yaml (optional)

The eight required files are written even when empty (`[]`) — mlopt's
"syntax" is the IR, not a text language, so the shell files are stubs
today; `ast.yaml` carries the op shapes and `transforms.yaml` the laws.

JSON is valid YAML, so plain maps and lists round-trip either way — but
**enum variants do not**: `serde_yaml` spells them as YAML *tags*
(`!Declarative`, `!Create`, `!KeepPath`, `!KindAt`, `!And`) where
`serde_json` spells them as maps (`{"Declarative": …}`).  `language_from_dir`
reads YAML, so the rows are written with tags — measured: the map form
fails to load with *"invalid type: map, expected a YAML tag starting
with '!'"*.

Usage:
    .venv/bin/python lab/handl-spike/dump_fragments.py [outdir] [--all]

``--all`` emits the whole shipped bank; the default emits a small,
deliberate slice (one structural law plus the op shapes it needs) so the
spike fixture stays reviewable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mlopt_core.laws import ALL_RULES
from mlopt_discovery.cellstore import ambient_store
from mlopt_discovery.handl import store_to_handl

#: The shell files `language_from_dir` requires (empty is valid).
_SHELL = (
    "tokens.yaml",
    "grammar.yaml",
    "eval.yaml",
    "types.yaml",
    "caps.yaml",
    "options.yaml",
    "lexer_rules.yaml",
)

#: The law the default slice carries — structural (no attrs, no guard,
#: no leaf), so it exercises shapes and matching without touching values.
_SLICE_LAW = "assoc_add"

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


def _emit(node: object, indent: int) -> str:
    """Render *node* as YAML, spelling enum variants as tags."""
    pad = " " * indent
    if isinstance(node, dict):
        if len(node) == 1:
            ((key, value),) = node.items()
            if key in _TAGS:
                inner = _emit(value, indent)
                # a tag on a scalar/flow value reads better inline
                if (
                    isinstance(value, (str, int, float))
                    or value is None
                ):
                    return f"{pad}!{key} {_scalar(value)}"
                return f"{pad}!{key}\n{inner}"
        if not node:
            return f"{pad}{{}}"
        out = []
        for key, value in node.items():
            rendered = _emit(value, indent + 2)
            if isinstance(value, (str, int, float)) or value is None:
                out.append(f"{pad}{key}: {_scalar(value)}")
            elif isinstance(value, dict) and not value:
                out.append(f"{pad}{key}: {{}}")
            elif isinstance(value, list) and not value:
                out.append(f"{pad}{key}: []")
            else:
                out.append(f"{pad}{key}:")
                out.append(rendered)
        return "\n".join(out)
    if isinstance(node, list):
        if not node:
            return f"{pad}[]"
        out = []
        for item in node:
            rendered = _emit(item, indent + 2)
            stripped = rendered[indent + 2 :]
            out.append(f"{pad}- {stripped}")
        return "\n".join(out)
    return f"{pad}{_scalar(node)}"


def _dump(path: Path, rows: list) -> None:
    """Write *rows* as one YAML sequence (the fragment file's shape)."""
    path.write_text(_emit(rows, 0) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Write the fragment directory; return the exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    everything = "--all" in args
    args = [a for a in args if a != "--all"]
    out = Path(args[0]) if args else Path(__file__).parent / "fragments"
    out.mkdir(parents=True, exist_ok=True)

    rules = (
        list(ALL_RULES)
        if everything
        else [r for r in ALL_RULES if r.name == _SLICE_LAW]
    )
    if not rules:
        raise SystemExit(f"no rule named {_SLICE_LAW!r} in ALL_RULES")

    frag = store_to_handl(ambient_store(rules=rules, handlers={}))
    for name in _SHELL:
        _dump(out / name, [])
    _dump(out / "ast.yaml", frag.get("ast", []))
    _dump(out / "transforms.yaml", frag.get("transforms", []))

    print(f"wrote {out}")
    print(f"  laws        : {len(frag.get('transforms', []))}")
    print(f"  op shapes   : {len(frag.get('ast', []))}")
    print(f"  natives     : {len(frag.get('natives', []))}")
    for row in frag.get("transforms", []):
        print(
            f"    - {row['id']}  (pass={row['pass']}, kind={row['node_kind']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
