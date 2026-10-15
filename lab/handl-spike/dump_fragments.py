"""Dump catopt's vocabulary as a handl fragment directory.

The bridge (`catopt_discovery.handl`) already emits handl-shaped rows;
this tool writes them into the *ten-file layout* `language_from_dir`
reads (`handl-kit-data-canonical`), so a handl host can assemble a
`Language` from catopt's vocabulary:

    tokens.yaml · grammar.yaml · eval.yaml · types.yaml · ast.yaml
    caps.yaml · options.yaml · lexer_rules.yaml
    transforms.yaml (optional) · lints.yaml / lsp.yaml (optional)

The eight required files are written even when empty (`[]`) — catopt's
"syntax" is the IR, not a text language, so the shell files are stubs
today; `ast.yaml` carries the op shapes and `transforms.yaml` the laws.

JSON is valid YAML, so the rows are dumped with `json.dump` and the
files still parse under `serde_yaml`.

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

from catopt_core.laws import ALL_RULES
from catopt_discovery.cellstore import ambient_store
from catopt_discovery.handl import store_to_handl

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


def _dump(path: Path, rows: list) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n")


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
