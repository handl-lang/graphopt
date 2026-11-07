"""Emit the canonical fragment vocabulary (plan 0033 phase 0).

Writes ``vocab/`` — the single source of truth for the discovery
vocabulary — from its current owners, one collector per registry
(see :mod:`mlopt_discovery.vocabdir` for the file layout, the merge
semantics and the load/upsert API):

    .venv/bin/python scripts/emit_vocab_fragments.py [outdir]
    .venv/bin/python scripts/emit_vocab_fragments.py --check [outdir]

With no ``outdir`` the default is the repo's ``vocab/``.  Emission
*merges*: owner rows upsert by key while hand- or machine-added
fragments survive.  ``--check`` reports drift instead of writing —
a bad ``schema:`` pin, an owner row that went missing or was edited,
or a stale ``ast.yaml``/``transforms.yaml`` projection (plan 0033
hard gates: fragment schema drift is a release blocker).
"""

from __future__ import annotations

import sys
from pathlib import Path

from mlopt_discovery import vocabdir

_REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    """Emit (or check) the vocabulary directory; return the exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in args
    args = [a for a in args if a != "--check"]
    out = Path(args[0]) if args else _REPO / "vocab"

    if check:
        drifted = vocabdir.check_vocab(out)
        if drifted:
            for rel in drifted[:50]:
                print(f"DRIFT {rel}", file=sys.stderr)
            if len(drifted) > 50:
                print(
                    f"… and {len(drifted) - 50} more", file=sys.stderr
                )
            print(
                f"{out}: {len(drifted)} drift(s) — "
                "run scripts/emit_vocab_fragments.py",
                file=sys.stderr,
            )
            return 1
        print(f"{out}: vocabulary fresh")
        return 0

    vocabdir.emit_all(out)
    print(f"wrote {out}")
    for name in vocabdir.CANONICAL_FILES:
        doc = vocabdir._read_canonical(out, name)
        rows = sum(
            len(v)
            for k, v in doc.items()
            if k != "schema" and isinstance(v, (list, dict))
        )
        print(f"  {name}: schema={doc['schema']} rows~{rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
