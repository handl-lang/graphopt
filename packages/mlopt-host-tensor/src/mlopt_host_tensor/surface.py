"""The host ``tensor.*`` op surface — one generated artifact, three readers.

``host_surface.json`` (this package's data file) is the graphopt-side
copy of the authoritative host-tensor surface, produced by
``scripts/dump_host_surface.py`` from the host-tensor checkout —
*read-only* there:

* the handwritten ``op "tensor.X" : <signature>`` declarations in
  ``host-tensor/src/index.hnl`` (the seam ops the Rust impls own);
* ``aten_port_gen --emit-spec`` — the generator's resolved spec for
  the ATen-ported ops (``tools/aten_port_list.txt`` +
  ``tools/aten_schemas.json`` → ``src/ops/aten_generated.rs``),
  carrying ``min_args``/``max_args``/``schema``/``status``.

Every entry is ``{"op": "tensor.<name>", "signature": …,
"min_args": …, "max_args": …|null, "status": …, "kind":
"handwritten"|"generated"}``.  Three consumers share it:

* :func:`host_ops` — this package's own readers (drift guards, the
  ``hnl_import`` covered-set check);
* ``mlopt_host_tensor.bindings.HOST_BINDINGS`` — the 138-op IR→host
  table's *target* names are validated against it (every binding key
  must be a real ``tensor.*`` op);
* a future ``emit.py``-style renderer — the mlopt-side signature
  table's adoption path is documented in ``docs/`` (the sibling
  mlopt repo is out of scope here).

Regenerate with::

    .venv/bin/python scripts/dump_host_surface.py

``HOST_TENSOR_DIR`` points at the checkout (default: the sibling
``../host-tensor``); ``--check`` mode (what the drift test runs when
the checkout is present) compares without writing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "collect_host_surface",
    "host_ops",
    "render_surface",
    "surface_json_path",
]

_SURFACE_JSON = Path(__file__).with_name("host_surface.json")

#: ``op "tensor.conv2d" : Value... -> Tensor`` — the handwritten decls.
#: The result side is ``(.+?)$`` (rest of line), NOT ``\S+$``:
#: ``tensor.shape`` returns ``(List Int)`` — a two-token composite type
#: a single-token pattern silently drops (237 ops → 238).
_OP_DECL = re.compile(
    r'^op "(tensor\.[A-Za-z0-9_]+)"\s*:\s*(.+?)\s*->\s*(.+?)\s*$',
    re.MULTILINE,
)


def surface_json_path() -> Path:
    """Return the checked-in artifact path."""
    return _SURFACE_JSON


def host_ops() -> dict[str, dict[str, Any]]:
    """``tensor.<name>`` → surface entry, from the shipped artifact."""
    data = json.loads(_SURFACE_JSON.read_text())
    return {op["op"]: op for op in data["ops"]}


def _find_host_tensor(explicit: str | Path | None = None) -> Path:
    """Locate the host-tensor checkout."""
    cand = (
        Path(explicit)
        if explicit is not None
        else Path(
            os.environ.get(
                "HOST_TENSOR_DIR",
                Path(__file__).resolve().parents[5] / "host-tensor",
            )
        )
    )
    if not (cand / "src" / "index.hnl").is_file():
        raise FileNotFoundError(
            f"host-tensor checkout not found at {cand} — set "
            "HOST_TENSOR_DIR"
        )
    return cand


def _index_hnl_ops(ht: Path) -> dict[str, dict[str, Any]]:
    """Handwritten ``op "tensor.X" : sig`` decls from ``index.hnl``."""
    text = (ht / "src" / "index.hnl").read_text()
    ops: dict[str, dict[str, Any]] = {}
    for m in _OP_DECL.finditer(text):
        name, params, result = m.group(1), m.group(2), m.group(3)
        ops[name] = {
            "op": name,
            "signature": f"{params.strip()} -> {result}",
            "min_args": None,
            "max_args": None,
            "status": "Handwritten",
            "kind": "handwritten",
            "reason": "",
        }
    return ops


def _emit_spec(ht: Path) -> list[dict[str, Any]]:
    """Run ``aten_port_gen --emit-spec``; return its ``ops`` array."""
    binary = ht / "target" / "debug" / "aten_port_gen"
    if not binary.is_file():
        raise FileNotFoundError(
            f"{binary} not built — "
            "cargo build --bin aten_port_gen (in host-tensor)"
        )
    out = subprocess.run(
        [str(binary), "--emit-spec"],
        cwd=ht,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"aten_port_gen --emit-spec failed: {out.stderr.strip()}"
        )
    spec = json.loads(out.stdout)
    return spec["ops"]


def collect_host_surface(
    host_tensor_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build the unified op→signature table from a host-tensor checkout.

    Generated (ATen-ported) entries win over same-named handwritten
    decls — the generator owns those names.  Sorted by op name for a
    stable artifact.
    """
    ht = _find_host_tensor(host_tensor_dir)
    ops = _index_hnl_ops(ht)
    for entry in _emit_spec(ht):
        name = entry["op"]
        ops[name] = {
            "op": name,
            "signature": entry.get("signature", ""),
            "schema": entry.get("schema", ""),
            "min_args": entry.get("min_args"),
            "max_args": entry.get("max_args"),
            "status": entry.get("status", ""),
            "kind": "generated",
            "reason": entry.get("reason", ""),
        }
    return sorted(ops.values(), key=lambda o: o["op"])


def render_surface(ops: list[dict[str, Any]]) -> str:
    """Canonical artifact text — what ``dump_host_surface.py`` writes."""
    doc = {
        "provenance": (
            "Generated by scripts/dump_host_surface.py from the "
            "host-tensor checkout (src/index.hnl op decls + "
            "aten_port_gen --emit-spec). Do not hand-edit; regenerate."
        ),
        "ops": ops,
    }
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"
