#!/usr/bin/env python3
"""Generate ``kit_torch/aten_classify.py`` from ATen schemas.

The torch bridge's export boundary classifies every ``aten::*`` op it
meets into an IR op name plus a positional-arg disposition: tensor
operands, ``Scalar``-typed Const operands, or named config attributes.
Those three tables (``_ATEN_TO_IR`` / ``_LOCAL_ATTR_SCHEMA`` /
``_SCALAR_OPERAND_OPS`` in ``torch_bridge.py``) were hand-maintained
and drifted from ATen reality — the audit
(``docs/reports/arch_audit_graphopt_packages.md``) found argpos/kwarg
bugs *because* of them.

This is the sdkgen answer, mirroring host-tensor's
``tools/aten_port_gen.rs`` + ``tools/aten_schemas.json`` +
``tools/aten_port_list.txt`` pipeline: the SAME schema dump host-tensor
consumes drives the mlopt-side classification, so schema drift becomes
a ``--check`` failure instead of a silent-wrong export.

Inputs (defaults resolve against the sibling ``handl-lang/host-tensor``
checkout; override with ``--schemas`` / ``--list``):

* ``aten_schemas.json`` — the offline ``torch._C._jit_get_all_schemas()``
  dump (name, overload, args[{name,type,default,kwonly,mutated}], ret).
* ``aten_port_list.txt`` — the curated ``tensor.<hnl> = aten::<query>``
  pins selecting which aten ops are in scope.

Derivation rules (what schemas CAN express):

* **IR name** — the bare aten name (overload suffix dropped), unless an
  ``IR_NAME_OVERRIDES`` entry says otherwise (``min.other`` →
  ``minimum``, ``view`` → ``reshape``, …).  The port list's host-tensor
  ``hnl`` names are NOT used: mlopt's IR vocabulary is the aten name
  space, and the hand tables in ``torch_bridge.py`` remain the override
  layer for ops outside the port surface.
* **Scalar operand positions** — a ``Scalar``/``Scalar?``-typed
  positional arg at index ≥ 1 in a Tensor-first overload marks the IR
  op as taking scalar Const operands (``pow``, ``eq.Scalar``,
  ``where.ScalarSelf``, …).  ``OP_OVERRIDES[*]["scalar_operands"]``
  excludes ops whose ``Scalar`` args are config, not operands
  (``celu.alpha``, ``softshrink.lambd`` — schema type cannot tell the
  difference; the bridge's binding signatures can).
* **Attr schema** — positional args whose type is config-valued
  (``int``/``SymInt``/``float``/``bool``/``str`` and the ``[]``/``[N]``
  list forms) map position → schema arg name.  ``Scalar`` positions are
  attrs only for ops NOT in the scalar-operand set.  Tensor-ish,
  ``ScalarType``/``Device``/``Layout``/``MemoryFormat``/``Generator``/
  ``Dimname``/``complex`` types are excluded — they have dedicated
  channels in the exporter (dtype/device/kwarg) or are never positional.
* Only **Tensor-first** overloads (``self``/first arg ``Tensor`` or
  ``Tensor[]``) contribute: scalar-first sibling overloads
  (``pow.int``, ``eq.float``, ``sort.int``) describe non-tensor calls
  that never appear in an FX graph, and their arg names would collide
  with the tensor overloads' positions.
* Positions are ranks among **non-kwonly** args — kwonly args arrive
  through ``node.kwargs``, not ``node.args``.

Outputs: ``kit-torch/src/kit_torch/aten_classify.py`` in the
sibling handl-lang/kit-torch checkout (P3 Step D-1e) —
``ATEN_TO_IR`` (qualified + bare names → IR op), ``ATTR_SCHEMA``
(IR op → {position: attr name}), ``SCALAR_OPERAND_OPS`` (frozenset of
IR ops), and ``CLASSIFY`` (per-overload provenance for audit/--emit-spec).

``--check`` regenerates in memory and byte-diffs against the checked-in
file — the drift gate.  ``--emit-spec`` prints the resolved
classification as JSON without touching the tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST_TOOLS = REPO_ROOT.parent / "host-tensor" / "tools"
DEFAULT_SCHEMAS = HOST_TOOLS / "aten_schemas.json"
DEFAULT_LIST = HOST_TOOLS / "aten_port_list.txt"
# The generated tables ship in handl-lang/kit-torch (P3 Step D-1e) —
# resolve against the sibling ``kit-torch`` checkout.
DEFAULT_OUT = (
    REPO_ROOT.parent
    / "kit-torch"
    / "src"
    / "kit_torch"
    / "aten_classify.py"
)

# ---------------------------------------------------------------------------
# Curated overrides — what schemas cannot express.
# ---------------------------------------------------------------------------

#: aten name (bare or ``name.overload``) → mlopt IR op name.  Mirrors the
#: semantic half of ``torch_bridge._ATEN_TO_IR`` restricted to the port
#: surface: overload-qualified entries win over bare-name entries.
IR_NAME_OVERRIDES: dict[str, str] = {
    # aten.t() is transpose(-2, -1); mT/swapaxes are named-axes
    # transposes — all carry the transpose schema's (dim0, dim1).
    "t": "transpose",
    "mT": "transpose",
    "swapaxes": "transpose",
    # moveaxis(source, destination) has movedim's positional layout.
    "moveaxis": "movedim",
    # aten.ger IS aten.outer.
    "ger": "outer",
    # aten.trace is the diagonal SUM — not the carrier ``trace``.
    "trace": "diag_sum",
    # clone/contiguous collapse; view/reshape share the shape channel.
    "clone": "contiguous",
    "view": "reshape",
    # min.other/max.other are the ELEMENTWISE spellings — the IR names
    # are the substrate's pointwise minimum/maximum (the reductions
    # stay min/max).
    "min.other": "minimum",
    "max.other": "maximum",
    # List-form split exports as split_with_sizes; the unsafe_*
    # spellings are functionalization-internal twins.
    "split_with_sizes": "split",
    "unsafe_split": "split",
    "unsafe_split_with_sizes": "split",
    # torch.tile(t, dims) IS repeat — same ``shape`` list attr.
    "tile": "repeat",
}

#: IR op → per-op corrections the schema cannot express.
#:
#: ``scalar_operands``: force membership (True) or exclusion (False)
#: in ``SCALAR_OPERAND_OPS``.  ``attrs``: REPLACE the generated
#: {position: name} table for the op (use ``{}`` to clear it — e.g.
#: ``to`` whose dtype/device args have dedicated exporter channels).
OP_OVERRIDES: dict[str, dict] = {
    # ``Scalar`` args that are CONFIG, not operands — the schema type
    # cannot tell ``celu.alpha`` from ``pow.exponent``; the bindings
    # read them as attrs.  Without the exclusion the scalar would mint
    # a Const operand the binding silently ignores.
    "celu": {"scalar_operands": False},
    "softshrink": {"scalar_operands": False},
    "hardshrink": {"scalar_operands": False},
    # Creator bounds are ATTRS, not Const operands — the
    # Tensor_Scalar/Scalar_Tensor sibling overloads would otherwise
    # pull the op into the scalar set and strand the bound in an
    # operand the ``*a``-ignoring binding never reads (the bridge's
    # ``_LOCAL_ATTR_SCHEMA`` already names start/end/steps).
    "linspace": {"scalar_operands": False},
    "logspace": {"scalar_operands": False},
    # ``to``'s prim_* overloads carry dtype as an ``int`` code plus
    # non_blocking/copy booleans at colliding positions — the exporter's
    # dedicated torch.dtype/torch.device channels own this op; naming
    # the positions would poison the dtype attr with a raw enum int.
    "to": {"attrs": {}},
    # Creator/view ops whose leading size list lands under the
    # ``shape`` attr convention (the bindings and ``_shape_of`` read
    # ``shape``/``dim``, never the schema name ``size``/``repeats``).
    "reshape": {"attrs": {1: "shape"}},
    "repeat": {"attrs": {1: "shape"}},
    "expand": {"attrs": {1: "shape"}},
}

# ---------------------------------------------------------------------------
# Schema model
# ---------------------------------------------------------------------------


def _strip_alias(ty: str) -> str:
    """``Tensor(a!)``/``Tensor(a -> *)`` → ``Tensor``; ``int[]?`` → ``int[]``."""
    ty = ty.strip()
    if ty.endswith("?"):
        ty = ty[:-1]
    paren = ty.find("(")
    if paren != -1 and ty.endswith(")"):
        ty = ty[:paren]
    return ty.strip()


def _is_tensor_first(args: list[dict]) -> bool:
    """Tensor-producing overloads take ``Tensor``/``Tensor[]`` first."""
    if not args:
        return False
    return _strip_alias(args[0]["type"]) in ("Tensor", "Tensor[]")


#: Arg types that land as named op ATTRIBUTES (everything else —
#: Tensor/Tensor[] operands, Scalar operands, and the dispatch-only
#: types ScalarType/Device/Layout/MemoryFormat/Generator/Dimname/
#: complex — are excluded).
_ATTR_TYPES = {
    "int",
    "SymInt",
    "float",
    "bool",
    "str",
    "Scalar",
}


def _is_attr_type(ty: str) -> bool:
    base = _strip_alias(ty)
    if base in _ATTR_TYPES:
        return True
    # int[], SymInt[], int[2], SymInt[1], float[], Scalar[], int[]?, …
    if base.endswith("[]") or (base.endswith("]") and "[" in base):
        head = base[: base.index("[")]
        return head in _ATTR_TYPES or head in ("int", "SymInt", "float")
    return False


def _is_scalar_type(ty: str) -> bool:
    return _strip_alias(ty) == "Scalar"


# ---------------------------------------------------------------------------
# Port list
# ---------------------------------------------------------------------------


def load_port_list(path: Path) -> dict[str, str]:
    """Aten name → pinned qualified name, from ``tensor.x = aten::y``."""
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        _lhs, rhs = line.split("=", 1)
        rhs = rhs.strip()
        if not rhs.startswith("aten::"):
            continue
        query = rhs[len("aten::") :]
        name = query.split(".", 1)[0]
        out.setdefault(name, query)
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(schemas: list[dict], ported: dict[str, str]) -> dict:
    """Build the per-IR-op classification.

    Returns ``{"aten_to_ir": {qname: ir}, "attrs": {ir: {pos: name}},
    "scalar_ops": set, "classify": {qname: record}}``.
    """
    # Group schemas by aten name, ported names only.
    by_name: dict[str, list[dict]] = {}
    for s in schemas:
        if s["name"] in ported:
            by_name.setdefault(s["name"], []).append(s)

    # IR name per qualified aten name.
    ir_of: dict[str, str] = {}
    for name, entries in by_name.items():
        for s in entries:
            q = name + (f".{s['overload']}" if s["overload"] else "")
            ir = (
                IR_NAME_OVERRIDES.get(q)
                or IR_NAME_OVERRIDES.get(name)
                or name
            )
            ir_of[q] = ir
        # Bare-name entry resolves like the default overload.
        ir_of.setdefault(
            name,
            IR_NAME_OVERRIDES.get(name) or name,
        )

    # Per-IR-op accumulation.  ``pos`` is the rank among non-kwonly
    # args (kwonly args arrive via node.kwargs, never node.args).
    scalar_positions: dict[str, set[int]] = {}
    attr_positions: dict[str, dict[int, str]] = {}
    seen_overloads: dict[str, list[str]] = {}
    for name, entries in by_name.items():
        # Pinned overload first so its arg names win position conflicts.
        pin = ported[name]
        pin_overload = pin.split(".", 1)[1] if "." in pin else ""
        ordered = sorted(
            entries,
            key=lambda s: s["overload"] != pin_overload,
        )
        for s in ordered:
            q = name + (f".{s['overload']}" if s["overload"] else "")
            ir = ir_of[q]
            seen_overloads.setdefault(ir, []).append(q)
            if not _is_tensor_first(s["args"]):
                continue
            pos = 0
            for a in s["args"]:
                if a["kwonly"]:
                    continue
                ty = a["type"]
                if _is_scalar_type(ty) and pos >= 1:
                    scalar_positions.setdefault(ir, set()).add(pos)
                elif _is_attr_type(ty):
                    attr_positions.setdefault(ir, {}).setdefault(
                        pos, a["name"]
                    )
                pos += 1

    # Apply OP_OVERRIDES: scalar_operands decides set membership;
    # Scalar positions of ops outside the set are attrs, not operands.
    scalar_ops = set()
    for ir in scalar_positions:
        forced = OP_OVERRIDES.get(ir, {}).get("scalar_operands")
        if forced is True or (
            forced is None and ir in scalar_positions
        ):
            scalar_ops.add(ir)

    # Second pass: Scalar-typed positions of non-scalar ops are attrs
    # too (celu.alpha, softshrink.lambd — config Scalars the schema
    # cannot distinguish from pow.exponent; OP_OVERRIDES marks them).
    for name, entries in by_name.items():
        pin = ported[name]
        pin_overload = pin.split(".", 1)[1] if "." in pin else ""
        ordered = sorted(
            entries, key=lambda s: s["overload"] != pin_overload
        )
        for s in ordered:
            q = name + (f".{s['overload']}" if s["overload"] else "")
            ir = ir_of[q]
            if ir in scalar_ops or not _is_tensor_first(s["args"]):
                continue
            pos = 0
            for a in s["args"]:
                if a["kwonly"]:
                    continue
                if _is_scalar_type(a["type"]):
                    attr_positions.setdefault(ir, {}).setdefault(
                        pos, a["name"]
                    )
                pos += 1

    attrs: dict[str, dict[int, str]] = {}
    for ir, table in attr_positions.items():
        attrs[ir] = {p: n for p, n in sorted(table.items())}
    for ir, ov in OP_OVERRIDES.items():
        if "attrs" in ov:
            attrs[ir] = dict(ov["attrs"])
        elif ir not in attrs and "scalar_operands" in ov:
            attrs.pop(ir, None)

    # Provenance records for --emit-spec / the audit report.
    records: dict[str, dict] = {}
    for name, entries in by_name.items():
        for s in entries:
            q = name + (f".{s['overload']}" if s["overload"] else "")
            records[q] = {
                "ir": ir_of[q],
                "schema": s["schema"],
                "overloads": seen_overloads.get(ir_of[q], []),
            }

    return {
        "aten_to_ir": ir_of,
        "attrs": attrs,
        "scalar_ops": scalar_ops,
        "classify": records,
    }


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

_HEADER = '''\
# GENERATED FILE — do not edit by hand.
#
# Regenerate:  .venv/bin/python scripts/torch_port_gen.py
# Drift gate:  .venv/bin/python scripts/torch_port_gen.py --check
#
# Source: host-tensor tools/aten_schemas.json (torch {torch}) filtered
# by tools/aten_port_list.txt ({n_ports} pinned aten ops).  Derivation
# rules and the curated override layer live in
# scripts/torch_port_gen.py — see its docstring.  torch_bridge.py
# merges these tables UNDER its hand-curated override dicts.

"""Schema-derived ATen -> IR classification for the torch bridge.

``ATEN_TO_IR`` maps qualified (``min.other``) and bare (``min``) aten
names to mlopt IR op names; ``ATTR_SCHEMA`` maps an IR op to
``{{argpos: attr_name}}`` for config-typed positional args;
``SCALAR_OPERAND_OPS`` names the IR ops whose ``Scalar``-typed
positional args mint ``Const`` operands.  Everything here is derived
from ``aten_schemas.json`` — drift vs the torch build shows up as a
``--check`` failure, not a silent-wrong export.
"""

from __future__ import annotations

SCHEMAS_TORCH = {torch!r}
PORTED_OPS = {n_ports}

'''


def emit(result: dict, torch_version: str, n_ports: int) -> str:
    """Render the deterministic ``aten_classify.py`` module text."""
    ir_of = result["aten_to_ir"]
    attrs = result["attrs"]
    scalar_ops = result["scalar_ops"]
    records = result["classify"]

    out = [_HEADER.format(torch=torch_version, n_ports=n_ports)]
    out.append("#: Qualified + bare aten name -> IR op name.\n")
    out.append("ATEN_TO_IR: dict[str, str] = {\n")
    for q in sorted(ir_of):
        out.append(f"    {q!r}: {ir_of[q]!r},\n")
    out.append("}\n\n")

    out.append(
        "#: IR op -> {argpos: attr name} for config-typed positional "
        "args.\n"
    )
    out.append("ATTR_SCHEMA: dict[str, dict[int, str]] = {\n")
    for ir in sorted(attrs):
        inner = ", ".join(f"{p}: {n!r}" for p, n in attrs[ir].items())
        out.append(f"    {ir!r}: {{{inner}}},\n")
    out.append("}\n\n")

    out.append(
        "#: IR ops whose Scalar-typed positional args are Const "
        "operands.\n"
    )
    out.append("SCALAR_OPERAND_OPS: frozenset[str] = frozenset({\n")
    for ir in sorted(scalar_ops):
        out.append(f"    {ir!r},\n")
    out.append("})\n\n")

    out.append(
        "#: Per-overload provenance — qualified aten name -> "
        "{ir, schema, overloads}.\n"
    )
    out.append("CLASSIFY: dict[str, dict] = {\n")
    for q in sorted(records):
        r = records[q]
        ovs = ", ".join(repr(o) for o in r["overloads"])
        out.append(
            f"    {q!r}: {{'ir': {r['ir']!r}, 'schema': "
            f"{r['schema']!r}, 'overloads': [{ovs}]}},\n"
        )
    out.append("}\n")
    return _format_with_ruff("".join(out))


def _format_with_ruff(text: str) -> str:
    """Run the emitted module through ``ruff format`` when available.

    The checked-in artifact must satisfy ``ruff format --check``
    (repo gate), and hand-rolling ruff's quote/wrap style is fragile —
    so the generator pipes its own output through the same formatter.
    ``ruff`` resolves as the ``.venv`` sibling of ``sys.executable``
    first, then ``PATH``; with neither, the raw text is returned (still
    valid Python — only the byte-diff gate cares about the style).
    """
    ruff = Path(sys.executable).with_name("ruff")
    if not ruff.exists():
        ruff = Path(shutil.which("ruff") or "")
    if not ruff or not ruff.exists():
        return text
    proc = subprocess.run(
        [
            str(ruff),
            "format",
            "--stdin-filename",
            "aten_classify.py",
            "-",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI: generate, ``--check`` for drift, or ``--emit-spec`` for JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schemas", type=Path, default=DEFAULT_SCHEMAS)
    ap.add_argument("--list", type=Path, default=DEFAULT_LIST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--check",
        action="store_true",
        help="byte-diff generated output against --out; exit 1 on drift",
    )
    ap.add_argument(
        "--emit-spec",
        action="store_true",
        help="print the resolved classification as JSON, write nothing",
    )
    args = ap.parse_args(argv)

    inv = json.loads(args.schemas.read_text())
    torch_version = inv.get("torch", "unknown")
    ported = load_port_list(args.list)
    result = classify(inv["schemas"], ported)

    if args.emit_spec:
        spec = {
            "torch": torch_version,
            "ported": len(ported),
            "aten_to_ir": result["aten_to_ir"],
            "attr_schema": result["attrs"],
            "scalar_operand_ops": sorted(result["scalar_ops"]),
            "overloads": result["classify"],
        }
        json.dump(spec, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    text = emit(result, torch_version, len(ported))
    if args.check:
        if not args.out.exists():
            print(f"MISSING: {args.out}", file=sys.stderr)
            return 1
        current = args.out.read_text()
        if current != text:
            print(
                f"DRIFT: {args.out} is stale — regenerate with "
                "scripts/torch_port_gen.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"wrote {args.out} "
        f"({len(result['aten_to_ir'])} names, "
        f"{len(result['attrs'])} attr schemas, "
        f"{len(result['scalar_ops'])} scalar-operand ops)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
