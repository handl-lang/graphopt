"""Differential parity: ``source-onnx`` (Rust) vs ``OnnxSource`` (Python).

ADR-0176's verification gate — the spike's recommendation, "two
producers, one wire contract" (``host/project/reports/
onnx_lifter_rust_spike.md`` §verification).  The Rust crate
``source-onnx`` lifts an ``.onnx`` model to the Term-JSON
envelope + a safetensors leaf side channel **without Python**; the
Python ``kit_onnx.OnnxSource`` (``lift_model`` +
``mlopt_core.term_json.term_to_json``) is the oracle.  This driver
runs the same model through both and compares.

The comparison boundary
-----------------------
Both sides emit the *same wire contract* — ``{"Op": …}`` /
``{"Arr": …}`` / ``{"Const": …}`` trees or the ``dag.program``
record — so the term comparison is **canonical-DAG isomorphism**:
each side's JSON is flattened to a node table, every node is
hash-consed to a canonical descriptor (op name + attrs + child
canonical ids), and the reachable canonical multiset + root id are
compared.  This is insensitive to the two legitimate envelope
differences the codec itself documents:

* **tree vs ``dag.program``** — both producers pick the envelope by
  their own sharing analysis (Python: ``Op.make`` interning;
  Rust: arena refcount).  Isomorphic terms can serialize either way;
  canonicalization erases the difference.
* **attrs channel** — Python ``term_to_json`` emits the canonical
  ``attrs`` field; Rust emits the same field, and older records may
  carry the legacy ``lit.__mlopt_attrs__`` record.  The
  canonicalizer merges both spellings (decode accepts exactly one —
  both present is a decline).

Everything else is compared *spelling-strict*: ``0`` vs ``0.0`` are
different canonical values (ints tag ``("int", …)``, floats tag
``("float", …)`` via ``float.hex()``), ``None`` dims are ``null``,
leaf ``(kind, name, shape)`` records and the leaf *payload*
(sha256 over the safetensors / NumPy bytes) must match exactly.

Side channels compared separately: graph ``inputs``
``(name, shape)``, ``input_dtypes``, and the leaf manifest
``(name, dtype-name, shape, nbytes, sha256)``.

Semantic spot-check
-------------------
Where the canonical terms *agree*, structural parity is the verdict.
Where they *diverge* — and always for small models — both terms are
decoded through ``term_from_json`` and evaluated on shared random
inputs + leaf payloads through a numpy evaluator (:func:`_eval`),
the same trick ``eqsat_differential.py`` uses: ``equal`` /
``diverge`` / ``unevaluated`` are the honest verdicts.  Ops outside
the evaluator's coverage raise :class:`_Unevaluated`, never a wrong
answer.

Refusals are compared too: a model both sides refuse is
``refuse-parity`` (family + op_type checked); one-sided refusal is a
divergence — the worst kind, since it means the tables disagree
about coverage.

Usage::

    .venv/bin/python bench/drivers/audit/onnx_lifter_differential.py
    .venv/bin/python bench/drivers/audit/onnx_lifter_differential.py --model add_relu
    .venv/bin/python bench/drivers/audit/onnx_lifter_differential.py --json /tmp/report.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_SOURCE_ONNX = Path(
    os.environ.get(
        "SOURCE_ONNX_REPO",
        Path.home() / "GitHub" / "handl-lang" / "source-onnx",
    )
)

from kit_onnx.arrays import DTYPE_NAMES  # noqa: E402
from kit_onnx.lift import lift_model  # noqa: E402
from mlopt_core.term import (  # noqa: E402
    Const,
    Op,
    Param,
    Var,
)
from mlopt_core.term_json import (  # noqa: E402
    term_from_json,
    term_to_json,
)

# Deep terms (whisper's 241-node DAG flattens deep) need headroom —
# set after imports; the limit is checked at call time.
sys.setrecursionlimit(100000)

try:
    import onnx as _onnx

    _RESNET_DEFAULT = (
        Path(_onnx.__file__).parent
        / "backend/test/data/light/light_resnet50.onnx"
    )
except Exception:  # pragma: no cover - onnx is a hard dep of kit-onnx
    _RESNET_DEFAULT = Path("/nonexistent/light_resnet50.onnx")

_WHISPER_DEFAULT = (
    Path.home() / "tmp/spike-artifacts/whisper_tiny_encoder.onnx"
)
_FIXTURES_DEFAULT = _SOURCE_ONNX / "tests" / "fixtures"

_BIN_CANDIDATES = [
    _SOURCE_ONNX / "target" / "release" / "examples" / "lift",
    _SOURCE_ONNX / "target" / "debug" / "examples" / "lift",
    Path.home()
    / ".cargo"
    / "shared-target"
    / "release"
    / "examples"
    / "lift",
    Path.home()
    / ".cargo"
    / "shared-target"
    / "debug"
    / "examples"
    / "lift",
]


# ---------------------------------------------------------------------------
#  Rust binary discovery
# ---------------------------------------------------------------------------


def _bin() -> Path | None:
    env = os.environ.get("SOURCE_ONNX_LIFT")
    if env:
        return Path(env)
    for cand in _BIN_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _ensure_bin() -> Path | None:
    """Return the ``lift`` example binary, building it if needed."""
    b = _bin()
    if b is not None:
        return b
    if not _SOURCE_ONNX.is_dir():
        return None
    subprocess.run(
        ["cargo", "build", "--example", "lift"],
        cwd=_SOURCE_ONNX,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return _bin()


# ---------------------------------------------------------------------------
#  The two producers
# ---------------------------------------------------------------------------

_NUMPY_TO_NAME = {
    np.dtype(dt): name for name, dt in DTYPE_NAMES.items()
}


def _py_refusal(e: BaseException) -> dict:
    """Classify a Python lift failure for refusal-parity comparison."""
    from kit_onnx.errors import (
        OnnxImportError,
        UnsupportedOnnxOpError,
    )

    if isinstance(e, UnsupportedOnnxOpError):
        return {"family": "unsupported", "op": e.op, "detail": e.detail}
    if isinstance(e, OnnxImportError):
        return {"family": "import", "detail": str(e)}
    return {"family": "error", "detail": f"{type(e).__name__}: {e}"}


def _shape_tuple(v: Var) -> tuple:
    """``Var.typ.shape`` as a JSON-able tuple (``None`` = unknown dim)."""
    shape = getattr(v.typ, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def _py_lift(path: Path) -> dict:
    """Python oracle: ``lift_model`` + ``term_to_json`` + leaf manifest."""
    try:
        res = lift_model(str(path))
    except (
        Exception
    ) as e:  # refusal or import error — both are data here
        return {"refusal": _py_refusal(e)}
    out: dict[str, Any] = {"ok": True}
    try:
        out["root"] = term_to_json(res.ir.root)
    except Exception as e:
        out["refusal"] = {
            "family": "codec",
            "detail": f"{type(e).__name__}: {e}",
        }
        return out
    out["inputs"] = [
        {"name": v.name, "shape": _shape_tuple(v)}
        for v in res.ir.inputs
    ]
    out["input_dtypes"] = dict(res.input_dtypes)
    leaves = {}
    for name, arr in res.leaves.items():
        dt_name = _NUMPY_TO_NAME.get(np.dtype(arr.dtype))
        payload = np.ascontiguousarray(arr).tobytes()
        leaves[name] = {
            "dtype": dt_name or f"np:{arr.dtype}",
            "shape": tuple(int(d) for d in arr.shape),
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    out["leaves"] = leaves
    return out


# -- safetensors side channel --------------------------------------------

_ST_DT = {
    "BOOL": "|b1",
    "U8": "|u1",
    "I8": "|i1",
    "I16": "<i2",
    "I32": "<i4",
    "I64": "<i8",
    "F16": "<f2",
    "BF16": None,  # no numpy spelling — reported, not decoded
    "F32": "<f4",
    "F64": "<f8",
}


def _parse_safetensors(path: Path) -> dict[str, dict]:
    """Parse a safetensors file into ``{name: {dtype, shape, sha256, nbytes}}``.

    The format is an 8-byte LE header length, a JSON header
    ``{name: {dtype, shape, data_offsets}}``, then the raw buffer —
    parsed directly so the harness needs no extra dependency.
    """
    buf = path.read_bytes()
    (hlen,) = struct.unpack("<Q", buf[:8])
    header = json.loads(buf[8 : 8 + hlen])
    base = 8 + hlen
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        payload = buf[base + s : base + e]
        out[name] = {
            "dtype": meta["dtype"],
            "shape": tuple(meta["shape"]),
            "nbytes": e - s,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return out


def _rs_refusal(doc: dict) -> dict:
    msg = doc.get("error", "")
    if msg.startswith("unsupported ONNX op "):
        # "unsupported ONNX op {op} (node {node:?}): {reason}"
        try:
            op = msg.split("unsupported ONNX op ", 1)[1].split(" ", 1)[
                0
            ]
            op = op.strip("'")
            reason = msg.split("): ", 1)[-1] if "): " in msg else ""
            return {"family": "unsupported", "op": op, "detail": reason}
        except Exception:
            pass
    if msg.startswith("onnx import error:"):
        return {"family": "import", "detail": msg}
    return {"family": "error", "detail": msg}


def _rs_lift(path: Path, workdir: Path) -> dict:
    """Rust producer: the ``lift`` example binary, JSON on stdout."""
    bin_path = _ensure_bin()
    if bin_path is None:
        return {"error": "source-onnx lift example binary not found"}
    st_path = workdir / f"{path.stem}.safetensors"
    try:
        proc = subprocess.run(
            [str(bin_path), str(path), "--st-out", str(st_path)],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return {"error": "lift timed out (>900s)"}
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "error": f"rc={proc.returncode} non-JSON stdout: "
            f"{proc.stdout[:200]!r} stderr={proc.stderr[:300]!r}"
        }
    if not doc.get("ok"):
        return {"refusal": _rs_refusal(doc)}
    # Leaf payload hashes come from the safetensors file itself.
    try:
        tensors = _parse_safetensors(st_path)
    except Exception as e:
        return {"error": f"safetensors parse failed: {e}"}
    manifest = {leaf["name"]: leaf for leaf in doc.get("leaves", [])}
    leaves = {}
    for name, meta in tensors.items():
        m = manifest.get(name, {})
        leaves[name] = {
            "dtype": m.get("dtype"),
            "st_dtype": meta["dtype"],
            "shape": meta["shape"],
            "nbytes": meta["nbytes"],
            "sha256": meta["sha256"],
        }
    # Manifest entries with no tensor payload (shouldn't happen) keep
    # their declared shape for the comparison.
    for name, m in manifest.items():
        if name not in leaves:
            leaves[name] = {
                "dtype": m.get("dtype"),
                "st_dtype": None,
                "shape": tuple(m.get("shape") or ()),
                "nbytes": m.get("nbytes"),
                "sha256": None,
            }
    doc["leaves"] = leaves
    doc.pop("safetensors", None)
    return doc


# ---------------------------------------------------------------------------
#  Canonical term graph — tree/dag envelopes unified, attrs merged
# ---------------------------------------------------------------------------


def _jcanon(v: Any) -> Any:
    """JSON value -> canonical hashable form, spelling-strict on numbers."""
    if v is None:
        return ("null",)
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        # .hex() keeps -0.0 distinct from 0.0 and is exact.
        return ("float", v.hex())
    if isinstance(v, str):
        return ("str", v)
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(_jcanon(x) for x in v))
    if isinstance(v, dict):
        keys = set(v)
        if keys == {"__tuple__"}:
            return ("tuple", tuple(_jcanon(x) for x in v["__tuple__"]))
        if keys == {"__list__"}:
            return ("list", tuple(_jcanon(x) for x in v["__list__"]))
        if keys == {"__nonfinite__"}:
            return ("nonfinite", v["__nonfinite__"])
        return (
            "map",
            tuple(sorted((k, _jcanon(x)) for k, x in v.items())),
        )
    return ("repr", repr(v))


def _attrs_of(node: dict) -> dict:
    """Merge the two attrs channels the codec accepts into one dict."""
    if "attrs" in node and node["attrs"] is not None:
        return node["attrs"]
    lit = node.get("lit")
    if isinstance(lit, dict) and "__mlopt_attrs__" in lit:
        return lit["__mlopt_attrs__"]
    return {}


def _canon_attrs(attrs: dict) -> tuple:
    return tuple(sorted((k, _jcanon(v)) for k, v in attrs.items()))


class _TermView:
    """A term JSON document as a flat node table + canonical ids.

    ``raw[i]`` is the node's local descriptor (no children);
    ``kids[i]`` the child indices; ``canon[i]`` the hash-consed
    canonical id (children's canon ids folded in).
    """

    def __init__(self, doc: dict):
        self.raw: list[tuple] = []
        self.kids: list[tuple[int, ...]] = []
        if "Opaque" in doc:
            payload = doc["Opaque"]
            if not (
                isinstance(payload, dict)
                and payload.get("__tag__") == "dag.program"
            ):
                raise ValueError(
                    f"unknown Opaque tag: {payload!r:.200}"
                )
            nodes = payload["nodes"]
            root = payload["root"]
            for rec in nodes:
                self._add_record(rec, arg_indices=True)
            self.root = root
        else:
            self.root = self._add_tree(doc)
        # Canonical ids — children always precede parents in both
        # flattenings (dag.program contract; tree is post-order).
        self.canon: list[Any] = []
        for i, r in enumerate(self.raw):
            if r[0] == "op":
                self.canon.append(
                    (
                        "op",
                        r[1],
                        r[2],
                        tuple(self.canon[k] for k in self.kids[i]),
                    )
                )
            else:
                self.canon.append(r)

    def _add_record(self, rec: dict, *, arg_indices: bool) -> int:
        """Append one dag.program node record; return its index."""
        if "Op" in rec:
            node = rec["Op"]
            attrs = _canon_attrs(_attrs_of(node))
            if arg_indices:
                kids = tuple(node["args"])
            else:
                kids = tuple(self._add_tree(a) for a in node["args"])
            self.raw.append(("op", node["name"], attrs))
            self.kids.append(kids)
        elif "Arr" in rec:
            val = rec["Arr"]["value"]
            self.raw.append(
                (
                    "leaf",
                    val.get("__mlopt_leaf__", "?"),
                    val.get("name"),
                    _jcanon(val.get("shape")),
                )
            )
            self.kids.append(())
        elif "Const" in rec:
            self.raw.append(("const", _jcanon(rec["Const"]["value"])))
            self.kids.append(())
        else:
            raise ValueError(f"bad node record: {rec!r:.200}")
        return len(self.raw) - 1

    def _add_tree(self, j: Any) -> int:
        """Flatten a tree-position term JSON node (post-order)."""
        if not isinstance(j, dict) or len(j) != 1:
            raise ValueError(f"bad tree node: {j!r:.200}")
        return self._add_record(j, arg_indices=False)

    def reachable(self) -> list[int]:
        """Indices reachable from ``root``, DFS order."""
        seen: set[int] = set()
        order: list[int] = []
        stack = [self.root]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            order.append(i)
            stack.extend(self.kids[i])
        return order


def _view_multiset(view: _TermView) -> Counter:
    return Counter(view.canon[i] for i in view.reachable())


def _op_census(view: _TermView) -> Counter:
    return Counter(
        view.raw[i][1]
        for i in view.reachable()
        if view.raw[i][0] == "op"
    )


def _leaf_census(view: _TermView) -> Counter:
    return Counter(
        view.raw[i][1:]
        for i in view.reachable()
        if view.raw[i][0] == "leaf"
    )


def _first_diffs(
    a: _TermView, b: _TermView, limit: int = 12
) -> list[str]:
    """Positional paired walk — localizes the first structural diffs.

    Pairs nodes by traversal (children zipped positionally), memoized
    on index pairs so DAG sharing cannot loop it.  Where local
    descriptors (kind + payload, children ignored) differ, a diff is
    recorded; children are still paired positionally to find more.
    This is a *reporting* walk, not the verdict — the verdict is the
    canonical-id comparison.
    """
    diffs: list[str] = []
    seen: set[tuple[int, int]] = set()

    def walk(i: int, j: int, path: str) -> None:
        if len(diffs) >= limit or (i, j) in seen:
            return
        seen.add((i, j))
        ra, rb = a.raw[i], b.raw[j]
        # raw never carries children (kids is separate) — the whole
        # record is the local payload: kind + name/attrs/value.
        pa, pb = ra, rb
        here = path or "<root>"
        if pa != pb:
            diffs.append(f"{here}: {pa!r:.160}  !=  {pb!r:.160}")
        ka, kb = a.kids[i], b.kids[j]
        if len(ka) != len(kb):
            diffs.append(f"{here}: arity {len(ka)} != {len(kb)}")
        for n, (x, y) in enumerate(zip(ka, kb, strict=False)):
            label = (
                ra[1]
                if ra[0] == "op"
                else f"{ra[0]}:{ra[2] if len(ra) > 2 else ''}"
            )
            walk(x, y, f"{here}/{label}[{n}]")

    walk(a.root, b.root, "")
    return diffs


def _compare_terms(py: dict, rs: dict) -> dict:
    """Canonical-DAG comparison of the two root terms."""
    try:
        va = _TermView(py["root"])
    except Exception as e:
        return {"verdict": "py-term-error", "detail": str(e)}
    try:
        vb = _TermView(rs["root"])
    except Exception as e:
        return {"verdict": "rs-term-error", "detail": str(e)}
    ca, cb = _view_multiset(va), _view_multiset(vb)
    out: dict[str, Any] = {
        "verdict": "iso"
        if va.canon[va.root] == vb.canon[vb.root]
        else "diff",
        "py_nodes": len(va.reachable()),
        "rs_nodes": len(vb.reachable()),
    }
    if out["verdict"] == "diff":
        out["py_only_canon"] = sum((ca - cb).values())
        out["rs_only_canon"] = sum((cb - ca).values())
        out["op_census_diff"] = {
            "py_only": dict(
                (_op_census(va) - _op_census(vb)).most_common(10)
            ),
            "rs_only": dict(
                (_op_census(vb) - _op_census(va)).most_common(10)
            ),
        }
        out["leaf_census_diff"] = {
            "py_only": [
                f"{k[0]}:{k[1]}"
                for k in (
                    _leaf_census(va) - _leaf_census(vb)
                ).elements()
            ][:10],
            "rs_only": [
                f"{k[0]}:{k[1]}"
                for k in (
                    _leaf_census(vb) - _leaf_census(va)
                ).elements()
            ][:10],
        }
        out["first_diffs"] = _first_diffs(va, vb)
    return out


# ---------------------------------------------------------------------------
#  Side-channel comparison (inputs / dtypes / leaves)
# ---------------------------------------------------------------------------


def _shape_eq(a: Any, b: Any) -> bool:
    """Shape lists equal, ``None`` dim == JSON null == missing."""

    def norm(s: Any) -> tuple:
        if s is None:
            return ("unknown",)
        return tuple("?" if d is None else int(d) for d in s)

    return norm(a) == norm(b)


def _compare_meta(py: dict, rs: dict) -> dict:
    diffs: list[str] = []

    def inputs_of(doc: dict) -> list[tuple[str, Any]]:
        return [
            (i["name"], i.get("shape")) for i in doc.get("inputs", [])
        ]

    pi, ri = inputs_of(py), inputs_of(rs)
    if [n for n, _ in pi] != [n for n, _ in ri]:
        diffs.append(f"input names: {pi} != {ri}")
    else:
        for (pn, ps), (_, rs_) in zip(pi, ri, strict=True):
            if not _shape_eq(ps, rs_):
                diffs.append(f"input {pn} shape: {ps} != {rs_}")

    pd, rd = py.get("input_dtypes", {}), rs.get("input_dtypes", {})
    if pd != rd:
        diffs.append(f"input_dtypes: {pd} != {rd}")

    pl, rl = py.get("leaves", {}), rs.get("leaves", {})
    if set(pl) != set(rl):
        diffs.append(
            f"leaf name sets differ: py-only={sorted(set(pl) - set(rl))[:8]} "
            f"rs-only={sorted(set(rl) - set(pl))[:8]}"
        )
    for name in sorted(set(pl) & set(rl)):
        p, r = pl[name], rl[name]
        if p["dtype"] != r.get("dtype"):
            diffs.append(
                f"leaf {name} dtype: {p['dtype']} != {r.get('dtype')}"
            )
        if tuple(p["shape"]) != tuple(r.get("shape") or ()):
            diffs.append(
                f"leaf {name} shape: {p['shape']} != {r.get('shape')}"
            )
        if p["sha256"] != r.get("sha256"):
            diffs.append(f"leaf {name} payload sha256 differs")
    return {"ok": not diffs, "diffs": diffs}


# ---------------------------------------------------------------------------
#  Semantic evaluator — numpy over the decoded mlopt terms
# ---------------------------------------------------------------------------


class _Unevaluated(Exception):
    """An op/leaf outside the evaluator's coverage — honest, not wrong."""


_ERF = np.vectorize(math.erf)


def _attr(at: dict, name: str, default: Any = None) -> Any:
    """Read one attr off an already-extracted attrs dict."""
    return (at or {}).get(name, default)


def _dims(at: dict, name: str = "dim") -> Any:
    d = at.get(name)
    if d is None:
        return None
    if isinstance(d, (int, np.integer)) and not isinstance(d, bool):
        return int(d)
    return tuple(int(x) for x in d)


def _f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _reshape(x: np.ndarray, shape: Any) -> np.ndarray:
    """IR ``reshape`` — ``shape`` may carry 0 (copy) and -1 (infer)."""
    shape = [int(s) for s in shape]
    for i, s in enumerate(shape):
        if s == 0:
            shape[i] = x.shape[i] if i < x.ndim else 0
    return np.reshape(x, shape)


def _expand_shape(shape: Any, x: np.ndarray) -> tuple:
    """``expand``/``reshape`` shapes may carry -1 → concrete extents."""
    shape = [int(s) for s in shape]
    if -1 in shape:
        known = 1
        for s in shape:
            if s > 0:
                known *= s
        shape[shape.index(-1)] = x.size // known
    # 0/-1 already resolved; -2 etc. are invalid — let numpy refuse.
    return tuple(shape)


def _pad_at(x: np.ndarray, at: dict) -> np.ndarray:
    pads = tuple(int(p) for p in _attr_at_seq(at, "pad"))
    mode = _attr(at, "mode", "constant")
    n = len(pads) // 2
    # ATen order: (last-dim begin, last-dim end, …, first-dim …).
    width = [
        (pads[2 * (n - 1 - i)], pads[2 * (n - 1 - i) + 1])
        for i in range(n)
    ]
    if mode == "replicate":
        return np.pad(x, width, mode="edge")
    if mode == "circular":
        return np.pad(x, width, mode="wrap")
    if mode == "constant":
        val = _attr(at, "value", 0.0)
        return np.pad(x, width, mode="constant", constant_values=val)
    raise _Unevaluated(f"pad mode {mode!r}")


def _attr_at_seq(at: dict, name: str) -> tuple:
    v = at.get(name)
    if v is None:
        raise _Unevaluated(f"missing attr {name!r}")
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return (int(v),)
    return tuple(int(x) for x in v)


def _slice_at(x: np.ndarray, at: dict) -> np.ndarray:
    dim = int(_attr(at, "dim", 0))
    start = int(_attr(at, "start", 0))
    step = int(_attr(at, "step", 1))
    end = _attr(at, "end", None)
    n = x.shape[dim]
    # ONNX open-bound sentinels the lifter may pass through.
    if end is None or end >= 2**62:
        end = n if step > 0 else -1
    if start <= -(2**62):
        start = 0 if step > 0 else n - 1
    # Normalize negatives (Python-slice semantics match ONNX here for
    # the in-range cases the fixtures exercise).
    sl = [slice(None)] * x.ndim
    sl[dim] = slice(start, end, step)
    return x[tuple(sl)]


def _topk(x: np.ndarray, at: dict):
    k = int(_attr(at, "k", 1))
    dim = int(_attr(at, "dim", -1))
    largest = bool(_attr(at, "largest", 1))
    d = dim % x.ndim
    moved = np.moveaxis(x, d, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    order = np.argsort(flat, axis=-1)
    if largest:
        order = order[..., ::-1]
    sel = order[..., :k]
    vals = np.take_along_axis(flat, sel, axis=-1)
    vals = vals.reshape(*moved.shape[:-1], k)
    vals = np.moveaxis(vals, -1, d)
    idx = np.moveaxis(sel.reshape(*moved.shape[:-1], k), -1, d)
    return vals, idx.astype(np.int64)


def _unary(op: str, x: np.ndarray, at: dict) -> np.ndarray:
    if op == "neg":
        return -x
    if op == "abs":
        return np.abs(x)
    if op == "sign":
        return np.sign(x)
    if op == "floor":
        return np.floor(x)
    if op == "ceil":
        return np.ceil(x)
    if op == "round":
        return np.round(x)
    if op == "sqrt":
        return np.sqrt(x)
    if op == "exp":
        return np.exp(x)
    if op == "log":
        return np.log(x)
    if op == "reciprocal":
        return 1.0 / x
    if op == "sin":
        return np.sin(x)
    if op == "cos":
        return np.cos(x)
    if op == "tan":
        return np.tan(x)
    if op == "asin":
        return np.arcsin(x)
    if op == "acos":
        return np.arccos(x)
    if op == "atan":
        return np.arctan(x)
    if op == "sinh":
        return np.sinh(x)
    if op == "cosh":
        return np.cosh(x)
    if op == "tanh":
        return np.tanh(x)
    if op == "asinh":
        return np.arcsinh(x)
    if op == "acosh":
        return np.arccosh(x)
    if op == "atanh":
        return np.arctanh(x)
    if op == "erf":
        return _ERF(np.asarray(x, dtype=np.float64))
    if op == "relu":
        return np.maximum(x, 0)
    if op == "sigmoid":
        return 1.0 / (1.0 + np.exp(-_f32(x)))
    if op == "softsign":
        return x / (1.0 + np.abs(x))
    if op == "mish":
        return x * np.tanh(np.logaddexp(0.0, x))
    if op == "softplus":
        return np.logaddexp(0.0, x)
    if op == "hardswish":
        return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0
    if op == "isnan":
        return np.isnan(x)
    if op == "isinf":
        return np.isinf(x)
    if op == "isposinf":
        return x == np.inf
    if op == "isneginf":
        return x == -np.inf
    if op == "logical_not":
        return ~np.asarray(x, dtype=bool)
    if op == "square":
        return x * x
    if op == "rsqrt":
        return 1.0 / np.sqrt(x)
    if op == "leaky_relu":
        return np.where(x > 0, x, _attr(at, "negative_slope", 0.01) * x)
    if op == "elu":
        a = float(_attr(at, "alpha", 1.0))
        return np.where(x > 0, x, a * (np.exp(x) - 1.0))
    if op == "celu":
        a = float(_attr(at, "alpha", 1.0))
        return np.maximum(x, 0.0) + a * (
            np.exp(np.minimum(x / a, 0.0)) - 1.0
        )
    if op == "selu":
        # IR selu: 1.0507 * (elu(x,1.6733) clipped below at -1.7581)
        return 1.0507009873554805 * np.maximum(
            np.where(x > 0, x, 1.6732632423543772 * (np.exp(x) - 1.0)),
            -1.7580993408473766,
        )
    if op == "hardsigmoid":
        return np.clip(x / 6.0 + 0.5, 0.0, 1.0)
    if op == "gelu":
        approx = _attr(at, "approximate", None)
        xf = _f32(x)
        if approx == "tanh":
            return (
                0.5
                * xf
                * (
                    1.0
                    + np.tanh(
                        math.sqrt(2.0 / math.pi)
                        * (xf + 0.044715 * xf**3)
                    )
                )
            )
        return 0.5 * xf * (1.0 + _ERF(xf / math.sqrt(2.0)))
    if op == "clone" or op == "identity":
        return x
    if op == "dropout":
        # Eval mode: identity (mask output is refused at lift time).
        return x
    raise _Unevaluated(f"unary op {op!r}")


def _binary(op: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if op == "add":
        return x + y
    if op == "sub":
        return x - y
    if op == "mul":
        return x * y
    if op == "div":
        return x / y
    if op == "pow":
        return np.power(x, y)
    if op == "eq":
        return x == y
    if op == "ne":
        return x != y
    if op == "lt":
        return x < y
    if op == "le":
        return x <= y
    if op == "gt":
        return x > y
    if op == "ge":
        return x >= y
    if op == "maximum":
        return np.maximum(x, y)
    if op == "minimum":
        return np.minimum(x, y)
    if op == "fmod":
        return np.fmod(x, y)
    if op == "remainder":
        return np.remainder(x, y)
    if op == "logical_and":
        return np.logical_and(x, y)
    if op == "logical_or":
        return np.logical_or(x, y)
    if op == "logical_xor":
        return np.logical_xor(x, y)
    if op == "bitwise_and":
        return np.bitwise_and(x, y)
    if op == "bitwise_or":
        return np.bitwise_or(x, y)
    if op == "bitwise_xor":
        return np.bitwise_xor(x, y)
    raise _Unevaluated(f"binary op {op!r}")


_EVAL_MISS = object()


def _eval(t: Any, env: dict, _memo: dict | None = None) -> Any:
    """Evaluate an mlopt term on numpy values (see module docstring).

    Memoized on ``id`` — ``Op.make`` interning means a lifted term is
    a DAG, and an unmemoized walk re-evaluates shared subtrees
    exponentially (whisper's encoder term is ~240 nodes but deeply
    shared).
    """
    if _memo is None:
        _memo = {}
    if isinstance(t, Op):
        key = id(t)
        hit = _memo.get(key, _EVAL_MISS)
        if hit is not _EVAL_MISS:
            return hit
        out = _eval_node(t, env, _memo)
        _memo[key] = out
        return out
    return _eval_node(t, env, _memo)


def _eval_node(t: Any, env: dict, memo: dict) -> Any:
    """One node's evaluation; children go through the memoized ``_eval``."""
    if isinstance(t, Var):
        return env["v:" + t.name]
    if isinstance(t, Param):
        return env["p:" + t.name]
    if isinstance(t, Const):
        v = t.value
        if isinstance(v, str):
            raise _Unevaluated(f"string const {v!r}")
        return np.asarray(v)
    if not isinstance(t, Op):
        raise _Unevaluated(f"leaf {t!r}")
    a = [_eval(x, env, memo) for x in t.args]
    op = t.op
    at = t.attrs or {}

    if op in (
        "neg",
        "abs",
        "sign",
        "floor",
        "ceil",
        "round",
        "sqrt",
        "exp",
        "log",
        "reciprocal",
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
        "erf",
        "relu",
        "sigmoid",
        "softsign",
        "mish",
        "softplus",
        "hardswish",
        "isnan",
        "isinf",
        "isposinf",
        "isneginf",
        "logical_not",
        "square",
        "rsqrt",
        "leaky_relu",
        "elu",
        "celu",
        "selu",
        "hardsigmoid",
        "gelu",
        "clone",
        "identity",
        "dropout",
    ):
        return _unary(op, np.asarray(a[0]), at)
    if op in (
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "maximum",
        "minimum",
        "fmod",
        "remainder",
        "logical_and",
        "logical_or",
        "logical_xor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    ):
        return _binary(op, np.asarray(a[0]), np.asarray(a[1]))
    if op == "where":
        return np.where(np.asarray(a[0], dtype=bool), a[1], a[2])
    if op in ("clamp", "clamp_min", "clamp_max"):
        x = np.asarray(a[0])
        lo = _attr(at, "min", None)
        hi = _attr(at, "max", None)
        if op == "clamp_min":
            lo = _attr(at, "min", None)
        if op == "clamp_max":
            hi = _attr(at, "max", None)
        return np.clip(
            x,
            -np.inf if lo is None else lo,
            np.inf if hi is None else hi,
        )
    if op == "to":
        name = _attr(at, "dtype")
        dt = DTYPE_NAMES.get(name)
        if dt is None:
            raise _Unevaluated(f"to dtype {name!r}")
        return np.asarray(a[0]).astype(dt)
    if op == "matmul":
        return np.matmul(np.asarray(a[0]), np.asarray(a[1]))
    if op == "linear":
        out = np.asarray(a[0]) @ np.swapaxes(np.asarray(a[1]), -1, -2)
        if len(a) > 2:
            out = out + a[2]
        return out
    if op == "reshape":
        return _reshape(np.asarray(a[0]), _attr_at_seq(at, "shape"))
    if op == "flatten":
        x = np.asarray(a[0])
        sd = int(_attr(at, "start_dim", 0))
        ed = int(_attr(at, "end_dim", x.ndim - 1))
        sd %= x.ndim
        ed %= x.ndim
        new = [
            *x.shape[:sd],
            int(np.prod(x.shape[sd : ed + 1])),
            *x.shape[ed + 1 :],
        ]
        return x.reshape(new)
    if op == "transpose":
        x = np.asarray(a[0])
        d0 = int(_attr(at, "dim0", 0)) % x.ndim
        d1 = int(_attr(at, "dim1", 1)) % x.ndim
        axes = list(range(x.ndim))
        axes[d0], axes[d1] = axes[d1], axes[d0]
        return np.transpose(x, axes)
    if op == "permute":
        x = np.asarray(a[0])
        d = _attr(at, "dim", None)
        if d is None:
            # ONNX Transpose with no perm attr reverses the dims.
            return np.transpose(x, axes=tuple(range(x.ndim))[::-1])
        return np.transpose(x, _attr_at_seq(at, "dim"))
    if op == "unsqueeze":
        return np.expand_dims(
            np.asarray(a[0]), int(_attr(at, "dim", 0))
        )
    if op == "squeeze":
        x = np.asarray(a[0])
        d = _attr(at, "dim", None)
        if d is None:
            return np.squeeze(x)
        return np.squeeze(x, axis=int(d) % x.ndim)
    if op == "expand":
        return np.broadcast_to(
            np.asarray(a[0]),
            _expand_shape(_attr_at_seq(at, "shape"), np.asarray(a[0])),
        )
    if op == "repeat":
        return np.tile(np.asarray(a[0]), _attr_at_seq(at, "shape"))
    if op == "concat":
        return np.concatenate(
            [np.asarray(x) for x in a], axis=int(_attr(at, "dim", 0))
        )
    if op == "select":
        x = np.asarray(a[0])
        dim = int(_attr(at, "dim", 0)) % x.ndim
        idx = int(_attr(at, "index", 0))
        idx %= x.shape[dim]
        return np.take(x, idx, axis=dim)
    if op == "slice":
        return _slice_at(np.asarray(a[0]), at)
    if op == "pad":
        return _pad_at(np.asarray(a[0]), at)
    if op == "flip":
        axes = _attr_at_seq(at, "dim")
        return np.flip(np.asarray(a[0]), axis=axes if axes else None)
    if op == "split":
        x = np.asarray(a[0])
        dim = int(_attr(at, "dim", 0))
        sizes = _attr_at_seq(at, "sizes")
        idx = int(_attr(at, "index", 0))
        parts = np.split(x, np.cumsum(sizes)[:-1], axis=dim)
        return parts[idx]
    if op == "getitem":
        # Projects output `index` of a multi-output term (topk, …) —
        # the arg evaluates to a tuple of outputs.
        v = a[0]
        if not isinstance(v, (tuple, list)):
            raise _Unevaluated("getitem of a single-output term")
        return v[int(_attr(at, "index", 0))]
    if op == "topk":
        return _topk(np.asarray(a[0]), at)
    if op == "gather":
        # IR `gather` is ONNX GatherElements — torch.gather semantics:
        # out[p] = x[p[:axis] + (idx[p],) + p[axis+1:]], output shape
        # = idx shape, idx dims <= x dims off-axis.
        x = np.asarray(a[0])
        idx = np.asarray(a[1])
        axis = int(_attr(at, "dim", 0)) % x.ndim
        if idx.ndim != x.ndim:
            raise _Unevaluated("gather rank mismatch")
        grids = np.indices(idx.shape)
        coords = [grids[d] for d in range(x.ndim)]
        coords[axis] = idx
        return x[tuple(coords)]
    if op == "index_select":
        return np.take(
            np.asarray(a[0]),
            np.asarray(a[1]).reshape(-1),
            axis=int(_attr(at, "dim", 0)),
        )
    if op == "embedding":
        return np.take(np.asarray(a[0]), np.asarray(a[1]), axis=0)
    if op == "cumsum":
        return np.cumsum(
            np.asarray(a[0]), axis=int(_attr(at, "dim", -1))
        )
    if op == "triu":
        return np.triu(
            np.asarray(a[0]), k=int(_attr(at, "diagonal", 0))
        )
    if op == "tril":
        return np.tril(
            np.asarray(a[0]), k=int(_attr(at, "diagonal", 0))
        )
    if op == "arange":
        dt_name = _attr(at, "dtype", None)
        dt = DTYPE_NAMES.get(dt_name) if dt_name else None
        return np.arange(
            float(_attr(at, "arg0", 0)),
            float(_attr(at, "arg1", 0)),
            float(_attr(at, "arg2", 1)),
            dtype=dt,
        )
    if op == "full":
        dt_name = _attr(at, "dtype", "float32")
        dt = DTYPE_NAMES.get(dt_name, np.float32)
        return np.full(
            _attr_at_seq(at, "shape"),
            _attr(at, "fill_value", 0.0),
            dtype=dt,
        )
    if op == "one_hot":
        x = np.asarray(a[0], dtype=np.int64)
        n = int(_attr(at, "num_classes", 0))
        return (x[..., None] == np.arange(n)).astype(np.int64)
    if op == "nonzero":
        # ONNX NonZero is ONE output: a (rank, nnz) index tensor.
        return np.stack(np.nonzero(np.asarray(a[0])))
    if op == "pack":
        return tuple(a)
    if op in ("sum", "mean", "prod", "amax", "amin"):
        x = np.asarray(a[0], dtype=np.float64)
        dim = _dims(at)
        keep = bool(_attr(at, "keepdim", 0))
        fn = {
            "sum": np.sum,
            "mean": np.mean,
            "prod": np.prod,
            "amax": np.max,
            "amin": np.min,
        }[op]
        if dim is None:
            out = fn(x)
            return (
                np.asarray(out).reshape((1,) * x.ndim) if keep else out
            )
        dims = (
            (dim,)
            if isinstance(dim, int)
            else tuple(d % x.ndim for d in dim)
        )
        return fn(x, axis=dims, keepdims=keep)
    if op in ("argmax", "argmin"):
        x = np.asarray(a[0])
        dim = int(_attr(at, "dim", 0)) % x.ndim
        keep = bool(_attr(at, "keepdim", 0))
        fn = np.argmax if op == "argmax" else np.argmin
        out = fn(x, axis=dim)
        if keep:
            out = np.expand_dims(out, dim)
        return out.astype(np.int64)
    if op == "linalg_vector_norm":
        x = _f32(np.asarray(a[0]))
        ord_ = _attr(at, "ord", 2)
        dim = _dims(at)
        keep = bool(_attr(at, "keepdim", 0))
        axes = (
            None
            if dim is None
            else (
                (dim,)
                if isinstance(dim, int)
                else tuple(d % x.ndim for d in dim)
            )
        )
        if ord_ == 1:
            out = np.sum(np.abs(x), axis=axes, keepdims=keep)
        elif ord_ == 2:
            out = np.sqrt(np.sum(x * x, axis=axes, keepdims=keep))
        else:
            raise _Unevaluated(f"linalg_vector_norm ord {ord_!r}")
        return out
    if op in ("softmax", "log_softmax"):
        x = _f32(np.asarray(a[0]))
        dim = int(_attr(at, "dim", -1)) % x.ndim
        z = x - np.max(x, axis=dim, keepdims=True)
        e = np.exp(z)
        p = e / np.sum(e, axis=dim, keepdims=True)
        if op == "log_softmax":
            return np.log(p)
        return p
    if op == "layer_norm":
        x = _f32(np.asarray(a[0]))
        scale = np.asarray(a[1]) if len(a) > 1 else 1.0
        bias = np.asarray(a[2]) if len(a) > 2 else 0.0
        dim = _attr_at_seq(at, "dim")
        nd = len(dim)
        axes = tuple(range(x.ndim - nd, x.ndim))
        eps = float(_attr(at, "eps", 1e-5))
        mu = np.mean(x, axis=axes, keepdims=True)
        var = np.mean((x - mu) ** 2, axis=axes, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * scale + bias
    if op == "batch_norm":
        x = _f32(np.asarray(a[0]))
        scale = np.asarray(a[1])
        bias = np.asarray(a[2])
        mean = np.asarray(a[3])
        var = np.asarray(a[4])
        eps = float(_attr(at, "eps", 1e-5))
        shape = [1] * x.ndim
        shape[1] = x.shape[1]
        return (x - mean.reshape(shape)) / np.sqrt(
            var.reshape(shape) + eps
        ) * scale.reshape(shape) + bias.reshape(shape)
    if op == "instance_norm":
        x = _f32(np.asarray(a[0]))
        scale = np.asarray(a[1]) if len(a) > 1 else 1.0
        bias = np.asarray(a[2]) if len(a) > 2 else 0.0
        axes = tuple(i for i in range(x.ndim) if i != 1)
        eps = float(_attr(at, "eps", 1e-5))
        mu = np.mean(x, axis=axes, keepdims=True)
        var = np.mean((x - mu) ** 2, axis=axes, keepdims=True)
        ch = [1] * x.ndim
        ch[1] = x.shape[1]
        return (x - mu) / np.sqrt(var + eps) * np.asarray(
            scale
        ).reshape(ch) + np.asarray(bias).reshape(ch)
    if op == "group_norm":
        x = _f32(np.asarray(a[0]))
        scale = np.asarray(a[1]) if len(a) > 1 else 1.0
        bias = np.asarray(a[2]) if len(a) > 2 else 0.0
        g = int(_attr(at, "num_groups", 1))
        eps = float(_attr(at, "eps", 1e-5))
        n, c = x.shape[0], x.shape[1]
        xg = x.reshape(n, g, c // g, *x.shape[2:])
        axes = tuple(range(2, xg.ndim))
        mu = np.mean(xg, axis=axes, keepdims=True)
        var = np.mean((xg - mu) ** 2, axis=axes, keepdims=True)
        y = (xg - mu) / np.sqrt(var + eps)
        y = y.reshape(n, c, *x.shape[2:])
        ch = [1] * x.ndim
        ch[1] = c
        return y * np.asarray(scale).reshape(ch) + np.asarray(
            bias
        ).reshape(ch)
    if op == "max_pool2d":
        return _max_pool2d(np.asarray(a[0]), at)
    if op == "einsum":
        # ONNX Einsum's equation attr is np.einsum syntax.
        return np.einsum(
            _attr(at, "equation"), *[np.asarray(x) for x in a]
        )
    if op == "conv1d":
        return _conv1d(
            np.asarray(a[0]),
            np.asarray(a[1]),
            np.asarray(a[2]) if len(a) > 2 else None,
            at,
        )
    raise _Unevaluated(f"op {op!r}")


def _conv1d(
    x: np.ndarray, w: np.ndarray, b: Any, at: dict
) -> np.ndarray:
    """NCHW→(N,C,L) conv1d via im2col — eval-mode, groups supported."""
    if x.ndim != 3 or w.ndim != 3:
        raise _Unevaluated("conv1d rank")
    n, c, _length = x.shape
    o, cg, k = w.shape
    groups = int(_attr(at, "groups", 1))
    if c % groups or o % groups or cg != c // groups:
        raise _Unevaluated("conv1d group split")
    stride = int(_attr_at_seq(at, "stride")[0])
    dil = int(_attr_at_seq(at, "dilation")[0])
    pads = _attr_at_seq(at, "padding")
    # IR conv padding is one symmetric value per spatial dim.
    pad_l = pad_r = int(pads[0])
    xp = np.pad(x, ((0, 0), (0, 0), (pad_l, pad_r)))
    lp = xp.shape[2]
    lo = (lp - dil * (k - 1) - 1) // stride + 1
    if lo <= 0:
        raise _Unevaluated("conv1d empty output")
    pos = np.arange(lo)[:, None] * stride + np.arange(k)[None, :] * dil
    win = xp[:, :, pos]  # (N, C, Lo, k)
    og = o // groups
    out = np.empty((n, o, lo), dtype=np.result_type(x, w, np.float32))
    for g in range(groups):
        wg = w[g * og : (g + 1) * og]  # (Og, Cg, k)
        xg = win[:, g * cg : (g + 1) * cg]  # (N, Cg, Lo, k)
        out[:, g * og : (g + 1) * og] = np.einsum(
            "nclk,ock->nol", xg, wg, optimize=True
        )
    if b is not None:
        out = out + np.asarray(b).reshape(1, o, 1)
    return out


def _max_pool2d(x: np.ndarray, at: dict) -> np.ndarray:
    """NCHW max-pool2d — kernel/stride/padding/dilation/ceil_mode attrs."""
    if x.ndim != 4:
        raise _Unevaluated("max_pool2d rank")
    k = _attr_at_seq(at, "kernel_size")
    s = _attr_at_seq(at, "stride") or (1, 1)
    p = _attr_at_seq(at, "padding") or (0, 0)
    d = _attr_at_seq(at, "dilation") or (1, 1)
    ceil_mode = bool(_attr(at, "ceil_mode", 0))
    kh, kw = int(k[0]), int(k[1])
    sh, sw = int(s[0]), int(s[1])
    ph, pw = int(p[0]), int(p[1])
    dh, dw = int(d[0]), int(d[1])
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    hp, wp = xp.shape[2], xp.shape[3]
    eff_h = dh * (kh - 1) + 1
    eff_w = dw * (kw - 1) + 1
    if ceil_mode:
        ho = -(-(hp - eff_h) // sh) + 1
        wo = -(-(wp - eff_w) // sw) + 1
    else:
        ho = (hp - eff_h) // sh + 1
        wo = (wp - eff_w) // sw + 1
    if ho <= 0 or wo <= 0:
        raise _Unevaluated("max_pool2d empty output")
    ph_idx = (
        np.arange(ho)[:, None] * sh + np.arange(kh)[None, :] * dh
    )  # (Ho, kh)
    pw_idx = (
        np.arange(wo)[:, None] * sw + np.arange(kw)[None, :] * dw
    )  # (Wo, kw)
    # Clip to the padded bounds — ONNX ceil_mode skips windows that
    # would start inside the padding; clipping is the closest cheap
    # approximation (fixtures exercise floor mode).
    ph_idx = np.clip(ph_idx, 0, hp - 1)
    pw_idx = np.clip(pw_idx, 0, wp - 1)
    win = xp[:, :, ph_idx[:, :, None, None], pw_idx[None, None, :, :]]
    # win: (N, C, Ho, kh, Wo, kw) -> (N, C, Ho, Wo, kh, kw)
    win = np.transpose(win, (0, 1, 2, 4, 3, 5))
    return win.max(axis=(-2, -1))


def _make_env(
    inputs: list[dict], leaf_arrays: dict[str, np.ndarray]
) -> dict:
    """Shared random inputs + leaf payloads for one model."""
    rng = np.random.default_rng(20260715)
    env: dict[str, np.ndarray] = {}
    for inp in inputs:
        name = inp["name"]
        shape = inp.get("shape") or ()
        shape = tuple(2 if d is None else int(d) for d in shape)
        dt_name = inp.get("dtype") or inp.get("input_dtype")
        dt = DTYPE_NAMES.get(dt_name, np.float32)
        if dt == np.dtype(bool):
            arr = rng.integers(0, 2, shape).astype(bool)
        elif np.issubdtype(np.dtype(dt), np.integer):
            arr = rng.integers(-3, 4, shape).astype(dt)
        else:
            arr = rng.standard_normal(shape).astype(
                np.float32 if dt == np.dtype(np.float16) else dt
            )
        env["v:" + name] = arr
    for name, arr in leaf_arrays.items():
        env["p:" + name] = arr
    return env


def _compare_eval(
    model: Path, py: dict, rs: dict, workdir: Path
) -> dict:
    """Decode both terms and evaluate on shared inputs."""
    try:
        t_py = term_from_json(py["root"])
        t_rs = term_from_json(rs["root"])
    except Exception as e:
        return {"verdict": "decode-error", "detail": str(e)}

    # Leaf payloads: Python side re-derives arrays from lift_model;
    # Rust side decodes the safetensors buffer.
    try:
        res = lift_model(str(model))
        leaf_arrays = {k: np.asarray(v) for k, v in res.leaves.items()}
    except Exception as e:
        return {"verdict": "py-releaf-error", "detail": str(e)}
    st = workdir / f"{model.stem}.safetensors"
    if not st.exists():
        return {
            "verdict": "eval-error",
            "side": "rs",
            "detail": f"safetensors side channel missing: {st}",
        }
    try:
        tensors = _parse_safetensors(st)
    except Exception as e:
        return {
            "verdict": "eval-error",
            "side": "rs",
            "detail": f"safetensors parse failed: {e}",
        }
    # Decode the safetensors payloads once — header + raw buffer.
    buf = st.read_bytes()
    (hlen,) = struct.unpack("<Q", buf[:8])
    header = json.loads(buf[8 : 8 + hlen])
    base = 8 + hlen
    rs_arrays: dict[str, np.ndarray] = {}
    for name, meta in tensors.items():
        dt = _ST_DT.get(meta["dtype"])
        if dt is None:
            continue
        s, e = header[name]["data_offsets"]
        arr = np.frombuffer(
            buf[base + s : base + e], dtype=np.dtype(dt)
        )
        rs_arrays[name] = arr.reshape(meta["shape"])

    inputs = py.get("inputs", [])
    # input dtype names ride the side channel, not the inputs list.
    dts = py.get("input_dtypes", {})
    inputs = [{**i, "dtype": dts.get(i["name"])} for i in inputs]

    env_py = _make_env(inputs, leaf_arrays)
    env_rs = _make_env(inputs, rs_arrays)
    try:
        out_py = _eval(t_py, env_py)
    except _Unevaluated as e:
        return {
            "verdict": "unevaluated",
            "side": "py",
            "detail": str(e),
        }
    except Exception as e:
        return {
            "verdict": "eval-error",
            "side": "py",
            "detail": f"{type(e).__name__}: {e}",
        }
    try:
        out_rs = _eval(t_rs, env_rs)
    except _Unevaluated as e:
        return {
            "verdict": "unevaluated",
            "side": "rs",
            "detail": str(e),
        }
    except Exception as e:
        return {
            "verdict": "eval-error",
            "side": "rs",
            "detail": f"{type(e).__name__}: {e}",
        }

    diff = _result_diff(out_py, out_rs)
    return {
        "verdict": "equal" if diff is None else "diverge",
        "detail": diff,
    }


def _result_diff(a: Any, b: Any, path: str = "") -> str | None:
    """First difference between two eval results, or ``None``."""
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        if not (
            isinstance(a, (tuple, list))
            and isinstance(b, (tuple, list))
        ):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            d = _result_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape:
        return f"{path}: shape {aa.shape} != {bb.shape}"
    if aa.dtype.kind == "f" or bb.dtype.kind == "f":
        if not np.allclose(
            aa.astype(np.float64),
            bb.astype(np.float64),
            rtol=1e-4,
            atol=1e-5,
            equal_nan=True,
        ):
            return f"{path}: values differ (max abs diff "
            f"{np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64))):.3e})"
        return None
    if not np.array_equal(aa, bb):
        return f"{path}: values differ"
    return None


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _corpus() -> list[tuple[str, Path]]:
    """(label, path) pairs — the spike's three tiers."""
    out: list[tuple[str, Path]] = []
    whisper = Path(
        os.environ.get("SOURCE_ONNX_WHISPER", _WHISPER_DEFAULT)
    )
    if whisper.exists():
        out.append(("whisper_tiny_encoder", whisper))
    resnet = Path(os.environ.get("SOURCE_ONNX_RESNET", _RESNET_DEFAULT))
    if resnet.exists():
        out.append(("light_resnet50", resnet))
    fixdir = Path(
        os.environ.get("SOURCE_ONNX_FIXTURES", _FIXTURES_DEFAULT)
    )
    if fixdir.is_dir():
        for f in sorted(fixdir.glob("*.onnx")):
            out.append((f"fixture:{f.stem}", f))
    return out


def _verdict(
    model: str,
    py: dict,
    rs: dict,
    term: dict,
    meta: dict,
    ev: dict | None,
) -> str:
    if "refusal" in py and "refusal" in rs:
        pf, rf = py["refusal"], rs["refusal"]
        same_family = pf.get("family") == rf.get("family")
        same_op = pf.get("op") in (None, rf.get("op")) or rf.get(
            "op"
        ) in (None, pf.get("op"))
        return (
            "refuse-parity"
            if same_family and same_op
            else "refuse-divergence"
        )
    if "refusal" in py or "refusal" in rs or "error" in rs:
        return "lift-divergence"
    if not meta["ok"]:
        return "meta-divergence"
    if term["verdict"] == "iso":
        return "parity" + ("+eval:" + ev["verdict"] if ev else "")
    return "term-divergence" + ("+eval:" + ev["verdict"] if ev else "")


def run_one(
    label: str, path: Path, workdir: Path, do_eval: bool
) -> dict:
    py = _py_lift(path)
    rs = _rs_lift(path, workdir)
    rec: dict[str, Any] = {"model": label, "path": str(path)}
    term: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    ev = None
    if py.get("ok") and rs.get("ok"):
        term = _compare_terms(py, rs)
        meta = _compare_meta(py, rs)
        if do_eval:
            ev = _compare_eval(path, py, rs, workdir)
    rec.update(
        {
            "py": "ok"
            if py.get("ok")
            else py.get("refusal", py.get("error")),
            "rs": "ok"
            if rs.get("ok")
            else rs.get("refusal", rs.get("error")),
            "term": term,
            "meta": meta,
            "eval": ev,
        }
    )
    rec["verdict"] = _verdict(label, py, rs, term, meta, ev)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model", help="substring filter on the corpus label"
    )
    ap.add_argument(
        "--no-eval", action="store_true", help="skip semantic eval"
    )
    ap.add_argument(
        "--json", type=Path, help="write the full report JSON here"
    )
    args = ap.parse_args()

    corpus = _corpus()
    if args.model:
        corpus = [c for c in corpus if args.model in c[0]]
    if not corpus:
        print("no corpus models found", file=sys.stderr)
        return 2

    workdir = Path(
        tempfile.mkdtemp(
            prefix="onnx_diff_", dir=Path.home() / ".cache"
        )
    )
    (workdir).mkdir(parents=True, exist_ok=True)

    records = []
    for label, path in corpus:
        try:
            rec = run_one(label, path, workdir, not args.no_eval)
        except Exception as e:
            rec = {
                "model": label,
                "path": str(path),
                "verdict": "harness-error",
                "detail": f"{type(e).__name__}: {e}",
            }
        records.append(rec)
        t = rec.get("term") or {}
        m = rec.get("meta") or {}
        e = rec.get("eval") or {}
        print(
            f"{rec['verdict']:<28} {label:<32} "
            f"term={t.get('verdict', '-')} nodes={t.get('py_nodes', '-')}/{t.get('rs_nodes', '-')} "
            f"meta={'ok' if m.get('ok', True) else 'DIFF'} "
            f"eval={e.get('verdict', '-')}"
        )
        for d in (m.get("diffs") or [])[:4]:
            print(f"    meta: {d}")
        for d in (t.get("first_diffs") or [])[:4]:
            print(f"    term: {d}")
        if e.get("detail"):
            print(f"    eval: {e['detail']}")

    # Cleanup scratch safetensors (they can be tens of MB).
    for f in workdir.glob("*.safetensors"):
        with contextlib.suppress(OSError):
            f.unlink()

    if args.json:
        args.json.write_text(json.dumps(records, indent=2, default=str))
        print(f"\nreport -> {args.json}")

    bad = [
        r
        for r in records
        if r["verdict"]
        not in ("parity", "parity+eval:equal", "refuse-parity")
    ]
    print(
        f"\n{len(records)} models: "
        f"{sum(1 for r in records if r['verdict'].startswith('parity'))} parity, "
        f"{sum(1 for r in records if r['verdict'] == 'refuse-parity')} refuse-parity, "
        f"{len(bad)} need review"
    )
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
