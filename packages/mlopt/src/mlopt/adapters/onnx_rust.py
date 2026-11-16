"""``.onnx`` source adapter — the ``source-onnx`` Rust lifter as frontend.

``mlopt optimize model.onnx`` / ``mlopt optimize onnx-rust:model.onnx``
runs the Rust ONNX lifter (``handl-lang/source-onnx``, ADR-0176) as a
subprocess — the same JSON-in/JSON-out bridge the differential harness
(``graphopt:bench/drivers/audit/onnx_lifter_differential.py``) uses —
then decodes the emitted Term-JSON through
:func:`mlopt_core.term_json.term_from_json` and feeds the
pipeline through an :class:`OnnxRustSource` — a
``mlopt_core.ports.Source`` answering ``to_ir`` with the pre-lifted
``(IR, param_values)``, exactly the seam ``adapters/hnl.py``'s
``HnlSource`` uses.  ``Optimizer.search`` then runs the SAME
saturate → extract → lower → verify flow a ``torch.export``-lifted
module runs; nothing downstream knows the frontend was ONNX.

Honest bounds, stated up front:

* **The lifter is a subprocess, not a library.**  mlopt is Python;
  ``source-onnx`` is a Rust crate.  The adapter resolves the ``lift``
  example binary (``--adapter-opt lift_bin=<path>``, the
  ``SOURCE_ONNX_LIFT`` env var, the sibling checkout's
  ``target/{release,debug}/examples/lift``, the shared cargo target
  dir) and builds it with ``cargo build --example lift`` when a
  ``source-onnx`` checkout is present but unbuilt.  No binary and no
  checkout → a named decline with the build instructions, never a
  silent fallback to the Python ``kit_onnx`` lifter (that oracle
  lives in graphopt's domain packages, which mlopt does not depend
  on — a fallback would hide which producer ran).
* **Refusals surface verbatim.**  The lifter's named refusals
  (``unsupported ONNX op …``, ``onnx import error: …`` — opset outside
  13..=21, non-default domains, the control-flow/random/quantized
  families, in-table attr-fold refusals) arrive as the document's
  ``error`` string and are raised unchanged, so the refusal the user
  sees is the lifter's own.
* **Weights are REAL.**  Initializers and shaped ``Constant`` outputs
  arrive as a safetensors side channel, parsed here into NumPy arrays
  keyed by ``Param`` name — the same ``(IR, leaves)`` pair
  ``kit_onnx.OnnxSource`` returns.  Half-precision payloads bind
  honestly: ``BF16`` decodes losslessly to float32 (bf16 is the high
  half of f32) and ``F16``/``BF16`` param tensors upcast to float32
  at bind time — the verify convention is f32 params against f32
  probes, and a ``c10::Half`` weight against an f32 probe fails
  torch's dtype check outright.
* **Inputs are seeded synthetic probes.**  ONNX graphs declare shapes,
  not data — the adapter mints deterministic ``torch.randn``-family
  inputs (``--adapter-opt seed=N``) in first-use order, like the
  ``.hnl`` adapter's synthetic leaves.  ``--adapter-opt
  inputs='{"name": [dims]}'`` overrides a declared shape (dynamic
  ``dim_param`` extents arrive as ``null`` and MUST be overridden);
  a shape that stays unknown declines at load, naming the input.
  ``--adapter-opt feed=<path.json>`` (the ``.hnl`` ``{"shape","data"}``
  format) supplies real input values instead.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlopt_core.ports import Source
from mlopt_core.term import IR, Op, Param, TensorType, Var

from mlopt.models import ModelSpec, register_adapter

__all__ = ["ADAPTER", "OnnxLift", "OnnxRustSource"]


# ---------------------------------------------------------------------------
#  Source-onnx checkout + ``lift`` binary discovery
# ---------------------------------------------------------------------------

#: mlopt repo root — ``packages/mlopt/src/mlopt/adapters/onnx_rust.py`` → parents[5].
_MLOPT_ROOT = Path(__file__).resolve().parents[5]


def _source_onnx_repo() -> Path | None:
    """The ``source-onnx`` checkout: env override, then the sibling dir."""
    env = os.environ.get("SOURCE_ONNX_REPO")
    if env:
        return Path(env)
    sibling = _MLOPT_ROOT.parent / "source-onnx"
    return sibling if sibling.is_dir() else None


def _bin_candidates(repo: Path | None) -> list[Path]:
    cands: list[Path] = []
    env = os.environ.get("SOURCE_ONNX_LIFT")
    if env:
        cands.append(Path(env))
    if repo is not None:
        cands += [
            repo / "target" / "release" / "examples" / "lift",
            repo / "target" / "debug" / "examples" / "lift",
        ]
    shared = Path.home() / ".cargo" / "shared-target"
    cands += [
        shared / "release" / "examples" / "lift",
        shared / "debug" / "examples" / "lift",
    ]
    return cands


def _find_binary(lift_bin: Any = None) -> Path | None:
    """Resolve the ``lift`` example binary without building.

    An explicit ``lift_bin`` that does not exist resolves to ``None``
    here; :func:`_ensure_binary` turns that into a named error rather
    than silently probing other locations.
    """
    if lift_bin:
        p = Path(str(lift_bin)).expanduser()
        return p if p.is_file() else None
    for cand in _bin_candidates(_source_onnx_repo()):
        if cand.is_file():
            return cand
    return None


def _ensure_binary(lift_bin: Any = None) -> Path:
    """Return the ``lift`` binary, building it when a checkout exists.

    Raises ``ValueError`` naming the honest remedies when no binary
    resolves and no checkout can build one.
    """
    b = _find_binary(lift_bin)
    if b is not None:
        return b
    if lift_bin:
        raise ValueError(f"onnx-rust: lift_bin={lift_bin!r} does not name an executable file")
    repo = _source_onnx_repo()
    if repo is not None and shutil.which("cargo") is not None:
        proc = subprocess.run(
            ["cargo", "build", "--example", "lift"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode == 0:
            b = _find_binary(lift_bin)
            if b is not None:
                return b
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise ValueError(
            f"onnx-rust: cargo build --example lift failed in {repo} "
            f"(rc={proc.returncode}): {tail[-1] if tail else 'no output'}"
        )
    raise ValueError(
        "onnx-rust: the source-onnx `lift` binary was not found — set "
        "--adapter-opt lift_bin=<path> or SOURCE_ONNX_LIFT, or clone "
        "handl-lang/source-onnx next to the mlopt checkout (or point "
        "SOURCE_ONNX_REPO at it) and build with "
        "`cargo build --example lift`"
    )


# ---------------------------------------------------------------------------
#  The lifter subprocess — one .onnx in, one JSON document out
# ---------------------------------------------------------------------------

_LIFT_TIMEOUT_S = 900


def _run_lift(model: Path, binary: Path, workdir: Path) -> dict[str, Any]:
    """Run ``lift <model> --st-out <workdir>/leaves.safetensors``.

    Returns the parsed document.  ``{"ok": false, "error": …}`` —
    the lifter's named refusal — raises ``ValueError`` carrying the
    error string verbatim.
    """
    st_path = workdir / f"{model.stem}.safetensors"
    try:
        proc = subprocess.run(
            [str(binary), str(model), "--st-out", str(st_path)],
            capture_output=True,
            text=True,
            timeout=_LIFT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        raise ValueError(f"onnx-rust: lift timed out (>{_LIFT_TIMEOUT_S}s) on {model}") from e
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"onnx-rust: lift produced non-JSON output (rc={proc.returncode}): "
            f"{proc.stdout[:200]!r} stderr={proc.stderr[:300]!r}"
        ) from e
    if not doc.get("ok"):
        # The lifter's named refusal — surface it verbatim.
        raise ValueError(f"onnx-rust: {doc.get('error', 'unknown lift error')}")
    doc["_safetensors_path"] = str(st_path)
    return doc


# ---------------------------------------------------------------------------
#  safetensors side channel → NumPy leaves (no extra dependency)
# ---------------------------------------------------------------------------

#: safetensors dtype tag → NumPy dtype.  ``BF16`` has no NumPy
#: spelling and is decoded separately (lossless u16 → f32 bit
#: widening) in :func:`_load_safetensors`; every other tag without a
#: mapping still declines by name, like the differential harness's
#: ``_ST_DT``.
_ST_NP_DT: dict[str, str] = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "|i1",
    "U8": "|u1",
    "BOOL": "|b1",
}


def _load_safetensors(path: Path) -> dict[str, Any]:
    """Parse a safetensors file into ``{name: np.ndarray}``.

    The format is an 8-byte LE header length, a JSON header
    ``{name: {dtype, shape, data_offsets}}``, then the raw buffer —
    parsed directly (the same reader the differential harness uses)
    so the adapter needs no ``safetensors`` dependency.
    """
    import numpy as np

    # memmap, not read_bytes: multi-GB payloads (arch: bound
    # checkpoints) must not pin a second full copy in RSS — the
    # per-tensor torch copies in ``_param_tensor`` are the only
    # resident weight bytes, and file-backed pages stay reclaimable.
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    if buf.size < 8:
        raise ValueError(f"onnx-rust: {path} is too small to be safetensors")
    (hlen,) = struct.unpack("<Q", buf[:8].tobytes())
    try:
        header = json.loads(buf[8 : 8 + hlen].tobytes())
    except json.JSONDecodeError as e:
        raise ValueError(f"onnx-rust: {path} has a corrupt safetensors header ({e})") from e
    base = 8 + hlen
    out: dict[str, Any] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype_tag = meta.get("dtype", "")
        s, e = meta["data_offsets"]
        payload = buf[base + s : base + e]
        if dtype_tag == "BF16":
            # bf16 is the high 16 bits of f32 — widen losslessly:
            # u16 << 16 reinterpreted as f32 is exact for every value.
            u16 = np.frombuffer(payload, dtype=np.dtype("<u2"))
            arr = (u16.astype(np.uint32) << np.uint32(16)).view(np.dtype("<f4"))
        else:
            np_dt = _ST_NP_DT.get(dtype_tag)
            if np_dt is None:
                raise ValueError(
                    f"onnx-rust: leaf {name!r} has safetensors dtype "
                    f"{dtype_tag!r} with no NumPy spelling — cannot "
                    "bind it as a param value"
                )
            arr = np.frombuffer(payload, dtype=np.dtype(np_dt))
        out[name] = arr.reshape(tuple(meta["shape"]))
    return out


def _param_tensor(arr: Any) -> Any:
    """One safetensors NumPy array → the torch ``Param`` value.

    torch tensors, not the raw NumPy views — the torch backend's
    lower/verify path calls ``.detach`` and ``np.frombuffer`` views
    are read-only.  Half-precision payloads (``F16`` — and ``BF16``,
    already widened to f32 by :func:`_load_safetensors`) upcast to
    float32: lossless (f16/bf16 ⊂ f32 exactly) and required by the
    verify convention — synthetic/feed probes run f32, and torch
    refuses ``Input type (float) and bias type (c10::Half)`` mixes.
    """
    import numpy as np
    import torch

    t = torch.tensor(np.ascontiguousarray(arr))
    if t.dtype in (torch.float16, torch.bfloat16):
        t = t.to(torch.float32)
    return t


# ---------------------------------------------------------------------------
#  Param-name sanitizing — the torch sink's setattr constraint
# ---------------------------------------------------------------------------


def _sanitize_leaf_name(name: str, used: set[str]) -> str:
    """One Param name → a ``setattr``-safe identifier.

    ONNX initializer names are graph value names (``conv1.weight``,
    ``layers.0.self_attn…``, ``onnx::Add_414``) — legal in the IR but
    not as ``torch.nn.Module`` attribute names, which is how
    ``kit_torch.torch_eval.IRModule`` binds params
    (``setattr(self, name, p)`` rejects ``"."``).  The Python
    ``OnnxSource`` never hit this because it only lowered through
    ``OnnxSink``; the mlopt pipeline lowers through ``IRModule``, so
    the adapter renames Param leaves *before* decode — consistently
    across the term, ``ir.params`` and ``param_values`` — and records
    the mapping in ``meta`` for provenance.
    """
    import re

    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = "_" + safe
    candidate = safe
    n = 2
    while candidate in used:
        candidate = f"{safe}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def _sanitize_param_names(root_json: Any) -> dict[str, str]:
    """Rename every ``param`` leaf's ``name`` in the Term JSON, in place.

    Walks both envelopes (tree ``{"Arr": …}`` and the ``dag.program``
    node table — leaf records are identical in both) and returns the
    ``{original: sanitized}`` map.  Only ``param`` leaves are renamed:
    ``Var`` names are env keys, never ``setattr`` targets.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()

    def rename(rec_name: str) -> str:
        if rec_name not in mapping:
            mapping[rec_name] = _sanitize_leaf_name(rec_name, used)
        return mapping[rec_name]

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            arr = node.get("Arr")
            if isinstance(arr, dict):
                value = arr.get("value")
                if (
                    isinstance(value, dict)
                    and value.get("__mlopt_leaf__") == "param"
                    and isinstance(value.get("name"), str)
                ):
                    value["name"] = rename(value["name"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(root_json)
    return mapping


# ---------------------------------------------------------------------------
#  Term census + leaf/input binding (the .hnl adapter's pattern)
# ---------------------------------------------------------------------------


def _census(term: Any) -> tuple[list[Var], dict[str, Param]]:
    """First-use-ordered ``Var`` leaves and name-keyed ``Param`` leaves."""
    vars_: list[Var] = []
    params: dict[str, Param] = {}
    seen_v: set[str] = set()
    # ``Op.make`` interning makes structurally equal subtrees the same
    # object, so lifted terms are DAGs — a shared subterm (residual
    # streams, the encoder output feeding every decoder cross-attn,
    # tied weights) would otherwise be re-walked once per *path*,
    # i.e. exponentially.  ``id``-memoize like ``term_json``'s
    # ``_has_sharing``: each distinct node is visited exactly once.
    seen_ops: set[int] = set()

    # Iterative, not recursive: unrolled transformer terms run thousands
    # of ops deep and would hit the recursion limit.
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Var):
            if t.name not in seen_v:
                seen_v.add(t.name)
                vars_.append(t)
            continue
        if isinstance(t, Param):
            params.setdefault(t.name, t)
            continue
        if isinstance(t, Op):
            key = id(t)
            if key in seen_ops:
                continue
            seen_ops.add(key)
            # reversed so the pop order matches the old recursive
            # preorder — ``vars_`` stays first-use-ordered.
            stack.extend(reversed(t.args))
    return vars_, params


def _concrete_shape(typ: TensorType) -> tuple[int, ...] | None:
    """The leaf's shape as ints, or ``None`` when it is not concrete."""
    shape = getattr(typ, "shape", None)
    if shape is None:
        return None
    if any(d is None or isinstance(d, bool) or not isinstance(d, int) for d in shape):
        return None
    return tuple(int(d) for d in shape)


def _parse_inputs_opt(raw: Any) -> dict[str, Any]:
    """``--adapter-opt inputs='{"x": [2, 3]}'`` → the shape-override map."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"inputs: not valid JSON ({e})") from e
    if not isinstance(raw, dict):
        raise ValueError(f"inputs: expected a JSON object, got {type(raw).__name__}")
    return dict(raw)


#: ONNX element-type name → torch dtype, for the synthetic input probes.
_TORCH_DT: dict[str, Any] = {}


def _torch_dtype(name: str) -> Any:
    """ONNX dtype name → ``torch.dtype`` (lazy torch import)."""
    import torch

    global _TORCH_DT
    if not _TORCH_DT:
        _TORCH_DT = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
    dt = _TORCH_DT.get(name)
    if dt is None:
        raise ValueError(
            f"onnx-rust: input dtype {name!r} has no torch spelling — "
            "cannot mint a probe input for it"
        )
    return dt


def _probe_input(shape: tuple[int, ...], dtype_name: str, gen: Any) -> Any:
    """One seeded synthetic input tensor — dtype-honest, shape-exact."""
    import torch

    dt = _torch_dtype(dtype_name)
    if dt.is_floating_point:
        return torch.randn(shape, generator=gen, dtype=dt)
    if dt == torch.bool:
        return torch.rand(shape, generator=gen) < 0.5
    # Integer probes: small non-negative values — safe as gather
    # indices, axes and sizes alike.
    return torch.randint(0, 3, shape, generator=gen, dtype=torch.int64).to(dt)


def _feed_tensor(spec: Any, name: str) -> Any:
    """One feed entry → a torch tensor (``{"shape","data"}`` or nested list).

    Same contract as the ``.hnl`` adapter's ``feed=<json>`` — real
    input values where the user has them.
    """
    import torch

    shape: tuple[int, ...] | None = None
    data: Any = spec
    if isinstance(spec, dict):
        if "shape" not in spec or "data" not in spec:
            raise ValueError(f"feed[{name!r}]: mapping needs 'shape'+'data'")
        shape = tuple(int(d) for d in spec["shape"])
        data = spec["data"]
    try:
        t = torch.as_tensor(data)
    except Exception as e:  # noqa: BLE001 — report the entry, not a traceback class
        raise ValueError(f"feed[{name!r}]: not array-like ({e})") from e
    if shape is not None:
        if t.numel() != _numel(shape):
            raise ValueError(f"feed[{name!r}]: {t.numel()} values do not fill shape {list(shape)}")
        t = t.reshape(shape)
    return t


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _load_feed(raw: Any) -> dict[str, Any]:
    """``--adapter-opt feed=<path.json>`` → ``{input: tensor-spec}``."""
    if raw is None:
        return {}
    if not isinstance(raw, (str, Path)):
        raise ValueError(f"feed: expected a JSON file path, got {type(raw).__name__}")
    path = Path(raw)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"feed: cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"feed: {path} is not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"feed: {path} must contain a JSON object of input values")
    return data


# ---------------------------------------------------------------------------
#  The Source port — the pre-lifted (IR, leaf values) the search consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnnxLift:
    """What ``load`` hands ``spec.module``: the lifted term + its IR.

    Carried for introspection (``doctor``/reports/stage records); the
    pipeline reads the IR through :class:`OnnxRustSource`, not this
    payload.
    """

    term: Any
    ir: IR
    path: Path
    doc: dict[str, Any]


class OnnxRustSource(Source):
    """``mlopt_core.ports.Source`` over a pre-lifted ONNX term.

    ``to_ir`` returns the ``(IR, param_values)`` the adapter built at
    load time — the same contract ``OnnxSource``/``HnlSource``
    fulfil.  ``model``/``example_inputs`` are accepted for the port
    signature and ignored: the lift already happened.
    """

    def __init__(self, ir: IR, param_values: dict[str, Any]) -> None:
        self._ir = ir
        self._param_values = param_values
        #: The lifter's ``input_dtypes`` side channel — ONNX declares
        #: element types the shape-only ``Var.typ`` cannot carry (the
        #: documented source-side workaround, same as ``OnnxSource``).
        self.input_dtypes: dict[str, str] = {}

    def to_ir(self, model: Any, example_inputs: Any) -> tuple[IR, dict[str, Any]]:
        """Return the lifted IR plus its param leaf values."""
        return self._ir, dict(self._param_values)


# ---------------------------------------------------------------------------
#  The lift — subprocess → Term JSON → (term, IR, inputs, param_values)
# ---------------------------------------------------------------------------


def _lift(
    path: Path,
    binary: Path,
    *,
    input_shapes: Any,
    feed: Any,
    seed: int,
) -> tuple[Any, IR, tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """``.onnx`` path → ``(term, ir, inputs, param_values, census)``."""
    from mlopt_core.term_json import term_from_json

    with tempfile.TemporaryDirectory(prefix="mlopt-onnx-") as td:
        doc = _run_lift(path, binary, Path(td))
        st_path = Path(doc.pop("_safetensors_path"))
        leaves_np = _load_safetensors(st_path)

    # Param names must be setattr-safe for IRModule — rename in the
    # JSON before decode so term, ir.params and param_values agree.
    rename = _sanitize_param_names(doc["root"])
    unrename = {new: old for old, new in rename.items()}
    term = term_from_json(doc["root"])
    vars_, params = _census(term)

    # Inputs: the lifter's declared order is the IR's; census Vars not
    # declared there (should not happen) append honestly at the end.
    declared = doc.get("inputs") or []
    by_name = {v.name: v for v in vars_}
    ordered: list[Var] = []
    seen: set[str] = set()
    overrides = _parse_inputs_opt(input_shapes)
    input_dtypes = {str(k): str(v) for k, v in (doc.get("input_dtypes") or {}).items()}
    feed_map = _load_feed(feed)

    import torch

    gen = torch.Generator().manual_seed(int(seed))

    inputs: list[Any] = []
    input_prov: dict[str, str] = {}
    unbound: list[str] = []
    for entry in declared:
        name = str(entry["name"])
        var = by_name.get(name)
        if var is None:
            # Declared input that never reached the term — the lifter
            # and the census disagree; decline rather than reorder.
            raise ValueError(
                f"onnx-rust: declared input {name!r} has no Var leaf in "
                "the lifted term — lifter/census disagreement"
            )
        seen.add(name)
        raw_shape = overrides.get(name, entry.get("shape"))
        if raw_shape is None:
            unbound.append(f"input {name!r} (shape undeclared)")
            continue
        shape = tuple(int(d) for d in raw_shape)
        if name in feed_map:
            val = _feed_tensor(feed_map[name], name)
            prov = "feed"
        else:
            val = _probe_input(shape, input_dtypes.get(name, "float32"), gen)
            prov = "synthetic"
        inputs.append(val)
        input_prov[name] = prov

    for v in vars_:
        if v.name in seen:
            continue
        # A Var the lift doc did not declare — synthesize from its own
        # type when concrete, else decline.
        shape = _concrete_shape(v.typ)
        if shape is None:
            unbound.append(f"Var {v.name!r} (shape={getattr(v.typ, 'shape', None)!r})")
            continue
        if v.name in feed_map:
            val = _feed_tensor(feed_map[v.name], v.name)
            prov = "feed"
        else:
            val = _probe_input(shape, "float32", gen)
            prov = "synthetic"
        inputs.append(val)
        input_prov[v.name] = prov
        ordered.append(v)

    if unbound:
        raise ValueError(
            "onnx-rust: unbound inputs — "
            + "; ".join(unbound)
            + ".  Pass --adapter-opt inputs='{\"name\": [dims]}' for "
            "dynamic dims, feed=<json> for real values"
        )

    # Params: every Param leaf must have a safetensors payload — the
    # weights are the artifact's own, never synthesized.
    param_values: dict[str, Any] = {}
    missing = [n for n in params if unrename.get(n, n) not in leaves_np]
    if missing:
        orig = [unrename.get(n, n) for n in missing]
        raise ValueError(
            "onnx-rust: param leaves missing from the safetensors "
            f"payload: {orig[:5]}{'…' if len(orig) > 5 else ''}"
        )
    for name in params:
        # torch tensors, not the raw NumPy views — the torch backend's
        # lower/verify path calls ``.detach`` on leaf values (the
        # ``TorchSource`` convention; ``np.frombuffer`` views are also
        # read-only, so copy through ``torch.tensor``).  ``name`` is
        # the sanitized leaf name; the safetensors key is the graph's
        # original value name.  Half dtypes upcast to f32 inside
        # ``_param_tensor`` (lossless; f32 probes are the convention).
        param_values[name] = _param_tensor(leaves_np[unrename.get(name, name)])

    ir = IR(
        root=term,
        inputs=[by_name[e["name"]] for e in declared if e["name"] in by_name] + ordered,
        input_names={v.name for v in vars_},
        params=params,
    )
    census = {
        "vars": {e["name"]: list(e.get("shape") or ()) for e in declared},
        "params": {n: list(_concrete_shape(p.typ) or ()) for n, p in params.items()},
        "input_values": input_prov,
        "leaf_manifest": {
            str(leaf["name"]): {
                "dtype": leaf.get("dtype"),
                "shape": leaf.get("shape"),
                "nbytes": leaf.get("nbytes"),
            }
            for leaf in (doc.get("leaves") or [])
        },
    }
    return (
        term,
        ir,
        tuple(inputs),
        param_values,
        {
            "census": census,
            "input_dtypes": input_dtypes,
            "param_rename": rename,
        },
    )


# ---------------------------------------------------------------------------
#  The adapter
# ---------------------------------------------------------------------------


class _OnnxRustAdapter:
    """Lifts ``.onnx`` files / ``onnx-rust:`` refs via the Rust lifter."""

    name = "onnx-rust"
    description = (
        ".onnx ONNX source — source-onnx Rust lifter (subprocess, "
        "Term-JSON), safetensors weights, seeded synthetic "
        "input probes"
    )
    available = True
    unavailable_reason = ""

    def claims(self, ref: str) -> bool:
        ref = ref.strip()
        return ref.endswith(".onnx") or ref.startswith("onnx-rust:")

    def _ref_path(self, ref: str) -> Path:
        ref = ref.strip()
        if ref.startswith("onnx-rust:"):
            ref = ref[len("onnx-rust:") :]
        return Path(ref)

    def _opts(self, opts: dict[str, Any]) -> dict[str, Any]:
        return {
            "lift_bin": opts.get("lift_bin"),
            "inputs": opts.get("inputs"),
            "feed": opts.get("feed"),
            "seed": opts.get("seed", 0),
        }

    def probe(self, ref: str) -> dict[str, Any]:
        """Cheap ``doctor`` report — binary resolution + file check."""
        path = self._ref_path(ref)
        binary = _find_binary(self._opts({}).get("lift_bin"))
        out: dict[str, Any] = {"path": str(path)}
        if not path.exists():
            out["file"] = "missing"
        else:
            out["file"] = "ok"
        if binary is None:
            out["lift_binary"] = "not found — see `mlopt optimize --help` / adapter docs"
        else:
            out["lift_binary"] = str(binary)
        return out

    def load(self, ref: str, **opts: Any) -> ModelSpec:
        path = self._ref_path(ref)
        if not path.exists():
            raise ValueError(f"onnx-rust: cannot read {path}: no such file")
        o = self._opts(opts)
        binary = _ensure_binary(o["lift_bin"])
        term, ir, inputs, param_values, extra = _lift(
            path,
            binary,
            input_shapes=o["inputs"],
            feed=o["feed"],
            seed=o["seed"],
        )
        census = extra["census"]
        provs = set(census["input_values"].values())
        if not provs:
            input_source = "none"
        elif provs == {"feed"}:
            input_source = "feed"
        elif "feed" in provs:
            input_source = "mixed"
        else:
            input_source = "synthetic"
        lift = OnnxLift(term=term, ir=ir, path=path, doc={})
        source = OnnxRustSource(ir, param_values)
        source.input_dtypes = extra["input_dtypes"]
        return ModelSpec(
            module=lift,
            inputs=inputs,
            meta={
                "ref": ref.strip(),
                "adapter": "onnx-rust",
                "path": str(path),
                "lift_binary": str(binary),
                # Weights are the artifact's own safetensors payload —
                # real, never synthesized.  Inputs are seeded probes.
                "leaf_source": "real",
                "input_source": input_source,
                "leaves": census,
                # ONNX value names → setattr-safe Param names (dots/
                # colons are illegal as torch Module attrs); empty
                # when every name was already safe.
                "param_rename": extra.get("param_rename") or {},
            },
            source=source,
        )


ADAPTER = _OnnxRustAdapter()
register_adapter(ADAPTER)
