"""``mlopt.arch/v1`` source adapter — architecture-as-data, zero ONNX.

``mlopt optimize arch:whisper-tiny-encoder`` (or
``arch:<path/to/spec.json>`` / a ``*.arch.json`` path) runs the
``source-onnx`` Rust assembler (``handl-lang/source-onnx``
``examples/arch.rs``, ADR-0176) as a subprocess — the same
JSON-in/JSON-out bridge ``adapters/onnx_rust.py`` uses — then decodes
the emitted Term-JSON through
:func:`mlopt_core.term_json.term_from_json` and feeds the
pipeline through the same pre-lifted ``Source`` port.  The lift chain
is HF safetensors → arch spec → Term → search → verify → deliver with
**no ONNX file and no Python model code**.

Honest bounds, stated up front:

* **The assembler is a subprocess, not a library** — same constraint
  as ``onnx-rust``.  The adapter resolves the ``arch`` example binary
  (``--adapter-opt arch_bin=<path>``, the ``SOURCE_ONNX_ARCH`` env
  var, the sibling checkout's ``target/{release,debug}/examples/arch``,
  the shared cargo target dir) and builds it with
  ``cargo build --example arch`` when a ``source-onnx`` checkout is
  present but unbuilt.  No binary and no checkout → a named decline.
* **Weights are REAL and required — resolved Rust-side.**  The
  ``arch`` CLI owns checkpoint resolution: ``--adapter-opt
  weights=<path>`` passes an explicit safetensors file (or shard
  ``*.index.json``) through ``--weights``; otherwise the spec's
  ``checkpoint.{repo,file}`` is resolved by the assembler itself
  through the local Hugging Face cache (``$HF_HUB_CACHE`` /
  ``$HF_HOME/hub`` / ``~/.cache/huggingface/hub``,
  ``models--<org>--<name>/snapshots/*``, ``<file>.index.json``
  shards followed) — cache only, never a download.  A miss is the
  assembler's named refusal.  With ``--st-out`` the CLI also emits
  the **bound payload**: a safetensors document keyed by leaf
  ``source`` names with each leaf's declared ``transform`` already
  applied and the manifest rewritten to ``identity`` — leaf values
  bind here with no checkpoint access and no transform logic.  No
  weights → a named decline listing the remedies.
* **Refusals surface verbatim.**  The assembler's named refusals
  (unknown ``family``, unimplemented ``activation``/
  ``linear_weight_transform``, spec/shape violations, safetensors
  binding failures) arrive as the document's ``error`` string and are
  raised unchanged.
* **Inputs are seeded synthetic probes** — identical contract to
  ``onnx-rust``: ``--adapter-opt inputs='{"name": [dims]}'`` overrides
  a declared shape, ``feed=<path.json>`` supplies real values,
  ``seed=N`` seeds the probes.  The spec's shapes are static today,
  so the probes are well-defined without overrides.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlopt_core.term import IR

from mlopt.adapters import onnx_rust as _onnx
from mlopt.models import ModelSpec, register_adapter

__all__ = ["ADAPTER", "ArchLift", "ArchRustSource"]


# ---------------------------------------------------------------------------
#  Source-onnx checkout + ``arch`` binary discovery (the lift mirror)
# ---------------------------------------------------------------------------


def _arch_bin_candidates(repo: Path | None) -> list[Path]:
    cands: list[Path] = []
    env = os.environ.get("SOURCE_ONNX_ARCH")
    if env:
        cands.append(Path(env))
    if repo is not None:
        cands += [
            repo / "target" / "release" / "examples" / "arch",
            repo / "target" / "debug" / "examples" / "arch",
        ]
    shared = Path.home() / ".cargo" / "shared-target"
    cands += [
        shared / "release" / "examples" / "arch",
        shared / "debug" / "examples" / "arch",
    ]
    return cands


def _find_arch_binary(arch_bin: Any = None) -> Path | None:
    """Resolve the ``arch`` example binary without building."""
    if arch_bin:
        p = Path(str(arch_bin)).expanduser()
        return p if p.is_file() else None
    for cand in _arch_bin_candidates(_onnx._source_onnx_repo()):
        if cand.is_file():
            return cand
    return None


def _ensure_arch_binary(arch_bin: Any = None) -> Path:
    """Return the ``arch`` binary, building it when a checkout exists."""
    b = _find_arch_binary(arch_bin)
    if b is not None:
        return b
    if arch_bin:
        raise ValueError(f"arch: arch_bin={arch_bin!r} does not name an executable file")
    repo = _onnx._source_onnx_repo()
    if repo is not None and shutil.which("cargo") is not None:
        proc = subprocess.run(
            ["cargo", "build", "--example", "arch"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode == 0:
            b = _find_arch_binary(arch_bin)
            if b is not None:
                return b
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise ValueError(
            f"arch: cargo build --example arch failed in {repo} "
            f"(rc={proc.returncode}): {tail[-1] if tail else 'no output'}"
        )
    raise ValueError(
        "arch: the source-onnx `arch` binary was not found — set "
        "--adapter-opt arch_bin=<path> or SOURCE_ONNX_ARCH, or clone "
        "handl-lang/source-onnx next to the mlopt checkout (or point "
        "SOURCE_ONNX_REPO at it) and build with "
        "`cargo build --example arch`"
    )


# ---------------------------------------------------------------------------
#  Spec resolution — arch: names → data/architectures/, paths verbatim
# ---------------------------------------------------------------------------


def _spec_candidates(name: str, repo: Path | None) -> list[Path]:
    """Candidate spec paths for an ``arch:`` ref's payload.

    A path that exists wins verbatim; otherwise the payload is a spec
    *name* resolved under the checkout's ``data/architectures/`` —
    ``whisper-tiny-encoder`` and ``whisper-tiny-encoder.json`` both
    spell the same file.
    """
    p = Path(name).expanduser()
    if p.is_file():
        return [p]
    if repo is None:
        return []
    base = repo / "data" / "architectures"
    stem = name[:-5] if name.endswith(".json") else name
    return [base / f"{stem}.json"]


def _resolve_spec_path(ref_payload: str) -> tuple[Path, Path | None]:
    """``(spec_path, repo)`` — declines by name when nothing resolves."""
    repo = _onnx._source_onnx_repo()
    for cand in _spec_candidates(ref_payload, repo):
        if cand.is_file():
            return cand, repo
    raise ValueError(
        f"arch: cannot resolve arch spec {ref_payload!r} — pass a "
        "path to an mlopt.arch/v1 JSON file, or a spec name shipped in "
        "source-onnx's data/architectures/ (clone handl-lang/source-onnx "
        "next to the mlopt checkout or set SOURCE_ONNX_REPO)"
    )


# ---------------------------------------------------------------------------
#  Weight resolution — --weights path, else the HF cache (never a download)
# ---------------------------------------------------------------------------


def _hf_cache_root() -> Path | None:
    """The huggingface_hub cache root, env-honest, ``None`` when unset."""
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    default = Path.home() / ".cache" / "huggingface" / "hub"
    return default if default.is_dir() else None


def _hf_cache_weights(repo: str, file: str) -> Path | None:
    """``models--<org>--<name>/snapshots/*/<file>``, newest snapshot wins.

    Cache-only by construction — this path never touches the network,
    so a miss is an honest decline, not a silent download.
    """
    root = _hf_cache_root()
    if root is None:
        return None
    model_dir = root / ("models--" + repo.replace("/", "--"))
    snaps = model_dir / "snapshots"
    if not snaps.is_dir():
        return None
    hits = [snap / file for snap in snaps.iterdir() if snap.is_dir() and (snap / file).is_file()]
    if not hits:
        return None
    # Newest snapshot directory by mtime, name as the tiebreak —
    # snapshot hashes are not ordered.
    return max(hits, key=lambda p: (p.stat().st_mtime, p.name))


def _checkpoint_of(spec_path: Path) -> dict[str, Any] | None:
    """The spec's ``checkpoint`` block, or ``None`` when undeclared."""
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ckpt = spec.get("checkpoint")
    return ckpt if isinstance(ckpt, dict) else None


def _resolve_weights(spec_path: Path, weights_opt: Any) -> tuple[str, Path | None]:
    """How the assembler gets its checkpoint: ``("path", p)`` or
    ``("checkpoint", None)``.

    ``weights=<path>`` wins (validated here so a typo declines before
    the subprocess); otherwise the spec's ``checkpoint.{repo,file}``
    resolves **Rust-side** through the HF cache — the assembler owns
    snapshot selection and shard indices now, so Python only reports
    the mode.  A spec with neither is the same named decline as
    before; weights are never synthesized.
    """
    if weights_opt:
        p = Path(str(weights_opt)).expanduser()
        if not p.is_file():
            raise ValueError(f"arch: weights={weights_opt!r} is not a file")
        return "path", p
    ckpt = _checkpoint_of(spec_path)
    if ckpt and ckpt.get("repo") and ckpt.get("file"):
        return "checkpoint", None
    raise ValueError(
        "arch: no weights — the arch assembler binds real "
        "safetensors leaves and emits no payload of its own.  Pass "
        "--adapter-opt weights=<model.safetensors>, or use a spec whose "
        "`checkpoint` block names an HF repo already in the local cache"
    )


# ---------------------------------------------------------------------------
#  The assembler subprocess — spec + weights in, JSON document out
# ---------------------------------------------------------------------------

_ARCH_TIMEOUT_S = 900


def _run_arch(
    spec: Path,
    mode: str,
    weights_path: Path | None,
    binary: Path,
    workdir: Path,
) -> dict[str, Any]:
    """Run ``arch <spec> [--weights <path> | --checkpoint] --st-out <payload>``.

    ``mode`` is ``"path"`` (explicit ``weights_path``) or
    ``"checkpoint"`` (Rust-side HF-cache resolution of the spec's
    ``checkpoint.{repo,file}``).  ``--st-out`` always emits the bound
    payload — the post-transform safetensors the leaf binding reads —
    into ``workdir``; its path rides the document as
    ``_weights_payload`` and the resolved checkpoint shard(s) as
    ``_checkpoint_files``.  ``{"ok": false, "error": …}`` — the
    assembler's named refusal — raises ``ValueError`` carrying the
    error string verbatim.
    """
    payload = workdir / "weights.safetensors"
    argv = [str(binary), str(spec)]
    if mode == "path":
        argv += ["--weights", str(weights_path)]
    else:
        argv += ["--checkpoint"]
    argv += ["--st-out", str(payload)]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_ARCH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        raise ValueError(f"arch: arch timed out (>{_ARCH_TIMEOUT_S}s) on {spec}") from e
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"arch: arch produced non-JSON output (rc={proc.returncode}): "
            f"{proc.stdout[:200]!r} stderr={proc.stderr[:300]!r}"
        ) from e
    if not doc.get("ok"):
        # The assembler's named refusal — surface it verbatim.
        raise ValueError(f"arch: {doc.get('error', 'unknown arch error')}")
    doc["_weights_payload"] = str(payload)
    doc["_checkpoint_files"] = (doc.get("weights") or {}).get("files") or []
    return doc


# ---------------------------------------------------------------------------
#  Leaf binding — manifest source/transform → torch param values
# ---------------------------------------------------------------------------


def _apply_transform(arr: Any, transform: str, leaf_name: str) -> Any:
    """One safetensors array → the Term leaf's value.

    ``transpose_last_two`` is the HF ``nn.Linear`` ``[out, in]`` →
    ``matmul`` ``[in, out]`` spelling; ``identity`` passes through.
    Anything else declines by name — the assembler only emits these
    two today, and an approximation would silently corrupt weights.
    """
    if transform == "identity":
        return arr
    if transform == "transpose_last_two":
        return arr.swapaxes(-1, -2)
    raise ValueError(
        f"arch: leaf {leaf_name!r} declares unknown weight "
        f"transform {transform!r} — cannot bind its values"
    )


def _bind_params(
    doc: dict[str, Any],
    leaves_np: dict[str, Any],
    payload_path: Path,
    rename: dict[str, str],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Manifest leaves → ``{sanitized param name: torch tensor}``.

    ``payload_path`` is the assembler's **bound payload** safetensors
    (``--st-out``): keyed by each leaf's ``source`` with the declared
    transform already applied and the manifest rewritten to
    ``identity``, so ``_apply_transform`` is a pass-through here —
    kept anyway so a doc minted before the payload contract still
    binds honestly.  ``rename`` is the ``{original: sanitized}`` map
    :func:`onnx_rust._sanitize_param_names` produced; each manifest
    leaf's ``source`` is the payload tensor carrying the values
    (aliases already resolved assembler-side).  A Param leaf with no
    manifest entry — or a manifest source missing from the payload —
    declines by name; weights are never synthesized.  ``leaves_np``
    is the caller-loaded ``{name: np.ndarray}`` view of the payload —
    passed in because the payload file lives in a TemporaryDirectory
    the caller has already torn down.
    """
    unrename = {new: old for old, new in rename.items()}
    manifest = {str(leaf["name"]): leaf for leaf in (doc.get("leaves") or [])}

    param_values: dict[str, Any] = {}
    missing = [n for n in params if unrename.get(n, n) not in manifest]
    if missing:
        orig = [unrename.get(n, n) for n in missing]
        raise ValueError(
            "arch: Param leaves with no weight manifest entry: "
            f"{orig[:5]}{'…' if len(orig) > 5 else ''}"
        )
    for name in params:
        orig = unrename.get(name, name)
        leaf = manifest[orig]
        source = str(leaf.get("source") or orig)
        if source not in leaves_np:
            raise ValueError(
                f"arch: safetensors tensor {source!r} (leaf "
                f"{orig!r}) is missing from the bound payload {payload_path}"
            )
        arr = _apply_transform(leaves_np[source], str(leaf.get("transform")), orig)
        want = leaf.get("shape")
        if want is not None and tuple(arr.shape) != tuple(int(d) for d in want):
            raise ValueError(
                f"arch: leaf {orig!r} binds {source!r} with shape "
                f"{arr.shape} after transform, spec wants {want}"
            )
        # torch tensors, not the raw NumPy views — the torch backend's
        # lower/verify path calls ``.detach`` and ``np.frombuffer``
        # views are read-only (the onnx-rust convention).  Half
        # payload dtypes upcast to f32 inside ``_param_tensor`` —
        # lossless, and required: the probes bind f32, and torch
        # refuses f32-input/c10::Half-weight mixes at verify.
        param_values[name] = _onnx._param_tensor(arr)

    # Declared aliases must also resolve — the assembler verified them
    # against the checkpoint and emits their raw tensors into the
    # payload when no leaf claims the key.
    for alias in doc.get("aliases") or []:
        source = str(alias.get("source") or "")
        if source and source not in leaves_np:
            raise ValueError(
                f"arch: alias {alias.get('name')!r} source "
                f"{source!r} is missing from the bound payload {payload_path}"
            )
    return param_values


# ---------------------------------------------------------------------------
#  The Source port — reuses onnx-rust's (it is the generic pre-lifted IR)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchLift:
    """What ``load`` hands ``spec.module``: the assembled term + its IR.

    Carried for introspection (``doctor``/reports/stage records); the
    pipeline reads the IR through the ``Source`` port, not this
    payload.
    """

    term: Any
    ir: IR
    spec_path: Path
    weights_path: Path
    doc: dict[str, Any]


#: The pre-lifted ``(IR, leaf values)`` port is producer-agnostic —
#: ``OnnxRustSource`` is exactly the contract ``ArchRustSource`` needs
#: (``to_ir`` returns the stored pair; ``input_dtypes`` is the same
#: side channel).  Aliased rather than subclassed so there is one
#: implementation to keep honest.
ArchRustSource = _onnx.OnnxRustSource


# ---------------------------------------------------------------------------
#  The lift — subprocess → Term JSON → (term, IR, inputs, param_values)
# ---------------------------------------------------------------------------


def _lift(
    spec_path: Path,
    mode: str,
    weights_path: Path | None,
    binary: Path,
    *,
    input_shapes: Any,
    feed: Any,
    seed: int,
) -> tuple[Any, IR, tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """``(spec, weights-mode)`` → ``(term, ir, inputs, param_values, extra)``."""
    import tempfile

    from mlopt_core.term_json import term_from_json

    with tempfile.TemporaryDirectory(prefix="mlopt-arch-") as td:
        doc = _run_arch(spec_path, mode, weights_path, binary, Path(td))
        payload_path = Path(doc.pop("_weights_payload"))
        checkpoint_files = doc.pop("_checkpoint_files")
        # The bound payload — post-transform leaf tensors keyed by
        # safetensors source name; the manifest in `doc` already
        # describes it (transform=identity).
        param_leaves_np = _onnx._load_safetensors(payload_path)

    # Param names must be setattr-safe for IRModule — rename in the
    # JSON before decode so term, ir.params and param_values agree.
    rename = _onnx._sanitize_param_names(doc["root"])
    term = term_from_json(doc["root"])
    vars_, params = _onnx._census(term)
    param_values = _bind_params(doc, param_leaves_np, payload_path, rename, params)
    # ``param_values`` are torch copies — the safetensors views were
    # only the bind source.  Drop the dict so the memmap's faulted
    # pages unmap now instead of riding along through search/lower
    # (llama-scale bound payloads are GBs; every reclaimable MB counts).
    del param_leaves_np

    # Inputs: identical contract to onnx-rust — the assembler's
    # declared order is the IR's; census Vars not declared there
    # append honestly at the end.
    declared = doc.get("inputs") or []
    by_name = {v.name: v for v in vars_}
    ordered: list[Any] = []
    seen: set[str] = set()
    overrides = _onnx._parse_inputs_opt(input_shapes)
    input_dtypes = {str(k): str(v) for k, v in (doc.get("input_dtypes") or {}).items()}
    feed_map = _onnx._load_feed(feed)

    import torch

    gen = torch.Generator().manual_seed(int(seed))

    inputs: list[Any] = []
    input_prov: dict[str, str] = {}
    unbound: list[str] = []
    for entry in declared:
        name = str(entry["name"])
        var = by_name.get(name)
        if var is None:
            raise ValueError(
                f"arch: declared input {name!r} has no Var leaf in "
                "the assembled term — assembler/census disagreement"
            )
        seen.add(name)
        raw_shape = overrides.get(name, entry.get("shape"))
        if raw_shape is None:
            unbound.append(f"input {name!r} (shape undeclared)")
            continue
        shape = tuple(int(d) for d in raw_shape)
        if name in feed_map:
            val = _onnx._feed_tensor(feed_map[name], name)
            prov = "feed"
        else:
            val = _onnx._probe_input(shape, input_dtypes.get(name, "float32"), gen)
            prov = "synthetic"
        inputs.append(val)
        input_prov[name] = prov

    for v in vars_:
        if v.name in seen:
            continue
        shape = _onnx._concrete_shape(v.typ)
        if shape is None:
            unbound.append(f"Var {v.name!r} (shape={getattr(v.typ, 'shape', None)!r})")
            continue
        if v.name in feed_map:
            val = _onnx._feed_tensor(feed_map[v.name], v.name)
            prov = "feed"
        else:
            val = _onnx._probe_input(shape, "float32", gen)
            prov = "synthetic"
        inputs.append(val)
        input_prov[v.name] = prov
        ordered.append(v)

    if unbound:
        raise ValueError(
            "arch: unbound inputs — "
            + "; ".join(unbound)
            + ".  Pass --adapter-opt inputs='{\"name\": [dims]}' for "
            "dynamic dims, feed=<json> for real values"
        )

    ir = IR(
        root=term,
        inputs=[by_name[e["name"]] for e in declared if e["name"] in by_name] + ordered,
        input_names={v.name for v in vars_},
        params=params,
    )
    census = {
        "vars": {e["name"]: list(e.get("shape") or ()) for e in declared},
        "params": {n: list(_onnx._concrete_shape(p.typ) or ()) for n, p in params.items()},
        "input_values": input_prov,
        "leaf_manifest": {
            str(leaf["name"]): {
                "dtype": leaf.get("dtype"),
                "shape": leaf.get("shape"),
                "source": leaf.get("source"),
                "transform": leaf.get("transform"),
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
            "weights_verified": doc.get("weights_verified"),
            "checkpoint_files": checkpoint_files,
            "weights_sharded": bool((doc.get("weights") or {}).get("sharded")),
        },
    )


# ---------------------------------------------------------------------------
#  The adapter
# ---------------------------------------------------------------------------


class _ArchRustAdapter:
    """Lifts ``arch:`` refs / ``*.arch.json`` specs via the Rust assembler."""

    #: Registry name ``arch`` — ``resolve_ref`` routes ``name:<rest>``
    #: by registry name, so the ``arch:`` ref scheme requires it (the
    #: ``onnx-rust:`` precedent: prefix == adapter name).
    name = "arch"
    description = (
        "mlopt.arch/v1 architecture spec — source-onnx Rust assembler "
        "(subprocess, Term-JSON), real safetensors weights "
        "(weights= opt or HF cache), seeded synthetic input probes"
    )
    available = True
    unavailable_reason = ""

    def claims(self, ref: str) -> bool:
        ref = ref.strip()
        if ref.startswith("arch:") or ref.endswith(".arch.json"):
            return True
        # A bare spec path: any ``*.json`` that exists and declares the
        # ``mlopt.arch/v1`` format tag.  The sniff is a bounded read —
        # ``claims`` runs once per bare-ref routing, never per load.
        if ref.endswith(".json"):
            p = Path(ref).expanduser()
            try:
                if p.is_file():
                    head = p.read_bytes()[:4096]
                    return b"mlopt.arch/v1" in head
            except OSError:
                return False
        return False

    def _ref_payload(self, ref: str) -> str:
        ref = ref.strip()
        if ref.startswith("arch:"):
            return ref[len("arch:") :].strip()
        return ref

    def _opts(self, opts: dict[str, Any]) -> dict[str, Any]:
        return {
            "arch_bin": opts.get("arch_bin"),
            "weights": opts.get("weights"),
            "inputs": opts.get("inputs"),
            "feed": opts.get("feed"),
            "seed": opts.get("seed", 0),
        }

    def probe(self, ref: str) -> dict[str, Any]:
        """Cheap ``doctor`` report — spec/binary/weights resolution."""
        payload = self._ref_payload(ref)
        binary = _find_arch_binary(self._opts({}).get("arch_bin"))
        out: dict[str, Any] = {"spec": payload}
        try:
            spec_path, _repo = _resolve_spec_path(payload)
        except ValueError as e:
            out["spec_file"] = f"unresolved — {e}"
        else:
            out["spec_file"] = str(spec_path)
            ckpt = _checkpoint_of(spec_path) or {}
            weights = _hf_cache_weights(str(ckpt.get("repo", "")), str(ckpt.get("file", "")))
            out["checkpoint"] = (
                f"{ckpt.get('repo')}/{ckpt.get('file')}" if ckpt.get("repo") else "undeclared"
            )
            out["weights"] = str(weights) if weights else "not in HF cache"
        if binary is None:
            out["arch_binary"] = "not found — see `mlopt optimize --help` / adapter docs"
        else:
            out["arch_binary"] = str(binary)
        return out

    def load(self, ref: str, **opts: Any) -> ModelSpec:
        payload = self._ref_payload(ref)
        spec_path, _repo = _resolve_spec_path(payload)
        o = self._opts(opts)
        binary = _ensure_arch_binary(o["arch_bin"])
        mode, weights_path = _resolve_weights(spec_path, o["weights"])
        term, ir, inputs, param_values, extra = _lift(
            spec_path,
            mode,
            weights_path,
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
        # ``weights_path`` for introspection/meta is the *checkpoint*
        # file: the explicit ``weights=`` path, or the shard the
        # assembler resolved (single-file checkpoints report their one
        # safetensors; sharded ones report every shard).
        ckpt_files = extra.get("checkpoint_files") or []
        resolved_weights = (
            weights_path
            if weights_path is not None
            else (Path(ckpt_files[0]) if len(ckpt_files) == 1 else None)
        )
        lift = ArchLift(
            term=term,
            ir=ir,
            spec_path=spec_path,
            weights_path=resolved_weights or spec_path,
            doc={},
        )
        source = ArchRustSource(ir, param_values)
        source.input_dtypes = extra["input_dtypes"]
        ckpt = _checkpoint_of(spec_path) or {}
        return ModelSpec(
            module=lift,
            inputs=inputs,
            meta={
                "ref": ref.strip(),
                "adapter": "arch",
                "spec_path": str(spec_path),
                "weights_path": (
                    str(resolved_weights)
                    if resolved_weights is not None
                    else (", ".join(ckpt_files) if ckpt_files else None)
                ),
                "checkpoint_files": [str(f) for f in ckpt_files],
                "weights_sharded": extra.get("weights_sharded"),
                "checkpoint": (f"{ckpt['repo']}/{ckpt['file']}" if ckpt.get("repo") else None),
                "arch_binary": str(binary),
                "weights_verified": extra.get("weights_verified"),
                # Weights are the checkpoint's own safetensors tensors —
                # real, never synthesized.  Inputs are seeded probes.
                "leaf_source": "real",
                "input_source": input_source,
                "leaves": census,
                # Arch/ONNX value names → setattr-safe Param names; empty
                # when every name was already safe.
                "param_rename": extra.get("param_rename") or {},
            },
            source=source,
        )


ADAPTER = _ArchRustAdapter()
register_adapter(ADAPTER)
