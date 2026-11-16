"""``python -m mlopt stage <name>`` — the kit-mlopt JSON protocol endpoint.

``handl-kit-mlopt``'s ``pipeline`` module (``lift`` / ``search`` /
``verify`` host functions) drives ``hnl run`` pipeline scripts by
spawning this subcommand once per stage:

.. code:: text

    <stage input JSON>  --stdin-->  python -m mlopt stage <stage>
    <stage output JSON> <--stdout-- (one JSON value, nothing else)

Protocol contract (the Rust side is ``kit-mlopt/src/lib.rs``):

* stdin carries **exactly one JSON value** — a model-ref string (or
  ``{"model": ...}`` object) for ``lift``, the pipeline record object
  for ``search`` / ``verify`` / ``deliver``.
* stdout carries **exactly one JSON value** — the stage's output
  record.  On failure the process exits non-zero with diagnostics on
  stderr (an ``{"ok": false, "error": ...}`` record is still emitted
  on stdout when one could be formed — the kit ignores stdout on a
  non-zero exit, but a human piping the protocol sees the reason).

Stage semantics — real pipeline stages over the existing machinery,
no reimplementation:

* ``lift`` resolves the model ref through the adapter catalog
  (``mlopt.models.resolve_ref`` — the same routing ``optimize``
  uses), loads the ``ModelSpec``, and runs the spec's ``Source`` port
  (``HnlSource`` for ``.hnl`` refs, ``TorchSource`` otherwise) to
  produce the lifted ``IR``.  The record carries the IR serialized
  with ``mlopt_core.term_json.term_to_json`` — the canonical
  kernel-Term-JSON envelope (v3; v≤2 records carry the
  ``mlopt_core.ir.term_to_data`` spellings, still decoded).  DAG-shared
  terms (e.g. one encoder
  output feeding many decoder cross-attentions) cross as the
  ``dag.program`` node-table envelope — linear in distinct nodes
  where the tree spelling is exponential; v1-era endpoints cannot
  decode it, hence the v2 record pin below.  v3 (ADR-0178 WP0b)
  re-spells every term payload in the canonical
  ``mlopt_core.term_json`` Term-JSON envelope (``{"Op"/"Arr"/
  "Const"/"Opaque"}``) — the codec ``engine-eqsat`` and the
  ``source-onnx`` lifter speak (``kit_discovery.hnl_codec`` is a
  re-export shim over the same codec); v≤2 payloads (the
  ``mlopt_core.ir.term_to_data`` spellings) are still decoded — the
  two envelopes are disjoint, so the reader dispatches on payload
  shape, never on a guess.
* ``search`` re-loads the spec (each stage is a fresh process — there
  is no shared memory) and runs ``Optimizer.search`` with a
  :class:`_RecordSource` that serves the *record's* lifted IR while
  taking fresh leaf values from the spec's real source.  The
  extracted term is serialized back into the record.
* ``verify`` re-loads the spec, rebuilds a ``SearchResult`` handle
  from the record's lifted IR + searched term + freshly materialised
  leaf values, and runs ``Optimizer.lower(verify=True)`` — the same
  substrate-enforced verify gate ``optimize`` uses.  A declined gate
  (``VerifyDeclinedError``) or a failed report exits non-zero.
* ``deliver`` takes the verify record and ships the verified term as
  a runnable artifact: it re-runs the SAME ``lower(verify=True)``
  gate (the JSON ``verified`` block is a claim, never trusted — a
  record whose ``verified.passed`` is not ``true`` is refused before
  any work), then emits the extracted term as a self-contained
  ``.hnl`` program through :func:`mlopt.emit.emit_program` — every
  leaf bound to its freshly materialised value — written to the
  record's ``out`` path.  A declined gate, an unemittable term, or
  an unwritable path exits non-zero with NO artifact.  The verify
  verdict covers the *term/module* — NOT the emitted program's
  hnl-run semantics (f64 wire, ``_DROP_ATTRS``, positional leaf
  binding) — so deliver additionally smoke-runs the artifact through
  ``hnl-run`` when the binary is available (a rejection is a decline
  with NO artifact) and reports ``artifact_executed`` honestly:
  ``true`` only when the program actually evaluated, ``false`` when
  no ``hnl-run`` binary was found (the note says the verdict's scope).

Honesty bounds:

* **Leaf values never cross the JSON boundary.**  Tensors are not
  JSON; every stage re-materialises them from the adapter (``feed`` /
  seeded-synthetic for ``.hnl``, ``torch.export`` params for torch
  models).  ``adapter_opts`` rides the record so later stages load
  the identical spec.  The record says so in ``notes``.
* **The e-graph does not cross either.**  ``verify``/``deliver``
  rebuild a ``SearchResult`` whose ``eg`` is ``None`` — the
  serialization boundary carries no saturated graph, and
  ``SearchResult.eg`` is nullable post-ADR-0178 (the eqsat engine
  never was an ``EGraph`` anyway).  ``lower`` never reads it (it
  consumes ``ir`` / ``term`` / ``param_values`` / ``stats`` only;
  the bound-propagation path ``isinstance``-guards the engine).
  ``stats`` is the JSON-safe subset of the search
  record; non-serializable entries are dropped and counted.
* Unbound leaves, unsupported refs, load/search/lower failures all
  exit non-zero naming the cause — a stage that cannot run says so.
* **No ``task=``/``task_tol=`` contract crosses the protocol.**  A
  ``TaskMetric`` is a callable — it cannot ride the JSON record —
  so ``search`` runs without ``task=`` and ``_rebuild_result``
  leaves ``SearchResult.task`` unset.  The pipeline path's only
  operative verify verdict is therefore the pointwise/bound gate
  (``VerifyReport`` + the propagated bound); the task-distance
  contract is reachable only through the in-process
  ``Optimizer.search(task=...)`` API, not ``mlopt stage``.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

__all__ = ["STAGES", "run_stage"]

#: The stage names ``python -m mlopt stage <name>`` accepts — the
#: kit-mlopt ``pipeline`` module's function table.  Extensible: add a
#: name here and a ``_stage_<name>`` handler below.
STAGES = ("lift", "search", "verify", "deliver")

#: The pipeline-record schema version this endpoint emits and
#: accepts.  Records carry ``"v": RECORD_VERSION``; a record with no
#: ``v`` is a v0 legacy record (the pre-versioning format — same
#: shape, so it reads fine).  A *higher* ``v`` is refused loudly:
#: the reader never guesses at a schema it does not know.
#:
#: v2: ``ir``/``term`` payloads may carry the ``dag.program``
#: node-table envelope (``mlopt_core.ir.term_to_data`` emits it for
#: terms whose tree expansion exceeds the serialization budget —
#: DAG-shared programs like the whisper encoder's cross-attention
#: fan-out that previously OOM'd the stage).  v1 payloads are still
#: accepted (the decoder reads both spellings); v1 *endpoints* cannot
#: read v2 records, which is exactly what the version gate is for.
#: v3: every term payload is spelled in the canonical
#: ``mlopt_core.term_json`` Term-JSON envelope (ADR-0178 WP0b —
#: ``{"Op"/"Arr"/"Const"/"Opaque"}``, the codec ``engine-eqsat`` and
#: ``source-onnx`` already speak; emitted with ``leaf="arr"`` — a
#: ``const-str`` leaf would misdecode as a scalar ``Const`` under
#: pre-shim ``hnl_codec`` readers, so it must never appear here).
#: v≤2 payloads (the
#: ``term_to_data`` spellings — ``{"op"/"var"/"param"/"const"/
#: "mvar"/"__tag__"}``) remain decodable: the envelopes are disjoint,
#: so :func:`_term_decode` dispatches on payload shape.
RECORD_VERSION = 3


class _StageError(Exception):
    """A stage-level failure — reported on stderr, exit 1."""


# ---------------------------------------------------------------------------
#  Record (de)serialization — terms through the term_json envelope
# ---------------------------------------------------------------------------
#
#  Emit side (v3): every term payload is ``mlopt_core.term_json``
#  Term-JSON — the canonical kernel-Term envelope (``{"Op"/"Arr"/
#  "Const"/"Opaque"}``, ``dag.program`` sharing under ``Opaque``),
#  the codec ``engine-eqsat`` and the ``source-onnx`` lifter speak
#  (ADR-0178 WP0b; ``kit_discovery.hnl_codec`` re-exports it).  The
#  leaf profile is pinned to ``"arr"`` — ``const-str`` leaves are an
#  engine-wire profile that pre-shim ``hnl_codec`` readers misdecode
#  as scalar ``Const`` payloads, so they must never appear in stage
#  records.
#
#  Decode side: BOTH envelopes are accepted.  v≤2 records carry the
#  ``mlopt_core.ir.term_to_data`` spellings (``{"op"/"var"/"param"/
#  "const"/"mvar"/"__tag__"}``); the two vocabularies are disjoint
#  (case-sensitive keys, different leaf records), so the decoder
#  dispatches on payload *shape* — a v3-stamped record that still
#  carries a legacy payload (a passthrough decline record, a hand
#  edit) decodes honestly instead of misreading.  ``term_from_json``
#  additionally accepts every legacy Term-JSON spelling forever
#  (``Foreign`` sources, ``lit.__mlopt_attrs__``, ``const-str``
#  leaves, nested-record dag ``dim``).  ``term_to_data``
#  decode is transitional — it retires with ``mlopt_core.ir`` at
#  ADR-0178 D5.


#: Top-level keys that mark a payload as Term-JSON.  The
#: legacy ``term_to_data`` spellings (``op``/``var``/``param``/
#: ``const``/``mvar``/``__tag__``) share no key with this set.
_TERM_JSON_KEYS = frozenset({"Op", "Arr", "Const", "Opaque"})


def _term_encode(term: Any) -> Any:
    """Encode a term as canonical Term-JSON (the v3 payload spelling).

    ``leaf="arr"`` is pinned explicitly: stage records must never
    carry ``const-str`` leaves (pre-shim ``hnl_codec`` readers would
    misdecode them as scalar ``Const`` values).
    """
    from mlopt_core.term_json import term_to_json

    return term_to_json(term, leaf="arr")


def _term_decode(d: Any, what: str = "term") -> Any:
    """Decode a term payload — Term-JSON or legacy term_to_data."""
    if isinstance(d, dict) and not _TERM_JSON_KEYS.isdisjoint(d):
        from mlopt_core.term_json import term_from_json

        try:
            return term_from_json(d)
        except (KeyError, TypeError, ValueError) as e:
            raise _StageError(
                f"record '{what}' payload failed term_json decode: {type(e).__name__}: {e}"
            ) from e
    from mlopt_core.ir import term_from_data

    try:
        return term_from_data(d)
    except (KeyError, TypeError, ValueError) as e:
        raise _StageError(f"record '{what}' failed to decode: {type(e).__name__}: {e}") from e


def _ir_encode(ir: Any) -> dict[str, Any]:
    """Serialize an ``IR``: root term + inputs + params, all term-encoded."""
    return {
        "root": _term_encode(ir.root),
        "inputs": [_term_encode(v) for v in ir.inputs],
        "input_names": sorted(ir.input_names),
        "params": {name: _term_encode(p) for name, p in ir.params.items()},
    }


def _ir_decode(d: Any) -> Any:
    """Rebuild an ``IR`` from :func:`_ir_encode` output."""
    from mlopt_core.term import IR

    if not isinstance(d, dict) or "root" not in d:
        raise _StageError(f"record 'ir' is not a lifted-IR object: {d!r:.200}")
    try:
        return IR(
            root=_term_decode(d["root"], "ir.root"),
            inputs=[_term_decode(v, "ir.inputs") for v in d.get("inputs", [])],
            input_names=set(d.get("input_names", [])),
            params={
                name: _term_decode(p, f"ir.params[{name!r}]")
                for name, p in d.get("params", {}).items()
            },
        )
    except _StageError:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise _StageError(f"record 'ir' failed to decode: {type(e).__name__}: {e}") from e


def _jsonable(value: Any) -> bool:
    """Whether *value* survives ``json.dumps`` unchanged."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _stats_to_data(stats: Any) -> dict[str, Any]:
    """The JSON-safe subset of a search ``stats`` record.

    Entries that do not round-trip through JSON (e.g. live e-graph or
    tensor references some passes stash) are dropped — the count is
    reported so the omission is visible, never silent.
    """
    out: dict[str, Any] = {}
    dropped = 0
    for k, v in dict(stats or {}).items():
        if _jsonable(v):
            out[str(k)] = v
        else:
            dropped += 1
    if dropped:
        out["_stats_keys_dropped"] = dropped
    return out


def _verified_to_data(verified: Any) -> dict[str, Any]:
    """The JSON-safe face of a ``VerifyReport``/``VerifyResult``."""
    out: dict[str, Any] = {"passed": bool(getattr(verified, "passed", False))}
    for key in ("max_abs", "max_rel", "rtol", "atol"):
        v = getattr(verified, key, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    per_output = getattr(verified, "per_output", None)
    if per_output is not None and _jsonable(per_output):
        out["per_output"] = per_output
    return out


# ---------------------------------------------------------------------------
#  Spec (re)loading — every stage is a fresh process, so the record must
#  carry everything needed to rebuild the ModelSpec identically.
# ---------------------------------------------------------------------------


def _base_record(model: str, opts: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "backend": "torch",
        "adapter_opts": dict(opts),
        "ok": False,
    }


def _resolve_spec(model: str, opts: dict[str, Any]) -> tuple[Any, str]:
    """Ref → ``(ModelSpec, adapter_name)`` — the ``optimize`` routing.

    Raises :class:`_StageError` naming the honest decline (no adapter,
    unavailable adapter, load failure).
    """
    import mlopt.models.catalog  # noqa: F401 — populates the registry
    from mlopt.models import resolve_ref

    adapter = resolve_ref(model)
    if adapter is None:
        raise _StageError(f"no adapter claims `{model}` — run `mlopt models` for the catalog")
    if not adapter.available:
        raise _StageError(f"adapter `{adapter.name}` unavailable: {adapter.unavailable_reason}")
    try:
        spec = adapter.load(model, backend="torch", **opts)
    except Exception as e:  # noqa: BLE001 — load failure is a stage failure
        raise _StageError(f"load failed: {type(e).__name__}: {e}") from e
    return spec, adapter.name


def _record_model_opts(rec: Any) -> tuple[str, dict[str, Any]]:
    """Extract ``(model, adapter_opts)`` from a pipeline record."""
    if not isinstance(rec, dict):
        raise _StageError(f"expected the pipeline record object, got {type(rec).__name__}")
    _check_record_version(rec)
    model = rec.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _StageError("record carries no 'model' ref — run `lift` first")
    opts = rec.get("adapter_opts")
    if opts is None:
        opts = {}
    if not isinstance(opts, dict):
        raise _StageError(f"record 'adapter_opts' is not an object: {opts!r:.200}")
    return model.strip(), dict(opts)


def _check_record_version(rec: dict[str, Any]) -> None:
    """Validate a record's ``v`` field — tolerant reader.

    Missing ``v`` is the v0 legacy format and always accepted; an
    int ``v`` up to :data:`RECORD_VERSION` is accepted; anything else
    (non-int, bool, or a newer schema) is a named decline — the
    reader never silently interprets a format it does not know.
    """
    v = rec.get("v")
    if v is None:
        return
    if isinstance(v, bool) or not isinstance(v, int):
        raise _StageError(f"record 'v' must be an int schema version, got {v!r:.100}")
    if v > RECORD_VERSION:
        raise _StageError(
            f"record 'v'={v} is newer than this endpoint's schema "
            f"(v{RECORD_VERSION}) — upgrade mlopt or regenerate the record"
        )


def _record_int(rec: dict[str, Any], key: str, default: int) -> int:
    raw = rec.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise _StageError(f"record '{key}' must be an int, got {raw!r:.100}")
    return raw


def _record_rules(rec: dict[str, Any]) -> Any:
    rules = rec.get("rules")
    if rules is not None and not isinstance(rules, str):
        raise _StageError(f"record 'rules' must be a preset name string, got {rules!r:.100}")
    return rules


def _lift_source(spec: Any) -> Any:
    """The spec's ``Source`` port — ``HnlSource`` for ``.hnl``, torch else."""
    src = getattr(spec, "source", None)
    if src is not None:
        return src
    from kit_torch.adapters import TorchSource

    return TorchSource()


class _RecordSource:
    """A ``Source`` port serving the lift record's IR with fresh leaves.

    ``to_ir`` delegates to the spec's real source for the leaf values
    (params/buffers — tensors cannot cross the JSON boundary) but
    returns the *record's* deserialized IR, so ``search`` saturates
    exactly the program ``lift`` emitted.  A re-export that disagrees
    with the record is recorded on ``.drifted`` — the stage surfaces
    it as a note rather than silently searching a different program.
    """

    def __init__(self, inner: Any, ir: Any) -> None:
        self._inner = inner
        self._ir = ir
        self.drifted = False

    def to_ir(self, model: Any, example_inputs: Any) -> tuple[Any, dict[str, Any]]:
        """``(record IR, fresh leaf values)`` — the Source contract."""
        fresh_ir, leaves = self._inner.to_ir(model, example_inputs)
        # Drift = ANY part of the lifted program disagreeing — the
        # root term AND the declared inputs/param table.  ``_ir_encode``
        # encodes all of it (leaves are name/shape-keyed terms), so a
        # re-export that only renames or reorders a leaf still counts.
        self.drifted = _ir_encode(fresh_ir) != _ir_encode(self._ir)
        return self._ir, leaves


def _optimizer(source: Any = None, rules: Any = None) -> Any:
    """``Optimizer(TorchBackend())`` — the same wiring ``optimize_model`` uses."""
    from kit_torch.backend import TorchBackend
    from mlopt_orchestrator.optimize import Optimizer

    kw: dict[str, Any] = {"backend": TorchBackend()}
    if source is not None:
        kw["source"] = source
    if rules is not None:
        kw["rules"] = rules
    return Optimizer(**kw)


# ---------------------------------------------------------------------------
#  The stages — each returns ``(record, exit_code)``
# ---------------------------------------------------------------------------


def _stage_lift(inp: Any) -> tuple[dict[str, Any], int]:
    """Model ref → lifted-IR record."""
    if isinstance(inp, str):
        model, adapter_opts = inp.strip(), {}
        rules = max_iterations = None
    elif isinstance(inp, dict):
        _check_record_version(inp)
        model, adapter_opts = _record_model_opts(inp)
        rules = _record_rules(inp)
        max_iterations = inp.get("max_iterations")
        if max_iterations is not None and (
            isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
        ):
            raise _StageError(
                f"record 'max_iterations' must be an int, got {max_iterations!r:.100}"
            )
    else:
        raise _StageError(
            f"lift expects a model-ref string or {{'model': ...}} object, got {type(inp).__name__}"
        )
    if not model:
        raise _StageError("lift: empty model ref")

    spec, adapter_name = _resolve_spec(model, adapter_opts)
    source = _lift_source(spec)
    try:
        ir, leaves = source.to_ir(spec.module, spec.inputs)
    except Exception as e:  # noqa: BLE001 — lift failure is a stage failure
        raise _StageError(f"lift (to_ir) failed: {type(e).__name__}: {e}") from e

    leaf_names = sorted(str(k) for k in leaves) if hasattr(leaves, "keys") else []
    meta = getattr(spec, "meta", None) or {}
    rec = _base_record(model, adapter_opts)
    rec.update(
        {
            "stage": "lift",
            "adapter": adapter_name,
            "ir": _ir_encode(ir),
            "leaf_names": leaf_names,
            "leaf_source": meta.get("leaf_source"),
            "input_source": meta.get("input_source"),
            "leaves_meta": meta.get("leaves"),
            "ok": True,
        }
    )
    notes = [
        "leaf values are re-materialised per stage from the adapter "
        "(tensors do not cross the JSON boundary)"
    ]
    if meta.get("leaf_source") in ("synthetic", "mixed"):
        notes.append(
            f"leaf values: {meta['leaf_source']} — .hnl artifacts carry "
            "shapes, not data; pass adapter_opts.feed for real values"
        )
    if meta.get("input_source") in ("synthetic", "mixed"):
        notes.append(
            f"input values: {meta['input_source']} — the spec declares "
            "shapes, not data; pass adapter_opts.inputs/feed for real values"
        )
    if rules is not None:
        rec["rules"] = rules
    if max_iterations is not None:
        rec["max_iterations"] = max_iterations
    rec["notes"] = notes
    return rec, 0


def _stage_search(inp: Any) -> tuple[dict[str, Any], int]:
    """Lift record → searched record (extracted term + search stats)."""
    if not isinstance(inp, dict):
        raise _StageError(f"search expects the lift record object, got {type(inp).__name__}")
    model, opts = _record_model_opts(inp)
    if "ir" not in inp:
        raise _StageError("search: record carries no lifted 'ir' — run `lift` first")
    ir = _ir_decode(inp["ir"])
    rules = _record_rules(inp)
    max_iterations = _record_int(inp, "max_iterations", 4)

    spec, _adapter_name = _resolve_spec(model, opts)
    inner = _lift_source(spec)
    source = _RecordSource(inner, ir)
    optz = _optimizer(source=source, rules=rules)

    t0 = time.time()
    try:
        res = optz.search(spec.module, spec.inputs, max_iterations=max_iterations)
    except Exception as e:  # noqa: BLE001 — search failure is a stage failure
        raise _StageError(f"search failed: {type(e).__name__}: {e}") from e
    search_s = time.time() - t0

    rec = dict(inp)
    rec.update(
        {
            "stage": "search",
            # Re-encode the lifted IR too: a v≤2 input record's
            # ``ir`` payload is legacy-spelled — the emitted record is
            # stamped v3, so its payloads are normalised to the v3
            # envelope rather than propagating the old spelling.
            "ir": _ir_encode(ir),
            "term": _term_encode(res.term),
            "n_enodes": (res.stats or {}).get("n_enodes"),
            "stats": _stats_to_data(res.stats),
            "search_s": round(search_s, 3),
            "ok": True,
        }
    )
    notes = list(rec.get("notes") or [])
    if source.drifted:
        notes.append(
            "re-export drift: the spec's fresh to_ir disagrees with the "
            "record's lifted IR — search ran on the record's term"
        )
    if notes:
        rec["notes"] = notes
    return rec, 0


def _rebuild_result(
    inp: dict[str, Any],
    spec: Any,
    ir: Any,
    term: Any,
    *,
    stage: str,
) -> tuple[Any, Any]:
    """Rebuild the ``SearchResult`` ``lower`` consumes + fresh leaves.

    Shared by ``verify`` and ``deliver`` — both stages are fresh
    processes, so each re-materialises leaf values from the spec's
    real source and re-assembles the handle ``Optimizer.lower`` reads
    (``ir`` / ``term`` / ``param_values`` / ``stats`` / ``task``).
    ``eg`` / ``root_eid`` are never read by ``lower`` — and post-
    ADR-0178 ``SearchResult.eg`` is nullable, so the honest handle
    carries ``eg=None`` / ``root_eid=-1`` rather than a fabricated
    empty ``EGraph`` (the last e-graph import in this package).
    """
    inner = _lift_source(spec)
    try:
        _fresh_ir, leaves = inner.to_ir(spec.module, spec.inputs)
    except Exception as e:  # noqa: BLE001
        raise _StageError(f"{stage} (to_ir) failed: {type(e).__name__}: {e}") from e

    from mlopt_core.pipeline import SearchResult

    res = SearchResult(
        ir=ir,
        eg=None,
        root_eid=-1,
        term=term,
        param_values=dict(leaves),
        stats=dict(inp.get("stats") or {}),
        source=getattr(spec, "source", None),
        model=spec.module,
    )
    return res, leaves


def _lower_decline_rec(
    inp: dict[str, Any], stage: str, e: Exception, t0: float, **extra: Any
) -> tuple[dict[str, Any], int]:
    """The ``ok:false`` record for a declined/failed ``lower`` gate."""
    # VerifyDeclinedError carries the declined LowerResult on
    # ``.result`` — the verify report is still inspectable.
    declined = getattr(e, "result", None)
    verified = getattr(declined, "verified", None) if declined is not None else None
    rec = dict(inp)
    rec.update(
        {
            "stage": stage,
            "ok": False,
            "error": f"{stage} declined: {type(e).__name__}: {e}",
            f"{stage}_s": round(time.time() - t0, 3),
            **extra,
        }
    )
    if verified is not None:
        rec["verified"] = _verified_to_data(verified)
    return rec, 1


def _stage_verify(inp: Any) -> tuple[dict[str, Any], int]:
    """Search record → verify report record (the real verify gate)."""
    if not isinstance(inp, dict):
        raise _StageError(f"verify expects the search record object, got {type(inp).__name__}")
    model, opts = _record_model_opts(inp)
    if "ir" not in inp:
        raise _StageError("verify: record carries no lifted 'ir' — run `lift` first")
    if "term" not in inp:
        raise _StageError("verify: record carries no searched 'term' — run `search` first")
    ir = _ir_decode(inp["ir"])
    term = _term_decode(inp["term"])

    spec, _adapter_name = _resolve_spec(model, opts)
    res, _leaves = _rebuild_result(inp, spec, ir, term, stage="verify")
    optz = _optimizer(rules=_record_rules(inp))

    t0 = time.time()
    try:
        lowered = optz.lower(res, spec.inputs, verify=True)
    except Exception as e:  # noqa: BLE001 — declined gate / lower failure
        return _lower_decline_rec(inp, "verify", e, t0)
    verify_s = time.time() - t0

    verified = getattr(lowered, "verified", None)
    passed = bool(getattr(verified, "passed", False)) if verified is not None else False
    rec = dict(inp)
    rec.update(
        {
            "stage": "verify",
            # Normalise payloads to the v3 envelope (see ``search``).
            "ir": _ir_encode(ir),
            "term": _term_encode(term),
            "ok": passed,
            "verify_s": round(verify_s, 3),
        }
    )
    if verified is not None:
        rec["verified"] = _verified_to_data(verified)
    return rec, 0 if passed else 1


def _artifact_feed(term: Any, ir: Any, leaves: Any, inputs: Any) -> dict[str, Any]:
    """``emit_program`` feed: param leaves + positional var leaves.

    ``leaves`` is the ``to_ir`` leaf-value dict (``Param`` name →
    tensor).  ``Var`` leaves bind positionally: ``ir.inputs`` order
    is the source contract's input order, zipped against ``inputs``
    (``spec.inputs``) by name — the same binding ``lower``'s verify
    oracle uses.  A var the record's term does not use is simply
    never referenced; an arity mismatch between ``ir.inputs`` and
    ``inputs`` is a named decline (``_StageError``), never a silent
    ``zip`` truncation.
    """
    feed: dict[str, Any] = dict(leaves or {})
    inputs = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    vars_ = list(getattr(ir, "inputs", ()) or ())
    if len(vars_) != len(inputs):
        # Positional binding cannot be honest under an arity mismatch:
        # ``zip`` would silently truncate and a used ``Var`` could go
        # unbound (the emit decline would catch it, but under a
        # generic "missing leaves" label).  Decline here, named.
        raise _StageError(
            "deliver declined: input arity mismatch — the record's IR "
            f"declares {len(vars_)} input leaves but the spec "
            f"materialised {len(inputs)} inputs; refusing to bind "
            "positionally"
        )
    try:
        from mlopt.emit import leaf_kinds

        used_vars = {n for n, kind in leaf_kinds(term) if kind == "var"}
    except Exception:  # noqa: BLE001 — emit import/shape failure declines later
        used_vars = set()
    for var, val in zip(vars_, inputs, strict=True):
        name = getattr(var, "name", None)
        if name is not None and (not used_vars or name in used_vars):
            feed[name] = val
    return feed


def _stage_deliver(inp: Any) -> tuple[dict[str, Any], int]:
    """Verify record → delivered-artifact record (the real delivery path).

    The record's ``verified`` block is a *claim* — deliver refuses a
    record whose claim is not ``passed is True`` and then re-runs the
    real ``lower(verify=True)`` gate anyway.  The shipped artifact is
    the verified extracted term rendered as a self-contained ``.hnl``
    program (:func:`mlopt.emit.emit_program`) at the record's ``out``
    path — runnable by ``hnl-run`` / the host-tensor backend.  The
    runner-selection/bench half of ``optimize --deliver``
    (region_fuse/cuda_graph timing) is deliberately not re-run here:
    it measures the torch module, not the term artifact.
    """
    if not isinstance(inp, dict):
        raise _StageError(f"deliver expects the verify record object, got {type(inp).__name__}")
    model, opts = _record_model_opts(inp)
    if "ir" not in inp:
        raise _StageError("deliver: record carries no lifted 'ir' — run `lift` first")
    if "term" not in inp:
        raise _StageError("deliver: record carries no searched 'term' — run `search` first")
    verified_claim = inp.get("verified")
    if not isinstance(verified_claim, dict):
        raise _StageError("deliver: record carries no 'verified' report — run `verify` first")
    if verified_claim.get("passed") is not True:
        rec = dict(inp)
        rec.update(
            {
                "stage": "deliver",
                "ok": False,
                "delivered": False,
                "error": "deliver refused: the record's verify report did not pass",
            }
        )
        return rec, 1
    out = inp.get("out")
    if not isinstance(out, str) or not out.strip():
        raise _StageError(
            "deliver: record carries no 'out' artifact path — set 'out' to the .hnl destination"
        )
    out = out.strip()
    ir = _ir_decode(inp["ir"])
    term = _term_decode(inp["term"])

    spec, _adapter_name = _resolve_spec(model, opts)
    res, leaves = _rebuild_result(inp, spec, ir, term, stage="deliver")
    optz = _optimizer(rules=_record_rules(inp))

    t0 = time.time()
    try:
        lowered = optz.lower(res, spec.inputs, verify=True)
    except Exception as e:  # noqa: BLE001 — declined gate / lower failure
        return _lower_decline_rec(inp, "deliver", e, t0, delivered=False)
    deliver_s = time.time() - t0

    verified = getattr(lowered, "verified", None)
    passed = bool(getattr(verified, "passed", False)) if verified is not None else False
    if not passed:
        # ``lower(verify=True)`` raises on a failed gate, so this is
        # defensive — a missing/failed report is still a decline,
        # never a ship.
        rec = dict(inp)
        rec.update(
            {
                "stage": "deliver",
                "ok": False,
                "delivered": False,
                "deliver_s": round(deliver_s, 3),
                "error": "deliver declined: lower returned no passing verify report",
            }
        )
        if verified is not None:
            rec["verified"] = _verified_to_data(verified)
        return rec, 1

    # Emit the VERIFIED term (``res.term`` — what the gate certified)
    # as a runnable .hnl program.  The feed re-binds every leaf to the
    # freshly materialised values — the same tensors the gate ran on.
    from mlopt.emit import (
        KitRunner,
        UnsupportedOpError,
        emit_program,
        find_hnl_run,
    )

    try:
        feed = _artifact_feed(res.term, ir, leaves, spec.inputs)
    except _StageError as e:
        rec = dict(inp)
        rec.update(
            {
                "stage": "deliver",
                "ok": False,
                "delivered": False,
                "deliver_s": round(time.time() - t0, 3),
                "error": str(e),
            }
        )
        return rec, 1
    try:
        src = emit_program(res.term, feed)
    except (UnsupportedOpError, ValueError) as e:
        rec = dict(inp)
        rec.update(
            {
                "stage": "deliver",
                "ok": False,
                "delivered": False,
                "deliver_s": round(deliver_s, 3),
                "error": f"deliver declined: emit failed — {type(e).__name__}: {e}",
            }
        )
        return rec, 1

    # Execution smoke BEFORE the artifact lands: the verify verdict
    # covers the term/module, not the emitted program's hnl-run
    # semantics — a program the kit rejects must not ship.  With no
    # ``hnl-run`` binary the artifact still ships, but the record
    # says ``artifact_executed: false`` and the note scopes the
    # verdict honestly.  The smoke evaluates the program; it is NOT
    # a numeric re-verify (``mlopt export --verify-run`` is that gate).
    artifact_executed = False
    binary = find_hnl_run()
    if binary is not None:
        try:
            KitRunner(binary).run(src)
        except Exception as e:  # noqa: BLE001 — a rejected program is a decline
            rec = dict(inp)
            rec.update(
                {
                    "stage": "deliver",
                    "ok": False,
                    "delivered": False,
                    "artifact_executed": False,
                    "deliver_s": round(deliver_s, 3),
                    "error": (
                        f"deliver declined: hnl-run rejected the artifact — {type(e).__name__}: {e}"
                    ),
                }
            )
            return rec, 1
        artifact_executed = True

    try:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(src + "\n")
    except OSError as e:
        rec = dict(inp)
        rec.update(
            {
                "stage": "deliver",
                "ok": False,
                "delivered": False,
                "deliver_s": round(deliver_s, 3),
                "error": f"deliver declined: cannot write {out}: {e}",
            }
        )
        return rec, 1

    rec = dict(inp)
    rec.update(
        {
            "stage": "deliver",
            # Normalise payloads to the v3 envelope (see ``search``).
            "ir": _ir_encode(ir),
            "term": _term_encode(term),
            "ok": True,
            "delivered": True,
            "deliver_s": round(deliver_s, 3),
            "verified": _verified_to_data(verified),
            "artifact_executed": artifact_executed,
            "artifact": {
                "path": out,
                "format": "hnl",
                "runnable_by": "hnl-run / host-tensor",
            },
        }
    )
    notes = list(rec.get("notes") or [])
    notes.append(
        "deliver: artifact is the verified extracted term as a "
        "self-contained .hnl program (emit_program); the "
        "region_fuse/cuda_graph runner + bench pass is the "
        "`optimize --deliver` path, not re-run per stage"
    )
    if artifact_executed:
        notes.append(
            "deliver: artifact smoke-executed once through hnl-run "
            "(run check, not a numeric re-verify — "
            "`mlopt export --verify-run` is the numeric gate); the "
            "verify verdict covers the term/module, and the smoke "
            "covers the emitted program's hnl-run semantics"
        )
    else:
        notes.append(
            "deliver: artifact NOT executed — no hnl-run binary "
            "found (set $MLOPT_HNL_RUN); the verify verdict covers "
            "the term/module, NOT the emitted program's hnl-run "
            "semantics (f64 wire, dropped attrs, positional leaf "
            "binding)"
        )
    rec["notes"] = notes
    return rec, 0


_STAGE_FNS = {
    "lift": _stage_lift,
    "search": _stage_search,
    "verify": _stage_verify,
    "deliver": _stage_deliver,
}


# ---------------------------------------------------------------------------
#  Entry point — one JSON value in, one JSON value out
# ---------------------------------------------------------------------------


def run_stage(name: str, inp: Any) -> int:
    """Run stage *name* over stdin-value *inp*; return the exit code.

    Prints exactly one JSON record to stdout on every path where a
    record could be formed; diagnostics go to stderr.  Unknown stage
    names are a usage error (exit 2) — ``cli``'s argparse ``choices``
    normally keeps them from reaching here.
    """
    fn = _STAGE_FNS.get(name)
    if fn is None:
        print(
            f"mlopt stage: unknown stage {name!r} (known: {', '.join(STAGES)})",
            file=sys.stderr,
        )
        return 2
    try:
        rec, code = fn(inp)
    except _StageError as e:
        print(f"mlopt stage {name}: {e}", file=sys.stderr)
        rec = {"stage": name, "ok": False, "error": str(e), "v": RECORD_VERSION}
        if isinstance(inp, dict) and isinstance(inp.get("model"), str):
            rec["model"] = inp["model"]
        code = 1
    except Exception as e:  # noqa: BLE001 — never leak a traceback as protocol
        print(f"mlopt stage {name}: {type(e).__name__}: {e}", file=sys.stderr)
        rec = {
            "stage": name,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "v": RECORD_VERSION,
        }
        if isinstance(inp, dict) and isinstance(inp.get("model"), str):
            rec["model"] = inp["model"]
        code = 1
    # Every emitted record carries the schema pin — including the
    # ``ok:false`` decline records the stages build via ``dict(inp)``
    # (a v0 input would otherwise propagate the missing field).
    rec["v"] = RECORD_VERSION
    print(json.dumps(rec))
    return code
