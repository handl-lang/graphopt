"""The census ledger — export, classify, verify, persist, diff.

The export boundary (``torch.export`` → ``export_to_ir``), the
ingested / census-only / rejected classification, the
``tools/data/intake_corpus.json`` + ``intake_tensors.pt`` side-file
round-trip (``catopt_discovery.census.corpus`` and
``pipeline.run_pipeline`` union it when present), and the census
delta against the baseline corpus.  The ``ingest`` orchestrator
itself lives in the package ``__init__`` (it drives ``candidates`` /
``_ingest_one`` through the package namespace the test suite patches).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from catopt_core.ir import (
    Op,
    TensorType,
    Var,
    term_from_data,
    term_to_data,
)
from catopt_torch.adapters import TorchSink
from catopt_torch.torch_bridge import export_to_ir

from catopt_discovery import TOOLS
from catopt_discovery.impact import (
    TermCase,
    _ir_of,
    _iter_subterms,
)

from .registry import Workload

#: The default side-file paths — the appended census + its tensors,
#: which stayed behind in ``tools/`` when the engine moved here.
_DEFAULT_JSON = TOOLS / "intake_corpus.json"
_DEFAULT_TENSORS = TOOLS / "intake_tensors.pt"

#: Verify tolerance for the lower-the-term-vs-original-model check.
_RTOL = 1e-4

#: Node cap for the firing/reach probe.  Deep stacks (the 105-node
#: TransformerEncoder(3), the 271-node full Transformer) still feed
#: the census and the matchers; the probe stays comparable in cost
#: to the corpus's own models (~80 nodes).
_PROBE_MAX_NODES = 120

# ---------------------------------------------------------------------------
#  Ingestion — export, classify, verify
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rejection:
    """A candidate the export boundary refused, with the reason."""

    name: str
    stage: str
    error: str


def _ops_of(term: Any) -> set[str]:
    """Return the distinct op names inside *term*."""
    return {s.op for s in _iter_subterms(term) if isinstance(s, Op)}


def _n_nodes(term: Any) -> int:
    """Return the op-node count of *term*."""
    return sum(1 for s in _iter_subterms(term) if isinstance(s, Op))


def _verify(
    model: nn.Module,
    ir: Any,
    tensors: dict,
    feed: tuple,
    sink: TorchSink,
) -> str:
    """Lower the exported IR and verify it against the module.

    Returns ``"pass"``, ``"FAIL"`` or an error tag — the same
    ``sink.lower`` + ``sink.verify`` path the corpus tests run.
    """
    try:
        lowered = sink.lower(
            _ir_of(ir.root, tuple(ir.inputs)), dict(tensors)
        )
        vr = sink.verify(model, lowered, feed, rtol=_RTOL)
    except Exception as e:
        return f"error:{type(e).__name__}: {e}"
    return "pass" if vr.passed else f"FAIL max_rel={vr.max_rel:.2e}"


def _ingest_one(
    cand: Workload, sink: TorchSink, supported: frozenset
) -> tuple[TermCase | None, dict | None, Rejection | None]:
    """Run one candidate through the boundary; classify the result."""
    try:
        model, x = cand.build()
    except Exception as e:
        return (
            None,
            None,
            Rejection(cand.name, "build", f"{type(e).__name__}: {e}"),
        )
    feed = x if isinstance(x, tuple) else (x,)
    try:
        ir, tensors = export_to_ir(model.eval().double(), feed)
    except Exception as e:
        return (
            None,
            None,
            Rejection(cand.name, "export", f"{type(e).__name__}: {e}"),
        )
    ops = _ops_of(ir.root)
    missing = sorted(ops - supported)
    rec: dict[str, Any] = {
        "name": f"intake:{cand.name}",
        "builder": cand.kind,
        "n_op_nodes": _n_nodes(ir.root),
        "ops": sorted(ops),
        "unsupported_ops": missing,
        "term": term_to_data(ir.root),
        "inputs": [
            {"name": v.name, "shape": list(v.typ.shape)}
            for v in ir.inputs
        ],
    }
    if missing:
        rec["status"] = "census-only"
        rec["verify"] = "skipped"
        rec["verify_note"] = f"unbound ops: {missing}"
    else:
        v = _verify(model, ir, tensors, feed, sink)
        rec["verify"] = v
        rec["status"] = "ingested" if v == "pass" else "verify-failed"
        rec["verify_note"] = "" if v == "pass" else v
    case = TermCase(
        source="intake",
        name=f"intake:{cand.name}",
        term=ir.root,
        inputs=tuple(ir.inputs),
        feed=tuple(feed),
        param_vals=dict(tensors),
    )
    return case, rec, None


# ---------------------------------------------------------------------------
#  Persistence — the side-file census
# ---------------------------------------------------------------------------


def write_intake(
    records: list[dict],
    rejections: list[Rejection],
    tensors_of: dict[str, dict],
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
) -> None:
    """Write the census contribution and the tensor blob.

    ``records`` carry the serialized terms; ``tensors_of`` maps each
    workload name to ``{"feed": [...], "params": {...}}`` — kept in a
    ``torch.save`` blob so the JSON stays diffable.
    """
    payload = {
        "format": 1,
        "tool": "law_intake",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "workloads": records,
        "rejections": [asdict(r) for r in rejections],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(tensors_of, tensors_path)


def load_records(path: Path = _DEFAULT_JSON) -> list[dict]:
    """Return the persisted workload records ([] when absent)."""
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("workloads", [])


def load_cases(
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
) -> list[TermCase]:
    """Rebuild intake ``TermCase``s from the side-file census.

    Terms decode through ``term_from_data``; feeds/params ride the
    companion ``torch.save`` blob (absent blob → empty feed — the
    census only reads ``.term`` anyway, and :func:`probe_cases`
    requires a feed).
    """
    tensor_blob: dict = {}
    if tensors_path.exists():
        tensor_blob = torch.load(tensors_path, weights_only=True)
    cases: list[TermCase] = []
    for rec in load_records(path):
        inputs = tuple(
            Var(i["name"], TensorType(tuple(i["shape"])))
            for i in rec["inputs"]
        )
        blob = tensor_blob.get(rec["name"], {})
        cases.append(
            TermCase(
                source="intake",
                name=rec["name"],
                term=term_from_data(rec["term"]),
                inputs=inputs,
                feed=tuple(blob.get("feed", ())),
                param_vals=dict(blob.get("params", {})),
            )
        )
    return cases


def probe_cases(
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
    max_nodes: int = _PROBE_MAX_NODES,
) -> list[TermCase]:
    """Return the intake cases eligible for the firing/reach probe.

    Eligibility is the intake contract: status ``"ingested"`` (all
    ops bound, lowered module verified), a persisted feed, and under
    the node cap that keeps the probe's saturation cost comparable
    to the corpus's own models.
    """
    status = {r["name"]: r["status"] for r in load_records(path)}
    nodes = {r["name"]: r["n_op_nodes"] for r in load_records(path)}
    return [
        c
        for c in load_cases(path, tensors_path)
        if status.get(c.name) == "ingested"
        and c.feed
        and nodes.get(c.name, 0) <= max_nodes
    ]


def _tensors_of(cases: list[TermCase]) -> dict[str, dict]:
    """Collect each case's feed/param tensors for ``torch.save``."""
    return {
        c.name: {"feed": list(c.feed), "params": dict(c.param_vals)}
        for c in cases
    }


# ---------------------------------------------------------------------------
#  Census delta — what the intake adds that the corpus did not have
# ---------------------------------------------------------------------------


def census_delta(base: list[TermCase], intake: list[TermCase]) -> dict:
    """Diff the intake terms' census keys against the base corpus.

    Returns the new op-tuples, new shapes and previously-absent ops,
    plus a per-workload novelty table for the report.
    """
    from catopt_discovery.census import (
        CorpusTerm,
        op_tuple_census,
        shape_census,
    )

    base_ct = [CorpusTerm(c.source, c.name, c.term) for c in base]
    in_ct = [CorpusTerm(c.source, c.name, c.term) for c in intake]
    base_op, _ = op_tuple_census(base_ct)
    base_sh, _ = shape_census(base_ct)
    in_op, _ = op_tuple_census(in_ct)
    in_sh, _ = shape_census(in_ct)
    base_ops = set()
    for c in base:
        base_ops |= _ops_of(c.term)
    in_ops = set()
    for c in intake:
        in_ops |= _ops_of(c.term)
    per_case = {}
    for c in intake:
        ct = CorpusTerm(c.source, c.name, c.term)
        op_c, _ = op_tuple_census([ct])
        sh_c, _ = shape_census([ct])
        per_case[c.name] = (
            sorted(set(op_c) - set(base_op)),
            len(set(sh_c) - set(base_sh)),
        )
    return {
        "n_base_tuples": len(base_op),
        "n_base_shapes": len(base_sh),
        "new_op_tuples": sorted(set(in_op) - set(base_op)),
        "new_shapes": len(set(in_sh) - set(base_sh)),
        "new_ops": sorted(in_ops - base_ops),
        "per_case": per_case,
    }
