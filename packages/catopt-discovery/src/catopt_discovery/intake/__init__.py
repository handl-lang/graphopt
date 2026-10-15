"""Workload intake — feed the census programs nobody wrote by hand.

The corpus is hand-curated: forty ``catopt_torch.models`` classes
registered in ``catopt_discovery.impact._model_cases``.  ``law-workload-gen.md``
proved resampling cannot mint new op-tuples and
``corpus-expansion-r2.md`` proved real architectures can — but every
expansion round meant writing a new builder by hand and registering
it in code.  The missing half of self-play is an **intake path**: a
tool that takes a ``torch.nn.Module`` + example input, runs the real
export boundary (``torch.export`` → ``export_to_ir``), and appends
whatever survives to the census — recording, just as honestly, what
the boundary rejects.

What counts as a candidate (``candidates()``):

* **torch-native modules** — ``nn.MultiheadAttention``,
  ``nn.TransformerEncoder`` stacks, ``nn.BatchNorm``,
  ``nn.Embedding``, the RNN family, pooling, losses, ….  Library
  code spells ops differently than our hand builders (``permute``,
  ``unflatten``, ``split_with_sizes``, ``feature_dropout``); every
  spelling the bridge cannot lower is a recorded finding, not a
  crash.
* **compound models** — hand-built but realistic assemblies (a
  shared-expert MoE, a ViT patch block, a bottleneck residual, a
  classifier loss head) at real depth.

Every candidate lands in exactly one class:

* **ingested** — exported, every op bound, and the lowered module
  verifies fp64 against the original.  These join the firing/reach
  probe under a node cap — deeper stacks still feed the census and
  the matchers.
* **census-only** — exported but at least one op has no torch
  binding, or the lowered module fails verification.  The term still
  feeds the census (a real program's shape is a real shape); the
  missing ops are the binding-gap backlog this tool exists to
  surface.
* **rejected** — ``torch.export`` / ``export_to_ir`` raised; the
  exception is recorded verbatim.

The census seam is a side-file, not a code edit: a run emits
``tools/data/intake_corpus.json`` (terms via ``catopt_core.ir.term_to_data``
plus per-workload metadata and the rejection backlog) and
``tools/data/intake_tensors.pt`` (the feeds/params ``torch.save``'d, so a
loaded term lowers and verifies exactly like an exported model).
``catopt_discovery.census.corpus()`` unions :func:`load_cases` when the
file exists — the file IS the appended census — and
``catopt_discovery.pipeline.run_pipeline`` reads the same file for its matcher
terms and (via :func:`probe_cases`) its firing probe.  Absent the
file, nothing changes: the baseline corpus is untouched.

Run::

    .venv/bin/python -m catopt_discovery.intake            # ingest + report
    .venv/bin/python -m catopt_discovery.intake --skip-pipeline
    .venv/bin/python -m catopt_discovery.intake --json /tmp/intake.json
    .venv/bin/python -m catopt_discovery.intake --holdout select_mul

CPU-only, a few minutes when the pipeline delta runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from catopt_torch.adapters import TorchSink

from catopt_discovery.impact import (
    TermCase,
    _bench_cases,
    model_cases,
)

from .ledger import (
    _DEFAULT_JSON,
    _DEFAULT_TENSORS,
    Rejection,
    _ingest_one,
    _tensors_of,
    census_delta,
    load_cases,
    load_records,
    probe_cases,
    write_intake,
)
from .ledger import (
    _PROBE_MAX_NODES as _PROBE_MAX_NODES,
)
from .ledger import (
    _RTOL as _RTOL,
)
from .ledger import (
    _n_nodes as _n_nodes,
)
from .ledger import (
    _ops_of as _ops_of,
)
from .ledger import (
    _verify as _verify,
)
from .models import (
    _SincKernel as _SincKernel,
)
from .registry import (
    _PURPOSE_BUILT,
    Workload,
    candidates,
    real_workloads,
)
from .report import (
    _delta_table,
    _eligible,
    _gap_table,
    _rejection_table,
    _run_delta,
    _status_table,
)
from .report import (
    _intake_fires as _intake_fires,
)

__all__ = [
    "Rejection",
    "Workload",
    "candidates",
    "census_delta",
    "ingest",
    "load_cases",
    "load_records",
    "main",
    "probe_cases",
    "real_workloads",
    "write_intake",
]


# ---------------------------------------------------------------------------
#  Ingestion — export, classify, verify
# ---------------------------------------------------------------------------


def ingest(
    cands: list[Workload] | None = None,
    sink: TorchSink | None = None,
) -> tuple[list[TermCase], list[dict], list[Rejection]]:
    """Run every candidate through the export boundary.

    Returns ``(cases, records, rejections)``: the ``TermCase``s of
    every term that exported (any status), the per-workload metadata
    records the side-file persists, and the hard rejections.
    """
    sink = sink or TorchSink()
    supported = sink.supported_ops
    cases: list[TermCase] = []
    records: list[dict] = []
    rejections: list[Rejection] = []
    for cand in cands if cands is not None else candidates():
        case, rec, rej = _ingest_one(cand, sink, supported)
        if rej is not None:
            rejections.append(rej)
            continue
        # _ingest_one yields a rejection xor a (case, rec) pair.
        cases.append(cast("TermCase", case))
        records.append(cast("dict", rec))
    return cases, records, rejections


# ---------------------------------------------------------------------------
#  Reporting + driver
# ---------------------------------------------------------------------------


def _report_pipeline(
    cases: list[TermCase],
    records: list[dict],
    args: argparse.Namespace,
    payload: dict,
) -> None:
    """Run the baseline-vs-enlarged pipeline delta and report it."""
    bench, _be = _bench_cases()
    models, _me = model_cases()
    probe = _eligible(cases, records)
    print(  # stdout-compat
        f"   probe-eligible intake cases: {len(probe)}"
    )
    res = _run_delta(
        [*bench, *models],
        models,
        cases,
        probe,
        args.vocab,
        args.holdout,
    )
    print(_delta_table(res))  # stdout-compat
    e = res["enlarged"]
    payload["pipeline"] = {
        "baseline_terms": res["baseline"]["n_terms"],
        "enlarged_terms": e["n_terms"],
        "baseline_tuples": res["baseline"]["n_tuples"],
        "enlarged_tuples": e["n_tuples"],
        "new_proposals": sorted(
            {ev.proposal.name for ev in e["ranked"]}
            - {ev.proposal.name for ev in res["baseline"]["ranked"]}
        ),
        "shippable": [
            ev.proposal.name for ev in e["ranked"] if ev.shippable
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Ingest candidates, measure the census delta, write the file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_JSON,
        help="side-file census (default tools/data/intake_corpus.json)",
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=_DEFAULT_TENSORS,
        help="feed/param tensor blob (tools/data/intake_tensors.pt)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="run the measurement without writing the side files",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vocab", choices=("hand", "derived"), default="derived"
    )
    parser.add_argument("--holdout", help="pipeline holdout rules")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="ingest only real workloads (drop the purpose-built spellings)",
    )
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    print(  # stdout-compat
        "== law_intake — real workloads into the census =="
    )
    sink = TorchSink()
    cands = real_workloads() if args.real_only else None
    if cands is not None:
        print(  # stdout-compat
            f"   --real-only: {len(cands)} real workloads "
            f"(of {len(candidates())}; "
            f"{len(_PURPOSE_BUILT)} purpose-built dropped)"
        )
    cases, records, rejections = ingest(cands=cands, sink=sink)
    n_in = sum(1 for r in records if r["status"] == "ingested")
    n_co = sum(1 for r in records if r["status"] == "census-only")
    n_vf = sum(1 for r in records if r["status"] == "verify-failed")
    print(  # stdout-compat
        f"   {len(records)} exported: {n_in} ingested, "
        f"{n_co} census-only, {n_vf} verify-failed; "
        f"{len(rejections)} rejected"
    )

    bench, _be = _bench_cases()
    models, _me = model_cases()
    delta = census_delta([*bench, *models], cases)
    print()  # stdout-compat
    print("-- intake table --")  # stdout-compat
    print(_status_table(records, delta))  # stdout-compat
    print()  # stdout-compat
    print(  # stdout-compat
        f"-- census delta (intake vs {len(bench)} bench + "
        f"{len(models)} models) --"
    )
    print(  # stdout-compat
        f"   {delta['n_base_tuples']} -> "
        f"{delta['n_base_tuples'] + len(delta['new_op_tuples'])} "
        f"op-tuples, +{delta['new_shapes']} shapes, "
        f"{len(delta['new_ops'])} new ops: {delta['new_ops']}"
    )
    print()  # stdout-compat
    print(  # stdout-compat
        "-- binding-gap backlog (exported, unbound) --"
    )
    print(_gap_table(records))  # stdout-compat
    print()  # stdout-compat
    print("-- export rejections --")  # stdout-compat
    print(_rejection_table(rejections))  # stdout-compat
    print()  # stdout-compat

    payload: dict[str, Any] = {
        "n_exported": len(records),
        "n_ingested": n_in,
        "n_census_only": n_co,
        "n_verify_failed": n_vf,
        "rejections": [asdict(r) for r in rejections],
        "census_delta": {
            "new_op_tuples": [
                f"{k[0]}({', '.join(k[1])})"
                for k in delta["new_op_tuples"]
            ],
            "new_shapes": delta["new_shapes"],
            "new_ops": delta["new_ops"],
        },
    }

    if not args.no_write:
        write_intake(
            records,
            rejections,
            _tensors_of(cases),
            args.out,
            args.tensors,
        )
        print(f"wrote {args.out} + {args.tensors}")  # stdout-compat
    else:
        print("(--no-write: side files not written)")  # stdout-compat

    if not args.skip_pipeline:
        print()  # stdout-compat
        print(  # stdout-compat
            "-- pipeline delta (baseline vs corpus+intake) --"
        )
        _report_pipeline(cases, records, args, payload)

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0
