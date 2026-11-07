"""Verify the phase-2 Models against their torch denotations (fp64 gate).

Plan 0033 phase 2 / ADR-0157: every ``models.yaml`` row whose ``hnl``
projection exists must reproduce its reference semantics on every
bound site at the shipped verify floor
(``handler_synth._VERIFY_RTOL`` = 1e-9, ``_VERIFY_ATOL`` = 1e-12 —
the same ``allclose`` gate the mint measured under) before its
verdict may read ``handl-shipped``.  Anything else is honestly
``python-shipped`` — ``TorchSink`` stays the delivery oracle.

For each model row:

* ``kind: "denotation"`` — the ``.hnl`` is the ``synop_*`` op's
  definitional spec projected; the reference is the same spec
  evaluated through ``mlopt_torch.torch_bridge._IR_TO_TORCH``.
* ``kind: "handler"`` — the ``.hnl`` is the minted ``handler_body``
  projected; the reference is the handler *pattern* spec (what the
  shipped ``verify_body`` measured the body against).

The bound-site domain is the mint's recorded ``provenance.domain``
(``{mvar: shape}`` entries — the shapes the handler was verified
under); rows without one (the five handler-refused prims) and rows
the ``tensor.*`` surface cannot express are verdicted
``python-shipped`` with the reason recorded — never silently skipped.

``eval_hnl`` is ``bench.drivers.kit.term_hnl.KitRunner`` — the
compiled ``hnl-run`` binary (``cargo build -p mlopt-kit --bin
hnl-run``; ``$MLOPT_HNL_RUN`` or ``--hnl-run`` overrides).  Without
it the script refuses loudly (exit 2): a parity gate with no parity
partner is not a gate.

    .venv/bin/python scripts/verify_vocab_models.py [vocabdir]
    .venv/bin/python scripts/verify_vocab_models.py --hnl-run PATH

Verdicts land in ``models.yaml``'s ``verdicts`` map via
``vocabdir.upsert_model_verdicts`` — each carries the ``hnl`` text it
measured, so a later spec change makes the stale verdict visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mlopt_core.cells import spec_from_data  # noqa: E402
from mlopt_core.laws import minted  # noqa: E402
from mlopt_discovery import (  # noqa: E402
    handler_synth,
    modelproj,
    vocabdir,
)


def _domains() -> dict[str, list]:
    """Map ``synop_*`` op → its minted handler's ``provenance.domain``."""
    out: dict[str, list] = {}
    for name, rec in minted.prim_records().items():
        prov = ((rec.get("handler") or {}).get("record") or {}).get(
            "provenance"
        ) or {}
        domain = prov.get("domain")
        if domain:
            out[name] = domain
    return out


def _handler_patterns() -> dict[str, Any]:
    """Map ``synop_*`` op → its minted handler's decoded pattern spec."""
    from mlopt_discovery.handler_synth import body_from_data

    out: dict[str, Any] = {}
    for name, rec in minted.prim_records().items():
        record = (rec.get("handler") or {}).get("record")
        if record and record.get("pattern") is not None:
            out[name] = body_from_data(record["pattern"])
    return out


def _python_shipped(row: dict, detail: str) -> dict:
    """Return the honest negative verdict in ``verify_model``'s shape."""
    return {
        "op": row["op"],
        "status": "python-shipped",
        "rtol": handler_synth._VERIFY_RTOL,
        "atol": handler_synth._VERIFY_ATOL,
        "max_err": float("inf"),
        "max_rel": float("inf"),
        "sites": 0,
        "detail": detail,
        "hnl": row.get("hnl"),
    }


def collect_verdicts(
    vocab: vocabdir.Vocab,
    eval_hnl,
    *,
    seed: int = 0,
    adversarial: int = 1,
) -> tuple[dict[str, dict], dict[str, int]]:
    """Verify every model row; return ``(verdicts, status_counts)``.

    *eval_hnl* is the ``.hnl``-source → decoded-JSON callable
    (``KitRunner.run``).  Rows are verified in id order so the
    committed ``verdicts`` map diff is stable.
    """
    domains = _domains()
    patterns = _handler_patterns()
    verdicts: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for row in sorted(vocab.models, key=lambda r: r["id"]):
        mid = row["id"]
        op = row["op"]
        hnl = row.get("hnl")
        domain = domains.get(op)
        if hnl is None:
            v = _python_shipped(
                row, f"unexpressible: {row.get('unsupported')}"
            )
        elif domain is None:
            v = _python_shipped(row, "no recorded bound-site domain")
        else:
            if row["kind"] == "denotation":
                ref_spec = spec_from_data(row["spec"])
            else:
                ref_spec = patterns.get(op)
            if ref_spec is None:
                v = _python_shipped(
                    row, "no handler pattern for the reference spec"
                )
            else:
                v = modelproj.verify_model(
                    name=op,
                    hnl=hnl,
                    ref_spec=ref_spec,
                    domain=domain,
                    mvars=row["mvars"],
                    eval_hnl=eval_hnl,
                    seed=seed,
                    adversarial=adversarial,
                    rtol=handler_synth._VERIFY_RTOL,
                    atol=handler_synth._VERIFY_ATOL,
                )
                v["hnl"] = hnl
        verdicts[mid] = v
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    return verdicts, counts


def main(argv: list[str] | None = None) -> int:
    """Run the gate over the vocabulary; write the verdicts."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "vocabdir",
        nargs="?",
        default=None,
        help="vocabulary directory (default: the repo vocab/)",
    )
    ap.add_argument(
        "--hnl-run",
        default=None,
        help="path to the hnl-run binary "
        "(default: $MLOPT_HNL_RUN or target/debug/hnl-run)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adversarial", type=int, default=1)
    args = ap.parse_args(argv)

    directory = Path(args.vocabdir) if args.vocabdir else None
    vocab = vocabdir.load_vocab(
        directory or vocabdir.default_vocab_dir() or _REPO / "vocab"
    )

    from bench.drivers.kit.term_hnl import KitRunner

    try:
        runner = KitRunner(binary=args.hnl_run)
    except FileNotFoundError as exc:
        print(f"verify_vocab_models: {exc}", file=sys.stderr)
        return 2

    verdicts, counts = collect_verdicts(
        vocab,
        runner.run,
        seed=args.seed,
        adversarial=args.adversarial,
    )
    vocabdir.upsert_model_verdicts(vocab.directory, verdicts)
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    shipped = [
        (mid, v["max_rel"])
        for mid, v in verdicts.items()
        if v["status"] == "handl-shipped"
    ]
    if shipped:
        worst = max(shipped, key=lambda kv: kv[1])
        print(
            f"  worst max_rel among handl-shipped: "
            f"{worst[0]} = {worst[1]:.3e}"
        )
    print(
        f"wrote {len(verdicts)} verdicts to {vocab.directory}/models.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
