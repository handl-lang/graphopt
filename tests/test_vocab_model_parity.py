"""Phase-2 parity, end to end: Models evaluated by the real ``hnl-run``.

``test_modelproj.py`` exercises the gate logic with an injected eval;
this file runs the actual boundary — ``bench.drivers.kit.term_hnl.
KitRunner`` → the compiled ``hnl-run`` binary (``cargo build -p
mlopt-kit --bin hnl-run``; ``$MLOPT_HNL_RUN`` overrides) — over a
sample of shipped Models and asserts the ``handl-shipped`` verdict
at the shipped fp64 floor.  Skips honestly when the binary is not
built (the same posture ``test_kit_bridge.py`` takes).

Marked ``e2e``: the suite crosses an external engine boundary (the
handl tree-walk runtime, via subprocess).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlopt_core.cells import spec_from_data
from mlopt_discovery import handler_synth, modelproj, vocabdir
from mlopt_discovery.handler_synth import body_from_data

pytestmark = pytest.mark.e2e

_REPO = Path(__file__).resolve().parents[1]
_VOCAB = _REPO / "vocab"


def _runner():
    try:
        from bench.drivers.kit.term_hnl import KitRunner
    except ImportError:  # pragma: no cover - bench always importable
        pytest.skip("bench.drivers.kit unavailable")
    try:
        return KitRunner()
    except FileNotFoundError as exc:
        pytest.skip(f"hnl-run not built: {exc}")


def _domains() -> dict[str, list]:
    from mlopt_core.laws import minted

    out: dict[str, list] = {}
    for name, rec in minted.prim_records().items():
        prov = ((rec.get("handler") or {}).get("record") or {}).get(
            "provenance"
        ) or {}
        if prov.get("domain"):
            out[name] = prov["domain"]
    return out


def _patterns() -> dict[str, object]:
    from mlopt_core.laws import minted

    out: dict[str, object] = {}
    for name, rec in minted.prim_records().items():
        record = (rec.get("handler") or {}).get("record")
        if record and record.get("pattern") is not None:
            out[name] = body_from_data(record["pattern"])
    return out


def _sample(rows: list[dict], kind: str, n: int = 3) -> list[dict]:
    return [r for r in rows if r["kind"] == kind and r.get("hnl")][:n]


def test_denotation_models_verify_handl_shipped() -> None:
    """A sample of ``synop_*`` denotations passes the real fp64 gate."""
    runner = _runner()
    vocab = vocabdir.load_vocab(_VOCAB)
    domains = _domains()
    checked = 0
    for row in _sample(list(vocab.models), "denotation"):
        if row["op"] not in domains:
            continue
        v = modelproj.verify_model(
            name=row["op"],
            hnl=row["hnl"],
            ref_spec=spec_from_data(row["spec"]),
            domain=domains[row["op"]],
            mvars=row["mvars"],
            eval_hnl=runner.run,
            adversarial=0,
            rtol=handler_synth._VERIFY_RTOL,
            atol=handler_synth._VERIFY_ATOL,
        )
        assert v["status"] == "handl-shipped", (row["id"], v)
        assert v["sites"] == 4
        checked += 1
    assert checked >= 1


def test_handler_models_verify_handl_shipped() -> None:
    """A sample of minted ``handler_body`` Models passes the gate."""
    runner = _runner()
    vocab = vocabdir.load_vocab(_VOCAB)
    domains = _domains()
    patterns = _patterns()
    checked = 0
    for row in _sample(list(vocab.models), "handler"):
        if row["op"] not in domains:
            continue
        v = modelproj.verify_model(
            name=row["op"],
            hnl=row["hnl"],
            ref_spec=patterns[row["op"]],
            domain=domains[row["op"]],
            mvars=row["mvars"],
            eval_hnl=runner.run,
            adversarial=0,
            rtol=handler_synth._VERIFY_RTOL,
            atol=handler_synth._VERIFY_ATOL,
        )
        assert v["status"] == "handl-shipped", (row["id"], v)
        checked += 1
    assert checked >= 1
