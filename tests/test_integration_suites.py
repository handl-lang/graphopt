"""The integration suites — delivery, referee, vocabulary.

Each suite re-reads committed measurements (no fabricated timings) and
states a falsifiable claim.  These tests pin the joins and the
arithmetic, not the measurements.
"""

from __future__ import annotations

import json

import pytest

from bench.registry import SUITES, get
from bench.suites.integration import delivery, referee, vocabulary

# -- catalog -----------------------------------------------------------


def test_integration_suites_are_registered():
    names = {s.name for s in SUITES}
    assert {"frontier", "delivery", "referee", "vocabulary"} <= names
    for name in ("delivery", "referee", "vocabulary"):
        spec = get(name)
        assert spec.intent == "integration"
        assert spec.harnessed
        assert spec.module.startswith("bench.suites.integration.")


# -- delivery ----------------------------------------------------------


def _pay(tmp_path, folds: dict) -> object:
    path = tmp_path / "pay.json"
    path.write_text(json.dumps({"folds": folds}))
    return path


def test_delivery_pairs_the_same_op_across_the_two_artifacts(tmp_path):
    pay = _pay(
        tmp_path,
        {
            "syn_op": {
                "worst": 2.0,
                "modules": [{"module": "M", "size": "small"}],
            }
        },
    )
    kernel = tmp_path / "prim_kernel.json"
    kernel.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "op": "syn_op",
                        "status": "measured",
                        "kernel_over_binding": 0.4,
                    },
                    {"op": "syn_other", "status": "skipped"},
                ]
            }
        )
    )
    rows = delivery.load_rows(pay, (kernel,))
    by_op = {r.op: r for r in rows}
    assert set(by_op) == {"syn_op"}
    assert by_op["syn_op"].step_wins is False
    assert by_op["syn_op"].kernel_wins is True
    assert by_op["syn_op"].flips is True


def test_delivery_findings_report_the_flip():
    row = delivery.Row(
        op="syn_op",
        step_worst=2.0,
        step_cells=2,
        kernel_ratio=0.4,
        dtype="fp32",
    )
    findings = delivery.build_findings([row])
    verdicts = [f.verdict.value for f in findings]
    assert verdicts == ["negative", "win", "win"]


def test_delivery_without_pairs_is_inconclusive():
    findings = delivery.build_findings(
        [delivery.Row("only_step", 2.0, 1, None, "-")]
    )
    assert findings[0].verdict.value == "inconclusive"


# -- referee -----------------------------------------------------------


def test_referee_overlap_math():
    overlap = referee.Overlap(
        substrate=referee.Bank("tensor", 10, frozenset({"add", "mul"})),
        kernel=referee.Bank("kernel", 5, frozenset({"mul", "kf_silu"})),
    )
    assert overlap.shared == {"mul"}
    assert overlap.kernel_only == {"kf_silu"}
    assert overlap.substrate_only == {"add"}
    assert overlap.jaccard == pytest.approx(1 / 3)
    assert overlap.untranslatable_fraction == pytest.approx(0.5)


def test_referee_without_kernelopt_says_so(monkeypatch):
    monkeypatch.setattr(referee, "load_overlap", lambda: None)
    findings = referee.build_findings(None)
    assert findings[0].verdict.value == "inconclusive"
    assert "kernelopt" in findings[0].headline


# -- vocabulary --------------------------------------------------------


def test_vocabulary_arm_is_the_mint_mode():
    run = vocabulary.Run(
        path="docs/x.json",
        corpus="zoo-prim",
        mint=vocabulary.VOCAB_MINT,
        admitted=3,
        original=3,
        known=0,
    )
    assert run.arm == "vocabulary"
    other = vocabulary.Run("docs/y.json", "zoo", "-", 1, 0, 1)
    assert other.arm == "fixed-bank"
    assert other.original_fraction == 0.0


def test_vocabulary_findings_split_the_arms():
    runs = [
        vocabulary.Run("a", "zoo", "-", 4, 0, 4),
        vocabulary.Run("b", "zoo-prim", vocabulary.VOCAB_MINT, 5, 5, 0),
    ]
    findings = vocabulary.build_findings(runs)
    assert findings[0].verdict.value == "negative"
    assert findings[1].verdict.value == "win"
    assert findings[2].value == pytest.approx(1.0)


def test_vocabulary_load_reads_the_committed_ledgers():
    runs = vocabulary.load_runs()
    assert runs, "expected the committed overnight ledgers to load"
    assert any(r.arm == "vocabulary" for r in runs)
    assert any(r.arm == "fixed-bank" for r in runs)
