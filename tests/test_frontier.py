"""The frontier's bound — Amdahl on dispatch, and its capture ratio.

These tests pin the arithmetic, not the measurements: the ceiling is
``1/(1-F)``, capture is ``(speedup-1)/(ceiling-1)``, and a ceiling with
no material room reports ``None`` instead of a noise ratio.
"""

from __future__ import annotations

import pytest

from bench.suites.integration.frontier import (
    CAPTURE_MIN_ROOM,
    Row,
    _regime_stats,
    load_rows,
)


def _row(
    device_share: float,
    *,
    eager_us: float = 1000.0,
    delivered_us: float = 1000.0,
    verified: bool = True,
) -> Row:
    return Row(
        case="zoo:Probe",
        eager_us=eager_us,
        delivered_us=delivered_us,
        device_share=device_share,
        kernels=1,
        verified=verified,
    )


def test_ceiling_is_amdahl_on_dispatch():
    # F = 0.5 -> at most 2x; F = 0.9 -> at most 10x.
    assert _row(0.5).ceiling == pytest.approx(2.0)
    assert _row(0.1).ceiling == pytest.approx(10.0)
    assert _row(1.0).ceiling == pytest.approx(1.0)


def test_capture_is_the_fraction_of_headroom_taken():
    # ceiling 2x, delivered 1.5x -> (1.5-1)/(2-1) = 0.5
    r = _row(0.5, eager_us=1500.0, delivered_us=1000.0)
    assert r.ceiling == pytest.approx(2.0)
    assert r.captured == pytest.approx(0.5)


def test_capture_is_negative_when_delivery_loses():
    r = _row(0.5, eager_us=1000.0, delivered_us=1200.0)
    assert r.captured is not None and r.captured < 0.0


def test_capture_is_none_below_the_noise_floor():
    # room = 0.05 < CAPTURE_MIN_ROOM -> undefined, not a ratio.
    r = _row(1.0 - 0.05, eager_us=1000.0, delivered_us=900.0)
    assert r.ceiling - 1.0 < CAPTURE_MIN_ROOM
    assert r.captured is None


def test_headroom_is_the_removable_wall_time():
    r = _row(0.8, eager_us=1000.0)
    assert r.headroom_us == pytest.approx(200.0)


def test_regimes_split_on_measured_dispatch_share():
    assert _row(0.98).regime == "compute-bound"
    assert _row(0.90).regime == "mixed"
    assert _row(0.75).regime == "launch-leaning"
    assert _row(0.40).regime == "launch-bound"


def test_regime_stats_count_wins_and_skip_undefined_capture():
    rows = [
        _row(0.40, eager_us=1500.0, delivered_us=1000.0),  # 1.5x, wins
        _row(0.40, eager_us=1000.0, delivered_us=1000.0),  # parity
        _row(0.98, eager_us=1000.0, delivered_us=1000.0),  # no room
    ]
    stats = {s["regime"]: s for s in _regime_stats(rows)}
    assert stats["launch-bound"]["n"] == 2
    assert stats["launch-bound"]["wins"] == 1
    assert stats["launch-bound"]["median_ceiling"] > 1.5
    assert stats["compute-bound"]["median_captured"] is None


def test_load_rows_drops_cases_the_profiler_never_saw(tmp_path):
    sweep = tmp_path / "sweep.json"
    kcount = tmp_path / "kcount.json"
    sweep.write_text(
        """[
          {"case": "zoo:A", "verified": true,
           "runs": [{"eager_us": 100.0, "gen_us": 50.0}]},
          {"case": "zoo:B", "verified": true,
           "runs": [{"eager_us": 100.0, "gen_us": 90.0}]}
        ]"""
    )
    kcount.write_text(
        """[{"case": "zoo:A", "kernels": 3, "device_share": 0.5}]"""
    )
    rows = load_rows(sweep, kcount)
    assert [r.case for r in rows] == ["zoo:A"]
    assert rows[0].speedup == pytest.approx(2.0)
    assert rows[0].verified is True
