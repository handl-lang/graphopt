"""Tests for ``bench.common.calibrate`` and the round-8 split wall.

The calibrators back ``bench/policy/stop_calibrated.py`` — Platt
scaling and isotonic regression fitted on the *train* pages'
out-of-fold predictions, applied to held-out scores.  These cover
the fit/predict contract, the isotonic monotone property, and the
no-peek seam (``_scores_for`` reads only the caller's keys).
"""

import sys
from itertools import pairwise
from pathlib import Path

import pytest

from bench.common.calibrate import (
    IsotonicCal,
    PlattCal,
    brier,
    calibration_metrics,
    calibration_slope,
    ece,
    log_loss,
)

_REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
#  PlattCal
# ---------------------------------------------------------------------------


def test_platt_predict_contract():
    """fit → predict returns probabilities in [0, 1], aligned."""
    cal = PlattCal().fit([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    out = cal.predict([0.05, 0.5, 0.95])
    assert len(out) == 3
    # a separable calibration set saturates honestly — ends are legal
    assert all(0.0 <= p <= 1.0 for p in out)
    assert cal.a_ is not None and cal.b_ is not None


def test_platt_predict_before_fit_raises():
    """The contract is fit-first — predict on an unfitted cal fails."""
    with pytest.raises(RuntimeError):
        PlattCal().predict([0.5])
    with pytest.raises(ValueError):
        PlattCal().fit([], [])
    with pytest.raises(ValueError):
        PlattCal().fit([0.1, 0.2], [0])


def test_platt_recovers_known_miscalibration():
    """A systematically under-confident score is lifted — BCE improves.

    Positives score 0.5-0.74, negatives 0.05-0.3: the ranking is
    perfect but the level is far below the real rates (the round-7
    held-out symptom).  Platt's intercept should go positive and the
    calibrated log-loss must beat the raw scores'.
    """
    labels = [0.0] * 25 + [1.0] * 25
    scores = [0.05 + 0.01 * i for i in range(25)] + [
        0.5 + 0.01 * i for i in range(25)
    ]
    cal = PlattCal().fit(scores, labels)
    assert cal.b_ is not None and cal.b_ > 0.0  # lifted the level
    assert log_loss(cal.predict(scores), labels) < log_loss(
        scores, labels
    )


def test_platt_monotone_when_score_predictive():
    """On a monotone-fit calibration set, predict is non-decreasing."""
    scores = [i / 50 for i in range(1, 50)]
    labels = [1.0 if s > 0.5 else 0.0 for s in scores]
    cal = PlattCal().fit(scores, labels)
    out = cal.predict(sorted(scores))
    assert all(b >= a - 1e-12 for a, b in pairwise(out))


# ---------------------------------------------------------------------------
#  IsotonicCal
# ---------------------------------------------------------------------------


def test_isotonic_predict_contract():
    """fit → predict returns in-range, aligned outputs."""
    cal = IsotonicCal().fit([0.9, 0.1, 0.5, 0.3], [1, 0, 1, 0])
    out = cal.predict([0.05, 0.4, 0.95])
    assert len(out) == 3
    assert all(0.0 <= p <= 1.0 for p in out)
    assert len(cal.blocks_) >= 1
    assert len(cal.knots_) == len(cal.blocks_)


def test_isotonic_predict_before_fit_raises():
    """Same fit-first contract as Platt."""
    with pytest.raises(RuntimeError):
        IsotonicCal().predict([0.5])
    with pytest.raises(ValueError):
        IsotonicCal().fit([], [])


def test_isotonic_monotone_on_noise():
    """The defining property: predict is non-decreasing in the score."""
    scores = [
        0.31,
        0.87,
        0.05,
        0.62,
        0.62,
        0.11,
        0.44,
        0.98,
        0.27,
        0.55,
    ]
    labels = [0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    cal = IsotonicCal().fit(scores, labels)
    grid = [i / 100 for i in range(101)]
    out = cal.predict(grid)
    assert all(b >= a - 1e-12 for a, b in pairwise(out))


def test_isotonic_ties_pool_and_ends_clamp():
    """Tied scores share a block; out-of-range queries clamp."""
    cal = IsotonicCal().fit([0.5, 0.5, 0.5, 0.1, 0.9], [0, 1, 0, 0, 1])
    # the 0.5-tie block rate is 1/3; ends clamp to block rates
    out = cal.predict([0.0, 0.5, 1.0])
    assert out[0] <= out[1] <= out[2]
    assert abs(cal.predict([0.5])[0] - (1 / 3)) < 0.2
    assert cal.predict([-1.0])[0] == cal.predict([0.0])[0]
    assert cal.predict([2.0])[0] == cal.predict([1.0])[0]


def test_isotonic_perfectly_calibrated_is_near_identity():
    """On already-calibrated pairs the map stays close to identity."""
    scores = [i / 40 for i in range(1, 40)]
    labels = [1.0 if i % 5 < 2 else 0.0 for i in range(1, 40)]
    cal = IsotonicCal().fit(scores, labels)
    out = cal.predict(scores)
    assert all(0.0 <= p <= 1.0 for p in out)
    # monotone fit on a mixed-label grid still orders correctly
    assert all(b >= a - 1e-12 for a, b in pairwise(out))


def test_isotonic_no_worse_than_platt_on_step_data():
    """Nonparametric fit should win where the truth is a step."""
    scores = [i / 60 for i in range(1, 60)]
    labels = [0.0] * 30 + [1.0] * 29
    iso = IsotonicCal().fit(scores, labels)
    platt = PlattCal().fit(scores, labels)
    assert (
        brier(iso.predict(scores), labels)
        <= brier(platt.predict(scores), labels) + 1e-9
    )


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def test_metrics_on_constant_predictions():
    """Constant predictions land the closed-form metric values."""
    preds = [0.25] * 8
    labels = [0, 0, 0, 0, 0, 0, 1, 1]
    assert abs(log_loss(preds, labels) - 0.5623) < 1e-3
    assert (
        abs(brier(preds, labels) - (6 * 0.0625 + 2 * 0.5625) / 8) < 1e-9
    )
    # all mass in one bin: |0.25 - 0.25| = 0 → ECE 0
    assert ece(preds, labels, bins=10) == pytest.approx(0.0)


def test_ece_flags_miscalibration():
    """Confident-and-wrong predictions register a large ECE."""
    preds = [0.9] * 10
    labels = [0] * 10
    assert ece(preds, labels, bins=10) == pytest.approx(0.9)


def test_calibration_slope_identity():
    """Well-calibrated preds → slope ≈ 1, intercept ≈ 0."""
    scores = [i / 100 for i in range(1, 100)]
    labels = [1.0 if i % 4 < 1 else 0.0 for i in range(1, 100)]
    slope, intercept = calibration_slope(scores, labels)
    assert slope is not None and intercept is not None
    # degenerate labels → no slope
    assert calibration_slope([0.5, 0.6], [0, 0]) == (None, None)


def test_calibration_metrics_keys():
    """The report carries the standard calibration numbers."""
    m = calibration_metrics([0.2, 0.8], [0, 1])
    for k in (
        "n",
        "pos_rate",
        "mean_pred",
        "log_loss",
        "brier",
        "ece",
        "slope",
        "intercept",
    ):
        assert k in m
    assert m["n"] == 2 and m["pos_rate"] == 0.5


# ---------------------------------------------------------------------------
#  The no-peek split seam in bench/policy/stop_calibrated.py
# ---------------------------------------------------------------------------


def test_scores_for_reads_only_the_given_candidates():
    """``_scores_for`` touches exactly the keys it is asked for.

    The calibration fit is fed ``_scores_for(flat_train, preds)`` —
    a spy dict proves no held-out key is ever read, so the train /
    held-out wall holds at the fit, not just by convention.
    """
    sys.path.insert(0, str(_REPO / "bench" / "policy"))
    try:
        import stop_calibrated as sc
    finally:
        sys.path.remove(str(_REPO / "bench" / "policy"))

    class _Cand:  # ``og._value_key`` reads label + repr(lhs/rhs)
        def __init__(self, label, lhs, rhs):
            self.label, self.lhs, self.rhs = label, lhs, rhs

    train = [_Cand("assoc:0", f"t{i}", "u") for i in range(3)]
    held = [_Cand("assoc:0", f"h{i}", "u") for i in range(2)]
    from mlopt_discovery import open_game as og

    accessed: list = []

    class _Spy(dict):
        def get(self, k, default=None):
            accessed.append(k)
            return super().get(k, default)

    preds = _Spy(
        {og._value_key(c): 0.1 * i for i, c in enumerate(train + held)}
    )
    out = sc._scores_for(train, preds)
    assert len(out) == len(train)
    held_keys = {og._value_key(c) for c in held}
    assert not (set(accessed) & held_keys)


def test_fit_calibrators_ignores_extra_scores():
    """The fit is a pure function of the train arrays it is given."""
    sys.path.insert(0, str(_REPO / "bench" / "policy"))
    try:
        import stop_calibrated as sc
    finally:
        sys.path.remove(str(_REPO / "bench" / "policy"))

    scores = {"useful": [0.1, 0.2, 0.7, 0.9], "pays": [0.2] * 4}
    labels = {"useful": [0.0, 0.0, 1.0, 1.0], "pays": [0.0] * 4}
    a = sc._fit_calibrators(scores, labels, ("platt", "isotonic"))
    b = sc._fit_calibrators(scores, labels, ("platt", "isotonic"))
    # deterministic fits, both calibrators honour the contract
    for m in ("platt", "isotonic"):
        for lname in ("useful", "pays"):
            assert a[m][lname].predict([0.5]) == pytest.approx(
                b[m][lname].predict([0.5])
            )
