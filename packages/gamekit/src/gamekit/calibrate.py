"""Score calibration — Platt scaling and isotonic regression.

Shared machinery for the policy-measurement drivers
(``bench/drivers/policy/stop_calibrated.py`` is the first consumer):
a predictor's raw score is mapped to a calibrated probability.  The
calibrator is fit on a held-out-quality calibration sample (e.g.
out-of-fold predictions on the *train* split) and applied to new
scores — the fit never sees eval data.  Moved from
``bench/common/calibrate.py`` at the gamekit extraction (plan 0032
lane A) — the calibrated-gate machinery is gamekit's.

Two calibrators, one contract — ``fit(scores, labels) -> self``,
``predict(scores) -> list[float]``:

* :class:`PlattCal` — the parametric answer: logistic regression on
  the score's logit, ``p̂ = sigmoid(a·logit(s) + b)`` (Platt 1999), fit by
  Newton iteration on the BCE.  Monotone non-decreasing iff
  ``a ≥ 0``; a negative ``a`` is *reported*, not prevented — it
  means the score is anti-predictive on the calibration set.
* :class:`IsotonicCal` — the nonparametric answer: pool-adjacent
  violators on the sorted ``(score, label)`` pairs, prediction by
  linear interpolation between block centres, clamped at the ends
  (an out-of-range score gets the nearest block's rate, not an
  extrapolation).  Monotone non-decreasing by construction.

And the metrics both are judged by: :func:`log_loss`,
:func:`brier`, :func:`ece`, and :func:`calibration_slope` — a
well-calibrated predictor lands slope ≈ 1 / intercept ≈ 0 on
held-out data.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence

__all__ = [
    "IsotonicCal",
    "PlattCal",
    "brier",
    "calibration_metrics",
    "calibration_slope",
    "ece",
    "log_loss",
]

_EPS = 1e-7  # logit-domain clamp — sigmoid outputs never reach 0/1


def _sigmoid(z: float) -> float:
    """Numerically tame sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float, eps: float = _EPS) -> float:
    """log(p / (1-p)) with the ends clamped inside (eps, 1-eps)."""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class PlattCal:
    """Logistic calibration on the score's logit — ``sigmoid(a·logit(s) + b)``.

    Two parameters fit by Newton iteration on the calibration set's
    BCE (deterministic — no sampling, no seed).  Fit attributes after
    :meth:`fit`: ``a_`` / ``b_`` (the slope/intercept), ``loss_`` (the
    calibration set's mean BCE), ``n_iter_``.
    """

    def __init__(
        self, *, iters: int = 200, ridge: float = 1e-9
    ) -> None:
        """Set the Newton iteration cap and the Hessian ridge."""
        self.iters = iters
        self.ridge = ridge
        self.a_: float | None = None
        self.b_: float | None = None
        self.loss_: float | None = None
        self.n_iter_: int = 0

    def fit(
        self, scores: Sequence[float], labels: Sequence[float]
    ) -> PlattCal:
        """Fit ``(a, b)`` on ``(score, label)`` pairs — Newton on BCE."""
        if len(scores) != len(labels) or not scores:
            raise ValueError("scores/labels must be nonempty, aligned")
        z = [_logit(s) for s in scores]
        y = [float(v) for v in labels]
        base = min(max(sum(y) / len(y), _EPS), 1.0 - _EPS)
        a, b = 0.0, math.log(base / (1.0 - base))
        n = len(z)
        for it in range(self.iters):
            ga = gb = haa = hab = hbb = 0.0
            for zi, yi in zip(z, y, strict=True):
                q = _sigmoid(a * zi + b)
                w = q * (1.0 - q)
                e = q - yi
                ga += e * zi
                gb += e
                haa += w * zi * zi
                hab += w * zi
                hbb += w
            haa += self.ridge
            hbb += self.ridge
            det = haa * hbb - hab * hab
            if abs(det) < 1e-24:
                break
            da = (hbb * ga - hab * gb) / det
            db = (-hab * ga + haa * gb) / det
            a -= da
            b -= db
            if abs(da) + abs(db) < 1e-12:
                self.n_iter_ = it + 1
                break
            self.n_iter_ = it + 1
        self.a_, self.b_ = a, b
        self.loss_ = (
            sum(
                -(
                    yi * math.log(max(q, _EPS))
                    + (1 - yi) * math.log(max(1.0 - q, _EPS))
                )
                for zi, yi in zip(z, y, strict=True)
                for q in (_sigmoid(a * zi + b),)
            )
            / n
        )
        return self

    def predict(self, scores: Sequence[float]) -> list[float]:
        """Map raw scores to calibrated probabilities."""
        if self.a_ is None or self.b_ is None:
            raise RuntimeError("PlattCal.predict before fit")
        a, b = self.a_, self.b_
        return [_sigmoid(a * _logit(s) + b) for s in scores]


def _pava(
    xs: Sequence[float], ys: Sequence[float]
) -> list[tuple[float, float, int, float]]:
    """Pool-adjacent violators — non-decreasing blocks over sorted *xs*.

    Returns ``(x_lo, x_hi, n, mean)`` blocks: adjacent equal scores
    pool, and any block whose mean exceeds its successor's merges —
    the standard isotonic fit for an increasing map.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    blocks: list[list[float]] = []
    for i in order:
        # equal scores are one point — pool them *before* the
        # violation merge, or the map is discontinuous at a tie
        if blocks and blocks[-1][1] == xs[i]:
            blocks[-1][2] += 1
            blocks[-1][3] += float(ys[i])
        else:
            blocks.append([xs[i], xs[i], 1, float(ys[i])])
        while len(blocks) >= 2:
            b2, b1 = blocks[-2], blocks[-1]
            if b2[3] / b2[2] <= b1[3] / b1[2]:
                break
            blocks[-2:] = [[b2[0], b1[1], b2[2] + b1[2], b2[3] + b1[3]]]
    return [(lo, hi, int(n), s / n) for lo, hi, n, s in blocks]


class IsotonicCal:
    """Nonparametric monotone calibration — PAVA + interpolation.

    The fitted map is a non-decreasing step/interp function over the
    calibration scores: :meth:`predict` linearly interpolates between
    block centres and clamps to the end block rates outside the
    fitted range.  Fit attributes: ``blocks_`` (the PAVA blocks),
    ``knots_`` (the interpolation points).
    """

    def __init__(self) -> None:
        """Start unfitted — :meth:`predict` raises until :meth:`fit`."""
        self.blocks_: list[tuple[float, float, int, float]] = []
        self.knots_: list[tuple[float, float]] = []

    def fit(
        self, scores: Sequence[float], labels: Sequence[float]
    ) -> IsotonicCal:
        """Fit the monotone map on ``(score, label)`` pairs."""
        if len(scores) != len(labels) or not scores:
            raise ValueError("scores/labels must be nonempty, aligned")
        self.blocks_ = _pava(scores, labels)
        self.knots_ = [
            ((lo + hi) / 2.0, m) for lo, hi, _n, m in self.blocks_
        ]
        return self

    def predict(self, scores: Sequence[float]) -> list[float]:
        """Map raw scores through the fitted monotone map."""
        if not self.knots_:
            raise RuntimeError("IsotonicCal.predict before fit")
        kx = [k[0] for k in self.knots_]
        ky = [k[1] for k in self.knots_]
        out: list[float] = []
        for s in scores:
            if s <= kx[0]:
                out.append(ky[0])
            elif s >= kx[-1]:
                out.append(ky[-1])
            else:
                j = bisect_left(kx, s)
                x0, x1 = kx[j - 1], kx[j]
                y0, y1 = ky[j - 1], ky[j]
                t = 0.0 if x1 <= x0 else (s - x0) / (x1 - x0)
                out.append(y0 + t * (y1 - y0))
        return out


def log_loss(
    preds: Sequence[float], labels: Sequence[float], eps: float = _EPS
) -> float:
    """Mean binary cross-entropy of *preds* against *labels*."""
    n = len(preds)
    if n == 0:
        return 0.0
    return (
        -sum(
            y * math.log(min(max(p, eps), 1.0))
            + (1.0 - y) * math.log(min(max(1.0 - p, eps), 1.0))
            for p, y in zip(preds, labels, strict=True)
        )
        / n
    )


def brier(preds: Sequence[float], labels: Sequence[float]) -> float:
    """Mean squared error of *preds* against *labels*."""
    n = len(preds)
    if n == 0:
        return 0.0
    return (
        sum((p - y) ** 2 for p, y in zip(preds, labels, strict=True))
        / n
    )


def ece(
    preds: Sequence[float], labels: Sequence[float], bins: int = 10
) -> float:
    """Compute expected calibration error — |conf - acc| over bins."""
    n = len(preds)
    if n == 0 or bins <= 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [
            i
            for i, p in enumerate(preds)
            if (lo <= p < hi) or (b == bins - 1 and p == 1.0)
        ]
        if not idx:
            continue
        mp = sum(preds[i] for i in idx) / len(idx)
        my = sum(labels[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(mp - my)
    return total


def calibration_slope(
    preds: Sequence[float], labels: Sequence[float]
) -> tuple[float | None, float | None]:
    """Logistic slope/intercept of *labels* on ``logit(preds)``.

    The held-out read on a calibrator: slope ≈ 1 and intercept ≈ 0
    is calibrated; slope < 1 is over-confident spread, intercept off
    zero is a systematic level shift.  Returns ``(None, None)`` when
    the labels are degenerate (all one class — no slope exists).
    """
    if not labels or min(labels) == max(labels):
        return None, None
    cal = PlattCal().fit(preds, labels)
    return cal.a_, cal.b_


def calibration_metrics(
    preds: Sequence[float], labels: Sequence[float], bins: int = 10
) -> dict[str, float | int | None]:
    """Return the calibration report for one (preds, labels) pair."""
    n = len(preds)
    slope, intercept = calibration_slope(preds, labels)
    return {
        "n": n,
        "pos_rate": round(sum(labels) / n, 4) if n else None,
        "mean_pred": round(sum(preds) / n, 4) if n else None,
        "log_loss": round(log_loss(preds, labels), 4),
        "brier": round(brier(preds, labels), 4),
        "ece": round(ece(preds, labels, bins), 4),
        "slope": round(slope, 4) if slope is not None else None,
        "intercept": (
            round(intercept, 4) if intercept is not None else None
        ),
    }
