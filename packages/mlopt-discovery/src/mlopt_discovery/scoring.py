"""The scoring spec — the referee's value function, as data.

Plan 0032 phase 1 (the API freeze): this module declares the *shape*
of the contract the phase-2 referees consume — it is not itself a
referee.  The seam it pins (see ``docs/public_api.md``):

* the **value spec** is the game layer's — a :class:`ScoringSpec` is
  pure data: which axes carry bars (:class:`ScoreGate` — the
  ``{axis, threshold}`` pair) and how the measured axes blend into
  the scalar ``value``;
* the **measurement** is the gauntlet's
  (:mod:`mlopt_discovery.evidence` and friends): a plain
  ``{axis: value}`` mapping emitted on the measurement side — plain
  data, so the substrate never imports this module;
* the **referee** is the composition — :func:`score` is where spec
  and measurement meet; phase-2 lane B lands the referee registry
  (``naive``/``oracle``/``pay``/``sweep``/``full``) on top.

The shipped axis vocabulary is :data:`AXES` — ``speed`` (delivered
latency), ``memory`` (resident + peak bytes), ``launch``
(kernel-launch/dispatch count) and ``params`` (parameter storage).
``AXES`` is the vocabulary the shipped referees draw on, not a closed
enum: a gate may name any axis a measurement can emit, and the
module imports nothing beyond the stdlib — it is a
gamekit-destined kernel module (see the "game kernel is
substrate-free" import-linter contract).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AXES",
    "DIRECTIONS",
    "GateVerdict",
    "ScoreGate",
    "ScoreVerdict",
    "ScoringSpec",
    "score",
]

#: The shipped axis vocabulary the referees draw on.  All four are
#: costs — lower is better — so a ``at_most`` bar is the common case.
AXES: tuple[str, ...] = ("speed", "memory", "launch", "params")

#: Gate directions: ``at_most`` bars a ceiling (latency, bytes);
#: ``at_least`` bars a floor (a speedup ratio, a coverage rate).
DIRECTIONS: tuple[str, ...] = ("at_most", "at_least")


def _finite(label: str, v: Any, *, positive: bool = False) -> float:
    """Validate and return a finite float; ``positive`` adds ``> 0``."""
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{label} must be a finite number — got {v!r}")
    if positive and v <= 0:
        raise ValueError(f"{label} must be positive — got {v!r}")
    return float(v)


@dataclass(frozen=True)
class ScoreGate:
    """One scored axis's bar — the ``{axis, threshold}`` contract row.

    ``direction="at_most"`` (the default) reads *threshold* as a
    ceiling — the measurement must not exceed it; ``"at_least"``
    reads it as a floor.  *axis* names a measured quantity — the
    shipped vocabulary is :data:`AXES`, but any axis a measurement
    emits is legal.
    """

    axis: str
    threshold: float
    direction: str = "at_most"

    def __post_init__(self) -> None:
        """Validate the gate — a malformed bar fails at declaration."""
        if not isinstance(self.axis, str) or not self.axis:
            raise ValueError(
                f"gate axis must be a non-empty str — got {self.axis!r}"
            )
        _finite(f"threshold of gate {self.axis!r}", self.threshold)
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"unknown gate direction {self.direction!r} — "
                f"expected one of {DIRECTIONS}"
            )

    def passed(self, value: float) -> bool:
        """Return whether *value* clears the bar."""
        if self.direction == "at_least":
            return value >= self.threshold
        return value <= self.threshold

    def margin(self, value: float) -> float:
        """Return signed headroom — ``>= 0`` passes."""
        if self.direction == "at_least":
            return value - self.threshold
        return self.threshold - value

    def ratio(self, value: float) -> float:
        """Return *value* normalised to the bar — ``<= 1`` is inside.

        ``at_most`` divides by the threshold; ``at_least`` inverts,
        so the ratio *drops* as the measurement clears the floor —
        one monotone "fraction of the bar" scale for the blend.  A
        zero/negative denominator reports ``inf`` rather than lying.
        """
        if self.direction == "at_least":
            if value <= 0:
                return math.inf
            return self.threshold / value
        if self.threshold == 0:
            return 0.0 if value <= 0 else math.inf
        return value / self.threshold


@dataclass(frozen=True)
class ScoringSpec:
    """The value function — the gates plus the blend weighting.

    ``gates`` are the pass bars a measurement must clear.
    ``weights`` blends measured axes into the scalar ``value`` —
    empty means "every gated axis at weight 1.0".  Each weighted axis
    normalises by its ``references`` entry, falling back to its
    gate's ``threshold`` — so ``value`` reads in fractions of the
    bar.  A weight on an axis that is neither gated nor referenced
    has nothing to normalise against and is refused at declaration.
    """

    gates: tuple[ScoreGate, ...]
    weights: Mapping[str, float] = field(default_factory=dict)
    references: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise *gates* to a tuple; validate weights/references."""
        object.__setattr__(self, "gates", tuple(self.gates))
        gated = {g.axis for g in self.gates}
        for axis, w in self.weights.items():
            w = _finite(f"weight of axis {axis!r}", w)
            if w < 0:
                raise ValueError(
                    f"weight of axis {axis!r} must be non-negative — "
                    f"got {w!r}"
                )
            if axis not in gated and axis not in self.references:
                raise ValueError(
                    f"weighted axis {axis!r} is neither gated nor "
                    "referenced — it has nothing to normalise against"
                )
        for axis, r in self.references.items():
            _finite(f"reference of axis {axis!r}", r, positive=True)

    def blend_weights(self) -> dict[str, float]:
        """Return the effective ``{axis: weight}`` blend."""
        if self.weights:
            return dict(self.weights)
        return {g.axis: 1.0 for g in self.gates}

    def reference(self, axis: str) -> float | None:
        """Return *axis*'s normaliser — explicit, else its gate's bar."""
        if axis in self.references:
            return float(self.references[axis])
        for g in self.gates:
            if g.axis == axis:
                return float(g.threshold)
        return None


@dataclass(frozen=True)
class GateVerdict:
    """One gate's outcome under a measurement.

    ``value``/``margin`` are ``nan`` when the axis was unmeasured —
    an absent measurement is a failed gate, never a skipped one.
    """

    gate: ScoreGate
    value: float
    passed: bool
    margin: float


@dataclass(frozen=True)
class ScoreVerdict:
    """The verdict — the spec applied to a measurement.

    ``passed`` is the conjunction of the gate verdicts; ``value`` is
    the weighted blend of normalised ratios (``nan`` when a weighted
    axis went unmeasured — the verdict cannot be priced); ``missing``
    names every needed axis the measurement did not carry.
    """

    passed: bool
    value: float
    gates: tuple[GateVerdict, ...]
    missing: tuple[str, ...] = ()


def score(
    spec: ScoringSpec, measured: Mapping[str, float]
) -> ScoreVerdict:
    """Score a ``{axis: value}`` measurement against *spec*.

    The measurement is plain data — the gauntlet emits it, the
    referee consumes this function's verdict.  A missing axis fails
    its gates honestly and is listed in ``missing``; an unmeasured
    *weighted* axis leaves ``value`` at ``nan`` (the row cannot be
    priced) without masking the gate outcomes that did resolve.
    """
    missing: list[str] = []
    verdicts: list[GateVerdict] = []
    for g in spec.gates:
        v = measured.get(g.axis)
        if v is None or not math.isfinite(v):
            missing.append(g.axis)
            verdicts.append(GateVerdict(g, math.nan, False, math.nan))
        else:
            fv = float(v)
            verdicts.append(
                GateVerdict(g, fv, g.passed(fv), g.margin(fv))
            )
    gates_by_axis = {g.axis: g for g in spec.gates}
    total = 0.0
    priced = True
    for axis, w in spec.blend_weights().items():
        v = measured.get(axis)
        ref = spec.reference(axis)
        if v is None or not math.isfinite(v) or ref is None or ref <= 0:
            priced = False
            if axis not in missing:
                missing.append(axis)
            continue
        gate = gates_by_axis.get(axis)
        direction = gate.direction if gate is not None else "at_most"
        if direction == "at_least":
            total += w * (math.inf if v <= 0 else ref / v)
        else:
            total += w * (v / ref)
    return ScoreVerdict(
        passed=all(v.passed for v in verdicts),
        value=total if priced else math.nan,
        gates=tuple(verdicts),
        missing=tuple(sorted(set(missing))),
    )
