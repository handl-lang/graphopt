"""``pay`` — the numeric oracle plus the typed-pay leg.

Adds the firing probe: the claim must fire on the probe cases (its
own accepted instances included), every changed extraction must
verify through the sink, and at least one case must strictly lower
cost under the extraction model.  Carries the shipped default
:class:`ScoringSpec` — ``launch <= 1.0x`` — so "must not regress the
launch axis" is data, and the blend prices the measured ratios into
the cell's ``value``.
"""

from __future__ import annotations

from mlopt_discovery.scoring import ScoreGate, ScoringSpec

from bench.referees.registry import RefereeSpec

SPEC = RefereeSpec(
    name="pay",
    components=("oracle", "pay"),
    spec=ScoringSpec(gates=(ScoreGate("launch", 1.0),)),
    title="Typed-pay — fires, verifies, pays (launch-gated)",
    note="changed extractions must verify; mean launch ratio <= 1.0x",
    record="the typed-pay gate — the paying claim is priced per axis",
)
