"""``sweep`` — the oracle and pay legs plus the adversarial sweep.

Adds the guarded-region enumeration over the *widened* leaf bank
(the retro's domain: ranks <= 3, extents {1, 2, 3}) — the same
``_binding_envs`` machinery the oracle window runs, fed a bank that
reaches the unequal-rank mid-insert corner.  This is the leg that
caught the grid-eq ``mul_unsq_nat`` guard's 212 bad accepts — the
strictness ablation the shipped ``flat-map-unsq`` guard was
synthesized to answer.
"""

from __future__ import annotations

from gamekit.scoring import ScoreGate, ScoringSpec

from bench.referees.registry import RefereeSpec

SPEC = RefereeSpec(
    name="sweep",
    components=("oracle", "pay", "sweep"),
    spec=ScoringSpec(gates=(ScoreGate("launch", 1.0),)),
    title="Adversarial sweep — the widened binding domain",
    note="guarded region over the deep bank; counterexamples refuse",
    record="the leg that measured the grid-eq guard's 212 bad "
    "accepts (retros/mul-unsq-nat.md)",
)
