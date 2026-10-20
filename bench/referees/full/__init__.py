"""``full`` — the whole dial: oracle + pay + sweep + cert replay.

Adds the certificate leg: ``verify_law`` must derive the equality
under the axiom set *and* the positional certificate must replay
standalone (``replayable``) — an e-graph-dependent witness is not a
rule derivation.  The strictest shipped arm: every stage a shipped
law cleared, applied to any claim a player produces.
"""

from __future__ import annotations

from mlopt_discovery.scoring import ScoreGate, ScoringSpec

from bench.referees.registry import RefereeSpec

SPEC = RefereeSpec(
    name="full",
    components=("oracle", "pay", "sweep", "cert"),
    spec=ScoringSpec(gates=(ScoreGate("launch", 1.0),)),
    title="Full gauntlet — every leg plus the cert replay",
    note="derivable AND standalone-replayable, or refused",
    record="the shipped-law bar — every measured leg plus the "
    "standalone derivation",
)
