"""Compatibility shim — the scoring spec moved to :mod:`gamekit`.

Plan 0032 lane A extracted the referee's value function *as data*
into ``packages/gamekit``: ``ScoreGate``/``ScoringSpec``/
:func:`score` now live in :mod:`gamekit.scoring`.  Every name this
module used to define is re-exported unchanged, so
``mlopt_discovery.scoring`` bindings keep resolving; new code should
bind the canonical path.
"""

from __future__ import annotations

from gamekit.scoring import AXES as AXES
from gamekit.scoring import DIRECTIONS as DIRECTIONS
from gamekit.scoring import GateVerdict as GateVerdict
from gamekit.scoring import ScoreGate as ScoreGate
from gamekit.scoring import ScoreVerdict as ScoreVerdict
from gamekit.scoring import ScoringSpec as ScoringSpec
from gamekit.scoring import score as score

__all__ = [
    "AXES",
    "DIRECTIONS",
    "GateVerdict",
    "ScoreGate",
    "ScoreVerdict",
    "ScoringSpec",
    "score",
]
