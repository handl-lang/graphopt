"""Compatibility shim — the advantage player moved to :mod:`gamekit`.

Plan 0032 lane A extracted the actor-critic ``LearnedPlayer`` into
``packages/gamekit``: :class:`AdvantagePlayer` now lives in
:mod:`gamekit.advantage`.  The name is re-exported unchanged, so
``mlopt_discovery.advantage`` bindings keep resolving; new code
should bind the canonical path.
"""

from __future__ import annotations

from gamekit.advantage import AdvantagePlayer as AdvantagePlayer

__all__ = [
    "AdvantagePlayer",
]
