"""Compatibility shim — the game layer moved to :mod:`gamekit`.

Plan 0032 lane A extracted the generic optimization game into
``packages/gamekit``: :class:`Board`, the episode driver
:func:`run_episode`, the arm comparison :func:`evaluate` and the
training loop :func:`train_policy` now live in
:mod:`gamekit.engine`.  Every name this module used to define is
re-exported unchanged, so ``mlopt_discovery.engine`` bindings keep
resolving; new code should bind the canonical path.
"""

from __future__ import annotations

from gamekit.engine import Board as Board
from gamekit.engine import Report as Report
from gamekit.engine import Trajectory as Trajectory
from gamekit.engine import evaluate as evaluate

#: ``imitate_policy`` is a documented driver that was never in
#: ``__all__`` — re-exported here for the same compat reason and
#: kept unlisted (the boundary note in docs/public_api.md stands).
from gamekit.engine import imitate_policy as imitate_policy
from gamekit.engine import run_episode as run_episode
from gamekit.engine import train_policy as train_policy

__all__ = [
    "Board",
    "Report",
    "Trajectory",
    "evaluate",
    "run_episode",
    "train_policy",
]
