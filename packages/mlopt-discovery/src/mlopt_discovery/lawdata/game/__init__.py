"""The game half of ``lawdata`` — destined for ``gamekit``.

Plan 0032 split: these are the tables a *player* is parameterized by —
the feature schemas (feature names, hash-bucket widths, history stats)
and the weight/score tables (priors, rewards, the referee's score
composition, the trained snapshots for arena / meta-arena / open-game /
player-arena).  They are domain-agnostic in *shape* — a linear policy
over any board consumes a schema + a weight table — so at repo
extraction this half travels with the Board/players engine into
``gamekit``.

* :mod:`mlopt_discovery.lawdata.game.features` — the player feature
  schemas.
* :mod:`mlopt_discovery.lawdata.game.weights` — prior, reward, referee
  and learned-player weight tables.

The one back-edge into the law half: ``features`` sizes its
action-identity columns off ``lawdata.laws.moves.ARENA_MOVE_ORDER``
(documented at the import site).
"""

from __future__ import annotations

from .features import (
    ARENA_FEATURES,
    ARENA_HASH_BUCKETS,
    ARENA_HISTORY_STATS,
    META_ARENA_FEATURES,
    META_ARENA_HASH_BUCKETS,
)
from .weights import (
    ARENA_PLAYER_CANDIDATES,
    ARENA_PLAYER_WEIGHTS,
    ARENA_REWARD,
    META_ARENA_PLAYER_WEIGHTS,
    META_ARENA_REWARD,
    OPEN_GAME_PLAYER_WEIGHTS,
    PLAYER_ARENA_PLAYER_WEIGHTS,
    PRIOR_WEIGHTS,
    REFEREE_SCORE,
)

__all__ = [
    "ARENA_FEATURES",
    "ARENA_HASH_BUCKETS",
    "ARENA_HISTORY_STATS",
    "ARENA_PLAYER_CANDIDATES",
    "ARENA_PLAYER_WEIGHTS",
    "ARENA_REWARD",
    "META_ARENA_FEATURES",
    "META_ARENA_HASH_BUCKETS",
    "META_ARENA_PLAYER_WEIGHTS",
    "META_ARENA_REWARD",
    "OPEN_GAME_PLAYER_WEIGHTS",
    "PLAYER_ARENA_PLAYER_WEIGHTS",
    "PRIOR_WEIGHTS",
    "REFEREE_SCORE",
]
