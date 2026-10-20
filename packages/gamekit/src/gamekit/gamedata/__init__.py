"""The game-side content tables — the extracted ``lawdata.game``.

Plan 0032 lane A: these are the tables a *player* is parameterized
by — the feature schemas (feature names, hash-bucket widths,
history stats), the weight/score tables (priors, rewards, the
referee's score composition, the trained snapshots for arena /
meta-arena / open-game / player-arena) and the shared move
vocabulary.  They are domain-agnostic in *shape* — a linear policy
over any board consumes a schema + a weight table — so this half
travelled with the Board/players engine into ``gamekit``.

* :mod:`gamekit.gamedata.features` — the player feature schemas.
* :mod:`gamekit.gamedata.weights` — prior, reward, referee and
  learned-player weight tables.
* :mod:`gamekit.gamedata.moves` — the shared move vocabulary
  (:data:`ARENA_MOVE_ORDER`): gamekit content the feature schema
  sizes its action-identity columns off; the domain side binds the
  same object through ``mlopt_discovery.lawdata.laws.moves``.

The old path ``mlopt_discovery.lawdata.game`` stays importable — a
``sys.modules`` alias in ``mlopt_discovery.lawdata`` points it here.
"""

from __future__ import annotations

from .features import (
    ARENA_FEATURES,
    ARENA_HASH_BUCKETS,
    ARENA_HISTORY_STATS,
    META_ARENA_FEATURES,
    META_ARENA_HASH_BUCKETS,
)
from .moves import ARENA_MOVE_ORDER
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
    "ARENA_MOVE_ORDER",
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
