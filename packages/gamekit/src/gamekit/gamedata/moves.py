"""The shared move vocabulary — the arena's fixed action order.

This table moved with the game half of ``lawdata`` at the gamekit
extraction (plan 0032 lane A): the player feature schema
(:mod:`gamekit.gamedata.features`) sizes its action-identity columns
off it, so the constant is gamekit *content*.  The domain side keeps
one binding — ``mlopt_discovery.lawdata.laws.moves`` re-exports the
same object (single copy; the consumer binds, never forks).
"""

from __future__ import annotations

__all__ = [
    "ARENA_MOVE_ORDER",
]

#: The arena move vocabulary in the fixed preference order the
#: scripted players and the feature schema share.  ``auto_cond``
#: first (the enumerator only offers it where a conditional verdict
#: or an unguarded object makes it live — the cheapest rescue of
#: already-measured constructions); ``compose`` next (premise-level
#: structure is the highest-information move); ``relax_guard``
#: widens a guarded region the sweeps re-referee; ``fold`` is the
#: generic mint; ``ingest`` grows the evidence scope mid-game;
#: ``specialize`` narrows a stored object; ``lift`` last — the most
#: speculative construction.  The table is data, not code: a learned
#: player must *beat* this ordering, and tuning it is a data edit.
ARENA_MOVE_ORDER: tuple[str, ...] = (
    "auto_cond",
    "compose",
    "relax_guard",
    "fold",
    "ingest",
    "specialize",
    "lift",
)
