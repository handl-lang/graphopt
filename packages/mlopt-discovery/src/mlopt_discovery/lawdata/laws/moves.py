"""The move vocabulary — arms, metavariables and action parameters.

The generator-arm inventory and its fixed enumeration order, the
corpus-mutating arms, the construction metavariable pool and literal
leaves, the ``saturate`` budget menu and the ``specialize`` move's
scalar/axis pin banks.  A learned player chooses among these; the
menu is data, not code.

:data:`ARENA_MOVE_ORDER` moved out at the gamekit extraction (plan
0032 lane A): it is the *player* move ranking — game-side content —
so its single home is :mod:`gamekit.gamedata.moves` now and this
module binds the same object (the consumer binds, never forks).
"""

from __future__ import annotations

from gamekit.gamedata.moves import ARENA_MOVE_ORDER

__all__ = [
    "ARENA_MOVE_ORDER",
    "CONST_LEAVES",
    "CORPUS_ARMS",
    "GENERATOR_ORDER",
    "META_SATURATE_BUDGETS",
    "MV_NAMES",
    "SPECIALIZE_AXES",
    "SPECIALIZE_SCALARS",
]

# ---------------------------------------------------------------------------
#  Meta-game — arm inventory and move vocabulary
# ---------------------------------------------------------------------------

#: The generator inventory in its fixed enumeration order — the
#: five ``pipeline.propose`` sources plus ``"build"`` (the
#: construction player; one draw is one play).
GENERATOR_ORDER: tuple = (
    "census-naturality",
    "census-mixed-view",
    "pattern-recognition",
    "shape-aware",
    "algebraic-grammar",
    "build",
)

#: The corpus-mutating arms — a draw is one verified generated
#: workload ingested into the arena corpus (``workload_gen``:
#: undirected resample/mutate; ``gap_gen``: a witness workload
#: synthesized for a ``no-instance`` candidate, which is then
#: re-adjudicated under the grown corpus).  They mutate the
#: evidence scope, so they sit outside the pipeline's enumeration;
#: the enumeration control schedules them last (fixed inventory
#: first, then corpus growth).
CORPUS_ARMS: tuple = ("workload_gen", "gap_gen")

#: Metavariable pool — distinct names bind distinct subterms.
MV_NAMES: tuple = ("U", "V", "W", "X")

#: Literal leaves offered as construction actions.
CONST_LEAVES: tuple = (0, 1)

#: The ``specialize`` move's value bank — the scalar constants a leaf
#: metavariable may be pinned to (identity and sign/direction probes;
#: deliberately small — a player that wants a richer pin writes a
#: richer bank, the bank is data not code).
SPECIALIZE_SCALARS: tuple[float, ...] = (0, 1, -1, 0.5, 2)

#: The ``specialize`` move's axis bank — the attr-metavariable pins
#: worth trying (the last/two-last axes are where view patterns
#: live; 0 keeps the first axis reachable).
SPECIALIZE_AXES: tuple[int, ...] = (-2, -1, 0, 1)

#: The ``saturate`` move's enumerated arms — the per-rule enode
#: budgets offered by ``meta_arena.legal_actions``.  ``None`` is the
#: unbudgeted closure step (the old game's whole move); the small
#: budget arm is the bounded variant.  A player that wants a
#: different budget spells it on the action — the menu is data.
META_SATURATE_BUDGETS: tuple = (None, 256)
