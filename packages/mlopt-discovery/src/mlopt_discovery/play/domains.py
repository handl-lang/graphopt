"""The shipped domain registry — populated once at package import.

Each entry is data (:func:`~.registry.spec_domain` over a cases
generator, a ``case -> Board`` factory and a player spec); the arms
live in the sibling modules.  ``bg.register_players()`` seats the
bilinear board's player arms explicitly here — the composition
root — not as an ``import bilinear_game`` side-effect.
"""

from __future__ import annotations

from .. import bilinear_game as bg
from .. import meta_arena as ma
from .. import meta_player as mp
from .. import open_game as og
from .. import player_arena as pa
from .fragment import (
    FRAGMENT_SPEC,
    _fragment_board,
    _fragment_cases,
    _fragment_scripted,
)
from .gen import _gen_board, _gen_cases
from .meta_joint import _joint_board, _joint_cases, _meta_board
from .players import _scripted_search
from .registry import DOMAINS, spec_domain
from .search import SEARCH_SPEC, _search_board, _search_cases
from .torch_arm import _torch_board, _torch_cases, _torch_deliver

DOMAINS.update(
    {
        "meta": spec_domain(
            mp.gen_cases,
            _meta_board,
            mp.META_SPEC,
            scripted=ma.ScriptedPlayer,
        ),
        "joint": spec_domain(
            _joint_cases,
            _joint_board,
            mp.META_SPEC,
            scripted=ma.ScriptedPlayer,
        ),
        "torch": spec_domain(
            _torch_cases,
            _torch_board,
            mp.META_SPEC,
            scripted=ma.ScriptedPlayer,
            deliver=_torch_deliver,
        ),
        "search": spec_domain(
            _search_cases,
            _search_board,
            SEARCH_SPEC,
            scripted=_scripted_search,
        ),
        "fragment": spec_domain(
            _fragment_cases,
            _fragment_board,
            FRAGMENT_SPEC,
            scripted=_fragment_scripted,
        ),
        "gen": spec_domain(
            _gen_cases,
            _gen_board,
            mp.META_SPEC,
            scripted=ma.ScriptedPlayer,
            deliver=_torch_deliver,
        ),
        "open": spec_domain(og.gen_cases, og.OpenArena, og.OPEN_SPEC),
        "player": spec_domain(
            pa.gen_cases,
            pa.PlayerArena,
            pa.PLAYER_SPEC,
            scripted=pa.GreedyEditor,
        ),
        "bilinear": spec_domain(
            bg.gen_cases,
            bg.BilinearBoard,
            bg.BILINEAR_SPEC,
            scripted=bg.strassen_line,
        ),
    }
)

# The bilinear board's player seats — explicit at the composition
# root, not an ``import bilinear_game`` side-effect.
bg.register_players()
