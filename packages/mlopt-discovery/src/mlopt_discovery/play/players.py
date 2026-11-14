"""Shared player helpers for the play arms.

``_random_policy`` is the uniform-random baseline factory every
domain's arm set binds; ``_scripted_search`` is the search domain's
fixed playbook.
"""

from __future__ import annotations

import random
from typing import Any

from mlopt_core.laws import DEFAULT as DEFAULT_SEARCH_RULES

from .search import _SearchState


def _random_policy(legal: Any) -> Any:
    """Return a uniform player over any domain's legal set."""

    class _R:
        def __init__(self, seed: int) -> None:
            self._r = random.Random(seed)

        def __call__(self, state: Any) -> Any | None:
            acts = legal(state)
            return acts[self._r.randrange(len(acts))] if acts else None

    return _R


def _scripted_search() -> Any:
    """Fire each rule once, matching-head-first — the playbook.

    Rules whose LHS root op occurs in the program go first (the
    only ones that can change the graph); the rest are dead budget.
    """
    heads = {
        r.name: getattr(r.lhs, "op", None) for r in DEFAULT_SEARCH_RULES
    }
    played: set = set()

    def go(state: _SearchState) -> Any | None:
        if state.done:
            return None
        for name in state.actions:
            if name not in played and heads.get(name) in state.ops:
                played.add(name)
                return name
        for name in state.actions:
            if name not in played:
                played.add(name)
                return name
        return None

    return go
