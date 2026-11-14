"""The ``fragment`` domain — the fragment-game board.

``fragment_game`` states played under the same player-spec seam;
the scripted arm is the domain's fixed playbook.
"""

from __future__ import annotations

import random
from typing import Any

from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    StateBroadcast,
    register_legal,
)
from mlopt_core.ir import Const, Op, TensorType, Var

from .. import fragment_game as fg

# ---------------------------------------------------------------------------


#: The fragment case pool — named so a domain spec's
#: ``corpus.kinds`` can select term shapes as data.  Built lazily
#: for the same shared-leaf reason as ``_search_pool``.
_FRAGMENT_POOL: tuple | None = None


def _fragment_pool() -> tuple:
    """Return the named fragment-case pool (built once)."""
    global _FRAGMENT_POOL
    if _FRAGMENT_POOL is None:
        x = Var("x", TensorType((4, 4)))
        y = Var("y", TensorType((4, 4)))
        _FRAGMENT_POOL = (
            ("addid", Op.make("add", x, Const(0))),
            ("mulid", Op.make("mul", x, Const(1))),
            ("negneg", Op.make("neg", Op.make("neg", x))),
            ("mulsig", Op.make("mul", x, Op.make("sigmoid", x))),
            (
                "addmul",
                Op.make(
                    "add",
                    Op.make("mul", x, y),
                    Op.make("mul", y, x),
                ),
            ),
            (
                "divabs",
                Op.make(
                    "div",
                    x,
                    Op.make("add", Op.make("abs", x), Const(1)),
                ),
            ),
        )
    return _FRAGMENT_POOL


def _fragment_cases(seed: int, n: int, *, kinds: Any = None) -> list:
    """Small terms that exercise law/handler/saturate fragment moves.

    *kinds* (domain-spec seam) selects pool names; ``None`` keeps
    the full pool.
    """
    rng = random.Random(seed)
    pool = list(_fragment_pool())
    if kinds is not None:
        known = {name for name, _ in pool}
        unknown = sorted(set(kinds) - known)
        if unknown:
            raise ValueError(
                f"corpus.kinds: unknown fragment pool {unknown!r} "
                f"(known: {sorted(known)!r})"
            )
        wanted = set(kinds)
        pool = [p for p in pool if p[0] in wanted]
    out = []
    for i in range(n):
        name, term = pool[rng.randrange(len(pool))]
        out.append((f"frag{i}:{name}", term))
    return out


def _fragment_board(
    case: Any,
    *,
    rules: Any = None,
    handlers: Any = None,
    max_moves: int = 256,
) -> fg.FragmentArena:
    """Board a fragment case over the canonical vocab directory.

    *rules*/*handlers*/*max_moves* are the domain-spec seams — a
    spec seats a rewrite-name subset, a handler-tag subset and the
    move budget as data.
    """
    return fg.FragmentArena(
        case[1],
        vocab_dir=None,
        rules=rules,
        handlers=handlers,
        max_moves=max_moves,
    )


def _frag_state(state: fg.FragmentState) -> dict[str, float]:
    """Broadcast the fragment board's progress features."""
    base = max(state.baseline_cost, 1e-9)
    best = (
        state.best_cost if state.best_cost == state.best_cost else base
    )
    return {
        "st:best": max(0.0, min(1.0, 1.0 - best / base)),
        "st:steps": state.steps / 8.0,
        "st:moves": min(len(state.moves) / 64.0, 1.0),
    }


def _frag_act(action: Any, _row: dict[str, float]) -> dict[str, float]:
    """One-hot the fragment move kind — the only action feature."""
    kind = getattr(action, "kind", "?")
    return {f"a:k:{kind}": 1.0}


FRAGMENT_FEATURES: tuple = (
    "bias",
    "st:best",
    "st:steps",
    "st:moves",
    "a:k:law",
    "a:k:prim",
    "a:k:schema",
    "a:k:handler",
    "a:k:saturate",
    "a:k:extract",
)

FRAGMENT_SPEC = DomainSpec.build(
    features=FRAGMENT_FEATURES,
    legal=fg.fragment_legal,
    parts=(
        Bias(),
        StateBroadcast(_frag_state),
        ActionFeats(_frag_act),
    ),
)
register_legal("fragment", fg.fragment_legal)


def _fragment_scripted() -> Any:
    """Saturate the fragment closure, then extract — the playbook."""
    played: set[str] = set()

    def go(state: fg.FragmentState) -> Any | None:
        if state.done:
            return None
        for m in state.moves:
            if m.kind == "saturate" and "sat" not in played:
                played.add("sat")
                return m
        for m in state.moves:
            if m.kind == "extract":
                return m
        return state.moves[0] if state.moves else None

    return go
