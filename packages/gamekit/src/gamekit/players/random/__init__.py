"""``random`` — the uniform-draw player, the honesty floor.

Plays every board: samples uniformly over the legal set the domain's
enumerator returns (the arenas' ``RandomPlayer`` semantics; uniform
unordered-pair draws on the contraction board).  No schema, no train
step, no artifact — the null hypothesis every learned arm must beat on
paired draws.

Measured record (``docs/results_learned_baseline.md``, the
memoized-verdict protocol): meta -0.566 reward/ep at 11.3 evals/ep —
it pays the full drain cost for what the scripted line banks in two
moves; open 0.134 reward/ep, 0.0096 useful/eval, and the page-ranking
AP is exactly the useful base rate — what a coin flip earns.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from gamekit.players.registry import PlayerSpec

__all__ = ["SPECS", "make"]

#: Uniform play is legal on every board — it is the paired-draw floor.
SPECS: tuple[PlayerSpec, ...] = (
    PlayerSpec(
        name="random",
        kind="chance",
        boards=(
            "meta",
            "joint",
            "torch",
            "search",
            "gen",
            "kernel",
            "open",
            "player",
            "contraction",
        ),
        schema="— (no features; the move is a uniform draw)",
        train="— (nothing to train)",
        artifact="— (no weights)",
        record=(
            "meta -0.566 reward/ep @ 11.3 evals/ep; open 0.134 "
            "reward/ep, 0.0096 useful/eval, AP = base rate "
            "(docs/results_learned_baseline.md)"
        ),
    ),
)


def _rand_cost(
    tensors: Any, sizes: dict[int, int], rng: random.Random
) -> float:
    """Contract a uniformly-random pair per step; return the cost."""
    from mlopt_torch.contraction_policy import merge_tensors, pair_cost

    ts = [frozenset(t) for t in tensors]
    total = 0.0
    while len(ts) > 1:
        cands = [
            (a, b)
            for a in range(len(ts))
            for b in range(a + 1, len(ts))
        ]
        a, b = cands[rng.randrange(len(cands))]
        total += pair_cost(ts[a], ts[b], sizes)
        ts = merge_tensors(ts, a, b)
    return total


def make(
    domain: str,
    *,
    spec: PlayerSpec,
    seed: int = 0,
    ctx: dict | None = None,
) -> Callable[[int], Any]:
    """Return the uniform arm factory for *domain* — ``i -> player``."""
    del spec, ctx
    if domain == "contraction":
        return lambda i: (
            lambda tensors, sizes: _rand_cost(
                tensors, sizes, random.Random(seed + 10_000 + i)
            )
        )
    from mlopt_discovery.play import DOMAINS, _random_policy

    cls = _random_policy(DOMAINS[domain].legal)
    return lambda i: cls(seed + 10_000 + i)
