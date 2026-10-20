"""Native contraction-ordering — the ``contraction_policy`` API in Rust.

Same signatures and semantics as
:mod:`mlopt_torch.contraction_policy`'s pure-combinatorial core:
a network is ``(tensors, sizes)``, a pair contraction costs
``∏ sizes`` over the union of index sets, and the merge appends the
symmetric difference.  ``greedy`` keeps ``random.Random`` on the
Python side — its draws are pre-computed here and passed through as
``picks``, so seeded streams stay identical.
"""

from __future__ import annotations

import random
from typing import Any

from mlopt_native._native import (
    contract_cost_of_order,
    contract_greedy,
    contract_merge,
    contract_pair_cost,
)

__all__ = [
    "cost_of_order",
    "greedy",
    "greedy_order",
    "merge_tensors",
    "pair_cost",
]


def pair_cost(
    x: frozenset[int], y: frozenset[int], sizes: dict[int, int]
) -> float:
    """Scalar multiplies to contract two tensors over index sets."""
    return contract_pair_cost(x, y, sizes)


def merge_tensors(
    ts: list[frozenset[int]], a: int, b: int
) -> list[frozenset[int]]:
    """Replace tensors ``a``/``b`` by their contraction (xor of sets)."""
    return [frozenset(t) for t in contract_merge(ts, a, b)]


def greedy_order(
    tensors: Any,
    sizes: dict[int, int],
    *,
    rng: random.Random | None = None,
    top_k: int = 1,
) -> tuple[list[tuple[int, int]], float]:
    """Return the cheapest-pair order and its cost — ``greedy`` + the order."""
    ts = [frozenset(t) for t in tensors]
    if rng is None:
        return contract_greedy(ts, sizes)
    picks = [
        rng.randrange(min(top_k, m * (m - 1) // 2))
        for m in range(len(ts), 1, -1)
    ]
    return contract_greedy(ts, sizes, picks)


def greedy(
    tensors: Any,
    sizes: dict[int, int],
    *,
    rng: random.Random | None = None,
    top_k: int = 1,
) -> float:
    """Contract the cheapest pair; ``rng`` randomises among the top-k."""
    return greedy_order(tensors, sizes, rng=rng, top_k=top_k)[1]


def cost_of_order(
    tensors: Any, sizes: dict[int, int], order: list[tuple[int, int]]
) -> float:
    """Replay an index-pair order and sum the pairwise cost."""
    return contract_cost_of_order(
        [frozenset(t) for t in tensors], sizes, order
    )
