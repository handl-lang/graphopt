"""``playbook`` / ``greedy`` — the scripted heuristics, hand-written floors.

Two arms, both unlearned:

* ``playbook`` is the fixed line each board's human-known-good order
  spells: ``saturate → extract`` on the meta-arena family
  (:class:`meta_arena.ScriptedPlayer`), matching-head-first rule
  fires on ``search``, the candidate drain on ``open`` (evaluate
  ``state.pending`` best-``dag_cost(lhs) - dag_cost(rhs)``-first, then
  ``stop`` — the heuristic the baseline builds by hand because the
  board ships none), and the greedy edit on the two-tier ``player``
  board.
* ``greedy`` is the immediate-payout argmax
  (:class:`meta_arena.GreedyPlayer`) on the meta-arena family, and the
  cheapest-pair contraction on the ``contraction`` board.

Measured record (``docs/results_learned_baseline.md``): scripted
meta -0.100 reward/ep in 2.0 steps — the line every learned arm is
priced against; greedy meta -0.600 — the cash-out-early failure the
declare→handle line exists to beat; open heuristic 0.360 reward/ep,
AP 0.045 — the drain that wins reward but loses useful-per-eval to
the trained prior.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gamekit.players.registry import PlayerSpec

__all__ = ["SPECS", "make"]

_SPELLED_BOARDS: tuple[str, ...] = (
    "meta",
    "joint",
    "torch",
    "gen",
    "search",
    "open",
    "player",
)

SPECS: tuple[PlayerSpec, ...] = (
    PlayerSpec(
        name="playbook",
        kind="scripted",
        boards=_SPELLED_BOARDS,
        schema="— (a fixed line, not a scorer)",
        train="— (hand-written; nothing to train)",
        artifact="— (no weights)",
        record=(
            "meta -0.100 reward/ep in 2.0 steps; open drain 0.360 "
            "reward/ep, AP 0.045 (docs/results_learned_baseline.md)"
        ),
    ),
    PlayerSpec(
        name="greedy",
        kind="scripted",
        boards=("meta", "joint", "torch", "gen", "contraction"),
        schema=(
            "immediate-reward scores (lawdata.META_ARENA_REWARD); "
            "cheapest-pair on contraction"
        ),
        train="— (hand-written; nothing to train)",
        artifact="— (no weights)",
        record=(
            "meta -0.600 — cashes out early, misses declare→handle; "
            "contraction cheapest-pair is the field-adjacent floor"
        ),
    ),
)


def _open_drain(ctx: dict) -> Callable[[int], Any]:
    """Build the open-board drain — best heuristic-score first, stop.

    Deposits ``ctx["score_fn"]`` — the same scorer the episode arm
    ranks by, so the leaderboard's AP column prices its ordering.
    """
    from mlopt_discovery import proposal as lp
    from mlopt_discovery.meta_arena import Action

    ctx["score_fn"] = lp._heuristic_score

    def mk(_i: int) -> Any:
        def player(state: Any) -> Any:
            best_j = None
            best_s = None
            for j in state.pending:
                s = lp._heuristic_score(state.pool[j])
                if best_s is None or s > best_s:
                    best_s, best_j = s, j
            if best_j is None:
                return Action("stop", {})
            return Action(
                "evaluate", {"i": best_j, "cand": state.pool[best_j]}
            )

        return player

    return mk


def _pair_greedy(tensors: Any, sizes: dict[int, int]) -> float:
    """Cheapest-pair contraction — the contraction board's playbook."""
    from mlopt_torch.contraction_policy import greedy

    return greedy(tensors, sizes)


def make(
    domain: str,
    *,
    spec: PlayerSpec,
    seed: int = 0,
    ctx: dict | None = None,
) -> Callable[[int], Any]:
    """Return the scripted arm factory for *domain* — ``i -> player``."""
    del seed
    ctx = {} if ctx is None else ctx
    if spec.name == "greedy":
        if domain == "contraction":
            return lambda i: _pair_greedy
        from mlopt_discovery import meta_arena as ma

        return lambda i: ma.GreedyPlayer()
    if domain == "open":
        return _open_drain(ctx)
    if domain == "search":
        from mlopt_discovery.play import _scripted_search

        return lambda i: _scripted_search()
    if domain == "player":
        from mlopt_discovery import player_arena as pa

        return lambda i: pa.GreedyEditor()
    from mlopt_discovery import meta_arena as ma

    return lambda i: ma.ScriptedPlayer()
