"""``linear`` — the shipped trained prior (``LearnedPlayer`` + baked weights).

The reusable :class:`~mlopt_discovery.players.LearnedPlayer` bound to
the domain's own :class:`~mlopt_discovery.players.DomainSpec` — the
feature schema, the composed featurizer and the legal-move enumerator
come from ``play.DOMAINS[d].spec``, the weight table from
``lawdata`` — so the same module plays every episode domain that
ships a spec.  Variants:

* ``linear`` — the shipped prior, softmax-sampled;
* ``linear-greedy`` — same weights, argmax decode;
* ``linear-cold`` — ``weights={}``, the untrained control the prior
  must beat (on ``search`` the spec ships no prior at all, so
  ``linear`` *is* the cold start there — the row says so).

Measured record (``docs/results_learned_baseline.md``): meta -0.0563
reward/ep (greedy decode -0.0500) — the best episode arm; open 0.327
reward/ep, **0.0121 useful/eval** — the best rate on the board — and
page-ranking AP 0.133, 3x the heuristic's 0.045.  Cold start: meta
-0.488, open AP ≈ the 0.012 base rate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mlopt_discovery.players.registry import EPISODE_BOARDS, PlayerSpec

__all__ = ["SPECS", "make"]

_TRAIN = (
    "scripts/train_meta_prior.py · scripts/train_open_prior.py · "
    "scripts/train_player_meta_arm.py"
)
_ARTIFACT = (
    "lawdata.META_ARENA_PLAYER_WEIGHTS / OPEN_GAME_PLAYER_WEIGHTS / "
    "PLAYER_ARENA_PLAYER_WEIGHTS (SEARCH_SPEC ships none — cold)"
)

SPECS: tuple[PlayerSpec, ...] = (
    PlayerSpec(
        name="linear",
        kind="linear",
        boards=EPISODE_BOARDS,
        schema=(
            "the domain's DomainSpec row — META_SPEC / OPEN_SPEC / "
            "SEARCH_SPEC / PLAYER_SPEC"
        ),
        train=_TRAIN,
        artifact=_ARTIFACT,
        record=(
            "meta -0.0563 reward/ep — best episode arm; open 0.327 "
            "reward/ep, 0.0121 useful/eval, AP 0.133 (3x heuristic) "
            "(docs/results_learned_baseline.md)"
        ),
    ),
    PlayerSpec(
        name="linear-greedy",
        kind="linear",
        boards=EPISODE_BOARDS,
        schema="same as ``linear`` — argmax decode",
        train=_TRAIN,
        artifact=_ARTIFACT,
        record=(
            "meta -0.0500; open 0.358 — greedy decode matches the "
            "heuristic drain on reward, still wins useful/eval"
        ),
        variant={"greedy": True},
    ),
    PlayerSpec(
        name="linear-cold",
        kind="linear",
        boards=EPISODE_BOARDS,
        schema="same as ``linear`` — ``weights={}``",
        train="— (the untrained control)",
        artifact="— (the point is no weights)",
        record=(
            "meta -0.488 reward/ep; open 0.170, AP ≈ base rate — "
            "the control the prior must beat"
        ),
        variant={"cold": True},
    ),
)


def make(
    domain: str,
    *,
    spec: PlayerSpec,
    seed: int = 0,
    ctx: dict | None = None,
) -> Callable[[int], Any]:
    """Return the learned arm factory for *domain* — ``i -> player``.

    The domain's spec supplies the enumerator, featurizer, schema and
    prior; ``variant.cold`` blanks the weights, ``variant.greedy``
    argmax-decodes the frozen copies.  On ``open`` the bound
    ``rank_score`` is deposited as ``ctx["score_fn"]`` — the
    candidate-only page ordering the leaderboard's AP column prices.
    """
    from mlopt_discovery.play import DOMAINS
    from mlopt_discovery.players import LearnedPlayer

    ctx = {} if ctx is None else ctx
    spec0 = DOMAINS[domain].spec
    if spec0 is None:
        raise ValueError(f"domain {domain!r} ships no DomainSpec")
    weights = {} if spec.variant.get("cold") else spec0.weights
    player = LearnedPlayer(seed, spec=spec0, weights=weights)
    if domain == "open":
        ctx["score_fn"] = player.rank_score
    greedy = bool(spec.variant.get("greedy"))
    return lambda i: player.frozen(seed + 20_000 + i, greedy=greedy)
