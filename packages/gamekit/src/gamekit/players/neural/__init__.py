"""``neural`` — the ``RuleValueNet`` MLP arm (open board only).

A small MLP over ``features(lhs) + rule_vector(as_rule)`` trained
per-run on the *measured* train-page outcomes
(``mlopt_torch.learned_policy.train_rule_value``, BCE on
``base_cost > cand_cost`` — the on-distribution version of
``proposal._train_scorer``), then played as a drain: evaluate the
pending member with the highest predicted value, ``stop`` when the
page is out.  The scorer itself is deposited as ``ctx["score_fn"]``
so the leaderboard's AP column prices its ordering too.

The arm needs ``ctx["train"]`` — ``[(candidate, outcome), ...]`` of
memoized oracle verdicts from the held-in pages; the comparison
runner builds it.  Without it there is nothing honest to play.

Measured record (``docs/results_learned_baseline.md``): AP 0.041 —
beats the coin flip, still loses to the cost-delta heuristic's 0.045
and badly to the trained linear prior's 0.133.  Capacity is not the
binding constraint — h32 ≈ h96 in the ablations; the feature row is.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gamekit.players.registry import PlayerSpec

__all__ = ["SPECS", "make"]

SPECS: tuple[PlayerSpec, ...] = (
    PlayerSpec(
        name="neural",
        kind="neural",
        boards=("open",),
        schema="compute_features(lhs) + rule_vector(as_rule) — the RuleValueNet input vector",
        train=(
            "trained in the run — train_rule_value(BCE) on the "
            "held-in pages' memoized verdicts "
            "(bench/drivers/policy/policy_baseline.py protocol)"
        ),
        artifact="— (no shipped weights; trained per comparison)",
        record=(
            "open AP 0.041 — beats the coin, loses to heuristic 0.045 "
            "and to trained-linear 0.133; h32 ≈ h96 — features bind, "
            "not capacity (docs/results_learned_baseline.md)"
        ),
    ),
)


def make(
    domain: str,
    *,
    spec: PlayerSpec,
    seed: int = 0,
    ctx: dict | None = None,
) -> Callable[[int], Any]:
    """Return the neural arm factory for ``open`` — ``i -> player``.

    Trains one ``RuleValueNet`` on ``ctx["train"]`` (the held-in
    pages' ``[(cand, outcome)]`` rows) and plays drain-by-predicted-
    value.  Raises ``ValueError`` when the ctx carries no training
    rows — an untrained neural player is a coin flip with extra
    steps, and the leaderboard does not fake it.
    """
    del spec
    ctx = {} if ctx is None else ctx
    train = ctx.get("train")
    if not train:
        raise ValueError(
            "neural needs ctx['train'] — [(cand, outcome)] from the "
            "held-in pages' memoized verdicts"
        )

    import torch
    from mlopt_core.features import compute_features
    from mlopt_core.trajectories import RuleSample, rule_vector
    from mlopt_torch.learned_policy import (
        input_vector,
        train_rule_value,
    )

    samples = [
        RuleSample(
            compute_features(c.lhs),
            rule_vector(c.as_rule()),
            c.as_rule().name,
            o.base_cost - o.cand_cost,
            c.label,
        )
        for c, o in train
    ]
    model = train_rule_value(
        samples,
        epochs=int(ctx.get("epochs", 300)),
        hidden=int(ctx.get("hidden", 32)),
        device=str(ctx.get("device", "cpu")),
        seed=seed,
    )
    model.eval()

    def scorer(cand: Any) -> float:
        """Predicted improvement — the trained net's page ordering."""
        x = torch.tensor(
            [
                input_vector(
                    compute_features(cand.lhs),
                    rule_vector(cand.as_rule()),
                )
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            return float(model(x)[0])

    ctx["score_fn"] = scorer

    from mlopt_discovery.meta_arena import Action

    def mk(_i: int) -> Any:
        def player(state: Any) -> Any:
            best_j = None
            best_s = None
            for j in state.pending:
                s = scorer(state.pool[j])
                if best_s is None or s > best_s:
                    best_s, best_j = s, j
            if best_j is None:
                return Action("stop", {})
            return Action(
                "evaluate", {"i": best_j, "cand": state.pool[best_j]}
            )

        return player

    return mk
