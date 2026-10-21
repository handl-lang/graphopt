"""``gated`` — the isolated stop channel: ranking + a side-channel quit.

The delivery the eight-round stop-policy loop kept converging on
(``docs/results_stop_isolated.md``): the ``stop`` row corrupts the
ranking whenever it shares the evaluate softmax, and every
weights-table fold of a working gate lands a degenerate point — so
the decision is delivered *outside* the softmax instead.  The player
is a :class:`~gamekit.players.GatedPlayer`: the inner ranker (the
domain's learned arm, its ``legal`` filtered to non-``stop`` moves)
proposes the ordering exactly as ``linear`` does, and the
``forced_action`` hook the engine consults first carries the quit
decision — a ``w_stop · φ_stop`` head or an injected score callable
compared against the calibrated break-even ``kappa``.

The arm is a *spec*, not a retraining recipe: ``ctx["gated"]`` feeds
``{spec, w_stop, kappa, score}`` — the stop row's own feature spec
(it may carry columns the ranker's schema lacks), the head weights,
the threshold and an optional ``(state, hist) -> float`` scorer.
Absent the config the channel is inert and the arm drains — the
honest never-stop control a registered player must still play.

Measured record (``docs/results_stop_isolated.md``, round 9 — the
memoized-verdict protocol): the channel reproduces the analytic
break-even gates and the calibrated head arms bit-for-bit against
their wrapper-gate twins, with the ranking softmax never seeing the
quit decision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gamekit.players._base import GatedPlayer, LearnedPlayer
from gamekit.players.registry import PlayerSpec

__all__ = ["SPECS", "make"]

SPECS: tuple[PlayerSpec, ...] = (
    PlayerSpec(
        name="gated",
        kind="gated",
        boards=("open",),
        schema=(
            "ranker: the domain's DomainSpec row; stop row: "
            "ctx['gated']['spec'] (may differ — the channel's schema)"
        ),
        train="— (scorer + κ are the payload; ctx['gated'] binds them)",
        artifact="lawdata stop-head table (weights + κ) — unbaked",
        record=(
            "the isolated-delivery arm — the quit decision rides the "
            "engine's forced_action channel, never the ranking "
            "softmax (docs/results_stop_isolated.md)"
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
    """Return the gated arm factory for *domain* — ``i -> player``.

    ``ctx['gated']`` carries the channel's payload:
    ``{weights, kappa, score, spec, rank_weights}`` —
    *weights*/*kappa* bind the linear stop head (``w·φ_stop`` over
    the *spec* featurizer, the domain's ``stop`` action row),
    *score* is an optional ``(state, hist) -> float`` override,
    *rank_weights* overrides the ranker's prior.  Unconfigured, the
    channel never fires early — the arm drains like a no-stop
    ``linear`` (the never-stop control the registered seat owes).
    ``ctx['score_fn']`` deposits the candidate-only page ranker.
    """
    del spec
    from mlopt_discovery.play import DOMAINS

    ctx = {} if ctx is None else ctx
    spec0 = DOMAINS[domain].spec
    if spec0 is None:
        raise ValueError(f"domain {domain!r} ships no DomainSpec")
    cfg = dict(ctx.get("gated") or {})
    stop_spec = cfg.get("spec", spec0)
    w_stop = cfg.get("weights")
    kappa = float(cfg.get("kappa", 0.0))
    score = cfg.get("score")
    w_rank = cfg.get("rank_weights", spec0.weights)
    legal = spec0.legal

    def _eval_only(state: Any) -> tuple:
        """Return the ranker's pick set — every non-stop legal move."""
        return tuple(
            a for a in legal(state) if getattr(a, "op", None) != "stop"
        )

    def _stop_move(state: Any) -> Any:
        return next(
            (
                a
                for a in legal(state)
                if getattr(a, "op", None) == "stop"
            ),
            None,
        )

    def _stop_feats(state: Any, hist: Any) -> dict:
        return stop_spec.featurize(state, _stop_move(state), hist)

    def factory(i: int) -> GatedPlayer:
        inner = LearnedPlayer(
            seed + 40_000 + i,
            spec=spec0,
            weights=w_rank,
            legal=_eval_only,
            learn=False,
        )
        return GatedPlayer(
            inner,
            legal=legal,
            stop_feats=_stop_feats,
            w_stop=w_stop,
            kappa=kappa,
            score=score,
        )

    if domain == "open":
        ctx["score_fn"] = LearnedPlayer(
            seed, spec=spec0, weights=w_rank
        ).rank_score
    return factory
