"""``distilled`` — the oe-all distilled contraction player.

The shipped contraction-ordering player:
:class:`mlopt_torch.contraction_policy.ContractionPolicy` over the
bundled ``contraction_policy_distilled.pt`` weights — a
``PairPolicyNet`` distilled on every trial trajectory of
``opt_einsum``'s ``RandomGreedy`` teacher on
:func:`random_bond_network` boards (scales 8-24), played as
best-of-N temperature-sampled rollouts (:meth:`best_order`).  The
``contraction`` board is an *order* game: the arm factory's player is
``(tensors, sizes) -> cost``, not an episode callable.

Variants: ``distilled-argmax`` is the single argmax pass
(:meth:`order`) — documented fallback, brittle at n ≥ 30.

Measured record (``project/retros/contraction-synthesis.md``): under
the guided-restart protocol the sampled player **beats the teacher**
at equal wall-clock at n = 40 — 0.89-1.01 oe-cost ratios across three
seeds, 0.89-0.98 at the best temperature — the first outright win
over the field's best cheap player.  Honest limits: the argmax pass
reads 0.9-1.3x oe-greedy, and the player is throughput-starved under
~50 ms budgets where the affine prior starves it to one episode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mlopt_discovery.players.registry import PlayerSpec

__all__ = ["SPECS", "make"]

_ARTIFACT = "mlopt_torch/artifacts/contraction_policy_distilled.pt"

SPECS: tuple[PlayerSpec, ...] = (
    PlayerSpec(
        name="distilled",
        kind="distilled",
        boards=("contraction",),
        schema=(
            "scale-free-v1 — state 7 + pair 9 "
            "(mlopt_torch.contraction_policy)"
        ),
        train="bench/drivers/probes/contraction/contraction_distill.py --save",
        artifact=_ARTIFACT,
        record=(
            "beats oe RandomGreedy at equal wall-clock, n=40: "
            "0.89-1.01 oe-cost ratios, three seeds "
            "(project/retros/contraction-synthesis.md)"
        ),
    ),
    PlayerSpec(
        name="distilled-argmax",
        kind="distilled",
        boards=("contraction",),
        schema="same as ``distilled`` — single argmax pass",
        train="bench/drivers/probes/contraction/contraction_distill.py --save",
        artifact=_ARTIFACT,
        record=(
            "0.9-1.3x oe-greedy — the plausible fallback, brittle "
            "at n ≥ 30, not the player"
        ),
        variant={"argmax": True},
    ),
)


def make(
    domain: str,
    *,
    spec: PlayerSpec,
    seed: int = 0,
    ctx: dict | None = None,
) -> Callable[[int], Any]:
    """Return the contraction arm factory — ``i -> (t, s) -> cost``.

    ``ctx`` carries ``samples`` (rollouts per board, default 64),
    ``temperature`` (default ``SAMPLE_TEMP``), ``device`` (default
    ``"cpu"``) and optionally ``path`` for a non-default artifact.
    """
    from mlopt_torch.contraction_policy import (
        SAMPLE_TEMP,
        load_contraction_policy,
    )

    ctx = {} if ctx is None else ctx
    pol = load_contraction_policy(
        ctx.get("path"), device=str(ctx.get("device", "cpu"))
    )
    samples = int(ctx.get("samples", 64))
    temp = float(ctx.get("temperature", SAMPLE_TEMP))
    argmax = bool(spec.variant.get("argmax"))

    def mk(i: int) -> Any:
        def play(tensors: Any, sizes: dict[int, int]) -> float:
            if argmax:
                return pol.cost(tensors, sizes)
            _order, cost = pol.best_order(
                tensors,
                sizes,
                samples=samples,
                seed=seed + 60_000 + i,
                temperature=temp,
            )
            return cost

        return play

    return mk
