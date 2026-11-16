"""``mlopt.play`` — seat the ``deliver`` board's players.

The delivery-as-game plugin (``mlopt.deliver_game``): :func:`register`
puts the shipped gamekit players onto the ``deliver`` board through
the registry seam — ``gamekit.players.registry.register_player`` —
following the kernelopt ``play.register()`` convention: **importing
this module mutates nothing**; ``deliver_game`` calls ``register()``
at run time (idempotent — the seams overwrite by name).

There is no ``play.DOMAINS`` registry here — that table lives in
``kit_discovery``, which the product does not depend on.  The
deliver board is driven directly by :func:`mlopt.deliver_game.
deliver_game` over ``gamekit.engine.run_episode``; what the registry
provides is the *player* roster — ``make(name, "deliver")`` returns
the ``i -> player`` arm factory the episode driver consumes.

Seats (shipped names unioned with the ``deliver`` board):

* ``random`` — uniform over ``legal_actions`` (pending configs +
  ``stop``).
* ``playbook`` — the scripted line: drain configs in
  ``deliver_game.CONFIGS`` preference order, stop early once a
  verified config beats eager by >1%.
* ``linear`` / ``linear-greedy`` / ``linear-cold`` — the
  ``LearnedPlayer`` over :data:`DELIVERY_SPEC`.  The weight prior is
  :data:`mlopt.weights.DELIVERY_PLAYER_WEIGHTS` — ``{}`` today, so
  all three are cold-start arms; ``scripts/train_delivery_prior.py``
  bakes the literal that warms them (kernelopt's
  ``KERNEL_PLAYER_WEIGHTS`` convention).  ``linear-cold`` stays the
  untrained control either way.
"""

from __future__ import annotations

import random
from typing import Any

from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    LearnedPlayer,
    RelFeats,
    StateBroadcast,
    register_legal,
    register_part,
)
from gamekit.players.registry import PlayerSpec, register_player

from mlopt.weights import DELIVERY_PLAYER_WEIGHTS

__all__ = [
    "DELIVERY_SPEC",
    "register",
]

_BOARD = "deliver"


# ---------------------------------------------------------------------------
#  The feature schema — what a delivery player sees
# ---------------------------------------------------------------------------


#: Broadcast state features — progress through the drain, the best
#: measured speedup so far, the board's eligibility flags, and the
#: workload-shape facts (``DeliveryState``'s ``n_inputs`` /
#: ``n_enodes`` / ``n_params`` / ``lowprec`` — which config kinds
#: pay is shape-dependent: cuda-graph wants static CUDA shapes,
#: compile wants fusible regions, etc.).
def _state_feats(state: Any) -> dict[str, float]:
    return {
        "st:steps": state.steps / 8.0,
        "st:n_pending": len(state.pending) / 8.0,
        "st:n_done": len(state.results) / 8.0,
        "st:speedup": min(state.best_speedup, 4.0),
        "st:cuda": float(state.cuda),
        "st:fusible": float(state.fusible),
        "st:compile": float(state.compile_ok),
        "st:alias": float(state.alias_spec),
        "st:n_inputs": state.n_inputs / 8.0,
        "st:n_enodes": min(state.n_enodes, 8192) / 8192.0,
        "st:n_params": min(state.n_params, 1 << 30) / float(1 << 30),
        "st:lowprec": float(state.lowprec),
    }


def _config_kind(name: str) -> str:
    """The move-kind bucket a config name scores under."""
    if "fuse" in name and "graph" in name:
        return "fuse+graph"
    if "fuse" in name:
        return "fuse"
    if "compile" in name and "graph" in name:
        return "compile+graph"
    if "compile" in name:
        return "compile"
    if "graph-alias" in name:
        return "graph-alias"
    if "graph" in name:
        return "graph"
    return "other"


_KINDS: tuple[str, ...] = (
    "fuse+graph",
    "graph-alias",
    "graph",
    "fuse",
    "compile+graph",
    "compile",
    "other",
)


def _act_feats(action: Any, row: dict[str, float]) -> dict[str, float]:
    """Action features — the config's kind one-hot, alias flag, stop."""
    if action == "stop":
        return {"a:stop": 1.0, "x:stop:speedup": row.get("st:speedup", 0.0)}
    name = str(action)
    out = {f"a:kind:{_config_kind(name)}": 1.0}
    if "alias" in name:
        out["a:alias"] = 1.0
    if "fuse" in name:
        out["a:fuse"] = 1.0
    if "graph" in name:
        out["a:graph"] = 1.0
    return out


def _rel_feats(state: Any, action: Any, row: dict[str, float]) -> dict[str, float]:
    """State-relative action features — is this config still pending?

    ``x:pending`` marks actions still in the unevaluated pool (every
    legal config is, by construction — the feature exists so a
    trained prior can learn *which kind* of pending move pays);
    ``x:stop:improve`` prices stopping against banked speedup.
    """
    if action == "stop":
        return {"x:stop:improve": row.get("st:speedup", 0.0)}
    return {"x:pending": float(str(action) in state.pending)}


DELIVERY_FEATURES: tuple[str, ...] = (
    "bias",
    "st:steps",
    "st:n_pending",
    "st:n_done",
    "st:speedup",
    "st:cuda",
    "st:fusible",
    "st:compile",
    "st:alias",
    "st:n_inputs",
    "st:n_enodes",
    "st:n_params",
    "st:lowprec",
    *(f"a:kind:{k}" for k in _KINDS),
    "a:alias",
    "a:fuse",
    "a:graph",
    "a:stop",
    "x:stop:speedup",
    "x:pending",
    "x:stop:improve",
)


#: The delivery spec — intercept, drain-progress/speedup broadcast,
#: config-kind action one-hots, pending/stop interactions.  ``weights``
#: is the baked ``mlopt.weights.DELIVERY_PLAYER_WEIGHTS`` table —
#: ``{}`` → ``None`` today (honest cold start); pasting the trainer's
#: literal warms every ``linear*`` arm at once.
DELIVERY_SPEC = DomainSpec.build(
    features=DELIVERY_FEATURES,
    legal=lambda state: (*state.pending, "stop"),
    parts=(
        Bias(),
        StateBroadcast(_state_feats),
        ActionFeats(_act_feats),
        RelFeats(_rel_feats),
    ),
    weights=dict(DELIVERY_PLAYER_WEIGHTS) or None,
)


# ---------------------------------------------------------------------------
#  Arm factories — the ``make(domain, *, spec, seed, ctx) -> i -> player``
# ---------------------------------------------------------------------------


def _check_domain(domain: str) -> None:
    if domain != _BOARD:
        raise ValueError(f"mlopt play seats only {_BOARD!r} — got {domain!r}")


def _make_random(domain: str, *, spec: PlayerSpec, seed: int = 0, ctx: Any = None) -> Any:
    """Uniform over the legal set — the honesty floor."""
    _check_domain(domain)
    del spec, ctx
    from mlopt.deliver_game import legal_actions  # lazy: torch-free path

    def mk(i: int) -> Any:
        rng = random.Random(seed + 10_000 + i)

        def player(state: Any) -> Any | None:
            acts = legal_actions(state)
            return acts[rng.randrange(len(acts))] if acts else None

        return player

    return mk


def _make_playbook(domain: str, *, spec: PlayerSpec, seed: int = 0, ctx: Any = None) -> Any:
    """The scripted line — preference-order drain, early stop on a win.

    Evaluates pending configs in ``deliver_game.CONFIGS`` order (the
    old heuristic's composition preference, now *measured* rather
    than assumed) and stops once a verified config beats eager past
    the ``WIN_MARGIN`` noise floor — the same gate ``committed()``
    applies, so the scripted line never stops on a noise-level win.
    """
    _check_domain(domain)
    del spec, seed, ctx
    from mlopt.deliver_game import CONFIGS

    def mk(_i: int) -> Any:
        def player(state: Any) -> Any | None:
            if state.committed_speedup > 1.0:
                return "stop"
            for name in CONFIGS:
                if name in state.pending:
                    return name
            return "stop" if state.pending or state.steps else None

        return player

    return mk


def _make_linear(domain: str, *, spec: PlayerSpec, seed: int = 0, ctx: Any = None) -> Any:
    """``LearnedPlayer`` over ``DELIVERY_SPEC`` — cold until trained."""
    _check_domain(domain)
    del ctx
    weights = {} if spec.variant.get("cold") else DELIVERY_SPEC.weights
    greedy = bool(spec.variant.get("greedy"))
    player = LearnedPlayer(seed, spec=DELIVERY_SPEC, weights=weights)
    return lambda i: player.frozen(seed + 20_000 + i, greedy=greedy)


# ---------------------------------------------------------------------------
#  Registration — explicit, idempotent, import-mutates-nothing
# ---------------------------------------------------------------------------


def _spec(
    name: str,
    kind: str,
    record: str,
    *,
    variant: dict | None = None,
) -> PlayerSpec:
    return PlayerSpec(
        name=name,
        kind=kind,
        boards=(_BOARD,),
        schema="mlopt.play.DELIVERY_SPEC — drain-progress + config-kind row",
        train="scripts/train_delivery_prior.py (imitate argmin oracle + REINFORCE)",
        artifact="mlopt.weights.DELIVERY_PLAYER_WEIGHTS ({} = cold)",
        record=record,
        variant=variant or {},
    )


def register() -> None:
    """Seat the ``deliver`` board's players — call, don't import."""
    register_part("deliver:state", lambda: StateBroadcast(_state_feats))
    register_part("deliver:action", lambda: ActionFeats(_act_feats))
    register_part("deliver:rel", lambda: RelFeats(_rel_feats))
    register_legal(_BOARD, lambda state: (*state.pending, "stop"))

    register_player(
        _spec("random", "chance", "uniform over pending configs + stop"),
        make=_make_random,
    )
    register_player(
        _spec(
            "playbook",
            "scripted",
            "CONFIGS-order drain; early stop at >1% over eager",
        ),
        make=_make_playbook,
    )
    register_player(
        _spec(
            "linear",
            "linear",
            "LearnedPlayer on DELIVERY_SPEC — prior = DELIVERY_PLAYER_WEIGHTS",
        ),
        make=_make_linear,
    )
    register_player(
        _spec(
            "linear-greedy",
            "linear",
            "same spec as linear, argmax decode",
            variant={"greedy": True},
        ),
        make=_make_linear,
    )
    register_player(
        _spec(
            "linear-cold",
            "linear",
            "the untrained control — weights={}",
            variant={"cold": True},
        ),
        make=_make_linear,
    )
