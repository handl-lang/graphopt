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

import math
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
#: measured speedup so far, the board's eligibility flags, the
#: workload-shape facts (``DeliveryState``'s ``n_inputs`` /
#: ``n_enodes`` / ``n_params`` / ``lowprec`` — which config kinds
#: pay is shape-dependent: cuda-graph wants static CUDA shapes,
#: compile wants fusible regions, etc.), and the v2 measured
#: signals: the construction-time eager median (``st:eager_ms`` —
#: v1 never exposed the absolute timing scale), scored/declined
#: referee counts, the margin-gated ``committed`` speedup, and the
#: big-model regime flag.  Action/rel features add the per-kind
#: measured speedups (``x:meas:*``), tried flags, kind×fusible
#: interactions, the ``CONFIGS`` preference index (``a:pref``) and
#: the stop-vs-run mirrors (``x:stop:*`` vs ``x:act:*``).
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
        # v2 measured signals — all observable at decision time.
        # ``eager_med_ms`` is measured at board construction (the
        # shipped fallback's own number) but v1 never exposed it;
        # the absolute scale (0.1 ms hnl kernels vs 250 ms whisper
        # encoders) is the single strongest eager-win predictor in
        # the v7 corpus.  ``n_scored``/``n_declined`` count referee
        # outcomes so far; ``committed`` is the margin-gated
        # speedup ``committed()`` would ship (``st:speedup`` is the
        # raw argmin ratio — they differ exactly when the best
        # config is inside the WIN_MARGIN noise band); ``big``
        # flags the regime where no runner paid in v7 (huge eager
        # step or huge term DAG — memory/launch bound).
        "st:eager_ms": _eager_feat(state),
        "st:n_scored": _n_scored(state) / 8.0,
        "st:n_declined": _n_declined(state) / 8.0,
        "st:committed": min(_committed(state), 4.0),
        "st:big": _big_model(state),
    }


def _eager_feat(state: Any) -> float:
    """Log-scaled eager median ms — 0.0 when unmeasured.

    ``eager_med_ms`` spans ~0.05 ms (tiny hnl kernels) to ~250 ms
    (whisper encoder steps); log10/3 maps that to roughly
    [-0.45, 0.85].  Every read is guarded — a state without the
    field (tests, foreign boards) contributes 0.0.
    """
    raw = getattr(state, "eager_med_ms", None)
    if raw is None:
        return 0.0
    try:
        emed = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if emed <= 0:
        return 0.0
    return math.log10(max(emed, 1e-4)) / 3.0


def _n_scored(state: Any) -> int:
    """Referee records with a measured ``med_ms`` so far."""
    results = getattr(state, "results", None) or {}
    return sum(1 for r in results.values() if isinstance(r, dict) and r.get("med_ms") is not None)


def _n_declined(state: Any) -> int:
    """Referee records that produced no measurement (decline/skip)."""
    results = getattr(state, "results", None) or {}
    return sum(1 for r in results.values() if isinstance(r, dict) and r.get("med_ms") is None)


def _committed(state: Any) -> float:
    """``committed_speedup`` when the state exposes it, else 1.0."""
    try:
        return float(getattr(state, "committed_speedup", 1.0) or 1.0)
    except Exception:  # noqa: BLE001 — a foreign state never sinks a row
        return 1.0


def _big_model(state: Any) -> float:
    """1.0 in the regime where v7 measured no runner paying.

    Huge eager steps (>30 ms — whisper/t5 encoder arms) and huge
    term DAGs (>300 enodes) were uniformly eager-wins in the v7
    corpus; the flag lets the stop-vs-run interactions price that
    regime without a hand threshold on the continuous features.
    """
    emed = getattr(state, "eager_med_ms", None) or 0.0
    enodes = getattr(state, "n_enodes", 0) or 0
    try:
        if float(emed) > 30.0 or int(enodes) > 300:
            return 1.0
    except (TypeError, ValueError):  # noqa: BLE001
        return 0.0
    return 0.0


def _measured_kinds(state: Any) -> tuple[dict[str, float], set[str]]:
    """``(kind -> best measured speedup, tried kinds)`` from results.

    The referee records carry ``med_ms`` plus the config's OWN
    paired-round eager median (``eager_med_ms``) — the drift-matched
    baseline ``committed_speedup`` compares against.  A kind's
    measured value is the best speedup any evaluated config of that
    kind posted (capped at 4.0); a kind is *tried* once any result
    exists for it, scored or declined.  Both are honest
    decision-time signals: they only ever read ``state.results``.
    """
    best: dict[str, float] = {}
    tried: set[str] = set()
    results = getattr(state, "results", None) or {}
    fallback_eager = getattr(state, "eager_med_ms", None)
    for name, rec in results.items():
        if not isinstance(rec, dict):
            continue
        kind = _config_kind(str(name))
        tried.add(kind)
        med = rec.get("med_ms")
        if med is None:
            continue
        try:
            med_f = float(med)
            base = rec.get("eager_med_ms") or fallback_eager
            if base is None:
                continue
            base_f = float(base)
        except (TypeError, ValueError):
            continue
        if med_f <= 0 or base_f <= 0:
            continue
        speedup = min(base_f / med_f, 4.0)
        if kind not in best or speedup > best[kind]:
            best[kind] = speedup
    return best, tried


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
    """Action features — the config's kind one-hot, alias flag, stop.

    ``a:pref`` is the config's position in ``deliver_game.CONFIGS``
    preference order (0 = richest composition first, 1 = ``compile``
    last) — a learned prior can then price "try the expensive
    composition first" separately from the kind identity itself.
    """
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
    try:
        from mlopt.deliver_game import CONFIGS  # lazy: torch-free path

        out["a:pref"] = CONFIGS.index(name) / max(1, len(CONFIGS) - 1)
    except (ImportError, ValueError):  # noqa: BLE001 — unlisted config
        out["a:pref"] = 1.0
    return out


def _rel_feats(state: Any, action: Any, row: dict[str, float]) -> dict[str, float]:
    """State-relative action features — pending, measured, stop price.

    v2 additions, all honest decision-time signals:

    * ``x:meas:<kind>`` — the best measured speedup (paired
      eager/config medians) posted so far by a config of *this*
      action's kind; 0.0 when the kind is unmeasured.  Lets the
      prior learn "this kind already paid/didn't on THIS board"
      rather than only the cross-case kind prior.
    * ``x:tried:<kind>`` — any referee record exists for the kind
      (scored or declined).
    * ``x:kfus:<kind>`` — the board's fusible flag scoped to the
      action's kind (fuse-family configs only exist when fusible,
      but the interaction lets the prior price fuse-vs-graph
      differently per board).
    * ``x:act:eager_ms`` / ``x:act:big`` — the board's eager scale
      and big-model flag emitted on *config* rows only; paired with
      the ``x:stop:*`` mirrors below they are the stop-vs-run
      interaction ("drain cheap boards, bail on huge ones").
    * stop rows carry ``x:stop:committed`` (the margin-gated
      speedup — not the raw argmin ``st:speedup``), ``n_pending``
      and the same eager/big mirrors.
    """
    if action == "stop":
        return {
            "x:stop:speedup": row.get("st:speedup", 0.0),
            "x:stop:improve": row.get("st:speedup", 0.0),
            "x:stop:committed": row.get("st:committed", 0.0),
            "x:stop:eager_ms": row.get("st:eager_ms", 0.0),
            "x:stop:big": row.get("st:big", 0.0),
            "x:stop:n_pending": len(getattr(state, "pending", ()) or ()) / 8.0,
        }
    name = str(action)
    kind = _config_kind(name)
    meas, tried = _measured_kinds(state)
    out = {
        "x:pending": float(name in (getattr(state, "pending", ()) or ())),
        "x:act:eager_ms": row.get("st:eager_ms", 0.0),
        "x:act:big": row.get("st:big", 0.0),
        f"x:meas:{kind}": meas.get(kind, 0.0),
        f"x:tried:{kind}": float(kind in tried),
        f"x:kfus:{kind}": float(bool(getattr(state, "fusible", False))),
    }
    return out


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
    "st:eager_ms",
    "st:n_scored",
    "st:n_declined",
    "st:committed",
    "st:big",
    *(f"a:kind:{k}" for k in _KINDS),
    "a:alias",
    "a:fuse",
    "a:graph",
    "a:pref",
    "a:stop",
    "x:stop:speedup",
    "x:pending",
    "x:stop:improve",
    *(f"x:meas:{k}" for k in _KINDS),
    *(f"x:tried:{k}" for k in _KINDS),
    *(f"x:kfus:{k}" for k in _KINDS),
    "x:act:eager_ms",
    "x:act:big",
    "x:stop:committed",
    "x:stop:eager_ms",
    "x:stop:big",
    "x:stop:n_pending",
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
