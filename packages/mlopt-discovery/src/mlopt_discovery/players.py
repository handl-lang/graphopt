"""Players for the generic optimization game.

:class:`LinearPolicy` is the reusable learner: a softmax linear
policy over whatever ``legal(state)`` enumerates, scored by an
injected ``featurizer``.  Training is episode REINFORCE with
reward-to-go credit (a move gets the terminal payout it armed), an
EMA baseline and clipped advantage; *epsilon* mixes uniform
exploration into the sample for sparse-reward boards.

The player/featurizer abstraction is *data*: a domain ships a
:class:`DomainSpec` — its feature schema, its featurizer's
**composable parts** (a state broadcast, the play-rate history, the
move-kind one-hot, the action's own params) and its legal-move
enumerator.  :class:`LearnedPlayer` is the one policy built from
that spec; :class:`PlayHist` is the shared play-rate history the
meta and open featurizers both read.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LEGALS",
    "PARTS",
    "RANKS",
    "ActionFeats",
    "Bias",
    "DomainSpec",
    "LearnedPlayer",
    "LinearPolicy",
    "OpOneHot",
    "PlayHist",
    "PlayRates",
    "RelFeats",
    "StateBroadcast",
    "compose_featurizer",
    "register_legal",
    "register_part",
    "register_rank",
    "spec_from_record",
]

#: REINFORCE constants — per-episode-return update normalized by
#: step count, EMA running baseline, clipped advantage.
_LR = 0.1
_BASELINE_EMA = 0.2
_CLIP = 40.0


def _dot(w: dict, feats: dict) -> float:
    """Return the linear score ``w·feats`` over a sparse feature dict."""
    s = 0.0
    for k, v in feats.items():
        wv = w.get(k)
        if wv:
            s += wv * v
    return s


def _action_key(action: Any) -> tuple:
    """Stable dedup key — the played-move mask, replayable."""
    params = getattr(action, "params", None)
    if params is None:
        return ("call", repr(action))
    return (
        getattr(action, "op", "?"),
        repr(sorted(params.items(), key=lambda kv: kv[0])),
    )


def _grad_row(
    feats: list[dict], exps: list[float], i: int
) -> tuple[dict, dict]:
    """``(φ_chosen, E_π[φ])`` under the softmax over unplayed moves."""
    z = sum(exps)
    ephi: dict[str, float] = {}
    for f, e in zip(feats, exps, strict=True):
        for k, v in f.items():
            ephi[k] = ephi.get(k, 0.0) + v * e / z
    return feats[i], ephi


@dataclass
class PlayHist:
    """Within-episode play counts — the shared adaptation channel.

    ``record`` tallies one played move of kind *op*; ``rate`` is that
    kind's share of the episode's plays so far.  Both the meta and the
    open featurizer render their ``h:<op>`` rows from it, so it lives
    with the player rather than with either domain.
    """

    counts: dict[str, int] = field(default_factory=dict)
    n: int = 0

    def record(self, op: str) -> None:
        """Tally one played move of kind *op*."""
        self.counts[op] = self.counts.get(op, 0) + 1
        self.n += 1

    def rate(self, op: str) -> float:
        """Share of this episode's plays of kind *op*."""
        return self.counts.get(op, 0) / max(self.n, 1)


#: The pre-consolidation name, kept as an alias for callers/tests.
_Hist = PlayHist


# ---------------------------------------------------------------------------
#  Composable feature parts — a domain's featurizer as data
# ---------------------------------------------------------------------------
#
#  Every part is a callable ``(state, action, hist, row) -> dict``: it
#  returns the slice of the feature row it owns, and ``row`` is the
#  partially-built row so a later part can read an earlier one's keys
#  (the interaction terms).  A domain composes an ordered tuple of
#  parts and the player reads the row they build — no per-domain
#  ``featurize`` function.


@dataclass(frozen=True)
class Bias:
    """The constant intercept every feature row carries."""

    def __call__(
        self, _state: Any, _action: Any, _hist: Any, _row: dict
    ) -> dict[str, float]:
        """Return ``{"bias": 1.0}``."""
        return {"bias": 1.0}


@dataclass(frozen=True)
class StateBroadcast:
    """A domain's ``st:*`` block — ``fn(state) -> dict``."""

    fn: Callable[[Any], dict[str, float]]

    def __call__(
        self, state: Any, _action: Any, _hist: Any, _row: dict
    ) -> dict[str, float]:
        """Return the board-summary features for *state*."""
        return self.fn(state)


@dataclass(frozen=True)
class PlayRates:
    """``h:<op>`` rows — the play rates of *ops* this episode."""

    ops: tuple[str, ...]

    def __call__(
        self, _state: Any, _action: Any, hist: PlayHist, _row: dict
    ) -> dict[str, float]:
        """Return ``hist.rate(op)`` for each tracked op."""
        return {f"h:{op}": hist.rate(op) for op in self.ops}


@dataclass(frozen=True)
class OpOneHot:
    """The ``op:<op>`` move-kind one-hot, filtered to the schema."""

    schema: frozenset

    def __call__(
        self, _state: Any, action: Any, _hist: Any, _row: dict
    ) -> dict[str, float]:
        """Return the move-kind one-hot when the schema names it."""
        key = f"op:{action.op}"
        return {key: 1.0} if key in self.schema else {}


@dataclass(frozen=True)
class ActionFeats:
    """A domain's action-param block — ``fn(action, row) -> dict``.

    *row* is the partially-built feature row, so an interaction term
    can read a broadcast value (``x:extract:improve = row["st:improve"]``).
    """

    fn: Callable[[Any, dict], dict[str, float]]

    def __call__(
        self, _state: Any, action: Any, _hist: Any, row: dict
    ) -> dict[str, float]:
        """Return the action's own params and interaction terms."""
        return self.fn(action, row)


@dataclass(frozen=True)
class RelFeats:
    """Action features *relative to the board* — ``fn(state, action, row)``.

    ``ActionFeats`` sees only the action and the partially-built row,
    so it cannot express *candidate-vs-pool* terms — "is this the
    best still-pending option" needs the rest of the board.  This is
    the part for them: its *fn* additionally receives ``state`` (the
    open game's pool and pending mask, say) while ``row`` still lets
    it read an earlier part's emissions.
    """

    fn: Callable[[Any, Any, dict], dict[str, float]]

    def __call__(
        self, state: Any, action: Any, _hist: Any, row: dict
    ) -> dict[str, float]:
        """Return the state-relative action features."""
        return self.fn(state, action, row)


def compose_featurizer(parts: tuple) -> Callable:
    """Compose ordered feature *parts* into one featurizer.

    Returns ``featurize(state, action, hist=None) -> dict``: each part
    updates the row in order, so a later part can read an earlier
    part's keys.  *hist* defaults to an empty :class:`PlayHist`.
    """

    def featurize(
        state: Any, action: Any, hist: PlayHist | None = None
    ) -> dict[str, float]:
        h = PlayHist() if hist is None else hist
        row: dict[str, float] = {}
        for part in parts:
            row.update(part(state, action, h, row))
        return row

    return featurize


# ---------------------------------------------------------------------------
#  The parts registry — a spec as *data* (the player-on-board term)
# ---------------------------------------------------------------------------
#
#  ``DomainSpec`` is ~70% data already (``features``, ``weights``,
#  ``lr``); the rest is callables — the composed ``featurize``, the
#  ``legal`` enumerator, the candidate ``rank`` scorer and the
#  ``fn`` inside ``StateBroadcast``/``ActionFeats`` parts.  The
#  registries below give every callable a *name*, so a spec reifies
#  to a record of tags — the term the player arena
#  (:mod:`mlopt_discovery.player_arena`) rewrites.  The idiom is the
#  codebase's own: ``meta_arena.ACTIONS``, ``genkernel.TIERS``,
#  ``lawdata.HANDLERS`` — a new binding is a registry entry, not a
#  player subclass.

#: Part tag -> constructor: ``PARTS[tag]()`` builds the part.
PARTS: dict[str, Callable[[], Any]] = {}

#: Domain tag -> legal-move enumerator ``state -> tuple``.
LEGALS: dict[str, Any] = {}

#: Domain tag -> candidate ranker ``cand -> dict``.
RANKS: dict[str, Any] = {}


def register_part(tag: str, ctor: Callable[[], Any]) -> Callable:
    """Register *ctor* as the constructor behind part tag *tag*.

    ``register_part("meta:state", lambda: StateBroadcast(f))`` — a
    part the shipped vocabulary owns is one registry entry; a spec
    record then names it by tag.
    """
    PARTS[tag] = ctor
    return ctor


def register_legal(tag: str, fn: Any) -> Any:
    """Register *fn* as the legal-move enumerator for domain *tag*."""
    LEGALS[tag] = fn
    return fn


def register_rank(tag: str, fn: Any) -> Any:
    """Register *fn* as the candidate ranker for domain *tag*."""
    RANKS[tag] = fn
    return fn


def _part_tag(part: Any) -> str:
    """Resolve *part* to its registry tag (equality, not identity)."""
    for tag, ctor in PARTS.items():
        if ctor() == part:
            return tag
    raise ValueError(f"unregistered feature part {part!r}")


def _fn_tag(fn: Any, table: dict, what: str) -> str:
    """Resolve *fn* to its registry tag (identity, then equality)."""
    for tag, f in table.items():
        if f is fn or f == fn:
            return tag
    raise ValueError(f"unregistered {what} {fn!r}")


@dataclass(frozen=True)
class DomainSpec:
    """Per-domain player data — one :class:`LearnedPlayer`, many specs.

    A domain declares its feature *schema*, the composable *parts*
    that produce a feature row, its legal-move *enumerator*, its
    default weight *prior* and (optionally) a candidate *rank*
    scorer.  The player is constructed from this data; adding a
    domain is a spec, never a player subclass.

    ``featurize`` is the composed ``(state, action, hist) -> dict``
    (:meth:`build` composes it from *parts*); it is a *stored*
    attribute, so a module can re-export the very same function
    object the registry binds.
    """

    features: tuple[str, ...]
    legal: Any
    parts: tuple
    featurize: Any
    weights: Mapping[str, float] | None = None
    rank: Any = None
    lr: float = _LR

    def new_hist(self) -> PlayHist:
        """Return an empty play-rate history for this domain."""
        return PlayHist()

    @classmethod
    def build(
        cls,
        *,
        features: Iterable[str],
        legal: Any,
        parts: tuple,
        weights: Mapping[str, float] | None = None,
        rank: Any = None,
        lr: float = _LR,
    ) -> DomainSpec:
        """Build a spec, composing *parts* into its featurizer."""
        return cls(
            features=tuple(features),
            legal=legal,
            parts=tuple(parts),
            featurize=compose_featurizer(parts),
            weights=weights,
            rank=rank,
            lr=lr,
        )

    def to_record(self) -> dict[str, Any]:
        """Reify the spec to a pure-data record — the rewritable term.

        Every callable field resolves to its registry tag
        (:data:`PARTS`, :data:`LEGALS`, :data:`RANKS`); the record is
        JSON-serializable — ``{features, legal, parts, weights, rank,
        lr}`` — and :meth:`build_from_record` restores the spec.
        Raises ``ValueError`` when a part or callable was never
        registered (a spec the record cannot name is not data).
        """
        return {
            "features": list(self.features),
            "legal": _fn_tag(self.legal, LEGALS, "legal enumerator"),
            "parts": [_part_tag(p) for p in self.parts],
            "weights": (
                dict(self.weights) if self.weights is not None else None
            ),
            "rank": (
                None
                if self.rank is None
                else _fn_tag(self.rank, RANKS, "rank scorer")
            ),
            "lr": self.lr,
        }

    @classmethod
    def build_from_record(cls, rec: Mapping[str, Any]) -> DomainSpec:
        """Rebuild a spec from its record — tags resolve to callables.

        The inverse of :meth:`to_record`: part tags construct through
        :data:`PARTS`, the ``legal``/``rank`` tags through
        :data:`LEGALS`/:data:`RANKS`, and the parts recompose into
        the featurizer through :meth:`build`.
        """
        rank = rec.get("rank")
        return cls.build(
            features=tuple(rec["features"]),
            legal=LEGALS[str(rec["legal"])],
            parts=tuple(PARTS[str(t)]() for t in rec["parts"]),
            weights=rec.get("weights"),
            rank=None if rank is None else RANKS[str(rank)],
            lr=float(rec.get("lr", _LR)),
        )


def spec_from_record(rec: Mapping[str, Any]) -> DomainSpec:
    """Rebuild a spec from its record — :meth:`DomainSpec.build_from_record`."""
    return DomainSpec.build_from_record(rec)


#: The shared intercept is the one part every domain's row carries —
#: registered here so any spec record can name it.
register_part("bias", Bias)


class LinearPolicy:
    """A softmax linear policy over a board's legal moves.

    ``score(move) = w·φ(state, move)`` where φ comes from the
    injected *featurizer* ``(state, action, hist) -> dict`` and *w*
    is a ``{feature_name: float}`` table over the *features* schema.
    A step samples the softmax over the *unplayed* legal set
    (``greedy`` → argmax; *temperature* scales logits; *epsilon*
    mixes uniform exploration).  ``learn=False`` freezes the policy
    for eval; ``frozen`` returns a non-learning copy sharing the
    trained weights.

    Episode boundary detection: a fresh state (``steps == 0`` after
    a progressed one) clears the played mask and the history — the
    domain's states must count steps, which both arenas do.
    """

    #: The domain's feature schema — a class attribute a preset
    #: binds (``MetaLearnedPlayer``); ``weights_dict`` reads it for
    #: the full named table.
    FEATURES: Iterable[str] = ()

    def __init__(
        self,
        seed: int = 0,
        *,
        legal: Any,
        featurizer: Any,
        weights: Mapping | None = None,
        lr: float = _LR,
        temperature: float = 1.0,
        greedy: bool = False,
        learn: bool = True,
        epsilon: float = 0.0,
        features: Iterable[str] | None = None,
    ) -> None:
        """Bind the enumerator, the featurizer, and the sampler."""
        self._rng = random.Random(seed)
        self._legal = legal
        self._featurizer = featurizer
        if features is not None:
            self.FEATURES = tuple(features)
        self._lr = lr
        self._temp = max(temperature, 1e-6)
        self._greedy = greedy
        self._learn = learn
        self._eps = epsilon
        self._w = dict(weights or {})
        self._baseline = 0.0
        self._played: set = set()
        self._hist: Any = None
        self._ep: list = []
        self._prev_steps = -1

    def fresh_hist(self) -> Any:
        """Return the domain's empty history object (overridable)."""
        return None

    def frozen(self, seed: int = 0, *, greedy: bool = False) -> Any:
        """Return a non-learning copy sharing the trained weights."""
        q = type(self)(
            seed=seed,
            legal=self._legal,
            featurizer=self._featurizer,
            weights=self.weights_dict(),
            temperature=self._temp,
            greedy=greedy,
            learn=False,
            epsilon=0.0,
        )
        q.FEATURES = self.FEATURES
        return q

    def weights_dict(self) -> dict[str, float]:
        """Return the weight table — ``{name: w}`` over the schema."""
        return {n: self._w.get(n, 0.0) for n in self.FEATURES}

    def _observe(self, state: Any) -> None:
        """Detect the episode boundary — a fresh board resets state."""
        if self._hist is None or (
            state.steps == 0 and self._prev_steps > 0
        ):
            self._played.clear()
            self._hist = self.fresh_hist()
        self._prev_steps = state.steps

    def _scored_unplayed(self, state: Any) -> tuple[list, list, list]:
        """Return ``(acts, feats, exps)`` over the unplayed legal set."""
        self._observe(state)
        acts = [
            a
            for a in self._legal(state)
            if _action_key(a) not in self._played
        ]
        feats = [self._featurizer(state, a, self._hist) for a in acts]
        logits = [_dot(self._w, f) / self._temp for f in feats]
        top = max(logits) if logits else 0.0
        exps = [math.exp(x - top) for x in logits]
        return acts, feats, exps

    def __call__(self, state: Any) -> Any | None:
        """Score the unplayed legal moves, sample one, remember it."""
        acts, feats, exps = self._scored_unplayed(state)
        if not acts:
            return None
        if self._greedy:
            i = exps.index(max(exps))
        elif self._eps and self._rng.random() < self._eps:
            i = self._rng.randrange(len(acts))
        else:
            i = self._sample(exps)
        if self._learn:
            self._ep.append(_grad_row(feats, exps, i))
        chosen = acts[i]
        self._played.add(_action_key(chosen))
        self._record(chosen)
        return chosen

    def _record(self, action: Any) -> None:
        """Post-play bookkeeping — the domain's history update hook."""
        rec = getattr(self._hist, "record", None)
        if rec is not None:
            rec(getattr(action, "op", "?"))

    def imitate(self, state: Any, action: Any) -> bool:
        """Behavior-cloning step toward a demonstrated *action*.

        Computes the same softmax over the unplayed legal set the
        sampler uses, then moves the weights toward the expert's
        features: ``w += lr·(φ_expert - E_π[φ])`` — a cross-entropy
        update with no baseline and no return, so demonstrations
        count identically regardless of what the episode later pays.
        Returns False when the expert move is not in the live legal
        set (the policy's played mask or the board's legality may
        diverge from the trace's) — the caller decides whether that
        step still replays on the board.  No-op when ``learn=False``.
        """
        acts, feats, exps = self._scored_unplayed(state)
        key = _action_key(action)
        hit = next(
            (i for i, a in enumerate(acts) if _action_key(a) == key),
            None,
        )
        if hit is None:
            return False
        if self._learn:
            _phi, ephi = _grad_row(feats, exps, hit)
            for k, v in feats[hit].items():
                self._w[k] = self._w.get(k, 0.0) + self._lr * v
            for k, v in ephi.items():
                self._w[k] = self._w.get(k, 0.0) - self._lr * v
        self._played.add(key)
        self._record(action)
        return True

    def _sample(self, exps: list[float]) -> int:
        """Draw an index proportionally to the unnormalized weights."""
        r = self._rng.random() * sum(exps)
        acc = 0.0
        for i, e in enumerate(exps):
            acc += e
            if r <= acc:
                return i
        return len(exps) - 1

    def finish_episode(
        self, total: float, rewards: Iterable[float] | None = None
    ) -> None:
        """REINFORCE — reward-to-go per step, not just the total.

        Per step ``t``: ``G_t = Σ_{k≥t} r_k`` — the remaining episode
        reward, so a move gets credit for the terminal payout it
        armed.  *rewards* comes from the trajectory's per-step
        reports; ``None`` falls back to the flat return.  The EMA
        baseline tracks the raw total; advantages are clipped.  No-op
        when ``learn=False``.
        """
        adv = total - self._baseline
        self._baseline += _BASELINE_EMA * adv
        if not (self._learn and self._ep):
            self._ep = []
            return
        rs = list(rewards) if rewards is not None else None
        n = len(self._ep)
        grad: dict[str, float] = {}
        for t, (phi, ephi) in enumerate(self._ep):
            g_t = (
                sum(rs[t:]) if rs is not None and t < len(rs) else total
            )
            a = max(-_CLIP, min(_CLIP, g_t - self._baseline))
            for k, v in phi.items():
                grad[k] = grad.get(k, 0.0) + self._lr * a / n * v
            for k, v in ephi.items():
                grad[k] = grad.get(k, 0.0) - self._lr * a / n * v
        for k, g in grad.items():
            self._w[k] = self._w.get(k, 0.0) + g
        self._ep = []


class LearnedPlayer(LinearPolicy):
    """The one learned player — a policy bound to a :class:`DomainSpec`.

    ``SPEC`` is the class-level default spec; a shipped preset is a
    subclass that sets it (``MetaLearnedPlayer`` / ``OpenLearnedPlayer``
    — a ``SPEC`` each, no player code), and ``spec=`` binds one
    explicitly.  The spec supplies the enumerator, the composed
    featurizer, the feature schema, the weight prior and the learning
    rate; *weights* / *legal* / *featurizer* / *lr* override it.
    ``rank_score`` is the spec's candidate ranker (the open game's
    page ordering), or ``None`` when the domain ships none.
    """

    SPEC: DomainSpec | None = None

    def __init__(
        self,
        seed: int = 0,
        *,
        spec: DomainSpec | None = None,
        weights: Mapping | None = None,
        lr: float | None = None,
        temperature: float = 1.0,
        greedy: bool = False,
        learn: bool = True,
        legal: Any = None,
        featurizer: Any = None,
        epsilon: float = 0.0,
    ) -> None:
        """Bind the spec's enumerator, featurizer, prior and schema."""
        spec = self.SPEC if spec is None else spec
        if spec is None:
            raise ValueError("LearnedPlayer requires a DomainSpec")
        self._spec = spec
        super().__init__(
            seed,
            legal=spec.legal if legal is None else legal,
            featurizer=spec.featurize
            if featurizer is None
            else featurizer,
            weights=spec.weights if weights is None else weights,
            lr=spec.lr if lr is None else lr,
            temperature=temperature,
            greedy=greedy,
            learn=learn,
            epsilon=epsilon,
            features=spec.features,
        )

    def fresh_hist(self) -> PlayHist:
        """Return the domain's empty play-rate history."""
        return self._spec.new_hist()

    def frozen(self, seed: int = 0, *, greedy: bool = False) -> Any:
        """Return a non-learning copy sharing the spec and weights."""
        return type(self)(
            seed=seed,
            spec=self._spec,
            legal=self._legal,
            featurizer=self._featurizer,
            weights=self.weights_dict(),
            temperature=self._temp,
            greedy=greedy,
            learn=False,
            epsilon=0.0,
        )

    def rank_score(self, cand: Any) -> float:
        """Score *cand* under the spec's candidate ranker.

        The open game's *candidate-only* page ordering is
        ``w · rank(cand)``: every ``evaluate`` move shares its state
        and move-kind features, so this scorer gives the ordering the
        shape block (``_cand_feats``) implies.  Parts that need the
        *pool* (``RelFeats`` — ``a:score_rank``/``a:score_rel``) or
        emit terms outside the ranker's vocabulary (``x:score:*``) do
        not figure here — the in-episode ordering uses them; the
        page-aware AP probe in ``lab/richer_features.py`` prices them.
        """
        if self._spec.rank is None:
            raise ValueError("this domain ships no candidate ranker")
        return _dot(self._w, self._spec.rank(cand))
