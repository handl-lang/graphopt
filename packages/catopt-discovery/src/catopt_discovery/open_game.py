"""The open game — law-shape evaluation as a playable domain.

ADR 0005's missing driver: the enumerated candidate stream
(:func:`catopt_discovery.enumerate.enumerate_candidates`) is a
*pool of moves*, and which ones to spend oracle evaluations on is
the policy question — evaluating everything is affordable at
depth <=2 (~0.025 s/candidate under the bounded oracle) but does
not scale to deeper pools or corpus-site measurement.  This domain
makes that ordering playable: an episode is a page of candidates,
a move evaluates one, the reward is the measured verdict — a
learned player that ranks paying shapes first directly raises the
game's yield-per-eval.

Board:

* ``evaluate`` — run :func:`catopt_discovery.proposal.evaluate` on
  the bound pool member and record the verdict.  Reward is the
  verdict tier from :data:`OPEN_GAME_REWARD`: a useful-new law
  pays, a true-but-shallow one pays less, a known/false one just
  costs the evaluation.
* ``stop`` — end the episode early (a player that has drained the
  paying members should take it).

The learned arm (:class:`OpenLearnedPlayer`) scores ``evaluate``
moves by the candidate's *shape* features — rewriting form, op
histogram, depth, metavariable count, the cost-delta heuristic —
plus episode-history rates.  That is exactly the prior a proposer
needs: "shapes of this form have been paying."

Cases (:func:`gen_cases`) are pages into the deterministic
enumerated stream — case *i* sees a different slice of the same
grammar, so the policy learns a transferable ordering, not one
pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any

from catopt_core.ir import Op, Var

from catopt_discovery import proposal as lp
from catopt_discovery.enumerate import enumerate_candidates
from catopt_discovery.meta_arena import Action
from catopt_discovery.meta_player import _Hist
from catopt_discovery.players import LinearPolicy

__all__ = [
    "OPEN_GAME_FEATURES",
    "OPEN_GAME_REWARD",
    "OpenArena",
    "OpenLearnedPlayer",
    "OpenState",
    "featurize",
    "gen_cases",
    "legal_actions",
]

# ---------------------------------------------------------------------------
#  The table — reward tiers and the feature schema (board constants)
# ---------------------------------------------------------------------------

#: Verdict tiers an evaluated candidate lands in, most valuable
#: first — the reward composition is ``step`` (the eval cost) plus
#: the tier the outcome earns.
OPEN_GAME_REWARD: dict[str, float] = {
    "step": -0.02,
    "known": 0.0,
    "false": 0.0,
    "true": 0.2,
    "useful": 1.0,
}

#: The enumerator's rewriting forms — the ``a:form:*`` one-hot
#: schema (labels carry ``"<form>:<n>"``).
FORMS: tuple[str, ...] = (
    "comm",
    "assoc",
    "rotate",
    "rekey",
    "factor-left",
    "factor-right",
    "unary-hom",
    "pull-left",
    "pull-right",
    "unary-swap",
    "involution",
    "unary-inverse",
    "unary-absorb",
    "self-op",
    "idempotent",
    "lit-self",
    "const-op",
    "id-right",
    "id-left",
    "annihilate-right",
    "annihilate-left",
    "cancel-self",
    "cancel-unary-right",
    "cancel-unary-left",
    "hom-cross",
    "bin-unary-right",
    "bin-unary-left",
    "unary-of-lit",
)

#: The feature schema — ``st:*`` state broadcasts, ``h:*``
#: within-episode play rates, ``op:*`` move-kind one-hots, and
#: ``a:*`` the candidate's own shape (rewriting-form one-hot, op
#: count, depth, meta count, heuristic score).
OPEN_GAME_FEATURES: tuple[str, ...] = (
    "bias",
    "st:pending_frac",
    "st:seen_frac",
    "st:useful_rate",
    "st:true_rate",
    "h:evaluate",
    "h:stop",
    "op:evaluate",
    "op:stop",
    "a:depth",
    "a:metas",
    "a:ops",
    "a:score",
    *(f"a:form:{f}" for f in FORMS),
)


def _term_depth(t: Any) -> int:
    if isinstance(t, Op):
        return 1 + max((_term_depth(a) for a in t.args), default=0)
    return 0


def _walk(t: Any, metas: set) -> int:
    """Count op nodes and collect metavar names in *t*."""
    if isinstance(t, Op):
        return 1 + sum(_walk(a, metas) for a in t.args)
    if isinstance(t, Var):
        metas.add(t.name)
    return 0


def _cand_feats(cand: lp.Candidate) -> dict[str, float]:
    """Shape features of one candidate — what the prior sees."""
    metas: set = set()
    n_ops = _walk(cand.lhs, metas) + _walk(cand.rhs, metas)
    feats: dict[str, float] = {
        "a:depth": float(
            max(_term_depth(cand.lhs), _term_depth(cand.rhs))
        ),
        "a:metas": float(len(metas)),
        "a:ops": float(n_ops),
        "a:score": lp._heuristic_score(cand) / 16.0,
    }
    form = cand.label.split(":", 1)[0]
    key = f"a:form:{form}"
    if key in OPEN_GAME_FEATURES:
        feats[key] = 1.0
    return feats


def _verdict(o: lp.Outcome) -> str:
    """Classify an outcome into its reward tier."""
    if not o.new:
        return "known"
    if not o.truth:
        return "false"
    return "useful" if o.useful else "true"


# ---------------------------------------------------------------------------
#  The board
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenState:
    """The read-only snapshot an open-game player decides on.

    ``pool`` carries the candidates themselves (they are data — the
    featurizer reads their shape without reaching the board).
    """

    pool: tuple
    pending: tuple[int, ...]
    verdicts: tuple[tuple[int, str], ...]
    steps: int
    done: bool
    n_seen: int = 0
    n_true: int = 0
    n_useful: int = 0
    pool_size: int = 0


class OpenArena:
    """One candidate pool, one evaluation-at-a-time episode.

    *case* is ``{"ops": ops, "offset": i, "page": n}`` — the pool is
    the page of :func:`enumerate_candidates` starting at *offset*;
    an empty pool is a done state (the whole grammar exhausted).
    """

    def __init__(self, case: dict) -> None:
        """Materialize the candidate page for *case*."""
        offset = case.get("offset", 0)
        self.pool = list(
            islice(
                enumerate_candidates(case.get("ops")),
                offset,
                offset + case.get("page", 48),
            )
        )
        self.verdicts: dict[int, str] = {}
        self.steps = 0
        self.done = not self.pool
        self._n_true = 0
        self._n_useful = 0

    def observe(self) -> OpenState:
        """Snapshot pending indices, verdicts, and the terminator."""
        return OpenState(
            pool=tuple(self.pool),
            pending=tuple(
                i
                for i in range(len(self.pool))
                if i not in self.verdicts
            ),
            verdicts=tuple(sorted(self.verdicts.items())),
            steps=self.steps,
            done=self.done,
            n_seen=len(self.verdicts),
            n_true=self._n_true,
            n_useful=self._n_useful,
            pool_size=len(self.pool),
        )

    def step(self, action: Any) -> tuple[OpenState, Any]:
        """Evaluate one candidate (``evaluate``) or end (``stop``)."""
        from catopt_discovery.engine import Report

        if self.done:
            raise RuntimeError("episode finished")
        if action.op == "stop":
            self.done = True
            return self.observe(), Report(
                action=action, reward=0.0, terminal=True
            )
        i = action.params["i"]
        outcome = lp.evaluate([self.pool[i]])[0]
        verdict = _verdict(outcome)
        self.verdicts[i] = verdict
        self._n_true += verdict in ("true", "useful")
        self._n_useful += verdict == "useful"
        self.done = not any(
            j not in self.verdicts for j in range(len(self.pool))
        )
        reward = OPEN_GAME_REWARD["step"] + OPEN_GAME_REWARD[verdict]
        self.steps += 1
        return self.observe(), Report(
            action=action,
            reward=reward,
            terminal=self.done,
            note=f"{verdict}:{self.pool[i].label}",
            detail={
                "label": self.pool[i].label,
                "verdict": verdict,
                "relation": outcome.relation,
                "num_true": outcome.num_true,
                "base_cost": outcome.base_cost,
                "cand_cost": outcome.cand_cost,
            },
        )


def legal_actions(state: OpenState) -> tuple[Action, ...]:
    """One ``evaluate`` per pending member, plus ``stop``.

    The candidate itself rides on ``params["cand"]`` — the
    featurizer reads its shape without re-entering the board.
    """
    acts = [
        Action("evaluate", {"i": i, "cand": state.pool[i]})
        for i in state.pending
    ]
    acts.append(Action("stop", {}))
    return tuple(acts)


def featurize(
    state: OpenState, action: Action, hist: _Hist | None = None
) -> dict[str, float]:
    """Feature row for one legal move — ``st``/``h``/``op``/``a``."""
    hist = _Hist() if hist is None else hist
    n = max(state.pool_size, 1)
    feats: dict[str, float] = {
        "bias": 1.0,
        "st:pending_frac": len(state.pending) / n,
        "st:seen_frac": state.n_seen / n,
        "st:useful_rate": state.n_useful / max(state.n_seen, 1),
        "st:true_rate": state.n_true / max(state.n_seen, 1),
        "h:evaluate": hist.rate("evaluate"),
        "h:stop": hist.rate("stop"),
        f"op:{action.op}": 1.0,
    }
    if action.op == "evaluate":
        cand = action.params.get("cand")
        if cand is not None:
            feats.update(_cand_feats(cand))
    return feats


class OpenLearnedPlayer(LinearPolicy):
    """The open-game preset — enumerate featurizer and schema bound."""

    FEATURES = OPEN_GAME_FEATURES

    def __init__(
        self,
        seed: int = 0,
        *,
        weights: dict | None = None,
        lr: float = 0.05,
        temperature: float = 1.0,
        greedy: bool = False,
        learn: bool = True,
        legal: Any = None,
        featurizer: Any = None,
        epsilon: float = 0.0,
    ) -> None:
        """Bind the enumerator, featurizer and sampler."""
        super().__init__(
            seed,
            legal=legal_actions if legal is None else legal,
            featurizer=featurize if featurizer is None else featurizer,
            weights={} if weights is None else weights,
            lr=lr,
            temperature=temperature,
            greedy=greedy,
            learn=learn,
            epsilon=epsilon,
        )

    def fresh_hist(self) -> _Hist:
        """Return the per-episode action histogram."""
        return _Hist()


# ---------------------------------------------------------------------------
#  Cases — pages into the enumerated stream
# ---------------------------------------------------------------------------


def gen_cases(
    seed: int, n: int, *, ops: tuple | None = None, page: int = 48
) -> list[dict]:
    """Generate *n* cases — successive pages of the grammar.

    ``ops`` defaults to the enumerator's op set; page size bounds
    the episode horizon.  Cases are ordered pages so every case
    faces a fresh slice — the learned ordering must transfer.
    """
    return [
        {"ops": ops, "offset": i * page, "page": page} for i in range(n)
    ]
