"""The open game — law-shape evaluation as a playable domain.

ADR 0005's missing driver: the enumerated candidate stream
(:func:`mlopt_discovery.enumerate.enumerate_candidates`) is a
*pool of moves*, and which ones to spend oracle evaluations on is
the policy question — evaluating everything is affordable at
depth <=2 (~0.025 s/candidate under the bounded oracle) but does
not scale to deeper pools or corpus-site measurement.  This domain
makes that ordering playable: an episode is a page of candidates,
a move evaluates one, the reward is the measured verdict — a
learned player that ranks paying shapes first directly raises the
game's yield-per-eval.

Board:

* ``evaluate`` — run :func:`mlopt_discovery.proposal.evaluate` on
  the bound pool member and record the verdict.  Reward is the
  verdict tier from :data:`OPEN_GAME_REWARD`: a useful-new law
  pays, a true-but-shallow one pays less, a known/false one just
  costs the evaluation.  A ``useful`` verdict also pays an ordering
  credit of ``OPEN_GAME_EFFICIENCY / evals`` — the paying laws found
  per oracle eval so far — so finding a paying shape *earlier* in
  the page is worth strictly more than finding it late.
* ``stop`` — end the episode early (a player that has drained the
  paying members should take it).

The learned arm (:class:`OpenLearnedPlayer` — the shared
:class:`~mlopt_discovery.players.LearnedPlayer` bound to
:data:`OPEN_SPEC`) scores ``evaluate`` moves by the candidate's
*shape* features — rewriting form, op histogram, depth, metavariable
count, the cost-delta heuristic — plus episode-history rates.  That
is exactly the prior a proposer needs: "shapes of this form have been
paying."  The arm ships
*warm*: it defaults to the baked prior
:data:`mlopt_discovery.lawdata.OPEN_GAME_PLAYER_WEIGHTS`, trained
over even pages by ``lab/train/train_open_prior.py`` and measured on the
held-out odd pages (``tests/test_discovery_open_prior.py``); pass
``weights={}`` for the cold start.

Cases (:func:`gen_cases`) are pages into the deterministic
enumerated stream — case *i* sees a different slice of the same
grammar, so the policy learns a transferable ordering, not one
pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from statistics import fmean
from typing import Any

from mlopt_core.ir import Op, Var

from mlopt_discovery import lawdata
from mlopt_discovery import proposal as lp
from mlopt_discovery.enumerate import enumerate_candidates
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    LearnedPlayer,
    OpOneHot,
    PlayRates,
    RelFeats,
    StateBroadcast,
    register_legal,
    register_part,
    register_rank,
)

__all__ = [
    "OPEN_EXTRA_FEATURES",
    "OPEN_EXTRA_PART_TAGS",
    "OPEN_GAME_EFFICIENCY",
    "OPEN_GAME_FEATURES",
    "OPEN_GAME_REWARD",
    "OPEN_PART_TAGS",
    "OPEN_SPEC",
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
#: the tier the outcome earns, plus the ordering credit
#: (:data:`OPEN_GAME_EFFICIENCY`) a ``useful`` verdict also pays.
#:
#: ``step`` is priced at ``-0.002`` — an order of magnitude below
#: the ~0.02 expected tier of a candidate page — so a candidate that
#: lands ``true`` (0.2) or ``useful`` (1.0) clearly out-earns the
#: eval that found it: the marginal candidate is *not* break-even,
#: and the reward is dominated by the verdict, not the eval count.
#:
#: Cheap evals alone do **not** make the game ordering-sensitive:
#: draining the whole page nets the same ``Σ tier`` whatever the
#: order.  The ordering signal is the efficiency credit below.
OPEN_GAME_REWARD: dict[str, float] = {
    "step": -0.002,
    "known": 0.0,
    "false": 0.0,
    "true": 0.2,
    "useful": 1.0,
}

#: Coefficient on the per-find ordering credit ``1 / evals``.
#:
#: Each ``useful`` verdict also pays ``OPEN_GAME_EFFICIENCY /
#: evals_spent`` — the paying laws found *per oracle eval so far*.
#: That is the real objective the game prices, and paying it *per
#: find* (not at the end) makes the reward order-sensitive for every
#: policy: the same useful law found at eval 1 pays ten times the
#: credit of one found at eval 10, so a player that ranks the paying
#: shapes first — whether it then ``stop``s or drains the page —
#: out-earns one that finds them late.  Summed over an episode it is
#: the dense form of the yield-per-eval ratio a terminal
#: ``useful / evals`` bonus would pay; the dense form is preferred
#: because it prices the *ordering*, not just the stopping.
OPEN_GAME_EFFICIENCY: float = 1.0

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
#: count, depth, meta count, heuristic score).  The trailing
#: ``a:score_*``/``x:score:*`` entries are the richer-vocabulary
#: extension (``lab/richer_features.py`` measured AP 0.210 vs the
#: incumbent schema's 0.133 — earned the schema and the bake).
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
    "a:score_rank",
    "a:score_rel",
    *(f"a:form:{f}" for f in FORMS),
    "x:score:depth",
    "x:score:ops",
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


#: Memoized heuristic score — the pool-relative parts re-read it once
#: per pending member per (state, action) featurize call, so one dict
#: lookup replaces the ``dag_cost`` walk on repeat visits.  ``Op`` is
#: ``eq=False``, so the key is effectively per-object: the memo wins
#: inside an episode (the same pool objects recur every step) and is
#: bounded by the stream's size.
_CAND_SCORE: dict[lp.Candidate, float] = {}


def _score(cand: lp.Candidate) -> float:
    """``lp._heuristic_score(cand)``, memoized — candidates recur."""
    v = _CAND_SCORE.get(cand)
    if v is None:
        v = lp._heuristic_score(cand)
        _CAND_SCORE[cand] = v
    return v


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
        "a:score": _score(cand) / 16.0,
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


def _state_feats(state: OpenState) -> dict[str, float]:
    """Return the ``st:*`` broadcast block — progress and verdict rates."""
    n = max(state.pool_size, 1)
    return {
        "st:pending_frac": len(state.pending) / n,
        "st:seen_frac": state.n_seen / n,
        "st:useful_rate": state.n_useful / max(state.n_seen, 1),
        "st:true_rate": state.n_true / max(state.n_seen, 1),
    }


def _act_feats(
    action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """Return the candidate's shape block — ``a:*`` (empty for ``stop``)."""
    if action.op != "evaluate":
        return {}
    cand = action.params.get("cand")
    return {} if cand is None else _cand_feats(cand)


# ---------------------------------------------------------------------------
#  Candidate vocabulary extensions — richer feature parts (registry data)
# ---------------------------------------------------------------------------
#
#  The shipped schema's whole candidate view is ``_cand_feats``: the
#  candidate *in isolation*.  The parts below encode what that view
#  cannot — where the candidate sits relative to the rest of the
#  *pending* pool (``a:score_rank`` / ``a:score_rel`` — "is this the
#  best option left on the board"), interactions of its heuristic
#  score with its shape (``x:score:*``), and pool-/history-level state
#  the ``st:*`` block lacked (what paying mass is still pending, how
#  many useful laws this episode has already banked).  Each is one
#  registered part — a spec record names them by tag
#  (:data:`OPEN_EXTRA_PART_TAGS`); they join ``OPEN_PARTS``/``OPEN_
#  GAME_FEATURES`` only if the richer-features measurement
#  (``lab/richer_features.py``) earns it.


def _pending_scores(state: OpenState) -> list[float]:
    """Raw heuristic scores of the still-pending pool members."""
    return [_score(state.pool[i]) for i in state.pending]


def _eval_cand(action: Action) -> Any:
    """Return the candidate an ``evaluate`` action binds, or None."""
    if action.op != "evaluate":
        return None
    return action.params.get("cand")


def _score_rank_feat(
    state: OpenState, action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """``a:score_rank`` — the candidate's rank quantile within pending.

    ``RelFeats`` (state + action): the fraction of pending members the
    candidate's heuristic score ties or beats — 1.0 for the best
    still-pending option, ~1/|pending| for the worst.  ``_cand_feats``
    cannot express it: it needs the rest of the pool.
    """
    cand = _eval_cand(action)
    if cand is None or not state.pending:
        return {}
    s = _score(cand)
    scores = _pending_scores(state)
    rank = sum(v <= s for v in scores) / len(scores)
    return {"a:score_rank": rank}


def _score_rel_feat(
    state: OpenState, action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """``a:score_rel`` — the candidate's score over the pending mean.

    Signed, in ``a:score`` units (heuristic / 16): positive marks a
    candidate above what remains on the board, negative below.
    """
    cand = _eval_cand(action)
    if cand is None or not state.pending:
        return {}
    s = _score(cand)
    mean = fmean(_pending_scores(state))
    return {"a:score_rel": (s - mean) / 16.0}


def _score_x_depth_feat(
    action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """``x:score:depth`` — ``a:score x a:depth``, rescaled to O(a:score).

    Interaction term: does the heuristic pay more for deep shapes than
    shallow ones?  Depth is /4-normalized so the term stays on the
    score's scale (one lr serves every feature).
    """
    cand = _eval_cand(action)
    if cand is None:
        return {}
    f = _cand_feats(cand)
    return {"x:score:depth": f["a:score"] * f["a:depth"] / 4.0}


def _score_x_ops_feat(
    action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """``x:score:ops`` — ``a:score x a:ops``, rescaled to O(a:score)."""
    cand = _eval_cand(action)
    if cand is None:
        return {}
    f = _cand_feats(cand)
    return {"x:score:ops": f["a:score"] * f["a:ops"] / 8.0}


def _pending_top_feat(state: OpenState) -> dict[str, float]:
    """``st:pending_top`` — the best heuristic score still pending.

    What the board still has to offer: a high value argues for
    spending another eval; near zero says the page's paying mass is
    already banked — the ``stop``-relevant half of the pool stats.
    """
    scores = _pending_scores(state)
    return {"st:pending_top": max(scores, default=0.0) / 16.0}


def _pending_mean_feat(state: OpenState) -> dict[str, float]:
    """``st:pending_mean`` — the mean heuristic score still pending."""
    scores = _pending_scores(state)
    return {"st:pending_mean": fmean(scores) / 16.0 if scores else 0.0}


def _useful_ct_feat(state: OpenState) -> dict[str, float]:
    """``st:useful_ct`` — useful finds banked this episode / pool size.

    The history-conditioned half the ``st:*`` block missed:
    ``st:useful_rate`` is a ratio (undefined at 0 evals, saturated by
    the first find); the raw count's gradient differs.
    """
    return {"st:useful_ct": state.n_useful / max(state.pool_size, 1)}


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

    def _efficiency(self) -> float:
        """Ordering credit for a useful law found at eval ``e``.

        The find is worth ``OPEN_GAME_EFFICIENCY / e`` — the useful
        laws found *per oracle eval spent so far*.  Paid the moment a
        paying law lands (not at the end), so the reward is
        order-sensitive for every policy, draining or stopping: the
        same useful law found at eval 1 pays ten times the credit of
        one found at eval 10.  Summed over an episode it is the
        dense form of the yield-per-eval objective the game prices.
        """
        return OPEN_GAME_EFFICIENCY / self.steps

    def step(self, action: Any) -> tuple[OpenState, Any]:
        """Evaluate one candidate (``evaluate``) or end (``stop``)."""
        from mlopt_discovery.engine import Report

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
        self.steps += 1
        reward = OPEN_GAME_REWARD["step"] + OPEN_GAME_REWARD[verdict]
        if verdict == "useful":
            reward += self._efficiency()
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


#: The open featurizer's composable parts, in row order: the
#: intercept, the ``st:*`` broadcast, the ``h:*`` play rates, the
#: move-kind one-hot, then the candidate's shape block plus the
#: pool-relative / interaction extensions (the measured AP winners).
OPEN_PARTS: tuple = (
    Bias(),
    StateBroadcast(_state_feats),
    PlayRates(("evaluate", "stop")),
    OpOneHot(frozenset(OPEN_GAME_FEATURES)),
    ActionFeats(_act_feats),
    RelFeats(_score_rank_feat),
    RelFeats(_score_rel_feat),
    ActionFeats(_score_x_depth_feat),
    ActionFeats(_score_x_ops_feat),
)

#: The open parts in the spec record's tag vocabulary — the names a
#: rewritable spec record carries (:meth:`DomainSpec.to_record`).
OPEN_PART_TAGS: tuple[str, ...] = (
    "bias",
    "open:state",
    "open:hist",
    "open:onehot",
    "open:action",
    "open:score-rank",
    "open:score-rel",
    "open:score-x-depth",
    "open:score-x-ops",
)

#: The richer-vocabulary features (see the extensions block above) —
#: name emitted per registered part tag.  Four of them earned the
#: schema (``a:score_rank``/``a:score_rel``/``x:score:depth``/
#: ``x:score:ops`` are in ``OPEN_GAME_FEATURES``); the ``st:*``
#: pool-stat parts measured dead weight (``lab/richer_features.py``)
#: and stay registered-but-unshipped.  ``DomainSpec.build`` for a
#: richer candidate spec dedupes names already in the schema.
OPEN_EXTRA_FEATURES: dict[str, tuple[str, ...]] = {
    "open:score-rank": ("a:score_rank",),
    "open:score-rel": ("a:score_rel",),
    "open:score-x-depth": ("x:score:depth",),
    "open:score-x-ops": ("x:score:ops",),
    "open:pending-top": ("st:pending_top",),
    "open:pending-mean": ("st:pending_mean",),
    "open:useful-ct": ("st:useful_ct",),
}

#: The extra parts' tags — the names a spec record (or the player
#: arena's vocabulary) carries for them.
OPEN_EXTRA_PART_TAGS: tuple[str, ...] = tuple(OPEN_EXTRA_FEATURES)

register_part("open:state", lambda: StateBroadcast(_state_feats))
register_part("open:hist", lambda: PlayRates(("evaluate", "stop")))
register_part(
    "open:onehot",
    lambda: OpOneHot(frozenset(OPEN_GAME_FEATURES)),
)
register_part("open:action", lambda: ActionFeats(_act_feats))
register_part("open:score-rank", lambda: RelFeats(_score_rank_feat))
register_part("open:score-rel", lambda: RelFeats(_score_rel_feat))
register_part(
    "open:score-x-depth", lambda: ActionFeats(_score_x_depth_feat)
)
register_part(
    "open:score-x-ops", lambda: ActionFeats(_score_x_ops_feat)
)
register_part(
    "open:pending-top", lambda: StateBroadcast(_pending_top_feat)
)
register_part(
    "open:pending-mean", lambda: StateBroadcast(_pending_mean_feat)
)
register_part("open:useful-ct", lambda: StateBroadcast(_useful_ct_feat))
register_legal("open", legal_actions)
register_rank("open", _cand_feats)

#: The open-game player spec — schema, parts, enumerator, prior, ranker.
OPEN_SPEC = DomainSpec.build(
    features=OPEN_GAME_FEATURES,
    legal=legal_actions,
    parts=OPEN_PARTS,
    weights=lawdata.OPEN_GAME_PLAYER_WEIGHTS,
    rank=_cand_feats,
    lr=0.05,
)

#: The composed featurizer — public for tests.
featurize = OPEN_SPEC.featurize


class OpenLearnedPlayer(LearnedPlayer):
    """The open-game preset — the open :class:`DomainSpec` bound.

    *weights* defaults to the shipped trained prior
    (:data:`lawdata.OPEN_GAME_PLAYER_WEIGHTS`) — the baked preset
    ``lab/train/train_open_prior.py`` regenerates.  Pass ``weights={}``
    explicitly for the cold start (the honest uniform baseline the
    held-out measurement compares against).  ``rank_score`` is the
    spec's candidate ranker (:func:`_cand_feats`) — the
    *candidate-only* slice of the page ordering
    :func:`mlopt_discovery.proposal.rank_pool` prices; the
    pool-relative parts (``RelFeats`` — ``a:score_rank``/``a:score_rel``)
    are outside its vocabulary, so the in-episode ordering is sharper
    than ``rank_score`` alone reports (``docs/results_richer_features.md``).
    """

    SPEC = OPEN_SPEC


# ---------------------------------------------------------------------------
#  Cases — pages into the enumerated stream
# ---------------------------------------------------------------------------


def gen_cases(
    seed: int,
    n: int,
    *,
    ops: tuple | None = None,
    page: int = 48,
    start: int = 0,
    step: int = 1,
) -> list[dict]:
    """Generate *n* cases — pages of the enumerated grammar.

    ``ops`` defaults to the enumerator's op set; page size bounds
    the episode horizon.  Cases are pages into the deterministic
    stream, so every case faces a fresh slice — the learned ordering
    must transfer.

    ``start`` is the first page index and ``step`` the page stride.
    A **held-out** split is two disjoint page sets: with ``step=2``
    the even pages (``start=0``) train and the odd pages
    (``start=1``) evaluate — disjoint, and *balanced* because the
    stream's ~15 useful candidates are front-loaded (13 of 15 in the
    first 58%), so a contiguous train/eval cut starves the eval half.
    ``seed`` is retained for the ``Domain.cases`` contract
    (``play.DOMAINS["open"].cases`` is this function) but the stream
    is fixed, so the page selection is ``start``/``step``: a seed
    cannot perturb it, and the prime-offset idiom (``seed + 7919``)
    would fall off the stream's ~1.1k-candidate end.
    """
    del seed  # the stream is deterministic — see the docstring
    return [
        {"ops": ops, "offset": (start + i * step) * page, "page": page}
        for i in range(n)
    ]
