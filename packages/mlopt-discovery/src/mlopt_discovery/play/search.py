"""The ``search`` domain — ``SearchEnv`` under the ``Board`` contract.

One rule per step under a horizon/patience bound, adapted to the
``mlopt_discovery.engine.Board`` protocol; the learned arm's
vocabulary is the fired-rule identity bucket.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from gamekit import engine
from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    StateBroadcast,
    register_legal,
    register_part,
)
from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import DEFAULT as DEFAULT_SEARCH_RULES
from mlopt_core.search_env import SearchEnv
from mlopt_core.traversal import expand as _term_expand

from .. import lawdata
from ..meta_player import _bucket

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SearchState:
    """The search board's observable: rule names, cursor, terminator."""

    actions: tuple[str, ...]
    ops: frozenset
    steps: int
    done: bool
    cost: float
    baseline: float


class SearchBoard:
    """``SearchEnv`` under the :class:`engine.Board` contract.

    Actions are rule names; ``observe`` returns a
    :class:`_SearchState`; ``step`` fires the rule once and reports
    the env's normalized cost improvement as reward.  Episode ends
    at the env's horizon/patience, or when the player drains.
    """

    def __init__(
        self,
        term: Any,
        rules: Any,
        *,
        horizon: int = 12,
        patience: int | None = None,
    ) -> None:
        """Bind the program, rules, and episode bounds.

        *patience* defaults to the ruleset size — a playbook that
        tries every rule once must not stall out mid-way; the
        episode still ends at *horizon* or when the player drains.
        """
        rules = tuple(rules)
        self.env = SearchEnv(
            term,
            rules,
            cost_fn=count_cost,
            horizon=horizon,
            patience=len(rules) if patience is None else patience,
        )
        self._h = horizon
        self._p = patience
        self._steps = 0
        self._done = True
        self._baseline = float("inf")
        self._ops = frozenset(t.op for t in _ops_walk(term))

    def observe(self) -> _SearchState:
        """Return the board view — reset lazily on first look."""
        if self._done and self._steps == 0:
            self.env.reset()
            self._done = False
            self._baseline = self.env.cost
        return _SearchState(
            actions=self.env.action_names,
            ops=self._ops,
            steps=self._steps,
            done=self._done,
            cost=self.env.cost,
            baseline=self._baseline,
        )

    def step(self, action: Any) -> tuple[_SearchState, engine.Report]:
        """Fire one rule; report the cost improvement as reward."""
        self.observe()  # ensure reset
        name = (
            action if isinstance(action, str) else action.params["rule"]
        )
        res = self.env.step(name)
        self._steps += 1
        self._done = res.done
        return self.observe(), engine.Report(
            action, reward=res.reward, terminal=res.done, cost=res.cost
        )


#: The search case pool — named so a domain spec's ``corpus.kinds``
#: can select paying-fold boards as data.  Built lazily: the Vars
#: are shared leaves, spelled once.
_SEARCH_POOL: tuple | None = None


def _search_pool() -> tuple:
    """Return the named search-case pool (built once)."""
    global _SEARCH_POOL
    if _SEARCH_POOL is None:
        x = Var("x", TensorType((4, 4)))
        y = Var("y", TensorType((4, 4)))
        _SEARCH_POOL = (
            ("silu", Op.make("mul", x, Op.make("sigmoid", x))),
            (
                "mixed",
                Op.make(
                    "add",
                    Op.make("mul", x, y),
                    Op.make("matmul", x, y),
                ),
            ),
            ("square", Op.make("mul", x, x)),
            (
                "divabs",
                Op.make(
                    "div",
                    x,
                    Op.make("add", Op.make("abs", x), Const(1)),
                ),
            ),
            ("addid", Op.make("add", x, Const(0))),
            ("mulid", Op.make("mul", x, Const(1))),
            ("negneg", Op.make("neg", Op.make("neg", x))),
        )
    return _SEARCH_POOL


def _search_cases(seed: int, n: int, *, kinds: Any = None) -> list:
    """Rule-fire boards — half the pool carries a paying fold.

    ``add(x,0)`` / ``mul(x,1)`` / ``neg(neg(x))`` collapse under
    the identity laws (cost→0); the silu spelling folds to the
    fused op; the rest are honest flat boards — the corpus the
    stoppable-vs-search signal lives on.  *kinds* (domain-spec
    seam) selects pool names; ``None`` keeps the full pool.
    """
    rng = random.Random(seed)
    pool = list(_search_pool())
    if kinds is not None:
        known = {name for name, _ in pool}
        unknown = sorted(set(kinds) - known)
        if unknown:
            raise ValueError(
                f"corpus.kinds: unknown search pool {unknown!r} "
                f"(known: {sorted(known)!r})"
            )
        wanted = set(kinds)
        pool = [p for p in pool if p[0] in wanted]
    out = []
    for i in range(n):
        name, term = pool[rng.randrange(len(pool))]
        out.append((f"search{i}:{name}", term))
    return out


def _search_board(
    case: Any,
    rules: Any = None,
    *,
    horizon: int = 12,
    patience: int | None = None,
) -> SearchBoard:
    """Board a search case under the rule-fire game.

    *rules*/*horizon*/*patience* are the domain-spec seams — a spec
    seats a ruleset profile and episode bounds as data.
    """
    return SearchBoard(
        case[1],
        DEFAULT_SEARCH_RULES if rules is None else rules,
        horizon=horizon,
        patience=patience,
    )


def _ops_walk(term: Any):
    """Yield every Op node reachable in *term* (args-first)."""
    for _p, n in _term_expand(term):
        if isinstance(n, Op):
            yield n


def _search_legal(state: _SearchState) -> tuple:
    """Every rule name is a legal move until the env says done."""
    return state.actions


def _search_state(state: _SearchState) -> dict[str, float]:
    """Return the ``st:*`` broadcast block — progress and improvement."""
    return {
        "st:steps": state.steps / 8.0,
        "st:improve": max(
            0.0, 1.0 - state.cost / max(state.baseline, 1e-9)
        ),
    }


def _search_act(
    action: Any, _row: dict[str, float]
) -> dict[str, float]:
    """Return the fired rule's identity bucket — the only action feature."""
    return {f"a:r:{_bucket(action, 'rule'):02d}": 1.0}


SEARCH_FEATURES: tuple = (
    "bias",
    "st:steps",
    "st:improve",
    *(
        f"a:r:{b:02d}"
        for b in range(lawdata.META_ARENA_HASH_BUCKETS["rule"])
    ),
)

#: The search spec's parts in the record vocabulary — every shipped
#: spec reifies to data (:meth:`DomainSpec.to_record`).
register_part("search:state", lambda: StateBroadcast(_search_state))
register_part("search:action", lambda: ActionFeats(_search_act))
register_legal("search", _search_legal)


#: The search adapter's spec — a rule-fire board has no move-kind
#: one-hot and no play-rate history, so its parts are just the
#: intercept, the ``st:*`` broadcast and the rule bucket.
SEARCH_SPEC = DomainSpec.build(
    features=SEARCH_FEATURES,
    legal=_search_legal,
    parts=(
        Bias(),
        StateBroadcast(_search_state),
        ActionFeats(_search_act),
    ),
)
