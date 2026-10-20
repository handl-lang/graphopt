"""The policy's own posterior as open-game feature columns.

``docs/results_stop_policy.md`` rounds 3-5 closed on one missing
variable: no feature can price the *pending* pool — "will this
candidate still pay" — so the stop decision has no observable to
learn from.  Round 3 named the closest thing the model itself
carries: its own softmax distribution over the pending candidates is
a weak value estimate, but "a ``RelFeats`` part cannot see the
weights — only ``state``, ``action`` and the partially-built row —
so ``x:stop:rank1_pred`` shipped the linear-score proxy".

This module lifts that barrier.  The injected seam
:func:`set_self_scores` binds a scorer ``(state, action, hist) ->
logit`` — typically the *current policy's* own ``w·φ`` over the
``evaluate`` rows — and the ``open:self`` part turns the logits it
returns over the pending set into features:

* on the ``stop`` row — the pool-priced quantities the quit decision
  needs: the model's best pending logit (``x:stop:self_best``), the
  top-3 mass (``x:stop:self_topk``), the rank-1 posterior share
  (``x:stop:self_p1``), the posterior-weighted logit
  (``x:stop:self_ev`` — expected value of the next pick under the
  model's own distribution), the posterior-weighted injected *value*
  (``x:stop:self_pay`` — expected yield per eval when the
  ``open:next-val`` table is fed), and the share of pending picks
  scored above uniform (``x:stop:self_above`` — the count-above-
  threshold column);
* on each ``evaluate`` row — that candidate's own posterior share
  (``x:self:post``) and its logit margin over the rest of the field
  (``x:self:gap``): the learned-units analogue of the heuristic
  ``a:score_rank``/``a:score_rel`` pair.

Two honest caveats the part cannot hide:

* the features are *self-referential* — the scorer reads the same
  weights the softmax trains, so the columns move as the policy
  learns (early training emits the uniform posterior, which is the
  honest "the model has no opinion yet");
* a nested featurize call cannot re-enter the part — the posterior
  it describes is over the *base* columns (the busy flag makes the
  recursive ``evaluate``-row featurizes emit nothing, so the scorer's
  logits are ``w·φ`` over the non-self schema).

Unfed (``set_self_scores(None)`` — the default), every column is
zero: a registered-but-unfed part is inert, not wrong — the
``open:next-val`` convention.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import fmean
from typing import Any

from gamekit.players import register_part

from mlopt_discovery.meta_arena import Action
from mlopt_discovery.open_game import _value as _cand_value

__all__ = [
    "SELF_FEATURES",
    "self_featurizer_part",
    "set_self_scores",
]

#: The feature names the ``open:self`` part emits — the stop-row
#: block is pool-priced (the model's own posterior over pending
#: ``evaluate`` moves), the eval-row block is the same posterior in
#: per-candidate units.  Every column is emitted on the rows it
#: describes — zero-valued when the seam is unfed or the pool is
#: drained — so a schema carrying the part stays rectangular.
SELF_FEATURES: tuple[str, ...] = (
    "x:stop:self_best",
    "x:stop:self_topk",
    "x:stop:self_p1",
    "x:stop:self_ev",
    "x:stop:self_pay",
    "x:stop:self_above",
    "x:self:post",
    "x:self:gap",
)

_STOP_ZERO: dict[str, float] = {
    f: 0.0 for f in SELF_FEATURES if f.startswith("x:stop:")
}
_EVAL_ZERO: dict[str, float] = {
    f: 0.0 for f in SELF_FEATURES if f.startswith("x:self:")
}


class _SelfState:
    """Module state for the injected scorer — ``set_value_preds``' twin.

    *score* is the injected ``(state, action, hist) -> float``
    callable or ``None`` (unfed → zero columns).  *busy* is the
    re-entrancy guard: while the part asks the scorer for the pending
    logits, the scorer's own ``evaluate``-row featurizes run the parts
    tuple again — the flag makes the nested ``self`` part emit
    nothing, so the posterior is over the base schema and no
    recursion is possible.
    """

    score: Callable[[Any, Any, Any], float] | None = None
    busy: bool = False


_SELF = _SelfState()


def set_self_scores(
    fn: Callable[[Any, Any, Any], float] | None,
) -> None:
    """Inject (or clear) the policy-posterior scorer.

    *fn* is a ``(state, action, hist) -> float`` callable returning
    the logit the model assigns *action* in *state* under *hist* —
    e.g. ``lambda s, a, h: _dot(player._w, spec.featurize(s, a, h))``
    binds the policy's own weights, live, so the columns track the
    training policy's posterior as it sharpens.  ``None`` clears the
    seam: the part falls back to zero-valued columns.
    """
    _SELF.score = fn
    _SELF.busy = False


def _pending_logits(state: Any, hist: Any) -> list[tuple[int, float]]:
    """Score every pending ``evaluate`` move under the injected scorer.

    Returns ``[(pool_index, logit)]``; empty when the seam is unfed
    or the pool is drained.  The busy flag guards the whole batch so
    the scorer's featurizes see this part as inert.
    """
    src = _SELF.score
    if src is None or not state.pending:
        return []
    _SELF.busy = True
    try:
        return [
            (
                i,
                float(
                    src(
                        state,
                        Action(
                            "evaluate",
                            {"i": i, "cand": state.pool[i]},
                        ),
                        hist,
                    )
                ),
            )
            for i in state.pending
        ]
    finally:
        _SELF.busy = False


def _stop_cols(state: Any, logits: list[tuple[int, float]]) -> dict:
    """Emit the pool-priced ``x:stop:self_*`` block over the logits."""
    if not logits:
        return dict(_STOP_ZERO)
    vals = [s for _i, s in logits]
    top = max(vals)
    exps = [math.exp(s - top) for s in vals]
    z = sum(exps)
    k = len(logits)
    pay = sum(
        e / z * _cand_value(state.pool[i])
        for e, (i, _s) in zip(exps, logits, strict=True)
    )
    return {
        "x:stop:self_best": top / 16.0,
        "x:stop:self_topk": fmean(sorted(vals, reverse=True)[:3])
        / 16.0,
        "x:stop:self_p1": 1.0 / z,
        "x:stop:self_ev": sum(
            e / z * s / 16.0 for e, s in zip(exps, vals, strict=True)
        ),
        "x:stop:self_pay": pay,
        "x:stop:self_above": sum(e / z > 1.0 / k for e in exps) / k,
    }


def _eval_cols(
    action: Any, logits: list[tuple[int, float]]
) -> dict[str, float]:
    """Emit the per-candidate ``x:self:*`` block — the row's share."""
    if not logits:
        return dict(_EVAL_ZERO)
    i = action.params.get("i")
    vals = [s for _j, s in logits]
    top = max(vals)
    exps = [math.exp(s - top) for s in vals]
    z = sum(exps)
    pos = next((k for k, (j, _s) in enumerate(logits) if j == i), None)
    if pos is None:
        return dict(_EVAL_ZERO)
    others = [s for k, s in enumerate(vals) if k != pos]
    return {
        "x:self:post": exps[pos] / z,
        "x:self:gap": (vals[pos] - fmean(others)) / 16.0
        if others
        else vals[pos] / 16.0,
    }


def self_featurizer_part(
    state: Any, action: Any, hist: Any, _row: dict
) -> dict[str, float]:
    """Emit the ``open:self`` columns — a raw 4-arg featurizer part.

    Unlike ``RelFeats`` (which drops *hist*), this part forwards the
    episode's play history to the injected scorer, so the posterior
    the columns describe is the one the policy is actually sampling
    under this step.  ``{}`` on the busy path (the scorer's own
    featurizes) and on non-``evaluate``/``stop`` ops; explicit zeros
    unfed.
    """
    if _SELF.busy:
        return {}
    if action.op == "stop":
        return _stop_cols(state, _pending_logits(state, hist))
    if action.op == "evaluate":
        return _eval_cols(action, _pending_logits(state, hist))
    return {}


#: Registry entry — the part is nameable by spec records
#: (``DomainSpec.to_record`` resolves it through :data:`PARTS`).
register_part("open:self", lambda: self_featurizer_part)
