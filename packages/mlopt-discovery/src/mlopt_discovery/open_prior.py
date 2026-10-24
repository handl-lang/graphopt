"""The shipped prior's *own* pending-pool view — the ``open:prior`` part.

``docs/results_stop_isolated.md`` (round 9) landed the isolated stop
channel (``gamekit.players.GatedPlayer`` + the ``forced_action``
hook): the quit decision can live outside the ranking softmax, and
its payload is *data* — a ``{feature: weight}`` head plus a κ.  What
it left open is whether the *evidence* that decision prices can ship
the same way: every priced column so far reads a runtime seam — the
injected value table behind ``open:next-val`` / ``open:vlabel``
(:func:`mlopt_discovery.open_game.set_value_preds`) or the live
policy weights behind ``open:self``
(:func:`mlopt_discovery.open_self.set_self_scores`).

This part removes the seam by construction.  It prices the pending
pool with the **shipped prior's own linear model** —
``_dot(OPEN_SPEC.weights, _cand_feats(c))``, the same ``rank_score``
the baked player orders candidates by.  Nothing is injected at
decode: the columns are computable from shipped tables alone, so a
spec record carrying ``open:prior`` reifies and rebuilds with the
decision quantities intact — the first priced stop row the loop has
measured that a weight-table-plus-constants artifact can carry
verbatim (``w_stop = {"x:stop:prior_best": 1.0}, κ = const`` is a
one-line payload, and κ = 0 on ``prior_margin`` is the prior's own
indifference point).

On the ``stop`` row (``RelFeats`` — every column zero on a drained
pool):

* ``x:stop:prior_best`` — ``max_pending v(c)`` where
  ``v(c) = w·_cand_feats(c)`` is the prior's candidate ranker score:
  the next eval's promise in the prior's own units — the
  ``x:stop:next_value``/``x:stop:vl_best`` quantity with no seam.
* ``x:stop:prior_top3`` — the mean of the top-3 pending ``v``: the
  paying *mass*, not just the peak.
* ``x:stop:prior_pool`` — ``Σ_pending v``: expected remaining yield
  in prior-score units — the ``vl_pool`` analogue.
* ``x:stop:prior_frac`` — ``prior_pool / Σ_page v``: the share of the
  page's starting prior-score mass still pending — the value-drain
  progress counter (0.0 when the page's total is non-positive).
* ``x:stop:prior_margin`` — the prior's *own* softmax margin:
  ``max_pending`` eval-row logit − the stop-row logit under the
  shipped weights.  The shared ``bias``/``st:*``/``h:*`` terms
  cancel, so this is
  ``w[op:evaluate] + max_pending w·(a-block + x-block) − w[op:stop]``
  — κ = 0 is exactly "stop when the prior's argmax would already
  pick stop"; κ > 0 is a hysteresis — quit unless the best eval
  beats stopping by a margin.
* ``x:stop:prior_p1`` — the prior's softmax share on its best
  pending eval vs the rest of the field (``e^lmax / (Σ e^l +
  e^l_stop)``): its own "keep going" conviction.

Registered as ``open:prior`` — unshipped, outside
``OPEN_EXTRA_FEATURES``/``OPEN_GAME_FEATURES`` until the measurement
(``bench/drivers/policy/stop_gate.py``) earns it.  Unlike the seam
parts there is no unfed state to fail loudly on: the part reads only
:attr:`DomainSpec.weights` of :data:`OPEN_SPEC` — the shipped table
itself — so a registered-but-unshipped schema prices the prior's
view honestly, always.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from gamekit.players import RelFeats, register_part, _dot

from mlopt_discovery.meta_arena import Action
from mlopt_discovery import open_game as og

__all__ = [
    "PRIOR_FEATURES",
    "prior_feats",
]

#: The feature names the ``open:prior`` part emits — all on the
#: ``stop`` row (the quit decision's own row; the eval rows carry
#: nothing the incumbent schema doesn't already emit).
PRIOR_FEATURES: tuple[str, ...] = (
    "x:stop:prior_best",
    "x:stop:prior_top3",
    "x:stop:prior_pool",
    "x:stop:prior_frac",
    "x:stop:prior_margin",
    "x:stop:prior_p1",
)

_ZERO: dict[str, float] = {f: 0.0 for f in PRIOR_FEATURES}


def _eval_block(state: Any, i: int) -> dict[str, float]:
    """The prior-visible ``evaluate`` block for pending member *i*.

    The candidate slice of the row the shipped prior scores:
    ``_cand_feats`` plus the pool-relative and interaction columns
    (``a:score_rank``/``a:score_rel``/``x:score:*``).  The shared
    ``bias``/``st:*``/``h:*`` terms are excluded — they cancel in
    every margin and softmax ratio the columns price.
    """
    cand = state.pool[i]
    act = Action("evaluate", {"i": i, "cand": cand})
    row = dict(og._cand_feats(cand))
    row.update(og._score_rank_feat(state, act, {}))
    row.update(og._score_rel_feat(state, act, {}))
    row.update(og._score_x_depth_feat(act, {}))
    row.update(og._score_x_ops_feat(act, {}))
    return row


def prior_feats(
    state: Any, action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """Emit the ``open:prior`` columns — a ``RelFeats`` function.

    ``stop`` rows get the pending pool priced by the shipped prior's
    own weights; every other row emits ``{}`` (the incumbent schema
    already carries the eval-row view).  A drained pool emits the
    explicit-zero block — "nothing left the prior would pay for".
    """
    if action.op != "stop":
        return {}
    w = dict(og.OPEN_SPEC.weights or {})
    if not w or not state.pending:
        return dict(_ZERO)
    cand_v = [
        _dot(w, og._cand_feats(state.pool[i])) for i in state.pending
    ]
    best = max(cand_v)
    pool = sum(cand_v)
    init = sum(_dot(w, og._cand_feats(c)) for c in state.pool)
    # The prior's own logits: shared row terms cancel, so the eval
    # logit is w[op:evaluate] + the candidate block and the stop
    # logit is w[op:stop].
    w_ev = w.get("op:evaluate", 0.0)
    l_stop = w.get("op:stop", 0.0)
    logits = [
        w_ev + _dot(w, _eval_block(state, i)) for i in state.pending
    ]
    top = max(logits)
    denom = sum(math.exp(l - top) for l in logits) + math.exp(
        l_stop - top
    )
    return {
        "x:stop:prior_best": best,
        "x:stop:prior_top3": fmean(sorted(cand_v, reverse=True)[:3]),
        "x:stop:prior_pool": pool,
        "x:stop:prior_frac": pool / init if init > 1e-9 else 0.0,
        "x:stop:prior_margin": top - l_stop,
        "x:stop:prior_p1": 1.0 / denom,
    }


#: Registry entry — the part is nameable by spec records
#: (:meth:`DomainSpec.to_record` resolves it through :data:`PARTS`).
register_part("open:prior", lambda: RelFeats(prior_feats))
