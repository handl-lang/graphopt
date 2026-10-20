"""The *expected-yield* stop columns — the candidate-value-label part.

``docs/results_stop_policy.md``'s five rounds and
``docs/results_stop_advantage.md``'s sixth closed on the same missing
variable from two directions: the stop row needs a *priced* answer to
"what does the pending pool still pay", and every feature family so
far has delivered an unpriced proxy — the heuristic's view
(``open:next-value``, candidate AP ~0.045), the raw label-probability
injection (``open:next-val`` — max/Σ/fraction of the injected
sigmoid, in *probability* units), or the policy's own posterior
(``open:self``).  The quantities the decision actually prices are
one step further: the injected per-candidate value ``v(c)`` is an
*expected payoff in reward units* (the driver feeds
``p(label)·payoff`` — e.g. ``p_useful·1.0``, or the two-net
``p_useful·1.0 + max(p_pays - p_useful, 0)·0.2`` expected-tier
composite ``bench/drivers/policy/stop_valuelabel.py`` builds), and the stop
row wants that expectation *net of the evals it would cost*:

* ``x:stop:vl_best`` — ``max_pending v(c)``: the expected payoff of
  the single best remaining eval — the margin the next spend clears
  or misses (vs ``x:stop:next_value``, which is the max of a
  probability, not a priced payoff).
* ``x:stop:vl_pool`` — ``Σ_pending v(c)``: expected remaining yield —
  the quantity "stop when nothing pays" actually wants.
* ``x:stop:vl_net`` — ``Σ_pending (v(c) - |step|)``: the expected
  reward of *draining the rest* — the drain-EV the oracle's
  best-prefix label approximates.  Unfed this column degenerates to
  ``-|step|·n_pending`` — a drain-progress counter in disguise (the
  ``x:stop:evals_left`` shape), which is honest: with no positive
  predictions, draining does lose.
* ``x:stop:vl_margin`` — ``max_pending v(c) - |step|``: the marginal
  next-eval EV — positive exactly when the best remaining eval is
  worth its cost at the injected estimate (the analytic break-even
  ``ev-margin`` gate's decision quantity).
* ``x:stop:vl_frac`` — ``vl_pool / Σ_page v``: value-drain progress,
  the ``x:stop:value_frac`` analogue in payoff units.
* ``x:vl:exp`` / ``x:vl:net`` (``evaluate`` rows) — this candidate's
  expected payoff, raw and net of its eval cost: the per-candidate
  half the ranker can weight directly.

The part reads the same injected table as ``open:next-val``
(:func:`mlopt_discovery.open_game.set_value_preds` — unfed, every
``v(c)`` is 0.0, so ``vl_net``/``vl_margin`` carry only their
``-|step|`` terms: a registered-but-unfed part prices the costs
honestly, not wrong).  Registered as ``open:vlabel`` —
unshipped, outside ``OPEN_EXTRA_FEATURES``/``OPEN_GAME_FEATURES``
until the measurement earns it.
"""

from __future__ import annotations

from typing import Any

from mlopt_discovery.meta_arena import Action
from mlopt_discovery.open_game import OPEN_GAME_REWARD
from mlopt_discovery.open_game import _value as _cand_value
from mlopt_discovery.players import RelFeats, register_part

__all__ = [
    "EVAL_COST",
    "VL_FEATURES",
    "vlabel_feats",
]

#: The per-eval cost in payoff units — ``|OPEN_GAME_REWARD["step"]|``.
#: The net/margin columns subtract it so a schema carrying the part
#: prices "is this eval worth its cost" directly.
EVAL_COST: float = -OPEN_GAME_REWARD["step"]

#: The feature names the ``open:vlabel`` part emits — the stop-row
#: block is the pool-priced expected-yield quantities, the eval-row
#: block the same expectation per candidate.
VL_FEATURES: tuple[str, ...] = (
    "x:stop:vl_best",
    "x:stop:vl_pool",
    "x:stop:vl_net",
    "x:stop:vl_margin",
    "x:stop:vl_frac",
    "x:vl:exp",
    "x:vl:net",
)

_STOP_ZERO: dict[str, float] = {
    f: 0.0 for f in VL_FEATURES if f.startswith("x:stop:")
}
_EVAL_ZERO: dict[str, float] = {
    f: 0.0 for f in VL_FEATURES if f.startswith("x:vl:")
}


def vlabel_feats(
    state: Any, action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """Emit the ``open:vlabel`` columns — a ``RelFeats`` function.

    ``stop`` rows get the pool-priced block over the injected
    per-candidate values; ``evaluate`` rows get the candidate's own
    expected payoff.  A drained pool emits the explicit-zero block
    (``vl_net``/`vl_margin` collapse to their ``-|step|`` terms, so a
    spent page reads "nothing left worth the cost").
    """
    if action.op == "stop":
        vals = [_cand_value(state.pool[i]) for i in state.pending]
        if not vals:
            return dict(_STOP_ZERO)
        pool = sum(vals)
        init = sum(_cand_value(c) for c in state.pool)
        return {
            "x:stop:vl_best": max(vals),
            "x:stop:vl_pool": pool,
            "x:stop:vl_net": pool - EVAL_COST * len(vals),
            "x:stop:vl_margin": max(vals) - EVAL_COST,
            "x:stop:vl_frac": pool / init if init > 1e-9 else 0.0,
        }
    if action.op == "evaluate":
        cand = action.params.get("cand")
        if cand is None:
            return dict(_EVAL_ZERO)
        v = _cand_value(cand)
        return {"x:vl:exp": v, "x:vl:net": v - EVAL_COST}
    return {}


#: Registry entry — the part is nameable by spec records
#: (:meth:`DomainSpec.to_record` resolves it through :data:`PARTS`).
register_part("open:vlabel", lambda: RelFeats(vlabel_feats))
