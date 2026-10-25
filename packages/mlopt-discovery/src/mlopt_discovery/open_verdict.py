"""The *verdict-visible* stop columns — certified evidence, not nets.

``docs/results_stop_valuelabel.md`` (round 7) closed with the same
named blocker for the third time: every stop-row family so far is a
*ranking* view (heuristic, injected net probabilities, the policy's
own posterior) and none is a *will-it-pay* one — the nets' held-out
APs on the labels that pay topped at ~0.10.  This part takes the
round-7 verdict's own direction literally: the signal must be
**verdict-conditioned or verifier-visible** — derived from what the
domain itself can certify, not from a learned output.

Two certified evidence sources feed the columns:

* the **episode's own banked verdicts** (``state.verdicts`` — oracle
  output, no injection): paying candidates cluster in the stream
  (~15 useful laws live in the front ~58% of it), so a page that has
  already certified a payer is *rich* — pending is more likely to
  hold another.  ``vd_*`` columns render that composition: banked
  payers, pending members sharing a form with a certified payer or
  with a certified-dead sample, and the *dead-ceiling* — the highest
  heuristic score an evaluated member carried while coming back
  non-paying.  A pending pool that cannot out-score what already
  failed is the episode's own certified "drained" evidence.
* the **corpus verdict table** (:func:`set_verdict_corpus` — the
  ``set_value_preds`` seam's twin, fed closed-form *counts* rather
  than net outputs): a ``signature -> (paying, useful, seen)`` map
  tallied from measured verdicts on a training corpus.  The
  signature is ``(form, lhs-op-spine, rhs-op-spine)`` — *which*
  concrete ops fill the equation skeleton's slots — the granularity
  at which a rewriting form is actually true (``factor-left`` over
  ``mul``/``add`` distributes; over ``sub``/``div`` it does not).
  This is the verifier's own statistics, not a fit: a candidate's
  "will it pay" is read off how often its exact shape was certified
  paying elsewhere.  Unfed, every ``cr_*``/``x:vd:*`` column is 0.0
  and the part is inert, not wrong — the ``open:next-val``
  convention.

Columns (``VD_FEATURES`` — all ``x:stop:*`` unless noted):

* ``vd_banked`` — payers (``useful`` + ``true``) certified this
  episode, /pool_size: page-richness evidence.
* ``vd_use`` — ``useful`` certified this episode, /pool_size.
* ``vd_live`` — fraction of pending sharing a *form* with a
  payer certified this episode (co-payer evidence).
* ``vd_dead`` — fraction of pending whose form was sampled this
  episode and never paid (certified-dead mass).
* ``vd_above`` — fraction of pending heuristic-scored above the
  dead-ceiling.
* ``vd_headroom`` — ``(best pending score - dead-ceiling)/16``:
  the certified "something left beats what already failed" margin.
* ``cr_top`` — max pending corpus pay-rate (signature level).
* ``cr_pool`` — Σ pending pay-rate: expected payers remaining.
* ``cr_use`` — max pending corpus ``useful``-rate.
* ``cr_dead`` — fraction of pending with a certified-dead
  signature (observed, zero pays).
* ``cr_cov`` — fraction of pending whose signature the corpus
  observed at all (the statistic's own coverage).
* ``cr_ftop`` / ``cr_fpool`` — the same max/Σ at *form* level:
  the coarse fallback where the fine signature was never seen.
* ``x:vd:rate`` / ``x:vd:seen`` / ``x:vd:live`` (``evaluate``
  rows) — this candidate's corpus pay-rate, its signature's
  observation count, and whether its form already paid this
  episode.

Registered as ``open:verdict`` — unshipped, outside
``OPEN_EXTRA_FEATURES``/``OPEN_GAME_FEATURES`` until the measurement
(``bench/drivers/policy/stop_verdict.py``) earns it.
"""

from __future__ import annotations

from typing import Any

from gamekit.players import RelFeats, register_part

from mlopt_core.ir import Op
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.open_game import _score as _cand_score

__all__ = [
    "VD_FEATURES",
    "set_verdict_corpus",
    "signature",
    "verdict_feats",
]

#: The feature names the ``open:verdict`` part emits.
VD_FEATURES: tuple[str, ...] = (
    "x:stop:vd_banked",
    "x:stop:vd_use",
    "x:stop:vd_live",
    "x:stop:vd_dead",
    "x:stop:vd_above",
    "x:stop:vd_headroom",
    "x:stop:cr_top",
    "x:stop:cr_pool",
    "x:stop:cr_use",
    "x:stop:cr_dead",
    "x:stop:cr_cov",
    "x:stop:cr_ftop",
    "x:stop:cr_fpool",
    "x:vd:rate",
    "x:vd:seen",
    "x:vd:live",
)

_STOP_ZERO: dict[str, float] = {
    f: 0.0 for f in VD_FEATURES if f.startswith("x:stop:")
}
_EVAL_ZERO: dict[str, float] = {
    f: 0.0 for f in VD_FEATURES if f.startswith("x:vd:")
}


def _ops_preorder(t: Any, out: list) -> list:
    """Collect the op names of *t* in preorder — the op spine."""
    if isinstance(t, Op):
        out.append(t.op)
        for a in t.args:
            _ops_preorder(a, out)
    return out


#: Memoized shape signature — the pool's members recur every
#: featurize call within an episode (the ``_CAND_SCORE`` seam).
_CAND_SIG: dict[Any, tuple] = {}


def signature(cand: Any) -> tuple:
    """``(form, lhs-op-spine, rhs-op-spine)`` — the shape's identity.

    The granularity at which a rewriting *form* is actually true:
    the form names the equation skeleton, the op spines name which
    concrete ops fill its slots (``factor-left`` over
    ``mul``/``add`` pays; over ``sub``/``div`` it cannot).  The
    corpus verdict table is keyed here.
    """
    sig = _CAND_SIG.get(cand)
    if sig is None:
        sig = (
            cand.label.split(":", 1)[0],
            tuple(_ops_preorder(cand.lhs, [])),
            tuple(_ops_preorder(cand.rhs, [])),
        )
        _CAND_SIG[cand] = sig
    return sig


#: The injected corpus verdict table —
#: ``{"sig": {signature: (pays, useful, seen)},
#:   "form": {form: (pays, useful, seen)}}`` — closed-form counts
#: tallied from measured verdicts on a training corpus, *not* a
#: net's outputs.  ``None`` means uninjected (``cr_*`` columns 0.0).
_verdict_corpus: dict | None = None


def set_verdict_corpus(table: dict | None) -> None:
    """Inject (or clear) the corpus verdict statistics.

    *table* is ``{"sig": {...}, "form": {...}}`` where each inner
    map keys a signature (:func:`signature`) or form string to its
    certified tally ``(payers, useful, evaluated)``.  ``None``
    clears the seam — the ``cr_*``/``x:vd:*`` columns fall back to
    zero, the ``vd_*`` block still reads the episode's own banked
    verdicts.
    """
    global _verdict_corpus
    _verdict_corpus = table


def _cand_rates(cand: Any) -> tuple[float, float, float]:
    """``(pay_rate, useful_rate, seen)`` of *cand*'s signature."""
    tab = _verdict_corpus
    if not tab:
        return 0.0, 0.0, 0.0
    pay, use, seen = tab.get("sig", {}).get(signature(cand), (0, 0, 0))
    return (pay / seen if seen else 0.0), (
        use / seen if seen else 0.0
    ), float(seen)


def _form_rate(cand: Any) -> float:
    """Form-level corpus pay-rate — the coarse fallback."""
    tab = _verdict_corpus
    if not tab:
        return 0.0
    pay, _use, seen = tab.get("form", {}).get(
        cand.label.split(":", 1)[0], (0, 0, 0)
    )
    return pay / seen if seen else 0.0


def _episode_forms(state: Any) -> tuple[set, set]:
    """Forms seen this episode split by verdict — ``(paying, dead)``.

    The episode's certified composition: a form that produced a
    paying verdict is *live* evidence for its pending siblings; a
    form sampled without paying is certified-dead evidence.
    """
    paying: set = set()
    dead: set = set()
    for i, v in state.verdicts:
        form = state.pool[i].label.split(":", 1)[0]
        if v in ("useful", "true"):
            paying.add(form)
        else:
            dead.add(form)
    return paying, dead


def _dead_ceiling(state: Any) -> float:
    """Highest heuristic score an evaluated non-payer carried.

    Certified "scores this high still failed" evidence: a pending
    pool unable to out-score it offers nothing the episode has not
    already seen fail.
    """
    best = 0.0
    seen_nonpay = False
    for i, v in state.verdicts:
        if v not in ("useful", "true"):
            seen_nonpay = True
            s = _cand_score(state.pool[i])
            if s > best:
                best = s
    return best if seen_nonpay else 0.0


def _stop_cols(state: Any) -> dict[str, float]:
    """Emit the ``x:stop:vd_*``/``x:stop:cr_*`` block."""
    n = max(state.pool_size, 1)
    pending = list(state.pending)
    if not pending:
        return dict(_STOP_ZERO)
    paying, dead = _episode_forms(state)
    ceiling = _dead_ceiling(state)
    cands = [state.pool[i] for i in pending]
    forms = [c.label.split(":", 1)[0] for c in cands]
    scores = [_cand_score(c) for c in cands]
    rates = [_cand_rates(c) for c in cands]
    frates = [_form_rate(c) for c in cands]
    k = len(cands)
    return {
        "x:stop:vd_banked": (state.n_useful + state.n_true) / n,
        "x:stop:vd_use": state.n_useful / n,
        "x:stop:vd_live": sum(f in paying for f in forms) / k,
        "x:stop:vd_dead": sum(f in dead for f in forms) / k,
        "x:stop:vd_above": sum(s > ceiling for s in scores) / k,
        "x:stop:vd_headroom": (max(scores) - ceiling) / 16.0,
        "x:stop:cr_top": max((r[0] for r in rates), default=0.0),
        "x:stop:cr_pool": sum(r[0] for r in rates),
        "x:stop:cr_use": max((r[1] for r in rates), default=0.0),
        "x:stop:cr_dead": sum(
            r[2] > 0 and r[0] == 0.0 for r in rates
        )
        / k,
        "x:stop:cr_cov": sum(r[2] > 0 for r in rates) / k,
        "x:stop:cr_ftop": max(frates, default=0.0),
        "x:stop:cr_fpool": sum(frates),
    }


def _eval_cols(state: Any, action: Action) -> dict[str, float]:
    """Emit the per-candidate ``x:vd:*`` block."""
    cand = action.params.get("cand")
    if cand is None:
        return dict(_EVAL_ZERO)
    rate, _use, seen = _cand_rates(cand)
    paying, _dead = _episode_forms(state)
    return {
        "x:vd:rate": rate,
        "x:vd:seen": seen,
        "x:vd:live": float(
            cand.label.split(":", 1)[0] in paying
        ),
    }


def verdict_feats(
    state: Any, action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """Emit the ``open:verdict`` columns — a ``RelFeats`` function.

    ``stop`` rows get the certified pool composition (episode
    verdicts + corpus statistics); ``evaluate`` rows get the
    candidate's own corpus rate and episode-liveness.  Explicit
    zeros on a drained pool or an unfed corpus seam.
    """
    if action.op == "stop":
        return _stop_cols(state)
    if action.op == "evaluate":
        return _eval_cols(state, action)
    return {}


#: Registry entry — the part is nameable by spec records
#: (:meth:`DomainSpec.to_record` resolves it through :data:`PARTS`).
register_part("open:verdict", lambda: RelFeats(verdict_feats))
