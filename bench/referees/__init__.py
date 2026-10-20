"""The referee registry — strictness arms over the admission gauntlet.

A *referee* is a named subset of the gauntlet's measurement
machinery, applied to a claimed equality ``lhs == rhs`` (a
:class:`~mlopt_core.egraph.Rewrite` — a pattern claim with optional
``check``/``cond``/``derive`` hooks, or a concrete term pair).  The
arms are a strictness dial, cumulative along the gauntlet's own
order — numeric oracle → typed-pay → adversarial sweep →
certificate replay:

* ``naive``  — the trust arm: the claim is accepted as declared,
  unmeasured.  The auto_cond-era referee: it is the arm that would
  have shipped ``mul_unsq_nat``'s grid-eq guard.
* ``oracle`` — the numeric oracle over the claim's accepted region
  (real matches + the synthesized binding window).
* ``pay``    — + the typed-pay leg: the claim fires on real probe
  terms, every changed extraction verifies, and it strictly lowers
  cost — plus the :class:`~mlopt_discovery.scoring.ScoringSpec`
  gates on the measured axes.
* ``sweep``  — + the adversarial sweep: the guarded region over the
  widened binding domain, the leg that caught the grid-eq guard's
  212 bad accepts (:mod:`project.retros` — mul-unsq-nat).
* ``full``   — + the certificate leg: :func:`mlopt_discovery.verifier.
  verify_law` must derive the equality and its certificate must
  replay standalone.

The registry mirrors ``bench/limits``: specs are data, folders load
lazily, and a referee that cannot judge a claim/environment raises
``SkipReferee`` — the row records the reason, never a silent drop.
"""

from bench.referees.registry import (  # noqa: F401
    COMPONENT_ORDER,
    COMPONENTS,
    ComponentVerdict,
    RefereeCtx,
    RefereeSpec,
    RefereeVerdict,
    SkipReferee,
    availability,
    judge,
    names,
    spec,
    specs,
)
