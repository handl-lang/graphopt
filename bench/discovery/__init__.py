"""The discovery benchmark — a versioned corpus of *named* optimizations.

Each case in ``cases/`` spells a program the way a practitioner writes
it (a conv+BatchNorm eval stack, a hand-rolled softmax attention, an
unrolled recurrence) and asks the engine two questions:

* **derivability** — does equality saturation + the non-local passes
  reach the named structure (the fold, the carrier, the fused kernel),
  and does the extraction pick it?
* **novelty** — is the certified form below the *registered floor* —
  the best known-good implementation registered as a baseline — on the
  case's cost axis?  Beating eager is table stakes; beating the
  registered floor is a real find, not a rediscovery.

Two lanes (:mod:`bench.discovery.run`): ``--fast`` prices everything
symbolically (``dag_cost`` — seconds, gate-suitable); ``--full`` lowers
and *measures* (wall time + the sink's equivalence gate).
"""
