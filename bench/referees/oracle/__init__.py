"""``oracle`` — the numeric oracle over the claim's accepted region.

One component: fp64 tri-state eval over real matches + the shipped
synthesized binding window (the ``360``-site domain the pipeline's
``num_true`` leg and auto_cond measure on).  Fast, and exactly as
shallow as the window it reads — the unequal-rank mid-insert corner
that falsified the grid-eq guard lives outside it.
"""

from __future__ import annotations

from bench.referees.registry import RefereeSpec

SPEC = RefereeSpec(
    name="oracle",
    components=("oracle",),
    title="Numeric oracle — the stock binding window",
    note="fp64 eval over real matches + the 360-site synth window",
    record="the pipeline's num_true leg — true on the measured "
    "window, blind past it",
)
