"""``naive`` — the trust arm: the claim is accepted as declared.

No components run; the verdict is "trusted, unmeasured".  This is
the auto_cond-era referee: the arm that ships the grid-eq
``mul_unsq_nat`` guard the adversarial sweep later measured at 212
bad accepts (``project/retros/mul-unsq-nat.md``) — the strictness
dial's bottom rung, kept as the honest baseline every measured arm
is compared against.
"""

from __future__ import annotations

from bench.referees.registry import RefereeSpec

SPEC = RefereeSpec(
    name="naive",
    components=(),
    title="Trust the declaration — no measurement",
    note="accepts every claim unmeasured",
    record="the auto_cond-era referee — 212 bad accepts on the "
    "mul_unsq_nat grid-eq guard",
)
