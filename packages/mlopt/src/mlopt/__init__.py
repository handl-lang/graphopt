"""mlopt — the product entrypoint.

Pipe models through the verified optimizer: lift → search → lower
(verify=True) → deliver, with an honest verdict (measured deltas,
verify status, declines) — never a silent fallback.

The machinery lives in the substrate/domain repos (eqsat, graphopt,
kernelopt); this package is composition only.
"""

__version__ = "0.1.0"
