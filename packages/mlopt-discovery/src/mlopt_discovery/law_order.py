"""Moved to ``discoverykit.law_order`` — compatibility shim (plan 0034 phase 1).

``sys.modules`` is swapped so ``mlopt_discovery.law_order`` *is* the
discoverykit module: attribute access, ``is`` comparisons and
``__module__`` strings behave exactly as before the move.  The
``__main__`` guard re-dispatches ``python -m mlopt_discovery.law_order``
to the new home; stay-side lazy imports inside the moved code still
spell ``mlopt_discovery.<stay>`` and resolve here.
"""

import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("discoverykit.law_order", run_name="__main__")
else:
    from discoverykit import law_order as _moved

    _sys.modules[__name__] = _moved
