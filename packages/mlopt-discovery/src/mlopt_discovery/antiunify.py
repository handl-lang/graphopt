"""Moved to ``discoverykit.antiunify`` — compatibility shim (plan 0034 phase 1).

``sys.modules`` is swapped so ``mlopt_discovery.antiunify`` *is* the
discoverykit module: attribute access, ``is`` comparisons and
``__module__`` strings behave exactly as before the move.  The
``__main__`` guard re-dispatches ``python -m mlopt_discovery.antiunify``
to the new home; stay-side lazy imports inside the moved code still
spell ``mlopt_discovery.<stay>`` and resolve here.
"""

import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("discoverykit.antiunify", run_name="__main__")
else:
    from discoverykit import antiunify as _moved

    _sys.modules[__name__] = _moved
