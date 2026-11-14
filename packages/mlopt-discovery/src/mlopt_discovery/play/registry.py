"""Moved to ``discoverykit.play.registry`` — compatibility shim (plan 0034 phase 1).

``sys.modules`` is swapped so ``mlopt_discovery.play.registry`` *is*
the discoverykit module — ``DOMAINS`` stays the one live dict every
consumer mutates (kernelopt's ``mlopt_kernel.play.register`` writes
``DOMAINS["kernel"]`` here).
"""

import sys as _sys

from discoverykit.play import registry as _moved

_sys.modules[__name__] = _moved
