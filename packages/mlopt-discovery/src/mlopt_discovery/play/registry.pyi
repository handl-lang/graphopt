"""Type stub for the ``mlopt_discovery.registry`` compatibility shim.

The ``.py`` swaps ``sys.modules[__name__]`` for the discoverykit
module at import time (the plan-0034 lazy seam); this stub resolves
that seam for ty.  ``__getattr__`` covers the underscore-private and
dynamically-bound names the star re-export cannot see.
"""

from typing import Any

from discoverykit.play.registry import *  # noqa: F403

def __getattr__(name: str) -> Any: ...
