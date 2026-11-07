"""``bench.benchkit.render`` — alias for ``benchkit.render`` + catalog.

The renderers themselves live in the ``benchkit`` package; this
subpackage aliases their modules so ``bench.benchkit.render.<mod>``
imports keep working, and adds the one graphopt-specific renderer:
``catalog`` — the suite catalog generated from ``bench.registry``.
"""

from __future__ import annotations

import importlib
import sys

import benchkit.render as _render
from benchkit.render import *  # noqa: F401,F403
from benchkit.render import __all__ as _all

_SUBMODULES = (
    "dashboard",
    "html",
    "markdown",
    "plots",
    "quarto",
    "results",
    "slidev",
)

_pkg = sys.modules[__name__]
for _name in _SUBMODULES:
    _mod = importlib.import_module(f"benchkit.render.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    setattr(_pkg, _name, _mod)

__all__ = list(_all) + ["catalog"]
