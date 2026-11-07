"""``bench.benchkit`` — compatibility alias for the ``benchkit`` package.

The benchmark harness was promoted to ``handl-lang/gamekit``
(``packages/benchkit``, plan 0032) so any domain repo can depend on
it.  This package re-exports it: every ``bench.benchkit.<mod>`` import
resolves to the real ``benchkit.<mod>`` module, and the public names
below are the same objects.  New code should ``import benchkit``
directly; this alias exists so the in-repo ``bench/`` lanes keep
working unchanged.

The one graphopt-specific piece left behind is
``bench.benchkit.render.catalog`` — the registry-driven docs catalog,
which lives in :mod:`bench.benchkit.render` (a real subpackage here,
not an alias).
"""

from __future__ import annotations

import importlib
import sys

import benchkit as _benchkit
from benchkit import *  # noqa: F401,F403 — re-export the public names
from benchkit import __all__ as _all

# Top-level submodules (everything except ``render``, which stays a
# real subpackage below for the graphopt-specific ``catalog``).
_SUBMODULES = (
    "baselines",
    "checkpoint",
    "compare",
    "env",
    "equality",
    "jsonl",
    "ledger",
    "model",
    "provenance",
    "regime",
    "report",
    "runner",
    "selftest",
    "stats",
    "table",
)

_pkg = sys.modules[__name__]
for _name in _SUBMODULES:
    _mod = importlib.import_module(f"benchkit.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    setattr(_pkg, _name, _mod)

__all__ = list(_all) + ["render"]
