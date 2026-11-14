"""Moved to ``discoverykit.lawdata`` — compatibility shim (plan 0034 phase 1).

``sys.modules`` is swapped so the old path *is* the discoverykit
package — every ``lawdata.X`` binding resolves to the same objects.
The ``.laws`` and ``.game`` submodule aliases are pre-registered so
``import mlopt_discovery.lawdata.laws.banks`` and the
``mlopt_discovery.lawdata.game`` half keep resolving to the single
copies (one table, one home — the plan-0032 seam, now across repos).
"""

import sys as _sys

import discoverykit.lawdata.laws as _laws
import gamekit.gamedata as _game
from discoverykit import lawdata as _moved

_sys.modules[__name__] = _moved
_sys.modules[__name__ + ".laws"] = _laws
for _name in ("banks", "handlers", "macros", "moves", "terms"):
    _mod = __import__(
        f"discoverykit.lawdata.laws.{_name}", fromlist=["*"]
    )
    _sys.modules[f"{__name__}.laws.{_name}"] = _mod
_sys.modules[__name__ + ".game"] = _game
_sys.modules[__name__ + ".game.features"] = _game.features
_sys.modules[__name__ + ".game.weights"] = _game.weights
