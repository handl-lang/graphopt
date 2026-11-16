"""mlopt model catalog — the explicit registration trigger.

Importing this module runs adapter discovery exactly once
(:func:`mlopt.adapters.load_all`) and populates the
``mlopt.models`` registry.  It is the single import that mutates the
registry — ``import mlopt.models`` or ``import mlopt.adapters`` alone
register nothing.
"""

from __future__ import annotations

from mlopt.adapters import load_all

load_all()
