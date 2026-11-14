"""Shared content-keyed shape memo for carrier term walks.

Every carrier module that falls back to ``mlopt_core.typing._shape_of``
on shared-DAG interned terms needs the same memo.  The copied
``_SHAPE_MEMO`` dicts (``xcarrier``, ``trace``, ``om_lower``, …) were
identical in kind — one dict per module, process-global either way —
so they now share a single memo here.  Terms are hash-consed Op
objects, so the memo is sound: ``_shape_of`` is a pure function of
the term's content.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.typing import _shape_of as _core_shape_of

__all__ = ["shape_of"]

_MEMO: dict = {}


def shape_of(term: Any):
    """``mlopt_core.typing._shape_of`` under the shared module memo."""
    return _core_shape_of(term, _MEMO)
