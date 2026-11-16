"""Model adapter plugins — auto-discovered catalog entries.

Each ``*.py`` module in this package is an adapter plugin exposing a
module-level ``ADAPTER`` object (or ``ADAPTERS`` tuple) satisfying the
``mlopt.models.ModelAdapter`` protocol.  Discovery is explicit:
``mlopt.models.catalog`` calls :func:`load_all` once at import — plain
``import mlopt.adapters`` mutates nothing.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlopt.models import ModelAdapter

_CACHE: dict[str, ModelAdapter] | None = None


def _plugin_modules() -> list[str]:
    return [
        info.name
        for info in pkgutil.iter_modules(__path__, __name__ + ".")
        if not info.name.endswith(".__init__")
    ]


def _module_adapters(module: object) -> list[ModelAdapter]:
    found: list[ModelAdapter] = []
    single = getattr(module, "ADAPTER", None)
    if single is not None:
        found.append(single)
    found.extend(getattr(module, "ADAPTERS", ()) or ())
    return found


class _UnavailableAdapter:
    """Pseudo-adapter for a plugin module whose import failed.

    Surfaces the failure in ``mlopt models`` (``pending — <reason>``)
    instead of crashing discovery: the module name becomes the
    adapter name, ``available`` is False, and ``load`` re-raises the
    recorded import error if anything calls it anyway.
    """

    def __init__(self, module_name: str, error: BaseException) -> None:
        self.name = module_name.rsplit(".", 1)[-1]
        self.description = f"adapter plugin {module_name} — import failed"
        self.available = False
        self.unavailable_reason = f"import failed: {type(error).__name__}: {error}"
        self._error = error

    def claims(self, ref: str) -> bool:
        return False

    def load(self, ref: str, **opts: Any) -> Any:
        raise RuntimeError(
            f"adapter {self.name} is unavailable: {self.unavailable_reason}"
        ) from self._error

    def probe(self, ref: str) -> dict[str, Any] | None:
        return {"available": False, "reason": self.unavailable_reason}


def load_all() -> dict[str, ModelAdapter]:
    """Import every adapter plugin and register its ``ADAPTER`` objects.

    Idempotent: the first call performs discovery and populates
    ``mlopt.models``; later calls return the cached snapshot.  Modules
    without an ``ADAPTER``/``ADAPTERS`` are imported but register
    nothing.  A module whose import raises becomes an *unavailable*
    pseudo-adapter (``mlopt models`` shows the reason; nothing crashes).
    Name collisions warn loudly inside ``register_adapter`` — the
    first registration wins, never a silent last-wins.
    """
    global _CACHE
    if _CACHE is not None:
        return dict(_CACHE)
    from mlopt.models import get_adapter, register_adapter

    loaded: dict[str, ModelAdapter] = {}
    for name in _plugin_modules():
        try:
            module = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 — fault isolation is the point
            adapter = _UnavailableAdapter(name, e)
            register_adapter(adapter)
            loaded[adapter.name] = adapter
            continue
        for found in _module_adapters(module):
            register_adapter(found)
            # register_adapter keeps the FIRST on a name collision —
            # mirror the registry so `loaded` matches it.
            kept = get_adapter(found.name)
            if kept is not None:
                loaded[found.name] = kept
    _CACHE = loaded
    return dict(loaded)
