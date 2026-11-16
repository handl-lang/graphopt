"""Model adapters — the product's catalog of liftable workloads.

An adapter turns a user-facing model reference (HF id, local path,
config) into ``(module, example_inputs, meta)`` for
:class:`mlopt_orchestrator.optimize.Optimizer`.  Registration is
explicit: ``register_adapter`` at import of the adapter module; the
``mlopt.models`` package ``__init__`` wires the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelSpec:
    """What ``load`` returns: the callable/module plus probe inputs.

    ``kwargs`` carries extra keyword arguments the adapter declares
    for the module's forward calls — ``optimize_model`` forwards them
    to the deliver pass's benchmark timer (``module(*inputs, **kwargs)``)
    so an adapter whose wrapped module needs call-time options (e.g. a
    cache flag) can declare them once.  The verify gate inside
    ``Optimizer.lower`` does not take them: an adapter whose *verify*
    path needs kwargs must bake them into the module itself.  Empty
    for every shipped adapter today.
    """

    module: Any
    inputs: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    #: Optional ``mlopt_core.ports.Source`` override — adapters whose
    #: lift is already a term (``.hnl`` sources) carry the port that
    #: answers ``to_ir`` with the pre-lifted ``(IR, leaf values)``;
    #: ``optimize_model`` hands it to ``Optimizer`` in place of the
    #: backend's ``TorchSource``.  ``None`` keeps the torch-export
    #: frontend for every shipped adapter.
    source: Any = None


class ModelAdapter(Protocol):
    """Lifts a model reference into an optimizable spec."""

    name: str
    description: str
    available: bool
    unavailable_reason: str

    def load(self, ref: str, **opts: Any) -> ModelSpec: ...

    def claims(self, ref: str) -> bool:
        """The adapter's own ref predicate — the bare-ref extension point.

        Routing resolves ``name:<rest>`` by registry name and bare
        *adapter names* by exact lookup first; ``claims`` is consulted
        only for bare refs nothing name-matched, letting an adapter
        claim refs outside the ``name:`` convention.
        """
        return False

    def probe(self, ref: str) -> dict[str, Any] | None:
        """Cheap metadata for ``mlopt doctor`` — no weight downloads.

        Optional: adapters that implement it let doctor report
        availability/lift details without calling ``load``.  Heavy
        (e.g. HF) adapters should implement this; doctor falls back to
        ``load`` only for available adapters without a probe.
        """
        return None


_ADAPTERS: dict[str, ModelAdapter] = {}


#: Attributes every registered adapter must expose — checked at
#: registration so a broken plugin fails loudly at discovery, not as
#: an AttributeError mid-command.
_REQUIRED_ATTRS = ("name", "description", "available", "unavailable_reason", "load")


def register_adapter(adapter: ModelAdapter) -> None:
    """Register *adapter*; validate the protocol surface and collisions.

    Raises ``TypeError`` when the ``ModelAdapter`` surface is missing
    (``name``/``description``/``available``/``unavailable_reason``/
    callable ``load``).  A second adapter claiming an already-registered
    ``name`` emits a loud ``RuntimeWarning`` and the FIRST registration
    wins — silent last-wins would hide plugin bugs.
    """
    import warnings

    missing = [a for a in _REQUIRED_ATTRS if not hasattr(adapter, a)]
    if missing:
        raise TypeError(f"adapter {adapter!r} missing ModelAdapter surface: {missing}")
    if not callable(getattr(adapter, "load", None)):
        raise TypeError(f"adapter {adapter!r}: 'load' is not callable")
    name = adapter.name
    if not isinstance(name, str) or not name.strip():
        raise TypeError(f"adapter name must be a non-empty string, got {name!r}")
    existing = _ADAPTERS.get(name)
    if existing is not None and existing is not adapter:
        warnings.warn(
            f"adapter name collision: {name!r} already registered "
            f"({existing!r}) — keeping the first, ignoring {adapter!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    _ADAPTERS[name] = adapter


def get_adapter(name: str) -> ModelAdapter | None:
    return _ADAPTERS.get(name)


def list_adapters() -> dict[str, ModelAdapter]:
    return dict(_ADAPTERS)


def resolve_ref(ref: str) -> ModelAdapter | None:
    """Map a model ref to an adapter — one semantics everywhere.

    * ``name:<rest>`` routes by adapter name (``whisper:openai/…``).
    * A bare *adapter name* (``kokoro``) resolves to that adapter —
      the same behaviour ``mlopt doctor`` has always had.
    * Any other bare ref goes through each *available* adapter's
      ``claims(ref)`` — the extension point for non-prefixed refs.

    Leading/trailing whitespace is stripped.  Returns ``None`` when
    nothing resolves — callers print the honest "no adapter" decline.
    """
    ref = ref.strip()
    if ":" in ref:
        name, _ = ref.split(":", 1)
        return _ADAPTERS.get(name.strip())
    exact = _ADAPTERS.get(ref)
    if exact is not None:
        return exact
    for adapter in _ADAPTERS.values():
        if adapter.available and adapter.claims(ref):
            return adapter
    return None
