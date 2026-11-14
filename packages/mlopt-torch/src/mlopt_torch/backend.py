"""TorchBackend — the canonical :class:`~mlopt_core.pipeline.Backend`.

One function returning the immutable backend value the orchestrator
consumes::

    from mlopt_torch import TorchBackend

    opt = Optimizer(backend=TorchBackend())   # works end to end
    mod, stats = opt.optimize(model, x)

The value bundles the four torch ports — :class:`TorchSource`
(``torch.export`` → IR), :class:`TorchSink` (IR → ``IRModule`` +
verify + the executor table), :class:`TorchComposer` (the per-block
structural machinery), :class:`TorchMeter` (wall-clock timing).

This is the *only* place torch defaults appear — a convenience
constructor.  ``Optimizer`` itself takes no default backend; the
deprecated ``optimize_*`` wrappers resolve through it.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.ops import OpTable
from mlopt_core.pipeline import Backend
from mlopt_core.ports import Composer, Meter, Sink, Source

from mlopt_torch.adapters import TorchSink, TorchSource
from mlopt_torch.composer import TorchComposer
from mlopt_torch.meter import TorchMeter

__all__ = ["TorchBackend"]


def TorchBackend(
    *,
    ops: OpTable | None = None,
    source: Source | None = None,
    sink: Sink | None = None,
    composer: Composer | None = None,
    meter: Meter | None = None,
    device_policy: Any = None,
) -> Backend:
    """Assemble the torch backend value.

    ``ops`` builds the default :class:`TorchSink` when ``sink`` is not
    given (the same precedence ``optimize_model`` always had: an
    explicit ``sink`` wins, ``ops`` is ignored).  Individual port
    arguments replace the defaults — mixing a custom sink with the
    stock composer/meter is the supported shape.

    ``device_policy`` (Phase B) is a
    :class:`~mlopt_torch.device.DevicePolicy`-shaped value — e.g.
    :class:`~mlopt_torch.device.DevicePolicy` or
    :class:`~mlopt_torch.device_placement.ShardedAwareDevicePolicy`.
    It is wired two ways so the answer is consistent on either
    substrate vintage: the returned ``Backend`` carries it on
    ``device_policy`` (consumed by ``Optimizer`` once the eqsat pin
    ships the field), AND ``source`` is wrapped in
    :class:`~mlopt_torch.device_placement.DeviceCheckedSource` so
    ``policy.resolve(model, x)`` runs before export TODAY — a refusal
    (mixed devices, DDP/FSDP/TP wrappers, sharded params) raises
    before the e-graph is built.  ``None`` (the default) leaves the
    historical unpoliced path untouched.
    """
    src = source if source is not None else TorchSource()
    if device_policy is not None:
        from mlopt_torch.device_placement import DeviceCheckedSource

        src = DeviceCheckedSource(src, device_policy)
    backend = Backend(
        source=src,
        sink=sink if sink is not None else TorchSink(ops=ops),
        composer=(
            composer if composer is not None else TorchComposer()
        ),
        meter=meter if meter is not None else TorchMeter(),
    )
    if device_policy is not None and "device_policy" in getattr(
        Backend, "__dataclass_fields__", {}
    ):
        # Frozen dataclass — set the Phase B field when the pinned
        # ``Backend`` carries it; on older pins the field does not
        # exist and only the source-wrap path is live (guarded by
        # field presence, not try/except).
        object.__setattr__(backend, "device_policy", device_policy)
    return backend
