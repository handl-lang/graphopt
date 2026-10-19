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
) -> Backend:
    """Assemble the torch backend value.

    ``ops`` builds the default :class:`TorchSink` when ``sink`` is not
    given (the same precedence ``optimize_model`` always had: an
    explicit ``sink`` wins, ``ops`` is ignored).  Individual port
    arguments replace the defaults — mixing a custom sink with the
    stock composer/meter is the supported shape.
    """
    return Backend(
        source=source if source is not None else TorchSource(),
        sink=sink if sink is not None else TorchSink(ops=ops),
        composer=(
            composer if composer is not None else TorchComposer()
        ),
        meter=meter if meter is not None else TorchMeter(),
    )
