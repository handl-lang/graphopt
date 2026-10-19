"""mlopt-torch — the PyTorch integration adapters + backend.

``torch.export`` → IR (:mod:`~mlopt_torch.torch_bridge`), the
executable ``IRModule`` + weight-chain folding, the batched executor
machinery (:mod:`~mlopt_torch.executors`), the small model zoo
(:mod:`~mlopt_torch.models`), and the typed reports + equivalence
gate (:mod:`~mlopt_torch.report`).

Plan 0007 adds the backend ports — :func:`~mlopt_torch.backend.TorchBackend`
(the :class:`~mlopt_core.pipeline.Backend` value:
:class:`TorchSource` + :class:`TorchSink` +
:class:`~mlopt_torch.composer.TorchComposer` +
:class:`~mlopt_torch.meter.TorchMeter`), the torch delivery runner
(:class:`~mlopt_torch.runners.TorchCompileRunner`), the torch halves
of the regime machinery (:mod:`~mlopt_torch.regime`) and calibration
(:mod:`~mlopt_torch.calibrate`), the production export helpers
(:mod:`~mlopt_torch.export`), and the torch autotune candidate
builders (:mod:`~mlopt_torch.autotune`).  The public names below
resolve lazily — importing ``mlopt_torch`` pulls no pipeline
machinery until the names are touched.
"""

import importlib
from typing import Any

# Importing the concrete-eval backend registers it with mlopt-core
# (:func:`mlopt_core.meta.register_concrete_eval`): any import of the
# torch adapter makes core's numeric candidate check work, and core
# itself imports neither ``torch`` nor ``mlopt_torch``.
from mlopt_torch import meta_eval as _meta_eval  # noqa: F401

# ---------------------------------------------------------------------------
# Lazy public surface — names resolve on first attribute access.
# ---------------------------------------------------------------------------

_DELEGATED = {
    "TorchBackend": "mlopt_torch.backend",
    "TorchComposer": "mlopt_torch.composer",
    "TorchMeter": "mlopt_torch.meter",
    "TorchCompileRunner": "mlopt_torch.runners",
    "TORCH_BUILDERS": "mlopt_torch.autotune",
    "RegimeDispatch": "mlopt_torch.regime",
    "regime_dispatch": "mlopt_torch.regime",
    "calibrate": "mlopt_torch.calibrate",
    "TargetProfile": "mlopt_torch.calibrate",
    "load_profile": "mlopt_torch.calibrate",
    "ContractionPolicy": "mlopt_torch.contraction_policy",
    "load_contraction_policy": "mlopt_torch.contraction_policy",
}


def __getattr__(name: str) -> Any:
    """Resolve the adapter-surface names lazily."""
    mod = _DELEGATED.get(name)
    if mod is not None:
        return getattr(importlib.import_module(mod), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
