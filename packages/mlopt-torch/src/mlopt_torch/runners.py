"""Torch delivery runners — the torch-coupled half of the Runner seam.

The delivery runners split by backend (plan 0007): the backend-neutral
:class:`Runner` protocol lives in :mod:`mlopt_core.ports`;
:class:`IdentityRunner` / :class:`ChainedRunner` /
:func:`runner_candidate` live in the orchestrator
(:mod:`mlopt_orchestrator.runners`); the torch-coupled
:class:`TorchCompileRunner` lives here; and the CUDA-device-coupled
:class:`CudaGraphRunner` lives in :mod:`mlopt_cuda`.

This module re-exports the neutral runner surface so the historical
``mlopt_orchestrator.runners`` path (and ``from mlopt_torch.runners
import …``) resolves it.  :class:`CudaGraphRunner` is deliberately NOT
re-exported: it lives in :mod:`mlopt_cuda`, and importing it here
would make ``mlopt_torch`` depend on ``mlopt_cuda`` (which in turn
depends on ``mlopt_torch``), a cycle.  Import it from its home::

    from mlopt_cuda import CudaGraphRunner
"""

from __future__ import annotations

from typing import Any, cast

import torch
from mlopt_core.ports import Runner
from mlopt_orchestrator.runners import (
    ChainedRunner,
    IdentityRunner,
    runner_candidate,
)

__all__ = [
    "ChainedRunner",
    "IdentityRunner",
    "Runner",
    "TorchCompileRunner",
    "runner_candidate",
]


class TorchCompileRunner:
    """Wrap the delivered executor in ``torch.compile``.

    Compile failures surface at first call — the module is invoked
    once here so a broken backend keeps the uncompiled executor
    rather than shipping a wrapper that fails later.
    ``stats["compiled"]`` records which ran.
    """

    name = "torch_compile"
    #: Marker ``optimize_model`` reads to price the fusion-region
    #: cost model during carrier selection — a compiled delivery
    #: bills every term under ``lowering="compiled"``.
    delivers_compiled = True

    def __init__(self, **compile_kwargs: Any) -> None:
        """Record the ``torch.compile`` keyword arguments."""
        self.compile_kwargs = dict(compile_kwargs)

    def apply(
        self,
        module: torch.nn.Module,
        example_input: Any,
        stats: dict[str, Any],
    ) -> torch.nn.Module:
        """Wrap *module* in ``torch.compile``, recording the outcome."""
        try:
            compiled = torch.compile(module, **self.compile_kwargs)
            compiled(example_input)
            stats["compiled"] = True
            return cast(torch.nn.Module, compiled)
        except Exception:
            stats["compiled"] = False
            return module
