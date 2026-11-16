"""Torch delivery runners — the torch-coupled half of the Runner seam.

The delivery runners split by backend (plan 0007): the backend-neutral
:class:`Runner` protocol lives in :mod:`mlopt_core.ports`;
:class:`IdentityRunner` / :class:`ChainedRunner` /
:func:`runner_candidate` live in the orchestrator
(:mod:`mlopt_orchestrator.runners`); the torch-coupled
:class:`TorchCompileRunner` lives here; and the CUDA-device-coupled
:class:`CudaGraphRunner` lives in :mod:`kit_cuda`.

This module re-exports the neutral runner surface so the historical
``mlopt_orchestrator.runners`` path (and ``from mlopt_torch.runners
import …``) resolves it.  :class:`CudaGraphRunner` is deliberately NOT
re-exported: it lives in :mod:`kit_cuda`, and importing it here
would make ``mlopt_torch`` depend on ``kit_cuda`` (which in turn
depends on ``mlopt_torch``), a cycle.  Import it from its home::

    from kit_cuda import CudaGraphRunner
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
    "RegionFusionRunner",
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


class RegionFusionRunner:
    """Fuse census-eligible elementwise regions into compiled kernels.

    The delivery-side half of ``docs/reports/kokoro_region_fuse.md``:
    :func:`mlopt_torch.region_fuse.fuse_module` rewrites the delivered
    ``IRModule``'s root — each eligible single-exit elementwise region
    becomes one ``fused_region`` op dispatched to a per-shape
    ``torch.compile``-compiled unrolled body — with per-region screening on
    real tensors and a whole-module fused-vs-unfused gate on
    ``example_input``.  Verify is untouched: ``lower(verify=True)``
    still compares the delivered module against a fresh UNFUSED
    lowering of the certified IR at the caller's ``rtol`` — set
    ``RegionFuseConfig.output_rtol`` ≤ that ``rtol`` (the measured
    fused-vs-unfused ``max_rel`` lands in ``stats["region_fuse"]``
    either way).

    Opt-in by construction: pass ``runner=RegionFusionRunner()`` (or
    a :class:`ChainedRunner` with it ahead of ``TorchCompileRunner``
    for a whole-module compile on top).  Non-:class:`IRModule`
    deliveries (carrier executors) decline honestly — the census
    needs the term, which only the generic lowering exposes.
    ``delivers_compiled`` is deliberately NOT set: fusion compiles
    region bodies, not the module spine, so carrier pricing keeps
    its uncompiled-delivery assumption.
    """

    name = "region_fuse"

    def __init__(
        self,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        """Store the :class:`RegionFuseConfig` (plus field overrides)."""
        import dataclasses

        from mlopt_torch.region_fuse import RegionFuseConfig

        cfg = config if config is not None else RegionFuseConfig()
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        self.config = cfg

    def apply(
        self,
        module: Any,
        example_input: Any,
        stats: dict[str, Any],
    ) -> Any:
        """Fuse *module* in place; record the ledger into *stats*."""
        from mlopt_torch.region_fuse import fuse_module

        rep = fuse_module(module, example_input, config=self.config)
        stats["region_fuse"] = rep
        return module
