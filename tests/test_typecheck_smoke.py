"""Smoke test: the domain packages and their public API import cleanly.

Pairs with the ty ratchet in pyproject.toml — if module-level code
breaks at import time, this fails before any typecheck output matters.
Plan 0008 removed the ``mlopt`` façade: ``import mlopt`` must fail
and every public name lives at its real package path.
"""

import importlib
import sys

import pytest


def test_modules_import():
    import mlopt_core
    import mlopt_orchestrator
    import mlopt_torch
    from mlopt_core.egraph import EGraph
    from mlopt_core.ir import IR
    from mlopt_orchestrator import Optimizer



    from mlopt_torch.torch_bridge import IRModule

    assert mlopt_core and mlopt_orchestrator and mlopt_torch
    assert EGraph and IR and IRModule
    assert Optimizer


def test_submodules_import():
    import mlopt_core.egraph
    import mlopt_core.ir
    import mlopt_core.laws
    import mlopt_orchestrator.optimize
    import mlopt_torch.models
    import mlopt_torch.torch_bridge

    assert mlopt_core.egraph.ENode
    assert mlopt_orchestrator.optimize.Optimizer


def test_mlopt_facade_is_gone():
    """Plan 0008 deleted the ``mlopt`` façade — the name fails."""
    sys.modules.pop("mlopt", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mlopt")
