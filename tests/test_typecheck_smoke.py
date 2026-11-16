"""Smoke test: the domain packages and their public API import cleanly.

Pairs with the ty ratchet in pyproject.toml — if module-level code
breaks at import time, this fails before any typecheck output matters.
Plan 0008 removed the old ``mlopt`` façade; the D-2 merge reinstated
``packages/mlopt`` as a workspace member, so ``import mlopt`` must
succeed — but as a composition-only product package that re-exports
nothing from the substrate/domain packages (their names live at
their real package paths).
"""

import importlib
import sys


def test_modules_import():
    import mlopt_core
    import mlopt_orchestrator
    import kit_torch
    from mlopt_core.egraph import EGraph
    from mlopt_core.ir import IR
    from mlopt_orchestrator import Optimizer



    from kit_torch.torch_bridge import IRModule

    assert mlopt_core and mlopt_orchestrator and kit_torch
    assert EGraph and IR and IRModule
    assert Optimizer


def test_submodules_import():
    import mlopt_core.egraph
    import mlopt_core.ir
    import mlopt_core.laws
    import mlopt_orchestrator.optimize

    assert mlopt_core.egraph.ENode
    assert mlopt_orchestrator.optimize.Optimizer


def test_mlopt_facade_is_composition_only():
    """Post-D-2 ``mlopt`` imports cleanly but exposes no substrate
    names — ``__init__`` carries only the product version."""
    sys.modules.pop("mlopt", None)
    mlopt = importlib.import_module("mlopt")
    assert mlopt.__version__
    assert not hasattr(mlopt, "Optimizer")
    assert not hasattr(mlopt, "EGraph")
    assert not hasattr(mlopt, "IR")
    assert not hasattr(mlopt, "mlopt_core")
    assert not hasattr(mlopt, "kit_torch")
