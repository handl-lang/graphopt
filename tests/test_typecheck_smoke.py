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
from types import ModuleType


def _mlopt_modules() -> dict[str, ModuleType]:
    """Every ``sys.modules`` entry for ``mlopt`` and its submodules."""
    return {
        name: mod
        for name, mod in sys.modules.items()
        if name == "mlopt" or name.startswith("mlopt.")
    }


def _purge_mlopt_modules() -> None:
    """Drop ``mlopt`` and every ``mlopt.*`` submodule from ``sys.modules``.

    Popping only the parent leaves submodule objects behind with stale
    parent references — a later ``import mlopt.x`` returns the orphaned
    module while ``import mlopt`` builds a fresh package, and attribute
    lookups across the two copies fail.
    """
    for name in _mlopt_modules():
        sys.modules.pop(name, None)


def _fresh_mlopt() -> ModuleType:
    """Import ``mlopt`` against an empty module table, then restore it.

    pytest imports every test module during collection — including
    ``packages/mlopt/tests``, whose module-level ``from mlopt.x import …``
    binds one ``mlopt.*`` instance.  Leaving the fresh instances this
    test creates behind would split module identity (``isinstance`` /
    monkeypatch targets diverge), so the pre-test ``sys.modules`` entries
    are restored on exit.
    """
    saved = _mlopt_modules()
    _purge_mlopt_modules()
    try:
        return importlib.import_module("mlopt")
    finally:
        _purge_mlopt_modules()
        sys.modules.update(saved)


def test_modules_import():
    import kit_torch
    import mlopt_core
    import mlopt_orchestrator
    from kit_torch.torch_bridge import IRModule
    from mlopt_core.egraph import EGraph
    from mlopt_core.ir import IR
    from mlopt_orchestrator import Optimizer

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
    mlopt = _fresh_mlopt()
    assert mlopt.__version__
    assert not hasattr(mlopt, "Optimizer")
    assert not hasattr(mlopt, "EGraph")
    assert not hasattr(mlopt, "IR")
    assert not hasattr(mlopt, "mlopt_core")
    assert not hasattr(mlopt, "kit_torch")


def test_purge_mlopt_modules_clears_submodules():
    """Regression: the facade test must drop ``mlopt.*`` too, not just
    the parent — orphaned submodules break later ``mlopt.x`` imports."""
    saved = _mlopt_modules()
    try:
        import mlopt.weights  # light submodule, no heavy deps

        assert any(n.startswith("mlopt.") for n in sys.modules)
        _purge_mlopt_modules()
        assert not any(
            n == "mlopt" or n.startswith("mlopt.") for n in sys.modules
        )
        mlopt = importlib.import_module("mlopt")
        assert mlopt.__version__
    finally:
        _purge_mlopt_modules()
        sys.modules.update(saved)
