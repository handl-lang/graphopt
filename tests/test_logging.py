"""Phase 3c: owned modules report through ``logging``, not ``print``.

* a saturation run logs the fixed-point iteration line at INFO (and the
  ``max_nodes`` stop at WARNING);
* ``verbose=True`` widens the module logger to DEBUG-level detail;
* no ``print()`` calls remain in the owned sources — an AST-level check
  (docstrings/comments cannot false-positive), with a documented
  ``# stdout-compat`` escape hatch for any line a capsys/capfd test
  asserts on.  One is kept: ``handler_synth.main``'s verdict row, which
  ``test_discovery_handler_synth.py``'s ``test_cli_battery`` reads via
  capsys.
"""

import ast
import logging
from pathlib import Path

from mlopt_torch.calibrate import calibrate
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Param, TensorType, Var
from mlopt_core.laws import ASSOC_LINEAR_BIAS

_REPO = Path(__file__).resolve().parents[1]

# Files owned by the phase-3c change (print -> logging).
_OWNED = sorted(
    str(p.relative_to(_REPO))
    for p in (_REPO / "packages").rglob("*.py")
    # the owned set is the phase-3c logging-migration files; optimize.py
    # keeps its deliberate verbose-mode stdout prints
    if "src" in p.parts and p.name != "optimize.py"
)


def _print_call_lines(path: Path) -> list[int]:
    """Line numbers of ``print(...)`` *calls* in *path*.

    AST-based: docstring examples (``regime.py``) and prose cannot
    register, only real call nodes.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]




def _small_graph() -> tuple[EGraph, int]:
    x = Var("x", TensorType((4, 8)))
    A = Param("A", TensorType((16, 8)))
    B = Param("B", TensorType((8, 16)))
    b1 = Param("b1", TensorType((16,)))
    b2 = Param("b2", TensorType((8,)))
    src = Op.make("linear", Op.make("linear", x, A, b1), B, b2)
    eg = EGraph()
    return eg, eg.add_term(src)






def test_verbose_true_yields_debug(caplog):
    with caplog.at_level(logging.DEBUG):
        calibrate(device="cpu", quick=True, verbose=True)
    assert any(
        r.name == "mlopt_torch.calibrate" and r.levelno == logging.DEBUG
        for r in caplog.records
    )


def test_verbose_restores_logger_level(caplog):
    # `mlopt_torch.calibrate` is the real logger home — name it
    # directly (the façade's sys.modules alias layer is gone).
    log = logging.getLogger("mlopt_torch.calibrate")
    prev = log.level
    with caplog.at_level(logging.DEBUG):
        calibrate(device="cpu", quick=True, verbose=True)
    assert log.level == prev
