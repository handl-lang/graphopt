"""mlopt-core — the dependency-free semantic engine.

The categorical optimizer's abstract core: typed term IR
(:mod:`~mlopt_core.ir`), canonical attribute contract
(:mod:`~mlopt_core.attrs`), shape/type inference
(:mod:`~mlopt_core.typing`), the e-graph engine
(:mod:`~mlopt_core.egraph`), equational + non-local laws
(:mod:`~mlopt_core.laws`), cost algebra
(:mod:`~mlopt_core.cost`), rule synthesis
(:mod:`~mlopt_core.meta`), the ``OpTable`` adapter registry
(:mod:`~mlopt_core.ops`), and the ports layer
(:mod:`~mlopt_core.ports`).  No torch, no numpy — pure Python.
"""

from mlopt_core.egraph import EGraph, ENode, Rewrite
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var
from mlopt_core.laws import CATEGORICAL_RULES, RuleSet

__all__ = [
    "CATEGORICAL_RULES",
    "IR",
    "Const",
    "EGraph",
    "ENode",
    "Op",
    "Param",
    "Rewrite",
    "RuleSet",
    "TensorType",
    "Var",
]
