"""``mlopt_kernel`` — the kernelopt spike package (plan 0032 phase 3).

A second domain for the game layer, built entirely on the substrate
side: a small *codegen* law bank (:mod:`mlopt_kernel.laws` — named
fused-kernel ops and arithmetic strength reductions, not a
language), a :class:`~mlopt_kernel.sink.KernelSink` implementing the
:class:`mlopt_core.ports.Sink` port over a torch composite
(verify-first, per the ONNX precedent), a launch/weight-dominated
:func:`~mlopt_kernel.cost.kernel_cost`, and
:class:`~mlopt_kernel.board.KernelBoard` — the
``gamekit.engine.Board``-shaped episode the ``kernel`` domain in
``mlopt_discovery.play`` boards cases on.
"""

from mlopt_kernel.board import KernelBoard, KernelReport, KernelState
from mlopt_kernel.cost import kernel_cost
from mlopt_kernel.laws import FUSED_KERNEL_OPS, KERNEL_RULES
from mlopt_kernel.sink import (
    KernelExecutor,
    KernelSink,
    KernelVerifyReport,
)

__all__ = [
    "FUSED_KERNEL_OPS",
    "KERNEL_RULES",
    "KernelBoard",
    "KernelExecutor",
    "KernelReport",
    "KernelSink",
    "KernelState",
    "KernelVerifyReport",
    "kernel_cost",
]
