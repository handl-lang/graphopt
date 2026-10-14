"""ONNX backend for catopt — ADR 0006, direction 1 (IR -> ONNX).

The first non-torch backend.  ``OnnxSink`` implements
``catopt_core.ports.Sink``: it declares the op surface the search may
select from, lowers a catopt IR to an ONNX graph, and verifies two
lowered programs against each other through ``onnxruntime``.

The experiment this package *is*: ADR 0003/0006 claim that a new
backend implements ``Sink`` and nothing in ``catopt-core`` changes.
The port boundary is what is under test, so the honest reading is
that the code here is half the deliverable and the answer to "did
core have to move?" is the other half.

    >>> from catopt_onnx import OnnxSink
    >>> sink = OnnxSink(dtype="float64")     # doctest: +SKIP
    >>> model = sink.build_model(ir, params) # doctest: +SKIP
    >>> executor = sink.lower(ir, params)    # doctest: +SKIP

Nothing here imports torch, ``catopt_torch``, ``catopt_carriers`` or
``catopt_orchestrator``: the ONNX backend is a *peer* of the torch
one, reachable through the same ports and depending only on
``catopt_core``.
"""

from __future__ import annotations

from catopt_onnx.arrays import (
    WORKING_DTYPES,
    numpy_to_onnx_type,
    to_numpy,
)
from catopt_onnx.errors import (
    OnnxLoweringError,
    UnsupportedOnnxOpError,
)
from catopt_onnx.executor import OnnxExecutor
from catopt_onnx.lower import GraphBuilder, lower_ir
from catopt_onnx.opmap import ONNX_BINDINGS
from catopt_onnx.report import (
    REL_FLOOR,
    OnnxVerifyReport,
    verify_equiv,
    verify_module,
)
from catopt_onnx.sink import OnnxSink

__all__ = [
    "ONNX_BINDINGS",
    "REL_FLOOR",
    "WORKING_DTYPES",
    "GraphBuilder",
    "OnnxExecutor",
    "OnnxLoweringError",
    "OnnxSink",
    "OnnxVerifyReport",
    "UnsupportedOnnxOpError",
    "lower_ir",
    "numpy_to_onnx_type",
    "to_numpy",
    "verify_equiv",
    "verify_module",
]
