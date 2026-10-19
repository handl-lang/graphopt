"""ONNX backend for mlopt — ADR 0006, both directions.

The first non-torch backend, now complete.  ``OnnxSink`` implements
``mlopt_core.ports.Sink`` (direction 1: IR -> ONNX, verified through
``onnxruntime``); ``OnnxSource`` implements
``mlopt_core.ports.Source`` (direction 2: ONNX graph -> IR, over
:mod:`mlopt_onnx.liftmap` — the inverse op vocabulary of the sink's
:mod:`mlopt_onnx.opmap`).  Together they close the loop the ADR
staged: ONNX models enter the optimizer untouched by torch, and
optimized programs leave for any ONNX runtime.

The experiment this package *is*: ADR 0003/0006 claim that a new
backend implements the ports and nothing in ``mlopt-core`` changes.
The port boundary is what is under test, so the honest reading is
that the code here is half the deliverable and the answer to "did
core have to move?" is the other half.

    >>> from mlopt_onnx import OnnxSink, OnnxSource
    >>> source = OnnxSource()                # doctest: +SKIP
    >>> ir, leaves = source.to_ir(model)     # doctest: +SKIP
    >>> sink = OnnxSink(dtype="float64",
    ...                 input_dtypes=source.input_dtypes)  # +SKIP
    >>> executor = sink.lower(ir, leaves)    # doctest: +SKIP

Nothing here imports torch, ``mlopt_torch``, ``mlopt_carriers`` or
``mlopt_orchestrator``: the ONNX backend is a *peer* of the torch
one, reachable through the same ports and depending only on
``mlopt_core``.
"""

from __future__ import annotations

from mlopt_onnx.arrays import (
    ONNX_TYPE_NAMES,
    WORKING_DTYPES,
    numpy_to_onnx_type,
    to_numpy,
)
from mlopt_onnx.errors import (
    OnnxImportError,
    OnnxLoweringError,
    UnsupportedOnnxOpError,
)
from mlopt_onnx.executor import OnnxExecutor
from mlopt_onnx.lift import GraphLifter, LiftResult, lift_model
from mlopt_onnx.liftmap import ONNX_LIFTS
from mlopt_onnx.lower import GraphBuilder, lower_ir
from mlopt_onnx.opmap import ONNX_BINDINGS
from mlopt_onnx.report import (
    REL_FLOOR,
    OnnxVerifyReport,
    verify_equiv,
    verify_module,
)
from mlopt_onnx.sink import OnnxSink
from mlopt_onnx.source import MAX_OPSET, MIN_OPSET, OnnxSource

__all__ = [
    "MAX_OPSET",
    "MIN_OPSET",
    "ONNX_BINDINGS",
    "ONNX_LIFTS",
    "ONNX_TYPE_NAMES",
    "REL_FLOOR",
    "WORKING_DTYPES",
    "GraphBuilder",
    "GraphLifter",
    "LiftResult",
    "OnnxExecutor",
    "OnnxImportError",
    "OnnxLoweringError",
    "OnnxSink",
    "OnnxSource",
    "OnnxVerifyReport",
    "UnsupportedOnnxOpError",
    "lift_model",
    "lower_ir",
    "numpy_to_onnx_type",
    "to_numpy",
    "verify_equiv",
    "verify_module",
]
