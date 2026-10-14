"""The ONNX source — ``catopt_core.ports.Source`` over ``onnx``.

ADR 0006's second stage: ONNX graphs *in*.  ``OnnxSource`` implements
the ``Source`` port — ``model -> (IR, leaves)`` — over an
``onnx.ModelProto`` (or a ``.onnx`` path), letting the optimizer
touch models that are not torch at all.  It is the peer of
``catopt_torch.adapters.TorchSource``, driven through the same port
and importing nothing from the torch family.

What this module does *not* do is as important as what it does.  The
import is not a parse-and-hope: :mod:`catopt_onnx.liftmap` is a
deliberately coherent subset of the ONNX op set, and everything
outside it — unsupported ops, computed (non-constant) attr-fold
inputs, non-tensor outputs, opsets outside the pinned window — is
*refused*, never silently dropped.  The ADR's rule runs both ways:
a silently narrower import would be a different program wearing the
original's name.

The dtype side-channel
----------------------
``Var.typ`` is shape-only — the port's ``(IR, leaves)`` return has
no element-type channel, the same gap the sink works around with its
``input_dtypes`` map.  ONNX inputs *do* declare element types, so
the source publishes them on :attr:`input_dtypes` after each
import — a parallel structure the caller wires back in:
``OnnxSink(input_dtypes=source.input_dtypes)`` keeps an ``int64``
index input an index input instead of a float tensor.  This is the
documented source-side workaround for the gap ADR 0006 measured
(``Source`` carries no element type), not a widening of core.
"""

from __future__ import annotations

from typing import Any

from catopt_core.ir import IR

from catopt_onnx.lift import (
    MAX_OPSET,
    MIN_OPSET,
    LiftResult,
    lift_model,
)

__all__ = ["MAX_OPSET", "MIN_OPSET", "OnnxSource"]


class OnnxSource:
    """``catopt_core.ports.Source`` over ONNX models.

    Parameters
    ----------
    None — the import's policy (the opset window, the refusal rules)
    is the table's own, not a knob.  The opset window itself is
    exposed as :data:`MIN_OPSET`/:data:`MAX_OPSET` for callers that
    want to probe before importing.

    Attributes
    ----------
    input_dtypes : dict[str, str]
        The element types the last import declared on the graph's
        inputs, keyed by ``Var`` name — see the module docstring for
        why this lives on the source rather than in the IR.

    """

    def __init__(self) -> None:
        """Initialise the source's per-import record."""
        self.input_dtypes: dict[str, str] = {}

    def to_ir(
        self, model: Any, example_inputs: Any = None
    ) -> tuple[IR, dict[str, Any]]:
        """Lift ``model`` (a ModelProto or ``.onnx`` path) to IR.

        Returns ``(ir, leaves)`` — the IR plus the concrete
        initializer/constant values keyed by ``Param`` name, the same
        pair ``TorchSource.to_ir`` returns as ``source_tensors``.
        ``example_inputs`` is ignored: ONNX graphs declare their own
        shapes (the lifter reads them through ONNX's shape
        inference), so the port's second argument has no role here —
        it is accepted so the ``Optimizer`` call shape is uniform.
        """
        result = lift_model(model)
        self.input_dtypes = dict(result.input_dtypes)
        return result.ir, result.leaves

    def lift(self, model: Any) -> LiftResult:
        """Lift ``model`` and return the full :class:`LiftResult`.

        The richer entry point when the caller wants ``input_dtypes``
        without reaching into the source's attribute — same import,
        one return value.
        """
        result = lift_model(model)
        self.input_dtypes = dict(result.input_dtypes)
        return result
