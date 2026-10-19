"""The ONNX sink — ``mlopt_core.ports.Sink`` over ``onnx``/``onnxruntime``.

ADR 0006's first stage: a *second* backend behind the port the torch
adapter already implements.  The sink declares its op surface
(:attr:`OnnxSink.supported_ops`), lowers an IR to an
:class:`~mlopt_onnx.executor.OnnxExecutor`, and verifies two lowered
programs against each other in its own runtime.

What this module does *not* do is as important as what it does.  It
imports no torch, no ``mlopt_torch``, no ``mlopt_carriers``, no
``mlopt_orchestrator``: the ONNX backend is a peer of the torch one,
not a layer on top of it.  ``mlopt_core.cost.backend_cost`` prices
extraction against :attr:`supported_ops`, so the search only selects
forms this table can lower — a narrower table is a narrower search,
not a broken delivery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import onnx
from mlopt_core.ir import IR
from mlopt_core.ops import OpTable
from mlopt_core.ports import Executor, ExecutorSpec, OpRegistry

from mlopt_onnx.arrays import DTYPE_NAMES, WORKING_DTYPES
from mlopt_onnx.executor import DEFAULT_PROVIDERS, OnnxExecutor
from mlopt_onnx.lower import lower_ir
from mlopt_onnx.opmap import ONNX_BINDINGS
from mlopt_onnx.report import OnnxVerifyReport, verify_module

__all__ = ["OnnxSink"]


class OnnxSink:
    """``mlopt_core.ports.Sink`` over the ONNX mapping.

    Parameters
    ----------
    dtype : str
        The floating working dtype — ``"float32"`` (the interchange
        default) or ``"float64"``.  Floating leaves and graph inputs
        are cast to it; integer and boolean leaves keep their dtype.
    input_dtypes : Mapping[str, str] | None
        Per-input element types, keyed by ``Var`` name.  The IR types
        inputs by *shape* only, so a program whose input is an index
        tensor has to say so here (``{"x": "int64"}``).
    opset : int
        The ONNX opset the mapping targets.  Pinned deliberately:
        ONNX ops change semantics across opsets, so the sink claims
        exactly one.
    ir_version : int
        The ONNX IR version stamped on the model.  ONNX's own latest
        ``IR_VERSION`` outruns released runtimes, so the sink pins a
        version every current onnxruntime accepts.
    name : str
        The graph name.
    providers : tuple[str, ...]
        onnxruntime execution providers for the sessions this sink
        builds.

    """

    def __init__(
        self,
        *,
        dtype: str = "float32",
        input_dtypes: Mapping[str, str] | None = None,
        opset: int = 18,
        ir_version: int = 10,
        name: str = "mlopt",
        providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    ) -> None:
        """Configure the sink's dtype, opset and runtime."""
        if dtype not in WORKING_DTYPES:
            raise ValueError(
                f"unknown working dtype {dtype!r}; expected one of "
                f"{sorted(WORKING_DTYPES)}"
            )
        unknown = sorted(
            var
            for var, element in (input_dtypes or {}).items()
            if element not in DTYPE_NAMES
        )
        if unknown:
            raise ValueError(
                f"unknown input dtype name(s) for {unknown}; "
                f"expected one of {sorted(DTYPE_NAMES)}"
            )
        self.dtype = dtype
        self.input_dtypes = dict(input_dtypes or {})
        self.opset = opset
        self.ir_version = ir_version
        self.name = name
        self.providers = providers
        self._ops = _op_table()

    @property
    def supported_ops(self) -> frozenset[str]:
        """Every IR op name this sink can lower.

        The bound ``mlopt_core.cost.backend_cost`` prices extraction
        against: any member using an op outside it is priced at
        ``+inf`` and can never win extraction.
        """
        return frozenset(ONNX_BINDINGS)

    @property
    def ops(self) -> OpRegistry:
        """The lowering registry — an ``OpTable`` of ONNX builders.

        ``OpTable`` is core's own ``OpRegistry`` implementation and
        the port is declared backend-neutral, so the ONNX mapping
        seats its builders there rather than re-wrapping the surface.
        The member name ``torch_bindings`` is historical (see
        ``mlopt_core.ports.Binding``); the values are ONNX node
        emitters.  The pipeline reads ``supported_ops`` and the
        optional ``specialize_causal`` hook, not this table.
        """
        return self._ops

    @property
    def executors(self) -> Mapping[str, ExecutorSpec]:
        """The ONNX executor table — empty, by design.

        The table names *carrier* lowerings (batched scan / om / omd
        schedules) that a backend may route a term to.  ONNX has no
        carrier machinery and no per-term schedule choice: there is
        one lowering, so the table is empty and the pipeline falls
        through to :meth:`lower`.
        """
        return {}

    def build_model(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> onnx.ModelProto:
        """Lower ``ir`` to an ONNX model without opening a session.

        The artifact on its own — what a delivery writes to disk.
        """
        return lower_ir(
            ir,
            params,
            dtype=self.dtype,
            input_dtypes=self.input_dtypes,
            opset=self.opset,
            ir_version=self.ir_version,
            name=self.name,
        )

    def lower(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> Executor:
        """Materialise ``ir`` as an executable ONNX program.

        Raises :class:`~mlopt_onnx.errors.UnsupportedOnnxOpError`
        when the program uses an op (or a spelling) the mapping
        cannot express — never a silently narrower graph.
        """
        return OnnxExecutor(
            self.build_model(ir, params), providers=self.providers
        )

    def verify(
        self,
        ref: Any,
        opt: Any,
        inputs: Any,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> OnnxVerifyReport:
        """Run ``ref`` and ``opt`` on ``inputs`` and compare.

        Both sides are runnables in *this* sink's runtime — the
        equivalence claim is stated where the delivery lives.
        """
        return verify_module(ref, opt, inputs, rtol=rtol, atol=atol)


def _op_table() -> OpTable:
    """Build the sink's ``OpRegistry`` view of the ONNX mapping.

    ``OpTable.core()`` supplies the backend-neutral fragments (the
    shape rules and the attribute schema); the lowering entries are
    replaced wholesale with the ONNX builders, so nothing torch-side
    leaks into ``supported_ops``.
    """
    table = OpTable.core()
    table.torch_bindings = dict(ONNX_BINDINGS)
    return table
