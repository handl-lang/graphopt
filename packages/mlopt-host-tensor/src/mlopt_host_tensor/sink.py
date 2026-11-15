"""The host-tensor sink — ``mlopt_core.ports.Sink`` over ``tensor.*``.

Phase 1 of plan 0036 (``docs/reports/host_tensor_sink_phase0.md``):
a *second* concrete backend behind the port the torch and ONNX
adapters already implement.  The sink declares its op surface
(:attr:`HostTensorSink.supported_ops` — exactly the 138 liftable-now
ops), lowers an IR to a :class:`~mlopt_host_tensor.executor.
HostTensorExecutor`, and verifies two lowered programs against each
other in its own runtime (the Rust op registry + NumPy).

What this module does *not* do is as important as what it does.  It
imports no torch, no ``mlopt_torch``, no ``mlopt_orchestrator``: the
host-tensor backend is a peer, not a layer.  ``supported_ops`` is the
extraction bound — ``mlopt_core.cost.backend_cost`` prices any member
using an op outside it at ``+inf`` — and :meth:`lower` declines
loudly (:class:`~mlopt_host_tensor.errors.UnsupportedHostOpError`)
rather than approximating.

The backend's runtime is the ``handl_host_tensor`` PyO3 extension
built from the sibling ``handl-lang/host-tensor`` checkout
(``cargo build --features python``); see
:mod:`mlopt_host_tensor._binding` for how the .so is located.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mlopt_core.ir import IR
from mlopt_core.ops import OpTable
from mlopt_core.ports import Executor, ExecutorSpec, OpRegistry

from mlopt_host_tensor.bindings import HOST_BINDINGS
from mlopt_host_tensor.errors import (
    HostTensorError,
    UnsupportedHostOpError,
)
from mlopt_host_tensor.executor import HostTensorExecutor
from mlopt_host_tensor.report import (
    HostTensorVerifyReport,
    verify_module,
)

__all__ = ["HostTensorSink"]


class HostTensorSink:
    """``mlopt_core.ports.Sink`` over the host-tensor op registry.

    Parameters
    ----------
    params_check : bool
        When ``True`` (default), :meth:`lower` refuses an IR whose
        ``Param`` leaves have no entry in ``params`` — the failure
        surfaces at lower time, not mid-``forward``.

    """

    def __init__(self, *, params_check: bool = True) -> None:
        """Configure the sink."""
        self._params_check = params_check
        self._ops = _op_table()

    @property
    def supported_ops(self) -> frozenset[str]:
        """Every IR op name this sink can lower.

        The 138 liftable-now ops of the phase-0 split.  The bound
        ``mlopt_core.cost.backend_cost`` prices extraction against it:
        any member using an op outside it is priced at ``+inf`` and
        can never win extraction.
        """
        return frozenset(HOST_BINDINGS)

    @property
    def ops(self) -> OpRegistry:
        """The lowering registry — an ``OpTable`` of host bindings.

        ``OpTable`` is core's own ``OpRegistry`` implementation; the
        member name ``torch_bindings`` is historical (see
        ``mlopt_core.ports.Binding``).  The values here are *arg-list
        builders*, not evaluators — the executor resolves them through
        the extension, so a search-time const fold that called them
        directly would get args, not values.  That matches the ONNX
        sink's precedent (its builders emit nodes); consumers that
        need evaluable bindings use the torch sink.
        """
        return self._ops

    @property
    def executors(self) -> Mapping[str, ExecutorSpec]:
        """The host-tensor executor table — empty, by design.

        There is one lowering and no carrier machinery, so the table
        is empty and the pipeline falls through to :meth:`lower`.
        """
        return {}

    def lower(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> Executor:
        """Materialise ``ir`` as a host-tensor executor.

        Raises :class:`~mlopt_host_tensor.errors.UnsupportedHostOpError`
        when the program uses an op outside :attr:`supported_ops`, and
        :class:`~mlopt_host_tensor.errors.HostTensorError` when a
        ``Param`` leaf has no value (``params_check``).
        """
        leaves = dict(params or {})
        if self._params_check:
            missing = sorted(
                p.name
                for p in _param_leaves(ir.root)
                if p.name not in leaves
            )
            if missing:
                raise HostTensorError(
                    f"host-tensor sink: no leaf value for param(s) "
                    f"{missing} — pass params= to lower()"
                )
        inputs = tuple(ir.inputs)
        # Reject unsupported ops at lower time — walk the term once.
        _check_supported(ir.root, frozenset(HOST_BINDINGS))
        return HostTensorExecutor(ir.root, inputs, leaves)

    def verify(
        self,
        ref: Any,
        opt: Any,
        inputs: Any,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> HostTensorVerifyReport:
        """Run ``ref`` and ``opt`` on ``inputs`` and compare.

        Both sides are runnables in *this* sink's runtime.
        """
        return verify_module(ref, opt, inputs, rtol=rtol, atol=atol)


def _param_leaves(term: Any) -> list[Any]:
    """Every ``Param`` leaf under ``term``.

    ``IR.params`` is not auto-populated — walk the term.
    """
    from mlopt_core.ir import Param

    out: list[Any] = []
    seen: set[int] = set()

    def walk(t: Any) -> None:
        if id(t) in seen:
            return
        seen.add(id(t))
        if isinstance(t, Param):
            out.append(t)
            return
        args = getattr(t, "args", None)
        if args:
            for a in args:
                walk(a)

    walk(term)
    return out


def _check_supported(term: Any, supported: frozenset[str]) -> None:
    """Walk ``term``; raise on the first op outside ``supported``.

    ``pack`` is the IR-internal multi-output pairing — expanded by the
    evaluator, never sent to a host op — so it passes the check.
    """
    from mlopt_core.ir import Op

    seen: set[int] = set()

    def walk(t: Any) -> None:
        if id(t) in seen:
            return
        seen.add(id(t))
        if not isinstance(t, Op):
            return
        if t.op != "pack" and t.op not in supported:
            raise UnsupportedHostOpError(
                f"host-tensor sink: op {t.op!r} is outside "
                f"supported_ops ({len(supported)} ops)"
            )
        for a in t.args:
            walk(a)

    walk(term)


def _op_table() -> OpTable:
    """Build the sink's ``OpRegistry`` view of the host bindings.

    ``OpTable.core()`` supplies the backend-neutral fragments (shape
    rules, attr schema); the lowering entries are replaced wholesale
    with the host builders.
    """
    table = OpTable.core()
    table.torch_bindings = dict(HOST_BINDINGS)
    return table
