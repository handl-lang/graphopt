"""host-tensor backend for mlopt — ``mlopt_core.ports.Sink`` over ``tensor.*``.

Phase 1 of plan 0036 (``docs/reports/host_tensor_sink_phase0.md`` in the
graphopt repo): lowers the liftable-now slice of the mlopt IR vocabulary
(elementwise / reduce-dim / linalg / shape / constructors) onto the
``handl-lang/host-tensor`` Rust op registry through its PyO3 binding
(``handl_host_tensor.call``).  The codec is the f64 row-major wire
record ``{"__tensor__": {"shape": [...], "data": [...], "dtype": "f64"}}``
on both sides.

Honest boundaries (phase-0 §1, updated for host-tensor ``0528649``):
the codec is f64-only — non-``float64`` inputs are upcast with a
warning, never silently — and non-finite elements (NaN/±inf) cross as
``{"__nonfinite__": …}`` tag objects in ``data``, round-tripping
exactly in both directions — see :mod:`mlopt_host_tensor.wire`.  Ops
outside :attr:`HostTensorSink.supported_ops` fail loudly at
lower/eval; the sink never silently approximates.
"""

from mlopt_host_tensor.sink import HostTensorSink

__all__ = ["HostTensorSink"]
