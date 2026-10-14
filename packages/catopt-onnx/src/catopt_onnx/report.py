"""The ONNX equivalence gate — NumPy, in the sink's own runtime.

``Sink.verify`` runs a reference and an optimized executable in *the
sink's* runtime; for the ONNX sink that runtime is ``onnxruntime``
and the comparison is NumPy.  The metric is deliberately the one
``catopt_torch.report`` already uses — ``max_rel = max|ref - out| /
(max|ref| + 1e-8)`` with ``passed = max_rel < rtol`` — so a delivery
gate means the same thing whichever backend lowered it.

Core names neither: :class:`OnnxVerifyReport` satisfies
``catopt_core.ports.VerifyResult`` structurally (``max_abs`` /
``max_rel`` / ``passed``), and ``catopt_core`` never imports this
module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "REL_FLOOR",
    "OnnxVerifyReport",
    "verify_equiv",
    "verify_module",
]

#: Denominator floor of the historical rel-diff metric — the torch
#: adapter's ``report._REL_FLOOR``, restated so a backend comparison
#: is the same statement about the same numbers.
REL_FLOOR = 1e-8


@dataclass(frozen=True)
class OnnxVerifyReport:
    """Result of one ONNX equivalence check.

    ``max_abs`` is the largest absolute difference over every output
    leaf; ``max_rel`` divides it by the largest reference magnitude
    (plus :data:`REL_FLOOR`); ``passed`` is the gate outcome.
    """

    max_abs: float
    max_rel: float
    passed: bool


def _leaves(out: Any) -> list[np.ndarray]:
    """Flatten an executor result to its array leaves."""
    if isinstance(out, (tuple, list)):
        return [np.asarray(o) for o in out]
    return [np.asarray(out)]


def _peak(arr: np.ndarray) -> float:
    """Return ``max|arr|``, or 0.0 for an empty tensor."""
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def verify_equiv(
    ref_out: Any,
    out: Any,
    rtol: float = 1e-4,
    atol: float | None = None,
) -> OnnxVerifyReport:
    """Compare two already-computed ONNX outputs.

    Gate semantics mirror ``catopt_torch.report.verify_equiv`` (not
    ``numpy.allclose``'s elementwise ``atol + rtol * |ref|``):

    * ``max_rel = max|ref - out| / (max|ref| + 1e-8)``
    * ``passed = max_rel < rtol``, and additionally
      ``max_abs <= atol`` when ``atol`` is given.

    Both sides may be a single array or a tuple of arrays (a
    multi-output graph); the leaf counts must agree.
    """
    refs = _leaves(ref_out)
    outs = _leaves(out)
    if len(refs) != len(outs):
        raise ValueError(
            f"output arity mismatch: reference has {len(refs)} "
            f"leaf/leaves, candidate has {len(outs)}"
        )
    max_abs = max(
        (_peak(r - o) for r, o in zip(refs, outs, strict=True)),
        default=0.0,
    )
    scale = max((_peak(r) for r in refs), default=0.0)
    max_rel = max_abs / (scale + REL_FLOOR)
    passed = max_rel < rtol and (atol is None or max_abs <= atol)
    return OnnxVerifyReport(max_abs, max_rel, passed)


def _run(mod: Any, args: Sequence[Any]) -> Any:
    """Run a lowered executor on positional ``args`` (duck-typed)."""
    fn = getattr(mod, "forward", None)
    if not callable(fn):
        fn = mod
    return fn(*args)


def verify_module(
    ref: Any,
    opt: Any,
    inputs: Any,
    *,
    rtol: float = 1e-4,
    atol: float | None = None,
) -> OnnxVerifyReport:
    """Run ``ref`` and ``opt`` on ``inputs``; compare their outputs.

    ``inputs`` is a positional-args tuple or a single value (wrapped
    into a one-tuple, matching the orchestrator's call sites).
    """
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    return verify_equiv(
        _run(ref, args), _run(opt, args), rtol=rtol, atol=atol
    )
