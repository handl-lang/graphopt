"""Refusal errors — the ONNX mapping declines what it cannot express.

ADR 0006 states the rule the whole package is built around: *refuse,
never drop*.  An IR op the mapping cannot express raises
:class:`UnsupportedOnnxOpError` rather than vanishing from the graph,
so a lowered artifact is either the whole program or no artifact at
all — a silently narrower graph would be a different program wearing
the original's name.
"""

from __future__ import annotations

__all__ = [
    "OnnxImportError",
    "OnnxLoweringError",
    "UnsupportedOnnxOpError",
]


class OnnxLoweringError(Exception):
    """Base class for every ONNX lowering failure."""


class OnnxImportError(Exception):
    """Base class for ONNX -> IR import failures (direction 2).

    Structural problems rather than op coverage: a declared opset
    outside the pinned window, a graph input with no tensor type, a
    non-tensor graph output, an output with no producer.  Op-level
    refusals raise :class:`UnsupportedOnnxOpError` — the same error
    the sink raises, since it is the same mapping in the other
    direction.
    """


class UnsupportedOnnxOpError(OnnxLoweringError):
    """An IR op (or op spelling) the ONNX mapping cannot express.

    ``op`` is the IR op name; ``detail`` names the specific reason
    when the op *is* mapped but this spelling is not (an unknown
    ``rounding_mode``, a missing static rank, an attribute value the
    ONNX op has no equivalent for).
    """

    def __init__(self, op: str, detail: str = "") -> None:
        """Record the offending op and an optional detail."""
        self.op = op
        self.detail = detail
        why = f" — {detail}" if detail else ""
        super().__init__(
            f"unsupported op {op!r}{why}: the ONNX mapping cannot "
            f"express it, so the model is refused rather than "
            f"silently narrowed (ADR 0006)"
        )
