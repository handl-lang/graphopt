"""The tensor wire codec — ``{"__tensor__": {shape, data, dtype}}``.

Identical record on both sides of the boundary (host-tensor
``src/ops/codec.rs``, lib-tensor ``codec.hnl``, ADR-0083 §2.1): dense
row-major ``f64`` payload, ``dtype`` optional on decode and ``"f64"``
the only accepted value.

Known holes, handled honestly rather than silently:

* **Non-finite inputs.** ``serde_json`` has no NaN/Inf literals, so a
  non-finite *input* tensor cannot cross ``call`` — :func:`to_wire`
  refuses it loudly.  (Rust-side ``f64`` ops *can* produce
  inf/NaN — ``serde_json`` then emits ``null`` in ``data``; the
  decoder maps ``null`` back to ``nan``, which is the closest honest
  reading and matches the codec's documented finite-only wire.)
* **Index dtypes.** ``argmax``/``argsort``-family results are f64 on
  the wire (no int dtype); consumers must not dtype-check them.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from mlopt_host_tensor.errors import HostTensorError

WIRE_TAG = "__tensor__"


def to_wire(a: Any) -> dict[str, Any]:
    """Encode an array-like as the ``{"__tensor__": …}`` wire record.

    Values are upcast to ``float64`` (lossless for f32/bf16 sources).
    Non-finite elements raise :class:`HostTensorError` — the JSON wire
    cannot carry them (see module docstring).
    """
    arr = np.asarray(a, dtype=np.float64)
    if arr.size and not np.isfinite(arr).all():
        raise HostTensorError(
            "non-finite values cannot cross the host-tensor wire "
            "(serde_json has no NaN/Inf literals)"
        )
    return {
        WIRE_TAG: {
            "shape": [int(d) for d in arr.shape],
            "data": arr.reshape(-1).tolist(),
            "dtype": "f64",
        }
    }


def from_wire(v: Any) -> np.ndarray:
    """Decode a wire record (or array-like) to a float64 ndarray."""
    if isinstance(v, dict) and WIRE_TAG in v:
        rec = v[WIRE_TAG]
        data = rec["data"]
        # `null` is the wire's non-finite marker (serde_json emits it
        # for NaN/±inf) — decode as nan, the documented hole.
        flat = np.array(
            [np.nan if x is None else float(x) for x in data],
            dtype=np.float64,
        )
        return flat.reshape([int(d) for d in rec["shape"]])
    return np.asarray(v, dtype=np.float64)


def tensor_arg(v: Any) -> Any:
    """Encode an evaluated operand for a Tensor-typed arg slot.

    ndarrays wire-encode; Python scalars wrap as rank-0 tensors (the
    broadcast-elementwise kernels stretch them like torch's 0-dim
    Const operands).
    """
    if isinstance(v, np.ndarray):
        return to_wire(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return to_wire(np.asarray(float(v)))
    raise HostTensorError(
        f"cannot encode {type(v).__name__} as tensor arg"
    )


def scalar_or_tensor(v: Any) -> Any:
    """Encode for ``Scalar | Tensor`` slots: numbers stay numbers."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    if isinstance(v, np.ndarray):
        return to_wire(v)
    raise HostTensorError(
        f"cannot encode {type(v).__name__} as scalar|tensor arg"
    )


def scalar(v: Any) -> float | int:
    """Extract a Python number from a scalar operand value."""
    if isinstance(v, np.ndarray):
        if v.size != 1:
            raise HostTensorError(
                f"expected a scalar operand, got shape {v.shape}"
            )
        v = v.reshape(-1)[0]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    raise HostTensorError(f"cannot read {type(v).__name__} as scalar")


def decode_result(text: str) -> Any:
    """Decode a ``call`` result JSON string to Python values.

    * ``{"__tensor__": …}`` → float64 ndarray
    * ``[{wire}, …]`` → tuple of ndarrays (multi-output ops)
    * ``{"values": wire, "indices": wire}`` → tuple (max/min.dim,
      sort — getitem consumers are out of scope for phase 1)
    * anything else → the JSON scalar/list as-is
    """
    v = json.loads(text)
    return _decode(v)


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        if WIRE_TAG in v:
            return from_wire(v)
        if set(v) == {"values", "indices"}:
            return (from_wire(v["values"]), from_wire(v["indices"]))
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return tuple(_decode(x) for x in v)
    return v
