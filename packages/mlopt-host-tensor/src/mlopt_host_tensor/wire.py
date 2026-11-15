"""The tensor wire codec — ``{"__tensor__": {shape, data, dtype}}``.

Identical record on both sides of the boundary (host-tensor
``src/ops/codec.rs``, lib-tensor ``codec.hnl``, ADR-0083 §2.1): dense
row-major ``f64`` payload, ``dtype`` optional on decode and ``"f64"``
the only accepted value.

Known holes, handled honestly rather than silently:

* **Non-finite elements.** ``serde_json`` has no NaN/Inf literals, so
  non-finite ``f64`` elements cross the wire as tagged objects —
  ``{"__nonfinite__": "nan" | "inf" | "-inf"}`` inside ``data``
  (host-tensor ``codec.rs`` since commit ``0528649``).  Both
  directions round-trip exactly: :func:`to_wire` emits the tags and
  :func:`from_wire` decodes them back to ``np.nan``/``np.inf``/
  ``-np.inf``.  A literal ``null`` element — the pre-tag serde_json
  spelling of a non-finite — is *rejected* loudly, matching the Rust
  decoder's ``decode_f64`` (old finite payloads still decode
  unchanged).
* **Dtype.** The codec carries ``f64`` only.  :func:`to_wire` upcasts
  any array-like to ``float64`` and stamps ``"dtype": "f64"`` — the
  stamp is true of the payload but the *source* dtype is not
  preserved: a non-``float64`` ndarray input warns at encode time
  (there is no int/bf16/f16 dtype on the wire — see
  :func:`to_wire`).
* **Index dtypes.** ``argmax``/``argsort``-family results are f64 on
  the wire (no int dtype); consumers must not dtype-check them.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np

from mlopt_host_tensor.errors import HostTensorError

WIRE_TAG = "__tensor__"
NONFINITE_TAG = "__nonfinite__"

_NONFINITE_VALUES = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}


def _encode_f64(x: float) -> Any:
    """Encode one ``data`` element for the wire.

    Finite ``f64``s stay numbers; NaN/±inf become the
    ``{"__nonfinite__": …}`` tag — the same spelling host-tensor's
    ``codec.rs::encode_f64`` emits.
    """
    if np.isnan(x):
        return {NONFINITE_TAG: "nan"}
    if x == np.inf:
        return {NONFINITE_TAG: "inf"}
    if x == -np.inf:
        return {NONFINITE_TAG: "-inf"}
    return float(x)


def _decode_f64(x: Any) -> float:
    """Decode one ``data`` element.

    A JSON number or a ``{"__nonfinite__": …}`` tag — mirroring
    ``codec.rs::decode_f64``.  Anything else (``null``, a string, a
    bool, an unknown tag) is a loud :class:`HostTensorError`, not a
    silent coercion.
    """
    if isinstance(x, bool):
        pass  # JSON true/false is not a number — fall through to the error
    elif isinstance(x, (int, float)):
        return float(x)
    elif isinstance(x, dict):
        tag = x.get(NONFINITE_TAG)
        if isinstance(tag, str) and tag in _NONFINITE_VALUES:
            return float(_NONFINITE_VALUES[tag])
    raise HostTensorError(
        "tensor data element is neither a number nor a valid "
        f"{NONFINITE_TAG!r} tag: {x!r}"
    )


def to_wire(a: Any) -> dict[str, Any]:
    """Encode an array-like as the ``{"__tensor__": …}`` wire record.

    Values are upcast to ``float64`` — the only dtype the codec
    carries.  The upcast is lossless for f16/bf16/f32 sources, but the
    wire records ``"dtype": "f64"`` and the source dtype is *not*
    preserved; a non-``float64`` ndarray input warns once per call
    site so the cast is never silent.

    Non-finite elements (NaN, ±inf) encode as
    ``{"__nonfinite__": "nan"|"inf"|"-inf"}`` tag objects, matching
    host-tensor's ``codec.rs`` — they round-trip exactly.
    """
    if isinstance(a, np.ndarray) and a.dtype != np.float64:
        warnings.warn(
            f"to_wire upcasts {a.dtype} input to f64 — the wire carries "
            "no dtype beyond 'f64', so the source dtype is not preserved",
            stacklevel=2,
        )
    arr = np.asarray(a, dtype=np.float64)
    flat = arr.reshape(-1)
    data: list[Any] = flat.tolist()
    for i in np.nonzero(~np.isfinite(flat))[0]:
        data[int(i)] = _encode_f64(float(flat[int(i)]))
    return {
        WIRE_TAG: {
            "shape": [int(d) for d in arr.shape],
            "data": data,
            "dtype": "f64",
        }
    }


def from_wire(v: Any) -> np.ndarray:
    """Decode a wire record (or array-like) to a float64 ndarray.

    ``data`` elements may be JSON numbers or ``{"__nonfinite__": …}``
    tags; ``null`` and anything else raise :class:`HostTensorError`.
    """
    if isinstance(v, dict) and WIRE_TAG in v:
        rec = v[WIRE_TAG]
        data = rec["data"]
        flat = np.array(
            [_decode_f64(x) for x in data], dtype=np.float64
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
