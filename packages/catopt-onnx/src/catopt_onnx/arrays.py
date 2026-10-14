"""Array bridging — the dtype policy, in one place.

ADR 0006 requires the dtype policy to be *stated*, not discovered.
This module is where it is stated.

The working dtype
-----------------
The sink works in one declared floating dtype: ``float32`` by default
(the interchange precision every ONNX runtime speaks) or ``float64``
opt-in.  Two rules keep that honest rather than lossy:

* **Floating leaves are cast to the working dtype.**  A ``float64``
  export lowered to a ``float32`` graph is the same program at
  interchange precision — the difference is a rounding, not a
  semantic change, and it is the difference the equivalence gate
  measures.
* **Non-floating leaves keep their dtype.**  Integer index tables,
  boolean masks and int64 shape operands are *identity-carrying*:
  rounding them would change the program, so they are mapped to the
  matching ONNX element type and passed through untouched.

Where ``float64`` is exact, and where it is not
-----------------------------------------------
``float64`` is exact on the arithmetic / linear-algebra core, but
onnxruntime's CPU execution provider does **not** ship a double
kernel for every ONNX op.  Measured gaps: ``Tan``, ``Asin``,
``Acos``, ``Atan``, ``Sinh``, ``Cosh``, ``Erf`` (and therefore the
composed ``gelu``), ``Conv`` and ``InstanceNormalization``.  The sink
does not paper over this — the session raises ``NOT_IMPLEMENTED`` —
so ``float32`` is the *portable* mode and ``float64`` the *exactness*
mode over the subset that runs.

Two further caveats the policy records:

* ONNX stores float **attributes** (``LeakyRelu.alpha``,
  ``HardSigmoid.alpha``, ``Selu.alpha``/``gamma``, …) at 32 bits
  whatever the tensor dtype.  In ``float64`` a program through one of
  those ops carries a ~1e-8 relative perturbation; in the default
  ``float32`` mode it does not.
* There is no weak typing and no promotion: ONNX requires both
  operands of an elementwise op to share an element type.  ``Const``
  scalars adopt their sibling tensor's dtype (the graph builder
  arranges that), but a *tensor* of a different dtype does not
  promote — ATen's ``float32 + int64`` has no ONNX equivalent unless
  the model casts explicitly.

Nothing here imports a tensor library.  ``to_numpy`` duck-types: an
object exposing ``detach``/``cpu``/``numpy`` (a torch tensor) and a
plain array-like both convert, so the sink stays usable from a
non-torch caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto

__all__ = [
    "DTYPE_NAMES",
    "WORKING_DTYPES",
    "numpy_to_onnx_type",
    "ort_type_to_numpy",
    "to_numpy",
]

#: Every dtype name the sink understands, keyed by the name the IR
#: records (``to(dtype="float64")``) and the names a caller may use
#: for ``input_dtypes``.
DTYPE_NAMES: dict[str, np.dtype] = {
    "float64": np.dtype(np.float64),
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int64": np.dtype(np.int64),
    "int32": np.dtype(np.int32),
    "int16": np.dtype(np.int16),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "bool": np.dtype(np.bool_),
}

#: The floating working dtypes a sink may be built in, by name.
WORKING_DTYPES: dict[str, np.dtype] = {
    "float32": DTYPE_NAMES["float32"],
    "float64": DTYPE_NAMES["float64"],
}

#: NumPy dtype -> ONNX tensor element type.
_NUMPY_TO_ONNX: dict[str, int] = {
    "float64": TensorProto.DOUBLE,
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "int64": TensorProto.INT64,
    "int32": TensorProto.INT32,
    "int16": TensorProto.INT16,
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
    "bool": TensorProto.BOOL,
}

#: onnxruntime's input type string -> NumPy dtype (the inverse of
#: :data:`_NUMPY_TO_ONNX` for the types the sink can declare).
_ORT_TO_NUMPY: dict[str, str] = {
    "tensor(double)": "float64",
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int16)": "int16",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}


def to_numpy(value: Any) -> np.ndarray:
    """Convert a leaf value to a NumPy array without importing torch.

    Accepts a NumPy array, a nested sequence, or any object exposing
    the torch-tensor protocol (``detach`` / ``cpu`` / ``numpy``).
    """
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if numpy is not None:
        return np.asarray(numpy())
    return np.asarray(value)


def numpy_to_onnx_type(arr: np.ndarray) -> int:
    """Map a NumPy array's dtype to its ONNX element type.

    Raises ``KeyError`` for a dtype the sink has no mapping for —
    a refusal, not a silent widening.
    """
    return _NUMPY_TO_ONNX[arr.dtype.name]


def ort_type_to_numpy(type_str: str) -> np.dtype:
    """Map an onnxruntime input type string back to a NumPy dtype."""
    return np.dtype(_ORT_TO_NUMPY[type_str])
