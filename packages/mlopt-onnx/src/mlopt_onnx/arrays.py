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
from mlopt_dtypes import DTYPES
from onnx import TensorProto

__all__ = [
    "DTYPE_NAMES",
    "ONNX_TYPE_NAMES",
    "WORKING_DTYPES",
    "numpy_to_onnx_type",
    "ort_type_to_numpy",
    "to_numpy",
]

#: Every dtype name the sink understands, keyed by the name the IR
#: records (``to(dtype="float64")``) and the names a caller may use
#: for ``input_dtypes``.  Derived from the shared ``mlopt_dtypes``
#: registry — the ``numpy`` column is exactly this vocabulary (names
#: NumPy cannot represent, like bfloat16, drop out).
DTYPE_NAMES: dict[str, np.dtype] = {
    name: np.dtype(rec.numpy)
    for name, rec in DTYPES.items()
    if rec.numpy is not None
}

#: The floating working dtypes a sink may be built in, by name.
WORKING_DTYPES: dict[str, np.dtype] = {
    "float32": DTYPE_NAMES["float32"],
    "float64": DTYPE_NAMES["float64"],
}

#: NumPy dtype -> ONNX tensor element type (the registry's ``numpy``
#: and ``onnx`` columns joined; a dtype ONNX has no element type for
#: — bfloat16, the float8 family — is absent, so the lookup refuses).
_NUMPY_TO_ONNX: dict[str, int] = {
    rec.numpy: getattr(TensorProto, rec.onnx)
    for rec in DTYPES.values()
    if rec.numpy is not None and rec.onnx is not None
}

#: ONNX tensor element type -> the name the IR records (the inverse
#: of :data:`_NUMPY_TO_ONNX`).  The source reads declared dtypes off
#: graph inputs and ``Cast`` targets; element types outside this
#: table (string, bfloat16, float8, complex) have no IR name and are
#: refused where they matter.
ONNX_TYPE_NAMES: dict[int, str] = {
    v: k for k, v in _NUMPY_TO_ONNX.items()
}

#: onnxruntime's input type string -> NumPy dtype (the inverse of
#: :data:`_NUMPY_TO_ONNX` for the types the sink can declare — the
#: registry's ``ort`` column).
_ORT_TO_NUMPY: dict[str, str] = {
    rec.ort: rec.numpy
    for rec in DTYPES.values()
    if rec.ort is not None and rec.numpy is not None
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
