"""The dtype registry — one canonical table for every backend.

Plan: ``docs/reports/arch_audit_graphopt_packages.md`` §3 found the
dtype mappings duplicated across the domain packages — torch dtype ↔
safetensors tag (``mlopt_torch.export``), IR name ↔ NumPy dtype ↔
ONNX element type ↔ onnxruntime type string (``mlopt_onnx.arrays``),
``hardware.dtype`` names (``mlopt_discovery.play``), and the
name-resolution idiom ``getattr(torch, name.split(".")[-1])``
(``mlopt_torch.torch_bridge``, ``mlopt_carriers.trace``).  This module
is the single table they all derive from.

Why a separate package
----------------------
``mlopt_core`` would be the natural home (it is the torch-free
substrate), but it ships from ``handl-lang/eqsat`` as a pinned git
dep — a new module there is a release, not a commit.  ``mlopt_onnx``
sits *below* ``mlopt_torch`` in the dependency-layers contract and
must stay torch/numpy-only-light, so the registry is its own leaf
package: pure stdlib, zero dependencies, importable by every backend.

The contract
------------
The registry keys on the **canonical IR name** — the short string the
term IR records for a dtype attribute (``to(dtype="float64")``),
which is ``str(torch.dtype)`` minus the ``torch.`` prefix.  Every
field is data, not a foreign object: this module imports neither
torch, numpy nor onnx.  Each backend resolves its own objects from
the strings:

* ``torch``  — the ``torch.dtype`` attribute name
  (``getattr(torch, rec.torch)``);
* ``numpy``  — a ``np.dtype()``-resolvable name, ``None`` where NumPy
  has no counterpart (bfloat16, the float8 family);
* ``onnx``   — the ``onnx.TensorProto`` enum attribute name
  (``getattr(TensorProto, rec.onnx)``), ``None`` for dtypes ONNX has
  no element type for *in this sink's policy* — note ``bfloat16`` is
  deliberately ``None``: ``TensorProto.BFLOAT16`` exists, but the
  ADR-0006 dtype policy refuses it (see ``mlopt_onnx.arrays``), and
  the registry preserves that refusal rather than widening it;
* ``ort``    — the onnxruntime input type string
  (``"tensor(float)"``);
* ``st``     — the safetensors dtype tag (``"F32"``);
* ``hnl``    — the host-tensor ``.hnl`` wire dtype tag.  The wire is
  f64-only today (``{"__tensor__": {"dtype": "f64"}}`` — host-tensor
  ADR-0083, enforced in ``host_tensor::ops::codec``), so only
  ``float64`` carries a tag;
* ``eps``    — the machine epsilon (``torch.finfo(dtype).eps`` /
  ``np.finfo(dtype).eps`` — the values agree), ``None`` for exact
  types;
* ``itemsize`` — bytes per element.

A name absent from :data:`DTYPES` is not an error: the torch bridge
still resolves arbitrary ``torch.dtype`` spellings through
``getattr`` (complex, quantized, …), and each backend keeps its own
refusal path for names it cannot represent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "DTYPES",
    "DTypeInfo",
    "canonical_name",
    "get",
    "names",
    "require",
]


@dataclass(frozen=True)
class DTypeInfo:
    """One registered dtype.

    Every field is a resolvable name or a measured constant; ``None``
    marks "this backend has no representation" (a refusal the consumer
    must keep, not paper over).
    """

    #: The canonical IR name — the registry key.
    name: str
    #: ``torch.dtype`` attribute name (``getattr(torch, torch)``).
    torch: str | None = None
    #: ``np.dtype()``-resolvable name.
    numpy: str | None = None
    #: ``onnx.TensorProto`` enum attribute name.
    onnx: str | None = None
    #: onnxruntime input type string (``"tensor(float)"``).
    ort: str | None = None
    #: safetensors container dtype tag (``"F32"``).
    st: str | None = None
    #: host-tensor wire dtype tag (``"f64"``; the wire is f64-only).
    hnl: str | None = None
    #: Machine epsilon for inexact types (``finfo().eps``); ``None``
    #: for exact (integer/bool) types.
    eps: float | None = None
    #: Bytes per element.
    itemsize: int | None = None


#: The registry.  Iteration order is the canonical listing order —
#: consumers that build name→object dicts inherit it, which keeps
#: e.g. ``mlopt_onnx.arrays.DTYPE_NAMES`` in its historical order.
_DTYPES: dict[str, DTypeInfo] = {
    # -- inexact -----------------------------------------------------
    "float64": DTypeInfo(
        "float64",
        torch="float64",
        numpy="float64",
        onnx="DOUBLE",
        ort="tensor(double)",
        st="F64",
        hnl="f64",
        eps=2.220446049250313e-16,
        itemsize=8,
    ),
    "float32": DTypeInfo(
        "float32",
        torch="float32",
        numpy="float32",
        onnx="FLOAT",
        ort="tensor(float)",
        st="F32",
        eps=1.1920928955078125e-07,
        itemsize=4,
    ),
    "float16": DTypeInfo(
        "float16",
        torch="float16",
        numpy="float16",
        onnx="FLOAT16",
        ort="tensor(float16)",
        st="F16",
        eps=0.0009765625,
        itemsize=2,
    ),
    "bfloat16": DTypeInfo(
        "bfloat16",
        torch="bfloat16",
        st="BF16",
        eps=0.0078125,
        itemsize=2,
        # NumPy has no bfloat16; ONNX BFLOAT16 exists but the ADR-0006
        # sink policy refuses it — both stay None on purpose.
    ),
    "float8_e4m3fn": DTypeInfo(
        "float8_e4m3fn",
        torch="float8_e4m3fn",
        st="F8_E4M3",
        eps=0.125,
        itemsize=1,
    ),
    "float8_e5m2": DTypeInfo(
        "float8_e5m2",
        torch="float8_e5m2",
        st="F8_E5M2",
        eps=0.25,
        itemsize=1,
    ),
    # -- exact ---------------------------------------------------------
    "int64": DTypeInfo(
        "int64",
        torch="int64",
        numpy="int64",
        onnx="INT64",
        ort="tensor(int64)",
        st="I64",
        itemsize=8,
    ),
    "int32": DTypeInfo(
        "int32",
        torch="int32",
        numpy="int32",
        onnx="INT32",
        ort="tensor(int32)",
        st="I32",
        itemsize=4,
    ),
    "int16": DTypeInfo(
        "int16",
        torch="int16",
        numpy="int16",
        onnx="INT16",
        ort="tensor(int16)",
        st="I16",
        itemsize=2,
    ),
    "int8": DTypeInfo(
        "int8",
        torch="int8",
        numpy="int8",
        onnx="INT8",
        ort="tensor(int8)",
        st="I8",
        itemsize=1,
    ),
    "uint8": DTypeInfo(
        "uint8",
        torch="uint8",
        numpy="uint8",
        onnx="UINT8",
        ort="tensor(uint8)",
        st="U8",
        itemsize=1,
    ),
    "bool": DTypeInfo(
        "bool",
        torch="bool",
        numpy="bool",
        onnx="BOOL",
        ort="tensor(bool)",
        st="BOOL",
        itemsize=1,
    ),
}

DTYPES: Mapping[str, DTypeInfo] = MappingProxyType(_DTYPES)
"""The registry — canonical name → :class:`DTypeInfo`."""


def canonical_name(name: Any) -> str:
    """Normalise a dtype spelling to its canonical short name.

    ``"torch.float64"`` → ``"float64"``; ``"float64"`` → ``"float64"``.
    This is the ``str(dtype).split(".")[-1]`` idiom the torch bridge
    and the carrier trace lowerings each spelled inline — one place
    now.
    """
    return str(name).split(".")[-1]


def names() -> tuple[str, ...]:
    """All registered canonical names, in registry order."""
    return tuple(_DTYPES)


def get(name: Any) -> DTypeInfo | None:
    """Look up ``name``; ``None`` when unregistered — never raises.

    Accepts any spelling :func:`canonical_name` accepts.
    """
    return _DTYPES.get(canonical_name(name))


def require(name: Any) -> DTypeInfo:
    """Look up ``name`` or raise ``KeyError`` listing the registry."""
    key = canonical_name(name)
    try:
        return _DTYPES[key]
    except KeyError:
        raise KeyError(
            f"unknown dtype {name!r}; registered: {sorted(_DTYPES)}"
        ) from None
