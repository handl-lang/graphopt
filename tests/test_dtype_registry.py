"""The shared dtype registry (``mlopt_dtypes``) and its consumers.

``docs/reports/arch_audit_graphopt_packages.md`` §3 found the dtype
tables duplicated across ``kit_torch.export`` (torch.dtype ↔
safetensors tag), ``kit_onnx.arrays`` (IR name ↔ NumPy ↔ ONNX ↔
onnxruntime) and the name-resolution idiom in ``torch_bridge`` /
``kit_carriers.trace``.  ``mlopt_dtypes`` is the single table; these
tests pin that every consumer's derived view still agrees with the
foreign dtype systems it serves — a registry edit that drifts from
torch/NumPy/ONNX/safetensors fails here, not in a delivered artifact.
"""

import numpy as np
import pytest
import torch
from mlopt_dtypes import DTYPES, canonical_name, get, names, require

# ---------------------------------------------------------------------------
#  registry shape
# ---------------------------------------------------------------------------


def test_registry_covers_the_shipped_vocabulary():
    assert names() == (
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint8",
        "bool",
    )


def test_every_entry_names_a_torch_dtype():
    for rec in DTYPES.values():
        assert rec.torch is not None
        assert isinstance(getattr(torch, rec.torch), torch.dtype)
        assert rec.torch == rec.name


def test_canonical_name_spellings():
    assert canonical_name("float64") == "float64"
    assert canonical_name("torch.float64") == "float64"
    assert canonical_name(torch.float16) == "float16"
    # get/require accept the same spellings.
    assert get("torch.bfloat16") is DTYPES["bfloat16"]
    assert get("complex64") is None
    assert require("torch.int32").name == "int32"
    with pytest.raises(KeyError, match="complex64"):
        require("complex64")


def test_eps_and_itemsize_match_torch_finfo():
    for rec in DTYPES.values():
        dt = getattr(torch, rec.torch)
        assert rec.itemsize == dt.itemsize
        if rec.eps is None:
            # Exact types: finfo refuses, the registry records nothing.
            with pytest.raises(TypeError):
                torch.finfo(dt)
        else:
            assert rec.eps == torch.finfo(dt).eps


# ---------------------------------------------------------------------------
#  consumer: kit_torch.export safetensors tables
# ---------------------------------------------------------------------------


def test_safetensors_tables_derive_from_registry():
    from kit_torch.export import _ST_TO_TORCH, _TORCH_TO_ST

    expected = {
        "bool": "BOOL",
        "uint8": "U8",
        "int8": "I8",
        "int16": "I16",
        "int32": "I32",
        "int64": "I64",
        "float16": "F16",
        "bfloat16": "BF16",
        "float32": "F32",
        "float64": "F64",
        "float8_e4m3fn": "F8_E4M3",
        "float8_e5m2": "F8_E5M2",
    }
    assert {
        name: _TORCH_TO_ST[getattr(torch, name)] for name in expected
    } == expected
    assert len(_TORCH_TO_ST) == len(expected)
    for tag, dt in _ST_TO_TORCH.items():
        assert _TORCH_TO_ST[dt] == tag


def test_safetensors_container_roundtrip_per_dtype(tmp_path):
    """Every registered safetensors tag survives write→read."""
    from kit_torch.export import (
        _read_safetensors,
        _write_safetensors,
    )

    for name, rec in DTYPES.items():
        if rec.st is None:
            continue
        dt = getattr(torch, rec.torch)
        # .to() covers dtypes arange can't mint directly (float8).
        t = torch.arange(6).reshape(2, 3).to(dt)
        path = tmp_path / f"w_{name}.safetensors"
        _write_safetensors({"w": t}, path, "{}")
        back = _read_safetensors(path)["w"]
        assert back.dtype == t.dtype
        assert torch.equal(back, t)


# ---------------------------------------------------------------------------
#  consumer: kit_onnx.arrays tables
# ---------------------------------------------------------------------------


def test_onnx_dtype_names_derive_from_registry():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from kit_onnx.arrays import (
        _NUMPY_TO_ONNX,
        _ORT_TO_NUMPY,
        DTYPE_NAMES,
        ONNX_TYPE_NAMES,
        WORKING_DTYPES,
        numpy_to_onnx_type,
        ort_type_to_numpy,
    )
    from onnx import TensorProto

    # Exactly the registry entries NumPy can name, in registry order.
    assert list(DTYPE_NAMES) == [
        n for n, r in DTYPES.items() if r.numpy is not None
    ]
    assert set(WORKING_DTYPES) == {"float32", "float64"}

    # The NumPy→ONNX join is the registry's numpy/onnx columns.
    assert {
        rec.numpy: getattr(TensorProto, rec.onnx)
        for rec in DTYPES.values()
        if rec.numpy is not None and rec.onnx is not None
    } == _NUMPY_TO_ONNX
    # ONNX type → IR name is its exact inverse.
    assert {v: k for k, v in _NUMPY_TO_ONNX.items()} == ONNX_TYPE_NAMES
    # onnxruntime type strings round-trip to the same NumPy dtype.
    assert {
        rec.ort: rec.numpy
        for rec in DTYPES.values()
        if rec.ort is not None and rec.numpy is not None
    } == _ORT_TO_NUMPY
    for name, rec in DTYPES.items():
        if rec.numpy is None:
            continue
        arr = np.zeros(1, dtype=DTYPE_NAMES[name])
        assert numpy_to_onnx_type(arr) == getattr(TensorProto, rec.onnx)
        if rec.ort is not None:
            assert ort_type_to_numpy(rec.ort) == np.dtype(rec.numpy)


# ---------------------------------------------------------------------------
#  consumer: torch_bridge name resolution + report eps floor
# ---------------------------------------------------------------------------


def test_creator_dtype_resolves_registered_and_foreign_names():
    from kit_torch.torch_bridge import _creator_dtype

    assert _creator_dtype({"dtype": "float64"}) is torch.float64
    assert _creator_dtype({"dtype": "torch.bfloat16"}) is torch.bfloat16
    assert _creator_dtype({"dtype": torch.float16}) is torch.float16
    # Unregistered spellings keep the historical getattr fallback.
    assert _creator_dtype({"dtype": "complex64"}) is torch.complex64
    assert _creator_dtype({}) is None


def test_verify_rtol_floor_unchanged():
    """The dtype-resolution floor still keys off machine epsilon."""
    from kit_torch.report import _dtype_rtol_floor

    bf16 = torch.zeros(1, dtype=torch.bfloat16)
    f32 = torch.zeros(1, dtype=torch.float32)
    i64 = torch.zeros(1, dtype=torch.int64)
    # bf16 floor = 8 * eps, fp32/int contribute nothing.
    assert _dtype_rtol_floor(bf16) == pytest.approx(
        8.0 * torch.finfo(torch.bfloat16).eps
    )
    assert _dtype_rtol_floor(f32, i64) == 0.0
    # Registry eps agrees with finfo for every registered float.
    for rec in DTYPES.values():
        if rec.eps is not None:
            assert rec.eps == torch.finfo(getattr(torch, rec.torch)).eps
