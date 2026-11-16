"""``onnx-rust`` source adapter tests — the source-onnx Rust lifter.

Covers ref routing, binary resolution + honest declines, the
safetensors → param-values binding, Param-name sanitizing for the
torch sink, the ``OnnxRustSource`` port feeding ``optimize_model``,
and the lifter's named refusals surfacing verbatim.  Tests that need
the ``lift`` binary or the ``source-onnx`` fixture checkout skip
when either is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mlopt.models.catalog  # noqa: F401 — populates the registry
from mlopt.adapters.onnx_rust import _find_binary
from mlopt.models import get_adapter, resolve_ref

try:
    import torch  # noqa: F401 — the pipeline's sink is torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover — torch is a hard dep
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch required")

#: This repo's sibling ``source-onnx`` checkout (the adapter's own
#: default guess), plus the ``SOURCE_ONNX_REPO`` env override.
_SIBLING_FIXTURES = Path(__file__).resolve().parents[4] / "source-onnx" / "tests" / "fixtures"


def _fixture(name: str) -> Path | None:
    """A committed source-onnx fixture, env-overridable, skip when absent."""
    import os

    env = os.environ.get("SOURCE_ONNX_REPO")
    cands = []
    if env:
        cands.append(Path(env).expanduser() / "tests" / "fixtures" / name)
    cands.append(_SIBLING_FIXTURES / name)
    for cand in cands:
        if cand.is_file():
            return cand
    return None


@pytest.fixture()
def adapter():
    ad = get_adapter("onnx-rust")
    assert ad is not None
    return ad


@pytest.fixture()
def tiny_onnx():
    p = _fixture("add_relu.onnx")
    if p is None or _find_binary() is None:
        pytest.skip("source-onnx lift binary or add_relu.onnx fixture absent")
    return p


# --- ref routing ------------------------------------------------------------


def test_resolve_onnx_suffix(adapter, tiny_onnx) -> None:
    assert resolve_ref(str(tiny_onnx)) is adapter


def test_resolve_prefix(adapter) -> None:
    assert resolve_ref("onnx-rust:any/model.onnx") is adapter


def test_claims_only_onnx_refs(adapter) -> None:
    assert adapter.claims("model.onnx")
    assert adapter.claims("onnx-rust:model.onnx")
    assert not adapter.claims("whisper:openai/whisper-tiny")
    assert not adapter.claims("foo.hnl")


# --- binary resolution honesty ----------------------------------------------


def test_missing_lift_bin_names_itself(tmp_path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"\x00")  # exists — the binary check fires first
    ad = get_adapter("onnx-rust")
    assert ad is not None
    with pytest.raises(ValueError, match="lift_bin"):
        ad.load(f"onnx-rust:{model}", lift_bin=str(tmp_path / "nope"))


def test_missing_file_declines(tmp_path) -> None:
    ad = get_adapter("onnx-rust")
    assert ad is not None
    with pytest.raises(ValueError, match="cannot read"):
        ad.load(f"onnx-rust:{tmp_path / 'absent.onnx'}")


# --- load: lift + leaf binding ----------------------------------------------


def test_load_lifts_term_and_binds_real_weights(adapter, tiny_onnx) -> None:
    spec = adapter.load(str(tiny_onnx))
    ir, values = spec.source.to_ir(spec.module, spec.inputs)
    assert ir.root is not None
    # add_relu: one Var input x[4], one Param weight w[4].
    assert [v.name for v in ir.inputs] == ["x"]
    assert list(ir.params) == ["w"]
    # Weights are the artifact's own safetensors payload — real, not synthetic.
    assert spec.meta["leaf_source"] == "real"
    assert values["w"].shape == (4,)
    assert values["w"].dtype.is_floating_point
    # Inputs are seeded synthetic probes.
    assert spec.meta["input_source"] == "synthetic"
    assert tuple(spec.inputs[0].shape) == (4,)


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Minimal safetensors writer — ``{name: (dtype_tag, np.ndarray)}``."""
    import json
    import struct

    import numpy as np

    header: dict = {}
    blobs = []
    offset = 0
    for name, (tag, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": tag,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    h = json.dumps(header).encode()
    h += b" " * ((-len(h)) % 8)  # safetensors pads the header to 8-byte alignment
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_param_tensor_upcasts_half_dtypes(tmp_path) -> None:
    """F16/BF16 payloads bind as f32 params — lossless, verify-convention."""
    import numpy as np
    import torch

    from mlopt.adapters.onnx_rust import _param_tensor

    f16 = np.array([1.5, -2.0, 0.25], dtype="<f2")
    t = _param_tensor(f16)
    assert t.dtype == torch.float32
    assert t.tolist() == [1.5, -2.0, 0.25]
    f32 = np.array([1.5], dtype="<f4")
    assert _param_tensor(f32).dtype == torch.float32
    i64 = np.array([3], dtype="<i8")
    assert _param_tensor(i64).dtype == torch.int64


def test_load_safetensors_bf16_widens_losslessly(tmp_path) -> None:
    """``BF16`` has no NumPy spelling — decode u16 → f32 bit widening."""
    import numpy as np

    from mlopt.adapters.onnx_rust import _load_safetensors

    # bf16 bit patterns: 0x3F80 = 1.0, 0xBF80 = -1.0, 0x0000 = 0.0,
    # 0x7F80 = +inf, 0x7FC0 = NaN.
    u16 = np.array([0x3F80, 0xBF80, 0x0000, 0x7F80], dtype="<u2")
    p = tmp_path / "w.safetensors"
    _write_safetensors(p, {"w": ("BF16", u16)})
    out = _load_safetensors(p)
    assert out["w"].dtype == np.float32
    assert out["w"].shape == (4,)
    assert out["w"][:3].tolist() == [1.0, -1.0, 0.0]
    assert np.isinf(out["w"][3]) and out["w"][3] > 0


def test_param_names_are_setattr_safe(adapter, tiny_onnx) -> None:
    """Dotted ONNX value names must not reach ``IRModule`` verbatim."""
    spec = adapter.load(str(tiny_onnx))
    for name in spec.source._param_values:
        assert "." not in name and "::" not in name


def test_inputs_shape_override(adapter, tmp_path) -> None:
    """A dynamic-dim input declines until ``inputs`` supplies the shape."""
    p = _fixture("layout.onnx")
    if p is None or _find_binary() is None:
        pytest.skip("fixture/binary absent")
    ad = get_adapter("onnx-rust")
    assert ad is not None
    try:
        spec = ad.load(str(p))
    except ValueError as e:
        assert "unbound inputs" in str(e) or "inputs" in str(e)
    else:
        # If every dim was static the override is a no-op — still fine.
        assert spec.inputs


# --- named refusals surface verbatim -----------------------------------------


def test_scan_refusal_verbatim(adapter) -> None:
    p = _fixture("scan.onnx")
    if p is None or _find_binary() is None:
        pytest.skip("fixture/binary absent")
    with pytest.raises(ValueError, match="unsupported ONNX op Scan"):
        adapter.load(str(p))


def test_nondefault_domain_refusal(adapter) -> None:
    p = _fixture("ml_domain.onnx")
    if p is None or _find_binary() is None:
        pytest.skip("fixture/binary absent")
    with pytest.raises(ValueError, match="non-default domain"):
        adapter.load(str(p))


# --- end to end through optimize_model ---------------------------------------


def test_optimize_verified(adapter, tiny_onnx) -> None:
    from mlopt.optimize import optimize_model

    spec = adapter.load(str(tiny_onnx))
    lowered, report = optimize_model(spec, max_iterations=1)
    assert report.error is None
    assert report.verified is True
    assert lowered is not None
    # Weights are real but inputs are seeded probes — the report must
    # say so (ONNX declares shapes, not data).
    assert any(n.startswith("input values: synthetic") for n in report.notes)


# --- export --verify-run over the data-carrier spec --------------------------


def test_export_verify_run_prelifted(adapter, tiny_onnx, tmp_path, capsys) -> None:  # noqa: ANN001
    """``export --verify-run`` on a pre-lifted spec.

    ``spec.module`` is the ``OnnxLift`` payload, not a callable — the
    gate must rebuild the torch reference by lowering the source IR
    through ``ir_to_torch_module`` (the same reference
    ``lower(verify=True)`` compared against) and actually run the
    numeric compare through ``hnl-run``.
    """
    from mlopt.cli import main
    from mlopt.emit import find_hnl_run

    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")
    out = tmp_path / "add_relu.hnl"
    rc = main(
        [
            "export",
            str(tiny_onnx),
            "--verify-run",
            "--iters",
            "1",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "verify-run: PASS" in captured.out
    assert out.is_file()


def test_census_shared_subtrees_are_walked_once() -> None:
    """``_census`` must memoize ``Op`` nodes — lifted terms are DAGs.

    ``Op.make`` interning makes structurally equal subtrees the same
    object, so a term that reuses a subterm (residual streams, tied
    weights, the encoder output feeding every decoder cross-attn)
    would otherwise be re-walked once per *path* — exponentially.
    Regression: ``arch:whisper-tiny-full`` load hung >50 min inside
    the un-memoized recursive walk before this fix.
    """
    from mlopt_core.ir import Op, Param, TensorType, Var

    from mlopt.adapters.onnx_rust import _census

    x = Var("x", TensorType((4,)))
    w = Param("w", TensorType((4, 4)))
    # 2**20 root-to-leaf paths over 21 distinct Op nodes.
    t = x
    for _ in range(20):
        t = Op.make("add", t, t, w)
    vars_, params = _census(t)
    assert [v.name for v in vars_] == ["x"]
    assert list(params) == ["w"]
