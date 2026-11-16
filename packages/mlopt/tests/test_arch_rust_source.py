"""``arch`` source adapter tests — the source-onnx ``mlopt.arch/v1`` assembler.

Covers ref routing, spec resolution, binary/weights honesty, the
safetensors manifest → param-values binding (``source``/``transform``),
Param-name sanitizing, and the ``Source`` port feeding
``optimize_model``.  Tests that need the ``arch`` binary, a shipped
spec, or a real checkpoint skip when any is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mlopt.models.catalog  # noqa: F401 — populates the registry
from mlopt.adapters.arch_rust import _find_arch_binary, _hf_cache_weights
from mlopt.models import get_adapter, resolve_ref

try:
    import torch  # noqa: F401 — the pipeline's sink is torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover — torch is a hard dep
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch required")

_SIBLING = Path(__file__).resolve().parents[4] / "source-onnx"
_SPEC = _SIBLING / "data" / "architectures" / "whisper-tiny-encoder.json"


def _spec_path() -> Path | None:
    import os

    env = os.environ.get("SOURCE_ONNX_REPO")
    cands = [Path(env).expanduser() / "data/architectures/whisper-tiny-encoder.json"] if env else []
    cands.append(_SPEC)
    for cand in cands:
        if cand.is_file():
            return cand
    return None


def _weights_path() -> Path | None:
    """The whisper-tiny checkpoint: ``ARCH_WEIGHTS`` env, else HF cache."""
    import os

    env = os.environ.get("ARCH_WEIGHTS")
    if env and Path(env).is_file():
        return Path(env)
    spec = _spec_path()
    if spec is None:
        return None
    ckpt = json.loads(spec.read_text())["checkpoint"]
    return _hf_cache_weights(ckpt["repo"], ckpt["file"])


@pytest.fixture()
def adapter():
    ad = get_adapter("arch")
    assert ad is not None
    return ad


@pytest.fixture()
def arch_ready():
    spec, weights = _spec_path(), _weights_path()
    if spec is None or weights is None or _find_arch_binary() is None:
        pytest.skip("arch binary, spec, or whisper-tiny weights absent")
    return spec, weights


# --- ref routing ------------------------------------------------------------


def test_resolve_arch_prefix(adapter) -> None:
    assert resolve_ref("arch:whisper-tiny-encoder") is adapter
    assert resolve_ref("arch:/tmp/x.arch.json") is adapter


def test_claims_arch_refs(adapter, tmp_path) -> None:
    assert adapter.claims("arch:whisper-tiny-encoder")
    assert adapter.claims("spec.arch.json")
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"format": "mlopt.arch/v1"}))
    assert adapter.claims(str(p))
    p2 = tmp_path / "other.json"
    p2.write_text(json.dumps({"format": "other/v9"}))
    assert not adapter.claims(str(p2))
    assert not adapter.claims("whisper:openai/whisper-tiny")
    assert not adapter.claims("model.onnx")


def test_bare_spec_path_routes(adapter, arch_ready) -> None:
    spec, _weights = arch_ready
    assert resolve_ref(str(spec)) is adapter


# --- honest declines ---------------------------------------------------------


def test_missing_spec_declines(adapter) -> None:
    with pytest.raises(ValueError, match="cannot resolve arch spec"):
        adapter.load("arch:no-such-spec-xyz")


def test_missing_weights_path_declines(adapter, arch_ready) -> None:
    with pytest.raises(ValueError, match="weights="):
        adapter.load("arch:whisper-tiny-encoder", weights="/nope.safetensors")


def _load_spec() -> dict | None:
    p = _spec_path()
    return json.loads(p.read_text()) if p is not None else None


def test_no_checkpoint_no_weights_declines(adapter, tmp_path) -> None:
    spec = _load_spec()
    if spec is None:
        pytest.skip("spec fixture absent")
    del spec["checkpoint"]
    p = tmp_path / "no-ckpt.arch.json"
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="no weights"):
        adapter.load(str(p))


def test_unknown_family_verbatim(adapter, tmp_path) -> None:
    spec = _load_spec()
    if spec is None:
        pytest.skip("spec fixture absent")
    spec["family"] = "llama"
    p = tmp_path / "badfam.arch.json"
    p.write_text(json.dumps(spec))
    weights = _weights_path()
    if weights is None:
        pytest.skip("whisper-tiny weights absent")
    # The assembler's refusal wording (source-onnx 8c6dd79+): the
    # family dispatch now reports the spec as uncombinable sections.
    with pytest.raises(ValueError, match='llama spec cannot combine "encoder"'):
        adapter.load(str(p), weights=str(weights))


def test_missing_arch_bin_names_itself(adapter, tmp_path) -> None:
    p = tmp_path / "s.arch.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="arch_bin"):
        adapter.load(str(p), arch_bin=str(tmp_path / "nope"), weights="x")


# --- load: assemble + leaf binding --------------------------------------------


def test_load_binds_real_weights(adapter, arch_ready) -> None:
    _spec, weights = arch_ready
    spec = adapter.load("arch:whisper-tiny-encoder")
    ir, values = spec.source.to_ir(spec.module, spec.inputs)
    assert ir.root is not None
    assert [v.name for v in ir.inputs] == ["input_features"]
    # whisper-tiny encoder: 67 weight leaves (conv1/2 w+b, pos embed,
    # 4 layers × (2 LN w+b + qkv/out w+b + fc1/fc2 w+b), final LN).
    assert len(ir.params) == 67
    assert spec.meta["leaf_source"] == "real"
    assert spec.meta["input_source"] == "synthetic"
    assert spec.meta["weights_verified"] is True
    assert spec.meta["weights_path"] == str(weights)
    # Linear weights arrive transposed: HF [out,in] → matmul [in,out].
    fc1 = values["model_encoder_layers_0_fc1_weight"]
    assert tuple(fc1.shape) == (384, 1536)
    assert fc1.dtype.is_floating_point
    assert tuple(spec.inputs[0].shape) == (1, 80, 3000)


def test_param_names_are_setattr_safe(adapter, arch_ready) -> None:
    spec = adapter.load("arch:whisper-tiny-encoder")
    for name in spec.source._param_values:
        assert "." not in name and "::" not in name


# --- end to end through optimize_model ---------------------------------------


def test_optimize_verified(adapter, arch_ready) -> None:
    from mlopt.optimize import optimize_model

    spec = adapter.load("arch:whisper-tiny-encoder")
    lowered, report = optimize_model(spec, max_iterations=1)
    assert report.error is None
    assert report.verified is True
    assert lowered is not None
    assert any(n.startswith("input values: synthetic") for n in report.notes)
