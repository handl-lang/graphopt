"""Unit tests for the kokoro adapter — claims/probe/ref-parsing only.

No weight downloads, no CUDA, no kokoro source tree, no GPU required:
``load`` is exercised end-to-end only on hosts with the source
checkout + HF cache + CUDA (see the module docstring for the honest
coverage note).
"""

from __future__ import annotations

import pytest

import mlopt.models.catalog  # noqa: F401 — populates the registry
from mlopt.adapters.kokoro import ADAPTER, _hf_id
from mlopt.models import get_adapter, resolve_ref


def test_adapter_surface() -> None:
    assert ADAPTER.name == "kokoro"
    assert ADAPTER.available is True
    assert ADAPTER.unavailable_reason == ""
    assert "kokoro" in ADAPTER.description.lower()


def test_claims_prefix_only() -> None:
    assert ADAPTER.claims("kokoro:hexgrad/Kokoro-82M")
    assert ADAPTER.claims("kokoro:hexgrad/Kokoro-82M-v1.1-zh")
    assert ADAPTER.claims("  kokoro:some-org/Kokoro-finetune  ")
    # bare HF ids and other families are not claimed
    assert not ADAPTER.claims("hexgrad/Kokoro-82M")
    assert not ADAPTER.claims("whisper:openai/whisper-base")
    assert not ADAPTER.claims("kokoro")
    assert not ADAPTER.claims("")


def test_hf_id_parsing() -> None:
    assert _hf_id("kokoro:hexgrad/Kokoro-82M") == "hexgrad/Kokoro-82M"
    assert _hf_id("hexgrad/Kokoro-82M") == "hexgrad/Kokoro-82M"
    assert _hf_id("  kokoro:hexgrad/Kokoro-82M ") == "hexgrad/Kokoro-82M"
    with pytest.raises(ValueError, match="empty HF model id"):
        _hf_id("kokoro:")
    with pytest.raises(ValueError, match="empty HF model id"):
        _hf_id("   ")


def test_probe_fields() -> None:
    info = ADAPTER.probe("kokoro:hexgrad/Kokoro-82M")
    assert info["adapter"] == "kokoro"
    assert info["available"] is True
    assert info["hf_id"] == "hexgrad/Kokoro-82M"
    assert info["family"] == "kokoro"
    assert info["voice"] == "af_heart"
    assert info["arms"] == ["decoder", "decoder-nc", "f0n", "textenc", "textenc-p", "full"]
    assert info["dtype_default"] == "float32"
    # stochastic arms are flagged honestly (SineGen draws in-graph)
    assert set(info["stochastic_arms"]) == {"decoder", "decoder-nc", "full"}
    # availability flags are honest booleans
    for key in (
        "torch",
        "huggingface_hub",
        "transformers",
        "loguru",
        "attrs",
        "numpy",
        "misaki",
    ):
        assert isinstance(info[key], bool), key
    assert isinstance(info["kokoro_src_present"], bool)
    assert info["kokoro_src"]  # a path string, env/default resolved


def test_probe_does_not_import_heavy_deps() -> None:
    # probe must stay cheap: find_spec only — it must not import
    # torch/kokoro/huggingface_hub as a side effect of parsing.  The
    # kokoro package may legitimately already be imported by an
    # earlier test in the session, so assert on the *delta* probe
    # introduces, not on global sys.modules state.
    import sys

    before = set(sys.modules)
    info = ADAPTER.probe("kokoro:hexgrad/Kokoro-82M")
    assert info["family"] == "kokoro"
    assert "kokoro" not in set(sys.modules) - before
    # transformers may legitimately be installed in the dev venv —
    # the flag must reflect reality either way.
    assert "transformers" not in sys.modules or info["transformers"]


def test_probe_bare_id_accepted() -> None:
    info = ADAPTER.probe("hexgrad/Kokoro-82M")
    assert info["hf_id"] == "hexgrad/Kokoro-82M"


def test_probe_empty_id_raises() -> None:
    with pytest.raises(ValueError, match="empty HF model id"):
        ADAPTER.probe("kokoro:")


def test_probe_unknown_repo_is_not_rejected() -> None:
    # probe is metadata-only: it parses any id; load() is where the
    # KModel.MODEL_NAMES weight-file check lives.
    info = ADAPTER.probe("kokoro:some-org/Kokoro-finetune")
    assert info["hf_id"] == "some-org/Kokoro-finetune"


def test_registry_kokoro_resolves_to_real_adapter() -> None:
    """pending.py is gone — the real adapter must win the registry.

    (History: alphabetical discovery + last-wins registration let a
    ``pending`` placeholder shadow this adapter; the placeholder file
    was deleted once both whisper and kokoro went live.)
    """
    adapter = get_adapter("kokoro")
    assert adapter is ADAPTER
    assert adapter.available is True
    assert resolve_ref("kokoro:hexgrad/Kokoro-82M") is ADAPTER
