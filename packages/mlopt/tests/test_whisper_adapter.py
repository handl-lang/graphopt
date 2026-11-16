"""Unit tests for the whisper adapter — claims/probe/ref-parsing only.

No weight downloads, no CUDA, no transformers/torch required:
``load`` is exercised end-to-end only on GPU hosts with the optional
dep installed (see the module docstring for the honest coverage note).
"""

from __future__ import annotations

import pytest

import mlopt.models.catalog  # noqa: F401 — populates the registry
from mlopt.adapters.whisper import (
    ADAPTER,
    _decoder_alias_positions,
    _dtype_hint,
    _hf_id,
    _size_hint,
)
from mlopt.models import get_adapter, resolve_ref


def test_adapter_surface() -> None:
    assert ADAPTER.name == "whisper"
    assert ADAPTER.available is True
    assert ADAPTER.unavailable_reason == ""
    assert "whisper" in ADAPTER.description.lower()


def test_claims_prefix_only() -> None:
    assert ADAPTER.claims("whisper:openai/whisper-turbo")
    assert ADAPTER.claims("whisper:openai/whisper-base")
    assert ADAPTER.claims("  whisper:openai/whisper-small  ")
    # bare HF ids and other families are not claimed
    assert not ADAPTER.claims("openai/whisper-base")
    assert not ADAPTER.claims("kokoro")
    assert not ADAPTER.claims("whisper")
    assert not ADAPTER.claims("")


def test_hf_id_parsing() -> None:
    assert _hf_id("whisper:openai/whisper-turbo") == "openai/whisper-turbo"
    assert _hf_id("openai/whisper-base") == "openai/whisper-base"
    assert _hf_id("  whisper:openai/whisper-small ") == "openai/whisper-small"
    with pytest.raises(ValueError, match="empty HF model id"):
        _hf_id("whisper:")


def test_size_hints() -> None:
    assert _size_hint("openai/whisper-tiny") == "tiny"
    assert _size_hint("openai/whisper-base") == "base"
    assert _size_hint("openai/whisper-base.en") == "base"
    assert _size_hint("openai/whisper-small") == "small"
    assert _size_hint("openai/whisper-medium") == "medium"
    assert _size_hint("openai/whisper-large") == "large"
    assert _size_hint("openai/whisper-large-v2") == "large-v2"
    assert _size_hint("openai/whisper-large-v3") == "large-v3"
    assert _size_hint("openai/whisper-large-v3-turbo") == "turbo"
    assert _size_hint("openai/whisper-turbo") == "turbo"
    assert _size_hint("distil-whisper/distil-large-v3") == "distil-large-v3"
    assert _size_hint("openai/whisper-base") == _size_hint("some-org/whisper-base-finetune")
    assert _size_hint("some-org/whisper-xl") is None
    assert _size_hint("bert-base-uncased") == "base"  # substring match, documented


def test_dtype_hints() -> None:
    assert _dtype_hint("openai/whisper-base") == "float32"
    assert _dtype_hint("openai/whisper-small") == "float32"
    assert _dtype_hint("openai/whisper-turbo") == "bfloat16"
    assert _dtype_hint("openai/whisper-large-v3-turbo") == "bfloat16"
    assert _dtype_hint("openai/whisper-large-v3") == "bfloat16"
    assert _dtype_hint("openai/whisper-large") == "bfloat16"


def test_probe_fields() -> None:
    info = ADAPTER.probe("whisper:openai/whisper-turbo")
    assert info["adapter"] == "whisper"
    assert info["available"] is True
    assert info["hf_id"] == "openai/whisper-turbo"
    assert info["family"] == "whisper"
    assert info["size"] == "turbo"
    assert info["n_mels"] == 128
    assert info["dtype_default"] == "bfloat16"
    # availability flags are honest booleans (transformers is not a dep)
    assert isinstance(info["transformers"], bool)
    assert isinstance(info["torch"], bool)


def test_probe_small_model() -> None:
    info = ADAPTER.probe("whisper:openai/whisper-base")
    assert info["size"] == "base"
    assert info["n_mels"] == 80
    assert info["dtype_default"] == "float32"


def test_probe_unknown_size_is_none_safe() -> None:
    info = ADAPTER.probe("whisper:some-org/custom-whisper-head")
    assert info["hf_id"] == "some-org/custom-whisper-head"
    assert info["size"] is None
    assert info["n_mels"] is None
    assert info["dtype_default"] == "float32"


def test_probe_bare_id_accepted() -> None:
    info = ADAPTER.probe("openai/whisper-small")
    assert info["hf_id"] == "openai/whisper-small"
    assert info["size"] == "small"


def test_probe_empty_id_raises() -> None:
    with pytest.raises(ValueError, match="empty HF model id"):
        ADAPTER.probe("whisper:")


def test_probe_does_not_require_transformers() -> None:
    # probe must stay cheap: it reports availability via find_spec and
    # must not import transformers/torch as a side effect of parsing.
    import sys

    info = ADAPTER.probe("whisper:openai/whisper-base")
    assert info["family"] == "whisper"
    # transformers may legitimately be absent in the dev venv — the
    # flag must reflect reality either way.
    assert "transformers" not in sys.modules or info["transformers"]


def test_registry_whisper_is_the_real_adapter() -> None:
    adapter = get_adapter("whisper")
    assert adapter is ADAPTER
    assert adapter.available is True
    assert resolve_ref("whisper:openai/whisper-turbo") is ADAPTER
    # kokoro is live too — pending.py was deleted once both adapters shipped
    kokoro = get_adapter("kokoro")
    assert kokoro is not None and kokoro.available is True


def test_probe_lists_encoder_arm() -> None:
    info = ADAPTER.probe("whisper:openai/whisper-turbo")
    assert info["arms"] == ["decoder", "encoder"]


def test_decoder_alias_positions_layout() -> None:
    """Pin the decoder arm's ``alias_inputs``/``mutable_inputs`` meta
    contract: ``(hs, mask, pos, sk[n], sv[n], ck[n], cv[n])`` — the 4*n
    persistent K/V buffers (positions 3..3+4n-1) alias by reference;
    the write-back self-K/V (sk/sv, positions 3..3+2n-1) are also
    ``mutable_inputs`` (in-graph ``index_copy_`` mutates them), while
    the read-only cross K/V (ck/cv) are alias-only.  Positions 0..2
    are rebuilt per step and must stay on the copy path."""
    n = 4  # whisper-tiny layer count — layout is what matters, not n
    alias, mutable = _decoder_alias_positions(n)
    assert alias == list(range(3, 3 + 4 * n))
    assert mutable == list(range(3, 3 + 2 * n))
    # mutable ⊂ alias: write-back positions must bind live storage or
    # the recorded index_copy_ lands in a private buffer and the
    # caller's cache never sees it.
    assert set(mutable) <= set(alias)
    # the per-step inputs (hs/mask/pos) are never aliased
    assert not set(alias) & {0, 1, 2}
    # ck/cv (positions 3+2n .. 3+4n-1) are alias but NOT mutable —
    # _KVRefLayer adopts them read-only.
    assert set(range(3 + 2 * n, 3 + 4 * n)) <= set(alias) - set(mutable)
    # the spec round-trips through deliver._alias_spec as ints
    from mlopt.deliver import _alias_spec

    spec, err = _alias_spec({"alias_inputs": alias}, "alias_inputs")
    assert err is None and spec == frozenset(alias)
    spec, err = _alias_spec({"mutable_inputs": mutable}, "mutable_inputs")
    assert err is None and spec == frozenset(mutable)


def test_unknown_arm_declined_before_download() -> None:
    """``arm=`` validation fires before any weight download — needs the
    optional deps importable but touches no weights."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    with pytest.raises(ValueError, match="unknown arm"):
        ADAPTER.load("whisper:openai/whisper-tiny", arm="bogus")
