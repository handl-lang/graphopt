"""The equality gate rejects agreement-on-a-mistake.

The regression worth encoding is P21's: a kernel and a reference that share
a wrong nibble order agree with each other to 1e-4 and disagree with the
incumbent (cosine 0.21). These tests rebuild that situation on CPU.
"""

from __future__ import annotations

import pytest
import torch

from bench.benchkit.equality import (
    GATE_TOLERANCE,
    cosine_gate,
    gate_all,
    require,
)


def _words(n_words: int = 8) -> torch.Tensor:
    """Random int4 words, 8 nibbles each."""
    return torch.randint(
        0, 2**31 - 1, (n_words,), dtype=torch.int32
    )


def _unpack(words: torch.Tensor, perm: list[int] | None = None) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nibbles = (words.unsqueeze(-1) >> shifts) & 0xF
    if perm is not None:
        nibbles = nibbles[..., perm]
    return nibbles.reshape(-1).float()


def test_identical_output_passes() -> None:
    t = torch.randn(64)
    verdict = cosine_gate("same", t, incumbent=t.clone())
    assert verdict.passed
    assert verdict.cosine == pytest.approx(1.0)
    assert verdict.relative_worst == pytest.approx(0.0, abs=1e-6)
    require(verdict)


def test_fp16_accumulate_order_still_passes() -> None:
    """The tolerance must survive legitimate rounding, not just identity."""
    t = torch.randn(1024) * 5
    noisy = (t + torch.randn(1024) * 1e-3).half().float()
    verdict = cosine_gate("rounded", noisy, incumbent=t)
    assert verdict.passed, verdict


def test_shared_wrong_nibble_order_fails() -> None:
    """P21: candidate and its own reference agree; the incumbent does not."""
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    plain = list(range(8))
    words = _words()

    incumbent = _unpack(words, awq_order)  # what vLLM consumes
    candidate = _unpack(words, plain)  # what our kernel consumed
    own_reference = _unpack(words, plain)  # ...and what it compared against

    # The bug is invisible to the self-comparison...
    assert cosine_gate("self", candidate, incumbent=own_reference).passed
    # ...and caught by the external one.
    verdict = cosine_gate("candidate", candidate, incumbent=incumbent)
    assert verdict.cosine < 0.8
    assert not verdict.passed
    with pytest.raises(AssertionError, match="incumbent"):
        require(verdict)


def test_shape_mismatch_is_an_error_not_a_verdict() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        cosine_gate("bad", torch.randn(4), incumbent=torch.randn(5))


def test_gate_all_reports_every_candidate() -> None:
    incumbent = torch.randn(128)
    good = lambda: incumbent.clone()  # noqa: E731
    bad = lambda: -incumbent  # noqa: E731
    verdicts = gate_all({"good": good, "bad": bad}, incumbent=incumbent)
    assert verdicts["good"].passed
    assert not verdicts["bad"].passed
    assert verdicts["bad"].cosine < -0.99
    assert GATE_TOLERANCE == 0.9999


def test_verdict_serialises_without_tensor_types() -> None:
    d = cosine_gate("x", torch.ones(8), incumbent=torch.ones(8)).to_dict()
    assert set(d) == {
        "name",
        "cosine",
        "passed",
        "tolerance",
        "relative_worst",
    }
    assert isinstance(d["cosine"], float)
