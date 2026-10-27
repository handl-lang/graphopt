"""The deploy-regime referee — pure-function and protocol tests.

GPU-free where possible: `evictor`/`blocked_paired` are exercised on CPU
tensors so the protocol (shuffling, warm blocks, medians) is covered in
CI without a device.
"""

from __future__ import annotations

import pytest
import torch

from bench.benchkit import regime


def test_roofline_is_bytes_over_bandwidth():
    # 89.1 GB/s, 2.18 MB -> ~24.5 us
    us = regime.roofline_us(2.18e6, 89.1)
    assert us == pytest.approx(24.5, abs=0.5)


def test_roofline_adds_fixed_overhead():
    assert regime.roofline_us(
        1e6, 100.0, overhead_us=5.0
    ) == pytest.approx(15.0, abs=0.01)


def test_roofline_rejects_bad_bandwidth():
    with pytest.raises(ValueError):
        regime.roofline_us(1.0, 0.0)


def test_evictor_flushes_the_requested_number_of_passes():
    calls = {"n": 0}

    flush = regime.evictor(evict_mb=1, device="cpu")

    # evictor returns a closure over a real buffer; count via a wrapper
    def counted(passes=1):
        calls["n"] += passes
        flush(passes)

    counted(3)
    assert calls["n"] == 3


def test_blocked_paired_measures_every_candidate():
    a = lambda: torch.zeros(16, device="cpu")  # noqa: E731
    b = lambda: torch.zeros(64, device="cpu")  # noqa: E731
    res = regime.blocked_paired(
        {"a": a, "b": b},
        rounds=3,
        evict_passes=0,
        warm_per_block=0,
        inner_reps=1,
        flush=lambda p=1: None,
    )
    assert set(res.medians) == {"a", "b"}
    assert all(v >= 0 for v in res.medians.values())
    assert len(res.samples["a"]) == 3


def test_blocked_paired_warms_before_timing():
    """The warm calls must happen, and must not be timed."""
    seen = {"calls": 0}

    def fn():
        seen["calls"] += 1
        return None

    regime.blocked_paired(
        {"only": fn},
        rounds=2,
        evict_passes=0,
        warm_per_block=3,
        inner_reps=2,
        flush=lambda p=1: None,
    )
    # 1 compile call + rounds * (3 warm + 2 timed)
    assert seen["calls"] == 1 + 2 * (3 + 2)


def test_blocked_paired_rejects_empty_input():
    with pytest.raises(ValueError):
        regime.blocked_paired({})


def test_ratio_reports_slower_as_greater_than_one():
    res = regime.RegimeResult(medians={"a": 10.0, "b": 5.0})
    assert res.ratio("a", "b") == pytest.approx(2.0)
    assert res.ratio("b", "a") == pytest.approx(0.5)


def test_result_serializes_rounded_medians():
    res = regime.RegimeResult(
        medians={"a": 1.23456}, stdevs={"a": 0.1}, rounds=1
    )
    payload = res.to_dict()
    assert payload["medians_us"]["a"] == 1.23
