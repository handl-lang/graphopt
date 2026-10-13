"""Tests for the declarative-derive synthesizer
(:mod:`catopt_discovery.dspec_synth`).

``synth_dspec`` closes the derive-as-data loop the way ``auto_cond``
closes the guard loop: measure a procedural hook's outputs over a
binding domain, enumerate candidate ``dspec`` expressions over the
cond-DSL bank, keep the ones reproducing every sample, mint the
smallest.  A refusal is a finding — the bank lacks the op the hook
needs — and the refusal guards (no value diversity, thin sample
sets, held-out validation) are what keep a coincidence from
shipping.
"""

from __future__ import annotations

from catopt_core.laws.cond import eval_derive
from catopt_discovery.dspec_synth import (
    MIN_SAMPLES,
    _near_misses,
    samples_from_hook,
    synth_dspec,
)


def _sum_hook(bound):
    a, c = bound.get("$attr:a"), bound.get("$attr:c")
    if not (isinstance(a, int) and isinstance(c, int)):
        return None
    return {"$attr:S": a + c}


def _bounds(n: int) -> list[dict]:
    return [
        {"q": "Q", "$attr:a": i, "$attr:c": i + 1}
        for i in range(1, n + 1)
    ]


class TestSynthesis:
    def test_finds_sum_of_attrs(self):
        spec = synth_dspec(
            samples_from_hook(_sum_hook, _bounds(6)), name="S"
        )
        assert spec == {"S": ("add", ("attr", "a"), ("attr", "c"))}

    def test_synth_verifies_through_eval_derive(self):
        bounds = _bounds(6)
        spec = synth_dspec(
            samples_from_hook(_sum_hook, bounds), name="S"
        )
        for b in bounds:
            assert eval_derive(spec, b) == {"$attr:S": _sum_hook(b)["$attr:S"]}

    def test_matches_heldout(self):
        # fit on even, validated on odd — a synth must agree on both
        spec = synth_dspec(
            samples_from_hook(_sum_hook, _bounds(8)), name="S"
        )
        assert spec is not None
        assert eval_derive(spec, {"$attr:a": 100, "$attr:c": 1}) == {
            "$attr:S": 101
        }


class TestRefusals:
    def test_thin_sample_set_refused(self):
        spec = synth_dspec(
            samples_from_hook(_sum_hook, _bounds(2)), name="S"
        )
        assert spec is None
        assert MIN_SAMPLES == 4

    def test_no_diversity_refused(self):
        def const_hook(_b):
            return {"$attr:D": 0}

        spec = synth_dspec(
            samples_from_hook(const_hook, _bounds(6)), name="D"
        )
        assert spec is None

    def test_bank_gap_refused(self):
        # max(a, c) is not in the bank — honest decline
        def max_hook(bound):
            a, c = bound.get("$attr:a"), bound.get("$attr:c")
            return {"$attr:M": max(a, c)}

        bounds = [
            {"$attr:a": i, "$attr:c": 7 - i} for i in range(1, 7)
        ]
        spec = synth_dspec(
            samples_from_hook(max_hook, bounds), name="M"
        )
        assert spec is None

    def test_empty_samples_refused(self):
        assert synth_dspec([]) is None

    def test_declining_hook_yields_no_samples(self):
        def never(_b):
            return None

        assert samples_from_hook(never, _bounds(6)) == []


class TestNearMisses:
    def test_near_misses_report_partial_agreement(self):
        def max_hook(bound):
            return {"$attr:M": max(bound["$attr:a"], bound["$attr:c"])}

        bounds = [
            {"$attr:a": i, "$attr:c": 7 - i} for i in range(1, 7)
        ]
        samples = samples_from_hook(max_hook, bounds)
        near = _near_misses(samples)
        assert near
        # the best near miss agrees on at least one sample
        assert near[0][1] >= 1
        assert near[0][1] < len(samples)
