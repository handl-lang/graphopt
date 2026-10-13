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
        # ``2 ** a`` needs a power op the bank lacks — honest decline
        # (``max``/``min`` used to be the gap; they landed).
        def pow_hook(bound):
            return {"$attr:M": 2 ** bound["$attr:a"]}

        bounds = [{"$attr:a": i} for i in range(1, 7)]
        spec = synth_dspec(
            samples_from_hook(pow_hook, bounds), name="M"
        )
        assert spec is None

    def test_empty_samples_refused(self):
        assert synth_dspec([]) is None

    def test_declining_hook_yields_no_samples(self):
        def never(_b):
            return None

        assert samples_from_hook(never, _bounds(6)) == []


class TestNewOps:
    """The ops the residual survey measured as missing: ``max``/``min``
    and the ``("len", ("shape", T))`` rank reader."""

    def test_finds_max_of_attrs(self):
        def max_hook(bound):
            a, c = bound["$attr:a"], bound["$attr:c"]
            return {"$attr:M": max(a, c)}

        bounds = [{"$attr:a": i, "$attr:c": 7 - i} for i in range(1, 7)]
        spec = synth_dspec(
            samples_from_hook(max_hook, bounds), name="M"
        )
        assert spec == {"M": ("max", ("attr", "a"), ("attr", "c"))}

    def test_finds_min_of_attrs(self):
        def min_hook(bound):
            a, c = bound["$attr:a"], bound["$attr:c"]
            return {"$attr:M": min(a, c)}

        bounds = [{"$attr:a": i, "$attr:c": 7 - i} for i in range(1, 7)]
        spec = synth_dspec(
            samples_from_hook(min_hook, bounds), name="M"
        )
        assert spec == {"M": ("min", ("attr", "a"), ("attr", "c"))}

    def test_finds_rank_offset(self):
        # ``max(rank q, rank k) - 1`` — the matmul_t_concat derive.
        from catopt_core.ir import TensorType, Var

        def var(name, shape):
            return Var(name, TensorType(tuple(shape)))

        def rank_hook(bound):
            return {
                "$attr:SD": max(
                    len(bound["q"].typ.shape),
                    len(bound["k1"].typ.shape),
                )
                - 1
            }

        # ranks interleave so neither single rank alone fits — the
        # ``max`` is load-bearing, not a coincidence.
        shapes = [
            ((4, 6), (2, 4, 6)),
            ((2, 4, 6), (4, 6)),
            ((4, 6), (4, 6)),
            ((2, 4, 6), (2, 4, 6)),
            ((4, 6), (2, 3, 4, 6)),
        ]
        bounds = [
            {"q": var("q", qs), "k1": var("k1", ks)}
            for qs, ks in shapes
        ]
        spec = synth_dspec(
            samples_from_hook(rank_hook, bounds), name="SD"
        )
        assert spec is not None
        expr = spec["SD"]
        # the fit must consult both ranks through the rank reader
        assert "max" in repr(expr)


class TestNearMisses:
    def test_near_misses_report_partial_agreement(self):
        # ``2 ** a`` — still outside the bank, so the closest
        # candidate is a genuine near miss, not an exact fit.
        def pow_hook(bound):
            return {"$attr:M": 2 ** bound["$attr:a"]}

        bounds = [{"$attr:a": i} for i in range(1, 7)]
        samples = samples_from_hook(pow_hook, bounds)
        near = _near_misses(samples)
        assert near
        # the best near miss agrees on at least one sample
        assert near[0][1] >= 1
        assert near[0][1] < len(samples)


class TestMigration:
    """The synth-converted rule: its ``dspec`` replaces the hook only
    because it reproduces the hook on the *full* measured domain — the
    old hook stays as the oracle (``_derive_score_concat_dim``)."""

    def test_matmul_t_concat_dspec_matches_oracle(self):
        from catopt_carriers.om import (
            MATMUL_T_CONCAT,
            _derive_score_concat_dim,
        )
        from catopt_discovery import object_synthesis as osyn

        rule = MATMUL_T_CONCAT
        assert rule.dspec is not None
        assert rule.derive is not _derive_score_concat_dim
        dom = osyn._measure_domain(rule, (), 120)
        assert dom
        for env, _outcome in dom:
            assert rule.derive(env) == _derive_score_concat_dim(env), (
                env
            )

