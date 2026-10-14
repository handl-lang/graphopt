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
    BINOPS,
    MIN_SAMPLES,
    _near_misses,
    samples_by_attr,
    samples_from_hook,
    synth_dspec,
    synth_dspec_multi,
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


class TestValueShapeSource:
    """The bank gained the carrier-aware rank reader
    ``("len", ("vshape", T))`` — the derive DSL's value-shape source.
    It is reachable only when a bound term's value shape differs from
    the cost convention; a resolver installs the mapping here."""

    #: (declared/cost shape, true value shape).  The cost shapes all
    #: share rank 3 while the value ranks vary, so the convention
    #: reader is constant and only the value-shape reader fits.
    _PAIRS = (
        ((2, 4, 6), (4, 6)),
        ((2, 4, 8), (4, 6, 8)),
        ((2, 4, 10), (4,)),
        ((2, 4, 12), (4, 6, 8, 10)),
        ((2, 4, 14), (4, 6)),
    )

    def _setup(self, monkeypatch, *, resolver):
        import catopt_core.typing as ty
        from catopt_core.ir import TensorType, Var

        bounds = [
            {"t": Var("t", TensorType(tuple(cs)))}
            for cs, _vs in self._PAIRS
        ]
        vmap = {
            b["t"]: vs
            for b, (_cs, vs) in zip(bounds, self._PAIRS, strict=True)
        }
        monkeypatch.setattr(
            ty,
            "_VALUE_SHAPE",
            (lambda x: vmap.get(x)) if resolver else None,
        )

        def rank_hook(bound):
            return {"$attr:R": len(vmap[bound["t"]])}

        return samples_from_hook(rank_hook, bounds)

    def test_finds_the_value_shape_rank(self, monkeypatch):
        samples = self._setup(monkeypatch, resolver=True)
        spec = synth_dspec(samples, name="R")
        assert spec is not None
        assert spec["R"] == ("len", ("vshape", "t"))

    def test_without_a_resolver_the_source_declines(self, monkeypatch):
        # the cost-convention rank reader cannot reproduce the value
        # ranks, so the synth honestly refuses
        samples = self._setup(monkeypatch, resolver=False)
        assert synth_dspec(samples, name="R") is None


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


class TestExtendedBank:
    """The ops the residual survey measured as missing, landed:
    ``mod`` (``%`` axis normalisation) and the ``("dim", S, i)`` dim
    readers (cost convention and carrier-aware)."""

    def test_mod_is_in_the_bank(self):
        assert "mod" in BINOPS

    def test_finds_mod_normalisation(self):
        # ``dim % rank`` — the axis normalisation a carrier-aware
        # derive spells; the rank bank cannot express it without ``%``.
        # The divisor is the rank reader (the observed-value literals
        # carry no ``3``), exactly the ``dim % rank`` idiom.
        from catopt_core.ir import TensorType, Var

        bounds = [
            {"$attr:a": i, "t": Var("t", TensorType((2, 3, 5)))}
            for i in range(1, 7)
        ]

        def mod_hook(bound):
            rank = len(bound["t"].typ.shape)
            return {"$attr:M": bound["$attr:a"] % rank}

        spec = synth_dspec(
            samples_from_hook(mod_hook, bounds), name="M"
        )
        assert spec is not None
        assert "mod" in repr(spec["M"])

    def test_finds_a_dim_reader(self):
        # ``("dim", ("shape", t), -1)`` — a concrete dim, not just the
        # rank the ``("len", ("shape", t))`` reader reaches.
        from catopt_core.ir import TensorType, Var

        bounds = [
            {"t": Var("t", TensorType((2, 3, i)))}
            for i in range(4, 10)
        ]

        def dim_hook(bound):
            return {"$attr:L": bound["t"].typ.shape[-1]}

        spec = synth_dspec(
            samples_from_hook(dim_hook, bounds), name="L"
        )
        assert spec is not None
        assert "dim" in repr(spec["L"])


class TestMultiOutput:
    """A hook returning several attrs is fitted per-attr — a ``dspec``
    is a ``{NAME: expr}`` map, so the flattened single-expr fit is not
    a spelling of the hook at all."""

    def _bounds(self):
        return [
            {"$attr:a": i, "$attr:c": 3 * i} for i in range(1, 7)
        ]

    def _pair_hook(self, bound):
        a, c = bound["$attr:a"], bound["$attr:c"]
        return {"$attr:S": a + c, "$attr:D": a - c}

    def test_groups_samples_by_attr(self):
        by = samples_by_attr(self._pair_hook, self._bounds())
        assert sorted(by) == ["D", "S"]
        assert len(by["S"]) == len(by["D"]) == 6

    def test_flattening_is_not_a_spelling_of_the_hook(self):
        # the two attrs have different values on one bound, so no
        # single expr reproduces the flattened list
        flat = samples_from_hook(self._pair_hook, self._bounds())
        assert len(flat) == 12
        assert synth_dspec(flat, name="S") is None

    def test_fits_each_attr_separately(self):
        bounds = self._bounds()
        spec = synth_dspec_multi(self._pair_hook, bounds)
        assert spec is not None
        assert sorted(spec) == ["D", "S"]
        for b in bounds:
            assert eval_derive(spec, b) == self._pair_hook(b)

    def test_partial_map_refused(self):
        # one attr fittable, one not (``2 ** a`` is outside the bank)
        def hook(bound):
            a, c = bound["$attr:a"], bound["$attr:c"]
            return {"$attr:S": a + c, "$attr:P": 2 ** a}

        assert synth_dspec_multi(hook, self._bounds()) is None

    def test_empty_hook_refused(self):
        assert synth_dspec_multi(lambda _b: None, self._bounds()) is None


class TestOpTermBank:
    """``oracle._term_bindings`` — the carrier op-term source the
    deepened measurement domain binds, so a metavariable resolves to a
    compound carrier member (the only condition under which a derive's
    ``_vshape`` read diverges from the cost convention)."""

    def test_op_terms_differ_in_value_shape(self):
        import catopt_carriers.xcarrier  # noqa: F401  registers _xshape
        from catopt_core.typing import _shape_of, value_shape_of
        from catopt_discovery import oracle as lvo

        terms = lvo._op_terms("q")
        assert terms
        for t in terms:
            assert _shape_of(t) != value_shape_of(t)

    def test_term_bindings_leads_with_the_op_terms(self):
        from catopt_discovery import oracle as lvo

        op_terms = lvo._op_terms("q")
        bank = lvo._term_bindings("q", set(), ())
        assert bank[: len(op_terms)] == op_terms
        assert len(bank) > len(op_terms)


class TestDeepDomain:
    """``_measure_domain(deep=True)`` unions the carrier op-term window
    with the leaf windows — bounded by ``synth_limit`` exactly as they
    are.  It is opt-in: the leaf domain is the guard-synthesis default."""

    def _rule(self):
        from catopt_carriers.om import MATMUL_T_CONCAT

        return MATMUL_T_CONCAT

    @staticmethod
    def _vshape_gap(dom):
        """Envs where some bound term's value shape differs."""
        import catopt_carriers.xcarrier  # noqa: F401  registers _xshape
        from catopt_core.typing import _shape_of, value_shape_of

        return [
            env
            for env, _o in dom
            if any(
                _shape_of(v) != value_shape_of(v)
                for k, v in env.items()
                if not k.startswith("$attr:")
            )
        ]

    def test_leaf_domain_has_no_vshape_gap(self):
        from catopt_discovery import object_synthesis as osyn

        dom = osyn._measure_domain(self._rule(), (), 60)
        assert dom
        assert self._vshape_gap(dom) == []

    def test_deep_domain_reaches_the_vshape_gap(self):
        from catopt_discovery import object_synthesis as osyn

        dom = osyn._measure_domain(self._rule(), (), 60, deep=True)
        assert self._vshape_gap(dom)

    def test_deep_domain_is_a_bounded_union(self):
        from catopt_discovery import object_synthesis as osyn

        limit = 60
        leaf = osyn._measure_domain(self._rule(), (), limit)
        deep = osyn._measure_domain(self._rule(), (), limit, deep=True)
        # the leaf window stays; the deep window is added, deduped,
        # and capped by the same limit — three windows at most
        assert len(deep) > len(leaf)
        assert len(deep) <= 3 * limit


class TestShippedDspecIsUnfaithfulOffTheLeafDomain:
    """The deepening's headline finding: ``MATMUL_T_CONCAT``'s shipped
    ``dspec`` reads the cost convention ``("shape", T)`` while its
    oracle hook reads ``_vshape`` — the two agree only on leaf sites.
    The deep window exposes the divergence; the correct spelling is
    ``("vshape", T)``."""

    def _rule(self):
        from catopt_carriers.om import MATMUL_T_CONCAT

        return MATMUL_T_CONCAT

    def test_agrees_on_the_leaf_domain(self):
        from catopt_carriers.om import _derive_score_concat_dim
        from catopt_discovery import object_synthesis as osyn

        rule = self._rule()
        dom = osyn._measure_domain(rule, (), 60)
        assert dom
        for env, _o in dom:
            assert rule.derive(env) == _derive_score_concat_dim(env)

    def test_diverges_on_carrier_member_sites(self):
        from catopt_carriers.om import _derive_score_concat_dim
        from catopt_discovery import object_synthesis as osyn

        rule = self._rule()
        dom = osyn._measure_domain(rule, (), 60, deep=True)
        bad = [
            env
            for env, _o in dom
            if rule.derive(env) != _derive_score_concat_dim(env)
        ]
        assert bad, "the deep window must expose the divergence"

    def test_the_vshape_spelling_would_agree(self):
        from catopt_carriers.om import _derive_score_concat_dim
        from catopt_discovery import object_synthesis as osyn

        rule = self._rule()
        dom = osyn._measure_domain(rule, (), 60, deep=True)
        fixed = {
            "SD": (
                "sub",
                (
                    "max",
                    ("len", ("vshape", "q")),
                    ("len", ("vshape", "k1")),
                ),
                1,
            )
        }
        for env, _o in dom:
            want = _derive_score_concat_dim(env)
            if want is None:
                continue
            assert eval_derive(fixed, env) == want

