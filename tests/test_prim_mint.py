"""The domain-agnostic prim-mint machinery — ``prim_mint.prim_pool``.

``mlopt_discovery.prim_mint`` is the promoted home of the
``synop_*`` primitive mint: the ``zoo-prim`` overnight driver wires
the torch-domain hooks (``_IR_TO_TORCH``, ``oracle.eval_instance``,
the substrate bank's denoted set, ``shipped_primitives.register``)
and every domain repo supplies its own.  These tests pin the
*hook-parametrized* surface the thin ``bench/drivers/overnight.py``
delegates do not reach — the custom prefix, the ``allowed_ops``
denoted gate, a non-core ``ShippedPrims`` table, the generic
denotation seater and the stub-hook pool paths — without spending
the real corpus/handler-mint machinery (the wrapped pool covers
those in ``test_overnight.TestCorpusZooPrim``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op, TensorType, Var
from mlopt_core.meta import instantiate_pattern

from mlopt_discovery import prim_mint as pm


def _v(name):
    """A (2,4) tensor leaf."""
    return Var(name, TensorType((2, 4)))


def _shape(a, b):
    """``sub(a, mul(a, sigmoid(b)))`` — the test_overnight shape."""
    return Op.make("sub", a, Op.make("mul", a, Op.make("sigmoid", b)))


def _stub_bindings(*ops):
    """A stub op table — names map to lambdas (never invoked here)."""
    return {op: (lambda *a, **k: None) for op in ops}


class TestHelpers:
    """The promoted term helpers, domain-free."""

    def test_synop_depth_parses_any_prefix(self):
        """``synop``/``ksynop``/``synopN`` tiers; non-synops are 0."""
        assert pm.synop_depth("synop_abc") == 1
        assert pm.synop_depth("synop2_abc") == 2
        assert pm.synop_depth("synop3_ff00") == 3
        assert pm.synop_depth("ksynop_abc") == 1
        assert pm.synop_depth("ksynop4_abc") == 4
        assert pm.synop_depth("add") == 0
        assert pm.synop_depth("kf_mul3") == 0

    def test_bare_call(self):
        """``f(v0, v1)`` over metavar args is a bare call."""
        assert pm.bare_call(Op.make("f", "x", "y"))
        assert not pm.bare_call(Op.make("f", Op.make("g", "x"), "y"))
        assert not pm.bare_call("x")

    def test_pat_spec_roundtrip(self):
        """Pattern -> spec tuple keeps mvar names and attrs."""
        pat = Op.make("mul", "a", Op.make("sigmoid", "b"))
        assert pm.pat_spec(pat) == (
            "mul",
            "a",
            ("sigmoid", "b"),
        )
        pat2 = Op.make("unsqueeze", "x", dim=-1)
        assert pm.pat_spec(pat2) == ("unsqueeze", "x", {"dim": -1})

    def test_pat_ops_and_sub_alphas(self):
        """``pat_ops`` counts the op set; ``sub_alphas`` proper subs."""
        pat = _shape("a", "b")
        assert pm.pat_ops(pat) == {"sub", "mul", "sigmoid"}
        assert len(pm.sub_alphas(pat)) == 2  # mul(...) and sigmoid(b)

    def test_spec_bind_and_freeze(self):
        """Attr metavars resolve through call attrs; freezing fixes them."""
        spec = ("unsqueeze", "x", {"dim": "a0"})
        bound = pm.spec_bind(spec, {"a0": -1}, ("a0",))
        assert bound == ("unsqueeze", "x", {"dim": -1})
        frozen = pm.freeze_spec(spec, {"$attr:a0": -2}, ("a0",))
        assert frozen == ("unsqueeze", "x", {"dim": -2})

    def test_op_subterms_and_modules_bound(self):
        """``{mod: [Op subterm]}`` and the per-module bind count."""
        roots = {"m1": _shape(_v("x"), _v("y")), "m2": _v("z")}
        subs = pm.op_subterms(roots)
        assert set(subs) == {"m1", "m2"}
        assert len(subs["m1"]) == 3 and subs["m2"] == []
        mods = pm.modules_bound(
            Op.make("sub", "a", Op.make("mul", "a", Op.make("sigmoid", "b"))),
            subs,
        )
        assert mods == {"m1"}


class TestDenoted:
    """The already-named gate's collector."""

    def test_bare_call_sides_collect(self):
        """``manual -> f(v*)`` folds and expansions both denote."""
        lhs = Op.make("mul", "a", Op.make("sigmoid", "b"))
        call = Op.make("kf_x", "a", "b")
        fold = Rewrite("f:fold", lhs, call)
        expand = Rewrite("f:expand", call, lhs)
        # neither side is a bare fused call — contributes nothing.
        unrelated = Rewrite(
            "f:other",
            Op.make("add", Op.make("mul", "a", "b"), "c"),
            Op.make("sub", Op.make("mul", "a", "b"), "c"),
        )
        got = pm.denoted_alphas([fold, expand, unrelated])
        from mlopt_discovery.antiunify import _alpha_norm

        assert got == frozenset({_alpha_norm(lhs)})

    def test_allowed_ops_filters_call_side(self):
        """An unlowe​rable call op's denotation does not count."""
        lhs = Op.make("mul", "a", Op.make("sigmoid", "b"))
        fold = Rewrite("f:fold", lhs, Op.make("hdlr_x", "a", "b"))
        assert pm.denoted_alphas([fold]) != frozenset()
        assert pm.denoted_alphas(
            [fold], allowed_ops=frozenset({"kf_mul3"})
        ) == frozenset()
        assert pm.denoted_alphas(
            [fold], allowed_ops=frozenset({"hdlr_x"})
        ) != frozenset()

    def test_extra_alphas_union(self):
        """Declared handler patterns union into the denoted set."""
        assert pm.denoted_alphas([], extra_alphas={"abc"}) == frozenset(
            {"abc"}
        )


class TestShippedTable:
    """The bootstrap vocabulary seam — ``ShippedPrims``."""

    def test_empty_table_is_identity(self):
        """No shipped ops: expansion/vocab/ops are the no-ops."""
        empty = pm.ShippedPrims(
            names=frozenset(), folds=(), fold_for=lambda op: None
        )
        pat = _shape("a", "b")
        assert pm.expand_synops(pat, empty) == pat
        assert pm.synop_ops(pat, empty) == set()
        assert pm.synop_vocab(empty) == {}

    def test_core_table_expands_a_shipped_synop(self):
        """A ``synop_*(v*)`` call re-expands to its manual denotation."""
        from mlopt_core.laws import minted

        name = sorted(minted.SYNOP_SHIPPED)[0]
        fold, _premise = minted.prim_fold(name)
        call = instantiate_pattern(
            fold.rhs,
            {m: _v(f"t{i}") for i, m in enumerate(fold.rhs.args)},
        )
        x = pm.expand_synops(call)
        # the expansion spells only manual ops — no synop survives.
        assert pm.pat_ops(x).isdisjoint(minted.SYNOP_SHIPPED)

    def test_core_vocab_prices_denotations(self):
        """Every shipped op's weight is its expansion's op-count."""
        vocab = pm.synop_vocab()
        assert vocab and all(w >= 2 for w in vocab.values())

    def test_register_denotation_seats_a_binding(self):
        """The generic seater: call pat + denotation pat -> binding."""
        call = Op.make("xsynop_t", "a", "b")
        den = Op.make("mul", "a", Op.make("sigmoid", "b"))
        table = {"mul": torch.mul, "sigmoid": torch.sigmoid}
        pm.register_denotation("xsynop_t", call, den, table)
        a, b = torch.randn(2, 4), torch.randn(2, 4)
        out = table["xsynop_t"](a, b)
        assert torch.allclose(out, a * torch.sigmoid(b))


class TestPoolHooks:
    """``prim_pool`` under stub hooks — the funnel without a domain."""

    def _roots(self):
        return {
            "m1": _shape(_v("x"), _v("y")),
            "m2": _shape(_v("p"), _v("q")),
        }

    def _pool(self, **kw):
        # real torch bindings: the spec eval in the emit gate and
        # the handler stage genuinely run on the mined shapes.
        table = {
            "sub": torch.sub,
            "mul": torch.mul,
            "sigmoid": torch.sigmoid,
        }
        args = {
            "bindings": table,
            "eval_pair": lambda a, b: "equal",
            "denoted": frozenset(),
            "prefix": "ksynop",
        }
        args.update(kw)
        return pm.prim_pool(self._roots(), 2, **args)

    def test_custom_prefix_mints(self):
        """``ksynop_*`` ops + ``prim:ksynop_*`` folds, sites equal."""
        cands, meta, stats = self._pool()
        labels = [c.label for c in cands]
        assert labels and all(
            lbl.startswith("prim:ksynop_") for lbl in labels
        ), labels
        rec = meta[0]["prim"]
        assert rec["synop"].startswith("ksynop_")
        assert rec["equal_sites"] >= 1
        assert stats["kbound"] >= 1
        # the fold's rule object carries the definitional premise.
        rule = cands[0].as_rule()
        assert [p.name for p in rule.premises] == [
            f"{rec['synop']}.expand"
        ]
        assert rule.premises[0].name.endswith(".expand")

    def test_denoted_gate_skips_named_shapes(self):
        """A shape in *denoted* is refused before the op mints."""
        from mlopt_discovery.antiunify import _alpha_norm

        pat = _shape("a", "b")
        _cands, meta, stats = self._pool(
            denoted=frozenset({_alpha_norm(pat)})
        )
        assert stats["denoted"] >= 1
        # the denoted 3-op shape was refused; other mined shapes
        # (the 2-op sub-shapes) still mint — check no pool member
        # records the excluded denotation.
        assert all(
            meta[i]["prim"]["alpha"] != _alpha_norm(pat)
            for i in meta
        )

    def test_unevaluable_ops_skip(self):
        """An op outside the binding table is an honest refusal."""
        _c, _m, stats = self._pool(bindings=_stub_bindings("sub"))
        assert stats["pool"] == 0
        assert stats["unevaluable"] >= 1

    def test_gate_failure_refuses(self):
        """A measured ``unequal`` (or no equal site) drops the shape."""
        _c, _m, stats = self._pool(
            eval_pair=lambda a, b: "unequal"
        )
        assert stats["pool"] == 0
        assert stats["gate_failed"] >= 1
        _c2, _m2, stats2 = self._pool(
            eval_pair=lambda a, b: "skip"
        )
        assert stats2["pool"] == 0
        assert stats2["gate_failed"] >= 1

    def test_kbind_under_threshold(self):
        """Shapes binding fewer than *k* modules never reach gates."""
        _c, _m, stats = self._pool()
        assert stats["mined"] >= stats["kbound"]
        _c2, _m2, stats2 = pm.prim_pool(
            self._roots(),
            3,  # nothing binds 3 modules — the corpus is 2
            bindings=_stub_bindings("sub", "mul", "sigmoid"),
            eval_pair=lambda a, b: "equal",
            denoted=frozenset(),
        )
        assert stats2["kbound"] == 0 and stats2["pool"] == 0

    def test_bootstrap_registers_and_seats(self):
        """The seating loop runs the shipped folds + the seater hook."""
        calls = []
        import mlopt_torch.torch_bridge as tb

        _c, _m, stats = pm.prim_pool(
            self._roots(),
            2,
            # the ambient torch table — any op a shipped denotation
            # spells is evaluable, whatever the seating embeds.
            bindings=tb._IR_TO_TORCH,
            eval_pair=lambda a, b: "equal",
            denoted=frozenset(),
            prefix="ksynop",
            bootstrap=True,
            register_shipped=lambda: calls.append("reg"),
        )
        assert calls == ["reg"]
        assert stats["bootstrap"] is True
        assert stats["synop_passes"] >= 1
