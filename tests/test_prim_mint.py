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

from pathlib import Path

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


class TestRecordTables:
    """Phase-3 record/table helpers — vocab fragments as shipped data."""

    def test_shipped_from_records_matches_core(self):
        """``prims.yaml``-shaped records rebuild the core table."""
        from mlopt_core.laws import minted

        table = pm.shipped_from_records(minted.prim_records())
        assert table.names == minted.SYNOP_SHIPPED
        # every premise RHS is the FULL manual denotation — no
        # synop survives the unfold, at any tier.
        for name in table.names:
            fold, expand = table.fold_for(name)
            assert fold.rhs.op == name
            assert not any(
                o.startswith("synop")
                for o in pm.pat_ops(expand.rhs)
            )

    def test_merge_shipped_unions_and_prefers_later(self):
        """``merge_shipped`` unions names; later tables win clashes."""
        from mlopt_core.laws import minted

        core = pm.core_shipped()
        vocab = pm.shipped_from_records(minted.prim_records())
        merged = pm.merge_shipped(core, vocab)
        assert merged.names == core.names | vocab.names

    def test_seat_shipped_fills_only_missing(self):
        """The generic seater respects a domain's own bindings."""
        from mlopt_core.laws import minted

        table = pm.shipped_from_records(minted.prim_records())
        name = sorted(table.names)[0]
        bindings = {name: "domain-owned", "mul": torch.mul}
        seated = pm.seat_shipped(table, bindings)
        assert seated == len(table.names) - 1
        assert bindings[name] == "domain-owned"
        others = [n for n in table.names if n != name]
        assert all(bindings[n] is not None for n in others)

    def test_prim_fold_of_matches_minted_for_shipped(self):
        """The record-generic fold twin agrees with ``minted.prim_fold``."""
        from mlopt_core.laws import minted

        name = sorted(
            minted.SYNOP_SHIPPED - minted.SYNOP2_SHIPPED
        )[0]
        fold, expand = pm.prim_fold_of(
            name, minted.prim_records()[name]
        )
        mfold, mexpand = minted.prim_fold(name)
        assert fold.lhs == mfold.lhs and fold.rhs == mfold.rhs
        assert expand.lhs == mexpand.lhs and expand.rhs == mexpand.rhs

    def test_minted_prim_record_carries_expand_only_when_deep(self):
        """Depth-1 records omit ``expand``; deeper tiers embed it."""
        pat = Op.make("mul", "a", Op.make("sigmoid", "b"))
        call = Op.make("ksynop_x", "a", "b")
        expand = Rewrite("ksynop_x.expand", call, pat)
        rule = Rewrite("prim:ksynop_x", pat, call)
        object.__setattr__(rule, "premises", (expand,))
        prim = {
            "alpha": "(mul a (sigmoid b))",
            "depth": 1,
            "mods": ["m1"],
            "sites": 2,
            "equal_sites": 2,
            "handler": {"verdict": "refused"},
        }
        rec = pm.minted_prim_record("ksynop_x", rule, prim)
        assert "expand" not in rec
        assert rec["fold"]["name"] == "prim:ksynop_x"
        deep = dict(prim, depth=2)
        rec2 = pm.minted_prim_record("ksynop2_x", rule, deep)
        assert rec2["expand"]["name"] == "ksynop_x.expand"


class TestVocabDirLoop:
    """Plan 0033 phase 3 — mint → emit → reload → remint.

    The fragment directory is the loop's memory: round 1 mints a
    depth-1 primitive into a scratch ``vocab/`` copy; round 2 loads
    that directory, seats the emitted prim's fold, and mints a
    ``ksynop2_*`` — a primitive composed over the vocabulary the
    first round wrote.  Stub hooks keep the funnel honest (the
    eval gate is measured, the handler grammar refuses on stub
    bindings — which itself exercises the refused-spec emission).
    """

    def _vocab_copy(self, tmp_path):
        import shutil

        dst = tmp_path / "vocab"
        shutil.copytree(
            Path(__file__).resolve().parents[1] / "vocab", dst
        )
        return dst

    @staticmethod
    def _zx_zy_zw():
        return {
            "zx": lambda *a, **k: None,
            "zy": lambda *a, **k: None,
            "zw": lambda *a, **k: None,
        }

    def _pool(self, roots, vocab_dir, **kw):
        args = {
            "bindings": self._zx_zy_zw(),
            "eval_pair": lambda a, b: "equal",
            "denoted": frozenset(),
            "prefix": "ksynop",
            "family_mint": False,
            "budget": 32,
            "vocab_dir": vocab_dir,
        }
        args.update(kw)
        return pm.prim_pool(roots, 2, **args)

    def test_emit_then_remint_mints_depth2(self, tmp_path):
        """Round 1 emits fragments; round 2 composes over them."""
        from mlopt_discovery import vocabdir

        vdir = self._vocab_copy(tmp_path)
        # round 1 — a depth-1 mint over a shape the bank never named.
        r1 = {
            "m1": Op.make("zx", _v("x"), Op.make("zy", _v("y"))),
            "m2": Op.make("zx", _v("p"), Op.make("zy", _v("q"))),
        }
        _c1, meta1, s1 = self._pool(r1, vdir)
        d1 = [m["prim"]["synop"] for m in meta1.values()]
        assert d1 and all(n.startswith("ksynop_") for n in d1)
        assert s1["vocab_emitted"] == len(d1)
        # the fragments landed: prim record, fold law, schema, op row.
        v = vocabdir.load_vocab(vdir, fresh=True)
        for n in d1:
            assert n in v.prims
            assert f"prim:{n}" in v.laws_by_name()
            assert f"fold:{n}" in {r.name for r in v.schemas}
        # the refused handler spec is vocabulary too.
        assert v.families.get("refused_specs")
        assert vocabdir.check_vocab(vdir) == []

        # round 2 — the emitted prim is a mintable atom: a larger
        # shared shape seats its fold and mints a depth-2 primitive.
        inner = lambda a, b: Op.make(  # noqa: E731
            "zx", a, Op.make("zy", b)
        )
        r2 = {
            "m1": Op.make("zw", inner(_v("x"), _v("y")), _v("w")),
            "m2": Op.make("zw", inner(_v("p"), _v("q")), _v("u")),
        }
        _c2, meta2, s2 = self._pool(
            r2, vdir, bootstrap=True, depth=3
        )
        d2 = [
            m["prim"]
            for m in meta2.values()
            if m["prim"]["depth"] == 2
        ]
        assert d2, "no depth-2 mint over the emitted vocabulary"
        assert any(p["synop"].startswith("ksynop2_") for p in d2)
        # the already-vocab'd depth-1 shape was named, not re-minted.
        assert s2["denoted"] >= 1
        # and the second round emitted its own fragments too.
        assert s2["vocab_emitted"] >= 1
        v2 = vocabdir.load_vocab(vdir, fresh=True)
        for p in d2:
            assert p["synop"] in v2.prims
        assert vocabdir.check_vocab(vdir) == []
