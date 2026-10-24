"""The shipped ``synop_*`` primitives — the zoo-prim mint, delivered.

``bench/drivers/overnight.py --corpus zoo-prim`` minted ``synop_*``
primitive ops for manual subgraph shapes recurring across the held-out
zoo — 21 folds admitted by the full gauntlet
(``docs/reports/results_overnight_prim_v2.json``).  The laws are
core-shipped, default-on: the verbatim records embed in
``mlopt_core.laws.minted`` and the folds ride ``MINTED_RULES`` into
``DEFAULT``.  :mod:`mlopt_discovery.shipped_primitives` is the
denotation registry — ``register()`` seats the torch bindings on the
ambient table so every ``TorchSink`` can lower a folded ``synop_*``.

The pins here cover the shipping contract end to end: records are
frozen and honest (the mint's own site-check measurements ride each
record), the folds are the same objects the mint admitted and live
in core ``DEFAULT``, registration is idempotent and reaches
already-built sinks, each fold fires and wins extraction on a real
bound module, the denotation verifies ``equal`` at every bound site
(the mint's own gate), and the search → lower → verify path delivers
an equivalent module on shipped defaults — no opt-in.
"""

from __future__ import annotations

import mlopt_carriers  # noqa: F401 — carrier bindings in the ambient table
import pytest
import torch
from mlopt_core.cost import count_cost
from mlopt_core.cost.backend import backend_cost
from mlopt_core.cost.params import dag_cost
from mlopt_core.egraph import EGraph, Rewrite
from mlopt_core.ir import Const, Op, generator
from mlopt_core.laws import DEFAULT, tags
from mlopt_discovery import shipped_primitives as sp

_N_PRIM = 21  # the admitted folds (the pool minted 24; 3 declined)

#: One bound module per prim family — GatedLinearAttention binds all
#: 21 admitted folds (the report's ``mods`` lists); it exports once
#: per session here and every per-fold pin runs against its term.
_GLA = "GatedLinearAttention"


@pytest.fixture(scope="module")
def gla_term():
    """``GatedLinearAttention``'s exported IR root — every prim binds it."""
    import mlopt_torch.torch_bridge as tb
    from mlopt_discovery import zoo as zoo_mod

    w = {w_.name: w_ for w_ in zoo_mod.zoo()}[_GLA]
    torch.manual_seed(0)
    model, feed = w.build()
    ir, _tensors = tb.export_to_ir(
        model.eval().double(),
        feed if isinstance(feed, tuple) else (feed,),
    )
    return ir.root


@pytest.fixture(scope="module")
def gla_subterms(gla_term):
    """The module's ``Op`` subterms — the folds' bind domain."""
    from mlopt_discovery.census import _iter_subterms

    return [s for s in _iter_subterms(gla_term) if isinstance(s, Op)]


def _ops_of(term):
    """Op-name multiset of a term, as a set."""
    out: set[str] = set()

    def go(t):
        if isinstance(t, Op):
            out.add(t.op)
            for a in t.args:
                go(a)

    go(term)
    return out


class TestRecords:
    """The embedded mint artifacts — frozen, complete, honest."""

    def test_twenty_one_records_named_by_content_hash(self):
        recs = sp.records()
        assert len(recs) == _N_PRIM
        assert set(recs) == set(sp.SYNOP_SHIPPED)
        for name, rec in recs.items():
            assert name.startswith("synop_")
            assert rec["fold"]["name"] == f"prim:{name}"

    def test_every_record_carries_its_mint_measurements(self):
        for rec in sp.records().values():
            # the fp64 denotation check the mint ran — all sites equal
            assert rec["sites"] == rec["equal_sites"] >= 1
            assert len(rec["mods"]) >= 2  # the k=2 bind gate
            assert rec["alpha"].startswith("(")
            # the grammar-minter verdict rides the record — minted or
            # honestly refused (the op ships on its denotation either
            # way; the verdict is provenance, not a gate)
            assert rec["handler"]["verdict"] in ("minted", "refused")

    def test_machine_minted_ledger_covers_the_shipped_ops(self):
        # the domain-side ledger is the op names this module
        # denotation-registers; the fold/premise names classify
        # through core's MACHINE_SHIPPED
        for synop in sp.SYNOP_SHIPPED:
            assert synop in sp.MACHINE_MINTED
            from mlopt_core.laws.provenance import MACHINE_SHIPPED

            assert f"prim:{synop}" in MACHINE_SHIPPED
            assert f"{synop}.expand" in MACHINE_SHIPPED


class TestMaterialize:
    """The records rebuild into the same objects the mint made."""

    def test_folds_are_pattern_to_synop(self):
        folds = sp.folds()
        assert len(folds) == _N_PRIM
        for f in folds:
            assert f.name.startswith("prim:synop_")
            assert isinstance(f.rhs, Op)
            assert f.rhs.op == sp.synop_name(f.name)
            # every rhs arg is a metavar leaf; attr metavars ride the
            # call's own attrs
            assert all(isinstance(a, str) for a in f.rhs.args)
            assert tags.FUSION in f.tags
            # the cert-leg premise attached exactly as as_rule minted
            premise = getattr(f, "premises", ())
            assert len(premise) == 1
            assert premise[0].name == sp.expand_name(f.rhs.op)

    def test_premises_expand_synop_to_pattern(self):
        by_synop = {f.rhs.op: f for f in sp.folds()}
        for p in sp.premises():
            assert p.name.endswith(".expand")
            fold = by_synop[p.name[: -len(".expand")]]
            assert p.lhs == fold.rhs and p.rhs == fold.lhs

    def test_folds_are_fully_serializable(self):
        from mlopt_core.laws.serialize import missing_hooks

        for f in sp.folds():
            assert missing_hooks(f) == ()

    def test_materialize_is_stable(self):
        assert sp.folds() == sp.folds()
        # interned term identity — the same objects, not copies
        assert sp.folds()[0] is sp.folds()[0]


class TestRegistration:
    """``declare``/``register`` — the ops and their denotations."""

    def test_declare_registers_gen_defs(self):
        sp.declare()
        for fold in sp.folds():
            g = generator(fold.rhs.op)
            assert g is not None
            assert g.n_in == len(fold.rhs.args) and g.n_out == 1
            assert "synthesized primitive" in g.law

    def test_register_seats_ambient_bindings(self):
        import mlopt_torch.torch_bridge as tb

        assert sp.register() == _N_PRIM
        for synop in sp.SYNOP_SHIPPED:
            assert callable(tb._IR_TO_TORCH[synop])

    def test_register_reaches_existing_sinks(self):
        from mlopt_torch.adapters import TorchSink

        sink = TorchSink()  # built BEFORE register() — ambient table
        sp.register()
        assert sink.supported_ops >= sp.SYNOP_SHIPPED

    def test_register_is_idempotent(self):
        sp.register()
        again = sp.register()
        assert again == _N_PRIM

    def test_denotation_replays_the_spelled_subgraph(self):
        """synop_5e58f26875 = v0*(v1*v2+v3*v4)+v5*v6 — eval directly."""
        sp.register()
        import mlopt_torch.torch_bridge as tb

        args = [torch.randn(2, 3).double() for _ in range(7)]
        got = tb._IR_TO_TORCH["synop_5e58f26875"](*args)
        want = args[0] * (args[1] * args[2] + args[3] * args[4]) + (
            args[5] * args[6]
        )
        assert torch.allclose(got, want)

    def test_denotation_threads_attr_metavars(self):
        """synop_4c657cb2c0 carries a155/a75 — bound at the call.

        Denotation: ``v0 + unsqueeze(v1,-1) * unsqueeze(
        select(linear(v2,v3,v4), dim=1, index=a155), dim=a75)``.
        """
        sp.register()
        import mlopt_torch.torch_bridge as tb

        v0 = torch.randn(2, 4, 1).double()
        v1 = torch.randn(2, 4).double()
        lin = torch.nn.Linear(4, 4).double()
        v2, v3, v4 = (
            torch.randn(2, 5, 4).double(),
            lin.weight,
            lin.bias,
        )
        got = tb._IR_TO_TORCH["synop_4c657cb2c0"](
            v0, v1, v2, v3, v4, a155=3, a75=-1
        )
        want = v0 + v1.unsqueeze(-1) * (
            torch.nn.functional.linear(v2, v3, v4)
            .select(dim=1, index=3)
            .unsqueeze(-1)
        )
        assert torch.allclose(got, want)


class TestSpecHelpers:
    """The spec render/bind helpers the bindings dispatch through."""

    def test_spec_keeps_metavar_names(self):
        fold = next(
            f for f in sp.folds() if f.rhs.op == "synop_5e58f26875"
        )
        s = sp.spec("synop_5e58f26875")
        assert s[0] == fold.lhs.op == "add"
        # leaf metavars stay their pattern names — binding keys
        from mlopt_discovery.genkernel import _eval_spec

        bindings = {"add": torch.add, "mul": torch.mul}
        env = {
            a: torch.tensor(float(i))
            for i, a in enumerate(fold.rhs.args, 1)
        }
        out = _eval_spec(s, env, bindings)
        assert torch.allclose(
            out,
            env["v223"]
            * (env["v345"] * env["v346"] + env["v358"] * env["v359"])
            + env["v215"] * env["v216"],
        )

    def test_spec_bind_substitutes_attr_mvars(self):
        bound = sp._spec_bind(
            ("select", "v0", {"dim": "a0", "other": 1}),
            {"a0": 3},
            ("a0",),
        )
        assert bound == ("select", "v0", {"dim": 3, "other": 1})
        # unbound metavar names pass through; list specs stay lists
        kept = sp._spec_bind(
            ["select", "v0", {"dim": "a9"}], {}, ("a0",)
        )
        assert kept == ["select", "v0", {"dim": "a9"}]
        assert sp._spec_bind("v0", {}, ("a0",)) == "v0"

    def test_spec_of_renders_const_and_leaf(self):
        assert sp._spec_of(Const(2.0)) == 2.0
        assert sp._spec_of("v7") == "v7"
        assert sp._spec_of(Op.make("add", "a", "b")) == (
            "add",
            "a",
            "b",
        )


class TestDeliver:
    """Each fold fires on its bound module and wins extraction."""

    def test_every_fold_fires_and_extracts(self, gla_term):
        import mlopt_torch.torch_bridge as tb

        sp.register()
        supported = set(tb._IR_TO_TORCH)
        priced = backend_cost(count_cost, supported)
        for fold in sp.folds():
            eg = EGraph()
            eid = eg.add_term(gla_term)
            eg.run([fold], eid, max_iterations=5)
            assert any(
                n.op == fold.rhs.op
                for ec in eg._classes.values()
                for n in ec.nodes
            ), fold.name
            term = eg.extract_best(eid, priced)
            assert fold.rhs.op in _ops_of(term), fold.name

    def test_denotation_equal_on_bound_sites(self, gla_subterms):
        """The mint's own gate, re-run at ship time: all sites equal."""
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.meta import instantiate_pattern
        from mlopt_discovery.oracle import eval_instance

        sp.register()
        for fold in sp.folds():
            sites = [
                m
                for s in gla_subterms
                if (m := _term_match(fold.lhs, s)) is not None
            ][:4]
            assert sites, fold.name
            for m in sites:
                out, _note = eval_instance(
                    instantiate_pattern(fold.lhs, m),
                    instantiate_pattern(fold.rhs, m),
                )
                assert out == "equal", f"{fold.name}: {out}"

    def test_fold_pays_on_a_bound_module(self, gla_term):
        """synop_bb8c54c201 cuts node count — the shipped fold beats."""
        import mlopt_torch.torch_bridge as tb

        sp.register()
        priced = backend_cost(count_cost, set(tb._IR_TO_TORCH))
        fold = next(
            f for f in sp.folds() if f.rhs.op == "synop_bb8c54c201"
        )
        eg0 = EGraph()
        base = dag_cost(
            eg0.extract_best(eg0.add_term(gla_term), priced), priced
        )
        eg = EGraph()
        eid = eg.add_term(gla_term)
        eg.run([fold], eid, max_iterations=5)
        got = dag_cost(eg.extract_best(eid, priced), priced)
        assert got < base

    def test_search_lower_verify_end_to_end(self):
        """HighwayGate on shipped DEFAULT: the fold delivers, no opt-in."""
        from mlopt_discovery import zoo as zoo_mod
        from mlopt_orchestrator.optimize import lower, search
        from mlopt_torch.adapters import TorchSink, TorchSource

        sp.register()
        w = {w_.name: w_ for w_ in zoo_mod.zoo()}["HighwayGate"]
        torch.manual_seed(0)
        model, feed = w.build()
        feeds = feed if isinstance(feed, tuple) else (feed,)
        res = search(
            model.eval().double(),
            feeds,
            source=TorchSource(),
            max_iterations=20,
        )
        assert "synop_bb8c54c201" in _ops_of(res.term)
        lr = lower(res, feeds, sink=TorchSink(), verify=True)
        assert lr.verified is not None and lr.verified.passed


class TestDeployment:
    """The opt-in seam and the provenance ledger."""

    def test_ruleset_is_a_named_ruleset(self):
        rs = sp.ruleset()
        assert rs.name == "prim_folds"
        assert len(rs.rules) == _N_PRIM
        assert all(isinstance(r, Rewrite) for r in rs)

    def test_folds_in_core_default_premises_never(self):
        # the promotion contract: the folds ARE core DEFAULT — the
        # default-on ship — while the definitional premises stay out
        # of every saturation set (expanding synop_* re-adds the
        # unfolded spelling); the op names are denotations, not laws
        names = {r.name for r in DEFAULT}
        assert {f.name for f in sp.folds()} <= names
        assert names.isdisjoint({p.name for p in sp.premises()})
        assert names.isdisjoint(sp.MACHINE_MINTED)

    def test_prim_default_unions_onto_default(self):
        rs = sp.prim_default()
        names = {r.name for r in rs}
        assert names >= {r.name for r in DEFAULT}
        assert names >= {f.name for f in sp.folds()}

    def test_prim_default_accepts_an_explicit_base(self):
        rs = sp.prim_default(base=sp.ruleset())
        assert {r.name for r in rs} == {f.name for f in sp.folds()}

    def test_law_provenance_classifies_across_both_libraries(self):
        assert sp.law_provenance("prim:synop_bb8c54c201") == "machine"
        assert sp.law_provenance("synop_bb8c54c201.expand") == "machine"
        assert sp.law_provenance("synop_bb8c54c201") == "machine"
        # delegates to the core registry for anything else
        assert sp.law_provenance("conv_bn_fold") == "machine"
        assert sp.law_provenance("select_mul") == "human"
        assert sp.law_provenance(sp.folds()[0]) == "machine"

    def test_synop_name_round_trip(self):
        assert sp.synop_name("prim:synop_x") == "synop_x"
        assert sp.synop_name("synop_x") == "synop_x"
        assert sp.expand_name("prim:synop_x") == "synop_x.expand"
        assert sp.expand_name("synop_x") == "synop_x.expand"
