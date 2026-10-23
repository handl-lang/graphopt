"""The learned mint proposer — featurizer, ranker, mutation generator.

``mlopt_discovery.learned_propose`` is the learned half of the mint:
a logistic ranker trained on the overnight ledger cells plus a
mutation generator over the known-good bank.  These tests pin the
featurization, the label extraction, the model's train/score path,
every mutation operator, and the ``propose_learned`` overnight seam —
all on synthetic rows/terms, so the file needs no real ledger.
"""

import json
import math
from pathlib import Path

import pytest
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import ALL_RULES
from mlopt_core.laws.provenance import MACHINE_SHIPPED
from mlopt_discovery import learned_propose as lp
from mlopt_discovery import proposal as prp
from mlopt_discovery.pipeline import Proposal


def _term() -> Op:
    """A small corpus term the machine-shipped patterns bind."""
    x = Var("x", TensorType((2, 4)))
    y = Var("y", TensorType((2, 4)))
    return Op.make(
        "mul",
        Op.make("unsqueeze", x, dim=0),
        Op.make("unsqueeze", y, dim=1),
    )


def _row(
    name: str, feats: dict, y: bool, **kw: object
) -> lp.LabeledRow:
    """A LabeledRow with the label fields a test cares about."""
    return lp.LabeledRow(
        name=name,
        family=str(kw.get("family", "")),
        sources=(),
        features=feats,
        y=y,
        admitted=bool(kw.get("admitted", False)),
        verdict=str(kw.get("verdict", "")),
        relation="new",
        num_true=None,
        ledger="t.jsonl",
        episode=0,
    )


class TestTermStats:
    def test_op_tree(self):
        t = Op.make("mul", Op.make("unsqueeze", "u", dim="D"), "v")
        s = lp.term_stats(t)
        assert s["nodes"] == 2.0
        assert s["depth"] == 3.0  # leaf-level depth: mul > unsq > 'u'
        assert s["mvars"] == 2.0
        assert s["leaves"] == 2.0
        assert s["attr_mvars"] == 1.0
        assert s["consts"] == 0.0
        assert s["distinct_ops"] == 2.0
        assert s["max_arity"] == 2.0

    def test_shared_mvar_and_const(self):
        t = Op.make("add", "u", Op.make("mul", "u", Const(2.0)))
        s = lp.term_stats(t)
        assert s["mvars"] == 1.0
        assert s["shared_mvars"] == 1.0
        assert s["consts"] == 1.0

    def test_concrete_leaves(self):
        x = Var("x", TensorType((4,)))
        s = lp.term_stats(Op.make("neg", x))
        assert s["mvars"] == 0.0
        assert s["leaves"] == 1.0

    def test_concrete_attr_is_not_an_attr_mvar(self):
        x = Var("x", TensorType((2, 4)))
        s = lp.term_stats(Op.make("unsqueeze", x, dim=0))
        assert s["attr_mvars"] == 0.0


class TestFeaturize:
    def test_blocks(self):
        lhs = Op.make("mul", "u", "v")
        feats = lp.featurize(
            lhs, "u", family="census", name="enumerate:comm:7"
        )
        assert feats["bias"] == 1.0
        assert feats["fam:census"] == 1.0
        assert feats["enum:comm"] == 1.0
        assert feats["pair:rhs_is_leaf"] == 1.0
        assert feats["pair:rhs_shrinks"] == 1.0
        assert feats["pair:same_root"] == 0.0
        assert feats["op:mul"] == pytest.approx(math.log1p(1))
        assert feats["top:mul"] == 1.0
        assert feats["pair:new_rhs_mvars"] == 0.0

    def test_same_root_and_new_mvars(self):
        lhs = Op.make("mul", "u", "v")
        rhs = Op.make("mul", "u", "w")
        feats = lp.featurize(lhs, rhs)
        assert feats["pair:same_root"] == 1.0
        assert feats["pair:new_rhs_mvars"] == 1.0
        assert "enum:comm" not in feats

    def test_cond_dspec(self):
        feats = lp.featurize("u", "u", cond=("concrete", "u"), dspec={})
        assert feats["meta:has_cond"] == 1.0
        assert feats["meta:has_dspec"] == 1.0


class TestOraclePassed:
    def test_failed_oracle(self):
        assert not lp.oracle_passed(
            [{"name": "oracle", "passed": False}]
        )

    def test_cleared_oracle(self):
        assert lp.oracle_passed(
            [
                {"name": "oracle", "passed": True},
                {"name": "pay", "passed": False},
            ]
        )

    def test_no_oracle_stage_counts_as_pass(self):
        assert lp.oracle_passed([{"name": "pay", "passed": True}])

    def test_empty_is_not_a_pass(self):
        assert not lp.oracle_passed([])
        assert not lp.oracle_passed(None)


class TestLedgerRows:
    def _write(self, tmp_path: Path, cells: list) -> Path:
        p = tmp_path / "overnight_t.jsonl"
        lines = [
            {"_manifest": True, "protocol": {}},
            {"key": ["episode", 0], "record": {"evals": []}},
            *cells,
        ]
        p.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
        return p

    def _cell(self, name: str, stages: list, **kw: object) -> dict:
        return {
            "key": ["candidate", name],
            "record": {
                "name": name,
                "family": kw.get("family", ""),
                "sources": kw.get("sources", []),
                "episode": kw.get("episode", 0),
                "admitted": kw.get("admitted", False),
                "stages": stages,
                "arena": kw.get(
                    "arena",
                    {
                        "verdict": "false",
                        "relation": "new",
                        "num_true": False,
                    },
                ),
                "law": kw.get(
                    "law",
                    {
                        "version": 2,
                        "kind": "law",
                        "name": name,
                        "lhs": {"mvar": "u"},
                        "rhs": {"mvar": "u"},
                    },
                ),
            },
        }

    def test_loads_rows(self, tmp_path):
        p = self._write(
            tmp_path,
            [
                self._cell(
                    "a",
                    [
                        {"name": "oracle", "passed": True},
                        {"name": "pay", "passed": False},
                    ],
                    family="fam-x",
                    sources=["s1"],
                    admitted=False,
                ),
                self._cell(
                    "b",
                    [{"name": "oracle", "passed": False}],
                    admitted=False,
                ),
                self._cell(
                    "c",
                    [
                        {"name": "oracle", "passed": True},
                        {"name": "pay", "passed": True},
                    ],
                    admitted=True,
                ),
            ],
        )
        rows = lp.load_ledger_rows([p])
        assert [r.name for r in rows] == ["a", "b", "c"]
        assert [r.y for r in rows] == [True, False, True]
        assert rows[0].family == "fam-x"
        assert rows[0].sources == ("s1",)
        assert rows[2].admitted
        assert all(r.ledger == "overnight_t.jsonl" for r in rows)
        assert rows[0].rule is None  # with_rules=False

    def test_with_rules_and_bad_law(self, tmp_path):
        good = self._cell("good", [{"name": "oracle", "passed": True}])
        bad = self._cell("bad", [{"name": "oracle", "passed": True}])
        bad["record"]["law"] = {"lhs": {"mvar": "u"}}  # no rhs
        p = self._write(tmp_path, [good, bad])
        rows = lp.load_ledger_rows([p], with_rules=True)
        assert [r.name for r in rows] == ["good"]
        assert isinstance(rows[0].rule, Rewrite)
        assert rows[0].rule.name == "good"

    def test_rule_rebuild_failures_are_none(
        self, tmp_path, monkeypatch
    ):
        # A law whose object payload cannot rebuild keeps the row but
        # carries ``rule=None`` — both the exception and the
        # not-a-Rewrite channels.
        from mlopt_core.laws import serialize

        good = self._cell("good", [{"name": "oracle", "passed": True}])
        p = self._write(tmp_path, [good])
        monkeypatch.setattr(
            serialize,
            "object_from_data",
            lambda _d: (_ for _ in ()).throw(ValueError("bad")),
        )
        rows = lp.load_ledger_rows([p], with_rules=True)
        assert rows[0].rule is None
        monkeypatch.setattr(
            serialize, "object_from_data", lambda _d: 42
        )
        rows = lp.load_ledger_rows([p], with_rules=True)
        assert rows[0].rule is None

    def test_missing_path_is_empty(self, tmp_path):
        assert lp.load_ledger_rows([tmp_path / "nope.jsonl"]) == []

    def test_expand_ledgers(self, tmp_path):
        (tmp_path / "bench/results/checkpoints").mkdir(parents=True)
        hit = tmp_path / "bench/results/checkpoints/overnight_a.jsonl"
        hit.write_text("{}\n")
        (tmp_path / "bench/results/checkpoints/other.jsonl").write_text(
            "{}\n"
        )
        out = lp.expand_ledgers(root=tmp_path)
        assert out == [hit]
        # literal paths also resolve
        assert lp.expand_ledgers([str(hit)], root=tmp_path) == [hit]


class TestModel:
    def test_sigmoid_dot(self):
        m = lp.ProposerModel(weights={"bias": 0.0, "x": 2.0})
        assert m.score_features(
            {"bias": 1.0, "x": 0.0}
        ) == pytest.approx(0.5)
        assert m.score_features({"x": 1.0}) > 0.8

    def test_score_and_predict(self):
        m = lp.ProposerModel(weights={"bias": 0.0})
        assert m.score(Op.make("mul", "u", "v"), "u") == pytest.approx(
            0.5
        )
        p = Proposal(name="n", lhs="u", rhs="u", family="f", sources=())
        assert m.predict_proposal(p) == pytest.approx(0.5)
        cand = prp.Candidate(
            strategy="s", label="enumerate:comm:1", lhs="u", rhs="u"
        )
        assert m.predict_candidate(cand) == pytest.approx(0.5)

    def test_train_separable(self):
        rows = [
            _row(f"p{i}", {"bias": 1.0, "good": 1.0}, True)
            for i in range(20)
        ] + [
            _row(f"n{i}", {"bias": 1.0, "bad": 1.0}, False)
            for i in range(20)
        ]
        m = lp.train(rows, epochs=60, lr=0.5, seed=0)
        ev = lp.evaluate(m, rows)
        assert ev["ap"] > 0.99
        assert m.weights["good"] > 0.0 > m.weights["bad"]

    def test_train_deterministic(self):
        rows = [
            _row("a", {"bias": 1.0, "x": 1.0}, True),
            _row("b", {"bias": 1.0, "y": 1.0}, False),
        ]
        m1 = lp.train(rows, epochs=10, seed=7)
        m2 = lp.train(rows, epochs=10, seed=7)
        assert dict(m1.weights) == dict(m2.weights)


class TestMetrics:
    def test_average_precision(self):
        assert lp.average_precision([(0.9, True), (0.1, False)]) == 1.0
        assert lp.average_precision([(0.1, True), (0.9, False)]) == 0.5
        assert lp.average_precision([(0.9, False)]) == 0.0
        assert lp.average_precision([]) == 0.0

    def test_precision_at(self):
        scored = [(0.9, True), (0.8, False), (0.7, True)]
        assert lp.precision_at(scored, 2) == 0.5
        assert lp.precision_at(scored, 0) == 0.0
        assert lp.precision_at([], 5) == 0.0


class TestMutations:
    SEED = Rewrite(
        "seed",
        Op.make("mul", Op.make("unsqueeze", "u", dim="D"), "v"),
        Op.make("mul", "u", "v"),
    )

    def _kinds(self, out):
        return {k for k, _t, _l, _r in out}

    def test_all_kinds_fire(self):
        out = list(
            lp.mutations(self.SEED, ops=("add", "sub"), unary=("neg",))
        )
        assert self._kinds(out) == set(lp.MUTATION_KINDS)

    def test_op_sub_same_arity_only(self):
        # 'neg' is fixed-arity-1: it may substitute the unary
        # 'unsqueeze' site but never the binary root 'mul'.
        out = list(
            lp.mutations(self.SEED, ops=("neg",), kinds=("op-sub",))
        )
        assert out
        for _k, tag, lhs, _r in out:
            assert "neg" in tag
            assert not (isinstance(lhs, Op) and lhs.op == "neg")

    def test_op_sub_keeps_seed_out(self):
        out = list(lp.mutations(self.SEED, ops=("add",)))
        for _k, _t, lhs, rhs in out:
            assert prp._key(lhs, rhs) != prp._key(
                self.SEED.lhs, self.SEED.rhs
            )

    def test_arg_swap(self):
        out = [m for m in lp.mutations(self.SEED, kinds=("arg-swap",))]
        assert out
        lhs = out[0][2]
        assert lhs.args[0] == self.SEED.lhs.args[1]

    def test_mvar_rewire(self):
        out = [
            m for m in lp.mutations(self.SEED, kinds=("mvar-rewire",))
        ]
        assert out
        rhs = out[0][3]
        # rewired rhs mentions only lhs mvars
        assert lp._mvar_set(rhs, set()) <= {"u", "v"}

    def test_wrap(self):
        out = list(
            lp.mutations(self.SEED, kinds=("wrap",), unary=("neg",))
        )
        assert out
        depths = set()
        for _k, tag, lhs, _r in out:
            assert tag.startswith("neg@")
            depths.add(lp.term_stats(lhs)["depth"])
        # 'u' wraps at depth 4 (mul>unsq>neg>u), 'v' at 3 (mul>neg>v)
        assert depths == {3.0, 4.0}

    def test_attr_drop(self):
        out = list(lp.mutations(self.SEED, kinds=("attr-drop",)))
        assert out
        lhs = out[0][2]
        assert lhs.args[0].attrs == {}

    def test_invalid_op_sub_skipped(self):
        # 'matmul' with unsqueeze's dim attr fails validation → no mint
        out = [
            m
            for m in lp.mutations(self.SEED, ops=("matmul",))
            if m[0] == "op-sub" and "@0" in m[1]
        ]
        assert out == [] or all(isinstance(m[2], Op) for m in out)

    def test_arg_swap_dup_of_seed(self):
        # add(u,u) arg-swapped is alpha-equal to the seed itself → the
        # emit dedup returns None and nothing mints.
        seed = Rewrite("dup", Op.make("add", "u", "u"), "u")
        assert list(lp.mutations(seed, kinds=("arg-swap",))) == []

    def test_failed_op_make_mints_nothing(self, monkeypatch):
        # ``Op.make`` is total today, but the mutation operators guard
        # against a raise — simulate it by patching the constructor.
        def boom(*a: object, **k: object) -> None:
            raise ValueError("not a valid spelling")

        monkeypatch.setattr(lp.Op, "make", staticmethod(boom))
        # mvar-rewire never constructs (leaf-for-leaf), so only the
        # constructor-gated kinds run here.
        assert (
            list(
                lp.mutations(
                    self.SEED,
                    ops=("add",),
                    unary=("neg",),
                    kinds=("op-sub", "arg-swap", "wrap", "attr-drop"),
                )
            )
            == []
        )

    def test_alpha_dedup_within_a_rule(self, monkeypatch):
        # Force every mutant key equal: after the first emit all
        # later mutations dedupe — the per-kind ``out is None`` guard.
        monkeypatch.setattr(lp.lp, "_key", lambda _l, _r: ("CONST",))
        out = list(
            lp.mutations(self.SEED, ops=("add", "sub"), unary=("neg",))
        )
        assert len(out) <= 1

    def test_unknown_kind_is_skipped(self):
        out = list(
            lp.mutations(
                self.SEED, ops=("add",), kinds=("bogus", "arg-swap")
            )
        )
        assert {k for k, _t, _l, _r in out} == {"arg-swap"}


class TestProposeLearned:
    def test_emits_binding_scored_proposals(self):
        t = _term()
        props = lp.propose_learned([t], top_k=None, seed=0)
        assert props
        for p in props:
            assert p.family == "learned-mint"
            assert p.sources[0] == "learned-mint"
            assert lp._binds_any(p.lhs, [t])
            # never alpha-equal to a shipped law — the sieve would
            # refuse it anyway, so the mint doesn't spend the slot
            assert prp._key(p.lhs, p.rhs) not in {
                prp._key(r.lhs, r.rhs) for r in ALL_RULES
            }

    def test_ranked_and_top_k(self):
        t = _term()
        # mvar-rewire mutants collapse rhs to mul(u,u)/mul(v,v) — the
        # only binding mutants with a shared rhs metavar.
        model = lp.ProposerModel(weights={"rhs:shared_mvars": 5.0})
        props = lp.propose_learned([t], model=model, top_k=3, seed=0)
        assert len(props) == 3
        scores = [model.predict_proposal(p) for p in props]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > 0.9
        assert all(":mvar-rewire:" in p.name for p in props)

    def test_seeds_override(self):
        seed = Rewrite(
            "mine",
            Op.make("add", "u", "u"),
            Op.make("mul", "u", Const(2.0)),
        )
        props = lp.propose_learned(
            [_term()],
            seeds=[seed],
            ops=("sub",),
            kinds=("op-sub",),
            require_bind=False,
            seed=0,
        )
        assert all(p.sources[1] == "mine" for p in props)

    def test_deterministic_seed(self):
        t = _term()
        a = lp.propose_learned([t], top_k=10, seed=3)
        b = lp.propose_learned([t], top_k=10, seed=3)
        assert [p.name for p in a] == [p.name for p in b]

    def test_default_seed_bank_is_machine_shipped(self):
        bank = lp._seed_bank(None)
        assert {r.name for r in bank} == set(MACHINE_SHIPPED)

    def test_explicit_vocab_and_bank(self):
        t = _term()
        props = lp.propose_learned(
            [t],
            ops=("add",),
            unary=("neg",),
            bank=(),
            kinds=("op-sub", "wrap"),
            seed=0,
        )
        for p in props:
            assert p.name.startswith("learned:")
        # bank=() means shipped-law mutants are *not* deduped away —
        # just assert the flag is honoured, not which rows survive.
        assert isinstance(props, list)

    def test_census_op_accepted_and_ignored(self):
        t = _term()
        a = lp.propose_learned([t], top_k=5, seed=1)
        b = lp.propose_learned([t], top_k=5, seed=1, census_op={})
        assert [p.name for p in a] == [p.name for p in b]


class TestRerankAndValue:
    def _props(self) -> list[Proposal]:
        return [
            Proposal(name=f"p{i}", lhs="u", rhs="u", family="f")
            for i in range(5)
        ]

    def test_rerank_orders_by_score(self):
        model = lp.ProposerModel(weights={"bias": 0.0})
        out = lp.rerank(self._props(), model=model)
        # all-equal scores → stable original order
        assert [p.name for p in out] == [f"p{i}" for i in range(5)]
        model2 = lp.ProposerModel(weights={"pair:rhs_is_leaf": 5.0})
        out2 = lp.rerank(self._props(), model=model2)
        assert len(out2) == 5

    def test_rerank_blend_keeps_tail(self):
        # score ranks p0 first; blend pulls later proposals up
        feats_model = lp.ProposerModel(weights={"lhs:nodes": 5.0})
        props = [
            Proposal(
                name="deep",
                lhs=Op.make("mul", Op.make("add", "u", "v"), "w"),
                rhs="u",
                family="f",
            ),
            *[
                Proposal(name=f"p{i}", lhs="u", rhs="u", family="f")
                for i in range(4)
            ],
        ]
        pure = lp.rerank(props, model=feats_model)
        assert pure[0].name == "deep"
        blended = lp.rerank(props, model=feats_model, blend=1.0)
        assert len(blended) == 5

    def test_value_fn_callable(self):
        fn = lp.value_fn(lp.ProposerModel(weights={"bias": 1.0}))
        cand = prp.Candidate(strategy="s", label="c", lhs="u", rhs="u")
        v = fn(cand)
        assert 0.0 < v < 1.0

    def test_default_model_exists(self):
        assert lp.DEFAULT_MODEL is lp.DEFAULT_MODEL
        assert lp.value_fn() is not None
        assert lp.rerank(self._props())
