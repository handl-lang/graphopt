"""Judge-time re-seating of minted ``synop*`` denotations.

``docs/reports/realcorpus_mint_r2.md``: the ``--resume`` judging pass
rebuilt prim-fold rules from their ``law`` records but never re-ran
``prim_pool``, so the minted ``synop*`` ops' torch bindings — a
mint-time side effect on the ambient ``_IR_TO_TORCH`` — were absent
and the oracle ``rerr``'d every env (all 24 prim folds).
``real_model_mint._seat_prim_denotations`` re-seats them from the
rules themselves; these tests pin the resume-path gap closed.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from mlopt_core.egraph import Rewrite  # noqa: E402
from mlopt_core.ir import Op, TensorType, Var  # noqa: E402
from mlopt_core.laws.serialize import (  # noqa: E402
    law_from_data,
    law_to_data,
)
from kit_discovery.oracle import eval_instance  # noqa: E402

import bench.drivers.mint.real_model_mint as rmm  # noqa: E402


def _vars() -> tuple[Var, Var]:
    """Two (2,3) fp leaves — the eval env's tensors."""
    return (
        Var("a", TensorType((2, 3))),
        Var("b", TensorType((2, 3))),
    )


def _prim_rule(name: str) -> Rewrite:
    """A depth-1 prim fold: ``mul(a,b) -> <name>(a,b)``."""
    return Rewrite(
        f"prim:{name}",
        Op.make("mul", "a", "b"),
        Op.make(name, "a", "b"),
    )


def _roundtrip(rule: Rewrite) -> Rewrite:
    """The resume path's rebuild — ``law_to_data`` -> ``law_from_data``."""
    return law_from_data(law_to_data(rule))


@pytest.fixture(autouse=True)
def _clean_seated_ops():
    """Undo the seat's process-global registrations after each test.

    ``_seat_prim_denotations`` mirrors the mint's side effects —
    ``op_def`` into ``ir._OP_REGISTRY`` and the binding into the
    ambient ``_IR_TO_TORCH`` — which ``vocabdir.check_vocab`` and the
    binding-coverage contracts audit globally; the fake ``synop_r2seat*``
    names must not leak past the test that minted them.
    """
    import kit_torch.torch_bridge as _tb
    from mlopt_core import ir as _ir

    reg_before = set(_ir._OP_REGISTRY)
    yield
    for k in set(_ir._OP_REGISTRY) - reg_before:
        if k.startswith("synop_r2seat"):
            _ir._OP_REGISTRY.pop(k, None)
    for k in [
        k for k in _tb._IR_TO_TORCH if k.startswith("synop_r2seat")
    ]:
        dict.pop(_tb._IR_TO_TORCH, k, None)


class TestSeatPrimDenotations:
    """The r2 ``rerr`` gap: unbound ``synop*`` RHS -> bound -> equal."""

    def test_unbound_synop_rhs_is_rhs_err(self):
        """Baseline — the r2 failure mode before seating."""
        import kit_torch.torch_bridge as tb

        name = "synop_r2seat02"
        tb._IR_TO_TORCH.pop(name, None)
        a, b = _vars()
        outcome, _note = eval_instance(
            Op.make("mul", a, b), Op.make(name, a, b)
        )
        assert outcome == "rhs-err"

    def test_seated_denotation_evaluates(self):
        """After seating, the same pair measures ``equal`` — not rerr."""
        import kit_torch.torch_bridge as tb

        name = "synop_r2seat03"
        tb._IR_TO_TORCH.pop(name, None)
        rule = _roundtrip(_prim_rule(name))
        seated = rmm._seat_prim_denotations([("t:0", rule)])
        assert seated == 1
        assert name in tb._IR_TO_TORCH
        a, b = _vars()
        outcome, note = eval_instance(
            Op.make("mul", a, b), Op.make(name, a, b)
        )
        assert outcome == "equal", note

    def test_seating_is_idempotent(self):
        """A second pass seats nothing — the binding already exists."""
        name = "synop_r2seat04"
        rule = _roundtrip(_prim_rule(name))
        assert rmm._seat_prim_denotations([("t:0", rule)]) == 1
        assert rmm._seat_prim_denotations([("t:0", rule)]) == 0

    def test_non_synop_rules_untouched(self):
        """A plain rewrite rule seats nothing and stays unbound."""
        rule = _roundtrip(
            Rewrite(
                "plain:add_to_mul",
                Op.make("add", "a", "b"),
                Op.make("mul", "a", "b"),
            )
        )
        assert rmm._seat_prim_denotations([("t:1", rule)]) == 0


class TestResumePremises:
    """realcorpus_mint_r3 F2 — ``rule.premises`` must survive resume.

    ``law_to_data`` cannot carry ``rule.premises`` (a ``Rewrite``
    attribute, not a codec field): the cert leg's only anchor for a
    minted fold is its ``synop_*.expand`` definitional premise, so a
    resume that rebuilds the bare 2-cell judges the claim
    *unanchored* — ``rmint:084``'s ``not derivable (fixed_point)``
    was exactly this artifact.  ``overnight.serialize_rule`` /
    ``rule_from_record`` are the premise-faithful codec;
    ``rmm._resume_rule`` additionally re-derives the definitional
    depth-1 expand for ledgers written before ``premise_laws``.
    """

    @staticmethod
    def _fold(name: str = "synop_r3prem01") -> Rewrite:
        """A depth-1 prim fold carrying its ``.expand`` premise."""
        fold = Rewrite(
            f"prim:{name}",
            Op.make("mul", "a", "b"),
            Op.make(name, "a", "b"),
        )
        expand = Rewrite(
            f"{name}.expand",
            Op.make(name, "a", "b"),
            Op.make("mul", "a", "b"),
            law=f"{name} unfolds to its mined denotation",
        )
        object.__setattr__(fold, "premises", (expand,))
        return fold

    @staticmethod
    def _prim_cand(rule: Rewrite, *, depth: int = 1) -> dict:
        """An rmm-shaped resume candidate for *rule*'s fold."""
        return {
            "id": "rmint:000",
            "law": law_to_data(rule),
            "prim": {
                "prim": {
                    "synop": rule.rhs.op,
                    "depth": depth,
                }
            },
        }

    def test_serialize_rule_carries_premise_laws(self):
        """The record keeps the audit name list AND full premise
        records — recursively serializable."""
        import bench.drivers.overnight as ov

        rec = ov.serialize_rule(self._fold())
        assert rec["premises"] == ["synop_r3prem01.expand"]
        assert len(rec["premise_laws"]) == 1
        assert rec["premise_laws"][0]["name"] == (
            "synop_r3prem01.expand"
        )

    def test_rule_from_record_restores_premises(self):
        """``premise_laws`` rebuilds the exact premise ``Rewrite``."""
        import bench.drivers.overnight as ov

        fold = self._fold()
        back = ov.rule_from_record(ov.serialize_rule(fold))
        assert [p.name for p in back.premises] == [
            "synop_r3prem01.expand"
        ]
        prem = back.premises[0]
        assert prem.lhs == back.rhs
        assert prem.rhs == back.lhs

    def test_rule_from_record_rule_without_premises(self):
        """A premise-free rule round-trips with no premise keys."""
        import bench.drivers.overnight as ov

        rule = Rewrite(
            "plain:add_to_mul",
            Op.make("add", "a", "b"),
            Op.make("mul", "a", "b"),
        )
        rec = ov.serialize_rule(rule)
        assert "premises" not in rec
        assert "premise_laws" not in rec
        back = ov.rule_from_record(rec)
        assert not getattr(back, "premises", ())

    def test_rule_from_record_legacy_names_only_depth1(self):
        """An overnight-style record (``"premises"`` names, no
        ``premise_laws``) re-derives the definitional depth-1 expand
        exactly — ``synop_call -> fold.lhs``."""
        import bench.drivers.overnight as ov

        fold = self._fold()
        rec = law_to_data(fold)
        rec["premises"] = ["synop_r3prem01.expand"]
        back = ov.rule_from_record(rec)
        assert [p.name for p in back.premises] == [
            "synop_r3prem01.expand"
        ]
        prem = back.premises[0]
        assert prem.lhs == back.rhs
        assert prem.rhs == back.lhs

    def test_rule_from_record_rejects_unrecoverable_premise(self):
        """A name-only premise that is not the definitional depth-1
        expand raises — an unanchored claim must not be judged as if
        anchored."""
        import bench.drivers.overnight as ov

        fold = self._fold()
        rec = law_to_data(fold)
        rec["premises"] = ["some_other.law"]
        with pytest.raises(ValueError, match="unanchored"):
            ov.rule_from_record(rec)

    def test_resume_rule_restores_prim_expand(self):
        """The r3 record shape — no premise keys at all — still
        rebuilds the fold WITH its ``.expand`` premise via the prim
        meta's definitional recovery."""
        rule = self._fold()
        cand = self._prim_cand(rule)
        # strip premise keys — the r3 ledger shape
        cand["law"] = law_to_data(rule)
        back = rmm._resume_rule(cand)
        assert back is not None
        assert [p.name for p in back.premises] == [
            "synop_r3prem01.expand"
        ]

    def test_resume_rule_premise_laws_record(self):
        """A new-shape record (``premise_laws`` present) restores
        premises without needing the prim-meta fallback."""
        import bench.drivers.overnight as ov

        rule = self._fold()
        cand = self._prim_cand(rule)
        cand["law"] = ov.serialize_rule(rule)
        back = rmm._resume_rule(cand)
        assert back is not None
        assert [p.name for p in back.premises] == [
            "synop_r3prem01.expand"
        ]

    def test_resume_rule_prim_without_meta_recovers_via_rhs(self):
        """A ``prim-fold`` record missing its ``prim`` meta still
        anchors — the fold's own ``synop*`` RHS names the expand."""
        rule = self._fold()
        cand = {
            "id": "rmint:000",
            "law": law_to_data(rule),
            "family": "prim-fold",
        }
        back = rmm._resume_rule(cand)
        assert back is not None
        assert [p.name for p in back.premises] == [
            "synop_r3prem01.expand"
        ]

    def test_resume_rule_depth2_without_premises_unfaithful(self):
        """A depth-N fold's premise RHS is the expanded denotation
        the record never stored — no silent bare-law rebuild."""
        name = "synop2_r3prem02"
        fold = Rewrite(
            f"prim:{name}",
            Op.make("mul", "a", "b"),
            Op.make(name, "a", "b"),
        )
        cand = self._prim_cand(fold, depth=2)
        cand["law"] = law_to_data(fold)
        assert rmm._resume_rule(cand) is None

    def test_resume_pool_marks_unfaithful_candidate(self, tmp_path):
        """``_resume_pool`` queues faithful rules and marks the
        unfaithful ones ``resume_skip`` — never a bare-law judge."""
        name = "synop2_r3prem03"
        fold = Rewrite(
            f"prim:{name}",
            Op.make("mul", "a", "b"),
            Op.make(name, "a", "b"),
        )
        cand = self._prim_cand(fold, depth=2)
        cand["law"] = law_to_data(fold)
        path = tmp_path / "prior.json"
        path.write_text(json.dumps({"candidates": [cand], "mint": {}}))
        _cands, to_judge, _adm, stats = rmm._resume_pool(str(path))
        assert to_judge == []
        assert stats["resume_unjudged_skipped"] == 1
        assert _cands[0]["admit"]["resume_skip"] is True
        assert "premise" in _cands[0]["admit"]["reason"]
