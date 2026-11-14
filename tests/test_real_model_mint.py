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

import pytest

torch = pytest.importorskip("torch")

from mlopt_core.egraph import Rewrite  # noqa: E402
from mlopt_core.ir import Op, TensorType, Var  # noqa: E402
from mlopt_core.laws.serialize import (  # noqa: E402
    law_from_data,
    law_to_data,
)
from mlopt_discovery.oracle import eval_instance  # noqa: E402

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
    import mlopt_torch.torch_bridge as _tb
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
        import mlopt_torch.torch_bridge as tb

        name = "synop_r2seat02"
        tb._IR_TO_TORCH.pop(name, None)
        a, b = _vars()
        outcome, _note = eval_instance(
            Op.make("mul", a, b), Op.make(name, a, b)
        )
        assert outcome == "rhs-err"

    def test_seated_denotation_evaluates(self):
        """After seating, the same pair measures ``equal`` — not rerr."""
        import mlopt_torch.torch_bridge as tb

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
