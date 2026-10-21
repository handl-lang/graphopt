"""Criterion-based selection — ``criteria_cost`` blends and the
``optimize_model(criteria=...)`` param.

The blend is a weighted sum of named cost axes
(:func:`mlopt_orchestrator.criteria.criteria_cost`).  Sum of
additive-per-node models is additive, so extraction's subtractive
local-cost recovery stays exact for the additive axes — the tests
below check that contract, the selection flip when two axes
disagree, the validation surface, and the stats record.
"""

import math

import pytest
import torch
from mlopt_core.cost import (
    _LAUNCH_S,
    _local_roofline,
    dag_cost,
    executor_cost_for,
    flops_cost,
    param_bytes_cost_for,
)
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Param, TensorType, Var
from mlopt_core.ports import CostFn, signature_conforms
from mlopt_orchestrator.criteria import (
    AXES,
    Blend,
    CompiledCriterion,
    Criteria,
    Criterion,
    DepthCriterion,
    FlopsCriterion,
    LatencyCriterion,
    MemoryCriterion,
    criteria_cost,
    peak_bytes_cost,
)
from mlopt_orchestrator import Optimizer


from mlopt_torch.backend import TorchBackend


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape) -> Param:
    return Param(name, TensorType(tuple(shape)))


def _member_storage() -> Op:
    """Compute-light, memory-heavy: one GEMM on a *stored* weight —
    2048 FLOPs, 16 stored values."""
    x = _v("x", 64, 4)
    return Op.make("matmul", x, _p("p_w", 4, 4))


def _member_compute() -> Op:
    """Compute-heavy, memory-free: a larger GEMM on a *data* operand
    plus a neg — 4608 FLOPs, nothing stored."""
    x = _v("x", 64, 4)
    y = _v("y", 4, 8)
    return Op.make("neg", Op.make("matmul", x, y))


def _two_member_graph():
    """One e-class holding the storage member and the compute member
    (the union asserts their equality — the same mechanism a
    witnessed rewrite uses)."""
    a = _member_storage()
    b = _member_compute()
    eg = EGraph()
    ea = eg.add_term(a)
    eb = eg.add_term(b)
    eg.union(ea, eb)
    eg.rebuild()
    return eg, eg.find(ea), a, b


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
#  The blend itself
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
#  Additive safety — the extract_best contract
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  Weighted blend changes selection — two axes disagree
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  optimize_model(criteria=...) — precedence + stats record
# ---------------------------------------------------------------------------


def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8)
    ).eval()


def test_optimize_model_criteria_records_stats():
    """criteria= builds the blend, stats records the normalised
    weights actually priced, and the module stays equivalent."""


    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = Optimizer(backend=TorchBackend(), criteria={"latency": 2.0}).optimize(m, x, verify=False, verbose=False)

    assert stats["criteria"] == {"latency": 1.0}
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_default_run_records_no_criteria():


    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        _mod, stats = Optimizer(backend=TorchBackend()).optimize(m, x, verify=False, verbose=False)

    assert stats["criteria"] is None


def test_optimize_model_explicit_cost_fn_beats_criteria():
    """Precedence: explicit cost_fn > criteria > default — a given
    cost_fn leaves the criteria record empty (none was priced)."""


    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = Optimizer(backend=TorchBackend(), criteria={"memory": 1.0}).optimize(m, x, cost_fn=flops_cost, verify=False, verbose=False)

    assert stats["criteria"] is None
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_bad_criteria_fails_loud():
    """An unknown axis surfaces as ValueError from the optimizer —
    validation isn't deferred to extraction."""


    m = _tiny_model()
    x = torch.rand(4, 8)
    with pytest.raises(ValueError, match="unknown criteria axes"):
        Optimizer(backend=TorchBackend(), criteria={"speed": 1.0}).optimize(m, x, verify=False, verbose=False)



# ---------------------------------------------------------------------------
#  Criterion objects — the pluggable axis contract
# ---------------------------------------------------------------------------


class _NamedCriterion:
    """A user criterion: any object with ``cost_fn(profile)`` — here
    an axis billing each op one credit (a count axis)."""

    name = "opcount"

    def cost_fn(self, profile=None):
        del profile  # uncalibrated axis

        def fn(term, memo=None):
            return float(sum(isinstance(t, Op) for t in _walk(term)))

        return fn


def _walk(term):
    seen, stack, out = set(), [term], []
    while stack:
        t = stack.pop()
        if isinstance(t, Op) and t not in seen:
            seen.add(t)
            out.append(t)
            stack.extend(t.args)
    return out


class _NamelessCriterion:
    """A criterion with no ``name`` — the class name stands in."""

    def cost_fn(self, profile=None):
        del profile
        return flops_cost


class _ConstCriterion:
    """A criterion whose cost fn takes NO memo — the blend calls it
    bare."""

    name = "const"

    def cost_fn(self, profile=None):
        del profile

        def fn(term):
            return 7.0

        return fn


class _WeirdSigCriterion:
    """A criterion whose built fn's signature is uninspectable —
    the blend falls back to a bare call."""

    name = "weird"

    def cost_fn(self, profile=None):
        del profile

        class _F:
            @property
            def __signature__(self):
                raise ValueError("uninspectable")

            def __call__(self, term):
                return 3.0

        return _F()


class _StorageCriterion:
    """A criterion declaring the storage-billing marker — propagated
    to the blend."""

    name = "storey"
    charges_param_only = True

    def cost_fn(self, profile=None):
        del profile
        return flops_cost










# ---------------------------------------------------------------------------
#  Composition — `crit * w`, `a + b`, Criteria/Blend containers
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
#  The memory axis — weights / peak / combined modes
# ---------------------------------------------------------------------------




















# ---------------------------------------------------------------------------
#  optimize_model(criteria=...) — new spec forms
# ---------------------------------------------------------------------------


def test_optimize_model_accepts_criterion_objects():
    """criteria=<Criterion> and criteria=<Blend> price through the
    same path as the dict spelling; stats records resolved axes."""


    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = Optimizer(backend=TorchBackend(), criteria=FlopsCriterion()).optimize(m, x, verify=False, verbose=False)

        assert stats["criteria"] == {"flops": 1.0}
        assert torch.allclose(mod(x), ref, atol=1e-5)
        mod, stats = Optimizer(backend=TorchBackend(), criteria=LatencyCriterion() * 0.5 + MemoryCriterion() * 0.5).optimize(m, x, verify=False, verbose=False)

        assert stats["criteria"] == {"latency": 0.5, "memory": 0.5}
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_criteria_list_and_peak_mode():
    """A list spec and a mode-qualified dict axis both run
    end-to-end."""


    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = Optimizer(backend=TorchBackend(), criteria=[FlopsCriterion(), (DepthCriterion(), 1.0)]).optimize(m, x, verify=False, verbose=False)

        assert stats["criteria"] == {"flops": 0.5, "depth": 0.5}
        assert torch.allclose(mod(x), ref, atol=1e-5)
        mod, stats = Optimizer(backend=TorchBackend(), criteria={"memory:peak": 1.0}).optimize(m, x, verify=False, verbose=False)

        assert stats["criteria"] == {"memory:peak": 1.0}
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_criteria_spec_errors_surface():
    """Bad spec types raise TypeError; bad members raise their own
    errors — validation isn't deferred to extraction."""


    m = _tiny_model()
    x = torch.rand(4, 8)
    with pytest.raises(TypeError, match="unsupported criteria spec"):
        Optimizer(backend=TorchBackend(), criteria=object()).optimize(m, x, verify=False, verbose=False)

    with pytest.raises(TypeError, match="not a Criterion"):
        Optimizer(backend=TorchBackend(), criteria=[object()]).optimize(m, x, verify=False, verbose=False)

