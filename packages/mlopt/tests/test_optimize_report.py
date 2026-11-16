"""OptimizeReport verify-contract tests — the honesty gate itself."""

from __future__ import annotations

from types import SimpleNamespace

import mlopt_orchestrator.optimize as orch_optimize

from mlopt.models import ModelSpec
from mlopt.optimize import optimize_model


class _FakeGate:
    def __init__(self, passed: bool) -> None:
        self.passed = passed


class _FakeOptimizer:
    """Stands in for Optimizer — returns canned search/lower results."""

    verified_value: object = None

    def __init__(self, *a: object, **kw: object) -> None:
        pass

    def search(self, model: object, x: object, **kw: object) -> object:
        return SimpleNamespace(term=None, n_enodes=0, param_values={})

    def lower(self, res: object, x: object, **kw: object) -> object:
        return SimpleNamespace(verified=_FakeOptimizer.verified_value)


def _spec() -> ModelSpec:
    return ModelSpec(module=object(), inputs=(object(),), meta={"ref": "test:fake"})


def test_failed_verify_reports_unverified(monkeypatch) -> None:  # noqa: ANN001
    """A VerifyReport with passed=False must NOT report verified=True."""
    _FakeOptimizer.verified_value = _FakeGate(passed=False)
    monkeypatch.setattr(orch_optimize, "Optimizer", _FakeOptimizer)
    lowered, report = optimize_model(_spec(), verify=True)
    assert lowered is not None
    assert report.verified is False


def test_passed_verify_reports_verified(monkeypatch) -> None:  # noqa: ANN001
    _FakeOptimizer.verified_value = _FakeGate(passed=True)
    monkeypatch.setattr(orch_optimize, "Optimizer", _FakeOptimizer)
    _, report = optimize_model(_spec(), verify=True)
    assert report.verified is True


def test_no_gate_reports_none(monkeypatch) -> None:  # noqa: ANN001
    _FakeOptimizer.verified_value = None
    monkeypatch.setattr(orch_optimize, "Optimizer", _FakeOptimizer)
    _, report = optimize_model(_spec(), verify=False)
    assert report.verified is None


def test_verify_declined_reports_failed_gate(monkeypatch) -> None:  # noqa: ANN001
    """VerifyDeclinedError is a FAILED gate — verified=False, verdict kept."""
    from mlopt_orchestrator import VerifyDeclinedError

    declined_gate = SimpleNamespace(passed=False, max_abs=0.5, max_rel=0.25)
    declined_result = SimpleNamespace(
        verified=declined_gate,
        stats={"accepted_by": "declined"},
    )

    class _DecliningOptimizer(_FakeOptimizer):
        def lower(self, res: object, x: object, **kw: object) -> object:
            raise VerifyDeclinedError("verify gate declined", result=declined_result)

    monkeypatch.setattr(orch_optimize, "Optimizer", _DecliningOptimizer)
    lowered, report = optimize_model(_spec(), verify=True)
    assert lowered is None
    assert report.verified is False  # a failed gate, not "no gate"
    assert report.verify_max_abs == 0.5
    assert report.verify_max_rel == 0.25
    assert report.verify_accepted_by == "declined"
    assert "VerifyDeclinedError" in (report.error or "")
    assert "verify=False" in report.render()
    assert "max_abs=5.000e-01" in report.render()


def test_verify_declined_without_result_still_false(monkeypatch) -> None:  # noqa: ANN001
    """A bare VerifyDeclinedError (no .result) still reports verified=False."""
    from mlopt_orchestrator import VerifyDeclinedError

    class _BareDecline(_FakeOptimizer):
        def lower(self, res: object, x: object, **kw: object) -> object:
            raise VerifyDeclinedError("no report produced")

    monkeypatch.setattr(orch_optimize, "Optimizer", _BareDecline)
    _, report = optimize_model(_spec(), verify=True)
    assert report.verified is False
    assert report.verify_max_abs is None
    assert report.verify_accepted_by is None
