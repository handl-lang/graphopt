"""Contract-honesty tests — backend refusal, ref semantics, plugin
isolation, registration validation, delivery gates, kwargs plumbing.

No torch/model downloads required; fakes stand in for adapters and
the optimizer.
"""

from __future__ import annotations

import pytest

import mlopt.models.catalog  # noqa: F401 — populates the registry once
from mlopt.models import _ADAPTERS, ModelSpec, get_adapter, register_adapter, resolve_ref

# --- ref semantics ---------------------------------------------------------


def test_resolve_ref_bare_adapter_name() -> None:
    # The audit bug: `doctor kokoro` worked but `optimize kokoro` did
    # not — resolve_ref must route bare adapter names everywhere.
    assert resolve_ref("kokoro") is get_adapter("kokoro")
    assert resolve_ref("whisper") is get_adapter("whisper")


def test_resolve_ref_strips_whitespace() -> None:
    assert resolve_ref("  kokoro  ") is get_adapter("kokoro")
    assert resolve_ref("  whisper:openai/whisper-tiny  ") is not None


def test_resolve_ref_prefixed_name_still_wins() -> None:
    assert resolve_ref("kokoro:hexgrad/Kokoro-82M") is get_adapter("kokoro")


# --- registration validation ------------------------------------------------


class _FakeAdapter:
    name = "fake-contract-adapter"
    description = "test"
    available = True
    unavailable_reason = ""

    def claims(self, ref: str) -> bool:
        return False

    def load(self, ref: str, **opts: object) -> ModelSpec:
        return ModelSpec(module=object())

    def probe(self, ref: str) -> dict | None:
        return None


def test_register_adapter_rejects_missing_surface() -> None:
    class _Bare:
        name = "bare"

    with pytest.raises(TypeError, match="missing ModelAdapter surface"):
        register_adapter(_Bare())  # ty: ignore[invalid-argument-type]


def test_register_adapter_rejects_uncallable_load() -> None:
    class _Bad:
        name = "bad-load"
        description = "x"
        available = True
        unavailable_reason = ""
        load = 42  # type: ignore[assignment]

    with pytest.raises(TypeError, match="not callable"):
        register_adapter(_Bad())  # ty: ignore[invalid-argument-type]


def test_register_adapter_rejects_empty_name() -> None:
    class _Nameless(_FakeAdapter):
        name = "   "

    with pytest.raises(TypeError, match="non-empty string"):
        register_adapter(_Nameless())


def test_register_adapter_collision_warns_first_wins() -> None:
    first = _FakeAdapter()
    second = _FakeAdapter()
    register_adapter(first)
    try:
        with pytest.warns(RuntimeWarning, match="collision"):
            register_adapter(second)
        assert get_adapter("fake-contract-adapter") is first
    finally:
        _ADAPTERS.pop("fake-contract-adapter", None)


# --- backend honesty ---------------------------------------------------------


def test_optimize_model_rejects_non_torch_backend() -> None:
    from mlopt.optimize import optimize_model

    spec = ModelSpec(module=object(), inputs=())
    with pytest.raises(ValueError, match="not wired"):
        optimize_model(spec, backend="onnx")


# --- delivery gates -----------------------------------------------------------


def test_deliver_game_kwargs_reach_bench() -> None:
    """``call_kwargs`` must reach every ``module(*inputs, **kwargs)``
    the delivery game times — a kwargs-needing module benches honestly."""
    from mlopt.deliver import _bench

    calls: list[dict] = []

    class _Mod:
        def __call__(self, x, *, flag=None):
            calls.append({"x": x, "flag": flag})
            return x

    _bench(_Mod(), (1,), iters=2, warmup=0, call_kwargs={"flag": "on"})
    assert calls and all(c["flag"] == "on" for c in calls)


def test_optimize_negative_budget_notes_report() -> None:
    """A negative --deliver-budget must surface in report.notes, not
    silently fall back to the heuristic."""
    import mlopt.optimize as optz

    class _Lowered:
        module = object()
        verified = None

    import mlopt_orchestrator.optimize as orch_opt

    class _FakeOptz:
        def search(self, *a, **k):
            return object()

        def lower(self, *a, **k):
            return _Lowered()

    orig_optz = orch_opt.Optimizer
    orch_opt.Optimizer = lambda **k: _FakeOptz()  # ty: ignore[invalid-assignment]
    try:
        # deliver path with a negative budget: game skipped, note added
        # — drive it far enough to observe report.notes.
        lowered, report = optz.optimize_model(
            ModelSpec(module=object(), inputs=()),
            verify=False,
            deliver=True,
            deliver_budget=-3,
        )
    finally:
        orch_opt.Optimizer = orig_optz
    assert any("deliver-budget" in n and "-3" in n for n in report.notes)


# --- plugin isolation ---------------------------------------------------------


def test_load_all_isolates_broken_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin module that raises on import becomes an unavailable
    pseudo-adapter — discovery never crashes."""
    import importlib

    import mlopt.adapters as adapters_pkg
    from mlopt.models import _ADAPTERS, list_adapters

    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "mlopt.adapters.whisper":
            raise ImportError("boom")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(adapters_pkg, "_CACHE", None)
    # Snapshot and clear the registry so the isolated run is observable.
    saved = dict(_ADAPTERS)
    _ADAPTERS.clear()
    try:
        loaded = adapters_pkg.load_all()
        assert "whisper" in loaded
        assert loaded["whisper"].available is False
        assert "import failed" in loaded["whisper"].unavailable_reason
        assert "kokoro" in list_adapters()
    finally:
        _ADAPTERS.clear()
        _ADAPTERS.update(saved)
        monkeypatch.setattr(adapters_pkg, "_CACHE", None)


# --- search timeout knob ------------------------------------------------------


def test_eqsat_timeout_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MLOPT_EQSAT_TIMEOUT`` is opt-in: unset → ``None`` (the engine's
    own 300 s default stands); set → seconds; bad values are named."""
    from mlopt.optimize import _eqsat_timeout_seconds

    monkeypatch.delenv("MLOPT_EQSAT_TIMEOUT", raising=False)
    assert _eqsat_timeout_seconds() is None
    monkeypatch.setenv("MLOPT_EQSAT_TIMEOUT", "900")
    assert _eqsat_timeout_seconds() == 900.0
    monkeypatch.setenv("MLOPT_EQSAT_TIMEOUT", "junk")
    with pytest.raises(ValueError, match="MLOPT_EQSAT_TIMEOUT"):
        _eqsat_timeout_seconds()
    monkeypatch.setenv("MLOPT_EQSAT_TIMEOUT", "-5")
    with pytest.raises(ValueError, match="positive"):
        _eqsat_timeout_seconds()


def test_optimize_model_wires_eqsat_timeout_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MLOPT_EQSAT_TIMEOUT`` reaches ``Optimizer(engine=...)`` as a
    zero-arg factory producing ``EqsatEngine(timeout=...)``; unset
    leaves ``engine=None`` (the default eqsat path)."""
    from types import SimpleNamespace

    import mlopt_orchestrator.optimize as orch_opt

    import mlopt.optimize as optz

    captured: dict = {}

    class _Optz:
        def __init__(self, **kw):  # noqa: ANN003, ANN204
            captured.clear()
            captured.update(kw)

        def search(self, *a, **k):  # noqa: ANN002, ANN003, ANN202
            return SimpleNamespace(term=None, n_enodes=0, param_values={})

        def lower(self, *a, **k):  # noqa: ANN002, ANN003, ANN202
            return SimpleNamespace(module=object(), verified=None)

    monkeypatch.setattr(orch_opt, "Optimizer", _Optz)
    spec = ModelSpec(module=object(), inputs=())

    monkeypatch.delenv("MLOPT_EQSAT_TIMEOUT", raising=False)
    optz.optimize_model(spec, verify=False)
    assert captured["engine"] is None

    monkeypatch.setenv("MLOPT_EQSAT_TIMEOUT", "777")
    _lowered, report = optz.optimize_model(spec, verify=False)
    engine = captured["engine"]
    assert engine is not None
    # functools.partial(EqsatEngine, timeout=...) — the orchestrator's
    # _resolve_engine calls zero-arg factories lazily at search time.
    assert engine.keywords["timeout"] == 777.0
    assert engine.func.__name__ == "EqsatEngine"
    assert any("MLOPT_EQSAT_TIMEOUT" in n and "777" in n for n in report.notes)
