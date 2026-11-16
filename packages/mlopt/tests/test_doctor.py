"""Adapter discovery + ``mlopt doctor`` tests — registry-level only."""

from __future__ import annotations

import mlopt.adapters
import mlopt.models.catalog  # noqa: F401 — populates registry
from mlopt.cli import main
from mlopt.doctor import diagnose
from mlopt.models import get_adapter, list_adapters


def test_load_all_is_idempotent() -> None:
    first = mlopt.adapters.load_all()
    second = mlopt.adapters.load_all()
    assert first == second
    assert set(first) <= set(list_adapters())


def test_discovery_registers_pending_adapters() -> None:
    adapters = list_adapters()
    assert "whisper" in adapters
    assert "kokoro" in adapters
    for adapter in adapters.values():
        for attr in ("name", "description", "available", "unavailable_reason", "load"):
            assert hasattr(adapter, attr), f"{adapter.name} missing {attr}"


def test_kokoro_probe_reports_metadata() -> None:
    adapter = get_adapter("kokoro")
    assert adapter is not None
    info = adapter.probe("hexgrad/Kokoro-82M")
    assert info is not None
    assert info["available"] is True
    assert info["family"] == "kokoro"


def test_doctor_available_kokoro_ref(capsys) -> None:  # noqa: ANN001
    out = diagnose("kokoro:hexgrad/Kokoro-82M")
    assert "adapter:   kokoro" in out
    assert "status:    available" in out
    assert "probe." in out


def test_doctor_available_bare_name() -> None:
    out = diagnose("kokoro")
    assert "adapter:   kokoro" in out
    assert "status:    available" in out


def test_doctor_unknown_ref() -> None:
    out = diagnose("bert-base-uncased")
    assert "adapter:   none" in out
    assert "mlopt models" in out


def test_doctor_command(capsys) -> None:  # noqa: ANN001
    assert main(["doctor", "kokoro:hexgrad/Kokoro-82M"]) == 0
    out = capsys.readouterr().out
    assert "kokoro" in out and "available" in out


def test_doctor_command_unknown(capsys) -> None:  # noqa: ANN001
    assert main(["doctor", "bert-base-uncased"]) == 0
    assert "no registered adapter" in capsys.readouterr().out
