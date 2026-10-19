"""Tests for the local artifact cache resolver.

``mlopt_discovery.cache_dir()`` / ``intake_dir()`` replace the old
in-tree ``tools/`` paths for the regenerable side-files (the intake
census and the evidence store).  The convention mirrors
``mlopt_core.profile.profiles_dir()``: an env override, then
``$XDG_CACHE_HOME``, then ``~/.cache``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlopt_discovery import CACHE_DIR_ENV, cache_dir, intake_dir


def test_cache_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "custom"))
    assert cache_dir() == tmp_path / "custom"


def test_cache_dir_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == tmp_path / "xdg" / "mlopt"


def test_cache_dir_home_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache_dir() == tmp_path / ".cache" / "mlopt"


def test_intake_dir_is_a_cache_subdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    assert intake_dir() == tmp_path / "intake"


def test_ledger_defaults_point_at_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The intake side-file defaults resolve into the cache, not the tree."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    import importlib

    from mlopt_discovery.intake import ledger

    reloaded = importlib.reload(ledger)
    assert reloaded._DEFAULT_JSON == tmp_path / "intake" / "intake_corpus.json"
    assert reloaded._DEFAULT_TENSORS == tmp_path / "intake" / "intake_tensors.pt"
    importlib.reload(ledger)
