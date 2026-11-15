"""``mlopt_torch.register()`` seats the backend on the core registry."""

from __future__ import annotations

import mlopt_torch
import pytest
from mlopt_core import backends
from mlopt_core.pipeline import Backend
from mlopt_core.ports import Sink, Source


@pytest.fixture
def clean_registry():
    snap = {
        name: reg for name, reg in backends.DEFAULT_REGISTRY.items()
    }
    backends.DEFAULT_REGISTRY.clear()
    yield
    backends.DEFAULT_REGISTRY.clear()
    for r in snap.values():
        backends.DEFAULT_REGISTRY.register(r)


def test_register_seats_torch_backend(clean_registry):
    mlopt_torch.register()
    reg = backends.get_backend("torch")
    assert isinstance(reg.source_instance(), Source)
    assert isinstance(reg.sink("torch"), Sink)
    assert "torch" in reg.targets
    assert [
        r.name for r in backends.find_backends_by_ops({"add", "mul"})
    ] == ["torch"]
    assert isinstance(reg.build_backend(), Backend)


def test_register_duplicate_declines(clean_registry):
    mlopt_torch.register()
    with pytest.raises(ValueError):
        mlopt_torch.register()
