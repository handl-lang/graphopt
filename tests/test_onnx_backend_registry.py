"""``mlopt_onnx.register()`` seats the backend on the core registry."""

from __future__ import annotations

import subprocess
import sys

import mlopt_onnx
import pytest
from mlopt_core import backends
from mlopt_core.ports import Sink, Source


def test_register_seats_onnx_backend():
    snap = {
        name: reg for name, reg in backends.DEFAULT_REGISTRY.items()
    }
    try:
        backends.DEFAULT_REGISTRY.clear()
        mlopt_onnx.register()
        reg = backends.get_backend("onnx")
        assert isinstance(reg.source_instance(), Source)
        assert isinstance(reg.sink("onnx"), Sink)
        assert "onnx" in reg.targets
        assert [
            r.name
            for r in backends.find_backends_by_ops({"add", "mul"})
        ] == ["onnx"]
        with pytest.raises(backends.BackendPortsMissing):
            reg.build_backend()
    finally:
        backends.DEFAULT_REGISTRY.clear()
        for r in snap.values():
            backends.DEFAULT_REGISTRY.register(r)


def test_import_does_not_register():
    # Fresh interpreter semantics: importing mlopt_onnx must not seat
    # anything — the register() call is explicit (kernelopt pattern).
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlopt_onnx, mlopt_core.backends as b;"
            "print(b.backend_names())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "onnx" not in out.stdout
