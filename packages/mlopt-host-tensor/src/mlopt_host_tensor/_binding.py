"""Load the ``handl_host_tensor`` PyO3 extension built from host-tensor.

The binding is a cargo cdylib (``cargo build --features python`` in the
``handl-lang/host-tensor`` checkout), not a pip package.  Resolution
order:

1. ``import handl_host_tensor`` — works when the .so's directory is on
   ``sys.path`` (e.g. copied next to a driver, or a maturin install).
2. ``$MLOPT_HOST_TENSOR_LIB`` — explicit path to the ``.so``.
3. The sibling-checkout convention: ``<repo>/../host-tensor/target/
   {release,debug}/handl_host_tensor.so`` — the dev layout this repo
   assumes (graphopt and host-tensor are siblings under
   ``handl-lang/``).  ``$MLOPT_HOST_TENSOR_REPO`` overrides the
   host-tensor checkout path; ``$MLOPT_HOST_TENSOR_TARGET`` overrides
   the cargo profile dir name.

Every failure raises :class:`HostTensorError` with the build command —
the sink never degrades silently.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from mlopt_host_tensor.errors import HostTensorError

_BUILD_HINT = (
    "build the binding: cd <host-tensor checkout> && "
    "cargo build --features python, then copy "
    "target/<profile>/libhandl_host_tensor.so to "
    "handl_host_tensor.so (or set MLOPT_HOST_TENSOR_LIB)"
)

#: ``packages/mlopt-host-tensor/src/mlopt_host_tensor/_binding.py`` →
#: graphopt repo root is four parents up.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _candidate_paths() -> list[Path]:
    """Return the .so locations to try, in order."""
    env = os.environ.get("MLOPT_HOST_TENSOR_LIB")
    if env:
        return [Path(env)]
    repo = Path(
        os.environ.get(
            "MLOPT_HOST_TENSOR_REPO", _REPO_ROOT.parent / "host-tensor"
        )
    )
    profile = os.environ.get("MLOPT_HOST_TENSOR_TARGET", "debug")
    target = repo / "target" / profile
    return [
        target / "handl_host_tensor.so",
        target / "libhandl_host_tensor.so",
    ]


def _load_from_path(path: Path) -> ModuleType:
    """Load the extension module from an explicit .so path."""
    spec = importlib.util.spec_from_file_location(
        "handl_host_tensor", path
    )
    if spec is None or spec.loader is None:
        raise HostTensorError(
            f"cannot load handl_host_tensor extension from {path}"
        )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: PyO3 modules expect to be in sys.modules.
    sys.modules["handl_host_tensor"] = mod
    spec.loader.exec_module(mod)
    return mod


def load() -> ModuleType:
    """Import (or load) the ``handl_host_tensor`` extension module.

    Raises :class:`HostTensorError` listing what was tried when no
    candidate resolves.
    """
    try:
        return importlib.import_module("handl_host_tensor")
    except ImportError:
        pass
    tried: list[str] = []
    for path in _candidate_paths():
        tried.append(str(path))
        if path.is_file():
            try:
                return _load_from_path(path)
            except Exception as exc:
                raise HostTensorError(
                    f"handl_host_tensor extension at {path} failed to "
                    f"load: {exc} — {_BUILD_HINT}"
                ) from exc
    raise HostTensorError(
        "handl_host_tensor extension not found — tried "
        + ", ".join(tried)
        + f"; {_BUILD_HINT}"
    )


#: The loaded module (lazy — import of this package must not require
#: the Rust build; only *using* the sink does).
_MODULE: ModuleType | None = None


def module() -> ModuleType:
    """Return the loaded extension, building the search once."""
    global _MODULE
    if _MODULE is None:
        _MODULE = load()
    return _MODULE


def call(op_name: str, args: list[Any]) -> Any:
    """Call ``tensor.<op_name>`` with JSON-able positional ``args``.

    Returns the decoded JSON result (dict/list/scalar).  ``op_name``
    is the bare host op name (``"add"`` → ``tensor.add``).
    """
    import json

    mod = module()
    return mod.call(f"tensor.{op_name}", json.dumps(args))


def host_ops() -> list[str]:
    """Return the ``tensor.*`` op names the extension registers."""
    return list(module().ops())
