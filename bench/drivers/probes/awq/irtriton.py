"""Term -> `compile-triton` -> a runnable Triton module.

The lowering lives in Rust; this is only the boundary. The term is
serialised (`lib_kernel_ir.to_json`), the binary answers with source, and
the source is executed into a fresh module namespace. Nothing here knows
what a GEMV is — if this file had to know, the lowering would not be
independent enough to be worth having.

    mod = compile_term(qgemv(k=4864, split=4))
    out = mod.run(x.reshape(-1), qw, qz, sc, n, k)
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import types
from pathlib import Path

from lib_kernel_ir import Term, to_json

__all__ = ["CRATE", "binary_path", "cached_module", "compile_source", "compile_term"]

CRATE = Path("/home/luis/GitHub/handl-lang/compile-triton")

#: Cargo may be configured with a shared `target-dir` (`~/.cargo/config.toml`
#: does exactly that on this machine), so the binary is not where a plain
#: `cargo build` would put it. Look in both places, and allow an override.
_CANDIDATES = (
    CRATE / "target" / "release" / "compile-triton",
    Path.home() / ".cargo" / "shared-target" / "release" / "compile-triton",
)

_CACHE: dict[tuple[str, int, int], types.ModuleType] = {}


def binary_path() -> Path:
    """Where `compile-triton` actually is, or a message that says how to fix."""
    override = os.environ.get("COMPILE_TRITON_BIN")
    if override:
        return Path(override)
    for candidate in _CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "compile-triton is not built — run `cargo build --offline "
        f"--release` in {CRATE} (looked in "
        + ", ".join(str(c) for c in _CANDIDATES)
        + ")"
    )


def compile_source(term: Term, *, block_n: int = 64, block_k: int = 128) -> str:
    """Run the Rust lowering and return the Triton source it emits."""
    binary = binary_path()
    proc = subprocess.run(
        [str(binary), "--block-n", str(block_n), "--block-k", str(block_k)],
        input=to_json(term),
        capture_output=True,
        text=True,
        cwd=str(CRATE),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "compile-triton refused the term: " + proc.stderr.strip()
        )
    return proc.stdout


def compile_term(
    term: Term, *, block_n: int = 64, block_k: int = 128
) -> types.ModuleType:
    """Lower `term` and load the emitted source as a module.

    The source goes to a real file: `@triton.jit` inspects the definition's
    source text and refuses a function that only ever existed in a string.
    """
    source = compile_source(term, block_n=block_n, block_k=block_k)
    directory = Path(tempfile.mkdtemp(prefix="compile-triton-"))
    path = directory / "generated_kernel.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(
        f"generated_{directory.name.replace('-', '_')}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the generated module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cached_module(
    term: Term, *, block_n: int = 64, block_k: int = 128
) -> types.ModuleType:
    """`compile_term`, memoised — a Triton module keeps its JIT cache."""
    key = (to_json(term), block_n, block_k)
    if key not in _CACHE:
        _CACHE[key] = compile_term(term, block_n=block_n, block_k=block_k)
    return _CACHE[key]
