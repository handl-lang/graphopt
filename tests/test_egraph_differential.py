"""Differential saturation: mlopt's ``EGraph`` vs ``handl-egraph``.

The run-level companion to ``test_handl_peer.py`` — that test diffs
single ``apply_rule`` firings through the peer; these tests drive
**whole saturation runs**: ``EGraph.run`` + ``extract_best`` against
the Rust port's ``EGraph::run``/``extract_best``, over the same
fixtures, through the compiled ``egraph-run`` subprocess door
(``packages/mlopt-kit/src/bin/egraph-run.rs``).  The fixture set and
the comparison live in ``lab/audit/egraph_differential.py`` — this file
is the suite-facing wrapper.

What is asserted: identical e-graph structure (Weisfeiler-Lehman class
signatures over canonical member encodings), identical
stats/provenance, and the same extracted winner — an extracted-*cost*
tie with both winners present in the other's root class is reported as
``tie`` (the documented member-order caveat), never silently.

Marked ``e2e``: the suite crosses an external engine boundary (the
compiled Rust e-graph, via subprocess).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from lab.audit import egraph_differential as ed

pytestmark = pytest.mark.e2e

_REPO = Path(__file__).resolve().parents[1]
_CANDIDATES = [
    _REPO / "target" / "debug" / "egraph-run",
    Path.home() / ".cargo" / "shared-target" / "debug" / "egraph-run",
]


def _find_bin() -> Path | None:
    env = os.environ.get("CATOPT_EGRAPH_RUN")
    if env:
        p = Path(env)
        return p if p.exists() else None
    for cand in _CANDIDATES:
        if cand.exists():
            return cand
    return None


def _build_bin() -> Path:
    """The compiled ``egraph-run``, building it if absent."""
    found = _find_bin()
    if found is not None:
        return found
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo not on PATH — egraph-run unavailable")
    env = dict(os.environ, CARGO_PROFILE_DEV_DEBUG="0")
    out = subprocess.run(
        [
            cargo,
            "build",
            "-p",
            "mlopt-kit",
            "--bin",
            "egraph-run",
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    found = _find_bin()
    if out.returncode != 0 or found is None:
        pytest.skip(f"egraph-run build failed: {out.stderr[-400:]}")
    return found


@pytest.fixture(scope="module")
def egraph_bin() -> Path:
    """The ``egraph-run`` binary path."""
    return _build_bin()


def _fixtures() -> list[dict]:
    return ed.fixtures()


@pytest.mark.parametrize(
    "fx", _fixtures(), ids=[f["name"] for f in _fixtures()]
)
def test_fixture_parity(fx: dict, egraph_bin: Path) -> None:
    """Same terms, same rules, same extraction — identical reports."""
    verdict, _py, _rs, detail = ed.compare(fx, egraph_bin)
    assert verdict in ("agree", "tie"), (
        f"{fx['name']}: {verdict} — {detail}"
    )
