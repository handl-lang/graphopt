"""Smoke test for the runnable example.

``examples/quickstart.py`` is the repo's one-command showcase — the
first thing a reader runs.  Nothing verified it before, so it could rot
silently.  This runs it as a **subprocess** with ``--quick`` (~15 s, no
coverage tracing) and asserts the pipeline completed and reported its
race, rather than importing it: an example is executed, not API.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "quickstart.py"


def test_quickstart_file_exists():
    assert EXAMPLE.is_file(), "the README's quickstart command must resolve"


def test_quickstart_runs_end_to_end():
    """The example exits 0 and prints the measured race."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE), "--quick"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = proc.stdout
    # the five acts, in order — a cheap structural check of the output
    assert "THE RACE" in out
    assert "faster than eager" in out
    assert "mlopt" in out
