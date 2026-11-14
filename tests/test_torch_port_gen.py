"""Drift gate for ``scripts/torch_port_gen.py``.

The generator derives the torch bridge's ATen classification tables
(``mlopt_torch.aten_classify``) from host-tensor's schema dump +
curated port list.  ``--check`` is the regression pin: it regenerates
in memory and byte-diffs against the checked-in artifact, so a schema
dump refresh or a port-list edit that nobody re-ran the generator for
fails here instead of silently reclassifying exports.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "torch_port_gen.py"
GENERATED = (
    ROOT
    / "packages"
    / "mlopt-torch"
    / "src"
    / "mlopt_torch"
    / "aten_classify.py"
)
SCHEMAS = ROOT.parent / "host-tensor" / "tools" / "aten_schemas.json"

pytestmark = pytest.mark.skipif(
    not SCHEMAS.exists(),
    reason="host-tensor schema dump not checked out alongside",
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GEN), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_check_passes_on_checked_in_artifact():
    """``--check`` is green while the generated file is in sync."""
    proc = _run("--check")
    assert proc.returncode == 0, proc.stderr
    assert "up to date" in proc.stdout


def test_check_detects_drift(tmp_path):
    """A stale artifact fails ``--check`` — the gate actually gates."""
    stale = tmp_path / "aten_classify.py"
    stale.write_text(GENERATED.read_text() + "\n# hand edit\n")
    proc = _run("--check", "--out", str(stale))
    assert proc.returncode == 1
    assert "DRIFT" in proc.stderr


def test_emit_spec_classification_pins():
    """Representative derivations: overload renames, scalar operands,
    and the config-Scalar exclusions schemas cannot express."""
    proc = _run("--emit-spec")
    assert proc.returncode == 0, proc.stderr
    spec = json.loads(proc.stdout)

    aten_to_ir = spec["aten_to_ir"]
    # Overload-qualified renames the bare-name fallback cannot express.
    assert aten_to_ir["min.other"] == "minimum"
    assert aten_to_ir["max.other"] == "maximum"
    assert aten_to_ir["pow.Tensor_Scalar"] == "pow"
    # Scalar-typed args that ARE operands.
    scalar_ops = set(spec["scalar_operand_ops"])
    assert {"pow", "where", "eq", "clamp", "leaky_relu"} <= scalar_ops
    # Scalar-typed args that are CONFIG — schema type alone would
    # misclassify them; the override layer excludes them.
    assert (
        not {"celu", "softshrink", "hardshrink", "linspace"}
        & scalar_ops
    )
    # Reduction dims are attributes, never scalar operands.
    assert "mean" not in scalar_ops and "sum" not in scalar_ops
    # Positional attr schemas name the config tails.
    attrs = spec["attr_schema"]
    assert attrs["mean"] == {"1": "dim", "2": "keepdim"}
    assert attrs["celu"] == {"1": "alpha"}
    assert attrs["logit"] == {"1": "eps"}


def test_generated_module_imports_and_merges():
    """The checked-in module is importable and the bridge's merged
    tables keep the hand overrides authoritative."""
    import mlopt_torch.aten_classify as ac
    from mlopt_torch import torch_bridge as tb

    assert ac.SCHEMAS_TORCH
    assert ac.PORTED_OPS > 0
    # Generated rename survives the merge.
    assert tb._ATEN_TO_IR["min.other"] == "minimum"
    # Hand override wins over any generated entry.
    assert tb._ATEN_TO_IR_OVERRIDES["trace"] == "diag_sum"
    assert tb._ATEN_TO_IR["trace"] == "diag_sum"
    # Scalar-operand union keeps hand-only entries (mul.Scalar etc.).
    assert "mul" in tb._SCALAR_OPERAND_OPS
    assert "pow" in tb._SCALAR_OPERAND_OPS
    # Local attr schema merges generated + hand per-op.
    assert tb._LOCAL_ATTR_SCHEMA["mean"][1] == "dim"
    assert tb._LOCAL_ATTR_SCHEMA["celu"][1] == "alpha"
