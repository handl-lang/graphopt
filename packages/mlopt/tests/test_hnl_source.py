"""``.hnl`` source adapter tests — ADR-0172 (no GPU/models needed).

Covers the ``hnl`` adapter's ref routing, the ``hnl_to_term`` lift,
leaf binding (feed / seeded synthetic / honest declines), the
``HnlSource`` port feeding ``optimize_model``, and ``doctor``'s probe.
"""

from __future__ import annotations

import json

import pytest

import mlopt.models.catalog  # noqa: F401 — populates the registry
from mlopt.models import get_adapter, resolve_ref

try:
    import torch  # noqa: F401 — the pipeline's sink is torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover — torch is a hard dep
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch required")


@pytest.fixture()
def adapter():
    ad = get_adapter("hnl")
    assert ad is not None
    return ad


@pytest.fixture()
def tiny_hnl(tmp_path):
    """A ``do``-block program: one ``Param`` weight, one ``Var`` input."""
    path = tmp_path / "tiny.hnl"
    path.write_text(
        "--- tiny test program\n"
        'from "host-tensor" import tensor\n'
        "do {\n"
        "  w :: tensor.reshape(tensor.from_list([1, 2, 3, 4, 5, 6]), [2, 3])\n"
        "  tensor.matmul(x, w)\n"
        "}\n"
    )
    return path


# --- ref routing ------------------------------------------------------------


def test_resolve_hnl_prefix(adapter) -> None:
    assert resolve_ref("hnl:foo.hnl") is adapter


def test_resolve_bare_hnl_suffix(tiny_hnl, adapter) -> None:
    assert resolve_ref(str(tiny_hnl)) is adapter


def test_claims_only_hnl_refs(adapter) -> None:
    assert adapter.claims("model.hnl")
    assert adapter.claims("hnl:anything")
    assert not adapter.claims("whisper:openai/whisper-tiny")
    assert not adapter.claims("some-model")


# --- load: lift + leaf binding ----------------------------------------------


def test_load_lifts_term_and_binds_synthetic(adapter, tiny_hnl) -> None:
    spec = adapter.load(f"hnl:{tiny_hnl}", leaves='{"x": [4, 2]}')
    assert len(spec.inputs) == 1
    assert tuple(spec.inputs[0].shape) == (4, 2)
    # The marshaled do-binding minted a Param leaf; its value is the
    # seeded synthetic — the .hnl artifact carries shapes, not data.
    assert "w" in spec.source._param_values
    assert spec.meta["leaf_source"] == "synthetic"
    assert spec.meta["leaves"]["param_values"] == {"w": "synthetic"}
    ir, values = spec.source.to_ir(spec.module, spec.inputs)
    assert ir.root is not None and values["w"].shape == (2, 3)


def test_load_feed_supplies_real_values(adapter, tiny_hnl, tmp_path) -> None:
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"w": {"shape": [2, 3], "data": list(range(6))}}))
    spec = adapter.load(f"hnl:{tiny_hnl}", leaves='{"x": [4, 2]}', feed=str(feed_path))
    # w is fed; x stays synthetic — the report says "mixed" honestly.
    assert spec.meta["leaf_source"] == "mixed"
    assert spec.meta["leaves"]["param_values"] == {"w": "feed"}
    w = spec.source._param_values["w"]
    assert w.flatten().tolist() == [float(i) for i in range(6)]


def test_load_var_feed_overrides_probe_input(adapter, tmp_path) -> None:
    path = tmp_path / "add.hnl"
    path.write_text("tensor.add(x, y)\n")
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"x": [[1.0, 2.0], [3.0, 4.0]]}))
    spec = adapter.load(
        f"hnl:{path}",
        leaves='{"x": [2, 2], "y": [2, 2]}',
        feed=str(feed_path),
    )
    # x binds positionally first (first-use order); its feed value is
    # the probe input, y stays synthetic.
    assert spec.inputs[0].flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert spec.meta["leaves"]["input_values"]["x"] == "feed"
    assert spec.meta["leaves"]["input_values"]["y"] == "synthetic"


def test_synthetic_false_declines_without_feed(adapter, tmp_path) -> None:
    path = tmp_path / "add.hnl"
    path.write_text("tensor.add(x, y)\n")
    with pytest.raises(ValueError, match="unbound leaves"):
        adapter.load(
            f"hnl:{path}",
            leaves='{"x": [2, 2], "y": [2, 2]}',
            synthetic=False,
        )


def test_shapeless_var_declines_naming_leaf(adapter, tmp_path) -> None:
    path = tmp_path / "add.hnl"
    path.write_text("tensor.add(x, y)\n")
    with pytest.raises(ValueError, match="Var 'x'"):
        adapter.load(f"hnl:{path}")


# --- load: honest declines ---------------------------------------------------


def test_missing_file_declines(adapter) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        adapter.load("hnl:/nonexistent-dir/nope.hnl")


def test_unsupported_op_declines(adapter, tmp_path) -> None:
    path = tmp_path / "bad.hnl"
    path.write_text("tensor.quantize(x, 4)\n")
    from kit_discovery.hnl_surface import UnsupportedOpError

    with pytest.raises(UnsupportedOpError, match="tensor.quantize"):
        adapter.load(f"hnl:{path}")


def test_syntax_error_declines(adapter, tmp_path) -> None:
    path = tmp_path / "syn.hnl"
    path.write_text("tensor.add(x, y\n")
    from kit_discovery.hnl_import import HnlImportError

    with pytest.raises(HnlImportError):
        adapter.load(f"hnl:{path}")


# --- export-artifact unwrap ---------------------------------------------------


def test_export_signature_unwraps(adapter, tmp_path) -> None:
    """``mlopt export`` artifacts: the ``export f := (params) ~>`` line
    is interface text — the body lifts, signature params become the
    term's Var leaves (unused ones simply never appear)."""
    path = tmp_path / "exp.hnl"
    path.write_text(
        "--- mlopt export — optimized: optimized term over tensor.*\n"
        'from "host-tensor" import tensor\n'
        "export optimized := (x, pos) ~> do {\n"
        "  w :: tensor.reshape(tensor.from_list([1, 0, 0, 1]), [2, 2])\n"
        "  tensor.matmul(x, w)\n"
        "}\n"
    )
    spec = adapter.load(f"hnl:{path}", leaves='{"x": [3, 2]}')
    assert len(spec.inputs) == 1  # `pos` is declared but unused
    assert "w" in spec.source._param_values


# --- codec round-trip ---------------------------------------------------------


def test_hnl_surface_emit_round_trip(adapter, tmp_path) -> None:
    """``hnl_surface.emit`` → file → ``load`` → the same op spine."""
    from kit_discovery import hnl_surface
    from mlopt_core.ir import Op, TensorType, Var

    x = Var("x", TensorType((2,)))
    y = Var("y", TensorType((2,)))
    z = Var("z", TensorType((2,)))
    term = Op.make("add", x, Op.make("mul", y, z))
    src = hnl_surface.emit(term)
    path = tmp_path / "rt.hnl"
    path.write_text(src + "\n")
    spec = adapter.load(f"hnl:{path}", leaves='{"x": [2], "y": [2], "z": [2]}')
    root = spec.source._ir.root
    assert root.op == "add"
    assert root.args[1].op == "mul"


# --- the pipeline -------------------------------------------------------------


def test_optimize_model_runs_verified_pipeline(adapter, tiny_hnl) -> None:
    from mlopt.optimize import optimize_model

    spec = adapter.load(f"hnl:{tiny_hnl}", leaves='{"x": [4, 2]}')
    lowered, report = optimize_model(spec, max_iterations=1, verify=True)
    assert lowered is not None
    assert report.error is None
    assert report.verified is True
    assert report.search_result is not None
    assert report.search_result.term is not None


def test_export_round_trip_through_mlopt_emit(adapter, tiny_hnl, tmp_path) -> None:
    """optimize → ``emit_function`` → the artifact re-imports."""
    from mlopt.emit import emit_function
    from mlopt.optimize import optimize_model

    spec = adapter.load(f"hnl:{tiny_hnl}", leaves='{"x": [4, 2]}')
    lowered, report = optimize_model(spec, max_iterations=1, verify=True)
    assert lowered is not None and report.verified is True
    term = report.search_result.term
    feed = dict(report.search_result.param_values or {})
    artifact = emit_function(term, feed, name="optimized")
    out = tmp_path / "optimized.hnl"
    out.write_text(artifact + "\n")
    spec2 = adapter.load(f"hnl:{out}", leaves='{"x": [4, 2]}')
    assert spec2.source._ir.root is not None


# --- doctor --------------------------------------------------------------------


def test_doctor_probe_reports_census(adapter, tiny_hnl, capsys) -> None:
    from mlopt.doctor import diagnose

    out = diagnose(f"hnl:{tiny_hnl}")
    assert "adapter:   hnl" in out
    assert "probe.parse: ok" in out
    assert "probe.vars: ['x']" in out
    assert "probe.params: ['w']" in out
    assert "feed=<json>" in out


def test_doctor_probe_reports_parse_decline(adapter, tmp_path) -> None:
    from mlopt.doctor import diagnose

    path = tmp_path / "bad.hnl"
    path.write_text("tensor.quantize(x, 4)\n")
    out = diagnose(f"hnl:{path}")
    assert "probe.parse: declined" in out


def test_cli_optimize_hnl_end_to_end(tiny_hnl, capsys) -> None:  # noqa: ANN001
    """``mlopt optimize hnl:<path>`` — the ADR-0172 headline path."""
    from mlopt.cli import main

    assert (
        main(
            [
                "optimize",
                f"hnl:{tiny_hnl}",
                "--iters",
                "1",
                "--adapter-opt",
                'leaves={"x": [4, 2]}',
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "verify=True" in out
    assert "leaf values: synthetic" in out


def test_cli_doctor_hnl(tmp_path, capsys) -> None:  # noqa: ANN001
    from mlopt.cli import main

    path = tmp_path / "tiny.hnl"
    path.write_text("tensor.add(x, y)\n")
    assert main(["doctor", f"hnl:{path}"]) == 0
    out = capsys.readouterr().out
    assert "adapter:   hnl" in out


# --- device placement + delivery metadata -----------------------------------


def test_device_opt_moves_inputs_and_params(adapter, tiny_hnl) -> None:
    """``device=cpu`` is explicit probe placement; ``device=cuda``
    follows torch availability — both move inputs AND bound leaf
    values together so they never disagree."""
    spec = adapter.load(
        f"hnl:{tiny_hnl}",
        leaves='{"x": [4, 2]}',
        device="cpu",
    )
    import torch

    assert spec.inputs[0].device == torch.device("cpu")
    assert spec.source._param_values["w"].device == torch.device("cpu")
    assert spec.meta["device"] == "cpu"


def test_device_default_matches_torch_probe(adapter, tiny_hnl) -> None:
    """No ``device`` opt: cuda when available, else cpu — the same
    convention whisper/kokoro use.  This is probe placement for the
    torch lowering only; ``hnl-run`` stays CPU-only."""
    import torch

    spec = adapter.load(f"hnl:{tiny_hnl}", leaves='{"x": [4, 2]}')
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert spec.meta["device"] == expected
    assert spec.inputs[0].device.type == expected


def test_unknown_device_declines(adapter, tiny_hnl) -> None:
    with pytest.raises(ValueError, match="unknown device"):
        adapter.load(f"hnl:{tiny_hnl}", leaves='{"x": [4, 2]}', device="tpu")


def test_meta_declares_alias_inputs(adapter, tiny_hnl) -> None:
    """``tensor.*`` ops are functional — positional inputs are
    caller-owned read-only buffers, so alias-aware CUDA capture may
    bind them by reference (fresh tensors per call fall back to the
    copy path).  This is about the torch lowering delivery path,
    NOT ``hnl-run``."""
    spec = adapter.load(f"hnl:{tiny_hnl}", leaves='{"x": [4, 2]}')
    assert spec.meta["alias_inputs"] is True
    # No in-place schemas in the codec surface → nothing is mutable.
    assert "mutable_inputs" not in spec.meta
