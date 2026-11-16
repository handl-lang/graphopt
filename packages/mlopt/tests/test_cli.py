"""CLI + registry tests — no torch/model deps required."""

from __future__ import annotations

import mlopt.models.catalog  # noqa: F401 — populates registry
from mlopt.cli import build_parser, main
from mlopt.models import get_adapter, list_adapters, resolve_ref


def test_catalog_lists_adapters() -> None:
    adapters = list_adapters()
    assert "whisper" in adapters
    assert "kokoro" in adapters
    assert adapters["whisper"].available
    assert adapters["kokoro"].available


def test_resolve_ref_by_name_prefix() -> None:
    assert get_adapter("whisper") is not None
    assert resolve_ref("whisper:openai/whisper-turbo") is not None


def test_resolve_ref_unknown() -> None:
    assert resolve_ref("bert-base-uncased") is None


def test_models_command(capsys) -> None:  # noqa: ANN001
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "whisper" in out and "ready" in out
    assert "kokoro" in out and "ready" in out


def test_optimize_declines_when_load_fails(capsys) -> None:  # noqa: ANN001
    # Point kokoro_src at a nonexistent tree so load() fails honestly
    # regardless of which extras/sources this venv happens to have.
    assert (
        main(
            [
                "optimize",
                "kokoro:hexgrad/Kokoro-82M",
                "--adapter-opt",
                "kokoro_src=/nonexistent/kokoro-src",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "load failed" in err


def test_optimize_unknown_ref(capsys) -> None:  # noqa: ANN001
    assert main(["optimize", "bert-base-uncased"]) == 2
    assert "no adapter" in capsys.readouterr().err


def test_export_sidecar_parser_args() -> None:
    args = build_parser().parse_args(
        ["export", "whisper:openai/whisper-tiny", "--sidecar", "--sidecar-dtype", "F32"]
    )
    assert args.sidecar is True
    assert args.sidecar_dtype == "F32"
    args2 = build_parser().parse_args(["export", "m"])
    assert args2.sidecar is False
    assert args2.sidecar_dtype == "F64"


def test_parser_requires_subcommand() -> None:
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_verify_run_binds_spec_inputs_by_name(capsys) -> None:  # noqa: ANN001
    """``_verify_run_export`` must not zip ``spec.inputs`` positionally
    onto the term's first-use Var order.

    Regression for ``arch:whisper-tiny-full``: the arch doc declares
    inputs ``[input_features, decoder_input_ids]`` while the term's
    first-use order is ``[decoder_input_ids, input_features]`` — the
    positional zip fed each Var the *other* input's probe, and
    ``input_features`` arrived rank-2 so the emitted conv1d→conv2d
    unsqueeze spelling hit ``expected rank-4 … got ranks 3 and 4``.
    Here ``spec.inputs``/``ir.inputs`` order ``[a, z]`` differs from
    the term's ``[z, a]`` first-use order on a non-commutative op, so
    only name binding produces the reference value.
    """
    import types

    import pytest
    from mlopt_core.ir import Op, TensorType, Var

    from mlopt.cli import _verify_run_export
    from mlopt.emit import find_hnl_run

    torch = pytest.importorskip("torch")
    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")

    T = TensorType
    a, z = Var("a", T((2,))), Var("z", T((2,)))
    term = Op.make("sub", z, a)  # leaf order [z, a]; value z - a
    t_a = torch.tensor([10.0, 20.0])
    t_z = torch.tensor([1.0, 2.0])
    spec = types.SimpleNamespace(
        module=lambda a_, z_: z_ - a_,  # inputs arrive in ir.inputs order
        inputs=(t_a, t_z),
        kwargs={},
        source=None,
    )
    search_result = types.SimpleNamespace(ir=types.SimpleNamespace(inputs=[a, z]), param_values={})
    rc = _verify_run_export(term, {}, spec, search_result)
    out = capsys.readouterr()
    assert rc == 0
    assert "verify-run: PASS" in out.out


def test_verify_run_tolerance_scales_with_reference_dtype() -> None:
    """``_verify_run_tolerance``: tight for f64, epsilon-scaled else."""
    import pytest

    torch = pytest.importorskip("torch")
    from mlopt.cli import _verify_run_tolerance

    atol, rtol, why = _verify_run_tolerance(torch.float64)
    assert (atol, rtol) == (1e-9, 1e-9)
    assert "f64" in why

    atol, rtol, why = _verify_run_tolerance(torch.float32)
    # 512·eps(f32) ≈ 6.1e-5 — the reassociation bound, not the old 1e-5.
    assert atol == rtol
    assert 5e-5 < atol < 1e-4
    assert "float32" in why and "eps" in why

    atol, rtol, _ = _verify_run_tolerance(torch.int64)
    assert (atol, rtol) == (0.0, 0.0)


def test_verify_run_tolerance_payload_dtype() -> None:
    """``payload_dtype="F32"`` widens the bound to the f32 scale —
    the sidecar quantizes Param weights on write, so an f64 reference
    must not demand f64 agreement from f32 payloads.  ``"F64"`` (and
    the default, i.e. no sidecar) never widens."""
    import pytest

    torch = pytest.importorskip("torch")
    from mlopt.cli import _verify_run_tolerance

    # f64 reference + F32 sidecar → the f32-scaled bound, rationale
    # names the payload.
    atol, rtol, why = _verify_run_tolerance(torch.float64, payload_dtype="F32")
    assert atol == rtol
    assert 5e-5 < atol < 1e-4
    assert "f64 reference" in why and "F32 sidecar" in why and "eps" in why

    # f64 reference + F64 sidecar (or none) → the tight bound unchanged.
    atol, rtol, why = _verify_run_tolerance(torch.float64, payload_dtype="F64")
    assert (atol, rtol) == (1e-9, 1e-9)
    assert "F32" not in why
    atol, rtol, _ = _verify_run_tolerance(torch.float64)
    assert (atol, rtol) == (1e-9, 1e-9)

    # f32 native reference already sits at the f32 bound — an F32
    # payload adds no further widening (never tightens either).
    atol, rtol, why = _verify_run_tolerance(torch.float32, payload_dtype="F32")
    assert 5e-5 < atol < 1e-4
    assert "F32 sidecar" in why

    # Non-floating reference: exact compare stays exact under an F64
    # payload, and an F32 payload honestly widens it (quantized
    # weights can move an integer-valued output off the exact value).
    atol, rtol, _ = _verify_run_tolerance(torch.int64, payload_dtype="F64")
    assert (atol, rtol) == (0.0, 0.0)
    atol, rtol, why = _verify_run_tolerance(torch.int64, payload_dtype="F32")
    assert 5e-5 < atol < 1e-4
    assert "F32 sidecar" in why


def test_verify_run_f64_reference_callable(capsys) -> None:  # noqa: ANN001
    """Callable-spec arm: the reference is upcast to f64 and the
    compare reports the tight f64 tolerance."""
    import types

    import pytest
    import torch
    from mlopt_core.ir import Op, TensorType, Var

    from mlopt.cli import _verify_run_export
    from mlopt.emit import find_hnl_run

    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")

    class _Sub(torch.nn.Module):
        def forward(self, x, y):  # noqa: ANN001, ANN202
            # spec.inputs arrive in ir.inputs order (a, z); the term's
            # name-bound eval is z - a, i.e. second minus first here.
            return y - x

    T = TensorType
    a, z = Var("a", T((2,))), Var("z", T((2,)))
    term = Op.make("sub", z, a)
    spec = types.SimpleNamespace(
        module=_Sub(),
        inputs=(torch.tensor([10.0, 20.0]), torch.tensor([1.0, 2.0])),
        kwargs={},
        source=None,
    )
    search_result = types.SimpleNamespace(ir=types.SimpleNamespace(inputs=[a, z]), param_values={})
    rc = _verify_run_export(term, {}, spec, search_result)
    out = capsys.readouterr()
    assert rc == 0
    assert "verify-run: PASS" in out.out
    assert "ref=f64" in out.out
    assert "dtype=torch.float64" in out.out
    assert "atol=1.0e-09" in out.out


def test_verify_run_native_reference_scales_tolerance(capsys) -> None:  # noqa: ANN001
    """Callable-spec arm fallback: a non-module callable cannot upcast,
    so the f32 reference compares at the epsilon-scaled bound."""
    import types

    import pytest
    import torch
    from mlopt_core.ir import Op, TensorType, Var

    from mlopt.cli import _verify_run_export
    from mlopt.emit import find_hnl_run

    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")

    T = TensorType
    a, z = Var("a", T((2,))), Var("z", T((2,)))
    term = Op.make("sub", z, a)
    spec = types.SimpleNamespace(
        # A plain function: no parameters(), no dtype cast → native arm.
        module=lambda a_, z_: z_ - a_,
        inputs=(torch.tensor([10.0, 20.0]), torch.tensor([1.0, 2.0])),
        kwargs={},
        source=None,
    )
    search_result = types.SimpleNamespace(ir=types.SimpleNamespace(inputs=[a, z]), param_values={})
    rc = _verify_run_export(term, {}, spec, search_result)
    out = capsys.readouterr()
    assert rc == 0
    assert "verify-run: PASS" in out.out
    assert "ref=native" in out.out
    assert "dtype=torch.float32" in out.out
    assert "512·eps" in out.out


def test_verify_run_prelifted_f64_reference(capsys) -> None:  # noqa: ANN001
    """Pre-lifted arm (onnx-rust/hnl/arch): the source-IR reference is
    lowered through ``ir_to_torch_module`` in f64."""
    import types

    import pytest
    import torch
    from mlopt_core.ir import IR, Op, TensorType, Var

    from mlopt.cli import _verify_run_export
    from mlopt.emit import find_hnl_run

    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")

    T = TensorType
    a, z = Var("a", T((2,))), Var("z", T((2,)))
    term = Op.make("sub", z, a)
    ir = IR(root=term, inputs=[a, z])
    spec = types.SimpleNamespace(
        module=object(),  # data-carrier payload, not callable
        inputs=(torch.tensor([10.0, 20.0]), torch.tensor([1.0, 2.0])),
        kwargs={},
        source=None,
    )
    search_result = types.SimpleNamespace(ir=ir, param_values={})
    rc = _verify_run_export(term, {}, spec, search_result)
    out = capsys.readouterr()
    assert rc == 0
    assert "verify-run: PASS" in out.out
    assert "ref=f64" in out.out
    assert "dtype=torch.float64" in out.out


def test_verify_run_f32_sidecar_payload_bound(capsys, tmp_path) -> None:  # noqa: ANN001
    """End-to-end: an ``"F32"`` sidecar quantizes the Param payload,
    so the f64 torch reference must compare at the f32-scaled bound —
    and ``"F64"`` keeps the tight bound on the same term."""
    import types

    import pytest
    import torch
    from mlopt_core.ir import Op, Param, TensorType, Var

    from mlopt.cli import _verify_run_export
    from mlopt.emit import find_hnl_run

    if find_hnl_run() is None:
        pytest.skip("hnl-run binary absent")

    T = TensorType
    n = 5000  # >= _SIDECAR_MIN_ELEMS so the Param leaf sidecars
    # f64-mantissa values: NOT f32-representable, so the F32 sidecar
    # rounds them and host-vs-reference diffs land at ~1e-8..1e-4 —
    # far over the 1e-9 tight bound, far under the f32-scaled one.
    p_vals = [0.123456789012345 + (i % 251) * 0.037 + i * 1e-7 for i in range(n)]
    p = Param("w", T((n,)))
    x = Var("x", T((n,)))
    term = Op.make("mul", p, x)
    feed = {"w": p_vals}
    x_t = torch.ones(n, dtype=torch.float64)
    p_ref = torch.tensor(p_vals, dtype=torch.float64)
    spec = types.SimpleNamespace(
        module=lambda x_: x_ * p_ref,  # native arm: f64 in, f64 out
        inputs=(x_t,),
        kwargs={},
        source=None,
    )
    st = tmp_path / "w.safetensors"

    rc = _verify_run_export(term, feed, spec, None, sidecar=st, sidecar_dtype="F32")
    out = capsys.readouterr()
    assert rc == 0
    assert "verify-run: PASS" in out.out
    assert "payload=F32" in out.out
    assert "atol=6.1e-05" in out.out
    assert "F32 sidecar" in out.out

    # Same term, F64 payload: the sidecar is lossless, so the tight
    # f64 bound still applies — the gate widens only on F32.
    rc = _verify_run_export(term, feed, spec, None, sidecar=st, sidecar_dtype="F64")
    out = capsys.readouterr()
    assert rc == 0
    assert "verify-run: PASS" in out.out
    assert "payload=F64" in out.out
    assert "atol=1.0e-09" in out.out
