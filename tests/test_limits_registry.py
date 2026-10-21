"""``bench/limits`` — the baselines registry and the limits table.

Plan 0031 workstream 3 (baselines, same shape): every opponent is a
folder under ``bench/limits/`` carrying a measured record, a
``BaselineSpec``, and a ``build`` adapter; the registry enumerates
them lazily, the runner crosses them with the discovery corpus, and
``docs/limits_table.{md,json}`` is generated — the "results —
limits" table is no longer hand-copied.  These tests pin the
registry contract (roster, case coverage, availability reporting,
skip-with-reason), a seconds-scale fast-lane run on the cheapest
cases, and the table emission.  All torch-eager — no external
engine — so no ``e2e`` mark.
"""

# ruff: noqa: RUF002 — math notation in prose.

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch  # noqa: F401 — importorskip equivalent: torch ships

from bench.discovery.cases import load_cases
from bench.limits import registry
from bench.limits.registry import (
    BaselineCtx,
    BaselineSpec,
    SkipBaseline,
)
from bench.limits.runner import run_case, run_suite
from bench.limits.table import render_markdown, write_outputs


def _case(name: str):
    return next(c for c in load_cases() if c.name == name)


# ---------------------------------------------------------------------------
#  registry contract
# ---------------------------------------------------------------------------


def test_registry_roster():
    """The shipped roster — every spec carries capability data."""
    names = registry.names()
    assert {
        "eager",
        "torch_compile",
        "opt_einsum",
        "fla",
        "paper_impl",
        "hand_carrier",
    } <= set(names)
    for name in names:
        s = registry.spec(name)
        assert isinstance(s, BaselineSpec)
        assert s.kind in {
            "eager",
            "compile",
            "order",
            "kernel",
            "impl",
            "carrier",
        }
        assert s.record, name  # the measured record is not empty


def test_registry_for_case_filters_honestly():
    """Case coverage filters the roster per case."""
    retnet = {s.name for s in registry.for_case("retnet_chunkwise")}
    assert {"eager", "paper_impl", "hand_carrier", "fla"} <= retnet
    weight = {s.name for s in registry.for_case("weight_chain_fold")}
    assert "opt_einsum" in weight
    # opt_einsum only prices the contraction case
    assert "opt_einsum" not in {
        s.name for s in registry.for_case("conv_bn_fold")
    }
    every = {s.name for s in registry.for_case("mul_unsq_nat")}
    assert "eager" in every and "paper_impl" in every


def test_registry_unknown_name():
    """Unknown names raise with the roster in the message."""
    with pytest.raises(ValueError, match="unknown baseline"):
        registry.spec("hal")
    case = _case("mul_unsq_nat")
    model, feed = __import__(
        "bench.discovery.runner", fromlist=["x"]
    )._build_fixture(case)
    ctx = BaselineCtx(
        case=case,
        model=model,
        feed=feed,
        sr=None,
        sink=None,
        source=None,
    )
    with pytest.raises(ValueError, match="unknown baseline"):
        registry.build("hal", ctx)


def test_availability_reports_reasons():
    """``fla`` reports its blocker; the installed roster is clear."""
    reason = registry.availability("fla", device="cpu")
    assert reason is not None
    assert "fla" in reason or "CUDA" in reason
    # the always-on baselines never need an excuse
    assert registry.availability("eager") is None
    assert registry.availability("paper_impl") is None
    # opt_einsum lives in the opt-in ``einsum`` dep group: installed it
    # needs no excuse; absent it reports its reason, same convention as
    # ``fla`` (``uv sync --group einsum`` to seat it).
    oe_reason = registry.availability("opt_einsum")
    if importlib.util.find_spec("opt_einsum") is None:
        assert oe_reason is not None and "opt_einsum" in oe_reason
    else:
        assert oe_reason is None


def test_build_declines_a_case_it_does_not_price():
    """``opt_einsum`` on conv_bn_fold skips with a reason."""
    case = _case("conv_bn_fold")
    ctx = BaselineCtx(
        case=case,
        model=None,
        feed=None,
        sr=None,
        sink=None,
        source=None,
    )
    with pytest.raises(SkipBaseline, match="does not price"):
        registry.build("opt_einsum", ctx)


# ---------------------------------------------------------------------------
#  fast lane — the cheap, honest rows
# ---------------------------------------------------------------------------


def test_fast_lane_mul_unsq_rows():
    """One case × every applicable baseline — ratios + caveats."""
    res = run_case(_case("mul_unsq_nat"), lane="fast")
    assert res.error == ""
    by_name = {r.baseline: r for r in res.rows}
    eager = by_name["eager"]
    assert eager.available and eager.ratio is not None
    # the fused form is strictly cheaper — eager reads as a WIN
    assert eager.ratio > 1.0
    impl = by_name["paper_impl"]
    assert impl.ratio == pytest.approx(1.0, rel=1e-6)  # the floor
    comp = by_name["torch_compile"]
    assert comp.available and comp.baseline_cost is None
    assert "symbolic" in comp.caveat or "measured" in comp.caveat
    # hand_carrier declines: no carrier executor claims a mul-unsq term
    hc = by_name["hand_carrier"]
    assert not hc.available and "no carrier" in hc.caveat


def test_fast_lane_retnet_carrier_skip_is_honest():
    """``hand_carrier`` declines retnet_chunkwise — a real routing truth.

    ``carrier_search`` extracts the bare ``om_decayed`` root, which no
    executor spec accepts (only the K-way chunked ``concat`` form
    routes to ``om_decayed_chunked``, and producing it needs the
    opt-in ``om_kway_laws``/``OM_CHUNKIFY`` set).  The delivered arm
    is already the generic lowering, so the row would duplicate mlopt.
    """
    res = run_case(
        _case("retnet_chunkwise"),
        lane="fast",
        baselines=["hand_carrier"],
    )
    assert res.error == ""
    (row,) = res.rows
    assert not row.available and "no carrier" in row.caveat


def test_fast_lane_preserves_case_order_and_covers():
    cases = [_case(n) for n in ("mul_unsq_nat", "conv_bn_fold")]
    out = run_suite(cases, lane="fast")
    assert [cl.case.name for cl in out] == [c.name for c in cases]
    for cl in out:
        assert cl.cost_after is not None
        assert cl.rows  # every case priced ≥1 baseline


def test_opt_einsum_arm_computes_the_chain():
    """The oe-greedy arm runs the chain correctly (fp64)."""
    # opt-in ``einsum`` dependency group (uv sync --group einsum)
    pytest.importorskip("opt_einsum")
    case = _case("weight_chain_fold")
    res = run_case(case, lane="fast", baselines=["opt_einsum", "eager"])
    by_name = {r.baseline: r for r in res.rows}
    assert by_name["opt_einsum"].available
    # smoke the arm numerically against the spelled chain
    from bench.limits.registry import build

    model, feed = __import__(
        "bench.discovery.runner", fromlist=["x"]
    )._build_fixture(case)
    arm = build(
        "opt_einsum",
        BaselineCtx(
            case=case,
            model=model,
            feed=feed,
            sr=None,
            sink=None,
            source=None,
        ),
    )
    import torch as _t

    with _t.no_grad():
        got = arm(feed)
        want = model(feed)
    assert _t.allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
#  table emission
# ---------------------------------------------------------------------------


def test_limits_markdown_and_json(tmp_path: Path):
    out = run_suite([_case("mul_unsq_nat")], lane="fast")
    md = render_markdown(out, lane="fast")
    assert "# Limits table" in md and "`mul_unsq_nat`" in md
    assert "eager" in md and "ratio" in md and "caveat" in md
    payload = write_outputs(
        out,
        lane="fast",
        md_path=tmp_path / "limits_table.md",
        json_path=tmp_path / "limits_table.json",
        device="cpu",
    )
    assert (tmp_path / "limits_table.md").exists()
    data = json.loads((tmp_path / "limits_table.json").read_text())
    assert data["schema"] == 1 and data["lane"] == "fast"
    assert data["env"]["git_sha"]
    assert data["cases"][0]["name"] == "mul_unsq_nat"
    assert payload["cases"] == data["cases"]
    rows = data["cases"][0]["baselines"]
    assert any(r["baseline"] == "eager" for r in rows)


def test_cli_smoke(tmp_path):
    """``python -m bench.limits.run --fast`` on the cheapest case."""
    from bench.limits.run import main

    rc = main(
        [
            "--fast",
            "--cases",
            "mul_unsq_nat",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads((tmp_path / "limits_table.json").read_text())
    assert payload["cases"][0]["name"] == "mul_unsq_nat"
    names = {r["baseline"] for r in payload["cases"][0]["baselines"]}
    assert "eager" in names and "paper_impl" in names


def test_cli_unknown_baseline(tmp_path):
    """An unknown baseline name exits 2 with the roster."""
    from bench.limits.run import main

    rc = main(["--baselines", "nope", "--out", str(tmp_path)])
    assert rc == 2


def test_cli_list_baselines(capsys):
    from bench.limits.run import main

    assert main(["--list-baselines"]) == 0
    out = capsys.readouterr().out
    assert "eager" in out and "fla" in out
