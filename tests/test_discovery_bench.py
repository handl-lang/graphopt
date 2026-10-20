"""The discovery benchmark — corpus loading, runner, scorecard.

``bench/discovery/`` is the named-optimization suite (plan 0031
workstream 1): each TOML case spells a "before" program, names its
literature citation, and registers the baselines the certified form
is priced against.  These tests pin the corpus contract (sources and
refs resolve, expect-spec parses), the fast lane's per-case verdicts
on the cheapest cases, and the scorecard emission.  All torch-eager —
no external engine — so no ``e2e`` mark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch  # noqa: F401 — importorskip equivalent: torch ships

from bench.discovery.cases import (
    DiscoveryCase,
    load_cases,
)
from bench.discovery.fixtures import SOURCES
from bench.discovery.refs import REFS
from bench.discovery.runner import run_case, run_suite
from bench.discovery.scorecard import render_markdown, write_outputs


def _case(name: str) -> DiscoveryCase:
    return next(c for c in load_cases() if c.name == name)


# ---------------------------------------------------------------------------
#  corpus contract
# ---------------------------------------------------------------------------


def test_corpus_loads_and_validates():
    cases = load_cases()
    assert len(cases) >= 19
    names = {c.name for c in cases}
    # the plan-0031 seed set
    assert {
        "conv_bn_fold",
        "retnet_chunkwise",
        "flash_attention",
        "blelloch_scan",
        "mul_unsq_nat",
        "weight_chain_fold",
        "swiglu_mlp",
    } <= names
    # the compiler-transform expansion (kernel folds, scale
    # commutation, weight-space merges, the pairing pass)
    assert {
        "rms_norm_fold",
        "rms_norm_nogain",
        "rms_norm_bridge",
        "softmax_fold",
        "silu_fold",
        "softsign_fold",
        "glu_fold",
        "sdpa_nomask",
        "mha_qkv_fuse",
        "weight_merge",
        "biased_chain_fold",
        "select_mul",
    } <= names
    for c in cases:
        assert c.source in SOURCES, c.name
        assert c.expect.signals, c.name  # every case names a signal
        assert c.baselines, c.name
        for b in c.baselines:
            if b.kind == "ref":
                assert b.ref in REFS, (c.name, b.ref)


def test_corpus_rejects_bad_toml(tmp_path: Path):
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n')
    with pytest.raises(ValueError, match="title"):
        load_cases(tmp_path)
    bad.write_text(
        'name = "x"\ntitle = "t"\ncategory = "bogus"\n'
        'citation = "c"\nsource = "weight_chain"\n'
    )
    with pytest.raises(ValueError, match="category"):
        load_cases(tmp_path)


def test_unknown_source_reports(tmp_path: Path):
    """A case naming an unregistered source fails loudly, not silently."""
    case = DiscoveryCase(
        name="ghost",
        title="t",
        category="speed",
        citation="c",
        source="does_not_exist",
    )
    res = run_case(case, lane="fast")
    assert res.error and "source" in res.error
    assert not res.found


# ---------------------------------------------------------------------------
#  fast lane — the cheap, honest rows
# ---------------------------------------------------------------------------


def test_conv_bn_fold_found_certified_ties_floor():
    """The minted composite fires on the spelled eval site; the
    extracted conv-only form is certified and *ties* the hand-folded
    floor — a rediscovery, not a find."""
    res = run_case(_case("conv_bn_fold"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:conv_bn_fold" in res.signals_seen
    assert "batch_norm" not in res.detail["extracted_ops"]
    assert res.cost_after < res.cost_before
    ref = next(a for a in res.baselines if a.name == "hand-folded conv")
    assert ref.cost is not None
    assert res.cost_after == pytest.approx(ref.cost, rel=1e-6)
    assert res.beat_best is False  # ties the floor — honest "no"


def test_mul_unsq_nat_fuses_the_views():
    res = run_case(_case("mul_unsq_nat"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    # two spelled unsqueezes collapse to one
    assert res.cost_after < res.cost_before


def test_weight_chain_fold_beats_eager_not_floor():
    res = run_case(_case("weight_chain_fold"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "matmul" in res.detail["extracted_ops"]
    eager = next(a for a in res.baselines if a.kind == "self")
    fused = next(a for a in res.baselines if a.kind == "ref")
    assert res.cost_after < eager.cost
    assert res.cost_after == pytest.approx(fused.cost, rel=1e-6)
    assert res.beat_best is False


def test_swiglu_pairing_is_a_find():
    """The pairing pass prices below even the hand-fused SwiGLU —
    strictly below the registered floor reads as beat."""
    res = run_case(_case("swiglu_mlp"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert res.signals_seen  # stat:paired_extract or law:swiglu_fuse
    assert res.beat_best is True


def _ref(res: object, kind: str = "ref"):
    return next(a for a in res.baselines if a.kind == kind)


def test_rms_norm_fold_reaches_the_kernel():
    """The gained manual RMSNorm folds to ``rms_norm`` — the
    reduction/gain chain gone — and ties the shipped ``F.rms_norm``
    floor (a rediscovery, not a find)."""
    res = run_case(_case("rms_norm_fold"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:rms_norm_fold" in res.signals_seen
    assert res.detail["extracted_ops"] == ["rms_norm"]
    assert res.cost_after < res.cost_before
    assert res.cost_after == pytest.approx(_ref(res).cost, rel=1e-6)
    assert res.beat_best is False


def test_rms_norm_nogain_fold():
    res = run_case(_case("rms_norm_nogain"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:rms_norm_fold_nogain" in res.signals_seen
    assert res.detail["extracted_ops"] == ["rms_norm"]
    assert res.beat_best is False


def test_rms_norm_misspellings_fold_via_bridges():
    """``x·x`` + ``pow(-0.5)`` — the recorded-miss spelling — folds
    only because the canonicalizer bridges mint the fold's redex
    first.  The bridge fires are part of the derivation."""
    res = run_case(_case("rms_norm_bridge"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:rms_norm_fold" in res.signals_seen
    assert res.detail["extracted_ops"] == ["rms_norm"]


def test_softmax_fold_reaches_the_kernel():
    res = run_case(_case("softmax_fold"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:softmax_fold" in res.signals_seen
    assert res.detail["extracted_ops"] == ["softmax"]
    assert res.cost_after == pytest.approx(_ref(res).cost, rel=1e-6)
    assert res.beat_best is False


def test_silu_fold_reaches_the_kernel():
    res = run_case(_case("silu_fold"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:silu_fold" in res.signals_seen
    assert res.detail["extracted_ops"] == ["silu"]
    assert res.beat_best is False


def test_softsign_fold_beats_the_torch_spelling():
    """``F.softsign`` itself exports decomposed (div/abs/add — aten
    has no softsign primitive), so the folded ``softsign`` op is
    strictly below the floor torch's own functional API registers."""
    res = run_case(_case("softsign_fold"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:softsign_fold" in res.signals_seen
    assert res.detail["extracted_ops"] == ["softsign"]
    assert res.beat_best is True


def test_glu_fold_reaches_the_kernel():
    res = run_case(_case("glu_fold"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:glu_fold" in res.signals_seen
    assert res.detail["extracted_ops"] == ["glu"]
    assert res.beat_best is False


def test_sdpa_nomask_fold():
    """The mask-free spelled attention — MultiheadAttention's default
    path — folds to ``sdpa``; no ``softmax``/``matmul`` survives."""
    res = run_case(_case("sdpa_nomask"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "sdpa" in res.detail["extracted_ops"]
    assert "softmax" not in res.detail["extracted_ops"]
    assert res.beat_best is False


def test_mha_qkv_pairing_fuses_the_gemms():
    """Three split projections of one input extract through the
    forced shared-GEMM member — the paired term won, hence
    ``stat:paired_extract`` (``qkv_fuse`` itself is SUBSUMED and
    cannot fire; the signal names the pass that did the work)."""
    res = run_case(_case("mha_qkv_fuse"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "stat:paired_extract" in res.signals_seen
    ops = res.detail["extracted_ops"]
    assert "sdpa" in ops and "split" in ops


def test_weight_merge_factors_the_weights():
    res = run_case(_case("weight_merge"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:weight_factor_linear" in res.signals_seen
    assert "add" in res.detail["extracted_ops"]  # the (A+B) weight add
    assert res.cost_after < res.cost_before
    assert res.beat_best is False


def test_biased_chain_compose():
    res = run_case(_case("biased_chain_fold"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:assoc_linear_bias" in res.signals_seen
    assert res.cost_after < res.cost_before


def test_select_mul_folds_but_costs_more_standalone():
    """Honest row: ``select_mul`` fires and the one-select fused
    member is picked, but standalone the fold multiplies the full
    tensors then slices — strictly costlier than multiplying two
    slices.  The row records the measured regression."""
    res = run_case(_case("select_mul"), lane="fast")
    assert res.error == ""
    assert res.found and res.certified
    assert "law:select_mul" in res.signals_seen
    assert res.cost_after > res.cost_before
    assert res.beat_best is False


def test_run_suite_preserves_order():
    names = ["mul_unsq_nat", "conv_bn_fold"]
    rows = run_suite([_case(n) for n in names], lane="fast")
    assert [r.name for r in rows] == names


# ---------------------------------------------------------------------------
#  scorecard emission
# ---------------------------------------------------------------------------


def test_scorecard_markdown_and_json(tmp_path: Path):
    rows = run_suite([_case("mul_unsq_nat")], lane="fast")
    md = render_markdown(rows, lane="fast")
    assert "`mul_unsq_nat`" in md and "yes" in md
    payload = write_outputs(
        rows,
        lane="fast",
        md_path=tmp_path / "scorecard.md",
        json_path=tmp_path / "scorecard.json",
        device="cpu",
    )
    assert (tmp_path / "scorecard.md").exists()
    data = json.loads((tmp_path / "scorecard.json").read_text())
    assert data["schema"] == 1 and data["lane"] == "fast"
    assert data["env"]["git_sha"]
    assert data["cases"][0]["name"] == "mul_unsq_nat"
    assert payload["cases"] == data["cases"]
