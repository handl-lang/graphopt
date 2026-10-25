"""The prim-delivery probe — contract tests.

``bench/drivers/probes/prim_delivery_probe.py`` times the shipped
``prim:synop*`` folds on their bound zoo modules.  These pins cover
the probe's plumbing, not the verdict (which is a measured quantity):
the ablation ruleset really excludes the prim family, the shipped
search delivers a synop-carrying term that lowers + verifies, every
arm evaluates ``allclose`` against the spelled module, and the timing
harness reports all four arms plus the compiled twins when asked.
"""

from __future__ import annotations

import bench.drivers.probes.prim_delivery_probe as probe


def test_noprim_ruleset_drops_the_whole_prim_family():
    rs = probe._noprim_rules()
    assert rs.name.endswith("-noprim")
    assert not any(r.name.startswith("prim:synop") for r in rs.rules)
    # the non-prim bank is untouched — DEFAULT minus exactly the folds
    from mlopt_orchestrator.optimize import default_rules

    n_prim = sum(
        1
        for r in default_rules().rules
        if r.name.startswith("prim:synop")
    )
    assert n_prim == len(probe.sp.SYNOP_SHIPPED)
    assert len(rs.rules) == len(default_rules().rules) - n_prim


def test_ops_of_counts_op_nodes():
    from mlopt_core.ir import Op

    term = Op.make(
        "add", Op.make("mul", "x", "y"), Op.make("mul", "u", "v")
    )
    assert probe._ops_of(term) == {"add": 1, "mul": 2}


def test_measure_small_module_delivers_a_verified_synop_module():
    """HighwayGate @ small: fold fires, extraction carries the op."""
    row = probe.measure_module(
        "HighwayGate",
        "small",
        compile_arms=False,
        warmup=5,
        iters=10,
    )
    # the fold family fires and the extraction carries synop ops —
    # which ones is bank-version-dependent (the shipped set grows), so
    # pin the property, not the name list
    assert row.synops and all(s.startswith("synop") for s in row.synops)
    assert any(row.prim_fires.values())
    assert row.verified == {"noprim": "pass", "synop": "pass"}
    # both lowered arms routed to *some* executor (which one is
    # extraction-dependent — the fold can change carrier routing)
    assert set(row.executors) == {"spelled", "tape", "noprim", "synop"}
    assert all(row.executors.values())
    # the folded term is strictly smaller than the spelled tape term
    assert row.n_ops_synop < row.n_ops_tape
    # every arm reproduces the spelled module's output
    assert set(row.allclose) == {"spelled", "tape", "noprim", "synop"}
    assert all(row.allclose.values())
    # every arm timed
    assert set(row.times) == {"spelled", "tape", "noprim", "synop"}
    for entry in row.times.values():
        assert entry["median"] > 0
        assert entry["samples"] == 10
    # the marginal + end-to-end ratios resolve to a verdict word
    assert probe._verdict(row).split()[0] in (
        "delivers",
        "tie",
        "loses",
    )


def test_report_and_json_render(tmp_path):
    row = probe.measure_module(
        "GRUFusedGate",
        "small",
        compile_arms=False,
        warmup=5,
        iters=10,
    )
    text = probe._report([row])
    assert "GRUFusedGate" in text and "synop" in text
    out = tmp_path / "prim.json"
    probe._dump_json(str(out), [row])
    import json

    payload = json.loads(out.read_text())
    assert payload[0]["module"] == "GRUFusedGate"
    assert payload[0]["synops"] == row.synops
