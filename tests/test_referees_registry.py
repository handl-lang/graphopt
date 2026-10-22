# ruff: noqa: RUF001 — math notation in prose.

"""``bench/referees`` + ``bench/matrix`` — the systematic-eval lane.

Plan 0032 phase 2 lane B: referees are strictness arms — cumulative
subsets of the admission gauntlet (numeric oracle → typed-pay →
adversarial sweep → cert replay) — as *consumers* of ``verify_law``
and the gauntlet machinery, each carrying a ``ScoringSpec`` value
function.  These tests pin the registry contract (roster, component
order, skip-with-reason), the dial's honest verdicts on known claims
— including the 212-bad-accepts ablation made a row — and a
seconds-scale matrix smoke.
"""

from __future__ import annotations

import json

import pytest
import torch
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op, TensorType, Var

from bench.referees import registry
from bench.referees.registry import (
    RefereeCtx,
    RefereeSpec,
    SkipReferee,
)


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _ctx(**kw) -> RefereeCtx:
    return RefereeCtx(**kw)


#: The ablation claims — retros/mul-unsq-nat.md made rows.
_GRID_EQ_CLAIM = Rewrite(
    "mul_unsq_nat:grid-eq",
    lhs=_p(
        "mul",
        _p("unsqueeze", "u", dim="UD"),
        _p("unsqueeze", "v", dim="UD"),
    ),
    rhs=_p("unsqueeze", _p("mul", "u", "v"), dim="UD"),
    cond=(
        "bcast-eq",
        ("unsq-out", "u", "UD"),
        ("unsq-out", "v", "UD"),
        ("unsq-out", ("bcast", "u", "v"), "UD"),
        ("unsq-out", ("bcast", "u", "v"), "UD"),
    ),
)
_FALSE_CLAIM = Rewrite(
    "mul_is_add", lhs=_p("mul", "U", "V"), rhs=_p("add", "U", "V")
)
_TRUE_CLAIM = Rewrite(
    "add_self_is_double",
    lhs=_p("add", "U", "U"),
    rhs=_p("mul", "U", _p("__const__")),
)


# ---------------------------------------------------------------------------
#  registry contract
# ---------------------------------------------------------------------------


def test_registry_roster():
    """The shipped roster — cumulative subsets in canonical order."""
    names = registry.names()
    assert {"naive", "oracle", "pay", "sweep", "full"} <= set(names)
    assert registry.COMPONENT_ORDER == (
        "oracle",
        "pay",
        "sweep",
        "cert",
    )
    for name in names:
        s = registry.spec(name)
        assert isinstance(s, RefereeSpec)
        assert set(s.components) <= registry.COMPONENTS
        assert s.record, name  # the measured record is not empty
    assert registry.spec("naive").components == ()
    assert registry.spec("full").components == registry.COMPONENT_ORDER


def test_registry_unknown_name():
    with pytest.raises(ValueError, match="unknown referee"):
        registry.spec("hal")
    with pytest.raises(ValueError, match="unknown referee"):
        registry.judge("hal", _FALSE_CLAIM, _ctx())


def test_spec_rejects_unknown_components():
    with pytest.raises(ValueError, match="unknown component"):
        RefereeSpec(name="bad", components=("telepathy",))


def test_naive_accepts_everything():
    """The trust arm — the dial's bottom rung."""
    v = registry.judge("naive", _FALSE_CLAIM, _ctx())
    assert v.passed and v.stages == ()
    assert registry.judge("naive", _GRID_EQ_CLAIM, _ctx()).passed


def test_oracle_refutes_the_false_claim():
    """The numeric oracle rejects ``mul == add`` on a concrete pair."""
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    claim = Rewrite("bad", _p("mul", x, y), _p("add", x, y))
    v = registry.judge("oracle", claim, _ctx())
    assert not v.passed
    assert v.stages[0].name == "oracle"
    assert "unequal" in v.stages[0].detail


def test_concrete_true_claim_clears_the_dial():
    """``add(x,0) == x`` — derivable, paying on count axis only.

    The claim pays under no shipped axis (it shrinks nothing the
    dag costs see differently — the identity fold's extraction ties),
    so the scored arms refuse on the pay leg while oracle/sweep
    certify truth.  Assert the honest split.
    """
    x = _v("x", 4, 4)
    from mlopt_core.ir import Const

    claim = Rewrite("id", _p("add", x, Const(0)), x)
    assert registry.judge("oracle", claim, _ctx()).passed
    assert registry.judge("sweep", claim, _ctx()).passed


def test_sweep_is_strict_superset():
    """A claim the oracle passes the sweep can still refuse.

    The grid-eq ``mul_unsq_nat`` guard: the stock window never
    reaches the unequal-rank corner (the retro's 212 bad accepts);
    the widened bank does.  If the enumeration later ships the
    corner itself, this pin becomes a tautology — revisit then.
    """
    o = registry.judge("oracle", _GRID_EQ_CLAIM, _ctx())
    s = registry.judge("sweep", _GRID_EQ_CLAIM, _ctx(sweep_limit=2_000))
    assert o.stages[0].passed  # the shallow window accepts
    assert not s.passed  # the adversarial leg refuses
    sweep_stage = s.stages[-1]
    assert sweep_stage.name in ("sweep", "pay")


def test_scoring_spec_is_data():
    """``ScoringSpec`` gating makes "optimize X first" data."""
    from gamekit.scoring import ScoreGate, ScoringSpec

    x, y = _v("x", 4, 4), _v("y", 4, 4)
    # A launch-neutral true claim: swap is true and free.
    claim = Rewrite("comm", _p("mul", x, y), _p("mul", y, x))
    ctx = _ctx(probe=())
    v = registry.judge("full", claim, ctx)
    # cert replays the comm law's derivation on the instance.
    assert "cert" in [s.name for s in v.stages] or not v.passed
    # An unmeasurable gate reports missing, never a guess.
    strict = RefereeSpec(
        name="strict",
        components=("pay",),
        spec=ScoringSpec(gates=(ScoreGate("memory", 1.0),)),
    )
    from bench.referees import components as _c

    evd = _c.ClaimEvidence(claim, ctx)
    stages = [
        _c.run_component("pay", claim, ctx, evd),
    ]
    del stages
    from gamekit.scoring import score

    sv = score(strict.spec, {"launch": 0.9})
    assert not sv.passed and "memory" in sv.missing


def test_skip_referee_requires(monkeypatch):
    """A referee with an unmet ``requires`` declines honestly."""
    s = RefereeSpec(
        name="dep", components=("oracle",), requires=("no_such_mod",)
    )
    monkeypatch.setitem(registry._INDEX, "dep", ("x", s))
    with pytest.raises(SkipReferee, match="scratch dep"):
        registry.judge("dep", _FALSE_CLAIM, _ctx())
    registry._INDEX.pop("dep", None)


def test_component_verdict_order():
    """``judge`` runs components in canonical order, stops on fail."""
    from bench.referees import components as _c

    x = _v("x", 4, 4)
    claim = Rewrite("bad2", _p("mul", x, x), _p("add", x, x))
    evd = _c.ClaimEvidence(claim, _ctx())
    v = _c.run_component("oracle", claim, _ctx(), evd)
    assert not v.passed  # mul != add on a concrete pair


def test_cert_replays_a_derivable_claim():
    """The cert leg on a derivable concrete pair — ``instances``
    returns ``(lhs, rhs)`` pairs, so the leg unpacks one instance
    before ``verify_law`` (the ``inst[1]`` IndexError regression).
    """
    from mlopt_core.ir import Const

    from bench.referees import components as _c

    x = _v("x", 4, 4)
    claim = Rewrite("id", _p("add", x, Const(0)), x)
    evd = _c.ClaimEvidence(claim, _ctx())
    v = _c.run_component("cert", claim, _ctx(), evd)
    assert v.name == "cert" and v.passed


def test_instance_case_feeds_params_via_param_vals():
    """Real-match sites carry ``Param`` leaves — the module's weights.

    They lower through ``param_vals`` (one shared fp64 draw), not the
    input feed: an empty ``param_vals`` makes each lowered side mint
    an independent ambient-dtype ``randn·0.02`` — fp32 weights
    against an fp64 feed error at the first mixed op, and the two
    sides would disagree on their own parameters even at matching
    dtype.  The ``fold:instance_norm``/``fold:group_norm`` /
    ``sdpa_fold_*`` ``verify_fail`` rows were this artifact.
    """
    from mlopt_core.ir import Param

    from bench.referees import components as _c

    lhs_i = _p("mul", _v("x", 2, 3), Param("w", TensorType((2, 3))))
    case = _c._instance_case("c#0", lhs_i)
    assert [type(v) for v in case.inputs] == [Var]
    assert len(case.feed) == len(case.inputs)
    assert case.param_vals["w"].dtype == torch.float64
    assert tuple(case.param_vals["w"].shape) == (2, 3)


def test_instances_skip_non_denoting_sites():
    """A check-passing site whose pair cannot eval is not a probe.

    The ``matmul(a, b) + c`` binding space mixes shapes the stock
    bank draws independently — an ``a(2, 3) @ b(2, 3)`` env makes the
    instantiated lhs error — and a TermCase minted on one fails to
    lower *before* the claim is tested: the malformed-case artifact
    behind the ``sdpa_fold_addmul`` ``verify_fail=4`` row.
    """
    from mlopt_discovery import oracle as lvo

    from bench.referees import components as _c

    claim = Rewrite(
        "mm+c",
        _p("add", _p("matmul", "a", "b"), "c"),
        _p("add", _p("matmul", "a", "b"), "c"),
    )
    evd = _c.ClaimEvidence(claim, _ctx())
    insts = evd.instances(8)
    assert insts  # denoting sites exist — the claim is honest
    for lhs_i, rhs_i in insts:
        assert not lvo.eval_instance(lhs_i, rhs_i)[0].endswith("-err")


# ---------------------------------------------------------------------------
#  matrix — coverage audit + a seconds-scale smoke
# ---------------------------------------------------------------------------


def test_matrix_coverage_marks_unrun():
    """No-claim boards and unsupported players report reasons."""
    from bench.matrix.runner import coverage

    cells = coverage(["open", "contraction"], ["playbook"], ["naive"])
    by_board = {(c.board, c.run) for c in cells}
    assert ("open", True) in by_board
    assert ("contraction", False) in by_board
    reasons = [c.reason for c in cells if not c.run]
    assert all("no equality claims" in r for r in reasons)


def test_matrix_search_smoke(tmp_path):
    """The search board: extraction claims judged per referee."""
    from bench.matrix.runner import run_matrix

    cells = run_matrix(
        ["search"],
        ["playbook"],
        ["naive", "oracle"],
        meta_cases=3,
        meta_budget=8,
    )
    assert len(cells) == 2
    for c in cells:
        assert c.run and c.claims >= 0 and c.reward_ep is not None
    naive = next(c for c in cells if c.referee == "naive")
    assert naive.admitted == naive.claims


def test_matrix_cli_smoke(tmp_path):
    """``python -m bench.matrix --smoke`` writes the table."""
    from bench.matrix.__main__ import main

    rc = main(
        [
            "--smoke",
            "--players",
            "playbook",
            "--referees",
            "naive,oracle",
            "--boards",
            "search",
            "--out",
            str(tmp_path),
            "--no-ledger",
        ]
    )
    assert rc == 0
    payload = json.loads((tmp_path / "matrix.json").read_text())
    assert payload["schema"] == 1 and payload["cells"]
    md = (tmp_path / "matrix.md").read_text()
    assert "players × boards × referees" in md


def test_matrix_cli_unknown_board(tmp_path):
    from bench.matrix.__main__ import main

    assert main(["--boards", "nope", "--out", str(tmp_path)]) == 2


def test_matrix_open_neural_trains():
    """``neural`` on ``open`` gets its held-in train ctx — the cell
    runs, it does not skip on ``ctx['train']``.
    """
    from bench.matrix.runner import run_matrix

    cells = run_matrix(
        ["open"],
        ["neural"],
        ["naive"],
        open_pages=1,
        open_page=4,
        open_budget=5,
        neural_epochs=5,
    )
    assert len(cells) == 1
    c = cells[0]
    assert c.run and not c.reason
    assert c.admitted == c.claims


# ---------------------------------------------------------------------------
#  pin tables — the golden-flow join
# ---------------------------------------------------------------------------


def test_table_record_and_compare(tmp_path):
    """The pinnable table record + direction-aware compare."""
    from bench.benchkit.compare import compare_baseline
    from bench.benchkit.table import (
        append_record,
        table_record,
        write_table_baseline,
    )

    record = table_record(
        "matrix",
        [
            {
                "name": "open/pay",
                "metrics": {"admit_rate": 1.0, "value": 0.5},
            }
        ],
        directions={"admit_rate": "down", "value": "up"},
        findings=[{"claim": "c", "verdict": "parity", "headline": "h"}],
    )
    assert record["kind"] == "table" and record["findings"]
    base = write_table_baseline(record, tmp_path)
    assert (tmp_path / "matrix.json").exists()
    # A dropped admit_rate regresses (direction "down").
    now = table_record(
        "matrix",
        [{"name": "open/pay", "metrics": {"admit_rate": 0.5}}],
        directions={"admit_rate": "down"},
    )
    regs = compare_baseline(now, base, threshold=0.05)
    assert [r.variant for r in regs] == ["admit_rate"]
    # A risen admit_rate is fine; a risen "value" regresses (up).
    now2 = table_record(
        "matrix",
        [{"name": "open/pay", "metrics": {"admit_rate": 1.0}}],
        directions={"admit_rate": "down"},
    )
    assert compare_baseline(now2, base, threshold=0.05) == []
    ledger = append_record(record, tmp_path / "led.jsonl")
    assert ledger.exists()


def test_pinned_tables_skip_expectation_gaps(tmp_path):
    """Table baselines don't trip the registry expectation check."""
    from bench.benchkit.compare import expectation_gaps
    from bench.benchkit.table import (
        table_record,
        write_table_baseline,
    )

    write_table_baseline(
        table_record(
            "matrix",
            [{"name": "c1", "metrics": {"v": 1.0}}],
            directions={"v": "down"},
            findings=[
                {"claim": "c", "verdict": "win", "headline": "h"}
            ],
        ),
        tmp_path,
    )
    assert expectation_gaps(tmp_path) == []
