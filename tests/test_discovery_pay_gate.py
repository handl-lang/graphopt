"""Tests for ``mlopt_discovery.pay_gate`` — the pay-prediction gate.

The gate is a scheduling heuristic, not a semantic change: these
tests pin the verdict rules on hand-built IR terms (torch-free) and
the two opt-in call sites (``gen_probe(pay_gate=...)`` early-return,
``zoo_probe(pay_gate=...)`` skipped rows).  The fitted rule and its
measured confusion live in ``docs/reports/pay_prediction_gate.md``.
"""

from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_discovery import pay_gate as pg
from mlopt_discovery import play
from mlopt_discovery import zoo as zoo_mod


def _v(*shape):
    return Var("x", TensorType(tuple(shape)))


def _op(name, *args):
    return Op.make(name, *args)


# ---------------------------------------------------------------------------
#  term_features
# ---------------------------------------------------------------------------


def test_features_counts_ops():
    x = _v(4, 4)
    term = _op(
        "add",
        _op("mul", x, _op("relu", x)),
        _op("linear", x),
    )
    f = pg.term_features(term)
    assert f["n_ops"] == 4
    assert f["compute_ops"] == 1  # linear
    assert f["other_ops"] == 3  # add, mul, relu
    assert f["select_ops"] == 0


def test_features_shared_subterms_counted_once():
    x = _v(4)
    shared = _op("mul", x, x)
    term = _op("add", shared, shared)
    f = pg.term_features(term)
    # the shared ``mul`` node is one op, not two
    assert f["n_ops"] == 2


# ---------------------------------------------------------------------------
#  predict — the three verdicts
# ---------------------------------------------------------------------------


def test_predict_skip_on_elementwise_bound():
    """Small pointwise+GEMM graph, no carrier structure → skip."""
    x = _v(8, 16)
    term = _op(
        "add",
        _op("mul", x, _op("tanh", x)),
        _op("linear", x),
    )
    v = pg.predict(term)
    assert v.cls == pg.SKIP
    assert not v.measure
    assert "elementwise-bound" in v.reasons[0]


def test_predict_skip_boundary_sizes():
    """Exactly at the bounds still skips; one over does not."""
    x = _v(4)
    # n_ops == _SKIP_MAX_OPS of pure pointwise → skip
    term = x
    for _ in range(pg._SKIP_MAX_OPS):
        term = _op("mul", term, x)
    # term has 24 muls + leaf = 24 ops? count: each mul is one op
    f = pg.term_features(term)
    assert f["n_ops"] == pg._SKIP_MAX_OPS
    assert pg.predict(term).cls == pg.SKIP
    term = _op("mul", term, x)  # 25 ops
    assert pg.predict(term).cls == pg.UNKNOWN


def test_predict_unknown_on_reduction():
    x = _v(4, 4)
    term = _op("softmax", _op("matmul", x, x))
    v = pg.predict(term)
    assert v.cls == pg.UNKNOWN
    assert v.measure


def test_predict_unknown_on_conv():
    x = _v(1, 4, 8, 8)
    term = _op("conv1d", x, x)
    assert pg.predict(term).cls == pg.UNKNOWN


def test_predict_unknown_on_quant_ops():
    x = _v(4)
    term = _op("clamp", _op("round", x), _op("detach", x))
    assert pg.predict(term).cls == pg.UNKNOWN


def test_predict_unknown_on_deep_linear_stack():
    """>=3 spelled ``linear`` nodes exempt the skip (LoRA/DEQ payers)."""
    x = _v(4, 4)
    y = _op("linear", x)
    y = _op("linear", y)
    y = _op("linear", y)
    term = _op("add", y, x)
    assert pg.predict(term).cls == pg.UNKNOWN


def test_predict_unknown_on_dequant_matmul():
    """One matmul + transpose + mul is the quantised-deploy spelling."""
    x = _v(4, 4)
    w = _op("transpose", _v(4, 4))
    term = _op("add", _op("matmul", x, w), _op("mul", x, w))
    v = pg.predict(term)
    assert v.cls == pg.UNKNOWN
    assert any("dequant" in r for r in v.reasons)


def test_predict_unknown_on_compute_heavy():
    """>5 GEMM/conv-family ops is not elementwise-bound → measure."""
    x = _v(4, 4)
    y = x
    for _ in range(pg._SKIP_MAX_COMPUTE + 1):
        y = _op("matmul", y, x)
    term = _op("add", y, x)
    v = pg.predict(term)
    assert v.cls == pg.UNKNOWN
    assert any("compute_ops" in r for r in v.reasons)


def test_predict_unknown_on_routing():
    x = _v(4, 4)
    term = _op("topk", _op("matmul", x, x), Const(2))
    v = pg.predict(term)
    assert v.cls == pg.UNKNOWN
    assert any("route_ops" in r for r in v.reasons)


def test_predict_unknown_on_einsum_readout():
    """einsum present without the scan signature → measure, not skip."""
    x = _v(4, 4)
    term = _op("einsum", x, x)
    assert pg.predict(term).cls == pg.UNKNOWN


def _scan_states(x, n):
    """``n`` *distinct* select nodes — ``Op.make`` hash-conses, so each
    state must differ to count as a separate op."""
    return [
        _op(
            "select",
            _op("add", x, _op("mul", x, Const(float(i)))),
            Const(0),
            i,
        )
        for i in range(n)
    ]


def test_predict_pay_seqt_trajectory():
    """select-chain feeding one stack → the measured seqt payers."""
    x = _v(4)
    term = _op("stack", *_scan_states(x, pg._SCAN_SELECT_MIN))
    v = pg.predict(term)
    assert v.cls == pg.PAY
    assert v.measure
    assert "seqt" in v.reasons[0]


def test_predict_pay_seqm_recurrence():
    """select-heavy recurrence, no einsum read-out, no cumsum → seqm."""
    x = _v(4)
    states = _scan_states(x, pg._SCAN_SELECT_MIN + 2)
    term = states[0]
    for s in states[1:]:
        term = _op("add", term, s)
    v = pg.predict(term)
    assert v.cls == pg.PAY
    assert "seqm" in v.reasons[0]


def test_predict_chunked_ssd_like_is_not_pay():
    """select+einsum+cumsum together is NOT the seqm signature."""
    x = _v(4)
    states = _scan_states(x, pg._SCAN_SELECT_MIN)
    term = _op(
        "add",
        _op("einsum", states[0], states[1]),
        _op("cumsum", states[2], Const(1)),
    )
    assert pg.predict(term).cls == pg.UNKNOWN


def test_should_measure_only_skips_on_skip():
    x = _v(4)
    skip_term = _op("mul", x, _op("relu", x))
    pay_term = _op("stack", *_scan_states(x, pg._SCAN_SELECT_MIN))
    unk_term = _op("softmax", x)
    assert not pg.should_measure(skip_term)
    assert pg.should_measure(pay_term)
    assert pg.should_measure(unk_term)


# ---------------------------------------------------------------------------
#  gen_probe pay_gate — the early return
# ---------------------------------------------------------------------------


def test_gen_probe_pay_gate_skips_before_board():
    """A skip verdict returns a declined row without touching the board."""
    x = _v(4)
    term = _op("mul", x, _op("relu", x))
    case = ("t", term, {}, "n", frozenset(), {"sink": None})
    row = play.gen_probe(case, pay_gate=True)
    assert row["delivered"] is False
    assert "pay-gate skip" in row["note"]
    assert row["pay_gate"]["class"] == pg.SKIP


def test_gen_probe_default_unchanged_signature():
    """``pay_gate`` defaults off — the measured path is untouched."""
    import inspect

    sig = inspect.signature(play.gen_probe)
    assert sig.parameters["pay_gate"].default is False


# ---------------------------------------------------------------------------
#  zoo wiring — real workloads, cheap verdicts only
# ---------------------------------------------------------------------------


def _wl(name, d=16, t=8, b=2):
    for wl in zoo_mod.zoo(d, t, b):
        if wl.name == name:
            return wl
    raise AssertionError(f"{name} not in zoo")


def test_zoo_pay_verdict_elementwise_head_skips():
    """FiLMHead is the measured elementwise-bound loser → skip."""
    v = play.zoo_pay_verdict(_wl("FiLMHead"))
    assert v is not None
    assert v["class"] == pg.SKIP


def test_zoo_pay_verdict_scan_carrier_measures():
    """GatedDeltaRule carries the seqm signature → never skipped."""
    v = play.zoo_pay_verdict(_wl("GatedDeltaRule"))
    assert v is not None
    assert v["class"] in (pg.PAY, pg.UNKNOWN)
    assert v["class"] != pg.SKIP


def test_zoo_pay_verdict_export_failure_is_none():
    """A workload whose build raises yields ``None`` — measure as today."""

    class _Bad:
        name = "Bad"

        def build(self):
            raise RuntimeError("no build")

    assert play.zoo_pay_verdict(_Bad()) is None


def test_zoo_probe_pay_gate_records_skips(monkeypatch):
    """``zoo_probe(pay_gate=True)`` records skipped rows, no search."""
    wl = _wl("FiLMHead")
    monkeypatch.setattr(zoo_mod, "zoo", lambda *a, **k: [wl])
    rows = play.zoo_probe(pay_gate=True, d=16, t=8, b=2)
    assert len(rows) == 1
    row = rows[0]
    assert row["case"] == "zoo:FiLMHead"
    assert row["delivered"] is False
    assert "pay-gate skip" in row["note"]
    assert row["pay_gate"]["class"] == pg.SKIP
