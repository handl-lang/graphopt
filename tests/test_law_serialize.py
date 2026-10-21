"""Tests for laws-as-data — ``mlopt_core.laws.serialize`` + lemma store.

The lemma-store seam: a law is a 2-cell — pattern pair, side
condition, provenance — and with ``cond`` the whole record is JSON.
This file pins:

* the term codec at its canonical IR home
  (``mlopt_core.ir.term_to_data`` / ``term_from_data`` — the same
  scheme ``rulecache`` persists, round-tripped over every shipped
  pattern);
* the ``Rewrite`` record round-trip — ``law_to_data`` /
  ``law_from_data`` — including the exact census of which of the 71
  shipped laws are *full-data* (all 71 now — the last four
  procedural hooks went declarative in stage 0 of plan 0017);
* the honesty contract — a ``serializable: false`` record rebuilds
  its pattern + cond but *not* the dropped hooks: a synthetic
  flagged law fires and mints ``fused(u, dim="ND")`` (the unbound
  attr metavar falls back to its literal name — a visible scar,
  not a silent veto);
* the sqlite ``lemmas`` table — ``store_lemma`` / ``admit_lemma`` /
  ``lemma_rows`` plus the ``--add-lemma`` / ``--admit`` CLI —
  closing the loop: a stored lemma admits into a live ``Rewrite``
  that fires identically to the shipped rule (cond and all).
"""

from __future__ import annotations

import json

from mlopt_core.egraph import Certificate, EGraph, Rewrite
from mlopt_core.egraph.certs import CERT_FORMAT
from mlopt_core.ir import (
    Op,
    Param,
    TensorType,
    Var,
)
from mlopt_core.laws.tensor import (
    FACTOR_MUL,
    SOFTMAX_FOLD,
)
from mlopt_discovery import evidence as le


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _json_roundtrip(data):
    return json.loads(json.dumps(data))


# ---------------------------------------------------------------------------
#  The term codec — canonical at the IR layer
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  alpha_key — the store's identity
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
#  missing_hooks — the serializability census
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  law_to_data / law_from_data — the record round-trip
# ---------------------------------------------------------------------------






def _flagged_rule():
    """A rule data cannot fully carry — a cond + procedural check +
    procedural derive composite.  No shipped law is flagged anymore
    (the census is clean), so the honesty path gets a synthetic
    stand-in."""
    return Rewrite(
        name="flagged_fused",
        lhs=Op.make("mul", "u", Op.make("rsqrt", "u")),
        rhs=Op.make("fused", "u", dim="ND"),
        cond=("rank", "u", ">=", 1),
        check=lambda bound: True,
        derive=lambda bound: {"$attr:ND": 1},
    )












# ---------------------------------------------------------------------------
#  Reconstructed laws fire — the proof the data is enough
# ---------------------------------------------------------------------------


def _factor_src():
    x, w = _v("x", 5, 4), _p("W", 4, 3)
    a, b = _p("a", 4, 3), _p("b", 4, 3)
    return (
        Op.make(
            "add",
            Op.make("matmul", x, Op.make("matmul", w, a)),
            Op.make("matmul", x, Op.make("matmul", w, b)),
        ),
        Op.make(
            "matmul", x, Op.make("matmul", w, Op.make("add", a, b))
        ),
    )


def _factor_decline():
    x, a, b = _v("x2", 16), _p("a2", 16), _p("b2", 16, 16)
    return (
        Op.make(
            "add",
            Op.make("matmul", x, a),
            Op.make("matmul", x, b),
        ),
        Op.make("matmul", x, Op.make("add", a, b)),
    )


def _fires(rule, src, want):
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    return eg.find(root) == eg.find(eg.add_term(want))








# ---------------------------------------------------------------------------
#  The lemmas table — store, admit, fire
# ---------------------------------------------------------------------------


def _conn(tmp_path):
    return le.connect(str(tmp_path / "laws.db"))


def test_lemmas_table_write_read_admit(tmp_path):
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, FACTOR_MUL, corpus_hash="abc")
        rows = le.lemma_rows(conn)
        assert len(rows) == 1
        assert rows[0]["alpha_key"] == key
        assert rows[0]["name"] == "factor_matmul"
        assert json.loads(rows[0]["law_json"])["serializable"] is True
        assert json.loads(rows[0]["derivation_json"]) == list(
            FACTOR_MUL.derivation
        )
        got = le.admit_lemma(conn, key)
        assert got is not None
        rule, data = got
        assert data["serializable"] is True
        src, merged = _factor_src()
        assert _fires(rule, src, merged)
        # and the stored cond still vetoes the rank-1 counterexample
        bad, bad_merged = _factor_decline()
        assert not _fires(rule, bad, bad_merged)
    finally:
        conn.close()


def test_admit_lemma_unknown_key_returns_none(tmp_path):
    conn = _conn(tmp_path)
    try:
        assert le.admit_lemma(conn, "nope") is None
    finally:
        conn.close()


def test_admitted_flagged_rule_is_flagged(tmp_path):
    conn = _conn(tmp_path)
    try:
        rule = _flagged_rule()
        key = le.store_lemma(conn, rule)
        admitted, data = le.admit_lemma(conn, key)
        assert data["serializable"] is False
        assert data["missing_hooks"] == ["check", "derive"]
        assert admitted.derive is None and admitted.cond == rule.cond
    finally:
        conn.close()


def test_admitted_softmax_fold_is_full_data(tmp_path):
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, SOFTMAX_FOLD)
        rule, data = le.admit_lemma(conn, key)
        assert data["serializable"] is True
        assert rule.dspec == SOFTMAX_FOLD.dspec
        # and the admitted rule mints the derived dim, not the scar
        u = _v("u", 4, 8)
        src = Op.make(
            "div",
            Op.make("exp", u),
            Op.make("sum", Op.make("exp", u), dim=(-1,), keepdim=True),
        )
        assert _fires(rule, src, Op.make("softmax", u, dim=-1))
    finally:
        conn.close()


def test_lemmas_table_carries_replayable_certificate(tmp_path):
    """``store_lemma`` materializes the recorded derivation;
    ``stored_certificate`` replays it strictly off the record."""
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, FACTOR_MUL)
        stored = json.loads(
            conn.execute(
                "SELECT law_json FROM lemmas WHERE alpha_key = ?",
                (key,),
            ).fetchone()["law_json"]
        )
        assert stored["derivation"] == list(FACTOR_MUL.derivation)
        assert stored["cert"] is not None
        cert = le.stored_certificate(stored)
        assert cert.replayable
        assert cert.n_steps == 1
        assert set(cert.rules_used) <= set(FACTOR_MUL.derivation)
        # and admit hands the same record back — cert intact
        _rule, record = le.admit_lemma(conn, key)
        assert record["cert"] == stored["cert"]
        assert le.stored_certificate(record) is not None
    finally:
        conn.close()


def test_store_lemma_cert_knob(tmp_path):
    """``cert=`` is explicit: ``None`` suppresses materialization, a
    Certificate embeds as given."""
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, FACTOR_MUL, cert=None)
        _rule, data = le.admit_lemma(conn, key)
        assert data["cert"] is None
        empty = Certificate(
            src=FACTOR_MUL.lhs,
            dst=FACTOR_MUL.rhs,
            root_eid=None,
            steps=[],
            rules={},
        )
        key = le.store_lemma(conn, FACTOR_MUL, cert=empty)
        _rule, data = le.admit_lemma(conn, key)
        assert data["cert"]["version"] == CERT_FORMAT
        assert data["cert"]["steps"] == []
    finally:
        conn.close()


def test_lemma_cli_store_then_admit(tmp_path, capsys):
    db = str(tmp_path / "laws.db")
    assert le.main(["--report", db, "--add-lemma", "assoc_matmul"]) == 0
    out = capsys.readouterr().out
    assert "stored assoc_matmul  [full-data]" in out
    key = next(
        line.split("=", 1)[1].strip()
        for line in out.splitlines()
        if "alpha_key" in line
    )
    assert le.main(["--report", db, "--admit", key]) == 0
    out = capsys.readouterr().out
    assert "admitted assoc_matmul  [full-data]" in out
    assert "cert: none recorded" in out


def test_lemma_cli_cert_roundtrip(tmp_path, capsys):
    """A derivation-carrying law stores + admits with its cert —
    ``--admit`` replays it strictly."""
    db = str(tmp_path / "laws.db")
    assert (
        le.main(["--report", db, "--add-lemma", "factor_matmul"]) == 0
    )
    out = capsys.readouterr().out
    assert (
        "cert: 1-step derivation ['distribute_matmul_over_add']" in out
    )
    key = next(
        line.split("=", 1)[1].strip()
        for line in out.splitlines()
        if "alpha_key" in line
    )
    assert le.main(["--report", db, "--admit", key]) == 0
    out = capsys.readouterr().out
    assert "cert: 1-step ['distribute_matmul_over_add']" in out
    assert "replayed strict" in out


def test_lemma_cli_reports_missing_hooks(tmp_path, capsys):
    """Every shipped law is full-data now, so the flagged report comes
    from a synthetic rule stored directly — ``--admit`` prints its
    missing hooks."""
    db = str(tmp_path / "laws.db")
    conn = le.connect(db)
    try:
        key = le.store_lemma(conn, _flagged_rule())
    finally:
        conn.close()
    assert le.main(["--report", db, "--admit", key]) == 0
    out = capsys.readouterr().out
    assert "missing hooks: check, derive" in out


def test_lemma_cli_errors(tmp_path, capsys):
    db = str(tmp_path / "laws.db")
    assert le.main(["--report", db, "--add-lemma", "nosuchlaw"]) == 1
    assert le.main(["--report", db, "--admit", "nope"]) == 1


def test_report_counts_lemmas(tmp_path):
    conn = _conn(tmp_path)
    try:
        le.store_lemma(conn, FACTOR_MUL)
        report = le.render_report(conn, "x.db")
        assert "1 lemmas" in report
    finally:
        conn.close()
