"""Tests for ``catopt_discovery.propose`` — the vocabulary proposal.

``propose`` extends the *proposal* side of ``vocab``'s need detection
(plan 0024 part B): a complete, checkable ``declare`` proposal — the
declared op with its arity/attrs, its observable semantics, and an
**observational semantics probe** (the spelled form the atom must
equal, evaluated on real ints).  ``mod`` is bank-derivable and its
check agrees; ``pow`` is primitive and its check is honestly ``None``.
The two worked examples are the measured ``mod`` need and the
carrier-aware shape-source (``vshape``) gap; the driver ranks the
backlog and pairs each need with its proposal.
"""

import json

import pytest

from catopt_discovery import propose as pp
from catopt_discovery import vocab as vc


def test_spelled_term_mod_is_bank_derivable():
    spelled = pp.spelled_term("mod")
    assert spelled == (
        "sub",
        ("attr", "a"),
        (
            "mul",
            ("attr", "b"),
            ("floordiv", ("attr", "a"), ("attr", "b")),
        ),
    )
    assert pp.render_term(spelled) == "sub(a, mul(b, floordiv(a, b)))"


def test_spelled_term_pow_is_primitive():
    assert pp.spelled_term("pow") is None


def test_spelled_term_unknown_raises():
    with pytest.raises(ValueError):
        pp.spelled_term("nope")


def test_render_term_covers_leaves():
    assert pp.render_term("T") == "T"
    assert pp.render_term(("attr", "a")) == "a"
    assert pp.render_term(("lit", 3)) == "3"
    assert pp.render_term(("len", ("shape", "x"))) == "len(shape(x))"


def test_check_spelled_mod_agrees_everywhere():
    check = pp.check_spelled("mod")
    assert check["agrees"] is True
    assert len(check["rows"]) == len(vc._PROBE_INPUTS)
    for row in check["rows"]:
        assert row["agree"] is True
        assert row["atom"] == row["spelled"]
    # the decline pair is carried, not dropped.
    assert {
        "in": [5, 0],
        "atom": None,
        "spelled": None,
        "agree": True,
    } in (check["rows"])


def test_check_spelled_pow_is_primitive():
    check = pp.check_spelled("pow")
    assert check["agrees"] is None
    assert check["spelled"] is None
    assert check["rows"][0] == {"in": [2, 3], "atom": 8}


def test_check_spelled_custom_inputs():
    check = pp.check_spelled("mod", inputs=((9, 4),))
    assert check["agrees"] is True
    assert check["rows"] == [
        {"in": [9, 4], "atom": 1, "spelled": 1, "agree": True}
    ]


def test_mod_is_now_expressible_so_it_is_not_a_need():
    """``mod`` left the backlog when the bank learned to spell it.

    The synth-domain work (`1bffe82`) extended the derive bank with
    ``mod`` (+ the value-level twin ``dspec_synth._vmod``: int-only
    operands, nonzero divisor — the handler's strictness).  A need is a
    *measured vocabulary gap*; an op the bank can spell is no longer one,
    so ``detect_need`` correctly refuses.  This test pins that fact —
    the honest refusal the ``mod_example`` docstring anticipated.
    """
    assert vc.detect_need(pp._mod_rank_hook, pp._rank_domain()) is None
    # the op is still a candidate with a spelled form (sugar, not a
    # primitive) — expressibility is what removed it from the backlog.
    assert pp.spelled_term("mod") is not None
    assert pp.check_spelled("mod")["agrees"] is True


def test_build_proposal_is_complete_and_checkable():
    # The need is *constructed*, not detected: `mod` is expressible now
    # (above), so no domain can produce this need anymore.  The record
    # still exercises `build_proposal`'s shape, which any future need
    # would take.
    need = vc.AtomNeed(
        op="mod",
        agreement_gain=0,
        rules_blocked=("_mod_rank_hook",),
    )
    prop = pp.build_proposal(need)
    assert prop["op"] == "declare"
    assert prop["admitted"] is False
    atom = prop["params"]["construction"]["params"]
    assert atom["name"] == "mod"
    assert atom["arity"] == 2
    assert atom["attrs"] == {}
    assert atom["operands"] == ["a", "b"]
    assert atom["primitive"] is False
    assert atom["spelled"] == "sub(a, mul(b, floordiv(a, b)))"
    assert atom["spelled_check"]["agrees"] is True
    # the observable semantics table is still carried.
    assert atom["observable"][0] == {"in": [2, 3], "out": 2}
    assert prop["evidence"]["op"] == "mod"
    # JSON-safe.
    assert json.loads(json.dumps(prop)) == prop


def test_build_proposal_pow_marks_primitive():
    need = vc.detect_need(pp._pow_rank_hook, pp._rank_domain())
    assert need is not None and need.op == "pow"
    atom = pp.build_proposal(need)["params"]["construction"]["params"]
    assert atom["primitive"] is True
    assert atom["spelled"] is None
    assert atom["spelled_check"]["agrees"] is None


def test_mod_example_reports_the_honest_refusal():
    """The ``mod`` example now demonstrates the *expressible* verdict.

    Its docstring anticipated exactly this: ``None`` in the ``need``
    slot "would mean the derive was already expressible — the honest
    refusal".  The bank learned ``mod`` (`1bffe82`), so the example
    keeps its name/kind and reports no need and no proposal.
    """
    ex = pp.mod_example()
    assert ex["name"] == "mod" and ex["kind"] == "bank-op"
    assert ex["need"] is None and ex["proposal"] is None


def test_shape_source_example_refuses_arity2_and_proposes_a_source():
    ex = pp.shape_source_example()
    assert ex["name"] == "vshape" and ex["kind"] == "shape-source"
    # the honest negative: no arity-2 bank op unblocks the gap.
    assert ex["arity2_refused"] is True
    assert ex["arity2_need"] is None
    atom = ex["proposal"]["params"]["construction"]["params"]
    assert atom["kind"] == "shape-source"
    assert atom["arity"] == 1
    assert atom["spelled"] == "shape(T)"
    assert atom["observable"]["agrees"] is True
    assert atom["observable"]["shape"] == [2, 3]
    assert ex["proposal"]["admitted"] is False


def test_propose_shape_source_evidence_roundtrip():
    need = pp.StructuralNeed("vshape", "shape-source", ("r1",), "why")
    prop = pp.propose_shape_source(need)
    assert prop["evidence"]["rules_blocked"] == ["r1"]
    assert json.loads(json.dumps(prop)) == prop


def test_examples_lists_both():
    exs = pp.examples()
    assert [e["name"] for e in exs] == ["mod", "vshape"]


def test_backlog_pairs_each_need_with_a_proposal():
    rows = pp.backlog(
        pp._synthetic_rules(), measure=pp._synthetic_measure
    )
    ops = {r["need"]["op"] for r in rows}
    # `pow` is still primitive; `mod` left the backlog when the bank
    # learned to spell it (`1bffe82`) — a need is a *measured* gap.
    assert "pow" in ops
    assert "mod" not in ops
    for row in rows:
        assert row["need"]["rules_blocked"]
        atom = row["proposal"]["params"]["construction"]["params"]
        assert atom["spelled_check"]["agrees"] in (True, None)
        assert row["proposal"]["admitted"] is False


def test_fmt_backlog_renders_needs_and_proposals():
    rows = pp.backlog(
        pp._synthetic_rules(), measure=pp._synthetic_measure
    )
    text = pp._fmt_backlog(rows)
    assert "ranked backlog" in text
    assert "pow" in text
    assert "spelled:" in text and "agrees:" in text
    assert "(primitive)" in text


def test_fmt_backlog_empty():
    assert "(no missing bank op detected)" in pp._fmt_backlog([])


def test_fmt_examples_renders_both_kinds():
    text = pp._fmt_examples(pp.examples())
    assert "worked examples" in text
    assert "shape source" in text
    # `mod` renders its expressible verdict now, not a declare line.
    assert "mod: no need (expressible)" in text


def test_main_synthetic_smoke(capsys):
    rc = pp.main(["--rules", "synthetic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ranked backlog" in out
    assert "worked examples" in out


def test_main_synthetic_json(tmp_path, capsys):
    out = tmp_path / "propose.json"
    rc = pp.main(
        ["--rules", "synthetic", "--no-examples", "--json", str(out)]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["examples"] == []
    # `mod` is expressible now (`1bffe82`); `pow` is the remaining need.
    assert {r["need"]["op"] for r in payload["backlog"]} >= {"pow"}


def test_main_carrier_mode_is_bounded():
    # max_rules=0 makes the carrier scan return an empty (honest) backlog
    # without importing/measuring anything — the branch is exercised.
    assert (
        pp.main(["--rules", "carrier", "--max-rules", "0", "--deadline", "0"])
        == 0
    )


def test_carrier_rules_gathers_every_family():
    rules = pp._carrier_rules()
    assert rules, "expected carrier laws"


def test_render_term_covers_a_bare_scalar():
    assert pp.render_term(5) == "5"


def test_rank_hooks_decline_without_a_bound_term():
    assert pp._mod_rank_hook({}) is None
    assert pp._pow_rank_hook({}) is None
    assert pp._vshape_rank_hook({}) is None


def test_shape_of_op_declines_on_an_unshaped_term():
    assert pp._shape_of_op("shape", {}) is None
    assert pp._shape_of_op("vshape", {}) is None


def test_fmt_examples_handles_an_expressible_example():
    exs = [
        {
            "name": "expressible",
            "kind": "bank-op",
            "need": None,
            "proposal": None,
        }
    ]
    text = pp._fmt_examples(exs)
    assert "no need (expressible)" in text


def test_fmt_proposal_without_a_check():
    prop = {
        "params": {
            "construction": {
                "params": {
                    "kind": "k",
                    "name": "n",
                    "arity": 1,
                    "semantics": "s",
                    "spelled": None,
                }
            }
        }
    }
    text = "\n".join(pp._fmt_proposal(prop))
    assert "(primitive)" in text
    assert "agrees" not in text


def test_fmt_examples_skips_a_missing_proposal():
    exs = [
        {
            "name": "x",
            "kind": "bank-op",
            "need": {"agreement_gain": 1, "rules_blocked": ["r"]},
            "proposal": None,
        }
    ]
    assert "x: gain 1" in pp._fmt_examples(exs)
