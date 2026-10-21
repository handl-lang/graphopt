"""The novelty sieve — the three-gate promotion filter's contract.

``bench/drivers/screen/novelty_sieve.py`` decides whether a minted
``Rewrite`` is worth a human's eyes: certified (its own rule carries a
replayable certificate), paying (a ``bench/discovery`` case's certified
extraction beats the best registered baseline *and* the derivation uses
the candidate), and underivable (the known bank — minus the candidate
and its alpha-siblings — does not already prove it).

These tests pin the gate semantics on synthetic laws where the answer
is known a priori:

* a candidate sitting inside the premise bank (or alpha-renamed copies
  of it) is *excluded* — otherwise every law self-derives;
* a composite the bank already proves (``silu(neg(neg(x))) =
  x·sigmoid(x)`` — double-neg then the silu fold) demotes to ``known``;
* a true-but-underivable law (``abs(abs(x)) = abs(x)`` — no bank rule
  mentions ``abs`` at all) with a measured win reaches ``review``;
* a false law (``neg(x) = x``) is refuted by the fp64 oracle before
  the gates even run;
* a law whose guard accepts nothing reports ``no-instance``.

The cert/derivability gates run on the real machinery —
``verify_law``, ``alpha_key``, the oracle — against the real shipped
bank.  The measured gate is exercised through the ``measurer`` seam
(the driver-side real run lives in ``docs/results_novelty_sieve.md``);
``measure_gate``'s own structure is covered directly.
"""

from __future__ import annotations

import pytest
from mlopt_core.ir import Op
from mlopt_core.laws import R
from mlopt_core.laws.serialize import alpha_key

from bench.drivers.screen import novelty_sieve as ns

# -- the synthetic laws ---------------------------------------------------
#
# ABS_ABS:  abs(abs(x)) -> abs(x).  True (idempotence), and the shipped
# bank contains no ``abs`` rule at all — underivable by construction.
ABS_ABS = R(
    "sieve_abs_abs",
    Op.make("abs", Op.make("abs", "x")),
    Op.make("abs", "x"),
    law="abs is idempotent",
)
#: The same law under metavariable renaming — an alpha-sibling.
ABS_ABS_RENAMED = R(
    "sieve_abs_abs_renamed",
    Op.make("abs", Op.make("abs", "w")),
    Op.make("abs", "w"),
    law="abs is idempotent (renamed spelling)",
)
# SILU_NN: silu(neg(neg(x))) -> mul(x, sigmoid(x)).  A composite the
# bank proves in two steps (double_neg, then the silu fold) — the
# proposal battery's known-derivable case spelled as a rule.
SILU_NN = R(
    "sieve_silu_nn",
    Op.make("silu", Op.make("neg", Op.make("neg", "x"))),
    Op.make("mul", "x", Op.make("sigmoid", "x")),
    law="silu of a double negation",
)
# NEG_ID: neg(x) -> x.  False — the oracle refutes it numerically.
NEG_ID = R(
    "sieve_neg_id", Op.make("neg", "x"), "x", law="false identity"
)


@pytest.fixture(scope="module")
def bank() -> list:
    """The real shipped bank — derivability is tested against it."""
    return ns.known_bank()


def _won(_c, _ctxs, **_kw):
    """A measured-gate stub reporting one attributed win."""
    return True, [
        ns.BeatRow(
            case="synthetic",
            certified=True,
            fired=1,
            used=True,
            cost_before=10.0,
            cost_after=4.0,
            floor=9.0,
            beat_best=True,
            beat=True,
        )
    ]


def _lost(_c, _ctxs, **_kw):
    """A measured-gate stub reporting no win."""
    return False, [
        ns.BeatRow(
            case="synthetic",
            certified=True,
            fired=1,
            used=True,
            cost_before=10.0,
            cost_after=12.0,
            floor=9.0,
            beat_best=False,
            beat=False,
        )
    ]


class TestSiblingExclusion:
    """The premise bank must not contain the candidate's own law."""

    def test_alpha_key_matches_across_renaming(self):
        assert alpha_key(ABS_ABS.lhs, ABS_ABS.rhs) == alpha_key(
            ABS_ABS_RENAMED.lhs, ABS_ABS_RENAMED.rhs
        )

    def test_candidate_in_bank_is_excluded(self, bank):
        premises, sibs = ns._premises(ABS_ABS, [*bank, ABS_ABS])
        assert "sieve_abs_abs" in sibs
        assert all(r.name != "sieve_abs_abs" for r in premises)

    def test_renamed_sibling_is_excluded(self, bank):
        premises, sibs = ns._premises(ABS_ABS, [*bank, ABS_ABS_RENAMED])
        assert "sieve_abs_abs_renamed" in sibs
        assert all(r.name != "sieve_abs_abs_renamed" for r in premises)

    def test_self_derivation_would_trivially_merge(self, bank):
        """Guard the guard: with the sibling left in, the candidate
        IS derivable — so the exclusion is what does the work."""
        got = ns.instance_of(ABS_ABS, [])
        assert got is not None
        _src, (lhs, rhs) = got
        with_self = ns.verify_law(
            lhs, rhs, [ABS_ABS_RENAMED], max_nodes=5_000
        )
        assert with_self.derivable

    def test_inverse_is_not_a_sibling(self, bank):
        """The reversed equation is a different law — it stays in the
        premises (a bank holding both directions honestly knows the
        equality)."""
        inv = R(
            "sieve_abs_abs_inv",
            Op.make("abs", "x"),
            Op.make("abs", Op.make("abs", "x")),
        )
        _p, sibs = ns._premises(ABS_ABS, [*bank, inv])
        assert "sieve_abs_abs_inv" not in sibs


class TestGates:
    """Per-gate verdicts on laws whose status is known a priori."""

    def test_underivable_true_law_reaches_review(self, bank):
        row = ns.sieve(ABS_ABS, bank, ctxs=[], measurer=_won)
        assert row.cert
        assert row.truth == "equal"
        assert row.derivable is False
        assert row.beat is True
        assert row.verdict == "review"

    def test_underivable_without_pay_is_unpaid(self, bank):
        row = ns.sieve(ABS_ABS, bank, ctxs=[], measurer=_lost)
        assert row.derivable is False
        assert row.beat is False
        assert row.verdict == "unpaid"

    def test_underivable_unexercised_is_unmeasured(self, bank):
        row = ns.sieve(ABS_ABS, bank, ctxs=[], measure=True)
        assert row.derivable is False
        assert row.beat is None
        assert row.verdict == "unmeasured"

    def test_composite_is_demoted_known(self, bank):
        row = ns.sieve(SILU_NN, bank, ctxs=[], measure=False)
        assert row.cert
        assert row.derivable is True
        assert row.verdict == "known"
        assert row.witness_rules

    def test_false_law_is_refuted(self, bank):
        row = ns.sieve(NEG_ID, bank, ctxs=[], measure=False)
        assert row.verdict == "refuted"
        assert row.truth == "unequal"

    def test_no_instance_law(self, bank):
        """A rule whose guard vetoes every binding → no-instance."""
        hopeless = R(
            "sieve_hopeless",
            Op.make("add", "x", "y"),
            "x",
            check=lambda _bound: False,
        )
        row = ns.sieve(hopeless, bank, ctxs=[], measure=False)
        assert row.verdict == "no-instance"


class TestRowShape:
    """The record is JSON-shaped for the queue doc."""

    def test_to_dict_keys(self, bank):
        row = ns.sieve(ABS_ABS, bank, ctxs=[], measurer=_won)
        d = row.to_dict()
        for k in ("cert", "derivable", "beat", "verdict", "name"):
            assert k in d
        assert d["verdict"] == "review"

    def test_render_has_queue_section(self, bank):
        rows = [
            ns.sieve(SILU_NN, bank, ctxs=[], measure=False),
            ns.sieve(ABS_ABS, bank, ctxs=[], measurer=_won),
        ]
        md = ns.render_markdown(rows)
        assert "## Review queue" in md
        assert "`sieve_abs_abs`" in md
        assert "known" in md


class TestMeasureGateStructure:
    """The measured gate's relevance filter and row shape."""

    def test_no_contexts_means_unmeasured(self):
        beat, rows = ns.measure_gate(ABS_ABS, [])
        assert beat is None
        assert rows == []

    def test_beatrow_defaults(self):
        r = ns.BeatRow(case="c")
        assert not r.beat and not r.certified and r.floor is None
