"""Lemma certificates — ``Rewrite.derivation`` as replayable proof data.

The axiom/lemma split records each non-kernel rule's ``derivation`` —
the shipped premises proving the law's instance.  These tests pin the
upgrade that turns the annotation into a *certificate*: saturate the
lemma's instance under the recorded premises alone, reconstruct the
positional derivation, and replay it strictly on real terms.  The
measured shape (``mlopt_discovery.lemma_cert``): every derivation-carrying
rule gets a linear certificate — one step, the recorded premise — and
the replayable direction is the one the *axiom* fires forward.

Also covers the certificate data codec (``cert_to_data`` /
``cert_from_data``): steps are JSON-safe records — terms through
``term_to_data``, bindings tagged term/attr — and a rebuilt
certificate replays under ``verify_certificate`` against a name→rule
map, closing the lemma-store loop (pattern + cond + derivation +
replayable proof).
"""

import json

import mlopt_core.laws.layout
import pytest
from mlopt_core import laws as R
from mlopt_core.egraph import (
    Certificate,
    CertificateVerificationError,
    CertStep,
    EGraph,
    cert_from_data,
    cert_to_data,
    verify_certificate,
)
from mlopt_core.ir import Op, TensorType, Var, op_repr

from mlopt_discovery import evidence as le
from mlopt_discovery import lemma_cert as llc


def _t(d=4):
    return TensorType((d, d))


def _merged_pair(lhs, rhs, rules):
    """Intern both sides, saturate under *rules*, return (eg, eids)."""
    eg = EGraph()
    r_lhs = eg.add_term(lhs)
    r_rhs = eg.add_term(rhs)
    eg.run(rules, r_lhs, max_iterations=20, max_nodes=50_000)
    assert eg.find(r_lhs) == eg.find(r_rhs), "sides did not merge"
    return eg, r_lhs, r_rhs


# ---------------------------------------------------------------------------
#  Lemmas certify as one-step derivations of their recorded premise
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  The serialization seam — certificates round-trip as JSON data
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
#  The honest boundary — merged but reconstruction-hard
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  The admission seam — a stored lemma carries its certificate
# ---------------------------------------------------------------------------


def test_stored_lemma_cert_roundtrips_through_admit(tmp_path):
    """The full loop: lemma -> record -> admit -> cert replays strict.

    ``store_lemma`` materializes ``silu_mul_form``'s recorded
    derivation (``silu_expand`` at path ``(0,)`` — the one composite
    step in the lemma set); ``admit_lemma`` rebuilds the Rewrite;
    ``stored_certificate`` decodes the record's ``cert`` field and
    verifies it strictly.  The replayed term is the certificate's
    ``dst`` — the proof travels with the law.
    """
    by_name = {r.name: r for r in R.ALL_RULES}
    rule = by_name["silu_mul_form"]
    conn = le.connect(str(tmp_path / "laws.db"))
    try:
        key = le.store_lemma(conn, rule)
        got = le.admit_lemma(conn, key)
        assert got is not None
        rebuilt, record = got
        assert rebuilt.name == "silu_mul_form"
        assert record["cert"] is not None
        cert = le.stored_certificate(record)
        assert cert.replayable
        assert cert.rules_used == list(rule.derivation)
        assert [(s.rule, s.path) for s in cert.steps] == [
            ("silu_expand", (0,))
        ]
        out = verify_certificate(cert.src, cert, strict=True)
        assert op_repr(out) == op_repr(cert.dst)
    finally:
        conn.close()


def test_right_factor_linear_stores_no_cert(tmp_path):
    """The honest boundary: ``cert: null`` where nothing replays.

    ``right_factor_linear`` carries no ``derivation`` annotation and
    under ``ALL_RULES`` its instance sides never even merge (the
    NT-bridge derivation exists only under ``--with-layout`` — and
    the annotation never claimed it).  ``materialize`` yields no
    certificate, and the stored record says so: ``cert`` is ``null``,
    not a stub.
    """
    by_name = {r.name: r for r in R.ALL_RULES}
    rfl = by_name["right_factor_linear"]
    assert rfl.derivation == ()
    row, cert = llc.materialize(rfl, list(R.ALL_RULES))
    assert cert is None
    assert row.verdict == "gap"
    conn = le.connect(str(tmp_path / "laws.db"))
    try:
        le.store_lemma(conn, rfl)
        stored = json.loads(le.lemma_rows(conn)[0]["law_json"])
        assert stored["cert"] is None
        assert le.stored_certificate(stored) is None
    finally:
        conn.close()
