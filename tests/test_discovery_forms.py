"""Tests for ``catopt_discovery.forms`` — the form-level enumerator.

``forms`` generates law *skeletons* over a meta-grammar (op-class slots
at nodes, metavariables / a literal-class at leaves) instead of
replaying the hand-written ``enumerate._FORMS`` list.  These tests pin
the meta-grammar (tree stratification, canonical metavariable and slot
labelling, the tautology / fresh-metavar filters), the **superset**
property (every shipped ``_f_*`` skeleton reappears as a member), the
instantiation flow (skeletons to ``proposal.Candidate``s through the
existing filters, alpha-key dedup and relation annotation), and the
falsification harness — the plan's test of whether form-level
enumeration admits laws from forms *outside* the shipped 27.
"""

import itertools
import json

import pytest
from catopt_core.ir import Const, Op, Var
from catopt_discovery import forms as fm
from catopt_discovery import proposal as lp
from catopt_discovery.enumerate import _FORMS, _form_pairs

#: A small vocabulary — 2 binary + 2 unary ops — so the op-class slot
#: count stays at 2 and the superset scan stays bounded.
_BIN = ("add", "mul")
_UN = ("neg", "exp")
_OPS = ("add", "mul", "sub", "div", "neg", "exp", "square", "sqrt")
_METAS = ("X", "Y", "Z")


def _skey(sk):
    return f"{fm._render(sk.lhs)} = {fm._render(sk.rhs)}"


def _form_keys():
    """Skeleton keys grouped by shipped form name."""
    per: dict[str, set] = {}
    for form, lhs, rhs in _form_pairs(_BIN, _UN, _METAS, (0,)):
        sl, sr = fm._canonical_slots(lhs, rhs, _METAS)
        if sl == sr:
            continue
        per.setdefault(form, set()).add(
            f"{fm._render(sl)} = {fm._render(sr)}"
        )
    return per


def _scan(max_forms=700_000, stop_at=None):
    """Stream the skeleton keys, optionally stopping once *stop_at* hit."""
    seen: set = set()
    remaining = set(stop_at) if stop_at is not None else None
    for sk in fm.enumerate_forms(
        max_nodes=3,
        bin_ops=_BIN,
        un_ops=_UN,
        literals=(0, 1),
        max_forms=max_forms,
    ):
        key = _skey(sk)
        seen.add(key)
        if remaining is not None:
            remaining.discard(key)
            if not remaining:
                return seen
    return seen


# ---------------------------------------------------------------------------
#  Meta-grammar helpers
# ---------------------------------------------------------------------------


def test_trees_by_count_stratifies():
    by_n = fm._trees_by_count(2, ("X", "Y", fm._LIT))
    assert by_n[0] == ["X", "Y", fm._LIT]
    # unary over 3 leaves + binary over 3x3 pairs.
    assert len(by_n[1]) == 3 + 9
    for t in by_n[2]:
        assert t[0] in ("bin", "un")
    # every tree at level n has exactly n op nodes
    for n in (1, 2):
        assert all(fm._nodes(t) == n for t in by_n[n])


def test_metas_nodes_helpers():
    t = ("bin", ("un", "X"), "Y")
    assert fm._metas_of(t) == frozenset({"X", "Y"})
    assert fm._metas_of(fm._LIT) == frozenset()
    assert fm._nodes(t) == 2
    assert fm._nodes("X") == 0 and fm._nodes(fm._LIT) == 0


def test_is_canonical_renames_first_appearance():
    assert fm._is_canonical(("bin", "X", "Y"), _METAS)
    assert fm._is_canonical(("bin", "X", "X"), _METAS)
    # Y before X is not first-appearance canonical.
    assert not fm._is_canonical(("bin", "Y", "X"), _METAS)
    # a literal-class leaf is transparent.
    assert fm._is_canonical(("bin", "X", fm._LIT), _METAS)


def test_kinds_joint_order():
    kinds = fm._kinds((("bin", "X", "Y"), ("un", "X")))
    assert kinds == ["b", "u"]


def test_labelings_are_canonical_partitions():
    # one bin node -> only slot 0
    assert list(fm._labelings(["b"], 2, 2)) == [(0,)]
    # two bin nodes -> both-same and both-distinct, first-appearance.
    assert set(fm._labelings(["b", "b"], 2, 2)) == {(0, 0), (0, 1)}
    # bounded by the vocabulary: one slot only.
    assert set(fm._labelings(["b", "b"], 1, 1)) == {(0, 0)}


def test_relabel_applies_slots():
    pair = (("bin", "X", "Y"), ("un", "X"))
    sl, sr = fm._relabel(pair, (0, 1))
    assert sl == ("bin", 0, "X", "Y")
    assert sr == ("un", 1, "X")


def test_render_slot_and_concrete():
    assert fm._render(("bin", 0, "X", fm._LIT)) == "B0(X, K)"
    assert fm._render(("un", 1, "X")) == "U1(X)"
    assert fm._render("X") == "X"
    assert fm._render(("add", "X", "Y")) == "add(X, Y)"


# ---------------------------------------------------------------------------
#  The skeleton stream
# ---------------------------------------------------------------------------


def test_stream_is_lazy_deterministic_and_capped():
    a = list(
        itertools.islice(
            fm.enumerate_forms(
                max_nodes=3,
                bin_ops=_BIN,
                un_ops=_UN,
                literals=(0, 1),
            ),
            12,
        )
    )
    assert len(a) == 12  # no need to exhaust
    b = list(
        itertools.islice(
            fm.enumerate_forms(
                max_nodes=3,
                bin_ops=_BIN,
                un_ops=_UN,
                literals=(0, 1),
            ),
            12,
        )
    )
    assert a == b
    capped = list(
        fm.enumerate_forms(
            max_nodes=3,
            bin_ops=_BIN,
            un_ops=_UN,
            literals=(0, 1),
            max_forms=25,
        )
    )
    assert len(capped) == 25


def test_stream_rejects_empty_vocabulary():
    with pytest.raises(ValueError, match="at least one binary"):
        list(
            fm.enumerate_forms(
                max_nodes=1,
                bin_ops=(),
                un_ops=_UN,
                literals=(0,),
            )
        )


def test_stream_is_tautology_free():
    # A skeleton is never a tautology: identical sides are dropped.
    for sk in itertools.islice(
        fm.enumerate_forms(
            max_nodes=3,
            bin_ops=_BIN,
            un_ops=_UN,
            literals=(0, 1),
        ),
        400,
    ):
        assert sk.lhs != sk.rhs
        # rhs uses no fresh metavariable.
        assert fm._metas_of(sk.rhs) <= fm._metas_of(sk.lhs)


# ---------------------------------------------------------------------------
#  Superset — the shipped 27 reappear
# ---------------------------------------------------------------------------


def test_every_shipped_form_skeleton_is_a_member():
    per = _form_keys()
    assert len(per) == len(_FORMS)  # every generator emits
    all_keys = {k for ks in per.values() for k in ks}
    seen = _scan(stop_at=all_keys)
    missing = {f: ks - seen for f, ks in per.items() if ks - seen}
    assert not missing, missing


def test_shipped_skeleton_keys_match_the_form_keys():
    per = _form_keys()
    union = {k for ks in per.values() for k in ks}
    assert fm.shipped_skeleton_keys(_BIN, _UN, _METAS) == union


# ---------------------------------------------------------------------------
#  Skeleton keys — concrete pair vs abstracted shape
# ---------------------------------------------------------------------------


def test_skeleton_key_abstracts_ops_and_shares_shape():
    # add(X,0)=X and mul(X,1)=X share the id-right skeleton.
    k1 = fm.skeleton_key(("add", "X", 0), "X", _METAS)
    k2 = fm.skeleton_key(("mul", "X", 1), "X", _METAS)
    assert k1 == k2 == "B0(X, K) = X"
    # unary-swap keeps its two distinct op-classes.
    k3 = fm.skeleton_key(
        ("neg", ("exp", "X")), ("exp", ("neg", "X")), _METAS
    )
    assert k3 == "U0(U1(X)) = U1(U0(X))"


def test_canonical_slots_drops_tautology_for_same_op():
    # the same op on both sides of a swap is a tautology skeleton.
    sl, sr = fm._canonical_slots(
        ("neg", ("neg", "X")), ("neg", ("neg", "X")), _METAS
    )
    assert sl == sr


# ---------------------------------------------------------------------------
#  Instantiation — skeletons to candidates
# ---------------------------------------------------------------------------


def test_slot_count_and_build_spec():
    sk = fm.Skeleton(
        ("bin", 0, ("un", 0, "X"), fm._LIT),
        ("un", 1, "X"),
        _BIN,
        _UN,
        (0, 1),
    )
    nb, nu, nl = fm._slot_count(sk)
    assert (nb, nu, nl) == (1, 2, 1)
    lhs = fm._build_spec(sk.lhs, ("add",), ("neg", "exp"), iter((0,)))
    assert lhs == ("add", ("neg", "X"), 0)


def test_assignments_are_a_bounded_product():
    sk = fm.Skeleton(("bin", 0, "X", fm._LIT), "X", _BIN, _UN, (0, 1))
    combos = list(fm._assignments(sk))
    # 2 bin ops x 2 literals, no unary slot.
    assert len(combos) == 4
    assert {ba for ba, _, _ in combos} == {("add",), ("mul",)}


def test_form_candidates_instantiated_deduped_and_annotated():
    cands = list(
        fm.enumerate_form_candidates(max_nodes=3, max_candidates=120)
    )
    assert cands, "expected a non-empty stream"
    assert all(c.strategy == "form" for c in cands)
    keys = [lp._key(c.lhs, c.rhs) for c in cands]
    assert len(keys) == len(set(keys))  # alpha-key dedup
    assert all(k[0] != k[1] for k in keys)  # no tautology
    for c in cands[:60]:
        assert c.note.startswith("form=")
        assert "relation=" in c.note
        for leaf in lp._leaves(c.lhs) | lp._leaves(c.rhs):
            assert isinstance(leaf, (Var, Const))


def test_form_candidates_are_lazy_and_capped():
    it = fm.enumerate_form_candidates(max_nodes=3)
    first = list(itertools.islice(it, 15))
    assert len(first) == 15
    capped = list(
        fm.enumerate_form_candidates(max_nodes=3, max_candidates=40)
    )
    assert len(capped) == 40


def test_form_candidates_flow_through_proposal_evaluate():
    cands = list(
        fm.enumerate_form_candidates(max_nodes=3, max_candidates=800)
    )
    outs = lp.evaluate(cands[:6])
    assert len(outs) == 6
    table = lp.yield_table(outs)
    assert table["form"]["proposed"] == 6
    # the id-right form's instantiation is true and pays.
    id_right = [c for c in cands if "(sub 'X', 0) = 'X'" in c.note]
    assert id_right
    assert fm._form_key(id_right[0]) == "B0(X, K) = X"
    (out,) = lp.evaluate(id_right[:1])
    assert out.num_true is True
    assert out.useful is True


def test_form_candidates_custom_vocabulary():
    cands = list(
        fm.enumerate_form_candidates(
            max_nodes=2,
            bin_ops=("add",),
            un_ops=("neg",),
            literals=(0,),
            max_candidates=50,
        )
    )
    assert cands
    for c in cands:
        for leaf in lp._leaves(c.lhs) | lp._leaves(c.rhs):
            assert isinstance(leaf, (Var, Const))
    # every op in the terms is from the vocabulary.
    for c in cands:
        assert set(_ops_of(c.lhs)) | set(_ops_of(c.rhs)) <= {
            "add",
            "neg",
        }


def _ops_of(term):
    if isinstance(term, Op):
        yield term.op
        for a in term.args:
            yield from _ops_of(a)


# ---------------------------------------------------------------------------
#  Falsification harness
# ---------------------------------------------------------------------------


def test_falsification_report_shape_and_honesty():
    rep = fm.falsification_report(
        budget=120, evaluate_n=30, max_nodes=3
    )
    assert rep["forms_shipped"] > 0
    assert rep["candidates"] == 120
    assert rep["evaluated"] == 30
    assert rep["paying"] >= rep["paying_new_form"]
    assert rep["paying_new_form"] == len(rep["new_forms"])
    # attribution: a paying law's form key is either shipped or listed new.
    shipped = rep["shipped"]
    for o in rep["outcomes"]:
        if o.useful:
            key = fm._form_key(o.candidate)
            assert key in shipped or key in rep["new_forms"]


def test_falsification_new_form_accounting():
    # The id-right form is shipped, so its paying law is not "new".
    rep = fm.falsification_report(
        budget=400, evaluate_n=120, max_nodes=3
    )
    for o in rep["outcomes"]:
        if o.useful:
            assert fm._form_key(o.candidate) in rep["shipped"]


# ---------------------------------------------------------------------------
#  Reporting + CLI
# ---------------------------------------------------------------------------


def test_form_key_extraction():
    cand = lp.Candidate(
        "form", "0", "X", "Y", note="form=B0(X, K) = X; rel=new"
    )
    assert fm._form_key(cand) == "B0(X, K) = X"


def test_fmt_report_both_outcomes():
    text = fm._fmt_report(
        {
            "forms_shipped": 3,
            "candidates": 10,
            "evaluated": 4,
            "paying": 0,
            "paying_new_form": 0,
            "new_forms": [],
        }
    )
    assert "falsified" in text
    text2 = fm._fmt_report(
        {
            "forms_shipped": 3,
            "candidates": 10,
            "evaluated": 4,
            "paying": 1,
            "paying_new_form": 1,
            "new_forms": ["U0(X) = X"],
        }
    )
    assert "new form: U0(X) = X" in text2


def test_parse_csv():
    assert fm._parse_csv("a,b,,c") == ("a", "b", "c")
    assert fm._parse_csv("0,1") == ("0", "1")


def test_main_smoke_and_json(tmp_path, capsys):
    out = tmp_path / "forms.json"
    rc = fm.main(
        [
            "--ops",
            "add,mul,neg,exp",
            "--budget",
            "80",
            "--evaluate",
            "8",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "form-level enumeration" in printed
    assert "form" in printed and "prop" in printed
    payload = json.loads(out.read_text())
    assert payload["candidates"] == 80
    assert payload["evaluated"] == 8
    assert payload["shipped_forms"]
    assert len(payload["outcomes"]) == 8
    assert all(
        o["form"].startswith(("B", "U")) for o in payload["outcomes"]
    )


def test_main_without_json(capsys):
    rc = fm.main(
        [
            "--ops",
            "add,neg",
            "--metas",
            "X,Y",
            "--budget",
            "30",
            "--evaluate",
            "4",
        ]
    )
    assert rc == 0
    assert "top-ranked" in capsys.readouterr().out
