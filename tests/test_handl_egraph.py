"""Phase-5 native saturation: the ``vocab/`` rewrite bank through
``handl-egraph``.

``mlopt_discovery.handl_egraph`` encodes the fragment vocabulary's
declarative rewrite rows into the ``egraph-run`` JSON spec and lets
the Rust ``EGraph::run`` saturate — matching, union, congruence,
extraction all on the handl side.  These tests pin:

* the wire encoder field-for-field against the audit driver's
  (``spec_term``/``build_spec`` vs ``_spec_term``/``spec_for`` — the
  two must never drift);
* the honest split of the vocab bank — declarative rows saturate,
  ``check``/``derive``-hooked rows are counted and named, never
  silently dropped;
* end-to-end parity: the Rust report on a vocab-rule spec equals the
  Python ``EGraph`` report (the canonical differential comparison —
  ``agree``/``tie`` only), and ``run_native`` returns the identical
  document ``egraph-run`` emits for the same spec.

The binary-dependent tests are marked ``e2e`` (the suite crosses the
compiled Rust engine boundary) and skip honestly when ``egraph-run``
is not built.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, Op, Var
from mlopt_discovery import handl_egraph as he

from bench.drivers.audit import egraph_differential as ed

_REPO = Path(__file__).resolve().parents[1]


def _find_bin() -> Path | None:
    return he.egraph_run_bin()


def _build_bin() -> Path:
    """The compiled ``egraph-run``, building it if absent."""
    found = _find_bin()
    if found is not None:
        return found
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo not on PATH — egraph-run unavailable")
    env = dict(__import__("os").environ, CARGO_PROFILE_DEV_DEBUG="0")
    out = subprocess.run(
        [
            cargo,
            "build",
            "-p",
            "mlopt-kit",
            "--bin",
            "egraph-run",
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    found = _find_bin()
    if out.returncode != 0 or found is None:
        pytest.skip(f"egraph-run build failed: {out.stderr[-400:]}")
    return found


@pytest.fixture(scope="module")
def egraph_bin() -> Path:
    """The ``egraph-run`` binary path."""
    return _build_bin()


@pytest.fixture(scope="module")
def vocab() -> he.VocabRewrites:
    """The canonical vocabulary's saturation-ready split."""
    return he.vocab_rewrites()


# ---------------------------------------------------------------------------
#  The wire encoder — pinned equal to the audit driver's
# ---------------------------------------------------------------------------


def _sample_terms() -> list:
    """Concrete + pattern terms covering every wire form."""
    x, y = Var("x", "f64"), Var("y", "f64")
    return [
        x,  # leaf -> {"leaf": repr}
        "a",  # pattern metavar -> {"mvar"}
        Const(2),
        Const(2.5),
        Op.make("add", x, y),
        Op.make("mul", x, Op.make("add", y, Const(1))),
        # pattern position: str attr -> amvar, tuple attr -> __tuple__
        Op.make("transpose", x, dim0=1, dim1="d"),
        Op.make("reshape", x, shape=(2, "S")),
        Op.make("select", x, dim=-1, index="i"),
    ]


@pytest.mark.parametrize("pattern", [False, True])
def test_spec_term_matches_audit_encoder(pattern: bool) -> None:
    """``spec_term`` is the audit driver's ``_spec_term``, verbatim."""
    for t in _sample_terms():
        assert he.spec_term(t, pattern=pattern) == ed._spec_term(
            t, pattern
        )


def test_build_spec_matches_spec_for(vocab: he.VocabRewrites) -> None:
    """``build_spec`` emits the identical document ``spec_for`` does."""
    r = vocab.rules[0]
    term = ed._instantiate(r.lhs)
    fx = {
        "name": f"vocab:{r.name}",
        "terms": [term],
        "rules": [r],
        "options": {"max_iterations": 20, "max_nodes": 5_000},
        "costs": {"default": 1.0, "leaf": 0.0},
    }
    ours = he.build_spec(
        fx["name"],
        fx["terms"],
        fx["rules"],
        root=fx.get("root", len(fx["terms"]) - 1),
        options=fx["options"],
        costs=fx["costs"],
        check_terms=[term],
    )
    theirs = ed.spec_for(fx, check_terms=[term])
    # spec_for leaves check_terms in loose codec form; build_spec
    # encodes them as spec terms — compare the shared fields exactly.
    assert {k: v for k, v in ours.items() if k != "check_terms"} == {
        k: v for k, v in theirs.items() if k != "check_terms"
    }
    assert ours["check_terms"] == [he.spec_term(term)]


# ---------------------------------------------------------------------------
#  The vocab split — the honest boundary
# ---------------------------------------------------------------------------


def test_vocab_rewrites_split_is_honest(
    vocab: he.VocabRewrites,
) -> None:
    """Every shipped row lands in exactly one bucket, with reasons."""
    names = [r.name for r in vocab.rules]
    assert len(names) == len(set(names)), "duplicate rule names"
    for rule in vocab.rules:
        assert rule.check is None and rule.derive is None, (
            f"{rule.name}: a hooked row leaked into the declarative set"
        )
    skipped_names = {n for n, _ in vocab.skipped}
    assert not (set(names) & skipped_names), "row in both buckets"
    for name, reasons in vocab.skipped:
        assert reasons, f"{name}: skipped without a recorded reason"
        assert set(reasons) <= {"check", "derive"}
    # epsilon rows are a subset of the declarative set
    assert vocab.epsilon <= len(vocab.rules)
    # the split is deterministic
    again = he.vocab_rewrites()
    assert [r.name for r in again.rules] == names
    assert again.skipped == vocab.skipped
    assert again.epsilon == vocab.epsilon


def test_vocab_rewrites_requires_a_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No resolvable vocab dir is a loud error, not an empty bank."""
    from mlopt_discovery import vocabdir

    monkeypatch.setattr(vocabdir, "default_vocab_dir", lambda: None)
    with pytest.raises(ValueError, match="no vocab directory"):
        he.vocab_rewrites()


# ---------------------------------------------------------------------------
#  Binary resolution + subprocess errors
# ---------------------------------------------------------------------------


def test_egraph_run_bin_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MLOPT_EGRAPH_RUN`` wins over the candidate scan."""
    fake = tmp_path / "egraph-run"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MLOPT_EGRAPH_RUN", str(fake))
    assert he.egraph_run_bin() == fake
    monkeypatch.setenv("CATOPT_EGRAPH_RUN", str(fake))
    monkeypatch.delenv("MLOPT_EGRAPH_RUN")
    assert he.egraph_run_bin() == fake


def test_run_native_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No binary -> HandlEgraphUnavailable, never a silent fallback."""
    monkeypatch.setattr(he, "egraph_run_bin", lambda: None)
    with pytest.raises(he.HandlEgraphUnavailable):
        he.run_native({"name": "x"})


@pytest.mark.e2e
def test_run_native_bad_spec_is_loud(egraph_bin: Path) -> None:
    """A spec the Rust side rejects surfaces as EgraphRunError."""
    spec = {
        "name": "bad",
        "terms": [{"mvar": "x"}],  # a metavar is not a concrete term
        "root": 0,
        "rules": [],
        "options": {},
        "costs": {},
    }
    with pytest.raises(he.EgraphRunError):
        he.run_native(spec, timeout=30)


# ---------------------------------------------------------------------------
#  Native saturation parity — every declarative vocab row, alone
# ---------------------------------------------------------------------------


def _vocab_fixture(rule, name: str) -> dict:
    """One vocab rule over a concrete instance of its own lhs."""
    return {
        "name": name,
        "terms": [ed._instantiate(rule.lhs)],
        "rules": [rule],
        "options": {"max_iterations": 30, "max_nodes": 10_000},
        "costs": {"default": 1.0, "leaf": 0.0},
    }


@pytest.mark.e2e
@pytest.mark.parametrize(
    "idx",
    range(len(he.vocab_rewrites().rules)),
    ids=[r.name for r in he.vocab_rewrites().rules],
)
def test_vocab_rule_parity(idx: int, egraph_bin: Path) -> None:
    """Each declarative vocab row saturates identically on both engines."""
    rule = he.vocab_rewrites().rules[idx]
    fx = _vocab_fixture(rule, f"vocab:{rule.name}")
    spec = he.build_spec(
        fx["name"],
        fx["terms"],
        fx["rules"],
        root=fx.get("root", len(fx["terms"]) - 1),
        options=fx["options"],
        costs=fx["costs"],
    )
    # run_native returns the identical document egraph-run emits.
    ours = he.run_native(spec)
    theirs = ed.run_rust(egraph_bin, spec)
    assert ours == theirs
    # and the canonical comparison against the Python EGraph agrees.
    verdict, _py, _rs, detail = ed.compare(fx, egraph_bin)
    assert verdict in ("agree", "tie"), (
        f"{rule.name}: {verdict} — {detail}"
    )


@pytest.mark.e2e
def test_vocab_compound_parity(egraph_bin: Path) -> None:
    """The algebraic core of the vocab bank saturating together."""
    vocab = he.vocab_rewrites()
    wanted = (
        "comm_add",
        "comm_mul",
        "assoc_add",
        "assoc_mul",
        "id_add",
        "id_mul",
        "double_neg",
        "sub_to_add",
        "silu_expand",
        "silu_fold",
        "pow_to_square",
        "square_to_pow",
        "mul_square",
    )
    by_name = {r.name: r for r in vocab.rules}
    rules = [by_name[n] for n in wanted if n in by_name]
    assert len(rules) == len(wanted), "vocab bank drifted"
    x, y = Var("x", "f64"), Var("y", "f64")
    term = Op.make(
        "add",
        Op.make("silu", x),
        Op.make("mul", Op.make("pow", y, 2), Const(1)),
    )
    fx = {
        "name": "vocab:compound",
        "terms": [term],
        "root": 0,
        "rules": rules,
        "options": {
            "max_iterations": 40,
            "max_nodes": 20_000,
            "rule_budgets": {
                "comm_add": 300,
                "comm_mul": 300,
                "assoc_add": 300,
                "assoc_mul": 300,
            },
        },
        "costs": {
            "ops": {
                "silu": 1.0,
                "mul": 2.0,
                "sigmoid": 2.0,
                "add": 1.0,
                "pow": 3.0,
                "square": 1.0,
                "neg": 1.0,
                "sub": 2.0,
            },
            "default": 1.0,
            "leaf": 0.0,
        },
    }
    verdict, _py, _rs, detail = ed.compare(fx, egraph_bin)
    assert verdict in ("agree", "tie"), (
        f"compound: {verdict} — {detail}"
    )


@pytest.mark.e2e
def test_saturate_extracts_the_folded_form(egraph_bin: Path) -> None:
    """End-to-end: ``x*1 + 0`` saturates natively to the leaf ``x``."""
    vocab = he.vocab_rewrites()
    by_name = {r.name: r for r in vocab.rules}
    rules = [by_name[n] for n in ("id_add", "id_mul") if n in by_name]
    x = Var("x", "f64")
    term = Op.make("add", Op.make("mul", x, Const(1)), Const(0))
    rep = he.saturate(
        [term],
        rules,
        name="vocab:identity-fold",
        options={"max_iterations": 30, "max_nodes": 10_000},
        costs={"default": 1.0, "leaf": 0.0},
    )
    assert rep["stop"] == "fixed_point"
    assert rep["best"] == {"leaf": repr(x)}
    # the Python oracle has the same member in the root class
    g = EGraph()
    root = g.add_term(term)
    g.run(
        rules,
        root,
        max_iterations=30,
        max_nodes=10_000,
        cost_fn=lambda t: 0.0 if not isinstance(t, Op) else 1.0,
    )
    eid, _en = g._locate(x)
    assert eid is not None and g.find(eid) == g.find(root)


@pytest.mark.e2e
def test_saturate_check_terms_probe(egraph_bin: Path) -> None:
    """``check_terms`` ride the spec and report root membership."""
    vocab = he.vocab_rewrites()
    by_name = {r.name: r for r in vocab.rules}
    rules = [by_name["id_mul"]]
    x = Var("x", "f64")
    term = Op.make("mul", x, Const(1))
    folded = x
    rep = he.saturate(
        [term],
        rules,
        name="vocab:probe",
        options={"max_iterations": 30, "max_nodes": 10_000},
        costs={"default": 1.0, "leaf": 0.0},
        check_terms=[term, folded],
    )
    checks = {
        json.dumps(c["term"], sort_keys=True): c for c in rep["checks"]
    }
    assert len(checks) == 2
    for chk in rep["checks"]:
        assert chk["in_root"] is True
