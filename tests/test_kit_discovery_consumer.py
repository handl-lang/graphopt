"""Consumer-bound tests for ``kit_discovery`` — the graphopt-only surface.

These tests exercise ``kit_discovery`` paths that read the *consumer*
checkout, which does not exist inside the kit-discovery repo itself:

* the ``bench`` law-case registry (``kit_discovery.verifier._law_cases``,
  ``kit_discovery.impact._bench_cases`` reached through
  ``instance_of``/``composite_candidates``/``rediscover``);
* the real bench + intake + model corpus (``census.run_census``,
  ``vocab.derive_vocabulary``, ``evidence.default_gauntlet_corpus``,
  ``arena.make_arena``);
* the ``mlopt_torch`` export lowering whose op spelling is
  torch-version sensitive (the slice-write ``fill_`` threader —
  torch 2.14 spells ``broadcast_to``, torch 2.15 does not).

They live here, in the consumer repo, for the same reason the other
domain packages' cross-stack tests stayed in graphopt: every import
resolves through this env's ``bench`` package, real corpus and pinned
torch.  Carried over verbatim from ``kit-discovery:tests/`` at the
D-1d split.
"""

import json

import pytest
import torch
from mlopt_core.ir import Const, IR, Op, Param, TensorType, Var, op_repr_dag
from mlopt_core.laws import ALL_RULES
from kit_discovery import arena as ar
from kit_discovery import census as cs
from kit_discovery import evidence as ev
from kit_discovery import meta_game as mg
from kit_discovery import proposal as pp
from kit_discovery import vocab as vc
from kit_discovery import verifier as vf
from kit_discovery.census import CorpusTerm, op_tuple_census
from mlopt_torch.adapters import TorchSink
from mlopt_torch.torch_bridge import export_to_ir, ir_to_torch_module

_BY_NAME = {r.name: r for r in ALL_RULES}


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _meta() -> dict:
    """A fixed evidence scope (no git probe in tests)."""
    return {"code_rev": "test"}


def _tiny_terms() -> list:
    """Six small, well-typed terms — the corpus every tool reads."""
    x = _v("x", 4, 4)
    y = _v("y", 4, 4)
    z = _v("z", 4, 4)
    w = Param("w", TensorType((4, 4)))
    return [
        _p("mul", _p("silu", x), y),
        _p("mul", x, _p("sigmoid", x)),
        _p("add", _p("mul", x, y), z),
        _p("mul", x, Const(1)),
        _p("select", _p("mul", x, y), dim=0, index=1),
        _p("add", _p("mul", x, w), y),
    ]


# ---------------------------------------------------------------------------
#  arena — the real board split (bench + intake + model corpus)
# ---------------------------------------------------------------------------


def test_make_arena_builds_the_real_board(tmp_path):
    """``make_arena`` splits the pipeline's own probe pool.

    The holdout is real-only (the intake ledger bars its
    purpose-built spellings), every held case is probe-eligible
    (feed-bearing), and nothing held out is observable in the state.
    """
    arena = ar.make_arena(seed=0, meta={"code_rev": "test"})
    assert arena.working and arena.holdout
    pb = ar.purpose_built_names()
    held = {c.name for c in arena.holdout}
    assert held and not (held & pb)
    assert all(c.feed for c in arena.holdout)
    state = arena.observe()
    blob = json.dumps(state.to_dict())
    # name-level isolation is strict; a *repr* only counts as a
    # leak when no working case shares that program verbatim.
    visible = {cv.name for cv in state.corpus} | set(state.pending)
    work_reprs = {cv.term for cv in state.corpus}
    for c in arena.holdout:
        assert c.name not in visible
        assert f'"{c.name}"' not in blob
        term_repr = op_repr_dag(c.term)
        if term_repr not in work_reprs:
            assert term_repr not in blob
    # the bench spellings and intake remainder are playable material
    assert state.pending or any(
        c.source == "bench" for c in state.corpus
    )


def test_make_arena_max_cases_is_deterministic_and_isolated(tmp_path):
    """The corpus cap thins working+pending; the holdout stays honest.

    ``max_cases`` subsamples *after* the honest split, so a held-out
    case can never leak back into working, the holdout keeps its
    real-only/probe-eligible contract, and the whole draw replays
    under the same seed.
    """
    a1 = ar.make_arena(
        seed=0, max_cases=12, max_holdout=6, meta=_meta()
    )
    a2 = ar.make_arena(
        seed=0, max_cases=12, max_holdout=6, meta=_meta()
    )

    def _wnames(a: ar.Arena) -> list:
        return [c.name for c in a.working]

    assert _wnames(a1) == _wnames(a2)  # seeded → replays
    assert [c.name for c in a1.holdout] == [c.name for c in a2.holdout]
    # the caps bind
    assert 0 < len(a1.working) <= 12
    assert len(a1.pending) <= 12
    assert 0 < len(a1.holdout) <= 6
    # holdout isolation under subsampling: no held name is playable
    held = {c.name for c in a1.holdout}
    assert held.isdisjoint({c.name for c in a1.working})
    assert held.isdisjoint(set(a1.pending))
    # and the holdout keeps its honest contract: real-only, feeds
    pb = ar.purpose_built_names()
    assert held and not (held & pb)
    assert all(c.feed for c in a1.holdout)
    # the thinned sets are subsets of the full board's honest splits
    full = ar.make_arena(seed=0, meta=_meta())
    assert set(_wnames(a1)) <= {c.name for c in full.working}
    assert held <= {c.name for c in full.holdout}
    assert set(a1.pending) <= set(full.pending)
    # the state view still hides every held case name
    blob = json.dumps(a1.observe().to_dict())
    assert all(f'"{n}"' not in blob for n in held)


# ---------------------------------------------------------------------------
#  census / vocab — the real-corpus report paths
# ---------------------------------------------------------------------------


def test_run_census_over_real_corpus():
    result = cs.run_census(top=10)
    assert result["n_terms"] > 0
    assert result["n_bench"] > 0 and result["n_models"] > 0
    assert result["n_terms"] == (
        result["n_bench"] + result["n_models"] + result["n_intake"]
    )
    assert result["n_op_nodes"] > 0
    assert result["n_shapes"] > 0
    ot = result["op_tuples"][0]
    assert set(ot) == {"op", "children", "count", "terms"}
    assert ot["count"] >= 1 and ot["terms"] >= 1
    sh = result["shapes"][0]
    assert set(sh) == {"shape", "count", "terms", "shared"}
    assert len(result["op_tuples"]) <= 10
    for row in result["sharing"]:
        assert set(row) >= {
            "op",
            "children",
            "sites",
            "shared_child",
            "shared_leaf",
        }


def test_main_prints_report_and_dumps_json(tmp_path, capsys):
    out = tmp_path / "census.json"
    rc = cs.main(["--json", str(out), "--top", "5"])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_terms"] > 0
    assert len(payload["op_tuples"]) <= 5
    text = capsys.readouterr().out
    assert "law_shape_census" in text
    assert "op-tuple census" in text
    assert "sharing census" in text


def test_census_main_without_json(capsys):
    """The report path prints and returns 0 without ``--json``."""
    rc = cs.main(["--top", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "op-tuple census" in out
    assert "wrote" not in out


def test_vocab_main_without_json(capsys):
    """The report path prints and returns 0 without ``--json``."""
    rc = vc.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "op vocabulary by property" in out
    assert "wrote" not in out


def test_derive_vocabulary_over_real_corpus():
    v = vc.derive_vocabulary()
    assert len(v.classes) > 40
    ops = {c.op for c in v.classes}
    assert {"add", "mul", "matmul"} <= ops
    assert "add" in v.pointwise


def test_main_derives_validates_dumps(tmp_path, capsys):
    out = tmp_path / "v.json"
    assert vc.main(["--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert len(payload["classes"]) > 40
    assert "corpus_ops" in payload["validation"]
    text = capsys.readouterr().out
    assert "derived vs hand tables" in text


# ---------------------------------------------------------------------------
#  composite candidates — instance_of rides the bench registry
# ---------------------------------------------------------------------------


def test_composite_dedups_alpha_identical_candidates():
    # ``factor_matmul`` and ``weight_factor_matmul`` spell the same
    # equality alpha-normally: both instances' ``comm_add`` composites
    # key identically, so only the first survives the ``seen`` dedup.
    uni = [
        _BY_NAME["factor_matmul"],
        _BY_NAME["weight_factor_matmul"],
        _BY_NAME["comm_add"],
    ]
    out = pp.composite_candidates(uni, max_cands=50)
    assert [c.label for c in out] == ["factor_matmul+comm_add"]


def test_composite_candidates_are_derivable():
    uni = [
        _BY_NAME[n]
        for n in ("sub_to_add", "double_neg", "comm_add", "id_add")
    ]
    out = pp.composite_candidates(uni, max_cands=10)
    assert out, "the small universe should compose at least once"
    for c in out:
        assert c.strategy == "composite"
        assert "+" in c.label
        res = vf.verify_law(c.lhs, c.rhs, uni)
        assert res.derivable is True


def test_composite_skips_non_instanced_and_caps():
    # mul_square has no bench/generic instance — it is skipped as a
    # source (and still appears as a r2 when applicable).
    uni = [
        _BY_NAME[n]
        for n in ("mul_square", "sub_to_add", "comm_add", "double_neg")
    ]
    out = pp.composite_candidates(uni, max_cands=1)
    assert 1 <= len(out) <= 2
    assert not any(c.label.startswith("mul_square+") for c in out)


# ---------------------------------------------------------------------------
#  verifier — the bench LAW_CASES registry + the real-corpus experiments
# ---------------------------------------------------------------------------


def test_instance_of_from_bench_registry():
    inst = vf.instance_of(_BY_NAME["sub_to_add"])
    assert inst is not None
    lhs, rhs = inst
    assert lhs.op == "sub"
    assert rhs.op == "add" and rhs.args[1].op == "neg"


def test_law_cases_adds_repo_to_syspath(monkeypatch):
    import sys

    from discoverykit import consumer_repo_root

    root = str(consumer_repo_root())
    kept = [p for p in sys.path if p != root]
    monkeypatch.setattr(sys, "path", kept)
    cases = vf._law_cases()
    assert "sub_to_add" in cases
    assert root in sys.path


def test_rediscover_three_categories():
    uni = [
        _BY_NAME[n]
        for n in (
            "sub_to_add",
            "double_neg",
            "id_add",
            "comm_add",
            "silu_expand",
            "silu_fold",
            "mul_square",
        )
    ]
    rows = vf.rediscover(uni, max_nodes=50_000)
    by = {r.name: r for r in rows}
    assert by["sub_to_add"].category == "primitive"
    assert by["silu_fold"].category == "derivable"
    assert by["silu_fold"].relation == "inverse"
    assert "silu_expand" in by["silu_fold"].result.witness_rules
    assert by["mul_square"].category == "no-instance"
    assert by["mul_square"].result is None
    assert by["mul_square"].lhs_repr == ""


def test_run_controls_small_and_empty():
    assert vf.run_controls([_BY_NAME["sub_to_add"]]) == {
        "n": 1,
        "failures": [],
    }
    assert vf.run_controls([]) == {"n": 0, "failures": []}


def test_main_runs_experiments_and_dumps(tmp_path, capsys):
    out = tmp_path / "law.json"
    rc = vf.main(["--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert {
        "rediscovery",
        "proposals",
        "controls",
        "structural",
        "summary",
    } <= set(payload)
    assert payload["controls"]["failures"] == []
    assert all(p["ok"] for p in payload["proposals"])
    counts = payload["summary"]["counts"]
    assert counts["derivable"] > 0 and counts["primitive"] > 0
    text = capsys.readouterr().out
    assert "law rediscovery" in text
    assert "soundness controls" in text


# ---------------------------------------------------------------------------
#  gauntlet / meta_game — the lazily-built real corpus
# ---------------------------------------------------------------------------


def test_default_gauntlet_corpus_builds():
    """The default corpus is the pipeline's real one — built lazily
    so the report path stays torch-free."""
    corpus = ev.default_gauntlet_corpus()
    assert len(corpus.real_terms) > 0
    assert len(corpus.probe) > 0
    assert len(corpus.base_rules) == len(ALL_RULES)
    assert corpus.census_op and corpus.cost_fn is not None


def test_generator_pools_derived_vocab():
    terms = _tiny_terms()
    cts = [CorpusTerm("t", f"m{i}", t) for i, t in enumerate(terms)]
    counts, _ = op_tuple_census(cts)
    pools = mg.generator_pools(
        {k: v for k, v in counts.items()}, terms, "derived"
    )
    assert set(pools) == set(mg.ENUMERATION_ORDER) - {"build"}


# ---------------------------------------------------------------------------
#  intake — the mlopt_torch export spelling (torch-version sensitive)
# ---------------------------------------------------------------------------


def _ops_of(term, acc=None):
    if acc is None:
        acc = []
    if isinstance(term, Op):
        acc.append(term)
        for a in term.args:
            _ops_of(a, acc)
    return acc


def _export_lower_verify(model, feed):
    """The intake's boundary pipeline: export → lower → sink.verify."""
    model = model.eval().double()
    ir, tensors = export_to_ir(model, feed)
    lowered = ir_to_torch_module(ir, param_values=tensors)
    vr = TorchSink().verify(model, lowered, feed)
    return ir, vr


class _SliceFill(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y[:, :4] = 0.0
        return y


def test_fill_tensor_through_slice_view():
    """``y[:, :4] = 0`` — ``fill_.Tensor`` through a ``slice`` view.

    Functionalisation spells a slice write as ``fill_(slice, lifted
    0-dim)``; the mutation threader must scatter the broadcast value
    onto the viewed base, not orphan the write.
    """
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_SliceFill(), feed)
    ops = {t.op for t in _ops_of(ir.root)}
    assert "slice_scatter" in ops and "broadcast_to" in ops
    assert vr.passed, vr
