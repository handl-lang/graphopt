"""Tests for ``mlopt_discovery.model_corpus`` — the per-model corpus.

The module closes the "corpus ≠ arbitrary models" gap: a target
``(model, feed)`` pair is exported through the same
``TorchSource.to_ir`` boundary every corpus term crossed, its
op-tuples/shapes are diffed against the standing corpus, mutants of
its own term are generated through ``workload_gen``'s gated
machinery, and every case — the model first — goes through the same
``_mint_tiers`` → ``profitable`` → ``gen_probe`` intake tail the zoo
rows use.  The tests pin:

* the census diff names tuples the corpus never contained;
* mutants pass ``valid_term``'s gates and differ from the model;
* boards carry the ``gen_probe`` contract (``case[1]`` term,
  ``case[4]`` supported, ``case[5]`` ex record);
* ``model_corpus`` reaches an honest pay/decline verdict row —
  delivery+verification are measured, never assumed.
"""

import math

import pytest
import torch
import torch.nn as nn
from mlopt_core.ir import Op
from mlopt_discovery import model_corpus as mc
from mlopt_discovery.census import CorpusTerm
from mlopt_discovery.impact import _iter_subterms


class _GatedRecurrentPool(nn.Module):
    """A recurrence/attention arrangement absent from the zoo.

    Fixed-unroll gated recurrence (``tanh`` + sigmoid-gated update)
    feeding a single-head additive-attention pool with a bilinear
    tail — composed only of ops the ambient sink lowers, but in a
    shape no registered workload spells.
    """

    def __init__(self, d: int = 16, t: int = 4) -> None:
        super().__init__()
        for n in ("w_in", "w_rec", "w_up", "q_w", "k_w", "v_w", "gate"):
            setattr(self, n, nn.Parameter(torch.randn(d, d) / d**0.5))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        h = torch.zeros(B, D, dtype=x.dtype, device=x.device)
        outs = []
        for i in range(T):
            xi = x[:, i]
            h = torch.tanh(xi @ self.w_in + h @ self.w_rec + self.bias)
            outs.append(h * torch.sigmoid(h @ self.w_up))
        H = torch.stack(outs, dim=1)
        q = H @ self.q_w
        k = H @ self.k_w
        v = H @ self.v_w
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)
        a = torch.softmax(scores, dim=-1)
        ctx = a @ v
        g = torch.sigmoid(ctx @ self.gate)
        y = ctx * g + H
        return (y * y).mean(dim=1)


@pytest.fixture(scope="module")
def model_case():
    """One exported (model, feed, TermCase) triple, built once."""
    torch.manual_seed(0)
    model = _GatedRecurrentPool().eval()
    feed = torch.randn(2, 4, 16, dtype=torch.float64)
    ir, leaves, feeds = mc.export_ir(model, feed)
    tc = mc._term_case("grp", ir, leaves, feeds)
    return model, feed, tc


def _tiny_corpus() -> list[CorpusTerm]:
    """A two-term stand-in corpus — keeps tests off the 40-model export."""
    from mlopt_core.ir import Const, Op, TensorType, Var

    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    t1 = Op.make("add", Op.make("mul", x, y), Const(1))
    t2 = Op.make("matmul", x, y)
    return [
        CorpusTerm("bench", "addmul", t1),
        CorpusTerm("bench", "mm", t2),
    ]


class TestExportAndCoverage:
    def test_export_ir_roundtrip(self, model_case):
        _model, _feed, tc = model_case
        assert tc.term is not None and tc.inputs
        assert tc.feed and all(torch.is_tensor(t) for t in tc.feed)
        assert tc.param_vals
        # the exported root evaluates — the denotation is real
        from mlopt_core.ir import Param, Var
        from mlopt_discovery.proposal import _eval_backend

        leaves = {
            leaf
            for leaf in _iter_subterms(tc.term)
            if isinstance(leaf, (Var, Param))
        }
        env = {
            leaf: torch.randn(
                tuple(leaf.typ.shape), dtype=torch.float64
            )
            for leaf in leaves
        }
        out = _eval_backend().eval_term(tc.term, env)
        assert isinstance(out, torch.Tensor)

    def test_coverage_finds_novel_tuples(self, model_case):
        _, _, tc = model_case
        rep = mc.coverage_report(tc, _tiny_corpus())
        assert rep["n_novel_op_tuples"] > 0
        ops = {t["op"] for t in rep["novel_op_tuples"]}
        # the recurrence's gated update and the attention pool are
        # compositions the two-term corpus cannot contain
        assert "matmul" in ops or "mul" in ops
        # and the model's own census is non-trivial
        assert rep["model_op_tuples"] >= 10

    def test_coverage_empty_when_corpus_contains_model(
        self, model_case
    ):
        _, _, tc = model_case
        corpus = [CorpusTerm("model", tc.name, tc.term)]
        rep = mc.coverage_report(tc, corpus)
        assert rep["n_novel_op_tuples"] == 0
        assert rep["n_novel_shapes"] == 0


class TestGenerateMutants:
    def test_mutants_verify_and_differ(self, model_case):
        _, _, tc = model_case
        muts, stats = mc.generate_mutants(
            tc, _tiny_corpus(), 3, seed=1, max_nodes=160
        )
        assert stats["accepted"] == len(muts)
        assert muts
        for mtc in muts:
            # every accepted mutant is a real TermCase: inputs, feed,
            # params populated, and a different term than the model
            assert mtc.inputs and mtc.feed and mtc.param_vals
            assert mtc.term is not tc.term
            # it evaluated — valid_term's oracle already gated it,
            # but pin the denotation cheaply here too
            assert isinstance(mtc.term, Op)

    def test_mutant_board_contract(self, model_case):
        _, _, tc = model_case
        muts, _ = mc.generate_mutants(
            tc, _tiny_corpus(), 1, seed=2, max_nodes=160
        )
        if not muts:
            pytest.skip("no mutant accepted under this seed")
        case = mc.mutant_board(muts[0], label="gen:test", gen_reps=2)
        assert case is not None
        # the gen_probe contract: term, supported bound, ex record
        assert case[0] == "gen:test"
        assert case[4]
        ex = case[5]
        for key in (
            "ir",
            "leaves",
            "model",
            "input",
            "gen_handlers",
            "sink",
        ):
            assert key in ex
        assert ex["_gen_claims_gated"] is True


class TestModelCorpus:
    def test_probe_false_pipeline(self, model_case, monkeypatch):
        """Census + generation + intake without the timing probe."""
        model, feed, _ = model_case
        rep = mc.model_corpus(
            "grp",
            model,
            feed,
            n_mutants=2,
            seed=3,
            corpus=_tiny_corpus(),
            gen_reps=2,
            probe=False,
        )
        assert rep["coverage"]["n_novel_op_tuples"] > 0
        assert rep["mutants"]["accepted"] >= 0
        # one board per accepted mutant plus the model row
        assert rep["n_boards"] == 1 + rep["mutants"]["accepted"]
        assert rep["rows"] == []

    def test_verdict_classification(self):
        assert mc._verdict({"delivered": False}) == "decline"
        assert (
            mc._verdict({"delivered": True, "verified": False})
            == "verify-fail"
        )
        assert (
            mc._verdict(
                {"delivered": True, "verified": True, "speedup": 1.5}
            )
            == "pay"
        )
        assert (
            mc._verdict(
                {"delivered": True, "verified": True, "speedup": 0.5}
            )
            == "decline"
        )
        assert (
            mc._verdict(
                {"delivered": True, "verified": True, "speedup": None}
            )
            == "decline"
        )

    def test_gen_probe_reaches_verdict(self, model_case):
        """One real ``gen_probe`` row on the model board itself.

        The full claim → deliver → verify → time path runs once on
        the actual model (``n_mutants=0`` keeps it to a single probe);
        the row must carry an honest delivered/verified/speedup
        verdict — pay or decline, never assumed.
        """
        model, feed, _ = model_case
        rep = mc.model_corpus(
            "grp",
            model,
            feed,
            n_mutants=0,
            seed=0,
            corpus=_tiny_corpus(),
            reps=2,
            gen_reps=2,
            probe=True,
        )
        assert len(rep["rows"]) == 1
        row = rep["rows"][0]
        assert row["case"] == "model:grp"
        assert row["delivered"] is True
        assert row["verified"] is True
        assert row["speedup"] is not None
        assert rep["verdicts"][mc._verdict(row)] == 1


class _TinyResidual(nn.Module):
    """A small fp32 module — fast through ``optimize``'s saturation."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return torch.relu(self.lin(x)) + x


def _probe_supported() -> bool:
    """True once the pinned eqsat ``search`` carries ``corpus_probe``.

    graphopt consumes eqsat as a pinned git dep — the hook lands in
    ``Optimizer.optimize``/``search`` upstream first, so these tests
    skip (honestly) until the pin carries it.
    """
    import inspect

    from mlopt_orchestrator import optimize as _opt_mod

    return (
        "corpus_probe" in inspect.signature(_opt_mod.search).parameters
    )


class TestOptimizeWiring:
    """The ``corpus_probe=`` seam — per-model coverage during optimize."""

    def test_probe_runs_and_report_lands_in_stats(self):
        if not _probe_supported():
            pytest.skip("eqsat pin predates the corpus_probe hook")
        from mlopt_orchestrator import Optimizer
        from mlopt_torch.backend import TorchBackend

        torch.manual_seed(0)
        model = _TinyResidual().eval()
        x = torch.randn(2, 8)
        seen = []

        def probe(name, m, feed):
            seen.append((name, m is model, feed is x))
            return {"marker": "probe-ran"}

        _mod, stats = Optimizer(backend=TorchBackend()).optimize(
            model,
            x,
            corpus_probe=probe,
            max_iterations=2,
            verify=False,
        )
        assert seen == [("_TinyResidual", True, True)]
        assert stats["corpus_probe"] == {"marker": "probe-ran"}

    def test_default_path_records_no_probe(self):
        """``corpus_probe`` unset → the historical stats, unchanged."""
        from mlopt_orchestrator import Optimizer
        from mlopt_torch.backend import TorchBackend

        torch.manual_seed(0)
        model = _TinyResidual().eval()
        x = torch.randn(2, 8)
        _mod, stats = Optimizer(backend=TorchBackend()).optimize(
            model, x, max_iterations=2, verify=False
        )
        assert "corpus_probe" not in stats

    def test_corpus_probe_adapter_end_to_end(self):
        """``mc.corpus_probe(...)`` as the hook: real census+board
        intake inside ``optimize``, report under ``stats``."""
        if not _probe_supported():
            pytest.skip("eqsat pin predates the corpus_probe hook")
        from mlopt_orchestrator import Optimizer
        from mlopt_torch.backend import TorchBackend

        torch.manual_seed(0)
        model = _TinyResidual().eval()
        x = torch.randn(2, 8)
        probe = mc.corpus_probe(
            n_mutants=0,
            seed=0,
            corpus=_tiny_corpus(),
            gen_reps=2,
            probe=False,
        )
        _mod, stats = Optimizer(backend=TorchBackend()).optimize(
            model,
            x,
            corpus_probe=probe,
            max_iterations=2,
            verify=True,
        )
        rep = stats["corpus_probe"]
        assert rep["name"] == "_TinyResidual"
        assert "coverage" in rep and "mutants" in rep
        # the adapter matched the model's fp32 dtype — no re-dtype
        assert all(p.dtype == torch.float32 for p in model.parameters())

    def test_model_corpus_is_a_valid_probe_directly(self):
        """The bare ``(name, model, feed)`` signature also fits —
        defaults apply (documented, not exercised end-to-end here)."""
        import inspect

        sig = inspect.signature(mc.model_corpus)
        params = list(sig.parameters)
        assert params[:3] == ["name", "model", "feed"]

    def test_corpus_probe_factory_binds_kwargs_and_dtype(
        self, monkeypatch
    ):
        """The adapter's ``(name, model, feed)`` closure — knob
        binding and dtype matching, without needing the eqsat pin."""
        seen = {}

        def fake_model_corpus(name, model, feed, **kw):
            seen.update(kw)
            seen["call"] = (name, model, feed)
            return {"ok": True}

        monkeypatch.setattr(mc, "model_corpus", fake_model_corpus)
        model = _TinyResidual().eval()  # fp32
        x = torch.randn(2, 8)
        probe = mc.corpus_probe(n_mutants=3, reps=7, probe=False)
        rep = probe("m", model, x)
        assert rep == {"ok": True}
        assert seen["call"] == ("m", model, x)
        assert seen["n_mutants"] == 3 and seen["reps"] == 7
        assert seen["probe"] is False
        # dtype auto-matched to the model — the no-re-dtype guarantee
        assert seen["dtype"] == "fp32"
        # an explicit dtype wins over the auto-match
        mc.corpus_probe(dtype="fp64")("m", model, x)
        assert seen["dtype"] == "fp64"

    def test_model_dtype_helper(self):
        assert mc._model_dtype(_TinyResidual()) == "fp32"
        assert (
            mc._model_dtype(_TinyResidual().to(torch.float64)) == "fp64"
        )
        # parameterless / non-module objects fall back to the
        # intake convention
        assert mc._model_dtype(object()) == "fp64"


class TestCliHelpers:
    def test_resolve_zoo_workload(self):
        pytest.importorskip("torch")
        from mlopt_discovery import zoo as _zoo

        name = _zoo.zoo()[0].name
        got, model, _feed = mc._resolve_workload(f"zoo:{name}")
        assert got == name
        assert isinstance(model, nn.Module)

    def test_resolve_unknown_rejected(self):
        with pytest.raises(ValueError, match="unknown zoo"):
            mc._resolve_workload("zoo:no-such-workload")
        with pytest.raises(ValueError, match="unknown workload kind"):
            mc._resolve_workload("bogus:thing")

    def test_input_spec(self):
        t = mc._input_from_spec("2,4")
        assert t.shape == (2, 4) and t.dtype == torch.float64
        pair = mc._input_from_spec("2,4;3,5")
        assert isinstance(pair, tuple) and len(pair) == 2
