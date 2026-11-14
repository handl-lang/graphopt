"""The public-API freeze — plan 0032 phase 1 contract tests.

``docs/public_api.md`` declares the names a downstream domain (a
board, a player, a backend sink, a referee) may bind; this file is
the mechanical pin.  Every frozen name must resolve at its canonical
path, hold its ``__all__`` seat, and keep its published signature —
a drift in any of them is a contract break, not a refactor.

Kept deliberately torch-free at *collection* time: the substrate and
game rows resolve without torch, and the torch-surface pins import
their modules lazily inside the test bodies (the first torch import
costs a few seconds once, inside the test run — never at collection).
The game-kernel modules it checks must stay substrate-free (the
matching import-linter contract lives in ``pyproject.toml``).
"""

from __future__ import annotations

import importlib
import inspect
import math

import pytest

# ---------------------------------------------------------------------------
#  The frozen surface — module -> the names downstream code may bind
#  (mirrors the tables in docs/public_api.md)
# ---------------------------------------------------------------------------

_PUBLIC_SURFACE: dict[str, tuple[str, ...]] = {
    "mlopt_core": ("API_VERSION",),
    "mlopt_core.ir": (
        "IR",
        "Op",
        "Var",
        "Param",
        "Const",
        "TensorType",
        "op_repr",
        "op_repr_dag",
        "term_to_data",
        "term_from_data",
    ),
    "mlopt_core.egraph": (
        "EGraph",
        "ENode",
        "EClass",
        "UnionFind",
        "Rewrite",
        "Certificate",
        "CertStep",
        "ProofEdge",
        "CertificateVerificationError",
        "cert_to_data",
        "cert_from_data",
        "verify_certificate",
    ),
    "mlopt_core.cost": (
        "dag_cost",
        "count_cost",
        "flops_cost",
        "param_bytes_cost",
        "param_bytes_cost_for",
        "backend_cost",
        "executor_cost_for",
        "fused_cost_for",
        "lowering_aware_cost_for",
        "depth_cost",
        "depth_cost_for",
        "roofline_cost",
        "roofline_cost_for",
        "launch_aware_cost",
        "fusion_regions",
        "fusion_member_key",
        "CostModel",
        "LOWERINGS",
    ),
    "mlopt_core.ports": (
        "Source",
        "Sink",
        "Capabilities",
        "Engine",
        "Composer",
        "Executor",
        "ExecutorSpec",
        "Runner",
        "Meter",
        "Verifier",
        "VerifyResult",
        "CostFn",
        "Criterion",
        "Policy",
        "Strategy",
        "TaskMetric",
        "PerformanceModel",
        "Profiler",
        "OpRegistry",
        "RuleSetLike",
        "RuleLike",
        "RuleSetProvider",
        "Binding",
        "TimingResult",
        "signature_conforms",
    ),
    "mlopt_core.laws": (
        "RuleSet",
        "preset",
        "DEFAULT",
        "FULL",
        "SYMMETRY",
        "ALL_RULES",
        "tags",
        "provenance",
    ),
    "mlopt_core.laws.provenance": (
        "MACHINE_SHIPPED",
        "MACHINE_STORED",
        "MACHINE_LAWS",
        "law_provenance",
    ),
    "mlopt_orchestrator": (
        "Optimizer",
        "search",
        "lower",
        "SearchResult",
        "LowerResult",
        "Backend",
        "discover_alternatives",
        "DEFAULT_RULES",
        "OptimizationResourceError",
        "Monolithic",
        "Compositional",
        "Autotuned",
        "IdentityRunner",
        "ChainedRunner",
        "runner_candidate",
        "Criterion",
        "Criteria",
        "Blend",
        "criteria_cost",
        "peak_bytes_cost",
        "LatencyCriterion",
        "MemoryCriterion",
        "FlopsCriterion",
        "DepthCriterion",
        "CompiledCriterion",
        "PredictedCriterion",
        "delivered_cost_for",
        "structural_key",
    ),
    "gamekit.engine": (
        "Board",
        "Report",
        "Trajectory",
        "run_episode",
        "evaluate",
        "train_policy",
    ),
    "gamekit.players": (
        "PlayerSpec",
        "LinearPolicy",
        "DomainSpec",
        "LearnedPlayer",
        "PlayHist",
        "Bias",
        "StateBroadcast",
        "PlayRates",
        "OpOneHot",
        "ActionFeats",
        "RelFeats",
        "compose_featurizer",
        "register_part",
        "register_legal",
        "register_rank",
        "spec_from_record",
        "PARTS",
        "LEGALS",
        "RANKS",
        "registry",
    ),
    "gamekit.players.registry": (
        "PlayerSpec",
        "BOARDS",
        "EPISODE_BOARDS",
        "names",
        "register_player",
        "spec",
        "specs",
        "for_board",
        "module",
        "make",
    ),
    "gamekit.scoring": (
        "AXES",
        "DIRECTIONS",
        "ScoreGate",
        "ScoringSpec",
        "GateVerdict",
        "ScoreVerdict",
        "score",
    ),
    "mlopt_discovery.verifier": ("verify_law", "LawResult"),
    "mlopt_discovery.play": (
        "DOMAINS",
        "Domain",
        "play",
        "spec_domain",
    ),
    # The torch delivery surface — the narrow user-facing set frozen
    # by the Phase-A decision (docs/public_api.md § Torch delivery):
    # optimize → export → load, plus the runner constructors a
    # ``runner=`` kwarg binds.  Everything else in ``mlopt_torch``
    # (IRModule, executors, adapters, autotune builders, calibrate,
    # the model zoo) stays public-but-unfrozen.
    "mlopt_torch": ("TorchBackend", "TorchCompileRunner"),
    "mlopt_torch.backend": ("TorchBackend",),
    "mlopt_torch.export": (
        "ARTIFACT_SCHEMA",
        "ExportError",
        "export_optimized",
        "load_optimized",
        "save_optimized",
        "save_optimized_weights",
    ),
    "mlopt_torch.runners": ("TorchCompileRunner",),
    "mlopt_cuda": ("CudaGraphRunner",),
}




# ---------------------------------------------------------------------------
#  The mechanical pin — every frozen name resolves at its canonical path
# ---------------------------------------------------------------------------


class TestFrozenSurface:
    """The whole table — the domain-seat half of the pin that lives in
    ``eqsat:tests/test_ported_public_api.py`` for the substrate rows.
    """

    @pytest.mark.parametrize("mod_name", sorted(_PUBLIC_SURFACE))
    def test_module_imports(self, mod_name):
        mod = importlib.import_module(mod_name)
        assert mod is not None

    @pytest.mark.parametrize(
        "mod_name,name",
        [
            (m, n)
            for m, names in _PUBLIC_SURFACE.items()
            for n in names
        ],
    )
    def test_frozen_name_resolves(self, mod_name, name):
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, name), (
            f"CONTRACT BREAK: {mod_name}.{name} — a frozen public name "
            "no longer resolves (docs/public_api.md)"
        )

    @pytest.mark.parametrize("mod_name", sorted(_PUBLIC_SURFACE))
    def test_frozen_names_hold_all_seats(self, mod_name):
        """Where a module declares ``__all__``, frozen names sit in it."""
        mod = importlib.import_module(mod_name)
        declared = getattr(mod, "__all__", None)
        if declared is None:
            pytest.skip(f"{mod_name} declares no __all__")
        missing = set(_PUBLIC_SURFACE[mod_name]) - set(declared)
        assert not missing, (
            f"CONTRACT BREAK: {mod_name} frozen names missing from "
            f"__all__: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
#  The version constant
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  Signature pins — the contract's shapes, not just its names
# ---------------------------------------------------------------------------


class TestSignaturePins:
    """The published call shapes — a kwarg lost is a contract break."""

    def test_egraph_run_shape(self):
        from mlopt_core.egraph import EGraph

        params = inspect.signature(EGraph.run).parameters
        for kw in (
            "rules",
            "root_eid",
            "max_iterations",
            "max_nodes",
            "rule_budgets",
            "stop",
            "patience",
            "cost_fn",
            "policy",
        ):
            assert kw in params, f"EGraph.run lost {kw!r}"

    def test_egraph_protocol_methods(self):
        from mlopt_core.egraph import EGraph

        for m in (
            "add_term",
            "find",
            "rebuild",
            "extract_best",
            "run",
            "certificate",
        ):
            assert callable(getattr(EGraph, m)), f"EGraph lost {m}"

    def test_search_engine_kwarg(self):
        """``engine=`` — engines are passed, never auto-detected."""
        from mlopt_orchestrator import search

        params = inspect.signature(search).parameters
        assert "engine" in params
        assert params["engine"].default is None
        assert "policy" in params
        assert "source" in params

    def test_verify_certificate_shape(self):
        from mlopt_core.egraph import verify_certificate

        params = inspect.signature(verify_certificate).parameters
        assert list(params)[:2] == ["src_term", "cert"]
        assert "strict" in params

    def test_dag_cost_shape(self):
        from mlopt_core.cost import dag_cost

        params = inspect.signature(dag_cost).parameters
        assert list(params)[:2] == ["term", "cost_fn"]
        assert "memo" in params

    def test_verify_law_shape(self):
        from mlopt_discovery.verifier import verify_law

        params = inspect.signature(verify_law).parameters
        assert list(params)[:3] == ["lhs", "rhs", "rules"]
        for kw in ("max_iterations", "max_nodes", "rule_budgets"):
            assert kw in params, f"verify_law lost {kw!r}"

    def test_registry_make_shape(self):
        from gamekit.players import registry

        params = inspect.signature(registry.make).parameters
        assert list(params)[:2] == ["name", "domain"]
        assert params["seed"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["ctx"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_law_provenance_shape(self):
        from mlopt_core.laws.provenance import law_provenance

        assert list(inspect.signature(law_provenance).parameters) == [
            "rule"
        ]

    # -- the torch delivery surface (Phase-A freeze) -----------------

    def test_optimizer_optimize_shape(self):
        """``Optimizer.optimize(model, x, *, strategy=None, **kw)`` —
        the entry point every delivery example binds."""
        from mlopt_orchestrator import Optimizer

        params = inspect.signature(Optimizer.optimize).parameters
        assert list(params)[:3] == ["self", "model", "x"]
        assert params["strategy"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["strategy"].default is None
        assert any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        ), "optimize must keep forwarding pipeline kwargs (**kw)"

    def test_torch_backend_factory_shape(self):
        """``TorchBackend()`` assembles the port value — the port
        slots plus ``device_policy`` are all keyword-only."""
        from mlopt_torch import TorchBackend

        params = inspect.signature(TorchBackend).parameters
        assert set(params) == {
            "ops",
            "source",
            "sink",
            "composer",
            "meter",
            "device_policy",
        }
        assert all(
            p.kind is inspect.Parameter.KEYWORD_ONLY
            for p in params.values()
        )
        assert all(p.default is None for p in params.values())

    def test_export_optimized_shape(self):
        from mlopt_torch.export import export_optimized

        params = inspect.signature(export_optimized).parameters
        assert list(params)[:3] == [
            "model",
            "optimized_model",
            "path",
        ]
        for kw in (
            "fmt",
            "example_input",
            "example_kwargs",
            "dynamic_shapes",
            "verify",
            "stats",
        ):
            assert params[kw].kind is inspect.Parameter.KEYWORD_ONLY, (
                f"export_optimized lost keyword-only {kw!r}"
            )
        assert params["fmt"].default == "module"
        assert params["verify"].default is True

    def test_load_optimized_shape(self):
        from mlopt_torch.export import load_optimized

        assert list(inspect.signature(load_optimized).parameters) == [
            "path"
        ]

    def test_save_optimized_weights_shape(self):
        from mlopt_torch.export import save_optimized_weights

        assert list(
            inspect.signature(save_optimized_weights).parameters
        ) == ["optimized_module", "path"]

    def test_runner_constructor_shapes(self):
        """The ``runner=`` values: zero-arg captures, kwargs compile,
        an iterable chain — the three call shapes docs promise."""
        from mlopt_cuda import CudaGraphRunner
        from mlopt_orchestrator import ChainedRunner, IdentityRunner
        from mlopt_torch import TorchCompileRunner

        cuda_params = inspect.signature(CudaGraphRunner).parameters
        assert list(cuda_params) == ["multi_device"]
        assert (
            cuda_params["multi_device"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        assert cuda_params["multi_device"].default is False
        assert not inspect.signature(IdentityRunner).parameters
        chained = inspect.signature(ChainedRunner).parameters
        assert list(chained) == ["runners"]
        compile_params = inspect.signature(TorchCompileRunner).parameters
        assert list(compile_params) == ["compile_kwargs"]
        assert (
            compile_params["compile_kwargs"].kind
            is inspect.Parameter.VAR_KEYWORD
        )


# ---------------------------------------------------------------------------
#  Protocols — the structural contracts check at runtime
# ---------------------------------------------------------------------------


class TestProtocols:
    """``Board``/``Sink``/``Engine`` stay runtime-checkable Protocols."""

    def test_board_runtime_checkable(self):
        from gamekit.engine import Board

        class _B:
            def observe(self):
                return None

            def step(self, action):
                return None, None

        assert isinstance(_B(), Board)

    def test_sink_port_members(self):
        from mlopt_core.ports import Capabilities, Sink

        for m in ("lower", "verify", "executors"):
            assert hasattr(Sink, m), f"Sink port lost {m}"
        for m in ("supported_ops", "ops"):
            assert hasattr(Capabilities, m), f"Capabilities lost {m}"

    def test_engine_port_members(self):
        from mlopt_core.ports import Engine

        for m in (
            "add_term",
            "find",
            "rebuild",
            "run",
            "extract_best",
        ):
            assert hasattr(Engine, m), f"Engine port lost {m}"


# ---------------------------------------------------------------------------
#  The scoring spec — the referee's value function, as data
# ---------------------------------------------------------------------------


class TestScoringSpec:
    """``score(spec, measured)`` — the spec/measurement seam."""

    def test_axes_vocabulary(self):
        from gamekit.scoring import AXES, DIRECTIONS

        assert AXES == ("speed", "memory", "launch", "params")
        assert DIRECTIONS == ("at_most", "at_least")

    def test_gate_pass_and_margin(self):
        from gamekit.scoring import ScoreGate

        g = ScoreGate("speed", 0.95)
        assert g.passed(0.9) and not g.passed(1.0)
        assert g.margin(0.9) == pytest.approx(0.05)
        floor = ScoreGate("win", 0.5, direction="at_least")
        assert floor.passed(0.6) and not floor.passed(0.4)
        assert floor.margin(0.6) == pytest.approx(0.1)

    def test_gate_ratio_normalises_to_the_bar(self):
        from gamekit.scoring import ScoreGate

        g = ScoreGate("memory", 4.0)
        assert g.ratio(2.0) == pytest.approx(0.5)
        floor = ScoreGate("win", 0.5, direction="at_least")
        assert floor.ratio(1.0) == pytest.approx(0.5)
        assert math.isinf(floor.ratio(0.0))
        zero = ScoreGate("launch", 0.0)
        assert zero.ratio(0.0) == 0.0
        assert math.isinf(zero.ratio(1.0))

    def test_gate_validation(self):
        from gamekit.scoring import ScoreGate

        with pytest.raises(ValueError, match="non-empty"):
            ScoreGate("", 1.0)
        with pytest.raises(ValueError, match="finite"):
            ScoreGate("speed", math.inf)
        with pytest.raises(ValueError, match="direction"):
            ScoreGate("speed", 1.0, direction="sideways")

    def test_score_pass_and_value(self):
        from gamekit.scoring import ScoreGate, ScoringSpec, score

        spec = ScoringSpec(gates=(ScoreGate("speed", 2.0),))
        v = score(spec, {"speed": 1.0})
        assert v.passed and v.value == pytest.approx(0.5)
        v = score(spec, {"speed": 4.0})
        assert not v.passed and v.value == pytest.approx(2.0)

    def test_score_blends_weighted_axes(self):
        from gamekit.scoring import ScoreGate, ScoringSpec, score

        spec = ScoringSpec(
            gates=(ScoreGate("speed", 2.0),),
            weights={"speed": 1.0, "memory": 0.5},
            references={"memory": 8.0},
        )
        v = score(spec, {"speed": 1.0, "memory": 4.0})
        assert v.passed
        # 1.0*(1/2) + 0.5*(4/8) = 0.75
        assert v.value == pytest.approx(0.75)

    def test_score_missing_measurement_is_honest(self):
        from gamekit.scoring import ScoreGate, ScoringSpec, score

        spec = ScoringSpec(gates=(ScoreGate("speed", 2.0),))
        v = score(spec, {})
        assert not v.passed
        assert v.missing == ("speed",)
        assert math.isnan(v.value)
        g = v.gates[0]
        assert not g.passed and math.isnan(g.value)

    def test_score_at_least_axis_inverts_the_ratio(self):
        from gamekit.scoring import ScoreGate, ScoringSpec, score

        spec = ScoringSpec(
            gates=(ScoreGate("win", 0.5, direction="at_least"),),
        )
        v = score(spec, {"win": 1.0})
        assert v.passed and v.value == pytest.approx(0.5)
        # a dead-floor measurement prices at +inf, not zero
        v = score(spec, {"win": 0.0})
        assert not v.passed and math.isinf(v.value)

    def test_score_unpriced_when_weighted_axis_missing(self):
        from gamekit.scoring import ScoringSpec, score

        spec = ScoringSpec(
            gates=(),
            weights={"memory": 1.0},
            references={"memory": 8.0},
        )
        v = score(spec, {})
        assert v.passed  # no gates to fail
        assert math.isnan(v.value)
        assert "memory" in v.missing

    def test_spec_rejects_unanchored_weight(self):
        from gamekit.scoring import ScoreGate, ScoringSpec

        with pytest.raises(ValueError, match="neither gated"):
            ScoringSpec(
                gates=(ScoreGate("speed", 1.0),),
                weights={"bogus": 1.0},
            )
        with pytest.raises(ValueError, match="finite"):
            ScoringSpec(gates=(), weights={"speed": math.nan})
        with pytest.raises(ValueError, match="non-negative"):
            ScoringSpec(
                gates=(ScoreGate("speed", 1.0),),
                weights={"speed": -1.0},
            )
        with pytest.raises(ValueError, match="positive"):
            ScoringSpec(gates=(), references={"memory": 0.0})

    def test_spec_normalises_gates_tuple(self):
        from gamekit.scoring import ScoreGate, ScoringSpec

        spec = ScoringSpec(gates=[ScoreGate("speed", 1.0)])
        assert isinstance(spec.gates, tuple)
        assert spec.reference("speed") == 1.0
        assert spec.reference("unknown") is None


# ---------------------------------------------------------------------------
#  The seam guards — things that must NOT become public-in-fact
# ---------------------------------------------------------------------------


class TestBoundaryNotes:
    """The documented internal/unfrozen edges stay where they are."""

    def test_mlopt_facade_still_gone(self):
        """Plan 0008: there is no ``mlopt`` façade."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("mlopt")

    def test_provenance_registry_classifies(self):
        """The registry contract — total over names, two classes."""
        from mlopt_core.laws.provenance import (
            MACHINE_LAWS,
            MACHINE_SHIPPED,
            MACHINE_STORED,
            law_provenance,
        )

        assert MACHINE_LAWS == MACHINE_SHIPPED | MACHINE_STORED
        assert law_provenance("conv_bn_fold") == "machine"
        assert law_provenance("select_mul") == "human"
        assert law_provenance("never-heard-of-it") == "human"

    def test_engine_report_floor(self):
        """``Report`` — the report floor the game contract names."""
        from gamekit.engine import Report

        r = Report(action="x", reward=1.5)
        assert r.terminal is False and r.applied is True
        assert r.reward == 1.5


# ---------------------------------------------------------------------------
#  The lane-A compat shims — the old discovery paths bind the same objects
# ---------------------------------------------------------------------------


class TestGamekitCompatShims:
    """``mlopt_discovery.{engine,scoring,advantage,players,lawdata.game}``
    re-export the gamekit modules — same objects, one home (plan 0032
    lane A)."""

    def test_engine_shim(self):
        import gamekit.engine
        import mlopt_discovery.engine as shim

        assert shim.Board is gamekit.engine.Board
        assert shim.run_episode is gamekit.engine.run_episode
        # reachable-but-unlisted stays reachable
        assert shim.imitate_policy is gamekit.engine.imitate_policy
        assert "imitate_policy" not in shim.__all__

    def test_scoring_shim(self):
        import gamekit.scoring
        import mlopt_discovery.scoring as shim

        assert shim.score is gamekit.scoring.score
        assert shim.ScoringSpec is gamekit.scoring.ScoringSpec

    def test_advantage_shim(self):
        import gamekit.advantage
        import mlopt_discovery.advantage as shim

        assert shim.AdvantagePlayer is gamekit.advantage.AdvantagePlayer

    def test_players_shim(self):
        import gamekit.players
        import mlopt_discovery.players as shim

        assert shim.LinearPolicy is gamekit.players.LinearPolicy
        assert shim._dot is gamekit.players._dot

    def test_players_submodules_are_the_same_module_object(self):
        """``mlopt_discovery.players.{registry,_base}`` alias — one
        registry ``_INDEX``, never a forked copy."""
        import gamekit.players._base
        import gamekit.players.registry
        import mlopt_discovery.players._base as pbase
        import mlopt_discovery.players.registry as preg

        assert preg is gamekit.players.registry
        assert pbase is gamekit.players._base

    def test_lawdata_game_alias(self):
        """``mlopt_discovery.lawdata.game`` aliases ``gamekit.gamedata``."""
        import gamekit.gamedata
        import gamekit.gamedata as game
        import gamekit.gamedata.features as features
        import gamekit.gamedata.weights as weights
        import mlopt_discovery.lawdata as data

        assert game is gamekit.gamedata
        assert data.game is gamekit.gamedata
        assert features is gamekit.gamedata.features
        assert weights is gamekit.gamedata.weights
        assert (
            data.ARENA_MOVE_ORDER
            is gamekit.gamedata.moves.ARENA_MOVE_ORDER
        )
