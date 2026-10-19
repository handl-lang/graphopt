"""mlopt-orchestrator — the backend-neutral optimizer orchestrator.

The public seam (plan 0006/0007): :class:`Optimizer` — the configured
entry point over an explicit
:class:`~mlopt_core.pipeline.Backend` (or explicit
``source``/``sink``/``composer``/``meter`` ports — there is no
default backend) — the phase verbs :func:`search` / :func:`lower`
and their result objects (:class:`SearchResult` /
:class:`LowerResult`), and the
:class:`~mlopt_core.ports.Strategy` implementations
:class:`Monolithic` / :class:`Compositional` / :class:`Autotuned`.

This package imports **no backend**: no torch, no
``mlopt_torch``, no ``mlopt_cuda``.  Backend specifics arrive
through the ports the caller passes — e.g.
``Optimizer(backend=TorchBackend())`` with
``mlopt_torch.backend.TorchBackend``.  The delivery runners live
with their backends (:class:`mlopt_torch.runners.TorchCompileRunner`,
:class:`mlopt_cuda.CudaGraphRunner`); the neutral
:class:`IdentityRunner` / :class:`ChainedRunner` /
:func:`runner_candidate` live in :mod:`mlopt_orchestrator.runners`.
"""

from mlopt_core.pipeline import Backend, LowerResult, SearchResult
from mlopt_core.ports import Criterion, Runner

from mlopt_orchestrator.carriers import (
    CarrierMachinery,
    get_carriers,
    register_carriers,
)
from mlopt_orchestrator.criteria import (
    Blend,
    CompiledCriterion,
    Criteria,
    DepthCriterion,
    FlopsCriterion,
    LatencyCriterion,
    MemoryCriterion,
    PredictedCriterion,
    criteria_cost,
    peak_bytes_cost,
)
from mlopt_orchestrator.diagram import (
    DEFAULT_MOVES,
    ContractionSearch,
    DEdge,
    DEnd,
    Diagram,
    DiagramMove,
    DNode,
    FactorShared,
    MergeProjs,
    ReorderCompose,
    SplitLeaf,
    diagram_of_graph,
    lift_diagram,
    optimize_diagram,
)
from mlopt_orchestrator.morphisms import (
    DEFAULT_MORPHISM_LAWS,
    BlockSig,
    CrossBlockCSE,
    InputSig,
    KVLatentShare,
    MorphismGraph,
    MorphismLaw,
    MorphismMatch,
    MorphismSearch,
    NormSig,
    ReifySpec,
    WeightRef,
    Wire,
    block_signature,
    lift_graph,
    optimize_morphisms,
)
from mlopt_orchestrator.optimize import (
    Autotuned,
    Compositional,
    Monolithic,
    OptimizationResourceError,
    Optimizer,
    delivered_cost_for,
    discover_alternatives,
    lower,
    search,
    structural_key,
)
from mlopt_orchestrator.runners import (
    ChainedRunner,
    IdentityRunner,
    runner_candidate,
)

__all__ = [
    "DEFAULT_MORPHISM_LAWS",
    "DEFAULT_MOVES",
    "DEFAULT_RULES",
    "Autotuned",
    "Backend",
    "Blend",
    "BlockSig",
    "CarrierMachinery",
    "ChainedRunner",
    "CompiledCriterion",
    "Compositional",
    "ContractionSearch",
    "Criteria",
    "Criterion",
    "CrossBlockCSE",
    "DEdge",
    "DEnd",
    "DNode",
    "DepthCriterion",
    "Diagram",
    "DiagramMove",
    "FactorShared",
    "FlopsCriterion",
    "IdentityRunner",
    "InputSig",
    "KVLatentShare",
    "LatencyCriterion",
    "LowerResult",
    "MemoryCriterion",
    "MergeProjs",
    "Monolithic",
    "MorphismGraph",
    "MorphismLaw",
    "MorphismMatch",
    "MorphismSearch",
    "NormSig",
    "OptimizationResourceError",
    "Optimizer",
    "PredictedCriterion",
    "ReifySpec",
    "ReorderCompose",
    "Runner",
    "SearchResult",
    "SplitLeaf",
    "WeightRef",
    "Wire",
    "block_signature",
    "criteria_cost",
    "default_rules",
    "delivered_cost_for",
    "diagram_of_graph",
    "discover_alternatives",
    "get_carriers",
    "lift_diagram",
    "lift_graph",
    "lower",
    "optimize_diagram",
    "optimize_morphisms",
    "peak_bytes_cost",
    "register_carriers",
    "runner_candidate",
    "search",
    "structural_key",
]


def __getattr__(name: str):
    """Lazily resolve the backend-coupled surfaces.

    ``DEFAULT_RULES`` composes ``mlopt_core.laws.DEFAULT`` with the
    carrier-package ``CARRIERS`` preset on first access, and
    ``default_rules`` is its accessor — both keep
    ``import mlopt_orchestrator`` free of the (torch-coupled)
    carriers package.
    """
    if name in ("DEFAULT_RULES", "default_rules"):
        from mlopt_orchestrator import optimize

        value = optimize.default_rules()
        resolved = {
            "DEFAULT_RULES": value,
            "default_rules": optimize.default_rules,
        }
        globals().update(resolved)
        return resolved[name]
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
