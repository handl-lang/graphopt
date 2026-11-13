"""``play`` — the optimization game as a tool.

One registry of *domains* (a board family plus its player
vocabulary), one entrypoint: pick a domain, pick arms, get the
paired-boards table.  Adding a domain is data — a case generator, a
``case -> Board`` factory and a player *spec* (the feature schema,
the featurizer's composable parts and the legal-move enumerator) —
no driver code and no player subclass:

.. code-block:: text

    SPEC = DomainSpec.build(
        features=FEATURES,          # the learned arm's schema
        legal=legal_actions,        # state -> tuple[action, ...]
        parts=(Bias(), StateBroadcast(...), ...),  # the feature row
        weights=PLAYER_WEIGHTS,     # the baked prior (optional)
    )
    DOMAINS[name] = spec_domain(
        cases=gen_cases,            # seed, n -> [case, ...]
        board_of=...,               # case -> Board
        spec=SPEC,
        scripted=...,               # optional fixed-playbook arm
    )

Shipped domains:

* ``meta`` — the meta-arena: fire/saturate/declare/handle/extract
  over the mined-spec corpus (:func:`meta_player.gen_cases`);
* ``joint`` — training graphs: a forward term plus every derived
  gradient under one ``joint`` root, played on the same meta-arena
  board (the reverse handler's programs are ordinary terms);
* ``search`` — the core :class:`~mlopt_core.search_env.SearchEnv`:
  one rule per step under a horizon/patience bound, adapted to the
  :class:`~mlopt_discovery.engine.Board` contract;
* ``torch`` — real ``nn.Module``s lifted through
  ``mlopt_torch.adapters.TorchSource`` and played under the real
  sink's ``supported_ops`` bound; the ``--deliver`` flag lowers
  the winning extraction to a verified runnable module;
* ``open`` — a page of enumerated law candidates, one oracle
  evaluation per move (:mod:`open_game`);
* ``player`` — the two-tier meta-game (:mod:`player_arena`): the
  state is an object-level *player spec record* under rewrite, and
  ``evaluate`` runs a nested train+held-out-eval on the ``meta`` or
  ``open`` board — the player itself is the term being rewritten;
* ``bilinear`` — the ``<2,2,2>`` matmul tensor's decomposition
  space (:mod:`bilinear_game`): cut/pick/flip/drop over a
  rank-decomposition, cost = multiplication count — the Strassen
  rediscovery spike (Q2).

Downstream domain repos add their own: importing the domain module
registers its :class:`Domain` into ``DOMAINS`` (kernelopt's
``mlopt_kernel.play`` seats ``kernel`` — plan 0032 phase 4).

Run it::

    python -m mlopt_discovery.play --domain meta --train-cases 40 \
        --eval-cases 10
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from typing import Any

from gamekit import engine
from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    LinearPolicy,
    StateBroadcast,
    register_legal,
    register_part,
)
from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_core.laws import DEFAULT as DEFAULT_SEARCH_RULES
from mlopt_core.search_env import SearchEnv
from mlopt_core.typing import (
    register_shape_rule as _register_shape_rule,
)

from . import bilinear_game as bg
from . import fragment_game as fg
from . import lawdata, training
from . import meta_arena as ma
from . import meta_player as mp
from . import open_game as og
from . import player_arena as pa
from .meta_player import _bucket

__all__ = [
    "DOMAINS",
    "Domain",
    "SearchBoard",
    "play",
    "spec_domain",
]


@dataclass(frozen=True)
class Domain:
    """One board family's registration — data, not code.

    A domain binds a *cases* generator, a ``case -> Board`` factory
    and the learned player's vocabulary: its legal-move enumerator
    (``legal``), featurizer and feature schema (``features``).  A
    domain that ships a :class:`~mlopt_discovery.players.DomainSpec`
    builds its record with :func:`spec_domain`, which fills those
    three from the spec and records it on ``spec``; a domain may set
    them directly instead.  ``scripted`` and ``deliver`` are extras.
    """

    cases: Any  # (seed, n) -> [case, ...]
    board_of: Any  # case -> Board
    legal: Any = None  # state -> tuple[action, ...]
    featurizer: Any = None  # (state, action, hist) -> dict
    features: Any = ()  # the learned arm's schema
    scripted: Any = None  # () -> player, or None
    deliver: Any = None  # (board, case, traj) -> artifact | None
    spec: DomainSpec | None = None  # the player spec, when one is bound


def spec_domain(
    cases: Any,
    board_of: Any,
    spec: DomainSpec,
    *,
    scripted: Any = None,
    deliver: Any = None,
) -> Domain:
    """Build a :class:`Domain` whose player fields come from *spec*."""
    return Domain(
        cases=cases,
        board_of=board_of,
        legal=spec.legal,
        featurizer=spec.featurize,
        features=spec.features,
        scripted=scripted,
        deliver=deliver,
        spec=spec,
    )


def play(
    domain: str,
    case: Any,
    player: Any,
    *,
    budget: int = 24,
) -> engine.Trajectory:
    """Play one episode of *domain*/*case* under *player*."""
    d = DOMAINS[domain]
    return engine.run_episode(d.board_of(case), player, budget)


def deliver(
    domain: str,
    case: Any,
    player: Any,
    *,
    budget: int = 24,
) -> dict:
    """Play an episode on *case*; return the lowered artifact.

    The product loop: the board's winning extraction resolved by
    ``MetaArena.deliverable`` (handled ops instantiate their
    kernels, unhandled abbreviations spell back), lowered through
    the domain's sink, verified against the source model.  Domains
    without a ``deliver`` hook raise — the game score needs no
    artifact; delivery does.
    """
    d = DOMAINS[domain]
    if d.deliver is None:
        raise ValueError(f"domain {domain!r} has no deliver hook")
    board = d.board_of(case)
    traj = engine.run_episode(board, player, budget)
    return d.deliver(board, case, traj)


# ---------------------------------------------------------------------------
#  Domain: meta — the mixed board
# ---------------------------------------------------------------------------


def _meta_board(case: Any) -> ma.MetaArena:
    """Probe-arity case tuple -> MetaArena (count_cost)."""
    return mp._board(case, count_cost, 20_000)


# ---------------------------------------------------------------------------
#  Domain: joint — forward + gradients on one board
# ---------------------------------------------------------------------------


def _joint_cases(seed: int, n: int) -> list:
    """Joint programs over shape-varied forward archetypes.

    Each case joints a forward (silu / softmax-spine / linear+tanh
    at a random width) with every derived gradient under one
    ``joint`` root — the reverse handler's programs are ordinary
    terms, so the meta-arena's move set applies unchanged.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d, b = 4 * rng.choice((1, 2)), 4 * rng.choice((1, 2))
        x = Var("x", TensorType((b, d)))
        w = Var("w", TensorType((d, d)))
        archetype = rng.randrange(4)
        if archetype == 0:
            fwd = Op.make("mul", x, Op.make("sigmoid", x))
        elif archetype == 1:
            fwd = Op.make("softmax", x, dim=-1)
        elif archetype == 2:
            fwd = Op.make(
                "div",
                Op.make("exp", x),
                Op.make("sum", Op.make("exp", x), dim=1, keepdim=True),
            )
        else:
            fwd = Op.make("tanh", Op.make("matmul", x, w))
        cot = training._cotangent_for(fwd)
        grads = training.backward(fwd, cotangent=cot)
        joint = Op.make(
            "joint",
            fwd,
            *[grads[k] for k in sorted(grads)],
            validate=False,
        )
        out.append((f"joint{i}:{archetype}", joint))
    return out


def _joint_board(case: Any) -> ma.MetaArena:
    """Board a joint program under the meta-arena's moves."""
    return ma.MetaArena(case[1], cost_fn=count_cost)


# ---------------------------------------------------------------------------
#  Domain: torch — real modules under the sink's bound
# ---------------------------------------------------------------------------


def _torch_modules() -> dict:
    """Return the torch domain's module zoo (``name -> d -> nn.Module``).

    Data, not code-paths: domain specs select builders by name
    (``corpus.modules``) instead of naming Python.  Torch imports
    lazily — the registry itself stays torch-free.
    """
    import torch
    import torch.nn as nn

    class _SpelledSiLU(nn.Module):
        """x·sigmoid(x) spelled out — the claim move's live site."""

        def forward(self, x):
            """Spell ``x·sigmoid(x)`` — no fused op named."""
            return x * torch.sigmoid(x)

    return {
        "spelled_silu": lambda d: _SpelledSiLU(),
        "mlp": lambda d: nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
        ),
        "relu_lin": lambda d: nn.Sequential(nn.Linear(d, d), nn.ReLU()),
        "lin": lambda d: nn.Linear(d, d),
        "softsign_mlp": lambda d: nn.Sequential(
            nn.Linear(d, d), nn.Softsign()
        ),
    }


def _torch_dtype(name: Any) -> Any:
    """Resolve a spec ``hardware.dtype`` spelling to a torch dtype."""
    import torch

    table = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name is None:
        return None
    if name not in table:
        raise ValueError(
            f"hardware.dtype: expected one of {sorted(table)!r}, "
            f"got {name!r}"
        )
    return table[name]


def _torch_cases(
    seed: int,
    n: int,
    *,
    modules: Any = None,
    widths: Any = None,
    batch: int = 4,
    device: Any = None,
    dtype: Any = None,
) -> list:
    """Small real ``nn.Module``s lifted to IR — the product board.

    Each case is a torch module exported through
    ``mlopt_torch.adapters.TorchSource`` and played under the real
    sink's ``supported_ops`` bound — kernel claims price at sink
    costs, not ambient vocabulary.  Modules that fail to export are
    skipped honestly.  Torch is imported lazily — the registry
    itself stays torch-free.

    *modules*/*widths*/*batch*/*device*/*dtype* are the domain-spec
    seams: a spec names zoo builders and shape/hardware bounds as
    data; the built-in ``torch`` domain passes none and gets the
    historical stream.
    """
    import torch
    from mlopt_torch.adapters import TorchSink, TorchSource

    zoo = _torch_modules()
    names = list(zoo) if modules is None else list(modules)
    for name in names:
        if name not in zoo:
            raise ValueError(
                f"corpus.modules: unknown torch builder {name!r} "
                f"(known: {sorted(zoo)!r})"
            )
    widths = [4, 8, 16] if widths is None else list(widths)
    if not widths or any(
        not isinstance(w, int) or w <= 0 for w in widths
    ):
        raise ValueError(
            f"shapes.widths: expected positive ints, got {widths!r}"
        )
    dt = _torch_dtype(dtype)
    supported = TorchSink().supported_ops
    src = TorchSource()
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # draw order matches the historical stream: width, then builder
        d = widths[rng.randrange(len(widths))]
        name = names[rng.randrange(len(names))]
        torch.manual_seed(seed * 10_007 + i)
        model = zoo[name](d).eval()
        if device is not None and dt is not None:
            model = model.to(device=device, dtype=dt)
        elif device is not None:
            model = model.to(device)
        elif dt is not None:
            model = model.to(dtype=dt)
        example = torch.randn(
            batch,
            d,
            device=device,
            dtype=dt,
        )
        ir, leaves = src.to_ir(model, example)
        out.append(
            (
                f"torch{i}:{name}",
                ir.root,
                None,
                (),
                supported,
                {
                    "ir": ir,
                    "leaves": leaves,
                    "input": example,
                    "model": model,
                },
            )
        )
    return out


def _torch_board(case: Any) -> ma.MetaArena:
    """Board a torch-lifted program under the real sink bound."""
    rules = case[2] if len(case) > 2 else None
    return ma.MetaArena(
        case[1],
        rules,
        cost_fn=count_cost,
        supported=case[4] if len(case) > 4 else None,
    )


def _is_plan_tape(module: Any) -> bool:
    """Return True when *module* evaluates through a plan tape.

    The two shapes ``CudaGraphRunner`` knows how to capture: an
    ``IRModule`` whose root flattened to an ``_EvalPlan`` (the
    ``_root_plan`` slot tape — the ~350 µs host-dispatch loss of
    ``docs/reports/t034_module_costing.md``), or a batched executor
    exposing ``capture_cuda_graph``.
    """
    if hasattr(module, "capture_cuda_graph"):
        return True
    try:
        from mlopt_torch.torch_bridge import IRModule
    except ImportError:  # pragma: no cover — torch-free env
        return False
    return isinstance(module, IRModule) and (
        getattr(module, "_root_plan", None) is not None
    )


def _max_abs_delta(
    a: Any, b: Any
) -> float:  # pragma: no cover — CUDA-only caller
    """Max |a - b| over tensors or (nested) tuples of them; NaN else."""
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return float("nan")
        return float((a - b).abs().max().item())
    if (
        isinstance(a, (tuple, list))
        and isinstance(b, (tuple, list))
        and len(a) == len(b)
    ):
        vals = [
            v
            for v in (
                _max_abs_delta(x, y) for x, y in zip(a, b, strict=True)
            )
            if v == v
        ]
        return max(vals) if vals else float("nan")
    return float("nan")


def _deliver_runner_arm(
    module: Any, ex: dict, *, reps: int = 50
) -> dict:
    """Attempt the ``CudaGraphRunner`` arm on the delivered module.

    ``docs/reports/t034_module_costing.md`` measured the assembled
    module's loss as ~350 µs of tape-interpreter host dispatch per
    call; the shipped ``CudaGraphRunner`` replays the *same*
    verified forward as one CUDA graph.  This arm runs it on the
    deliver path: capture is attempted only when every example input
    is already a CUDA tensor (the case's device is the deliver
    contract — a CPU case is never silently promoted), only after
    ``verify`` passed, and every failure degrades to the eager
    module plus a ``runner_note`` — capture never fails a deliver.

    Additive keys: ``runner``/``cuda_graph`` record the attempt,
    ``plan_tape`` whether the module is the dispatch-bound shape,
    ``deliver_us``/``graph_us`` the honest eager-vs-replay medians
    (the replay's per-call static-input copies are *inside* the
    timed ``wrapped(*xs)`` — ``_GraphReplayModule.forward`` does
    ``buf.copy_(t)`` then ``graph.replay()``, so the copy cost is
    measured, not hidden), ``graph_max_abs`` the replay-vs-eager
    output delta, and ``module`` is overridden with the replay
    wrapper only when capture *and* the numeric check succeeded.
    """
    import torch

    out: dict[str, Any] = {"runner": "cuda_graph"}
    xin = ex.get("input")
    xs = (
        tuple(xin)
        if isinstance(xin, (tuple, list))
        else (() if xin is None else (xin,))
    )
    out["plan_tape"] = _is_plan_tape(module)
    if not (
        xs
        and all(isinstance(t, torch.Tensor) and t.is_cuda for t in xs)
    ):
        out["cuda_graph"] = False
        out["runner_note"] = "inputs not CUDA tensors"
        return out
    try:  # pragma: no cover — CUDA-only body
        from mlopt_cuda import CudaGraphRunner
    except ImportError:  # pragma: no cover — optional dep absent
        out["cuda_graph"] = False
        out["runner_note"] = "mlopt-cuda not installed"
        return out
    stats: dict[str, Any] = {}
    try:  # pragma: no cover — CUDA-only body
        wrapped = CudaGraphRunner().apply(module, xin, stats)
    except Exception as exc:  # pragma: no cover — CUDA-only
        out["cuda_graph"] = False
        out["runner_note"] = f"apply raised {type(exc).__name__}: {exc}"
        return out
    if not stats.get("cuda_graph"):  # pragma: no cover — CUDA-only
        failure = stats.get("cuda_graph_failure")
        out["cuda_graph"] = False
        out["runner_note"] = (
            f"capture declined ({failure})"
            if failure is not None
            else "capture declined"
        )
        return out
    try:  # pragma: no cover — CUDA-only body
        t_eager = _time_module(module, xin, reps)
        t_graph = _time_module(wrapped, xin, reps)
        with torch.no_grad():
            out_eager = module(*xs)
            out_graph = wrapped(*xs)
    except Exception as exc:  # pragma: no cover — CUDA-only
        out["cuda_graph"] = False
        out["runner_note"] = (
            f"post-capture check failed: {type(exc).__name__}: {exc}"
        )
        return out
    out.update(  # pragma: no cover — CUDA-only body
        {
            "cuda_graph": True,
            "deliver_us": round(t_eager, 1),
            "graph_us": round(t_graph, 1),
            "graph_speedup": (
                round(t_eager / t_graph, 3) if t_graph else None
            ),
            "graph_max_abs": _max_abs_delta(out_eager, out_graph),
            "module": wrapped,
        }
    )
    return out  # pragma: no cover — CUDA-only success path


def _torch_deliver(board: ma.MetaArena, case: Any, traj: Any) -> dict:
    """Lower the winning extraction and verify it against the model.

    ``deliverable`` resolves handled claims to their kernels and
    unhandled abbreviations to spelled form; ``TorchSink.lower``
    rebuilds the module from it; ``verify`` runs both on the case's
    example input.  The game score becomes a delivered artifact.
    On a verified CUDA deliverable the ``CudaGraphRunner`` arm
    (:func:`_deliver_runner_arm`) additionally attempts graph
    capture — additive keys only, eager module on any decline.
    """
    from mlopt_core.ir import IR
    from mlopt_torch.adapters import TorchSink

    ex = case[5]
    best = board.eg.extract_best(board.root, board.feasible_cost)
    if best is None:
        return {"delivered": False, "reward": traj.total}
    term = board.deliverable(best)
    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    sink = ex.get("sink") or TorchSink()
    module = sink.lower(ir, params=ex["leaves"])
    rep = sink.verify(ex["model"], module, (ex["input"],))
    t = traj.terminal
    out = {
        "delivered": True,
        "module": module,
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "cost": getattr(t, "cost", None),
        "cert": getattr(t, "certificate_ok", None),
        "reward": traj.total,
    }
    if rep.passed:
        out.update(_deliver_runner_arm(module, ex))
    return out


# ---------------------------------------------------------------------------
#  Domain: gen — generated kernels over real torch modules
# ---------------------------------------------------------------------------
#
#  The "discovery builds lowerings" story end to end: spell fusion
#  patterns the ambient handler table has no name for (``x·relu(x)``,
#  ``x·gelu(x)``, …) plus stock-model pieces; the board mints a gen
#  handler + kernel per occurring elementwise subterm; ``claim``
#  binds it; ``deliver`` lowers it through a ``TorchSink`` whose op
#  table includes the generated binding — a runnable module calling
#  a kernel nobody wrote.


def _fusion_pattern(t: Any, names: dict) -> Any:
    """Return the elementwise-closure pattern at *t*, leaves abstracted.

    A subterm fuses iff its *head* is elementwise — children below
    the elementwise boundary (matmuls, reductions, …) become bound
    metavariables, consistently named by subtree identity.  So
    ``mul(y, relu(y))`` over ``y = mm(…)`` yields
    ``("mul", "X1", ("relu", "X1"))`` — ``X1`` binds the whole
    folded matmul, and the generated kernel computes it once.
    """
    from mlopt_core.ir import Const

    from mlopt_discovery import genkernel as gk

    if isinstance(t, Op) and t.op in gk.ELEMENTWISE:
        kids = [_fusion_pattern(a, names) for a in t.args]
        if t.attrs:
            kids.append(dict(t.attrs))
        return (t.op, *kids)
    if isinstance(t, Const):
        return t.value
    key = repr(t)
    if key not in names:
        names[key] = f"X{len(names) + 1}"
    return names[key]


def _fusion_sites(term: Any) -> list:
    """Elementwise-connex subterms of size ≥2, leaves abstracted.

    Each entry is a *pattern tuple* (metavar leaves for the
    non-elementwise boundary), the shape ``gen_handlers`` consumes.
    """
    from mlopt_discovery import genkernel as gk

    out: list = []

    def walk(t: Any) -> None:
        if isinstance(t, Op) and t.op in gk.ELEMENTWISE:
            spec = _fusion_pattern(t, {})
            ops = {e for e in _flat_ops(spec)}
            if len(ops) >= 2:
                out.append(spec)
        if isinstance(t, Op):
            for a in t.args:
                walk(a)

    walk(term)
    return out


def _reduction_pattern(t: Any, names: dict) -> Any:
    """Spell a reduction-headed subterm as a fusion spec.

    The head and its attr dict stay concrete (``dim``/``keepdim``
    are part of the site's meaning — a claim binds that spelling,
    not every reduction of the same op); each arg is spelled by the
    pointwise tier's own boundary rule, so non-elementwise children
    abstract to metavars and the elementwise body stays spelled —
    ``mean(mul(x, relu(x)))`` → ``("mean", ("mul", "X1",
    ("relu", "X1")), {…})``.  :func:`genkernel.reduction_spec` is
    the accept gate.
    """
    kids = [_fusion_pattern(a, names) for a in t.args]
    if t.attrs:
        kids.append(dict(t.attrs))
    return (t.op, *kids)


def _reduction_sites(term: Any) -> list:
    """Reduction-bodied subterms of *term*, spec'd for the tier.

    The reduction tier's site collector — the mirror of
    ``_fusion_sites`` one level up: every ``Op`` whose head is a
    reduce-dim op (``genkernel.reduce_heads``) mints a candidate
    spec; :func:`genkernel.reduction_spec` keeps only the
    elementwise-fusable bodies (>=2 ops, non-head ops all
    elementwise).  Sites are raw-IR subterms, so minted entries
    carry no ``leaf_env`` — ``_handler_env`` rebinds them by
    concrete match like every other tier.
    """
    from mlopt_discovery import genkernel as gk

    heads = gk.reduce_heads()
    out: list = []

    def walk(t: Any) -> None:
        if isinstance(t, Op) and t.op in heads:
            spec = _reduction_pattern(t, {})
            if gk.reduction_spec(spec):
                out.append(spec)
        if isinstance(t, Op):
            for a in t.args:
                walk(a)

    walk(term)
    return out


def _decayed_sites(term: Any) -> list:
    """``matmul(mul(s,d),v)``/``om_decayed`` subterms, spec'd.

    The decayed-tier site collector — the elementwise walk cannot
    see a contraction site, so each ``mul`` feeding a ``matmul``
    (and each already-lifted ``om_decayed``) mints a pattern whose
    operands are metavars binding whole subterms at match time —
    the kernel then receives the evaluated operand tensors.
    """
    out: list = []

    def walk(t: Any) -> None:
        if isinstance(t, Op):
            if t.op == "om_decayed":
                out.append(("om_decayed", "S", "D", "V"))
            elif (
                t.op == "matmul"
                and isinstance(t.args[0], Op)
                and t.args[0].op == "mul"
            ):
                out.append(("matmul", ("mul", "S", "D"), "V"))
            for a in t.args:
                walk(a)

    walk(term)
    return out


def _scan_sites(term: Any) -> list:
    """``applyd``-carrier members of the saturated graph, spec'd.

    The scan tier's site source is not raw IR — carriers only exist
    once the lifts run.  This collector runs a bounded carrier-
    producing saturation (the om laws that introduce ``applyd``)
    and harvests every ``applyd`` member's decoded spec, pinned
    ``kind=unrolled`` — the machine-folded carriers' ambient
    reading, not the fold-over-steps variant the seq form declares.

    Each site is a ``{"spec": ..., "leaf_env": ...}`` record: the
    leaf env binds every spec metavar — the abstracted leaves AND
    the abstracted state — to the decoded subterm it stands for.
    The member the spec came from exists only inside this
    throwaway graph, so no raw-IR match can ever rebind it; the
    record carries the binding the referee needs.

    Sites also carry ``"chain"`` when the applyd member's e-class
    contains a raw-IR subterm of *term* — the outermost one, found
    by mapping every subterm through ``_class_of_term``.  That
    spelled subterm is the deliver-time route's LHS, the same role
    ``"chain"`` plays for the seq/seqm collectors: saturation
    proved it equal to the carrier member, so delivery may rewrite
    the extracted class member to the minted kernel call.  A class
    with no raw member (a carrier synthesized wholesale) records
    no route and delivers spelled, as before.
    """
    from mlopt_carriers.om import OM_CHUNKIFY, OM_LAWS
    from mlopt_core.egraph import EGraph
    from mlopt_core.laws.tensor import SPLIT_SCALAR_RULES

    from mlopt_discovery.meta_arena import _spec_of

    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(
        list(OM_LAWS) + list(OM_CHUNKIFY) + list(SPLIT_SCALAR_RULES),
        root,
        max_iterations=10,
        max_nodes=60_000,
        rule_budgets={
            "om_decayed_chunkify_rows": 20,
            "om_decayed_chunkify_keys": 20,
        },
    )
    # e-class -> outermost raw-IR subterm realising it.  The walk is
    # pre-order, so the first subterm seen per class is the largest
    # spelled member — the route's maximal LHS.
    raw_by_class: dict = {}
    cls_memo: dict = {}
    seen_raw: set = set()

    def _mark(t: Any) -> None:
        if not isinstance(t, Op) or id(t) in seen_raw:
            return
        seen_raw.add(id(t))
        cid = eg._class_of_term(t, cls_memo)
        if cid is not None and cid not in raw_by_class:
            raw_by_class[cid] = t
        for a in t.args:
            _mark(a)

    _mark(term)
    out: list = []
    carrier_ops = ("aff_diag", "affd_compose")
    for cls in eg._classes.values():
        for n in cls.nodes:
            if n.op != "applyd":
                continue
            member = _carrier_member(eg, n, carrier_ops)
            if member is None:
                continue
            f_term = member.args[0]
            if not isinstance(f_term, Op):
                continue
            names: dict = {}
            f_spec = _spec_of(f_term, names)
            leaf_env = _spec_leaf_env(f_term, names)
            # the state abstracts to a metavar — the handler's
            # pattern is ``applyd(<map>, X_h)`` over ANY state;
            # whatever the fold produced there is recorded too
            state = f"X{len(names) + 1}"
            leaf_env[state] = member.args[1]
            spec = ("applyd", f_spec, state, {"kind": "unrolled"})
            rec: dict = {"spec": spec, "leaf_env": leaf_env}
            chain = raw_by_class.get(eg.find(cls.id))
            if chain is not None:
                rec["chain"] = chain
            # dedup on the WHOLE record, not the spec tuple alone:
            # two carrier members can share one spec skeleton while
            # ``leaf_env`` binds different operands — distinct folds
            # the referee must measure and distinct chains the
            # delivery route must rewrite.
            if rec in out:
                continue
            out.append(rec)
    return out


def _spec_leaf_env(term: Any, names: dict) -> dict:
    """Bind each metavar ``_spec_of`` minted to its leaf subterm.

    The inverse of leaf abstraction: every non-``Op``/non-``Const``
    leaf maps to ``names[repr(leaf)]`` — the env a spec harvested
    post-saturation carries so its metavars evaluate without a
    match (``eval_term`` on the term, the literal itself on a raw
    scalar leaf like a ``pow`` exponent).
    """
    env: dict = {}

    def walk(t: Any) -> None:
        if isinstance(t, Op):
            for a in t.args:
                walk(a)
        elif not isinstance(t, Const):
            env[names[repr(t)]] = t

    walk(term)
    return env


def _carrier_member(eg: Any, node: Any, carrier_ops: tuple) -> Any:
    """Decode an ``applyd`` enode preferring carrier-shaped children.

    ``any_term`` prefers leaves — useless for the map slot, which
    must surface as ``aff_diag``/``affd_compose``.  Children decode
    to the carrier op member when their class carries one, else the
    class rep.
    """
    args = []
    for c in node.children:
        cls = eg._classes[eg.find(c)]
        pick = next((m for m in cls.nodes if m.op in carrier_ops), None)
        if pick is not None:
            args.append(_carrier_member(eg, pick, carrier_ops))
        else:
            args.append(eg.any_term(eg.find(c)))
    if any(a is None for a in args):
        return None
    return Op.make(node.op, *args, **dict(node.attrs))


# ---------------------------------------------------------------------------
#  Seq-form scan sites — raw recurrence chains, not saturated carriers
# ---------------------------------------------------------------------------

#: Cap on tied spine readings kept per node and seq sites emitted per
#: term — ambiguity is rare; the caps bound the collector's work on
#: pathological terms without dropping the common case.
_SEQ_SPINE_CAP = 4
_SEQ_SITE_CAP = 16


def _seq_step_cands(t: Any) -> tuple:
    """``(map, state, inp)`` readings of one ``add(mul(·,·),·)`` step.

    The diagonal recurrence ``h' = a ⊙ h + x`` arrives unrolled as
    ``add(mul(a, h), x)`` — either add operand may hold the product
    and either mul factor may be the state, so a node yields up to
    four candidate decompositions.  The spine walk keeps only the
    longest readings; :func:`_seq_site` arbitrates the rest.  The
    ``a * h`` spelling (map first, state second) is tried first so
    the common torch order wins ties deterministically.
    """
    if not (
        isinstance(t, Op)
        and t.op == "add"
        and len(t.args) == 2
        and not t.attrs
    ):
        return ()
    c0, c1 = t.args
    out = []
    for prod, inn in ((c0, c1), (c1, c0)):
        if (
            isinstance(prod, Op)
            and prod.op == "mul"
            and len(prod.args) == 2
            and not prod.attrs
        ):
            f0, f1 = prod.args
            out.append((f0, f1, inn))
            out.append((f1, f0, inn))
    # Matrix-state form: ``h' = h + f(h)`` — the delta-rule spelling
    # ``s = s + beta·k⊗(v - kᵀs)`` puts the state as a DIRECT add
    # operand, never a ``mul`` factor, so the product readings above
    # cannot see it.  ``None`` marks the implicit unit map; the
    # seq-matrix collector (:func:`_seqm_site`) is the only consumer
    # that accepts it — :func:`_seq_site` declines ``None`` maps.
    out.append((None, c0, c1))
    out.append((None, c1, c0))
    return tuple(out)


def _seq_spines(t: Any, memo: dict) -> list:
    """Decomposition alternatives of *t*, longest first.

    Each entry is ``(steps, base)`` — *steps* a tuple of
    ``(map, inp, state)`` triples in CHRONOLOGICAL order (the
    outermost ``add`` is the last step, so the walk prepends inward)
    and *base* the initial-state term the fold starts from.  A node
    with no step reading is its own base; tied longest readings are
    all kept (up to :data:`_SEQ_SPINE_CAP`) so the seq-ification
    pass can arbitrate readings the spine length cannot.
    """
    key = id(t)
    if key in memo:
        return memo[key]
    cands = _seq_step_cands(t)
    results = []
    for m, s, inn in cands:
        for sub_steps, sub_base in _seq_spines(s, memo):
            results.append(((*sub_steps, (m, inn, s)), sub_base))
    if results:
        top = max(len(r[0]) for r in results)
        memo[key] = [r for r in results if len(r[0]) == top][
            :_SEQ_SPINE_CAP
        ]
    else:
        memo[key] = [((), t)]
    return memo[key]


def _seq_view_lift(t: Any):
    """Lift a view-op chain over a ``select`` onto the select's child.

    ``unsqueeze``/``squeeze`` directly wrapping a per-step ``select``
    (possibly nested) commute onto the sequence tensor: the view is
    re-expressed on the WHOLE child with its ``dim`` re-based into
    child coordinates, and the select's dim shifted to match.  The
    result is ``(viewed_child, select_dim, index)`` — feed it through
    the ordinary operand path so the metavar binds the viewed
    sequence (the step axis's position in *viewed-child* coords).

    Why lift instead of rendering the view in the step expr: the
    seq fold's state space is the operand broadcast minus the step
    axis.  A view INSIDE the step expr (``unsqueeze`` of a ``(b,d)``
    slice against a ``(b,d,d)`` state) has no flat-space meaning —
    but the same view lifted onto the sequence child grows the
    operand's own shape so the broadcast *includes* the state dims,
    and every downstream consumer (``shape_of`` checks,
    ``seq_scan_denotation``, the Triton kernel's materialised rows)
    is unchanged.  ``unsqueeze(gg,-1)`` on a ``(b,T,d)`` operand
    becomes ``unsqueeze(gg,3)`` selected at dim 1 — the fold then
    reads ``(b,d,d)`` step slices, exactly the recurrence's state.

    Returns ``None`` — the caller falls back to the generic spec
    path, which keeps the view node in the skeleton and lets
    ``scan_spec`` decline honestly — when the chain does not end at
    a ``select``, a ``dim`` attr is missing/non-int, ``shape_of``
    cannot type a child, or a normalised dim falls out of range.
    Other view ops (``reshape``/``view``/``expand``/``permute``)
    and dim-less ``squeeze`` stay outside the grammar.
    """
    from mlopt_core.typing import INVALID, shape_of

    views: list = []
    node = t
    while (
        isinstance(node, Op)
        and node.op in ("unsqueeze", "squeeze")
        and len(node.args) == 1
        and set(node.attrs) <= {"dim"}
        and type(node.attrs.get("dim")) is int
    ):
        views.append(node)
        node = node.args[0]
    if (
        not views
        or not isinstance(node, Op)
        or node.op != "select"
        or len(node.args) != 1
        or type(node.attrs.get("dim")) is not int
        or type(node.attrs.get("index")) is not int
    ):
        return None
    child = node.args[0]
    idx = node.attrs["index"]
    sh = shape_of(child)
    if sh is INVALID or not isinstance(sh, tuple):
        return None
    rank = len(sh)
    d = node.attrs["dim"]
    d0 = d if d >= 0 else d + rank
    if not 0 <= d0 < rank:
        return None
    # Fold the views inside-out (``views`` is outermost-first).  At
    # each step the invariant is ``select(child, d0, idx)`` == the
    # tensor the remaining inner views produce; *d0* is the step
    # axis in current-child coords, *rank-1* the sliced rank the
    # view's ``dim`` normalises against.
    for v in reversed(views):
        vd = v.attrs["dim"]
        sh = shape_of(child)
        if sh is INVALID or not isinstance(sh, tuple):
            return None
        r = len(sh) - 1  # rank of the post-select tensor
        if v.op == "unsqueeze":
            # insertion position in sliced coords, 0..r
            p = vd if vd >= 0 else vd + r + 1
            if not 0 <= p <= r:
                return None
            # child-coord insertion: before the step axis when the
            # slice position precedes it, else after — either side
            # of the boundary yields the same result axis order, and
            # this choice keeps the step axis index stable (``dims``
            # stays a singleton across operands).
            q = p if p < d0 else p + 1
            child = Op.make("unsqueeze", child, dim=q)
            if q <= d0:
                d0 += 1
        else:  # squeeze — removes the sliced axis at position k
            k = vd if vd >= 0 else vd + r
            if not 0 <= k < r:
                return None
            j = k if k < d0 else k + 1  # the same axis in child coords
            child = Op.make("squeeze", child, dim=j)
            if j < d0:
                d0 -= 1
    sh = shape_of(child)
    if (
        sh is INVALID
        or not isinstance(sh, tuple)
        or not 0 <= d0 < len(sh)
    ):
        return None
    return child, d0, idx


def _seq_meta(
    names: dict,
    operands: dict,
    events: list,
    child: Any,
    dim: int,
    idx: int,
) -> str:
    """Mint/reuse the sequence-operand metavar for ``(child, dim)``."""
    key = (child, dim)
    meta = names.get(key)
    if meta is None:
        meta = f"Q{len(names) + 1}"
        names[key] = meta
        operands[meta] = (child, dim)
    events.append((meta, idx))
    return meta


def _seq_const_meta(consts: dict, t: Any) -> str:
    """Mint/reuse the loop-invariant metavar for leaf *t*.

    *consts* maps the leaf term to its ``"P"``-prefixed meta — equal
    leaves (``Param``/``Var`` are value-equality dataclasses) share
    one meta, so a weight used at every step binds once.
    """
    meta = consts.get(t)
    if meta is None:
        meta = f"P{len(consts) + 1}"
        consts[t] = meta
    return meta


def _seq_select_free(t: Any) -> bool:
    """Whether *t* contains no ``select`` node anywhere.

    Iterative over the interned DAG — the lifted projection operand
    is evaluated ONCE by ``leaf_env``, so a ``select`` surviving
    inside it would freeze one step's slice into every step.  Fixed
    -index selects are technically loop-invariant, but rejecting
    them keeps the lift's contract simple: the operand child must
    be a pure sequence tensor.
    """
    stack = [t]
    seen: set = set()
    while stack:
        n = stack.pop()
        if not isinstance(n, Op) or id(n) in seen:
            continue
        seen.add(id(n))
        if n.op == "select":
            return False
        stack.extend(n.args)
    return True


def _seq_step_select(t: Any):
    """``(child, dim, index)`` when *t* is a bare per-step ``select``."""
    if (
        isinstance(t, Op)
        and t.op == "select"
        and len(t.args) == 1
        and type(t.attrs.get("dim")) is int
        and type(t.attrs.get("index")) is int
    ):
        return t.args[0], t.attrs["dim"], t.attrs["index"]
    return None


def _seq_projection_lift(t: Any):
    """Pull a per-step ``select`` out through a projection op.

    ``matmul``/``linear`` over a ``select``ed sequence commutes with
    the slice whenever the selected axis is not contracted:
    ``matmul(select(x, d, i), w)`` on a non-``k`` axis *d* equals
    ``select(matmul(x, w), d', i)`` — the projection hoists onto the
    WHOLE sequence tensor and the fold slices the projected rows,
    exactly the spelling a hand-hoisted recurrence already uses.
    Returns ``(lifted_term, out_dim, index)`` — feed through the
    ordinary operand path — or ``None`` to decline.

    Accepted shapes (verified against ``shape_of``, torch matmul /
    ``F.linear`` semantics):

    * ``matmul(a, b)`` with the select a DIRECT left arg: *b* must
      have rank ≤ 2 (no batch dims to misalign) and the select dim
      must not be the contracted last axis of *a*; batch and ``m``
      dims keep their position in the output, so ``out_dim = d0``.
    * ``matmul(a, b)`` with the select a DIRECT right arg: *b* must
      be a rank-2 ``(k, n)`` matrix and the select must hit the
      free ``n`` axis; the output's last axis is ``n``.
    * ``linear(x, w[, b])`` with the select a DIRECT first arg:
      ``F.linear`` contracts only the last axis of *x*, so any
      leading dim lifts with ``out_dim = d0``.

    The lifted term must be ``select``-free — a surviving ``select``
    would freeze one step's slice into the once-evaluated operand
    (see :func:`_seq_select_free`).  Selects nested deeper inside an
    argument, selects in BOTH matmul args, and every other op keep
    the generic decline.
    """
    from mlopt_core.typing import INVALID, shape_of

    if not (isinstance(t, Op) and not t.attrs):
        return None
    if t.op == "matmul" and len(t.args) == 2:
        a, b = t.args
        sa = _seq_step_select(a)
        sb = _seq_step_select(b)
        if (sa is None) == (sb is None):
            return None  # zero or two direct selects — decline
        if sa is not None:
            c, d, idx = sa
            other, pos = b, 0
        else:
            c, d, idx = sb
            other, pos = a, 1
        sh = shape_of(c)
        so = shape_of(other)
        if (
            sh is INVALID
            or so is INVALID
            or not isinstance(sh, tuple)
            or not isinstance(so, tuple)
        ):
            return None
        ra, rb = len(sh), len(so)
        d0 = d if d >= 0 else d + ra
        if not 0 <= d0 < ra:
            return None
        if pos == 0:
            # left operand: last axis is the contracted ``k``; every
            # other axis keeps its position in the (…, m, n) output
            # provided *b* brings no batch dims of its own.
            if rb > 2 or d0 == ra - 1:
                return None
            out_dim = d0
        else:
            # right operand: only a rank-2 ``(k, n)`` right tensor
            # has a free axis — ``n``, its last dim — and it lands
            # last in the output.  ``ra`` is the SELECT CHILD's rank
            # (the lifted right operand), ``rb`` the left arg's.
            if ra != 2 or d0 != ra - 1:
                return None
            out_dim = (rb if rb >= 2 else 1) - 1
        lifted = (
            Op.make("matmul", c, other)
            if pos == 0
            else Op.make("matmul", other, c)
        )
    elif t.op == "linear" and len(t.args) in (2, 3):
        s = _seq_step_select(t.args[0])
        if s is None:
            return None
        c, d, idx = s
        sh = shape_of(c)
        if sh is INVALID or not isinstance(sh, tuple):
            return None
        ra = len(sh)
        d0 = d if d >= 0 else d + ra
        if not 0 <= d0 < ra - 1:
            return None  # the contracted in-features axis
        out_dim = d0
        lifted = Op.make("linear", c, *t.args[1:])
    else:
        return None
    if not _seq_select_free(lifted):
        return None
    sh2 = shape_of(lifted)
    if (
        sh2 is INVALID
        or not isinstance(sh2, tuple)
        or not 0 <= out_dim < len(sh2)
    ):
        return None
    return lifted, out_dim, idx


def _seq_shape_first(_op: Any, shapes: list) -> Any:
    """Shape rule for ops that return their first operand's shape."""
    return shapes[0] if shapes else None


# ``shape_of`` rules the seq commute-lift relies on.  These ops
# provably preserve their input's shape, but the substrate's rule
# table (``mlopt_core.typing``, owned by the eqsat repo) does not
# list them yet — ``register_shape_rule`` is the documented
# extension point and the rule is semantically exact, not a patch:
# ``layer_norm``/``rms_norm``/``clone`` always return ``x.shape``.
for _norm_op in ("layer_norm", "rms_norm", "clone"):
    _register_shape_rule(_norm_op, _seq_shape_first)
del _norm_op


def _seq_axis_idx(d: Any, rank: int):
    """Normalise an int axis into ``[0, rank)``; ``None`` out of range."""
    if type(d) is not int:
        return None
    d0 = d if d >= 0 else d + rank
    return d0 if 0 <= d0 < rank else None


def _seq_commute_lift(t: Any):
    """Pull a per-step ``select`` out through an axis-safe op.

    The projection lift (:func:`_seq_projection_lift`) covers
    ``matmul``/``linear``; this is the same hoist for every other op
    whose denotation commutes with the step-axis slice —
    ``op(select(x, d, i), …) == select(op(x, …), D, i)`` — so the
    operand binds the WHOLE-tensor ``op(x, …)`` evaluated once and
    the fold slices it per step.  Each case below states the law it
    relies on; anything not listed, any select that is not a DIRECT
    argument, and any ``argN`` positional attr spelling (which would
    alias the dims being remapped) keeps the generic decline.

    * ``transpose``/``permute`` — pure axis permutations: the select
      axis is re-based through the permutation and re-inserted last.
    * ``softmax``/``log_softmax`` — normalise along ONE axis; they
      commute iff that axis is not the step axis.
    * ``layer_norm``/``rms_norm`` — normalise over the LAST
      ``len(normalized_shape)`` axes; they commute iff the step axis
      sits strictly before that trailing block (and the block's
      sizes match the child's trailing dims).
    * ``sum``/``mean``/``amax``/``amin`` — reductions commute iff no
      reduced axis is the step axis; ``keepdim`` decides whether
      the output axis shifts.
    * ``slice``/``narrow``/``chunk``/``split`` — range restrictions
      along one axis commute iff that axis is not the step axis.
    * ``unbind`` — removes one axis; commutes iff it is not the step
      axis (the output axis count drops by one).
    * ``concat`` — EVERY argument must be a bare ``select`` on the
      SAME child axis with the SAME index (the multi-arg analogue of
      the both-args boundary); the lifted concat joins the children
      along the remapped concat axis.
    * ``to``/``clone``/``contiguous``/``alias``/``dropout``/
      ``broadcast``/``float``/``type_as`` — axis-preserving
      passthroughs; they commute unconditionally.

    Returns ``(lifted_term, out_dim, index)`` — feed through the
    ordinary operand path — or ``None`` to decline.  The lifted term
    must be ``select``-free (:func:`_seq_select_free`) and type via
    ``shape_of`` exactly like the projection lift's output.
    """
    from mlopt_core.typing import INVALID, shape_of

    if not isinstance(t, Op) or not t.args:
        return None
    if t.op == "concat":
        return _seq_concat_lift(t)
    s = _seq_step_select(t.args[0])
    if s is None:
        return None
    c, d, idx = s
    sh = shape_of(c)
    if sh is INVALID or not isinstance(sh, tuple) or not sh:
        return None
    r = len(sh)
    d0 = _seq_axis_idx(d, r)
    if d0 is None:
        return None
    rs = r - 1  # rank of the per-step (post-select) tensor

    def up(j: int) -> int:
        """Sliced-tensor axis -> child-tensor axis (skips ``d0``)."""
        return j if j < d0 else j + 1

    ctx = _SeqLiftCtx(t, c, d0, idx, rs, r, sh, up)
    if t.op in _SEQ_PASSTHROUGH_OPS:
        # axis-preserving passthroughs need no dim bookkeeping
        return _seq_lift_finish(
            (Op.make(t.op, c, *t.args[1:], **t.attrs), ctx.d0), ctx
        )
    # Every remaining case remaps an axis attr — positional ``argN``
    # spellings would alias those dims in child coordinates, so they
    # decline rather than lift under the wrong axis.
    if _seq_argn_attrs(t.attrs):
        return None
    fn = _SEQ_AXIS_LIFTS.get(t.op)
    if fn is None:
        return None
    return _seq_lift_finish(fn(ctx), ctx)


@dataclass(frozen=True)
class _SeqLiftCtx:
    """Shared context for the commute-lift op handlers.

    ``c`` is the select's child (the whole sequence tensor), ``d0``
    its normalised step axis, ``idx`` the step index, ``rs``/``r``
    the sliced/child ranks, ``sh`` the child's ``shape_of`` and
    ``up`` the sliced-axis -> child-axis map (skips ``d0``).
    """

    t: Any
    c: Any
    d0: int
    idx: int
    rs: int
    r: int
    sh: tuple
    up: Any


def _seq_argn_attrs(attrs: dict) -> bool:
    """Whether *attrs* carries a positional ``argN`` spelling."""
    return any(
        isinstance(k, str) and k.startswith("arg") and k[3:].isdigit()
        for k in attrs
    )


def _seq_lift_finish(res: Any, ctx: _SeqLiftCtx):
    """Shared tail: build ``(lifted, out_dim, idx)`` or decline.

    *res* is the op handler's ``(lifted_term, out_dim)`` pair (or
    ``None``).  The lifted term must be ``select``-free — a surviving
    ``select`` would freeze one step's slice into the once-evaluated
    operand — and ``shape_of`` must type it with ``out_dim`` in
    range, the same contract the projection lift keeps.
    """
    from mlopt_core.typing import INVALID, shape_of

    if res is None:
        return None
    lifted, out_dim = res
    if not _seq_select_free(lifted):
        return None
    sh2 = shape_of(lifted)
    if (
        sh2 is INVALID
        or not isinstance(sh2, tuple)
        or not 0 <= out_dim < len(sh2)
    ):
        return None
    return lifted, out_dim, ctx.idx


def _seq_lift_attrs(
    ctx: _SeqLiftCtx, drop: tuple, **set_attrs: Any
) -> dict:
    """Copy *ctx.t*'s attrs minus *drop*, plus *set_attrs*."""
    out = {k: v for k, v in ctx.t.attrs.items() if k not in drop}
    out.update(set_attrs)
    return out


def _seq_lift_transpose(ctx: _SeqLiftCtx):
    """``transpose`` — swap two axes; the select axis keeps position."""
    a = _seq_axis_idx(ctx.t.attrs.get("dim0", -2), ctx.rs)
    b = _seq_axis_idx(ctx.t.attrs.get("dim1", -1), ctx.rs)
    if a is None or b is None:
        return None
    lifted = Op.make(
        "transpose",
        ctx.c,
        **_seq_lift_attrs(
            ctx, ("dim0", "dim1"), dim0=ctx.up(a), dim1=ctx.up(b)
        ),
    )
    return lifted, ctx.d0


def _seq_lift_permute(ctx: _SeqLiftCtx):
    """``permute`` — rebuild the permutation with the step axis last."""
    dims = ctx.t.attrs.get("dim", ctx.t.attrs.get("dims"))
    if not (
        isinstance(dims, (tuple, list))
        and len(dims) == ctx.rs
        and all(type(v) is int for v in dims)
    ):
        return None
    ps = [_seq_axis_idx(v, ctx.rs) for v in dims]
    if any(v is None for v in ps) or sorted(ps) != list(range(ctx.rs)):
        return None
    lifted = Op.make(
        "permute",
        ctx.c,
        **_seq_lift_attrs(
            ctx,
            ("dim", "dims"),
            dim=(*tuple(ctx.up(v) for v in ps), ctx.d0),
        ),
    )
    return lifted, ctx.r - 1


def _seq_lift_softmax(ctx: _SeqLiftCtx):
    """``softmax``/``log_softmax`` — mixes only along ``dim``."""
    k = _seq_axis_idx(ctx.t.attrs.get("dim", -1), ctx.rs)
    if k is None:
        return None
    kc = ctx.up(k)
    if kc == ctx.d0:
        return None
    lifted = Op.make(
        ctx.t.op, ctx.c, **_seq_lift_attrs(ctx, (), dim=kc)
    )
    return lifted, ctx.d0


def _seq_lift_norm(ctx: _SeqLiftCtx):
    """``layer_norm``/``rms_norm`` — normalise the trailing block."""
    ns = ctx.t.attrs.get("dim")
    if not (
        isinstance(ns, tuple) and ns and all(type(v) is int for v in ns)
    ):
        return None
    ln = len(ns)
    if ln > ctx.rs or ctx.d0 >= ctx.r - ln:
        return None
    if tuple(ns) != tuple(ctx.sh[ctx.r - ln :]):
        return None
    lifted = Op.make(ctx.t.op, ctx.c, *ctx.t.args[1:], **ctx.t.attrs)
    return lifted, ctx.d0


def _seq_reduce_dims(ctx: _SeqLiftCtx) -> list | None:
    """Normalised reduced-axis list in sliced coords, or ``None``."""
    dim = ctx.t.attrs.get("dim")
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    if not dims or any(type(v) is not int for v in dims):
        return None
    rd = []
    for v in dims:
        j = _seq_axis_idx(v, ctx.rs)
        if j is None or j in rd:
            return None
        rd.append(j)
    return rd


def _seq_lift_reduce(ctx: _SeqLiftCtx):
    """``sum``/``mean``/``amax``/``amin`` — reduce non-step axes."""
    rd = _seq_reduce_dims(ctx)
    if rd is None:
        return None
    rc = {ctx.up(j) for j in rd}
    if ctx.d0 in rc:
        return None
    lifted = Op.make(
        ctx.t.op,
        ctx.c,
        **_seq_lift_attrs(ctx, (), dim=tuple(ctx.up(j) for j in rd)),
    )
    if ctx.t.attrs.get("keepdim", False):
        return lifted, ctx.d0
    return lifted, ctx.d0 - sum(1 for x in rc if x < ctx.d0)


def _seq_lift_range(ctx: _SeqLiftCtx):
    """``slice``/``narrow``/``chunk``/``split`` — range one axis."""
    kd = ctx.t.attrs.get("dim", 0 if ctx.t.op == "slice" else -1)
    k = _seq_axis_idx(kd, ctx.rs)
    if k is None:
        return None
    kc = ctx.up(k)
    if kc == ctx.d0:
        return None
    lifted = Op.make(
        ctx.t.op, ctx.c, **_seq_lift_attrs(ctx, (), dim=kc)
    )
    return lifted, ctx.d0


def _seq_lift_unbind(ctx: _SeqLiftCtx):
    """``unbind`` — removes one non-step axis; positions shift."""
    k = _seq_axis_idx(ctx.t.attrs.get("dim", 0), ctx.rs)
    if k is None:
        return None
    kc = ctx.up(k)
    if kc == ctx.d0:
        return None
    lifted = Op.make(
        ctx.t.op, ctx.c, **_seq_lift_attrs(ctx, (), dim=kc)
    )
    return lifted, ctx.d0 if ctx.d0 < kc else ctx.d0 - 1


#: Op -> commute-lift handler for the axis-remapping families.
_SEQ_AXIS_LIFTS = {
    "transpose": _seq_lift_transpose,
    "permute": _seq_lift_permute,
    "softmax": _seq_lift_softmax,
    "log_softmax": _seq_lift_softmax,
    "layer_norm": _seq_lift_norm,
    "rms_norm": _seq_lift_norm,
    "sum": _seq_lift_reduce,
    "mean": _seq_lift_reduce,
    "amax": _seq_lift_reduce,
    "amin": _seq_lift_reduce,
    "slice": _seq_lift_range,
    "narrow": _seq_lift_range,
    "chunk": _seq_lift_range,
    "split": _seq_lift_range,
    "unbind": _seq_lift_unbind,
}


#: Ops that preserve every axis position exactly — the commute lift
#: needs no dim bookkeeping for them.
_SEQ_PASSTHROUGH_OPS = frozenset(
    {
        "to",
        "clone",
        "contiguous",
        "alias",
        "dropout",
        "broadcast",
        "float",
        "type_as",
    }
)


def _seq_concat_lift(t: Any):
    """Lift ``concat`` when EVERY arg is a per-step ``select``.

    ``concat([select(x,d,i), …], dim=c)`` with all selects on the
    same child axis ``d`` (never the joined axis) and the same index
    ``i`` equals ``select(concat([x,…], dim=c'), d, i)`` — the
    children are joined along the remapped concat axis and the fold
    slices the joined sequence.  Mixed select axes, mixed indices,
    a select ON the joined axis, or any non-select argument all
    decline: a single outer ``select`` cannot reproduce them.
    """
    from mlopt_core.typing import INVALID, shape_of

    if len(t.args) < 2 or any(
        isinstance(k, str) and k.startswith("arg") and k[3:].isdigit()
        for k in t.attrs
    ):
        return None
    sels = [_seq_step_select(a) for a in t.args]
    if any(s is None for s in sels):
        return None
    children = []
    d0s = []
    idxs = []
    shapes = []
    r = None
    for cc, dd, ii in sels:
        sh = shape_of(cc)
        if sh is INVALID or not isinstance(sh, tuple) or not sh:
            return None
        rr = len(sh)
        if r is None:
            r = rr
        elif rr != r:
            return None
        d0 = _seq_axis_idx(dd, rr)
        if d0 is None:
            return None
        children.append(cc)
        d0s.append(d0)
        idxs.append(ii)
        shapes.append(sh)
    if r is None or len(set(d0s)) != 1 or len(set(idxs)) != 1:
        return None
    d0 = d0s[0]
    idx = idxs[0]
    # concat dim is spelled in the rank-(r-1) argument/output coords
    cd_raw = t.attrs.get("dim", -1)
    cd = _seq_axis_idx(cd_raw, r - 1)
    if cd is None:
        return None
    # joined axis in child coords — identical for every arg since
    # all selects sit on the same child axis ``d0``
    jc = cd if cd < d0 else cd + 1
    # the lifted concat must itself be well-formed: every non-joined
    # child axis (including the shared step axis) must agree
    for k in range(r):
        if k == jc:
            continue
        if len({sh[k] for sh in shapes}) != 1:
            return None
    lifted = Op.make(
        "concat",
        *children,
        dim=jc,
        **{k2: v for k2, v in t.attrs.items() if k2 != "dim"},
    )
    if not _seq_select_free(lifted):
        return None
    sh2 = shape_of(lifted)
    if (
        sh2 is INVALID
        or not isinstance(sh2, tuple)
        or not 0 <= d0 < len(sh2)
    ):
        return None
    return lifted, d0, idx


def _seq_bcast(sh1: tuple, sh2: tuple):
    """Right-aligned broadcast of two shapes; ``None`` on conflict."""
    if len(sh1) < len(sh2):
        sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
    elif len(sh2) < len(sh1):
        sh2 = (1,) * (len(sh1) - len(sh2)) + sh2
    out = []
    for x, y in zip(sh1, sh2, strict=True):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            return None
    return tuple(out)


def _seq_deindex(
    t: Any,
    names: dict,
    operands: dict,
    events: list,
    state: Any = None,
    consts: dict | None = None,
    pullout: bool = False,
):
    """Spec of *t* with per-step ``select``s abstracted to metas.

    Every ``select(child, dim=D, index=i)`` node becomes a sequence
    operand metavar bound to *child* — recorded in *operands* as
    ``meta -> (child, dim)`` and in *events* as ``(meta, index)`` so
    the caller can verify the index walks the step axis in step
    order.  ``rsub(x, s)`` normalises to ``sub(s, x)`` — the
    elementwise spelling the scan renderer knows.  Any other bare
    leaf or non-scalar ``Const`` declines the whole term with
    ``None`` — EXCEPT ``Param``/``Var`` leaves when *consts* is
    given: they are loop-invariant, so they mint ``"P"`` metas
    bound whole (recorded in *consts*, never in *operands*/*events*
    — the fold reads the same value every step).

    ``unsqueeze``/``squeeze`` chains wrapping a ``select`` are
    lifted onto the child first (:func:`_seq_view_lift`) — the
    metavar then binds the *viewed* sequence and no view op reaches
    the step-expr skeleton.  With *pullout*, a ``select`` sitting
    DIRECTLY inside ``matmul``/``linear`` is pulled through the
    projection (:func:`_seq_projection_lift`) and one inside any
    other axis-safe op through the commutation lift
    (:func:`_seq_commute_lift`) — the operand binds the hoisted
    whole-tensor op result, never a per-step slice.

    *state* (matrix-state form only): the exact subterm the fold's
    accumulator denotes at this step — an occurrence of it inside
    the step expression abstracts to the ``"H"`` metavar instead of
    recursing into the previous chain link.  The check is *equality*,
    not identity: ``Op.make`` hash-conses ops, but ``Var``/``Param``
    leaves are plain frozen dataclasses — a second, structurally
    equal ``Var("H0", …)`` minted by a sibling recurrence interns the
    SAME ``matmul`` node while the spine's state factor stays the
    other object, so ``t is state`` silently misses (measured: a
    gated recurrence built first made the plain one's site decline).
    Structural equality is also the *sound* reading — an equal term
    denotes the same value the accumulator holds at this step.
    """
    if state is not None and t == state:
        return "H"
    if isinstance(t, Op):
        if t.op in ("unsqueeze", "squeeze") and len(t.args) == 1:
            lifted = _seq_view_lift(t)
            if lifted is not None:
                child, dim, idx = lifted
                return _seq_meta(
                    names, operands, events, child, dim, idx
                )
        if (
            t.op == "select"
            and len(t.args) == 1
            and type(t.attrs.get("dim")) is int
            and type(t.attrs.get("index")) is int
        ):
            child, dim, idx = (
                t.args[0],
                t.attrs["dim"],
                t.attrs["index"],
            )
            return _seq_meta(names, operands, events, child, dim, idx)
        if t.op == "rsub" and len(t.args) == 2 and not t.attrs:
            b = _seq_deindex(
                t.args[1],
                names,
                operands,
                events,
                state=state,
                consts=consts,
                pullout=pullout,
            )
            a = _seq_deindex(
                t.args[0],
                names,
                operands,
                events,
                state=state,
                consts=consts,
                pullout=pullout,
            )
            if a is None or b is None:
                return None
            return ("sub", b, a)
        if pullout:
            proj = _seq_projection_lift(t)
            if proj is None:
                proj = _seq_commute_lift(t)
            if proj is not None:
                child, dim, idx = proj
                return _seq_meta(
                    names, operands, events, child, dim, idx
                )
        parts = []
        for a_ in t.args:
            p = _seq_deindex(
                a_,
                names,
                operands,
                events,
                state=state,
                consts=consts,
                pullout=pullout,
            )
            if p is None:
                return None
            parts.append(p)
        spec = [t.op, *parts]
        if t.attrs:
            spec.append(dict(t.attrs))
        return tuple(spec)
    if isinstance(t, Const):
        return None if isinstance(t.value, str) else t.value
    if consts is not None and isinstance(t, (Param, Var)):
        # loop-invariant leaf — a weight or input the fold reads
        # whole at every step; recorded in *consts*, not *operands*
        return _seq_const_meta(consts, t)
    return None


def _seq_site(steps: tuple, base: Any):
    """Seq-form spec record for one spine, or ``None``.

    The fold ``acc = a_t * acc + b_t`` reproduces the unrolled chain
    exactly when every step's ``a``/``b`` deindex to the SAME spec
    skeleton — per-step ``select``s becoming sequence operands —
    and each select's index equals its chronological position along
    a single shared step axis whose extent is the step count.  The
    shape check via ``shape_of`` is what makes the axis claim
    sound: equal-rank operands with the step extent at one dim mean
    the fold's broadcast slice reads exactly the term the original
    ``select`` read.  Anything unverifiable declines honestly.
    """
    from mlopt_core.typing import INVALID, shape_of

    names: dict = {}
    operands: dict = {}
    consts: dict = {}
    a0 = b0 = None
    events: list = []
    for k, (at, bt, _st) in enumerate(steps):
        ea: list = []
        eb: list = []
        sa = _seq_deindex(
            at, names, operands, ea, consts=consts, pullout=True
        )
        sb = _seq_deindex(
            bt, names, operands, eb, consts=consts, pullout=True
        )
        if sa is None or sb is None:
            return None
        if k == 0:
            a0, b0 = sa, sb
        elif sa != a0 or sb != b0:
            return None
        events.append(ea + eb)
    if not operands:
        return None
    dims = {d for _c, d in operands.values()}
    if len(dims) != 1:
        return None
    dim = dims.pop()
    t_len = len(steps)
    rank = None
    eff = dim
    state_sh: tuple | None = None
    for _m, (child, _d) in operands.items():
        sh = shape_of(child)
        if sh is INVALID or not isinstance(sh, tuple):
            return None
        if rank is None:
            rank = len(sh)
            eff = dim if dim >= 0 else dim + rank
        if (
            len(sh) != rank
            or not -rank <= dim < rank
            or sh[eff] != t_len
        ):
            return None
        parts = sh[:eff] + sh[eff + 1 :]
        state_sh = (
            parts if state_sh is None else _seq_bcast(state_sh, parts)
        )
        if state_sh is None:
            return None
    if state_sh is None:
        return None  # operands is non-empty — unreachable, narrows ty
    for k, evs in enumerate(events):
        for _m, idx in evs:
            if idx != k:
                return None
    # Loop-invariant leaves minted as ``"P"`` metas must broadcast
    # into the fold's state shape — the same contract the per-step
    # ``_eval_spec``/kernel broadcast applies at run time.  A const
    # that cannot is not a per-step operand the fold can read
    # honestly; decline rather than emit a spec that raises on call.
    for t_const, _m in consts.items():
        csh = shape_of(t_const)
        if (
            csh is INVALID
            or not isinstance(csh, tuple)
            or _seq_bcast(csh, state_sh) != state_sh
        ):
            return None
    h_meta = "H"
    attrs: dict = {"kind": "seq", "axis": eff}
    if consts:
        attrs["const"] = tuple(consts.values())
    spec = (
        "applyd",
        ("aff_diag", a0, b0),
        h_meta,
        attrs,
    )
    leaf_env = {m: c for m, (c, _d) in operands.items()}
    leaf_env.update({m: t for t, m in consts.items()})
    leaf_env[h_meta] = base
    return spec, leaf_env, eff


def _seq_scan_sites(term: Any) -> list:
    """Seq-form scan specs minted from raw recurrence chains.

    The unrolled collectors above only see what saturation already
    built; the *sequence* form — ``applyd(aff_diag(a, b), h)`` whose
    operands carry a step axis and whose declared op is the fold
    ``acc = a_t * acc + b_t`` — has a different, earlier source: the
    exported recurrence itself.  Models spell ``h_t = a_t ⊙ h_{t-1}
    + b_t`` as an unrolled ``add(mul(·, ·), ·)`` chain over per-step
    ``select(x, dim, t)`` slices; this collector walks those chains,
    abstracts the selects into sequence operands and mints the seq
    spec whose fold IS the chain — sound by construction, since the
    spec is only emitted when the deindexed skeletons agree step for
    step and the select indices walk the axis in order.

    Sites are ``{"spec", "leaf_env", "axis"}`` records: ``leaf_env``
    binds each operand metavar to its full sequence tensor (the
    ``select`` child) and ``H`` to the initial-state term, and
    ``axis`` is the resolved step axis — merged into the handler
    entry so ``handler_denotation`` and the Triton binding agree on
    it.  Chains that are strict prefixes of a longer emitted chain
    are skipped (the maximal chain covers them); ambiguous or
    unverifiable readings decline rather than guess.
    """
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    memo: dict = {}
    cands: list = []
    seen_nodes: set = set()

    def visit(t: Any) -> None:
        if not isinstance(t, Op) or id(t) in seen_nodes:
            return
        seen_nodes.add(id(t))
        for steps, base in _seq_spines(t, memo):
            if len(steps) >= 2:
                cands.append((t, steps, base))
        for a in t.args:
            visit(a)

    visit(term)
    out: list = []
    emitted_roots: set = set()
    emitted_states: set = set()
    for root, steps, base in cands:
        if id(root) in emitted_roots or id(root) in emitted_states:
            continue
        site = _seq_site(steps, base)
        if site is None:
            continue  # next tied reading of the same root
        spec, leaf_env, axis = site
        rec = {
            "spec": spec,
            "leaf_env": leaf_env,
            "axis": axis,
            # the unrolled chain this fold denotes — the
            # deliver-time route's LHS (the intake IR has no
            # ``applyd`` node for a claim to match, so the
            # rewrite fires on the extracted term instead).
            "chain": root,
        }
        # dedup on the whole record — spec AND leaf_env AND chain —
        # not the spec tuple alone: the fold's denotation binds the
        # ``leaf_env`` operands, so two same-shaped chains over
        # different sequences (e.g. a bidirectional scan's forward
        # and flipped backward recurrences) are two distinct folds
        # that must each mint a site.
        if rec not in out:
            out.append(rec)
        for _m, _i, s in steps:
            emitted_states.add(id(s))
        emitted_roots.add(id(root))
        if len(out) >= _SEQ_SITE_CAP:
            break
    return out


def _spec_has_meta(spec: Any, meta: str) -> bool:
    """Whether *meta* occurs as a leaf of spec *spec*."""
    if spec == meta:
        return True
    if isinstance(spec, (tuple, list)):
        return any(
            _spec_has_meta(e, meta)
            for e in spec[1:]
            if not isinstance(e, dict)
        )
    return False


def _seqm_site(steps: tuple, base: Any):
    """Seq-matrix spec record for one spine, or ``None``.

    The matrix-state analogue of :func:`_seq_site`: the fold is
    ``acc = a_t * acc + b_t(acc)`` where the state term may occur
    inside *b_t* — the delta-rule read ``kᵀ·acc`` — abstracted to
    the ``"H"`` metavar by :func:`_seq_deindex`'s *state* argument.
    Steps spelled ``h' = h + f(h)`` carry ``at is None`` (the
    implicit unit map); ``a_t`` must never depend on the
    accumulator, and *b_t* must — a state-free ``b`` is the plain
    seq form's job (or nothing's).

    The operand checks differ from :func:`_seq_site` in ONE way:
    there is no common broadcast frame.  Matrix-state operands live
    in different spaces (``k ∈ R^dk``, ``v ∈ R^dv``, ``acc ∈
    R^(dk,dv)``) that only meet inside the step expr, so instead of
    equal-rank/single-frame ``shape_of`` checks each operand is
    verified on its own: its select dim is in range and its extent
    there is the step count.  The per-operand axis is recorded in
    ``axes`` — the binding and denotation slice each operand at ITS
    own normalized dim, exactly what the original ``select`` read.
    Everything else — identical skeletons step to step, index ==
    chronological position — is the same soundness argument.
    """
    from mlopt_core.typing import INVALID, shape_of

    names: dict = {}
    operands: dict = {}
    axes: dict = {}
    a0 = b0 = None
    events: list = []
    for k, (at, bt, st) in enumerate(steps):
        ea: list = []
        eb: list = []
        sa = (
            1
            if at is None
            else _seq_deindex(at, names, operands, ea, state=st)
        )
        sb = _seq_deindex(bt, names, operands, eb, state=st)
        if sa is None or sb is None:
            return None
        if _spec_has_meta(sa, "H"):
            return None  # a_t must be accumulator-free
        if k == 0:
            a0, b0 = sa, sb
        elif sa != a0 or sb != b0:
            return None
        events.append(ea + eb)
    if not operands or not _spec_has_meta(b0, "H"):
        return None
    t_len = len(steps)
    for m, (child, d) in operands.items():
        sh = shape_of(child)
        if sh is INVALID or not isinstance(sh, tuple) or not sh:
            return None
        eff = d if d >= 0 else d + len(sh)
        if not 0 <= eff < len(sh) or sh[eff] != t_len:
            return None
        axes[m] = eff
    for k, evs in enumerate(events):
        for _m, idx in evs:
            if idx != k:
                return None
    a_spec = a0 if a0 is not None else 1
    spec = (
        "applyd",
        ("aff_diag", a_spec, b0),
        "H",
        {"kind": "seqm", "axis": next(iter(axes.values()))},
    )
    leaf_env = {m: c for m, (c, _d) in operands.items()}
    leaf_env["H"] = base
    return spec, leaf_env, axes


def _seqm_scan_sites(term: Any) -> list:
    """Seq-matrix specs minted from state-feedback recurrence chains.

    Same traversal and caps as :func:`_seq_scan_sites`; the site
    record additionally carries ``axes`` — each operand metavar's
    normalised step axis — which :func:`_seqm_site` resolved per
    operand (there is no shared frame to derive it from).  Records
    are ``{"spec", "leaf_env", "axis", "axes"}``.
    """
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    memo: dict = {}
    cands: list = []
    seen_nodes: set = set()

    def visit(t: Any) -> None:
        if not isinstance(t, Op) or id(t) in seen_nodes:
            return
        seen_nodes.add(id(t))
        for steps, base in _seq_spines(t, memo):
            if len(steps) >= 2:
                cands.append((t, steps, base))
        for a in t.args:
            visit(a)

    visit(term)
    out: list = []
    emitted_roots: set = set()
    emitted_states: set = set()
    for root, steps, base in cands:
        if id(root) in emitted_roots or id(root) in emitted_states:
            continue
        site = _seqm_site(steps, base)
        if site is None:
            continue
        spec, leaf_env, axes = site
        rec = {
            "spec": spec,
            "leaf_env": leaf_env,
            "axis": next(iter(axes.values())),
            "axes": axes,
            # see ``_seq_scan_sites``: the chain root is the
            # deliver-time route's LHS.
            "chain": root,
        }
        # whole-record dedup, same contract as ``_seq_scan_sites``:
        # the spec tuple does not see ``leaf_env``, so same-shaped
        # chains over different operands would collapse into one
        # site and the second fold would never be offered.
        if rec not in out:
            out.append(rec)
        for _m, _i, s in steps:
            emitted_states.add(id(s))
        emitted_roots.add(id(root))
        if len(out) >= _SEQ_SITE_CAP:
            break
    return out


def _cumsum_sites(term: Any) -> list:
    """Emitted-scan specs minted from ``cumsum``/``cumprod`` ops.

    The select-chain collectors read recurrences spelled as unrolled
    ``add(mul(·, ·), ·)`` chains; a ``cumsum`` op is ONE node — no
    spine, no site — even though the prefix sum IS the same
    recurrence with ``a ≡ 1`` emitted at every step.  This collector
    reads the op directly and mints the ``kind="seqs"`` spec whose
    fold denotes every prefix state (``out_t = acc_t``), the form
    ``genkernel.seqs_denotation``/the ``seqs`` Triton piece implement.

    ``cumsum(x, dim)`` spells ``acc_t = 1·acc_{t-1} + x_t`` from a
    zero carry; ``cumprod`` spells ``acc_t = x_t·acc_{t-1} + 0`` from
    a unit carry — the carry is a ``Const`` leaf in ``leaf_env``
    (the scan's neutral element, not a guessed parameter).  The
    ``"H"`` metavar binds it; ``"X1"`` binds the scanned child and
    ``axis`` the normalised step dim (``shape_of``-checked, the same
    honesty contract the seq collectors keep).  ``chain`` is the
    ``cumsum``/``cumprod`` node itself — delivery rewrites it to the
    minted kernel call exactly as the unrolled chains are routed.
    """
    from mlopt_core.ir import Const, Op
    from mlopt_core.typing import INVALID, shape_of

    out: list = []
    seen: set = set()

    def visit(t: Any) -> None:
        if not isinstance(t, Op) or id(t) in seen:
            return
        seen.add(id(t))
        for a in t.args:
            visit(a)
        if (
            t.op not in ("cumsum", "cumprod")
            or len(t.args) != 1
            or set(t.attrs) - {"dim"}
            or type(t.attrs.get("dim")) is not int
        ):
            return
        child = t.args[0]
        sh = shape_of(child)
        if sh is INVALID or not isinstance(sh, tuple) or not sh:
            return
        eff = t.attrs["dim"] % len(sh)
        if t.op == "cumsum":
            a_spec, b_spec, carry = 1, "X1", 0.0
        else:
            a_spec, b_spec, carry = "X1", 0, 1.0
        spec = (
            "applyd",
            ("aff_diag", a_spec, b_spec),
            "H",
            {"kind": "seqs", "axis": eff},
        )
        out.append(
            {
                "spec": spec,
                "leaf_env": {"X1": child, "H": Const(carry)},
                "axis": eff,
                "chain": t,
            }
        )

    visit(term)
    return out[:_SEQ_SITE_CAP]


def _all_scan_sites(term: Any) -> list:
    """All scan site forms: carriers + raw recurrences, both folds."""
    return [
        *_scan_sites(term),
        *_seq_scan_sites(term),
        *_seqm_scan_sites(term),
        *_cumsum_sites(term),
    ]


#: Site collectors, keyed by codegen-tier name — the term-traversal
#: knowledge that pairs with ``genkernel.TIERS``.  A tier without a
#: collector mints nothing here; the ``scan`` collector joins the
#: saturated-graph carriers with the raw recurrence chains that mint
#: the sequence form.
TIER_SITES: dict[str, Any] = {
    "gen": _fusion_sites,
    "decayed": _decayed_sites,
    "scan": _all_scan_sites,
    "reduction": _reduction_sites,
}


def _mint_tiers(root: Any, covered: set) -> dict:
    """Mint handler entries across every codegen tier.

    The registry walk replacing per-tier call sites: each tier's
    site collector gathers specs, its ``handlers`` builder mints
    the ``{tag: {pattern, kernel, args}}`` table — a tier without
    a collector contributes nothing.
    """
    from mlopt_discovery import genkernel as gk

    out: dict = {}
    for tier in gk.TIERS:
        sites_fn = TIER_SITES.get(tier.name)
        if sites_fn is not None:
            out.update(tier.handlers(sites_fn(root), covered=covered))
    return out


def _flat_ops(spec: Any):
    """Yield op names inside a fusion pattern tuple."""
    if isinstance(spec, (tuple, list)) and spec:
        yield spec[0]
        for e in spec[1:]:
            if not isinstance(e, dict):
                yield from _flat_ops(e)


def _gen_cases(
    seed: int,
    n: int,
    *,
    builders: Any = None,
    ambient_handlers: Any = None,
) -> list:
    """Torch modules over fusion-spelling and stock-model builders.

    The fusion builders spell pointwise patterns the ambient
    handler table has no fused op for — the candidates ``claim``
    can only reach through a *generated* kernel.  Stock pieces
    (mlp/attn chains) give the board real shape.

    *builders* (domain-spec seam) selects builder names from the
    table below; *ambient_handlers* substitutes the shipped
    ``lawdata.HANDLERS`` table the cases seat (``None`` keeps it).
    """
    import torch
    import torch.nn as nn
    from mlopt_torch.adapters import TorchSource

    from mlopt_discovery import genkernel as gk
    from mlopt_discovery import lawdata
    from mlopt_discovery.meta_arena import _canon_concrete

    class _Fuse(nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            self.kind = kind

        def forward(self, x):
            """Spell the pointwise pattern — no fused op named."""
            if self.kind == "mulrelu":
                return x * torch.relu(x)
            if self.kind == "mulgelu":
                return x * torch.nn.functional.gelu(x)
            if self.kind == "mulsoftplus":
                return x * torch.nn.functional.softplus(x)
            if self.kind == "bigfuse":
                return (
                    x * torch.relu(x)
                    + torch.sigmoid(x)
                    - torch.tanh(x) * torch.abs(x)
                    + x * x
                )
            return x * torch.tanh(x)

    class _SwigluMLP(nn.Module):
        """Stock llama-MLP shape: ``down(silu(gate) * up)`` — spelled."""

        def __init__(self, d: int = 64) -> None:
            super().__init__()
            self.gate = nn.Linear(d, d * 2, bias=False)
            self.up = nn.Linear(d, d * 2, bias=False)
            self.down = nn.Linear(d * 2, d, bias=False)

        def forward(self, x):
            """Spell the swiglu — the site a claim can name."""
            g = self.gate(x)
            return self.down(g * torch.sigmoid(g) * self.up(x))

    class _ChainMLP(nn.Module):
        """A spelled matmul chain + pointwise tail — the reassoc story."""

        def __init__(self, d: int = 64) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.b = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.c = nn.Parameter(torch.randn(d, d) / d**0.5)

        def forward(self, x):
            """Three spelled mms, then a pointwise tail."""
            y = x @ self.a @ self.b @ self.c
            return y * torch.relu(y)

    class _TwoIslands(nn.Module):
        """Two fusion sites separated by matmuls — the composition case.

        No single elementwise-connex pattern covers both tails, so
        one extraction needs *two* minted kernels: ``claim`` each
        site and the delivered module calls both ``gen_*_k`` — and a
        claimed op can feed another claim's metavar (the inner
        fusion's e-class member is what the outer kernel receives).
        """

        def __init__(self, d: int = 128) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.v = nn.Parameter(torch.randn(d, d) / d**0.5)

        def forward(self, x):
            """Chain → fuse → chain → fuse, tails rejoined."""
            h = x @ self.w
            a = h * torch.relu(h)
            y = a @ self.v
            b = y * torch.tanh(y)
            return b + a

    class _ChainFuse(nn.Module):
        """Fold-able mm chain + a 6-op pointwise tail needing gen.

        The folded chain is a laws win; the tail's fused form exists
        only via ``claim(gen_*)`` — the win that needs the generated
        kernel.
        """

        def __init__(self, d: int = 512) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.b = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.c = nn.Parameter(torch.randn(d, d) / d**0.5)

        def forward(self, x):
            """Chain, then a long elementwise tail over the result."""
            y = x @ self.a @ self.b @ self.c
            return (
                y * torch.relu(y)
                - torch.nn.functional.softplus(y)
                + torch.tanh(y * y)
                * torch.nn.functional.softplus(y * torch.sigmoid(y))
            )

    rng = random.Random(seed)
    src = TorchSource()
    ambient = (
        lawdata.HANDLERS
        if ambient_handlers is None
        else ambient_handlers
    )
    covered = {_canon_concrete(h["pattern"]) for h in ambient.values()}
    table = [
        ("mulrelu", lambda: _Fuse("mulrelu"), (256, 512)),
        ("mulgelu", lambda: _Fuse("mulgelu"), (256, 512)),
        ("mulsoftplus", lambda: _Fuse("mulsoftplus"), (256, 512)),
        ("multanh", lambda: _Fuse("multanh"), (256, 512)),
        ("bigfuse", lambda: _Fuse("bigfuse"), (256, 512)),
        ("swiglu_mlp", _SwigluMLP, (16, 64)),
        ("chain_mlp", _ChainMLP, (16, 64)),
        ("chainfuse", _ChainFuse, (256, 512)),
        ("twosite", _TwoIslands, (256, 128)),
        (
            "mlp",
            lambda d=64: nn.Sequential(
                nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d)
            ),
            (8, 64),
        ),
    ]
    if builders is not None:
        known = {b[0] for b in table}
        unknown = sorted(set(builders) - known)
        if unknown:
            raise ValueError(
                f"corpus.builders: unknown gen builder {unknown!r} "
                f"(known: {sorted(known)!r})"
            )
        wanted = set(builders)
        table = [b for b in table if b[0] in wanted]
    out = []
    for i in range(n):
        name, build, shape = table[rng.randrange(len(table))]
        model = build().eval()
        gen = torch.Generator().manual_seed(seed * 997 + i)
        x = torch.randn(*shape, generator=gen)
        ir, leaves = src.to_ir(model, x)
        genh = _mint_tiers(ir.root, covered)
        sink = gk.gen_sink(genh, compile_kernels=False)
        handlers = dict(ambient) | genh
        out.append(
            (
                f"gen{i}:{name}",
                ir.root,
                {"gen": len(genh)},
                name,
                sink.supported_ops,
                {
                    "ir": ir,
                    "leaves": leaves,
                    "model": model,
                    "input": x,
                    "sink": sink,
                    "handlers": handlers,
                    "gen_handlers": genh,
                    "ambient_handlers": ambient,
                },
            )
        )
    return out


def _var_env(ir: Any, leaves: dict, feed: Any) -> dict:
    """Build the metavar→value env ``profitable`` times against.

    Param leaves plus the input vars bound positionally to the
    feed — a param-only env makes every site touching the feed
    eval-fail and silently skip.
    """
    feeds = tuple(feed) if isinstance(feed, (tuple, list)) else (feed,)
    var_env = dict(leaves)
    var_env.update(
        {v.name: t for v, t in zip(ir.inputs, feeds, strict=False)}
    )
    return var_env


def _fold_cost_gate(case: Any, handlers: dict) -> dict:
    """Judge shipped prim folds by measured delivery, not op-count.

    T-019: the ``prim:synop*`` folds ride ``DEFAULT`` into saturation
    unconditionally — no claim offer, no ``profitable`` consult — so
    a folded member won extraction on the cost model while its
    delivered body lost wall-clock (the p26 re-sweep: 7 prim rows,
    median 0.943 → 0.744).  This is the same referee claims pass
    through, applied to the fold entries before they join the
    board's ``supported`` bound:

    * an entry whose pattern has **no concrete site** in the term
      cannot fire — it keeps its seat (it costs nothing);
    * an entry **with a site** is timed by ``profitable`` — the
      delivered binding (``kernel_for`` = the ``hybrid_sink`` codegen
      path) against the spelled denotation on the case's real
      values — and only a measured win stays feasible;
    * an entry the referee **cannot evaluate** — no concrete
      occurrence — keeps its seat: the gate refuses a *measured*
      loss, it does not invent one.  The attr-metavar specs (5 of
      57) DO get sited: ``genkernel.concretize_at_site`` binds the
      attr metavars from the matched site's real attrs and the
      referee times the frozen denotation — the same callable the
      delivery path's ``spec_bind`` produces for that site.  The
      *delivered* table keeps the polymorphic original; only the
      measurement sees the freeze.
    """
    from mlopt_discovery import genkernel as gk

    term = case[1]
    ex = case[5]
    ungated: dict[str, dict] = {}
    measured: dict[str, dict] = {}
    for name, h in handlers.items():
        if gk._handler_env(h, term) is not None:
            measured[name] = h
        elif h.get("attr_mvars"):
            concrete = gk.concretize_at_site(h, term)
            if concrete is None:
                ungated[name] = h
            else:
                measured[name] = concrete
        else:
            ungated[name] = h
    if not measured:
        return ungated
    keep = gk.profitable(
        measured,
        term,
        _var_env(ex["ir"], ex["leaves"], ex["input"]),
        reps=20,
        kernel_for=_triton_kernel_for,
    )
    # ``keep`` holds the *concretized* copies — the delivered table
    # must carry the polymorphic originals (one site's frozen attrs
    # would mis-denote another site's call).
    return {
        **ungated,
        **{name: handlers[name] for name in keep},
    }


def _claim_measurable(tag: str, h: dict) -> bool:
    """Check a minted claim's spelled body is an honest denotation.

    The shared predicate behind both cost gates: ``gen_*``
    pointwise and ``decayed_*`` bodies evaluate through the
    ordinary bindings (except ``om_decayed``-headed specs, whose
    spelled side lives in ``_MEASURED_BINDINGS``, not
    ``_IR_TO_TORCH`` — unmeasurable stays unpriced); ``red_*``
    bodies evaluate through the ordinary bindings — the spelled
    spec IS the denotation (the reduction tier has no separate
    codegen; delivery is the compiled spelled body), so they are
    measurable like ``gen_*``/``decayed_*``.  ``scan_*`` claims
    are measurable in BOTH forms: *unrolled* compose-tree specs'
    spelled eval IS the carrier semantics, and *sequence*-form
    scans carry a registered spelled fold denotation
    (``genkernel.handler_denotation`` /
    ``seq_scan_denotation``) that ``profitable``'s
    ``denotation_for`` hook and ``gen_bindings`` both use — the
    ambient map application is never the referee's or the
    sink's stand-in for the fold.
    """
    from mlopt_discovery import genkernel as gk

    if tag.startswith(("gen_", "decayed_", "red_")):
        return "om_decayed" not in gk._spec_ops(h.get("pattern"))
    return tag.startswith("scan_")


def _claim_cost_gate(case: Any) -> None:
    """Drop minted ``gen_*`` claims that don't pay — the offer guard.

    ``_zoo_case`` already filters minted handlers through
    ``genkernel.profitable`` at intake; the synthetic ``_gen_cases``
    stream minted every fusion site *unpriced*, so ``claim`` offers
    included kernels that lose to their spelled body — the 0.38x /
    0.86x regimes the genkernel retro measured.  This applies the
    same measured-cost referee to the pointwise ``gen_*`` tier once
    per case, mutating ``ex["gen_handlers"]`` before the board's
    handler store is built: an unprofitable claim is never offered,
    so extraction cannot pick it and ``claim_tags`` never names it.

    Scope is every minted tier whose spelled body is an honest
    denotation of the claim:

    * ``gen_*`` pointwise fusions — the spelled spec-walk IS the
      unfused body;
    * ``decayed_*`` — the spelled ``matmul(mul(s, d), v)`` evaluates
      through the ordinary bindings, so kernel-vs-spelled is the
      same comparison the delivery ships (this is the offer that
      lost 0.47x on twosite before the gate existed).  ``om_decayed``-
      headed specs are excluded: that op lives in
      ``_MEASURED_BINDINGS``, not ``_IR_TO_TORCH``, so the referee's
      spelled side cannot evaluate them — unmeasurable stays
      unpriced rather than silently dropped;
    * ``scan_*`` — BOTH forms are measurable now.  *Unrolled*
      compose-tree specs' spelled eval IS the carrier semantics
      (``hybrid_sink``'s own rule).  *Sequence*-form scans carry a
      registered spelled fold denotation —
      ``genkernel.seq_scan_denotation`` spells the recurrence in
      plain torch and ``gen_bindings`` delivers it wherever the
      Triton kernel does not bind — so ``profitable``'s
      ``denotation_for`` hook times the generated fold kernel
      against the fold itself, never the ambient map application
      the ``("applyd", …)`` spec tuple evals to.  A seq claim
      with no bound codegen declines outright (delivery IS the
      denotation — there is no faster arm to beat);
    * ``red_*`` reduction-bodied fusions — the spelled
      ``mean``/``sum``/``softmax``-headed body evaluates through the
      ordinary bindings and IS the denotation (the tier ships the
      compiled spelled body, no separate codegen);
    * the shipped ``prim:*`` folds already pass
      :func:`_fold_cost_gate` on the same board build.

    ``_zoo_case`` sets ``ex["_gen_claims_gated"]`` — its intake
    ``profitable`` sweep already judged every minted tier, so the
    board does not pay the referee twice.
    """
    from mlopt_discovery import genkernel as gk

    ex = case[5]
    if ex.get("_gen_claims_gated"):
        return
    ex["_gen_claims_gated"] = True

    minted = {
        k: v
        for k, v in ex["gen_handlers"].items()
        if _claim_measurable(k, v)
    }
    if not minted:
        return
    keep = gk.profitable(
        minted,
        case[1],
        _var_env(ex["ir"], ex["leaves"], ex["input"]),
        reps=20,
        kernel_for=_triton_kernel_for,
        denotation_for=_claim_denotation_for,
    )
    dropped = set(minted) - set(keep)
    if not dropped:
        return
    ex["gen_handlers"] = {
        k: v for k, v in ex["gen_handlers"].items() if k not in dropped
    }
    # reseat the delivery sink on the filtered table — the same
    # consistency the prim fold gate maintains
    if ex.get("sink") is not None:
        ex["sink"] = gk.gen_sink(
            ex["gen_handlers"], compile_kernels=False
        )


def _gen_board(case: Any) -> ma.MetaArena:
    """Board a gen case under its generated handler table.

    The bank merges the shipped ``synop*`` primitives' denotations as
    generated-kernel handler rows — one entry per minted op, kernel =
    the op itself.  The ``prim:synop*`` folds already ride ``DEFAULT``
    into saturation, but a folded ``synop_*`` member could never win
    extraction while no sink could lower it: the merged table is what
    makes the ops *feasible* (``supported``) and *deliverable* —
    ``gen_probe``'s ``hybrid_sink`` reads the same ``ex["gen_handlers"]``
    and binds each op to a fused-body codegen (``triton_bindings``
    where the denotation is elementwise, the ``torch.compile``d spelled
    body otherwise), not the spec-walk binding
    ``shipped_primitives.register()`` seats.  The merge mutates the
    case record so the delivery sink sees the identical table, and
    reseats ``ex["sink"]`` on it for the gen domain's deliver arm.
    """
    from mlopt_core.cells import Provenance
    from mlopt_core.cost.basic import count_cost
    from mlopt_core.laws import DEFAULT

    from mlopt_discovery import cellstore as cs
    from mlopt_discovery import genkernel as gk
    from mlopt_discovery import lawdata
    from mlopt_discovery import shipped_primitives as sp

    ex = case[5]
    _claim_cost_gate(case)

    def novel(op: str) -> bool:
        return op.startswith(("claim_", "foldabs_", "gen_"))

    prim = sp.handler_entries()
    if not any(tag in ex["gen_handlers"] for tag in prim):
        # T-019 fold cost gate: the ``prim:*`` folds fire
        # unconditionally through ``DEFAULT``, so the measured-cost
        # referee ``profitable`` applies to claims must also judge
        # the folds — a ``synop_*`` member stays feasible only when
        # its *delivered* binding beats the spelled site it replaces
        # on the case's real values (the 7 prim-delivering zoo rows
        # measured 0.744x median ungated — extraction preferred
        # members that lose wall-clock).
        ex["gen_handlers"] = {
            **ex["gen_handlers"],
            **_fold_cost_gate(case, prim),
        }
        if "sink" in ex:
            ex["sink"] = gk.gen_sink(
                ex["gen_handlers"], compile_kernels=False
            )
    kept = {k: v for k, v in prim.items() if k in ex["gen_handlers"]}
    supported = frozenset(case[4]) | {
        h["kernel"] for h in kept.values()
    }
    ambient = ex.get("ambient_handlers", lawdata.HANDLERS)
    store = cs.ambient_store(rules=DEFAULT, handlers=ambient).merge(
        cs.handlers_fragment(
            "gen",
            ex["gen_handlers"],
            provenance=Provenance("minted"),
        )
    )
    return ma.MetaArena(
        case[1],
        supported=supported,
        corpus=[case[1]],
        cost_fn=gk.novel_cost(count_cost, novel, 0.5),
        store=store,
    )


def _probe_play(board: Any) -> Any:
    """Play the maximal automated line; return the best extraction.

    Every claim offer, then ``saturate`` (the shipped laws —
    reassociation &c.), then ``extract``.
    """
    from mlopt_discovery import meta_arena as ma

    st = board.observe()
    for a in [a for a in ma.legal_actions(st) if a.op == "claim"]:
        board.step(a)
    board.step(ma.Action.saturate())
    board.step(ma.Action.extract())
    return board.eg.extract_best(board.root, board.feasible_cost)


def gen_probe(case: Any, *, budget: int = 12, reps: int = 200) -> dict:
    """Measure the novelty claim: baseline vs generated-kernel module.

    For every claim the board offers: play it, extract, deliver
    through the case's gen sink (bindings compiled — the fused
    kernel is a real inductor graph), verify numerically against
    the source module, and time both sides.  The row answers the
    product question: did the search produce a *runnable module
    that is faster* than the spelled source, not just a new
    spelling.
    """
    import torch

    from mlopt_discovery import genkernel as gk

    # measurement integrity: each gen kernel is a distinct code
    # object recompiled under varying global state; dynamo's
    # default limit (8) would silently degrade late compiles —
    # and late *baseline* compiles — to eager.  Raise it so both
    # sides are measured as compiled, not corrupted by the cache.
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)

    ex = case[5]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    board = _gen_board(case)
    tags = list(board.observe().claim_tags)
    best = _probe_play(board)
    rows: list[dict] = []
    base_model = ex["model"].to(device)
    xin = (
        tuple(t.to(device) for t in ex["input"])
        if isinstance(ex["input"], (tuple, list))
        else ex["input"].to(device)
    )
    try:
        t_base = _time_module(base_model, xin, reps)
    except Exception as exc:
        return {
            "case": case[0],
            "delivered": False,
            "note": f"baseline failed: {type(exc).__name__}: {exc}",
        }
    if best is None:
        return {"case": case[0], "delivered": False, "claims": tags}
    import torch as _torch

    compiled = _torch.compile(base_model)
    t_comp = _time_module(compiled, xin, reps)
    sink = gk.hybrid_sink(ex["gen_handlers"], compile_kernels=True)
    term = board.deliverable(best)
    from mlopt_core.ir import IR

    from mlopt_discovery import shipped_primitives as sp
    from mlopt_discovery.meta_arena import _ops_of

    prim_ops = sorted(_ops_of(term) & set(sp.SYNOP_SHIPPED))

    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    try:
        rep, t_gen, module = _probe_delivery(
            sink, ir, ex, base_model, xin, device, reps
        )
    except Exception as exc:
        return {
            "case": case[0],
            "claims": tags,
            "delivered": False,
            "note": f"deliver failed: {type(exc).__name__}: {exc}",
            "baseline_us": t_base,
            "compiled_us": t_comp,
        }
    row = {
        "case": case[0],
        "claims": tags,
        "delivered": True,
        "verified": rep.passed,
        "prim_ops": prim_ops,
        "max_abs": rep.max_abs,
        "cost": getattr(board.observe(), "cost", float("nan")),
        "baseline_us": t_base,
        "compiled_us": t_comp,
        "gen_us": t_gen,
        "speedup": round(t_base / t_gen, 3) if t_gen else None,
        "vs_compiled": round(t_comp / t_gen, 3) if t_gen else None,
        "device": device,
    }
    # T-030/T-031 probe arm: the raw ``gen_us`` above is the per-leaf
    # lowering's wall — on launch-bound zoo rows that is ~75% dispatch
    # the rewrite bank cannot reach (the frontier ceiling counts it,
    # the delivered body doesn't take it).  The shipped
    # ``CudaGraphRunner`` is the delivery-side answer: capture the
    # *same* verified module into a CUDA graph and time the replay.
    # Additive keys only — ``gen_us``/``speedup`` stay the lowering
    # number; ``gen_graph_us`` is the runner-wrapped arm.  Optional
    # dep (``mlopt-cuda`` is not in this package's requirements) and
    # CUDA-only, so every failure path degrades to no keys.
    graph = _probe_graph_arm(module, xin, t_base, t_comp, reps)
    row.update(graph)
    rows.append(row)
    return rows[0]


def _probe_graph_arm(
    module: Any,
    xin: Any,
    t_base: float,
    t_comp: float,
    reps: int,
) -> dict:
    """Time the delivered module under ``CudaGraphRunner``; {} on decline.

    Returns ``{}`` unless the runner actually captured (``stats
    ["cuda_graph"]``) — an honest decline records nothing rather than
    a ``gen_graph_us`` that is just the eager path re-timed.  When it
    binds, the row also carries the replay-vs-eager verify delta so a
    capture that changed numerics could not pass silently.
    """
    import torch

    try:
        from mlopt_cuda import CudaGraphRunner
    except ImportError:  # pragma: no cover — optional dep absent
        return {}
    args = xin if isinstance(xin, (tuple, list)) else (xin,)
    if not (args and torch.is_tensor(args[0]) and args[0].is_cuda):
        return {}
    stats: dict = {}
    try:  # pragma: no cover — CUDA-only body
        wrapped = CudaGraphRunner().apply(module, xin, stats)
        if not stats.get("cuda_graph"):
            return {}
        t_graph = _time_module(wrapped, xin, reps)
        with torch.no_grad():
            out_eager = module(*args)
            out_graph = wrapped(*args)
        graph_max_abs = (
            (out_eager - out_graph).abs().max().item()
            if torch.is_tensor(out_eager)
            else float("nan")
        )
    except Exception:  # pragma: no cover — CUDA-only
        return {}
    return {
        "gen_graph_us": round(t_graph, 1),
        "graph_speedup": round(t_base / t_graph, 3)
        if t_graph
        else None,
        "graph_vs_compiled": (
            round(t_comp / t_graph, 3) if t_graph else None
        ),
        "graph_max_abs": graph_max_abs,
    }


#: Referee audit log — one record per ``_triton_kernel_for`` consult.
#: The ``kernel_for`` hook declines silently otherwise: a claim whose
#: Triton arm never binds is graded on the compiled spelled body and
#: nothing records why the generated kernel never ran (the "8 of 8 no
#: binding" row in ``docs/reports/zoo_chain.md``).  Each record carries
#: the claim tag, the kernel name, the bound-value dtype, the per-tier
#: attempt table and the final outcome; bench audit drivers
#: (``bench/drivers/probes/triton_binding_audit.py``) read it to answer
#: "did the generated kernel ever bind, and if not, why".  Appended
#: unconditionally — a claim sweep is tens of records, not a stream.
TRITON_REFEREE_LOG: list[dict] = []


def _binding_attempts(tag: str, h: dict) -> tuple[Any, list[dict]]:
    """Return the first binding any tier accepts, and the attempt trail.

    Returns ``(fn, attempts)``; each attempt is
    ``{"tier", "outcome", "reason"}`` — ``bound`` /
    ``declined`` / ``no_builder``.  A decline distinguishes a spec the
    tier's recogniser rejects (outside its grammar) from one it
    accepts whose generated module then fails to load — the same
    silence ``triton_bindings`` and friends return ``{}`` for.
    """
    from mlopt_discovery import genkernel as gk

    attempts: list[dict] = []
    for tier in gk.TIERS:
        if tier.triton is None:
            attempts.append(
                {
                    "tier": tier.name,
                    "outcome": "no_builder",
                    "reason": "tier.triton is None — the registry "
                    "has no generated-kernel builder for this tier",
                }
            )
            continue
        fn = tier.triton({tag: h}).get(h["kernel"])
        if fn is not None:
            attempts.append({"tier": tier.name, "outcome": "bound"})
            return fn, attempts
        # the tier's *codegen* recogniser first (``<tier>_triton_spec``
        # — e.g. ``reduction_triton_spec`` gates the subset the Triton
        # piece lowers, narrower than the mint grammar ``reduction_spec``)
        recognises = getattr(gk, f"{tier.name}_triton_spec", None)
        if recognises is None:
            recognises = getattr(gk, f"{tier.name}_spec", None)
        if (
            recognises is not None
            and recognises(h["pattern"]) is not None
        ):
            reason = "spec recognised; generated module failed to load"
        else:
            reason = "spec declined — outside the tier's grammar"
        attempts.append(
            {"tier": tier.name, "outcome": "declined", "reason": reason}
        )
    return None, attempts


def _triton_kernel_for(tag: str, h: dict, vals: dict) -> Any:
    """Return the delivery-path callable a claim would run.

    ``profitable``'s ``kernel_for``: consult each tier's generated
    binding builder — the callable the ``hybrid_sink`` delivery
    actually ships — on CUDA-bound values (the probe's device).
    Falls back to ``None`` (the compiled-spelled default) when
    Triton/CUDA or every tier's codegen declines.  Every consult is
    recorded in :data:`TRITON_REFEREE_LOG`.
    """
    import torch

    from mlopt_discovery import genkernel as gk

    record: dict = {
        "tag": tag,
        "kernel": h.get("kernel"),
        "pattern_head": (
            h["pattern"][0]
            if isinstance(h["pattern"], (tuple, list))
            else h["pattern"]
        ),
        "dtype": next(
            (
                str(v.dtype).removeprefix("torch.")
                for v in vals.values()
                if torch.is_tensor(v)
            ),
            None,
        ),
    }
    TRITON_REFEREE_LOG.append(record)
    if not (torch.cuda.is_available() and _has_triton()):
        record["outcome"] = "unavailable"
        record["reason"] = "no CUDA or no triton import"
        return None
    try:
        fn, attempts = _binding_attempts(tag, h)
    except Exception as exc:  # a builder's own failure is a decline too
        record["outcome"] = "declined"
        record["reason"] = (
            f"binding builder raised: {type(exc).__name__}: {exc}"
        )
        return None
    record["attempts"] = attempts
    if fn is None:
        record["outcome"] = "declined"
        return None
    try:
        mvs = gk._spec_metas(h["pattern"])
        cvals = {m: vals[m].cuda() for m in mvs}
        ctuple = tuple(cvals[m] for m in mvs)
    except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
        record["outcome"] = "env_bind_failed"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return None
    record["outcome"] = "bound"

    def kern(*a: Any) -> Any:
        return fn(*ctuple)

    return kern, cvals


def _claim_denotation_for(tag: str, h: dict, vals: dict) -> Any:
    """``profitable``'s ``denotation_for``: the seq fold, or ``None``.

    Returns the positional spelled-fold callable
    ``genkernel.handler_denotation`` registers for ``kind="seq"``
    scan specs — the denotation the minted op declares — so the
    referee times the generated fold kernel against the fold
    itself, never the ambient ``applyd`` map application the spec
    tuple evals to.  ``None`` for every other claim: their spelled
    spec eval already IS the denotation.
    """
    from mlopt_discovery import genkernel as gk

    return gk.handler_denotation(h)


def _probe_delivery(
    sink: Any,
    ir: Any,
    ex: dict,
    base_model: Any,
    xin: Any,
    device: str,
    reps: int,
) -> Any:
    """Lower → verify → time the delivered module; returns (rep, µs, mod)."""
    module = sink.lower(ir, params=ex["leaves"]).to(device)
    rep = sink.verify(
        base_model,
        module,
        xin if isinstance(xin, (tuple, list)) else (xin,),
    )
    return rep, _time_module(module, xin, reps), module


def _as_dtype(obj: Any, td: Any) -> Any:
    """Cast a feed (tensor, or a tuple/list of them) to ``td``.

    Feeds carry ints next to tensors, so a blind ``.to`` raises; only
    tensors move, everything else passes through untouched.
    """
    import torch

    if torch.is_tensor(obj):
        return obj.to(td)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_as_dtype(x, td) for x in obj)
    return obj


def _zoo_case(wl: Any, *, dtype: str = "fp64") -> Any:
    """Build a held-out zoo workload as a gen-domain case tuple.

    ``dtype`` is the **intake** dtype.  ``fp64`` is the convention every
    earlier table was measured at; ``fp32`` is what a caller ships, and
    the two disagree on 12 of 65 zoo modules — a folded body's sign is
    the backend kernel's, not the IR's.  The IR is the same either way;
    only the leaves, the feeds and the refinery's measurements move.
    """
    import torch
    from mlopt_torch.adapters import TorchSource

    from mlopt_discovery import genkernel as gk
    from mlopt_discovery.meta_arena import _canon_concrete

    td = {"fp32": torch.float32, "fp64": torch.float64}[dtype]
    model, feed = wl.build()
    model = model.to(td)  # the intake convention is fp64 feeds
    feed = _as_dtype(feed, td)
    ir, leaves = TorchSource().to_ir(model, feed)
    covered = {
        _canon_concrete(h["pattern"]) for h in lawdata.HANDLERS.values()
    }
    genh = _mint_tiers(ir.root, covered)
    # the measured-cost referee: mint only claims whose generated
    # kernel actually beats its spelled site on the real values;
    # the env must bind the input vars too — a param-only env makes
    # every site touching the feed eval-fail and silently skip.
    # Only *measurable* claims are priced (``_claim_measurable`` —
    # the same predicate ``_claim_cost_gate`` applies).  Sequence-
    # form scans ARE measurable: ``denotation_for`` hands the
    # referee the registered fold denotation
    # (``genkernel.handler_denotation``) as the spelled arm, so the
    # kernel is graded fold-vs-fold — and ``gen_sink`` below binds
    # the same denotation wherever the Triton kernel does not, so
    # delivery can never silently ship the ambient map
    # application.  Other unmeasurable claims (``om_decayed``-
    # headed decayed specs) keep their seat unpriced, exactly as
    # the board-build gate leaves them — an undeliverable binding
    # fails loudly there, never silently.
    var_env = _var_env(ir, leaves, feed)
    priced = {k: v for k, v in genh.items() if _claim_measurable(k, v)}
    unpriced = {
        k: v for k, v in genh.items() if not _claim_measurable(k, v)
    }
    genh = {
        **unpriced,
        **gk.profitable(
            priced,
            ir.root,
            var_env,
            reps=20,
            kernel_for=_triton_kernel_for,
            denotation_for=_claim_denotation_for,
        ),
    }
    sink = gk.gen_sink(genh, compile_kernels=False)
    return (
        f"zoo:{wl.name}",
        ir.root,
        {"gen": len(genh)},
        wl.name,
        sink.supported_ops,
        {
            "ir": ir,
            "leaves": leaves,
            "model": model,
            "input": feed,
            "sink": sink,
            "handlers": dict(lawdata.HANDLERS) | genh,
            "gen_handlers": genh,
            # intake already ran ``profitable`` over every minted
            # tier — ``_gen_board``'s offer gate must not re-pay it
            "_gen_claims_gated": True,
        },
    )


def _zoo_rows(args: Any) -> None:
    """Print the zoo probe rows — the ``--zoo`` CLI arm."""
    for row in zoo_probe(d=args.zoo_d, t=args.zoo_t, b=args.zoo_b):
        print(f"zoo[{row['case']}]: {row}")  # stdout-compat


def zoo_probe(
    *, reps: int = 100, d: int = 16, t: int = 8, b: int = 2
) -> list[dict]:
    """``gen_probe`` over the held-out model zoo — the real corpus.

    Untraceable workloads are reported honestly rather than dropped
    silently; each surviving row is the full claim → deliver →
    verify → time table vs eager *and* ``torch.compile``.  ``d``/``t``
    /``b`` scale the workloads — generated kernels pay at larger
    shapes.
    """
    from mlopt_discovery import zoo as _zoo

    rows: list[dict] = []
    for wl in _zoo.zoo(d, t, b):
        try:
            case = _zoo_case(wl)
        except Exception as exc:
            rows.append(
                {
                    "case": f"zoo:{wl.name}",
                    "delivered": False,
                    "note": (
                        f"export declined: {type(exc).__name__}: {exc}"
                    ),
                }
            )
            continue
        rows.append(gen_probe(case, reps=reps))
    return rows


def _time_module(model: Any, x: Any, reps: int) -> float:
    """Median per-call latency (µs) of ``model(*x)`` on its device.

    ``x`` is a tuple of positional inputs — a single-tensor feed is
    ``(x,)``.
    """
    import torch

    args = x if isinstance(x, (tuple, list)) else (x,)
    dev = args[0].device
    for _ in range(10):
        model(*args)
    if dev.type == "cuda":  # pragma: no cover — CUDA-only host path
        torch.cuda.synchronize()
    import time

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(*args)
        if dev.type == "cuda":  # pragma: no cover — CUDA-only host path
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


# ---------------------------------------------------------------------------
#  The scan-tier probe — generated fold kernel vs today's delivery
# ---------------------------------------------------------------------------


def _spelled_scan(a: Any, b: Any, h: Any) -> Any:
    """Fold the torch-loop spelling: ``h_t = a_t*h + b_t`` (axis 1)."""
    for t in range(a.shape[1]):
        h = a.select(1, t) * h + b.select(1, t)
    return h


def _scan_carrier_ir(batch: int, steps: int, width: int) -> Any:
    """``applyd(<balanced affd_compose tree>, h)`` over select leaves.

    The form ``affd`` lifts leave behind — per-step ``aff_diag``
    leaves bracketed into a balanced compose tree (in-order leaves
    are reverse-chronological, so later steps sit in the LEFT
    subtree: ``affd_compose(f, g)`` applies ``g`` first).
    """
    from mlopt_core.ir import IR

    a = Var("a", TensorType((batch, steps, width)))
    b = Var("b", TensorType((batch, steps, width)))
    h = Var("h", TensorType((batch, width)))
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", a, dim=1, index=t),
            Op.make("select", b, dim=1, index=t),
        )
        for t in range(steps)
    ]

    def tree(ls: list) -> Any:
        if len(ls) == 1:
            return ls[0]
        k = len(ls) // 2
        return Op.make("affd_compose", tree(ls[k:]), tree(ls[:k]))

    return IR(
        root=Op.make("applyd", tree(leaves), h),
        inputs=[a, b, h],
        input_names={"a", "b", "h"},
        params={},
    )


#: Default probe shapes — AdaLNBlock-style ``(B, T, d)`` fp64 scans.
_SCAN_PROBE_SHAPES = ((2, 64, 128), (8, 128, 512))


def scan_probe(
    *,
    shapes: Any = None,
    reps: int = 100,
    fused: bool = True,
) -> list[dict]:
    """Time the generated scan kernel against today's delivery paths.

    The honest-number arm for the scan tier: per ``(B, T, d)`` shape
    — fp64 CUDA, AdaLNBlock-style token axis — the row reports median
    µs for the spelled torch loop (eager and ``torch.compile``-fused),
    the carrier executor path (generic ``TorchSink`` tuple-passing
    eval of the ``affd_compose`` tree, the level-batched
    ``to_batched_scan_module`` slot-gather schedule, and — with
    *fused* — its ``scan_fused`` compiled schedule), and the
    generated Triton fold kernel from
    :func:`~mlopt_discovery.genkernel.scan_triton_bindings`.
    Every arm is verified against the spelled loop before timing.
    On a host without CUDA+triton the row reports ``delivered:
    False`` honestly.
    """
    import torch
    from mlopt_carriers.scan_lower import to_batched_scan_module
    from mlopt_torch.adapters import TorchSink

    from mlopt_discovery import genkernel as gk

    if shapes is None:
        shapes = _SCAN_PROBE_SHAPES
    if not (torch.cuda.is_available() and _has_triton()):
        return [{"delivered": False, "note": "needs cuda + triton"}]
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)
    kern = gk.scan_triton_bindings(
        {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_probe",
                "args": ("A", "B", "H"),
            }
        }
    )["sc_probe"]
    compiled = torch.compile(_spelled_scan)
    dt, dev = torch.float64, "cuda"
    rows: list[dict] = []
    for batch, steps, width in shapes:
        a = torch.rand(batch, steps, width, device=dev, dtype=dt)
        a = a * 0.5 + 0.3
        b = torch.randn(batch, steps, width, device=dev, dtype=dt)
        h = torch.randn(batch, width, device=dev, dtype=dt)
        want = _spelled_scan(a, b, h)
        ir = _scan_carrier_ir(batch, steps, width)
        generic = TorchSink().lower(ir, params={})
        batched = to_batched_scan_module(ir)
        arms: dict[str, Any] = {
            "spelled": _spelled_scan,
            "spelled_compiled": compiled,
            "executor_generic": generic,
            "executor_batched": batched,
            "triton_scan": kern,
        }
        if fused:
            arms["executor_fused"] = to_batched_scan_module(
                ir, fused="compile"
            )
        row: dict[str, Any] = {
            "shape": (batch, steps, width),
            "delivered": True,
        }
        for name, fn in arms.items():
            torch.testing.assert_close(
                fn(a, b, h), want, atol=1e-9, rtol=1e-9
            )
            row[f"{name}_us"] = _time_module(fn, (a, b, h), reps)
        row["vs_spelled"] = round(
            row["spelled_us"] / row["triton_scan_us"], 3
        )
        rows.append(row)
    return rows


def _has_triton() -> bool:
    """Check triton is importable (the codegen tiers' runtime)."""
    import importlib.util

    return importlib.util.find_spec("triton") is not None


# ---------------------------------------------------------------------------
#  The decayed-tier probe — fused (s ⊙ d) @ v vs today's delivery
# ---------------------------------------------------------------------------


def _spelled_decay(s: Any, d: Any, v: Any) -> Any:
    """Evaluate the spelled body: ``(s * d) @ v``, verbatim."""
    return (s * d) @ v


def _decay_carrier_ir(batch: int, steps: int, width: int) -> Any:
    """``om_decayed(s, d, v)`` over leaf vars — the lifted site.

    RetNet inner-chunk layout: scores ``(B, T, T)``, a ``(T, T)``
    decay matrix broadcast over the batch, values ``(B, T, dv)``.
    """
    from mlopt_core.ir import IR

    s = Var("s", TensorType((batch, steps, steps)))
    d = Var("d", TensorType((steps, steps)))
    v = Var("v", TensorType((batch, steps, width)))
    return IR(
        root=Op.make("om_decayed", s, d, v),
        inputs=[s, d, v],
        input_names={"s", "d", "v"},
        params={},
    )


#: Default probe shapes — ``(B, T, dv)`` RetNet inner-chunk fp64
#: sites.  The last is the memory-bound case: the spelled mul
#: materialises a ``(8, 512, 512)`` fp64 intermediate (~134 MB).
_DECAY_PROBE_SHAPES = (
    (2, 64, 128),
    (8, 128, 256),
    (4, 256, 128),
    (8, 512, 512),
)


def decay_probe(
    *,
    shapes: Any = None,
    reps: int = 100,
) -> list[dict]:
    """Time the generated decayed kernel vs today's delivery paths.

    The honest-number arm for the decayed tier: per ``(B, T, dv)``
    shape — fp64 CUDA, ``s`` ``(B,T,T)``, broadcast decay ``d``
    ``(T,T)``, ``v`` ``(B,T,dv)`` — the row reports median µs for
    the spelled eager ``(s*d)@v``, its ``torch.compile`` (inductor
    may already fuse the mul into the matmul — if it wins, the row
    says so), the reference executor path (generic ``TorchSink``
    eval of ``om_decayed`` — the spelled body verbatim), and the
    generated Triton kernel from
    :func:`~mlopt_discovery.genkernel.decayed_triton_bindings`.
    Every arm is verified against the spelled body before timing.
    On a host without CUDA+triton the row reports ``delivered:
    False`` honestly.
    """
    import torch
    from mlopt_torch.adapters import TorchSink

    from mlopt_discovery import genkernel as gk

    if shapes is None:
        shapes = _DECAY_PROBE_SHAPES
    if not (torch.cuda.is_available() and _has_triton()):
        return [{"delivered": False, "note": "needs cuda + triton"}]
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)
    kern = gk.decayed_triton_bindings(
        {
            "d": {
                "pattern": ("om_decayed", "S", "D", "V"),
                "kernel": "dc_probe",
                "args": ("S", "D", "V"),
            }
        }
    )["dc_probe"]
    compiled = torch.compile(_spelled_decay)
    dt, dev = torch.float64, "cuda"
    rows: list[dict] = []
    for batch, steps, width in shapes:
        s = torch.randn(batch, steps, steps, device=dev, dtype=dt)
        d = torch.rand(steps, steps, device=dev, dtype=dt)
        v = torch.randn(batch, steps, width, device=dev, dtype=dt)
        want = _spelled_decay(s, d, v)
        executor = TorchSink().lower(
            _decay_carrier_ir(batch, steps, width), params={}
        )
        arms: dict[str, Any] = {
            "spelled": _spelled_decay,
            "spelled_compiled": compiled,
            "executor": executor,
            "triton_decayed": kern,
        }
        row: dict[str, Any] = {
            "shape": (batch, steps, width),
            "delivered": True,
        }
        for name, fn in arms.items():
            torch.testing.assert_close(
                fn(s, d, v), want, atol=1e-9, rtol=1e-9
            )
            row[f"{name}_us"] = _time_module(fn, (s, d, v), reps)
        row["vs_spelled"] = round(
            row["spelled_us"] / row["triton_decayed_us"], 3
        )
        row["vs_compiled"] = round(
            row["spelled_compiled_us"] / row["triton_decayed_us"], 3
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
#  Domain: search — SearchEnv adapted to the Board contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SearchState:
    """The search board's observable: rule names, cursor, terminator."""

    actions: tuple[str, ...]
    ops: frozenset
    steps: int
    done: bool
    cost: float
    baseline: float


class SearchBoard:
    """``SearchEnv`` under the :class:`engine.Board` contract.

    Actions are rule names; ``observe`` returns a
    :class:`_SearchState`; ``step`` fires the rule once and reports
    the env's normalized cost improvement as reward.  Episode ends
    at the env's horizon/patience, or when the player drains.
    """

    def __init__(
        self,
        term: Any,
        rules: Any,
        *,
        horizon: int = 12,
        patience: int | None = None,
    ) -> None:
        """Bind the program, rules, and episode bounds.

        *patience* defaults to the ruleset size — a playbook that
        tries every rule once must not stall out mid-way; the
        episode still ends at *horizon* or when the player drains.
        """
        rules = tuple(rules)
        self.env = SearchEnv(
            term,
            rules,
            cost_fn=count_cost,
            horizon=horizon,
            patience=len(rules) if patience is None else patience,
        )
        self._h = horizon
        self._p = patience
        self._steps = 0
        self._done = True
        self._baseline = float("inf")
        self._ops = frozenset(t.op for t in _ops_walk(term))

    def observe(self) -> _SearchState:
        """Return the board view — reset lazily on first look."""
        if self._done and self._steps == 0:
            self.env.reset()
            self._done = False
            self._baseline = self.env.cost
        return _SearchState(
            actions=self.env.action_names,
            ops=self._ops,
            steps=self._steps,
            done=self._done,
            cost=self.env.cost,
            baseline=self._baseline,
        )

    def step(self, action: Any) -> tuple[_SearchState, engine.Report]:
        """Fire one rule; report the cost improvement as reward."""
        self.observe()  # ensure reset
        name = (
            action if isinstance(action, str) else action.params["rule"]
        )
        res = self.env.step(name)
        self._steps += 1
        self._done = res.done
        return self.observe(), engine.Report(
            action, reward=res.reward, terminal=res.done, cost=res.cost
        )


#: The search case pool — named so a domain spec's ``corpus.kinds``
#: can select paying-fold boards as data.  Built lazily: the Vars
#: are shared leaves, spelled once.
_SEARCH_POOL: tuple | None = None


def _search_pool() -> tuple:
    """Return the named search-case pool (built once)."""
    global _SEARCH_POOL
    if _SEARCH_POOL is None:
        x = Var("x", TensorType((4, 4)))
        y = Var("y", TensorType((4, 4)))
        _SEARCH_POOL = (
            ("silu", Op.make("mul", x, Op.make("sigmoid", x))),
            (
                "mixed",
                Op.make(
                    "add",
                    Op.make("mul", x, y),
                    Op.make("matmul", x, y),
                ),
            ),
            ("square", Op.make("mul", x, x)),
            (
                "divabs",
                Op.make(
                    "div",
                    x,
                    Op.make("add", Op.make("abs", x), Const(1)),
                ),
            ),
            ("addid", Op.make("add", x, Const(0))),
            ("mulid", Op.make("mul", x, Const(1))),
            ("negneg", Op.make("neg", Op.make("neg", x))),
        )
    return _SEARCH_POOL


def _search_cases(seed: int, n: int, *, kinds: Any = None) -> list:
    """Rule-fire boards — half the pool carries a paying fold.

    ``add(x,0)`` / ``mul(x,1)`` / ``neg(neg(x))`` collapse under
    the identity laws (cost→0); the silu spelling folds to the
    fused op; the rest are honest flat boards — the corpus the
    stoppable-vs-search signal lives on.  *kinds* (domain-spec
    seam) selects pool names; ``None`` keeps the full pool.
    """
    rng = random.Random(seed)
    pool = list(_search_pool())
    if kinds is not None:
        known = {name for name, _ in pool}
        unknown = sorted(set(kinds) - known)
        if unknown:
            raise ValueError(
                f"corpus.kinds: unknown search pool {unknown!r} "
                f"(known: {sorted(known)!r})"
            )
        wanted = set(kinds)
        pool = [p for p in pool if p[0] in wanted]
    out = []
    for i in range(n):
        name, term = pool[rng.randrange(len(pool))]
        out.append((f"search{i}:{name}", term))
    return out


def _search_board(
    case: Any,
    rules: Any = None,
    *,
    horizon: int = 12,
    patience: int | None = None,
) -> SearchBoard:
    """Board a search case under the rule-fire game.

    *rules*/*horizon*/*patience* are the domain-spec seams — a spec
    seats a ruleset profile and episode bounds as data.
    """
    return SearchBoard(
        case[1],
        DEFAULT_SEARCH_RULES if rules is None else rules,
        horizon=horizon,
        patience=patience,
    )


def _ops_walk(term: Any):
    """Yield every Op node reachable in *term* (args-first)."""
    if isinstance(term, Op):
        yield term
        for a in term.args:
            yield from _ops_walk(a)


def _search_legal(state: _SearchState) -> tuple:
    """Every rule name is a legal move until the env says done."""
    return state.actions


def _search_state(state: _SearchState) -> dict[str, float]:
    """Return the ``st:*`` broadcast block — progress and improvement."""
    return {
        "st:steps": state.steps / 8.0,
        "st:improve": max(
            0.0, 1.0 - state.cost / max(state.baseline, 1e-9)
        ),
    }


def _search_act(
    action: Any, _row: dict[str, float]
) -> dict[str, float]:
    """Return the fired rule's identity bucket — the only action feature."""
    return {f"a:r:{_bucket(action, 'rule'):02d}": 1.0}


SEARCH_FEATURES: tuple = (
    "bias",
    "st:steps",
    "st:improve",
    *(
        f"a:r:{b:02d}"
        for b in range(lawdata.META_ARENA_HASH_BUCKETS["rule"])
    ),
)

#: The search spec's parts in the record vocabulary — every shipped
#: spec reifies to data (:meth:`DomainSpec.to_record`).
register_part("search:state", lambda: StateBroadcast(_search_state))
register_part("search:action", lambda: ActionFeats(_search_act))
register_legal("search", _search_legal)


#: The search adapter's spec — a rule-fire board has no move-kind
#: one-hot and no play-rate history, so its parts are just the
#: intercept, the ``st:*`` broadcast and the rule bucket.
SEARCH_SPEC = DomainSpec.build(
    features=SEARCH_FEATURES,
    legal=_search_legal,
    parts=(
        Bias(),
        StateBroadcast(_search_state),
        ActionFeats(_search_act),
    ),
)


# ---------------------------------------------------------------------------
#  Domain: fragment — vocab/ fragment-application moves (plan 0033 ph.4)
# ---------------------------------------------------------------------------


#: The fragment case pool — named so a domain spec's
#: ``corpus.kinds`` can select term shapes as data.  Built lazily
#: for the same shared-leaf reason as ``_search_pool``.
_FRAGMENT_POOL: tuple | None = None


def _fragment_pool() -> tuple:
    """Return the named fragment-case pool (built once)."""
    global _FRAGMENT_POOL
    if _FRAGMENT_POOL is None:
        x = Var("x", TensorType((4, 4)))
        y = Var("y", TensorType((4, 4)))
        _FRAGMENT_POOL = (
            ("addid", Op.make("add", x, Const(0))),
            ("mulid", Op.make("mul", x, Const(1))),
            ("negneg", Op.make("neg", Op.make("neg", x))),
            ("mulsig", Op.make("mul", x, Op.make("sigmoid", x))),
            (
                "addmul",
                Op.make(
                    "add",
                    Op.make("mul", x, y),
                    Op.make("mul", y, x),
                ),
            ),
            (
                "divabs",
                Op.make(
                    "div",
                    x,
                    Op.make("add", Op.make("abs", x), Const(1)),
                ),
            ),
        )
    return _FRAGMENT_POOL


def _fragment_cases(seed: int, n: int, *, kinds: Any = None) -> list:
    """Small terms that exercise law/handler/saturate fragment moves.

    *kinds* (domain-spec seam) selects pool names; ``None`` keeps
    the full pool.
    """
    rng = random.Random(seed)
    pool = list(_fragment_pool())
    if kinds is not None:
        known = {name for name, _ in pool}
        unknown = sorted(set(kinds) - known)
        if unknown:
            raise ValueError(
                f"corpus.kinds: unknown fragment pool {unknown!r} "
                f"(known: {sorted(known)!r})"
            )
        wanted = set(kinds)
        pool = [p for p in pool if p[0] in wanted]
    out = []
    for i in range(n):
        name, term = pool[rng.randrange(len(pool))]
        out.append((f"frag{i}:{name}", term))
    return out


def _fragment_board(
    case: Any,
    *,
    rules: Any = None,
    handlers: Any = None,
    max_moves: int = 256,
) -> fg.FragmentArena:
    """Board a fragment case over the canonical vocab directory.

    *rules*/*handlers*/*max_moves* are the domain-spec seams — a
    spec seats a rewrite-name subset, a handler-tag subset and the
    move budget as data.
    """
    return fg.FragmentArena(
        case[1],
        vocab_dir=None,
        rules=rules,
        handlers=handlers,
        max_moves=max_moves,
    )


def _frag_state(state: fg.FragmentState) -> dict[str, float]:
    """Broadcast the fragment board's progress features."""
    base = max(state.baseline_cost, 1e-9)
    best = (
        state.best_cost if state.best_cost == state.best_cost else base
    )
    return {
        "st:best": max(0.0, min(1.0, 1.0 - best / base)),
        "st:steps": state.steps / 8.0,
        "st:moves": min(len(state.moves) / 64.0, 1.0),
    }


def _frag_act(action: Any, _row: dict[str, float]) -> dict[str, float]:
    """One-hot the fragment move kind — the only action feature."""
    kind = getattr(action, "kind", "?")
    return {f"a:k:{kind}": 1.0}


FRAGMENT_FEATURES: tuple = (
    "bias",
    "st:best",
    "st:steps",
    "st:moves",
    "a:k:law",
    "a:k:prim",
    "a:k:schema",
    "a:k:handler",
    "a:k:saturate",
    "a:k:extract",
)

FRAGMENT_SPEC = DomainSpec.build(
    features=FRAGMENT_FEATURES,
    legal=fg.fragment_legal,
    parts=(
        Bias(),
        StateBroadcast(_frag_state),
        ActionFeats(_frag_act),
    ),
)
register_legal("fragment", fg.fragment_legal)


def _fragment_scripted() -> Any:
    """Saturate the fragment closure, then extract — the playbook."""
    played: set[str] = set()

    def go(state: fg.FragmentState) -> Any | None:
        if state.done:
            return None
        for m in state.moves:
            if m.kind == "saturate" and "sat" not in played:
                played.add("sat")
                return m
        for m in state.moves:
            if m.kind == "extract":
                return m
        return state.moves[0] if state.moves else None

    return go


def _random_policy(legal: Any) -> Any:
    """Return a uniform player over any domain's legal set."""

    class _R:
        def __init__(self, seed: int) -> None:
            self._r = random.Random(seed)

        def __call__(self, state: Any) -> Any | None:
            acts = legal(state)
            return acts[self._r.randrange(len(acts))] if acts else None

    return _R


def _scripted_search() -> Any:
    """Fire each rule once, matching-head-first — the playbook.

    Rules whose LHS root op occurs in the program go first (the
    only ones that can change the graph); the rest are dead budget.
    """
    heads = {
        r.name: getattr(r.lhs, "op", None) for r in DEFAULT_SEARCH_RULES
    }
    played: set = set()

    def go(state: _SearchState) -> Any | None:
        if state.done:
            return None
        for name in state.actions:
            if name not in played and heads.get(name) in state.ops:
                played.add(name)
                return name
        for name in state.actions:
            if name not in played:
                played.add(name)
                return name
        return None

    return go


DOMAINS: dict[str, Domain] = {
    "meta": spec_domain(
        mp.gen_cases,
        _meta_board,
        mp.META_SPEC,
        scripted=ma.ScriptedPlayer,
    ),
    "joint": spec_domain(
        _joint_cases,
        _joint_board,
        mp.META_SPEC,
        scripted=ma.ScriptedPlayer,
    ),
    "torch": spec_domain(
        _torch_cases,
        _torch_board,
        mp.META_SPEC,
        scripted=ma.ScriptedPlayer,
        deliver=_torch_deliver,
    ),
    "search": spec_domain(
        _search_cases,
        _search_board,
        SEARCH_SPEC,
        scripted=_scripted_search,
    ),
    "fragment": spec_domain(
        _fragment_cases,
        _fragment_board,
        FRAGMENT_SPEC,
        scripted=_fragment_scripted,
    ),
    "gen": spec_domain(
        _gen_cases,
        _gen_board,
        mp.META_SPEC,
        scripted=ma.ScriptedPlayer,
        deliver=_torch_deliver,
    ),
    "open": spec_domain(og.gen_cases, og.OpenArena, og.OPEN_SPEC),
    "player": spec_domain(
        pa.gen_cases,
        pa.PlayerArena,
        pa.PLAYER_SPEC,
        scripted=pa.GreedyEditor,
    ),
    "bilinear": spec_domain(
        bg.gen_cases,
        bg.BilinearBoard,
        bg.BILINEAR_SPEC,
        scripted=bg.strassen_line,
    ),
}

# The bilinear board's player seats — explicit at the composition
# root (this module), not an ``import bilinear_game`` side-effect.
bg.register_players()


def _arms(domain: Domain, player: LinearPolicy, seed: int) -> dict:
    """Build the arm set for one domain: baselines + learned pair."""
    rnd = _random_policy(domain.legal)
    arms: dict[str, Any] = {
        "random": lambda i: rnd(seed + 10_000 + i),
        "learned": lambda i: player.frozen(seed + 20_000 + i),
        "learned-greedy": lambda i: player.frozen(
            seed + 30_000 + i, greedy=True
        ),
    }
    if domain.scripted is not None:
        arms = {"scripted": lambda i: domain.scripted(), **arms}
    return arms


def _table(table: dict[str, list[dict]]) -> str:
    """Render per-arm means over eval rows — generic edition."""
    head = f"{'player':<15} {'cases':>6} {'reward':>9} {'cost':>7} {'cert':>5} {'steps':>6}"
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        n = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        certs = sum(1 for r in rows if r["cert"])
        lines.append(
            f"{name:<15} {len(rows):>6} "
            f"{sum(r['reward'] for r in rows) / n:>9.2f} "
            f"{(sum(costs) / len(costs)) if costs else float('nan'):>7.2f} "
            f"{certs:>5} "
            f"{sum(r['steps'] for r in rows) / n:>6.1f}"
        )
    return "\n".join(lines)


def _extras(args: Any, d: Any, ev: list, fallback: Any) -> None:
    """Run the ``--*probe`` / ``--deliver`` / ``--zoo`` report arms."""
    if args.deliver:
        arm = d.scripted() if d.scripted is not None else fallback
        res = deliver(args.domain, ev[0], arm, budget=args.budget)
        shown = {k: v for k, v in res.items() if k != "module"}
        print(f"deliver[{ev[0][0]}]: {shown}")  # stdout-compat
    if args.probe:
        for case in ev:
            if DOMAINS[args.domain] is DOMAINS["gen"]:
                print(  # stdout-compat
                    f"probe[{case[0]}]: {gen_probe(case)}"
                )
    if args.zoo:
        _zoo_rows(args)
    _kernel_probes(args)


def _kernel_probes(args: Any) -> None:
    """Run the generated-kernel timing arms (CUDA + triton)."""
    if args.scan_probe:
        for row in scan_probe():
            print(  # stdout-compat
                f"scan[{row.get('shape', '-')}]: {row}"
            )
    if args.decay_probe:
        for row in decay_probe():
            print(  # stdout-compat
                f"decay[{row.get('shape', '-')}]: {row}"
            )


def main(argv: list[str] | None = None) -> int:
    """Train the learned arm on one case stream, eval on another."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--domain",
        default="meta",
        help=(
            "a registered domain name — built-ins plus any seated "
            "via --domain-spec or a downstream register() plugin "
            "(kernelopt's 'kernel')"
        ),
    )
    ap.add_argument(
        "--domain-spec",
        action="append",
        default=[],
        metavar="PATH|NAME",
        help=(
            "load a domain-spec file (JSON path or packaged "
            "domains/<name>.json) and register its domain before "
            "play — repeatable (mlopt_discovery.domain_spec)"
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=40)
    ap.add_argument("--eval-cases", type=int, default=10)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument(
        "--deliver",
        action="store_true",
        help="lower + verify the first eval case's extraction",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="run the gen domain's claim→deliver→time probe per eval case",
    )
    ap.add_argument(
        "--zoo",
        action="store_true",
        help="run the claim→deliver→time probe over the held-out zoo",
    )
    ap.add_argument(
        "--zoo-d",
        type=int,
        default=16,
        help="zoo model width (default 16 — the honest small probe)",
    )
    ap.add_argument(
        "--zoo-t",
        type=int,
        default=8,
        help="zoo sequence length (default 8 — kernels pay ≥64)",
    )
    ap.add_argument(
        "--zoo-b",
        type=int,
        default=2,
        help="zoo batch size (default 2 — the honest small probe)",
    )
    ap.add_argument(
        "--scan-probe",
        action="store_true",
        help=(
            "time the generated Triton scan kernel against the "
            "spelled loop and the carrier executor path (CUDA)"
        ),
    )
    ap.add_argument(
        "--decay-probe",
        action="store_true",
        help=(
            "time the generated fused (s·d)@v Triton kernel against "
            "eager, torch.compile, and the om_decayed executor (CUDA)"
        ),
    )
    ap.add_argument(
        "--player",
        default=None,
        metavar="NAME",
        help=(
            "evaluate a registered players/ module by name instead of "
            "training the learned arm — eval-only, paired against "
            "``random`` (see players.registry.names())"
        ),
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    for spec_arg in args.domain_spec:
        from . import domain_spec

        # ``domains=DOMAINS``: under ``python -m mlopt_discovery.play``
        # this module *is* ``__main__`` — a different object than the
        # ``mlopt_discovery.play`` domain_spec would import, so the
        # registry must be passed explicitly.
        domain_spec.register_file(spec_arg, domains=DOMAINS)
    if args.domain not in DOMAINS:
        print(  # stdout-compat
            f"unknown domain {args.domain!r} — known: "
            f"{sorted(DOMAINS)!r}"
        )
        return 2
    d = DOMAINS[args.domain]
    train = d.cases(args.seed, args.train_cases)
    ev = d.cases(args.seed + 7919, args.eval_cases)
    if args.player is not None:
        from gamekit.players import registry

        try:
            mk = registry.make(
                args.player, args.domain, seed=args.seed, ctx={}
            )
        except ValueError as e:
            print(f"--player: {e}")  # stdout-compat
            return 2
        rnd = _random_policy(d.legal)
        arms = {
            args.player: mk,
            "random": lambda i: rnd(args.seed + 10_000 + i),
        }
        fallback = mk(0)
        totals: list = []
    else:
        player = LinearPolicy(
            seed=args.seed,
            legal=d.legal,
            featurizer=d.featurizer,
            features=d.features,
            temperature=args.temperature,
            epsilon=args.epsilon,
        )
        totals = engine.train_policy(
            player, train, d.board_of, budget=args.budget
        )
        arms = _arms(d, player, args.seed)
        fallback = player.frozen(0)
    table = engine.evaluate(arms, ev, d.board_of, budget=args.budget)
    print(f"== play — domain {args.domain} ==")  # stdout-compat
    print(_table(table))  # stdout-compat
    if args.player is not None:
        print(  # stdout-compat
            f"eval-only — registered player {args.player!r}"
        )
    else:
        decade = max(len(totals) // 10, 1)
        means = [
            round(sum(totals[i : i + decade]) / decade, 2)
            for i in range(0, len(totals) - decade + 1, decade)
        ]
        print(f"train decade means: {means}")  # stdout-compat
    _extras(args, d, ev, fallback)
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(
                {"table": table, "train_totals": totals}, f, indent=1
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
