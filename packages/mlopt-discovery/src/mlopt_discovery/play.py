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
    DOMAINS[name] = _spec_domain(
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
  ``open`` board — the player itself is the term being rewritten.
* ``kernel`` — the kernelopt spike (plan 0032 phase 3): the codegen
  law bank of ``mlopt_kernel`` as the move set, the launch/weight
  cost model, and the gauntlet as referee — the kernel-engineering
  board of ``project/notes/kernel-engineer-board.md``; the
  ``--deliver`` flag lowers the winning extraction through
  ``KernelSink`` and verifies it numerically.

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
    RelFeats,
    StateBroadcast,
    register_legal,
    register_part,
)
from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import DEFAULT as DEFAULT_SEARCH_RULES
from mlopt_core.search_env import SearchEnv

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
]


@dataclass(frozen=True)
class Domain:
    """One board family's registration — data, not code.

    A domain binds a *cases* generator, a ``case -> Board`` factory
    and the learned player's vocabulary: its legal-move enumerator
    (``legal``), featurizer and feature schema (``features``).  A
    domain that ships a :class:`~mlopt_discovery.players.DomainSpec`
    builds its record with :func:`_spec_domain`, which fills those
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


def _spec_domain(
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


def _torch_cases(seed: int, n: int) -> list:
    """Small real ``nn.Module``s lifted to IR — the product board.

    Each case is a torch module exported through
    ``mlopt_torch.adapters.TorchSource`` and played under the real
    sink's ``supported_ops`` bound — kernel claims price at sink
    costs, not ambient vocabulary.  Modules that fail to export are
    skipped honestly.  Torch is imported lazily — the registry
    itself stays torch-free.
    """
    import torch
    import torch.nn as nn
    from mlopt_torch.adapters import TorchSink, TorchSource

    class _SpelledSiLU(nn.Module):
        """x·sigmoid(x) spelled out — the claim move's live site."""

        def forward(self, x):
            """Spell ``x·sigmoid(x)`` — no fused op named."""
            return x * torch.sigmoid(x)

    rng = random.Random(seed)
    supported = TorchSink().supported_ops
    src = TorchSource()
    builders = [
        ("spelled_silu", lambda d: _SpelledSiLU()),
        (
            "mlp",
            lambda d: nn.Sequential(
                nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
            ),
        ),
        (
            "relu_lin",
            lambda d: nn.Sequential(nn.Linear(d, d), nn.ReLU()),
        ),
        ("lin", lambda d: nn.Linear(d, d)),
        (
            "softsign_mlp",
            lambda d: nn.Sequential(nn.Linear(d, d), nn.Softsign()),
        ),
    ]
    out = []
    for i in range(n):
        d = 4 * rng.choice((1, 2, 4))
        kind, mk = builders[rng.randrange(len(builders))]
        torch.manual_seed(seed * 10_007 + i)
        model = mk(d).eval()
        example = torch.randn(4, d)
        ir, leaves = src.to_ir(model, example)
        out.append(
            (
                f"torch{i}:{kind}",
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


def _torch_deliver(board: ma.MetaArena, case: Any, traj: Any) -> dict:
    """Lower the winning extraction and verify it against the model.

    ``deliverable`` resolves handled claims to their kernels and
    unhandled abbreviations to spelled form; ``TorchSink.lower``
    rebuilds the module from it; ``verify`` runs both on the case's
    example input.  The game score becomes a delivered artifact.
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
    return {
        "delivered": True,
        "module": module,
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "cost": getattr(t, "cost", None),
        "cert": getattr(t, "certificate_ok", None),
        "reward": traj.total,
    }


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
    """
    from mlopt_carriers.om import OM_CHUNKIFY, OM_LAWS
    from mlopt_core.egraph import EGraph
    from mlopt_core.laws.tensor import SPLIT_SCALAR_RULES

    from mlopt_discovery.meta_arena import _spec_of

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
            if any(spec == s["spec"] for s in out):
                continue
            out.append({"spec": spec, "leaf_env": leaf_env})
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


#: Site collectors, keyed by codegen-tier name — the term-traversal
#: knowledge that pairs with ``genkernel.TIERS``.  A tier without a
#: collector mints nothing here; the ``scan`` collector is the only
#: one whose sites live in the saturated graph, not raw IR.
TIER_SITES: dict[str, Any] = {
    "gen": _fusion_sites,
    "decayed": _decayed_sites,
    "scan": _scan_sites,
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


def _gen_cases(seed: int, n: int) -> list:
    """Torch modules over fusion-spelling and stock-model builders.

    The fusion builders spell pointwise patterns the ambient
    handler table has no fused op for — the candidates ``claim``
    can only reach through a *generated* kernel.  Stock pieces
    (mlp/attn chains) give the board real shape.
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
    covered = {
        _canon_concrete(h["pattern"]) for h in lawdata.HANDLERS.values()
    }
    builders = [
        ("mulrelu", lambda: _Fuse("mulrelu"), (256, 512)),
        ("mulgelu", lambda: _Fuse("mulgelu"), (256, 512)),
        ("mulsoftplus", lambda: _Fuse("mulsoftplus"), (256, 512)),
        ("multanh", lambda: _Fuse("multanh"), (256, 512)),
        ("bigfuse", lambda: _Fuse("bigfuse"), (256, 512)),
        ("swiglu_mlp", _SwigluMLP, (16, 64)),
        ("chain_mlp", _ChainMLP, (16, 64)),
        ("chainfuse", _ChainFuse, (256, 512)),
        (
            "mlp",
            lambda d=64: nn.Sequential(
                nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d)
            ),
            (8, 64),
        ),
    ]
    out = []
    for i in range(n):
        name, build, shape = builders[rng.randrange(len(builders))]
        model = build().eval()
        gen = torch.Generator().manual_seed(seed * 997 + i)
        x = torch.randn(*shape, generator=gen)
        ir, leaves = src.to_ir(model, x)
        genh = _mint_tiers(ir.root, covered)
        sink = gk.gen_sink(genh, compile_kernels=False)
        handlers = dict(lawdata.HANDLERS) | genh
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
                },
            )
        )
    return out


def _gen_board(case: Any) -> ma.MetaArena:
    """Board a gen case under its generated handler table."""
    from mlopt_core.cells import Provenance
    from mlopt_core.cost.basic import count_cost
    from mlopt_core.laws import DEFAULT

    from mlopt_discovery import cellstore as cs
    from mlopt_discovery import genkernel as gk
    from mlopt_discovery import lawdata

    ex = case[5]

    def novel(op: str) -> bool:
        return op.startswith(("claim_", "foldabs_", "gen_"))

    store = cs.ambient_store(
        rules=DEFAULT, handlers=lawdata.HANDLERS
    ).merge(
        cs.handlers_fragment(
            "gen",
            ex["gen_handlers"],
            provenance=Provenance("minted"),
        )
    )
    return ma.MetaArena(
        case[1],
        supported=case[4],
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

    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    try:
        rep, t_gen = _probe_delivery(
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
    rows.append(
        {
            "case": case[0],
            "claims": tags,
            "delivered": True,
            "verified": rep.passed,
            "max_abs": rep.max_abs,
            "cost": getattr(board.observe(), "cost", float("nan")),
            "baseline_us": t_base,
            "compiled_us": t_comp,
            "gen_us": t_gen,
            "speedup": round(t_base / t_gen, 3) if t_gen else None,
            "vs_compiled": round(t_comp / t_gen, 3) if t_gen else None,
            "device": device,
        }
    )
    return rows[0]


def _first_binding(tag: str, h: dict) -> Any:
    """Return the first generated kernel binding any tier accepts."""
    from mlopt_discovery import genkernel as gk

    for tier in gk.TIERS:
        if tier.triton is not None:
            fn = tier.triton({tag: h}).get(h["kernel"])
            if fn is not None:
                return fn
    return None


def _triton_kernel_for(tag: str, h: dict, vals: dict) -> Any:
    """Return the delivery-path callable a claim would run.

    ``profitable``'s ``kernel_for``: consult each tier's generated
    binding builder — the callable the ``hybrid_sink`` delivery
    actually ships — on CUDA-bound values (the probe's device).
    Falls back to ``None`` (the compiled-spelled default) when
    Triton/CUDA or every tier's codegen declines.
    """
    import torch

    from mlopt_discovery import genkernel as gk

    if not (torch.cuda.is_available() and _has_triton()):
        return None
    try:
        fn = _first_binding(tag, h)
        if fn is None:
            return None
        mvs = gk._spec_metas(h["pattern"])
        cvals = {m: vals[m].cuda() for m in mvs}
        ctuple = tuple(cvals[m] for m in mvs)
    except (AttributeError, KeyError, RuntimeError, TypeError):
        return None

    def kern(*a: Any) -> Any:
        return fn(*ctuple)

    return kern, cvals


def _probe_delivery(
    sink: Any,
    ir: Any,
    ex: dict,
    base_model: Any,
    xin: Any,
    device: str,
    reps: int,
) -> Any:
    """Lower → verify → time the delivered module; returns (rep, µs)."""
    module = sink.lower(ir, params=ex["leaves"]).to(device)
    rep = sink.verify(
        base_model,
        module,
        xin if isinstance(xin, (tuple, list)) else (xin,),
    )
    return rep, _time_module(module, xin, reps)


def _zoo_case(wl: Any) -> Any:
    """Build a held-out zoo workload as a gen-domain case tuple."""
    from mlopt_torch.adapters import TorchSource

    from mlopt_discovery import genkernel as gk
    from mlopt_discovery.meta_arena import _canon_concrete

    model, feed = wl.build()
    model = model.double()  # the intake convention is fp64 feeds
    ir, leaves = TorchSource().to_ir(model, feed)
    covered = {
        _canon_concrete(h["pattern"]) for h in lawdata.HANDLERS.values()
    }
    genh = _mint_tiers(ir.root, covered)
    # the measured-cost referee: mint only claims whose generated
    # kernel actually beats its spelled site on the real values;
    # the env must bind the input vars too — a param-only env makes
    # every site touching the feed eval-fail and silently skip.
    feeds = tuple(feed) if isinstance(feed, (tuple, list)) else (feed,)
    var_env = dict(leaves)
    var_env.update(
        {v.name: t for v, t in zip(ir.inputs, feeds, strict=False)}
    )
    genh = gk.profitable(
        genh,
        ir.root,
        var_env,
        reps=20,
        kernel_for=_triton_kernel_for,
    )
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
    if dev.type == "cuda":
        torch.cuda.synchronize()
    import time

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(*args)
        if dev.type == "cuda":
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


def _search_cases(seed: int, n: int) -> list:
    """Rule-fire boards — half the pool carries a paying fold.

    ``add(x,0)`` / ``mul(x,1)`` / ``neg(neg(x))`` collapse under
    the identity laws (cost→0); the silu spelling folds to the
    fused op; the rest are honest flat boards — the corpus the
    stoppable-vs-search signal lives on.
    """
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    pool = [
        Op.make("mul", x, Op.make("sigmoid", x)),
        Op.make("add", Op.make("mul", x, y), Op.make("matmul", x, y)),
        Op.make("mul", x, x),
        Op.make("div", x, Op.make("add", Op.make("abs", x), Const(1))),
        Op.make("add", x, Const(0)),
        Op.make("mul", x, Const(1)),
        Op.make("neg", Op.make("neg", x)),
    ]
    return [
        (f"search{i}", pool[rng.randrange(len(pool))]) for i in range(n)
    ]


def _search_board(case: Any) -> SearchBoard:
    """Board a search case under the rule-fire game."""
    return SearchBoard(case[1], DEFAULT_SEARCH_RULES)


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


# ---------------------------------------------------------------------------
#  Domain: kernel — codegen laws under the kernel cost model
# ---------------------------------------------------------------------------
#
#  The kernelopt spike's board (plan 0032 phase 3): candidates are
#  codegen-law applications — named fused-kernel ops and strength
#  reductions from ``mlopt_kernel.laws.KERNEL_RULES`` — the referee
#  is the existing gauntlet (``verifier.verify_law`` derives the
#  scheduled candidate against the spelled source), and the score is
#  ``mlopt_kernel.cost.kernel_cost`` improvement.  Cases are spelled
#  programs carrying live fusion/strength-reduction sites plus honest
#  flat boards; ``deliver`` lowers the winning extraction through
#  ``KernelSink`` and verifies it numerically — verify-first.


def _kernel_cases(seed: int, n: int) -> list:
    """Generate spelled programs over the kernel bank's live sites.

    Each archetype exercises a different paying move — the gate
    fusion (``a·sigmoid(b)``), the softsign fusion (``a/(|b|+c)``), the
    FMA fusion (``a·b + c``), and the strength reductions — plus
    compositions and honest flat boards, so the player's signal is
    *which* move pays, not whether any does.
    """
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 8)))
    y = Var("y", TensorType((4, 8)))
    w = Var("w", TensorType((8, 4)))
    b = Var("b", TensorType((4,)))
    ln_center = Op.make(
        "sub", x, Op.make("mean", x, dim=-1, keepdim=True)
    )
    pool = [
        ("silu", Op.make("mul", x, Op.make("sigmoid", x))),
        (
            "gate",
            Op.make("mul", y, Op.make("sigmoid", x)),
        ),
        (
            "softsign",
            Op.make(
                "div",
                x,
                Op.make("add", Op.make("abs", x), Const(1)),
            ),
        ),
        (
            "fma",
            Op.make("add", Op.make("mul", x, y), Op.make("tanh", y)),
        ),
        (
            "swiglu",
            Op.make("mul", Op.make("mul", x, Op.make("sigmoid", x)), y),
        ),
        (
            "softmax",
            Op.make(
                "div",
                Op.make("exp", x),
                Op.make("sum", Op.make("exp", x), dim=-1, keepdim=True),
            ),
        ),
        (
            "layernorm",
            Op.make(
                "mul",
                ln_center,
                Op.make(
                    "rsqrt",
                    Op.make(
                        "add",
                        Op.make(
                            "mean",
                            Op.make("pow", ln_center, Const(2)),
                            dim=-1,
                            keepdim=True,
                        ),
                        Const(1e-5),
                    ),
                ),
            ),
        ),
        (
            "mm_bias",
            Op.make("add", Op.make("matmul", x, w), b),
        ),
        (
            "mm_sb",
            Op.make(
                "add",
                Op.make("mul", Op.make("matmul", x, w), Const(0.5)),
                b,
            ),
        ),
        ("square", Op.make("pow", x, Const(2))),
        ("recip", Op.make("div", x, Const(4.0))),
        (
            "mixed",
            Op.make(
                "add",
                Op.make("mul", x, Op.make("sigmoid", x)),
                Op.make("div", Op.make("pow", x, Const(2)), Const(4.0)),
            ),
        ),
        (
            "flat",
            Op.make("matmul", x, w),
        ),
    ]
    out = []
    for i in range(n):
        kind, term = pool[rng.randrange(len(pool))]
        names = {v.name for v in _vars_of(term)}
        vars_in = [v for v in (x, y, w, b) if v.name in names]
        out.append(
            (
                f"kernel{i}:{kind}",
                term,
                kind,
                (),
                (),
                {"inputs": vars_in},
            )
        )
    return out


def _vars_of(term: Any):
    """Yield every ``Var`` leaf reachable in *term*."""
    if isinstance(term, Var):
        yield term
    elif isinstance(term, Op):
        for a in term.args:
            yield from _vars_of(a)


def _kernel_board(case: Any) -> Any:
    """Board a kernel case under the codegen bank + gauntlet referee.

    ``step_cost=0.05`` is the kernel board's oracle-call economics:
    firing a codegen law is a compile/measure call, so moves pay a
    small charge and ``stop`` — declining the remaining evaluations —
    is the skill the board scores.
    """
    from mlopt_kernel import KernelBoard

    from mlopt_discovery.verifier import verify_law

    return KernelBoard(case[1], verifier=verify_law, step_cost=0.05)


def _kernel_legal(state: Any) -> tuple:
    """Return the codegen laws plus ``stop`` — ending is a move."""
    return (*state.actions, "stop")


#: The op vocabulary the kernel cases draw from — the board's
#: site-detection signal, broadcast as binary ``st:op:*`` flags.
_KERNEL_OP_FEATURES: tuple[str, ...] = (
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "sigmoid",
    "tanh",
    "pow",
    "matmul",
    "exp",
    "relu",
    "mean",
    "sum",
    "rsqrt",
)


def _kernel_state(state: Any) -> dict[str, float]:
    """Return the ``st:*`` broadcast — progress, improvement, sites."""
    out = {
        "st:steps": state.steps / 8.0,
        "st:improve": max(
            0.0, 1.0 - state.cost / max(state.baseline, 1e-9)
        ),
        "st:const": float(state.has_const),
        "st:sites": float(len(state.matches)),
    }
    for op in _KERNEL_OP_FEATURES:
        out[f"st:op:{op}"] = float(op in state.ops)
    return out


def _kernel_act(
    action: Any, _row: dict[str, float]
) -> dict[str, float]:
    """Return the fired law's identity bucket — action features."""
    return {f"a:r:{_bucket(action, 'rule'):02d}": 1.0}


def _kernel_rel(
    state: Any, action: Any, _row: dict[str, float]
) -> dict[str, float]:
    """Return the state-x-action interaction — does this law apply here?.

    ``x:m`` flags a rule whose checked pattern occurs in the program
    (the board's ``matches`` — a matching codegen is the one that can
    pay its compile cost); ``x:stop`` gives the stop move its own
    weight, and ``x:stop:imp`` prices stopping against the
    improvement already banked.
    """
    if action == "stop":
        return {
            "x:stop": 1.0,
            "x:stop:imp": _row.get("st:improve", 0.0),
        }
    return {"x:m": float(action in state.matches)}


KERNEL_FEATURES: tuple = (
    "bias",
    "st:steps",
    "st:improve",
    "st:const",
    "st:sites",
    *(f"st:op:{op}" for op in _KERNEL_OP_FEATURES),
    "x:m",
    "x:stop",
    "x:stop:imp",
    *(
        f"a:r:{b:02d}"
        for b in range(lawdata.META_ARENA_HASH_BUCKETS["rule"])
    ),
)

register_part("kernel:state", lambda: StateBroadcast(_kernel_state))
register_part("kernel:action", lambda: ActionFeats(_kernel_act))
register_part("kernel:rel", lambda: RelFeats(_kernel_rel))
register_legal("kernel", _kernel_legal)

#: The kernel spec — intercept, progress/improvement/site broadcast,
#: the match interaction, and the rule bucket; the shipped prior is
#: ``lawdata.KERNEL_PLAYER_WEIGHTS`` (match-oracle imitation +
#: REINFORCE; learned-greedy 0.511 > scripted 0.333 on held-out).
KERNEL_SPEC = DomainSpec.build(
    features=KERNEL_FEATURES,
    legal=_kernel_legal,
    parts=(
        Bias(),
        StateBroadcast(_kernel_state),
        RelFeats(_kernel_rel),
        ActionFeats(_kernel_act),
    ),
    weights=lawdata.KERNEL_PLAYER_WEIGHTS,
)


def _scripted_kernel() -> Any:
    """Play the playbook: fire every matching law once, bank order.

    ``kern_fuse_*`` lhs heads name the op the pattern needs; a law
    whose head is absent can never match, so firing it wastes the
    step — matching-head-first is the honest scripted line.
    """
    from mlopt_kernel import KERNEL_RULES

    heads = {r.name: getattr(r.lhs, "op", None) for r in KERNEL_RULES}
    played: set = set()

    def go(state: Any) -> Any | None:
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


def _kernel_deliver(board: Any, case: Any, traj: Any) -> dict:
    """Lower the winning extraction through ``KernelSink``; verify it.

    The deliver half of the kernel board: the spelled source and the
    candidate the episode's extraction picked are both lowered in the
    sink's runtime and compared on random feeds — verify-first.
    """
    import torch
    from mlopt_core.ir import IR
    from mlopt_kernel import KernelSink

    best = board.extract_best()
    if best is None:
        return {"delivered": False, "reward": traj.total}
    inputs = list(case[5]["inputs"])
    sink = KernelSink()
    ref = sink.lower(
        IR(root=case[1], inputs=inputs, params={}), params={}
    )
    opt = sink.lower(IR(root=best, inputs=inputs, params={}), params={})
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in inputs
    )
    rep = sink.verify(ref, opt, feeds)
    t = traj.terminal
    return {
        "delivered": True,
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "cost": getattr(t, "cost", None),
        "cert": getattr(t, "certificate_ok", None),
        "reward": traj.total,
    }


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
    "meta": _spec_domain(
        mp.gen_cases,
        _meta_board,
        mp.META_SPEC,
        scripted=ma.ScriptedPlayer,
    ),
    "joint": _spec_domain(
        _joint_cases,
        _joint_board,
        mp.META_SPEC,
        scripted=ma.ScriptedPlayer,
    ),
    "torch": _spec_domain(
        _torch_cases,
        _torch_board,
        mp.META_SPEC,
        scripted=ma.ScriptedPlayer,
        deliver=_torch_deliver,
    ),
    "search": _spec_domain(
        _search_cases,
        _search_board,
        SEARCH_SPEC,
        scripted=_scripted_search,
    ),
    "gen": _spec_domain(
        _gen_cases,
        _gen_board,
        mp.META_SPEC,
        scripted=ma.ScriptedPlayer,
        deliver=_torch_deliver,
    ),
    "kernel": _spec_domain(
        _kernel_cases,
        _kernel_board,
        KERNEL_SPEC,
        scripted=_scripted_kernel,
        deliver=_kernel_deliver,
    ),
    "open": _spec_domain(og.gen_cases, og.OpenArena, og.OPEN_SPEC),
    "player": _spec_domain(
        pa.gen_cases,
        pa.PlayerArena,
        pa.PLAYER_SPEC,
        scripted=pa.GreedyEditor,
    ),
}


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
    ap.add_argument("--domain", choices=sorted(DOMAINS), default="meta")
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
