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
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import DEFAULT as DEFAULT_SEARCH_RULES
from mlopt_core.search_env import SearchEnv

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


def _seq_deindex(t: Any, names: dict, operands: dict, events: list):
    """Spec of *t* with per-step ``select``s abstracted to metas.

    Every ``select(child, dim=D, index=i)`` node becomes a sequence
    operand metavar bound to *child* — recorded in *operands* as
    ``meta -> (child, dim)`` and in *events* as ``(meta, index)`` so
    the caller can verify the index walks the step axis in step
    order.  ``rsub(x, s)`` normalises to ``sub(s, x)`` — the
    elementwise spelling the scan renderer knows.  Any other bare
    leaf (a whole-tensor reference the fold cannot slice honestly)
    or non-scalar ``Const`` declines the whole term with ``None``.

    ``unsqueeze``/``squeeze`` chains wrapping a ``select`` are
    lifted onto the child first (:func:`_seq_view_lift`) — the
    metavar then binds the *viewed* sequence and no view op reaches
    the step-expr skeleton.
    """
    if isinstance(t, Op):
        if t.op in ("unsqueeze", "squeeze") and len(t.args) == 1:
            lifted = _seq_view_lift(t)
            if lifted is not None:
                child, dim, idx = lifted
                key = (child, dim)
                meta = names.get(key)
                if meta is None:
                    meta = f"Q{len(names) + 1}"
                    names[key] = meta
                    operands[meta] = (child, dim)
                events.append((meta, idx))
                return meta
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
            key = (child, dim)
            meta = names.get(key)
            if meta is None:
                meta = f"Q{len(names) + 1}"
                names[key] = meta
                operands[meta] = (child, dim)
            events.append((meta, idx))
            return meta
        if t.op == "rsub" and len(t.args) == 2 and not t.attrs:
            b = _seq_deindex(t.args[1], names, operands, events)
            a = _seq_deindex(t.args[0], names, operands, events)
            if a is None or b is None:
                return None
            return ("sub", b, a)
        parts = []
        for a_ in t.args:
            p = _seq_deindex(a_, names, operands, events)
            if p is None:
                return None
            parts.append(p)
        spec = [t.op, *parts]
        if t.attrs:
            spec.append(dict(t.attrs))
        return tuple(spec)
    if isinstance(t, Const):
        return None if isinstance(t.value, str) else t.value
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
    a0 = b0 = None
    events: list = []
    for k, (at, bt, _st) in enumerate(steps):
        ea: list = []
        eb: list = []
        sa = _seq_deindex(at, names, operands, ea)
        sb = _seq_deindex(bt, names, operands, eb)
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
    for k, evs in enumerate(events):
        for _m, idx in evs:
            if idx != k:
                return None
    h_meta = "H"
    spec = (
        "applyd",
        ("aff_diag", a0, b0),
        h_meta,
        {"kind": "seq", "axis": eff},
    )
    leaf_env = {m: c for m, (c, _d) in operands.items()}
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
        if not any(spec == s["spec"] for s in out):
            out.append(
                {"spec": spec, "leaf_env": leaf_env, "axis": axis}
            )
        for _m, _i, s in steps:
            emitted_states.add(id(s))
        emitted_roots.add(id(root))
        if len(out) >= _SEQ_SITE_CAP:
            break
    return out


def _all_scan_sites(term: Any) -> list:
    """Both scan site forms: saturated carriers + raw recurrences."""
    return [*_scan_sites(term), *_seq_scan_sites(term)]


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
        ("twosite", _TwoIslands, (256, 128)),
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
    )
    return rows[0]


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
    """Lower → verify → time the delivered module; returns (rep, µs)."""
    module = sink.lower(ir, params=ex["leaves"]).to(device)
    rep = sink.verify(
        base_model,
        module,
        xin if isinstance(xin, (tuple, list)) else (xin,),
    )
    return rep, _time_module(module, xin, reps)


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


def _fragment_cases(seed: int, n: int) -> list:
    """Small terms that exercise law/handler/saturate fragment moves."""
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    pool = [
        Op.make("add", x, Const(0)),
        Op.make("mul", x, Const(1)),
        Op.make("neg", Op.make("neg", x)),
        Op.make("mul", x, Op.make("sigmoid", x)),
        Op.make("add", Op.make("mul", x, y), Op.make("mul", y, x)),
        Op.make("div", x, Op.make("add", Op.make("abs", x), Const(1))),
    ]
    return [
        (f"frag{i}", pool[rng.randrange(len(pool))]) for i in range(n)
    ]


def _fragment_board(case: Any) -> fg.FragmentArena:
    """Board a fragment case over the canonical vocab directory."""
    return fg.FragmentArena(case[1], vocab_dir=None, max_moves=256)


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
