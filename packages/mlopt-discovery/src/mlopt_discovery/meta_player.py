"""The learned meta-arena player — Q3's arm.

Plan 0021's open question: can a *learned* player beat the scripted
``saturate → extract`` playbook on the mixed board?  The probe's
baseline table says the headroom is real — every baseline ties at
baseline on the fold-site cases while a ``declare`` exists that
wins — but no baseline plays the line, because the paying sequence
is four moves deep: ``declare`` a fresh fold of a mined spec,
``saturate`` so the minted rule fires, ``handle`` to bind a
supported kernel, ``extract`` to collect.

``MetaLearnedPlayer`` is a linear softmax policy over the legal move
set, trained by episode-return REINFORCE — the shared
:class:`~mlopt_discovery.players.LearnedPlayer`, bound to the
:data:`META_SPEC` declared below (the feature schema, the featurizer's
composable parts and ``meta_arena.legal_actions``).  The board is
small (~dozens of legal moves vs the construction arena's ~50k), so
the softmax is *exact* over the unplayed set — no perceptual cap, no
played-mass correction; the gradient row is ``φ_chosen - E_π[φ]``
computed directly.

``gen_cases`` builds the training corpus: seeded variants of the
sites the handlers cover (a ``mul(x, sigmoid(x))`` under a ruleset
with ``silu_fold`` stripped pays exactly one line — the
declare→saturate→handle chain), controls where nothing pays (the
policy must learn *not* to burn steps), and a swiglu site whose
kernel enters the bound explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Callable, Iterable
from typing import Any

from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    LearnedPlayer,
    OpOneHot,
    PlayHist,
    PlayRates,
    StateBroadcast,
    register_legal,
    register_part,
)
from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import DEFAULT

from . import lawdata
from . import meta_arena as ma

__all__ = [
    "META_PART_TAGS",
    "META_SPEC",
    "MetaLearnedPlayer",
    "evaluate",
    "featurize",
    "gen_cases",
    "player_table",
    "train_player",
]

#: REINFORCE default — the spec's learning rate.
_LR = 0.1

#: The pre-consolidation name, kept as an alias for callers/tests.
_Hist = PlayHist


# ---------------------------------------------------------------------------
#  Featurization — (MetaState, Action) -> feature row over the schema
# ---------------------------------------------------------------------------


def _bucket(name: Any, kind: str) -> int:
    """Stable-hash *name* into ``META_ARENA_HASH_BUCKETS[kind]``."""
    width = lawdata.META_ARENA_HASH_BUCKETS[kind]
    h = hashlib.sha256(str(name).encode("utf-8")).digest()
    return int.from_bytes(h[:4]) % width


def _state_feats(state: ma.MetaState) -> dict[str, float]:
    """Return the ``st:*`` broadcast block — the summary per move."""
    total = state.n_enodes + max(state.budget_left, 0)
    rel = (
        state.best_cost / state.baseline_cost
        if state.baseline_cost > 0
        else 1.0
    )
    return {
        "st:steps": state.steps / 32.0,
        "st:enodes_frac": state.n_enodes / max(total, 1),
        "st:classes_frac": state.n_classes / max(state.n_enodes, 1),
        "st:improve": max(0.0, 1.0 - rel),
        "st:declared": len(state.declared) / 8.0,
        "st:handled": len(state.handles) / 8.0,
        "st:unhandled": (
            len(state.declared_bodies) - len(state.handles)
        )
        / 8.0,
        "st:handleable": _handleable(state) / 8.0,
        "st:sites": len(state.sites) / 8.0,
        "st:claimable": state.claimable / 8.0,
        "st:specs": len(state.specs) / 32.0,
        "st:fires": sum(c for _, c in state.rule_fires) / 64.0,
    }


def _handleable(state: ma.MetaState) -> int:
    """Mined specs some handler's canonical pattern covers.

    The precondition of the paying line — ``declare`` a fresh fold
    of a covered spec, then ``handle`` binds it to the kernel.  A
    spec is coverable iff its alpha-canonical form equals a handler
    pattern's (``meta_arena._handle_actions``' own test).
    """
    pats = [p for _, p in state.handler_specs]
    return sum(
        1
        for s in state.specs
        if any(ma._canon_spec(s) == p for p in pats)
    )


def _spec_index(action: ma.Action) -> float:
    """Mined-spec position a ``declare`` targets (``foldabs_i``)."""
    cons = action.params.get("construction") or {}
    params = cons.get("params") or {}
    suffix = str(params.get("name", "")).rsplit("_", 1)[-1]
    return int(suffix) / 32.0 if suffix.isdigit() else 0.5


def _act_feats(
    action: ma.Action, row: dict[str, float]
) -> dict[str, float]:
    """Return the move's own params and interactions — ``a:*``/``x:*``.

    *row* is the partially-built feature row (the state broadcasts and
    play rates), so an interaction term can read a broadcast value.
    The move-kind one-hot is a separate part (:class:`OpOneHot`).
    """
    op = action.op
    out: dict[str, float] = {}
    if op == "fire":
        out[f"a:r:{_bucket(action.params.get('rule'), 'rule'):02d}"] = (
            1.0
        )
    elif op == "saturate":
        budget = action.params.get("budget")
        if budget is None:
            out["a:unbounded"] = 1.0
        else:
            out["a:budget"] = float(budget) / 512.0
        out["x:pressure:saturate"] = row["st:enodes_frac"]
    elif op == "declare":
        out["a:spec_i"] = _spec_index(action)
        out["x:specs:declare"] = row["st:specs"]
        out["x:handleable:declare"] = row["st:handleable"]
        out["x:claimable:declare"] = row["st:claimable"]
    elif op == "handle":
        out[
            f"a:h:{_bucket(action.params.get('handler'), 'handler')}"
        ] = 1.0
        out[f"a:o:{_bucket(action.params.get('object'), 'object')}"] = (
            1.0
        )
        out["x:declared:handle"] = row["st:declared"]
        out["x:unhandled:handle"] = row["st:unhandled"]
    elif op == "claim":
        out[
            f"a:h:{_bucket(action.params.get('handler'), 'handler')}"
        ] = 1.0
        out["x:handleable:claim"] = row["st:handleable"]
        out["x:claimable:claim"] = row["st:claimable"]
    elif op == "extract":
        out["x:extract:improve"] = row["st:improve"]
        out["x:late:extract"] = row["st:steps"]
    return out


def _macro_feats(
    action: ma.Action, row: dict[str, float]
) -> dict[str, float]:
    """Return the macro payload block — ``x:macro:*`` + identity reuse.

    ``docs/results_macro_moves.md``'s diagnosed blindness: every
    ``op == "macro"`` offer featurized to the same broadcast row, so
    the softmax could learn *when* a macro pays but never *which*.
    This part emits the per-offer signal:

    * ``x:macro:len`` — the compressed line's length (``seq``/8):
      a one-step ``claim→extract`` prices differently from the
      spelled four-move line;
    * ``x:macro:certified`` / ``x:macro:score`` — the miner's ranking
      evidence the offer carries on ``params["macro"]`` (the stat
      payload ``macro_moves.instantiate`` writes:
      ``certified`` = share of source episodes that certified;
      ``score`` = count x mean terminal reward, the miner's rank
      key, rescaled /8);
    * ``a:h:<bucket>`` — the *same* handler-identity channel a
      spelled ``claim``/``handle`` uses, emitted for each claim and
      handle leg in the seq.  This is what lets the policy prefer
      ``claim(swiglu)`` over the rank-1 ``claim(silu)`` on a swiglu
      board — the failure mode the macro doc's honest notes named
      ("viability is legality, not payoff");
    * ``x:claimable:claim`` / ``x:handleable:claim`` — the context
      terms a spelled claim gets, emitted when the seq *contains* a
      claim, so the macro row inherits the board-fit pricing.

    ``op:macro`` itself comes free through :class:`OpOneHot` once the
    schema names it.  ``a:macro`` is deliberately absent — it would
    be the same flag twice.  A hand-spelled ``Action("macro",
    {"seq": ...})`` with no ``Macro`` record still gets the
    structural terms; the stat columns read as 0 (missing weight).
    """
    if action.op != "macro":
        return {}
    seq = tuple(action.params.get("seq") or ())
    out: dict[str, float] = {"x:macro:len": len(seq) / 8.0}
    m = action.params.get("macro")
    if m:
        if isinstance(m, dict):
            out["x:macro:certified"] = float(m.get("certified", 0.0))
            out["x:macro:score"] = float(m.get("score", 0.0)) / 8.0
        else:  # a Macro-like record duck-typed in by the caller
            out["x:macro:certified"] = float(
                getattr(m, "certified", 0.0)
            )
            out["x:macro:score"] = float(getattr(m, "score", 0.0)) / 8.0
    has_claim = False
    for sub in seq:
        op, params = (
            (str(sub["op"]), dict(sub.get("params") or {}))
            if isinstance(sub, dict)
            else (str(sub.op), dict(getattr(sub, "params", {})))
        )
        if op in ("claim", "handle"):
            tag = params.get("handler")
            if tag is not None:
                out[f"a:h:{_bucket(tag, 'handler')}"] = 1.0
            has_claim = has_claim or op == "claim"
    if has_claim:
        out["x:claimable:claim"] = row.get("st:claimable", 0.0)
        out["x:handleable:claim"] = row.get("st:handleable", 0.0)
    return out


#: The meta featurizer's composable parts, in row order: the intercept,
#: the ``st:*`` broadcast, the ``h:*`` play rates, the move-kind
#: one-hot, then the action's params and interactions — and the macro
#: payload last (it reads no ``st:*`` keys it doesn't guard, so the
#: part order only fixes row ordering).
META_PARTS: tuple = (
    Bias(),
    StateBroadcast(_state_feats),
    PlayRates(("fire", "saturate", "declare", "handle", "claim")),
    OpOneHot(frozenset(lawdata.META_ARENA_FEATURES)),
    ActionFeats(_act_feats),
    ActionFeats(_macro_feats),
)

#: The meta parts in the spec record's tag vocabulary — the names a
#: rewritable spec record carries (:meth:`DomainSpec.to_record`).
META_PART_TAGS: tuple[str, ...] = (
    "bias",
    "meta:state",
    "meta:hist",
    "meta:onehot",
    "meta:action",
    "meta:macro",
)

register_part("meta:state", lambda: StateBroadcast(_state_feats))
register_part(
    "meta:hist",
    lambda: PlayRates(
        ("fire", "saturate", "declare", "handle", "claim")
    ),
)
register_part(
    "meta:onehot",
    lambda: OpOneHot(frozenset(lawdata.META_ARENA_FEATURES)),
)
register_part("meta:action", lambda: ActionFeats(_act_feats))
register_part("meta:macro", lambda: ActionFeats(_macro_feats))
register_legal("meta", ma.legal_actions)

#: The meta-arena player spec — schema, parts, enumerator, prior.
META_SPEC = DomainSpec.build(
    features=lawdata.META_ARENA_FEATURES,
    legal=ma.legal_actions,
    parts=META_PARTS,
    weights=lawdata.META_ARENA_PLAYER_WEIGHTS,
    lr=_LR,
)

#: The composed featurizer — public for tests.
featurize = META_SPEC.featurize


# ---------------------------------------------------------------------------
#  The player
# ---------------------------------------------------------------------------


class MetaLearnedPlayer(LearnedPlayer):
    """The meta-arena preset — the meta :class:`DomainSpec` bound.

    A :class:`~mlopt_discovery.players.LearnedPlayer` over
    ``meta_arena.legal_actions`` scored by :data:`META_SPEC`'s
    featurizer; *weights* defaults to the shipped trained prior
    :data:`lawdata.META_ARENA_PLAYER_WEIGHTS` (pass ``weights={}``
    for the uniform cold start).  ``learn=False`` freezes the policy
    for eval; ``frozen`` returns a non-learning copy.
    """

    SPEC = META_SPEC


def _key(action: ma.Action) -> tuple:
    """Stable dedup key — the played-move mask, replayable."""
    params = action.params
    return (
        action.op,
        repr(sorted(params.items(), key=lambda kv: kv[0])),
    )


# ---------------------------------------------------------------------------
#  Training and evaluation — disjoint case streams
# ---------------------------------------------------------------------------


def gen_cases(seed: int, n: int) -> list:
    """Seeded case variants — the mixed board's paying structures.

    Kinds: ``silu`` sites under a ruleset with ``silu_fold``
    stripped (the declare→handle line is the only path back — silu
    stays in the ambient vocabulary through ``silu_expand`` /
    ``silu_mul_form``); ``swiglu`` sites whose kernel enters the
    bound explicitly; ``softsign`` sites under ``sans`` where the
    handler kernel is *out* of vocabulary — an honest negative, the
    line must decline; and ``control`` terms under ``DEFAULT`` where
    nothing pays and the best play is a cheap extract.  Case tuples
    follow ``meta_probe``'s arity — ``(name, term, rules, decls,
    supported)``.
    """
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
    silu_spelled = Op.make("mul", x, Op.make("sigmoid", x))
    builders = [
        ("silu", lambda: (silu_spelled, sans, None)),
        (
            "silu_nested",
            lambda: (
                Op.make("tanh", Op.make("add", silu_spelled, y)),
                sans,
                None,
            ),
        ),
        (
            "swiglu",
            lambda: (
                Op.make("mul", silu_spelled, y),
                sans,
                ma._supported_or(None, silu_spelled, sans) | {"swiglu"},
            ),
        ),
        (
            "softsign",
            lambda: (
                Op.make(
                    "div",
                    x,
                    Op.make("add", Op.make("abs", x), Const(1)),
                ),
                sans,
                None,
            ),
        ),
        (
            "control",
            lambda: (
                Op.make(
                    "add", Op.make("mul", x, y), Op.make("matmul", x, y)
                ),
                DEFAULT,
                None,
            ),
        ),
    ]
    out = []
    for i in range(n):
        kind, build = builders[rng.randrange(len(builders))]
        term, rules, sup = build()
        out.append((f"gen{i}:{kind}", term, rules, (), sup))
    return out


def _board(case: list | tuple, cf: Any, max_nodes: int) -> ma.MetaArena:
    """Build the arena for one case tuple (probe arity)."""
    rules = case[2] if len(case) > 2 else None
    sup = case[4] if len(case) > 4 else None
    return ma.MetaArena(
        case[1],
        rules,
        cost_fn=cf,
        supported=sup,
        max_nodes=max_nodes,
    )


def train_player(
    player: MetaLearnedPlayer,
    cases: Iterable,
    *,
    budget: int,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> list[float]:
    """Play one episode per case; REINFORCE on each trajectory total."""
    cf = cost_fn or count_cost
    totals: list[float] = []
    for case in cases:
        traj = ma.run_episode(
            _board(case, cf, max_nodes), player, budget
        )
        player.finish_episode(
            traj.total, rewards=[r.reward for r in traj.reports]
        )
        totals.append(traj.total)
    return totals


def evaluate(
    arms: dict[str, Callable[[int], Any]],
    cases: list,
    *,
    budget: int,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> dict[str, list[dict]]:
    """Play every arm on the same cases; paired boards, probe rows."""
    cf = cost_fn or count_cost
    out: dict[str, list[dict]] = {}
    for name, mk in arms.items():
        rows: list[dict] = []
        for i, case in enumerate(cases):
            arena = _board(case, cf, max_nodes)
            traj = ma.run_episode(arena, mk(i), budget)
            t = traj.terminal
            rows.append(
                {
                    "case": case[0],
                    "steps": len(traj.reports),
                    "reward": round(traj.total, 3),
                    "cost": t.cost if t is not None else None,
                    "cert": t.certificate_ok if t is not None else None,
                    "used_declared": (
                        t.used_declared if t is not None else ()
                    ),
                }
            )
        out[name] = rows
    return out


def player_table(table: dict[str, list[dict]]) -> str:
    """Render per-arm means over the eval rows."""
    head = (
        f"{'player':<15} {'cases':>6} {'reward':>9} {'cost':>7}"
        f" {'cert':>5} {'decl':>5} {'steps':>6}"
    )
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        n = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        certs = sum(1 for r in rows if r["cert"])
        decls = sum(1 for r in rows if r["used_declared"])
        lines.append(
            f"{name:<15} {len(rows):>6} "
            f"{sum(r['reward'] for r in rows) / n:>9.2f} "
            f"{(sum(costs) / len(costs)) if costs else float('nan'):>7.2f} "
            f"{certs:>5} {decls:>5} "
            f"{sum(r['steps'] for r in rows) / n:>6.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Train on one case stream, evaluate on a disjoint one."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=40)
    ap.add_argument("--eval-cases", type=int, default=10)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--max-nodes", type=int, default=20_000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=_LR)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    train = gen_cases(args.seed, args.train_cases)
    ev = gen_cases(args.seed + 7919, args.eval_cases)
    player = MetaLearnedPlayer(
        seed=args.seed,
        temperature=args.temperature,
        lr=args.lr,
        epsilon=args.epsilon,
    )
    totals = train_player(
        player,
        train,
        budget=args.budget,
        max_nodes=args.max_nodes,
    )
    arms: dict[str, Callable[[int], Any]] = {
        "scripted": lambda i: ma.ScriptedPlayer(),
        "greedy": lambda i: ma.GreedyPlayer(),
        "random": lambda i: ma.RandomPlayer(
            random.Random(args.seed + 10_000 + i)
        ),
        "learned": lambda i: player.frozen(args.seed + 20_000 + i),
        "learned-greedy": lambda i: player.frozen(
            args.seed + 30_000 + i, greedy=True
        ),
    }
    table = evaluate(
        arms, ev, budget=args.budget, max_nodes=args.max_nodes
    )
    print(  # stdout-compat
        "== meta-arena player — learned vs baselines =="
    )
    print(player_table(table))  # stdout-compat
    decade = max(len(totals) // 10, 1)
    means = [
        round(sum(totals[i : i + decade]) / decade, 2)
        for i in range(0, len(totals) - decade + 1, decade)
    ]
    print(f"train decade means: {means}")  # stdout-compat
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(
                {
                    "table": table,
                    "train_totals": totals,
                    "weights": player.weights_dict(),
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
