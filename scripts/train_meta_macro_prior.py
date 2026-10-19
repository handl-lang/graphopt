"""Retrain the meta prior on the shipped macro legal set.

``docs/results_macro_moves.md``'s open loop, promoted: the minted
macros pay the scripted arm (+0.088 held-out, 7W/0L) but were
invisible to the *frozen* shipped prior, and a lab-only retrain
regressed the base set.  Macros now ship — ``lawdata.MACROS`` rides
``meta_arena.legal_actions`` — so this driver asks whether a prior
*trained* against the shipped macro-offering set learns to spend
them, and whether the bake gate is earned on the prod action space.

Protocol (mirrors ``scripts/train_meta_prior.py`` exactly — the
``legal=`` seam is gone because the seam shipped):

* **mine** — ``macro_miner.collect_trajectories`` + ``mine_macros`` over
  ``mp.gen_cases(seed, 24)`` *train* cases only (the held-out stream
  ``gen_cases(seed + 7919)`` is never mined from); the minted set is
  reported alongside the shipped :data:`lawdata.MACROS` table (drift
  check — the two should coincide);
* **train** — a *cold* ``MetaLearnedPlayer`` (``weights={}``) on the
  shipped legal set — ``meta_arena.legal_actions`` already offers the
  macros — 30 epochs x ``gen_cases(seed, 40)``, budget 12,
  episode-return REINFORCE;
* **measure** — held-out ``gen_cases(seed + 7919, 16)`` budget 12: the
  new arm's sampled/greedy reward, its *macro pick rate* (applied
  ``op == "macro"`` steps), the shipped V1 prior for pairing, and the
  same V2 weights played on the *macro-free* base set
  (``v2-base`` — ``meta_arena._base_legal``; the regression clause:
  a caller that suppresses the offers must not get a worse policy).

Usage::

    .venv/bin/python scripts/train_meta_macro_prior.py
    .venv/bin/python scripts/train_meta_macro_prior.py \
        --epochs 30 --json /tmp/meta_macro.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# ``lab/`` and ``scripts/`` are not installed packages — the repo root
# goes on ``sys.path`` so ``lab.mint.*`` / ``scripts.*`` imports resolve
# under the documented ``python scripts/…`` invocation (the sanctioned
# shim shape; cf. ``lab/probes/conv_fold_wall_regimes_probe.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from mlopt_core.cost.basic import count_cost
from mlopt_discovery import lawdata
from mlopt_discovery import macro_moves as mm
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import meta_player as mp
from mlopt_discovery.macro_moves import Macro

from lab.mint.macro_miner import (
    MacroScripted,
    collect_trajectories,
    mine_macros,
)
from scripts.train_meta_prior import (
    EPOCHS,
    HELD_OUT_OFFSET,
    SEED,
    _literal,
    baked_table,
)

__all__ = ["main", "run_eval", "train"]

TRAIN_CASES = 40
EVAL_CASES = 16
MINE_CASES = 24
BUDGET = 12
MACRO_CAP = 8


def mine_train_macros(
    *,
    seed: int = SEED,
    mine_cases: int = MINE_CASES,
) -> tuple[list[Macro], list[dict]]:
    """Mine the offer set off the *train* stream — held-out untouched."""
    train = mp.gen_cases(seed, mine_cases)
    trajs = collect_trajectories(train, seed=seed)
    return mine_macros(trajs, top=MACRO_CAP)


def train(
    *,
    seed: int = SEED,
    train_cases: int = TRAIN_CASES,
    epochs: int = EPOCHS,
    budget: int = BUDGET,
) -> tuple[mp.MetaLearnedPlayer, list[float], dict]:
    """Train a cold meta prior against the shipped macro legal set.

    Same loop as ``train_meta_prior.train`` — the player's default
    enumerator is ``meta_arena.legal_actions``, which now offers
    :data:`lawdata.MACROS` itself; the trajectory reports are kept so
    the macro pick rate *during training* is measurable, not just at
    eval.
    """
    torch.manual_seed(seed)
    player = mp.MetaLearnedPlayer(seed, weights={})
    cases = mp.gen_cases(seed, train_cases)
    totals: list[float] = []
    macro_picks = steps = 0
    for _ in range(epochs):
        for case in cases:
            traj = ma.run_episode(
                mp._board(case, count_cost, 20_000), player, budget
            )
            player.finish_episode(
                traj.total, rewards=[r.reward for r in traj.reports]
            )
            totals.append(traj.total)
            macro_picks += sum(
                1
                for r in traj.reports
                if r.applied and r.action.op == "macro"
            )
            steps += len(traj.reports)
    stats = {
        "macro_picks": macro_picks,
        "steps": steps,
        "macro_rate": round(macro_picks / max(steps, 1), 4),
        "episodes_with_macro": None,  # per-episode split lives in eval
    }
    return player, totals, stats


def _play(
    mk: Any,
    ev: list,
    *,
    seed: int,
    budget: int,
    max_nodes: int = 20_000,
) -> list[dict]:
    """Play arm *mk* on *ev*; per-case rows including macro pick counts."""
    rows: list[dict] = []
    for i, case in enumerate(ev):
        traj = ma.run_episode(
            mp._board(case, count_cost, max_nodes), mk(i), budget
        )
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
                "macros": sum(
                    1
                    for r in traj.reports
                    if r.applied and r.action.op == "macro"
                ),
            }
        )
    return rows


def _random_arm(seed: int, legal: Any) -> Any:
    """Uniform player over a given legal enumerator (seeded)."""

    def make(i: int) -> Any:
        return ma.RandomPlayer(random.Random(seed + 10_000 + i), legal)

    return make


def run_eval(
    player: mp.MetaLearnedPlayer,
    macros: list[Macro],
    *,
    seed: int = SEED,
    eval_cases: int = EVAL_CASES,
    budget: int = BUDGET,
) -> dict[str, Any]:
    """Held-out arms: V2 sampled/greedy/base, shipped V1, baselines.

    Every learned arm plays the *shipped* legal set — macros are prod
    offers now.  ``v2-base`` is the regression clause: the *new*
    weights through the macro-free :func:`meta_arena._base_legal`,
    what a caller that suppresses the offers would get.  ``v1``
    replays the shipped prior from its own captured table, so the
    pairing survives the bake.
    """
    torch.manual_seed(seed)
    ev = mp.gen_cases(seed + HELD_OUT_OFFSET, eval_cases)
    v1 = mp.MetaLearnedPlayer(
        seed, weights=dict(lawdata.META_ARENA_PLAYER_WEIGHTS)
    )
    v2_base = mp.MetaLearnedPlayer(
        seed, weights=player.weights_dict(), legal=ma._base_legal
    )
    cold = mp.MetaLearnedPlayer(seed, weights={})
    arms = {
        "random": _random_arm(seed, ma.legal_actions),
        "cold": lambda i: cold.frozen(seed + 20_000 + i),
        "v1-shipped": lambda i: v1.frozen(seed + 40_000 + i),
        "v2-base": lambda i: v2_base.frozen(seed + 45_000 + i),
        "v2+macros": lambda i: player.frozen(seed + 50_000 + i),
        "v2+macros-greedy": lambda i: player.frozen(
            seed + 60_000 + i, greedy=True
        ),
        "scripted": lambda i: ma.ScriptedPlayer(),
        "scripted+macros": lambda i: MacroScripted(macros),
    }
    table = {
        name: _play(mk, ev, seed=seed, budget=budget)
        for name, mk in arms.items()
    }
    means: dict[str, dict] = {}
    for name, rows in table.items():
        k = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        means[name] = {
            "reward": round(sum(r["reward"] for r in rows) / k, 4),
            "cost": round(sum(costs) / len(costs), 3)
            if costs
            else None,
            "steps": round(sum(r["steps"] for r in rows) / k, 2),
            "cert": sum(1 for r in rows if r["cert"]),
            "decl": sum(1 for r in rows if r["used_declared"]),
            "macro_picks": sum(r["macros"] for r in rows),
            "macro_cases": sum(1 for r in rows if r["macros"]),
        }
    return {"means": means, "table": table}


def _block_means(totals: list[float], blocks: int) -> list[float]:
    """Mean train reward per block — the visible training curve."""
    size = max(len(totals) // blocks, 1)
    return [
        round(sum(totals[i : i + size]) / len(totals[i : i + size]), 3)
        for i in range(0, len(totals), size)
    ]


def _means_table(means: dict[str, dict]) -> str:
    """Render the held-out arm table — reward, cost, macro picks."""
    head = (
        f"{'arm':<18} {'reward':>8} {'cost':>7} {'steps':>6}"
        f" {'cert':>5} {'decl':>5} {'macro':>10}"
    )
    lines = [head, "-" * len(head)]
    for name, r in means.items():
        cost = (
            f"{r['cost']:7.2f}"
            if r["cost"] is not None
            else f"{'—':>7}"
        )
        lines.append(
            f"{name:<18} {r['reward']:>8.3f} {cost}"
            f" {r['steps']:>6.2f} {r['cert']:>5} {r['decl']:>5}"
            f" {r['macro_picks']:>3}/{r['macro_cases']:>2}c"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Mine, retrain, evaluate, print the candidate baked literal."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mine-cases", type=int, default=MINE_CASES)
    ap.add_argument("--train-cases", type=int, default=TRAIN_CASES)
    ap.add_argument("--eval-cases", type=int, default=EVAL_CASES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    macros, _rows = mine_train_macros(
        seed=args.seed, mine_cases=args.mine_cases
    )
    shipped = {
        tuple(str(s["op"]) for s in r["seq"]) for r in lawdata.MACROS
    }
    print(  # stdout-compat
        f"== minted {len(macros)} macros off gen_cases"
        f"({args.seed}, {args.mine_cases}) — train stream only; "
        f"shipped table: {len(lawdata.MACROS)} rows =="
    )
    for m in macros:
        ops = tuple(str(s["op"]) for s in m.seq)
        flag = "" if ops in shipped else "  (not shipped)"
        print(  # stdout-compat
            f"  {m.name} {m.op_str():<44} score={m.score:.3f}{flag}"
        )

    player, totals, tstats = train(
        seed=args.seed,
        train_cases=args.train_cases,
        epochs=args.epochs,
        budget=args.budget,
    )
    curve = _block_means(totals, args.blocks)
    print(  # stdout-compat
        f"\n== train — cold MetaLearnedPlayer, shipped legal set"
        f" (macros offered), {args.train_cases} cases x"
        f" {args.epochs} ep, budget {args.budget} =="
    )
    print(f"train block means: {curve}")  # stdout-compat
    print(  # stdout-compat
        f"train macro picks: {tstats['macro_picks']}"
        f" / {tstats['steps']} steps"
        f" (rate {tstats['macro_rate']:.3f})"
    )

    res = run_eval(
        player,
        macros,
        seed=args.seed,
        eval_cases=args.eval_cases,
        budget=args.budget,
    )
    print(  # stdout-compat
        f"\n== held-out — gen_cases({args.seed}+{HELD_OUT_OFFSET},"
        f" {args.eval_cases}), budget {args.budget} =="
    )
    print(_means_table(res["means"]))  # stdout-compat
    table = baked_table(player)
    print("== candidate baked snapshot ==")  # stdout-compat
    print(_literal(table))  # stdout-compat
    if args.json:
        payload = {
            "seed": args.seed,
            "protocol": {
                "mine_cases": args.mine_cases,
                "train_cases": args.train_cases,
                "epochs": args.epochs,
                "eval_cases": args.eval_cases,
                "budget": args.budget,
                "macro_cap": MACRO_CAP,
                "held_out": f"gen_cases(seed+{HELD_OUT_OFFSET})",
            },
            "minted": [
                {"name": m.name, "ops": m.ops, "score": m.score}
                for m in macros
            ],
            "curve": curve,
            "train_macro_stats": tstats,
            "eval_means": res["means"],
            "eval_table": res["table"],
            "weights": table,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1, default=repr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
