"""Train the meta-arena policy and emit the baked prior.

The sibling of ``scripts/train_open_prior.py`` for the meta board:
trains a *cold* :class:`mlopt_discovery.meta_player.MetaLearnedPlayer`
over a fixed case stream and prints the trained weight table as the
Python literal that bakes into
:data:`mlopt_discovery.lawdata.META_ARENA_PLAYER_WEIGHTS` (the
``MetaLearnedPlayer`` default).

The split is the registry's: train on ``meta_player.gen_cases(seed,
n)``, evaluate on the disjoint ``gen_cases(seed + 7919, n)`` the
``play`` driver uses.  ``torch.manual_seed`` fixes the numeric
oracle's tensors, so the run is reproducible from the seed alone.

The run is honest: the tool prints the *train curve* (mean episode
reward per block) and the held-out table — random, the cold start
(``{}``), the cold-greedy arm, the trained arm, and the hand-written
``scripted`` playbook — so the bake is earned by a measurement.

Usage::

    python scripts/train_meta_prior.py
    python scripts/train_meta_prior.py --epochs 30 --json /tmp/meta.json
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

import torch
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import meta_player as mp

#: Defaults — the case counts and horizon the meta driver uses.
TRAIN_CASES = 40
EVAL_CASES = 16
EPOCHS = 30
BUDGET = 12
SEED = 0
#: The held-out case stream's seed offset — ``play``'s convention.
HELD_OUT_OFFSET = 7919
#: Round the baked weights to this many decimals.
ROUND = 4


def train(
    *,
    seed: int = SEED,
    train_cases: int = TRAIN_CASES,
    epochs: int = EPOCHS,
    budget: int = BUDGET,
) -> tuple[mp.MetaLearnedPlayer, list[float]]:
    """Train a *cold* meta policy over the seeded case stream.

    The player starts from ``weights={}`` (the cold start), so the
    bake is a genuine training run.  ``torch.manual_seed(seed)`` fixes
    the numeric oracle's tensors; the sampler is a seeded
    ``random.Random``.
    """
    torch.manual_seed(seed)
    player = mp.MetaLearnedPlayer(seed, weights={})
    cases = mp.gen_cases(seed, train_cases)
    totals: list[float] = []
    for _ in range(epochs):
        totals.extend(mp.train_player(player, cases, budget=budget))
    return player, totals


def baked_table(player: mp.MetaLearnedPlayer) -> dict[str, float]:
    """Return the snapshot — rounded, nonzero weights only."""
    return {
        name: round(w, ROUND)
        for name, w in player.weights_dict().items()
        if abs(w) > 10.0**-ROUND
    }


def _block_means(totals: list[float], blocks: int) -> list[float]:
    """Mean train reward per block — the visible training curve."""
    size = max(len(totals) // blocks, 1)
    return [
        round(sum(totals[i : i + size]) / len(totals[i : i + size]), 3)
        for i in range(0, len(totals), size)
    ]


def _random_arm(seed: int) -> Any:
    """Return a uniform player over the meta board's legal set."""

    def make(i: int) -> Any:
        r = random.Random(seed + 10_000 + i)

        def player(state: Any) -> Any | None:
            acts = ma.legal_actions(state)
            return acts[r.randrange(len(acts))] if acts else None

        return player

    return make


def _held_out(
    player: mp.MetaLearnedPlayer,
    *,
    seed: int = SEED,
    eval_cases: int = EVAL_CASES,
    budget: int = BUDGET,
) -> dict[str, float]:
    """Play the arms on the held-out stream; mean reward per arm."""
    torch.manual_seed(seed)
    ev = mp.gen_cases(seed + HELD_OUT_OFFSET, eval_cases)
    cold = mp.MetaLearnedPlayer(seed, weights={})
    arms = {
        "random": _random_arm(seed),
        "cold": lambda i: cold.frozen(seed + 20_000 + i),
        "cold-greedy": lambda i: cold.frozen(
            seed + 30_000 + i, greedy=True
        ),
        "trained": lambda i: player.frozen(seed + 40_000 + i),
        "trained-greedy": lambda i: player.frozen(
            seed + 50_000 + i, greedy=True
        ),
        "scripted": lambda i: ma.ScriptedPlayer(),
    }
    table = mp.evaluate(arms, ev, budget=budget)
    return {
        name: round(sum(r["reward"] for r in rows) / len(rows), 3)
        for name, rows in table.items()
    }


def _literal(table: dict[str, float]) -> str:
    """Render the snapshot as the ``lawdata`` assignment literal."""
    head = "META_ARENA_PLAYER_WEIGHTS: dict[str, float] = {"
    rows = "".join(
        f'\n    "{name}": {value!r},' for name, value in table.items()
    )
    return f"{head}{rows}\n}}" if table else f"{head}}}"


def main(argv: list[str] | None = None) -> int:
    """Train, print the curve, the literal, and the held-out table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-cases", type=int, default=TRAIN_CASES)
    ap.add_argument("--eval-cases", type=int, default=EVAL_CASES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    player, totals = train(
        seed=args.seed,
        train_cases=args.train_cases,
        epochs=args.epochs,
        budget=args.budget,
    )
    table = baked_table(player)
    held = _held_out(
        player,
        seed=args.seed,
        eval_cases=args.eval_cases,
        budget=args.budget,
    )
    curve = _block_means(totals, args.blocks)
    print(  # stdout-compat
        f"== train meta prior — seed {args.seed}, "
        f"{args.train_cases} cases x {args.epochs} epochs =="
    )
    print(f"train block means: {curve}")  # stdout-compat
    print("== held-out (seed + 7919) ==")  # stdout-compat
    for name, reward in held.items():
        print(f"  {name:16s} reward={reward:8.3f}")  # stdout-compat
    print("== baked snapshot ==")  # stdout-compat
    print(_literal(table))  # stdout-compat
    if args.json:
        payload: dict[str, Any] = {
            "weights": table,
            "curve": curve,
            "held_out": held,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
