"""Train the kernel-board policy and emit the baked prior.

The sibling of ``scripts/train_meta_prior.py`` for the ``kernel``
domain — the kernelopt spike's open arm from
``project/notes/kernel-engineer-board.md``: the shipped
``KERNEL_PLAYER_WEIGHTS`` was trained on the 5-law bank, so its
``a:r:*`` buckets cover only those rules' hashes (the expanded
bank's laws ride the untrained ``x:m`` match feature alone).
This driver retrains the prior on the full bank.

The recipe is the shipped one's: warm-start by imitating the
*match-then-stop oracle* (fire every law in ``state.matches`` once,
in bank order, then ``stop`` -- the scripted optimum on this board,
whose demonstrations are free) through
:func:`gamekit.engine.imitate_policy`, then REINFORCE on the kernel
case stream.  Cold REINFORCE alone is the documented plateau
(~0.20-0.28 at 600 cases) -- the imitation phase is what carries
the prior, and this run re-measures that dependence.

The split is the registry's: train on
``DOMAINS["kernel"].cases(seed, n)``, evaluate on the disjoint
``cases(seed + 7919, n)`` stream the ``play``/leaderboard drivers
use.  The held-out table rows the arms honestly — random, cold,
cold-greedy, the *shipped* prior (the incumbent this bake must
beat), trained, trained-greedy, and the scripted playbook.

Usage::

    .venv/bin/python scripts/train_kernel_prior.py
    .venv/bin/python scripts/train_kernel_prior.py \
        --epochs 10 --imitate-epochs 5 --json /tmp/kernel.json
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from gamekit import engine
from gamekit.players import LearnedPlayer
from mlopt_discovery import lawdata, play

#: Defaults — the kernel domain's own horizon and case stream.
TRAIN_CASES = 64
EVAL_CASES = 32
EPOCHS = 8
IMITATE_EPOCHS = 4
BUDGET = 8
SEED = 0
#: The held-out case stream's seed offset — ``play``'s convention.
HELD_OUT_OFFSET = 7919
#: Round the baked weights to this many decimals.
ROUND = 4


def _match_oracle() -> Any:
    """Return a fresh match-then-stop expert — the scripted optimum.

    Fires each law the board's ``matches`` flags once (bank order —
    deeper fusions first), then ``stop``: the line a perfect player
    plays on this board, and the demonstrations imitation consumes.
    A fresh copy per episode — the played set is per-board state.
    """
    played: set = set()

    def go(state: Any) -> Any | None:
        if state.done:
            return None
        for name in state.actions:
            if name in state.matches and name not in played:
                played.add(name)
                return name
        return "stop"

    return go


def train(
    *,
    seed: int = SEED,
    train_cases: int = TRAIN_CASES,
    epochs: int = EPOCHS,
    imitate_epochs: int = IMITATE_EPOCHS,
    budget: int = BUDGET,
) -> tuple[LearnedPlayer, list[float], dict]:
    """Imitation warm-start, then REINFORCE over the case stream.

    The player starts from ``weights={}`` (the cold start), so the
    bake is a genuine training run on the current law bank.  Returns
    ``(player, totals, imitation)`` — the arm, the per-episode
    returns and the demonstrated/imitated counts.
    """
    d = play.DOMAINS["kernel"]
    player = LearnedPlayer(seed, spec=d.spec, weights={})
    stream = d.cases(seed, train_cases)
    imitation = {"shown": 0, "imitated": 0}
    for _ in range(imitate_epochs):
        r = engine.imitate_policy(
            player,
            lambda i: _match_oracle(),
            stream,
            d.board_of,
            budget=budget,
        )
        imitation["shown"] += r["shown"]
        imitation["imitated"] += r["imitated"]
    totals: list[float] = []
    for _ in range(epochs):
        totals.extend(
            engine.train_policy(
                player, stream, d.board_of, budget=budget
            )
        )
    return player, totals, imitation


def cold_train(
    *,
    seed: int = SEED,
    train_cases: int = TRAIN_CASES,
    epochs: int = EPOCHS,
    budget: int = BUDGET,
) -> tuple[LearnedPlayer, list[float]]:
    """Cold REINFORCE only — the imitation-dependence control arm."""
    d = play.DOMAINS["kernel"]
    player = LearnedPlayer(seed, spec=d.spec, weights={})
    stream = d.cases(seed, train_cases)
    totals: list[float] = []
    for _ in range(epochs):
        totals.extend(
            engine.train_policy(
                player, stream, d.board_of, budget=budget
            )
        )
    return player, totals


def baked_table(player: LearnedPlayer) -> dict[str, float]:
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
    """Return a uniform player over the kernel board's legal set."""
    d = play.DOMAINS["kernel"]

    def make(i: int) -> Any:
        r = random.Random(seed + 10_000 + i)

        def player(state: Any) -> Any | None:
            acts = d.spec.legal(state)
            return acts[r.randrange(len(acts))] if acts else None

        return player

    return make


def _held_out(
    trained: LearnedPlayer,
    cold_rl: LearnedPlayer | None = None,
    *,
    seed: int = SEED,
    eval_cases: int = EVAL_CASES,
    budget: int = BUDGET,
) -> dict[str, float]:
    """Play the arms on the held-out stream; mean reward per arm."""
    d = play.DOMAINS["kernel"]
    ev = d.cases(seed + HELD_OUT_OFFSET, eval_cases)
    shipped = LearnedPlayer(
        seed, spec=d.spec, weights=lawdata.KERNEL_PLAYER_WEIGHTS
    )
    cold = LearnedPlayer(seed, spec=d.spec, weights={})
    arms: dict[str, Any] = {
        "random": _random_arm(seed),
        "cold": lambda i: cold.frozen(seed + 20_000 + i),
        "shipped": lambda i: shipped.frozen(seed + 30_000 + i),
        "shipped-greedy": lambda i: shipped.frozen(
            seed + 35_000 + i, greedy=True
        ),
        "trained": lambda i: trained.frozen(seed + 40_000 + i),
        "trained-greedy": lambda i: trained.frozen(
            seed + 50_000 + i, greedy=True
        ),
        "scripted": lambda i: play._scripted_kernel(),
    }
    if cold_rl is not None:
        arms["cold-rl"] = lambda i: cold_rl.frozen(
            seed + 60_000 + i, greedy=True
        )
    table = engine.evaluate(arms, ev, d.board_of, budget=budget)
    return {
        name: round(sum(r["reward"] for r in rows) / len(rows), 3)
        for name, rows in table.items()
    }


def _literal(table: dict[str, float]) -> str:
    """Render the snapshot as the ``gamedata`` assignment literal."""
    head = "KERNEL_PLAYER_WEIGHTS: dict[str, float] = {"
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
    ap.add_argument(
        "--imitate-epochs", type=int, default=IMITATE_EPOCHS
    )
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument(
        "--cold-rl",
        action="store_true",
        help="also train a cold-REINFORCE-only control arm",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    player, totals, imitation = train(
        seed=args.seed,
        train_cases=args.train_cases,
        epochs=args.epochs,
        imitate_epochs=args.imitate_epochs,
        budget=args.budget,
    )
    cold_rl = None
    cold_totals: list[float] = []
    if args.cold_rl:
        cold_rl, cold_totals = cold_train(
            seed=args.seed,
            train_cases=args.train_cases,
            epochs=args.epochs,
            budget=args.budget,
        )
    table = baked_table(player)
    held = _held_out(
        player,
        cold_rl,
        seed=args.seed,
        eval_cases=args.eval_cases,
        budget=args.budget,
    )
    curve = _block_means(totals, args.blocks)
    print(  # stdout-compat
        f"== train kernel prior — seed {args.seed}, "
        f"{args.train_cases} cases x {args.epochs} epochs (+ "
        f"{args.imitate_epochs} imitation ep) =="
    )
    print(  # stdout-compat
        f"imitation: {imitation['imitated']}/{imitation['shown']}"
        " demonstrated moves cloned"
    )
    print(f"train block means: {curve}")  # stdout-compat
    if cold_totals:
        print(  # stdout-compat
            "cold-rl block means:"
            f" {_block_means(cold_totals, args.blocks)}"
        )
    print("== held-out (seed + 7919) ==")  # stdout-compat
    for name, reward in held.items():
        print(f"  {name:16s} reward={reward:8.3f}")  # stdout-compat
    print("== baked snapshot ==")  # stdout-compat
    print(_literal(table))  # stdout-compat
    if args.json:
        payload: dict[str, Any] = {
            "weights": table,
            "curve": curve,
            "imitation": imitation,
            "cold_curve": (
                _block_means(cold_totals, args.blocks)
                if cold_totals
                else None
            ),
            "held_out": held,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
