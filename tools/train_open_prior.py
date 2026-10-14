"""Train the open-game policy and emit the baked prior.

The RL arms ship *cold*: ``lawdata.OPEN_GAME_PLAYER_WEIGHTS`` is
empty, so every run trains only within its own episodes.  This tool
trains the open-game policy
(:class:`catopt_discovery.open_game.OpenLearnedPlayer`) over a fixed
page split and prints the trained weight table as a Python literal —
the snapshot that bakes into
:data:`catopt_discovery.lawdata.OPEN_GAME_PLAYER_WEIGHTS`.

The split is the held-out one the open-game measurement uses: **even
pages train, odd pages evaluate** (``gen_cases(start=0, step=2)`` vs
``start=1, step=2``).  The enumerated stream is deterministic, so the
run is reproducible from the fixed seed alone — ``gen_cases`` ignores
its seed (the page region is ``start``/``step``), and the policy's
sampler is a seeded ``random.Random``.

The run is honest: the tool prints the *train curve* (mean episode
reward per block) so a flat or diverging run is visible, not hidden,
and it reports the held-out odd-page reward of the baked snapshot
against the cold (``{}``) and random arms so the bake is earned by a
measurement, not assumed.

Usage::

    python tools/train_open_prior.py            # print curve + literal
    python tools/train_open_prior.py --epochs 5 --json /tmp/prior.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from catopt_discovery import engine, play
from catopt_discovery import open_game as og

#: Defaults — the held-out split's page size and the even-page count.
PAGE = 16
TRAIN_PAGES = 36
EPOCHS = 5
SEED = 0
#: Round the baked weights to this many decimals — a shipped snapshot
#: is a clean literal, and the policy only ever reads ``w·φ``.
ROUND = 4


def _budget(page: int) -> int:
    """Episode horizon — one eval per pool member plus ``stop``."""
    return page + 1


def train(
    *,
    seed: int = SEED,
    train_pages: int = TRAIN_PAGES,
    page: int = PAGE,
    epochs: int = EPOCHS,
) -> tuple[og.OpenLearnedPlayer, list[float]]:
    """Train a *cold* open-game policy over the even train pages.

    Returns the trained player and the per-episode train totals (the
    curve is ``epochs x train_pages`` long).  The player starts from
    ``weights={}`` — the cold start — so the bake is a genuine
    training run, not a fine-tune of a prior that is already baked.

    ``torch.manual_seed(seed)`` fixes the numeric oracle's tensors
    (``proposal.evaluate`` draws them from torch's global RNG), so the
    run is reproducible from *seed* alone — the sampler is a seeded
    ``random.Random`` and the enumerated stream is deterministic.
    """
    torch.manual_seed(seed)
    player = og.OpenLearnedPlayer(seed=seed, weights={})
    cases = og.gen_cases(seed, train_pages, page=page, start=0, step=2)
    budget = _budget(page)
    totals: list[float] = []
    for _ in range(epochs):
        totals.extend(
            engine.train_policy(
                player, cases, og.OpenArena, budget=budget
            )
        )
    return player, totals


def baked_table(player: og.OpenLearnedPlayer) -> dict[str, float]:
    """Return the snapshot — rounded, nonzero weights only.

    ``LinearPolicy`` reads a missing feature as ``0``, so dropping the
    near-zero entries keeps the shipped literal honest and compact
    without changing the ordering.
    """
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


def _held_out(
    player: og.OpenLearnedPlayer,
    *,
    seed: int = SEED,
    train_pages: int = TRAIN_PAGES,
    page: int = PAGE,
) -> dict[str, dict[str, float]]:
    """Play the arms on the held-out odd pages; tally yield per arm."""
    torch.manual_seed(seed)
    ev = og.gen_cases(seed, train_pages, page=page, start=1, step=2)
    budget = _budget(page)
    cold = og.OpenLearnedPlayer(seed=seed, weights={})
    d = play.DOMAINS["open"]
    arms = {
        "random": play._arms(d, player, seed)["random"],
        "cold": lambda i: cold.frozen(seed + 10_000 + i),
        "baked": lambda i: player.frozen(seed + 20_000 + i),
        "baked-greedy": lambda i: player.frozen(
            seed + 30_000 + i, greedy=True
        ),
    }
    out: dict[str, dict[str, float]] = {}
    for name, mk in arms.items():
        found = evals = 0
        reward = 0.0
        for i, case in enumerate(ev):
            traj = engine.run_episode(og.OpenArena(case), mk(i), budget)
            reward += traj.total
            for r in traj.reports:
                evals += 1
                if r.note and r.note.split(":", 1)[0] == "useful":
                    found += 1
        n = len(ev)
        out[name] = {
            "reward": round(reward / n, 4),
            "useful_per_ep": round(found / n, 3),
            "useful_per_eval": round(found / max(evals, 1), 4),
        }
    return out


def _literal(table: dict[str, float]) -> str:
    """Render the snapshot as the ``lawdata`` assignment literal."""
    head = "OPEN_GAME_PLAYER_WEIGHTS: dict[str, float] = {"
    rows = "".join(
        f'\n    "{name}": {value!r},' for name, value in table.items()
    )
    return f"{head}{rows}\n}}" if table else f"{head}}}"


def main(argv: list[str] | None = None) -> int:
    """Train, print the curve, the literal, and the held-out table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-pages", type=int, default=TRAIN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    player, totals = train(
        seed=args.seed,
        train_pages=args.train_pages,
        page=args.page,
        epochs=args.epochs,
    )
    table = baked_table(player)
    held = _held_out(
        player,
        seed=args.seed,
        train_pages=args.train_pages,
        page=args.page,
    )
    curve = _block_means(totals, args.blocks)
    print(  # stdout-compat
        f"== train open prior — seed {args.seed}, "
        f"{args.train_pages} even pages x {args.epochs} epochs =="
    )
    print(f"train block means: {curve}")  # stdout-compat
    print("== held-out (odd pages) ==")  # stdout-compat
    for name, row in held.items():
        print(  # stdout-compat
            f"  {name:14s} reward={row['reward']:8.4f} "
            f"use/ep={row['useful_per_ep']:.3f} "
            f"use/eval={row['useful_per_eval']:.4f}"
        )
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
