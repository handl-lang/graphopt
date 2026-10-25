"""Does a grown bank help the **same** player? — the G* experiment.

The claim the stack rests on is a loop: play mints rules, the minted
rules enter the bank, and the grown bank makes the *same* player better
without retraining. The first half is measured
(`docs/reports/primitives_finds.md` — 21/21 machine-original). The second
half has never been measured. This driver measures it.

Protocol — paired boards, one frozen player:

* **two banks.** ``v0`` = ``DEFAULT`` minus every ``prim:synop*`` fold
  (the bank before the vocabulary grew); ``v1`` = ``DEFAULT`` (with the
  57 minted folds seated).
* **three arms**, each trained once and then **frozen** (weights fixed,
  no learning at eval): ``train-v0``, ``train-v1`` (the control that
  shows what the new moves are worth when the player has seen them), and
  ``random`` (the bank effect with no learning at all).
* **evaluation** crosses every frozen arm with both banks' boards on the
  same held-out cases. The G* number is the cell
  ``train-v0`` on **v1** minus the same player on **v0**: growth, same
  player, no retraining.

Why it can be zero or negative: featurization is per-vocabulary. A move
the player has never seen scores through the same feature names, so the
new rules can only help if the features already cover them.

Run::

    .venv/bin/python -m bench.drivers.policy.bank_growth
    .venv/bin/python -m bench.drivers.policy.bank_growth --json /tmp/bg.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from gamekit import engine
from gamekit.players import LinearPolicy
from mlopt_core.laws import DEFAULT
from mlopt_discovery import play as P

PRIM_PREFIX = "prim:synop"


def banks(drop: str = PRIM_PREFIX) -> dict[str, tuple[Any, ...]]:
    """Return two banks: everything, and everything minus *drop* (a regex).

    ``drop`` names the growth under test.  The default is the minted prim
    family; a control run drops the laws that actually *pay* on the
    search pool, which is how the instrument is shown to be sensitive.
    """
    v1 = tuple(DEFAULT)
    pattern = re.compile(drop)
    v0 = tuple(r for r in v1 if not pattern.search(r.name))
    return {"v0": v0, "v1": v1}


def board_of(rules: tuple[Any, ...]):
    """Return a board factory bound to one bank — the experiment's seam."""

    def _board(case: Any) -> Any:
        return P.SearchBoard(case[1], rules)

    return _board


def _arm(player: LinearPolicy):
    """Return a frozen-arm factory: every episode gets fixed weights."""
    frozen = player.frozen(0)
    return lambda _i: frozen


def _decades(totals: list[float]) -> list[float]:
    step = max(len(totals) // 5, 1)
    return [
        round(sum(totals[i : i + step]) / step, 3)
        for i in range(0, len(totals) - step + 1, step)
    ]


def run(
    *,
    seed: int = 0,
    train_cases: int = 24,
    eval_cases: int = 32,
    budget: int = 24,
    epochs: int = 3,
    drop: str = PRIM_PREFIX,
) -> dict[str, Any]:
    """Train one player per bank, then cross them on both banks."""
    domain = P.DOMAINS["search"]
    train = domain.cases(seed, train_cases)
    held = domain.cases(seed + 7919, eval_cases)
    bs = banks(drop)

    players: dict[str, LinearPolicy] = {}
    train_totals: dict[str, list[float]] = {}
    for name, rules in bs.items():
        player = LinearPolicy(
            seed=seed,
            legal=domain.legal,
            featurizer=domain.featurizer,
            features=domain.features,
            temperature=1.0,
            epsilon=0.1,
        )
        totals: list[float] = []
        for _ in range(epochs):
            totals += engine.train_policy(
                player, train, board_of(rules), budget=budget
            )
        players[name] = player
        train_totals[name] = totals

    rnd = P._random_policy(domain.legal)
    arms: dict[str, Any] = {
        "train-v0": _arm(players["v0"]),
        "train-v1": _arm(players["v1"]),
        "random": lambda i: rnd(seed + 10_000 + i),
    }

    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for bank_name, rules in bs.items():
        table = engine.evaluate(
            arms, held, board_of(rules), budget=budget
        )
        cells[bank_name] = {
            arm: {
                "reward": round(
                    statistics.mean(r["reward"] for r in rows), 4
                ),
                "steps": round(
                    statistics.mean(r["steps"] for r in rows), 2
                ),
                "n": len(rows),
            }
            for arm, rows in table.items()
        }

    gstar = (
        cells["v1"]["train-v0"]["reward"]
        - cells["v0"]["train-v0"]["reward"]
    )
    return {
        "seed": seed,
        "cases": {"train": len(train), "eval": len(held)},
        "budget": budget,
        "epochs": epochs,
        "drop": drop,
        "bank_sizes": {k: len(v) for k, v in bs.items()},
        "cells": cells,
        "gstar_delta": round(gstar, 4),
        "train_decade_means": {
            k: _decades(v) for k, v in train_totals.items()
        },
    }


def summary(result: dict[str, Any]) -> str:
    """Render the cross as a table plus the G* number."""
    lines = [
        f"banks {result['bank_sizes']} · "
        f"cases {result['cases']} · budget {result['budget']} · "
        f"epochs {result['epochs']}",
        "",
        "| frozen player | board v0 | board v1 | delta (v1-v0) |",
        "|---|---|---|---|",
    ]
    for arm in ("train-v0", "train-v1", "random"):
        v0 = result["cells"]["v0"][arm]["reward"]
        v1 = result["cells"]["v1"][arm]["reward"]
        lines.append(
            f"| {arm} | {v0:.3f} | {v1:.3f} | {v1 - v0:+.3f} |"
        )
    lines += [
        "",
        "**G\\* delta (train-v0, grown bank, no retraining): "
        f"{result['gstar_delta']:+.4f}**",
        "",
        "| bank | train decade means |",
        "|---|---|",
    ]
    for bank, means in result["train_decade_means"].items():
        lines.append(f"| {bank} | {means} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the experiment and print the cross."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=24)
    ap.add_argument("--eval-cases", type=int, default=32)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument(
        "--drop",
        default=PRIM_PREFIX,
        help="regex naming the growth under test (default: prim folds)",
    )
    args = ap.parse_args(argv)
    result = run(
        seed=args.seed,
        train_cases=args.train_cases,
        eval_cases=args.eval_cases,
        budget=args.budget,
        epochs=args.epochs,
        drop=args.drop,
    )
    print(summary(result))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
