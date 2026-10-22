"""Strassen rediscovery probe — the Q2 go/no-go measurement.

Runs the ``bilinear`` domain's arms over paired episodes and
reports the deliverable number: the best *valid* rank each arm
reached (``Report.cost`` — 8 is the naive scheme, 7 is
Strassen's, the proven optimum).  Arms:

* ``scripted`` — the recorded Strassen line (cut the committed
  scheme, commit the 7 Strassen terms): the expressibility
  witness, must reach 7 in 29 moves;
* ``playbook`` — the greedy residual-repair policy (cheapest
  ``u⊗v⊗w`` patch under L1): the honest local-search arm;
* ``random`` — uniform draws over the Brent-equation move set;
* ``linear`` — the domain's spec-driven learned arm, cold
  (``BILINEAR_SPEC`` ships no prior — the untrained baseline).

Usage::

    .venv/bin/python bench/drivers/probes/strassen_probe.py \
        --episodes 60 --budget 48

The honest-scope caveat is part of the result: this board tests
rediscovery *inside a move vocabulary built for bilinear
algorithms*, not from-scratch discovery — every legal move is a
Brent-equation identity.
"""

from __future__ import annotations

import argparse
import sys
import time
from statistics import fmean


def _arms(domain: str, seed: int) -> dict:
    """Build the arm factories: scripted + the registry players."""
    from gamekit.players import registry
    from mlopt_discovery import play

    d = play.DOMAINS[domain]
    arms = {
        "scripted": lambda i: d.scripted(),
        "playbook": registry.make(
            "playbook", domain, seed=seed, ctx={}
        ),
        "random": registry.make("random", domain, seed=seed, ctx={}),
        "linear": registry.make("linear", domain, seed=seed, ctx={}),
    }
    return arms


def _episode(board_of, case, player, budget: int) -> dict:
    """One episode — best valid rank, reward, step count."""
    from gamekit import engine

    traj = engine.run_episode(board_of(case), player, budget)
    best = min(
        (r.detail["best_rank"] for r in traj.reports),
        default=None,
    )
    term = traj.terminal
    return {
        "steps": len(traj.reports),
        "reward": round(traj.total, 2),
        "best_rank": best,
        "terminal_rank": (
            term.detail["rank"] if term is not None else None
        ),
        "won": bool(term is not None and term.cost <= 7),
    }


def main(argv: list | None = None) -> int:
    """Paired episodes per arm; report the rank distribution."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="bilinear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument(
        "--det-episodes",
        type=int,
        default=10,
        help="episodes for the deterministic arms (scripted/playbook)",
    )
    args = ap.parse_args(argv)

    from mlopt_discovery import play

    d = play.DOMAINS[args.domain]
    cases = d.cases(args.seed + 7919, args.episodes)
    arms = _arms(args.domain, args.seed)

    print(
        f"== strassen probe — domain {args.domain} =="
    )  # stdout-compat
    print(
        f"{'arm':<10} {'eps':>4} {'won':>4} {'pct<=7':>8} "
        f"{'rank~':>6} {'reward~':>9} {'steps~':>6} {'s':>6}"
    )
    print("-" * 52)
    summary = {}
    for name, mk in arms.items():
        n = (
            args.det_episodes
            if name in ("scripted", "playbook")
            else args.episodes
        )
        t0 = time.time()
        rows = [
            _episode(
                d.board_of, cases[i % len(cases)], mk(i), args.budget
            )
            for i in range(n)
        ]
        won = sum(r["won"] for r in rows)
        ranks = [
            r["best_rank"] for r in rows if r["best_rank"] is not None
        ]
        row = {
            "eps": n,
            "won": won,
            "pct7": round(100.0 * won / n, 1),
            "mean_rank": round(fmean(ranks), 2) if ranks else None,
            "mean_reward": round(fmean(r["reward"] for r in rows), 2),
            "mean_steps": round(fmean(r["steps"] for r in rows), 1),
            "secs": round(time.time() - t0, 1),
        }
        summary[name] = row
        print(  # stdout-compat
            f"{name:<10} {n:>4} {won:>4} {row['pct7']:>7.1f}% "
            f"{row['mean_rank']:>6} {row['mean_reward']:>9} "
            f"{row['mean_steps']:>6} {row['secs']:>6}"
        )
    print(  # stdout-compat
        "verdict: "
        + (
            "GO — search reached rank ≤ 7"
            if any(
                summary[a]["won"]
                for a in ("random", "linear", "playbook")
            )
            else "NO-GO — only the scripted line reached rank 7"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
