"""Train the learned meta-arm on the ``player`` domain.

``docs/results_player_on_board.md``'s open thread: ``PLAYER_SPEC`` +
``legal_actions`` give the generic ``LearnedPlayer`` a board to learn
on, but each ``evaluate`` probe is seconds-to-minutes of nested
train+eval, so the shipped demo arm is the scripted ``GreedyEditor``.
This driver trains the meta-arm and checks whether it emits a child
as good as ``GreedyEditor``'s (``['open:action']``, held-out 0.2827).

Two phases, honestly separated:

* **train** — REINFORCE over ``PlayerArena`` episodes whose nested
  oracle is a *distilled stub score* injected through the board's
  ``score_fn`` seam (the seam ``tests/test_discovery_player_arena``
  uses): ``score = 0.28ish`` iff ``open:action`` is the only
  load-bearing part — the leaderboard's own part-importance finding
  (``action`` LOO -0.128, every other part LOO ≈ 0) rendered as a
  cheap callable.  A stub episode costs milliseconds, so the meta-
  training cap is the *episode count*, not wall time.  The stub
  teaches move sequencing (which edits, when to probe, when to
  extract); it cannot make the emitted child's real held-out score
  any better — only the real oracle measures that.
* **held-out** — the trained policy, frozen, plays fresh
  ``PlayerArena`` episodes with **no** ``score_fn``: the canonical
  nested eval (3 epochs x 36 even pages, odd-page held-out, memoized
  per-candidate verdicts — the leaderboard protocol, and the same
  measurement ``GreedyEditor``'s 0.2827 was taken under).  The
  terminal report's ``child_score`` is the number that decides.

Usage::

    timeout 1800 .venv/bin/python scripts/train_player_meta_arm.py
    timeout 1800 .venv/bin/python scripts/train_player_meta_arm.py \
        --skip-real --json /tmp/player_arm.json   # train curve only
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from gamekit import engine
from gamekit.players import LearnedPlayer
from mlopt_discovery import player_arena as pa

__all__ = ["GREEDY_CHILD_SCORE", "distilled_open", "main", "train"]

SEED = 0
#: Meta-episode step budget — the scripted editor spent 9.
BUDGET = 16
#: ``GreedyEditor``'s emitted child under the canonical nested eval —
#: the held-out bar the learned arm must meet.
GREEDY_CHILD_SCORE = 0.2827


def distilled_open(record: dict, *, seed: int, cfg: dict) -> dict:
    """Stub oracle distilled from the leaderboard's three tiers.

    The schema leaderboard's measured landscape over part subsets:
    ``open:action`` absent → the blind-schema plateau (~0.10);
    ``action`` present and ``onehot`` absent → the winner tier
    (~0.28 — *every* ``onehot``-free ``action``-bearing subset tied
    at 0.2827); ``onehot`` present → the incumbent tier (~0.23 —
    the sampled-reward cost ``onehot`` buys its greedy/AP gains
    with).  A first draft scored ``action``-bearing records by
    part count alone; the trained arm then dropped ``action``
    before ``onehot`` (both read as "extra part") and could not
    recover — the *tiers* are the load-bearing shape.  This is a
    protocol stand-in for the nested train+eval — 32 subsets mapped
    to three published values, not a per-record cheat: the emitted
    child's real score is measured only in the held-out phase.
    """
    del seed, cfg
    parts = set(record["parts"])
    if "open:action" not in parts:
        score = 0.0989
    elif "open:onehot" not in parts:
        score = 0.2827
    else:
        score = 0.2301
    return {"score": score, "greedy": score, "evals_ep": 0.0}


def _train_cases(
    seed: int, n: int, max_evals: int, score_fn: Any
) -> list[dict]:
    """Stub-oracle player-board cases — seeds and eval caps varied."""
    return [
        {
            "name": f"stub{i}:open",
            "objective": "open",
            "seed": seed + i,
            "max_evals": max_evals - (i % 3),
            "score_fn": score_fn,
        }
        for i in range(n)
    ]


def train(
    *,
    seed: int = SEED,
    cases: int = 12,
    epochs: int = 150,
    imitate_epochs: int = 6,
    budget: int = BUDGET,
    max_evals: int = 8,
    score_fn: Any = distilled_open,
    epsilon: float = 0.0,
) -> tuple[LearnedPlayer, list[float], list[dict], dict]:
    """Warm-start by cloning ``GreedyEditor``, then REINFORCE.

    Cold REINFORCE alone stalls: the paying line (strip the dead
    parts, keep ``open:action``, extract) is ~5 moves deep in a
    ~40-action legal set, and the EMA-baseline gradient mostly sees
    meandering episodes, which pushes ``a:dir:drop`` *down*.
    :func:`engine.imitate_policy` replays the scripted editor's own
    episodes on the stub board (its demonstrations are free under
    the stub) and takes a cross-entropy step toward each move —
    the codebase's bootstrap idiom.  REINFORCE then trains from
    that warm start on its own returns.

    Returns ``(player, totals, traces, imitation)`` — the arm (cold
    start, ``weights={}``), the per-episode returns, one action
    trace per epoch boundary, and the shown/imitated counts.
    """
    player = LearnedPlayer(
        seed, spec=pa.PLAYER_SPEC, weights={}, epsilon=epsilon
    )
    board_of = pa.PlayerArena
    stream = _train_cases(seed, cases, max_evals, score_fn)
    imitation = {"shown": 0, "imitated": 0}
    for _ in range(imitate_epochs):
        r = engine.imitate_policy(
            player,
            lambda i: pa.GreedyEditor(),
            stream,
            board_of,
            budget=budget,
        )
        imitation["shown"] += r["shown"]
        imitation["imitated"] += r["imitated"]
    totals: list[float] = []
    traces: list[dict] = []
    for ep in range(epochs):
        epoch_totals = engine.train_policy(
            player, stream, board_of, budget=budget
        )
        totals.extend(epoch_totals)
        if ep in {0, epochs // 2, epochs - 1}:
            board = board_of(stream[0])
            traj = engine.run_episode(
                board, player.frozen(seed + 70_000 + ep), budget
            )
            traces.append(
                {
                    "epoch": ep,
                    "ops": [r.action.op for r in traj.reports],
                    "reward": round(traj.total, 4),
                }
            )
    return player, totals, traces, imitation


def _held_out_episode(
    player: LearnedPlayer, case: dict, *, greedy: bool, budget: int
) -> dict:
    """Play one real-oracle episode; return the terminal child row."""
    board = pa.PlayerArena(case)
    arm = player.frozen(case["seed"] + 80_000, greedy=greedy)
    traj = engine.run_episode(board, arm, budget)
    t = traj.terminal
    row: dict[str, Any] = {
        "case": case.get("name", case["seed"]),
        "greedy": greedy,
        "steps": len(traj.reports),
        "reward": round(traj.total, 4),
        "ops": [r.action.op for r in traj.reports],
        "evals": board.evals,
    }
    if t is not None:
        d = t.detail
        row.update(
            {
                "child_parts": list(d["child"]["parts"]),
                "child_score": round(d["child_score"], 4),
                "parent_score": round(d["parent_score"], 4),
                "delta": round(d["delta"], 4),
            }
        )
    return row


def held_out(
    player: LearnedPlayer,
    *,
    seeds: tuple = (0, 1),
    budget: int = BUDGET,
    max_evals: int = 8,
) -> list[dict]:
    """Frozen arms on real-oracle episodes — the canonical nested eval."""
    rows: list[dict] = []
    for s in seeds:
        case = {
            "name": f"real{s}:open",
            "objective": "open",
            "seed": s,
            "max_evals": max_evals,
        }
        for greedy in (False, True):
            rows.append(
                _held_out_episode(
                    player, case, greedy=greedy, budget=budget
                )
            )
    return rows


def _block_means(totals: list[float], blocks: int) -> list[float]:
    """Mean train reward per block — the visible training curve."""
    size = max(len(totals) // blocks, 1)
    return [
        round(sum(totals[i : i + size]) / len(totals[i : i + size]), 3)
        for i in range(0, len(totals), size)
    ]


def main(argv: list[str] | None = None) -> int:
    """Train on the stub oracle; check the child under the real one."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imitate-epochs", type=int, default=6)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--max-evals", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument(
        "--skip-real",
        action="store_true",
        help="train curve only — skip the real-oracle held-out check",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    player, totals, traces, imitation = train(
        seed=args.seed,
        cases=args.cases,
        epochs=args.epochs,
        imitate_epochs=args.imitate_epochs,
        budget=args.budget,
        max_evals=args.max_evals,
        epsilon=args.epsilon,
    )
    curve = _block_means(totals, 6)
    print(  # stdout-compat
        f"== train meta-arm — cold PLAYER_SPEC, stub oracle, "
        f"{args.cases} cases x {args.epochs} ep (+ "
        f"{args.imitate_epochs} imitation ep), budget {args.budget} =="
    )
    print(  # stdout-compat
        f"imitation: {imitation['imitated']}/{imitation['shown']}"
        f" demonstrated moves cloned"
    )
    print(f"train block means: {curve}")  # stdout-compat
    for tr in traces:
        print(
            f"  trace[ep {tr['epoch']}]: {tr['ops']}"
        )  # stdout-compat

    rows: list[dict] = []
    if not args.skip_real:
        print(  # stdout-compat
            "\n== held-out — real nested eval (canonical cfg),"
            " frozen arms =="
        )
        rows = held_out(
            player,
            seeds=tuple(args.eval_seeds),
            budget=args.budget,
            max_evals=args.max_evals,
        )
        for r in rows:
            print(json.dumps(r))  # stdout-compat
        best = max(
            (r.get("child_score", float("-inf")) for r in rows),
            default=float("-inf"),
        )
        print(  # stdout-compat
            f"best emitted child: {best:.4f} vs GreedyEditor"
            f" {GREEDY_CHILD_SCORE:.4f}"
        )
    weights = {
        n: round(w, 4)
        for n, w in player.weights_dict().items()
        if abs(w) > 1e-4
    }
    print("== trained weight snapshot ==")  # stdout-compat
    print(
        json.dumps(weights, indent=1, sort_keys=True)
    )  # stdout-compat
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "protocol": {
                        "cases": args.cases,
                        "epochs": args.epochs,
                        "imitate_epochs": args.imitate_epochs,
                        "budget": args.budget,
                        "max_evals": args.max_evals,
                        "oracle": "distilled_open stub (score_fn seam)",
                    },
                    "curve": curve,
                    "imitation": imitation,
                    "traces": traces,
                    "held_out": rows,
                    "greedy_child_bar": GREEDY_CHILD_SCORE,
                    "weights": weights,
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
