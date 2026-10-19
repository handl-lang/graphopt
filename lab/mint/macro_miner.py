"""Mint macro actions from played trajectories — moves the way laws are minted.

The discovery loop ``antiunify`` runs on rules, run on *moves*:
collect :class:`~mlopt_discovery.meta_arena.Trajectory` objects from
played episodes (scripted / greedy / random / trained arms plus the
claim- and declare-line probes), count canonical action n-grams
across episodes, rank by ``count x payoff`` (the mean terminal-step
reward of the episodes the gram rode in — "did the subsequence
precede a paying certified extract?"), and mint the top grams as
:data:`lab.mint.macro_moves.Macro` templates — data replayed through
the refereed ``macro`` handler, sound by construction.

Smoke test: the minted table must surface the known winning line
``declare→saturate→handle→extract`` (or its ``claim``-fused form
``claim→extract``) — if it does, the miner works.

Eval: held-out ``meta_player.gen_cases(seed + 7919, n)`` under
budget 12 — ``scripted`` vs ``scripted+macros`` vs ``trained-linear``
vs ``trained+macros`` (frozen shipped prior; macro offers enter
through the ``legal=`` seam, the policy's weights untouched).

Run::

    .venv/bin/python lab/mint/macro_miner.py
    .venv/bin/python lab/mint/macro_miner.py --mine-cases 24 --json /tmp/macro.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mlopt_core.cost.basic import count_cost
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import meta_player as mp
from mlopt_discovery.meta_arena import Action

from lab.mint.macro_moves import (
    Macro,
    instantiate,
    legal_with_macros,
    viable,
)

__all__ = [
    "MacroScripted",
    "collect_trajectories",
    "main",
    "mine_macros",
    "run_eval",
]

SEED = 0
HELD_OUT_OFFSET = 7919  # ``play``'s held-out seed offset (meta)
EVAL_CASES = 16
BUDGET = 12
MINE_CASES = 24
MINE_BUDGET = 24
LENGTHS = (2, 3, 4, 5)
MIN_COUNT = 2
TOP_MACROS = 8
DECLARE_CAP = 10
RNG_EPISODES = 2


# ---------------------------------------------------------------------------
#  Trajectory collection — every arm's episodes on the train stream
# ---------------------------------------------------------------------------


def _scripted_episode(
    board: ma.MetaArena, moves: list
) -> ma.Trajectory:
    """Play a fixed move list on *board*; stop at terminal/drained."""
    reports = []
    for a in moves:
        if board.observe().done:
            break
        _st, rep = board.step(a)
        reports.append(rep)
    return ma.Trajectory(reports)


def collect_trajectories(
    cases: list,
    *,
    seed: int = SEED,
    budget: int = MINE_BUDGET,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
    declare_cap: int = DECLARE_CAP,
    rng_episodes: int = RNG_EPISODES,
) -> list[ma.Trajectory]:
    """Play the mining arms over *cases*; return every trajectory.

    Arms per case: the scripted playbook, greedy, seeded randoms, the
    frozen shipped prior (sampled + greedy), one ``claim→extract``
    per claimable handler tag, and one declare-line episode per mined
    candidate (``declare→saturate→handle*→extract`` — the winning
    trace the smoke test must rediscover).
    """
    cf = cost_fn or count_cost
    trained = mp.MetaLearnedPlayer(seed)
    trajs: list[ma.Trajectory] = []
    for ci, case in enumerate(cases):

        def board(case: Any = case) -> ma.MetaArena:
            return mp._board(case, cf, max_nodes)

        trajs.append(
            ma.run_episode(board(), ma.ScriptedPlayer(), budget)
        )
        trajs.append(ma.run_episode(board(), ma.GreedyPlayer(), budget))
        for r in range(rng_episodes):
            trajs.append(
                ma.run_episode(
                    board(),
                    ma.RandomPlayer(
                        random.Random(seed + 1000 * ci + r)
                    ),
                    budget,
                )
            )
        trajs.append(
            ma.run_episode(
                board(), trained.frozen(seed + 5000 + ci), budget
            )
        )
        trajs.append(
            ma.run_episode(
                board(),
                trained.frozen(seed + 9000 + ci, greedy=True),
                budget,
            )
        )
        probe = board()
        for tag in probe.observe().claim_tags:
            trajs.append(
                _scripted_episode(
                    board(), [Action.claim(tag), Action.extract()]
                )
            )
        for cand in ma.declare_candidates(board())[:declare_cap]:
            trajs.append(
                ma._declare_episode(board(), cand, handle=True)
            )
    return trajs


# ---------------------------------------------------------------------------
#  n-gram mining — canonical action signatures over applied moves
# ---------------------------------------------------------------------------


def _sig(action: Action) -> tuple:
    """Canonical n-gram signature — op plus its *identity* params.

    ``declare`` keys on its spelled spec's coverage skeleton (the
    fresh name is incidental); ``handle``/``claim`` on the handler
    tag; ``fire`` on the rule; ``saturate`` on its rules/budget.
    Two winning lines that differ only in the minted fold name merge.
    """
    op = action.op
    p = action.params
    if op == "declare":
        cons = p.get("construction") or {}
        spelled = (cons.get("params") or {}).get("spelled")
        return (op, repr(ma._canon_spec(spelled)))
    if op in ("handle", "claim"):
        return (op, str(p.get("handler", "")))
    if op == "fire":
        return (op, str(p.get("rule", "")))
    if op == "saturate":
        return (op, repr(p.get("rules")), repr(p.get("budget")))
    return (op,)


def mine_macros(
    trajs: list[ma.Trajectory],
    *,
    lengths: tuple = LENGTHS,
    min_count: int = MIN_COUNT,
    top: int = TOP_MACROS,
) -> tuple[list[Macro], list[dict]]:
    """Count canonical n-grams over applied moves; mint the top set.

    An occurrence's *payoff* is its episode's terminal-step reward —
    the delta the line's certified ``extract`` collected (0 when the
    episode ran out).  ``score = count x max(0, mean payoff)``.  The
    representative sequence minted is the occurrence from the
    highest-total certified episode.  Returns ``(macros, rows)`` —
    the minted templates and the full ranked table for the report.
    """
    stats: dict[tuple, dict] = {}
    for t in trajs:
        reports = [r for r in t.reports if r.applied]
        acts = [r.action for r in reports]
        terminal_reward = t.terminal.reward if t.terminal else 0.0
        certified = bool(t.certificate_ok)
        for n in lengths:
            for i in range(len(acts) - n + 1):
                gram = acts[i : i + n]
                key = tuple(_sig(a) for a in gram)
                s = stats.setdefault(
                    key,
                    {
                        "count": 0,
                        "certified": 0,
                        "terminals": [],
                        "rep": None,
                        "rep_total": float("-inf"),
                    },
                )
                s["count"] += 1
                s["certified"] += int(certified)
                s["terminals"].append(terminal_reward)
                if certified and t.total > s["rep_total"]:
                    s["rep_total"] = t.total
                    s["rep"] = [
                        {"op": a.op, "params": dict(a.params)}
                        for a in gram
                    ]
    rows: list[dict] = []
    for key, s in stats.items():
        if s["count"] < min_count or s["rep"] is None:
            continue
        payoff = sum(s["terminals"]) / len(s["terminals"])
        rows.append(
            {
                "ops": tuple(k[0] for k in key),
                "key": key,
                "count": s["count"],
                "certified": round(s["certified"] / s["count"], 3),
                "mean_terminal": round(payoff, 4),
                "score": round(s["count"] * max(0.0, payoff), 4),
                "rep": s["rep"],
            }
        )
    rows.sort(key=lambda r: (-r["score"], -r["count"], r["ops"]))
    macros = [
        Macro(
            name=f"m{r + 1}",
            seq=tuple(row["rep"]),
            count=row["count"],
            certified=row["certified"],
            mean_terminal=row["mean_terminal"],
            score=row["score"],
            ops=row["ops"],
        )
        for r, row in enumerate(rows[:top])
    ]
    return macros, rows


# ---------------------------------------------------------------------------
#  The macro-scripted arm — minted lines first, playbook fallback
# ---------------------------------------------------------------------------


class MacroScripted:
    """``ScriptedPlayer`` + minted macros — try each viable macro once.

    Macros play in rank order; :func:`viable` keeps the line off
    cases where its recorded params cannot bind.  When the minted set
    is drained (or declined — a played macro is not retried), the
    human playbook ``saturate → extract`` finishes the episode.
    """

    def __init__(self, macros: Any) -> None:
        """Bind the ranked minted set."""
        self._macros = list(macros)
        self._tried: set[str] = set()
        self._fallback = deque([Action.saturate(), Action.extract()])
        self.played: list[str] = []

    def __call__(self, state: ma.MetaState) -> Action | None:
        """Return the next viable minted macro, else the playbook."""
        for m in self._macros:
            if m.name in self._tried:
                continue
            self._tried.add(m.name)
            if viable(m, state):
                self.played.append(m.name)
                return instantiate(m)
        return self._fallback.popleft() if self._fallback else None


# ---------------------------------------------------------------------------
#  Held-out eval — macro offers vs the base move set
# ---------------------------------------------------------------------------


def _per_case_delta(macro_rows: list, base_rows: list) -> dict:
    """Pair the two arms' rows per case; tally wins/losses and stalls.

    A *cost win* is a case where the macro arm extracted a strictly
    cheaper certified program than the base arm — the cases where the
    base playbook stalled at baseline and the minted line paid.
    """
    wins = losses = 0
    detail: list[dict] = []
    for mr, br in zip(macro_rows, base_rows, strict=True):
        d_reward = mr["reward"] - br["reward"]
        mc, bc = mr["cost"], br["cost"]
        if d_reward > 1e-9:
            wins += 1
        elif d_reward < -1e-9:
            losses += 1
        detail.append(
            {
                "case": mr["case"],
                "base_reward": br["reward"],
                "macro_reward": mr["reward"],
                "base_cost": bc,
                "macro_cost": mc,
                "used_declared": mr["used_declared"],
            }
        )
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(macro_rows) - wins - losses,
        "cost_wins": sum(
            1
            for d in detail
            if d["macro_cost"] is not None
            and (
                d["base_cost"] is None
                or d["macro_cost"] < d["base_cost"]
            )
        ),
        "rows": detail,
    }


def run_eval(
    macros: list[Macro],
    *,
    seed: int = SEED,
    cases: int = EVAL_CASES,
    budget: int = BUDGET,
    cap: int = TOP_MACROS,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> dict[str, Any]:
    """Play the four arms on the held-out stream; pair the rows.

    ``scripted`` vs ``scripted+macros`` (the playbook plus the minted
    set) and ``trained-linear`` vs ``trained+macros`` (the frozen
    shipped prior with macro offers entering through its ``legal=``
    seam — weights untouched, the featurizer scores a macro with the
    generic broadcast row).  ``greedy``/``greedy+macros`` ride along
    as context arms.
    """
    ev = mp.gen_cases(seed + HELD_OUT_OFFSET, cases)
    macro_legal = legal_with_macros(macros, cap=cap)
    trained = mp.MetaLearnedPlayer(seed)
    trained_m = mp.MetaLearnedPlayer(seed, legal=macro_legal)
    arms = {
        "scripted": lambda i: ma.ScriptedPlayer(),
        "scripted+macros": lambda i: MacroScripted(macros),
        "greedy": lambda i: ma.GreedyPlayer(),
        "greedy+macros": lambda i: ma.GreedyPlayer(legal=macro_legal),
        "trained-linear": lambda i: trained.frozen(seed + 40_000 + i),
        "trained+macros": lambda i: trained_m.frozen(seed + 50_000 + i),
    }
    table = mp.evaluate(arms, ev, budget=budget, max_nodes=max_nodes)
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
            "used_declared": sum(1 for r in rows if r["used_declared"]),
        }
    paired = _per_case_delta(
        table["scripted+macros"], table["scripted"]
    )
    return {
        "means": means,
        "paired_vs_scripted": paired,
        "table": table,
    }


def _macro_table(rows: list[dict], macros: list[Macro]) -> str:
    """Render the minted table — seq, freq, payoff-correlation."""
    head = (
        f"{'rank':<5} {'macro':<7} {'seq':<44} {'freq':>5}"
        f" {'cert%':>6} {'term-rw':>8} {'score':>7}"
    )
    lines = [head, "-" * len(head)]
    for i, m in enumerate(macros):
        lines.append(
            f"{i + 1:<5} {m.name:<7} {m.op_str():<44} {m.count:>5}"
            f" {m.certified:>6.2f} {m.mean_terminal:>8.3f}"
            f" {m.score:>7.3f}"
        )
    return "\n".join(lines)


def _eval_table(means: dict[str, dict]) -> str:
    """Render per-arm means over the held-out rows."""
    head = (
        f"{'arm':<17} {'reward':>8} {'cost':>7} {'steps':>6}"
        f" {'cert':>5} {'decl':>5}"
    )
    lines = [head, "-" * len(head)]
    for name, r in means.items():
        cost = (
            f"{r['cost']:7.2f}"
            if r["cost"] is not None
            else f"{'—':>7}"
        )
        lines.append(
            f"{name:<17} {r['reward']:>8.3f} {cost}"
            f" {r['steps']:>6.2f} {r['cert']:>5} {r['used_declared']:>5}"
        )
    return "\n".join(lines)


def _rediscovery(macros: list[Macro]) -> dict:
    """Report the smoke test — was the known winning line minted."""
    spelled = ("declare", "saturate", "handle", "extract")
    fused = ("claim", "extract")
    hits = {
        "spelled": [m.name for m in macros if m.ops == spelled],
        "fused": [m.name for m in macros if m.ops == fused],
        "has_handle_tail": [
            m.name
            for m in macros
            if m.ops[-2:] == ("handle", "extract")
        ],
        "has_claim_head": [
            m.name for m in macros if m.ops[0] == "claim"
        ],
    }
    hits["rediscovered"] = bool(hits["spelled"] or hits["fused"])
    return hits


def main(argv: list[str] | None = None) -> int:
    """Mine macros on train cases, eval them on the held-out stream."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mine-cases", type=int, default=MINE_CASES)
    ap.add_argument("--mine-budget", type=int, default=MINE_BUDGET)
    ap.add_argument("--eval-cases", type=int, default=EVAL_CASES)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument(
        "--lengths", type=int, nargs="+", default=list(LENGTHS)
    )
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    ap.add_argument("--top", type=int, default=TOP_MACROS)
    ap.add_argument("--offer-cap", type=int, default=TOP_MACROS)
    ap.add_argument("--declare-cap", type=int, default=DECLARE_CAP)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    train = mp.gen_cases(args.seed, args.mine_cases)
    trajs = collect_trajectories(
        train,
        seed=args.seed,
        budget=args.mine_budget,
        declare_cap=args.declare_cap,
    )
    print(  # stdout-compat
        f"== macro mining — {len(trajs)} trajectories over "
        f"{len(train)} train cases =="
    )
    macros, rows = mine_macros(
        trajs,
        lengths=tuple(args.lengths),
        min_count=args.min_count,
        top=args.top,
    )
    print(_macro_table(rows, macros))  # stdout-compat
    hits = _rediscovery(macros)
    print(  # stdout-compat
        f"rediscovery: spelled={hits['spelled']} fused={hits['fused']} "
        f"→ {'YES' if hits['rediscovered'] else 'NO'}"
    )

    res = run_eval(
        macros,
        seed=args.seed,
        cases=args.eval_cases,
        budget=args.budget,
        cap=args.offer_cap,
    )
    print(  # stdout-compat
        f"\n== held-out eval — gen_cases({args.seed}+{HELD_OUT_OFFSET},"
        f" {args.eval_cases}), budget {args.budget} =="
    )
    print(_eval_table(res["means"]))  # stdout-compat
    p = res["paired_vs_scripted"]
    print(  # stdout-compat
        f"scripted+macros vs scripted: wins={p['wins']} "
        f"losses={p['losses']} ties={p['ties']} "
        f"cost-wins={p['cost_wins']}"
    )
    for d in p["rows"]:
        if d["macro_reward"] != d["base_reward"] or d["used_declared"]:
            print(  # stdout-compat
                f"  {d['case']:<18} {d['base_reward']:>7.3f} → "
                f"{d['macro_reward']:>7.3f} "
                f"cost {d['base_cost']} → {d['macro_cost']} "
                f"declared={d['used_declared']}"
            )
    if args.json:
        payload = {
            "seed": args.seed,
            "protocol": {
                "mine_cases": args.mine_cases,
                "mine_budget": args.mine_budget,
                "lengths": args.lengths,
                "eval_cases": args.eval_cases,
                "budget": args.budget,
                "held_out": f"gen_cases(seed+{HELD_OUT_OFFSET})",
            },
            "minted": [
                {
                    "name": m.name,
                    "ops": m.ops,
                    "seq": m.seq,
                    "count": m.count,
                    "certified": m.certified,
                    "mean_terminal": m.mean_terminal,
                    "score": m.score,
                }
                for m in macros
            ],
            "rediscovery": hits,
            "eval_means": res["means"],
            "paired_vs_scripted": {
                k: v for k, v in p.items() if k != "rows"
            },
            "paired_rows": p["rows"],
            "ranked_rows": [
                {k: v for k, v in r.items() if k != "rep"}
                for r in rows[: 3 * args.top]
            ],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1, default=repr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
