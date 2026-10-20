"""Controlled learned-policy baseline — identical held-out budgets.

The claim under measurement (``docs/results.md``'s honest note):
"learned policies beat random but lose to the cost-delta heuristic".
This script re-measures it under one protocol: random, a
cost-delta-heuristic arm, a cold linear policy, the shipped trained
linear priors, and the neural ``RuleValueNet`` — all on the *same*
held-out cases and budgets.

Domains and splits (the ones the shipped priors were baked under):

* **meta** — ``meta_player.gen_cases(seed + 7919, n)``, budget 12.
  Arms: ``scripted``, ``greedy``, ``random``, ``cold-linear``
  (``weights={}``), ``trained-linear`` (the baked
  ``META_ARENA_PLAYER_WEIGHTS``) and ``trained-greedy``.
* **open** — odd pages ``open_game.gen_cases(seed, n, page=16,
  start=1, step=2)``, budget 17; the even pages are the train half
  (never evaluated here).  Arms: ``random``, ``heuristic``
  (rank ``state.pending`` by ``dag_cost(lhs) - dag_cost(rhs)``
  descending, drain the page, then ``stop`` — built here, the board
  ships none), ``cold-linear``, ``trained-linear`` (the baked
  ``OPEN_GAME_PLAYER_WEIGHTS``) and ``trained-greedy``.
* **neural** — ``RuleValueNet`` (MLP 32) trained by
  ``train_rule_value`` (BCE, 300 epochs, full-batch, CPU) on
  ``RuleSample`` rows built from the *even*-page candidates'
  measured outcomes (``features = compute_features(lhs)``,
  ``rule_vector(as_rule())``, ``delta = base_cost - cand_cost``),
  then scored over the held-out pool exactly as
  ``proposal._learned_scores`` does.

Two controls the naive comparison misses, both kept visible:

* **Oracle-draw pairing.**  ``proposal.evaluate`` draws its numeric
  tensors from torch's global RNG, so a candidate's verdict is a
  *draw*, not a constant — two arms evaluating the same candidate
  see different draws unless the RNG is aligned, and per-episode
  reseeding only aligns draws by *step position*, not by candidate.
  This script measures every held-out pool member once under a
  per-candidate seed, memoizes the verdict, and runs every arm's
  episodes under a patched ``lp.evaluate`` returning the memoized
  outcome — literally identical verdicts for every arm, and
  verdicts consistent with the ranking pass.  (The pinned tests do
  not do this; their numbers carry the draw noise.)
* **Stopping is a confound.**  The reward pays a per-find ordering
  credit and a cheap ``-0.002`` step cost, so an arm that stops
  early and one that drains the page are not directly comparable:
  ``evals_per_ep`` is reported alongside reward and yield.

Measured pins this run is compared against (``docs/results.md``):
open AP heuristic 0.045 / trained 0.133 / neural 0.023, open reward
random 0.134 / cold 0.143 / trained 0.304, meta reward random -0.57 /
cold -0.49 / trained -0.06 / scripted -0.10.

Usage::

    .venv/bin/python bench/policy/policy_baseline.py
    .venv/bin/python bench/policy/policy_baseline.py --open-pages 18 --json /tmp/b.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any

import torch
from mlopt_core.features import compute_features
from mlopt_core.trajectories import RuleSample, rule_vector
from mlopt_discovery import engine, play
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import meta_player as mp
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from mlopt_torch.learned_policy import input_vector, train_rule_value

#: Defaults — the shipped priors' measurement protocol.
SEED = 0
HELD_OUT_OFFSET = 7919  # ``play``'s held-out seed offset (meta only)
META_EVAL_CASES = 16
META_BUDGET = 12
PAGE = 16
OPEN_PAGES = 36  # the odd half; the even half trains
OPEN_BUDGET = PAGE + 1
NEURAL_EPOCHS = 300  # ``proposal._train_scorer``'s CPU default
NEURAL_HIDDEN = 32


# ---------------------------------------------------------------------------
#  Meta arena — the 6-arm paired eval
# ---------------------------------------------------------------------------


def _meta_random_arm(seed: int) -> Any:
    """Return a uniform player over the meta board's legal set."""

    def make(i: int) -> Any:
        r = random.Random(seed + 10_000 + i)

        def player(state: Any) -> Any | None:
            acts = ma.legal_actions(state)
            return acts[r.randrange(len(acts))] if acts else None

        return player

    return make


def run_meta(seed: int, n: int, budget: int) -> dict[str, dict]:
    """Play the six arms on the held-out meta stream; mean rows."""
    torch.manual_seed(seed)
    ev = mp.gen_cases(seed + HELD_OUT_OFFSET, n)
    trained = mp.MetaLearnedPlayer(seed)  # default = shipped preset
    cold = mp.MetaLearnedPlayer(seed, weights={})
    arms = {
        "scripted": lambda i: ma.ScriptedPlayer(),
        "greedy": lambda i: ma.GreedyPlayer(),
        "random": _meta_random_arm(seed),
        "cold-linear": lambda i: cold.frozen(seed + 20_000 + i),
        "trained-linear": lambda i: trained.frozen(seed + 40_000 + i),
        "trained-greedy": lambda i: trained.frozen(
            seed + 50_000 + i, greedy=True
        ),
    }
    table = mp.evaluate(arms, ev, budget=budget)
    out: dict[str, dict] = {}
    for name, rows in table.items():
        k = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        out[name] = {
            "reward": round(sum(r["reward"] for r in rows) / k, 4),
            "cost": round(sum(costs) / len(costs), 2)
            if costs
            else None,
            "steps": round(sum(r["steps"] for r in rows) / k, 2),
            "cert": sum(1 for r in rows if r["cert"]),
            "used_declared": sum(1 for r in rows if r["used_declared"]),
        }
    return out


# ---------------------------------------------------------------------------
#  Open game — the arms (incl. the heuristic ranker the board lacks)
# ---------------------------------------------------------------------------


def _heuristic_arm() -> Any:
    """Rank pending by the cost-delta heuristic; drain, then stop.

    The episode-level version of ``rank_pool``'s heuristic scorer:
    evaluate ``state.pending`` in descending ``dag_cost(lhs) -
    dag_cost(rhs)`` order, then ``stop``.  Mirrors the ``_scripted``
    pattern of ``tests/test_discovery_open_learned.py``.
    """

    def make(i: int) -> Any:
        del i

        def player(state: Any) -> Any:
            best_j = None
            best_s = None
            for j in state.pending:
                s = lp._heuristic_score(state.pool[j])
                if best_s is None or s > best_s:
                    best_s, best_j = s, j
            if best_j is None:
                return Action("stop", {})
            return Action(
                "evaluate", {"i": best_j, "cand": state.pool[best_j]}
            )

        return player

    return make


def _cand_key(cand: Any) -> tuple:
    """Stable identity of a candidate — same proposal, same verdict."""
    return (cand.label, repr(cand.lhs), repr(cand.rhs))


def _measure_pools(
    pools: list[list], seed: int
) -> tuple[list[list], dict]:
    """Evaluate every pool member once under a per-candidate seed.

    Returns ``(per_pool_outcomes, by_key)``.  Each candidate gets
    ``torch.manual_seed(seed + index)`` before its own
    ``lp.evaluate`` call, so the verdicts are deterministic, and
    ``by_key`` lets a patched ``lp.evaluate`` replay them —
    literally identical oracle draws for every arm that later
    evaluates the same candidate, whatever order it asks in.
    """
    by_key: dict = {}
    per_pool: list[list] = []
    idx = 0
    for pool in pools:
        outs: list = []
        for cand in pool:
            torch.manual_seed(seed + idx)
            idx += 1
            outcome = lp.evaluate([cand])[0]
            by_key.setdefault(_cand_key(cand), outcome)
            outs.append(outcome)
        per_pool.append(outs)
    return per_pool, by_key


def _patched_evaluate(by_key: dict) -> Any:
    """Return an ``lp.evaluate`` replaying the memoized verdicts."""

    def evaluate(cands: Any, **kw: Any) -> list:
        del kw
        return [by_key[_cand_key(c)] for c in cands]

    return evaluate


def _play_open(mk: Any, ev: list[dict], budget: int) -> dict:
    """Play one arm over the held-out pages; tally yield and reward.

    Episodes run under the caller's patched ``lp.evaluate``, so the
    verdicts are the memoized ones — identical across arms.  Each
    arm gets *fresh* arenas: a board drains, and a shared arena
    would start the next arm already ``done``.
    """
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
    return {
        "reward": round(reward / n, 4),
        "useful_per_ep": round(found / n, 4),
        "useful_per_eval": round(found / max(evals, 1), 4),
        "evals_per_ep": round(evals / n, 3),
        "found": found,
    }


def _ap(scores: list[float], outcomes: dict) -> float:
    """Average precision of a best-first ordering under *scores*."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return lp._average_precision(order, outcomes)


def _neural_scores(
    pool: list,
    train_pool: list,
    train_outcomes: list,
    *,
    epochs: int,
    hidden: int,
    seed: int,
) -> list[float]:
    """Train ``RuleValueNet`` on even-page outcomes; score the pool.

    The samples are the *measured* candidate rows — the on-distribution
    version of ``proposal._train_scorer``'s ``rule_samples`` corpus:
    ``(features(lhs), rule_vector(as_rule), base_cost - cand_cost)``,
    BCE target ``delta > 0`` (the shipped default, ``binary=True``).
    """
    samples = [
        RuleSample(
            compute_features(c.lhs),
            rule_vector(c.as_rule()),
            c.as_rule().name,
            o.base_cost - o.cand_cost,
            c.label,
        )
        for c, o in zip(train_pool, train_outcomes, strict=True)
    ]
    model = train_rule_value(
        samples,
        epochs=epochs,
        hidden=hidden,
        device="cpu",
        seed=seed,
    )
    x = torch.tensor(
        [
            input_vector(
                compute_features(c.lhs), rule_vector(c.as_rule())
            )
            for c in pool
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return [float(v) for v in model(x)]


def run_open(
    seed: int,
    half: int,
    page: int,
    budget: int,
    *,
    neural_epochs: int,
    neural_hidden: int,
    skip_neural: bool,
) -> dict:
    """Play the arms on the odd pages; rank the held-out pool."""
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    eval_outcomes, by_key = _measure_pools(
        ev_pools, seed + HELD_OUT_OFFSET
    )

    train_pools: list[list] = []
    train_outcomes: list[list] = []
    if not skip_neural:
        train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
        train_outcomes, train_by_key = _measure_pools(
            train_pools, seed + 1_000_003
        )
        by_key.update(train_by_key)

    flat_pool = [c for pool in ev_pools for c in pool]
    # Index the held-out outcomes by flat pool position.
    flat_outcomes = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1

    orig_evaluate = lp.evaluate
    lp.evaluate = _patched_evaluate(by_key)
    try:
        baked = og.OpenLearnedPlayer(seed)  # default = shipped preset
        cold = og.OpenLearnedPlayer(seed, weights={})
        rnd = play._random_policy(og.legal_actions)
        arms = {
            "random": lambda i: rnd(seed + 10_000 + i),
            "heuristic": _heuristic_arm(),
            "cold-linear": lambda i: cold.frozen(seed + 20_000 + i),
            "trained-linear": lambda i: baked.frozen(seed + 30_000 + i),
            "trained-greedy": lambda i: baked.frozen(
                seed + 40_000 + i, greedy=True
            ),
        }
        table = {
            name: _play_open(mk, ev, budget)
            for name, mk in arms.items()
        }

        scorers: dict[str, list[float]] = {
            "heuristic": [lp._heuristic_score(c) for c in flat_pool],
            "cold-linear": [cold.rank_score(c) for c in flat_pool],
            "trained-linear": [baked.rank_score(c) for c in flat_pool],
        }
        neural: dict[str, Any] = {"skipped": skip_neural}
        if not skip_neural:
            flat_train = [c for pool in train_pools for c in pool]
            flat_train_outcomes = [
                o for outs in train_outcomes for o in outs
            ]
            scorers["neural"] = _neural_scores(
                flat_pool,
                flat_train,
                flat_train_outcomes,
                epochs=neural_epochs,
                hidden=neural_hidden,
                seed=seed,
            )
            neural.update(
                {
                    "train_samples": len(flat_train),
                    "train_positives": sum(
                        1
                        for o in flat_train_outcomes
                        if o.base_cost > o.cand_cost
                    ),
                    "epochs": neural_epochs,
                    "hidden": neural_hidden,
                }
            )
    finally:
        lp.evaluate = orig_evaluate

    n_pos = sum(1 for o in flat_outcomes.values() if o.useful)
    ranking: dict[str, Any] = {
        "pool": len(flat_pool),
        "useful": n_pos,
        "random_ap": round(n_pos / max(len(flat_pool), 1), 4),
    }
    for name, scores in scorers.items():
        ranking[f"{name}_ap"] = round(_ap(scores, flat_outcomes), 4)
    return {"episodes": table, "ranking": ranking, "neural": neural}


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _print_meta(table: dict[str, dict]) -> None:
    """Render the meta-arm means."""
    head = (
        f"{'arm':<15} {'reward':>9} {'cost':>8} {'steps':>6}"
        f" {'cert':>5} {'decl':>5}"
    )
    print(
        "== meta arena — held-out reward/cost/steps =="
    )  # stdout-compat
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for name, r in table.items():
        cost = (
            f"{r['cost']:8.2f}"
            if r["cost"] is not None
            else f"{'—':>8}"
        )
        print(  # stdout-compat
            f"{name:<15} {r['reward']:>9.4f} {cost} "
            f"{r['steps']:>6.2f} {r['cert']:>5} {r['used_declared']:>5}"
        )


def _print_open(res: dict) -> None:
    """Render the open episode table and the ranking APs."""
    t = res["episodes"]
    head = (
        f"{'arm':<15} {'reward':>9} {'use/ep':>7} {'evals/ep':>9}"
        f" {'use/eval':>9} {'found':>6}"
    )
    print("== open game — held-out odd pages ==")  # stdout-compat
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for name, r in t.items():
        print(  # stdout-compat
            f"{name:<15} {r['reward']:>9.4f} {r['useful_per_ep']:>7.4f} "
            f"{r['evals_per_ep']:>9.3f} {r['useful_per_eval']:>9.4f} "
            f"{r['found']:>6}"
        )
    r = res["ranking"]
    print(  # stdout-compat
        f"== ranking AP — pool={r['pool']} useful={r['useful']} =="
    )
    for key in sorted(k for k in r if k.endswith("_ap")):
        print(f"  {key:<20} {r[key]:.4f}")  # stdout-compat


def main(argv: list[str] | None = None) -> int:
    """Run both held-out baselines and print the tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--meta-cases", type=int, default=META_EVAL_CASES)
    ap.add_argument("--meta-budget", type=int, default=META_BUDGET)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--open-budget", type=int, default=OPEN_BUDGET)
    ap.add_argument("--neural-epochs", type=int, default=NEURAL_EPOCHS)
    ap.add_argument("--neural-hidden", type=int, default=NEURAL_HIDDEN)
    ap.add_argument("--skip-meta", action="store_true")
    ap.add_argument("--skip-open", action="store_true")
    ap.add_argument("--skip-neural", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    payload: dict[str, Any] = {
        "seed": args.seed,
        "protocol": {
            "meta_cases": args.meta_cases,
            "meta_budget": args.meta_budget,
            "meta_held_out": f"gen_cases(seed+{HELD_OUT_OFFSET})",
            "open_pages": args.open_pages,
            "open_page": args.page,
            "open_budget": args.open_budget,
            "open_held_out": "odd pages (start=1, step=2)",
            "open_train": "even pages (start=0, step=2)",
            "oracle_pairing": "memoized per-candidate-seeded verdicts",
        },
    }
    if not args.skip_meta:
        meta = run_meta(args.seed, args.meta_cases, args.meta_budget)
        payload["meta"] = meta
        _print_meta(meta)
    if not args.skip_open:
        open_res = run_open(
            args.seed,
            args.open_pages,
            args.page,
            args.open_budget,
            neural_epochs=args.neural_epochs,
            neural_hidden=args.neural_hidden,
            skip_neural=args.skip_neural,
        )
        payload["open"] = open_res
        _print_open(open_res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
