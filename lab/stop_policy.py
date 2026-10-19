"""The stopping policy — can the learned arm learn *when to quit*.

``docs/results_richer_features.md``'s flat axis: reward sits at
~0.27 across every schema arm because the trained prior drains
~15 of 16 evals — ordering sharpens (AP 0.133 -> 0.210) but the
``stop`` timing is untrained.  A pure ``st:*`` broadcast cannot fix
it (it cancels in the softmax — the ``open:pending-*`` parts
measured dead for exactly that reason), so ``open_game.py`` ships
the ``open:stop`` part: action-level columns

* ``x:marginal`` on ``evaluate`` rows — the candidate's pending
  rank quantile x the payoff a useful find collects *this* step
  (tier + ordering credit): decays as the episode ages;
* ``x:stop:marginal`` / ``x:stop:evals_left`` / ``x:stop:finds``
  on the ``stop`` row — the opportunity cost of quitting now, the
  evals left, and the finds banked.

The measurement mirrors ``lab/richer_features.py`` exactly: cold
retrain per arm (5 epochs x 36 even pages of ``engine.train_policy``
under the baseline's memoized verdicts), held-out odd pages, budget
17, page-aware AP (fresh-page ``w.featurize(evaluate)`` rows pooled
across pages — ``rank_score`` cannot see the pool-relative columns).

Bake rule: the extended schema + weights join
``OPEN_GAME_PLAYER_WEIGHTS`` only if the stop arm's held-out
reward/ep AND page-aware AP both clear the incumbent bake
(0.273 / 0.210 — the richer-features `core4`/`all` rows).

Runtime: ~145 s one-time pool memoization, ~10-15 s per arm.

Usage::

    timeout 1800 .venv/bin/python lab/stop_policy.py \
        --json /tmp/stop.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import policy_baseline as pb
import schema_leaderboard as sl
from mlopt_discovery import engine, play
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import PARTS, DomainSpec, _dot

#: Defaults — the shipped bake protocol under memoized verdicts.
SEED = 0
PAGE = 16
OPEN_PAGES = 36  # the even half trains, the odd half evaluates
BUDGET = PAGE + 1
EPOCHS = 5  # the shipped bake used 5
_SAMPLED_SEED = 40_000  # schema_leaderboard's eval-seed offsets
_GREEDY_SEED = 50_000

#: The arms — incumbent schema (retrained cold, the control) and the
#: stop-column extension; a ``x:marginal``-only arm isolates the
#: evaluate-side column from the stop-row block.
ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("incumbent", ()),
    ("inc+stop", ("open:stop",)),
)


def _extra_spec(tags: tuple[str, ...]) -> DomainSpec:
    """Incumbent parts + the extra *tags*; schema extended likewise."""
    parts = og.OPEN_PARTS + tuple(PARTS[t]() for t in tags)
    feats = og.OPEN_GAME_FEATURES + tuple(
        f
        for t in tags
        for f in og.OPEN_EXTRA_FEATURES[t]
        if f not in og.OPEN_GAME_FEATURES
    )
    return DomainSpec.build(
        features=feats,
        legal=og.legal_actions,
        parts=parts,
        weights=None,
        rank=og._cand_feats,
        lr=og.OPEN_SPEC.lr,
    )


def _page_scores(
    spec: DomainSpec, weights: dict[str, float], pools: list[list]
) -> list[float]:
    """Fresh-page ``w . featurize(evaluate(cand))`` per pool member."""
    scores: list[float] = []
    for pool in pools:
        st = sl._FixedArena(pool).observe()
        for i, c in enumerate(pool):
            f = spec.featurize(
                st, Action("evaluate", {"i": i, "cand": c})
            )
            scores.append(_dot(weights, f))
    return scores


def _play_pools(
    mk: Any, pools: list[list], budget: int
) -> dict[str, Any]:
    """``sl._play_open_pools`` plus the ``stop`` pick count."""
    found = evals = stops = 0
    reward = 0.0
    for i, pool in enumerate(pools):
        traj = engine.run_episode(sl._FixedArena(pool), mk(i), budget)
        reward += traj.total
        for r in traj.reports:
            evals += 1
            stops += r.action.op == "stop"
            if r.note and r.note.split(":", 1)[0] == "useful":
                found += 1
    n = len(pools)
    return {
        "reward": round(reward / n, 4),
        "useful_per_ep": round(found / n, 4),
        "useful_per_eval": round(found / max(evals, 1), 4),
        "evals_per_ep": round(evals / n, 3),
        "found": found,
        "stops": stops,
    }


def run(
    seed: int, half: int, page: int, budget: int, epochs: int
) -> dict[str, Any]:
    """Measure pools once; train each arm cold; eval held-out."""
    t0 = time.time()
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    eval_outcomes, by_key = pb._measure_pools(
        ev_pools, seed + pb.HELD_OUT_OFFSET
    )
    _t, train_keys = pb._measure_pools(train_pools, seed + 1_000_003)
    by_key.update(train_keys)
    del _t

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1
    measure_s = time.time() - t0

    rows: list[dict] = []
    curves: dict[str, list[float]] = {}
    weights: dict[str, dict] = {}
    orig_evaluate = lp.evaluate
    lp.evaluate = pb._patched_evaluate(by_key)
    try:
        baked = og.OpenLearnedPlayer(seed)
        anchors = {
            "baked-prior": {
                "sampled": sl._play_open_pools(
                    lambda i: baked.frozen(seed + 30_000 + i),
                    ev_pools,
                    budget,
                ),
                "ap": round(
                    pb._ap(
                        _page_scores(
                            og.OPEN_SPEC,
                            baked.weights_dict(),
                            ev_pools,
                        ),
                        flat_outcomes,
                    ),
                    4,
                ),
            },
            "heuristic": {
                "sampled": sl._play_open_pools(
                    pb._heuristic_arm(), ev_pools, budget
                ),
                "ap": round(
                    pb._ap(
                        [lp._heuristic_score(c) for c in flat_pool],
                        flat_outcomes,
                    ),
                    4,
                ),
            },
            "random": {
                "sampled": sl._play_open_pools(
                    lambda i: play._random_policy(og.legal_actions)(
                        seed + 10_000 + i
                    ),
                    ev_pools,
                    budget,
                ),
                "ap": round(
                    pb._ap(
                        [0.0] * len(flat_pool), flat_outcomes
                    ),
                    4,
                ),
            },
        }

        for arm_i, (arm_id, tags) in enumerate(ARMS):
            t1 = time.time()
            spec = _extra_spec(tags)
            player = og.OpenLearnedPlayer(
                seed=seed, spec=spec, weights={}
            )
            curve: list[float] = []
            for _ in range(epochs):
                totals = engine.train_policy(
                    player, train_pools, sl._FixedArena, budget=budget
                )
                curve.append(
                    round(sum(totals) / max(len(totals), 1), 4)
                )
            sampled = _play_pools(
                lambda i, p=player: p.frozen(
                    seed + _SAMPLED_SEED + i
                ),
                ev_pools,
                budget,
            )
            greedy = _play_pools(
                lambda i, p=player: p.frozen(
                    seed + _GREEDY_SEED + i, greedy=True
                ),
                ev_pools,
                budget,
            )
            ap = round(
                pb._ap(
                    _page_scores(
                        spec, player.weights_dict(), ev_pools
                    ),
                    flat_outcomes,
                ),
                4,
            )
            rows.append(
                {
                    "id": arm_id,
                    "extra_parts": list(tags),
                    "n_features": len(spec.features),
                    "reward_ep": sampled["reward"],
                    "reward_greedy_ep": greedy["reward"],
                    "useful_ep": sampled["useful_per_ep"],
                    "useful_eval": sampled["useful_per_eval"],
                    "evals_ep": sampled["evals_per_ep"],
                    "stops": sampled["stops"],
                    "stops_greedy": greedy["stops"],
                    "ap": ap,
                }
            )
            curves[arm_id] = curve
            weights[arm_id] = {
                n: round(w, 4)
                for n, w in player.weights_dict().items()
                if abs(w) > 1e-4
            }
            print(  # stdout-compat
                f"  [{arm_i + 1}/{len(ARMS)}] {arm_id:<16} "
                f"reward={sampled['reward']:>8.4f} "
                f"greedy={greedy['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"ap={ap:.4f} ({time.time() - t1:.0f}s)"
            )
    finally:
        lp.evaluate = orig_evaluate

    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": rows,
        "anchors": anchors,
        "train_curve": curves,
        "weights": weights,
        "pool": {"held_out": len(flat_pool), "useful": n_useful},
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "epochs": epochs,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the arm table."""
    head = (
        f"{'arm':<16} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
    )
    print("== stopping policy — held-out odd pages ==")  # stdout-compat
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for r in res["rows"]:
        print(  # stdout-compat
            f"{r['id']:<16} {r['reward_ep']:>8.4f} "
            f"{r['reward_greedy_ep']:>8.4f} {r['useful_eval']:>9.4f} "
            f"{r['evals_ep']:>8.3f} {r['stops']:>6} {r['ap']:>7.4f}"
        )
    print("-- anchors --")  # stdout-compat
    for name, a in res["anchors"].items():
        s = a["sampled"]
        print(  # stdout-compat
            f"{name:<16} {s['reward']:>8.4f} {'—':>8} "
            f"{s['useful_per_eval']:>9.4f} "
            f"{s['evals_per_ep']:>8.3f} {'—':>6} {a['ap']:>7.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the stopping-policy measurement and print the table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    res = run(
        args.seed,
        args.open_pages,
        args.page,
        args.page + 1,
        args.epochs,
    )
    _print(res)
    payload = {
        "seed": args.seed,
        "protocol": {
            "train": "even pages (start=0, step=2)",
            "held_out": "odd pages (start=1, step=2)",
            "page": args.page,
            "pages": args.open_pages,
            "budget": args.page + 1,
            "epochs": args.epochs,
            "oracle_pairing": "memoized per-candidate-seeded verdicts",
            "ap": "page-aware w.featurize(evaluate) on fresh pools",
            "arms": [a for a, _ in ARMS],
        },
        "results": res,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
