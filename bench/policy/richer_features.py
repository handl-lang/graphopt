"""Richer open-domain features — does new vocabulary raise the ceiling.

``docs/results_schema_leaderboard.md`` found the shipped open schema's
load-bearing part is ``action`` (the ``a:*`` block) — but every schema
it enumerated was a subset of the same shipped vocabulary, so the
search could only *drop* information, never add any.  This demo grows
the vocabulary: ``open_game.py`` now registers seven extra parts —
each emitting one feature the shipped schema cannot express:

* ``open:score-rank`` / ``open:score-rel`` — cross-candidate:
  the candidate's heuristic-score rank quantile within the *pending*
  pool, and its score minus the pending mean ("is this the best
  still-pending option right now").  ``RelFeats`` parts — they read
  ``state`` *and* ``action``.
* ``open:score-x-depth`` / ``open:score-x-ops`` — interactions:
  ``a:score x a:depth``, ``a:score x a:ops`` (does the heuristic pay
  more for deep/wide shapes?).
* ``open:pending-top`` / ``open:pending-mean`` — pool-level state:
  the best / mean heuristic score still pending (what the board still
  has to offer — the ``stop``-relevant signal).
* ``open:useful-ct`` — history-conditioned: useful finds banked this
  episode / pool size (``st:useful_rate`` is a *rate*; the count's
  gradient differs).

The measurement is per-feature, both directions: each feature alone
added to the incumbent parts (``inc+<tag>``), and each dropped from
the full set (``all-<tag>`` — leave-one-out).  Training is *cold* at
the shipped bake recipe: 5 epochs x 36 even pages of
``engine.train_policy`` under the baseline's memoized verdicts;
held-out odd pages, budget 17.

AP honesty note: ``rank_score`` (``spec.rank`` = ``_cand_feats``) only
sees the candidate — it cannot emit the pool-relative features.  The
AP reported here is therefore *page-aware*: each held-out pool's
fresh-page ``featurize(state, evaluate(cand))`` row is dotted with the
trained weights, the per-candidate scores pooled across pages.  For
the incumbent schema this is provably the ``rank_score`` ordering
(``st:*``/``h:*``/``op:*``/``bias`` are identical on every fresh page);
for richer schemas it prices the pool-context the new features add.

Anchors: the baked shipped prior (0.3271 reward, 0.1332 AP), the
heuristic drain, random.  A feature set *earns* the bake
(``lawdata.OPEN_GAME_PLAYER_WEIGHTS``) only if its AP beats the baked
prior's by more than ~1 sigma (~0.01).

Runtime: ~145 s one-time pool memoization, ~10-15 s per arm.

Usage::

    timeout 1800 .venv/bin/python bench/policy/richer_features.py \
        --json /tmp/richer.json
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


def _extra_spec(tags: tuple[str, ...]) -> DomainSpec:
    """Incumbent parts + the extra *tags*; schema extended likewise.

    Feature names already in ``OPEN_GAME_FEATURES`` (the extensions
    that earned the bake) are not re-listed — a part can appear twice
    harmlessly (``row.update`` is idempotent) but the schema tuple
    stays a set.
    """
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


def _arm_grid() -> list[tuple[str, tuple[str, ...]]]:
    """``(arm_id, extra_tags)``: incumbent, singles, all, LOO."""
    tags = og.OPEN_EXTRA_PART_TAGS
    arms: list[tuple[str, tuple[str, ...]]] = [("incumbent", ())]
    arms += [(f"inc+{t.split(':', 1)[1]}", (t,)) for t in tags]
    arms.append(("all", tags))
    arms += [
        (f"all-{t.split(':', 1)[1]}", tuple(u for u in tags if u != t))
        for t in tags
    ]
    # Composite subsets — the candidate-level (a:*/x:*) core, and the
    # minimal AP-driving triple if the singles/LOO implicate one.
    arms.append(
        (
            "core4",
            (
                "open:score-rank",
                "open:score-rel",
                "open:score-x-depth",
                "open:score-x-ops",
            ),
        )
    )
    arms.append(
        (
            "core3",
            ("open:score-rank", "open:score-rel", "open:score-x-depth"),
        )
    )
    return arms


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


def _eval_spec(
    player: og.OpenLearnedPlayer,
    spec: DomainSpec,
    ev_pools: list[list],
    flat_outcomes: dict,
    budget: int,
    seed: int,
) -> tuple[dict, dict, float]:
    """Play sampled + greedy frozen arms; return them and the AP."""
    sampled = sl._play_open_pools(
        lambda i: player.frozen(seed + _SAMPLED_SEED + i),
        ev_pools,
        budget,
    )
    greedy = sl._play_open_pools(
        lambda i: player.frozen(seed + _GREEDY_SEED + i, greedy=True),
        ev_pools,
        budget,
    )
    ap = round(
        pb._ap(
            _page_scores(spec, player.weights_dict(), ev_pools),
            flat_outcomes,
        ),
        4,
    )
    return sampled, greedy, ap


def run(
    seed: int, half: int, page: int, budget: int, epochs: int
) -> dict[str, Any]:
    """Measure pools once; train every feature-set arm cold; report."""
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
        # Anchors — the same eval machinery the candidates use.
        baked = og.OpenLearnedPlayer(seed)
        anchors = {
            "baked-prior": {
                "sampled": sl._play_open_pools(
                    lambda i: baked.frozen(seed + 30_000 + i),
                    ev_pools,
                    budget,
                ),
                "greedy": sl._play_open_pools(
                    lambda i: baked.frozen(
                        seed + 35_000 + i, greedy=True
                    ),
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
                        [0.0] * len(flat_pool),
                        flat_outcomes,
                    ),
                    4,
                ),
            },
        }

        for arm_i, (arm_id, tags) in enumerate(_arm_grid()):
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
            sampled, greedy, ap = _eval_spec(
                player, spec, ev_pools, flat_outcomes, budget, seed
            )
            rows.append(
                {
                    "id": arm_id,
                    "extra_parts": list(tags),
                    "n_features": len(spec.features),
                    "reward_ep": sampled["reward"],
                    "reward_greedy_ep": greedy["reward"],
                    "useful_ep": sampled["useful_per_ep"],
                    "evals_ep": sampled["evals_per_ep"],
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
                f"  [{arm_i + 1}/{len(_arm_grid())}] {arm_id:<22} "
                f"reward={sampled['reward']:>8.4f} "
                f"greedy={greedy['reward']:>8.4f} ap={ap:.4f} "
                f"({time.time() - t1:.0f}s)"
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
    """Render the arm table and the per-feature LOO deltas."""
    head = (
        f"{'arm':<22} {'reward':>8} {'greedy':>8} {'use/ep':>7}"
        f" {'evals/ep':>8} {'AP':>7} {'feats':>6}"
    )
    print("== richer features — held-out odd pages ==")  # stdout-compat
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for r in res["rows"]:
        print(  # stdout-compat
            f"{r['id']:<22} {r['reward_ep']:>8.4f} "
            f"{r['reward_greedy_ep']:>8.4f} {r['useful_ep']:>7.4f} "
            f"{r['evals_ep']:>8.3f} {r['ap']:>7.4f} "
            f"{r['n_features']:>6}"
        )
    print("-- anchors --")  # stdout-compat
    for name, a in res["anchors"].items():
        g = a.get("greedy")
        print(  # stdout-compat
            f"{name:<22} {a['sampled']['reward']:>8.4f} "
            f"{g['reward'] if g else '—':>8} "
            f"{a['sampled']['useful_per_ep']:>7.4f} "
            f"{a['sampled']['evals_per_ep']:>8.3f} {a['ap']:>7.4f}"
        )
    by_id = {r["id"]: r for r in res["rows"]}
    inc, full = by_id.get("incumbent"), by_id.get("all")
    if inc and full:
        print("-- per-feature deltas (AP) --")  # stdout-compat
        for t in og.OPEN_EXTRA_PART_TAGS:
            short = t.split(":", 1)[1]
            solo = by_id[f"inc+{short}"]["ap"] - inc["ap"]
            loo = by_id[f"all-{short}"]["ap"] - full["ap"]
            print(  # stdout-compat
                f"  {short:<14} solo dAP={solo:+.4f}  "
                f"LOO dAP={loo:+.4f}"
            )


def main(argv: list[str] | None = None) -> int:
    """Run the richer-features measurement and print the table."""
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
        },
        "results": res,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
