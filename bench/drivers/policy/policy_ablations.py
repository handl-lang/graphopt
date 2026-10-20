"""Learned-policy ablations — same held-out protocol as the baseline.

Extends ``bench/drivers/policy/policy_baseline.py``'s measurement with the four knobs
the learned-policy bet rests on.  Every arm reuses the baseline's
plumbing: held-out **odd pages** ``og.gen_cases(seed, n, page=16,
start=1, step=2)``, even pages train, the same per-candidate-seeded
**verdict memoization** (``_measure_pools`` + a patched
``lp.evaluate``) so oracle draws are literally identical across arms —
*and across the training episodes*, which removes the epoch-level
draw noise the shipped bake carried (a controlled-difference choice,
documented in ``docs/results_learned_ablations.md``).

The ablations:

1.  **Heuristic bootstrap** — ``cold`` vs ``primed`` (behavior-cloned
    toward the heuristic-drain expert via ``engine.imitate_policy``)
    vs ``primed+rl`` (primed, then 5-epoch REINFORCE) vs ``cold+rl``
    (5-epoch REINFORCE from scratch — the shipped-prior recipe
    re-derived under memoized verdicts).  Question: does warm-start
    beat from-scratch on reward/ep, useful/ep and AP?
2.  **Exploration schedule** — from the shipped prior as init:
    ``prior-frozen`` (anchor, no training), ``eps-0.1``, ``eps-0.3``
    (epsilon-greedy sampling during training), ``temp-anneal``
    (softmax temperature 4 → 0.25 across epochs, reset to 1.0 for
    eval) and ``imitate-again`` (a second imitation pass over the
    prior).  Question: does continued exploration find MORE useful
    laws than frozen?
3.  **Reward design** — cold 5-epoch REINFORCE under: the standard
    reward (``cold+rl`` reused), ``no-eff`` (``OPEN_GAME_EFFICIENCY``
    zeroed — no per-find ordering credit) and ``no-useful``
    (``OPEN_GAME_REWARD["useful"]`` zeroed — the tier pays nothing,
    only the ordering credit remains).  All evaluated under the
    *standard* reward.  Question: does the ordering credit cause the
    trained prior's AP win?
4.  **Neural capacity x data** — ``RuleValueNet`` ``hidden {32, 96}``
    x training on {even-page measured outcomes (on-distribution),
    corpus ``rule_samples`` over ``families.mixture_programs`` (the
    ``train_search_policy.py`` path)}; AP on the held-out pool.
    Question: is the 0.023 -> 0.041 gap data or capacity?
5.  **Yield per eval** — ``useful_per_eval`` rides on every episode
    row: "do rewards improve useful-candidate yield per evaluation".

Usage::

    .venv/bin/python bench/drivers/policy/policy_ablations.py
    .venv/bin/python bench/drivers/policy/policy_ablations.py --json /tmp/ab.json
    .venv/bin/python bench/drivers/policy/policy_ablations.py --only bootstrap,reward
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Any

import policy_baseline as pb
import torch
from mlopt_core.features import compute_features
from mlopt_core.laws import all_rules
from mlopt_core.trajectories import (
    RuleSample,
    rule_samples,
    rule_vector,
)
from mlopt_discovery import engine, families, play
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp
from mlopt_torch.learned_policy import input_vector, train_rule_value

#: Defaults — the baseline protocol.
SEED = 0
PAGE = 16
OPEN_PAGES = 36
OPEN_BUDGET = PAGE + 1
EPOCHS = 5
NEURAL_EPOCHS = 300
CORPUS_PER_FAMILY = (
    30  # train_search_policy's default is 60; halved for CPU budget
)
TEMP_SCHEDULE = (4.0, 2.0, 1.0, 0.5, 0.25)
EVAL_SEED_OFFSET = 60_000


# ---------------------------------------------------------------------------
#  Shared helpers — training, eval, scoring
# ---------------------------------------------------------------------------


def _imitate_heuristic(
    player: og.OpenLearnedPlayer, cases: list[dict], budget: int
) -> dict[str, int]:
    """Behavior-clone *player* toward the heuristic-drain expert.

    The expert is the baseline's ``_heuristic_arm`` maker — pick the
    pending candidate with the top cost-delta score, drain, then
    ``stop``.  ``imitate_policy`` replays it on real boards, so the
    demonstrations see true states.
    """
    return engine.imitate_policy(
        player, pb._heuristic_arm(), cases, og.OpenArena, budget=budget
    )


def _train(
    player: og.OpenLearnedPlayer,
    cases: list[dict],
    budget: int,
    epochs: int,
    temps: tuple[float, ...] | None = None,
) -> list[float]:
    """Run *epochs* REINFORCE passes; return the per-epoch mean total.

    *temps* optionally schedules the softmax temperature per epoch —
    ``LinearPolicy._temp`` is the one knob the sampler exposes, so the
    anneal arm sets it directly (documented; there is no public
    setter).
    """
    means: list[float] = []
    for e in range(epochs):
        if temps is not None:
            player._temp = max(temps[e], 1e-6)
        totals = engine.train_policy(
            player, cases, og.OpenArena, budget=budget
        )
        means.append(round(sum(totals) / max(len(totals), 1), 4))
    return means


def _eval_trained(
    player: og.OpenLearnedPlayer,
    flat_pool: list,
    flat_outcomes: dict,
    ev: list[dict],
    budget: int,
    seed: int,
    tag: str,
    out: dict[str, dict],
    aps: dict[str, float],
) -> None:
    """Play *player* frozen (sampled + greedy); record AP and weights."""
    out[tag] = pb._play_open(
        lambda i: player.frozen(seed + EVAL_SEED_OFFSET + i),
        ev,
        budget,
    )
    out[f"{tag}-greedy"] = pb._play_open(
        lambda i: player.frozen(
            seed + EVAL_SEED_OFFSET + 500 + i, greedy=True
        ),
        ev,
        budget,
    )
    aps[tag] = round(
        pb._ap(
            [player.rank_score(c) for c in flat_pool], flat_outcomes
        ),
        4,
    )


def _nonzero_weights(player: og.OpenLearnedPlayer) -> dict[str, float]:
    """Return the player's rounded nonzero weight table."""
    return {
        n: round(w, 4)
        for n, w in player.weights_dict().items()
        if abs(w) > 1e-4
    }


@contextlib.contextmanager
def _reward_patch(
    efficiency: float | None = None, useful: float | None = None
) -> Any:
    """Temporarily retune the open-game reward table for training.

    ``OPEN_GAME_EFFICIENCY`` is a module global ``OpenArena`` reads
    per find and ``OPEN_GAME_REWARD["useful"]`` is a dict entry read
    per step — both restore on exit, so evaluation always runs under
    the standard table.
    """
    old_eff = og.OPEN_GAME_EFFICIENCY
    old_useful = og.OPEN_GAME_REWARD["useful"]
    if efficiency is not None:
        og.OPEN_GAME_EFFICIENCY = efficiency
    if useful is not None:
        og.OPEN_GAME_REWARD["useful"] = useful
    try:
        yield
    finally:
        og.OPEN_GAME_EFFICIENCY = old_eff
        og.OPEN_GAME_REWARD["useful"] = old_useful


def _neural_ap(
    samples: list[RuleSample],
    flat_pool: list,
    flat_outcomes: dict,
    *,
    epochs: int,
    hidden: int,
    seed: int,
) -> float:
    """Train ``RuleValueNet`` on *samples*; AP-score the held-out pool."""
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
            for c in flat_pool
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        scores = [float(v) for v in model(x)]
    return round(pb._ap(scores, flat_outcomes), 4)


def _corpus_samples(per_family: int, seed: int) -> tuple[list, int]:
    """Corpus ``RuleSample``s — the ``train_search_policy.py`` path.

    ``mixture_programs`` over every family x *per_family*, one
    ``rule_samples`` row per (program, shipped rule) — off-distribution
    relative to the candidate stream (corpus programs, not candidate
    LHS terms).
    """
    rules = all_rules()
    progs = families.mixture_programs(
        list(families.FAMILIES), per_family, seed
    )
    samples = [s for p in progs for s in rule_samples(p, rules)]
    return samples, len(progs)


# ---------------------------------------------------------------------------
#  The ablations — one orchestrated run under one patched evaluate
# ---------------------------------------------------------------------------


def run_ablations(
    seed: int,
    half: int,
    page: int,
    budget: int,
    *,
    epochs: int,
    neural_epochs: int,
    corpus_per_family: int,
    only: set[str],
) -> dict[str, Any]:
    """Measure pools once, then run the enabled ablation arms."""
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    eval_outcomes, by_key = pb._measure_pools(
        ev_pools, seed + pb.HELD_OUT_OFFSET
    )
    train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
    train_outcomes, train_by_key = pb._measure_pools(
        train_pools, seed + 1_000_003
    )
    by_key.update(train_by_key)

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1
    flat_train = [c for pool in train_pools for c in pool]
    flat_train_outcomes = [o for outs in train_outcomes for o in outs]

    results: dict[str, Any] = {}
    orig_evaluate = lp.evaluate
    lp.evaluate = pb._patched_evaluate(by_key)
    try:
        if "bootstrap" in only:
            results["bootstrap"] = _ablation_bootstrap(
                seed,
                train_cases,
                ev,
                budget,
                epochs,
                flat_pool,
                flat_outcomes,
            )
        if "explore" in only:
            results["exploration"] = _ablation_explore(
                seed,
                train_cases,
                ev,
                budget,
                epochs,
                flat_pool,
                flat_outcomes,
            )
        if "reward" in only:
            results["reward"] = _ablation_reward(
                seed,
                train_cases,
                ev,
                budget,
                epochs,
                flat_pool,
                flat_outcomes,
            )
        if "capacity" in only:
            results["capacity"] = _ablation_capacity(
                seed,
                flat_pool,
                flat_outcomes,
                flat_train,
                flat_train_outcomes,
                neural_epochs,
                corpus_per_family,
            )
    finally:
        lp.evaluate = orig_evaluate
    results["pool"] = {
        "held_out": len(flat_pool),
        "useful": sum(1 for o in flat_outcomes.values() if o.useful),
        "train": len(flat_train),
        "train_positives": sum(
            1 for o in flat_train_outcomes if o.base_cost > o.cand_cost
        ),
    }
    return results


def _anchors(seed: int, ev: list[dict], budget: int) -> dict[str, dict]:
    """Return the reference arms — random, heuristic, baked prior."""
    baked = og.OpenLearnedPlayer(seed)
    rnd = play._random_policy(og.legal_actions)
    return {
        "random": pb._play_open(
            lambda i: rnd(seed + 10_000 + i), ev, budget
        ),
        "heuristic": pb._play_open(pb._heuristic_arm(), ev, budget),
        "trained-linear(baked)": pb._play_open(
            lambda i: baked.frozen(seed + 30_000 + i), ev, budget
        ),
        "trained-greedy(baked)": pb._play_open(
            lambda i: baked.frozen(seed + 40_000 + i, greedy=True),
            ev,
            budget,
        ),
    }


def _ablation_bootstrap(
    seed: int,
    train_cases: list[dict],
    ev: list[dict],
    budget: int,
    epochs: int,
    flat_pool: list,
    flat_outcomes: dict,
) -> dict[str, Any]:
    """Ablation 1 — heuristic bootstrap vs cold start."""
    eps: dict[str, dict] = _anchors(seed, ev, budget)
    aps: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    meta: dict[str, Any] = {}

    cold = og.OpenLearnedPlayer(seed + 1, weights={})
    eps["cold"] = pb._play_open(
        lambda i: cold.frozen(seed + 20_000 + i), ev, budget
    )
    aps["cold"] = round(
        pb._ap([cold.rank_score(c) for c in flat_pool], flat_outcomes),
        4,
    )

    primed = og.OpenLearnedPlayer(seed + 2, weights={})
    meta["primed_imitation"] = _imitate_heuristic(
        primed, train_cases, budget
    )
    _eval_trained(
        primed,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "primed",
        eps,
        aps,
    )

    primed_rl = og.OpenLearnedPlayer(seed + 3, weights={})
    meta["primed_rl_imitation"] = _imitate_heuristic(
        primed_rl, train_cases, budget
    )
    curves["primed+rl"] = _train(primed_rl, train_cases, budget, epochs)
    _eval_trained(
        primed_rl,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "primed+rl",
        eps,
        aps,
    )

    cold_rl = og.OpenLearnedPlayer(seed + 4, weights={})
    curves["cold+rl"] = _train(cold_rl, train_cases, budget, epochs)
    _eval_trained(
        cold_rl,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "cold+rl",
        eps,
        aps,
    )

    meta["weights"] = {
        "primed": _nonzero_weights(primed),
        "primed+rl": _nonzero_weights(primed_rl),
        "cold+rl": _nonzero_weights(cold_rl),
    }
    return {
        "episodes": eps,
        "ap": aps,
        "train_curve": curves,
        "meta": meta,
    }


def _ablation_explore(
    seed: int,
    train_cases: list[dict],
    ev: list[dict],
    budget: int,
    epochs: int,
    flat_pool: list,
    flat_outcomes: dict,
) -> dict[str, Any]:
    """Ablation 2 — continued exploration from the shipped prior."""
    eps: dict[str, dict] = {}
    aps: dict[str, float] = {}
    curves: dict[str, list[float]] = {}

    baked = og.OpenLearnedPlayer(seed)
    eps["prior-frozen"] = pb._play_open(
        lambda i: baked.frozen(seed + 30_000 + i), ev, budget
    )
    eps["prior-frozen-greedy"] = pb._play_open(
        lambda i: baked.frozen(seed + 40_000 + i, greedy=True),
        ev,
        budget,
    )
    aps["prior-frozen"] = round(
        pb._ap([baked.rank_score(c) for c in flat_pool], flat_outcomes),
        4,
    )

    eps01 = og.OpenLearnedPlayer(seed + 11, epsilon=0.1)
    curves["eps-0.1"] = _train(eps01, train_cases, budget, epochs)
    _eval_trained(
        eps01,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "eps-0.1",
        eps,
        aps,
    )

    eps03 = og.OpenLearnedPlayer(seed + 12, epsilon=0.3)
    curves["eps-0.3"] = _train(eps03, train_cases, budget, epochs)
    _eval_trained(
        eps03,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "eps-0.3",
        eps,
        aps,
    )

    anneal = og.OpenLearnedPlayer(
        seed + 13, temperature=TEMP_SCHEDULE[0]
    )
    curves["temp-anneal"] = _train(
        anneal, train_cases, budget, epochs, temps=TEMP_SCHEDULE
    )
    anneal._temp = 1.0  # eval decodes at the standard temperature
    _eval_trained(
        anneal,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "temp-anneal",
        eps,
        aps,
    )

    imit = og.OpenLearnedPlayer(seed + 14)
    imitation = _imitate_heuristic(imit, train_cases, budget)
    _eval_trained(
        imit,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "imitate-again",
        eps,
        aps,
    )

    return {
        "episodes": eps,
        "ap": aps,
        "train_curve": curves,
        "meta": {
            "init": "shipped OPEN_GAME_PLAYER_WEIGHTS",
            "imitate_again": imitation,
            "temp_schedule": list(TEMP_SCHEDULE),
        },
    }


def _ablation_reward(
    seed: int,
    train_cases: list[dict],
    ev: list[dict],
    budget: int,
    epochs: int,
    flat_pool: list,
    flat_outcomes: dict,
) -> dict[str, Any]:
    """Ablation 3 — which reward term trains the better ranker.

    All three arms train *cold* for *epochs* epochs; only the
    training-time reward table differs.  Evaluation always runs under
    the standard reward — the ablation is about what the signal
    teaches, not what the eval pays.
    """
    eps: dict[str, dict] = {}
    aps: dict[str, float] = {}
    curves: dict[str, list[float]] = {}

    std = og.OpenLearnedPlayer(seed + 21, weights={})
    curves["std"] = _train(std, train_cases, budget, epochs)
    _eval_trained(
        std,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "std",
        eps,
        aps,
    )

    with _reward_patch(efficiency=0.0):
        no_eff = og.OpenLearnedPlayer(seed + 22, weights={})
        curves["no-eff"] = _train(no_eff, train_cases, budget, epochs)
    _eval_trained(
        no_eff,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "no-eff",
        eps,
        aps,
    )

    with _reward_patch(useful=0.0):
        no_useful = og.OpenLearnedPlayer(seed + 23, weights={})
        curves["no-useful"] = _train(
            no_useful, train_cases, budget, epochs
        )
    _eval_trained(
        no_useful,
        flat_pool,
        flat_outcomes,
        ev,
        budget,
        seed,
        "no-useful",
        eps,
        aps,
    )

    return {
        "episodes": eps,
        "ap": aps,
        "train_curve": curves,
        "meta": {
            "note": (
                "no-eff: OPEN_GAME_EFFICIENCY=0 during training "
                "(useful pays only its 1.0 tier); no-useful: "
                "OPEN_GAME_REWARD['useful']=0 during training (a "
                "useful find pays only the ordering credit). Both "
                "evaluated under the standard reward."
            ),
            "weights": {
                "std": _nonzero_weights(std),
                "no-eff": _nonzero_weights(no_eff),
                "no-useful": _nonzero_weights(no_useful),
            },
        },
    }


def _ablation_capacity(
    seed: int,
    flat_pool: list,
    flat_outcomes: dict,
    flat_train: list,
    flat_train_outcomes: list,
    neural_epochs: int,
    corpus_per_family: int,
) -> dict[str, Any]:
    """Ablation 4 — RuleValueNet hidden x data-source AP matrix."""
    on_dist = [
        RuleSample(
            compute_features(c.lhs),
            rule_vector(c.as_rule()),
            c.as_rule().name,
            o.base_cost - o.cand_cost,
            c.label,
        )
        for c, o in zip(flat_train, flat_train_outcomes, strict=True)
    ]
    corpus, n_progs = _corpus_samples(corpus_per_family, seed)
    matrix: dict[str, dict[str, float]] = {}
    for hidden in (32, 96):
        matrix[str(hidden)] = {
            "on-distribution": _neural_ap(
                on_dist,
                flat_pool,
                flat_outcomes,
                epochs=neural_epochs,
                hidden=hidden,
                seed=seed,
            ),
            "corpus": _neural_ap(
                corpus,
                flat_pool,
                flat_outcomes,
                epochs=neural_epochs,
                hidden=hidden,
                seed=seed,
            ),
        }
    return {
        "matrix_ap": matrix,
        "meta": {
            "on_dist_samples": len(on_dist),
            "corpus_samples": len(corpus),
            "corpus_programs": n_progs,
            "corpus_per_family": corpus_per_family,
            "epochs": neural_epochs,
            "reference": {
                "heuristic_ap": round(
                    pb._ap(
                        [lp._heuristic_score(c) for c in flat_pool],
                        flat_outcomes,
                    ),
                    4,
                )
            },
        },
    }


# ---------------------------------------------------------------------------
#  Rendering + driver
# ---------------------------------------------------------------------------


def _print_table(name: str, res: dict[str, Any]) -> None:
    """Render one ablation's episode rows and APs."""
    eps = res.get("episodes", {})
    aps = res.get("ap", {})
    head = (
        f"{'arm':<24} {'reward':>9} {'use/ep':>7} {'evals/ep':>9}"
        f" {'use/eval':>9} {'found':>6} {'AP':>7}"
    )
    print(f"== {name} ==")  # stdout-compat
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for arm, r in eps.items():
        stem = arm[:-7] if arm.endswith("-greedy") else arm
        ap = aps.get(stem)
        ap_s = f"{ap:7.4f}" if ap is not None else f"{'—':>7}"
        print(  # stdout-compat
            f"{arm:<24} {r['reward']:>9.4f} {r['useful_per_ep']:>7.4f} "
            f"{r['evals_per_ep']:>9.3f} {r['useful_per_eval']:>9.4f} "
            f"{r['found']:>6} {ap_s}"
        )
    for arm, curve in res.get("train_curve", {}).items():
        print(f"  train[{arm}]: {curve}")  # stdout-compat
    if "matrix_ap" in res:
        print("  hidden | on-distribution | corpus")  # stdout-compat
        for h, row in res["matrix_ap"].items():
            print(  # stdout-compat
                f"  {h:>6} | {row['on-distribution']:>15.4f} "
                f"| {row['corpus']:>6.4f}"
            )


def main(argv: list[str] | None = None) -> int:
    """Run the enabled ablations and print the tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--open-budget", type=int, default=OPEN_BUDGET)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--neural-epochs", type=int, default=NEURAL_EPOCHS)
    ap.add_argument(
        "--corpus-per-family", type=int, default=CORPUS_PER_FAMILY
    )
    ap.add_argument(
        "--only",
        type=str,
        default="bootstrap,explore,reward,capacity",
        help="comma-separated subset of ablations to run",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    results = run_ablations(
        args.seed,
        args.open_pages,
        args.page,
        args.open_budget,
        epochs=args.epochs,
        neural_epochs=args.neural_epochs,
        corpus_per_family=args.corpus_per_family,
        only=only,
    )
    payload: dict[str, Any] = {
        "seed": args.seed,
        "protocol": {
            "open_pages": args.open_pages,
            "open_page": args.page,
            "open_budget": args.open_budget,
            "open_held_out": "odd pages (start=1, step=2)",
            "open_train": "even pages (start=0, step=2)",
            "epochs": args.epochs,
            "neural_epochs": args.neural_epochs,
            "corpus_per_family": args.corpus_per_family,
            "oracle_pairing": (
                "memoized per-candidate-seeded verdicts — identical "
                "draws for every arm AND every training episode"
            ),
        },
        "results": results,
    }
    for name, res in results.items():
        if isinstance(res, dict) and (
            "episodes" in res or "matrix_ap" in res
        ):
            _print_table(name, res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
