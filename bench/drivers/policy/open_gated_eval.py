"""The ``open-gated`` seat's measured record — the round-10 replay.

``gamekit@fd9137f`` ships the named seat ``open-gated``
(``gamekit.players.gated`` ``variant={"baked": True}`` →
:data:`gamekit.gamedata.OPEN_GATED_PAYLOAD`, the ``vhead:baked`` bake
candidate — the shipped open prior ranking, the ridge stop head at
κ=0, both over incumbent spec records).  Its ``PlayerSpec.record``
quotes the round-10 numbers (``docs/results_stop_gate.md``: held-out
**0.2893 reward/ep, 0.2101 AP, 26 stops, 14.56 evals/ep**), but the
seat itself was never played in-tree — this driver plays it through
the identical memoized-verdict protocol and records what it earns.

Protocol (``stop_gate.py`` round 10, unchanged): the held-out half is
the odd pages — ``gen_cases(seed, 36, page=16, start=1, step=2)`` → 36
episodes, ``budget = page + 1 = 17`` — under the oracle table patched
to per-candidate-seeded verdicts (``seed + HELD_OUT_OFFSET``, exactly
``pb._measure_pools``).  With ``--cache`` the driver reuses the
recorded verdict pickle ``(eval_outcomes, train_outcomes, by_key)``
so the seat faces literally the same oracle draws the arm table did;
without it the pools are re-measured by ``stop_gate``'s own
``_measure_pools_par`` (per-candidate seeds preserved, ``--workers``
and the ``--cand-timeout`` censorship cap available — the round-10
run used 8 workers / 1 200 s, ~40 min wall).

Three rows are priced, all through the shipped code paths:

* ``open-gated`` — ``registry.make("open-gated", "open", ctx={})``
  played by ``sp._play_pools`` (the seat's inner seeds
  ``seed + 40_000 + i`` are the sampled arm's, by construction);
* ``payload:decode-check`` — the same ``OPEN_GATED_PAYLOAD`` pushed
  through ``stop_gate``'s ``_play_payload`` (the round-10 decode
  path).  Must land *identical* to the seat row — that equality is
  the proof the shipped seat is the measured artifact;
* ``anchor:baked-prior`` — the ungated shipped-prior anchor replayed
  (recorded 0.2758 / 13.056 evals / 5 useful) — the oracle-table
  identity check: if the cache's verdicts drifted, this row moves.

AP is the round-10 page-aware number (``sp._page_scores`` of the
baked rank table on fresh pools vs the memoized outcomes); the
seat's own ``ctx["score_fn"]`` (the candidate-only ``rank_score``
seam the registry deposits) is priced alongside for completeness.

Usage::

    .venv/bin/python bench/drivers/policy/open_gated_eval.py \
        --cache /tmp/gate_cache.pkl.censored-bak \
        --json /tmp/open_gated_seat.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import policy_baseline as pb
import schema_leaderboard as sl
import stop_gate as sg
import stop_policy as sp
from gamekit.gamedata import OPEN_GATED_PAYLOAD
from gamekit.players import registry
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp

#: The round-10 constants — identical protocol to ``stop_gate.run``.
SEED = sg.SEED
PAGE = sg.PAGE
OPEN_PAGES = sg.OPEN_PAGES
BUDGET = PAGE + 1

#: The recorded round-10 rows this replay is checked against.
RECORDED = {
    "vhead:baked": {
        "reward": 0.2893,
        "reward_greedy": 0.3524,
        "useful_per_ep": 0.1667,
        "useful_per_eval": 0.0115,
        "evals_per_ep": 14.556,
        "stops": 26,
        "stops_greedy": 27,
        "ap": 0.2101,
    },
    "anchor:baked-prior": {
        "reward": 0.2758,
        "useful_per_ep": 0.1389,
        "useful_per_eval": 0.0106,
        "evals_per_ep": 13.056,
        "found": 5,
        "ap": 0.2101,
    },
}


def run(
    seed: int,
    half: int,
    page: int,
    budget: int,
    cache: str | None = None,
    workers: int = 1,
    cand_timeout: int = 1200,
) -> dict[str, Any]:
    """Load or measure the odd-page verdicts; play the shipped seat.

    Only the *held-out* half is needed — the seat's head is baked, no
    train-drain fit runs here.  The cache is ``stop_gate``'s
    ``(eval_outcomes, train_outcomes, by_key)`` pickle; when it is
    absent or does not cover the eval pools the missing verdicts are
    measured with the identical per-candidate seeding.
    """
    t0 = time.time()
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    loaded = None
    if cache and Path(cache).exists():
        import pickle

        with open(cache, "rb") as f:
            loaded = pickle.load(f)
    if loaded is not None:
        eval_outcomes, _train_outcomes, by_key = loaded
    else:
        eval_outcomes, by_key = sg._measure_pools_par(
            ev_pools,
            seed + pb.HELD_OUT_OFFSET,
            workers,
            None if cache is None else cache + ".ev",
            cand_timeout,
        )
        if cache:
            import pickle

            with open(cache, "wb") as f:
                pickle.dump((eval_outcomes, None, by_key), f)
    measure_s = time.time() - t0

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    n_censored = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            n_censored += o.relation == "censored"
            pos += 1

    orig_evaluate = lp.evaluate
    lp.evaluate = pb._patched_evaluate(by_key)
    try:
        # 1. The shipped seat — the named registry player, ctx empty
        #    (the baked payload binds inside ``gated.make``).
        ctx: dict[str, Any] = {}
        mk = registry.make(
            "open-gated", "open", seed=seed, ctx=ctx
        )
        seat = sp._play_pools(mk, ev_pools, budget)

        # 2. The decode check — the same shipped payload through the
        #    round-10 ``_play_payload`` path (sampled + greedy twins).
        pay_sampled, pay_greedy = sg._play_payload(
            OPEN_GATED_PAYLOAD, ev_pools, budget, seed
        )

        # 3. The oracle-table identity anchor — the ungated shipped
        #    prior, the round-10 ``baked-prior`` anchor row.
        baked = og.OpenLearnedPlayer(seed)
        w_baked = baked.weights_dict()
        anchor = sl._play_open_pools(
            lambda i: baked.frozen(seed + 30_000 + i),
            ev_pools,
            budget,
        )

        # AP — the round-10 page-aware ranker pricing, plus the
        # seat's deposited candidate-only scorer for comparison.
        ap = round(
            pb._ap(
                sp._page_scores(og.OPEN_SPEC, w_baked, ev_pools),
                flat_outcomes,
            ),
            4,
        )
        ap_rank_score = round(
            pb._ap(
                [ctx["score_fn"](c) for c in flat_pool],
                flat_outcomes,
            ),
            4,
        )
    finally:
        lp.evaluate = orig_evaluate

    return {
        "seat": seat,
        "payload_check": {"sampled": pay_sampled, "greedy": pay_greedy},
        "anchor": anchor,
        "ap": ap,
        "ap_rank_score": ap_rank_score,
        "recorded": RECORDED,
        "pool": {
            "held_out": len(flat_pool),
            "useful": sum(
                1 for o in flat_outcomes.values() if o.useful
            ),
            "censored": n_censored,
        },
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "cache_hit": loaded is not None,
            "cand_timeout_s": cand_timeout,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the seat-vs-record comparison."""
    s, a = res["seat"], res["anchor"]
    p = res["payload_check"]
    rec, rec_a = (
        res["recorded"]["vhead:baked"],
        res["recorded"]["anchor:baked-prior"],
    )
    print("== open-gated seat — held-out odd pages (round-10 replay) ==")
    head = (
        f"{'row':<28} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
    )
    print(head)
    print("-" * len(head))
    print(
        f"{'open-gated (seat)':<28} {s['reward']:>8.4f} {'—':>8} "
        f"{s['useful_per_eval']:>9.4f} {s['evals_per_ep']:>8.3f} "
        f"{s['stops']:>6} {res['ap']:>7.4f}"
    )
    print(
        f"{'payload decode-check':<28} {p['sampled']['reward']:>8.4f} "
        f"{p['greedy']['reward']:>8.4f} "
        f"{p['sampled']['useful_per_eval']:>9.4f} "
        f"{p['sampled']['evals_per_ep']:>8.3f} "
        f"{p['sampled']['stops']:>6} {res['ap']:>7.4f}"
    )
    print(
        f"{'anchor:baked-prior':<28} {a['reward']:>8.4f} {'—':>8} "
        f"{a['useful_per_eval']:>9.4f} {a['evals_per_ep']:>8.3f} "
        f"{'—':>6} {res['ap']:>7.4f}"
    )
    print("-- recorded round-10 (results_stop_gate.md) --")
    print(
        f"{'vhead:baked':<28} {rec['reward']:>8.4f} "
        f"{rec['reward_greedy']:>8.4f} {rec['useful_per_eval']:>9.4f} "
        f"{rec['evals_per_ep']:>8.3f} {rec['stops']:>6} "
        f"{rec['ap']:>7.4f}"
    )
    print(
        f"{'baked-prior anchor':<28} {rec_a['reward']:>8.4f} "
        f"{'—':>8} {rec_a['useful_per_eval']:>9.4f} "
        f"{rec_a['evals_per_ep']:>8.3f} {'—':>6} {rec_a['ap']:>7.4f}"
    )
    print(  # stdout-compat
        f"-- pool: {res['pool']['held_out']} held-out candidates, "
        f"{res['pool']['useful']} useful, "
        f"{res['pool']['censored']} censored; "
        f"cache_hit={res['meta']['cache_hit']} "
        f"measure={res['meta']['measure_seconds']}s; "
        f"score_fn AP={res['ap_rank_score']} --"
    )
    ident = all(
        s[k] == p["sampled"][k]
        for k in ("reward", "useful_per_ep", "evals_per_ep", "stops")
    )
    print(  # stdout-compat
        f"-- seat == payload decode: {ident} --"
    )


def main(argv: list[str] | None = None) -> int:
    """Replay the round-10 held-out protocol for the shipped seat."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument(
        "--cache",
        type=str,
        default=None,
        help="verdict pickle (eval_outcomes, train_outcomes, by_key) — "
        "stop_gate's --cache format; reusing the round-10 file replays "
        "the recorded oracle table bit-for-bit",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="process-pool workers for a fresh verdict pass",
    )
    ap.add_argument(
        "--cand-timeout",
        type=int,
        default=1200,
        help="per-candidate eval cap in seconds for a fresh pass "
        "(round 10 used 1200 — 3 candidates censored)",
    )
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    res = run(
        args.seed,
        args.open_pages,
        args.page,
        args.page + 1,
        cache=args.cache,
        workers=args.workers,
        cand_timeout=args.cand_timeout,
    )
    _print(res)
    payload = {
        "seed": args.seed,
        "protocol": {
            "held_out": "odd pages (start=1, step=2)",
            "page": args.page,
            "pages": args.open_pages,
            "budget": args.page + 1,
            "oracle_pairing": "memoized per-candidate-seeded verdicts",
            "seat": "registry.make('open-gated','open',ctx={})",
        },
        "results": res,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
