"""The stopping policy, round 10 — the gate as *data*.

``docs/results_stop_isolated.md`` (round 9) landed the delivery:
``GatedPlayer`` + the ``forced_action`` hook move the quit decision
out of the ranking softmax, and honest-κ arms clear both bake bars —
but their payloads ride a ``score`` *callable* (the injected value
table, an in-process calibrator), so "serializable" was asserted, not
shown.  This round prices exactly that question: **can a stop
decision outside the shared softmax be expressed as pure data?**

The serializable payload the ``gated`` seat already accepts is
``{spec, w_stop, kappa, rank_weights}`` — and every piece of it has a
JSON form: ``DomainSpec.to_record()`` reifies the stop-row spec to
part tags, ``w_stop``/``rank_weights`` are ``{feature: weight}``
dicts, κ is a float, and the ``score`` callable can be *named* — a
``SCORERS``-style registry entry — the same registry idiom the
codebase uses for parts/legals/ranks.  This driver measures the arms
in *exactly* that form: each row's player is built from the
``json.dumps`` round-trip of its payload, so a clearing arm is
serializable by construction, not by argument.

The evidence candidates — priced from **shipped machinery only**,
no ``set_value_preds`` at decode:

* ``open:prior`` (new — :mod:`mlopt_discovery.open_prior`): the stop
  row carries the *shipped prior's own* pending-pool pricing —
  ``x:stop:prior_best``/``_top3``/``_pool``/``_frac`` (the pending
  pool under ``w·_cand_feats``), ``x:stop:prior_margin`` (the prior's
  own softmax margin: best eval logit − stop logit; κ=0 is its
  literal indifference point) and ``x:stop:prior_p1`` (its "keep
  going" conviction);
* ``open:next-value`` — the registered heuristic-priced columns
  (``x:stop:best_score``/``topk_score``), equally seam-free;
* ``self-margin`` — the ranker's own ``w·_cand_feats`` margin as a
  *named scorer* (the payload is the ranker weights + κ);
* the learned stop heads of round 9 in payload form: ``vreg`` (the
  ridge value head, κ=0 — the pure ``w_stop`` linear path) and
  ``head-iso`` (logistic head + isotonic knots + break-even p* — a
  named scorer whose params are data: weight dict + knot list).

κ for the analytic gates is a **named constant fit at bake**: chosen
on the train pages by the analytic-drain shortcut (the greedy drain
visits states independent of κ, so each candidate κ is priced by
prefix-sum — the *fit* is one pass, like the head's own fit), never
at runtime.  Baselines are the non-serializable earners they must
beat honestly: ``ev-margin``/``ev-pool`` on the raw ``delta``
injection (round 7's 0.2821) and ``useful``-isotonic @1.5κ (round
9's 0.2929 — needs the per-candidate table shipped, the seam-fed
serialization round 9 flagged "deployed-else-broken").

Bake rule (unchanged): held-out reward/ep ≥ 0.273 AND page-aware AP
≥ 0.210 — never-stop or AP-collapse is an honest negative.  The
``vhead-fold`` row is the shared-softmax A/B port control.

Runtime: ~150-190 s one-time pool memoization, ~15 s predictors
(``delta`` + ``useful`` only), ~10-20 s drains, ~0.5-1.5 s per arm.

Usage::

    timeout 1800 .venv/bin/python bench/drivers/policy/stop_gate.py \
        --json /tmp/gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import policy_baseline as pb
import schema_leaderboard as sl
import stop_advantage as sa
import stop_calibrated as sc
import stop_isolated as si
import stop_policy as sp
import stop_valuelabel as svl
from gamekit import advantage as adv
from gamekit import engine
from gamekit.calibrate import IsotonicCal
from gamekit.players import (
    GatedPlayer,
    PARTS,
    DomainSpec,
    PlayHist,
    _dot,
    registry,
    spec_from_record,
)
from mlopt_discovery import play
from mlopt_discovery import open_game as og
from mlopt_discovery import open_prior as opr
from mlopt_discovery import open_self as osel
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action

#: Defaults — identical protocol to ``stop_isolated.py``.
SEED = svl.SEED
PAGE = svl.PAGE
OPEN_PAGES = svl.OPEN_PAGES
EPOCHS = svl.EPOCHS
_SAMPLED_SEED = svl._SAMPLED_SEED
_GREEDY_SEED = svl._GREEDY_SEED
_KAPPA_GRID = 25  # the fold/τ convention — bake-time fit only

#: The extra-part vocabulary this driver knows — round 7's plus the
#: shipped ``open:prior`` part.
EXTRA: dict[str, tuple[str, ...]] = {
    **svl.EXTRA,
    "open:prior": opr.PRIOR_FEATURES,
}

#: The serializable gate arms — ``(id, column source, column, ranker,
#: extra part tags)``.  ``ranker`` is ``"inc"`` (the cold-retrained
#: incumbent) or ``"baked"`` (the shipped prior — the deployment
#: pairing).  Single-column gates ride the pure ``w_stop`` linear
#: path: ``w_stop = {column: 1.0}``, κ fit on the train drains.
GATE_ARMS: tuple[tuple[str, str, str, str], ...] = (
    ("heur-best", "x:stop:best_score", "inc", ("open:next-value",)),
    ("heur-top3", "x:stop:topk_score", "inc", ("open:next-value",)),
    ("prior-best", "x:stop:prior_best", "inc", ("open:prior",)),
    ("prior-best", "x:stop:prior_best", "baked", ("open:prior",)),
    ("prior-top3", "x:stop:prior_top3", "inc", ("open:prior",)),
    ("prior-top3", "x:stop:prior_top3", "baked", ("open:prior",)),
    ("prior-pool", "x:stop:prior_pool", "inc", ("open:prior",)),
    ("prior-pool", "x:stop:prior_pool", "baked", ("open:prior",)),
    ("prior-frac", "x:stop:prior_frac", "inc", ("open:prior",)),
    ("prior-frac", "x:stop:prior_frac", "baked", ("open:prior",)),
    ("prior-margin", "x:stop:prior_margin", "inc", ("open:prior",)),
    (
        "prior-margin",
        "x:stop:prior_margin",
        "baked",
        ("open:prior",),
    ),
    ("prior-p1", "x:stop:prior_p1", "inc", ("open:prior",)),
    ("prior-p1", "x:stop:prior_p1", "baked", ("open:prior",)),
)


def _spec_for(tags: tuple[str, ...]) -> DomainSpec:
    """Incumbent parts + the extra *tags*; schema extended likewise."""
    parts = og.OPEN_PARTS + tuple(PARTS[t]() for t in tags)
    feats = og.OPEN_GAME_FEATURES + tuple(
        f
        for t in tags
        for f in EXTRA[t]
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


def _self_v(w: dict) -> Any:
    """``state -> max_pending w·_cand_feats`` — the ranker's own margin."""

    def v(state: Any) -> float:
        vals = [
            _dot(w, og._cand_feats(state.pool[i]))
            for i in state.pending
        ]
        return max(vals) if vals else -math.inf

    return v


def _drain_stats(
    spec: DomainSpec,
    w_rank: dict,
    rank_spec: DomainSpec,
    pools: list[list],
    by_key: dict,
    budget: int,
    fns: dict[str, Any] | None = None,
) -> list[dict]:
    """Greedy drain per pool: stop rows, gate values, true rewards.

    Under the *greedy* ranker the eval order is deterministic and
    independent of any gate threshold, so one drain records every
    decision state the gates can see: the spec's ``stop`` row, the
    per-candidate *extra* value series (``fns`` — scorer-valued
    gates), the oracle continue-label, the pending count and the
    *exact* per-eval reward the arena pays (step + tier + the
    ``1/evals`` ordering credit).  A κ-gate's trajectory is then a
    prefix-sum lookup — the bake-time κ fit costs one pass.
    """
    out: list[dict] = []
    for pool in pools:
        arena = sl._FixedArena(pool)
        hist = PlayHist()
        rec: dict[str, Any] = {
            "rows": [],
            "vals": {n: [] for n in (fns or {})},
            "labels": [],
            "prefix_v": [],
            "rewards": [],
            "pend": [],
        }
        for _ in range(budget):
            state = arena.observe()
            if state.done or not state.pending:
                break
            rec["rows"].append(
                spec.featurize(state, Action("stop", {}), hist)
            )
            for n, fn in (fns or {}).items():
                rec["vals"][n].append(fn(state))
            order = sp._eval_order(rank_spec, w_rank, state)
            pv = sp._prefix_value(state, order, by_key)
            rec["labels"].append(pv > 0.0)
            rec["prefix_v"].append(pv)
            i = order[0]
            verdict = og._verdict(
                by_key[pb._cand_key(state.pool[i])]
            )
            r = og.OPEN_GAME_REWARD["step"] + og.OPEN_GAME_REWARD[
                verdict
            ]
            if verdict == "useful":
                r += og.OPEN_GAME_EFFICIENCY / (state.steps + 1)
            rec["rewards"].append(r)
            rec["pend"].append(len(state.pending))
            act = Action("evaluate", {"i": i, "cand": state.pool[i]})
            arena.step(act)
            hist.record(act.op)
        out.append(rec)
    return out


def _stop_at(series: list[float], kappa: float) -> int:
    """First drain step the gate fires at — ``len`` = full drain."""
    for t, f in enumerate(series):
        if f <= kappa:
            return t
    return len(series)


def _kappa_fit(
    series: list[list[float]], rewards: list[list[float]]
) -> tuple[float, float, list[list[float]]]:
    """Bake-time κ: the train-reward-max grid point, exact by prefix-sum.

    *series*/*rewards* are per-pool along the greedy drain.  For each
    grid κ the gated episode stops at the first state whose gate
    value drops to κ — the reward is the prefix sum at that state —
    so the sweep is arithmetic, not replay.  Returns ``(κ*, train
    reward, [(κ, reward)])``; the grid always includes the analytic
    0 point so a "named constant" κ is priced against it honestly.
    """
    flat = [f for s in series for f in s if math.isfinite(f)]
    if not flat:
        return 0.0, -1e9, []
    lo, hi = min(flat), max(flat)
    grid = sorted(
        {
            lo + (hi - lo) * k / (_KAPPA_GRID - 1)
            for k in range(_KAPPA_GRID)
        }
        | {0.0}
    )
    curve: list[list[float]] = []
    best_k, best_r = grid[0], -1e9
    for k in grid:
        r = sum(
            sum(rw[: _stop_at(s, k)])
            for s, rw in zip(series, rewards, strict=True)
        ) / len(series)
        curve.append([round(k, 6), round(r, 4)])
        if r > best_r:
            best_r, best_k = r, k
    return best_k, round(best_r, 4), curve


# ---------------------------------------------------------------------------
#  The serializable payload — the gated seat's config as pure data
# ---------------------------------------------------------------------------
#
#  ``{rank: {spec: <record>, weights: {...}},
#    stop: {spec: <record>, weights: {...}, kappa: float,
#           scorer: <name>, scorer_params: {...}}}`` — every leaf is a
#  JSON literal.  ``_player_from_payload`` builds the player from the
#  ``json.dumps`` round-trip, so every measured row *is* the
#  serialized artifact playing.


def _payload(
    *,
    rank_weights: dict,
    rank_spec: DomainSpec,
    stop_spec: DomainSpec | None = None,
    w_stop: dict | None = None,
    kappa: float = 0.0,
    scorer: str | None = None,
    scorer_params: dict | None = None,
) -> dict:
    """The gated arm as pure data — ``ctx['gated']``, JSON-shaped."""
    return {
        "rank": {
            "spec": rank_spec.to_record(),
            "weights": rank_weights,
        },
        "stop": {
            "spec": (
                None if stop_spec is None else stop_spec.to_record()
            ),
            "weights": w_stop,
            "kappa": kappa,
            "scorer": scorer,
            "scorer_params": scorer_params,
        },
    }


def _sc_self_margin(
    _spec: Any, _params: dict, rank_w: dict
) -> Any:
    """The ranker's own ``max_pending w·_cand_feats`` — a named scorer."""

    def score(state: Any, _hist: Any) -> float:
        return _self_v(rank_w)(state)

    return score


def _sc_cal_head(spec: DomainSpec, params: dict, _rank_w: dict) -> Any:
    """The calibrated continue-head — a named scorer over data params.

    *params* is ``{"weights": w, "knots": [[x, y], ...]}`` — the
    isotonic map's knot list is the whole calibrator (shipped
    ``gamekit.calibrate.IsotonicCal`` rebuilds from it verbatim).
    """
    cal = IsotonicCal()
    cal.knots_ = [tuple(k) for k in params["knots"]]
    w = params["weights"]

    def score(state: Any, hist: Any) -> float:
        stop = next(
            a for a in og.legal_actions(state) if a.op == "stop"
        )
        return cal.predict(
            [si._sig(_dot(w, spec.featurize(state, stop, hist)))]
        )[0]

    return score


#: The scorer-name registry — the serialization seam for
#: callable-path arms.  Same idiom as ``PARTS``/``LEGALS``/``RANKS``:
#: the payload names the rule, the shipped code binds it.
SCORERS: dict[str, Any] = {
    "self-margin": _sc_self_margin,
    "cal-head": _sc_cal_head,
}


def _player_from_payload(
    payload: dict, seed: int, *, greedy: bool = False
) -> GatedPlayer:
    """Rebuild the gated player from *payload*'s JSON round-trip."""
    rec = json.loads(json.dumps(payload))
    rspec = spec_from_record(rec["rank"]["spec"])
    inner = og.OpenLearnedPlayer(
        seed=seed,
        spec=rspec,
        weights=rec["rank"]["weights"],
        legal=sp._eval_only,
        learn=False,
        greedy=greedy,
    )
    s = rec["stop"]
    if s.get("scorer") is not None:
        sspec = (
            spec_from_record(s["spec"]) if s.get("spec") else None
        )
        score = SCORERS[s["scorer"]](
            sspec, s["scorer_params"] or {}, rec["rank"]["weights"]
        )
        return GatedPlayer(
            inner, legal=og.legal_actions, score=score, kappa=s["kappa"]
        )
    sspec = spec_from_record(s["spec"])
    return GatedPlayer(
        inner,
        legal=og.legal_actions,
        stop_feats=si._stop_feats_of(sspec),
        w_stop=s["weights"],
        kappa=s["kappa"],
    )


def _row(
    arm_id: str,
    kind: str,
    sampled: dict,
    greedy: dict,
    ap: float,
    **kw: Any,
) -> dict:
    """Shape one arm row — the rounds-5/9 table shape."""
    return {
        "id": arm_id,
        "kind": kind,
        "reward_ep": sampled["reward"],
        "reward_greedy_ep": greedy["reward"],
        "useful_ep": sampled["useful_per_ep"],
        "useful_eval": sampled["useful_per_eval"],
        "evals_ep": sampled["evals_per_ep"],
        "stops": sampled["stops"],
        "stops_greedy": greedy["stops"],
        "ap": ap,
        **kw,
    }


def _play_payload(
    payload: dict, ev_pools: list[list], budget: int, seed: int
) -> tuple[dict, dict]:
    """Sampled + greedy play of the *decoded* payload — the artifact."""
    sampled = sp._play_pools(
        lambda i: _player_from_payload(
            payload, seed + _SAMPLED_SEED + i
        ),
        ev_pools,
        budget,
    )
    greedy = sp._play_pools(
        lambda i: _player_from_payload(
            payload, seed + _GREEDY_SEED + i, greedy=True
        ),
        ev_pools,
        budget,
    )
    return sampled, greedy


def _measure_cand_chunk(args: tuple) -> list:
    """Worker for the parallel memoization — a slice of candidates.

    Replicates ``pb._measure_pools`` exactly: each candidate gets
    ``torch.manual_seed(seed + global_idx)`` before its own
    ``lp.evaluate``, so the verdicts are identical to the serial
    pass.  Per-*candidate* tasks (not per-pool) matter here: the
    deep-offset pools' candidates can each saturate the bounded
    egraph for ~30-90 s under the grown law bank, and a per-pool
    shard strands one worker on ~16 of them serially.  Returns
    ``[(global_idx, key, outcome)]``.

    ``cand_timeout`` (0 = off) SIGALRM-boxes each ``lp.evaluate``:
    candidates whose post-merge certificate path runs unbounded come
    back ``relation="censored"`` — verify_law's own refusal class
    (resource exhaustion → derivable=False), counted in the report.
    """
    import signal

    import torch

    torch.set_num_threads(1)
    seed, cands, cand_timeout = args

    def _alrm(_sig, _frame):
        raise TimeoutError

    if cand_timeout:
        signal.signal(signal.SIGALRM, _alrm)
    out = []
    for idx, cand in cands:
        torch.manual_seed(seed + idx)
        try:
            signal.alarm(cand_timeout)
            res = lp.evaluate([cand])[0]
        except TimeoutError:
            res = lp.Outcome(candidate=cand, relation="censored")
        finally:
            signal.alarm(0)
        out.append((idx, pb._cand_key(cand), res))
    return out


def _measure_pools_par(
    pools: list[list],
    seed: int,
    workers: int,
    part_cache: str | None = None,
    cand_timeout: int = 0,
) -> tuple[list[list], dict]:
    """``pb._measure_pools`` sharded over a process pool — same memo.

    The verdicts are deterministic per-candidate seeds
    (``torch.manual_seed(seed + idx)``), so sharding changes nothing
    but wall time.  One candidate per task — the prim-vocabulary
    families (factor-*/pull-*) run ~5-25 s+ each mid-stream, so the
    tail is bounded by the single slowest candidate, never a chunk
    of them.  *part_cache* pickles the finished ``{idx: (key,
    outcome)}`` map incrementally — a killed run resumes where it
    stopped instead of losing the whole pass.
    """
    import pickle
    from concurrent.futures import ProcessPoolExecutor

    jobs = [
        (g, cand)
        for g, cand in enumerate(
            c for pool in pools for c in pool
        )
    ]
    by_idx: dict[int, tuple] = {}
    cpath = Path(part_cache) if part_cache else None
    if cpath and cpath.exists():
        with open(cpath, "rb") as f:
            by_idx.update(pickle.load(f))
    todo = [(g, c) for g, c in jobs if g not in by_idx]
    if todo:
        tasks = [(seed, [j], cand_timeout) for j in todo]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            from concurrent.futures import as_completed

            futs = [ex.submit(_measure_cand_chunk, t) for t in tasks]
            for fut in as_completed(futs):
                for idx, k, o in fut.result():
                    by_idx[idx] = (k, o)
                done += 1
                if cpath and done % 64 == 0:
                    with open(cpath, "wb") as f:
                        pickle.dump(by_idx, f)
        if cpath:
            with open(cpath, "wb") as f:
                pickle.dump(by_idx, f)
    per_pool: list[list] = []
    by_key: dict = {}
    g = 0
    for pool in pools:
        outs: list = []
        for _c in pool:
            k, o = by_idx[g]
            g += 1
            outs.append(o)
            by_key.setdefault(k, o)
        per_pool.append(outs)
    return per_pool, by_key


def run(
    seed: int,
    half: int,
    page: int,
    budget: int,
    epochs: int,
    cache: str | None = None,
    workers: int = 1,
    cand_timeout: int = 0,
) -> dict[str, Any]:
    """Measure pools once; fit κ/heads on train; eval held-out.

    *cache* pickles the memoized verdicts (deterministic under the
    per-candidate seeds) — rerun-safe on this box where the oracle
    pass can run ~10-20 min.
    """
    t0 = time.time()
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    loaded = None
    if cache and Path(cache).exists():
        import pickle

        with open(cache, "rb") as f:
            loaded = pickle.load(f)
    if loaded is not None:
        eval_outcomes, train_outcomes, by_key = loaded
    else:
        measure = (
            _measure_pools_par
            if workers > 1
            else pb._measure_pools
        )
        if workers > 1:
            eval_outcomes, by_key = measure(
                ev_pools,
                seed + pb.HELD_OUT_OFFSET,
                workers,
                None if cache is None else cache + ".ev",
                cand_timeout,
            )
            train_outcomes, train_keys = measure(
                train_pools,
                seed + 1_000_003,
                workers,
                None if cache is None else cache + ".tr",
                cand_timeout,
            )
        else:
            eval_outcomes, by_key = measure(
                ev_pools, seed + pb.HELD_OUT_OFFSET
            )
            train_outcomes, train_keys = measure(
                train_pools, seed + 1_000_003
            )
        by_key.update(train_keys)
        if cache:
            import pickle

            with open(cache, "wb") as f:
                pickle.dump((eval_outcomes, train_outcomes, by_key), f)

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    n_censored = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            n_censored += o.relation == "censored"
            pos += 1
    for outs in train_outcomes:
        for o in outs:
            n_censored += o.relation == "censored"
    measure_s = time.time() - t0

    # The baselines' predictors — ``delta`` (the round-7 gate signal)
    # and ``useful`` + its isotonic map (round 9's best earner).  Both
    # are the non-serializable reference: the injected table itself.
    t_net = time.time()
    flat_train = [c for pool in train_pools for c in pool]
    flat_train_outcomes = [o for outs in train_outcomes for o in outs]
    all_cands = flat_train + flat_pool
    preds_by_label: dict[str, dict] = {}
    predictor: dict[str, Any] = {"nets": {}}
    for lname in ("delta", "useful"):
        preds, rep = sa._fit_preds(
            flat_train,
            flat_train_outcomes,
            all_cands,
            train_pools,
            page,
            seed,
            lname,
        )
        preds_by_label[lname] = preds
        predictor["nets"][lname] = {
            "train_positives": rep["train_positives"],
            "pos_weight": rep["pos_weight"],
        }
    s_u, y_u = sc._calibration_frame(
        flat_train,
        flat_train_outcomes,
        preds_by_label["useful"],
        sp.VALUE_LABELS["useful"],
    )
    cal_u = IsotonicCal().fit(s_u, y_u)
    inj_delta = sc._injection("delta", preds_by_label)
    inj_useful_iso = sc._injection(
        "useful",
        {"useful": sc._apply(cal_u, preds_by_label["useful"])},
    )
    predictor["seconds"] = round(time.time() - t_net, 1)

    rows_out: list[dict] = []
    gates: dict[str, dict] = {}
    heads: dict[str, dict] = {}
    curves: dict[str, list[float]] = {}
    orig_evaluate = lp.evaluate
    lp.evaluate = pb._patched_evaluate(by_key)
    osel.set_self_scores(None)
    try:
        inc_spec = _spec_for(())
        heur_spec = _spec_for(("open:next-value",))
        prior_spec = _spec_for(("open:prior",))

        baked = og.OpenLearnedPlayer(seed)
        w_baked = baked.weights_dict()
        anchors = {
            "baked-prior": {
                "sampled": sl._play_open_pools(
                    lambda i: baked.frozen(seed + 30_000 + i),
                    ev_pools,
                    budget,
                ),
                "ap": round(
                    pb._ap(
                        sp._page_scores(
                            og.OPEN_SPEC, w_baked, ev_pools
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
                    pb._ap([0.0] * len(flat_pool), flat_outcomes),
                    4,
                ),
            },
        }
        baked_ap = anchors["baked-prior"]["ap"]

        # The incumbent-schema ranker — the rounds-4/9 recipe.
        inc_player = og.OpenLearnedPlayer(
            seed=seed, spec=inc_spec, weights={}
        )
        curve: list[float] = []
        for _ in range(epochs):
            totals = engine.train_policy(
                inc_player, train_pools, sl._FixedArena, budget=budget
            )
            curve.append(round(sum(totals) / max(len(totals), 1), 4))
        w_inc = inc_player.weights_dict()
        inc_ap = round(
            pb._ap(
                sp._page_scores(inc_spec, w_inc, ev_pools),
                flat_outcomes,
            ),
            4,
        )
        curves["incumbent"] = curve
        og.set_value_preds(None)
        inc_sampled = sp._play_pools(
            lambda i: inc_player.frozen(seed + _SAMPLED_SEED + i),
            ev_pools,
            budget,
        )
        inc_greedy = sp._play_pools(
            lambda i: inc_player.frozen(
                seed + _GREEDY_SEED + i, greedy=True
            ),
            ev_pools,
            budget,
        )
        rows_out.append(
            _row("incumbent", "rl", inc_sampled, inc_greedy, inc_ap)
        )
        print(  # stdout-compat
            f"  [incumbent] reward={inc_sampled['reward']:>8.4f} "
            f"evals={inc_sampled['evals_per_ep']:>5.2f} ap={inc_ap:.4f}"
        )

        # ``ac`` — the closest serializable arm's credit mechanism.
        ac_player = adv.AdvantagePlayer(
            seed=seed,
            spec=inc_spec,
            weights={},
            state_feats=lambda s, h, s_=inc_spec: s_.featurize(
                s, Action("stop", {}), h
            ),
            v_lr=svl.V_LR,
            v_passes=svl.V_PASSES,
        )
        curve = []
        for _ in range(epochs):
            totals = engine.train_policy(
                ac_player, train_pools, sl._FixedArena, budget=budget
            )
            curve.append(round(sum(totals) / max(len(totals), 1), 4))
        w_ac = ac_player.weights_dict()
        ac_ap = round(
            pb._ap(
                sp._page_scores(inc_spec, w_ac, ev_pools), flat_outcomes
            ),
            4,
        )
        curves["ac"] = curve
        ac_sampled = sp._play_pools(
            lambda i: ac_player.frozen(seed + _SAMPLED_SEED + i),
            ev_pools,
            budget,
        )
        ac_greedy = sp._play_pools(
            lambda i: ac_player.frozen(
                seed + _GREEDY_SEED + i, greedy=True
            ),
            ev_pools,
            budget,
        )
        rows_out.append(
            _row("ac", "ac", ac_sampled, ac_greedy, ac_ap)
        )
        print(  # stdout-compat
            f"  [ac] reward={ac_sampled['reward']:>8.4f} "
            f"evals={ac_sampled['evals_per_ep']:>5.2f} ap={ac_ap:.4f}"
        )

        # Channel controls: the oracle ceiling, the never-stop drain.
        orc = sp._play_pools(
            lambda i: si._mk_iso(
                w_inc,
                inc_spec,
                seed + _SAMPLED_SEED + i,
                score=si._oracle_score(by_key, inc_spec, w_inc),
                kappa=0.0,
            ),
            ev_pools,
            budget,
        )
        orc_g = sp._play_pools(
            lambda i: si._mk_iso(
                w_inc,
                inc_spec,
                seed + _GREEDY_SEED + i,
                greedy=True,
                score=si._oracle_score(by_key, inc_spec, w_inc),
                kappa=0.0,
            ),
            ev_pools,
            budget,
        )
        rows_out.append(
            _row("oracle-chan", "oracle", orc, orc_g, inc_ap)
        )
        dr = sp._play_pools(
            lambda i: si._mk_iso(w_inc, inc_spec, seed + _SAMPLED_SEED + i),
            ev_pools,
            budget,
        )
        dr_g = sp._play_pools(
            lambda i: si._mk_iso(
                w_inc, inc_spec, seed + _GREEDY_SEED + i, greedy=True
            ),
            ev_pools,
            budget,
        )
        rows_out.append(_row("iso-drain", "control", dr, dr_g, inc_ap))
        print(  # stdout-compat
            f"  [oracle]={orc['reward']:.4f} [drain]={dr['reward']:.4f}"
        )

        # The non-serializable baselines — the injected-table gates.
        og.set_value_preds(inj_delta)
        for rule in ("margin", "pool"):
            bs = sp._play_pools(
                lambda i, r=rule: si._mk_iso(
                    w_inc,
                    inc_spec,
                    seed + _SAMPLED_SEED + i,
                    score=si._ev_score(r, 1.0),
                    kappa=0.0,
                ),
                ev_pools,
                budget,
            )
            bg = sp._play_pools(
                lambda i, r=rule: si._mk_iso(
                    w_inc,
                    inc_spec,
                    seed + _GREEDY_SEED + i,
                    greedy=True,
                    score=si._ev_score(r, 1.0),
                    kappa=0.0,
                ),
                ev_pools,
                budget,
            )
            rows_out.append(
                _row(
                    f"ev-{rule}:delta",
                    "gate-injected",
                    bs,
                    bg,
                    inc_ap,
                )
            )
            print(  # stdout-compat
                f"  [ev-{rule}:delta] reward={bs['reward']:>8.4f} "
                f"evals={bs['evals_per_ep']:>5.2f} stops={bs['stops']}"
            )
        og.set_value_preds(inj_useful_iso)
        bs = sp._play_pools(
            lambda i: si._mk_iso(
                w_inc,
                inc_spec,
                seed + _SAMPLED_SEED + i,
                score=si._ev_score("pool", 1.5),
                kappa=0.0,
            ),
            ev_pools,
            budget,
        )
        bg = sp._play_pools(
            lambda i: si._mk_iso(
                w_inc,
                inc_spec,
                seed + _GREEDY_SEED + i,
                greedy=True,
                score=si._ev_score("pool", 1.5),
                kappa=0.0,
            ),
            ev_pools,
            budget,
        )
        rows_out.append(
            _row(
                "ev-pool:useful/iso@1.5x",
                "gate-injected",
                bs,
                bg,
                inc_ap,
            )
        )
        og.set_value_preds(None)
        print(  # stdout-compat
            f"  [ev-pool:useful/iso] reward={bs['reward']:>8.4f} "
            f"evals={bs['evals_per_ep']:>5.2f} stops={bs['stops']}"
        )

        # ----------------------------------------------------------
        #  The serializable arms.  Every row's player is rebuilt from
        #  the JSON round-trip of its payload — the number IS the
        #  serialized artifact's number.
        # ----------------------------------------------------------

        # The κ-fit drains — one greedy pass per (stop spec, ranker).
        t_dr = time.time()
        drains = {
            ("inc", "inc"): _drain_stats(
                inc_spec,
                w_inc,
                inc_spec,
                train_pools,
                by_key,
                budget,
                fns={"self": _self_v(w_inc)},
            ),
            ("inc", "baked"): _drain_stats(
                inc_spec,
                w_baked,
                inc_spec,
                train_pools,
                by_key,
                budget,
                fns={"self": _self_v(w_baked)},
            ),
            ("next", "inc"): _drain_stats(
                heur_spec, w_inc, inc_spec, train_pools, by_key, budget
            ),
            ("prior", "inc"): _drain_stats(
                prior_spec, w_inc, inc_spec, train_pools, by_key, budget
            ),
            ("prior", "baked"): _drain_stats(
                prior_spec,
                w_baked,
                inc_spec,
                train_pools,
                by_key,
                budget,
            ),
        }
        print(  # stdout-compat
            f"  [drains] {time.time() - t_dr:.0f}s — κ fits by prefix-sum"
        )

        # The single-column gates — ``w_stop = {col: 1}``, κ* from the
        # train drain, ranker per the arm row.  Played through the
        # decoded payload (w_stop linear path — the purest form).
        w_of = {"inc": w_inc, "baked": w_baked}
        ap_of = {"inc": inc_ap, "baked": baked_ap}
        for arm_id, col, ranker, tags in GATE_ARMS:
            spec = _spec_for(tags)
            dkey = (
                "prior" if "open:prior" in tags else "next",
                ranker,
            )
            recs = drains[dkey]
            series = [[r.get(col, 0.0) for r in d["rows"]] for d in recs]
            rewards = [d["rewards"] for d in recs]
            kstar, train_r, curve_pts = _kappa_fit(series, rewards)
            pl = _payload(
                rank_weights=w_of[ranker],
                rank_spec=inc_spec,
                stop_spec=spec,
                w_stop={col: 1.0},
                kappa=kstar,
            )
            sampled, greedy = _play_payload(pl, ev_pools, budget, seed)
            aid = f"gate:{arm_id}:{ranker}"
            rows_out.append(
                _row(
                    aid,
                    "gate-column",
                    sampled,
                    greedy,
                    ap_of[ranker],
                    kappa=kstar,
                    column=col,
                )
            )
            gates[aid] = {
                "kappa": round(kstar, 6),
                "train_reward": train_r,
                "curve": curve_pts,
                "column": col,
            }
            print(  # stdout-compat
                f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"stops={sampled['stops']:>3} k*={kstar:.4f}"
            )

        # The analytic κ=0 prior-margin arm — the named constant,
        # no fit at all (the prior's own indifference point).
        for ranker in ("inc", "baked"):
            pl = _payload(
                rank_weights=w_of[ranker],
                rank_spec=inc_spec,
                stop_spec=prior_spec,
                w_stop={"x:stop:prior_margin": 1.0},
                kappa=0.0,
            )
            sampled, greedy = _play_payload(pl, ev_pools, budget, seed)
            aid = f"gate:prior-margin@0:{ranker}"
            rows_out.append(
                _row(
                    aid,
                    "gate-column",
                    sampled,
                    greedy,
                    ap_of[ranker],
                    kappa=0.0,
                    column="x:stop:prior_margin",
                )
            )
            print(  # stdout-compat
                f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"stops={sampled['stops']:>3} k=0 analytic"
            )

        # ``self-margin`` — the ranker's own score as a named scorer.
        for ranker in ("inc", "baked"):
            recs = drains[("inc", ranker)]
            series = [d["vals"]["self"] for d in recs]
            rewards = [d["rewards"] for d in recs]
            kstar, train_r, curve_pts = _kappa_fit(series, rewards)
            pl = _payload(
                rank_weights=w_of[ranker],
                rank_spec=inc_spec,
                kappa=kstar,
                scorer="self-margin",
            )
            sampled, greedy = _play_payload(pl, ev_pools, budget, seed)
            aid = f"gate:self-margin:{ranker}"
            rows_out.append(
                _row(
                    aid,
                    "gate-scorer",
                    sampled,
                    greedy,
                    ap_of[ranker],
                    kappa=kstar,
                )
            )
            gates[aid] = {
                "kappa": round(kstar, 6),
                "train_reward": train_r,
                "curve": curve_pts,
            }
            print(  # stdout-compat
                f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"stops={sampled['stops']:>3} k*={kstar:.4f}"
            )

        # The learned heads as payloads.  ``vreg`` is the pure
        # ``w_stop`` linear path (κ=0); ``head-iso`` is the named
        # scorer (weights + knot list + p*).  Each fit reads the
        # drain matching its ranker.
        for tags, key in (((), "inc"), (("open:prior",), "prior")):
            spec = _spec_for(tags)
            for ranker in ("inc", "baked"):
                recs = drains[(key, ranker)]
                hrows = [r for d in recs for r in d["rows"]]
                hlabels = [b for d in recs for b in d["labels"]]
                hvals = [v for d in recs for v in d["prefix_v"]]
                hpend = [n for d in recs for n in d["pend"]]
                w_v, rmse = si._fit_value_head(
                    hrows, hvals, spec.features
                )
                pl = _payload(
                    rank_weights=w_of[ranker],
                    rank_spec=inc_spec,
                    stop_spec=spec,
                    w_stop=w_v,
                    kappa=0.0,
                )
                sampled, greedy = _play_payload(
                    pl, ev_pools, budget, seed
                )
                aid = f"vhead{'+' + key if key != 'inc' else ''}:{ranker}"
                rows_out.append(
                    _row(
                        aid,
                        "head-wstop",
                        sampled,
                        greedy,
                        ap_of[ranker],
                        kappa=0.0,
                    )
                )
                heads[aid] = {
                    "train_rmse": rmse,
                    "weights": {
                        n: round(wv, 4)
                        for n, wv in w_v.items()
                        if abs(wv) > 1e-4
                    },
                }
                print(  # stdout-compat
                    f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                    f"evals={sampled['evals_per_ep']:>5.2f} "
                    f"stops={sampled['stops']:>3} (vreg)"
                )
                # The calibrated logistic head — named scorer payload.
                w_h, hloss = sp._fit_head(spec, hrows, hlabels, seed)
                pstar, vbar, cbar = si._breakeven_p(
                    hvals, hlabels, hpend
                )
                cal = IsotonicCal().fit(
                    [si._sig(_dot(w_h, r)) for r in hrows],
                    [float(b) for b in hlabels],
                )
                pl = _payload(
                    rank_weights=w_of[ranker],
                    rank_spec=inc_spec,
                    stop_spec=spec,
                    kappa=pstar,
                    scorer="cal-head",
                    scorer_params={
                        "weights": w_h,
                        "knots": [list(k) for k in cal.knots_],
                    },
                )
                sampled, greedy = _play_payload(
                    pl, ev_pools, budget, seed
                )
                aid = (
                    f"head-iso{'+' + key if key != 'inc' else ''}"
                    f":{ranker}"
                )
                rows_out.append(
                    _row(
                        aid,
                        "head-scorer",
                        sampled,
                        greedy,
                        ap_of[ranker],
                        kappa=pstar,
                    )
                )
                heads[aid] = {
                    "pstar": round(pstar, 4),
                    "vbar": vbar,
                    "cbar": cbar,
                    "loss": hloss,
                    "n_knots": len(cal.knots_),
                    "weights": {
                        n: round(wv, 4)
                        for n, wv in w_h.items()
                        if abs(wv) > 1e-4
                    },
                }
                print(  # stdout-compat
                    f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                    f"evals={sampled['evals_per_ep']:>5.2f} "
                    f"stops={sampled['stops']:>3} p*={pstar:.4f}"
                )

        # The shared-softmax A/B — the vreg head folded into the one
        # weight table (the port control; expect the degenerate point).
        recs = drains[("inc", "inc")]
        hrows = [r for d in recs for r in d["rows"]]
        hvals = [v for d in recs for v in d["prefix_v"]]
        hlabels = [b for d in recs for b in d["labels"]]
        w_v, _rm = si._fit_value_head(hrows, hvals, inc_spec.features)
        fold_c, fold_cal = sp._calibrate_fold_c(
            inc_spec, w_inc, w_v, train_pools, budget, seed
        )
        w_fold = sp._fold_weights(inc_spec, w_inc, w_v, fold_c)
        fs = sp._play_pools(
            lambda i: og.OpenLearnedPlayer(
                seed=seed + _SAMPLED_SEED + i,
                spec=inc_spec,
                weights=w_fold,
                learn=False,
            ),
            ev_pools,
            budget,
        )
        fg = sp._play_pools(
            lambda i: og.OpenLearnedPlayer(
                seed=seed + _GREEDY_SEED + i,
                spec=inc_spec,
                weights=w_fold,
                learn=False,
                greedy=True,
            ),
            ev_pools,
            budget,
        )
        rows_out.append(
            _row(
                "vhead-fold",
                "fold",
                fs,
                fg,
                inc_ap,
                kappa=round(fold_c, 4),
            )
        )
        heads["vhead-fold"] = {
            "c": round(fold_c, 4),
            "cal_reward": fold_cal,
        }
        print(  # stdout-compat
            f"    vhead-fold                  "
            f"reward={fs['reward']:>8.4f} stops={fs['stops']:>3}"
        )

        # The registry-seat proof — the same payload played through
        # ``registry.make("gated", "open", ctx)``, the consumption
        # path ``--player gated`` uses.  Must land identical to the
        # direct-build row (same inner seeds: ``seed + 40_000 + i``).
        seat = {}
        mk = registry.make(
            "gated",
            "open",
            seed=seed,
            ctx={
                "gated": {
                    "spec": inc_spec,
                    "weights": w_v,
                    "kappa": 0.0,
                    "rank_weights": w_inc,
                }
            },
        )
        seat_played = sp._play_pools(mk, ev_pools, budget)
        seat["vhead:inc"] = seat_played
        print(  # stdout-compat
            f"    seat:vhead:inc              "
            f"reward={seat_played['reward']:>8.4f} "
            f"stops={seat_played['stops']:>3}"
        )
    finally:
        lp.evaluate = orig_evaluate
        og.set_value_preds(None)
        osel.set_self_scores(None)

    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": rows_out,
        "anchors": anchors,
        "seat": seat,
        "gates": gates,
        "heads": heads,
        "train_curve": curves,
        "predictor": predictor,
        "pool": {
            "held_out": len(flat_pool),
            "useful": n_useful,
            "censored": n_censored,
        },
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "cand_timeout_s": cand_timeout,
            "epochs": epochs,
            "kappa_grid": _KAPPA_GRID,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the arm table."""
    print("== the serialized gate — held-out odd pages ==")  # stdout-compat
    head = (
        f"{'arm':<32} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
    )
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for r in res["rows"]:
        print(  # stdout-compat
            f"{r['id']:<32} {r['reward_ep']:>8.4f} "
            f"{r['reward_greedy_ep']:>8.4f} {r['useful_eval']:>9.4f} "
            f"{r['evals_ep']:>8.3f} {r['stops']:>6} {r['ap']:>7.4f}"
        )
    print("-- anchors --")  # stdout-compat
    for name, a in res["anchors"].items():
        s = a["sampled"]
        print(  # stdout-compat
            f"{name:<32} {s['reward']:>8.4f} {'—':>8} "
            f"{s['useful_per_eval']:>9.4f} "
            f"{s['evals_per_ep']:>8.3f} {'—':>6} {a['ap']:>7.4f}"
        )
    for aid, g in res.get("gates", {}).items():
        print(  # stdout-compat
            f"-- gate {aid}: k*={g['kappa']} "
            f"train_reward={g['train_reward']} col={g.get('column')} --"
        )
    for aid, h in res.get("heads", {}).items():
        if "pstar" in h:
            print(  # stdout-compat
                f"-- head {aid}: p*={h['pstar']} V+={h['vbar']} "
                f"C-={h['cbar']} knots={h['n_knots']} --"
            )
        elif "train_rmse" in h:
            print(  # stdout-compat
                f"-- vreg {aid}: rmse={h['train_rmse']} --"
            )
    for aid, s in res.get("seat", {}).items():
        print(  # stdout-compat
            f"-- seat {aid}: reward={s['reward']} "
            f"stops={s['stops']} (registry.make path) --"
        )
    if res["pool"].get("censored"):
        print(  # stdout-compat
            f"-- censored: {res['pool']['censored']} candidates "
            f"hit the {res['meta'].get('cand_timeout_s')}s eval cap "
            f"(relation='censored', derivable=False) --"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the serialized-gate measurement and print the table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument(
        "--cache",
        type=str,
        default=None,
        help="pickle cache for the memoized verdicts (rerun-safe)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="process-pool workers for the verdict memoization "
        "(identical outcomes — the seeds are per-candidate)",
    )
    ap.add_argument(
        "--cand-timeout",
        type=int,
        default=0,
        help="per-candidate eval cap in seconds; timed-out "
        "candidates report relation='censored' (derivable=False "
        "— the machinery's own resource-refusal class) and are "
        "counted under pool.censored",
    )
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    res = run(
        args.seed,
        args.open_pages,
        args.page,
        args.page + 1,
        args.epochs,
        cache=args.cache,
        workers=args.workers,
        cand_timeout=args.cand_timeout,
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
            "delivery": (
                "GatedPlayer — every arm rebuilt from the JSON "
                "round-trip of its payload "
                "{rank:{spec,weights}, stop:{spec,weights,kappa,"
                "scorer,params}}"
            ),
            "kappa": "bake-time fit on train drains (prefix-sum); "
            "analytic κ=0 rows included",
        },
        "results": res,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
