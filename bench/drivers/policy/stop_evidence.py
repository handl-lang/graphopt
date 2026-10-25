"""The stopping policy, round 12 — evidence x the serializable gate.

Two halves, each proven, one payload.  Round 10
(``docs/results_stop_gate.md``) showed the ``gated`` seat's config is
pure data — every arm played from its ``json.dumps`` round-trip — and
that the ``baked`` rank table carries the bake-gate AP (0.2101) the
cold-retrained incumbent cannot reach (0.2074): ``vhead:baked``
cleared honestly at 0.2893 reward / 0.2101 AP with 26 stops.  Round
11 (``docs/results_stop_verdict.md``) found the evidence that
*transfers*: the ``open:verdict`` part's ``vd_*``/``cr_*`` stop
columns — ``x:stop:vd_live`` alone posts decision-row AP 0.631 — and
its ``head+vd-ep`` arm cleared the reward bar (0.2866) with real
stops (32, 8.0 evals/ep) but sat on the incumbent rank floor.  This
driver measures the composition: **the ``open:verdict`` stop spec
over the ``baked`` rank table, played as serialized ``gated``
payloads** — the gate is pure data (``{spec record, w_stop, κ,
scorer+params}``) and every row is rebuilt from its JSON round-trip.

The corpus seam rides honestly, exactly as round 11 bound it: the
train-side drains for the ``+vd`` spec read *jackknifed* corpus
tables (each train pool's rows minus its own page's contribution —
the count-table OOF analogue), while eval drains and eval play read
the full train tally — held-out candidates only ever see
train-certified counts.  The ``+vd-ep`` variants clear the seam
entirely: episode-only columns, no corpus table needed at decode —
the more deployable artifact.

Arms:

* ``gate:<col>:<ranker>`` — the 13 ``x:stop:vd_*``/``x:stop:cr_*``
  columns as single-literal gates (``w_stop={col:1}``, κ fit on the
  train drains by prefix-sum — bake-time only);
* ``vhead[+vd|+vd-ep]:<ranker>`` — the ridge ``w_stop`` literal head
  (κ=0) over the incumbent / verdict / episode-only stop rows —
  ``vhead:baked`` is the round-10 winner rerun in the same protocol;
* ``head-iso+…:<ranker>`` — the logistic head + isotonic knots +
  break-even p* as the named ``cal-head`` scorer payload;
* ``headk+…:<ranker>`` — the logistic head as a pure ``w_stop``
  literal with κ prefix-fit on its own train-drain score series —
  the payload-expressed form of round 11's τ-calibrated gate;
* ``head+vd`` / ``head+vd-ep`` — the round-11 mechanism replayed
  verbatim (``sp._calibrate_tau`` + ``sp._mk_gate``, incumbent
  ranker) for the same-run comparison.

Bake rule unchanged: held-out reward/ep ≥ 0.273 ∧ page-aware AP
≥ 0.210, and the arm must actually stop — never-stop clears are the
round-7 confound and do not count.

Runtime: with ``--cache`` warm (the ``stop_gate.py`` verdict pickle —
``gen_cases`` is identical, so the whole oracle pass is skipped) the
run is minutes; cold, the memoization is the same ~30-60 min the
round-10 doc prices (``--workers`` shards it, ``--cand-timeout``
caps the unbounded certificate path as ``relation="censored"``).

Usage::

    timeout 2400 .venv/bin/python bench/drivers/policy/stop_evidence.py \
        --json /tmp/evidence.json --cache /tmp/gate_cache.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent)
)

import policy_baseline as pb
import schema_leaderboard as sl
import stop_gate as sg
import stop_isolated as si
import stop_policy as sp
import stop_valuelabel as svl
import stop_verdict as sv
from gamekit import engine
from gamekit.calibrate import IsotonicCal
from gamekit.players import PARTS, DomainSpec, _dot, registry
from mlopt_discovery import open_game as og
from mlopt_discovery import open_self as osel
from mlopt_discovery import open_verdict as ov
from mlopt_discovery import play
from mlopt_discovery import proposal as lp

#: Defaults — identical protocol to ``stop_gate.py``/``stop_verdict.py``.
SEED = svl.SEED
PAGE = svl.PAGE
OPEN_PAGES = svl.OPEN_PAGES
EPOCHS = svl.EPOCHS
_SAMPLED_SEED = svl._SAMPLED_SEED
_GREEDY_SEED = svl._GREEDY_SEED

#: The extra-part vocabulary — ``stop_gate``'s (shipped extras +
#: ``open:self``/``open:vlabel``/``open:prior``) plus the round-11
#: ``open:verdict`` part.
EXTRA: dict[str, tuple[str, ...]] = {
    **sg.EXTRA,
    "open:verdict": ov.VD_FEATURES,
}

#: The stop-row columns the single-literal gates price.
VD_STOP_COLS: tuple[str, ...] = tuple(
    f for f in ov.VD_FEATURES if f.startswith("x:stop:")
)

#: The stop-spec axis — ``(id suffix, part tags, corpus binding)``.
#: ``""`` is the incumbent schema (round 10's spec); ``"+vd"`` reads
#: the corpus tables (jackknifed at fit, full at eval); ``"+vd-ep"``
#: is the same spec with the seam cleared — the episode-only ablation.
STOP_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("", (), "none"),
    ("+vd", ("open:verdict",), "jack"),
    ("+vd-ep", ("open:verdict",), "ep"),
)

RANKERS: tuple[str, ...] = ("inc", "baked")


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


def _drain_stats_vd(
    spec: DomainSpec,
    w_rank: dict,
    rank_spec: DomainSpec,
    pools: list[list],
    by_key: dict,
    budget: int,
    tables: list[dict | None],
) -> list[dict]:
    """``sg._drain_stats`` with a per-pool corpus-table binding.

    ``tables[i]`` (the jackknifed table for train pools, the full
    table for eval pools, ``None`` for the episode-only ablation) is
    bound before the pool's drain is featurized — the ``vd_*``
    columns ignore it (they read the episode's own banked verdicts)
    but the ``cr_*`` columns are exactly its content, so the head's
    fit sees held-out-quality statistics.
    """
    out: list[dict] = []
    for pool, tab in zip(pools, tables, strict=True):
        ov.set_verdict_corpus(tab)
        out.extend(
            sg._drain_stats(
                spec, w_rank, rank_spec, [pool], by_key, budget
            )
        )
    return out


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
    """Measure pools once; fit κ/heads on train drains; eval held-out.

    *cache* pickles the memoized verdicts (the ``stop_gate.py``
    convention — ``(eval, train, by_key)``, deterministic under the
    per-candidate seeds); a warm cache skips the whole oracle pass.
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
        if workers > 1:
            eval_outcomes, by_key = sg._measure_pools_par(
                ev_pools,
                seed + pb.HELD_OUT_OFFSET,
                workers,
                None if cache is None else cache + ".ev",
                cand_timeout,
            )
            train_outcomes, train_keys = sg._measure_pools_par(
                train_pools,
                seed + 1_000_003,
                workers,
                None if cache is None else cache + ".tr",
                cand_timeout,
            )
        else:
            eval_outcomes, by_key = pb._measure_pools(
                ev_pools, seed + pb.HELD_OUT_OFFSET
            )
            train_outcomes, train_keys = pb._measure_pools(
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

    # The corpus verdict table — counts over the *train* pages'
    # certified verdicts (never the held-out half), plus the
    # per-page jackknifed variants the train drains need (the
    # count-table analogue of round 5's OOF train preds).
    t_tab = time.time()
    corpus = sv._corpus_table(train_pools, train_outcomes)
    train_tables = []
    for pool, outs in zip(train_pools, train_outcomes, strict=True):
        drop: dict[str, dict] = {"sig": {}, "form": {}}
        for c, o in zip(pool, outs, strict=True):
            v = og._verdict(o)
            for kind, key in (
                ("sig", ov.signature(c)),
                ("form", c.label.split(":", 1)[0]),
            ):
                p, u, n = drop[kind].get(key, (0, 0, 0))
                drop[kind][key] = (
                    p + (v in ("useful", "true")),
                    u + (v == "useful"),
                    n + 1,
                )
        train_tables.append(sv._jackknife(corpus, drop))
    eval_tables = [corpus] * len(ev_pools)
    sig_cov = sum(ov.signature(c) in corpus["sig"] for c in flat_pool)
    corpus_report = {
        "n_signatures": len(corpus["sig"]),
        "n_forms": len(corpus["form"]),
        "train_payers": sum(t[0] for t in corpus["sig"].values()),
        "train_useful": sum(t[1] for t in corpus["sig"].values()),
        "eval_sig_coverage": f"{sig_cov}/{len(flat_pool)}",
        "eval_rate_ap_pays": round(
            sp._ap_binary(
                [ov._cand_rates(c)[0] for c in flat_pool],
                [
                    og._verdict(o) in ("useful", "true")
                    for o in (
                        flat_outcomes[i] for i in range(len(flat_pool))
                    )
                ],
            ),
            4,
        ),
        "seconds": round(time.time() - t_tab, 1),
    }

    rows_out: list[dict] = []
    gates: dict[str, dict] = {}
    heads: dict[str, dict] = {}
    curves: dict[str, list[float]] = {}
    payloads: dict[str, dict] = {}
    seat: dict[str, dict] = {}
    orig_evaluate = lp.evaluate
    lp.evaluate = pb._patched_evaluate(by_key)
    osel.set_self_scores(None)
    ov.set_verdict_corpus(None)
    try:
        inc_spec = _spec_for(())
        vd_spec = _spec_for(("open:verdict",))
        spec_of = {"": inc_spec, "+vd": vd_spec, "+vd-ep": vd_spec}
        bind_of = {"": None, "+vd": corpus, "+vd-ep": None}

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

        # The incumbent-schema ranker — the rounds-4/9/10 recipe.
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
            sg._row("incumbent", "rl", inc_sampled, inc_greedy, inc_ap)
        )
        print(  # stdout-compat
            f"  [incumbent] reward={inc_sampled['reward']:>8.4f} "
            f"evals={inc_sampled['evals_per_ep']:>5.2f} ap={inc_ap:.4f}"
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
            sg._row("oracle-chan", "oracle", orc, orc_g, inc_ap)
        )
        dr = sp._play_pools(
            lambda i: si._mk_iso(
                w_inc, inc_spec, seed + _SAMPLED_SEED + i
            ),
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
        rows_out.append(
            sg._row("iso-drain", "control", dr, dr_g, inc_ap)
        )
        print(  # stdout-compat
            f"  [oracle]={orc['reward']:.4f} [drain]={dr['reward']:.4f}"
        )

        # ----------------------------------------------------------
        #  The drains — one greedy pass per (stop spec, ranker) on the
        #  train pools (κ fits are prefix-sums over them), plus the
        #  held-out drains the heads' decision-row APs are priced on.
        # ----------------------------------------------------------
        t_dr = time.time()
        drains = {
            ("", "inc"): sg._drain_stats(
                inc_spec, w_inc, inc_spec, train_pools, by_key, budget
            ),
            ("", "baked"): sg._drain_stats(
                inc_spec, w_baked, inc_spec, train_pools, by_key, budget
            ),
            ("+vd", "inc"): _drain_stats_vd(
                vd_spec,
                w_inc,
                inc_spec,
                train_pools,
                by_key,
                budget,
                train_tables,
            ),
            ("+vd", "baked"): _drain_stats_vd(
                vd_spec,
                w_baked,
                inc_spec,
                train_pools,
                by_key,
                budget,
                train_tables,
            ),
            ("+vd-ep", "inc"): _drain_stats_vd(
                vd_spec,
                w_inc,
                inc_spec,
                train_pools,
                by_key,
                budget,
                [None] * len(train_pools),
            ),
            ("+vd-ep", "baked"): _drain_stats_vd(
                vd_spec,
                w_baked,
                inc_spec,
                train_pools,
                by_key,
                budget,
                [None] * len(train_pools),
            ),
        }
        edrains = {
            ("", "inc"): sg._drain_stats(
                inc_spec, w_inc, inc_spec, ev_pools, by_key, budget
            ),
            ("", "baked"): sg._drain_stats(
                inc_spec, w_baked, inc_spec, ev_pools, by_key, budget
            ),
            ("+vd", "inc"): _drain_stats_vd(
                vd_spec,
                w_inc,
                inc_spec,
                ev_pools,
                by_key,
                budget,
                eval_tables,
            ),
            ("+vd", "baked"): _drain_stats_vd(
                vd_spec,
                w_baked,
                inc_spec,
                ev_pools,
                by_key,
                budget,
                eval_tables,
            ),
            ("+vd-ep", "inc"): _drain_stats_vd(
                vd_spec,
                w_inc,
                inc_spec,
                ev_pools,
                by_key,
                budget,
                [None] * len(ev_pools),
            ),
            ("+vd-ep", "baked"): _drain_stats_vd(
                vd_spec,
                w_baked,
                inc_spec,
                ev_pools,
                by_key,
                budget,
                [None] * len(ev_pools),
            ),
        }
        print(  # stdout-compat
            f"  [drains] {time.time() - t_dr:.0f}s — "
            f"κ/head fits by prefix-sum"
        )

        w_of = {"inc": w_inc, "baked": w_baked}
        ap_of = {"inc": inc_ap, "baked": baked_ap}

        # ----------------------------------------------------------
        #  The serializable arms.  Every row's player is rebuilt from
        #  the JSON round-trip of its payload — the number IS the
        #  serialized artifact playing.
        # ----------------------------------------------------------

        # The single-literal gates — ``w_stop = {col: 1}`` over the
        # verdict spec, κ* from the train drain, ranker per the column.
        for col in VD_STOP_COLS:
            cname = col.rsplit(":", 1)[-1].replace("_", "-")
            for ranker in RANKERS:
                recs = drains[("+vd", ranker)]
                series = [
                    [r.get(col, 0.0) for r in d["rows"]] for d in recs
                ]
                rewards = [d["rewards"] for d in recs]
                kstar, train_r, _pts = sg._kappa_fit(series, rewards)
                pl = sg._payload(
                    rank_weights=w_of[ranker],
                    rank_spec=inc_spec,
                    stop_spec=vd_spec,
                    w_stop={col: 1.0},
                    kappa=kstar,
                )
                ov.set_verdict_corpus(corpus)
                sampled, greedy = sg._play_payload(
                    pl, ev_pools, budget, seed
                )
                aid = f"gate:{cname}:{ranker}"
                rows_out.append(
                    sg._row(
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
                    "column": col,
                }
                payloads[aid] = pl
                print(  # stdout-compat
                    f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                    f"evals={sampled['evals_per_ep']:>5.2f} "
                    f"stops={sampled['stops']:>3} k*={kstar:.4f}"
                )

        # The learned heads as payloads — per (stop spec, ranker):
        # ``vhead`` (ridge → w_stop literals, κ=0 — round 10's arm),
        # ``head-iso`` (logistic + isotonic knots + break-even p* →
        # the named ``cal-head`` scorer) and ``headk`` (the same
        # logistic head as a pure w_stop literal, κ prefix-fit on its
        # own score series — the payload-expressed round-11 gate).
        for skey, _tags, _bind in STOP_SPECS:
            spec = spec_of[skey]
            for ranker in RANKERS:
                recs = drains[(skey, ranker)]
                erecs = edrains[(skey, ranker)]
                hrows = [r for d in recs for r in d["rows"]]
                hlabels = [b for d in recs for b in d["labels"]]
                hvals = [v for d in recs for v in d["prefix_v"]]
                hpend = [n for d in recs for n in d["pend"]]
                rewards = [d["rewards"] for d in recs]
                erows = [r for d in erecs for r in d["rows"]]
                elabels = [b for d in erecs for b in d["labels"]]

                # ``vhead`` — the ridge literal head, κ=0.
                w_v, rmse = si._fit_value_head(
                    hrows, hvals, spec.features
                )
                pl = sg._payload(
                    rank_weights=w_of[ranker],
                    rank_spec=inc_spec,
                    stop_spec=spec,
                    w_stop=w_v,
                    kappa=0.0,
                )
                ov.set_verdict_corpus(bind_of[skey])
                sampled, greedy = sg._play_payload(
                    pl, ev_pools, budget, seed
                )
                aid = f"vhead{skey}:{ranker}"
                rows_out.append(
                    sg._row(
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
                    "eval_ap": round(
                        sp._ap_binary(
                            [_dot(w_v, r) for r in erows], elabels
                        ),
                        4,
                    ),
                    "weights": {
                        n: round(wv, 4)
                        for n, wv in w_v.items()
                        if abs(wv) > 1e-4
                    },
                }
                payloads[aid] = pl
                print(  # stdout-compat
                    f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                    f"evals={sampled['evals_per_ep']:>5.2f} "
                    f"stops={sampled['stops']:>3} (vreg)"
                )

                # The logistic head — shared by ``head-iso``/``headk``.
                w_h, hloss = sp._fit_head(spec, hrows, hlabels, seed)
                head_ap = round(
                    sp._ap_binary(
                        [_dot(w_h, r) for r in erows], elabels
                    ),
                    4,
                )

                # ``head-iso`` — the calibrated named scorer.
                pstar, vbar, cbar = si._breakeven_p(
                    hvals, hlabels, hpend
                )
                cal = IsotonicCal().fit(
                    [si._sig(_dot(w_h, r)) for r in hrows],
                    [float(b) for b in hlabels],
                )
                pl = sg._payload(
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
                ov.set_verdict_corpus(bind_of[skey])
                sampled, greedy = sg._play_payload(
                    pl, ev_pools, budget, seed
                )
                aid = f"head-iso{skey}:{ranker}"
                rows_out.append(
                    sg._row(
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
                    "eval_ap": head_ap,
                    "weights": {
                        n: round(wv, 4)
                        for n, wv in w_h.items()
                        if abs(wv) > 1e-4
                    },
                }
                payloads[aid] = pl
                print(  # stdout-compat
                    f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                    f"evals={sampled['evals_per_ep']:>5.2f} "
                    f"stops={sampled['stops']:>3} p*={pstar:.4f}"
                )

                # ``headk`` — the same head as a w_stop literal, κ
                # prefix-fit on the train drains (the payload form of
                # round 11's τ-calibrated gate).
                series = [
                    [_dot(w_h, r) for r in d["rows"]] for d in recs
                ]
                kstar, train_r, _pts = sg._kappa_fit(series, rewards)
                pl = sg._payload(
                    rank_weights=w_of[ranker],
                    rank_spec=inc_spec,
                    stop_spec=spec,
                    w_stop=w_h,
                    kappa=kstar,
                )
                ov.set_verdict_corpus(bind_of[skey])
                sampled, greedy = sg._play_payload(
                    pl, ev_pools, budget, seed
                )
                aid = f"headk{skey}:{ranker}"
                rows_out.append(
                    sg._row(
                        aid,
                        "head-wstop-k",
                        sampled,
                        greedy,
                        ap_of[ranker],
                        kappa=kstar,
                    )
                )
                heads[aid] = {
                    "kappa": round(kstar, 6),
                    "train_reward": train_r,
                    "loss": hloss,
                    "eval_ap": head_ap,
                    "weights": {
                        n: round(wv, 4)
                        for n, wv in w_h.items()
                        if abs(wv) > 1e-4
                    },
                }
                payloads[aid] = pl
                print(  # stdout-compat
                    f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                    f"evals={sampled['evals_per_ep']:>5.2f} "
                    f"stops={sampled['stops']:>3} k*={kstar:.4f}"
                )

                # The fitted-spec probe — solo/LOO decision-row APs
                # on the held-out drain, on the deployment pairing
                # (verdict spec x baked ranker).
                if skey == "+vd" and ranker == "baked":
                    erecs_inc = edrains[("+vd", "inc")]
                    erows_inc = [
                        r for d in erecs_inc for r in d["rows"]
                    ]
                    elabels_inc = [
                        b for d in erecs_inc for b in d["labels"]
                    ]
                    probe: dict[str, dict] = {}
                    for f in VD_STOP_COLS:
                        solo = sp._ap_binary(
                            [r.get(f, 0.0) for r in erows], elabels
                        )
                        solo_inc = sp._ap_binary(
                            [r.get(f, 0.0) for r in erows_inc],
                            elabels_inc,
                        )
                        wl, _l = sp._fit_head(
                            spec,
                            hrows,
                            hlabels,
                            seed,
                            names=tuple(
                                n for n in spec.features if n != f
                            ),
                        )
                        loo = sp._ap_binary(
                            [_dot(wl, r) for r in erows], elabels
                        )
                        probe[f] = {
                            "solo_ap": round(solo, 4),
                            "solo_ap_inc": round(solo_inc, 4),
                            "loo_ap": round(loo, 4),
                            "loo_delta": round(loo - head_ap, 4),
                        }
                    heads[aid]["feature_probe"] = probe

        # The round-11 mechanism, verbatim — the supervised stop head
        # over the verdict spec, τ replay-calibrated on train pools,
        # incumbent ranker, played through ``sp._mk_gate``.  The
        # same-run comparison for the payload arms above.
        for skey, bind in (("+vd", corpus), ("+vd-ep", None)):
            recs = drains[(skey, "inc")]
            erecs = edrains[(skey, "inc")]
            hrows = [r for d in recs for r in d["rows"]]
            hlabels = [b for d in recs for b in d["labels"]]
            erows = [r for d in erecs for r in d["rows"]]
            elabels = [b for d in erecs for b in d["labels"]]
            w_h, hloss = sp._fit_head(vd_spec, hrows, hlabels, seed)
            ov.set_verdict_corpus(bind)
            tau, cal_reward = sp._calibrate_tau(
                vd_spec,
                inc_player,
                w_h,
                hrows,
                train_pools,
                budget,
                seed,
                rank_spec=inc_spec,
            )
            sampled = sp._play_pools(
                lambda i, wh=w_h, ta=tau: sp._mk_gate(
                    inc_player,
                    vd_spec,
                    wh,
                    ta,
                    seed + _SAMPLED_SEED + i,
                    rank_spec=inc_spec,
                ),
                ev_pools,
                budget,
            )
            greedy = sp._play_pools(
                lambda i, wh=w_h, ta=tau: sp._mk_gate(
                    inc_player,
                    vd_spec,
                    wh,
                    ta,
                    seed + _GREEDY_SEED + i,
                    greedy=True,
                    rank_spec=inc_spec,
                ),
                ev_pools,
                budget,
            )
            aid = f"head{skey}"
            rows_out.append(
                sg._row(
                    aid, "head-gate", sampled, greedy, inc_ap, kappa=tau
                )
            )
            heads[aid] = {
                "tau": round(tau, 4),
                "cal_reward": cal_reward,
                "loss": hloss,
                "n_rows": len(hrows),
                "pos_rate": round(
                    sum(hlabels) / max(len(hlabels), 1), 4
                ),
                "eval_ap": round(
                    sp._ap_binary(
                        [_dot(w_h, r) for r in erows], elabels
                    ),
                    4,
                ),
                "weights": {
                    n: round(wv, 4)
                    for n, wv in w_h.items()
                    if abs(wv) > 1e-4
                },
            }
            print(  # stdout-compat
                f"    {aid:<30} reward={sampled['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"stops={sampled['stops']:>3} tau={tau:.4f} (r11 replay)"
            )

        # The registry-seat proof — the combined payload played
        # through ``registry.make("gated", "open", ctx)``, the
        # consumption path ``--player gated`` uses.
        pl = payloads["vhead+vd:baked"]
        ov.set_verdict_corpus(corpus)
        mk = registry.make(
            "gated",
            "open",
            seed=seed,
            ctx={
                "gated": {
                    "spec": vd_spec,
                    "weights": pl["stop"]["weights"],
                    "kappa": pl["stop"]["kappa"],
                    "rank_weights": w_baked,
                }
            },
        )
        seat["vhead+vd:baked"] = sp._play_pools(mk, ev_pools, budget)
        print(  # stdout-compat
            f"    seat:vhead+vd:baked         "
            f"reward={seat['vhead+vd:baked']['reward']:>8.4f} "
            f"stops={seat['vhead+vd:baked']['stops']:>3}"
        )
    finally:
        lp.evaluate = orig_evaluate
        og.set_value_preds(None)
        osel.set_self_scores(None)
        ov.set_verdict_corpus(None)

    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": rows_out,
        "anchors": anchors,
        "seat": seat,
        "gates": gates,
        "heads": heads,
        "payloads": payloads,
        "train_curve": curves,
        "corpus": corpus_report,
        "pool": {
            "held_out": len(flat_pool),
            "useful": n_useful,
            "censored": n_censored,
        },
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "cand_timeout_s": cand_timeout,
            "epochs": epochs,
            "kappa_grid": sg._KAPPA_GRID,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the arm table."""
    cp = res.get("corpus")
    if cp:
        print(  # stdout-compat
            f"== corpus table: {cp['n_signatures']} signatures "
            f"({cp['n_forms']} forms), payers={cp['train_payers']} "
            f"useful={cp['train_useful']}, eval sig coverage "
            f"{cp['eval_sig_coverage']}, rate AP(pays)="
            f"{cp['eval_rate_ap_pays']} ({cp['seconds']}s) =="
        )
    print(  # stdout-compat
        "== evidence x serializable gate — held-out odd pages =="
    )
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
                f"C-={h['cbar']} knots={h['n_knots']} "
                f"eval_ap={h.get('eval_ap')} --"
            )
        elif "tau" in h:
            print(  # stdout-compat
                f"-- head {aid}: tau={h['tau']} "
                f"cal_reward={h['cal_reward']} loss={h['loss']} "
                f"eval_ap={h.get('eval_ap')} --"
            )
        elif "kappa" in h:
            print(  # stdout-compat
                f"-- head {aid}: k*={h['kappa']} "
                f"train_reward={h['train_reward']} "
                f"eval_ap={h.get('eval_ap')} --"
            )
        else:
            print(  # stdout-compat
                f"-- vreg {aid}: rmse={h['train_rmse']} "
                f"eval_ap={h.get('eval_ap')} --"
            )
        for f, p in h.get("feature_probe", {}).items():
            print(  # stdout-compat
                f"    probe {f:<22} solo={p['solo_ap']:.4f} "
                f"solo:inc={p['solo_ap_inc']:.4f} "
                f"loo={p['loo_ap']:.4f} d={p['loo_delta']:+.4f}"
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
    """Run the evidence-gate measurement and print the table."""
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
                "scorer,params}}; round-11 replays ride "
                "sp._mk_gate verbatim"
            ),
            "kappa": "bake-time fit on train drains (prefix-sum)",
            "corpus": (
                "train-page verdict counts keyed by "
                "(form, op-spine) signature; train drains "
                "jackknifed, eval rows/play read the full table; "
                "+vd-ep arms run with the seam cleared"
            ),
        },
        "results": res,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
