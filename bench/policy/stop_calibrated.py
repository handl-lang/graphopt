"""The stopping policy, round 8 — calibrate the predictor first.

``docs/results_stop_valuelabel.md`` (round 7) priced the candidate
value *labels* as expected yield and isolated the blocker precisely:
the ``open:vlabel`` columns carry real signal, but the verdict-label
nets are **miscalibrated in magnitude on held-out pages** — they
under-predict (the balanced ``pos_weight`` fit memorizes its few
in-sample positives and transfers almost nothing), so the analytic
break-even gates ``ev-margin``/``ev-pool`` (κ = ``|step|`` = 0.002)
execute the wrong operating point: they quit at 3-9 evals and lose
0.09-0.19 reward.  The blocker is *label calibration*, not the
threshold and not the features.

This round prices exactly that: fit a calibrator on the predictor's
scores — **Platt scaling** (logistic on the score's logit,
``p̂ = sigmoid(a·logit(s) + b)``, :class:`bench.common.calibrate.PlattCal`)
and **isotonic regression** (pool-adjacent violators,
:class:`~bench.common.calibrate.IsotonicCal`) — then re-run the
break-even gates on ``p̂`` with κ unchanged, and re-run round 7's
best prior arms (``head+vlabel``, ``ac+vlabel``) under the
calibrated injection.

The split is the same train/held-out wall as every prior round, now
with a third layer: the calibration set is the *train pages'* own
predictions — which are already out-of-fold (``stop_advantage``'s
``OOF_FOLDS`` protocol: each train page scored by a net that never
saw it), so the calibrator fits held-out-quality pairs — and is
*evaluated* on the odd pages.  No eval-page number enters any fit.

The ``raw`` gate rows replicate round 7's miscalibrated break-even
as the port control (``ev`` should land ~0.126/9.06 and ~0.094/7.33
evals); the κ sweep (0.5x-2x ``|step|``) draws the calibrated
curve's shape around the analytic break-even.

Bake rule (unchanged): a delivery joins
``OPEN_GAME_PLAYER_WEIGHTS`` only if held-out reward/ep AND
page-aware AP both clear the incumbent bake (0.273 / 0.210) —
never-stop or AP-collapse is an honest negative.

Runtime: ~150 s one-time pool memoization, ~15-30 s predictors +
calibration fits, gates ~1 s each, trained arms ~5-30 s each.

Usage::

    timeout 1800 .venv/bin/python bench/policy/stop_calibrated.py \
        --json /tmp/cal.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import policy_baseline as pb
import schema_leaderboard as sl
import stop_advantage as sa
import stop_policy as sp
import stop_valuelabel as svl
from mlopt_discovery import advantage as adv
from mlopt_discovery import engine, play
from mlopt_discovery import open_game as og
from mlopt_discovery import open_self as osel
from mlopt_discovery import open_vlabel as ovl
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import DomainSpec, _dot

from bench.common.calibrate import (
    IsotonicCal,
    PlattCal,
    calibration_metrics,
)

#: Defaults — identical protocol to ``stop_valuelabel.py``.
SEED = svl.SEED
PAGE = svl.PAGE
OPEN_PAGES = svl.OPEN_PAGES
EPOCHS = svl.EPOCHS
V_LR = svl.V_LR
V_PASSES = svl.V_PASSES
_SAMPLED_SEED = svl._SAMPLED_SEED
_GREEDY_SEED = svl._GREEDY_SEED

#: The calibrators under test — both fit on the train pages' OOF
#: predictions only, both judged on the held-out pages.
CALIBRATORS: dict[str, Any] = {
    "platt": PlattCal,
    "isotonic": IsotonicCal,
}

#: The injected modes the gates price — ``ev`` is the two-net
#: expected-tier composite (round 7's default); ``useful``/``pays``/
#: ``delta`` the single-label injections (``delta`` is the control:
#: its raw gate already earned in round 7).
MODES: tuple[str, ...] = ("ev", "useful", "pays", "delta")

#: The κ sweep around the analytic break-even ``|step|`` — cheap
#: because verdicts are memoized.
KAPPA_MULTS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

#: The gate rule names — ``stop_valuelabel``'s ``evm``/``evp``.
GATE_MODES: tuple[str, ...] = ("margin", "pool")

#: The trained arms re-run under each calibrated injection — round
#: 7's earning ``head+vlabel`` gate arm, its ``open:stop`` sibling,
#: the flat-RL ``vlabel`` and the per-step-advantage twin
#: ``ac+vlabel``.  ``(id, part tags, kind)``; head arms hold the
#: ranker at the incumbent schema (``rank_tags=()``) as in rounds 4-7.
TRAIN_ARMS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("vlabel", ("open:vlabel",), "rl"),
    ("ac+vlabel", ("open:vlabel",), "ac"),
    ("head+vlabel", ("open:vlabel",), "head"),
    ("head+stop+vlabel", ("open:stop", "open:vlabel"), "head"),
)


def _scores_for(cands: list, preds: dict) -> list[float]:
    """Predicted scores for *cands* — reads ONLY those keys.

    The no-peek seam the calibration fit is built on: the caller
    passes the *train* candidates, so the calibrator's view is the
    train pages' own (out-of-fold) predictions; held-out keys are
    never touched.
    """
    return [preds.get(og._value_key(c), 0.0) for c in cands]


def _calibration_frame(
    cands: list, outcomes: list, preds: dict, label_fn: Any
) -> tuple[list[float], list[float]]:
    """``(scores, labels)`` for the calibration fit — train rows only."""
    return (
        _scores_for(cands, preds),
        [float(label_fn(o)) for o in outcomes],
    )


def _fit_calibrators(
    train_scores: dict[str, list[float]],
    train_labels: dict[str, list[float]],
    methods: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """``{method: {label: fitted calibrator}}`` — train arrays only."""
    return {
        m: {
            lname: CALIBRATORS[m]().fit(s, train_labels[lname])
            for lname, s in train_scores.items()
        }
        for m in methods
    }


def _apply(cal: Any, preds: dict) -> dict:
    """Map a whole ``{key: raw}`` table through the calibrator."""
    keys = list(preds)
    return dict(
        zip(keys, cal.predict([preds[k] for k in keys]), strict=True)
    )


def _injection(mode: str, preds_by_label: dict[str, dict]) -> dict:
    """The ``set_value_preds`` table for *mode* — raw or calibrated.

    The same payoff conversion round 7 applied to the raw sigmoids:
    ``useful``/``pays``/``delta`` scale their single net's p (or p̂)
    by the tier unit; ``ev`` composes the expected tier payoff
    ``p_u·1.0 + max(p_p - p_u, 0)·0.2`` — now over *calibrated* p̂.
    """
    if mode == "ev":
        pu, pp = preds_by_label["useful"], preds_by_label["pays"]
        return {
            k: pu.get(k, 0.0) * og.OPEN_GAME_REWARD["useful"]
            + max(pp.get(k, 0.0) - pu.get(k, 0.0), 0.0)
            * og.OPEN_GAME_REWARD["true"]
            for k in set(pu) | set(pp)
        }
    return {
        k: v * svl.PAYOFF_UNIT[mode]
        for k, v in preds_by_label[mode].items()
    }


def _play_gates(
    ranker: Any,
    rank_spec: DomainSpec,
    ev_pools: list[list],
    budget: int,
    seed: int,
    rank_ap: float,
    value_mode: str,
    cal_tag: str,
    kappa_mults: tuple[float, ...],
) -> list[dict]:
    """Play the break-even gates at every κ — under the bound table.

    The gate reads ``og._value`` at decision time, so the caller's
    ``set_value_preds`` binding is what this prices; ``cal_tag``
    records which table (``raw`` or a calibrator name) was bound.
    """
    rows: list[dict] = []
    for gm in GATE_MODES:
        for mult in kappa_mults:
            kappa = ovl.EVAL_COST * mult
            sampled = sp._play_pools(
                lambda i, r=ranker, rs=rank_spec, k=kappa, m=gm: (
                    svl._mk_ev_gate(
                        r, rs, k, m, seed + _SAMPLED_SEED + i
                    )
                ),
                ev_pools,
                budget,
            )
            greedy = sp._play_pools(
                lambda i, r=ranker, rs=rank_spec, k=kappa, m=gm: (
                    svl._mk_ev_gate(
                        r,
                        rs,
                        k,
                        m,
                        seed + _GREEDY_SEED + i,
                        greedy=True,
                    )
                ),
                ev_pools,
                budget,
            )
            rows.append(
                {
                    "id": f"{value_mode}-{gm}:{cal_tag}@{mult:g}x",
                    "kind": "evgate",
                    "value_mode": value_mode,
                    "calibrator": cal_tag,
                    "gate_mode": gm,
                    "kappa_mult": mult,
                    "kappa": round(kappa, 4),
                    "reward_ep": sampled["reward"],
                    "reward_greedy_ep": greedy["reward"],
                    "useful_ep": sampled["useful_per_ep"],
                    "useful_eval": sampled["useful_per_eval"],
                    "evals_ep": sampled["evals_per_ep"],
                    "stops": sampled["stops"],
                    "stops_greedy": greedy["stops"],
                    "ap": rank_ap,
                }
            )
            print(  # stdout-compat
                f"    gate {value_mode}-{gm}:{cal_tag}@{mult:g}x "
                f"reward={sampled['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"stops={sampled['stops']:>3}"
            )
    return rows


def run(
    seed: int,
    half: int,
    page: int,
    budget: int,
    epochs: int,
    modes: tuple[str, ...] = MODES,
    cal_names: tuple[str, ...] = tuple(CALIBRATORS),
    kappa_mults: tuple[float, ...] = KAPPA_MULTS,
) -> dict[str, Any]:
    """Measure pools once; fit nets + calibrators; eval held-out."""
    t0 = time.time()
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    eval_outcomes, by_key = pb._measure_pools(
        ev_pools, seed + pb.HELD_OUT_OFFSET
    )
    train_outcomes, train_keys = pb._measure_pools(
        train_pools, seed + 1_000_003
    )
    by_key.update(train_keys)

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1
    measure_s = time.time() - t0

    # The predictors — round 7's protocol bit-for-bit (full-table net
    # per label + OOF train preds).  ``preds`` for a train key is the
    # OOF score — the held-out-quality calibration input.
    t_net = time.time()
    flat_train = [c for pool in train_pools for c in pool]
    flat_train_outcomes = [o for outs in train_outcomes for o in outs]
    all_cands = flat_train + flat_pool
    preds_by_label: dict[str, dict] = {}
    predictor: dict[str, Any] = {"nets": {}}
    for lname in sp.VALUE_LABELS:
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
    flat_ordered = [flat_outcomes[i] for i in range(len(flat_pool))]
    ev_labels = {
        n: [bool(lf(o)) for o in flat_ordered]
        for n, lf in sp.VALUE_LABELS.items()
    }
    for lname, preds in preds_by_label.items():
        scores = _scores_for(flat_pool, preds)
        predictor["nets"][lname].update(
            {
                f"ap_{n}": round(sp._ap_binary(scores, ev_labels[n]), 4)
                for n in sp.VALUE_LABELS
            }
        )
    predictor["seconds"] = round(time.time() - t_net, 1)

    # The calibration stage — the only new machinery.  Fit on the
    # TRAIN pages' predictions (already OOF — each train page scored
    # by a net that never saw it) vs the train verdict labels; the
    # held-out pages enter only as the *evaluation* of the fitted
    # maps.  ``_scores_for``/``_fit_calibrators`` take train rows only
    # — the no-peek wall is structural, not procedural.
    t_cal = time.time()
    train_scores: dict[str, list[float]] = {}
    train_labels: dict[str, list[float]] = {}
    ev_scores: dict[str, list[float]] = {}
    for lname, lf in sp.VALUE_LABELS.items():
        s_tr, y_tr = _calibration_frame(
            flat_train, flat_train_outcomes, preds_by_label[lname], lf
        )
        train_scores[lname] = s_tr
        train_labels[lname] = y_tr
        ev_scores[lname] = _scores_for(flat_pool, preds_by_label[lname])
    cals = _fit_calibrators(train_scores, train_labels, cal_names)
    cal_report: dict[str, dict] = {}
    for lname in sp.VALUE_LABELS:
        y_ev = [float(v) for v in ev_labels[lname]]
        s_ev = ev_scores[lname]
        entry: dict[str, Any] = {
            "raw": {
                "held_out": {
                    **calibration_metrics(s_ev, y_ev),
                    "ap": round(
                        sp._ap_binary(s_ev, ev_labels[lname]), 4
                    ),
                }
            }
        }
        for m in cal_names:
            cal = cals[m][lname]
            s_hat_tr = cal.predict(train_scores[lname])
            s_hat_ev = cal.predict(s_ev)
            params: dict[str, Any] = {}
            if isinstance(cal, PlattCal):
                params = {"a": round(cal.a_, 4), "b": round(cal.b_, 4)}
            elif isinstance(cal, IsotonicCal):
                params = {"blocks": len(cal.blocks_)}
            entry[m] = {
                "params": params,
                "train": calibration_metrics(
                    s_hat_tr, train_labels[lname]
                ),
                "held_out": {
                    **calibration_metrics(s_hat_ev, y_ev),
                    "ap": round(
                        sp._ap_binary(s_hat_ev, ev_labels[lname]), 4
                    ),
                },
            }
        cal_report[lname] = entry
    predictor["calibration_seconds"] = round(time.time() - t_cal, 2)

    # Pre-compute each calibrator's full prediction table once (keys
    # are candidate-structural — the same map serves gates, features
    # and head rows).
    cal_tables: dict[str, dict[str, dict]] = {
        m: {
            lname: _apply(cals[m][lname], preds_by_label[lname])
            for lname in sp.VALUE_LABELS
        }
        for m in cal_names
    }

    rows_out: list[dict] = []
    gate_rows: list[dict] = []
    curves: dict[str, list[float]] = {}
    weights: dict[str, dict] = {}
    heads: dict[str, dict] = {}
    vdiag: dict[str, dict] = {}
    orig_evaluate = lp.evaluate
    lp.evaluate = pb._patched_evaluate(by_key)
    osel.set_self_scores(None)
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
                        sp._page_scores(
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
                    pb._ap([0.0] * len(flat_pool), flat_outcomes),
                    4,
                ),
            },
        }

        # One incumbent-schema ranker serves every gate and every head
        # arm — the same cold-RL recipe round 7's evm/evp/head rows
        # trained (identical seed, identical order: the weights are
        # the same object either way, so training it once is exact,
        # not approximate).
        inc_spec = svl._spec_for(())
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
        weights["incumbent"] = {
            n: round(w, 4) for n, w in w_inc.items() if abs(w) > 1e-4
        }
        og.set_value_preds(None)  # incumbent schema never reads it
        inc_sampled = sp._play_pools(
            lambda i, p=inc_player: p.frozen(seed + _SAMPLED_SEED + i),
            ev_pools,
            budget,
        )
        inc_greedy = sp._play_pools(
            lambda i, p=inc_player: p.frozen(
                seed + _GREEDY_SEED + i, greedy=True
            ),
            ev_pools,
            budget,
        )
        rows_out.append(
            {
                "id": "incumbent",
                "kind": "rl",
                "value_mode": None,
                "calibrator": None,
                "extra_parts": [],
                "rl_epochs": epochs,
                "reward_ep": inc_sampled["reward"],
                "reward_greedy_ep": inc_greedy["reward"],
                "useful_ep": inc_sampled["useful_per_ep"],
                "useful_eval": inc_sampled["useful_per_eval"],
                "evals_ep": inc_sampled["evals_per_ep"],
                "stops": inc_sampled["stops"],
                "stops_greedy": inc_greedy["stops"],
                "ap": inc_ap,
            }
        )
        print(  # stdout-compat
            f"  [incumbent] reward={inc_sampled['reward']:>8.4f} "
            f"evals={inc_sampled['evals_per_ep']:>5.2f} ap={inc_ap:.4f}"
        )
        # The incumbent-schema ``ac`` control (round 6/7's actor-critic
        # baseline) — binding-free like the incumbent.
        ac_player = adv.AdvantagePlayer(
            seed=seed,
            spec=inc_spec,
            weights={},
            state_feats=lambda s, h, s_=inc_spec: s_.featurize(
                s, Action("stop", {}), h
            ),
            v_lr=V_LR,
            v_passes=V_PASSES,
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
        weights["ac"] = {
            n: round(w, 4) for n, w in w_ac.items() if abs(w) > 1e-4
        }
        vdiag["ac"] = {
            "v_err": round(ac_player._v_err, 4),
            "v_size": len(ac_player._v),
            "v_top": {
                n: round(wv, 4)
                for n, wv in sorted(
                    ac_player._v.items(), key=lambda kv: -abs(kv[1])
                )[:8]
                if abs(wv) > 1e-4
            },
        }
        ac_sampled = sp._play_pools(
            lambda i, p=ac_player: p.frozen(seed + _SAMPLED_SEED + i),
            ev_pools,
            budget,
        )
        ac_greedy = sp._play_pools(
            lambda i, p=ac_player: p.frozen(
                seed + _GREEDY_SEED + i, greedy=True
            ),
            ev_pools,
            budget,
        )
        rows_out.append(
            {
                "id": "ac",
                "kind": "ac",
                "value_mode": None,
                "calibrator": None,
                "extra_parts": [],
                "rl_epochs": epochs,
                "reward_ep": ac_sampled["reward"],
                "reward_greedy_ep": ac_greedy["reward"],
                "useful_ep": ac_sampled["useful_per_ep"],
                "useful_eval": ac_sampled["useful_per_eval"],
                "evals_ep": ac_sampled["evals_per_ep"],
                "stops": ac_sampled["stops"],
                "stops_greedy": ac_greedy["stops"],
                "ap": ac_ap,
            }
        )
        print(  # stdout-compat
            f"  [ac] reward={ac_sampled['reward']:>8.4f} "
            f"evals={ac_sampled['evals_per_ep']:>5.2f} ap={ac_ap:.4f}"
        )

        for mode in modes:
            # Raw-table gates — the round-7 port control (identical
            # ranker, identical seeds → the raw κ=1x rows should
            # reproduce round 7's miscalibrated break-evens).
            og.set_value_preds(_injection(mode, preds_by_label))
            gate_rows.extend(
                _play_gates(
                    inc_player,
                    inc_spec,
                    ev_pools,
                    budget,
                    seed,
                    inc_ap,
                    mode,
                    "raw",
                    kappa_mults,
                )
            )
            for m in cal_names:
                og.set_value_preds(_injection(mode, cal_tables[m]))
                gate_rows.extend(
                    _play_gates(
                        inc_player,
                        inc_spec,
                        ev_pools,
                        budget,
                        seed,
                        inc_ap,
                        mode,
                        m,
                        kappa_mults,
                    )
                )
                # The trained arms under this (mode, calibrator)
                # injection — the vlabel features the stop row reads
                # are now p̂-priced.
                for arm_id, tags, kind in TRAIN_ARMS:
                    t1 = time.time()
                    key = f"{arm_id}:{mode}/{m}"
                    spec = svl._spec_for(tags)
                    # ``head`` arms hold the *ranker* at the incumbent
                    # schema (``rank_tags=()`` — the round-4 protocol);
                    # ``rl``/``ac`` arms train on their own schema.
                    # The head ranker is the incumbent player itself —
                    # same seed, same recipe, identical weights.
                    rank_spec = inc_spec if kind == "head" else spec
                    if kind == "head":
                        player = inc_player
                        curve = curves["incumbent"]
                    elif kind == "ac":
                        player = adv.AdvantagePlayer(
                            seed=seed,
                            spec=rank_spec,
                            weights={},
                            state_feats=lambda s, h, s_=rank_spec: (
                                s_.featurize(s, Action("stop", {}), h)
                            ),
                            v_lr=V_LR,
                            v_passes=V_PASSES,
                        )
                        curve = []
                    else:
                        player = og.OpenLearnedPlayer(
                            seed=seed, spec=rank_spec, weights={}
                        )
                        curve = []
                    for _ in range(0 if kind == "head" else epochs):
                        totals = engine.train_policy(
                            player,
                            train_pools,
                            sl._FixedArena,
                            budget=budget,
                        )
                        curve.append(
                            round(sum(totals) / max(len(totals), 1), 4)
                        )
                    w_rank = player.weights_dict()
                    curves[key] = curve
                    weights[key] = {
                        n: round(w, 4)
                        for n, w in w_rank.items()
                        if abs(w) > 1e-4
                    }
                    fold_sampled = fold_greedy = None
                    if kind == "head":
                        hrows, hlabels = sp._collect_head_rows(
                            spec,
                            w_rank,
                            train_pools,
                            by_key,
                            budget,
                            rank_spec=rank_spec,
                        )
                        erows, elabels = sp._collect_head_rows(
                            spec,
                            w_rank,
                            ev_pools,
                            by_key,
                            budget,
                            rank_spec=rank_spec,
                        )
                        w_head, hloss = sp._fit_head(
                            spec, hrows, hlabels, seed
                        )
                        eval_ap = sp._ap_binary(
                            [_dot(w_head, r) for r in erows], elabels
                        )
                        probe: dict[str, dict] = {}
                        for f in svl.PROBE_FEATURES:
                            if f not in spec.features:
                                continue
                            solo = sp._ap_binary(
                                [r.get(f, 0.0) for r in erows], elabels
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
                                "loo_ap": round(loo, 4),
                                "loo_delta": round(loo - eval_ap, 4),
                            }
                        tau, cal_reward = sp._calibrate_tau(
                            spec,
                            player,
                            w_head,
                            hrows,
                            train_pools,
                            budget,
                            seed,
                            rank_spec=rank_spec,
                        )
                        heads[key] = {
                            "tau": round(tau, 4),
                            "cal_reward": cal_reward,
                            "loss": hloss,
                            "n_rows": len(hrows),
                            "pos_rate": round(
                                sum(hlabels) / max(len(hlabels), 1), 4
                            ),
                            "eval_ap": round(eval_ap, 4),
                            "eval_pos_rate": round(
                                sum(elabels) / max(len(elabels), 1), 4
                            ),
                            "feature_probe": probe,
                            "weights": {
                                n: round(wv, 4)
                                for n, wv in w_head.items()
                                if abs(wv) > 1e-4
                            },
                        }
                        sampled = sp._play_pools(
                            lambda i, p=player, s=spec, wh=w_head, ta=tau, rs=rank_spec: (
                                sp._mk_gate(
                                    p,
                                    s,
                                    wh,
                                    ta,
                                    seed + _SAMPLED_SEED + i,
                                    rank_spec=rs,
                                )
                            ),
                            ev_pools,
                            budget,
                        )
                        greedy = sp._play_pools(
                            lambda i, p=player, s=spec, wh=w_head, ta=tau, rs=rank_spec: (
                                sp._mk_gate(
                                    p,
                                    s,
                                    wh,
                                    ta,
                                    seed + _GREEDY_SEED + i,
                                    greedy=True,
                                    rank_spec=rs,
                                )
                            ),
                            ev_pools,
                            budget,
                        )
                        fold_c, fold_cal = sp._calibrate_fold_c(
                            spec,
                            w_rank,
                            w_head,
                            train_pools,
                            budget,
                            seed,
                        )
                        w_fold = sp._fold_weights(
                            spec, w_rank, w_head, fold_c
                        )
                        weights[key + "-fold"] = {
                            n: round(wv, 4)
                            for n, wv in w_fold.items()
                            if abs(wv) > 1e-4
                        }
                        heads[key]["fold"] = {
                            "c": round(fold_c, 4),
                            "cal_reward": fold_cal,
                        }
                        fold_sampled = sp._play_pools(
                            lambda i, s=spec, wf=w_fold: (
                                og.OpenLearnedPlayer(
                                    seed=seed + _SAMPLED_SEED + i,
                                    spec=s,
                                    weights=wf,
                                    learn=False,
                                )
                            ),
                            ev_pools,
                            budget,
                        )
                        fold_greedy = sp._play_pools(
                            lambda i, s=spec, wf=w_fold: (
                                og.OpenLearnedPlayer(
                                    seed=seed + _GREEDY_SEED + i,
                                    spec=s,
                                    weights=wf,
                                    learn=False,
                                    greedy=True,
                                )
                            ),
                            ev_pools,
                            budget,
                        )
                    else:
                        sampled = sp._play_pools(
                            lambda i, p=player: p.frozen(
                                seed + _SAMPLED_SEED + i
                            ),
                            ev_pools,
                            budget,
                        )
                        greedy = sp._play_pools(
                            lambda i, p=player: p.frozen(
                                seed + _GREEDY_SEED + i, greedy=True
                            ),
                            ev_pools,
                            budget,
                        )
                    ap = round(
                        pb._ap(
                            sp._page_scores(
                                rank_spec, w_rank, ev_pools
                            ),
                            flat_outcomes,
                        ),
                        4,
                    )
                    rows_out.append(
                        {
                            "id": key,
                            "kind": kind,
                            "value_mode": mode,
                            "calibrator": m,
                            "extra_parts": list(tags),
                            "rl_epochs": epochs,
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
                    if kind == "ac":
                        vdiag[key] = {
                            "v_err": round(player._v_err, 4),
                            "v_size": len(player._v),
                            "v_top": {
                                n: round(wv, 4)
                                for n, wv in sorted(
                                    player._v.items(),
                                    key=lambda kv: -abs(kv[1]),
                                )[:8]
                                if abs(wv) > 1e-4
                            },
                        }
                    if (
                        fold_sampled is not None
                        and fold_greedy is not None
                    ):
                        rows_out.append(
                            {
                                "id": key + "-fold",
                                "kind": "fold",
                                "value_mode": mode,
                                "calibrator": m,
                                "extra_parts": list(tags),
                                "rl_epochs": epochs,
                                "reward_ep": fold_sampled["reward"],
                                "reward_greedy_ep": fold_greedy[
                                    "reward"
                                ],
                                "useful_ep": fold_sampled[
                                    "useful_per_ep"
                                ],
                                "useful_eval": fold_sampled[
                                    "useful_per_eval"
                                ],
                                "evals_ep": fold_sampled[
                                    "evals_per_ep"
                                ],
                                "stops": fold_sampled["stops"],
                                "stops_greedy": fold_greedy["stops"],
                                "ap": ap,
                            }
                        )
                    print(  # stdout-compat
                        f"  [{key:<30}] reward={sampled['reward']:>8.4f} "
                        f"greedy={greedy['reward']:>8.4f} "
                        f"evals={sampled['evals_per_ep']:>5.2f} "
                        f"ap={ap:.4f} ({time.time() - t1:.0f}s)"
                    )
    finally:
        lp.evaluate = orig_evaluate
        og.set_value_preds(None)
        osel.set_self_scores(None)

    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": rows_out,
        "gate_rows": gate_rows,
        "anchors": anchors,
        "train_curve": curves,
        "weights": weights,
        "heads": heads,
        "vdiag": vdiag,
        "predictor": predictor,
        "calibration": cal_report,
        "pool": {"held_out": len(flat_pool), "useful": n_useful},
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "epochs": epochs,
            "modes": list(modes),
            "calibrators": list(cal_names),
            "kappa_mults": list(kappa_mults),
            "eval_cost": ovl.EVAL_COST,
            "v_lr": V_LR,
            "v_passes": V_PASSES,
            "neural": {
                "epochs": sa.NEURAL_EPOCHS,
                "hidden": sa.NEURAL_HIDDEN,
                "oof_folds": sa.OOF_FOLDS,
            },
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the calibration + gate + arm tables."""
    pred = res.get("predictor")
    if pred:
        print(  # stdout-compat
            f"== value predictors (OOF {sa.OOF_FOLDS} folds, "
            f"{pred['seconds']}s) =="
        )
        for lname, pr in pred["nets"].items():
            aps = " ".join(
                f"{n}={pr[f'ap_{n}']:.4f}" for n in sp.VALUE_LABELS
            )
            print(  # stdout-compat
                f"  {lname:<7} pos={pr['train_positives']:>3} "
                f"pw={pr['pos_weight']:>6.1f} AP[{aps}]"
            )
    cal = res.get("calibration", {})
    if cal:
        print(  # stdout-compat
            "== calibration — held-out odd pages (fit on train OOF) =="
        )
        for lname, entry in cal.items():
            for tag in ("raw", *CALIBRATORS):
                if tag not in entry:
                    continue
                ho = entry[tag]["held_out"]
                print(  # stdout-compat
                    f"  {lname:<7} {tag:<9} ll={ho['log_loss']:>7.4f} "
                    f"brier={ho['brier']:>7.4f} ece={ho['ece']:>6.4f} "
                    f"slope={ho['slope']} int={ho['intercept']} "
                    f"mpred={ho['mean_pred']} pos={ho['pos_rate']} "
                    f"ap={ho['ap']}"
                )
    print(
        "== break-even gates — held-out odd pages =="
    )  # stdout-compat
    head = (
        f"{'gate':<26} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6}"
    )
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for r in res["gate_rows"]:
        print(  # stdout-compat
            f"{r['id']:<26} {r['reward_ep']:>8.4f} "
            f"{r['reward_greedy_ep']:>8.4f} {r['useful_eval']:>9.4f} "
            f"{r['evals_ep']:>8.3f} {r['stops']:>6}"
        )
    print("== trained arms — held-out odd pages ==")  # stdout-compat
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
    for arm_id, h in res.get("heads", {}).items():
        print(  # stdout-compat
            f"-- head {arm_id}: tau={h['tau']} "
            f"cal_reward={h['cal_reward']} pos_rate={h['pos_rate']} "
            f"loss={h['loss']} eval_ap={h.get('eval_ap')} "
            f"fold c={h['fold']['c']} cal={h['fold']['cal_reward']} --"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the calibrated stop-policy measurement and print tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument(
        "--modes",
        type=str,
        default=",".join(MODES),
        help="comma list of injected value modes "
        "(ev,useful,pays,delta)",
    )
    ap.add_argument(
        "--calibrators",
        type=str,
        default=",".join(CALIBRATORS),
        help="comma list of calibrators (platt,isotonic)",
    )
    ap.add_argument(
        "--kappas",
        type=str,
        default=",".join(f"{k:g}" for k in KAPPA_MULTS),
        help="comma list of κ multipliers on |step|",
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
        modes=tuple(args.modes.split(",")),
        cal_names=tuple(args.calibrators.split(",")),
        kappa_mults=tuple(float(k) for k in args.kappas.split(",")),
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
            "calibration": (
                "fit on train pages' out-of-fold predictions vs "
                "train verdict labels; evaluated on held-out pages"
            ),
            "modes": args.modes,
            "calibrators": args.calibrators,
            "kappa_mults": args.kappas,
            "arms": [a for a, *_ in TRAIN_ARMS],
        },
        "results": res,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
