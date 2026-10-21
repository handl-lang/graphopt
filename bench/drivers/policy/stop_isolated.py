"""The stopping policy, round 9 — the isolated stop channel.

``docs/results_stop_calibrated.md`` (round 8) and ADR 0013 left the
blocker stated exactly: at calibrated probabilities the break-even
rule *rationally drains* (cost asymmetry), and **no serializable arm
clears both bake bars** (reward >= 0.273 and AP >= 0.210) — while every
feature family routed through the shared ``stop`` logit corrupts the
ranking (five measured AP collapses).  The recurring corruption
channel is the softmax itself.  This round attacks the *delivery*:
the quit decision moves to a **side channel the engine consults
before the policy's own pick** — ``GatedPlayer`` exposes
``forced_action(state)`` (:func:`gamekit.engine.run_episode` calls it
first; the inner ranker binds an eval-only ``legal`` so ``stop``
never enters a softmax row).

Three scorer families ride the channel — every one is *honest
break-even* (κ = ``|step|`` = 0.002 in the injected units), none is
reward-calibrated:

* ``iso-{mode}-{margin|pool}:{cal}`` — the analytic break-even gates
  of rounds 7-8 re-delivered through the hook: continue while the
  best pending eval clears its cost (``margin``) or while draining
  the rest still nets positive (``pool``), over the raw / Platt /
  isotonic injected tables.  The raw κ=1x rows are the port control:
  they must reproduce round 8's ``_EVGate`` numbers bit-identically.
* ``iso-head-*:{platt|isotonic}`` — the supervised continue-head
  (``stop_policy._fit_head`` logistic over the spec's ``stop`` row,
  oracle continue-labels) with its logit *calibrated* to
  ``P(continuation nets > 0)`` on the train pages' OOF decision rows
  (:mod:`gamekit.calibrate`), then thresholded at the
  decision-theoretic ``p* = C^- / (V+ + C^-)``: ``V+`` is the mean
  best-prefix value a correct continue banks, ``C^-`` the mean drain
  cost of a wrong continue (``|step| * pending`` at dead states) —
  both measured on the collected train rows, not tuned by reward.
* ``iso-vreg-*`` — a ridge-regression head fit directly on the
  *best-prefix continuation value* (reward units, already net of
  step costs); stop iff ``w.phi <= 0`` — κ=0 is the literal
  break-even.  Delivered through the linear ``w_stop`` path (the
  scalar ``score`` arms exercise the callable path).

Controls: ``incumbent`` / ``ac`` retrain (port controls — must land
0.2728 / 0.2761), ``oracle`` (the hook carrying the best-prefix
oracle — the 0.3877 ceiling), ``iso-drain`` (channel bound, scorer
None — the never-stop twin), ``iso-always`` (κ=+inf — sanity that
the channel fires).  Every head arm also ships its ``-fold`` twin —
the same head folded into the shared softmax — the channel-isolation
A/B the hypothesis wants.  ``rl-gated`` trains the ranker *with the
channel armed* (episodes truncate where the gate says stop), the
training-time-isolation arm.  Value-reading arms get the honest
``-unfed`` replay (the seam cleared, as a weight bake would deploy).

Bake rule (unchanged): a delivery joins
``OPEN_GAME_PLAYER_WEIGHTS``-shape artifacts only if held-out
reward/ep AND page-aware AP both clear the incumbent bake
(0.273 / 0.210) — never-stop or AP-collapse is an honest negative.

Runtime: ~150 s one-time pool memoization, ~15-30 s predictors +
calibration fits, ~0.5-1.5 s per arm row.

Usage::

    timeout 1800 .venv/bin/python bench/drivers/policy/stop_isolated.py \
        --json /tmp/iso.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import policy_baseline as pb
import schema_leaderboard as sl
import stop_advantage as sa
import stop_calibrated as sc
import stop_policy as sp
import stop_valuelabel as svl
import torch
from mlopt_discovery import play
from mlopt_discovery import open_game as og
from mlopt_discovery import open_self as osel
from mlopt_discovery import open_vlabel as ovl
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from gamekit import advantage as adv
from gamekit import engine
from gamekit.calibrate import IsotonicCal, PlattCal, calibration_metrics
from gamekit.players import GatedPlayer, PlayHist, _dot

#: Defaults — identical protocol to ``stop_calibrated.py``.
SEED = svl.SEED
PAGE = svl.PAGE
OPEN_PAGES = svl.OPEN_PAGES
EPOCHS = svl.EPOCHS
V_LR = svl.V_LR
V_PASSES = svl.V_PASSES
_SAMPLED_SEED = svl._SAMPLED_SEED
_GREEDY_SEED = svl._GREEDY_SEED

#: The injected payoff tables the gates price, and the calibrators
#: mapping each label net's OOF scores to ``p̂`` — round 8's set.
MODES: tuple[str, ...] = sc.MODES
CALS: tuple[str, ...] = ("raw", "platt", "isotonic")
KAPPA_MULTS: tuple[float, ...] = sc.KAPPA_MULTS
GATE_RULES: tuple[str, ...] = ("margin", "pool")

#: The per-eval cost — the honest break-even in payoff units.
EVAL_COST: float = ovl.EVAL_COST

#: The head-arm schemas — ``(id, stop-spec part tags, (mode, cal) of
#: the bound injection or None)``.  ``open:vlabel`` arms read the
#: injected table; the binding is bound for collection AND play.
HEAD_ARMS: tuple[tuple[str, tuple[str, ...], tuple[str, str] | None], ...] = (
    ("head", (), None),
    ("head+stop", ("open:stop",), None),
    ("head+vl:delta/raw", ("open:vlabel",), ("delta", "raw")),
    ("head+vl:useful/iso", ("open:vlabel",), ("useful", "isotonic")),
    ("head+vl:ev/platt", ("open:vlabel",), ("ev", "platt")),
)

#: The head-score calibrators — logistic head → sigmoid → ``p̂``.
HEAD_CALS: dict[str, Any] = {"platt": PlattCal, "isotonic": IsotonicCal}


def _sig(z: float) -> float:
    """Numerically tame sigmoid — head logits into (0, 1) for Platt."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
    return math.exp(z) / (1.0 + math.exp(z))


def _stop_feats_of(spec: Any) -> Any:
    """``(state, hist) -> dict`` — *spec*'s row for the stop move."""

    def row(state: Any, hist: Any) -> dict:
        stop = next(
            a for a in og.legal_actions(state) if a.op == "stop"
        )
        return spec.featurize(state, stop, hist)

    return row


def _mk_iso(
    w_rank: dict,
    rank_spec: Any,
    seed: int,
    *,
    greedy: bool = False,
    stop_feats: Any = None,
    w_stop: dict | None = None,
    kappa: float = 0.0,
    score: Any = None,
) -> GatedPlayer:
    """Build one isolated-channel player: eval-only ranker + the hook."""
    inner = og.OpenLearnedPlayer(
        seed=seed,
        spec=rank_spec,
        weights=w_rank,
        legal=sp._eval_only,
        learn=False,
        greedy=greedy,
    )
    return GatedPlayer(
        inner,
        legal=og.legal_actions,
        stop_feats=stop_feats,
        w_stop=w_stop,
        kappa=kappa,
        score=score,
    )


def _ev_score(rule: str, mult: float) -> Any:
    """The analytic break-even as a ``(state, hist) -> score`` callable.

    ``pool``: expected yield of draining the rest, ``Σ_pending(v - κ)``;
    ``margin``: the next eval's break-even margin, ``max v - κ``.  The
    hook stops when the score drops to 0 — identical to
    ``stop_valuelabel._EVGate`` (continue iff ``> 0``), so the raw
    rows are the delivery-exactness port control.
    """
    kappa = EVAL_COST * mult

    def score(state: Any, _hist: Any) -> float:
        vals = [og._value(state.pool[i]) for i in state.pending]
        if not vals:
            return -math.inf
        if rule == "pool":
            return sum(v - kappa for v in vals)
        return max(vals) - kappa

    return score


def _oracle_score(by_key: dict, rank_spec: Any, w_rank: dict) -> Any:
    """The best-prefix oracle as a side-channel score (upper bound)."""

    def score(state: Any, _hist: Any) -> float:
        if not state.pending:
            return -math.inf
        order = sp._eval_order(rank_spec, w_rank, state)
        return sp._prefix_value(state, order, by_key)

    return score


def _collect_rows_v(
    spec: Any,
    w: dict,
    pools: list[list],
    by_key: dict,
    budget: int,
    rank_spec: Any,
) -> tuple[list[dict], list[bool], list[float], list[int]]:
    """``(stop_row, label, prefix_value, n_pending)`` along greedy drains.

    ``stop_policy._collect_head_rows`` plus the two row-level
    quantities the honest κ needs: the oracle best-prefix *value*
    (the vreg head's regression target, and ``V+``'s sample) and the
    pending count (``C^-``'s drain length).  Labels are
    ``prefix_value > 0`` — the oracle's own continue call.
    """
    rows: list[dict] = []
    labels: list[bool] = []
    vals: list[float] = []
    pend: list[int] = []
    for pool in pools:
        arena = sl._FixedArena(pool)
        hist = PlayHist()
        for _ in range(budget):
            state = arena.observe()
            if state.done or not state.pending:
                break
            rows.append(spec.featurize(state, Action("stop", {}), hist))
            order = sp._eval_order(rank_spec, w, state)
            pv = sp._prefix_value(state, order, by_key)
            labels.append(pv > 0.0)
            vals.append(pv)
            pend.append(len(state.pending))
            i = order[0]
            act = Action("evaluate", {"i": i, "cand": state.pool[i]})
            arena.step(act)
            hist.record(act.op)
    return rows, labels, vals, pend


def _fit_value_head(
    rows: list[dict],
    vals: list[float],
    names: tuple[str, ...],
    lam: float = 1e-2,
) -> tuple[dict[str, float], float]:
    """Ridge regression ``w.phi ~= best-prefix value`` — reward units.

    Same standardize-then-map-back shape as ``sp._fit_head``: the
    closed form solves on standardized columns (``bias`` excluded —
    the appended intercept column owns it), then maps to raw feature
    units so ``_dot`` scores the row directly and ``w["bias"]`` is the
    intercept.  Returns ``(w, rmse)``.
    """
    names = [n for n in names if n != "bias"]
    x = torch.tensor(
        [[r.get(n, 0.0) for n in names] for r in rows],
        dtype=torch.float64,
    )
    y = torch.tensor(vals, dtype=torch.float64)
    mu0, sd0 = x.mean(0), x.std(0, unbiased=False)
    const = sd0 < 1e-9
    mu = torch.where(const, torch.zeros_like(mu0), mu0)
    sd = torch.where(const, torch.ones_like(sd0), sd0)
    xz = (x - mu) / sd
    xt = torch.cat(
        [xz, torch.ones(len(rows), 1, dtype=torch.float64)], dim=1
    )
    reg = torch.eye(len(names) + 1, dtype=torch.float64) * lam
    reg[-1, -1] = 0.0
    sol = torch.linalg.solve(xt.T @ xt + reg, xt.T @ y)
    w = {n: float(sol[i]) / float(sd[i]) for i, n in enumerate(names)}
    w["bias"] = float(sol[-1]) - sum(
        float(sol[i]) * float(mu[i]) / float(sd[i])
        for i in range(len(names))
    )
    pred = xt @ sol
    rmse = float(torch.sqrt(((pred - y) ** 2).mean()))
    return (
        {n: wv for n, wv in w.items() if abs(wv) > 1e-9},
        round(rmse, 4),
    )


def _breakeven_p(
    vals: list[float], labels: list[bool], pend: list[int]
) -> tuple[float, float, float]:
    """The honest head threshold ``p* = C^-/(V+ + C^-)``.

    ``V+`` = mean oracle best-prefix value on continue-labelled rows —
    what a right continue banks.  ``C^-`` = ``|step| * mean pending``
    on stop-labelled rows — what a wrong continue spends draining a
    dead board (upper bound: the gate may stop earlier).  The rule
    ``continue iff p̂ >= p*`` prices the asymmetric table from the
    collected train rows — not a reward-max search.
    """
    pos = [v for v, lab in zip(vals, labels, strict=True) if lab]
    neg = [n for n, lab in zip(pend, labels, strict=True) if not lab]
    vbar = fmean(pos) if pos else 0.0
    cbar = EVAL_COST * (fmean(neg) if neg else 0.0)
    pstar = cbar / (vbar + cbar) if vbar + cbar > 0 else 1.0
    return pstar, round(vbar, 4), round(cbar, 4)


def _cal_score(cal: Any, w_head: dict, feats: Any) -> Any:
    """``p̂(continue pays)`` — the calibrated head as a score callable."""

    def score(state: Any, hist: Any) -> float:
        return cal.predict([_sig(_dot(w_head, feats(state, hist)))])[0]

    return score


def _injection_for(
    mode: str, cal_name: str, preds_by_label: dict, cal_tables: dict
) -> dict:
    """The ``set_value_preds`` table for ``(mode, cal)`` — r8's shape."""
    src = preds_by_label if cal_name == "raw" else cal_tables[cal_name]
    return sc._injection(mode, src)


def run(
    seed: int,
    half: int,
    page: int,
    budget: int,
    epochs: int,
    modes: tuple[str, ...] = MODES,
    cal_names: tuple[str, ...] = CALS,
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

    # The predictors — round 7/8's protocol bit-for-bit (full-table
    # net per label + OOF train preds), then the calibrator per label
    # fit on the train pages' OOF scores only.
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
        scores = sc._scores_for(flat_pool, preds)
        predictor["nets"][lname].update(
            {
                f"ap_{n}": round(sp._ap_binary(scores, ev_labels[n]), 4)
                for n in sp.VALUE_LABELS
            }
        )
    train_scores: dict[str, list[float]] = {}
    train_labels: dict[str, list[float]] = {}
    for lname, lf in sp.VALUE_LABELS.items():
        s_tr, y_tr = sc._calibration_frame(
            flat_train, flat_train_outcomes, preds_by_label[lname], lf
        )
        train_scores[lname] = s_tr
        train_labels[lname] = y_tr
    cals = sc._fit_calibrators(
        train_scores, train_labels, tuple(c for c in cal_names if c != "raw")
    )
    cal_tables: dict[str, dict[str, dict]] = {
        m: {
            lname: sc._apply(cals[m][lname], preds_by_label[lname])
            for lname in sp.VALUE_LABELS
        }
        for m in cal_names
        if m != "raw"
    }
    predictor["seconds"] = round(time.time() - t_net, 1)

    rows_out: list[dict] = []
    curves: dict[str, list[float]] = {}
    weights: dict[str, dict] = {}
    heads: dict[str, dict] = {}
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

        # The incumbent-schema ranker — identical recipe to rounds
        # 4-8: every iso arm's ``inner`` is a frozen copy of it, so
        # the ranking the channel gates is the incumbent's own.
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
            {
                "id": "incumbent",
                "kind": "rl",
                "value_mode": None,
                "calibrator": None,
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

        # The ``ac`` control — the closest serializable arm's credit
        # mechanism, on the incumbent schema (no channel).
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
            {
                "id": "ac",
                "kind": "ac",
                "value_mode": None,
                "calibrator": None,
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

        # --------------------------------------------------------------
        #  The isolated-channel rows.  Every ``iso-*`` arm shares one
        #  shape: a frozen incumbent ranker on the eval-only legal set,
        #  plus a side-channel score the engine's ``forced_action``
        #  hook reads.  The ranking softmax never sees ``stop``.
        # --------------------------------------------------------------

        # ``oracle`` — the ceiling through the channel: the hook fires
        # ``stop`` the moment the memoized-verdict best prefix is
        # non-positive (the ranker's own greedy order).
        for tag, greedy_inner in (("oracle", False),):
            orc_sampled = sp._play_pools(
                lambda i: _mk_iso(
                    w_inc,
                    inc_spec,
                    seed + _SAMPLED_SEED + i,
                    score=_oracle_score(by_key, inc_spec, w_inc),
                    kappa=0.0,
                ),
                ev_pools,
                budget,
            )
            orc_greedy = sp._play_pools(
                lambda i: _mk_iso(
                    w_inc,
                    inc_spec,
                    seed + _GREEDY_SEED + i,
                    greedy=True,
                    score=_oracle_score(by_key, inc_spec, w_inc),
                    kappa=0.0,
                ),
                ev_pools,
                budget,
            )
            rows_out.append(
                {
                    "id": tag,
                    "kind": "oracle",
                    "value_mode": None,
                    "calibrator": None,
                    "reward_ep": orc_sampled["reward"],
                    "reward_greedy_ep": orc_greedy["reward"],
                    "useful_ep": orc_sampled["useful_per_ep"],
                    "useful_eval": orc_sampled["useful_per_eval"],
                    "evals_ep": orc_sampled["evals_per_ep"],
                    "stops": orc_sampled["stops"],
                    "stops_greedy": orc_greedy["stops"],
                    "ap": inc_ap,
                }
            )
            print(  # stdout-compat
                f"  [oracle] reward={orc_sampled['reward']:>8.4f} "
                f"evals={orc_sampled['evals_per_ep']:>5.2f}"
            )
            del greedy_inner

        # ``iso-drain`` / ``iso-always`` — the channel's degenerate
        # controls: scorer absent (never fires early — the drain twin)
        # and κ=+inf (fires at step 0 — the stop-on-sight twin).
        for tag, kw in (
            ("iso-drain", {"w_stop": None}),
            ("iso-always", {"score": lambda s, h: -1.0, "kappa": 1e9}),
        ):
            ds = sp._play_pools(
                lambda i, k=kw: _mk_iso(
                    w_inc, inc_spec, seed + _SAMPLED_SEED + i, **k
                ),
                ev_pools,
                budget,
            )
            dg = sp._play_pools(
                lambda i, k=kw: _mk_iso(
                    w_inc,
                    inc_spec,
                    seed + _GREEDY_SEED + i,
                    greedy=True,
                    **k,
                ),
                ev_pools,
                budget,
            )
            rows_out.append(
                {
                    "id": tag,
                    "kind": "control",
                    "value_mode": None,
                    "calibrator": None,
                    "reward_ep": ds["reward"],
                    "reward_greedy_ep": dg["reward"],
                    "useful_ep": ds["useful_per_ep"],
                    "useful_eval": ds["useful_per_eval"],
                    "evals_ep": ds["evals_per_ep"],
                    "stops": ds["stops"],
                    "stops_greedy": dg["stops"],
                    "ap": inc_ap,
                }
            )
            print(  # stdout-compat
                f"  [{tag}] reward={ds['reward']:>8.4f} "
                f"evals={ds['evals_per_ep']:>5.2f} stops={ds['stops']}"
            )

        # The break-even gates through the channel — every
        # (mode, calibrator) injection, both rules, the κ sweep.  Raw
        # κ=1x rows must reproduce round 8's ``_EVGate`` numbers.
        for mode in modes:
            for cal_name in cal_names:
                og.set_value_preds(
                    _injection_for(
                        mode, cal_name, preds_by_label, cal_tables
                    )
                )
                for rule in GATE_RULES:
                    for mult in kappa_mults:
                        arm_id = f"iso-{mode}-{rule}:{cal_name}@{mult:g}x"
                        sampled = sp._play_pools(
                            lambda i, m=mode, r=rule, mu=mult: _mk_iso(
                                w_inc,
                                inc_spec,
                                seed + _SAMPLED_SEED + i,
                                score=_ev_score(r, mu),
                                kappa=0.0,
                            ),
                            ev_pools,
                            budget,
                        )
                        greedy = sp._play_pools(
                            lambda i, r=rule, mu=mult: _mk_iso(
                                w_inc,
                                inc_spec,
                                seed + _GREEDY_SEED + i,
                                greedy=True,
                                score=_ev_score(r, mu),
                                kappa=0.0,
                            ),
                            ev_pools,
                            budget,
                        )
                        rows_out.append(
                            {
                                "id": arm_id,
                                "kind": "gate",
                                "value_mode": mode,
                                "calibrator": cal_name,
                                "gate_rule": rule,
                                "kappa_mult": mult,
                                "reward_ep": sampled["reward"],
                                "reward_greedy_ep": greedy["reward"],
                                "useful_ep": sampled["useful_per_ep"],
                                "useful_eval": sampled[
                                    "useful_per_eval"
                                ],
                                "evals_ep": sampled["evals_per_ep"],
                                "stops": sampled["stops"],
                                "stops_greedy": greedy["stops"],
                                "ap": inc_ap,
                            }
                        )
                        print(  # stdout-compat
                            f"    {arm_id:<34} "
                            f"reward={sampled['reward']:>8.4f} "
                            f"evals={sampled['evals_per_ep']:>5.2f} "
                            f"stops={sampled['stops']:>3}"
                        )

        # The learned head through the channel — logistic continue-head
        # on the spec's stop row (oracle labels), its logit calibrated
        # to p̂ on the train OOF decision rows, thresholded at the
        # honest break-even p* — plus the ridge value head (κ=0).
        for arm_id, tags, inj in HEAD_ARMS:
            spec = svl._spec_for(tags)
            if inj is not None:
                og.set_value_preds(
                    _injection_for(
                        inj[0], inj[1], preds_by_label, cal_tables
                    )
                )
            else:
                og.set_value_preds(None)
            hrows, hlabels, hvals, hpend = _collect_rows_v(
                spec, w_inc, train_pools, by_key, budget, inc_spec
            )
            erows, elabels, evals_v, _ep = _collect_rows_v(
                spec, w_inc, ev_pools, by_key, budget, inc_spec
            )
            w_head, hloss = sp._fit_head(spec, hrows, hlabels, seed)
            eval_ap = sp._ap_binary(
                [_dot(w_head, r) for r in erows], elabels
            )
            pstar, vbar, cbar = _breakeven_p(hvals, hlabels, hpend)
            feats = _stop_feats_of(spec)
            train_sig = [_sig(_dot(w_head, r)) for r in hrows]
            for cal_name in HEAD_CALS:
                cal = HEAD_CALS[cal_name]().fit(
                    train_sig, [float(b) for b in hlabels]
                )
                ev_p_hat = cal.predict([_sig(_dot(w_head, r)) for r in erows])
                cal_ho = calibration_metrics(
                    ev_p_hat, [float(b) for b in elabels]
                )
                key = f"iso-{arm_id}:{cal_name}"
                sampled = sp._play_pools(
                    lambda i, c=cal, wh=w_head, f=feats, p=pstar: (
                        _mk_iso(
                            w_inc,
                            inc_spec,
                            seed + _SAMPLED_SEED + i,
                            score=_cal_score(c, wh, f),
                            kappa=p,
                        )
                    ),
                    ev_pools,
                    budget,
                )
                greedy = sp._play_pools(
                    lambda i, c=cal, wh=w_head, f=feats, p=pstar: (
                        _mk_iso(
                            w_inc,
                            inc_spec,
                            seed + _GREEDY_SEED + i,
                            greedy=True,
                            score=_cal_score(c, wh, f),
                            kappa=p,
                        )
                    ),
                    ev_pools,
                    budget,
                )
                rows_out.append(
                    {
                        "id": key,
                        "kind": "iso-head",
                        "value_mode": inj[0] if inj else None,
                        "calibrator": cal_name,
                        "extra_parts": list(tags),
                        "reward_ep": sampled["reward"],
                        "reward_greedy_ep": greedy["reward"],
                        "useful_ep": sampled["useful_per_ep"],
                        "useful_eval": sampled["useful_per_eval"],
                        "evals_ep": sampled["evals_per_ep"],
                        "stops": sampled["stops"],
                        "stops_greedy": greedy["stops"],
                        "ap": inc_ap,
                    }
                )
                heads[key] = {
                    "pstar": round(pstar, 4),
                    "vbar": vbar,
                    "cbar": cbar,
                    "loss": hloss,
                    "n_rows": len(hrows),
                    "pos_rate": round(
                        sum(hlabels) / max(len(hlabels), 1), 4
                    ),
                    "eval_ap": round(eval_ap, 4),
                    "eval_pos_rate": round(
                        sum(elabels) / max(len(elabels), 1), 4
                    ),
                    "cal_held_out": cal_ho,
                    "weights": {
                        n: round(wv, 4)
                        for n, wv in w_head.items()
                        if abs(wv) > 1e-4
                    },
                }
                print(  # stdout-compat
                    f"  [{key:<30}] reward={sampled['reward']:>8.4f} "
                    f"evals={sampled['evals_per_ep']:>5.2f} "
                    f"stops={sampled['stops']:>3} "
                    f"p*={pstar:.4f}"
                )
            # The fold twin — the same head folded into the shared
            # softmax (the channel-isolation A/B control).
            fold_c, fold_cal = sp._calibrate_fold_c(
                spec, w_inc, w_head, train_pools, budget, seed
            )
            w_fold = sp._fold_weights(spec, w_inc, w_head, fold_c)
            heads[arm_id] = {
                **heads.get(arm_id, {}),
                "fold": {"c": round(fold_c, 4), "cal_reward": fold_cal},
            }
            fold_sampled = sp._play_pools(
                lambda i, s=spec, wf=w_fold: og.OpenLearnedPlayer(
                    seed=seed + _SAMPLED_SEED + i,
                    spec=s,
                    weights=wf,
                    learn=False,
                ),
                ev_pools,
                budget,
            )
            fold_greedy = sp._play_pools(
                lambda i, s=spec, wf=w_fold: og.OpenLearnedPlayer(
                    seed=seed + _GREEDY_SEED + i,
                    spec=s,
                    weights=wf,
                    learn=False,
                    greedy=True,
                ),
                ev_pools,
                budget,
            )
            rows_out.append(
                {
                    "id": f"iso-{arm_id}-fold",
                    "kind": "fold",
                    "value_mode": inj[0] if inj else None,
                    "calibrator": None,
                    "extra_parts": list(tags),
                    "reward_ep": fold_sampled["reward"],
                    "reward_greedy_ep": fold_greedy["reward"],
                    "useful_ep": fold_sampled["useful_per_ep"],
                    "useful_eval": fold_sampled["useful_per_eval"],
                    "evals_ep": fold_sampled["evals_per_ep"],
                    "stops": fold_sampled["stops"],
                    "stops_greedy": fold_greedy["stops"],
                    "ap": inc_ap,
                }
            )
            # The value-regression twin — κ=0, the literal break-even,
            # delivered through the linear ``w_stop`` path.
            w_vreg, vrmse = _fit_value_head(
                hrows, hvals, spec.features
            )
            vreg_ap = sp._ap_binary(
                [_dot(w_vreg, r) for r in erows], elabels
            )
            ev_rmse = round(
                math.sqrt(
                    fmean(
                        [
                            (_dot(w_vreg, r) - v) ** 2
                            for r, v in zip(erows, evals_v, strict=True)
                        ]
                    )
                ),
                4,
            )
            key = f"iso-v{arm_id}"
            sampled = sp._play_pools(
                lambda i, s=spec, wv=w_vreg: _mk_iso(
                    w_inc,
                    inc_spec,
                    seed + _SAMPLED_SEED + i,
                    stop_feats=_stop_feats_of(s),
                    w_stop=wv,
                    kappa=0.0,
                ),
                ev_pools,
                budget,
            )
            greedy = sp._play_pools(
                lambda i, s=spec, wv=w_vreg: _mk_iso(
                    w_inc,
                    inc_spec,
                    seed + _GREEDY_SEED + i,
                    greedy=True,
                    stop_feats=_stop_feats_of(s),
                    w_stop=wv,
                    kappa=0.0,
                ),
                ev_pools,
                budget,
            )
            rows_out.append(
                {
                    "id": key,
                    "kind": "iso-vreg",
                    "value_mode": inj[0] if inj else None,
                    "calibrator": None,
                    "extra_parts": list(tags),
                    "reward_ep": sampled["reward"],
                    "reward_greedy_ep": greedy["reward"],
                    "useful_ep": sampled["useful_per_ep"],
                    "useful_eval": sampled["useful_per_eval"],
                    "evals_ep": sampled["evals_per_ep"],
                    "stops": sampled["stops"],
                    "stops_greedy": greedy["stops"],
                    "ap": inc_ap,
                }
            )
            heads[key] = {
                "train_rmse": vrmse,
                "eval_rmse": ev_rmse,
                "eval_ap": round(vreg_ap, 4),
                "kappa": 0.0,
                "weights": {
                    n: round(wv, 4)
                    for n, wv in w_vreg.items()
                    if abs(wv) > 1e-4
                },
            }
            print(  # stdout-compat
                f"  [{key:<30}] reward={sampled['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"stops={sampled['stops']:>3} (vreg)"
            )

        # ``rl-gated`` — the ranker trains *with the channel armed*:
        # episodes truncate where the gate says stop, so the ranking
        # never sees dead-tail states.  The scorer is the isotonic
        # ``useful`` pool gate (round 8's best earning signal) and,
        # as a control, the raw ``delta`` one.
        for rmode, rcal in (("useful", "isotonic"), ("delta", "raw")):
            key = f"rl-gated:{rmode}/{rcal}"
            og.set_value_preds(
                _injection_for(rmode, rcal, preds_by_label, cal_tables)
            )
            inner_train = og.OpenLearnedPlayer(
                seed=seed, spec=inc_spec, weights={}, legal=sp._eval_only
            )
            armed = GatedPlayer(
                inner_train,
                legal=og.legal_actions,
                score=_ev_score("pool", 1.0),
                kappa=0.0,
            )
            curve = []
            for _ in range(epochs):
                totals = engine.train_policy(
                    armed, train_pools, sl._FixedArena, budget=budget
                )
                curve.append(
                    round(sum(totals) / max(len(totals), 1), 4)
                )
            w_rlg = inner_train.weights_dict()
            curves[key] = curve
            rlg_ap = round(
                pb._ap(
                    sp._page_scores(inc_spec, w_rlg, ev_pools),
                    flat_outcomes,
                ),
                4,
            )
            sampled = sp._play_pools(
                lambda i, m=rmode, c=rcal, w=w_rlg: _mk_iso(
                    w,
                    inc_spec,
                    seed + _SAMPLED_SEED + i,
                    score=_ev_score("pool", 1.0),
                    kappa=0.0,
                ),
                ev_pools,
                budget,
            )
            greedy = sp._play_pools(
                lambda i, w=w_rlg: _mk_iso(
                    w,
                    inc_spec,
                    seed + _GREEDY_SEED + i,
                    greedy=True,
                    score=_ev_score("pool", 1.0),
                    kappa=0.0,
                ),
                ev_pools,
                budget,
            )
            rows_out.append(
                {
                    "id": key,
                    "kind": "rl-gated",
                    "value_mode": rmode,
                    "calibrator": rcal,
                    "reward_ep": sampled["reward"],
                    "reward_greedy_ep": greedy["reward"],
                    "useful_ep": sampled["useful_per_ep"],
                    "useful_eval": sampled["useful_per_eval"],
                    "evals_ep": sampled["evals_per_ep"],
                    "stops": sampled["stops"],
                    "stops_greedy": greedy["stops"],
                    "ap": rlg_ap,
                }
            )
            print(  # stdout-compat
                f"  [{key:<30}] reward={sampled['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"stops={sampled['stops']:>3} ap={rlg_ap:.4f}"
            )

        # The honest unfed replay — every arm that read the value seam
        # is replayed with ``set_value_preds(None)`` exactly as a
        # weights bake would deploy it today.  (κ=1x gates + every
        # head+vl arm, gated and folded.)
        og.set_value_preds(None)
        unfed_rows: list[dict] = []
        for mode in modes:
            for rule in GATE_RULES:
                us = sp._play_pools(
                    lambda i, r=rule: _mk_iso(
                        w_inc,
                        inc_spec,
                        seed + _SAMPLED_SEED + i,
                        score=_ev_score(r, 1.0),
                        kappa=0.0,
                    ),
                    ev_pools,
                    budget,
                )
                unfed_rows.append(
                    {
                        "id": f"iso-{mode}-{rule}@1x-unfed",
                        "reward_ep": us["reward"],
                        "evals_ep": us["evals_per_ep"],
                        "stops": us["stops"],
                    }
                )
                print(  # stdout-compat
                    f"    unfed iso-{mode}-{rule}@1x "
                    f"reward={us['reward']:>8.4f} "
                    f"evals={us['evals_per_ep']:>5.2f}"
                )
    finally:
        lp.evaluate = orig_evaluate
        og.set_value_preds(None)
        osel.set_self_scores(None)

    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": rows_out,
        "unfed_rows": unfed_rows,
        "anchors": anchors,
        "train_curve": curves,
        "weights": weights,
        "heads": heads,
        "predictor": predictor,
        "pool": {"held_out": len(flat_pool), "useful": n_useful},
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "epochs": epochs,
            "modes": list(modes),
            "calibrators": list(cal_names),
            "kappa_mults": list(kappa_mults),
            "eval_cost": EVAL_COST,
            "neural": {
                "epochs": sa.NEURAL_EPOCHS,
                "hidden": sa.NEURAL_HIDDEN,
                "oof_folds": sa.OOF_FOLDS,
            },
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the arm table."""
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
    print("== isolated stop channel — held-out odd pages ==")  # stdout-compat
    head = (
        f"{'arm':<36} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
    )
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for r in res["rows"]:
        print(  # stdout-compat
            f"{r['id']:<36} {r['reward_ep']:>8.4f} "
            f"{r['reward_greedy_ep']:>8.4f} {r['useful_eval']:>9.4f} "
            f"{r['evals_ep']:>8.3f} {r['stops']:>6} {r['ap']:>7.4f}"
        )
    print("-- anchors --")  # stdout-compat
    for name, a in res["anchors"].items():
        s = a["sampled"]
        print(  # stdout-compat
            f"{name:<36} {s['reward']:>8.4f} {'—':>8} "
            f"{s['useful_per_eval']:>9.4f} "
            f"{s['evals_per_ep']:>8.3f} {'—':>6} {a['ap']:>7.4f}"
        )
    print("-- unfed replays (seam cleared) --")  # stdout-compat
    for r in res.get("unfed_rows", []):
        print(  # stdout-compat
            f"{r['id']:<36} {r['reward_ep']:>8.4f} {'—':>8} "
            f"{'—':>9} {r['evals_ep']:>8.3f} {r['stops']:>6}"
        )
    for arm_id, h in res.get("heads", {}).items():
        if "pstar" in h:
            print(  # stdout-compat
                f"-- head {arm_id}: p*={h['pstar']} V+={h['vbar']} "
                f"C-={h['cbar']} eval_ap={h['eval_ap']} "
                f"cal_ho={h['cal_held_out']} --"
            )
        elif "train_rmse" in h:
            print(  # stdout-compat
                f"-- vreg {arm_id}: rmse={h['train_rmse']}/"
                f"{h['eval_rmse']} eval_ap={h['eval_ap']} --"
            )


def main(argv: list[str] | None = None) -> int:
    """Run the isolated-channel measurement and print the table."""
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
            "delivery": (
                "GatedPlayer — the engine's forced_action hook; "
                "the ranker's legal set never carries stop"
            ),
            "kappa": "honest break-even |step| (no reward tuning)",
        },
        "results": res,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
