"""The stopping policy, round 7 — candidate *value labels* priced.

``docs/results_stop_advantage.md`` (round 6) closed on the closest
serializable arm yet — ``ac+nv`` 0.2764 / AP 0.2074, missing the
0.210 bar by 0.003 — and re-named the evidence blocker: every stop-row
signal so far is a *ranking* view (the heuristic, the injected net's
raw probability, the policy's own posterior), none a *will-it-pay*
one.  This driver prices the remaining label-side lever named there:
the injected per-candidate value is an **expected payoff** — the
verdict-label predictors' output in reward units — and the stop row
sees the pool-priced decision quantities directly:

* the new ``open:vlabel`` part (:mod:`mlopt_discovery.open_vlabel`)
  emits ``x:stop:vl_pool`` (Σ pending expected payoff — the
  *expected remaining yield*), ``x:stop:vl_net`` (yield net of the
  drain's eval costs), ``x:stop:vl_margin`` (best pending eval's EV),
  ``x:stop:vl_best``/``x:stop:vl_frac``, and per-candidate
  ``x:vl:exp``/``x:vl:net`` on the evaluate rows;
* ``--value-label`` picks the injection: ``useful``/``pays``/
  ``delta`` inject ``p(label)·unit`` (unit = the tier payoff the
  label prices — 1.0 / 0.2-floor / 1.0), and ``ev`` — the default —
  injects the two-net expected tier payoff
  ``p_useful·1.0 + max(p_pays - p_useful, 0)·0.2``;
* two **analytic break-even gates** join the arm table — no learned
  head, no reward-calibrated τ: ``ev-margin`` stops when the best
  pending eval's expected payoff no longer covers its cost
  (``max_pending v ≤ |step|``), ``ev-pool`` when draining the rest
  nets ≤ 0 (``Σ_pending (v - |step|) ≤ 0``).  They price the
  asymmetric table's own break-even — the calibrated-threshold
  variant the learned head's reward-max τ never lands on.

The ``ac`` twins rerun round 6's per-step-advantage credit on the
new columns (``open:vlabel`` rows are the ``state_feats`` V reads);
``head+*`` arms reuse ``stop_policy``'s corrected gate +
serializable fold unchanged.  Protocol identical to
``stop_advantage.py``: cold retrain per arm under memoized
per-candidate verdicts, even pages train / odd pages held out
(page 16, 36 pages, budget 17), page-aware AP.

Bake rule (unchanged): a delivery joins
``OPEN_GAME_PLAYER_WEIGHTS`` only if held-out reward/ep AND
page-aware AP both clear the incumbent bake (0.273 / 0.210) —
never-stop or AP-collapse is an honest negative.

Runtime: ~150-175 s one-time pool memoization, ~4-7 s predictors
(3 labels x 7 nets), ~1-10 s per arm — ~3.5 min per label run.

Usage::

    timeout 1800 .venv/bin/python bench/policy/stop_valuelabel.py \
        --json /tmp/vl.json                    # --value-label ev (default)
    ... --value-label useful --json /tmp/vl_u.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import policy_baseline as pb
import schema_leaderboard as sl
import stop_advantage as sa
import stop_policy as sp
from mlopt_discovery import advantage as adv
from mlopt_discovery import engine, play
from mlopt_discovery import open_game as og
from mlopt_discovery import open_self as osel
from mlopt_discovery import open_vlabel as ovl
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import PARTS, DomainSpec, _dot

#: Defaults — identical protocol to ``stop_advantage.py``.
SEED = 0
PAGE = 16
OPEN_PAGES = 36  # the even half trains, the odd half evaluates
BUDGET = PAGE + 1
EPOCHS = 5  # the shipped bake used 5
V_LR = 0.05  # the value function's delta-rule rate
V_PASSES = 4  # delta-rule sweeps over each episode's rows
_SAMPLED_SEED = 40_000  # schema_leaderboard's eval-seed offsets
_GREEDY_SEED = 50_000

#: The extra-part vocabulary this driver knows — ``stop_advantage``'s
#: (shipped extras + ``open:self``) plus the round-7
#: ``open:vlabel`` part.
EXTRA: dict[str, tuple[str, ...]] = {
    **sa.EXTRA,
    "open:vlabel": ovl.VL_FEATURES,
}

#: The injected table is in *expected-payoff* units — what one eval
#: of the candidate is predicted to collect — so the break-even
#: κ is ``|step|`` under every label.  For the probability labels the
#: unit is the tier the label prices: ``useful`` pays 1.0 (ordering
#: credit excluded — conservative), ``pays`` is floored at the
#: ``true`` tier 0.2 (a payer is useful|true; the split is
#: unpredictable, so the cheap tier is the honest floor), ``delta``
#: keeps 1.0 (the round-4/5 control semantics — sigmoid-as-payoff).
#: ``ev`` composes the expected tier payoff from the two verdict
#: nets: ``p_useful·1.0 + max(p_pays - p_useful, 0)·0.2``.
PAYOFF_UNIT: dict[str, float] = {
    "delta": 1.0,
    "useful": og.OPEN_GAME_REWARD["useful"],
    "pays": og.OPEN_GAME_REWARD["true"],
    "ev": 1.0,
}

#: The arms — ``(id, part tags, kind, rl_epochs, rank_tags)``; same
#: kinds as ``stop_advantage`` plus two gate-only kinds:
#:
#: * ``evm`` / ``evp`` — the analytic break-even gates: a cold-RL
#:   incumbent-schema ranker (``rank_tags=()`` — the round-4 protocol,
#:   ranking held apart from the decision) whose ``evaluate`` picks
#:   are gated by the injected table directly — ``evm`` continues
#:   while ``max_pending v(c) > |step|`` (the next eval still clears
#:   its cost), ``evp`` while ``Σ_pending (v(c) - |step|) > 0``
#:   (draining the rest still nets positive).  No head fit, no
#:   reward-calibrated τ — the asymmetric cost table's own
#:   break-even, so the arms price *signal quality* alone.
ARMS: tuple[
    tuple[str, tuple[str, ...], str, int, tuple[str, ...]], ...
] = (
    ("incumbent", (), "rl", EPOCHS, ()),
    ("ac", (), "ac", EPOCHS, ()),
    ("nv", ("open:next-val",), "rl", EPOCHS, ()),
    ("ac+nv", ("open:next-val",), "ac", EPOCHS, ()),
    ("vlabel", ("open:vlabel",), "rl", EPOCHS, ()),
    ("ac+vlabel", ("open:vlabel",), "ac", EPOCHS, ()),
    (
        "ac+self+vlabel",
        ("open:self", "open:vlabel"),
        "ac",
        EPOCHS,
        (),
    ),
    ("head+vlabel", ("open:vlabel",), "head", EPOCHS, ()),
    (
        "head+stop+vlabel",
        ("open:stop", "open:vlabel"),
        "head",
        EPOCHS,
        (),
    ),
    ("ev-margin", (), "evm", EPOCHS, ()),
    ("ev-pool", (), "evp", EPOCHS, ()),
)

#: The stop-row columns the per-feature probe reads — the new
#: expected-yield family plus ``stop_policy``'s known set.
PROBE_FEATURES: tuple[str, ...] = (
    tuple(f for f in ovl.VL_FEATURES if f.startswith("x:stop:"))
    + sp.PROBE_FEATURES
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


class _EVGate:
    """The analytic break-even gate — the injected table as decision.

    No learned head and no calibrated threshold: *kappa* is the
    per-eval cost in the injected value's units, so the gate prices
    the asymmetric reward table's own break-even directly.
    ``mode="margin"`` continues while the best pending eval's
    expected payoff covers its cost (``max v > κ``); ``mode="pool"``
    continues while draining the rest still nets positive
    (``Σ (v - κ) > 0``) — looser, since it lets small-value tails
    keep the episode alive.
    """

    def __init__(self, inner: Any, kappa: float, mode: str) -> None:
        """Bind the eval-only ranker, the break-even cost, the rule."""
        self._inner = inner
        self._k = kappa
        self._mode = mode

    def __call__(self, state: Any) -> Any:
        """Stop when the injected expected payoff no longer covers cost."""
        if not state.pending:
            return Action("stop", {})
        vals = [og._value(state.pool[i]) for i in state.pending]
        if self._mode == "pool":
            cont = sum(v - self._k for v in vals) > 0.0
        else:
            cont = max(vals) > self._k
        if not cont:
            return Action("stop", {})
        act = self._inner(state)
        if act is None:
            return Action("stop", {})
        return act


def _mk_ev_gate(
    player: Any,
    rank_spec: DomainSpec,
    kappa: float,
    mode: str,
    seed: int,
    *,
    greedy: bool = False,
) -> _EVGate:
    """Build one gated player: frozen incumbent ranker + the EV gate."""
    inner = og.OpenLearnedPlayer(
        seed=seed,
        spec=rank_spec,
        weights=player.weights_dict(),
        legal=sp._eval_only,
        learn=False,
        greedy=greedy,
    )
    return _EVGate(inner, kappa, mode)


def run(
    seed: int,
    half: int,
    page: int,
    budget: int,
    epochs: int,
    value_label: str = "ev",
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

    # The injected expected-payoff table — round 5's OOF protocol
    # (``stop_advantage._fit_preds`` per label: full table + OOF train
    # preds), plus the *label→payoff* conversion this round adds
    # (``PAYOFF_UNIT`` / the ``ev`` composite) so every injected value
    # is an expected payoff in reward units and the break-even κ is
    # ``|step|`` under all modes.
    t_net = time.time()
    flat_train = [c for pool in train_pools for c in pool]
    flat_train_outcomes = [o for outs in train_outcomes for o in outs]
    all_cands = flat_train + flat_pool
    preds_by_label: dict[str, dict] = {}
    predictor: dict[str, Any] = {"injected": value_label, "nets": {}}
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
    if value_label == "ev":
        pu, pp = preds_by_label["useful"], preds_by_label["pays"]
        injected = {
            k: pu.get(k, 0.0) * og.OPEN_GAME_REWARD["useful"]
            + max(pp.get(k, 0.0) - pu.get(k, 0.0), 0.0)
            * og.OPEN_GAME_REWARD["true"]
            for k in pu
        }
    else:
        unit = PAYOFF_UNIT[value_label]
        injected = {
            k: v * unit for k, v in preds_by_label[value_label].items()
        }
    flat_ordered = [flat_outcomes[i] for i in range(len(flat_pool))]
    ev_labels = {
        n: [bool(lf(o)) for o in flat_ordered]
        for n, lf in sp.VALUE_LABELS.items()
    }
    for lname, preds in preds_by_label.items():
        scores = [preds.get(og._value_key(c), 0.0) for c in flat_pool]
        predictor["nets"][lname].update(
            {
                f"ap_{n}": round(sp._ap_binary(scores, ev_labels[n]), 4)
                for n in sp.VALUE_LABELS
            }
        )
    predictor["payoff_unit"] = PAYOFF_UNIT[value_label]
    og.set_value_preds(injected)
    kappa = ovl.EVAL_COST
    predictor["seconds"] = round(time.time() - t_net, 1)

    rows_out: list[dict] = []
    curves: dict[str, list[float]] = {}
    weights: dict[str, dict] = {}
    heads: dict[str, dict] = {}
    vdiag: dict[str, dict] = {}
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

        for arm_i, (
            arm_id,
            tags,
            kind,
            n_epochs,
            rank_tags,
        ) in enumerate(ARMS):
            t1 = time.time()
            spec = _spec_for(tags)
            rank_spec = (
                _spec_for(rank_tags)
                if kind in ("head", "evm", "evp")
                else spec
            )
            if kind == "ac":
                player = adv.AdvantagePlayer(
                    seed=seed,
                    spec=rank_spec,
                    weights={},
                    state_feats=lambda s, h, s_=rank_spec: s_.featurize(
                        s, Action("stop", {}), h
                    ),
                    v_lr=V_LR,
                    v_passes=V_PASSES,
                )
            else:
                player = og.OpenLearnedPlayer(
                    seed=seed, spec=rank_spec, weights={}
                )
            # The self-posterior seam — bound exactly as
            # ``stop_advantage`` binds it (the ranker's own live w·φ).
            if "open:self" in tags:
                osel.set_self_scores(
                    sa._mk_self_scorer(rank_spec.featurize, player._w)
                )
            else:
                osel.set_self_scores(None)
            curve: list[float] = []
            for _ in range(n_epochs):
                totals = engine.train_policy(
                    player, train_pools, sl._FixedArena, budget=budget
                )
                curve.append(
                    round(sum(totals) / max(len(totals), 1), 4)
                )
            w_rank = player.weights_dict()
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
                w_head, hloss = sp._fit_head(spec, hrows, hlabels, seed)
                eval_ap = sp._ap_binary(
                    [_dot(w_head, r) for r in erows], elabels
                )
                probe: dict[str, dict] = {}
                for f in PROBE_FEATURES:
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
                        names=tuple(n for n in spec.features if n != f),
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
                heads[arm_id] = {
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
                # The serializable fold — same delivery as rounds 5-6.
                fold_c, fold_cal = sp._calibrate_fold_c(
                    spec, w_rank, w_head, train_pools, budget, seed
                )
                w_fold = sp._fold_weights(spec, w_rank, w_head, fold_c)
                weights[arm_id + "-fold"] = {
                    n: round(wv, 4)
                    for n, wv in w_fold.items()
                    if abs(wv) > 1e-4
                }
                heads[arm_id]["fold"] = {
                    "c": round(fold_c, 4),
                    "cal_reward": fold_cal,
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
            elif kind in ("evm", "evp"):
                mode = "margin" if kind == "evm" else "pool"
                sampled = sp._play_pools(
                    lambda i, p=player, rs=rank_spec, m=mode: (
                        _mk_ev_gate(
                            p, rs, kappa, m, seed + _SAMPLED_SEED + i
                        )
                    ),
                    ev_pools,
                    budget,
                )
                greedy = sp._play_pools(
                    lambda i, p=player, rs=rank_spec, m=mode: (
                        _mk_ev_gate(
                            p,
                            rs,
                            kappa,
                            m,
                            seed + _GREEDY_SEED + i,
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
                    sp._page_scores(rank_spec, w_rank, ev_pools),
                    flat_outcomes,
                ),
                4,
            )
            rows_out.append(
                {
                    "id": arm_id,
                    "kind": kind,
                    "extra_parts": list(tags),
                    "rank_parts": (
                        list(rank_tags)
                        if kind in ("head", "evm", "evp")
                        else list(tags)
                    ),
                    "rl_epochs": n_epochs,
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
                for n, w in w_rank.items()
                if abs(w) > 1e-4
            }
            if kind == "ac":
                vdiag[arm_id] = {
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
            if fold_sampled is not None and fold_greedy is not None:
                rows_out.append(
                    {
                        "id": arm_id + "-fold",
                        "kind": "fold",
                        "extra_parts": list(tags),
                        "rank_parts": list(rank_tags),
                        "rl_epochs": n_epochs,
                        "n_features": len(spec.features),
                        "reward_ep": fold_sampled["reward"],
                        "reward_greedy_ep": fold_greedy["reward"],
                        "useful_ep": fold_sampled["useful_per_ep"],
                        "useful_eval": fold_sampled["useful_per_eval"],
                        "evals_ep": fold_sampled["evals_per_ep"],
                        "stops": fold_sampled["stops"],
                        "stops_greedy": fold_greedy["stops"],
                        "ap": ap,
                    }
                )
            print(  # stdout-compat
                f"  [{arm_i + 1}/{len(ARMS)}] {arm_id:<16} "
                f"reward={sampled['reward']:>8.4f} "
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
        "anchors": anchors,
        "train_curve": curves,
        "weights": weights,
        "heads": heads,
        "vdiag": vdiag,
        "predictor": predictor,
        "pool": {"held_out": len(flat_pool), "useful": n_useful},
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "epochs": epochs,
            "value_label": value_label,
            "kappa": kappa,
            "v_lr": V_LR,
            "v_passes": V_PASSES,
            "neural": {
                "epochs": sa.NEURAL_EPOCHS,
                "hidden": sa.NEURAL_HIDDEN,
            },
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the arm table."""
    head = (
        f"{'arm':<16} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
    )
    pred = res.get("predictor")
    if pred:
        print(  # stdout-compat
            f"== value predictors (injected={pred['injected']} "
            f"unit={pred['payoff_unit']} OOF {sa.OOF_FOLDS} folds, "
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
    for arm_id, h in res.get("heads", {}).items():
        print(  # stdout-compat
            f"-- head {arm_id}: tau={h['tau']} "
            f"cal_reward={h['cal_reward']} pos_rate={h['pos_rate']} "
            f"loss={h['loss']} eval_ap={h.get('eval_ap')} "
            f"fold c={h['fold']['c']} cal={h['fold']['cal_reward']} --"
        )
        for f, p in h.get("feature_probe", {}).items():
            print(  # stdout-compat
                f"    probe {f:<22} solo={p['solo_ap']:.4f} "
                f"loo={p['loo_ap']:.4f} d={p['loo_delta']:+.4f}"
            )
        for n, wv in sorted(
            h["weights"].items(), key=lambda kv: -abs(kv[1])
        )[:10]:
            print(f"    {n:<22} {wv:>9.4f}")  # stdout-compat
    for arm_id, v in res.get("vdiag", {}).items():
        print(  # stdout-compat
            f"-- ac {arm_id}: v_err={v['v_err']} "
            f"v_size={v['v_size']} v_top={v['v_top']} --"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the stopping-policy measurement and print the table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument(
        "--value-label",
        choices=(*sorted(sp.VALUE_LABELS), "ev"),
        default="ev",
        help="which trained predictor feeds set_value_preds "
        "('ev' = the two-net expected tier payoff)",
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
        value_label=args.value_label,
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
            "value_label": args.value_label,
            "arms": [a for a, *_ in ARMS],
        },
        "results": res,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
