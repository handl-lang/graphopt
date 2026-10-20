"""The stopping policy, round 6 — the observability attack.

``docs/results_stop_policy.md``'s five rounds closed on two missing
pieces: *evidence* (no feature prices the pending pool — the
heuristic view tops out at candidate AP 0.045, and the verdict-label
predictor memorizes its 7 positives) and *delivery* (the gate's
absolute threshold does not serialize — every weight-table delivery,
softmax columns / CE imitation / the head fold, lands a degenerate
point).  This driver prices the two remaining levers, both new:

* **the policy's own posterior as evidence** — the ``open:self`` part
  (:mod:`mlopt_discovery.open_self`): the model's softmax over the
  pending ``evaluate`` moves IS a weak value estimate, and round 3's
  blocker — a ``RelFeats`` part cannot see the weights — is lifted
  by injecting the scorer (``open_self.set_self_scores``).  The
  ``stop`` row gets the pool-priced quantities (best/top-3 pending
  logit, rank-1 posterior share, posterior-weighted logit and
  injected value, the above-uniform count); each ``evaluate`` row
  gets its posterior share and margin.
* **per-step advantage** — :class:`mlopt_discovery.advantage.
  AdvantagePlayer`: the REINFORCE EMA-total baseline becomes a
  learned ``V(s) = u·φ_stop(s)`` (delta-rule fit on the episode's
  own reward-to-go rows), so the ``stop`` action's credit isolates
  to ``-V(s)`` instead of inheriting the episode total — and the
  ``evaluate`` rows' credit is state-conditioned too, which is the
  variance reduction the round-3-5 shared-softmax AP collapses
  priced.

Protocol identical to ``stop_policy.py``: cold retrain per arm
(``engine.train_policy`` under the baseline's memoized verdicts),
even pages train / odd pages held out (page 16, 36 pages, budget 17),
page-aware AP.  The ``open:next-val`` arms inject the same
out-of-fold ``RuleValueNet`` sigmoid table round 5 built
(``--value-label``, default ``delta`` — the only injected label
whose gate earned).  Head arms reuse ``stop_policy``'s corrected
gate machinery under the round-4 protocol (``rank_tags=()`` — the
ranker is the incumbent schema, so the stop decision is measured
apart from the ranking).

Round-6b adds the delivery check the arms above were missing:
every injected-schema weights table is also replayed **unfed** —
``set_self_scores(None)`` + ``set_value_preds(None)``, the state
``OPEN_GAME_PLAYER_WEIGHTS`` would deploy it in — so a row that
clears the gate *bound* but collapses unfed is priced as the
non-serializable delivery it is.  ``head+self+stop`` (gate over
``open:self`` + ``open:stop``) joins the head family.

Bake rule (unchanged): a delivery joins
``OPEN_GAME_PLAYER_WEIGHTS`` only if held-out reward/ep AND
page-aware AP both clear the incumbent bake (0.273 / 0.210) —
never-stop or AP-collapse is an honest negative.

Runtime: ~150 s one-time pool memoization, ~10-90 s per arm.

Usage::

    timeout 1800 .venv/bin/python bench/policy/stop_advantage.py \
        --json /tmp/adv.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import policy_baseline as pb
import schema_leaderboard as sl
import stop_policy as sp
import torch
from mlopt_core.features import compute_features
from mlopt_core.trajectories import RuleSample, rule_vector
from mlopt_discovery import advantage as adv
from mlopt_discovery import engine, play
from mlopt_discovery import open_game as og
from mlopt_discovery import open_self as oself
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import PARTS, DomainSpec, _dot
from mlopt_torch.learned_policy import input_vector, train_rule_value

#: Defaults — identical protocol to ``stop_policy.py``.
SEED = 0
PAGE = 16
OPEN_PAGES = 36  # the even half trains, the odd half evaluates
BUDGET = PAGE + 1
EPOCHS = 5  # the shipped bake used 5
V_LR = 0.05  # the value function's delta-rule rate
V_PASSES = 4  # delta-rule sweeps over each episode's rows
NEURAL_EPOCHS = 300  # ``train_rule_value`` — policy_baseline's recipe
NEURAL_HIDDEN = 32
OOF_FOLDS = 6  # predictor folds — out-of-fold rows for calibration
_SAMPLED_SEED = 40_000  # schema_leaderboard's eval-seed offsets
_GREEDY_SEED = 50_000

#: The extra-part vocabulary this driver knows — the shipped extras
#: plus the round-6 ``open:self`` part (registered in
#: ``mlopt_discovery.open_self``, outside ``OPEN_EXTRA_FEATURES``).
EXTRA: dict[str, tuple[str, ...]] = {
    **og.OPEN_EXTRA_FEATURES,
    "open:self": oself.SELF_FEATURES,
}

#: The arms — ``(id, part tags, kind, rl_epochs, rank_tags)``.
#:
#: * ``rl`` — plain episode REINFORCE (the incumbent control and the
#:   ``self``/``self+nv`` schema arms — rounds 3-5's collapse is the
#:   expected outcome under flat credit; they are measured anyway);
#: * ``ac`` — the actor-critic player: same softmax decode, per-step
#:   advantage ``G_t - V(s_t)`` with V fit on the ``stop`` row's state
#:   summary — the credit-assignment mechanism;
#: * ``head`` — ``stop_policy``'s supervised gate over the arm's own
#:   stop-row columns under the round-4 protocol (``rank_tags=()`` —
#:   incumbent ranker): the honest read on whether the self-posterior
#:   columns carry stop-decision signal at all.
ARMS: tuple[
    tuple[str, tuple[str, ...], str, int, tuple[str, ...]], ...
] = (
    ("incumbent", (), "rl", EPOCHS, ()),
    ("ac", (), "ac", EPOCHS, ()),
    ("ac+stop", ("open:stop",), "ac", EPOCHS, ()),
    ("self", ("open:self",), "rl", EPOCHS, ()),
    ("ac+self", ("open:self",), "ac", EPOCHS, ()),
    ("nv", ("open:next-val",), "rl", EPOCHS, ()),
    ("ac+nv", ("open:next-val",), "ac", EPOCHS, ()),
    ("self+nv", ("open:self", "open:next-val"), "rl", EPOCHS, ()),
    (
        "ac+self+nv",
        ("open:self", "open:next-val"),
        "ac",
        EPOCHS,
        (),
    ),
    ("head+self", ("open:self",), "head", EPOCHS, ()),
    (
        "head+self+nv",
        ("open:self", "open:next-val"),
        "head",
        EPOCHS,
        (),
    ),
    (
        "head+self+stop",
        ("open:self", "open:stop"),
        "head",
        EPOCHS,
        (),
    ),
)

#: The stop-row columns the per-feature probe reads — the new
#: self-posterior family plus ``stop_policy``'s known set.
PROBE_FEATURES: tuple[str, ...] = (
    oself.SELF_FEATURES + sp.PROBE_FEATURES
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


def _mk_self_scorer(featurize: Any, w: dict) -> Any:
    """Bind the injected posterior scorer — ``w · featurize``, memoized.

    The memo key carries the state's pool identity, its step count,
    the evaluated member and the play-history length, plus a hash of
    the weight table — the scorer reads *w* live, so the fingerprint
    keeps episode-to-episode weight updates from serving stale
    logits while a single state's posterior is computed once and
    shared by every row featurized under it.
    """
    memo: dict = {}

    def score(state: Any, action: Any, hist: Any) -> float:
        wkey = hash(frozenset(w.items()))
        key = (
            id(state.pool),
            state.steps,
            action.params.get("i"),
            getattr(hist, "n", -1),
            wkey,
        )
        v = memo.get(key)
        if v is None:
            v = _dot(w, featurize(state, action, hist))
            if len(memo) < 400_000:
                memo[key] = v
        return v

    return score


def _fit_preds(
    flat_train: list,
    flat_train_outcomes: list,
    all_cands: list,
    train_pools: list[list],
    page: int,
    seed: int,
    label: str,
) -> tuple[dict, dict]:
    """Train the value net + OOF preds — ``stop_policy`` round 5's block.

    Replicates the round-5 protocol for the single injected *label*:
    the full-table net (``delta`` rides the shipped ``binary=True``
    path; ``useful``/``pays`` carry the balanced ``pos_weight``), then
    ``OOF_FOLDS`` held-out-quality train preds so τ calibration and
    head rows see the same signal quality held-out pages get.
    """
    samples = [
        RuleSample(
            compute_features(c.lhs),
            rule_vector(c.as_rule()),
            c.as_rule().name,
            o.base_cost - o.cand_cost,
            c.label,
        )
        for c, o in zip(flat_train, flat_train_outcomes, strict=True)
    ]
    lf = sp.VALUE_LABELS[label]
    ys = [lf(o) for o in flat_train_outcomes]
    n_pos = int(sum(ys))
    pw = (len(ys) - n_pos) / max(n_pos, 1)
    kw: dict[str, Any] = (
        {} if label == "delta" else {"labels": ys, "pos_weight": pw}
    )
    net = train_rule_value(
        samples,
        epochs=NEURAL_EPOCHS,
        hidden=NEURAL_HIDDEN,
        device="cpu",
        seed=seed,
        **kw,
    )
    fpp = max(len(train_pools) // OOF_FOLDS, 1)

    def _fold_of(i: int) -> int:
        return (i // page) // fpp

    oof: dict = {}
    for f in range(OOF_FOLDS):
        fsamples = [
            s for i, s in enumerate(samples) if _fold_of(i) != f
        ]
        fys = [y for i, y in enumerate(ys) if _fold_of(i) != f]
        held = [c for i, c in enumerate(flat_train) if _fold_of(i) == f]
        fkw: dict[str, Any] = (
            {}
            if label == "delta"
            else {"labels": fys, "pos_weight": pw}
        )
        fnet = train_rule_value(
            fsamples,
            epochs=NEURAL_EPOCHS,
            hidden=NEURAL_HIDDEN,
            device="cpu",
            seed=seed + 7 + f,
            **fkw,
        )
        xh = torch.tensor(
            [
                input_vector(
                    compute_features(c.lhs), rule_vector(c.as_rule())
                )
                for c in held
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            ph = torch.sigmoid(fnet(xh))
        for c, p in zip(held, ph, strict=True):
            oof[og._value_key(c)] = float(p)
    x_net = torch.tensor(
        [
            input_vector(
                compute_features(c.lhs), rule_vector(c.as_rule())
            )
            for c in all_cands
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        probs = torch.sigmoid(net(x_net))
    preds = {
        og._value_key(c): float(p)
        for c, p in zip(all_cands, probs, strict=True)
    }
    preds.update(oof)  # train candidates get held-out-quality preds
    report = {
        "label": label,
        "train_samples": len(samples),
        "train_positives": n_pos,
        "pos_weight": round(pw, 2),
        "epochs": NEURAL_EPOCHS,
        "hidden": NEURAL_HIDDEN,
        "oof_folds": OOF_FOLDS,
    }
    return preds, report


def run(
    seed: int,
    half: int,
    page: int,
    budget: int,
    epochs: int,
    value_label: str = "delta",
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

    # The injected value table — ``stop_policy`` round 5's protocol,
    # injected-label net only (default ``delta``: the only label whose
    # gate earned).  Train preds are out-of-fold — honest calibration.
    t_net = time.time()
    flat_train = [c for pool in train_pools for c in pool]
    flat_train_outcomes = [o for outs in train_outcomes for o in outs]
    all_cands = flat_train + flat_pool
    preds, predictor = _fit_preds(
        flat_train,
        flat_train_outcomes,
        all_cands,
        train_pools,
        page,
        seed,
        value_label,
    )
    og.set_value_preds(preds)
    predictor["seconds"] = round(time.time() - t_net, 1)

    rows_out: list[dict] = []
    curves: dict[str, list[float]] = {}
    weights: dict[str, dict] = {}
    heads: dict[str, dict] = {}
    vdiag: dict[str, dict] = {}
    specs: dict[str, tuple[DomainSpec, DomainSpec]] = {}
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
            rank_spec = _spec_for(rank_tags) if kind == "head" else spec
            specs[arm_id] = (spec, rank_spec)
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
            # The self-posterior seam — the scorer is the *ranker's*
            # own w·φ over the pending evaluate rows, read live (the
            # columns track the training weights; the frozen eval
            # arms share the trained table).  Head arms bind it the
            # same way: the stop row's self columns are then the
            # ranker's own posterior, the honest "model's view".
            if "open:self" in tags:
                oself.set_self_scores(
                    _mk_self_scorer(rank_spec.featurize, player._w)
                )
            else:
                oself.set_self_scores(None)
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
                # The serializable fold — the gate's stop-only columns
                # plus a calibrated ``op:stop`` intercept as an
                # ordinary weight table (``stop_policy`` round 5).
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
                        if kind == "head"
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
                f"  [{arm_i + 1}/{len(ARMS)}] {arm_id:<14} "
                f"reward={sampled['reward']:>8.4f} "
                f"greedy={greedy['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"ap={ap:.4f} ({time.time() - t1:.0f}s)"
            )

        # Delivery honesty — the *unfed replay*.  Every row above is
        # measured with the injected seams bound (``set_self_scores``
        # for ``open:self`` arms, ``set_value_preds`` for
        # ``open:next-val``).  ``OPEN_GAME_PLAYER_WEIGHTS`` serializes
        # only the ``{feature: weight}`` table — a bake of any arm
        # whose schema carries an injected part deploys today with the
        # seam unfed, its columns emitting zeros, so the shipped
        # function is *not* the measured one.  This replays each
        # weights-table candidate exactly as the bake would run it
        # (the rounded near-zero-dropped table ``weights`` stores is
        # what ``train_open_prior.py`` serializes): the ``-unfed``
        # rows are the honest serializable numbers.  The seams stay
        # cleared for the whole pass — a column family not in the
        # arm's schema cannot be emitted anyway, so clearing both is
        # the same replay as clearing the arm's own seam.
        oself.set_self_scores(None)
        og.set_value_preds(None)
        for arm_id, tags, kind, _e, _rt in ARMS:
            if not ({"open:self", "open:next-val"} & set(tags)):
                continue
            spec, _rank_spec = specs[arm_id]
            candidates = (
                []
                if kind == "head"  # the ranker alone is no delivery
                else [(arm_id, weights[arm_id])]
            )
            if arm_id + "-fold" in weights:
                candidates.append(
                    (arm_id + "-fold", weights[arm_id + "-fold"])
                )
            for cand_id, w in candidates:
                usampled = sp._play_pools(
                    lambda i, s=spec, wf=w: og.OpenLearnedPlayer(
                        seed=seed + _SAMPLED_SEED + i,
                        spec=s,
                        weights=wf,
                        learn=False,
                    ),
                    ev_pools,
                    budget,
                )
                ugreedy = sp._play_pools(
                    lambda i, s=spec, wf=w: og.OpenLearnedPlayer(
                        seed=seed + _GREEDY_SEED + i,
                        spec=s,
                        weights=wf,
                        learn=False,
                        greedy=True,
                    ),
                    ev_pools,
                    budget,
                )
                uap = round(
                    pb._ap(
                        sp._page_scores(spec, w, ev_pools),
                        flat_outcomes,
                    ),
                    4,
                )
                rows_out.append(
                    {
                        "id": cand_id + "-unfed",
                        "kind": "unfed",
                        "extra_parts": list(tags),
                        "rank_parts": (
                            list(_rt) if kind == "head" else list(tags)
                        ),
                        "rl_epochs": n_epochs,
                        "n_features": len(spec.features),
                        "reward_ep": usampled["reward"],
                        "reward_greedy_ep": ugreedy["reward"],
                        "useful_ep": usampled["useful_per_ep"],
                        "useful_eval": usampled["useful_per_eval"],
                        "evals_ep": usampled["evals_per_ep"],
                        "stops": usampled["stops"],
                        "stops_greedy": ugreedy["stops"],
                        "ap": uap,
                    }
                )
                print(  # stdout-compat
                    f"  [unfed] {cand_id:<19} "
                    f"reward={usampled['reward']:>8.4f} "
                    f"greedy={ugreedy['reward']:>8.4f} "
                    f"evals={usampled['evals_per_ep']:>5.2f} "
                    f"ap={uap:.4f}"
                )
    finally:
        lp.evaluate = orig_evaluate
        og.set_value_preds(None)
        oself.set_self_scores(None)

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
            "v_lr": V_LR,
            "v_passes": V_PASSES,
            "neural": {
                "epochs": NEURAL_EPOCHS,
                "hidden": NEURAL_HIDDEN,
            },
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the arm table."""
    head = (
        f"{'arm':<15} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
    )
    pred = res.get("predictor")
    if pred:
        print(  # stdout-compat
            f"== value predictor: label={pred['label']} "
            f"pos={pred['train_positives']}/{pred['train_samples']} "
            f"pw={pred['pos_weight']} (OOF {pred['oof_folds']} folds, "
            f"{pred['seconds']}s) =="
        )
    print("== stopping policy — held-out odd pages ==")  # stdout-compat
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for r in res["rows"]:
        print(  # stdout-compat
            f"{r['id']:<15} {r['reward_ep']:>8.4f} "
            f"{r['reward_greedy_ep']:>8.4f} {r['useful_eval']:>9.4f} "
            f"{r['evals_ep']:>8.3f} {r['stops']:>6} {r['ap']:>7.4f}"
        )
    print("-- anchors --")  # stdout-compat
    for name, a in res["anchors"].items():
        s = a["sampled"]
        print(  # stdout-compat
            f"{name:<15} {s['reward']:>8.4f} {'—':>8} "
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
                f"    probe {f:<21} solo={p['solo_ap']:.4f} "
                f"loo={p['loo_ap']:.4f} d={p['loo_delta']:+.4f}"
            )
        for n, wv in sorted(
            h["weights"].items(), key=lambda kv: -abs(kv[1])
        )[:10]:
            print(f"    {n:<21} {wv:>9.4f}")  # stdout-compat
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
        choices=sorted(sp.VALUE_LABELS),
        default="delta",
        help="which trained predictor feeds set_value_preds",
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
