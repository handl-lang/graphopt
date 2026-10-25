"""The stopping policy, round 8 — *verdict-visible* usefulness.

``docs/results_stop_valuelabel.md`` (round 7) closed by re-naming the
blocker for the third time: every stop-row family so far is a
*ranking* view — the heuristic's ``open:next-value``, the injected
nets' ``open:next-val``/``open:vlabel`` (held-out AP <= 0.10 on the
labels that pay), the policy's own ``open:self`` posterior — and the
verdict named the remaining lever: **verdict-conditioned or
verifier-visible usefulness**.  This driver prices it.  The new
``open:verdict`` part (:mod:`mlopt_discovery.open_verdict`) emits
columns built from what the domain *certifies*, not from net
outputs:

* episode-banked verdicts (``state.verdicts`` — real oracle output):
  ``x:stop:vd_banked``/``vd_use`` (payers certified this episode —
  the page-richness evidence, since paying shapes cluster in the
  stream), ``x:stop:vd_live``/``vd_dead`` (the fraction of pending
  sharing a rewriting *form* with a certified payer / a certified
  dead sample), and the *dead-ceiling* columns
  ``x:stop:vd_above``/``vd_headroom`` — "does anything pending
  out-score what already failed", the episode's own certified
  drained-evidence;
* the corpus verdict table (``set_verdict_corpus`` — the
  ``set_value_preds`` seam's twin, fed closed-form **counts** not a
  learned net): ``signature -> (payers, useful, seen)`` tallied from
  the *train* pages' measured verdicts, where the signature is
  ``(form, lhs-op-spine, rhs-op-spine)`` — which concrete ops fill
  the equation skeleton's slots, the granularity at which a form is
  actually true (``factor-left`` over ``mul``/``add`` distributes;
  over ``sub``/``div`` it cannot).  ``x:stop:cr_top``/``cr_pool``/
  ``cr_use`` price the pending pool's certified hit-rates;
  ``x:stop:cr_dead``/``cr_cov`` report the table's certified-dead
  mass and its coverage; ``x:stop:cr_ftop``/``cr_fpool`` are the
  coarse *form*-level fallback.  ``x:vd:*`` columns put the same
  rates on the ``evaluate`` rows.

Protocol identical to rounds 5-7: cold retrain per arm under
memoized per-candidate verdicts, even pages train / odd held out
(page 16, 36 pages, budget 17), page-aware AP.  Two honesty details
the corpus seam adds: the *eval* head rows read the full train-side
table (held-out candidates only ever see train-certified counts —
no leakage), while *train* head rows read a **jackknifed** table —
each train pool's rows score against the corpus minus that page's
own contribution (the count-table analogue of round 5's OOF train
preds), so the head's fit sees held-out-quality statistics.

The test the round-7 verdict set: a column whose decision-row
(continue-label) usefulness AP *transfers* held-out — past the
~0.40 coin-flip ceiling every ranking-view family has hit.

Runtime: ~8-9 min one-time pool memoization on this box,
~1-5 s per arm.

Usage::

    timeout 2400 .venv/bin/python bench/drivers/policy/stop_verdict.py \
        --json /tmp/vd.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import policy_baseline as pb
import schema_leaderboard as sl
import stop_advantage as sa
import stop_policy as sp
from gamekit import advantage as adv
from gamekit import engine
from gamekit.players import PARTS, DomainSpec, PlayHist, _dot
from mlopt_discovery import open_game as og
from mlopt_discovery import open_self as osel
from mlopt_discovery import open_verdict as ov
from mlopt_discovery import play
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action

#: Defaults — identical protocol to ``stop_valuelabel.py``.
SEED = 0
PAGE = 16
OPEN_PAGES = 36  # the even half trains, the odd half evaluates
BUDGET = PAGE + 1
EPOCHS = 5  # the shipped bake used 5
V_LR = 0.05  # the value function's delta-rule rate
V_PASSES = 4  # delta-rule sweeps over each episode's rows
_SAMPLED_SEED = 40_000  # schema_leaderboard's eval-seed offsets
_GREEDY_SEED = 50_000

#: The extra-part vocabulary this driver knows —
#: ``stop_advantage``'s (shipped extras + ``open:self``) plus the
#: round-8 ``open:verdict`` part.
EXTRA: dict[str, tuple[str, ...]] = {
    **sa.EXTRA,
    "open:verdict": ov.VD_FEATURES,
}

#: The arms — ``(id, part tags, kind, rl_epochs, rank_tags)``; the
#: kinds are rounds 5-7's: ``rl`` flat REINFORCE, ``ac`` the
#: per-step-advantage twin, ``head`` the supervised stop gate over
#: the arm's own columns (incumbent ranker — the mechanism under
#: test is the stop decision).  ``vd-ep`` is the same probe with the
#: corpus seam cleared: the episode-only columns honest ablation —
#: which half of the family carries the signal.
ARMS: tuple[
    tuple[str, tuple[str, ...], str, int, tuple[str, ...]], ...
] = (
    ("incumbent", (), "rl", EPOCHS, ()),
    ("ac", (), "ac", EPOCHS, ()),
    ("verdict", ("open:verdict",), "rl", EPOCHS, ()),
    ("ac+verdict", ("open:verdict",), "ac", EPOCHS, ()),
    ("head+verdict", ("open:verdict",), "head", EPOCHS, ()),
    (
        "head+stop+verdict",
        ("open:stop", "open:verdict"),
        "head",
        EPOCHS,
        (),
    ),
    ("head+vd-ep", ("open:verdict",), "head-ep", EPOCHS, ()),
)

#: The stop-row columns the per-feature probe reads — the new
#: certified-evidence family plus ``stop_policy``'s known set.
PROBE_FEATURES: tuple[str, ...] = (
    tuple(f for f in ov.VD_FEATURES if f.startswith("x:stop:"))
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


def _corpus_table(
    pools: list[list],
    outcomes: list[list],
) -> dict[str, dict]:
    """Tally ``signature -> (payers, useful, seen)`` over *outcomes*.

    The verdict table the domain can certify — counts, not a fit.
    Two granularities: the op-spine signature (``ov.signature``) and
    the coarse rewriting *form*.
    """
    sig_tab: dict[tuple, list] = {}
    form_tab: dict[str, list] = {}
    for pool, outs in zip(pools, outcomes, strict=True):
        for c, o in zip(pool, outs, strict=True):
            v = og._verdict(o)
            pay = v in ("useful", "true")
            use = v == "useful"
            for tab, key in (
                (sig_tab, ov.signature(c)),
                (form_tab, c.label.split(":", 1)[0]),
            ):
                p, u, n = tab.get(key, (0, 0, 0))
                tab[key] = (p + pay, u + use, n + 1)
    return {"sig": sig_tab, "form": form_tab}


def _jackknife(table: dict[str, dict], drop: dict) -> dict[str, dict]:
    """The corpus table minus one pool's contribution.

    *drop* is a per-key ``(pays, useful, seen)`` decrement map pair
    (``{"sig": {...}, "form": {...}}``) — the count-table analogue of
    round 5's out-of-fold train predictions: train decision rows
    must not read statistics that include their own page's verdicts.
    """
    out: dict[str, dict] = {}
    for kind in ("sig", "form"):
        t: dict = {}
        for k, (p, u, n) in table[kind].items():
            dp, du, dn = drop[kind].get(k, (0, 0, 0))
            t[k] = (p - dp, u - du, n - dn)
        out[kind] = t
    return out


def _collect_vd_rows(
    spec: DomainSpec,
    w: dict[str, float],
    pools: list[list],
    by_key: dict,
    budget: int,
    rank_spec: DomainSpec | None = None,
    tables: list[dict] | None = None,
) -> tuple[list[dict], list[bool]]:
    """``sp._collect_head_rows`` with a per-pool corpus table.

    Binds ``tables[i]`` (the jackknifed table for train pools, the
    full table for eval pools) before featurizing each pool's
    decision rows, so every row reads honestly-scoped statistics.
    ``None`` leaves whatever table the caller bound.
    """
    rows: list[dict] = []
    labels: list[bool] = []
    for pi, pool in enumerate(pools):
        if tables is not None:
            ov.set_verdict_corpus(tables[pi])
        arena = sl._FixedArena(pool)
        hist = PlayHist()
        for _ in range(budget):
            state = arena.observe()
            if state.done or not state.pending:
                break
            rows.append(spec.featurize(state, Action("stop", {}), hist))
            order = sp._eval_order(
                spec if rank_spec is None else rank_spec, w, state
            )
            labels.append(
                sp._prefix_value(state, order, by_key) > 0.0
            )
            i = order[0]
            act = Action("evaluate", {"i": i, "cand": state.pool[i]})
            arena.step(act)
            hist.record(act.op)
    return rows, labels


def run(
    seed: int,
    half: int,
    page: int,
    budget: int,
    epochs: int,
    cache: str | None = None,
) -> dict[str, Any]:
    """Measure pools once; train each arm cold; eval held-out.

    *cache* pickles the memoized verdicts (deterministic under the
    per-candidate seeds — the ``stop_gate.py`` rerun-safe convention
    on this contended box).
    """
    t0 = time.time()
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    loaded = None
    if cache and Path(cache).exists():
        with open(cache, "rb") as f:
            loaded = pickle.load(f)
    if loaded is not None:
        eval_outcomes, train_outcomes, by_key = loaded
    else:
        eval_outcomes, by_key = pb._measure_pools(
            ev_pools, seed + pb.HELD_OUT_OFFSET
        )
        train_outcomes, train_keys = pb._measure_pools(
            train_pools, seed + 1_000_003
        )
        by_key.update(train_keys)
        if cache:
            with open(cache, "wb") as f:
                pickle.dump(
                    (eval_outcomes, train_outcomes, by_key), f
                )

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1
    measure_s = time.time() - t0

    # The corpus verdict table — counts over the *train* pages'
    # certified verdicts (never the held-out half), plus the
    # per-page jackknifed variants the train head rows need.
    t_tab = time.time()
    corpus = _corpus_table(train_pools, train_outcomes)
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
        train_tables.append(_jackknife(corpus, drop))
    eval_tables = [corpus] * len(ev_pools)
    ov.set_verdict_corpus(corpus)
    n_sig = len(corpus["sig"])
    sig_cov = sum(
        ov.signature(c) in corpus["sig"] for c in flat_pool
    )
    corpus_report = {
        "n_signatures": n_sig,
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
                if kind.startswith("head")
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
            if "open:self" in tags:
                osel.set_self_scores(
                    sa._mk_self_scorer(rank_spec.featurize, player._w)
                )
            else:
                osel.set_self_scores(None)
            # The corpus seam: cleared for the episode-only
            # ablation, the full train table otherwise (the same
            # honesty level as round 5's injected preds — train
            # episodes read train-certified statistics).
            ov.set_verdict_corpus(
                None if kind == "head-ep" else corpus
            )
            curve: list[float] = []
            for _ in range(n_epochs):
                totals = engine.train_policy(
                    player, train_pools, sl._FixedArena, budget=budget
                )
                curve.append(
                    round(sum(totals) / max(len(totals), 1), 4)
                )
            w_rank = player.weights_dict()
            if kind in ("head", "head-ep"):
                hrows, hlabels = _collect_vd_rows(
                    spec,
                    w_rank,
                    train_pools,
                    by_key,
                    budget,
                    rank_spec=rank_spec,
                    tables=(
                        [None] * len(train_pools)
                        if kind == "head-ep"
                        else train_tables
                    ),
                )
                erows, elabels = _collect_vd_rows(
                    spec,
                    w_rank,
                    ev_pools,
                    by_key,
                    budget,
                    rank_spec=rank_spec,
                    tables=(
                        [None] * len(ev_pools)
                        if kind == "head-ep"
                        else eval_tables
                    ),
                )
                ov.set_verdict_corpus(
                    None if kind == "head-ep" else corpus
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
                        if kind.startswith("head")
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
                n: round(wv, 4)
                for n, wv in w_rank.items()
                if abs(wv) > 1e-4
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
            print(  # stdout-compat
                f"  [{arm_i + 1}/{len(ARMS)}] {arm_id:<18} "
                f"reward={sampled['reward']:>8.4f} "
                f"greedy={greedy['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"ap={ap:.4f} ({time.time() - t1:.0f}s)"
            )
    finally:
        lp.evaluate = orig_evaluate
        osel.set_self_scores(None)
        ov.set_verdict_corpus(None)

    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": rows_out,
        "anchors": anchors,
        "train_curve": curves,
        "weights": weights,
        "heads": heads,
        "vdiag": vdiag,
        "corpus": corpus_report,
        "pool": {"held_out": len(flat_pool), "useful": n_useful},
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "epochs": epochs,
            "v_lr": V_LR,
            "v_passes": V_PASSES,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the arm table."""
    head = (
        f"{'arm':<18} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
    )
    cp = res.get("corpus")
    if cp:
        print(  # stdout-compat
            f"== corpus table: {cp['n_signatures']} signatures "
            f"({cp['n_forms']} forms), payers={cp['train_payers']} "
            f"useful={cp['train_useful']}, eval sig coverage "
            f"{cp['eval_sig_coverage']}, rate AP(pays)="
            f"{cp['eval_rate_ap_pays']} ({cp['seconds']}s) =="
        )
    print("== stopping policy — held-out odd pages ==")  # stdout-compat
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for r in res["rows"]:
        print(  # stdout-compat
            f"{r['id']:<18} {r['reward_ep']:>8.4f} "
            f"{r['reward_greedy_ep']:>8.4f} {r['useful_eval']:>9.4f} "
            f"{r['evals_ep']:>8.3f} {r['stops']:>6} {r['ap']:>7.4f}"
        )
    print("-- anchors --")  # stdout-compat
    for name, a in res["anchors"].items():
        s = a["sampled"]
        print(  # stdout-compat
            f"{name:<18} {s['reward']:>8.4f} {'—':>8} "
            f"{s['useful_per_eval']:>9.4f} "
            f"{s['evals_per_ep']:>8.3f} {'—':>6} {a['ap']:>7.4f}"
        )
    for arm_id, h in res.get("heads", {}).items():
        print(  # stdout-compat
            f"-- head {arm_id}: tau={h['tau']} "
            f"cal_reward={h['cal_reward']} pos_rate={h['pos_rate']} "
            f"loss={h['loss']} eval_ap={h.get('eval_ap')} --"
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
    """Run the verdict-features measurement and print the table."""
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
        help="pickle of the memoized verdicts — rerun-safe",
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
            "corpus": (
                "train-page verdict counts keyed by "
                "(form, op-spine) signature; train head rows "
                "jackknifed, eval rows read the full table"
            ),
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
