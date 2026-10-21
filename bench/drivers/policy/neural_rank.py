"""Neural-arm ranking attack — swap BCE for a per-page ordering loss.

``docs/results_neural_corpus.md`` located the residual neural gap in
*ordering credit*, not coverage: on the grown ondist+wide pool the
BCE-trained ``RuleValueNet`` finds the same 6 of 7 useful candidates
as the heuristic drain (16 evals/ep each) yet earns 0.309 vs 0.360
reward/episode — it ranks useful candidates lower *within a page*.
BCE is a pointwise objective: it scores each candidate against a
fixed threshold and never compares two candidates on the same slate.
This driver keeps everything else fixed — same ``RuleValueNet``
(hidden 32), same featurization, same grown ondist+wide training
pool, same 300-epoch full-batch Adam recipe, same seed — and swaps
only the loss for per-page ranking objectives:

* ``bce-useful`` — BCE pointwise on the ``useful`` label itself
  (``train_rule_value``'s ``labels`` seam, ``pos_weight`` rebalanced).
  With ``rank-useful`` this completes the 2x2 that separates the
  *label* change from the *loss-form* change.
* ``rank-delta`` — pairwise RankNet on the BCE label
  (``delta > 0`` outranks the rest).  The loss-only ablation: any
  movement vs the BCE row is attributable to the objective form.
* ``rank-useful`` — pairwise RankNet on the AP target itself
  (``useful`` outranks everything).  Sparse — 12 useful rows in the
  whole training pool — but it is the metric's own ordering.
* ``rank-graded`` — pairwise RankNet on a 3-tier relevance
  (``useful`` > ``delta > 0`` > rest): useful candidates are pushed
  above merely-positive ones, the exact ordering the reward pays.
* ``listnet-graded`` — listwise softmax cross-entropy per page
  (ListNet top-1) on the same 3-tier relevance.

Groups are the *pages* the candidates arrived on — the even pages
keep their 16-candidate structure, the 8-candidate stream tail is one
group, and the wide-grammar stream is chunked into 16s.  Per-group
losses are averaged (a page counts once regardless of pair count).

Held-out protocol is identical to ``policy_baseline`` /
``neural_corpus``: odd pages ``gen_cases(seed, 36, page=16, start=1,
step=2)``, per-candidate-seeded verdicts memoized through a patched
``lp.evaluate``, AP on the flat 576-candidate pool, episodes with the
nets as drain-rankers (argmax over ``state.pending``, drain,
``stop``).  Verdicts are replayed from ``--cache`` (the
``neural_corpus`` checkpoint — this driver measures nothing new; a
miss is still measured under the same spawned-worker guard).

Usage::

    .venv/bin/python bench/drivers/policy/neural_rank.py \
        --json /tmp/nr.json --cache /tmp/neural_corpus_cache.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

import neural_corpus as nc
import policy_baseline as pb
import torch
import torch.nn.functional as F
from mlopt_core.features import compute_features
from mlopt_core.trajectories import rule_vector
from mlopt_discovery import open_game as og
from mlopt_discovery import play
from mlopt_discovery import proposal as lp
from mlopt_torch.learned_policy import (
    RuleValueNet,
    input_vector,
    train_rule_value,
)

SEED = 0
PAGE = 16
OPEN_PAGES = 36
OPEN_BUDGET = PAGE + 1
EPOCHS = 300
HIDDEN = 32
LR = 1e-2  # train_rule_value's default — identical optimizer recipe


# ---------------------------------------------------------------------------
#  Relevance grades
# ---------------------------------------------------------------------------


def _delta(o: Any) -> float:
    """``base_cost - cand_cost``; non-finite counts as no gain."""
    d = o.base_cost - o.cand_cost
    return d if math.isfinite(d) else 0.0


def _grades(outcomes: list, kind: str) -> list[float]:
    """Per-candidate relevance for one group, by label *kind*."""
    if kind == "delta":
        return [1.0 if _delta(o) > 0 else 0.0 for o in outcomes]
    if kind == "useful":
        return [1.0 if o.useful else 0.0 for o in outcomes]
    if kind == "graded":
        return [
            2.0 if o.useful else (1.0 if _delta(o) > 0 else 0.0)
            for o in outcomes
        ]
    raise ValueError(f"unknown grade kind {kind!r}")


# ---------------------------------------------------------------------------
#  Ranking losses + trainer (RuleValueNet, loss-only change)
# ---------------------------------------------------------------------------


def _pairwise_loss(s: torch.Tensor, y: torch.Tensor) -> Any:
    """RankNet logistic loss over ordered pairs within one group.

    ``mean softplus(-(s_i - s_j))`` over pairs with ``y_i > y_j`` —
    the score margin is a logit: pairs push apart in score space.
    ``None`` when the group has no discordant pair (uniform grades).
    """
    diff = s[:, None] - s[None, :]
    mask = y[:, None] > y[None, :]
    if not bool(mask.any()):
        return None
    return F.softplus(-diff[mask]).mean()


def _listnet_loss(s: torch.Tensor, y: torch.Tensor) -> Any:
    """ListNet top-1 loss: cross-entropy of softmax(s) vs softmax(y).

    ``None`` when the group is all-zero relevance (a uniform target
    would score every ordering identically — no signal).
    """
    if float(y.sum()) <= 0.0:
        return None
    return -(F.softmax(y, dim=0) * F.log_softmax(s, dim=0)).sum()


_GROUP_LOSS = {"pairwise": _pairwise_loss, "listnet": _listnet_loss}


def train_rule_rank(
    groups: list[list],
    group_outcomes: list[list],
    *,
    kind: str,
    loss: str,
    hidden: int = HIDDEN,
    epochs: int = EPOCHS,
    lr: float = LR,
    seed: int = SEED,
) -> Any:
    """Fit a ``RuleValueNet`` under a per-group ranking objective.

    Same architecture, input standardisation, optimizer, epoch count
    and seeding as ``train_rule_value`` — only the loss differs.
    *groups*/*group_outcomes* are aligned lists of pages; each page's
    candidates get grades by *kind* and the group loss is *loss*
    (``"pairwise"`` RankNet or ``"listnet"`` softmax).  Pages with no
    discordant pair contribute nothing.
    """
    torch.manual_seed(seed)
    xs: list[list[float]] = []
    idx: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for cands, outs in zip(groups, group_outcomes, strict=True):
        ix = []
        for c in cands:
            xs.append(
                input_vector(
                    compute_features(c.lhs), rule_vector(c.as_rule())
                )
            )
            ix.append(len(xs) - 1)
        idx.append(torch.tensor(ix))
        ys.append(torch.tensor(_grades(outs, kind)))
    x = torch.tensor(xs, dtype=torch.float32)
    model = RuleValueNet(hidden)
    with torch.no_grad():
        model.mean.copy_(x.mean(0))
        model.std.copy_(x.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = _GROUP_LOSS[loss]
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        s = model(x)
        terms = [
            t
            for t in (
                loss_fn(s[ix], y) for ix, y in zip(idx, ys, strict=True)
            )
            if t is not None
        ]
        total = torch.stack(terms).mean()
        total.backward()
        opt.step()
    return model.eval()


def _scores(model: Any, pool: list) -> list[float]:
    """Score every held-out pool candidate in one forward pass."""
    x = torch.tensor(
        [
            input_vector(
                compute_features(c.lhs), rule_vector(c.as_rule())
            )
            for c in pool
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return [float(v) for v in model(x)]


# ---------------------------------------------------------------------------
#  The run
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[neural_rank] {msg}", file=sys.stderr, flush=True)


def run(
    seed: int,
    half: int,
    page: int,
    *,
    wide: int,
    epochs: int,
    hidden: int,
    lr: float,
    skip_episodes: bool,
    cache: str | None,
) -> dict[str, Any]:
    """Rebuild the ondist+wide pools from the cache; train the arms."""
    t0 = time.time()
    store = nc._load_store(cache)
    nc._install_eval_limit()

    # --- held-out pool: odd pages, memoized verdicts ----------------
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    ev_pools = nc._stream_pool(ev)
    raw_pool = [c for pool in ev_pools for c in pool]
    banned = {nc._instance_key(c) for c in raw_pool}
    flat_raw = nc._measure(
        raw_pool, seed + pb.HELD_OUT_OFFSET, store, "heldout", cache
    )
    flat_pool, flat_eval, heldout_skipped = nc._kept(raw_pool, flat_raw)
    by_key: dict = {}
    flat_outcomes = dict(enumerate(flat_eval))
    for c, o in zip(flat_pool, flat_eval, strict=True):
        by_key.setdefault(pb._cand_key(c), o)
    n_useful = sum(1 for o in flat_eval if o.useful)

    # --- even-page train pool (keeps its page structure) ------------
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    train_pools = nc._stream_pool(train_cases)
    raw_train = [c for pool in train_pools for c in pool]
    train_raw = nc._measure(
        raw_train, seed + nc.TRAIN_SEED_OFFSET, store, "train", cache
    )
    banned |= {nc._instance_key(c) for c in raw_train}
    # Re-split the flat measured rows back into pages so the ranking
    # groups match the pages the candidates arrived on.
    base_groups: list[list] = []
    base_gout: list[list] = []
    pos = 0
    train_skipped = 0
    for pool in train_pools:
        n = len(pool)
        cands, outs, sk = nc._kept(pool, train_raw[pos : pos + n])
        pos += n
        train_skipped += sk
        if cands:
            base_groups.append(cands)
            base_gout.append(outs)
    base_cands = [c for g in base_groups for c in g]
    train_out_flat = [o for g in base_gout for o in g]

    # --- stream tail: one 8-candidate group -------------------------
    tail = nc._tail_candidates()
    tail_raw = nc._measure(tail, seed + 2_000_003, store, "tail", cache)
    banned |= {nc._instance_key(c) for c in tail}
    tail, tail_out, tail_skipped = nc._kept(tail, tail_raw)

    # --- wide grammar: stream-chunked into pages --------------------
    groups = list(base_groups)
    gouts = list(base_gout)
    wide_cands: list = []
    wide_out: list = []
    wide_skipped = 0
    if wide:
        _log(f"enumerating + measuring {wide} wide-grammar candidates")
        wide_cands = nc._wide_candidates(wide, banned)
        wide_raw = nc._measure(
            wide_cands, seed + 2_100_001, store, "wide", cache
        )
        wide_cands, wide_out, wide_skipped = nc._kept(
            wide_cands, wide_raw
        )
        for i in range(0, len(wide_cands), page):
            groups.append(wide_cands[i : i + page])
            gouts.append(wide_out[i : i + page])
    if tail:
        groups.append(tail)
        gouts.append(tail_out)

    flat_outcomes_train = [*train_out_flat, *tail_out, *wide_out]
    flat_samples = nc._samples(
        [*base_cands, *tail, *wide_cands],
        flat_outcomes_train,
    )

    pools = {
        "even-pages": {
            **nc._pool_stats(base_cands, train_out_flat),
            "skipped": train_skipped,
            "pages": len(base_groups),
        },
        "stream-tail": {
            **nc._pool_stats(tail, tail_out),
            "skipped": tail_skipped,
        },
        "wide-grammar": {
            **nc._pool_stats(wide_cands, wide_out),
            "skipped": wide_skipped,
        },
    }
    _log(
        f"pools rebuilt: {len(flat_samples)} rows in "
        f"{len(groups)} ranking groups"
    )

    # --- BCE control (must reproduce neural_corpus' 0.0595) ---------
    _log("training BCE control (ondist+wide)")
    bce = train_rule_value(
        flat_samples,
        epochs=epochs,
        hidden=hidden,
        device="cpu",
        seed=seed,
    )
    aps: dict[str, float] = {
        "bce-ondist+wide": round(
            pb._ap(_scores(bce, flat_pool), flat_outcomes), 4
        )
    }
    models: dict[str, Any] = {"bce": bce}
    model_scores: dict[str, list[float]] = {
        "bce-ondist+wide": _scores(bce, flat_pool)
    }

    # Pointwise control on the AP target itself — the BCE half of the
    # 2x2 (label x loss-form).  ``useful`` is ~0.5% of the pool, so the
    # fit needs the ``pos_weight`` rebalance or it collapses to zero.
    _log("training BCE on the useful label")
    useful_labels = [
        1.0 if o.useful else 0.0 for o in flat_outcomes_train
    ]
    n_pos = sum(useful_labels)
    bce_useful = train_rule_value(
        flat_samples,
        labels=useful_labels,
        pos_weight=(len(useful_labels) - n_pos) / max(n_pos, 1),
        epochs=epochs,
        hidden=hidden,
        device="cpu",
        seed=seed,
    )
    aps["bce-useful"] = round(
        pb._ap(_scores(bce_useful, flat_pool), flat_outcomes), 4
    )
    models["bce-useful"] = bce_useful
    model_scores["bce-useful"] = _scores(bce_useful, flat_pool)

    # --- ranking arms -----------------------------------------------
    for name, kind, loss in [
        ("rank-delta", "delta", "pairwise"),
        ("rank-useful", "useful", "pairwise"),
        ("rank-graded", "graded", "pairwise"),
        ("listnet-graded", "graded", "listnet"),
    ]:
        _log(f"training {name} ({loss}/{kind})")
        m = train_rule_rank(
            groups,
            gouts,
            kind=kind,
            loss=loss,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        aps[name] = round(
            pb._ap(_scores(m, flat_pool), flat_outcomes), 4
        )
        models[name] = m
        model_scores[name] = _scores(m, flat_pool)

    anchors = {
        "heuristic": round(
            pb._ap(
                [lp._heuristic_score(c) for c in flat_pool],
                flat_outcomes,
            ),
            4,
        ),
        "trained-linear": round(
            pb._ap(
                [
                    og.OpenLearnedPlayer(seed).rank_score(c)
                    for c in flat_pool
                ],
                flat_outcomes,
            ),
            4,
        ),
        "random": round(n_useful / max(len(flat_pool), 1), 4),
    }
    model_scores["heuristic"] = [
        lp._heuristic_score(c) for c in flat_pool
    ]

    # --- within-page rank diagnostic -------------------------------
    # The quantity the "ordering credit" hypothesis names: where does
    # each scorer place each useful candidate inside its own page?
    # (AP is pool-global; the episode reward only pays for in-page
    # order.)  flat_pool is the ev_pools concatenation, so page p is
    # positions [p*page, (p+1)*page) — valid only when no held-out row
    # was skipped (a drop would shift every later page boundary).
    page_rank: dict[str, dict[str, float]] = {}
    useful_ix = [i for i, o in flat_outcomes.items() if o.useful]
    for name, scores in (
        model_scores.items() if not heldout_skipped else {}
    ):
        ranks = []
        for i in useful_ix:
            lo = (i // page) * page
            hi = min(lo + page, len(flat_pool))
            # 1 + how many page-mates score strictly higher.
            r = 1 + sum(
                1
                for j in range(lo, hi)
                if j != i and scores[j] > scores[i]
            )
            ranks.append((i, r))
        page_rank[name] = {
            "mean": (
                round(sum(r for _, r in ranks) / len(ranks), 2)
                if ranks
                else 0.0
            ),
            "top1": sum(1 for _, r in ranks if r == 1),
            "top3": sum(1 for _, r in ranks if r <= 3),
            # pool index -> in-page rank (the index identifies which
            # useful candidate each rank belongs to — e.g. the
            # derivable one no policy can reach).
            "ranks": {
                str(i): r for i, r in sorted(ranks, key=lambda t: t[0])
            },
        }

    episodes: dict[str, dict] = {}
    if not skip_episodes:
        orig = lp.evaluate
        lp.evaluate = pb._patched_evaluate(by_key)
        try:
            rnd = play._random_policy(og.legal_actions)
            episodes["random"] = pb._play_open(
                lambda i: rnd(seed + 10_000 + i), ev, OPEN_BUDGET
            )
            episodes["heuristic"] = pb._play_open(
                pb._heuristic_arm(), ev, OPEN_BUDGET
            )
            baked = og.OpenLearnedPlayer(seed)
            episodes["trained-linear"] = pb._play_open(
                lambda i: baked.frozen(seed + 30_000 + i),
                ev,
                OPEN_BUDGET,
            )
            for name, m in models.items():
                episodes[f"neural-{name}"] = pb._play_open(
                    nc._neural_ranker_arm(m), ev, OPEN_BUDGET
                )
        finally:
            lp.evaluate = orig

    return {
        "held_out": {
            "pool": len(flat_pool),
            "useful": n_useful,
            "skipped": heldout_skipped,
        },
        "pools": pools,
        "groups": len(groups),
        "ap": aps,
        "anchors": anchors,
        "page_rank": page_rank,
        "episodes": episodes,
        "train_rows": len(flat_samples),
        "wall_s": round(time.time() - t0, 1),
    }


def _print(res: dict[str, Any]) -> None:
    """Render the AP table and the episode rows."""
    print(  # stdout-compat
        f"== held-out AP (pool={res['held_out']['pool']}, "
        f"useful={res['held_out']['useful']}, "
        f"groups={res['groups']}, rows={res['train_rows']}) =="
    )
    for name, ap in sorted(res["ap"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<18} {ap:.4f}")  # stdout-compat
    print("-- anchors --")  # stdout-compat
    for name, ap in res["anchors"].items():
        print(f"  {name:<18} {ap:.4f}")  # stdout-compat
    print("-- in-page rank of the 7 useful held-out candidates --")
    for name, pr in res["page_rank"].items():
        ranks = ",".join(f"{i}:{r}" for i, r in pr["ranks"].items())
        print(  # stdout-compat
            f"  {name:<18} mean={pr['mean']:>5.2f} top1={pr['top1']} "
            f"top3={pr['top3']}  idx:rank=[{ranks}]"
        )
    for name, r in res.get("episodes", {}).items():
        print(  # stdout-compat
            f"  ep {name:<22} reward={r['reward']:.4f} "
            f"use/ep={r['useful_per_ep']:.4f} "
            f"evals={r['evals_per_ep']:.2f} found={r['found']}"
        )


def main(argv: list[str] | None = None) -> int:
    """Train the ranking arms and print the AP table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument(
        "--wide",
        type=int,
        default=1728,
        help="widened-grammar candidates to add (0 disables)",
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--skip-episodes", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument(
        "--cache",
        type=str,
        default="/tmp/neural_corpus_cache.pt",
        help="neural_corpus verdict checkpoint (replayed, not rebuilt)",
    )
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    res = run(
        args.seed,
        args.open_pages,
        args.page,
        wide=args.wide,
        epochs=args.epochs,
        hidden=args.hidden,
        lr=args.lr,
        skip_episodes=args.skip_episodes,
        cache=args.cache,
    )
    _print(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "protocol": {
                        "open_pages": args.open_pages,
                        "open_page": args.page,
                        "open_held_out": "odd pages (start=1, step=2)",
                        "open_train": "even pages + tail + wide, paged",
                        "model": "RuleValueNet(hidden=32)",
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "oracle_pairing": (
                            "memoized per-candidate-seeded verdicts"
                        ),
                    },
                    "results": res,
                },
                f,
                indent=1,
                default=str,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
