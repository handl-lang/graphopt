"""The stopping policy — can the learned arm learn *when to quit*.

``docs/results_richer_features.md``'s flat axis: reward sits at
~0.27 across every schema arm because the trained prior drains
~15 of 16 evals — ordering sharpens (AP 0.133 -> 0.210) but the
``stop`` timing is untrained.  Round 1 (the ``open:stop`` feature
columns alone, 5 epochs) failed: reward *dropped* 0.2728 -> 0.2444
and the arm stopped *less* — the REINFORCE episode-return gradient
has no reason to isolate the ~0.03 of quitting headroom.  This
extension prices the two remaining mechanisms:

* **(a) features + more training** — ``inc+stop-long``: the same
  ``open:stop`` part at 4x the epochs (20), the undertraining
  hypothesis;
* **(b) an explicit stop head** — ``head`` / ``head+stop``: keep the
  ranking softmax for the ``evaluate`` picks; a *second* linear scorer
  over the spec's ``stop``-row features decides quit-vs-continue,
  trained supervised against the *oracle-optimal* label — "does the
  best continuation under the ranker's own greedy order still earn
  more than the step cost", computed exactly from the memoized
  verdicts — and threshold-calibrated on the train pages;
* **(b, bakeable)** — ``imitate-stop`` / ``imitate-cold``: fold the
  same oracle supervision into the single weight vector through
  ``engine.imitate_policy`` — a cross-entropy step toward the expert
  action, so the ``x:stop:*`` / ``op:stop`` weights can carry the
  timing (the only delivery that still serializes as one
  ``OPEN_GAME_PLAYER_WEIGHTS`` table).

The measurement mirrors round 1 exactly: cold retrain per arm
(``engine.train_policy`` under the baseline's memoized verdicts),
held-out odd pages, budget 17, page-aware AP (fresh-page
``w.featurize(evaluate)`` rows pooled — ``rank_score`` cannot see the
pool-relative columns).

Bake rule: the extended schema + weights join
``OPEN_GAME_PLAYER_WEIGHTS`` only if the stop arm's held-out
reward/ep AND page-aware AP both clear the incumbent bake
(0.273 / 0.210 — the richer-features `core4`/`all` rows).

Runtime: ~145 s one-time pool memoization, ~10-60 s per arm.

Usage::

    timeout 1800 .venv/bin/python lab/stop_policy.py \
        --json /tmp/stop.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import policy_baseline as pb
import schema_leaderboard as sl
import torch
from mlopt_discovery import engine, play
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import PARTS, DomainSpec, PlayHist, _dot

#: Defaults — the shipped bake protocol under memoized verdicts.
SEED = 0
PAGE = 16
OPEN_PAGES = 36  # the even half trains, the odd half evaluates
BUDGET = PAGE + 1
EPOCHS = 5  # the shipped bake used 5
LONG_EPOCHS = (
    20  # arm (a): 4x — was the failed arm simply undertrained?
)
IMITATE_PASSES = 6  # oracle-stop demonstration passes per imitate arm
HEAD_EPOCHS = 800  # full-batch logistic iterations for the stop head
_TAU_GRID = 25  # threshold-calibration points over the train scores
_SAMPLED_SEED = 40_000  # schema_leaderboard's eval-seed offsets
_GREEDY_SEED = 50_000
_CAL_SEED = 60_000  # the threshold-calibration decode offset

#: The arms — ``(id, extra part tags, kind, rl_epochs)``.
#:
#: * ``rl`` — the round-1 mechanism: REINFORCE only (the incumbent
#:   control, the failed ``inc+stop``, and its 4x-epochs variant);
#: * ``rl+imitate`` / ``imitate`` — oracle-stop supervision folded
#:   into the one weight vector (bakeable delivery of the head's
#:   signal: the expert evaluates in the ranker's greedy order while
#:   the memoized verdicts say continuing pays, then ``stop``s);
#: * ``head`` — the explicit second scorer: a logistic stop head over
#:   the spec's ``stop`` row + a train-calibrated threshold gates a
#:   ranker whose softmax never sees ``stop``.
ARMS: tuple[tuple[str, tuple[str, ...], str, int], ...] = (
    ("incumbent", (), "rl", EPOCHS),
    ("inc+stop", ("open:stop",), "rl", EPOCHS),
    ("inc+stop-long", ("open:stop",), "rl", LONG_EPOCHS),
    ("imitate-stop", ("open:stop",), "rl+imitate", EPOCHS),
    ("imitate-cold", ("open:stop",), "imitate", 0),
    ("head", (), "head", EPOCHS),
    ("head+stop", ("open:stop",), "head", EPOCHS),
)


def _extra_spec(tags: tuple[str, ...]) -> DomainSpec:
    """Incumbent parts + the extra *tags*; schema extended likewise."""
    parts = og.OPEN_PARTS + tuple(PARTS[t]() for t in tags)
    feats = og.OPEN_GAME_FEATURES + tuple(
        f
        for t in tags
        for f in og.OPEN_EXTRA_FEATURES[t]
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


def _page_scores(
    spec: DomainSpec, weights: dict[str, float], pools: list[list]
) -> list[float]:
    """Fresh-page ``w . featurize(evaluate(cand))`` per pool member."""
    scores: list[float] = []
    for pool in pools:
        st = sl._FixedArena(pool).observe()
        for i, c in enumerate(pool):
            f = spec.featurize(
                st, Action("evaluate", {"i": i, "cand": c})
            )
            scores.append(_dot(weights, f))
    return scores


def _play_pools(
    mk: Any, pools: list[list], budget: int
) -> dict[str, Any]:
    """``sl._play_open_pools`` plus the ``stop`` pick count."""
    found = evals = stops = 0
    reward = 0.0
    for i, pool in enumerate(pools):
        traj = engine.run_episode(sl._FixedArena(pool), mk(i), budget)
        reward += traj.total
        for r in traj.reports:
            evals += 1
            stops += r.action.op == "stop"
            if r.note and r.note.split(":", 1)[0] == "useful":
                found += 1
    n = len(pools)
    return {
        "reward": round(reward / n, 4),
        "useful_per_ep": round(found / n, 4),
        "useful_per_eval": round(found / max(evals, 1), 4),
        "evals_per_ep": round(evals / n, 3),
        "found": found,
        "stops": stops,
    }


# ---------------------------------------------------------------------------
#  Mechanism (b) — the oracle-priced stop signal
# ---------------------------------------------------------------------------
#
#  Both deliveries share one supervision source: under the memoized
#  verdicts the *exact* value of continuing is computable — evaluate
#  the pending set in the ranker's greedy order, sum the reward each
#  prefix banks (step cost + verdict tier + the ``1/steps`` ordering
#  credit a ``useful`` find collects at its real eval index), and the
#  episode should continue iff the best nonempty prefix nets > 0.
#  That is the "stop where expected-marginal-yield < step cost" rule
#  with the expectation replaced by the oracle — the cleanest signal
#  the stop axis can get under this protocol.


def _eval_only(state: Any) -> tuple:
    """``legal_actions`` minus ``stop`` — the gated ranker's pick set."""
    return tuple(
        a for a in og.legal_actions(state) if a.op == "evaluate"
    )


def _eval_order(
    spec: DomainSpec, w: dict[str, float], state: Any
) -> list[int]:
    """Pending indices in the ranker's greedy (descending-score) order."""

    def key(i: int) -> float:
        f = spec.featurize(
            state, Action("evaluate", {"i": i, "cand": state.pool[i]})
        )
        return _dot(w, f)

    return sorted(state.pending, key=key, reverse=True)


def _prefix_value(state: Any, order: list[int], by_key: dict) -> float:
    """Best-prefix value of draining *order* from *state* — oracle-priced.

    The episode should continue iff this is positive: every prefix is
    priced at the real reward composition (step cost + tier + the
    ordering credit a ``useful`` verdict collects at its actual eval
    index ``state.steps + j``).
    """
    best = cum = 0.0
    for j, i in enumerate(order, start=1):
        v = og._verdict(by_key[pb._cand_key(state.pool[i])])
        cum += og.OPEN_GAME_REWARD["step"] + og.OPEN_GAME_REWARD[v]
        if v == "useful":
            cum += og.OPEN_GAME_EFFICIENCY / (state.steps + j)
        if cum > best:
            best = cum
    return best


def _oracle_expert(
    spec: DomainSpec, w: dict[str, float], by_key: dict
) -> Any:
    """Expert factory for ``imitate_policy``: ranker's order, oracle stop.

    Evaluates the top-ranked still-pending candidate while the best
    continuation prefix nets > 0 under the memoized verdicts, then
    plays ``stop`` — the joint policy the stop axis wants.
    """

    def make(_i: int) -> Any:
        def expert(state: Any) -> Action:
            if not state.pending:
                return Action("stop", {})
            order = _eval_order(spec, w, state)
            if _prefix_value(state, order, by_key) <= 0.0:
                return Action("stop", {})
            i = order[0]
            return Action("evaluate", {"i": i, "cand": state.pool[i]})

        return expert

    return make


def _collect_head_rows(
    spec: DomainSpec,
    w: dict[str, float],
    pools: list[list],
    by_key: dict,
    budget: int,
) -> tuple[list[dict], list[bool]]:
    """``(stop_row, continue_label)`` pairs along greedy drain traces.

    Every prefix of the ranker's greedy order is a visited state, so
    one drain per page covers the whole pending-count range — early
    states label ``True`` (a paying prefix remains), late states
    ``False`` (only duds left).  The features are the spec's ``stop``
    row itself, so a head over them prices exactly the quantity the
    softmax could not isolate.
    """
    rows: list[dict] = []
    labels: list[bool] = []
    for pool in pools:
        arena = sl._FixedArena(pool)
        hist = PlayHist()
        for _ in range(budget):
            state = arena.observe()
            if state.done or not state.pending:
                break
            rows.append(spec.featurize(state, Action("stop", {}), hist))
            order = _eval_order(spec, w, state)
            labels.append(_prefix_value(state, order, by_key) > 0.0)
            i = order[0]
            act = Action("evaluate", {"i": i, "cand": state.pool[i]})
            arena.step(act)
            hist.record(act.op)
    return rows, labels


def _fit_head(
    spec: DomainSpec,
    rows: list[dict],
    labels: list[bool],
    seed: int,
    epochs: int = HEAD_EPOCHS,
) -> tuple[dict[str, float], float]:
    """Logistic stop head over the ``stop``-row features -> weight dict.

    Full-batch BCE on standardized columns (constant columns pass
    through raw — ``bias`` stays the intercept), then the solution is
    mapped back to raw feature units so :func:`_dot` scores the row
    directly.  Returns ``(w, final_loss)``.
    """
    names = list(spec.features)
    x = torch.tensor(
        [[r.get(n, 0.0) for n in names] for r in rows],
        dtype=torch.float64,
    )
    y = torch.tensor(
        [float(lab) for lab in labels], dtype=torch.float64
    )
    mu0, sd0 = x.mean(0), x.std(0, unbiased=False)
    const = sd0 < 1e-9
    mu = torch.where(const, torch.zeros_like(mu0), mu0)
    sd = torch.where(const, torch.ones_like(sd0), sd0)
    xz = (x - mu) / sd
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(len(names), generator=g, dtype=torch.float64) * 0.01
    v.requires_grad_()
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([v, b], lr=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    loss = torch.tensor(0.0, dtype=torch.float64)
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(xz @ v + b, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        w = {n: float(v[i]) / float(sd[i]) for i, n in enumerate(names)}
        # fold the intercept: score = v·((x-mu)/sd) + b
        w["bias"] = (
            w.get("bias", 0.0)
            + float(b)
            - sum(
                float(v[i]) * float(mu[i]) / float(sd[i])
                for i in range(len(names))
            )
        )
    return (
        {n: wv for n, wv in w.items() if abs(wv) > 1e-9},
        round(float(loss.detach()), 4),
    )


class _StopGate:
    """The explicit stop head — mechanism (b) as a player wrapper.

    The inner ranker (``learn=False``, ``legal`` = ``evaluate`` rows
    only) picks *which* candidate; this wrapper decides *whether* to
    keep spending: ``stop`` iff ``w_head . phi_stop(state) >= tau``
    where ``phi_stop`` is the spec's ``stop``-row features under the
    episode's own play history.  ``stop``'s semantics are untouched —
    the action is the board's own, played at head-chosen times.
    """

    def __init__(
        self,
        inner: Any,
        spec: DomainSpec,
        w_head: dict[str, float],
        tau: float,
    ) -> None:
        """Bind the ranker, the head weights and the threshold."""
        self._inner = inner
        self._spec = spec
        self._w = w_head
        self._tau = tau
        self._hist = PlayHist()
        self._prev = -1

    def __call__(self, state: Any) -> Any:
        """Stop when the head scores quitting above the threshold."""
        if state.steps == 0 and self._prev > 0:
            self._hist = PlayHist()
        self._prev = state.steps
        if not state.pending:
            return Action("stop", {})
        row = self._spec.featurize(
            state, Action("stop", {}), self._hist
        )
        if _dot(self._w, row) >= self._tau:
            self._hist.record("stop")
            return Action("stop", {})
        act = self._inner(state)
        if act is None:
            return Action("stop", {})
        self._hist.record(act.op)
        return act


def _mk_gate(
    player: Any,
    spec: DomainSpec,
    w_head: dict[str, float],
    tau: float,
    seed: int,
    *,
    greedy: bool = False,
) -> _StopGate:
    """Build one gated player: frozen ranker (eval-only) + the head."""
    inner = og.OpenLearnedPlayer(
        seed=seed,
        spec=spec,
        weights=player.weights_dict(),
        legal=_eval_only,
        learn=False,
        greedy=greedy,
    )
    return _StopGate(inner, spec, w_head, tau)


def _calibrate_tau(
    spec: DomainSpec,
    player: Any,
    w_head: dict[str, float],
    rows: list[dict],
    pools: list[list],
    budget: int,
    seed: int,
) -> tuple[float, float]:
    """Pick the stop threshold maximizing train reward — greedy decode.

    The grid spans the head's score range on the collected decision
    rows; each candidate tau is priced by actually playing the gated
    arm over the train pools (identical memoized verdicts), so the
    threshold is calibrated on the metric it must earn.
    """
    scores = sorted(_dot(w_head, r) for r in rows)
    lo, hi = scores[0], scores[-1]
    best_tau, best_reward = lo, -1e9
    for k in range(_TAU_GRID):
        tau = lo + (hi - lo) * k / max(_TAU_GRID - 1, 1)
        total = 0.0
        for i, pool in enumerate(pools):
            gate = _mk_gate(
                player,
                spec,
                w_head,
                tau,
                seed + _CAL_SEED + i,
                greedy=True,
            )
            traj = engine.run_episode(
                sl._FixedArena(pool), gate, budget
            )
            total += traj.total
        r = total / len(pools)
        if r > best_reward:
            best_reward, best_tau = r, tau
    return best_tau, round(best_reward, 4)


def run(
    seed: int, half: int, page: int, budget: int, epochs: int
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
    _t, train_keys = pb._measure_pools(train_pools, seed + 1_000_003)
    by_key.update(train_keys)
    del _t

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1
    measure_s = time.time() - t0

    rows_out: list[dict] = []
    curves: dict[str, list[float]] = {}
    weights: dict[str, dict] = {}
    heads: dict[str, dict] = {}
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
                        _page_scores(
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

        for arm_i, (arm_id, tags, kind, n_epochs) in enumerate(ARMS):
            t1 = time.time()
            spec = _extra_spec(tags)
            player = og.OpenLearnedPlayer(
                seed=seed, spec=spec, weights={}
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
            if kind in ("rl+imitate", "imitate"):
                for _ in range(IMITATE_PASSES):
                    w_rank = player.weights_dict()
                    engine.imitate_policy(
                        player,
                        _oracle_expert(spec, w_rank, by_key),
                        train_pools,
                        sl._FixedArena,
                        budget=budget,
                    )
                w_rank = player.weights_dict()
            if kind == "head":
                hrows, hlabels = _collect_head_rows(
                    spec, w_rank, train_pools, by_key, budget
                )
                w_head, hloss = _fit_head(spec, hrows, hlabels, seed)
                tau, cal_reward = _calibrate_tau(
                    spec,
                    player,
                    w_head,
                    hrows,
                    train_pools,
                    budget,
                    seed,
                )
                heads[arm_id] = {
                    "tau": round(tau, 4),
                    "cal_reward": cal_reward,
                    "loss": hloss,
                    "n_rows": len(hrows),
                    "pos_rate": round(
                        sum(hlabels) / max(len(hlabels), 1), 4
                    ),
                    "weights": {
                        n: round(wv, 4)
                        for n, wv in w_head.items()
                        if abs(wv) > 1e-4
                    },
                }
                sampled = _play_pools(
                    lambda i, p=player, s=spec, wh=w_head, ta=tau: (
                        _mk_gate(p, s, wh, ta, seed + _SAMPLED_SEED + i)
                    ),
                    ev_pools,
                    budget,
                )
                greedy = _play_pools(
                    lambda i, p=player, s=spec, wh=w_head, ta=tau: (
                        _mk_gate(
                            p,
                            s,
                            wh,
                            ta,
                            seed + _GREEDY_SEED + i,
                            greedy=True,
                        )
                    ),
                    ev_pools,
                    budget,
                )
            else:
                sampled = _play_pools(
                    lambda i, p=player: p.frozen(
                        seed + _SAMPLED_SEED + i
                    ),
                    ev_pools,
                    budget,
                )
                greedy = _play_pools(
                    lambda i, p=player: p.frozen(
                        seed + _GREEDY_SEED + i, greedy=True
                    ),
                    ev_pools,
                    budget,
                )
            ap = round(
                pb._ap(
                    _page_scores(spec, w_rank, ev_pools),
                    flat_outcomes,
                ),
                4,
            )
            rows_out.append(
                {
                    "id": arm_id,
                    "kind": kind,
                    "extra_parts": list(tags),
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
            print(  # stdout-compat
                f"  [{arm_i + 1}/{len(ARMS)}] {arm_id:<16} "
                f"reward={sampled['reward']:>8.4f} "
                f"greedy={greedy['reward']:>8.4f} "
                f"evals={sampled['evals_per_ep']:>5.2f} "
                f"ap={ap:.4f} ({time.time() - t1:.0f}s)"
            )
    finally:
        lp.evaluate = orig_evaluate

    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": rows_out,
        "anchors": anchors,
        "train_curve": curves,
        "weights": weights,
        "heads": heads,
        "pool": {"held_out": len(flat_pool), "useful": n_useful},
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "epochs": epochs,
            "long_epochs": LONG_EPOCHS,
            "imitate_passes": IMITATE_PASSES,
            "head_epochs": HEAD_EPOCHS,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def _print(res: dict[str, Any]) -> None:
    """Render the arm table."""
    head = (
        f"{'arm':<16} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
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
            f"loss={h['loss']} --"
        )
        for n, wv in sorted(
            h["weights"].items(), key=lambda kv: -abs(kv[1])
        ):
            print(f"    {n:<22} {wv:>9.4f}")  # stdout-compat


def main(argv: list[str] | None = None) -> int:
    """Run the stopping-policy measurement and print the table."""
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
