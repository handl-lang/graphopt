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
  verdicts — and threshold-calibrated on the train pages.  (Round 4
  corrected the gate: it is ``stop iff w.phi <= tau`` — the head is
  fit on *continue* labels, so the round-2/3 ``>= tau`` reading
  inverted the rule and could only ever calibrate to never-stop or
  stop-on-confident-continue);
* **(b, bakeable)** — ``imitate-stop`` / ``imitate-cold``: fold the
  same oracle supervision into the single weight vector through
  ``engine.imitate_policy`` — a cross-entropy step toward the expert
  action, so the ``x:stop:*`` / ``op:stop`` weights can carry the
  timing (the only delivery that still serializes as one
  ``OPEN_GAME_PLAYER_WEIGHTS`` table).

Round 3 adds the missing variable round 2 diagnosed — *next-candidate
value*: the ``open:next-value`` part emits ``x:stop:best_score`` /
``x:stop:topk_score`` / ``x:stop:rank1_pred`` /
``x:stop:remaining_frac`` / ``x:stop:decay`` on the ``stop`` row — the
heuristic ordering's own pricing of what the pending pool still
offers (the policy's next-pick probability is what
``x:stop:rank1_pred`` wants, but a ``RelFeats`` part cannot see the
weights; it ships the linear-score softmax proxy).  The arms rerun
the round-1/2 mechanisms on the enriched stop row — cold RL at the
bake recipe (``inc+nextval``, ``inc+stop+nextval``), the 20-epoch
undertraining check (``inc+nextval-long`` — does the AP collapse
repeat), and the supervised gate retrained over the new columns
(``head+nextval``, ``head+stop+nextval``) — plus a per-feature probe:
each new column's solo AP against the oracle continue-labels and its
leave-one-out cost in the refit head, scored on held-out decision
rows.

Round 4 closes the loop round 3 opened: the value columns were still
the *heuristic's* view (candidate AP ~0.045) — the missing evidence
is a per-candidate "will it still pay" answer, which is exactly
``RuleValueNet``'s job.  This round trains it on-distribution (the
``bench/policy/policy_baseline.py`` neural path: ``RuleSample`` rows over the
even-page candidates' memoized verdicts, ``train_rule_value`` BCE
300 epochs hidden 32), injects sigmoid predictions into the new
``open:next-val`` part (``x:stop:next_value`` — the max predicted
value still pending; ``x:stop:pool_value`` — the remaining expected
yield; ``x:stop:value_frac`` — its share of the page's starting
value), and reruns the two live mechanisms: cold RL at the bake
recipe (``inc+nv``, ``inc+stop+nv``) and the supervised gate
(``head+nv``, ``head+stop+nv``).  The head arms hold the *ranker* at
the incumbent schema (round-4-only protocol: ``rank_tags=()``) — the
arm prices the stop decision, and round 3 showed RL over
stop-varying columns can collapse the shared ranking (AP 0.0215), a
confound outside the mechanism under test.

Round 5 prices the label gap round 4 closed on and the in-sample
optimism it flagged.  The same ``RuleValueNet`` is retrained on the
*verdict* labels — ``useful`` (truth ∧ strictly-cost-lower ∧
novelty, ~1.2% positive — the balanced ``pos_weight`` keeps BCE off
the all-zero point), ``pays`` (``useful|true`` — the tiers the stop
decision actually prices, ~4.5%), ``delta`` kept as the control —
and ``--value-label`` picks which net's sigmoids feed
``set_value_preds``.  Train-page preds are now **out-of-fold**
(``OOF_FOLDS`` nets, each page scored by a net that never saw it) so
the τ calibration and head-row collection see held-out-quality
signal, not memorized near-oracle preds — round 4's ``head+nv``
over-stop is re-priced under honest calibration.  Every head arm
also ships a **fold** variant: the head's stop-only columns +
calibrated ``op:stop`` intercept as an ordinary weight table (the
only delivery the gate's timing can serialize into
``OPEN_GAME_PLAYER_WEIGHTS``).

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

    timeout 1800 .venv/bin/python bench/policy/stop_policy.py \
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
from mlopt_core.features import compute_features
from mlopt_core.trajectories import RuleSample, rule_vector
from mlopt_discovery import engine, play
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import PARTS, DomainSpec, PlayHist, _dot
from mlopt_torch.learned_policy import input_vector, train_rule_value

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
NEURAL_EPOCHS = 300  # ``train_rule_value`` — policy_baseline's recipe
NEURAL_HIDDEN = 32
_TAU_GRID = 25  # threshold-calibration points over the train scores
_FOLD_C_LO, _FOLD_C_HI = -6.0, 4.0  # the fold's ``op:stop`` grid
OOF_FOLDS = 6  # predictor folds — out-of-fold rows for calibration
_SAMPLED_SEED = 40_000  # schema_leaderboard's eval-seed offsets
_GREEDY_SEED = 50_000
_CAL_SEED = 60_000  # the threshold-calibration decode offset

#: Round 5 — the per-candidate labels the value predictor can be
#: trained on.  ``delta`` is the round-4 label (``delta_cost > 0`` —
#: the shipped ``train_rule_value`` BCE, reported as the control);
#: ``useful`` is the real verdict — truth ∧ strictly-cost-lower ∧
#: novelty, i.e. ``_verdict == "useful"`` (~1.2% positive — the
#: balanced ``pos_weight`` keeps the fit off the all-zero point);
#: ``pays`` is the stop decision's priced quantity — every verdict
#: tier that out-earns its step cost (``useful`` +1.0 and ``true``
#: +0.2 vs -0.002), ~4.5% positive.  ``--value-label`` picks which
#: net feeds ``set_value_preds``; all three report held-out APs.
VALUE_LABELS: dict[str, Any] = {
    "delta": lambda o: float(o.base_cost > o.cand_cost),
    "useful": lambda o: float(og._verdict(o) == "useful"),
    "pays": lambda o: float(og._verdict(o) in ("useful", "true")),
}

#: The arms — ``(id, head/stop-row part tags, kind, rl_epochs,
#: rank_tags)``.  ``rank_tags`` is read by ``head`` arms only: it names
#: the part tags the *ranker* is trained on — equal to the head's tags
#: in rounds 2-3 (one shared spec), ``()`` in round 4 so the value
#: signal's stop-decision test is not hostage to the RL corruption of
#: the shared ranking the round-3 nextval arms showed (AP collapses).
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
ARMS: tuple[
    tuple[str, tuple[str, ...], str, int, tuple[str, ...]], ...
] = (
    ("incumbent", (), "rl", EPOCHS, ()),
    ("inc+stop", ("open:stop",), "rl", EPOCHS, ()),
    ("inc+stop-long", ("open:stop",), "rl", LONG_EPOCHS, ()),
    ("imitate-stop", ("open:stop",), "rl+imitate", EPOCHS, ()),
    ("imitate-cold", ("open:stop",), "imitate", 0, ()),
    ("head", (), "head", EPOCHS, ()),
    ("head+stop", ("open:stop",), "head", EPOCHS, ("open:stop",)),
    # Round 3 — the next-candidate-value family, same mechanisms.
    ("inc+nextval", ("open:next-value",), "rl", EPOCHS, ()),
    (
        "inc+stop+nextval",
        ("open:stop", "open:next-value"),
        "rl",
        EPOCHS,
        (),
    ),
    ("inc+nextval-long", ("open:next-value",), "rl", LONG_EPOCHS, ()),
    (
        "head+nextval",
        ("open:next-value",),
        "head",
        EPOCHS,
        ("open:next-value",),
    ),
    (
        "head+stop+nextval",
        ("open:stop", "open:next-value"),
        "head",
        EPOCHS,
        ("open:stop", "open:next-value"),
    ),
    # Round 4 — the RuleValueNet value signal.  The head arms hold the
    # ranker at the incumbent schema (``rank_tags=()``): the mechanism
    # under test is the stop decision, and the round-3 shared-spec
    # collapse is a confound outside it.  ``imitate-nv`` is the only
    # bakeable delivery of a working stop rule (the folded weights
    # serialize); ``head+nvh`` is the ablation — the round-3
    # heuristic-priced columns under the same incumbent ranker — so
    # the predictor's contribution reads arm-to-arm.
    ("inc+nv", ("open:next-val",), "rl", EPOCHS, ()),
    (
        "inc+stop+nv",
        ("open:stop", "open:next-val"),
        "rl",
        EPOCHS,
        (),
    ),
    ("imitate-nv", ("open:next-val",), "rl+imitate", EPOCHS, ()),
    ("head+nv", ("open:next-val",), "head", EPOCHS, ()),
    ("head+nvh", ("open:next-value",), "head", EPOCHS, ()),
    (
        "head+stop+nv",
        ("open:stop", "open:next-val"),
        "head",
        EPOCHS,
        (),
    ),
)

#: The next-candidate-value columns — the round-3 probe reads them.
NEXT_VALUE_FEATURES: tuple[str, ...] = og.OPEN_EXTRA_FEATURES[
    "open:next-value"
]

#: The predicted-value columns — the round-4 probe reads them.
NEXT_VAL_FEATURES: tuple[str, ...] = og.OPEN_EXTRA_FEATURES[
    "open:next-val"
]

#: Every stop-row column family the per-feature probe knows.
PROBE_FEATURES: tuple[str, ...] = (
    NEXT_VALUE_FEATURES + NEXT_VAL_FEATURES
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
    rank_spec: DomainSpec | None = None,
) -> tuple[list[dict], list[bool]]:
    """``(stop_row, continue_label)`` pairs along greedy drain traces.

    Every prefix of the ranker's greedy order is a visited state, so
    one drain per page covers the whole pending-count range — early
    states label ``True`` (a paying prefix remains), late states
    ``False`` (only duds left).  The features are the spec's ``stop``
    row itself, so a head over them prices exactly the quantity the
    softmax could not isolate.  *rank_spec* supplies the order's
    featurizer when it differs from the head's schema (round 4's
    incumbent-ranker arms); default ``spec`` is the shared-spec
    protocol of rounds 2-3.
    """
    rs = spec if rank_spec is None else rank_spec
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
            order = _eval_order(rs, w, state)
            labels.append(_prefix_value(state, order, by_key) > 0.0)
            i = order[0]
            act = Action("evaluate", {"i": i, "cand": state.pool[i]})
            arena.step(act)
            hist.record(act.op)
    return rows, labels


def _ap_binary(scores: list[float], labels: list[bool]) -> float:
    """Average precision of *scores* against binary *labels*.

    The decision-row analogue of ``pb._ap``: the oracle
    continue-labels are booleans, not ``Outcome`` objects, so the
    AP sum is over ``labels[i]`` hits directly.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    hits = 0
    total = 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            hits += 1
            total += hits / rank
    n_pos = sum(1 for lab in labels if lab)
    return total / n_pos if n_pos else 0.0


def _fit_head(
    spec: DomainSpec,
    rows: list[dict],
    labels: list[bool],
    seed: int,
    epochs: int = HEAD_EPOCHS,
    names: tuple[str, ...] | None = None,
) -> tuple[dict[str, float], float]:
    """Logistic stop head over the ``stop``-row features -> weight dict.

    Full-batch BCE on standardized columns (constant columns pass
    through raw — ``bias`` stays the intercept), then the solution is
    mapped back to raw feature units so :func:`_dot` scores the row
    directly.  *names* restricts the fit to a schema subset — the
    leave-one-out probe refits without one column.  Returns
    ``(w, final_loss)``.
    """
    if names is None:
        names = spec.features
    names = list(names)
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
    keep spending: ``stop`` iff ``w_head . phi_stop(state) <= tau``
    where ``phi_stop`` is the spec's ``stop``-row features under the
    episode's own play history.  ``stop``'s semantics are untouched —
    the action is the board's own, played at head-chosen times.

    The comparison direction was ``>=`` through round 3 — an
    inversion: the head is fit on *continue* labels (high score =
    "draining still pays"), so ``stop iff score >= tau`` quits first
    exactly where continuing is worth most.  Under a perfect head the
    only non-degenerate calibrated point is the never-stop end — the
    rounds-2/3 "the reward-optimal tau is never-stop" result was
    forced by construction, not measured.  ``<=`` is the correct
    reading: quit when the head's continue-score drops below the
    threshold.
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
        """Stop when the continue-score drops to the threshold."""
        if state.steps == 0 and self._prev > 0:
            self._hist = PlayHist()
        self._prev = state.steps
        if not state.pending:
            return Action("stop", {})
        row = self._spec.featurize(
            state, Action("stop", {}), self._hist
        )
        if _dot(self._w, row) <= self._tau:
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
    rank_spec: DomainSpec | None = None,
) -> _StopGate:
    """Build one gated player: frozen ranker (eval-only) + the head.

    The ranker binds *rank_spec* (default *spec* — the shared-schema
    protocol); the head reads the stop row under *spec*.
    """
    inner = og.OpenLearnedPlayer(
        seed=seed,
        spec=spec if rank_spec is None else rank_spec,
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
    rank_spec: DomainSpec | None = None,
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
                rank_spec=rank_spec,
            )
            traj = engine.run_episode(
                sl._FixedArena(pool), gate, budget
            )
            total += traj.total
        r = total / len(pools)
        if r > best_reward:
            best_reward, best_tau = r, tau
    return best_tau, round(best_reward, 4)


def _fold_weights(
    spec: DomainSpec,
    w_rank: dict[str, float],
    w_head: dict[str, float],
    c: float,
) -> dict[str, float]:
    """Fold the stop head into one weight table — the bakeable arm.

    The gate is ``stop iff w_head . phi_stop <= tau``.  Only features
    emitted *exclusively* on the ``stop`` row can carry that rule in a
    shared weight table — broadcast columns (``st:*``, the play-rate
    history) shift every logit together and cancel in the softmax.
    The fold keeps the ranker weights, negates the head's weights on
    the stop-only ``x:stop:*`` columns (stop logit falls as the head's
    continue-score rises), and lets ``op:stop`` carry the calibrated
    intercept *c*.  The result is an ordinary spec-sized
    ``{feature: weight}`` table — the shape
    ``OPEN_GAME_PLAYER_WEIGHTS`` serializes — whose softmax argmax
    implements ``stop iff -w_head.phi_xstop + c > max evaluate-logit``:
    a *relative* threshold (the best pending eval logit moves it),
    not the gate's absolute one — that gap is what this arm prices.
    """
    w = dict(w_rank)
    for f in spec.features:
        if f.startswith("x:stop:"):
            w[f] = w.get(f, 0.0) - w_head.get(f, 0.0)
    w["op:stop"] = c
    return w


def _calibrate_fold_c(
    spec: DomainSpec,
    w_rank: dict[str, float],
    w_head: dict[str, float],
    pools: list[list],
    budget: int,
    seed: int,
) -> tuple[float, float]:
    """Pick the fold's ``op:stop`` intercept by train reward — greedy."""
    best_c, best_reward = _FOLD_C_LO, -1e9
    for k in range(_TAU_GRID):
        c = _FOLD_C_LO + (_FOLD_C_HI - _FOLD_C_LO) * k / max(
            _TAU_GRID - 1, 1
        )
        w = _fold_weights(spec, w_rank, w_head, c)
        total = 0.0
        for i, pool in enumerate(pools):
            pl = og.OpenLearnedPlayer(
                seed=seed + _CAL_SEED + 500 + i,
                spec=spec,
                weights=w,
                learn=False,
                greedy=True,
            )
            traj = engine.run_episode(sl._FixedArena(pool), pl, budget)
            total += traj.total
        r = total / len(pools)
        if r > best_reward:
            best_reward, best_c = r, c
    return best_c, round(best_reward, 4)


def run(
    seed: int,
    half: int,
    page: int,
    budget: int,
    epochs: int,
    value_label: str = "useful",
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

    # Round 5 — the verdict-label predictors.  Round 4's closing
    # diagnosis named the label gap as the binding constraint: the
    # same ``RuleValueNet`` nails its own ``delta > 0`` training
    # label (held-out AP 0.93) and misses ``useful`` (0.028).  This
    # round trains the identical net shape on the *verdict* labels
    # (``VALUE_LABELS``): ``useful`` — the real verdict — is ~1.2%
    # positive (7/576), so the fit carries the balanced
    # ``pos_weight`` (``train_rule_value``'s ``labels``/``pos_weight``
    # seam, added for this round); ``pays`` — useful|true — is the
    # stop decision's actually-priced quantity (~4.5%).  Every net
    # reports held-out AP vs *each* label; ``value_label`` picks the
    # net whose sigmoid preds feed ``set_value_preds``.
    t_net = time.time()
    flat_train = [c for pool in train_pools for c in pool]
    flat_train_outcomes = [o for outs in train_outcomes for o in outs]
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
    all_cands = flat_train + flat_pool
    x_net = torch.tensor(
        [
            input_vector(
                compute_features(c.lhs), rule_vector(c.as_rule())
            )
            for c in all_cands
        ],
        dtype=torch.float32,
    )
    x_ev = x_net[len(flat_train) :]
    flat_ordered = [flat_outcomes[i] for i in range(len(flat_pool))]
    ev_labels = {
        n: [bool(lf(o)) for o in flat_ordered]
        for n, lf in VALUE_LABELS.items()
    }
    nets: dict[str, Any] = {}
    predictor: dict[str, Any] = {
        "injected": value_label,
        "train_samples": len(samples),
        "epochs": NEURAL_EPOCHS,
        "hidden": NEURAL_HIDDEN,
        "oof_folds": OOF_FOLDS,
        "nets": {},
    }
    for lname, lf in VALUE_LABELS.items():
        ys = [lf(o) for o in flat_train_outcomes]
        n_pos = int(sum(ys))
        # ``delta`` replicates round 4 exactly — the shipped
        # ``binary=True`` path (no explicit labels, no pos_weight).
        # Rare-positive labels get the balanced weight.
        if lname == "delta":
            net = train_rule_value(
                samples,
                epochs=NEURAL_EPOCHS,
                hidden=NEURAL_HIDDEN,
                device="cpu",
                seed=seed,
            )
            pw = 1.0
        else:
            pw = (len(ys) - n_pos) / max(n_pos, 1)
            net = train_rule_value(
                samples,
                epochs=NEURAL_EPOCHS,
                hidden=NEURAL_HIDDEN,
                device="cpu",
                seed=seed,
                labels=ys,
                pos_weight=pw,
            )
        nets[lname] = net
        with torch.no_grad():
            logits = net(x_ev)
        probs = torch.sigmoid(logits)
        score_p = list(map(float, probs))
        score_l = list(map(float, logits))
        predictor["nets"][lname] = {
            "train_positives": n_pos,
            "pos_weight": round(pw, 2),
            "ap_useful": round(pb._ap(score_p, flat_outcomes), 4),
            **{
                f"ap_{n}_logit": round(
                    _ap_binary(score_l, ev_labels[n]), 4
                )
                for n in VALUE_LABELS
            },
            # in-sample train AP — the memorization read
            "ap_train_logit": round(
                _ap_binary(
                    list(
                        map(
                            float,
                            net(x_net[: len(flat_train)]).detach(),
                        )
                    ),
                    [bool(v) for v in ys],
                ),
                4,
            ),
        }
    # Out-of-fold train preds for the *injected* label: 300 epochs on
    # ~576 rows memorizes (in-sample AP ≈ 1.0), so in-sample sigmoids
    # would show the τ calibration and the head-row collection a
    # near-oracle signal held-out pages do not have — round 4 priced
    # exactly that optimism (``head+nv`` over-stopped: 10.78 evals,
    # -0.028).  Each train page's preds come from a net trained on
    # the other folds — same label, same recipe, held-out quality.
    lf_inj = VALUE_LABELS[value_label]
    ys_inj = [lf_inj(o) for o in flat_train_outcomes]
    n_pos_inj = int(sum(ys_inj))
    pw_inj = (len(ys_inj) - n_pos_inj) / max(n_pos_inj, 1)
    fpp = max(len(train_pools) // OOF_FOLDS, 1)  # pages per fold

    def _fold_of(i: int) -> int:
        return (i // page) // fpp

    oof: dict = {}
    for f in range(OOF_FOLDS):
        fsamples = [
            s for i, s in enumerate(samples) if _fold_of(i) != f
        ]
        fys = [y for i, y in enumerate(ys_inj) if _fold_of(i) != f]
        held = [c for i, c in enumerate(flat_train) if _fold_of(i) == f]
        if value_label == "delta":
            fnet = train_rule_value(
                fsamples,
                epochs=NEURAL_EPOCHS,
                hidden=NEURAL_HIDDEN,
                device="cpu",
                seed=seed + 7 + f,
            )
        else:
            fnet = train_rule_value(
                fsamples,
                epochs=NEURAL_EPOCHS,
                hidden=NEURAL_HIDDEN,
                device="cpu",
                seed=seed + 7 + f,
                labels=fys,
                pos_weight=pw_inj,
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
    with torch.no_grad():
        probs_all = torch.sigmoid(nets[value_label](x_net))
    preds = {
        og._value_key(c): float(p)
        for c, p in zip(all_cands, probs_all, strict=True)
    }
    preds.update(oof)  # train candidates get held-out-quality preds
    og.set_value_preds(preds)
    predictor["seconds"] = round(time.time() - t_net, 1)

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

        for arm_i, (
            arm_id,
            tags,
            kind,
            n_epochs,
            rank_tags,
        ) in enumerate(ARMS):
            t1 = time.time()
            spec = _extra_spec(tags)
            rank_spec = (
                _extra_spec(rank_tags) if kind == "head" else spec
            )
            player = og.OpenLearnedPlayer(
                seed=seed, spec=rank_spec, weights={}
            )
            fold_sampled = fold_greedy = None
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
                    spec,
                    w_rank,
                    train_pools,
                    by_key,
                    budget,
                    rank_spec=rank_spec,
                )
                erows, elabels = _collect_head_rows(
                    spec,
                    w_rank,
                    ev_pools,
                    by_key,
                    budget,
                    rank_spec=rank_spec,
                )
                w_head, hloss = _fit_head(spec, hrows, hlabels, seed)
                eval_ap = _ap_binary(
                    [_dot(w_head, r) for r in erows], elabels
                )
                # Round-3/4 probe — which stop-row column carries the
                # timing: solo AP of the raw column, and the AP cost
                # of dropping it from a refit head, both scored on
                # the held-out decision rows.  Arms whose schema lacks
                # the family probe nothing — an absent column is a
                # constant, not a feature.
                probe: dict[str, dict] = {}
                for f in PROBE_FEATURES:
                    if f not in spec.features:
                        continue
                    solo = _ap_binary(
                        [r.get(f, 0.0) for r in erows], elabels
                    )
                    wl, _l = _fit_head(
                        spec,
                        hrows,
                        hlabels,
                        seed,
                        names=tuple(n for n in spec.features if n != f),
                    )
                    loo = _ap_binary(
                        [_dot(wl, r) for r in erows], elabels
                    )
                    probe[f] = {
                        "solo_ap": round(solo, 4),
                        "loo_ap": round(loo, 4),
                        "loo_delta": round(loo - eval_ap, 4),
                    }
                tau, cal_reward = _calibrate_tau(
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
                sampled = _play_pools(
                    lambda i, p=player, s=spec, wh=w_head, ta=tau, rs=rank_spec: (
                        _mk_gate(
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
                greedy = _play_pools(
                    lambda i, p=player, s=spec, wh=w_head, ta=tau, rs=rank_spec: (
                        _mk_gate(
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
                # Round 5 — the serializable fold: the head's
                # stop-only columns + a calibrated ``op:stop``
                # intercept as an ordinary weight table (see
                # ``_fold_weights`` — the only delivery of the
                # gate's timing that ``OPEN_GAME_PLAYER_WEIGHTS``
                # could actually carry).  C calibrated on the train
                # pages exactly like τ.
                fold_c, fold_cal = _calibrate_fold_c(
                    spec, w_rank, w_head, train_pools, budget, seed
                )
                w_fold = _fold_weights(spec, w_rank, w_head, fold_c)
                weights[arm_id + "-fold"] = {
                    n: round(wv, 4)
                    for n, wv in w_fold.items()
                    if abs(wv) > 1e-4
                }
                heads[arm_id]["fold"] = {
                    "c": round(fold_c, 4),
                    "cal_reward": fold_cal,
                }
                fold_sampled = _play_pools(
                    lambda i, s=spec, wf=w_fold: og.OpenLearnedPlayer(
                        seed=seed + _SAMPLED_SEED + i,
                        spec=s,
                        weights=wf,
                        learn=False,
                    ),
                    ev_pools,
                    budget,
                )
                fold_greedy = _play_pools(
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
                    _page_scores(rank_spec, w_rank, ev_pools),
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

    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": rows_out,
        "anchors": anchors,
        "train_curve": curves,
        "weights": weights,
        "heads": heads,
        "predictor": predictor,
        "pool": {"held_out": len(flat_pool), "useful": n_useful},
        "meta": {
            "measure_seconds": round(measure_s, 1),
            "epochs": epochs,
            "long_epochs": LONG_EPOCHS,
            "imitate_passes": IMITATE_PASSES,
            "head_epochs": HEAD_EPOCHS,
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
        f"{'arm':<16} {'reward':>8} {'greedy':>8} {'use/eval':>9}"
        f" {'evals/ep':>8} {'stops':>6} {'AP':>7}"
    )
    pred = res.get("predictor")
    if pred:
        print(  # stdout-compat
            f"== RuleValueNet predictors: {pred['train_samples']} "
            f"even-page samples, injected={pred['injected']} "
            f"(OOF {pred['oof_folds']} folds, {pred['seconds']}s) =="
        )
        for lname, pr in pred["nets"].items():
            aps = " ".join(
                f"{n}={pr[f'ap_{n}_logit']:.4f}" for n in VALUE_LABELS
            )
            print(  # stdout-compat
                f"  {lname:<7} pos={pr['train_positives']:>3} "
                f"pw={pr['pos_weight']:>6.1f} AP_sigmoid_useful="
                f"{pr['ap_useful']:.4f} AP_logit[{aps}] "
                f"AP_train={pr['ap_train_logit']:.4f}"
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
            f"loss={h['loss']} eval_ap={h.get('eval_ap')} --"
        )
        for f, p in h.get("feature_probe", {}).items():
            print(  # stdout-compat
                f"    probe {f:<22} solo={p['solo_ap']:.4f} "
                f"loo={p['loo_ap']:.4f} d={p['loo_delta']:+.4f}"
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
    ap.add_argument(
        "--value-label",
        choices=sorted(VALUE_LABELS),
        default="useful",
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
