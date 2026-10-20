"""Dedicated RL arms under the controlled held-out protocol.

The baseline (``bench/drivers/policy/policy_baseline.py``) and the ablations
(``bench/drivers/policy/policy_ablations.py``) measured *linear* policies trained by
the players' built-in REINFORCE and a *supervised* ``RuleValueNet``.
This script runs the dedicated neural-RL arms the repo actually
ships, under the same held-out protocol, and asks whether the
dedicated-RL machinery beats the trained linear prior:

* **open** — ``mlopt_torch.rl.PolicyNet`` (the ``SearchEnv``
  REINFORCE net: MLP 64->64->1 over ``(state (+) rule)`` inputs,
  Categorical sampling, gamma 0.95, Adam 3e-3, cross-episode EMA
  advantage baseline) ported to the open-domain episode loop
  (``OpenArena`` + memoized ``proposal.evaluate``).  ``train_rl_policy``
  only ever ran the search domain; the port keeps the net and the
  update rule verbatim and adapts only the featurization — three input
  encodings:

  - ``policynet`` — the ``PolicyNet`` contract *extended*: each
    ``evaluate`` action is ``state_vector(compute_features(cand.lhs),
    progress) (+) rule_vector(cand.as_rule()) (+)`` the candidate's
    ``a:*``/``x:*`` block — the same keys ``spec.featurize`` emits,
    in ``OPEN_GAME_FEATURES`` order.  The bare contract collapsed
    (see ``docs/results_learned_rl.md``): it lacked the
    discriminative shape features — heuristic score, depth, metas,
    form one-hots — that the linear policy's ``a:*`` row carries.
    The bare form is load-bearing for the search domain
    (``SearchEnv``/``RLPolicy`` score *rules*, which have no ``a:*``
    block), so it is not replaced globally — the port appends the
    missing features, sourced from ``spec.featurize`` itself so a
    schema extension is picked up automatically.
  - ``struct`` — the pre-fix ``policynet`` row (``compute_features``
    ``(+)`` ``rule_vector``, no ``a:*`` block), kept as the encoding
    control under the shipped recipe.
  - ``full`` — the *same* MLP shape fed the linear policy's own
    feature rows (``og.featurize(state, action, hist)`` over
    ``OPEN_GAME_FEATURES``) — isolates "is the linear prior's edge
    the features or the model class".

  Each encoding trains at two budgets: ``--rl-episodes`` (180 = the
  5-epoch linear recipe) and ``--rl-episodes-long`` (1440 = 8x).
  Held-out reporting is unchanged: odd pages, budget 17, memoized
  oracle draws, reward/ep + useful/ep + useful/eval + AP via
  ``rank_pool``'s average precision — the AP scorer is the net's
  ``evaluate``-action logit on the fresh page.

* **contraction** — ``contraction_policy.train_rl`` (REINFORCE with
  the greedy-completion critic, entropy bonus, batch 24) re-run at
  *full* scale on CPU (1800 iterations — measured ~2 min here, no
  reduction needed) on ``contraction_scale.random_network``
  instances, evaluated with ``_ladder`` at n = 20/30/40 next to the
  bundled curriculum-RL artifact and the shipped distilled player
  (``load_contraction_policy`` — the 0.91x-vs-opt_einsum anchor).

* **az** — ``contraction_az.train_search`` (expert iteration: PUCT
  self-play teaches the two-head net) on CPU at the shipped defaults
  (200 iterations, batch 24, sims 24) over ``random_bond_network``
  instances, then the equal-wall-clock board of
  ``contraction_einsum._run_board`` plus ``puct_best_order`` at
  reduced budgets, scored under *both* cost models.  ``opt_einsum``
  must be installed (``uv pip install opt_einsum``); the section is
  skipped with a note when it is absent.

Usage::

    .venv/bin/python bench/drivers/policy/policy_rl_arms.py --json /tmp/rl_arms.json
    .venv/bin/python bench/drivers/policy/policy_rl_arms.py --only open
    .venv/bin/python bench/drivers/policy/policy_rl_arms.py --only contraction,az
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import policy_ablations as pa
import policy_baseline as pb
import torch
from mlopt_core.features import compute_features
from mlopt_core.trajectories import RULE_VECTOR_LEN, rule_vector
from mlopt_discovery import open_game as og
from mlopt_discovery import play
from mlopt_discovery import proposal as lp
from mlopt_discovery.meta_arena import Action
from mlopt_torch import rl as rl_mod
from mlopt_torch.learned_policy import input_vector
from torch import nn
from torch.distributions import Categorical

from bench.drivers.probes.contraction import contraction_policy as cp
from bench.drivers.probes.contraction import contraction_scale as cs

#: Defaults — the baseline protocol plus the RL recipes' shipped knobs.
SEED = 0
PAGE = 16
OPEN_PAGES = 36
OPEN_BUDGET = PAGE + 1
RL_EPISODES = 180  # 5 epochs x 36 pages — the linear recipe's budget
RL_EPISODES_LONG = 1440  # 8x — the "does bigger RL help" arm
NET_HIDDEN = 64
NET_LR = 3e-3
NET_GAMMA = 0.95
LINEAR_EPOCHS = 5  # the cold+rl linear anchor
CONTRACTION_ITERS = 1800
CONTRACTION_BATCH = 24
AZ_ITERS = 200
AZ_BATCH = 24
AZ_SIMS = 24
AZ_EPOCHS = 4
EVAL_SEED_OFFSET = 60_000  # policy_ablations' eval-seed offset


# ---------------------------------------------------------------------------
#  Open domain — PolicyNet REINFORCE ported to OpenArena episodes
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: int) -> nn.Module:
    """Return the ``PolicyNet`` shape — ``in_dim -> hidden -> hidden -> 1``."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


#: The candidate-level block the bare ``PolicyNet`` contract lacked —
#: the ``a:*``/``x:*`` keys ``spec.featurize`` emits, in schema order.
#: Sourced from ``OPEN_GAME_FEATURES`` so a schema extension (a new
#: ``a:*``/``x:*`` part joining the spec) widens this block
#: automatically.
_CAND_KEYS: tuple[str, ...] = tuple(
    k
    for k in og.OPEN_GAME_FEATURES
    if k.startswith("a:") or k.startswith("x:")
)


class _OpenNetFeaturizer:
    """Per-action input rows for the open-domain encodings.

    ``mode="policynet"`` reuses ``rl.PolicyNet``'s input contract —
    ``state_vector(compute_features(cand.lhs), progress)`` concatenated
    with ``rule_vector(cand.as_rule())`` — the candidate's LHS plays
    the program role, its ``as_rule()`` the rule role — **plus** the
    candidate's ``a:*``/``x:*`` block (``:data:`_CAND_KEYS```, emitted
    by ``spec.featurize``).  ``stop`` is an all-zero row: its logit is
    a learnable constant.

    ``mode="struct"`` is the bare pre-fix contract (no ``a:*`` block)
    — the encoding that collapsed; kept as the control.

    ``mode="full"`` feeds the linear policy's own feature row —
    ``og.featurize(state, action, hist)`` ordered by
    ``OPEN_GAME_FEATURES`` — so the MLP gets exactly the information
    the trained linear prior has.
    """

    def __init__(self, mode: str) -> None:
        """Bind *mode* (``policynet`` | ``struct`` | ``full``)."""
        self.mode = mode
        # Content-keyed row cache — ``id(cand)`` is *wrong* here: each
        # episode builds a fresh arena, dead candidates' ids get
        # reused, and the collision order depends on arm-run order
        # (a silent nondeterminism).  ``pb._cand_key`` is the stable
        # content identity the memoized verdicts use too.
        self._cache: dict[tuple, list[float]] = {}

    def dim(self) -> int:
        """Input width for *mode*."""
        if self.mode == "full":
            return len(og.OPEN_GAME_FEATURES)
        n = rl_mod._STATE_DIM + RULE_VECTOR_LEN
        if self.mode == "policynet":
            n += len(_CAND_KEYS)
        return n

    def _cand_row(
        self, cand: Any, progress: float, extra: list[float]
    ) -> list[float]:
        """Return the structural row for one ``evaluate`` action."""
        key = pb._cand_key(cand)
        hit = self._cache.get(key)
        if hit is None:
            hit = [
                *compute_features(cand.lhs).to_vector(),
                *rule_vector(cand.as_rule()),
            ]
            self._cache[key] = hit
        n_feat = rl_mod._STATE_DIM - 1
        return [*hit[:n_feat], progress, *hit[n_feat:], *extra]

    def rows(
        self, state: Any, acts: Any, hist: Any
    ) -> list[list[float]]:
        """One input row per legal action (evaluate-per-pending, stop)."""
        if self.mode == "full":
            feats = [og.featurize(state, a, hist) for a in acts]
            return [
                [f.get(k, 0.0) for k in og.OPEN_GAME_FEATURES]
                for f in feats
            ]
        progress = state.n_seen / max(state.pool_size, 1)
        zero = [0.0] * self.dim()
        out: list[list[float]] = []
        for a in acts:
            if a.op != "evaluate":
                out.append(zero)
                continue
            extra = (
                [
                    og.featurize(state, a, hist).get(k, 0.0)
                    for k in _CAND_KEYS
                ]
                if self.mode == "policynet"
                else []
            )
            out.append(
                self._cand_row(a.params["cand"], progress, extra)
            )
        return out

    def rank_rows(self, cands: list) -> list[list[float]]:
        """Rows for the fresh-page AP scorer (progress = 0)."""
        if self.mode == "struct":
            return [self._cand_row(c, 0.0, []) for c in cands]
        # A fresh state over the pool itself — ``featurize`` reads
        # the candidate off the action and the broadcast block off
        # the state, so a synthetic untouched page is exact.  Pool-
        # relative features (``a:score_rank`` & co., if baked) see the
        # whole page pending, which is the right context for an
        # ordering probe.
        state = og.OpenState(
            pool=tuple(cands),
            pending=tuple(range(len(cands))),
            verdicts=(),
            steps=0,
            done=False,
            pool_size=len(cands),
        )
        return self.rows(
            state,
            [
                og.Action("evaluate", {"i": i, "cand": c})
                for i, c in enumerate(cands)
            ],
            og.OPEN_SPEC.new_hist(),
        )


def _open_net(feat: _OpenNetFeaturizer, hidden: int) -> nn.Module:
    """Build the scorer — the ``PolicyNet`` shape at the row's width.

    ``struct`` still fits the shipped class exactly; the extended
    ``policynet`` and ``full`` rows use the same ``in->h->h->1`` MLP
    at their own input width (``PolicyNet(hidden).net`` is precisely
    ``_mlp(_STATE_DIM + RULE_VECTOR_LEN, hidden)``).
    """
    if feat.mode == "struct":
        return rl_mod.PolicyNet(hidden).net
    return _mlp(feat.dim(), hidden)


def train_open_net(
    cases: list[dict],
    budget: int,
    *,
    mode: str,
    adv_mode: str,
    episodes: int,
    hidden: int = NET_HIDDEN,
    lr: float = NET_LR,
    gamma: float = NET_GAMMA,
    seed: int,
    device: str = "cpu",
    epoch_len: int = 0,
) -> tuple[nn.Module, list[float]]:
    """REINFORCE over open-domain episodes — the ``train_reinforce`` loop.

    One sampled episode per case (cycling), returns-to-go discounted
    by *gamma* and an Adam step per episode — the shipped recipe, with
    only the featurization adapted to the arena.  ``adv_mode`` picks
    the advantage baseline: ``"shipped"`` is whatever
    ``mlopt_torch.rl`` trains with — since the stop-collapse fix, a
    cross-episode :class:`rl.EmaBaseline` (``G_t - ema``, ``ema`` a
    0.2-EMA of episode totals, clipped); ``"ema"`` is the same
    baseline inlined here as an implementation-parity cross-check;
    ``"per-ep"`` is the retired per-episode standardization
    (``rl._advantage``), kept as the regression control — it loses
    the cross-episode reward level and collapses to ``stop`` on
    sparse pages.  The caller patches ``lp.evaluate`` with the
    memoized verdicts, so a candidate's verdict is identical for
    every arm and epoch.
    """
    torch.manual_seed(seed)
    feat = _OpenNetFeaturizer(mode)
    model = _open_net(feat, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    baseline = rl_mod.EmaBaseline()
    ema = 0.0
    means: list[float] = []
    block: list[float] = []
    for ep in range(episodes):
        arena = og.OpenArena(cases[ep % len(cases)])
        hist = og.OPEN_SPEC.new_hist()
        logps: list[torch.Tensor] = []
        rews: list[float] = []
        while len(rews) < budget:
            state = arena.observe()
            if state.done:
                break
            acts = og.legal_actions(state)
            x = torch.tensor(
                feat.rows(state, acts, hist),
                dtype=torch.float32,
                device=device,
            )
            dist = Categorical(logits=model(x).squeeze(-1))
            i = dist.sample()
            logps.append(dist.log_prob(i))
            action = acts[int(i.item())]
            _state, rep = arena.step(action)
            rews.append(rep.reward)
            hist.record(getattr(action, "op", "?"))
        running, returns = 0.0, []
        for r in reversed(rews):
            running = r + gamma * running
            returns.append(running)
        returns.reverse()
        adv_t = torch.tensor(
            returns, dtype=torch.float32, device=device
        )
        if adv_mode == "shipped":
            adv = baseline.advantage(adv_t)
        elif adv_mode == "ema":
            if len(adv_t):
                ema += 0.2 * (float(adv_t[0]) - ema)
            adv = adv_t - ema
        else:  # "per-ep" — the retired recipe, regression control
            adv = rl_mod._advantage(adv_t)
        loss = -(torch.stack(logps) * adv).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        block.append(sum(rews))
        if epoch_len and len(block) == epoch_len:
            means.append(round(sum(block) / len(block), 4))
            block = []
    if block:
        means.append(round(sum(block) / len(block), 4))
    return model.eval(), means


def _fresh_state(pool: list) -> Any:
    """Return a fresh ``OpenState`` over *pool* — all pending."""
    return og.OpenState(
        pool=tuple(pool),
        pending=tuple(range(len(pool))),
        verdicts=(),
        steps=0,
        done=False,
        pool_size=len(pool),
    )


def _enriched_row(c: Any, state: Any, i: int) -> list[float]:
    """``input_vector(lhs, rule)`` ++ the spec's ``a:*``/``x:*`` keys."""
    f = og.featurize(
        state, Action("evaluate", {"i": i, "cand": c}), None
    )
    return [
        *input_vector(
            compute_features(c.lhs), rule_vector(c.as_rule())
        ),
        *[f.get(k, 0.0) for k in _CAND_KEYS],
    ]


def _ruleval_enriched_ap(
    train_pools: list[list],
    train_outcomes: list[list],
    ev_pools: list[list],
    flat_outcomes: dict,
    *,
    epochs: int,
    hidden: int,
    seed: int,
) -> float:
    """Supervised BCE AP arm on the enriched rows — ``ruleval+a*``.

    The ``RuleValueNet`` recipe (BCE on ``delta > 0``, standardized
    inputs, Adam 1e-2, MLP ``in->h->h->1``) fed the *enriched* row —
    the shipped structural input concatenated with the candidate's
    ``a:*``/``x:*`` block (the pool-relative columns the bare input
    cannot see).  Scores each held-out pool's fresh-page
    ``evaluate`` rows; the AP is the ordering-only read — no episode.
    """
    torch.manual_seed(seed)
    xs: list[list[float]] = []
    ys: list[float] = []
    for pool, outs in zip(train_pools, train_outcomes, strict=True):
        st = _fresh_state(pool)
        for i, (c, o) in enumerate(zip(pool, outs, strict=True)):
            xs.append(_enriched_row(c, st, i))
            ys.append(1.0 if o.base_cost > o.cand_cost else 0.0)
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    mean = x.mean(0)
    std = x.std(0).clamp_min(1e-6)
    model = _mlp(x.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model((x - mean) / std).squeeze(-1), y)
        loss.backward()
        opt.step()
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for pool in ev_pools:
            st = _fresh_state(pool)
            xs2 = torch.tensor(
                [_enriched_row(c, st, i) for i, c in enumerate(pool)],
                dtype=torch.float32,
            )
            scores.extend(
                float(v) for v in model((xs2 - mean) / std).squeeze(-1)
            )
    return round(pb._ap(scores, flat_outcomes), 4)


def _net_player(
    model: nn.Module, mode: str, seed: int, greedy: bool
) -> Any:
    """Return a ``run_episode``-contract player decoding *model*'s logits."""
    torch.manual_seed(seed)
    feat = _OpenNetFeaturizer(mode)
    hist = og.OPEN_SPEC.new_hist()

    def player(state: Any) -> Any:
        acts = og.legal_actions(state)
        x = torch.tensor(
            feat.rows(state, acts, hist),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = model(x).squeeze(-1)
        if greedy:
            i = int(torch.argmax(logits))
        else:
            i = int(Categorical(logits=logits).sample())
        action = acts[i]
        hist.record(getattr(action, "op", "?"))
        return action

    return player


def _net_ap(
    model: nn.Module,
    mode: str,
    pools: list[list],
    flat_outcomes: dict,
) -> float:
    """AP of the net's fresh-page evaluate ordering on each pool."""
    feat = _OpenNetFeaturizer(mode)
    scores: list[float] = []
    for pool in pools:
        x = torch.tensor(
            feat.rank_rows(list(pool)), dtype=torch.float32
        )
        with torch.no_grad():
            scores.extend(float(v) for v in model(x).squeeze(-1))
    return round(pb._ap(scores, flat_outcomes), 4)


def _eval_net(
    model: nn.Module,
    mode: str,
    tag: str,
    ev_pools: list[list],
    flat_outcomes: dict,
    ev: list[dict],
    budget: int,
    seed: int,
    eps: dict[str, dict],
    aps: dict[str, float],
) -> None:
    """Play *model* sampled + greedy on the held-out pages; AP too."""
    eps[tag] = pb._play_open(
        lambda i: _net_player(
            model, mode, seed + EVAL_SEED_OFFSET + i, False
        ),
        ev,
        budget,
    )
    eps[f"{tag}-greedy"] = pb._play_open(
        lambda i: _net_player(
            model, mode, seed + EVAL_SEED_OFFSET + 500 + i, True
        ),
        ev,
        budget,
    )
    aps[tag] = _net_ap(model, mode, ev_pools, flat_outcomes)


def run_open(
    seed: int,
    half: int,
    page: int,
    budget: int,
    *,
    episodes: int,
    episodes_long: int,
    linear_epochs: int,
    hidden: int,
) -> dict[str, Any]:
    """Train the net RL arms on even pages; eval on the odd pages."""
    t0 = time.perf_counter()
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    eval_outcomes, by_key = pb._measure_pools(
        ev_pools, seed + pb.HELD_OUT_OFFSET
    )
    train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
    _train_outcomes, train_by_key = pb._measure_pools(
        train_pools, seed + 1_000_003
    )
    by_key.update(train_by_key)
    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1
    measured_s = time.perf_counter() - t0

    eps: dict[str, dict] = {}
    aps: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    meta: dict[str, Any] = {"measure_seconds": round(measured_s, 1)}

    orig_evaluate = lp.evaluate
    lp.evaluate = pb._patched_evaluate(by_key)
    try:
        # --- anchors re-run under the same memoized draws -----------
        baked = og.OpenLearnedPlayer(seed)
        cold = og.OpenLearnedPlayer(seed, weights={})
        rnd = play._random_policy(og.legal_actions)
        eps["random"] = pb._play_open(
            lambda i: rnd(seed + 10_000 + i), ev, budget
        )
        eps["heuristic"] = pb._play_open(
            pb._heuristic_arm(), ev, budget
        )
        eps["cold-linear"] = pb._play_open(
            lambda i: cold.frozen(seed + 20_000 + i), ev, budget
        )
        eps["trained-linear"] = pb._play_open(
            lambda i: baked.frozen(seed + 30_000 + i), ev, budget
        )
        eps["trained-greedy"] = pb._play_open(
            lambda i: baked.frozen(seed + 40_000 + i, greedy=True),
            ev,
            budget,
        )
        aps["heuristic"] = round(
            pb._ap(
                [lp._heuristic_score(c) for c in flat_pool],
                flat_outcomes,
            ),
            4,
        )
        aps["cold-linear"] = round(
            pb._ap(
                [cold.rank_score(c) for c in flat_pool], flat_outcomes
            ),
            4,
        )
        aps["trained-linear"] = round(
            pb._ap(
                [baked.rank_score(c) for c in flat_pool], flat_outcomes
            ),
            4,
        )

        # --- the linear-RL anchor (the ablations' strongest RL arm) --
        cold_rl = og.OpenLearnedPlayer(seed + 4, weights={})
        curves["cold+rl"] = pa._train(
            cold_rl, train_cases, budget, linear_epochs
        )
        pa._eval_trained(
            cold_rl,
            flat_pool,
            flat_outcomes,
            ev,
            budget,
            seed,
            "cold+rl",
            eps,
            aps,
        )

        # --- the supervised enriched arm ----------------------------
        # RuleValueNet's BCE recipe on the ``input_vector (+) a:*/x:*``
        # row — does *supervised* ordering exploit the pool-relative
        # columns the RL nets cannot?
        t1 = time.perf_counter()
        aps["ruleval+a*"] = _ruleval_enriched_ap(
            train_pools,
            _train_outcomes,
            ev_pools,
            flat_outcomes,
            epochs=300,
            hidden=32,
            seed=seed,
        )
        meta.setdefault("train_seconds", {})["ruleval+a*"] = round(
            time.perf_counter() - t1, 1
        )

        # --- the dedicated RL nets ---------------------------------
        # Bare tag = the shipped recipe; ``-ema`` = the inlined parity
        # cross-check; ``-per-ep`` = the retired per-episode
        # standardization, kept as the collapse control.
        arms = []
        for mode in ("policynet", "full"):
            stem = "rl-net" if mode == "policynet" else "rl-net-full"
            for adv in ("per-ep", "shipped", "ema"):
                for eps_n, suf in (
                    (episodes, ""),
                    (episodes_long, "-long"),
                ):
                    arms.append(
                        (
                            mode,
                            adv,
                            eps_n,
                            f"{stem}"
                            f"{'-ema' if adv == 'ema' else ''}"
                            f"{'-per-ep' if adv == 'per-ep' else ''}"
                            f"{suf}",
                        )
                    )
        # ``rl-struct`` — the retired bare ``PolicyNet`` input
        # (``compute_features (+) rule_vector``, no ``a:*`` block),
        # under the shipped recipe: the encoding-collapse control.
        for eps_n, suf in ((episodes, ""), (episodes_long, "-long")):
            arms.append(("struct", "shipped", eps_n, f"rl-struct{suf}"))
        for mode, adv, eps_n, tag in arms:
            t1 = time.perf_counter()
            model, curve = train_open_net(
                train_cases,
                budget,
                mode=mode,
                adv_mode=adv,
                episodes=eps_n,
                hidden=hidden,
                seed=seed,
                epoch_len=len(train_cases),
            )
            curves[tag] = curve
            meta.setdefault("train_seconds", {})[tag] = round(
                time.perf_counter() - t1, 1
            )
            _eval_net(
                model,
                mode,
                tag,
                ev_pools,
                flat_outcomes,
                ev,
                budget,
                seed,
                eps,
                aps,
            )
    finally:
        lp.evaluate = orig_evaluate

    n_pos = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "episodes_table": eps,
        "ap": aps,
        "train_curve": curves,
        "pool": {"held_out": len(flat_pool), "useful": n_pos},
        "meta": meta,
    }


# ---------------------------------------------------------------------------
#  Contraction domain — the RL curriculum re-run at full scale on CPU
# ---------------------------------------------------------------------------


def _contraction_ladder(
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, nn.Module],
    device: str,
) -> dict[str, Any]:
    """``cp._ladder`` per board; mean ratio-to-best-found + pairwise."""
    per_scale: dict[str, Any] = {}
    for n in scales:
        ratios: dict[str, list[float]] = {}
        pair: dict[str, list[float]] = {}
        for k in range(instances):
            tensors, sizes = cs.random_network(n, seed + k)
            found = cp._ladder(tensors, sizes, n, models, device)
            best = min(found.values())
            for name, v in found.items():
                ratios.setdefault(name, []).append(v / best)
            for name in models:
                for base in ("greedy", "restart", "search"):
                    if base in found:
                        pair.setdefault(f"{name}/{base}", []).append(
                            found[name] / found[base]
                        )
        per_scale[str(n)] = {
            "ratio_to_best": {
                p: round(statistics.fmean(v), 4)
                for p, v in sorted(ratios.items())
            },
            "pairwise": {
                c: round(statistics.fmean(v), 4)
                for c, v in sorted(pair.items())
            },
        }
    return per_scale


def _load_artifact_model(name: str, device: str) -> tuple[Any, dict]:
    """Load a bundled artifact's ``PairPolicyNet`` + meta.

    The bundled ``.pt`` payloads predate the catopt -> mlopt rename and
    carry ``format='catopt-contraction-policy/1'`` — accepted by the
    shipped ``load_contraction_policy`` as a legacy alias, but this
    reads the payload directly so the raw ``format_tag`` can be
    recorded in the report alongside the meta.
    """
    from importlib import resources

    from mlopt_torch import contraction_policy as mcp
    from mlopt_torch.contraction_policy import PairPolicyNet

    path = resources.files("mlopt_torch") / "artifacts" / name
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = dict(payload["meta"])
    if meta.get("feature_contract") != mcp._FEATURE_CONTRACT:
        raise ValueError(
            f"{path}: feature contract {meta.get('feature_contract')!r}"
            f" != {mcp._FEATURE_CONTRACT!r}"
        )
    model = PairPolicyNet(int(meta["hidden"]))
    model.load_state_dict(payload["state_dict"])
    meta["format_tag"] = payload.get("format")
    return model.to(device).eval(), meta


def run_contraction(
    seed: int,
    scales: tuple[int, ...],
    instances: int,
    *,
    iterations: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    """Re-run the RL curriculum at full scale; race the artifacts."""
    t0 = time.perf_counter()
    rl = cp.train_rl(
        iterations=iterations,
        batch=batch,
        device=device,
        seed=seed,
        log_every=max(iterations // 3, 1),
    )
    train_s = time.perf_counter() - t0
    bundled, bundled_meta = _load_artifact_model(
        "contraction_policy_curriculum.pt", device
    )
    shipped, shipped_meta = _load_artifact_model(
        "contraction_policy_distilled.pt", device
    )
    models = {
        "rl-fresh": rl,
        "rl-bundled": bundled,
        "distilled": shipped,
    }
    meta = {
        "train_seconds": round(train_s, 1),
        "iterations": iterations,
        "batch": batch,
        "train_family": "contraction_scale.random_network",
        "shipped_meta": shipped_meta,
        "bundled_meta": bundled_meta,
    }
    t0 = time.perf_counter()
    table = _contraction_ladder(scales, instances, seed, models, device)
    meta["eval_seconds"] = round(time.perf_counter() - t0, 1)
    return {"ladder": table, "meta": meta}


# ---------------------------------------------------------------------------
#  AZ domain — PUCT + learned value head on bond networks, CPU scale
# ---------------------------------------------------------------------------


def _az_board(
    tensors: Any,
    sizes: dict[int, int],
    board_models: dict[str, Any],
    az_model: Any,
    device: str,
    budget: float,
    seed: int,
    *,
    sims: int,
    puct_analytic: bool,
) -> dict[str, Any]:
    """Run the einsum board plus the PUCT arms at equal wall-clock.

    ``board_models`` are ``PairPolicyNet``s played by
    ``ce._run_board``'s anytime ``policy_best_order``; *az_model* is
    the two-head net and is played by ``az.puct_best_order`` — the two
    have different forward signatures, so they cannot share the loop.
    """
    from bench.drivers.probes.contraction import contraction_az as az
    from bench.drivers.probes.contraction import contraction_einsum as ce

    e = ce.to_einsum(tensors, sizes)
    ref = cs.greedy(tensors, sizes)
    out = ce._run_board(
        tensors, sizes, board_models, device, budget, seed
    )
    b1, b8 = az._puct_prior(az_model, tensors, sizes, device)
    order, secs, _eps, work = az.puct_best_order(
        az_model,
        tensors,
        sizes,
        ref,
        budget,
        device,
        sims=sims,
        seed=seed,
        per_sim=b8,
        per_sim_b1=b1,
    )
    out["puct"] = ce._res(tensors, sizes, e, order, secs, work)
    if puct_analytic:
        a_b1, a_b8 = az._puct_prior(
            az_model, tensors, sizes, device, use_net_value=False
        )
        order, secs, _eps, work = az.puct_best_order(
            az_model,
            tensors,
            sizes,
            ref,
            budget,
            device,
            sims=sims,
            seed=seed,
            per_sim=a_b8,
            per_sim_b1=a_b1,
            use_net_value=False,
        )
        out["puct-analytic"] = ce._res(
            tensors, sizes, e, order, secs, work
        )
    return out


def run_az(
    seed: int,
    scales: tuple[int, ...],
    instances: int,
    budgets: tuple[float, ...],
    *,
    iterations: int,
    batch: int,
    sims: int,
    epochs: int,
    rl_iterations: int,
    device: str,
    puct_analytic: bool,
) -> dict[str, Any]:
    """Train the AZ net + a bond-family RL policy; equal-budget board."""
    from bench.drivers.probes.contraction import contraction_az as az
    from bench.drivers.probes.contraction import contraction_einsum as ce

    t0 = time.perf_counter()
    with ce._on_family(ce.random_bond_network):
        current = cp.train_rl(
            iterations=rl_iterations,
            batch=batch,
            device=device,
            seed=seed,
        )
    rl_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    azm = az.train_search(
        batch=batch,
        iterations=iterations,
        sims=sims,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    az_s = time.perf_counter() - t0

    board_models = {
        "rl-bond": current,
        "distilled": _load_artifact_model(
            "contraction_policy_distilled.pt", device
        )[0],
    }

    rows: dict[str, Any] = {}
    t0 = time.perf_counter()
    for budget in budgets:
        for n in scales:
            per_player_our: dict[str, list[float]] = {}
            per_player_oe: dict[str, list[float]] = {}
            pair_our: dict[str, list[float]] = {}
            pair_oe: dict[str, list[float]] = {}
            works: dict[str, list[float]] = {}
            for k in range(instances):
                tensors, sizes = ce.random_bond_network(n, seed + k)
                res = _az_board(
                    tensors,
                    sizes,
                    board_models,
                    azm,
                    device,
                    budget,
                    seed + k,
                    sims=sims,
                    puct_analytic=puct_analytic,
                )
                best_our = min(r.our for r in res.values())
                best_oe = min(
                    r.oe for r in res.values() if r.oe < float("inf")
                )
                for name, r in res.items():
                    per_player_our.setdefault(name, []).append(
                        r.our / best_our
                    )
                    if r.oe == r.oe and r.oe < float("inf"):
                        per_player_oe.setdefault(name, []).append(
                            r.oe / best_oe
                        )
                    works.setdefault(name, []).append(r.work)
                learned = [
                    p
                    for p in ("puct", "rl-bond", "distilled")
                    if p in res
                ]
                for name in learned:
                    for base in res:
                        if base == name:
                            continue
                        if res[base].our < float("inf"):
                            pair_our.setdefault(
                                f"{name}/{base}", []
                            ).append(res[name].our / res[base].our)
                        if res[base].oe < float("inf"):
                            pair_oe.setdefault(
                                f"{name}/{base}", []
                            ).append(res[name].oe / res[base].oe)
            key = f"{int(budget * 1000)}ms_n{n}"
            rows[key] = {
                "ratio_to_best_our": {
                    p: round(statistics.fmean(v), 4)
                    for p, v in sorted(per_player_our.items())
                },
                "ratio_to_best_oe": {
                    p: round(statistics.fmean(v), 4)
                    for p, v in sorted(per_player_oe.items())
                },
                "pairwise_our": {
                    c: round(statistics.fmean(v), 4)
                    for c, v in sorted(pair_our.items())
                },
                "pairwise_oe": {
                    c: round(statistics.fmean(v), 4)
                    for c, v in sorted(pair_oe.items())
                },
                "work": {
                    p: round(statistics.fmean(v), 1)
                    for p, v in sorted(works.items())
                },
            }
    return {
        "boards": rows,
        "meta": {
            "rl_train_seconds": round(rl_s, 1),
            "az_train_seconds": round(az_s, 1),
            "eval_seconds": round(time.perf_counter() - t0, 1),
            "az_iterations": iterations,
            "az_sims": sims,
            "az_batch": batch,
            "rl_iterations": rl_iterations,
            "family": "random_bond_network (einsum-valid)",
            "puct_analytic": puct_analytic,
        },
    }


# ---------------------------------------------------------------------------
#  Rendering + driver
# ---------------------------------------------------------------------------


def _print_open(res: dict[str, Any]) -> None:
    """Render the open episode table, APs and training curves."""
    head = (
        f"{'arm':<24} {'reward':>9} {'use/ep':>7} {'evals/ep':>9}"
        f" {'use/eval':>9} {'found':>6} {'AP':>7}"
    )
    print("== open domain — held-out odd pages ==")  # stdout-compat
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat
    for arm, r in res["episodes_table"].items():
        stem = arm[:-7] if arm.endswith("-greedy") else arm
        ap = res["ap"].get(stem)
        ap_s = f"{ap:7.4f}" if ap is not None else f"{'—':>7}"
        print(  # stdout-compat
            f"{arm:<24} {r['reward']:>9.4f} {r['useful_per_ep']:>7.4f} "
            f"{r['evals_per_ep']:>9.3f} {r['useful_per_eval']:>9.4f} "
            f"{r['found']:>6} {ap_s}"
        )
    for arm, curve in res["train_curve"].items():
        print(f"  train[{arm}]: {curve}")  # stdout-compat
    ep_stems = {
        a[:-7] if a.endswith("-greedy") else a
        for a in res["episodes_table"]
    }
    for name, ap in res["ap"].items():
        if name not in ep_stems:
            print(f"  AP[{name}] = {ap:.4f}")  # stdout-compat
    r = res["pool"]
    print(  # stdout-compat
        f"  pool={r['held_out']} useful={r['useful']} "
        f"(random AP = {r['useful'] / r['held_out']:.4f})"
    )


def _print_contraction(res: dict[str, Any]) -> None:
    """Render the per-scale ratio-to-best and pairwise tables."""
    print(
        "== contraction — RL curriculum vs artifacts =="
    )  # stdout-compat
    print(  # stdout-compat
        f"  (trained in {res['meta']['train_seconds']}s, "
        f"eval {res['meta']['eval_seconds']}s; ratio to best-found)"
    )
    for n, rows in res["ladder"].items():
        print(f"  n={n}")  # stdout-compat
        for p, v in rows["ratio_to_best"].items():
            print(f"    {p:<22} {v:.4f}")  # stdout-compat
        for c, v in rows["pairwise"].items():
            print(f"    {c:<22} {v:.4f}  (pairwise)")  # stdout-compat


def _print_az(res: dict[str, Any]) -> None:
    """Render the equal-wall-clock AZ board means."""
    m = res["meta"]
    print(  # stdout-compat
        "== AZ — PUCT + value head, equal wall-clock "
        f"(az {m['az_iterations']} iters in {m['az_train_seconds']}s, "
        f"eval {m['eval_seconds']}s) =="
    )
    for key, rows in res["boards"].items():
        print(f"  {key}")  # stdout-compat
        for p, v in rows["ratio_to_best_our"].items():
            print(f"    our {p:<16} {v:.4f}")  # stdout-compat
        for c, v in rows["pairwise_oe"].items():
            if c.startswith("puct/") or c.endswith("/oe-rand-greedy"):
                print(
                    f"    oe  {c:<20} {v:.4f}  (pairwise)"
                )  # stdout-compat


def main(argv: list[str] | None = None) -> int:
    """Run the enabled RL arms and print the tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--open-budget", type=int, default=OPEN_BUDGET)
    ap.add_argument("--rl-episodes", type=int, default=RL_EPISODES)
    ap.add_argument(
        "--rl-episodes-long", type=int, default=RL_EPISODES_LONG
    )
    ap.add_argument("--linear-epochs", type=int, default=LINEAR_EPOCHS)
    ap.add_argument("--hidden", type=int, default=NET_HIDDEN)
    ap.add_argument(
        "--contraction-iters", type=int, default=CONTRACTION_ITERS
    )
    ap.add_argument(
        "--contraction-batch", type=int, default=CONTRACTION_BATCH
    )
    ap.add_argument("--contraction-scales", default="20,30,40")
    ap.add_argument("--az-iters", type=int, default=AZ_ITERS)
    ap.add_argument("--az-batch", type=int, default=AZ_BATCH)
    ap.add_argument("--az-sims", type=int, default=AZ_SIMS)
    ap.add_argument("--az-epochs", type=int, default=AZ_EPOCHS)
    ap.add_argument("--az-scales", default="20,30")
    ap.add_argument(
        "--budgets",
        default="200,1000",
        help="comma-separated per-instance wall-clock ms (AZ board)",
    )
    ap.add_argument(
        "--az-rl-iters",
        type=int,
        default=1800,
        help="iterations for the AZ board's bond-family 'rl-bond' arm",
    )
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument(
        "--puct-analytic",
        action="store_true",
        help="also run PUCT with the analytic greedy-completion value",
    )
    ap.add_argument(
        "--only",
        default="open,contraction,az",
        help="comma-separated subset of arms to run",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    device = "cpu"
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    results: dict[str, Any] = {}
    if "open" in only:
        results["open"] = run_open(
            args.seed,
            args.open_pages,
            args.page,
            args.open_budget,
            episodes=args.rl_episodes,
            episodes_long=args.rl_episodes_long,
            linear_epochs=args.linear_epochs,
            hidden=args.hidden,
        )
        _print_open(results["open"])
    if "contraction" in only:
        results["contraction"] = run_contraction(
            args.seed,
            tuple(
                int(x)
                for x in args.contraction_scales.split(",")
                if x.strip()
            ),
            args.instances,
            iterations=args.contraction_iters,
            batch=args.contraction_batch,
            device=device,
        )
        _print_contraction(results["contraction"])
    if "az" in only:
        results["az"] = run_az(
            args.seed,
            tuple(
                int(x) for x in args.az_scales.split(",") if x.strip()
            ),
            args.instances,
            tuple(
                1e-3 * float(x)
                for x in args.budgets.split(",")
                if x.strip()
            ),
            iterations=args.az_iters,
            batch=args.az_batch,
            sims=args.az_sims,
            epochs=args.az_epochs,
            rl_iterations=args.az_rl_iters,
            device=device,
            puct_analytic=args.puct_analytic,
        )
        _print_az(results["az"])

    payload = {
        "seed": args.seed,
        "protocol": {
            "open_held_out": "odd pages (start=1, step=2)",
            "open_train": "even pages (start=0, step=2)",
            "open_page": args.page,
            "open_pages": args.open_pages,
            "open_budget": args.open_budget,
            "oracle_pairing": (
                "memoized per-candidate-seeded verdicts — identical "
                "draws for every arm and every training episode"
            ),
            "rl_episodes": args.rl_episodes,
            "rl_episodes_long": args.rl_episodes_long,
            "net": "PolicyNet shape (in->64->64->1), Adam 3e-3, "
            "gamma 0.95, cross-episode EMA advantage baseline "
            "(0.2 decay, ±40 clip)",
            "contraction_iters": args.contraction_iters,
            "az_iters": args.az_iters,
            "az_sims": args.az_sims,
            "device": device,
        },
        "results": results,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
