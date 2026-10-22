"""The credit-assignment probe — does RL learn enabling chains?

The Q3 measurement arm: a curriculum where the rule bank is
*deliberately incomplete* and the win is reachable only through a
two-step chain — mint the intermediate premise the goal rewrite
needs (a move that pays nothing and costs a step), then apply the
now-enabled rewrite (the move that pays).  The enabling mint's
immediate reward is *negative* (it spends budget on a law that
enables nothing by itself); only the chain pays.  The question is
whether the shipped learners' credit assignment reaches back one
step — the mint move only ever earns through the *next* move's
terminal payout.

The board is synthetic on purpose — option (a) of the Q3 design:
the cleanest possible measurement, no e-graph, no oracle, no torch
in the board itself.  The state is an abstract bank/apply game:

* ``mint(cand=c)`` — admit candidate premise *c* into the bank.
  Costs ``-MINT_COST`` and enables nothing observable.  ``k``
  candidates per board; exactly one is the goal's premise.
* ``apply(law=l)`` — fire a banked law once.  ``base`` pays a small
  ``BASE_PAY`` (the chain-free fallback), the goal's premise pays
  ``WIN_PAY`` and *ends the episode* (the win), every other minted
  law is a dead apply (0.0 — enables nothing, costs budget).
* ``stop`` — end the episode.

The board is honest about its diagnostics: the state publishes
``needed`` — the missing-premise report a real admission loop
surfaces (``missing_hooks``'s sibling).  What it does *not* publish
is the strategy: that the negative-reward mint is the purchase the
+1.0 apply needs.  The scripted arm is told the chain; the learned
arms must assign the terminal payout back across the mint.

The measurement is the *premise delta*: the same held-out cases
played twice — as generated (the chain required) and with the
needed premise pre-minted in the bank (the win degenerates to a
single greedy-visible apply).  An arm that can act on the chain but
cannot *learn* it scores high pre-minted and low without; an arm
that learned the enabling chain closes the delta.

Arms (one protocol — seeded, paired held-out boards, budget 6):

* ``random`` — uniform draws over the legal set.
* ``flood`` — bank-everything baseline: applies ``base``, then mints
  candidates until the budget drains.  Shows flooding the bank does
  not buy the chain under the budget.
* ``greedy`` — the myopic immediate-reward argmax: fires a banked
  paying law, never mints (minting is immediate-negative).  Wins
  every pre-minted board, never wins an unpreminted one — the arm
  the delta was built to expose.
* ``scripted`` — knows the chain: ``apply base`` (the free payout),
  ``mint needed``, ``apply needed``.  The oracle ceiling.
* ``linear-cold`` — ``LearnedPlayer`` on ``CHAIN_SPEC``, untrained.
* ``linear`` — the same arm after ``engine.train_policy`` REINFORCE
  episodes on unpreminted boards (reward-to-go credit, EMA
  baseline).
* ``linear-greedy`` — the trained table, argmax-decoded.
* ``advantage`` — ``AdvantagePlayer``: the same softmax policy with
  a learned state-value baseline, ``G_t - V(s_t)`` per-step credit —
  the arm built for exactly this variance question.
* ``neural`` / ``neural-greedy`` — the ``PolicyNet`` recipe ported
  to the board: MLP ``in->h->h->1`` over the spec's own feature
  rows, Categorical sampling, gamma-discounted returns-to-go, EMA
  baseline, Adam.

Usage::

    .venv/bin/python bench/drivers/policy/credit_chain.py
    .venv/bin/python bench/drivers/policy/credit_chain.py \\
        --train 400 --eval 96 --json /tmp/chain.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent)
)

from gamekit import engine
from gamekit.advantage import AdvantagePlayer
from gamekit.players import (
    Bias,
    DomainSpec,
    LearnedPlayer,
    OpOneHot,
    PlayRates,
    RelFeats,
    StateBroadcast,
)
from mlopt_discovery.meta_arena import Action

__all__ = [
    "BASE_LAW",
    "BASE_PAY",
    "CHAIN_FEATURES",
    "CHAIN_SPEC",
    "MINT_COST",
    "OPS",
    "WIN_PAY",
    "ChainArena",
    "ChainState",
    "gen_cases",
    "greedy_player",
    "legal_actions",
    "measure",
    "scripted_player",
    "train_net",
]

#: The always-banked fallback law — the chain-free consolation payout.
BASE_LAW = "base"
#: Payoff table.  The mint is *negative* — the enabling purchase must
#: be bought before it can pay.
WIN_PAY = 1.0
BASE_PAY = 0.2
MINT_COST = 0.05
#: The move vocabulary.
OPS = ("mint", "apply", "stop")


@dataclass(frozen=True)
class ChainState:
    """The chain board's observable — bank, applied set, diagnostics.

    ``needed`` is the board's missing-premise report: the premise
    the winning rewrite requires, observable to every arm.  ``bank``
    is the law set in force; ``applied`` the laws already fired this
    episode (each fires once — the legal enumerator masks them).
    ``steps`` counts played moves (the learned arms' episode-
    boundary signal); ``done`` terminates the episode.
    """

    candidates: tuple[str, ...]
    bank: frozenset
    applied: frozenset
    needed: str
    won: bool
    steps: int
    done: bool
    budget: int


class ChainArena:
    """The enabling-chain board under the :class:`engine.Board` contract.

    ``case`` is a ``gen_cases`` row — ``{"name", "needed", "k",
    "pre_minted"}``.  ``pre_minted`` seats the needed premise in the
    opening bank: the win degenerates to a one-step apply, which is
    the delta arm of the measurement.
    """

    def __init__(self, case: dict) -> None:
        """Bind the case: candidates, the needed premise, the bank."""
        self.name = str(case["name"])
        self.needed = str(case["needed"])
        self.candidates = tuple(f"c{j}" for j in range(int(case["k"])))
        self._budget = int(case.get("budget", 6))
        self._mint_cost = float(case.get("mint_cost", MINT_COST))
        bank = {BASE_LAW}
        if case.get("pre_minted"):
            bank.add(self.needed)
        self._bank = frozenset(bank)
        self._applied: frozenset = frozenset()
        self._steps = 0
        self._won = False
        self._done = False

    def observe(self) -> ChainState:
        """Return the board snapshot — the read the players featurize."""
        return ChainState(
            candidates=self.candidates,
            bank=self._bank,
            applied=self._applied,
            needed=self.needed,
            won=self._won,
            steps=self._steps,
            done=self._done,
            budget=self._budget,
        )

    def step(self, action: Any) -> tuple[ChainState, engine.Report]:
        """Apply *action*; return ``(state', report)``.

        Illegal moves (unbanked apply, repeat mint, off-vocabulary op)
        are honest no-ops: ``applied=False``, zero reward, the move
        still spends a step of budget.
        """
        op = getattr(action, "op", None)
        params = getattr(action, "params", {}) or {}
        reward = 0.0
        applied = False
        note = "illegal"
        if self._done:
            return self.observe(), engine.Report(
                action, 0.0, terminal=True, applied=False, note="done"
            )
        self._steps += 1
        if op == "stop":
            self._done = True
            applied = True
            note = "stop"
        elif op == "mint":
            cand = str(params.get("cand", ""))
            if cand in self.candidates and cand not in self._bank:
                self._bank = self._bank | {cand}
                reward = -self._mint_cost
                applied = True
                note = f"minted {cand}"
        elif op == "apply":
            law = str(params.get("law", ""))
            if law in self._bank and law not in self._applied:
                self._applied = self._applied | {law}
                applied = True
                if law == self.needed:
                    reward = WIN_PAY
                    self._won = True
                    self._done = True
                    note = "win"
                elif law == BASE_LAW:
                    reward = BASE_PAY
                    note = "base"
                else:
                    note = "dead"
        if self._steps >= self._budget:
            self._done = True
        state = self.observe()
        return state, engine.Report(
            action,
            reward=reward,
            terminal=self._done,
            applied=applied,
            note=note,
            detail={"won": self._won},
        )


def gen_cases(
    seed: int,
    n: int,
    *,
    k: int = 8,
    pre_minted: bool = False,
    budget: int = 6,
) -> list[dict]:
    """Return *n* chain boards — which candidate is needed varies.

    The draw is the needed premise's index, uniform over the *k*
    candidates; ``pre_minted`` is a flag on the same draws, so
    ``gen_cases(s, n, pre_minted=True)`` pairs board-for-board with
    the unpreminted stream — the delta is within-board.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        needed = f"c{rng.randrange(k)}"
        out.append(
            {
                "name": f"chain{i}:{needed}",
                "needed": needed,
                "k": k,
                "pre_minted": pre_minted,
                "budget": budget,
            }
        )
    return out


def legal_actions(state: ChainState) -> tuple[Action, ...]:
    """Enumerate the legal set — mints, banked applies, ``stop``."""
    acts = [
        Action("mint", {"cand": c})
        for c in state.candidates
        if c not in state.bank
    ]
    acts += [
        Action("apply", {"law": law})
        for law in sorted(state.bank)
        if law not in state.applied
    ]
    acts.append(Action("stop", {}))
    return tuple(acts)


# ---------------------------------------------------------------------------
#  Feature schema — the learned arms' vocabulary
# ---------------------------------------------------------------------------


def _state_feats(state: ChainState) -> dict[str, float]:
    """Return the ``st:*`` broadcast — progress and the armed flag."""
    width = len(state.candidates) + 1  # candidates + the base law
    return {
        "st:steps": state.steps / max(state.budget, 1),
        "st:banked": len(state.bank) / width,
        "st:applied": len(state.applied) / width,
        "st:needed_minted": 1.0 if state.needed in state.bank else 0.0,
        "st:won": 1.0 if state.won else 0.0,
    }


def _rel_feats(
    state: ChainState, action: Action, row: dict[str, float]
) -> dict[str, float]:
    """Return the state-relative ``a:*``/``x:*`` block.

    The discriminative columns of the whole probe: ``a:mint:needed``
    marks the mint the goal named (the move whose credit is entirely
    deferred), ``a:apply:needed`` the terminal apply it enables, and
    ``x:stop:armed`` the premature-quit signature — stopping while
    the armed premise sits unspent.
    """
    del row
    out: dict[str, float] = {}
    if action.op == "mint":
        if action.params.get("cand") == state.needed:
            out["a:mint:needed"] = 1.0
    elif action.op == "apply":
        law = action.params.get("law")
        if law == state.needed:
            out["a:apply:needed"] = 1.0
        elif law == BASE_LAW:
            out["a:apply:base"] = 1.0
        else:
            out["a:apply:dead"] = 1.0
    elif action.op == "stop" and (
        state.needed in state.bank and state.needed not in state.applied
    ):
        out["x:stop:armed"] = 1.0
    return out


CHAIN_FEATURES: tuple[str, ...] = (
    "bias",
    "st:steps",
    "st:banked",
    "st:applied",
    "st:needed_minted",
    "st:won",
    "h:mint",
    "h:apply",
    "h:stop",
    "op:mint",
    "op:apply",
    "op:stop",
    "a:mint:needed",
    "a:apply:needed",
    "a:apply:base",
    "a:apply:dead",
    "x:stop:armed",
)

#: The chain domain's spec — cold (no baked prior): the whole point
#: is whether training finds the chain from nothing.
CHAIN_SPEC = DomainSpec.build(
    features=CHAIN_FEATURES,
    legal=legal_actions,
    parts=(
        Bias(),
        StateBroadcast(_state_feats),
        PlayRates(OPS),
        OpOneHot(frozenset(f"op:{op}" for op in OPS)),
        RelFeats(_rel_feats),
    ),
)

featurize = CHAIN_SPEC.featurize


# ---------------------------------------------------------------------------
#  Arms — scripted, myopic, flood, chance
# ---------------------------------------------------------------------------


def scripted_player(state: ChainState) -> Action:
    """Play the known chain: ``base`` first, then mint→apply needed.

    The oracle ceiling: banks the free ``BASE_PAY`` before the win
    closes the episode (``apply(needed)`` is terminal), so the
    optimal line is base → mint → apply.
    """
    if BASE_LAW in state.bank and BASE_LAW not in state.applied:
        return Action("apply", {"law": BASE_LAW})
    if state.needed not in state.bank:
        return Action("mint", {"cand": state.needed})
    return Action("apply", {"law": state.needed})


def greedy_player(state: ChainState) -> Action:
    """Myopic immediate-reward argmax — the arm the delta exposes.

    Fires the best *immediate* payout among banked laws
    (``needed`` > ``base`` > stop); minting is immediate-negative so
    the chain is invisible to it.  Wins every pre-minted board,
    settles for ``BASE_PAY`` otherwise.
    """
    if state.needed in state.bank and state.needed not in state.applied:
        return Action("apply", {"law": state.needed})
    if BASE_LAW in state.bank and BASE_LAW not in state.applied:
        return Action("apply", {"law": BASE_LAW})
    return Action("stop", {})


def _flood_player() -> Any:
    """Bank-everything baseline: ``base``, then mints till the budget."""
    minted: set = set()

    def player(state: ChainState) -> Action:
        if BASE_LAW in state.bank and BASE_LAW not in state.applied:
            return Action("apply", {"law": BASE_LAW})
        for c in state.candidates:
            if c not in state.bank and c not in minted:
                minted.add(c)
                return Action("mint", {"cand": c})
        for law in sorted(state.bank):
            if law not in state.applied:
                return Action("apply", {"law": law})
        return Action("stop", {})

    return player


def _random_player(seed: int) -> Any:
    """Uniform draws over the legal set — the honesty floor."""
    rng = random.Random(seed)

    def player(state: ChainState) -> Action:
        acts = legal_actions(state)
        return acts[rng.randrange(len(acts))]

    return player


# ---------------------------------------------------------------------------
#  The neural arm — the PolicyNet recipe on the spec's own feature rows
# ---------------------------------------------------------------------------


def _net_rows(state: ChainState, acts: tuple, hist: Any) -> list:
    """One input row per legal action — the spec's own featurizer."""
    return [
        [featurize(state, a, hist).get(k, 0.0) for k in CHAIN_FEATURES]
        for a in acts
    ]


def train_net(
    cases: list[dict],
    budget: int,
    *,
    episodes: int,
    hidden: int = 32,
    lr: float = 3e-3,
    gamma: float = 0.95,
    seed: int = 0,
) -> tuple[Any, list[float]]:
    """REINFORCE over chain episodes — the shipped net recipe verbatim.

    One sampled episode per case (cycling), returns-to-go discounted
    by *gamma*, a cross-episode EMA baseline (the ``adv_mode="ema"``
    arm of ``policy_rl_arms``), one Adam step per episode.  Returns
    the eval-mode model and the per-100-episode mean totals.
    """
    import torch
    from torch import nn
    from torch.distributions import Categorical

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(len(CHAIN_FEATURES), hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = 0.0
    means: list[float] = []
    block: list[float] = []
    for ep in range(episodes):
        arena = ChainArena(cases[ep % len(cases)])
        hist = CHAIN_SPEC.new_hist()
        logps: list = []
        rews: list[float] = []
        while len(rews) < budget:
            state = arena.observe()
            if state.done:
                break
            acts = legal_actions(state)
            x = torch.tensor(
                _net_rows(state, acts, hist), dtype=torch.float32
            )
            dist = Categorical(logits=model(x).squeeze(-1))
            i = dist.sample()
            logps.append(dist.log_prob(i))
            action = acts[int(i.item())]
            _state, rep = arena.step(action)
            rews.append(rep.reward)
            hist.record(action.op)
        running, returns = 0.0, []
        for r in reversed(rews):
            running = r + gamma * running
            returns.append(running)
        returns.reverse()
        adv = torch.tensor(returns, dtype=torch.float32)
        if len(adv):
            ema += 0.2 * (float(adv[0]) - ema)
        adv = adv - ema
        loss = -(torch.stack(logps) * adv).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        block.append(sum(rews))
        if len(block) == 100:
            means.append(round(sum(block) / len(block), 4))
            block = []
    if block:
        means.append(round(sum(block) / len(block), 4))
    return model.eval(), means


def _net_player(model: Any, seed: int, greedy: bool) -> Any:
    """Return a ``run_episode``-contract player decoding *model*'s logits."""
    import torch
    from torch.distributions import Categorical

    torch.manual_seed(seed)
    hist = CHAIN_SPEC.new_hist()
    prev_steps = -1

    def player(state: ChainState) -> Action:
        nonlocal prev_steps, hist
        if prev_steps < 0 or (state.steps == 0 and prev_steps > 0):
            hist = CHAIN_SPEC.new_hist()
        prev_steps = state.steps
        acts = legal_actions(state)
        x = torch.tensor(
            _net_rows(state, acts, hist), dtype=torch.float32
        )
        with torch.no_grad():
            logits = model(x).squeeze(-1)
        if greedy:
            i = int(torch.argmax(logits))
        else:
            i = int(Categorical(logits=logits).sample())
        action = acts[i]
        hist.record(action.op)
        return action

    return player


# ---------------------------------------------------------------------------
#  The measurement — paired boards, with/without the premise
# ---------------------------------------------------------------------------


def _eval_rows(arms: dict, cases: list[dict], budget: int) -> dict:
    """Play every arm on *cases*; return per-arm aggregate rows."""
    out: dict[str, dict] = {}
    for name, mk in arms.items():
        rewards: list[float] = []
        wins: list[bool] = []
        steps: list[int] = []
        for i, case in enumerate(cases):
            traj = engine.run_episode(ChainArena(case), mk(i), budget)
            rewards.append(traj.total)
            wins.append(any(r.detail.get("won") for r in traj.reports))
            steps.append(len(traj.reports))
        n = max(len(cases), 1)
        out[name] = {
            "cases": len(cases),
            "reward": round(sum(rewards) / n, 4),
            "win": round(sum(wins) / n, 4),
            "steps": round(sum(steps) / n, 2),
        }
    return out


def _arms(
    seed: int,
    lin: Any,
    adv: Any,
    model: Any,
) -> dict[str, Any]:
    """Assemble the arm factories ``i -> player`` for one eval pass."""
    return {
        "random": lambda i: _random_player(seed + 10_000 + i),
        "flood": lambda i: _flood_player(),
        "greedy": lambda i: greedy_player,
        "scripted": lambda i: scripted_player,
        "linear-cold": lambda i: LearnedPlayer(
            seed + 40_000 + i, spec=CHAIN_SPEC, weights={}, learn=False
        ),
        "linear": lambda i: lin.frozen(seed + 20_000 + i),
        "linear-greedy": lambda i: lin.frozen(
            seed + 30_000 + i, greedy=True
        ),
        "advantage": lambda i: adv.frozen(seed + 50_000 + i),
        "neural": lambda i: _net_player(
            model, seed + 60_000 + i, False
        ),
        "neural-greedy": lambda i: _net_player(
            model, seed + 70_000 + i, True
        ),
    }


def measure(
    *,
    seed: int = 0,
    train_cases: int = 400,
    eval_cases: int = 96,
    k: int = 8,
    budget: int = 6,
    epsilon: float = 0.15,
    net_episodes: int = 1200,
    hidden: int = 32,
) -> dict[str, Any]:
    """Run the full protocol; return the paired result document.

    The learned arms train on *unpreminted* boards only — the chain
    curriculum.  Eval runs the same held-out seed twice: as drawn
    (``nopre`` — the win needs the mint) and pre-minted (``pre`` —
    the win is one greedy-visible apply).  ``delta`` is the
    per-arm pre-minus-nopre reward gap: for scripted it is the mint
    cost saved; for an arm that cannot learn the chain it is the
    whole win.
    """
    train = gen_cases(seed, train_cases, k=k, budget=budget)
    ev_no = gen_cases(seed + 7919, eval_cases, k=k, budget=budget)
    ev_pre = gen_cases(
        seed + 7919, eval_cases, k=k, budget=budget, pre_minted=True
    )
    lin = LearnedPlayer(
        seed, spec=CHAIN_SPEC, weights={}, epsilon=epsilon
    )
    totals_lin = engine.train_policy(
        lin, train, ChainArena, budget=budget
    )
    adv = AdvantagePlayer(
        seed,
        spec=CHAIN_SPEC,
        weights={},
        epsilon=epsilon,
        state_feats=lambda s, _h: _state_feats(s),
    )
    totals_adv = engine.train_policy(
        adv, train, ChainArena, budget=budget
    )
    model, net_means = train_net(
        train, budget, episodes=net_episodes, hidden=hidden, seed=seed
    )
    arms = _arms(seed, lin, adv, model)
    nopre = _eval_rows(arms, ev_no, budget)
    pre = _eval_rows(arms, ev_pre, budget)
    table = {
        name: {
            "nopre": nopre[name],
            "pre": pre[name],
            "delta": round(
                pre[name]["reward"] - nopre[name]["reward"], 4
            ),
            "win_delta": round(
                pre[name]["win"] - nopre[name]["win"], 4
            ),
        }
        for name in arms
    }
    return {
        "seed": seed,
        "train_cases": train_cases,
        "eval_cases": eval_cases,
        "k": k,
        "budget": budget,
        "epsilon": epsilon,
        "net_episodes": net_episodes,
        "train_lin": [
            round(
                sum(totals_lin[i : i + 50])
                / min(50, max(len(totals_lin) - i, 1)),
                3,
            )
            for i in range(0, len(totals_lin), 50)
        ],
        "train_adv": [
            round(
                sum(totals_adv[i : i + 50])
                / min(50, max(len(totals_adv) - i, 1)),
                3,
            )
            for i in range(0, len(totals_adv), 50)
        ],
        "train_net": net_means,
        "table": table,
    }


def _render(doc: dict[str, Any]) -> str:
    """Render the paired arms table — reward, win-rate, delta."""
    lines = [
        f"{'player':<14} {'r_nopre':>8} {'win':>6} {'r_pre':>8} "
        f"{'win':>6} {'delta':>7} {'dwin':>6}",
        "-" * 62,
    ]
    for name, row in doc["table"].items():
        lines.append(
            f"{name:<14} {row['nopre']['reward']:>8.3f} "
            f"{row['nopre']['win']:>6.2f} {row['pre']['reward']:>8.3f} "
            f"{row['pre']['win']:>6.2f} {row['delta']:>7.3f} "
            f"{row['win_delta']:>6.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Train the learned arms on the chain curriculum; eval both ways."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=400)
    ap.add_argument("--eval-cases", type=int, default=96)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--net-episodes", type=int, default=1200)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    doc = measure(
        seed=args.seed,
        train_cases=args.train_cases,
        eval_cases=args.eval_cases,
        k=args.k,
        budget=args.budget,
        epsilon=args.epsilon,
        net_episodes=args.net_episodes,
        hidden=args.hidden,
    )
    print(
        "== credit-chain — enabling-chain curriculum =="
    )  # stdout-compat
    print(_render(doc))  # stdout-compat
    print(f"train linear means/50: {doc['train_lin']}")  # stdout-compat
    print(
        f"train advantage means/50: {doc['train_adv']}"
    )  # stdout-compat
    print(
        f"train neural means/100: {doc['train_net']}"
    )  # stdout-compat
    if args.json:
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
