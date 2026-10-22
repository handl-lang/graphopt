"""Strassen rediscovery, take two — a lookahead (MCTS) arm.

Q2 follow-up to ``bench/drivers/probes/strassen_probe.py`` /
``docs/results_strassen.md``.  That measurement showed the board
provably expresses Strassen (the scripted line reaches verified
rank 7 in 29 moves) while greedy, random and cold-linear arms
stall at rank 8 — a ~15-move coordinated valley through invalid
space with no positive prefix.  The open question: is one-step
greediness the whole problem, or is the valley deeper than
search?

This driver seats a minimal UCB-tree planner on the same board:

* **selection** — UCB1 over the board's own ``legal_actions``
  (children Q min-max normalised per node, untried actions
  first);
* **expansion** — lazy, one node per simulation;
* **rollout** — a dig/repair cycle on a cloned board: at valid
  states take a *collapse* when one exists (two terms with
  proportional factors in two modes → one flip zeroes a factor →
  ``drop`` banks a free rank -1 — the mechanism behind
  ``tests/test_discovery_bilinear_game.py``'s engineered-zero
  pin), else flip or cut to leave the current scheme; at invalid
  states dig deeper with probability ``p_dig`` else repair to
  validity with tempered top-``k`` patches (softmax over the L1
  residual table, the same candidate table ``_GreedyRepair``
  uses);
* **backup** — leaf value is the best quality
  ``-rank - 16·‖R‖₁`` seen along the rollout; a rollout that
  reaches a valid scheme of rank ≤ 7 terminates the sim board —
  the win is discovered, not assumed.

The tree re-roots on each committed move, so a discovered
winning line persists across decisions.

Usage::

    .venv/bin/python bench/drivers/probes/strassen_mcts.py \
        --sims 64,256,1024 --episodes 8 --budget 48

The measurement: does sims-per-move compute cross the valley
(best valid rank / episodes to first rank 7), or does lookahead
at this scale still stall at 8?  Both answers are real results.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from statistics import fmean

from mlopt_discovery import bilinear_game as bg
from mlopt_discovery.meta_arena import Action

#: Legal flip shifts — a collapse flip is only offered when its
#: λ is in the board's own ``FLIP_LAMBDAS``.
_FLAM = set(bg.FLIP_LAMBDAS)


def _board_from(state: bg.BilinearState) -> bg.BilinearBoard:
    """Rebuild a mutable board from a read-only snapshot."""
    b = bg.BilinearBoard(("sim", state.terms))
    b.pending = list(state.pending)
    b.steps = state.steps
    b.done = state.done
    b.best_rank = state.best_rank
    return b


def _quality(state: bg.BilinearState) -> float:
    """``-rank - 16·‖R‖₁`` — the score the reward differentiates."""
    return -len(state.terms) - bg.BILINEAR_RESIDUAL_WEIGHT * float(
        state.norm
    )


def collapse_moves(terms: tuple) -> list:
    """Return flip params that zero a factor → ``drop`` banks rank -1.

    Two committed terms with proportional factors in two modes
    ``(m, s)`` collapse under the flip identity: ``f_s(t_i) -=
    λ·f_s(t_j)`` with ``λ = f_s(t_i)/f_s(t_j)`` zeroes term *i*'s
    *s*-factor while the pair's tensor sum — hence validity — is
    preserved.  The zeroed term is then ``drop``-able: rank -1 at
    zero residual cost.  Only legal λ (the board's
    ``FLIP_LAMBDAS``) count.
    """
    out = []
    for i in range(len(terms)):
        for j in range(len(terms)):
            if i == j:
                continue
            ti, tj = terms[i], terms[j]
            for m in range(3):
                rho = bg._ratio(tj[m], ti[m])
                if rho is None or rho == 0:
                    continue
                for s in range(3):
                    if s == m:
                        continue
                    lam = bg._ratio(ti[s], tj[s])
                    if lam is None or lam == 0 or lam not in _FLAM:
                        continue
                    out.append(
                        {"i": i, "j": j, "m": m, "s": s, "lam": lam}
                    )
    return out


class _Node:
    """One tree node: a state plus visit statistics per action."""

    __slots__ = ("acts", "children", "n", "state", "w")

    def __init__(self, state: bg.BilinearState) -> None:
        self.state = state
        self.acts: tuple | None = None
        self.children: dict = {}
        self.n = None
        self.w = None


class MCTSPlayer:
    """UCB-tree planner over the bilinear move graph.

    ``sims`` simulations per committed move; the tree re-roots on
    the committed child when the next observed state matches (a
    discovered winning line survives).  All randomness flows
    through two seeded generators — same seed, same play.
    """

    def __init__(
        self,
        *,
        sims: int = 64,
        seed: int = 0,
        c: float = 1.4,
        rollout_steps: int = 36,
        tau: float = 4.0,
        topk: int = 8,
        p_dig: float = 0.35,
        p_flip: float = 0.15,
    ) -> None:
        self.sims = sims
        self.c = c
        self.rollout_steps = rollout_steps
        self.tau = tau
        self.topk = topk
        self.p_dig = p_dig
        self.p_flip = p_flip
        self.rng = random.Random(seed)
        self._root: _Node | None = None
        self._plan: list = []
        self._np = None
        self._npr = None
        self._mat = None

    # -- lazy numpy: the candidate-term L1 table (the playbook's own)
    def _tables(self) -> None:
        if self._mat is None:
            import numpy as np

            self._np = np
            self._npr = np.random.default_rng(
                self.rng.randrange(1 << 60)
            )
            self._mat = bg._patch_matrix().astype(np.float32)

    # -- tree machinery -------------------------------------------------

    def _expand(self, nd: _Node) -> None:
        nd.acts = bg.legal_actions(nd.state)
        nd.n = self._np.zeros(len(nd.acts))
        nd.w = self._np.zeros(len(nd.acts))

    def _select(self, nd: _Node) -> int:
        """UCB1: untried actions first, else norm-Q + c·√(ln N/n)."""
        np = self._np
        n, w = nd.n, nd.w
        untried = np.flatnonzero(n == 0)
        if len(untried):
            return int(self.rng.choice(untried.tolist()))
        total = float(n.sum())
        q = w / n
        lo, hi = float(q.min()), float(q.max())
        s = (q - lo) / (hi - lo + 1e-9)
        u = self.c * np.sqrt(math.log(total) / n)
        return int(np.argmax(s + u))

    def _sim(self, root: _Node, state: bg.BilinearState) -> None:
        """One selection→expansion→rollout→backup pass."""
        board = _board_from(state)
        path = []
        nd = root
        while not nd.state.done:
            if nd.acts is None:
                self._expand(nd)
            if not nd.acts:
                break
            a = self._select(nd)
            st2, _rep = board.step(nd.acts[a])
            child = nd.children.get(a)
            if child is None:
                child = _Node(st2)
                nd.children[a] = child
                path.append((nd, a))
                nd = child
                break
            path.append((nd, a))
            nd = child
        v = (
            _quality(nd.state)
            if nd.state.done
            else self._rollout(board)
        )
        for parent, a in path:
            parent.n[a] += 1.0
            parent.w[a] += v

    # -- rollout policy: dig / repair / free-collapse --------------------

    def _collapse(self, board: bg.BilinearBoard, st) -> bool:
        """Take a collapse if this valid state offers one."""
        cm = collapse_moves(st.terms)
        if not cm:
            return False
        board.step(Action("flip", self.rng.choice(cm)))
        st2 = board.observe()
        drops = [
            i
            for i, t in enumerate(st2.terms)
            if any(bg._is_zero(f) for f in t)
        ]
        board.step(Action("drop", {"i": drops[0]}))
        return True

    def _try_flip(self, board: bg.BilinearBoard, st) -> bool:
        """One random legal flip (blind pick, retry on the raise)."""
        for _ in range(4):
            i = self.rng.randrange(len(st.terms))
            j = self.rng.randrange(len(st.terms))
            m = self.rng.randrange(3)
            s = self.rng.randrange(3)
            if i == j or m == s:
                continue
            try:
                board.step(
                    Action(
                        "flip",
                        {
                            "i": i,
                            "j": j,
                            "m": m,
                            "s": s,
                            "lam": self.rng.choice(bg.FLIP_LAMBDAS),
                        },
                    )
                )
                return True
            except ValueError:
                continue
        return False

    def _patch_pick(self, board: bg.BilinearBoard, st) -> Action:
        """Tempered top-k L1 patch — returns the staged ``pick`` for u.

        Samples a whole dictionary term by softmax over
        ``-‖R - u⊗v⊗w‖₁/τ`` within the best ``topk`` candidates,
        then commits its three factors through the staged picks
        (the remaining two ride ``self._plan``).
        """
        np = self._np
        res = np.array(
            [float(x) for x in st.residual], dtype=np.float32
        )
        sc = np.abs(res[None, :] - self._mat).sum(axis=1)
        k = min(self.topk, sc.size)
        idx = np.argpartition(sc, k - 1)[:k]
        w = np.exp(-(sc[idx] - sc[idx].min()) / self.tau)
        w /= w.sum()
        j = int(self._npr.choice(idx, p=w))
        nf = len(bg.FACTORS)
        iu, rem = divmod(j, nf * nf)
        iv, iw = divmod(rem, nf)
        self._plan = [iv, iw]
        return Action("pick", {"k": iu})

    def _rollout(self, board: bg.BilinearBoard) -> float:
        """Play the dig/repair cycle to the cap; return best quality seen."""
        self._plan = []
        repairing = False
        d = 0
        best = _quality(board.observe())
        while not board.done and d < self.rollout_steps:
            st = board.observe()
            if st.pending:
                a = (
                    Action("pick", {"k": self._plan.pop(0)})
                    if self._plan
                    else Action("abort", {})
                )
            elif st.valid:
                if self._collapse(board, st):
                    d += 2
                    continue
                repairing = False
                if self.rng.random() < self.p_flip and self._try_flip(
                    board, st
                ):
                    d += 1
                    continue
                a = Action(
                    "cut", {"i": self.rng.randrange(len(st.terms))}
                )
            elif (
                not repairing
                and self.rng.random() < self.p_dig
                and st.terms
            ):
                a = Action(
                    "cut", {"i": self.rng.randrange(len(st.terms))}
                )
            else:
                repairing = True
                a = self._patch_pick(board, st)
            board.step(a)
            d += 1
            best = max(best, _quality(board.observe()))
        return best

    # -- the player contract ---------------------------------------------

    def __call__(self, state: bg.BilinearState) -> Action:
        """Search ``sims`` rollouts from *state*; commit most-visited."""
        self._tables()
        if self._root is None or self._root.state != state:
            self._root = _Node(state)
        root = self._root
        for _ in range(self.sims):
            self._sim(root, state)
        if not root.children:
            return Action("stop", {})
        best_n = max(root.n[a] for a in root.children)
        tied = [a for a in root.children if root.n[a] == best_n]
        a = tied[0] if len(tied) == 1 else self.rng.choice(tied)
        self._root = root.children[a]
        return root.acts[a]


def _arms(domain: str, seed: int, sims: list) -> dict:
    """Build the arm factories: the anchors plus one MCTS per budget."""
    from gamekit.players import registry
    from mlopt_discovery import play

    d = play.DOMAINS[domain]
    arms = {
        "scripted": lambda i: d.scripted(),
        "playbook": registry.make(
            "playbook", domain, seed=seed, ctx={}
        ),
        "random": registry.make("random", domain, seed=seed, ctx={}),
        "linear": registry.make("linear", domain, seed=seed, ctx={}),
    }
    for s in sims:
        arms[f"mcts{s}"] = lambda i, s=s: MCTSPlayer(
            sims=s, seed=seed * 1009 + i
        )
    return arms


def _episode(board_of, case, player, budget: int) -> dict:
    """One episode — best valid rank, reward, step count."""
    from gamekit import engine

    traj = engine.run_episode(board_of(case), player, budget)
    best = min(
        (r.detail["best_rank"] for r in traj.reports),
        default=None,
    )
    term = traj.terminal
    return {
        "steps": len(traj.reports),
        "reward": round(traj.total, 2),
        "best_rank": best,
        "won": bool(term is not None and term.cost <= 7),
    }


def main(argv: list | None = None) -> int:
    """Paired episodes per arm; report the rank distribution per budget."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="bilinear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument(
        "--sims",
        default="64,256",
        help="comma-separated simulations-per-move budgets",
    )
    ap.add_argument(
        "--det-episodes",
        type=int,
        default=10,
        help="episodes for the deterministic anchors",
    )
    args = ap.parse_args(argv)

    from mlopt_discovery import play

    d = play.DOMAINS[args.domain]
    sims = [int(s) for s in args.sims.split(",") if s]
    cases = d.cases(args.seed + 7919, args.episodes)
    arms = _arms(args.domain, args.seed, sims)

    print(
        f"== strassen mcts — domain {args.domain} ==",
        flush=True,
    )  # stdout-compat
    print(
        f"{'arm':<10} {'eps':>4} {'won':>4} {'pct<=7':>8} "
        f"{'rank~':>6} {'reward~':>9} {'steps~':>6} {'s':>7}"
    )
    print("-" * 54)
    summary = {}
    for name, mk in arms.items():
        n = (
            args.det_episodes
            if name in ("scripted", "playbook")
            else args.episodes
        )
        t0 = time.time()
        rows = []
        first_win = None
        for i in range(n):
            r = _episode(
                d.board_of, cases[i % len(cases)], mk(i), args.budget
            )
            rows.append(r)
            if r["won"] and first_win is None:
                first_win = i
        won = sum(r["won"] for r in rows)
        ranks = [
            r["best_rank"] for r in rows if r["best_rank"] is not None
        ]
        row = {
            "eps": n,
            "won": won,
            "pct7": round(100.0 * won / n, 1),
            "mean_rank": round(fmean(ranks), 2) if ranks else None,
            "mean_reward": round(fmean(r["reward"] for r in rows), 2),
            "mean_steps": round(fmean(r["steps"] for r in rows), 1),
            "first_win": first_win,
            "secs": round(time.time() - t0, 1),
        }
        summary[name] = row
        print(  # stdout-compat
            f"{name:<10} {n:>4} {won:>4} {row['pct7']:>7.1f}% "
            f"{row['mean_rank']:>6} {row['mean_reward']:>9} "
            f"{row['mean_steps']:>6} {row['secs']:>7}",
            flush=True,
        )
    mcts_arms = [a for a in summary if a.startswith("mcts")]
    verdict = (
        "LOOKAHEAD FIXED IT — MCTS reached rank ≤ 7"
        if any(summary[a]["won"] for a in mcts_arms)
        else "STILL VALLEY — no MCTS arm reached rank ≤ 7; "
        "the blocker is deeper than search at this scale"
    )
    print("verdict: " + verdict, flush=True)  # stdout-compat
    return 0


if __name__ == "__main__":
    sys.exit(main())
