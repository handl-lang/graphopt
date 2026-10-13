"""The measured claim for the open game — does the learned arm beat
random at finding useful-new laws per evaluation, on held-out pages?

The open domain (:mod:`catopt_discovery.open_game`) plays the
enumerated candidate stream as pages: a move evaluates one candidate,
the reward is its verdict tier.  This module *measures* the claim the
game exists to support — that a learned ordering of the page raises
yield-per-eval — and pins it as a regression test.

Two honest problems with the naive recipe, both handled here:

1. **``gen_cases`` ignores its seed.**  The enumerated stream is
   deterministic, so ``gen_cases(seed, n)`` and
   ``gen_cases(seed + 7919, n)`` return the *same* pages — the
   prime-offset idiom is not a held-out split.  ``gen_cases`` now
   takes ``start``/``step``; the split here is disjoint by page
   parity (even pages train, odd pages evaluate).

2. **The useful candidates are ~1.3% of the stream and front-loaded**
   (13 of the 15 useful laws sit in the first 58% of the stream), so
   a contiguous train/eval cut leaves the eval half nearly empty (3
   useful in 584 candidates).  The even/odd split balances the two
   halves (8 useful train, 7 useful eval).

The measured result (seed 0, deterministic under ``torch.manual_seed``):

* the learned arm beats random on reward/episode, useful/episode and
  useful/eval — and never loses to random across train seeds 0-5
  (measured offline); the reward margin is modest (≈+0.05/episode)
  because the game's per-eval cost is calibrated to the *average*
  candidate, so ordering barely moves the episode total;
* the learned *ranking* (average precision) beats random but stays
  well below the cost-delta heuristic, which already finds every
  useful candidate — so the honest verdict is "learned beats random,
  heuristic still beats learned".

A run is ~30 s (bounded): 36 training pages + 36 eval pages, 3 arms.
"""

from __future__ import annotations

import pytest
import torch
from catopt_discovery import engine, play
from catopt_discovery import open_game as og
from catopt_discovery import proposal as lp

#: Page size — the episode horizon (budget = page + 1).
PAGE = 16
#: Pages per half of the even/odd held-out split.
HALF = 36
BUDGET = PAGE + 1
SEED = 0


def _split() -> tuple[list[dict], list[dict]]:
    """Disjoint even/odd pages — the genuinely held-out split."""
    train = og.gen_cases(SEED, HALF, page=PAGE, start=0, step=2)
    ev = og.gen_cases(SEED, HALF, page=PAGE, start=1, step=2)
    return train, ev


def _play_arm(mk, ev: list[dict]) -> dict:
    """Play one arm over every eval page; tally yield and reward."""
    found = evals = 0
    reward = 0.0
    for i, case in enumerate(ev):
        traj = engine.run_episode(og.OpenArena(case), mk(i), BUDGET)
        reward += traj.total
        for r in traj.reports:
            evals += 1
            if r.note and r.note.split(":", 1)[0] == "useful":
                found += 1
    n = len(ev)
    return {
        "reward": reward / n,
        "useful_per_ep": found / n,
        "useful_per_eval": found / max(evals, 1),
        "evals_per_ep": evals / n,
        "found": found,
    }


@pytest.fixture(scope="module")
def measured() -> dict:
    """Train once, evaluate the arms, and rank the held-out pool."""
    torch.manual_seed(SEED)
    train, ev = _split()
    player = og.OpenLearnedPlayer(seed=SEED)
    engine.train_policy(player, train, og.OpenArena, budget=BUDGET)
    arms = play._arms(play.DOMAINS["open"], player, SEED)
    table = {name: _play_arm(arms[name], ev) for name in arms}
    pool: list = []
    outcomes: dict = {}
    for case in ev:
        st = og.OpenArena(case).observe()
        cands = list(st.pool)
        outs = lp.evaluate(cands)
        for c, o in zip(cands, outs, strict=True):
            outcomes[len(pool)] = o
            pool.append(c)
    ranking = lp.rank_pool(pool, outcomes, scorer=player.rank_score)
    return {
        "player": player,
        "eval": ev,
        "table": table,
        "ranking": ranking,
    }


class TestHeldOutYield:
    """The learned arm's episode yield on the held-out pages."""

    def test_learned_beats_random_on_reward(self, measured):
        """Learned reward/episode clears random by a real margin."""
        t = measured["table"]
        assert t["learned"]["reward"] > t["random"]["reward"]
        assert t["learned"]["reward"] - t["random"]["reward"] > 0.02

    def test_learned_beats_random_on_useful_per_eval(self, measured):
        """Learned finds at least as many useful laws per eval."""
        t = measured["table"]
        assert (
            t["learned"]["useful_per_eval"]
            >= t["random"]["useful_per_eval"]
        )

    def test_learned_beats_random_on_useful_per_episode(self, measured):
        """Learned finds at least as many useful laws per page."""
        t = measured["table"]
        assert (
            t["learned"]["useful_per_ep"]
            >= t["random"]["useful_per_ep"]
        )

    def test_greedy_finds_more_but_costs_more(self, measured):
        """The honest counterweight: greedy evaluates the whole page,
        finds *more* useful laws, yet scores *less* reward — the
        per-eval cost dominates once the page is sparse."""
        t = measured["table"]
        assert (
            t["learned-greedy"]["useful_per_ep"]
            >= t["random"]["useful_per_ep"]
        )
        assert t["learned-greedy"]["reward"] < t["random"]["reward"]


class TestRanking:
    """Average precision of the page ordering, learned vs the rest."""

    def test_heuristic_ranking_beats_random(self, measured):
        """The cost-delta heuristic is a strong ranker on this pool."""
        r = measured["ranking"]
        assert r["useful"] > 0
        assert r["heuristic_ap"] > r["random_ap"]

    def test_learned_ranking_beats_random(self, measured):
        """The trained policy's ordering clears the random baseline."""
        r = measured["ranking"]
        assert r["learned_ap"] > r["random_ap"]

    def test_learned_ranking_below_heuristic(self, measured):
        """The honest negative: the learned prior does *not* beat the
        cost-delta heuristic — the heuristic already finds every
        useful candidate, so the learned arm's edge over random comes
        from coverage, not a sharper shape ranking."""
        r = measured["ranking"]
        assert r["learned_ap"] < r["heuristic_ap"]


def test_gen_cases_start_step():
    """``start``/``step`` select the page region — the held-out lever.

    Defaults reproduce the original behaviour (pages 0, 1, 2, …);
    ``start=1, step=2`` walks the odd pages, disjoint from the even
    ones the train half uses.
    """
    default = og.gen_cases(0, 3, page=10)
    assert [c["offset"] for c in default] == [0, 10, 20]
    odd = og.gen_cases(0, 3, page=10, start=1, step=2)
    assert [c["offset"] for c in odd] == [10, 30, 50]


def test_split_pages_are_disjoint():
    """The train/eval halves share no page — genuinely held out."""
    train, ev = _split()
    train_off = {c["offset"] for c in train}
    ev_off = {c["offset"] for c in ev}
    assert train_off and ev_off
    assert not (train_off & ev_off)
    assert all(c["page"] == PAGE for c in train + ev)


def test_rank_score_matches_step0_ordering():
    """``rank_score`` is the policy's step-0 ordering over a page.

    The scorer is ``w · _cand_feats(cand)`` — the ``evaluate`` moves
    share every state/move-kind feature, so it agrees with the full
    featurizer's ordering.
    """
    torch.manual_seed(0)
    player = og.OpenLearnedPlayer(seed=0)
    for name, val in (("a:ops", 1.0), ("a:depth", -1.0)):
        player._w[name] = val
    st = og.OpenArena(_split()[1][0]).observe()
    cands = list(st.pool)
    acts = [a for a in og.legal_actions(st) if a.op == "evaluate"]
    full = [og.featurize(st, a, player.fresh_hist()) for a in acts]
    order_full = sorted(
        range(len(full)),
        key=lambda i: (
            -sum(player._w.get(k, 0.0) * v for k, v in full[i].items())
        ),
    )
    order_score = sorted(
        range(len(cands)), key=lambda i: -player.rank_score(cands[i])
    )
    assert order_full == order_score
