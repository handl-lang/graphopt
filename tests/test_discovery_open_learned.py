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

The reward was recalibrated so the *ordering* pays.  The old table
priced an eval at ``-0.02`` — the page's ~0.02 expected tier — so the
marginal candidate was break-even and the metric rewarded *stopping*,
not ranking.  Now ``step = -0.002`` (the verdict dominates) and each
``useful`` verdict also pays an ordering credit
``OPEN_GAME_EFFICIENCY / evals`` — the paying laws found per oracle
eval so far.  A useful law found at eval 1 pays ten times one found at
eval 10, so the same page in a better order scores strictly higher.

The measured result (seed 0, deterministic under ``torch.manual_seed``):

* ``learned`` beats ``random`` on reward/episode, useful/episode and
  useful/eval;
* ``learned-greedy`` (ranks the whole page, then drains it) beats
  ``random`` on reward/episode — its better ordering reaches the
  paying laws earlier, which the credit pays more; a scripted
  rank-first page beats the same page rank-last, and a rank-and-stop
  arm beats the naive pool-order drain;
* the cheap eval is a real trade-off, reported not hidden: a *ranked*
  drain still outscores a rank-and-stop policy, because with each
  eval worth ~0.018 in expectation the extra true-but-shallow finds
  pay for themselves.  The reward prices the *ordering* — where the
  paying laws are found — not a blanket preference for stopping;
* the learned *ranking* (average precision) beats random but stays
  well below the cost-delta heuristic, which already finds every
  useful candidate — the honest verdict is "learned beats random,
  heuristic still beats learned".

A run is ~40 s (bounded): 36 training pages + 36 eval pages, 3 arms.
"""

from __future__ import annotations

import pytest
import torch
from catopt_discovery import engine, play
from catopt_discovery import open_game as og
from catopt_discovery import proposal as lp
from catopt_discovery.meta_arena import Action

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


def _scripted(order_per_page: dict[int, list[int]]):
    """A player that plays a fixed per-page order, then stops.

    The order is a list of pool indices; ``stop`` is returned once
    every index is evaluated.  Used to pin the reward's ordering
    sensitivity: the *same* page drained in two orders, or drained
    vs. stopped, must score differently.
    """

    def make(i: int):
        order = order_per_page[i]

        def player(state):
            for j in order:
                if j in state.pending:
                    return Action(
                        "evaluate", {"i": j, "cand": state.pool[j]}
                    )
            return Action("stop", {})

        return player

    return make


def _rank_orders(
    per_page: list[tuple[list, list[int]]],
) -> dict[str, dict[int, list[int]]]:
    """Build the scripted orderings over the useful-bearing pages.

    ``first``/``last`` drain the whole page with the paying laws
    ranked first/last (same set, same eval count — only the order
    differs); ``stop`` evaluates the paying laws only; ``drain``
    walks the page in pool order.
    """
    pages = [k for k, (_, useful) in enumerate(per_page) if useful]
    orders: dict[str, dict[int, list[int]]] = {
        "first": {},
        "last": {},
        "stop": {},
        "drain": {},
    }
    for k, page in enumerate(pages):
        cands, useful = per_page[page]
        rest = [j for j in range(len(cands)) if j not in useful]
        orders["first"][k] = useful + rest
        orders["last"][k] = rest + useful
        orders["stop"][k] = list(useful)
        orders["drain"][k] = list(range(len(cands)))
    return orders


@pytest.fixture(scope="module")
def measured() -> dict:
    """Train once, evaluate the arms, and rank the held-out pool."""
    torch.manual_seed(SEED)
    train, ev = _split()
    # The cold start (``{}``), explicitly: this module pins the
    # *training* claim — does training from scratch move the arm
    # past random?  The shipped baked prior
    # (``lawdata.OPEN_GAME_PLAYER_WEIGHTS``) is what the player
    # defaults to, and is measured in
    # ``tests/test_discovery_open_prior.py``.
    player = og.OpenLearnedPlayer(seed=SEED, weights={})
    engine.train_policy(player, train, og.OpenArena, budget=BUDGET)
    arms = play._arms(play.DOMAINS["open"], player, SEED)
    table = {name: _play_arm(arms[name], ev) for name in arms}
    pool: list = []
    outcomes: dict = {}
    per_page: list[tuple[list, list[int]]] = []
    for case in ev:
        st = og.OpenArena(case).observe()
        cands = list(st.pool)
        outs = lp.evaluate(cands)
        per_page.append(
            (cands, [i for i, o in enumerate(outs) if o.useful])
        )
        for c, o in zip(cands, outs, strict=True):
            outcomes[len(pool)] = o
            pool.append(c)
    ranking = lp.rank_pool(pool, outcomes, scorer=player.rank_score)
    orders = _rank_orders(per_page)
    useful_pages = [ev[k] for k, (_, u) in enumerate(per_page) if u]
    ordered = {
        name: _play_arm(_scripted(order), useful_pages)
        for name, order in orders.items()
    }
    return {
        "player": player,
        "eval": ev,
        "table": table,
        "ranking": ranking,
        "ordered": ordered,
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

    def test_learned_greedy_beats_random_on_reward(self, measured):
        """The greedy ranking arm now out-earns random.

        Under the old break-even reward the two tied (both drained
        the page for ≈0); with the cheap eval and the ordering credit
        the greedy arm's ranked drain finds the paying laws earlier
        and scores strictly more.
        """
        t = measured["table"]
        assert t["learned-greedy"]["reward"] > t["random"]["reward"]

    def test_greedy_finds_more_and_scores_more(self, measured):
        """The greedy arm drains the page — and now pays for it.

        It finds at least as many useful laws as random and scores
        more reward: with the eval cheap and the ordering credit
        rewarding early finds, a ranked drain is no longer penalised
        into a loss.
        """
        t = measured["table"]
        assert (
            t["learned-greedy"]["useful_per_ep"]
            >= t["random"]["useful_per_ep"]
        )
        assert t["learned-greedy"]["reward"] > t["random"]["reward"]


class TestOrderingSensitivity:
    """The reward pays for *where* a useful law is found."""

    def test_rank_first_drain_beats_rank_last(self, measured):
        """Same page, same evals, better order — strictly more reward.

        ``first`` and ``last`` evaluate the identical candidate set
        (the whole page) for the identical eval count; only the
        order differs, so the tier sum and the eval cost are equal
        and the gap is exactly the ordering credit.
        """
        o = measured["ordered"]
        assert o["first"]["evals_per_ep"] == o["last"]["evals_per_ep"]
        assert o["first"]["found"] == o["last"]["found"]
        assert o["first"]["reward"] > o["last"]["reward"]

    def test_rank_and_stop_beats_draining_the_page(self, measured):
        """Rank the paying laws, then stop — beats evaluating all.

        ``stop`` evaluates only the paying candidates and ends; the
        naive ``drain`` walks the whole page.  Same useful set, far
        fewer evals, and the credit pays the early finds more.
        """
        o = measured["ordered"]
        assert o["stop"]["found"] == o["drain"]["found"]
        assert o["stop"]["evals_per_ep"] < o["drain"]["evals_per_ep"]
        assert o["stop"]["reward"] > o["drain"]["reward"]

    def test_ranked_drain_still_outscores_rank_and_stop(self, measured):
        """The honest trade-off the cheap eval buys.

        ``first`` ranks the paying laws first and *keeps going*: it
        banks the true-but-shallow tiers the ``stop`` arm skips, and
        with each eval worth ~0.018 in expectation that pays.  The
        reward prices the ordering, not a blanket preference for
        stopping — a policy that stops is not automatically rewarded.
        """
        o = measured["ordered"]
        assert o["first"]["reward"] > o["stop"]["reward"]


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
        """The honest negative: the *cold-start-trained* prior does
        not beat the cost-delta heuristic — one epoch from ``{}``
        leaves the shape ranking no sharper than the heuristic, so
        its edge over random comes from coverage.  (The shipped baked
        prior, trained further, does clear the heuristic — see
        ``tests/test_discovery_open_prior.py``.)"""
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
