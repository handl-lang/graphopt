"""The baked open-game prior — shipped, and measured on held-out pages.

``lawdata.OPEN_GAME_PLAYER_WEIGHTS`` is the trained preset
:class:`mlopt_discovery.open_game.OpenLearnedPlayer` now defaults to.
This module measures that the bake is *earned*: on the held-out odd
pages (the even pages trained — see ``gen_cases(start=, step=)``) the
baked prior beats both the cold start (``weights={}``, the honest
uniform baseline) and the random arm on the game's reward, and its
page *ordering* (``rank_score``) beats the cold-start ordering, the
random baseline, and — the strong positive — the cost-delta
heuristic.

The run is deterministic (the stream is fixed, the sampler is a seeded
``random.Random``) and bounded: 36 held-out pages x 4 arms, plus one
average-precision pass over the same pool.  Regenerate the preset with
``python scripts/train_open_prior.py``.

Honest negatives kept visible:

* the baked arm *drains* the page (~15 evals/episode vs the cold
  arm's ~9), so the cold arm's useful-per-eval ratio stays
  competitive even though it finds far fewer useful laws and scores
  far less reward — the game prices the ordering (the credit), not
  the raw ratio;
* the baked prior beats the heuristic on this pool, but only by
  training well past the single cold epoch the other module's
  honest-negative test pins.
"""

from __future__ import annotations

import pytest
import torch
from mlopt_discovery import engine, lawdata, play
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp

#: Page size — the episode horizon (budget = page + 1).
PAGE = 16
#: Held-out pages (the odd half of the even/odd split).
HALF = 36
BUDGET = PAGE + 1
SEED = 0


def _held_out() -> list[dict]:
    """The odd pages — disjoint from the even pages the prior trained on."""
    return og.gen_cases(SEED, HALF, page=PAGE, start=1, step=2)


def _play_arm(mk, ev: list[dict]) -> dict:
    """Play one arm over every held-out page; tally yield and reward."""
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


def _pool(ev: list[dict]) -> tuple[list, dict]:
    """Concatenate the held-out pages' candidates and their outcomes."""
    pool: list = []
    outcomes: dict = {}
    for case in ev:
        st = og.OpenArena(case).observe()
        cands = list(st.pool)
        outs = lp.evaluate(cands)
        for c, o in zip(cands, outs, strict=True):
            outcomes[len(pool)] = o
            pool.append(c)
    return pool, outcomes


@pytest.fixture(scope="module")
def measured() -> dict:
    """Play the arms on the held-out pages and rank the pool."""
    # ``proposal.evaluate`` draws its numeric-oracle tensors from
    # torch's global RNG, so the verdicts (and the arms' yields) are
    # only reproducible once the seed is fixed — same discipline as
    # ``test_discovery_open_learned``.
    torch.manual_seed(SEED)
    ev = _held_out()
    baked = og.OpenLearnedPlayer(seed=SEED)  # default = shipped preset
    cold = og.OpenLearnedPlayer(seed=SEED, weights={})
    d = play.DOMAINS["open"]
    table = {
        "random": _play_arm(play._arms(d, baked, SEED)["random"], ev),
        "cold": _play_arm(lambda i: cold.frozen(SEED + 10_000 + i), ev),
        "baked": _play_arm(
            lambda i: baked.frozen(SEED + 20_000 + i), ev
        ),
        "baked-greedy": _play_arm(
            lambda i: baked.frozen(SEED + 30_000 + i, greedy=True), ev
        ),
    }
    pool, outcomes = _pool(ev)
    ranking = {
        "baked": lp.rank_pool(pool, outcomes, scorer=baked.rank_score),
        "cold": lp.rank_pool(pool, outcomes, scorer=cold.rank_score),
    }
    return {"table": table, "ranking": ranking}


class TestPreset:
    """The shipped preset is bound and the cold start stays reachable."""

    def test_preset_is_baked(self):
        """``OpenLearnedPlayer`` defaults to the trained preset."""
        assert lawdata.OPEN_GAME_PLAYER_WEIGHTS
        player = og.OpenLearnedPlayer(seed=0)
        live = {
            k: v for k, v in player.weights_dict().items() if v != 0.0
        }
        assert live == lawdata.OPEN_GAME_PLAYER_WEIGHTS

    def test_cold_start_reachable(self):
        """``weights={}`` is the uniform cold start, still reachable."""
        cold = og.OpenLearnedPlayer(seed=0, weights={})
        assert set(cold.weights_dict().values()) == {0.0}

    def test_baked_differs_from_cold(self):
        """The default arm is genuinely warm, not the cold start."""
        baked = og.OpenLearnedPlayer(seed=0).weights_dict()
        cold = og.OpenLearnedPlayer(seed=0, weights={}).weights_dict()
        assert baked != cold


@pytest.mark.slow
class TestHeldOutYield:
    """The baked prior's episode yield on the held-out odd pages."""

    def test_baked_beats_cold_on_reward(self, measured):
        """The bake clears the cold start it is trained from."""
        t = measured["table"]
        assert t["baked"]["reward"] > t["cold"]["reward"]

    def test_baked_beats_random_on_reward(self, measured):
        """The baked prior clears random by a real margin."""
        t = measured["table"]
        assert t["baked"]["reward"] > t["random"]["reward"]
        assert t["baked"]["reward"] - t["random"]["reward"] > 0.05

    def test_baked_beats_random_on_useful_per_episode(self, measured):
        """The baked arm finds more useful laws per page than random."""
        t = measured["table"]
        assert (
            t["baked"]["useful_per_ep"] > t["random"]["useful_per_ep"]
        )

    def test_baked_beats_cold_on_useful_per_episode(self, measured):
        """...and more than the cold start, too."""
        t = measured["table"]
        assert t["baked"]["useful_per_ep"] > t["cold"]["useful_per_ep"]

    def test_baked_beats_random_on_useful_per_eval(self, measured):
        """The baked arm's per-eval yield clears random."""
        t = measured["table"]
        assert (
            t["baked"]["useful_per_eval"]
            > t["random"]["useful_per_eval"]
        )

    def test_baked_greedy_beats_random_on_reward(self, measured):
        """The greedy ranking of the baked prior also out-earns random."""
        t = measured["table"]
        assert t["baked-greedy"]["reward"] > t["random"]["reward"]

    def test_cold_use_per_eval_stays_comparable(self, measured):
        """Honest trade-off: the cold arm's useful/eval is not crushed.

        The cold policy stops early — it evaluates ~9 of the 16
        candidates per page against the baked arm's ~15 — so its
        useful-per-eval ratio stays within half of the baked arm's
        even though it finds far fewer useful laws and scores far
        less reward.  The game prices the *ordering* (the credit), so
        the baked arm's early finds win on reward while the cold arm
        keeps a competitive ratio by stopping.  Reported, not hidden.
        """
        t = measured["table"]
        assert t["cold"]["evals_per_ep"] < t["baked"]["evals_per_ep"]
        assert (
            t["cold"]["useful_per_eval"]
            >= t["baked"]["useful_per_eval"] * 0.5
        )


@pytest.mark.slow
class TestRanking:
    """Average precision of the baked prior's page ordering."""

    def test_pool_has_useful_candidates(self, measured):
        """The held-out pool carries positive examples — AP is defined."""
        r = measured["ranking"]["baked"]
        assert r["useful"] > 0

    def test_baked_ranking_beats_random(self, measured):
        """The baked ordering clears the random baseline."""
        r = measured["ranking"]["baked"]
        assert r["learned_ap"] > r["random_ap"]

    def test_baked_ranking_beats_cold(self, measured):
        """The baked ordering is sharper than the cold-start one."""
        rb = measured["ranking"]["baked"]
        rc = measured["ranking"]["cold"]
        assert rb["learned_ap"] > rc["learned_ap"]

    def test_baked_ranking_beats_heuristic(self, measured):
        """The strong positive: the baked prior clears the heuristic.

        The cold-start-trained prior does not (the other module's
        honest-negative test); training the same arm further on the
        even pages produces a shape ranking whose average precision
        on the held-out pool is ~2-3x the cost-delta heuristic's,
        stable across train seeds.
        """
        r = measured["ranking"]["baked"]
        assert r["learned_ap"] > r["heuristic_ap"]
