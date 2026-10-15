"""The baked meta-arena prior — shipped, and measured on held-out cases.

``lawdata.META_ARENA_PLAYER_WEIGHTS`` is the trained preset
:class:`catopt_discovery.meta_player.MetaLearnedPlayer` now defaults
to.  This module measures that the bake is *earned*: on the held-out
case stream (``gen_cases(7919, n)``, disjoint from the
``gen_cases(0, n)`` training stream) the trained prior beats the cold
start (``weights={}``), the random arm, the cold-greedy arm, and the
hand-written ``scripted`` playbook on the board's reward.

The meta board prices *spend* (a per-step cost and a per-enode price),
so a good line extracts cheaply — the learned table is legible
(``op:extract`` high, ``op:fire`` low).  The run is deterministic
(``torch.manual_seed`` fixes the numeric oracle's tensors) and cheap —
no training here, just one held-out evaluation.  Regenerate the preset
with ``python lab/train/train_meta_prior.py``.
"""

from __future__ import annotations

import random

import pytest
import torch
from catopt_discovery import lawdata
from catopt_discovery import meta_arena as ma
from catopt_discovery import meta_player as mp

#: Held-out case count and horizon (the driver's defaults).
EVAL_CASES = 16
BUDGET = 12
SEED = 0
#: ``play``'s held-out seed offset.
HELD_OUT_OFFSET = 7919


def _random_arm(seed: int):
    """A uniform player over the meta board's legal set."""

    def make(i: int):
        r = random.Random(seed + 10_000 + i)

        def player(state):
            acts = ma.legal_actions(state)
            return acts[r.randrange(len(acts))] if acts else None

        return player

    return make


@pytest.fixture(scope="module")
def measured() -> dict[str, float]:
    """Play every arm on the held-out stream; mean reward per arm."""
    torch.manual_seed(SEED)
    ev = mp.gen_cases(SEED + HELD_OUT_OFFSET, EVAL_CASES)
    trained = mp.MetaLearnedPlayer(SEED)  # default = shipped preset
    cold = mp.MetaLearnedPlayer(SEED, weights={})
    arms = {
        "random": _random_arm(SEED),
        "cold": lambda i: cold.frozen(SEED + 20_000 + i),
        "cold-greedy": lambda i: cold.frozen(
            SEED + 30_000 + i, greedy=True
        ),
        "trained": lambda i: trained.frozen(SEED + 40_000 + i),
        "trained-greedy": lambda i: trained.frozen(
            SEED + 50_000 + i, greedy=True
        ),
        "scripted": lambda i: ma.ScriptedPlayer(),
    }
    table = mp.evaluate(arms, ev, budget=BUDGET)
    return {
        name: sum(r["reward"] for r in rows) / len(rows)
        for name, rows in table.items()
    }


class TestPreset:
    """The shipped preset is bound and the cold start stays reachable."""

    def test_preset_is_baked(self):
        """``MetaLearnedPlayer`` defaults to the trained preset."""
        assert lawdata.META_ARENA_PLAYER_WEIGHTS
        player = mp.MetaLearnedPlayer(0)
        live = {
            k: v for k, v in player.weights_dict().items() if v != 0.0
        }
        assert live == lawdata.META_ARENA_PLAYER_WEIGHTS

    def test_cold_start_reachable(self):
        """``weights={}`` is the uniform cold start, still reachable."""
        cold = mp.MetaLearnedPlayer(0, weights={})
        assert set(cold.weights_dict().values()) == {0.0}


class TestHeldOutReward:
    """The trained prior's reward on the held-out case stream."""

    def test_trained_beats_random(self, measured):
        """The bake clears the random arm."""
        assert measured["trained"] > measured["random"]

    def test_trained_beats_cold(self, measured):
        """The bake clears the cold start it is trained from."""
        assert measured["trained"] > measured["cold"]

    def test_trained_beats_cold_greedy(self, measured):
        """...and the cold arm's greedy ordering, too."""
        assert measured["trained"] > measured["cold-greedy"]

    def test_trained_beats_scripted(self, measured):
        """The learned line out-earns the hand-written playbook."""
        assert measured["trained"] > measured["scripted"]

    def test_trained_greedy_beats_random(self, measured):
        """The greedy ordering of the baked prior also clears random."""
        assert measured["trained-greedy"] > measured["random"]
