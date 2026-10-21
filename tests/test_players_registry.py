"""The players package — the registry and the module facade.

Plan 0031 workstream 2 turned ``players.py`` into the ``players/``
package: the base machinery moved to ``players/_base.py`` (the
``gamekit.players`` surface is re-exported unchanged — the
import-compat pins below are the point of the reorganization, not
the internals), and every player is now a folder with a
:class:`~gamekit.players.registry.PlayerSpec`, a measured
record, and a ``make`` arm factory the play CLI's ``--player`` flag
and ``bench/players`` consume.
"""

from __future__ import annotations

from itertools import islice
from types import SimpleNamespace

import pytest

from mlopt_core.cost.basic import count_cost
from gamekit import engine
from mlopt_discovery import meta_player as mp
from mlopt_discovery import open_game as og
from mlopt_discovery import play
from mlopt_discovery.enumerate import enumerate_candidates
from gamekit.players import (
    _action_key,
    _BASELINE_EMA,
    _dot,
    _grad_row,
    DomainSpec,
    LearnedPlayer,
    LinearPolicy,
    PlayHist,
    PlayerSpec,
    registry,
)

# ---------------------------------------------------------------------------
#  The re-export — the pre-package import surface is unchanged
# ---------------------------------------------------------------------------


def test_base_surface_reexports():
    """``from gamekit.players import …`` — old paths live."""
    assert _dot({"a": 2.0}, {"a": 3.0}) == 6.0
    assert _action_key("plain") == ("call", "'plain'")
    phi, ephi = _grad_row([{"x": 1.0}, {"y": 2.0}], [1.0, 1.0], 0)
    assert phi == {"x": 1.0} and ephi == {"x": 0.5, "y": 1.0}
    assert _BASELINE_EMA == 0.2
    assert issubclass(LearnedPlayer, LinearPolicy)
    assert isinstance(DomainSpec, type)
    assert isinstance(PlayHist(), PlayHist)


def test_torch_free_package_import():
    """``import gamekit.players`` stays stdlib-only."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gamekit.players; "
            "assert 'torch' not in sys.modules, 'torch leaked'; "
            "print('players-torch-free')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "players-torch-free" in proc.stdout, (
        f"torch-free players import failed:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


# ---------------------------------------------------------------------------
#  The registry — names, specs, boards, errors
# ---------------------------------------------------------------------------


def test_registry_roster():
    """The shipped roster — every spec carries its capability data."""
    names = registry.names()
    assert {
        "random",
        "playbook",
        "greedy",
        "linear",
        "linear-cold",
        "linear-greedy",
        "neural",
        "distilled",
        "distilled-argmax",
    } <= set(names)
    for name in names:
        s = registry.spec(name)
        assert isinstance(s, PlayerSpec)
        assert s.boards, name
        assert s.kind in {
            "chance",
            "scripted",
            "linear",
            "neural",
            "distilled",
            "gated",
        }
        assert s.record, name  # the measured record is not empty


def test_registry_for_board():
    """Board support filters the roster honestly."""
    open_players = {s.name for s in registry.for_board("open")}
    assert {"random", "playbook", "linear", "neural"} <= open_players
    assert "distilled" not in open_players
    contraction = {s.name for s in registry.for_board("contraction")}
    assert {"distilled", "greedy", "random"} <= contraction
    assert "linear" not in contraction


def test_registry_unknown_name():
    """Unknown names raise with the roster in the message."""
    with pytest.raises(ValueError, match="unknown player"):
        registry.spec("hal")
    with pytest.raises(ValueError, match="random"):
        registry.make("hal", "meta")


def test_registry_board_rejection():
    """``make`` refuses a board the spec does not list."""
    with pytest.raises(ValueError, match="does not play board"):
        registry.make("distilled", "meta")


# ---------------------------------------------------------------------------
#  Arm factories — each module builds a playable arm
# ---------------------------------------------------------------------------


def test_random_plays_a_meta_episode():
    """``random`` on meta — a real episode terminates."""
    mk = registry.make("random", "meta", seed=0, ctx={})
    case = mp.gen_cases(1, 1)[0]
    traj = engine.run_episode(
        mp._board(case, count_cost, 20_000), mk(0), 6
    )
    assert len(traj.reports) >= 1


def test_playbook_meta_and_search():
    """``playbook`` — the scripted line on meta and search."""
    mk = registry.make("playbook", "meta", seed=0, ctx={})
    case = mp.gen_cases(1, 1)[0]
    traj = engine.run_episode(
        mp._board(case, count_cost, 20_000), mk(0), 6
    )
    assert len(traj.reports) == 2  # saturate, extract — the fixed line
    smk = registry.make("playbook", "search", seed=0, ctx={})
    case = play.DOMAINS["search"].cases(0, 1)[0]
    traj = engine.run_episode(
        play.DOMAINS["search"].board_of(case), smk(0), 4
    )
    assert len(traj.reports) >= 1


def test_greedy_and_linear_variants_on_meta():
    """``greedy`` + the three ``linear`` variants build frozen arms."""
    mk = registry.make("greedy", "meta", seed=0, ctx={})
    assert callable(mk(0))
    trained = registry.make("linear", "meta", seed=0, ctx={})(0)
    cold = registry.make("linear-cold", "meta", seed=0, ctx={})(0)
    greedy = registry.make("linear-greedy", "meta", seed=0, ctx={})(0)
    w = trained.weights_dict()
    assert any(v != 0.0 for v in w.values())  # the shipped prior
    assert not any(v != 0.0 for v in cold.weights_dict().values())
    assert isinstance(trained, LearnedPlayer)
    assert greedy is not None


def test_playbook_open_drains_and_deposits_scorer():
    """``playbook`` on open — drain heuristics + the AP scorer."""
    ctx: dict = {}
    mk = registry.make("playbook", "open", seed=0, ctx=ctx)
    assert callable(ctx["score_fn"])
    case = og.gen_cases(0, 1, page=4)[0]
    traj = engine.run_episode(og.OpenArena(case), mk(0), 5)
    assert len(traj.reports) >= 1


def test_linear_open_deposits_rank_scorer():
    """``linear`` on open — the spec's ranker becomes the AP scorer."""
    ctx: dict = {}
    registry.make("linear", "open", seed=0, ctx=ctx)
    assert ctx["score_fn"] is not None
    case = og.gen_cases(0, 1, page=4)[0]
    cand = og.OpenArena(case).pool[0]
    assert isinstance(ctx["score_fn"](cand), float)


def test_neural_requires_and_uses_train_ctx():
    """``neural`` — no honest play without held-in training rows."""
    with pytest.raises(ValueError, match=r"ctx\['train'\]"):
        registry.make("neural", "open", seed=0, ctx={})
    cands = list(islice(enumerate_candidates(), 2))
    outcomes = [
        SimpleNamespace(base_cost=10.0, cand_cost=5.0),
        SimpleNamespace(base_cost=10.0, cand_cost=12.0),
    ]
    ctx = {
        "train": list(zip(cands, outcomes, strict=True)),
        "epochs": 3,
        "hidden": 8,
        "device": "cpu",
    }
    mk = registry.make("neural", "open", seed=0, ctx=ctx)
    assert callable(ctx["score_fn"])
    case = og.gen_cases(0, 1, page=4)[0]
    player = mk(0)
    state = og.OpenArena(case).observe()
    action = player(state)
    assert action.op == "evaluate"  # best predicted value first


def test_distilled_contraction_arms():
    """``distilled`` + ``distilled-argmax`` score real bond networks."""
    from mlopt_torch.contraction_policy import (
        cost_of_order,
        greedy,
        random_bond_network,
    )

    tensors, sizes = random_bond_network(6, 0)
    mk = registry.make(
        "distilled", "contraction", seed=0, ctx={"samples": 4}
    )
    cost = mk(0)(tensors, sizes)
    assert isinstance(cost, float) and cost > 0
    mk2 = registry.make(
        "distilled-argmax", "contraction", seed=0, ctx={}
    )
    cost2 = mk2(0)(tensors, sizes)
    assert isinstance(cost2, float) and cost2 > 0
    assert cost2 >= min(cost, greedy(tensors, sizes)) * 0.5
    # an order the policy returns replays under the shared model
    mk3 = registry.make("random", "contraction", seed=0, ctx={})
    c3 = mk3(0)(tensors, sizes)
    assert isinstance(c3, float) and c3 >= 0
    del cost_of_order


# ---------------------------------------------------------------------------
#  The play CLI — ``--player <name>``
# ---------------------------------------------------------------------------


def test_play_player_flag_eval_only(capsys):
    """``--player playbook`` — eval-only arm table, no training."""
    rc = play.main(
        [
            "--domain",
            "search",
            "--player",
            "playbook",
            "--train-cases",
            "2",
            "--eval-cases",
            "2",
            "--budget",
            "4",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "playbook" in out and "random" in out
    assert "eval-only" in out
    assert "train decade means" not in out


def test_play_player_flag_rejects_unsupported(capsys):
    """``--player`` on a board the spec does not list exits 2."""
    rc = play.main(
        [
            "--domain",
            "meta",
            "--player",
            "distilled",
            "--eval-cases",
            "1",
            "--budget",
            "2",
        ]
    )
    assert rc == 2
    assert "does not play board" in capsys.readouterr().out
