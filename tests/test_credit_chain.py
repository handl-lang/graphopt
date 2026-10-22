"""The credit-chain probe's contract tests.

``bench/drivers/policy/credit_chain.py`` is the Q3 arm — the
synthetic board where the win is a two-step enabling chain (mint
the needed premise at negative immediate reward, then apply it for
the terminal payout).  These tests pin:

* **board contract** — :class:`ChainArena` satisfies
  :class:`engine.Board`; ``observe`` carries ``done``/``steps``;
  the legal enumerator is deterministic and always offers ``stop``;
* **the chain property** — ``apply(needed)`` is *not* legal until
  the premise is minted, no mint-free play sequence can win, and
  the scripted two-step line reaches ``won`` through a terminal
  report; ``pre_minted`` cases open with the premise banked so the
  win is a one-step apply (the delta arm);
* **the payoff table** — mint is immediate-negative, the ``base``
  law pays the chain-free fallback, dead applies pay nothing, and
  budget exhaustion terminates;
* **the arms run** — scripted banks the optimum on both conditions,
  the myopic greedy arm wins only pre-minted (the delta the probe
  exists to expose), random/cold-learned episodes complete, and a
  few REINFORCE episodes move ``LinearPolicy``/``AdvantagePlayer``
  weights and the net forward pass trains;
* **the driver's protocol** — ``measure`` returns the paired
  with/without-premise table at toy scale, and ``main`` emits the
  rendered table and JSON.
"""

from __future__ import annotations

import json

import pytest
from gamekit import engine
from gamekit.players import LearnedPlayer
from mlopt_discovery.meta_arena import Action

from bench.drivers.policy import credit_chain as cc


def _case(needed="c3", k=8, **kw):
    """A single-case row — the driver's ``gen_cases`` payload shape."""
    case = {
        "name": "t",
        "needed": needed,
        "k": k,
        "pre_minted": False,
        "budget": 6,
    }
    case.update(kw)
    return case


# ---------------------------------------------------------------------------
#  Board contract
# ---------------------------------------------------------------------------


def test_board_satisfies_the_engine_contract():
    board = cc.ChainArena(_case())
    assert isinstance(board, engine.Board)
    state = board.observe()
    assert not state.done
    assert state.steps == 0
    acts = cc.legal_actions(state)
    assert acts == cc.legal_actions(state)  # deterministic
    assert acts[-1].op == "stop"  # the terminator is always offered


def test_illegal_moves_are_honest_noops():
    board = cc.ChainArena(_case())
    board.observe()
    _s, rep = board.step(Action("mint", {"cand": "nope"}))
    assert not rep.applied
    assert rep.reward == 0.0
    _s, rep = board.step(Action("apply", {"law": "c0"}))  # unbanked
    assert not rep.applied
    _s, rep = board.step(Action("leap", {}))
    assert not rep.applied and rep.note == "illegal"


def test_repeat_mint_and_repeat_apply_are_legal_masked():
    board = cc.ChainArena(_case(k=2))
    _s, rep = board.step(Action("mint", {"cand": "c0"}))
    assert rep.applied and rep.reward == -cc.MINT_COST
    state = board.observe()
    assert "c0" in state.bank
    # a repeat mint is not re-offered, and plays as a no-op if forced
    assert Action("mint", {"cand": "c0"}) not in cc.legal_actions(state)
    _s, rep = board.step(Action("mint", {"cand": "c0"}))
    assert not rep.applied
    _s, rep = board.step(Action("apply", {"law": "c0"}))
    assert rep.applied and rep.reward == 0.0  # dead apply
    state = board.observe()
    assert Action("apply", {"law": "c0"}) not in cc.legal_actions(state)
    _s, rep = board.step(Action("apply", {"law": "c0"}))
    assert not rep.applied


def test_budget_exhaustion_terminates():
    board = cc.ChainArena(_case(k=8, budget=2))
    board.step(Action("mint", {"cand": "c0"}))
    _s, rep = board.step(Action("mint", {"cand": "c1"}))
    assert rep.terminal
    assert board.observe().done


# ---------------------------------------------------------------------------
#  The chain property — the win needs mint → apply
# ---------------------------------------------------------------------------


def test_win_requires_the_mint_first():
    board = cc.ChainArena(_case())
    state = board.observe()
    # the winning apply is not on the table until the premise is banked
    assert Action(
        "apply", {"law": state.needed}
    ) not in cc.legal_actions(state)
    # drain every non-mint move the board offers: still no win
    for law in sorted(state.bank):
        _s, rep = board.step(Action("apply", {"law": law}))
        assert not rep.detail["won"]
    board.step(Action("stop", {}))
    assert not board.observe().won


def test_the_scripted_line_wins_through_a_terminal_report():
    board = cc.ChainArena(_case())
    traj = engine.run_episode(board, cc.scripted_player, 6)
    assert traj.terminal is not None
    assert traj.terminal.detail["won"]
    # base (+0.2) → mint (-0.05) → apply needed (+1.0) = 1.15
    assert traj.total == pytest.approx(
        cc.BASE_PAY - cc.MINT_COST + cc.WIN_PAY
    )
    kinds = [r.action.op for r in traj.reports]
    assert kinds == ["apply", "mint", "apply"]


def test_pre_minted_boards_open_armed():
    board = cc.ChainArena(_case(pre_minted=True))
    state = board.observe()
    assert state.needed in state.bank
    _s, rep = board.step(Action("apply", {"law": state.needed}))
    assert rep.detail["won"] and rep.terminal
    assert rep.reward == cc.WIN_PAY


def test_gen_cases_pairs_premise_conditions():
    no = cc.gen_cases(11, 12, k=8)
    pre = cc.gen_cases(11, 12, k=8, pre_minted=True)
    assert [c["needed"] for c in no] == [c["needed"] for c in pre]
    assert all(not c["pre_minted"] for c in no)
    assert all(c["pre_minted"] for c in pre)


# ---------------------------------------------------------------------------
#  Arms — the delta the probe exists to expose
# ---------------------------------------------------------------------------


def test_scripted_wins_both_conditions():
    for pre in (False, True):
        board = cc.ChainArena(_case(pre_minted=pre))
        traj = engine.run_episode(board, cc.scripted_player, 6)
        assert traj.terminal is not None and traj.terminal.detail["won"]


def test_greedy_is_the_delta_arm():
    # pre-minted: the win is one greedy-visible apply
    board = cc.ChainArena(_case(pre_minted=True))
    traj = engine.run_episode(board, cc.greedy_player, 6)
    assert traj.terminal is not None and traj.terminal.detail["won"]
    assert traj.total == pytest.approx(cc.WIN_PAY)
    # unpreminted: minting is immediate-negative, so the chain is
    # invisible — the arm cashes the base law and stops
    board = cc.ChainArena(_case())
    traj = engine.run_episode(board, cc.greedy_player, 6)
    assert traj.terminal is None or not any(
        r.detail.get("won") for r in traj.reports
    )
    assert traj.total == pytest.approx(cc.BASE_PAY)


def test_flood_buys_nothing_under_budget():
    board = cc.ChainArena(_case(k=8, budget=6))
    traj = engine.run_episode(board, cc._flood_player(), 6)
    assert not any(r.detail.get("won") for r in traj.reports)


def test_random_and_cold_learned_episodes_complete():
    cases = cc.gen_cases(3, 4, k=4)
    for case in cases:
        traj = engine.run_episode(
            cc.ChainArena(case), cc._random_player(0), 6
        )
        assert len(traj.reports) <= 6
        cold = LearnedPlayer(
            0, spec=cc.CHAIN_SPEC, weights={}, learn=False
        )
        traj = engine.run_episode(cc.ChainArena(case), cold, 6)
        assert len(traj.reports) <= 6


def test_linear_policy_learns_a_step_or_two():
    """A few REINFORCE episodes move weights — the learner hooks up."""
    cases = cc.gen_cases(5, 24, k=4, budget=4)
    player = LearnedPlayer(
        0, spec=cc.CHAIN_SPEC, weights={}, epsilon=0.3
    )
    engine.train_policy(player, cases, cc.ChainArena, budget=4)
    assert any(abs(w) > 1e-9 for w in player.weights_dict().values())


def test_advantage_player_trains():
    cases = cc.gen_cases(7, 24, k=4, budget=4)
    player = cc.AdvantagePlayer(
        0,
        spec=cc.CHAIN_SPEC,
        weights={},
        epsilon=0.3,
        state_feats=lambda s, _h: cc._state_feats(s),
    )
    engine.train_policy(player, cases, cc.ChainArena, budget=4)
    assert any(abs(w) > 1e-9 for w in player.weights_dict().values())


def test_neural_arm_trains_and_plays():
    cases = cc.gen_cases(9, 8, k=4, budget=4)
    model, means = cc.train_net(cases, 4, episodes=24, hidden=8, seed=0)
    assert means  # per-100 means fall back to the partial block
    player = cc._net_player(model, 0, greedy=True)
    traj = engine.run_episode(cc.ChainArena(cases[0]), player, 4)
    assert len(traj.reports) <= 4


# ---------------------------------------------------------------------------
#  The driver — paired measurement at toy scale
# ---------------------------------------------------------------------------


def test_measure_returns_the_paired_table():
    doc = cc.measure(
        seed=0,
        train_cases=12,
        eval_cases=6,
        k=4,
        budget=4,
        net_episodes=24,
        hidden=8,
    )
    for row in doc["table"].values():
        assert set(row) == {"nopre", "pre", "delta", "win_delta"}
        assert row["nopre"]["cases"] == 6
    assert doc["table"]["scripted"]["nopre"]["win"] == 1.0
    assert doc["table"]["scripted"]["pre"]["win"] == 1.0
    assert doc["table"]["greedy"]["nopre"]["win"] == 0.0
    assert doc["table"]["greedy"]["pre"]["win"] == 1.0


def test_main_prints_and_writes_json(tmp_path, capsys):
    out = tmp_path / "chain.json"
    rc = cc.main(
        [
            "--train-cases",
            "8",
            "--eval-cases",
            "4",
            "--k",
            "4",
            "--budget",
            "4",
            "--net-episodes",
            "16",
            "--hidden",
            "8",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    doc = json.loads(out.read_text())
    assert "scripted" in doc["table"]
    printed = capsys.readouterr().out
    assert "credit-chain" in printed and "scripted" in printed
