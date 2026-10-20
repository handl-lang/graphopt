"""``bench/players`` — the player-comparison runner + leaderboard.

Plan 0031 workstream 2's automatic leaderboard: same-protocol
episodes across ``--players`` × ``--boards`` — paired held-out cases
on the episode boards, memoized per-candidate oracle verdicts on
``open``, pairwise-cost contraction boards with the ``opt_einsum``
baselines — emitting ``leaderboard_players.{md,json}``.  These tests
run the fast lane: tiny case counts, page sizes and rollout budgets
so the whole file is seconds-scale (the open board's oracle
measurement dominates at ~0.25 s/candidate).
"""

from __future__ import annotations

import json

from bench.players.leaderboard import render_markdown, write_outputs
from bench.players.runner import (
    Row,
    run_board,
    run_contraction,
    run_episode_board,
    run_open,
)


def test_episode_board_meta():
    """Meta — paired held-out cases, per-arm rows with wall time."""
    rows = run_episode_board(
        "meta", ["playbook", "random"], seed=0, cases=4, budget=8
    )
    assert [r.player for r in rows] == ["playbook", "random"]
    for r in rows:
        assert r.board == "meta" and r.n == 4
        assert r.reward_ep is not None
        assert r.evals_ep is not None and r.evals_ep > 0
        assert r.wall_s >= 0.0


def test_episode_board_unsupported_player():
    """An explicit player that cannot play the board says so."""
    rows = run_episode_board(
        "meta", ["distilled"], seed=0, cases=2, budget=4
    )
    assert rows[0].notes == "does not play this board"
    assert rows[0].reward_ep is None


def test_open_board_memoized_protocol():
    """Open — memoized verdicts, drain notes, AP column."""
    rows, meta = run_open(
        ["random", "playbook", "linear"],
        seed=0,
        pages=2,
        page=6,
        budget=7,
        neural_epochs=3,
        neural_hidden=8,
        device="cpu",
    )
    assert meta["pool"] == 12 and meta["train_pairs"] == 0
    by_name = {r.player: r for r in rows}
    assert by_name["playbook"].evals_ep == 6.0
    assert "drains every page" in by_name["playbook"].notes
    assert by_name["random"].ap == meta["random_ap"]
    assert by_name["linear"].ap is not None
    assert by_name["playbook"].ap is not None
    for r in rows:
        assert r.reward_ep is not None and r.wall_s >= 0.0


def test_open_board_neural_arm():
    """Open + neural — trains on the held-in pages, plays a drain."""
    rows, meta = run_open(
        ["neural"],
        seed=0,
        pages=2,
        page=6,
        budget=7,
        neural_epochs=3,
        neural_hidden=8,
        device="cpu",
    )
    assert meta["train_pairs"] == 12
    r = rows[0]
    assert r.reward_ep is not None
    assert r.ap is not None  # ctx["score_fn"] deposited


def test_contraction_board():
    """Contraction — pairwise cost vs the oe/our-greedy floor."""
    rows = run_contraction(
        ["greedy", "distilled", "random"],
        seed=0,
        scales=(6,),
        instances=2,
        samples=4,
        device="cpu",
    )
    by_name = {r.player: r for r in rows}
    assert "greedy" in by_name and "distilled" in by_name
    d = by_name["distilled"]
    assert d.cost is not None and d.cost > 0
    assert d.ratio is not None and d.n == 2
    assert "vs oe-greedy" in d.notes or "vs our-greedy" in d.notes
    # the field baseline rows report too when opt_einsum installs
    if "oe-greedy" in by_name:
        assert by_name["oe-greedy"].ratio == 1.0


def test_run_board_dispatch_and_markdown():
    """Dispatch + the Markdown render — tables per board."""
    rows, meta = run_board(
        "meta",
        ["playbook"],
        seed=0,
        meta_cases=2,
        meta_budget=4,
        open_pages=2,
        open_page=6,
        open_budget=7,
        neural_epochs=3,
        neural_hidden=8,
        scales=(6,),
        instances=1,
        samples=2,
        device="cpu",
    )
    assert rows and meta["cases"] == 2
    md = render_markdown(rows, protocol={"seed": 0}, meta=meta)
    assert "# Player leaderboard" in md
    assert "## `meta`" in md and "`playbook`" in md
    assert "notes" in md


def test_write_outputs_and_main(tmp_path):
    """The JSON side-file + the CLI end-to-end on the cheap boards."""
    row = Row(board="meta", player="random", reward_ep=-0.5)
    payload = write_outputs(
        [row],
        md_path=tmp_path / "lb.md",
        json_path=tmp_path / "lb.json",
        protocol={"seed": 0},
        device="cpu",
    )
    data = json.loads((tmp_path / "lb.json").read_text())
    assert data["schema"] == 1 and data["rows"][0]["player"] == "random"
    assert payload["env"]["git_sha"]
    assert "| `random` |" in (tmp_path / "lb.md").read_text()


def test_cli_smoke(tmp_path):
    """``python -m bench.players.run --smoke`` on two cheap boards."""
    from bench.players.run import main

    rc = main(
        [
            "--smoke",
            "--boards",
            "meta,contraction",
            "--players",
            "playbook,distilled",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(
        (tmp_path / "leaderboard_players.json").read_text()
    )
    boards = {r["board"] for r in payload["rows"]}
    assert boards == {"meta", "contraction"}
    names = {r["player"] for r in payload["rows"]}
    assert "playbook" in names and "distilled" in names


def test_cli_unknown_board(tmp_path):
    """An unknown board name exits 2 with the known list."""
    from bench.players.run import main

    rc = main(["--boards", "nope", "--out", str(tmp_path)])
    assert rc == 2
