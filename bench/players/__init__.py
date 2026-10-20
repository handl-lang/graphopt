"""The player comparison — one protocol, every registered player.

Plan 0031 workstream 2's referee: same-protocol episodes across
``--players`` x ``--boards`` drawn from
:mod:`gamekit.players.registry`, with the memoized-verdict
protocol of ``bench/drivers/policy/policy_baseline.py`` (paired oracle draws, held-out
pages) and ``bench/benchkit.env`` provenance.  ``run.py`` is the CLI;
``runner.py`` the per-board drivers; ``leaderboard.py`` the
``docs/leaderboard_players.md`` + JSON emission.
"""
