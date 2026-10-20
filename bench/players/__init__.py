"""The player comparison — one protocol, every registered player.

Plan 0031 workstream 2's referee: same-protocol episodes across
``--players`` x ``--boards`` drawn from
:mod:`mlopt_discovery.players.registry`, with the memoized-verdict
protocol of ``bench/policy/policy_baseline.py`` (paired oracle draws, held-out
pages) and ``bench/benchkit.env`` provenance.  ``run.py`` is the CLI;
``runner.py`` the per-board drivers; ``leaderboard.py`` the
``docs/leaderboard_players.md`` + JSON emission.
"""
