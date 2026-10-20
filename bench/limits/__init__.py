"""``bench/limits`` — the baselines registry and the limits table.

Plan 0031 workstream 3: opponents organized like players.  Each
baseline is a folder in this package — a measured record in its
docstring, a :class:`~bench.limits.registry.BaselineSpec`, and a
``build(ctx)`` adapter producing the callable a practitioner would
ship.  The registry maps *name → spec* with lazy folder loading; the
runner crosses the registered baselines with the discovery corpus and
:mod:`bench.limits.table` emits ``docs/limits_table.{md,json}`` — the
"results — limits" table, generated rather than hand-copied.

This is deliberately a sibling of :mod:`bench.discovery` and
:mod:`bench.players`, *not* a reuse of the ``bench/baselines/``
directory — that directory already holds the pinned golden reports
``python -m bench compare``/``gate`` reads (a different kind of
baseline).
"""
