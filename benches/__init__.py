"""catopt benchmarks — systematic, presentable measurement of the engine.

Layout:

* ``benches.benchkit`` — the harness: timing, provenance, findings, and
  the renderers (Markdown / HTML dashboard / Quarto / Slidev).
* ``benches.suites`` — the benchmarks themselves, grouped by intent
  (``correctness`` / ``speedup`` / ``e2e`` / …; the directory is the
  intent).
* ``benches.registry`` — the suite catalog the CLI drives.
* ``benches.cli`` — ``python -m benches``.

Run ``python -m benches list`` to see the catalog.
"""

from __future__ import annotations

__all__: list[str] = []
