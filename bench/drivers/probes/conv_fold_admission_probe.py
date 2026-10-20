"""``CONV_FOLD_RULES`` (+ ``BATCH_NORM_RULES`` re-confirm) admission sweep.

The conv channel-scale/shift anchors and the eval-mode BN expansion
family are opt-in pending the admission floor — the same gauntlet the
promoted laws cleared: stored verbatim (``kind="law"``, ``cert=None``)
into a scratch evidence store, then ``evidence.run_gauntlet`` over the
real corpus (``default_gauntlet_corpus()``) per rule.  The admission
rule is the f8e5615 floor: truth-clean AND extraction actually pays
(``typed-pay``: fires > 0, all well-typed, paid > 0).

Expected, per the BN sweep's account (``project/retros/
bn-admission-sweep.md``): truth green, typed-pay refuses — a fused
conv still prices cheaper than the spelled scale/shift subgraph, so
the anchors' real value is *enablement* (they are what lets the
fixpoint mint conv2d-headed composites), not extraction pay.

Run with
``.venv/bin/python bench/drivers/probes/conv_fold_admission_probe.py [bn|conv]``.
"""

from __future__ import annotations

import sys
import time
from typing import Any


def _sweep(rules: list[Any], label: str, corpus: Any, db: str) -> None:
    """Store + gauntlet every rule; print the stage-by-stage account."""
    from mlopt_discovery import evidence as ev

    conn = ev.connect(db)
    try:
        keys = [
            ev.store_object(conn, r, kind="law", cert=None)
            for r in rules
        ]
        n_fail: dict[str, int] = {}
        n_pass = 0
        for r, key in zip(rules, keys, strict=True):
            t0 = time.monotonic()
            rep = ev.run_gauntlet(conn, key, corpus=corpus)
            dt = time.monotonic() - t0
            verdict = (
                "usable" if rep.usable else f"REFUSED @ {rep.reason}"
            )
            if rep.usable:
                n_pass += 1
            else:
                stage = rep.reason.split(":", 1)[0]
                n_fail[stage] = n_fail.get(stage, 0) + 1
            print(f"  {r.name:<44} {dt:5.1f}s  {verdict}")
        print(
            f"  {label}: {n_pass}/{len(rules)} usable; "
            f"refusals by stage: {n_fail or '{}'}"
        )
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    """Run the sweep over the requested families (default: both)."""
    from mlopt_core.laws import BATCH_NORM_RULES, CONV_FOLD_RULES
    from mlopt_discovery import cache_dir
    from mlopt_discovery import evidence as ev

    which = argv[1] if len(argv) > 1 else "all"
    t0 = time.monotonic()
    corpus = ev.default_gauntlet_corpus()
    print(
        f"corpus: {len(corpus.real_terms)} terms / "
        f"{len(corpus.probe)} probes / {len(corpus.base_rules)} "
        f"base rules ({time.monotonic() - t0:.1f}s)"
    )
    db = str(cache_dir() / "conv_fold_admission.db")
    if which in ("conv", "all"):
        print(f"== CONV_FOLD_RULES ({len(CONV_FOLD_RULES)}) ==")
        _sweep(list(CONV_FOLD_RULES), "CONV_FOLD_RULES", corpus, db)
    if which in ("bn", "all"):
        print(f"== BATCH_NORM_RULES ({len(BATCH_NORM_RULES)}) ==")
        _sweep(list(BATCH_NORM_RULES), "BATCH_NORM_RULES", corpus, db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
