"""Bound-site domain provenance for Models the prim records lack.

``verify_vocab_models._domains`` reads each shipped prim's measured
bound-site domain from ``minted.prim_records()`` —
``handler.record.provenance.domain``, the ``{mvar: shape}`` entries
``prim_mint.prim_domain`` sampled off the mint's bound sites.  Five
shipped prims minted before the phase-3 refusal ledger recorded no
such domain: their handler stage refused (``"no candidate verified
within budget=256"`` — the v2 run whose records embed in
``mlopt_core.laws.minted``), so ``provenance.domain`` was never
written and the denotation Models verdicted ``python-shipped`` with
``no recorded bound-site domain``.

The ``--prim-vocab`` rerun (``docs/reports/results_overnight_prim_
v4.json``) minted handlers for the same content-hashed shapes and
recorded the measured domain in each ``handler.record.
provenance.domain`` — the same bound sites the v2 mint measured
(the corpus and pattern are identical; only the handler budget
differed).  ``attr_bindings`` are the representative-site freeze the
same record carries: ``prim_mint.freeze_spec(spec, sites[0], …)``
concretized the spec's attr metavariables (``{"dim": "a97"}`` →
``{"dim": 1}``), the convention every minted handler verifies under.

This table backfills that measured data so the fp64 gate
(``modelproj.verify_model``) can run honestly: the domain is the
mint's own record, the attr bindings the mint's own freeze — nothing
here is guessed.  Per-site attr variation was never persisted (the
bound substs' ``$attr:`` keys are stripped by ``prim_domain``), so
the verdict measures the recorded representative binding, the same
bar the shipped minted handlers cleared.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DOMAIN_PROVENANCE"]

#: ``op -> {"domain": [{mvar: shape}, …], "attr_bindings": {mvar: value}}``
#: — the v4 mint's measured bound-site domain and representative-site
#: attr freeze for the five handler-refused prims.
DOMAIN_PROVENANCE: dict[str, dict[str, Any]] = {
    "synop_4c5296b3ba": {
        "domain": [
            {
                "v215": [2, 16, 1],
                "v223": [2, 16, 4],
                "v268": [2, 16, 4],
                "v345": [2, 16, 4],
                "v346": [2, 16, 4],
                "v355": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v223": [2, 16, 4],
                "v268": [2, 16, 4],
                "v345": [2, 16, 4],
                "v346": [2, 16, 4],
                "v355": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v223": [2, 16, 4],
                "v268": [2, 16, 4],
                "v345": [2, 16, 4],
                "v346": [2, 16, 4],
                "v355": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v223": [2, 16, 4],
                "v268": [2, 16, 4],
                "v345": [2, 16, 4],
                "v346": [2, 16, 4],
                "v355": [2, 4],
            },
        ],
        "attr_bindings": {"a97": 1},
    },
    "synop_4c657cb2c0": {
        "domain": [
            {
                "v13": [2, 16, 4],
                "v419": [2, 16],
                "v633": [2, 8, 16],
                "v634": [4, 16],
                "v635": [4],
            },
            {
                "v13": [2, 16, 4],
                "v419": [2, 16],
                "v633": [2, 8, 16],
                "v634": [4, 16],
                "v635": [4],
            },
            {
                "v13": [2, 16, 4],
                "v419": [2, 16],
                "v633": [2, 8, 16],
                "v634": [4, 16],
                "v635": [4],
            },
            {
                "v13": [2, 16, 4],
                "v419": [2, 16],
                "v633": [2, 8, 16],
                "v634": [4, 16],
                "v635": [4],
            },
        ],
        "attr_bindings": {"a155": 7, "a75": 1},
    },
    "synop_976f9836d7": {
        "domain": [
            {
                "v215": [2, 16, 1],
                "v223": [2, 16, 4],
                "v267": [2, 16, 4],
                "v347": [2, 16, 1],
                "v348": [2, 1, 4],
                "v370": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v223": [2, 16, 4],
                "v267": [2, 16, 4],
                "v347": [2, 16, 1],
                "v348": [2, 1, 4],
                "v370": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v223": [2, 16, 4],
                "v267": [2, 16, 4],
                "v347": [2, 16, 1],
                "v348": [2, 1, 4],
                "v370": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v223": [2, 16, 4],
                "v267": [2, 16, 4],
                "v347": [2, 16, 1],
                "v348": [2, 1, 4],
                "v370": [2, 4],
            },
        ],
        "attr_bindings": {"a101": 1},
    },
    "synop_98a002d64b": {
        "domain": [
            {
                "v14": [2, 16, 4],
                "v201": [2, 16, 4],
                "v472": [2, 16, 4],
                "v473": [2, 16, 4],
                "v480": [2, 16, 1],
                "v500": [2, 4],
            },
            {
                "v14": [2, 16, 4],
                "v201": [2, 16, 4],
                "v472": [2, 16, 4],
                "v473": [2, 16, 4],
                "v480": [2, 16, 1],
                "v500": [2, 4],
            },
            {
                "v14": [2, 16, 4],
                "v201": [2, 16, 4],
                "v472": [2, 16, 4],
                "v473": [2, 16, 4],
                "v480": [2, 16, 1],
                "v500": [2, 4],
            },
            {
                "v14": [2, 16, 4],
                "v201": [2, 16, 4],
                "v472": [2, 16, 4],
                "v473": [2, 16, 4],
                "v480": [2, 16, 1],
                "v500": [2, 4],
            },
        ],
        "attr_bindings": {"a128": 1},
    },
    "synop_f08a53f9a0": {
        "domain": [
            {
                "v215": [2, 16, 1],
                "v216": [2, 1, 4],
                "v223": [2, 16, 4],
                "v267": [2, 16, 4],
                "v347": [2, 16, 1],
                "v381": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v216": [2, 1, 4],
                "v223": [2, 16, 4],
                "v267": [2, 16, 4],
                "v347": [2, 16, 1],
                "v381": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v216": [2, 1, 4],
                "v223": [2, 16, 4],
                "v267": [2, 16, 4],
                "v347": [2, 16, 1],
                "v381": [2, 4],
            },
            {
                "v215": [2, 16, 1],
                "v216": [2, 1, 4],
                "v223": [2, 16, 4],
                "v267": [2, 16, 4],
                "v347": [2, 16, 1],
                "v381": [2, 4],
            },
        ],
        "attr_bindings": {"a103": 1},
    },
}
