"""The baked meta-arena player prior — graphopt's own trained table.

:data:`META_ARENA_PLAYER_WEIGHTS` is the weight snapshot
:class:`mlopt_discovery.meta_player.MetaLearnedPlayer` defaults to
(``DomainSpec.build(weights=...)``).  It lives here — a
graphopt-owned module beside ``meta_player`` — rather than in
``lawdata``/``gamekit.gamedata`` because the *recipe* that produced
it is domain machinery: ``meta_player.imitate_oracle`` /
``plan_oracles`` / ``curriculum_order`` replay the simulated oracle
line per board, a fit the shared game layer cannot run (it knows
boards, not meta-arena claim semantics).  ``lawdata.
META_ARENA_PLAYER_WEIGHTS`` still resolves to the gamekit-shipped
macro-move prior — this table supersedes it as the spec's default;
the two are *different bakes of the same schema*, kept under
distinct homes so the lawdata/gamekit split invariant (no table in
both halves) stays clean.

Bake (T-009, 2026-11-08 measurement — re-baked after the
macro move-shape legs + widened handler buckets landed; the
``silu``/``square`` handler collision fix *forces* this re-bake —
the ``a:h:*`` bucket indices remapped, so the superseded table
would misprice every claim/handle row):

    .venv/bin/python -m mlopt_discovery.meta_player \
        --seed 0 --train-cases 400 --eval-cases 40 --budget 24 \
        --temperature 1.2 --epsilon 0.25 \
        --imitate-epochs 3 --epochs 5 --json run.json

— 3 oracle-imitation passes (1200/1200 demonstrated moves cloned)
then 5 REINFORCE passes over ``gen_cases(0, 400)`` on the
10-kind corpus.  Held-out on ``gen_cases(7919, 40)`` at the same
budget: **+0.33 mean reward (learned-greedy; +0.30 sampled),
declared-use 26/40**; replicated at ``gen_cases(31337, 40)``:
**+0.33/+0.31** — vs the superseded bake's +0.32/+0.30 and
+0.33/+0.29 on the same stream.  **Honest read: FLAT** — the new
``x:macro:spelled``/``saturating``/``nclaims``/``reach_*`` and
role-qualified ``x:mc:*``/``x:mh:*`` legs (plus the ``a:h:4..7``
widened buckets that separate ``silu``→7 from ``square``→3) did
not close the ~0.03 dual/tri residual; the bake ships because the
bucket remap invalidated the old table, not because the features
earned.  The learned weights do use the legs (``x:macro:spelled``
-0.85, ``reach_sites`` +2.23, ``x:mc:7`` +1.00, ``x:mh:7`` +0.79,
``a:h:7`` +1.80) — the policy can now *name* the spelled-vs-claim
distinction, it just doesn't convert it to held-out reward on
this corpus.  See
``project/reports/2026-11-08-meta-arena-macro-legs.md``.

Weights are rounded to 4 decimals and near-zero entries dropped —
``LinearPolicy`` reads a missing feature as ``0``.
"""

from __future__ import annotations

#: The trained meta-arena policy — feature → logit weight.
META_ARENA_PLAYER_WEIGHTS: dict[str, float] = {
    "op:fire": -7.9034,
    "op:saturate": -1.8773,
    "op:declare": -1.7958,
    "op:handle": -0.0337,
    "op:claim": -0.1504,
    "op:extract": 8.7441,
    "op:macro": 3.017,
    "a:budget": -0.5015,
    "a:unbounded": -0.8745,
    "a:spec_i": -0.0295,
    "a:r:00": -0.7135,
    "a:r:01": -0.7071,
    "a:r:02": -0.0395,
    "a:r:03": -0.6255,
    "a:r:04": -0.5356,
    "a:r:05": -0.8548,
    "a:r:06": -0.4262,
    "a:r:07": -0.169,
    "a:r:08": -0.6748,
    "a:r:09": 0.0279,
    "a:r:10": -0.6639,
    "a:r:11": -0.1727,
    "a:r:12": -0.5881,
    "a:r:13": -0.4454,
    "a:r:14": -0.3541,
    "a:r:15": -0.9611,
    "a:h:0": 3.3391,
    "a:h:3": 2.8592,
    "a:h:7": 1.8018,
    "a:o:0": -0.0199,
    "a:o:2": -0.0137,
    "x:extract:improve": 5.8383,
    "x:late:extract": 0.5405,
    "x:pressure:saturate": -0.0003,
    "x:declared:handle": -0.0265,
    "x:unhandled:handle": -0.0086,
    "x:specs:declare": -0.114,
    "x:handleable:declare": -0.1195,
    "x:handleable:claim": -0.069,
    "x:claimable:declare": -0.1207,
    "x:claimable:claim": -3.6652,
    "x:macro:len": 2.0942,
    "x:macro:certified": 3.023,
    "x:macro:score": 2.8136,
    "x:macro:spelled": -0.8462,
    "x:macro:saturating": -0.845,
    "x:macro:nclaims": 0.2357,
    "x:macro:reach_sites": 2.2291,
    "x:macro:reach_claimable": 2.5509,
    "x:mc:0": -0.0603,
    "x:mc:7": 1.0031,
    "x:mh:0": 0.0001,
    "x:mh:7": 0.7911,
}
