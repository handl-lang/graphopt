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

Bake (T-009, 2026-11-07 measurement):

    .venv/bin/python -m mlopt_discovery.meta_player \
        --seed 0 --train-cases 400 --eval-cases 40 --budget 24 \
        --temperature 1.2 --epsilon 0.25 \
        --imitate-epochs 3 --epochs 5 --json run.json

— 3 oracle-imitation passes (1950/1950 demonstrated moves cloned)
then 5 REINFORCE passes over ``gen_cases(0, 400)``.  Held-out on
``gen_cases(7919, 40)`` at the same budget: **+0.17 mean reward,
22 literal ``claim`` moves, declared-use 20/40** — vs the previous
shipped prior (the macro-move bake, ``op:macro``-route, 0 literal
claims) at +0.13/+0.14 on the same stream, and vs scripted -0.10.
Replicated at a second held-out offset (``gen_cases(31337, 40)``:
+0.22/+0.23 vs shipped +0.17/+0.18) and under the
``test_discovery_meta_prior`` protocol (budget 12, 16 cases:
+0.14 vs +0.11).  The policy is legible: ``op:claim`` +5.9,
``op:extract`` +4.6, ``x:extract:improve`` +5.9, ``op:fire`` -6.3,
``op:macro`` -1.2 — it plays the composed claim→extract line
itself rather than the mined-macro shortcut.

Weights are rounded to 4 decimals and near-zero entries dropped —
``LinearPolicy`` reads a missing feature as ``0``.
"""

from __future__ import annotations

#: The trained meta-arena policy — feature → logit weight.
META_ARENA_PLAYER_WEIGHTS: dict[str, float] = {
    "a:budget": -0.4338,
    "a:h:0": 4.1878,
    "a:h:3": 1.4326,
    "a:o:0": -0.0193,
    "a:o:2": -0.0137,
    "a:r:00": -0.7304,
    "a:r:01": -0.575,
    "a:r:02": -0.0236,
    "a:r:03": -0.3384,
    "a:r:04": -0.5562,
    "a:r:05": -0.622,
    "a:r:06": -0.3055,
    "a:r:07": -0.2115,
    "a:r:08": -0.5548,
    "a:r:09": -0.1127,
    "a:r:10": -0.5385,
    "a:r:11": -0.1386,
    "a:r:12": -0.3621,
    "a:r:13": -0.2369,
    "a:r:14": -0.224,
    "a:r:15": -0.7256,
    "a:spec_i": -0.0254,
    "a:unbounded": -0.7802,
    "op:claim": 5.9149,
    "op:declare": -1.3955,
    "op:extract": 4.6002,
    "op:fire": -6.2559,
    "op:handle": -0.033,
    "op:macro": -1.1825,
    "op:saturate": -1.6478,
    "x:claimable:claim": 0.1946,
    "x:claimable:declare": -0.1278,
    "x:declared:handle": -0.0263,
    "x:extract:improve": 5.9056,
    "x:handleable:claim": 0.1946,
    "x:handleable:declare": -0.1278,
    "x:late:extract": 0.5101,
    "x:macro:certified": -1.1755,
    "x:macro:len": -0.3954,
    "x:macro:score": -0.5132,
    "x:pressure:saturate": -0.0002,
    "x:specs:declare": -0.094,
    "x:unhandled:handle": -0.0085,
}
