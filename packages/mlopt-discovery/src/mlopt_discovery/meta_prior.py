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

Bake (T-009, 2026-11-07 measurement — re-baked after the
``lawdata.MACROS`` ``claim_swiglu_extract`` extension):

    .venv/bin/python -m mlopt_discovery.meta_player \
        --seed 0 --train-cases 400 --eval-cases 40 --budget 24 \
        --temperature 1.2 --epsilon 0.25 \
        --imitate-epochs 3 --epochs 5 --json run.json

— 3 oracle-imitation passes (1200/1200 demonstrated moves cloned)
then 5 REINFORCE passes over ``gen_cases(0, 400)``.  Held-out on
``gen_cases(7919, 40)`` at the same budget: **+0.20 mean reward
(learned-greedy; +0.18 sampled), declared-use 20/40** — vs the
previous bake's +0.19/+0.17 on the same stream, the pre-extension
recorded bake at +0.17, and scripted -0.10.  Replicated at a
second held-out offset (``gen_cases(31337, 40)``: +0.27/+0.23 vs
+0.25/+0.22 pre-extension).  The policy keeps both routes legible:
``op:claim`` +2.1 *and* ``op:macro`` +2.2 (was -1.2) — the oracle's
paying lines are now the offered macro compressions on every
claimable board, so the warm-start demonstrates ``macro`` where it
is strictly cheaper and literal ``claim`` where it is not;
``op:extract`` +6.5, ``x:extract:improve`` +5.9, ``op:fire`` -7.1.

Weights are rounded to 4 decimals and near-zero entries dropped —
``LinearPolicy`` reads a missing feature as ``0``.
"""

from __future__ import annotations

#: The trained meta-arena policy — feature → logit weight.
META_ARENA_PLAYER_WEIGHTS: dict[str, float] = {
    "a:budget": -0.4982,
    "a:h:0": 4.0523,
    "a:h:3": 2.9992,
    "a:o:0": -0.0198,
    "a:o:2": -0.0137,
    "a:r:00": -0.6721,
    "a:r:01": -0.6215,
    "a:r:02": -0.0424,
    "a:r:03": -0.4762,
    "a:r:04": -0.6268,
    "a:r:05": -0.7369,
    "a:r:06": -0.3384,
    "a:r:07": -0.2047,
    "a:r:08": -0.5744,
    "a:r:09": -0.0742,
    "a:r:10": -0.6171,
    "a:r:11": -0.1631,
    "a:r:12": -0.5017,
    "a:r:13": -0.3289,
    "a:r:14": -0.2801,
    "a:r:15": -0.8325,
    "a:spec_i": -0.0294,
    "a:unbounded": -0.8957,
    "op:claim": 2.1395,
    "op:declare": -1.776,
    "op:extract": 6.467,
    "op:fire": -7.0912,
    "op:handle": -0.0336,
    "op:macro": 2.1868,
    "op:saturate": -1.8921,
    "x:claimable:claim": 3.1339,
    "x:claimable:declare": -0.1292,
    "x:declared:handle": -0.0265,
    "x:extract:improve": 5.9117,
    "x:handleable:claim": 3.1339,
    "x:handleable:declare": -0.1292,
    "x:late:extract": 0.5262,
    "x:macro:certified": 2.1929,
    "x:macro:len": -3.7153,
    "x:macro:score": 2.9974,
    "x:pressure:saturate": -0.0003,
    "x:specs:declare": -0.1138,
    "x:unhandled:handle": -0.0086,
}
