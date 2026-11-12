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

Bake (coverage corpus, 2026-11-12 measurement — re-baked after
``gen_cases`` grew the ``corpus_cov`` gap-targeting kinds: the
shipped stream is no longer the 10-kind base corpus the prior
three bakes trained on, so the table was re-fit on the widened
distribution):

    .venv/bin/python -m mlopt_discovery.meta_player \
        --seed 23418 --train-cases 400 --eval-cases 40 --budget 24 \
        --temperature 1.2 --epsilon 0.25 \
        --imitate-epochs 3 --epochs 5 --json run.json

— 3 oracle-imitation passes (1416/1416 demonstrated moves cloned)
then 5 REINFORCE passes over ``gen_cases(23418, 400)``; held-out
lands on ``gen_cases(31337, 40)`` because ``main`` evals
``seed + 7919``.  The canonical ``--seed 0`` replicate (eval
``gen_cases(7919, 40)``) was run alongside; its weights are
statistically indistinguishable on the extended stream and
slightly worse on the base-only stream, so the 23418 table —
the better base-stream retentive of the two — ships.

Held-out, extended corpus (``coverage=True``, the shipped
``gen_cases`` default): shipped prior +0.086/+0.101 @7919 and
+0.115/+0.131 @31337 (learned/learned-greedy mean reward);
this bake +0.104/+0.123 @7919 (seed-0 replicate +0.109/+0.123)
and +0.144/+0.156 @31337 — **+0.02..+0.03 earned, replicated on
both offsets and both eval arms**.  On the base-only stream
(``coverage=False``) the same bake reads +0.290/+0.311 @7919
and +0.288/+0.308 @31337 vs the superseded table's
+0.299/+0.329 and +0.293/+0.326 — a -0.01..-0.02 retention
cost, the honest price of specializing to the widened corpus.
The large ``op:saturate``/``a:unbounded`` logits are real
(both eval arms, both offsets) but bear watching — the
coverage boards reward saturate-then-extract lines the base
corpus never priced.  See
``docs/reports/2026-11-12-meta-arena-corpus-coverage-gen.md``.

Weights are rounded to 4 decimals and near-zero entries dropped —
``LinearPolicy`` reads a missing feature as ``0``.
"""

from __future__ import annotations

#: The trained meta-arena policy — feature → logit weight.
META_ARENA_PLAYER_WEIGHTS: dict[str, float] = {
    "op:fire": -8.7279,
    "op:saturate": 3.3608,
    "op:declare": -1.9874,
    "op:handle": -0.0338,
    "op:claim": -0.3042,
    "op:extract": 5.3597,
    "op:macro": 2.3333,
    "a:budget": -0.98,
    "a:unbounded": 5.3205,
    "a:spec_i": -0.0343,
    "a:r:00": -0.7286,
    "a:r:01": -0.7086,
    "a:r:02": -0.0844,
    "a:r:03": -0.7058,
    "a:r:04": -0.5643,
    "a:r:05": -0.9259,
    "a:r:06": -0.5516,
    "a:r:07": -0.2443,
    "a:r:08": -0.7015,
    "a:r:09": 0.0435,
    "a:r:10": -0.8479,
    "a:r:11": -0.1838,
    "a:r:12": -0.6972,
    "a:r:13": -0.4654,
    "a:r:14": -0.3495,
    "a:r:15": -1.0126,
    "a:h:0": 3.343,
    "a:h:3": 2.7096,
    "a:h:7": 1.7352,
    "a:o:0": -0.02,
    "a:o:2": -0.0137,
    "x:extract:improve": 5.7526,
    "x:late:extract": 0.5639,
    "x:pressure:saturate": 0.0024,
    "x:declared:handle": -0.0265,
    "x:unhandled:handle": -0.0086,
    "x:specs:declare": -0.1306,
    "x:handleable:declare": -0.117,
    "x:handleable:claim": -0.4819,
    "x:claimable:declare": -0.1172,
    "x:claimable:claim": -4.8152,
    "x:macro:len": 2.7303,
    "x:macro:certified": 2.3403,
    "x:macro:score": 2.5058,
    "x:macro:spelled": -1.7652,
    "x:macro:saturating": -1.7612,
    "x:macro:nclaims": 0.2959,
    "x:macro:reach_sites": 2.3756,
    "x:macro:reach_claimable": 3.7667,
    "x:mc:0": -0.0445,
    "x:mc:7": 1.228,
    "x:mh:0": -0.0071,
    "x:mh:7": 0.4991,
}
