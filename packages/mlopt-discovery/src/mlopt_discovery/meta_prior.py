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
``gen_cases`` corpus-coverage extension):

    .venv/bin/python -m mlopt_discovery.meta_player \
        --seed 0 --train-cases 400 --eval-cases 40 --budget 24 \
        --temperature 1.2 --epsilon 0.25 \
        --imitate-epochs 3 --epochs 5 --json run.json

— 3 oracle-imitation passes (1200/1200 demonstrated moves cloned)
then 5 REINFORCE passes over ``gen_cases(0, 400)``.  The corpus now
carries the handler tags and site multiplicities the five base
kinds never produced: ``square`` (claimable-but-op-count-neutral —
the first boards where a claim must be *refused*), ``square_mix``
/``tri`` (a paying silu/swiglu site beside a square site), and
``dual`` /``dual_swiglu`` (two disjoint silu-family sites whose
oracle argmax is the spelled ``declare→saturate→handle→extract``
macro, not a ``claim_*_extract`` compression).  Held-out on
``gen_cases(7919, 40)`` at the same budget: **+0.32 mean reward
(learned-greedy; +0.30 sampled), declared-use 26/40** — vs the
previous bake's +0.20/+0.18 on the *old* five-kind stream; on the
extended stream the superseded weights score +0.29 greedy, so the
re-bake itself is worth ~+0.03 and the corpus lift the rest
(oracle ceiling +0.35).  Replicated at the second held-out offset
(``gen_cases(31337, 40)``: +0.33/+0.29 vs +0.30 greedy for the old
weights; oracle +0.35).  Per-tag reads: ``square`` boards decline
at the oracle's -0.05 exactly, ``dual_swiglu`` matches the oracle
(+0.52); the residual gap sits on ``dual``/``tri`` where the
winning offer is the four-step spelled macro — feature-blind
against the ``claim_*_extract`` compressions (see
``project/reports/2026-11-07-meta-arena-corpus-coverage.md``).

Weights are rounded to 4 decimals and near-zero entries dropped —
``LinearPolicy`` reads a missing feature as ``0``.
"""

from __future__ import annotations

#: The trained meta-arena policy — feature → logit weight.
META_ARENA_PLAYER_WEIGHTS: dict[str, float] = {
    "op:fire": -7.5837,
    "op:saturate": -1.903,
    "op:declare": -1.7976,
    "op:handle": -0.0337,
    "op:claim": 0.0575,
    "op:extract": 8.3429,
    "op:macro": 2.918,
    "a:budget": -0.4988,
    "a:unbounded": -0.9055,
    "a:spec_i": -0.0297,
    "a:r:00": -0.7455,
    "a:r:01": -0.716,
    "a:r:02": -0.0124,
    "a:r:03": -0.4875,
    "a:r:04": -0.5899,
    "a:r:05": -0.7478,
    "a:r:06": -0.3671,
    "a:r:07": -0.1997,
    "a:r:08": -0.5982,
    "a:r:09": -0.0186,
    "a:r:10": -0.6612,
    "a:r:11": -0.2504,
    "a:r:12": -0.5414,
    "a:r:13": -0.4186,
    "a:r:14": -0.3523,
    "a:r:15": -0.877,
    "a:h:0": 3.4027,
    "a:h:3": 3.0712,
    "a:o:0": -0.0199,
    "a:o:2": -0.0137,
    "x:extract:improve": 5.8534,
    "x:late:extract": 0.5341,
    "x:pressure:saturate": -0.0003,
    "x:declared:handle": -0.0265,
    "x:unhandled:handle": -0.0086,
    "x:specs:declare": -0.1146,
    "x:handleable:declare": -0.1268,
    "x:handleable:claim": 1.0249,
    "x:claimable:declare": -0.1273,
    "x:claimable:claim": -1.0537,
    "x:macro:len": -0.2548,
    "x:macro:certified": 2.924,
    "x:macro:score": 2.7804,
}
