"""Weight and score tables — priors, rewards and learned presets.

The fixed prior weights added to the learned logits, the referee's
score composition, the arena / meta-arena reward tables, the learned
player's per-step candidate cap, and the weight tables a trained
snapshot serializes into (arena, meta-arena and open-game).  All
data: a policy tunes these by edit, never by a code change.
"""

from __future__ import annotations

__all__ = [
    "ARENA_PLAYER_CANDIDATES",
    "ARENA_PLAYER_WEIGHTS",
    "ARENA_REWARD",
    "META_ARENA_PLAYER_WEIGHTS",
    "META_ARENA_REWARD",
    "OPEN_GAME_PLAYER_WEIGHTS",
    "PLAYER_ARENA_PLAYER_WEIGHTS",
    "PRIOR_WEIGHTS",
    "REFEREE_SCORE",
]

# ---------------------------------------------------------------------------
#  Meta-game — priors and score
# ---------------------------------------------------------------------------

#: Fixed prior weights added to the learned logits.  Each name is a
#: corpus-informed bias the policy may learn to override:
#:
#: * ``child`` — census child frequency (op, metavariable *and*
#:   const children are counted — the census's ``·``/``const``
#:   entries);
#: * ``reuse`` — "the op occurs on the other side";
#: * ``bound_mv`` — "metavariable already bound" (RHS);
#: * ``fresh_mv`` — "fresh metavariable" (LHS — distinct bindings
#:   are what naturality and factoring preconditions need);
#: * ``commute`` — RHS root repeats the LHS root op;
#: * ``swap_nest`` — RHS root lifts an op one level down on the LHS
#:   (the naturality move ``f(g u, g v) -> g(f u v)``);
#: * ``depth_decay`` — op-placement prior decays with hole depth:
#:   the corpus's frequent shapes are shallow (≤ ~3 ops), so below
#:   the seeded skeleton level leaves dominate and constructed LHSs
#:   stay applicable.
PRIOR_WEIGHTS: dict[str, float] = {
    "child": 2.0,
    "reuse": 2.0,
    "bound_mv": 1.2,
    "fresh_mv": 0.4,
    "commute": 1.0,
    "swap_nest": 1.2,
    "depth_decay": 0.55,
}

#: The referee's score composition — ``base + fire·min(fires,
#: fire_cap) + paid·paid + rel_drop·rel_drop``, applied only to a
#: *true* candidate (the truth gate already scored a false one 0).
REFEREE_SCORE: dict[str, int | float] = {
    "base": 1.0,
    "fire": 0.2,
    "fire_cap": 40,
    "paid": 2.0,
    "rel_drop": 10.0,
}

#: The arena reward composition — partial credit per cleared gauntlet
#: stage, a bonus on the honest ``usable`` verdict, and the holdout
#: firing/pay columns.  Weights are shaped so *the objective
#: dominates the shaping signal*: the 24-step probe measured a
#: stage-clear-dominated reward letting a churn policy outscore a
#: paying playbook (greedy 72 reward / 0 pay vs fixed 23.6 / 2
#: paid).  Stage clears stay as a dense orienting signal but a
#: paying or admitted object outweighs a full budget of churn.
#: ``project/retros/arena-depth.md``.
ARENA_REWARD: dict[str, float] = {
    "stage": 0.1,
    "usable": 10.0,
    "fire": 0.5,
    "paid": 5.0,
}

#: The meta-arena reward composition — the weights of plan 0021's
#: mixed board, shaped like :data:`ARENA_REWARD`: the objective
#: dominates the spend signal.  ``step`` is the flat per-applied-
#: action cost; ``enode`` the per-enode price of graph growth (the
#: measured schedule dimension — a ``fire``/``saturate`` pays for
#: the board it spends); ``delta`` scales the terminal payout —
#: ``(baseline - extracted)/baseline``, collected only when the
#: certificate replays; ``saturate_est`` is the greedy player's
#: enode-spend estimate for an unbudgeted ``saturate`` (a scoring
#: prior, not a budget).
META_ARENA_REWARD: dict[str, float] = {
    "step": 0.05,
    "enode": 0.001,
    "delta": 1.0,
    "saturate_est": 512.0,
}

#: The learned player's weight table — ``{feature_name: logit
#: weight}`` over :data:`ARENA_FEATURES`.  Empty is the honest cold
#: start (uniform policy); a table trained by
#: ``arena_player.train_player`` serializes here by name.
ARENA_PLAYER_WEIGHTS: dict[str, float] = {}

#: The learned player's per-step candidate cap — how many of the
#: enumerated legal moves the policy scores before choosing (a
#: perceptual limit, not a legality change; the slice is re-drawn
#: each step).  ``None`` scores the full ~50k-move board — the
#: unbounded form; the probe uses the cap to keep a step at the
#: gauntlet's pace rather than the scorer's.
ARENA_PLAYER_CANDIDATES: int | None = 2048

#: The shipped meta-arena prior — V2 (macro-aware, 2026-10-19).
#: :class:`mlopt_discovery.meta_player.MetaLearnedPlayer` defaults to
#: it (pass ``weights={}`` for the uniform cold start, the baseline
#: the measurement keeps).  A trained snapshot:
#: ``scripts/train_meta_macro_prior.py --init v1 --epochs 30 --scale
#: 1.8`` — seed 0, 40 cases x 30 epochs of episode-return REINFORCE on
#: the shipped macro-offering legal set, *warm-started* from the V1
#: table (the cold-start retrain scrambled base competence —
#: ``docs/results_macro_bake2.md``), then multiplied by a uniform
#: 1.8 logit scale.  The scale is a decode-sharpness transform: the
#: softmax's argmax is scale-invariant so the greedy decode is
#: untouched, while the sampled decode concentrates on the learned
#: ranking — the fix for the "which offer" mass-splitting the prior
#: bake attempt could not cross.  Reproducible, not hand-tuned.
#:
#: The seeded held-out split (train ``gen_cases(0, 40)``, evaluate
#: ``gen_cases(7919, 16)``, budget 12) earns the bake — the gate
#: table (V1 = the pre-macro table this replaces):
#:
#: ======  ========  ====================  ======================
#: arm     reward    bar                   verdict
#: ======  ========  ====================  ======================
#: greedy  +0.117    >= -0.056 & +0.088    clears both anchors
#: sampled +0.101    >= +0.088             clears scripted+macros
#: v2-base -0.032    >= V1 (-0.056)        macro-free caller ok
#: ======  ========  ====================  ======================
#:
#: (context anchors: V1 -0.056 sampled on the same shipped legal
#: set, scripted playbook -0.100, scripted+macros +0.088.)  Macro
#: pick-rate held out: sampled 11 plays in 7 of 16 cases, greedy
#: 7/7 — selective: zero picks where ``viable`` offers nothing
#: bindable.  The learned line stays legible: ``op:extract`` and
#: ``op:macro`` high, ``x:macro:certified``/``a:h:3`` up — the
#: policy prices the miner's evidence.
#: Regenerate with::
#:
#:     python scripts/train_meta_macro_prior.py \
#:         --init v1 --epochs 30 --scale 1.8
#:
#: Weights are rounded to 4 decimals and near-zero entries dropped —
#: ``LinearPolicy`` reads a missing feature as ``0``.
META_ARENA_PLAYER_WEIGHTS: dict[str, float] = {
    "op:fire": -5.5097,
    "op:saturate": -0.9514,
    "op:declare": -0.1926,
    "op:handle": -0.0316,
    "op:claim": 0.0281,
    "op:extract": 4.1594,
    "op:macro": 2.4981,
    "a:budget": -0.2214,
    "a:unbounded": -0.5086,
    "a:spec_i": -0.0051,
    "a:r:00": -0.6219,
    "a:r:01": -0.4323,
    "a:r:02": -0.0039,
    "a:r:03": -0.3169,
    "a:r:04": -0.5321,
    "a:r:05": -0.51,
    "a:r:06": -0.2971,
    "a:r:07": -0.2612,
    "a:r:08": -0.4722,
    "a:r:09": -0.123,
    "a:r:10": -0.5062,
    "a:r:11": -0.1272,
    "a:r:12": -0.3481,
    "a:r:13": -0.2397,
    "a:r:14": -0.117,
    "a:r:15": -0.6008,
    "a:h:0": 0.1714,
    "a:h:3": 3.0383,
    "a:o:0": -0.0181,
    "a:o:2": -0.0135,
    "x:extract:improve": 0.3224,
    "x:late:extract": 0.2452,
    "x:declared:handle": -0.0255,
    "x:unhandled:handle": -0.0083,
    "x:specs:declare": -0.0163,
    "x:handleable:declare": -0.0489,
    "x:handleable:claim": 0.5695,
    "x:claimable:declare": -0.0489,
    "x:claimable:claim": 0.5695,
    "x:macro:len": 0.9288,
    "x:macro:certified": 2.5054,
    "x:macro:score": 1.3048,
}

#: The open-game learned prior — the shipped preset
#: :class:`mlopt_discovery.open_game.OpenLearnedPlayer` defaults to
#: (empty ⇒ uniform, the cold start the measurement keeps as a
#: baseline).  A trained snapshot: ``scripts/train_open_prior.py``
#: (seed 0, 36 even train pages x 5 epochs, cold start) emits exactly
#: this literal — the table is reproducible, not hand-tuned.
#:
#: The even/odd page split makes the bake earn its place: on the
#: held-out odd pages (seed 0) the baked prior scores 0.285
#: reward/episode against 0.143 for the cold start and 0.134 for
#: random, finds 0.167 useful laws/episode (cold 0.083, random
#: 0.083), and its page *ordering* (``rank_score``) beats the
#: cost-delta heuristic on average precision (0.133-era schema;
#: see ``docs/results_richer_features.md`` — the extended schema
#: measures AP 0.210 under the controlled protocol).  Pinned by
#: ``tests/test_discovery_open_prior.py``.  Regenerate with::
#:
#:     python scripts/train_open_prior.py
#:
#: Weights are rounded to 4 decimals and near-zero entries dropped —
#: ``LinearPolicy`` reads a missing feature as ``0``.
OPEN_GAME_PLAYER_WEIGHTS: dict[str, float] = {
    "op:evaluate": 0.1003,
    "op:stop": -0.1003,
    "a:depth": 0.1695,
    "a:metas": 0.1729,
    "a:ops": 0.3013,
    "a:score": 0.0628,
    "a:score_rank": 0.08,
    "a:score_rel": 0.0249,
    "a:form:comm": 0.0098,
    "a:form:assoc": -0.0083,
    "a:form:rotate": 0.0318,
    "a:form:rekey": -0.0434,
    "a:form:factor-left": -0.0013,
    "a:form:factor-right": 0.0057,
    "a:form:unary-hom": -0.0032,
    "a:form:pull-left": 0.0025,
    "a:form:pull-right": 0.023,
    "a:form:unary-swap": -0.0026,
    "a:form:involution": -0.0011,
    "a:form:unary-inverse": 0.0123,
    "a:form:self-op": -0.0104,
    "a:form:idempotent": 0.0034,
    "a:form:lit-self": 0.0099,
    "a:form:const-op": -0.0219,
    "a:form:id-right": 0.0154,
    "a:form:id-left": 0.0122,
    "a:form:annihilate-right": -0.0141,
    "a:form:annihilate-left": 0.0066,
    "a:form:cancel-self": 0.0023,
    "a:form:cancel-unary-right": -0.0091,
    "a:form:cancel-unary-left": -0.0086,
    "a:form:hom-cross": 0.0228,
    "a:form:bin-unary-right": 0.001,
    "a:form:bin-unary-left": 0.0478,
    "a:form:unary-of-lit": 0.0178,
    "x:score:depth": 0.0048,
    "x:score:ops": 0.0093,
}

#: The player-domain learned prior — the meta-arm
#: :data:`mlopt_discovery.player_arena.PLAYER_SPEC` now defaults to
#: (the ``LearnedPlayer`` over spec records; pass ``weights={}`` for
#: the uniform cold start the measurement keeps as a baseline).  A
#: trained snapshot: ``scripts/train_player_meta_arm.py`` (seed 0,
#: 12 stub-oracle cases x 150 REINFORCE epochs on a 6-epoch
#: ``GreedyEditor`` imitation warm start — the nested oracle stubbed
#: through the board's ``score_fn`` seam, see the driver for the
#: exact tier table) emits exactly this literal — reproducible, not
#: hand-tuned.
#:
#: The bake is earned on the *real* oracle: frozen on fresh
#: ``PlayerArena`` episodes under the canonical nested eval (3
#: epochs x 36 even pages, odd-page held-out, memoized verdicts),
#: the greedy decode emits ``['open:action']`` — GreedyEditor's own
#: child, held-out 0.2827 vs parent 0.2301 (delta +0.0526) on seed 0
#: and 0.3063 vs 0.3003 on seed 1; the sampled decode emits
#: positive-or-zero deltas (never a worse child).  The learned line
#: is legible: ``op:evaluate`` high (probe), ``a:dir:drop`` up with
#: ``a:part:open:action`` the *least* dropped part — the editor's
#: ablate-then-extract game, learned.  Regenerate with::
#:
#:     python scripts/train_player_meta_arm.py
#:
#: Weights are rounded to 4 decimals and near-zero entries dropped —
#: ``LinearPolicy`` reads a missing feature as ``0``.
PLAYER_ARENA_PLAYER_WEIGHTS: dict[str, float] = {
    "op:toggle_feature": 0.5881,
    "op:edit_weight": -3.6214,
    "op:set_lr": -2.2034,
    "op:set_weights": -1.4736,
    "op:evaluate": 5.2226,
    "op:extract": 1.4877,
    "a:delta": -0.0044,
    "a:lr": -0.2332,
    "a:k": 2.742,
    "a:dir:add": -1.953,
    "a:dir:drop": 2.541,
    "a:part:bias": 1.313,
    "a:part:open:action": -0.6754,
    "a:part:open:hist": -0.0591,
    "a:part:open:onehot": -0.4028,
    "a:part:open:state": 0.4124,
    "a:w:a:depth": -0.4003,
    "a:w:a:metas": -0.4032,
    "a:w:a:ops": -0.4014,
    "a:w:a:score": -0.4001,
    "a:w:bias": -0.4023,
    "a:w:h:evaluate": -0.4063,
    "a:w:op:evaluate": -0.4017,
    "a:w:op:stop": -0.4037,
    "a:w:st:useful_rate": -0.4024,
    "a:table:baked": -0.7396,
    "a:table:cold": -0.734,
}

#  The kernel board's prior moved with the domain — plan 0032 phase
#  4 extracted the kernelopt surface into the private
#  ``handl-lang/kernelopt`` repo; ``KERNEL_PLAYER_WEIGHTS`` now lives
#  at ``mlopt_kernel.weights``.
