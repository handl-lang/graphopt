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

#: The shipped meta-arena prior —
#: :class:`mlopt_discovery.meta_player.MetaLearnedPlayer` defaults to
#: it (pass ``weights={}`` for the uniform cold start, the baseline
#: the measurement keeps).  A trained snapshot:
#: ``scripts/train_meta_prior.py`` (seed 0, 40 cases x 30 epochs, cold
#: start) emits exactly this literal — reproducible, not hand-tuned.
#:
#: The seeded held-out split (train ``gen_cases(0, 40)``, evaluate
#: ``gen_cases(7919, 16)``) makes the bake earn its place: the trained
#: prior scores -0.06 mean reward/episode against -0.49 for the cold
#: start, -0.57 for random and -0.60 for the cold-greedy arm (the
#: hand-written ``scripted`` playbook scores -0.10).  The learned
#: line is legible — ``op:extract`` high, ``op:fire`` low: the board
#: prices spend, so the paying move is the cheap extract.
#: Regenerate with::
#:
#:     python scripts/train_meta_prior.py
#:
#: Weights are rounded to 4 decimals and near-zero entries dropped —
#: ``LinearPolicy`` reads a missing feature as ``0``.
META_ARENA_PLAYER_WEIGHTS: dict[str, float] = {
    "op:fire": -2.7853,
    "op:saturate": -0.4063,
    "op:declare": -0.2087,
    "op:handle": -0.0149,
    "op:claim": 0.2662,
    "op:extract": 3.1491,
    "a:budget": -0.0816,
    "a:unbounded": -0.2431,
    "a:spec_i": -0.0024,
    "a:r:00": -0.3249,
    "a:r:01": -0.1761,
    "a:r:03": -0.1637,
    "a:r:04": -0.2088,
    "a:r:05": -0.265,
    "a:r:06": -0.145,
    "a:r:07": -0.1296,
    "a:r:08": -0.2153,
    "a:r:09": -0.1006,
    "a:r:10": -0.2656,
    "a:r:11": -0.0578,
    "a:r:12": -0.2285,
    "a:r:13": -0.1204,
    "a:r:14": -0.0764,
    "a:r:15": -0.3076,
    "a:h:0": 0.1346,
    "a:h:3": 0.1166,
    "a:o:0": -0.023,
    "a:o:2": 0.0081,
    "x:extract:improve": 0.1599,
    "x:late:extract": 0.1336,
    "x:declared:handle": -0.006,
    "x:unhandled:handle": -0.0028,
    "x:specs:declare": -0.0125,
    "x:handleable:declare": -0.0192,
    "x:handleable:claim": 0.0551,
    "x:claimable:declare": -0.0192,
    "x:claimable:claim": 0.0551,
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
