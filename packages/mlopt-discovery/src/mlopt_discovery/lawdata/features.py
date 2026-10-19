"""The player feature schemas — feature names and bucket widths.

``arena_player.LearnedPlayer`` and ``meta_player.MetaLearnedPlayer``
are linear softmax policies over their boards' legal actions:
``score(move) = weight · feature``.  The *schema* — the feature names
and the bucket widths the identity-flavoured features hash into — is
content, so it lives here; the learned weights are the tables in
:mod:`mlopt_discovery.lawdata.weights`.
"""

from __future__ import annotations

from .moves import ARENA_MOVE_ORDER

__all__ = [
    "ARENA_FEATURES",
    "ARENA_HASH_BUCKETS",
    "ARENA_HISTORY_STATS",
    "META_ARENA_FEATURES",
    "META_ARENA_HASH_BUCKETS",
]

# ---------------------------------------------------------------------------
#  The learned arena player — feature schema
# ---------------------------------------------------------------------------
#
#  ``arena_player.LearnedPlayer`` is a linear softmax policy over
#  ``arena.legal_actions``: score(move) = weight · feature.  The
#  *schema* — the feature names and the bucket widths the identity-
#  flavoured features hash into — is content, so it lives here; the
#  learned weights are a ``{name: float}`` table over the same names.

#: Bucket widths for the identity-flavoured action features.  A
#: linear policy cannot name "the softsign kernel" — vocabulary is
#: dynamic — so kernel / carrier / apply / premise identities enter
#: the schema as stable-hash buckets (the same name lands in the same
#: bucket on any board; collisions merge two names' weights, which a
#: linear probe tolerates).  Widths are content: a finer identity
#: split is a data edit, not a code edit.
ARENA_HASH_BUCKETS: dict[str, int] = {
    "kernel": 16,
    "carrier": 8,
    "apply": 8,
    "premise": 16,
}

#: The per-op outcome stats the player accumulates within an episode
#: by diffing consecutive observations (a new or re-gauntleted
#: ``ObjectView`` is the action's measured result): ``n`` is the
#: log-scaled play count; ``usable`` / ``pay`` / ``truthfail`` /
#: ``decline`` are rates over that op's plays.
ARENA_HISTORY_STATS: tuple[str, ...] = (
    "n",
    "usable",
    "pay",
    "truthfail",
    "decline",
)

#: The feature schema — ``weight[ARENA_FEATURES[i]]`` is the learned
#: player's scorer.  ``st:*`` rows broadcast board summaries to every
#: move; ``op:*`` the op one-hot; ``h:<op>:<stat>`` the within-episode
#: outcome rates (nonzero only on that op's rows); ``a:*`` the move's
#: own parameters (spec size / coverage, kernel, carrier and premise
#: identity buckets, the specialize/self flags, the ingest share, the
#: ``a:rescue`` "guard the conditionals" signature); ``t:*`` the
#: referenced object's guard/verdict state for the object-targeting
#: ops.  Feature names are data: a trained weight table
#: (:data:`ARENA_PLAYER_WEIGHTS`) slots in by name, and a richer
#: feature is a new row here plus a small featurizer edit.
ARENA_FEATURES: tuple[str, ...] = (
    # bias and state broadcasts
    "bias",
    "st:progress",
    "st:objects",
    "st:usable",
    "st:unguarded",
    "st:truthfail",
    "st:condprop",
    "st:pending",
    "st:trend",
    "st:decline",
    "st:guarded",
    # op one-hot
    *(f"op:{op}" for op in ARENA_MOVE_ORDER),
    # op x state interactions — a broadcast state feature shifts
    # every logit together and cancels in the softmax, so state can
    # only steer the policy through a product term ("auto_cond when
    # unguarded objects exist", "lift while the episode is paying").
    *(f"x:{op}:trend" for op in ARENA_MOVE_ORDER),
    "x:ingest:pending",
    "x:auto_cond:unguarded",
    "x:auto_cond:condprop",
    "x:compose:objects",
    "x:relax_guard:guarded",
    "x:specialize:usable",
    # per-op within-episode outcome rates
    *(
        f"h:{op}:{stat}"
        for op in ARENA_MOVE_ORDER
        for stat in ARENA_HISTORY_STATS
    ),
    # shared structural scalars (fold/lift/compose spelled spec)
    "a:spec_sz",
    "a:mvars",
    "a:whole",
    "a:cases",
    # fold kernel: unary vs all-metavar form + identity buckets
    "a:k_unary",
    "a:k_full",
    "a:k_arity",
    *(f"a:k:{b:02d}" for b in range(ARENA_HASH_BUCKETS["kernel"])),
    # lift carrier: statefulness + carrier/apply identity buckets
    "a:c_stateful",
    *(f"a:c:{b}" for b in range(ARENA_HASH_BUCKETS["carrier"])),
    *(f"a:ap:{b}" for b in range(ARENA_HASH_BUCKETS["apply"])),
    # compose: specialization, arity, self-compose, premise buckets
    "a:compose_spec",
    "a:rest",
    "a:self",
    *(f"a:p1:{b:02d}" for b in range(ARENA_HASH_BUCKETS["premise"])),
    *(f"a:p2:{b:02d}" for b in range(ARENA_HASH_BUCKETS["premise"])),
    # ingest share and the conditional-rescue signature
    "a:ingest_frac",
    "a:rescue",
    # the referenced object's state (auto_cond/relax_guard/specialize)
    "t:known",
    "t:usable",
    "t:guard",
    "t:f_truth",
    "t:f_fulldata",
    "t:f_typedpay",
    "t:f_novelty",
    "t:fires",
    "t:paid",
    "t:cleared",
    "t:conditional",
)

# ---------------------------------------------------------------------------
#  The learned meta-arena player — feature schema
# ---------------------------------------------------------------------------
#
#  ``meta_player.MetaLearnedPlayer`` is a linear softmax policy over
#  ``meta_arena.legal_actions`` — the same architecture as
#  ``arena_player.LearnedPlayer``, featurizing ``MetaState`` instead of
#  ``ArenaState``.  The schema and bucket widths are content, so they
#  live here; the learned weights are a ``{name: float}`` table.

#: Bucket widths for the identity-flavoured action features —
#: ``fire``'s rule name, ``handle``'s handler tag and target object.
#: A linear policy cannot name "the silu handler" — vocabulary is
#: dynamic — so identities enter as stable-sha256 buckets (the same
#: name lands in the same bucket on any board; collisions merge two
#: names' weights, which a linear probe tolerates).
META_ARENA_HASH_BUCKETS: dict[str, int] = {
    "rule": 16,
    "handler": 4,
    "object": 8,
}

#: The feature schema — ``weight[META_ARENA_FEATURES[i]]`` is the
#: learned player's scorer.  ``st:*`` rows broadcast board summaries
#: to every move; ``op:*`` the move-kind one-hot; ``h:*`` the
#: within-episode play rates (the adaptation channel — "declares
#: haven't been tried yet" is a feature); ``a:*`` the move's own
#: parameters (saturate budget, spec index, rule/handler/object
#: identity buckets); ``x:*`` the interaction terms — a broadcast
#: state feature cancels in the softmax, so state can only steer the
#: policy through products like ``x:extract:improve``.
META_ARENA_FEATURES: tuple[str, ...] = (
    "bias",
    # state broadcasts
    "st:steps",
    "st:enodes_frac",
    "st:classes_frac",
    "st:improve",
    "st:declared",
    "st:handled",
    "st:unhandled",
    "st:handleable",
    "st:sites",
    "st:claimable",
    "st:specs",
    "st:fires",
    # within-episode play rates
    "h:fire",
    "h:saturate",
    "h:declare",
    "h:handle",
    "h:claim",
    # op one-hots
    "op:fire",
    "op:saturate",
    "op:declare",
    "op:handle",
    "op:claim",
    "op:extract",
    # action parameters
    "a:budget",
    "a:unbounded",
    "a:spec_i",
    *(f"a:r:{b:02d}" for b in range(META_ARENA_HASH_BUCKETS["rule"])),
    *(f"a:h:{b}" for b in range(META_ARENA_HASH_BUCKETS["handler"])),
    *(f"a:o:{b}" for b in range(META_ARENA_HASH_BUCKETS["object"])),
    # interactions
    "x:extract:improve",
    "x:late:extract",
    "x:pressure:saturate",
    "x:declared:handle",
    "x:unhandled:handle",
    "x:specs:declare",
    "x:handleable:declare",
    "x:handleable:claim",
    "x:claimable:declare",
    "x:claimable:claim",
)
