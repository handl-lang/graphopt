"""Discovery content tables — everything the player may learn, as data.

The discovery engine's hardcoded tables are *content*, not code:
shape banks an oracle enumerates, a per-op constant domain a guarded
sweep can mint, the schema library a proposal strategy replays, the
generator-arm inventory a guide schedules, the prior and reward
weights a player's score is composed of.  A learned policy could
plausibly choose any of them — so they live here, in one pure-data
module (the ``mlopt_core.opmeta`` idiom: the table is the single
home, consumers bind projections or the table itself, and a
consistency test — ``tests/test_lawdata.py`` — pins the relation).

What is deliberately *not* here: budgets and tolerances (resource
knobs, not content), enumeration *algorithms* (the per-op attr-domain
generators in ``mlopt_discovery.oracle._attr_options`` are shape-
dependent code), recognizer hooks (``pipeline._RECOGNIZERS`` maps a
census signature to a *function* — a hook, not a record), nn.Module
corpus registries (``impact._model_cases``, ``intake.candidates``,
``zoo`` — kernel bodies are code), and store/SQL schema vocabulary
(``evidence``'s column lists are the database contract).

Term specs
----------

Several tables carry *terms* (seed programs, schema equalities) in
the declarative spec language :func:`object_synthesis.term_from_spec`
reads, extended with two leaf forms the consumers resolve first::

    spec := str                        # metavariable leaf ("A", "x")
          | int | float                # Const leaf
          | ("var", name, (dims, ...)) # Var(name, TensorType(dims))
          | ("param", name, (dims, ..))# Param(name, TensorType(dims))
          | (op, *specs)               # Op.make(op, *specs)
          | (op, *specs, {attrs})      # trailing dict = attr map

so a new seed term, schema or candidate law is a *new data row*, not
a Python edit — the shape a model would write a new entry in.
"""

from __future__ import annotations

from .banks import (
    ATTR_KIND_OVERRIDES,
    ATTR_KINDS,
    BASE_SHAPES,
    CHAIN_VIEWED_SHAPES,
    COHERENCE_CLUSTER,
    COHERENCE_CORPUS_MODELS,
    CONST_DOMAIN,
    FALLBACK_AXES,
    FALLBACK_SHAPES,
    FAMILIES,
    FREE_SENTINELS,
    GRAMMAR_BINARY_OPS,
    GRAMMAR_LITERALS,
    GRAMMAR_UNARY_OPS,
    INSTANCE_ATTR_DEFAULTS,
    INSTANCE_LEAF_SHAPES,
    INSTANCE_SCALAR_MVARS,
    JOINT_SHAPES,
    JOINT_VIEW_SHAPES,
    LINEAR_HELD_SHAPES,
    LINEAR_TRAIN_SHAPES,
    PURPOSE_BUILT,
    SHAPE_POOL,
    TUPLE_SOURCE_SHAPE,
    TUPLE_SOURCES,
    VIEWED_SHAPES,
)
from .features import (
    ARENA_FEATURES,
    ARENA_HASH_BUCKETS,
    ARENA_HISTORY_STATS,
    META_ARENA_FEATURES,
    META_ARENA_HASH_BUCKETS,
)
from .handlers import (
    HANDLERS,
    REVERSE,
)
from .moves import (
    ARENA_MOVE_ORDER,
    CONST_LEAVES,
    CORPUS_ARMS,
    GENERATOR_ORDER,
    META_SATURATE_BUDGETS,
    MV_NAMES,
)
from .moves import (
    SPECIALIZE_AXES as SPECIALIZE_AXES,
)
from .moves import (
    SPECIALIZE_SCALARS as SPECIALIZE_SCALARS,
)
from .terms import (
    CANDIDATE_LAWS,
    GRAMMAR_SCHEMAS,
    SEED_TERMS,
    SHAPE_SCHEMAS,
)
from .terms import (
    SYNTHETIC_CASES as SYNTHETIC_CASES,
)
from .weights import (
    ARENA_PLAYER_CANDIDATES,
    ARENA_PLAYER_WEIGHTS,
    ARENA_REWARD,
    META_ARENA_PLAYER_WEIGHTS,
    META_ARENA_REWARD,
    OPEN_GAME_PLAYER_WEIGHTS,
    PRIOR_WEIGHTS,
    REFEREE_SCORE,
)

__all__ = [
    "ARENA_FEATURES",
    "ARENA_HASH_BUCKETS",
    "ARENA_HISTORY_STATS",
    "ARENA_MOVE_ORDER",
    "ARENA_PLAYER_CANDIDATES",
    "ARENA_PLAYER_WEIGHTS",
    "ARENA_REWARD",
    "ATTR_KINDS",
    "ATTR_KIND_OVERRIDES",
    "BASE_SHAPES",
    "CANDIDATE_LAWS",
    "CHAIN_VIEWED_SHAPES",
    "COHERENCE_CLUSTER",
    "COHERENCE_CORPUS_MODELS",
    "CONST_DOMAIN",
    "CONST_LEAVES",
    "CORPUS_ARMS",
    "FALLBACK_AXES",
    "FALLBACK_SHAPES",
    "FAMILIES",
    "FREE_SENTINELS",
    "GENERATOR_ORDER",
    "GRAMMAR_BINARY_OPS",
    "GRAMMAR_LITERALS",
    "GRAMMAR_SCHEMAS",
    "GRAMMAR_UNARY_OPS",
    "HANDLERS",
    "INSTANCE_ATTR_DEFAULTS",
    "INSTANCE_LEAF_SHAPES",
    "INSTANCE_SCALAR_MVARS",
    "JOINT_SHAPES",
    "JOINT_VIEW_SHAPES",
    "LINEAR_HELD_SHAPES",
    "LINEAR_TRAIN_SHAPES",
    "META_ARENA_FEATURES",
    "META_ARENA_HASH_BUCKETS",
    "META_ARENA_PLAYER_WEIGHTS",
    "META_ARENA_REWARD",
    "META_SATURATE_BUDGETS",
    "MV_NAMES",
    "OPEN_GAME_PLAYER_WEIGHTS",
    "PRIOR_WEIGHTS",
    "PURPOSE_BUILT",
    "REFEREE_SCORE",
    "REVERSE",
    "SEED_TERMS",
    "SHAPE_POOL",
    "SHAPE_SCHEMAS",
    "TUPLE_SOURCES",
    "TUPLE_SOURCE_SHAPE",
    "VIEWED_SHAPES",
]
