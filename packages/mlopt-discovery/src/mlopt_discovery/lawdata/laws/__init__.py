"""The law half of ``lawdata`` — destined for ``graphopt``/``lawmint``.

Plan 0032 split: these are the *domain* tables — the content the
tensor-law boards enumerate and the mint machinery draws from.  They
are specific to the mlopt domains (the meta/joint/search/open arenas,
the proposal pipeline, the verifier's instantiation banks), so at repo
extraction this half stays with the domain code in ``graphopt`` (the
``lawmint`` engine), not with the generic game layer.

* :mod:`mlopt_discovery.lawdata.laws.banks` — the enumeration banks:
  shape pools, attr-kind tables, the const domain, instantiation
  defaults, grammar alphabets, family/coherence pools.
* :mod:`mlopt_discovery.lawdata.laws.handlers` — the ``handle`` move's
  handler table and the reverse-mode VJP table.
* :mod:`mlopt_discovery.lawdata.laws.macros` — the shipped ``MACROS``
  minted-move table.
* :mod:`mlopt_discovery.lawdata.laws.moves` — the move vocabulary:
  generator/corpus arms, metavariable and literal pools, the arena
  move order, saturate budgets, specialize pin banks.
* :mod:`mlopt_discovery.lawdata.laws.terms` — the proposal corpus:
  seed terms, schema libraries, candidate laws, synthetic cases.
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
from .handlers import (
    HANDLERS,
    REVERSE,
)
from .macros import (
    MACROS,
)
from .moves import (
    ARENA_MOVE_ORDER,
    CONST_LEAVES,
    CORPUS_ARMS,
    GENERATOR_ORDER,
    META_SATURATE_BUDGETS,
    MV_NAMES,
    SPECIALIZE_AXES,
    SPECIALIZE_SCALARS,
)
from .terms import (
    CANDIDATE_LAWS,
    GRAMMAR_SCHEMAS,
    SEED_TERMS,
    SHAPE_SCHEMAS,
    SYNTHETIC_CASES,
)

__all__ = [
    "ARENA_MOVE_ORDER",
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
    "MACROS",
    "META_SATURATE_BUDGETS",
    "MV_NAMES",
    "PURPOSE_BUILT",
    "REVERSE",
    "SEED_TERMS",
    "SHAPE_POOL",
    "SHAPE_SCHEMAS",
    "SPECIALIZE_AXES",
    "SPECIALIZE_SCALARS",
    "SYNTHETIC_CASES",
    "TUPLE_SOURCES",
    "TUPLE_SOURCE_SHAPE",
    "VIEWED_SHAPES",
]
