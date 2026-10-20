"""mlopt_core.laws — rewrite laws and graph passes, split by domain.

The machinery ↔ bank seam (plan 0032)
------------------------------------

This package holds two things that the repo split sends to *different*
repos — the machinery is substrate, the bank is domain content:

**Machinery → ``eqsat``** (the gauntlet-facing substrate; domain-free):

* :mod:`mlopt_core.laws.base` — the ``R`` rewrite constructor plus the
  shared check-hook machinery (``_SHAPE_MEMO``, ``_shape_of``, small
  term predicates).
* :mod:`mlopt_core.laws.cond` — the declarative side-condition and
  derive DSLs: a ``cond`` is a serializable tuple tree interpreted
  against the same ``bound`` dict ``check`` sees, and a ``dspec`` is a
  serializable ``{NAME: expr}`` map evaluated to the
  ``{"$attr:NAME": value}`` extras ``derive`` returns (``R(...,
  cond=...)`` / ``R(..., dspec=...)`` fold them into the rule's hooks
  at construction).
* :mod:`mlopt_core.laws.serialize` — laws as data: the
  ``Rewrite`` ↔ JSON record codec (``law_to_data`` /
  ``law_from_data``), the ``missing_hooks`` honesty flag for
  procedural ``check``/``derive`` remainders, and ``alpha_key``, the
  structural identity the lemma store keys rows by.
* :mod:`mlopt_core.laws.tags` — the rule-tag constants
  (``SYMMETRY`` / ``EXPANSIVE`` / ``SUBSUMED`` / ``FUSION`` / …).
* :mod:`mlopt_core.laws.ruleset` — :class:`RuleSet`, the composable
  rule-set value (+/-/& algebra, ``named``/``tagged`` subsets, the
  ``priorities`` scheduling map with ``EARLY``/``NORMAL``/``LATE``).
  The machinery stays; the named *presets* (``SIMPLIFICATION`` /
  ``CATEGORICAL`` / ``FUSION`` / ``SYMMETRY`` / ``CARRIERS`` /
  ``WITH_LAYOUT`` / ``DEFAULT`` / ``FULL``) are compositions over bank
  rows and travel with the bank.
* :mod:`mlopt_core.laws.provenance` — the authorship registry
  machinery: :func:`law_provenance` classifies every law as
  human-authored or machine-admitted, registry-level rather than a
  per-law field.  The registry travels; the *rows*
  (``MACHINE_SHIPPED`` / ``MACHINE_STORED`` contents) name bank laws
  and travel with the bank.

**Bank → ``graphopt``** (the tensor-domain contents):

* :mod:`mlopt_core.laws.tensor` — the tensor-algebra laws:
  ``SIMPLIFICATION_RULES``, ``CATEGORICAL_RULES``, ``SDPA_FOLD_RULES``,
  ``ALL_RULES`` / ``all_rules()``.
* :mod:`mlopt_core.laws.minted` — the machine-admitted laws replayed
  from embedded premise-closure records.
* :mod:`mlopt_core.laws.scan` — the scan-monoid law sets: ``SCAN_LAWS``
  (dense affine carrier) and ``SCAN_DIAG_LAWS`` (diagonal-affine).
* :mod:`mlopt_core.laws.attention` — the attention-path laws:
  ``ATTENTION_RULES`` (rotary composition and scale commutation, the
  right-multiply absorb, the score-scale migration) — opt-in via the
  ``attention`` preset, never in ``DEFAULT``/``all_rules()``.
* :mod:`mlopt_core.laws.layout` — the transpose/layout laws:
  ``LAYOUT_RULES`` (pointwise commutation, involution, the
  product-transpose law, and the ``linear``/NT-GEMM bridge), folded
  into ``ALL_RULES``.
* :mod:`mlopt_core.laws.pairing` — the non-local passes over the whole
  e-graph (pairing, weight sharing).  These are diagram-level passes,
  not equational laws — domain heuristics, bank side.
* :mod:`mlopt_core.laws.factored` — the factored-parameter path:
  ``offer_low_rank_factors`` detects numerically low-rank weight
  values and offers the ``(x@A)@B`` / ``linear(linear(x,A),B)``
  member into the consumer's e-class under a certified bound.
* :mod:`mlopt_core.laws.specials` — the exact-elision sibling:
  ``offer_weight_specials`` detects structurally-special stored
  weights (identity, diagonal, zero, dead/duplicate slices,
  block-diagonal) and offers exact members under a certified zero
  bound.
* :mod:`mlopt_core.laws.headshare` — ``share_duplicate_attention_heads``:
  the compute-level fold for bitwise-equal attention heads — an
  ``sdpa`` over provably-identical heads runs on the ``k`` unique
  ones and re-expands by gather (exact, witnessed).

The seam is documentary for now — extraction moves the bank modules
physically (``laws/bank/`` or a sibling bank package); every name
below stays re-exported at this root until then, so callers bind
``mlopt_core.laws.X`` unchanged.

This package is the canonical rewrite surface — the historical
``mlopt_core.rules`` module (and its ``mlopt.rules`` façade alias)
is gone.  The loose group lists (``*_RULES`` / ``*_LAWS``,
``all_rules()``) remain as thin list aliases during migration; the
first-class values are the ``RuleSet`` presets.
"""

# Public surface (in ``__all__``) plus the private side-condition /
# memo helpers historically reachable through the removed
# ``mlopt_core.rules`` shim (and its ``mlopt.rules`` alias).  The
# private names are re-exported (hence ``noqa: F401``) so
# ``mlopt_core.laws._x`` keeps resolving, but stay out of ``__all__``.
from mlopt_core.laws import provenance as provenance
from mlopt_core.laws import tags as tags
from mlopt_core.laws.attention import (  # noqa: F401
    ATTENTION_RULES,
    LINEAR_MM_ABSORB,
    LINEAR_MM_ABSORB_BIAS,
    LINEAR_MM_ABSORB_BIAS_REV,
    LINEAR_MM_ABSORB_REV,
    LINEAR_OUT_SCALE,
    LINEAR_OUT_SCALE_REV,
    NATURALITY_SCALAR_LEFT,
    NATURALITY_SCALAR_LEFT_REV,
    ROPE_CAT_COMPOSE,
    ROPE_CAT_SCALE_IN,
    ROPE_CAT_SCALE_OUT,
    ROPE_RH_SCALE_IN,
    ROPE_RH_SCALE_OUT,
    _check_left_scale,
    _check_linear_mm_absorb,
    _check_linear_mm_absorb_bias,
    _check_linear_out_scale,
    _check_rope_cat_compose,
    _check_rope_scale,
    _check_uniform,
    _half_bounds,
    _is_uniform,
    _replant,
    _rope_axis,
    _rope_cat,
    _rope_rh,
    _slice,
    _to_end,
    _view_scale_rules,
)
from mlopt_core.laws.base import (  # noqa: F401
    _SHAPE_MEMO,
    R,
    _is_channel_scale,
    _is_row_scale,
    _is_scalar,
    _shape_of,
)
from mlopt_core.laws.factored import (
    offer_low_rank_factors,
)
from mlopt_core.laws.headshare import (
    headshare_keys_hold,
    share_duplicate_attention_heads,
)
from mlopt_core.laws.layout import (  # noqa: F401
    LAYOUT_RULES,
    LINEAR_FROM_MM_T,
    LINEAR_FROM_MM_T_BARE,
    LINEAR_TO_MM_T,
    _bound_axes,
    _check_commute_binary,
    _check_commute_unary,
    _check_involution,
    _check_linear_is_mm_t,
    _check_mm_t_is_linear,
    _check_mm_transposes,
    _check_transpose_matmul,
    _is_swap,
    _linear_shapes_ok,
)
from mlopt_core.laws.pairing import (  # noqa: F401
    _CONV_ATTR_KEYS,
    _pair_shared_input,
    _term_has_var,
    _wshape,
    pair_shared_input_convs,
    pair_shared_input_linears,
    share_duplicate_param_slices,
    share_duplicate_params,
)
from mlopt_core.laws.provenance import (
    MACHINE_LAWS,
    MACHINE_SHIPPED,
    MACHINE_STORED,
    law_provenance,
)
from mlopt_core.laws.ruleset import (
    CARRIER_SEARCH,
    CARRIERS,
    CATEGORICAL,
    DEFAULT,
    EARLY,
    FULL,
    FUSION,
    LATE,
    NORMAL,
    PRESETS,
    SIMPLIFICATION,
    SYMMETRY,
    WITH_LAYOUT,
    RuleSet,
    preset,
)
from mlopt_core.laws.scan import (  # noqa: F401
    _AFFD_UNIT,
    AFF_ASSOC,
    AFF_ASSOC_REV,
    AFF_COMPOSE_UNFOLD,
    AFF_LIFT,
    AFF_LIFT_STEP,
    AFF_UNLIFT,
    AFFD_ASSOC,
    AFFD_ASSOC_REV,
    AFFD_COMPOSE_UNFOLD,
    AFFD_LIFT,
    AFFD_LIFT_POST,
    AFFD_LIFT_POST_SWAP,
    AFFD_LIFT_STEP,
    AFFD_LIFT_STEP_POST,
    AFFD_LIFT_STEP_POST_SWAP,
    AFFD_LIFT_STEP_SWAP,
    AFFD_LIFT_SWAP,
    AFFD_LIFT_UNIT,
    AFFD_LIFT_UNIT_POST,
    AFFD_LIFT_UNIT_STEP,
    AFFD_LIFT_UNIT_STEP_POST,
    AFFD_UNLIFT,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
    _affd_state_like,
    _affd_unit_state_like,
    _derive_affd_unit,
)
from mlopt_core.laws.specials import (
    offer_weight_specials,
)
from mlopt_core.laws.tensor import (  # noqa: F401
    _QK_SCORES,
    _QKV_CAT,
    _REPEAT_KV,
    _REPEAT_V,
    ALL_RULES,
    ALL_RULES_WITH_LAYOUT,
    ASSOC_ADD,
    ASSOC_LINEAR,
    ASSOC_LINEAR_BIAS,
    ASSOC_LINEAR_BIAS_REV,
    ASSOC_LINEAR_REV,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    ASSOC_MUL,
    BATCH_NORM_EXPAND,
    BATCH_NORM_EXPAND_NOBIAS,
    BATCH_NORM_EXPAND_PLAIN,
    BATCH_NORM_RULES,
    CATEGORICAL_RULES,
    CHUNK_SINGLE,
    COMM_ADD,
    COMM_MUL,
    CONV_BN_FOLD,
    CONV_FOLD_RULES,
    DISTRIBUTE_MUL,
    DISTRIBUTE_MUL_ADD,
    DIV_ADD_FACTOR,
    DIV_SQRT_TO_RSQRT,
    DOUBLE_NEG,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    GLU_FOLD,
    GQA_ABSORB,
    ID_ADD,
    ID_MUL,
    LINEAR_CHANNEL_SCALE,
    LINEAR_CHANNEL_SCALE_REV,
    LINEAR_CHANNEL_TO_ROW_SCALE,
    LINEAR_ROW_SCALE,
    LINEAR_ROW_SCALE_REV,
    MATMUL_DIV_DISTRIBUTE,
    MINTED_RULES,
    MUL_DIV_SWAP,
    MUL_RESHAPE_INERT_L,
    MUL_SQUARE,
    MUL_UNSQ_NAT,
    MUL_UNSQ_PAD_L,
    MUL_UNSQ_PAD_R,
    NATURALITY_SCALAR,
    NATURALITY_SCALAR_REV,
    PARALLEL_MUL_FUSE,
    POINTWISE_FACTOR_RULES,
    POW_TO_RSQRT,
    POW_TO_SQUARE,
    QKV_FUSE,
    QKV_FUSE_ASYM,
    RECIP_SQRT_TO_RSQRT,
    RIGHT_DISTRIBUTE,
    RIGHT_FACTOR,
    RIGHT_FACTOR_LINEAR,
    RMS_NORM_FOLD,
    RMS_NORM_FOLD_NOGAIN,
    SDPA_EXPAND_RULES,
    SDPA_FOLD_DIV_NOMASK,
    SDPA_FOLD_MUL_NOMASK,
    SDPA_FOLD_NOMASK,
    SDPA_FOLD_RULES,
    SELECT_MUL,
    SILU_EXPAND,
    SILU_FOLD,
    SILU_MUL_FORM,
    SIMPLIFICATION_RULES,
    SOFTMAX_FOLD,
    SOFTSIGN_FOLD,
    SQUARE_EXPAND,
    SQUARE_TO_POW,
    SUB_TO_ADD,
    SUB_UNSQ_PAD_L,
    SWIGLU_FUSE,
    TRANSPOSE_NOOP,
    WEIGHT_DISTRIBUTE,
    WEIGHT_DISTRIBUTE_LINEAR,
    WEIGHT_FACTOR,
    WEIGHT_FACTOR_LINEAR,
    _check_bn_eval,
    _check_glu_fold,
    _check_glu_shaped,
    _check_gqa_absorb,
    _check_linear_bias_compose,
    _check_repeat_chain,
    _check_rms_consts,
    _check_rms_fold,
    _check_rms_fold_nogain,
    _check_score_transpose,
    _check_sdpa_base,
    _check_sdpa_mf,
    _check_sdpa_mf_scaled,
    _check_sdpa_scaled,
    _check_softmax_dim,
    _const_val,
    _derive_bn,
    _derive_rms_norm,
    _derive_scale_div,
    _derive_scale_mul,
    _derive_scale_one,
    _derive_split_sizes,
    _head,
    _head_v,
    _make_sdpa_fold_rules,
    _scale_of,
    all_rules,
)

__all__ = [
    "AFFD_ASSOC",
    "AFFD_ASSOC_REV",
    "AFFD_COMPOSE_UNFOLD",
    "AFFD_LIFT",
    "AFFD_LIFT_POST",
    "AFFD_LIFT_POST_SWAP",
    "AFFD_LIFT_STEP",
    "AFFD_LIFT_STEP_POST",
    "AFFD_LIFT_STEP_POST_SWAP",
    "AFFD_LIFT_STEP_SWAP",
    "AFFD_LIFT_SWAP",
    "AFFD_LIFT_UNIT",
    "AFFD_LIFT_UNIT_POST",
    "AFFD_LIFT_UNIT_STEP",
    "AFFD_LIFT_UNIT_STEP_POST",
    "AFFD_UNLIFT",
    "AFF_ASSOC",
    "AFF_ASSOC_REV",
    "AFF_COMPOSE_UNFOLD",
    "AFF_LIFT",
    "AFF_LIFT_STEP",
    "AFF_UNLIFT",
    "ALL_RULES",
    "ALL_RULES_WITH_LAYOUT",
    "ASSOC_ADD",
    "ASSOC_LINEAR",
    "ASSOC_LINEAR_BIAS",
    "ASSOC_LINEAR_BIAS_REV",
    "ASSOC_LINEAR_REV",
    "ASSOC_MATMUL",
    "ASSOC_MATMUL_REV",
    "ASSOC_MUL",
    "ATTENTION_RULES",
    "BATCH_NORM_EXPAND",
    "BATCH_NORM_EXPAND_NOBIAS",
    "BATCH_NORM_EXPAND_PLAIN",
    "BATCH_NORM_RULES",
    "CARRIERS",
    "CARRIER_SEARCH",
    "CATEGORICAL",
    "CATEGORICAL_RULES",
    "CHUNK_SINGLE",
    "COMM_ADD",
    "COMM_MUL",
    "CONV_BN_FOLD",
    "CONV_FOLD_RULES",
    "DEFAULT",
    "DISTRIBUTE_MUL",
    "DISTRIBUTE_MUL_ADD",
    "DIV_ADD_FACTOR",
    "DIV_SQRT_TO_RSQRT",
    "DOUBLE_NEG",
    "EARLY",
    "FACTOR_MUL",
    "FACTOR_MUL_ADD",
    "FULL",
    "FUSION",
    "GLU_FOLD",
    "GQA_ABSORB",
    "ID_ADD",
    "ID_MUL",
    "LATE",
    "LAYOUT_RULES",
    "LINEAR_CHANNEL_SCALE",
    "LINEAR_CHANNEL_SCALE_REV",
    "LINEAR_CHANNEL_TO_ROW_SCALE",
    "LINEAR_FROM_MM_T",
    "LINEAR_FROM_MM_T_BARE",
    "LINEAR_MM_ABSORB",
    "LINEAR_MM_ABSORB_BIAS",
    "LINEAR_MM_ABSORB_BIAS_REV",
    "LINEAR_MM_ABSORB_REV",
    "LINEAR_OUT_SCALE",
    "LINEAR_OUT_SCALE_REV",
    "LINEAR_ROW_SCALE",
    "LINEAR_ROW_SCALE_REV",
    "LINEAR_TO_MM_T",
    "MACHINE_LAWS",
    "MACHINE_SHIPPED",
    "MACHINE_STORED",
    "MATMUL_DIV_DISTRIBUTE",
    "MINTED_RULES",
    "MUL_DIV_SWAP",
    "MUL_RESHAPE_INERT_L",
    "MUL_SQUARE",
    "MUL_UNSQ_NAT",
    "MUL_UNSQ_PAD_L",
    "MUL_UNSQ_PAD_R",
    "NATURALITY_SCALAR",
    "NATURALITY_SCALAR_LEFT",
    "NATURALITY_SCALAR_LEFT_REV",
    "NATURALITY_SCALAR_REV",
    "NORMAL",
    "PARALLEL_MUL_FUSE",
    "POINTWISE_FACTOR_RULES",
    "POW_TO_RSQRT",
    "POW_TO_SQUARE",
    "PRESETS",
    "QKV_FUSE",
    "QKV_FUSE_ASYM",
    "RECIP_SQRT_TO_RSQRT",
    "RIGHT_DISTRIBUTE",
    "RIGHT_FACTOR",
    "RIGHT_FACTOR_LINEAR",
    "RMS_NORM_FOLD",
    "RMS_NORM_FOLD_NOGAIN",
    "ROPE_CAT_COMPOSE",
    "ROPE_CAT_SCALE_IN",
    "ROPE_CAT_SCALE_OUT",
    "ROPE_RH_SCALE_IN",
    "ROPE_RH_SCALE_OUT",
    "SCAN_DIAG_LAWS",
    "SCAN_LAWS",
    "SDPA_EXPAND_RULES",
    "SDPA_FOLD_DIV_NOMASK",
    "SDPA_FOLD_MUL_NOMASK",
    "SDPA_FOLD_NOMASK",
    "SDPA_FOLD_RULES",
    "SELECT_MUL",
    "SILU_EXPAND",
    "SILU_FOLD",
    "SILU_MUL_FORM",
    "SIMPLIFICATION",
    "SIMPLIFICATION_RULES",
    "SOFTMAX_FOLD",
    "SOFTSIGN_FOLD",
    "SQUARE_EXPAND",
    "SQUARE_TO_POW",
    "SUB_TO_ADD",
    "SUB_UNSQ_PAD_L",
    "SWIGLU_FUSE",
    "SYMMETRY",
    "TRANSPOSE_NOOP",
    "WEIGHT_DISTRIBUTE",
    "WEIGHT_DISTRIBUTE_LINEAR",
    "WEIGHT_FACTOR",
    "WEIGHT_FACTOR_LINEAR",
    "WITH_LAYOUT",
    "R",
    "RuleSet",
    "all_rules",
    "headshare_keys_hold",
    "law_provenance",
    "offer_low_rank_factors",
    "offer_weight_specials",
    "pair_shared_input_convs",
    "pair_shared_input_linears",
    "preset",
    "provenance",
    "share_duplicate_attention_heads",
    "share_duplicate_param_slices",
    "share_duplicate_params",
    "tags",
]
