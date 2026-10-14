"""Enumeration banks — the pools the search draws content from.

Shape banks an oracle enumerates (``catopt_discovery.oracle``), the
per-op constant domain a guarded sweep mints, the attribute-kind
tables that name and type the sweep's attrs, the instantiation
defaults ``verifier.generic_instance`` reads, the gap-synthesis pools
``gap_gen`` samples, the algebraic-grammar alphabet, the training
family shapes, the coherence-probe neighbourhoods, and the intake
corpus-circularity ledger.  Pure data; consumers bind projections or
the table itself.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ATTR_KINDS",
    "ATTR_KIND_OVERRIDES",
    "BASE_SHAPES",
    "CHAIN_VIEWED_SHAPES",
    "COHERENCE_CLUSTER",
    "COHERENCE_CORPUS_MODELS",
    "CONST_DOMAIN",
    "FALLBACK_AXES",
    "FALLBACK_SHAPES",
    "FAMILIES",
    "FREE_SENTINELS",
    "GRAMMAR_BINARY_OPS",
    "GRAMMAR_LITERALS",
    "GRAMMAR_UNARY_OPS",
    "INSTANCE_ATTR_DEFAULTS",
    "INSTANCE_LEAF_SHAPES",
    "INSTANCE_SCALAR_MVARS",
    "LINEAR_HELD_SHAPES",
    "LINEAR_TRAIN_SHAPES",
    "PURPOSE_BUILT",
    "SHAPE_POOL",
    "TUPLE_SOURCES",
    "TUPLE_SOURCE_SHAPE",
    "VIEWED_SHAPES",
]

# ---------------------------------------------------------------------------
#  Oracle enumeration banks (catopt_discovery.oracle)
# ---------------------------------------------------------------------------

#: Shapes offered to a metavariable that sits under a view op.
VIEWED_SHAPES: tuple = (
    (4,),
    (2, 3),
    (3, 4),
    (2, 2),
    (2, 3, 4),
    (2, 3, 1),
)

#: Extra viewed-bank shapes for patterns that contain an
#: operand-*chained* view (``reshape(expand(unsqueeze(...)))``) — the
#: rank-4 ``(b, t, h_kv, d)`` / ``(b, t, h, d)`` pair.  The chain's
#: intermediate tensors are rank-4+, and ``gqa_absorb_repeat``'s
#: corner (``repeat-heads``: ``q[-2] == k[-2] * r``) needs the head-dim
#: product only a rank-4 pair supplies.  Scoped to chained patterns so
#: chain-free enumerations are byte-identical.
CHAIN_VIEWED_SHAPES: tuple = (
    (2, 3, 2, 4),
    (2, 3, 4, 4),
)

#: Sentinel shapes the free operand's derived bank always carries —
#: the scalar corner and two generic mismatches.
FREE_SENTINELS: tuple = ((), (4,), (7, 7))

#: Axes offered when the operand's shape is not visible (a free
#: metavariable bound later, or a rank-0 operand): the common last-
#: and first-axis spellings.
FALLBACK_AXES: tuple = (-2, -1, 0, 1)

#: Trailing tuples offered for a shape-typed attr whose operand
#: shape is unknown — plausible normalized-shape spellings.
FALLBACK_SHAPES: tuple = ((1,), (2,), (4,), (2, 4))

#: Per-op literal-constant domain — the values a metavariable's
#: *parent op* admits, keyed by the op name.  The leaf bank's generic
#: scalar corner mints a single ``Const(0.5)``; a shipped guard can
#: demand a specific literal that corner never reaches, so the
#: guarded-region sweep can never construct the site — the
#: *domain-gapped* class of ``project/retros/cap-policy.md``.  Each
#: entry is a measured widening:
#:
#: * ``pow`` — the exponent (``2`` for the square/RMSNorm spelling,
#:   ``-0.5`` for the reciprocal-root; ``0.5`` / ``1`` round it out).
#: * ``masked_fill`` — the softmax mask sentinel; ``-inf`` is the one
#:   that clears the strict ``const-cmp F < -1e30`` (``-1e30`` itself
#:   does not: the comparison is strict), the other two are the
#:   finite analogues.
#: * ``div`` — the scalar identities (``1`` for the numerator of the
#:   ``1 / sqrt(x)`` spelling, ``0`` for the additive-identity probe).
#:
#: Only the ops a shipped guard actually constrains carry an entry;
#: ``mul`` / ``add`` identities were measured and left out (the bank
#: feeds the enumeration — a widening with no rescue to show for it
#: perturbs the capped order).  ``project/retros/value-bank.md``.
CONST_DOMAIN: dict[str, tuple[int | float, ...]] = {
    "pow": (2, -0.5, 0.5, 1),
    "masked_fill": (-float("inf"), -1e30, 1e9),
    "div": (1, 0),
}

#: Canonical attr names -> the value kind the sweep can enumerate.
#: ``ATTR_SCHEMA`` *names* every positional attr; this table *types*
#: the names.  ``dim`` defaults to a plain axis — the reduction ops
#: and the normalized-shape spellings override below.
ATTR_KINDS: dict[str, str] = {
    # axes — valid values are ``-rank..rank-1`` of the operand
    "dim": "axis",
    "dim0": "axis",
    "dim1": "axis",
    "start_dim": "axis",
    "end_dim": "axis",
    "source": "axis",
    "destination": "axis",
    # small ints — indices, counts, bounds, kernel sizes
    "index": "int",
    "start": "int",
    "end": "int",
    "step": "int",
    "length": "int",
    "chunks": "int",
    "k": "int",
    "sections": "int",
    "num_groups": "int",
    "num_classes": "int",
    "num_layers": "int",
    "groups": "int",
    "shifts": "int",
    "diagonal": "int",
    "correction": "int",
    "upscale_factor": "int",
    "downscale_factor": "int",
    "stride": "int",
    "padding": "int",
    "dilation": "int",
    "m": "int",
    # float scalars — scales, epsilons, rates
    "scale": "float",
    "eps": "float",
    "momentum": "float",
    "p": "float",
    "dropout_p": "float",
    "dropout": "float",
    "rtol": "float",
    "atol": "float",
    "alpha": "float",
    "beta": "float",
    "threshold": "float",
    "min": "float",
    "max": "float",
    "input_scale": "float",
    "value": "float",
    "ord": "float",
    # boolean flags
    "keepdim": "bool",
    "is_causal": "bool",
    "enable_gqa": "bool",
    "train": "bool",
    "training": "bool",
    "largest": "bool",
    "sorted": "bool",
    "descending": "bool",
    "accumulate": "bool",
    "equal_nan": "bool",
    "cudnn_enabled": "bool",
    "has_biases": "bool",
    "bidirectional": "bool",
    "batch_first": "bool",
    "use_input_stats": "bool",
    "right": "bool",
    "out_int32": "bool",
    # shape-typed tuples — normalized_shape and friends
    "shape": "shape",
    "sizes": "shape",
    "size": "shape",
    "pad": "shape",
    # int-or-tuple axis lists (roll's ``dims``)
    "dims": "red-dims",
    # honestly unenumerable — string payloads, not scalars
    "equation": "str",
    "mode": "str",
    "reduce": "str",
    "layout": "str",
}

#: ``(op, canonical-attr)`` pairs whose value kind differs from the
#: name default — the attr names are honest but not typed, and these
#: are the measured exceptions.
ATTR_KIND_OVERRIDES: dict[tuple[str, str], str] = {
    # the norms' ``dim`` attr is aten's normalized_shape list, not an
    # axis (``rms_norm(x, ns, w, eps)`` / ``layer_norm``'s arg1).
    ("layer_norm", "dim"): "shape",
    ("rms_norm", "dim"): "shape",
    # ``eye(n)`` / ``eye.m(n, m)`` — sizes, not axes.
    ("eye", "dim"): "int",
    # unfold's ``size`` is a kernel extent; upsample's ``size`` stays
    # a shape tuple.
    ("unfold", "size"): "int",
}

#: Shared operand shape for the tuple-producing sources below.
TUPLE_SOURCE_SHAPE: tuple = (2, 4)

#: Tuple-producing ops for a metavariable under ``getitem`` — the
#: corpus's real ``getitem`` matches pick elements out of ``topk`` /
#: ``var_mean`` / ``cummax`` (a bare ``Var`` only covers the dim-0
#: tensor index).  Each entry is ``(op, attrs)`` built over a
#: ``TUPLE_SOURCE_SHAPE`` operand.
TUPLE_SOURCES: tuple = (
    ("topk", {"k": 2}),
    ("var_mean", {"dim": (-1,), "correction": 0, "keepdim": True}),
    ("cummax", {"dim": 0}),
)

# ---------------------------------------------------------------------------
#  Instantiation defaults (catopt_discovery.verifier.generic_instance)
# ---------------------------------------------------------------------------

#: Leaf shapes for the SDPA-fold family's metavariables — the only
#: rules the bench registry does not cover.  ``(B,H,T,D)`` scores feed
#: ``(B,H,T,T)`` masks and a ``(B,H,T,D)`` value.
INSTANCE_LEAF_SHAPES: dict[str, tuple[int, ...]] = {
    "Q": (2, 4, 8, 4),
    "K": (2, 4, 8, 4),
    "V": (2, 4, 8, 4),
    "M": (2, 4, 8, 8),
    "MK": (2, 4, 8, 8),
}

#: Attr-metavariable defaults for the SDPA-fold family.
INSTANCE_ATTR_DEFAULTS: dict[str, Any] = {
    "TD1": -2,
    "TD2": -1,
    "SD": -1,
    "DP": 0.5,
    "DT": True,
}

#: Scalar metavariable defaults — the ``Const`` a name mints.
INSTANCE_SCALAR_MVARS: dict[str, int | float] = {
    "S": 0.5,
    "F": float("-inf"),
}

# ---------------------------------------------------------------------------
#  Gap-synthesis pools (catopt_discovery.gap_gen)
# ---------------------------------------------------------------------------

#: Leaf-shape pool for metavariable instantiation.  Uniform draws
#: cover the pointwise laws (all metavars one shape); the broadcastable
#: entries (``(1,)``, ``(4,1)``, ``()``) give mixed-view candidates a
#: non-view operand that composes on BOTH sides of the rewrite.
SHAPE_POOL: tuple = (
    (4, 16),
    (16, 16),
    (2, 8, 16),
    (8, 8),
    (16,),
    (4, 8),
    (4, 1),
    (1, 16),
    (1,),
    (),
    (2, 4, 4),
)

#: Uniform base shapes tried first, in order, before random draws.
BASE_SHAPES: tuple = ((4, 16), (16, 16), (2, 8, 16))

# ---------------------------------------------------------------------------
#  The algebraic-grammar alphabet (catopt_discovery.grammar)
# ---------------------------------------------------------------------------

#: Binary ops the search may place at a two-child node.
GRAMMAR_BINARY_OPS: tuple = (
    "add",
    "mul",
    "sub",
    "div",
    "pow",
    "matmul",
)

#: Unary ops the search may wrap a leaf in (and swap a unary node to).
GRAMMAR_UNARY_OPS: tuple = (
    "neg",
    "exp",
    "square",
    "sqrt",
    "rsqrt",
    "sigmoid",
    "silu",
    "tanh",
)

#: Literal constants the search may substitute for a leaf.  Integer
#: spelling, matching the hand-written grammar (``Const(0)``, not
#: ``Const(0.0)``) — the e-graph keys leaves by ``repr``, so a float
#: spelling would make a law miss its own grammar rule and look novel.
GRAMMAR_LITERALS: tuple = (0, 1, 2)

# ---------------------------------------------------------------------------
#  Training families (catopt_discovery.families)
# ---------------------------------------------------------------------------

#: The family names, in report order.
FAMILIES: tuple[str, ...] = ("chain", "dup", "linear")

#: ``(d_in, d_hidden, d_out)`` shapes for the linear family — training.
LINEAR_TRAIN_SHAPES: tuple = (
    (8, 16, 8),
    (8, 24, 8),
    (6, 12, 6),
    (10, 20, 10),
)

#: Held-out linear shapes (widths unseen in training).
LINEAR_HELD_SHAPES: tuple = (
    (7, 18, 7),
    (9, 21, 9),
    (12, 26, 12),
    (11, 22, 11),
)

# ---------------------------------------------------------------------------
#  Coherence probe neighbourhoods
# ---------------------------------------------------------------------------

#: The select_mul / softmax_fold / transpose_push_mul neighbourhood:
#: the folds and the layout/pointwise naturality laws they touch
#: (``catopt_discovery.coherence``'s hand-picked cluster).
COHERENCE_CLUSTER: tuple = (
    "select_mul",
    "softmax_fold",
    "sdpa_fold_add",
    "sdpa_fold_addmul",
    "comm_mul",
    "comm_add",
    "transpose_push_mul",
    "transpose_pull_mul",
    "transpose_push_add",
    "transpose_pull_add",
    "transpose_push_exp",
    "transpose_transpose_dd",
    "transpose_matmul",
)

#: The coherence2 reach corpus — small real exports chosen to cover
#: the interesting law neighbourhoods: silu/swiglu, matmul chains,
#: linear folds, softmax/select, residual adds.
COHERENCE_CORPUS_MODELS: tuple = (
    "SwiGLU",
    "ResidualMLP",
    "GatedResidualBlock",
    "ManualSoftmaxAttention",
    "MatrixChain",
    "ParallelLinear",
    "NormLinear",
)

# ---------------------------------------------------------------------------
#  The intake ledger (catopt_discovery.intake)
# ---------------------------------------------------------------------------

#: The corpus-circularity ledger — the workload names written to spell
#: one specific *pattern* (an auto-cond guard's region), not to stand
#: for a real program.  Round 3's view-identity elementwise spellings
#: exist to fill the guards' empty regions ("Round 3 supplies the
#: no-op spellings directly"); every round-4 workload spells one
#: wrap / mirror / shared-factor / grammar / scalar-corner pattern.
#: ``intake.candidates`` reads it to set ``Workload.purpose_built``;
#: ``real_workloads`` filters on the flag.
PURPOSE_BUILT: frozenset[str] = frozenset(
    {
        # --- round 3 — view-identity elementwise spellings ----------
        "BroadcastPadLeft",
        "BroadcastPadRight",
        "BroadcastPadSub",
        "FullSliceScale",
        "NoopTransposeScale",
        "NoopTransposeAdd",
        "InertReshapeScale",
        "SingleChunkScale",
        "SquaredDistance",
        # --- round 4 — unsqueeze-wrap (gated broadcast) -------------
        "ChannelGateBroadcast",
        "LiftedScalarScale",
        "HeadGateBroadcast",
        "LiftedScalarScaleRight",
        "LiftedScalarCenter",
        "ContrastiveCenter",
        "PairwiseSubLift",
        # --- round 4 — chunked-projection mirrors -------------------
        "ScaledChunkProjection",
        "SingleChunkGate",
        "ChunkHalfScale",
        # --- round 4 — transposed-add mirrors -----------------------
        "NoopTransposeResidual",
        "ScalarTransposeBias",
        "Rank3TransposeAdd",
        # --- round 4 — select naturality ----------------------------
        "SelectGateSum",
        "SelectGateDiff",
        # --- round 4 — shared-factor algebra ------------------------
        "SharedFactorMixture",
        "SharedFactorContrast",
        "SharedFactorMixtureRight",
        "SharedFactorContrastRight",
        "QuadraticFeature",
        # --- round 4 — reshape / neg / exp / square grammar ---------
        "DoubleReshapeHead",
        "NegatedSum",
        "NegDistributeHead",
        "SubNegBias",
        "NegatedScale",
        "ExpProductHead",
        "ExpSumHead",
        "SquareNegHead",
        "SquareMulHead",
        "SigmoidNegGate",
        "PowOneHead",
        "SubAddFactorHead",
        "DivAddHead",
        # --- round 4 — scalar-corner annihilators -------------------
        "ScalarAnnihilator",
        "ScalarSelfCancel",
        "ScalarSelfRatio",
        "ScalarInverseSum",
    }
)
