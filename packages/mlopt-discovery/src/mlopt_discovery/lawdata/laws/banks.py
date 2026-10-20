"""Enumeration banks — the pools the search draws content from.

Shape banks an oracle enumerates (``mlopt_discovery.oracle``), the
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
    "JOINT_SHAPES",
    "JOINT_VIEW_SHAPES",
    "LINEAR_HELD_SHAPES",
    "LINEAR_TRAIN_SHAPES",
    "PURPOSE_BUILT",
    "SHAPE_POOL",
    "TUPLE_SOURCES",
    "TUPLE_SOURCE_SHAPE",
    "VIEWED_SHAPES",
]

# ---------------------------------------------------------------------------
#  Oracle enumeration banks (mlopt_discovery.oracle)
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
    "shifts": "int",
    "diagonal": "int",
    "correction": "int",
    "upscale_factor": "int",
    "downscale_factor": "int",
    "stride": "int",
    "padding": "int",
    "m": "int",
    # strictly positive ints — a 0 draw can only mint an ill-typed op
    # (``dilation=0`` / ``groups=0`` raise in every torch op that
    # carries them, so the name default is honest; ``stride`` keeps
    # ``int`` because ``as_strided`` does admit 0).
    "dilation": "pos-int",
    "groups": "pos-int",
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
    # conv2d's stride is strictly positive — a 0 draw can only mint
    # an ill-typed conv (measured: every accepted site of the conv
    # anchors in the 360-window died on ``non-positive stride``).
    # Scoped per-op like the rest of the table; a different op whose
    # ``stride`` admits 0 (``as_strided``) keeps the name default.
    ("conv2d", "stride"): "pos-int",
}

#: Correlated operand-shape rows for ops whose operands form a shape
#: contract — the *joint* corner the independent leaf banks cannot
#: mint.  Keyed ``(op, arity)``; each row is one concrete joint shape
#: assignment in operand order (a row shorter than the node's args
#: binds a leading prefix; non-metavar arg positions carry their
#: expected shape but bind nothing).  The enumeration binds the
#: metavariable args of a tabulated node *jointly* from these rows —
#: a guard whose accepted region is a relation across operands
#: (``conv2d``'s ``x[1] == W[1]·groups``, ``batch_norm``'s per-channel
#: stat vectors all equalling ``x[1]``) reaches its corner in one
#: draw instead of never: independent draws need the coincidence of
#: every relation at once, and measured over the conv+BN composite
#: that corner lay outside the domain entirely (``0 accepted / 2000
#: declined`` — ``project/retros/binding-env-cap.md``).
#:
#: Only ops a measured starvation named carry entries — a covered
#: metavariable's domain IS the row set, so a row for an op whose
#: sweeps already measure shrinks a working domain for nothing (the
#: ``CONST_DOMAIN`` discipline).  ``conv2d`` / ``batch_norm`` are the
#: opt-in families whose guarded sweeps starved; a new entry is a
#: data row, never an engine edit.
JOINT_SHAPES: dict[
    tuple[str, int], tuple[tuple[tuple[int, ...], ...], ...]
] = {
    # x (N,C,H,W), W (O,C,kh,kw), b (O,) — the biased conv site.
    ("conv2d", 3): (
        ((2, 4, 8, 8), (6, 4, 3, 3), (6,)),  # groups=1: x[1] == W[1]
        ((2, 4, 8, 8), (6, 2, 3, 3), (6,)),  # groups=2: x[1] == W[1]·2
        ((1, 3, 6, 6), (5, 3, 3, 3), (5,)),
    ),
    # The bias-free export arity.
    ("conv2d", 2): (
        ((2, 4, 8, 8), (6, 4, 3, 3)),
        ((2, 4, 8, 8), (6, 2, 3, 3)),
        ((1, 3, 6, 6), (5, 3, 3, 3)),
    ),
    # batch_norm(x, w, b, rm, rv): x (N,C,·) and the four per-channel
    # vectors — every row's vector extent IS x's channel axis.
    ("batch_norm", 5): (
        ((2, 6, 8, 8), (6,), (6,), (6,), (6,)),
        ((2, 4, 8, 8), (4,), (4,), (4,), (4,)),
        ((2, 5, 6, 6), (5,), (5,), (5,), (5,)),
    ),
    # batch_norm(x, w, rm, rv) — the bias-free export arity.
    ("batch_norm", 4): (
        ((2, 6, 8, 8), (6,), (6,), (6,)),
        ((2, 4, 8, 8), (4,), (4,), (4,)),
        ((2, 5, 6, 6), (5,), (5,), (5,)),
    ),
    # batch_norm(x, rm, rv) — the affine-free arity.
    ("batch_norm", 3): (
        ((2, 6, 8, 8), (6,), (6,)),
        ((2, 4, 8, 8), (4,), (4,), (4,)),
        ((2, 5, 6, 6), (5,), (5,), (5,)),
    ),
}

#: Broadcast-view shapes appended to the leaf banks when a pattern
#: carries a joint-shape node — the *channel view* a structured site's
#: sibling operand takes (``sv ~ (1, C, 1, 1)`` multiplying a conv
#: output, the per-channel scale the conv+BN fold rides).  The
#: independent bank mints rank-≤3 shapes and broadcast pads, never a
#: same-rank singleton view; the extents here span the channel counts
#: :data:`JOINT_SHAPES` rows produce, so a row and a view draw agree
#: at a low index-sum.  Scoped to joint patterns (via the
#: ``extra_viewed`` channel, exactly like ``CHAIN_VIEWED_SHAPES``) so
#: joint-free enumerations are byte-identical.
JOINT_VIEW_SHAPES: tuple = (
    (1, 4, 1, 1),
    (1, 5, 1, 1),
    (1, 6, 1, 1),
    (1, 4, 1),
    (1, 1, 4),
)

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
#  Instantiation defaults (mlopt_discovery.verifier.generic_instance)
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
#  Gap-synthesis pools (mlopt_discovery.gap_gen)
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
#  The algebraic-grammar alphabet (mlopt_discovery.grammar)
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
#  Training families (mlopt_discovery.families)
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
#: (``mlopt_discovery.coherence``'s hand-picked cluster).
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
#  The intake ledger (mlopt_discovery.intake)
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
