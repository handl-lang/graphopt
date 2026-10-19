"""The candidate registry — ``Workload`` thunks so import is free.

``candidates()`` returns the full intake inventory — the torch-native
``nn.*`` modules and the compound assemblies from
:mod:`mlopt_discovery.intake.models` — as ``Workload`` thunks (a
fresh module + example input per call, so importing the registry
constructs nothing).  ``real_workloads`` is the honest "real-only"
filter over the corpus-circularity ledger.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlopt_discovery import lawdata

from .models import (
    _GRN,
    _AddcmulHead,
    _ALiBiAttention,
    _AllReduce,
    _AngleLog,
    _ArgSort,
    _AxialAttention,
    _BitwiseHead,
    _BooleanIndex,
    _Bottleneck,
    _BroadcastPadLeft,
    _BroadcastPadRight,
    _BroadcastPadSub,
    _BroadcastTensors,
    _BroadcastTo,
    _Bucketize,
    _CDist,
    _CeilFloor,
    _CenterNorm,
    _ChannelGateBroadcast,
    _ChannelShuffle,
    _ChunkHalfScale,
    _CircularPad,
    _ClampMaxHead,
    _ComparisonHead,
    _ConformerBlock,
    _ContrastiveCenter,
    _ConvNeXtBlock,
    _CoshAcosh,
    _CountHead,
    _CrossAttentionBlock,
    _CumMaxMin,
    _CumProdGate,
    _CumsumScan,
    _CumulativeGateRNN,
    _DeepNorm,
    _DepthwiseConv2d,
    _DigammaLn,
    _DivAddHead,
    _DoubleReshapeHead,
    _EinsumAttention,
    _ErfFeatures,
    _ExpandAs,
    _ExpProductHead,
    _ExpSumHead,
    _EyeInit,
    _FakeQuant,
    _FillSelect,
    _FillWhole,
    _FlipConv,
    _FullSliceScale,
    _GeluErf,
    _GeoMeanPool,
    _GQAAttention,
    _HeadGateBroadcast,
    _Heaviside,
    _HyperbolicHead,
    _IndexMoE,
    _IndexPut,
    _InertReshapeScale,
    _IsNegPos,
    _KthMode,
    _LerpMix,
    _LiftedScalarCenter,
    _LiftedScalarScale,
    _LiftedScalarScaleRight,
    _LinearAttention,
    _Log10Expm1,
    _LogBase,
    _LogicCombo,
    _LogSoftmaxHead,
    _LossHead,
    _ManualAttention,
    _ManualNLL,
    _ManualRMSNorm,
    _MaskedFillInplace,
    _MaskedFillTensor,
    _MaskedFillView,
    _MaskLogic,
    _MatrixInverse,
    _MBConvBlock,
    _MedianPool,
    _MinFmax,
    _MinPair,
    _MoEGate,
    _MoveDimStack,
    _MQAAttention,
    _MultiLatentAttention,
    _NanGuard,
    _NanStats,
    _NarrowHead,
    _NegatedScale,
    _NegatedSum,
    _NegDistributeHead,
    _NewCreators,
    _NoopTransposeAdd,
    _NoopTransposeResidual,
    _NoopTransposeScale,
    _NormalizeHead,
    _OneHotEmbed,
    _OuterPositional,
    _PairwiseSubLift,
    _PatchMerging,
    _PostNormBlock,
    _PowOneHead,
    _PreNormBlock,
    _ProdAminPool,
    _QKNorm,
    _QuadraticFeature,
    _Rank3TransposeAdd,
    _RegluMlp,
    _RelativePositionBias,
    _Relu6Head,
    _RemainderHead,
    _RepeatHead,
    _RepeatInterleave,
    _RoPEAttention,
    _RotaryEmbedding,
    _RunningNorm,
    _SanitizeHead,
    _ScalarAnnihilator,
    _ScalarInverseSum,
    _ScalarRsub,
    _ScalarSelfCancel,
    _ScalarSelfRatio,
    _ScalarTransposeBias,
    _ScaledChunkProjection,
    _ScaleNorm,
    _ScatterAdd,
    _ScatterReduce,
    _SdpaBiasParam,
    _SelectGateDiff,
    _SelectGateSum,
    _SelectiveScan,
    _SelectWrite,
    _SharedExpertMoE,
    _SharedFactorContrast,
    _SharedFactorContrastRight,
    _SharedFactorMixture,
    _SharedFactorMixtureRight,
    _ShiftedWindow,
    _SigmoidNegGate,
    _SignTruncFrac,
    _SincKernel,
    _SingleChunkGate,
    _SingleChunkScale,
    _SliceFill,
    _SlidingUnfold,
    _SlidingWindowMask,
    _SoftsignHead,
    _SortSelect,
    _SpatialTransformer,
    _SqrtHead,
    _SquaredDistance,
    _SquareMulHead,
    _SquareNegHead,
    _StdNorm,
    _SubAddFactorHead,
    _SubNegBias,
    _SwiGLU,
    _SwiGLUBlock,
    _TakeAlong,
    _TensorSplit,
    _TinyCNN,
    _TraceNorm,
    _TracePenalty,
    _TrigHead,
    _TriuAttention,
    _TypeAs,
    _UnbindHead,
    _VarNorm,
    _VarStdHead,
    _ViTPatchBlock,
    _XlogyKL,
    _ZeroInit,
)

# ---------------------------------------------------------------------------
#  The candidate registry — thunks so import constructs nothing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    """One intake candidate: a name and a ``(model, input)`` thunk.

    ``kind`` is the registry class (``"torch-native"`` / ``"compound"``);
    ``purpose_built`` is the provenance flag — true for a micro-module
    written to spell one specific pattern (a guard region) rather than
    to represent a real network.  :data:`_PURPOSE_BUILT` is the ledger;
    :func:`real_workloads` is the honest filter over it.
    """

    name: str
    build: Callable[[], tuple[nn.Module, Any]]
    kind: str = "torch-native"
    purpose_built: bool = False


#: The corpus-circularity ledger — the workload names written to spell
#: one specific *pattern* (an auto-cond guard's region), not to stand
#: for a real program.  Everything else — the torch-native ``nn.*``
#: modules and the realistic compound assemblies — is what an honest
#: "real-only" measurement keeps.  The table lives in
#: :mod:`mlopt_discovery.lawdata` (:data:`PURPOSE_BUILT`); this is
#: the same object bound to the ledger's own name — :func:`candidates`
#: reads it to set ``Workload.purpose_built``, and
#: :func:`real_workloads` filters on the flag.
_PURPOSE_BUILT: frozenset[str] = lawdata.PURPOSE_BUILT


def _r(*shape: int) -> torch.Tensor:
    """Return a fresh fp64 CPU example input.

    Called with no shape it yields a rank-0 (scalar) tensor — the
    binding the scalar-corner guards (``rank(A) == 0``) need.
    """
    return torch.randn(tuple(shape), dtype=torch.float64)


def _enc_layer(d: int) -> nn.TransformerEncoderLayer:
    """Return one batch-first encoder layer at the intake size."""
    return nn.TransformerEncoderLayer(
        d, 4, 2 * d, batch_first=True, dropout=0.0
    )


def candidates() -> list[Workload]:
    """Return the intake registry — torch natives + compound models.

    Thunks build a fresh module and example input per call, so the
    registry is cheap to import and every ingestion sees the same
    seeded weights the report records.
    """
    d = 16
    reg = [
        # --- torch-native: attention / transformer family -----------
        Workload(
            "nn.MultiheadAttention",
            lambda: (
                nn.MultiheadAttention(d, 4, batch_first=True),
                (_r(2, 8, d), _r(2, 8, d), _r(2, 8, d)),
            ),
        ),
        Workload(
            "nn.TransformerEncoderLayer",
            lambda: (_enc_layer(d), _r(2, 8, d)),
        ),
        Workload(
            "nn.TransformerEncoder(d=3)",
            lambda: (
                nn.TransformerEncoder(_enc_layer(d), 3),
                _r(2, 8, d),
            ),
        ),
        Workload(
            "nn.TransformerDecoderLayer",
            lambda: (
                nn.TransformerDecoderLayer(
                    d, 4, 2 * d, batch_first=True, dropout=0.0
                ),
                (_r(2, 8, d), _r(2, 8, d)),
            ),
        ),
        # --- torch-native: norms ------------------------------------
        Workload(
            "nn.BatchNorm1d", lambda: (nn.BatchNorm1d(d), _r(4, d))
        ),
        Workload(
            "nn.BatchNorm2d",
            lambda: (nn.BatchNorm2d(8), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.GroupNorm", lambda: (nn.GroupNorm(4, 8), _r(1, 8, 8, 8))
        ),
        Workload("nn.LayerNorm", lambda: (nn.LayerNorm(d), _r(4, d))),
        Workload(
            "nn.InstanceNorm2d",
            lambda: (nn.InstanceNorm2d(8), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.LocalResponseNorm",
            lambda: (nn.LocalResponseNorm(4), _r(1, 8, 8, 8)),
        ),
        # --- torch-native: conv / pooling / reshaping ---------------
        Workload(
            "nn.ConvTranspose2d",
            lambda: (nn.ConvTranspose2d(8, 8, 3), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Conv3d",
            lambda: (nn.Conv3d(4, 4, 3), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.MaxPool2d", lambda: (nn.MaxPool2d(2), _r(1, 8, 8, 8))
        ),
        Workload(
            "nn.AdaptiveAvgPool2d",
            lambda: (nn.AdaptiveAvgPool2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.PixelShuffle",
            lambda: (nn.PixelShuffle(2), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Upsample(nearest)",
            lambda: (nn.Upsample(scale_factor=2), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Upsample(bilinear)",
            lambda: (
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                ),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload("nn.Unfold", lambda: (nn.Unfold(2), _r(1, 8, 8, 8))),
        Workload(
            "nn.Fold",
            lambda: (nn.Fold((8, 8), (2, 2)), _r(1, 8 * 4, 49)),
        ),
        Workload(
            "nn.Dropout2d", lambda: (nn.Dropout2d(0.3), _r(1, 8, 8, 8))
        ),
        # --- torch-native: embeddings / pairwise / losses -----------
        Workload(
            "nn.Embedding",
            lambda: (nn.Embedding(32, d), torch.randint(0, 32, (2, 8))),
        ),
        Workload(
            "nn.Bilinear",
            lambda: (nn.Bilinear(d, d, d), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CosineSimilarity",
            lambda: (nn.CosineSimilarity(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CrossEntropyLoss",
            lambda: (
                nn.CrossEntropyLoss(),
                (_r(4, 8), torch.randint(0, 8, (4,))),
            ),
        ),
        Workload(
            "nn.MSELoss",
            lambda: (nn.MSELoss(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CTCLoss",
            lambda: (
                nn.CTCLoss(),
                (
                    F.log_softmax(_r(8, 2, 6), -1),
                    torch.randint(1, 6, (2, 3)),
                    torch.tensor([8, 8]),
                    torch.tensor([3, 3]),
                ),
            ),
        ),
        # --- torch-native: recurrent family -------------------------
        Workload(
            "nn.LSTM",
            lambda: (nn.LSTM(8, 8, batch_first=True), _r(2, 6, 8)),
        ),
        Workload(
            "nn.GRU",
            lambda: (nn.GRU(8, 8, batch_first=True), _r(2, 6, 8)),
        ),
        Workload(
            "nn.RNNCell",
            lambda: (nn.RNNCell(8, 8), (_r(2, 8), _r(2, 8))),
        ),
        # --- torch-native: assorted pointwise -----------------------
        Workload("nn.PReLU", lambda: (nn.PReLU(), _r(4, d))),
        Workload("nn.Mish", lambda: (nn.Mish(), _r(4, d))),
        Workload("nn.Hardswish", lambda: (nn.Hardswish(), _r(4, d))),
        # --- compound models ----------------------------------------
        Workload(
            "SharedExpertMoE",
            lambda: (_SharedExpertMoE(d, 8, 4), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ViTPatchBlock",
            lambda: (_ViTPatchBlock(8, d, 4), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "TinyCNN",
            lambda: (_TinyCNN(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "Bottleneck",
            lambda: (_Bottleneck(8, 4), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "LossHead",
            lambda: (
                _LossHead(d, 8),
                (_r(4, d), torch.randint(0, 8, (4,))),
            ),
            kind="compound",
        ),
        Workload(
            "ManualNLL",
            lambda: (
                _ManualNLL(d, 8),
                (_r(4, d), torch.randint(0, 8, (4,))),
            ),
            kind="compound",
        ),
        Workload(
            "LogSoftmaxHead",
            lambda: (_LogSoftmaxHead(d, 8), _r(4, d)),
            kind="compound",
        ),
        # --- round 2 — torch-native: activation spellings ----------
        Workload(
            "nn.LeakyReLU", lambda: (nn.LeakyReLU(0.05), _r(4, d))
        ),
        Workload("nn.CELU", lambda: (nn.CELU(), _r(4, d))),
        Workload("nn.SELU", lambda: (nn.SELU(), _r(4, d))),
        Workload("nn.Softplus", lambda: (nn.Softplus(), _r(4, d))),
        Workload("nn.Softsign", lambda: (nn.Softsign(), _r(4, d))),
        Workload(
            "nn.Hardsigmoid", lambda: (nn.Hardsigmoid(), _r(4, d))
        ),
        Workload("nn.Hardtanh", lambda: (nn.Hardtanh(-2, 2), _r(4, d))),
        Workload("nn.ReLU6", lambda: (nn.ReLU6(), _r(4, d))),
        Workload(
            "nn.RReLU", lambda: (nn.RReLU(0.1, 0.3).eval(), _r(4, d))
        ),
        Workload("nn.LogSigmoid", lambda: (nn.LogSigmoid(), _r(4, d))),
        Workload("nn.Softmin", lambda: (nn.Softmin(-1), _r(4, d))),
        Workload("nn.Tanhshrink", lambda: (nn.Tanhshrink(), _r(4, d))),
        Workload("nn.Softshrink", lambda: (nn.Softshrink(), _r(4, d))),
        # --- round 2 — pooling / upsampling / conv gaps ------------
        Workload(
            "nn.AvgPool1d", lambda: (nn.AvgPool1d(2), _r(1, 8, 16))
        ),
        Workload(
            "nn.AvgPool2d", lambda: (nn.AvgPool2d(2), _r(1, 8, 8, 8))
        ),
        Workload(
            "nn.AdaptiveMaxPool2d",
            lambda: (nn.AdaptiveMaxPool2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.MaxPool1d", lambda: (nn.MaxPool1d(2), _r(1, 8, 16))
        ),
        Workload(
            "nn.MaxPool3d",
            lambda: (nn.MaxPool3d(2), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.FractionalMaxPool2d",
            lambda: (
                nn.FractionalMaxPool2d(2, output_size=4),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload(
            "nn.Upsample(bicubic)",
            lambda: (
                nn.Upsample(
                    scale_factor=2, mode="bicubic", align_corners=False
                ),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload(
            "nn.Upsample(linear)",
            lambda: (
                nn.Upsample(
                    scale_factor=2, mode="linear", align_corners=False
                ),
                _r(1, 8, 16),
            ),
        ),
        Workload(
            "nn.Upsample(area)",
            lambda: (
                nn.Upsample(scale_factor=2, mode="area"),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload(
            "nn.Upsample(nearest-exact)",
            lambda: (
                nn.Upsample(scale_factor=2, mode="nearest-exact"),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload(
            "nn.ConvTranspose1d",
            lambda: (nn.ConvTranspose1d(8, 8, 3), _r(1, 8, 16)),
        ),
        Workload(
            "nn.ConvTranspose3d",
            lambda: (nn.ConvTranspose3d(4, 4, 3), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.Conv1d(grouped)",
            lambda: (
                nn.Conv1d(8, 8, 3, padding=1, groups=4),
                _r(1, 8, 16),
            ),
        ),
        # --- round 2 — recurrent cells / embeddings / distances ----
        Workload(
            "nn.GRUCell",
            lambda: (nn.GRUCell(8, 8), (_r(2, 8), _r(2, 8))),
        ),
        Workload(
            "nn.LSTMCell",
            lambda: (
                nn.LSTMCell(8, 8),
                (_r(2, 8), (_r(2, 8), _r(2, 8))),
            ),
        ),
        Workload(
            "nn.EmbeddingBag",
            lambda: (
                nn.EmbeddingBag(32, d),
                torch.randint(0, 32, (2, 8)),
            ),
        ),
        Workload(
            "nn.PairwiseDistance",
            lambda: (nn.PairwiseDistance(), (_r(4, d), _r(4, d))),
        ),
        # --- round 2 — losses --------------------------------------
        Workload(
            "nn.L1Loss", lambda: (nn.L1Loss(), (_r(4, d), _r(4, d)))
        ),
        Workload(
            "nn.SmoothL1Loss",
            lambda: (nn.SmoothL1Loss(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.HuberLoss",
            lambda: (nn.HuberLoss(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.BCELoss",
            lambda: (
                nn.BCELoss(),
                (
                    torch.rand(4, 8).double(),
                    torch.rand(4, 8).double(),
                ),
            ),
        ),
        Workload(
            "nn.BCEWithLogitsLoss",
            lambda: (
                nn.BCEWithLogitsLoss(),
                (_r(4, 8), torch.rand(4, 8).double()),
            ),
        ),
        Workload(
            "nn.KLDivLoss",
            lambda: (
                nn.KLDivLoss(),
                (
                    F.log_softmax(_r(4, 8), -1),
                    F.softmax(_r(4, 8), -1),
                ),
            ),
        ),
        Workload(
            "nn.SoftMarginLoss",
            lambda: (
                nn.SoftMarginLoss(),
                (
                    _r(4, 8),
                    torch.randint(0, 2, (4, 8)).double() * 2 - 1,
                ),
            ),
        ),
        Workload(
            "nn.MarginRankingLoss",
            lambda: (
                nn.MarginRankingLoss(),
                (_r(4), _r(4), torch.ones(4)),
            ),
        ),
        Workload(
            "nn.TripletMarginLoss",
            lambda: (
                nn.TripletMarginLoss(),
                (_r(4, d), _r(4, d), _r(4, d)),
            ),
        ),
        Workload(
            "nn.PoissonNLLLoss",
            lambda: (
                nn.PoissonNLLLoss(),
                (_r(4, 8), torch.rand(4, 8).double() + 0.1),
            ),
        ),
        Workload(
            "nn.GaussianNLLLoss",
            lambda: (
                nn.GaussianNLLLoss(),
                (
                    _r(4, 8),
                    torch.rand(4, 8).double(),
                    torch.rand(4, 8).double() + 0.5,
                ),
            ),
        ),
        Workload(
            "nn.CosineEmbeddingLoss",
            lambda: (
                nn.CosineEmbeddingLoss(),
                (_r(4, d), _r(4, d), torch.ones(4)),
            ),
        ),
        # --- round 2 — padding modes / misc natives ----------------
        Workload(
            "nn.ReflectionPad2d",
            lambda: (nn.ReflectionPad2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.ReplicationPad2d",
            lambda: (nn.ReplicationPad2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.ConstantPad2d",
            lambda: (nn.ConstantPad2d(1, 0.5), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.PixelUnshuffle",
            lambda: (nn.PixelUnshuffle(2), _r(1, 8, 8, 8)),
        ),
        # --- round 2 — compound models ------------------------------
        Workload(
            "CircularPad",
            lambda: (_CircularPad(), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "SpatialTransformer",
            lambda: (_SpatialTransformer(), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "ShiftedWindowRoll",
            lambda: (_ShiftedWindow(), _r(1, 8, 8, d)),
            kind="compound",
        ),
        Workload(
            "ALiBiAttention",
            lambda: (
                _ALiBiAttention(4),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "GQA-sdpa",
            lambda: (
                _GQAAttention(),
                (_r(1, 8, 8, d), _r(1, 2, 8, d), _r(1, 2, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "EinsumAttention",
            lambda: (
                _EinsumAttention(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "ManualAttention",
            lambda: (
                _ManualAttention(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "VarNorm",
            lambda: (_VarNorm(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "StdNorm",
            lambda: (_StdNorm(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "VarStdHead",
            lambda: (_VarStdHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "FakeQuant",
            lambda: (_FakeQuant(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SincKernel",
            lambda: (_SincKernel(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ScalarRsub",
            lambda: (_ScalarRsub(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "CumsumScan",
            lambda: (_CumsumScan(), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "CumProdGate",
            lambda: (_CumProdGate(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "CumMaxMin",
            lambda: (_CumMaxMin(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "IndexAddMoE",
            lambda: (
                _IndexMoE(d),
                (_r(8, d), torch.randint(0, 8, (8,))),
            ),
            kind="compound",
        ),
        Workload(
            "ScatterAdd",
            lambda: (
                _ScatterAdd(),
                (_r(8, 8), torch.randint(0, 4, (8,))),
            ),
            kind="compound",
        ),
        Workload(
            "MedianPool",
            lambda: (_MedianPool(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SortSelect",
            lambda: (_SortSelect(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ArgSort",
            lambda: (_ArgSort(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "KthMode",
            lambda: (_KthMode(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "TakeAlongGather",
            lambda: (_TakeAlong(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "NormalizeHead",
            lambda: (_NormalizeHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SlidingUnfold",
            lambda: (_SlidingUnfold(), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "MoveDimStack",
            lambda: (_MoveDimStack(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "TensorSplitHead",
            lambda: (_TensorSplit(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "AngleLog",
            lambda: (_AngleLog(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "Log10Expm1",
            lambda: (_Log10Expm1(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "LogBase",
            lambda: (_LogBase(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "NanGuard",
            lambda: (_NanGuard(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SanitizeHead",
            lambda: (_SanitizeHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "NanStats",
            lambda: (_NanStats(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "MaskLogic",
            lambda: (_MaskLogic(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "LogicCombo",
            lambda: (_LogicCombo(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "Bucketize",
            lambda: (_Bucketize(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ChannelShuffle",
            lambda: (_ChannelShuffle(), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "FlipConv",
            lambda: (_FlipConv(), _r(1, 4, 16)),
            kind="compound",
        ),
        Workload(
            "GeoMeanPool",
            lambda: (_GeoMeanPool(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "LerpMix",
            lambda: (_LerpMix(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "AddcmulHead",
            lambda: (_AddcmulHead(), (_r(4, d), _r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "TriuAttention",
            lambda: (
                _TriuAttention(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "TracePenalty",
            lambda: (_TracePenalty(), (_r(8, d), _r(d, d))),
            kind="compound",
        ),
        Workload(
            "OuterPositional",
            lambda: (_OuterPositional(), _r(2, d, d)),
            kind="compound",
        ),
        Workload(
            "RepeatHead",
            lambda: (_RepeatHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "RemainderHead",
            lambda: (_RemainderHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ProdAminPool",
            lambda: (_ProdAminPool(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "MinFmax",
            lambda: (_MinFmax(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "BroadcastTo",
            lambda: (_BroadcastTo(), (_r(4, d), _r(4, d, 4))),
            kind="compound",
        ),
        Workload(
            "CountNonzero",
            lambda: (_CountHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "BooleanIndex",
            lambda: (_BooleanIndex(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SignTruncFrac",
            lambda: (_SignTruncFrac(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "CeilFloor",
            lambda: (_CeilFloor(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "Heaviside",
            lambda: (_Heaviside(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SdpaBiasParam",
            lambda: (
                _SdpaBiasParam(8),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "Cdist",
            lambda: (_CDist(), (_r(4, 8), _r(6, 8))),
            kind="compound",
        ),
        Workload(
            "EyeInit",
            lambda: (_EyeInit(), _r(d, d)),
            kind="compound",
        ),
        Workload(
            "UnbindHead",
            lambda: (_UnbindHead(), _r(2, 8)),
            kind="compound",
        ),
        Workload(
            "SliceFill",
            lambda: (_SliceFill(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "FillWhole",
            lambda: (_FillWhole(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "FillSelect",
            lambda: (_FillSelect(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ZeroInit",
            lambda: (_ZeroInit(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "MaskedFillInplace",
            lambda: (_MaskedFillInplace(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "MaskedFillTensor",
            lambda: (
                _MaskedFillTensor(),
                (_r(4, d), torch.randn((), dtype=torch.float64)),
            ),
            kind="compound",
        ),
        Workload(
            "MaskedFillView",
            lambda: (_MaskedFillView(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SelectWrite",
            lambda: (_SelectWrite(), (_r(4, d), _r(d))),
            kind="compound",
        ),
        Workload(
            "IndexPut",
            lambda: (_IndexPut(), (_r(4, d), _r(2, d))),
            kind="compound",
        ),
        Workload(
            "NarrowHead",
            lambda: (_NarrowHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "NewCreators",
            lambda: (_NewCreators(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "TypeAs",
            lambda: (_TypeAs(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "ExpandAs",
            lambda: (_ExpandAs(), (_r(4, d), _r(4, d, 4))),
            kind="compound",
        ),
        Workload(
            "IsNegPos",
            lambda: (_IsNegPos(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "TrigHead",
            lambda: (_TrigHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "HyperbolicHead",
            lambda: (_HyperbolicHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "CoshAcosh",
            lambda: (_CoshAcosh(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "DigammaLn",
            lambda: (_DigammaLn(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "XlogyKL",
            lambda: (_XlogyKL(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "RegluMLP",
            lambda: (_RegluMlp(d), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SwiGLU",
            lambda: (_SwiGLU(d), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "OneHotEmbed",
            lambda: (_OneHotEmbed(), torch.randint(0, 16, (4, 8))),
            kind="compound",
        ),
        Workload(
            "BitwiseHead",
            lambda: (
                _BitwiseHead(),
                (
                    torch.randint(0, 8, (4, 8)),
                    torch.randint(0, 8, (4, 8)),
                ),
            ),
            kind="compound",
        ),
        Workload(
            "RepeatInterleave",
            lambda: (_RepeatInterleave(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "DepthwiseConv2d",
            lambda: (_DepthwiseConv2d(), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        # --- round 3 — torch-native norm gaps ----------------------
        Workload("nn.RMSNorm", lambda: (nn.RMSNorm(d), _r(4, d))),
        Workload(
            "nn.BatchNorm3d",
            lambda: (nn.BatchNorm3d(4), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.TransformerEncoderLayer(norm_first)",
            lambda: (
                nn.TransformerEncoderLayer(
                    d,
                    4,
                    2 * d,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=True,
                ),
                _r(2, 8, d),
            ),
        ),
        # --- round 3 — transformer variants ------------------------
        Workload(
            "RotaryEmbedding",
            lambda: (_RotaryEmbedding(), _r(1, 4, 8, d)),
            kind="compound",
        ),
        Workload(
            "RoPEAttention",
            lambda: (_RoPEAttention(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "MQAAttention",
            lambda: (_MQAAttention(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "RelativePositionBias",
            lambda: (_RelativePositionBias(4, 8), _r(2, 4, 8, 8)),
            kind="compound",
        ),
        Workload(
            "MoEGate",
            lambda: (_MoEGate(d, 4), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "PreNormBlock",
            lambda: (_PreNormBlock(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "PostNormBlock",
            lambda: (_PostNormBlock(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "SwiGLUBlock",
            lambda: (_SwiGLUBlock(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "CrossAttentionBlock",
            lambda: (
                _CrossAttentionBlock(d, 4),
                (_r(2, 8, d), _r(2, 6, d)),
            ),
            kind="compound",
        ),
        Workload(
            "SlidingWindowMask",
            lambda: (
                _SlidingWindowMask(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "MultiLatentAttention",
            lambda: (_MultiLatentAttention(d, 4, 8), _r(2, 8, d)),
            kind="compound",
        ),
        # --- round 3 — normalisation variants ----------------------
        Workload(
            "ManualRMSNorm",
            lambda: (_ManualRMSNorm(d), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "QKNorm",
            lambda: (
                _QKNorm(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "DeepNorm",
            lambda: (_DeepNorm(d), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "CenterNorm",
            lambda: (_CenterNorm(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ScaleNorm",
            lambda: (_ScaleNorm(d), _r(4, d)),
            kind="compound",
        ),
        # --- round 3 — conv/attention hybrids ----------------------
        Workload(
            "ConvNeXtBlock",
            lambda: (_ConvNeXtBlock(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "MBConvBlock",
            lambda: (_MBConvBlock(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "GRN",
            lambda: (_GRN(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "ConformerBlock",
            lambda: (_ConformerBlock(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "AxialAttention",
            lambda: (_AxialAttention(8, 2), _r(1, 8, 4, 4)),
            kind="compound",
        ),
        Workload(
            "PatchMerging",
            lambda: (_PatchMerging(d), _r(1, 4, d)),
            kind="compound",
        ),
        # --- round 3 — recurrent / scan-heavy ----------------------
        Workload(
            "SelectiveScan",
            lambda: (_SelectiveScan(d), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "LinearAttention",
            lambda: (
                _LinearAttention(),
                (_r(2, 8, d), _r(2, 8, d), _r(2, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "CumulativeGateRNN",
            lambda: (_CumulativeGateRNN(), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "RunningNorm",
            lambda: (_RunningNorm(), _r(2, 8, d)),
            kind="compound",
        ),
        # --- round 3 — view-identity elementwise spellings ---------
        Workload(
            "BroadcastPadLeft",
            lambda: (_BroadcastPadLeft(), (_r(4, 8), _r(1, 4, 8))),
            kind="compound",
        ),
        Workload(
            "BroadcastPadRight",
            lambda: (_BroadcastPadRight(), (_r(1, 4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "BroadcastPadSub",
            lambda: (_BroadcastPadSub(), (_r(4, 8), _r(1, 4, 8))),
            kind="compound",
        ),
        Workload(
            "FullSliceScale",
            lambda: (_FullSliceScale(), (_r(2, 4, 8), _r(2, 4, 8))),
            kind="compound",
        ),
        Workload(
            "NoopTransposeScale",
            lambda: (_NoopTransposeScale(), (_r(8, 1, 1), _r(8, 1, 1))),
            kind="compound",
        ),
        Workload(
            "NoopTransposeAdd",
            lambda: (_NoopTransposeAdd(), (_r(8, 1, 1), _r(8, 1, 1))),
            kind="compound",
        ),
        Workload(
            "InertReshapeScale",
            lambda: (_InertReshapeScale(), (_r(8, 8), _r(1, 8, 8))),
            kind="compound",
        ),
        Workload(
            "SingleChunkScale",
            lambda: (_SingleChunkScale(), (_r(2, 8), _r(2, 8))),
            kind="compound",
        ),
        Workload(
            "SquaredDistance",
            lambda: (_SquaredDistance(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 3 — ops absent from the census ------------------
        Workload(
            "ErfFeatures",
            lambda: (_ErfFeatures(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "GeluErf",
            lambda: (_GeluErf(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SoftsignHead",
            lambda: (_SoftsignHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "Relu6Head",
            lambda: (_Relu6Head(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SqrtHead",
            lambda: (_SqrtHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "TraceNorm",
            lambda: (_TraceNorm(), _r(8, 8)),
            kind="compound",
        ),
        Workload(
            "ScatterReduce",
            lambda: (
                _ScatterReduce(),
                (_r(8, 8), torch.randint(0, 4, (8,))),
            ),
            kind="compound",
        ),
        Workload(
            "MinPair",
            lambda: (_MinPair(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "ClampMaxHead",
            lambda: (_ClampMaxHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ComparisonHead",
            lambda: (_ComparisonHead(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "AllReduce",
            lambda: (_AllReduce(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "BroadcastTensors",
            lambda: (_BroadcastTensors(), (_r(4, 8), _r(4, 8, 3))),
            kind="compound",
        ),
        Workload(
            "MatrixInverse",
            lambda: (_MatrixInverse(), _r(2, 8, 8)),
            kind="compound",
        ),
        # --- round 4 — the unsqueeze-wrap (gated broadcast) family --
        Workload(
            "ChannelGateBroadcast",
            lambda: (
                _ChannelGateBroadcast(),
                (_r(8, 8, 8), _r(8, 8, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "LiftedScalarScale",
            lambda: (_LiftedScalarScale(), (_r(8, 8, 8), _r())),
            kind="compound",
        ),
        Workload(
            "HeadGateBroadcast",
            lambda: (_HeadGateBroadcast(), (_r(8, 8, 8), _r(8, 8, 8))),
            kind="compound",
        ),
        Workload(
            "LiftedScalarScaleRight",
            lambda: (_LiftedScalarScaleRight(), (_r(8, 8, 8), _r())),
            kind="compound",
        ),
        Workload(
            "LiftedScalarCenter",
            lambda: (_LiftedScalarCenter(), (_r(8, 8, 8), _r())),
            kind="compound",
        ),
        Workload(
            "ContrastiveCenter",
            lambda: (_ContrastiveCenter(), (_r(8, 8, 8), _r(8, 8, 8))),
            kind="compound",
        ),
        Workload(
            "PairwiseSubLift",
            lambda: (_PairwiseSubLift(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 4 — chunked-projection mirrors -------------------
        Workload(
            "ScaledChunkProjection",
            lambda: (_ScaledChunkProjection(d), (_r(4, d), _r())),
            kind="compound",
        ),
        Workload(
            "SingleChunkGate",
            lambda: (_SingleChunkGate(d), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "ChunkHalfScale",
            lambda: (_ChunkHalfScale(d), (_r(4, d), _r())),
            kind="compound",
        ),
        # --- round 4 — transposed-add mirrors -----------------------
        Workload(
            "NoopTransposeResidual",
            lambda: (
                _NoopTransposeResidual(),
                (_r(8, 1, 1), _r(8, 1, 1)),
            ),
            kind="compound",
        ),
        Workload(
            "ScalarTransposeBias",
            lambda: (_ScalarTransposeBias(), (_r(), _r(8, 1, 1))),
            kind="compound",
        ),
        Workload(
            "Rank3TransposeAdd",
            lambda: (
                _Rank3TransposeAdd(),
                (_r(1, 8, 1, 1), _r(8, 1, 1)),
            ),
            kind="compound",
        ),
        # --- round 4 — select naturality ----------------------------
        Workload(
            "SelectGateSum",
            lambda: (_SelectGateSum(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "SelectGateDiff",
            lambda: (_SelectGateDiff(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 4 — shared-factor algebra ------------------------
        Workload(
            "SharedFactorMixture",
            lambda: (
                _SharedFactorMixture(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "SharedFactorContrast",
            lambda: (
                _SharedFactorContrast(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "SharedFactorMixtureRight",
            lambda: (
                _SharedFactorMixtureRight(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "SharedFactorContrastRight",
            lambda: (
                _SharedFactorContrastRight(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "QuadraticFeature",
            lambda: (_QuadraticFeature(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 4 — reshape / neg / exp / square grammar ---------
        Workload(
            "DoubleReshapeHead",
            lambda: (_DoubleReshapeHead(), _r(2, 3, 8)),
            kind="compound",
        ),
        Workload(
            "NegatedSum",
            lambda: (_NegatedSum(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "NegDistributeHead",
            lambda: (_NegDistributeHead(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "SubNegBias",
            lambda: (_SubNegBias(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "NegatedScale",
            lambda: (_NegatedScale(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "ExpProductHead",
            lambda: (_ExpProductHead(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "ExpSumHead",
            lambda: (_ExpSumHead(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "SquareNegHead",
            lambda: (_SquareNegHead(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "SquareMulHead",
            lambda: (_SquareMulHead(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "SigmoidNegGate",
            lambda: (_SigmoidNegGate(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "PowOneHead",
            lambda: (_PowOneHead(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "SubAddFactorHead",
            lambda: (
                _SubAddFactorHead(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "DivAddHead",
            lambda: (_DivAddHead(), (_r(4, 8), _r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 4 — scalar-corner annihilators -------------------
        Workload(
            "ScalarAnnihilator",
            lambda: (_ScalarAnnihilator(), _r()),
            kind="compound",
        ),
        Workload(
            "ScalarSelfCancel",
            lambda: (_ScalarSelfCancel(), _r()),
            kind="compound",
        ),
        Workload(
            "ScalarSelfRatio",
            lambda: (_ScalarSelfRatio(), _r()),
            kind="compound",
        ),
        Workload(
            "ScalarInverseSum",
            lambda: (_ScalarInverseSum(), _r()),
            kind="compound",
        ),
    ]
    # Provenance is data: the ledger above marks the micro-modules
    # written to spell a pattern, so the honest "real-only" filter is
    # one predicate over the registry.
    return [
        replace(w, purpose_built=True)
        if w.name in _PURPOSE_BUILT
        else w
        for w in reg
    ]


def real_workloads(
    cands: list[Workload] | None = None,
) -> list[Workload]:
    """Return the registry minus the purpose-built spelling workloads.

    The honest subset for a full-vs-real corpus comparison: workloads
    that represent a real network — a torch-native ``nn.*`` module or
    a realistic compound assembly — not a micro-module written to
    spell one pattern.  ``purpose_built`` is the ledger
    (:data:`_PURPOSE_BUILT`); the filter excludes exactly those names.
    """
    src = candidates() if cands is None else cands
    return [w for w in src if not w.purpose_built]
