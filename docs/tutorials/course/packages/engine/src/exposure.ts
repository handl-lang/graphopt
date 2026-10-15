/** Exposure — where discovery can pay at all.
 *
 * Mirrors `project/retros/exposed-math.md`: discovery only pays where a
 * model's math is *visible* to the IR.  The screen is bimodal — the
 * fused-op class (`nn.*` natives that export to one opaque node) scores
 * exactly 0% FLOPs-weighted law coverage, while the spelled-math class
 * scores ≥95%.  The real corpus: 321 exported workloads, **81 at 0%**,
 * **140 at ≥95%**.
 *
 * The workload slice and the op classifier below are the measured zoo
 * table (`exposed-math.md`) plus the fused-op class; the histogram is
 * computed live over that slice.  The corpus totals are the cited
 * measured numbers.
 */

/** The measured corpus totals (`exposed-math.md`). */
export const CORPUS = { total: 321, zero: 81, high: 140 };

/** The honest yield: paying unpublished structure per exposed workload. */
export const YIELD_PCT = 1.5;

export interface Workload {
  name: string;
  lhs: number;
  note: string;
}

/** A representative slice: the fused-op tail, the zoo, and the middle. */
export const WORKLOADS: readonly Workload[] = [
  // the 0% fused-op class — "the e-graph literally cannot see the computation"
  { name: "ResNetBlock", lhs: 0, note: "one fused conv stack — nothing to rewrite" },
  { name: "TinyCNN", lhs: 0, note: "fused `nn.*` natives" },
  { name: "MBConvBlock", lhs: 0, note: "fused depthwise/pointwise convs" },
  { name: "Bottleneck", lhs: 0, note: "fused convs" },
  { name: "DepthwiseConv", lhs: 0, note: "a single opaque node" },
  { name: "LayerNormHead", lhs: 0, note: "`nn.LayerNorm` exports as one node" },
  { name: "ConvStack", lhs: 0, note: "conv-only" },
  { name: "PoolHead", lhs: 0, note: "`nn.*Pool*` natives" },
  // the middle — visible ≠ reachable
  { name: "TimeCondConv", lhs: 3.5, note: "conv2d carries 96% of the FLOPs, invisible" },
  { name: "FNOBlock", lhs: 11.6, note: "conv2d 73%, fft_rfft2 invisible" },
  { name: "SelectiveScan", lhs: 33, note: "cumsum/cumprod shares the laws can't touch" },
  { name: "LinearAttention", lhs: 51, note: "cumsum carries 44% of its FLOPs invisible" },
  { name: "SinusoidalEncoding", lhs: 69, note: "sin/cos invisible" },
  { name: "SincKernel", lhs: 79, note: "sinc/i0 invisible" },
  // the zoo — the most exposed models, all ≥80%
  { name: "SplineKAN", lhs: 80.2, note: "sin 19%, einsum" },
  { name: "ExpertChoiceMoE", lhs: 85.3, note: "gather, topk, scatter_add, einsum" },
  { name: "FiLMHead", lhs: 86.2, note: "layer_norm 14%" },
  { name: "CapsuleRouting", lhs: 91.3, note: "zeros 9%, einsum" },
  { name: "TalkingHeadsAttention", lhs: 96.2, note: "permute 4%" },
  { name: "DeltaRuleBlock", lhs: 97.2, note: "the census find — 58 strict LHS sites" },
  { name: "HighwayGate", lhs: 98.6, note: "rsub 1%" },
  { name: "CosineAttention", lhs: 98.6, note: "expand_as, vector_norm, clamp" },
  { name: "MLPMixerBlock", lhs: 98.7, note: "layer_norm 1%" },
  { name: "AffineCoupling", lhs: 99.0, note: "clamp 1%" },
  { name: "AdaLNBlock", lhs: 99.3, note: "layer_norm 1%" },
  { name: "MoSHead", lhs: 99.8, note: "einsum 0%" },
  { name: "LoRAAdapter", lhs: 100, note: "fully spelled — pays −20%" },
  { name: "GatedAttentionUnit", lhs: 100, note: "fully spelled — pays −9.5%" },
  { name: "DeepEquilibrium", lhs: 100, note: "fully spelled — and pays 0" },
  { name: "GCNLayer", lhs: 100, note: "fully spelled — pays −7.7%" },
  { name: "AdditiveAttention", lhs: 100, note: "fully spelled — and pays 0" },
  { name: "CovarianceHead", lhs: 100, note: "fully spelled — and pays 0" },
  { name: "ChunkedRetention", lhs: 100, note: "fully spelled — pays −48%" },
  { name: "SoftSlotMoE", lhs: 100, note: "fully spelled — and pays 0" },
];

export interface OpClass {
  op: string;
  visible: boolean;
  share: number;
  where: string;
}

/** Ops the law LHSs can name — and the black boxes they cannot. */
export const OP_CLASSES: readonly OpClass[] = [
  { op: "matmul", visible: true, share: 0, where: "associativity, the whole contraction family" },
  {
    op: "add / mul / sub / div",
    visible: true,
    share: 0,
    where: "commutativity, distribution, folds",
  },
  { op: "sigmoid", visible: true, share: 0, where: "silu_fold, swiglu_fuse" },
  { op: "softmax", visible: true, share: 0, where: "softmax_fold" },
  {
    op: "sdpa",
    visible: true,
    share: 0,
    where: "sdpa_fold — but 0 strict matches: visible ≠ reachable",
  },
  { op: "conv2d", visible: false, share: 96, where: "TimeCondConv — one fused node" },
  { op: "conv1d", visible: false, share: 27, where: "WaveNetGate" },
  { op: "layer_norm", visible: false, share: 14, where: "FiLMHead" },
  { op: "sin / cos", visible: false, share: 19, where: "SplineKAN / SinusoidalEncoding" },
  { op: "cumsum", visible: false, share: 44, where: "LinearAttention — an atom-synth need" },
  { op: "einsum", visible: false, share: 0, where: "SoftSlotMoE, MoSHead" },
  { op: "gather / scatter_add / topk", visible: false, share: 0, where: "ExpertChoiceMoE" },
  { op: "fft_rfft2 / irfft2", visible: false, share: 0, where: "FNOBlock" },
  {
    op: "deform_conv2d",
    visible: false,
    share: 0,
    where: "BiRefNet — exports, but no lowering binding",
  },
];

export interface Bin {
  label: string;
  count: number;
}

const BINS: ReadonlyArray<{ label: string }> = [
  { label: "0%" },
  { label: "1–25%" },
  { label: "25–50%" },
  { label: "50–75%" },
  { label: "75–95%" },
  { label: "95–100%" },
];

const bucket = (lhs: number): number => {
  if (lhs <= 0) return 0;
  if (lhs <= 25) return 1;
  if (lhs <= 50) return 2;
  if (lhs <= 75) return 3;
  if (lhs < 95) return 4;
  return 5;
};

/** The exposure screen, playable: the histogram is computed live. */
export class Exposure {
  mode: "workloads" | "ops" = "workloads";

  get workloads(): readonly Workload[] {
    return WORKLOADS;
  }

  get ops(): readonly OpClass[] {
    return OP_CLASSES;
  }

  /** The bimodal histogram, computed live over the workload slice. */
  get histogram(): Bin[] {
    const counts = Array.from({ length: BINS.length }, () => 0);
    for (const w of WORKLOADS) counts[bucket(w.lhs)] += 1;
    return BINS.map((b, i) => ({ label: b.label, count: counts[i] }));
  }

  get maxBin(): number {
    return Math.max(1, ...this.histogram.map((b) => b.count));
  }

  /** The two modes the screen separates, over the live slice. */
  get modes(): { zero: number; high: number; total: number } {
    return {
      zero: WORKLOADS.filter((w) => w.lhs === 0).length,
      high: WORKLOADS.filter((w) => w.lhs >= 95).length,
      total: WORKLOADS.length,
    };
  }

  get visibleFlops(): number {
    return WORKLOADS.reduce((p, w) => p + w.lhs, 0);
  }

  get invisibleFlops(): number {
    return WORKLOADS.reduce((p, w) => p + (100 - w.lhs), 0);
  }

  /** Exposure is necessary, not sufficient: fully-visible models that pay 0. */
  get visibleButNull(): Workload[] {
    return WORKLOADS.filter((w) => w.lhs === 100 && w.note.includes("pays 0"));
  }

  setMode(m: "workloads" | "ops"): void {
    this.mode = m;
  }
}
