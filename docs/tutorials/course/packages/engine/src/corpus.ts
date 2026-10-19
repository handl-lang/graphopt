/** The corpus and the intake — where programs come from.
 *
 * Plan 0028 §D1: the system is measured on **real exported programs**,
 * not hand-written terms.  `python -m mlopt_discovery.intake` takes a
 * candidate `nn.Module` through `torch.export` → `export_to_ir` →
 * classify → persist, and every candidate lands in exactly one class:
 * `ingested` (bound, lowered, verified fp64), `census-only` (exported
 * but an op is unbound), or `rejected` (`torch.export` raised).
 *
 * The headline numbers are the plan's (§D1): **166 workloads, 114
 * verified**, and the intake's own finding — **three lowering bugs** it
 * caught (`nn.MultiheadAttention`, `nn.TransformerDecoderLayer`,
 * `nn.LSTM`), each a boundary that emitted a term that was never the
 * program.  The browser below is a labelled *slice* of the corpus
 * (`project/retros/workload-intake.md` + `intake-defect-fixes.md`);
 * the class counts are computed live from it.
 */

/** The class a workload lands in — the intake's three-way split. */
export type IntakeStatus = "ingested" | "census-only" | "rejected";

/** One workload in the browser — a labelled slice of the real corpus. */
export interface IntakeWorkload {
  id: string;
  name: string;
  /** The architecture family. */
  family: string;
  /** Distinct ops the export spells. */
  ops: number;
  /** Op nodes in the exported graph. */
  nodes: number;
  status: IntakeStatus;
  note: string;
  /** True when the counts are a real measurement (vs illustrative). */
  sourced: boolean;
  source: string;
}

/** The plan's headline corpus totals (plan 0028 §D1). */
export const INTAKE = {
  /** Real `nn.*` workloads in the corpus. */
  total: 166,
  /** Workloads that bind, lower and verify fp64. */
  verified: 114,
  /** Lowering defects the intake verify gate caught. */
  defects: 3,
} as const;

/** A labelled slice of the corpus — the intake's own retro examples. */
export const INTAKE_WORKLOADS: readonly IntakeWorkload[] = [
  {
    id: "encoder-layer",
    name: "nn.TransformerEncoderLayer",
    family: "transformer",
    ops: 18,
    nodes: 35,
    status: "ingested",
    note: "permute / unflatten / dropout / sdpa — library spellings the hand corpus never wrote",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "moe",
    name: "SharedExpertMoE",
    family: "routing",
    ops: 12,
    nodes: 47,
    status: "ingested",
    note: "the topk → eq → any → to dispatch spine",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "vit",
    name: "ViTPatchBlock",
    family: "vision",
    ops: 15,
    nodes: 38,
    status: "ingested",
    note: "conv2d → flatten → transpose → encoder",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "batchnorm",
    name: "nn.BatchNorm2d",
    family: "norm",
    ops: 6,
    nodes: 6,
    status: "ingested",
    note: "a batched-norm chain, lowered and verified",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "groupnorm",
    name: "nn.GroupNorm",
    family: "norm",
    ops: 7,
    nodes: 7,
    status: "ingested",
    note: "the group-norm spelling, bound cleanly",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "logsoftmax",
    name: "LogSoftmaxHead",
    family: "head",
    ops: 9,
    nodes: 9,
    status: "ingested",
    note: "linear → log_softmax — the loss-head tail",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "tinycnn",
    name: "TinyCNN",
    family: "vision",
    ops: 11,
    nodes: 24,
    status: "census-only",
    note: "blocked on the pooling trio — max_pool2d / adaptive_avg_pool2d are unbound",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "losshead",
    name: "LossHead",
    family: "loss",
    ops: 8,
    nodes: 12,
    status: "census-only",
    note: "cross_entropy_loss / nll_loss_nd block every training-step graph",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "gru",
    name: "nn.GRU",
    family: "recurrent",
    ops: 10,
    nodes: 20,
    status: "census-only",
    note: "gru.input is exported but unbound",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "manual-nll",
    name: "ManualNLL",
    family: "loss",
    ops: 7,
    nodes: 11,
    status: "census-only",
    note: "the loss family is the second binding-gap backlog",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "fold",
    name: "nn.Fold",
    family: "vision",
    ops: 6,
    nodes: 14,
    status: "census-only",
    note: "col2im exports but the bridge has no binding for it",
    sourced: true,
    source: "workload-intake.md",
  },
  {
    id: "ctcloss",
    name: "nn.CTCLoss",
    family: "loss",
    ops: 5,
    nodes: 5,
    status: "rejected",
    note: "aten._ctc_loss.Tensor — a dynamic output shape torch.export refuses",
    sourced: true,
    source: "workload-intake.md",
  },
];

/** The three lowering defects the intake caught — boundary, not engine. */
export const DEFECTS: readonly { model: string; symptom: string; fix: string }[] = [
  {
    model: "nn.MultiheadAttention",
    symptom: "mat1 and mat2 shapes cannot be multiplied (16×16 and 4×48)",
    fix: "aten elides `chunk`'s dim default (0); the boundary minted no dim, so the binding read the last axis",
  },
  {
    model: "nn.TransformerDecoderLayer",
    symptom: "split_with_sizes expects [16, 32] to sum to 16",
    fix: "the same dropped `dim` default, spelled `split_with_sizes` on the cross-attention packed split",
  },
  {
    model: "nn.LSTM",
    symptom: "mat1 and mat2 must have the same dtype, Float and Double",
    fix: "`zeros`' `torch.dtype` kwarg was dropped, so h0/c0 materialised fp32 under the fp64 harness",
  },
];

/** One funnel row: a class and how many of the slice landed in it. */
export interface FunnelRow {
  key: IntakeStatus;
  label: string;
  count: number;
}

/** The intake browser, playable: the class counts are computed live. */
export class Intake {
  sel = 0;

  get workloads(): readonly IntakeWorkload[] {
    return INTAKE_WORKLOADS;
  }

  get picked(): IntakeWorkload {
    return INTAKE_WORKLOADS[this.sel];
  }

  pick(i: number): void {
    this.sel = Math.max(0, Math.min(INTAKE_WORKLOADS.length - 1, i));
  }

  /** The three classes and how many of the slice landed in each. */
  get classes(): { ingested: number; censusOnly: number; rejected: number; total: number } {
    let ingested = 0;
    let censusOnly = 0;
    let rejected = 0;
    for (const w of INTAKE_WORKLOADS) {
      if (w.status === "ingested") ingested += 1;
      else if (w.status === "census-only") censusOnly += 1;
      else rejected += 1;
    }
    return { ingested, censusOnly, rejected, total: INTAKE_WORKLOADS.length };
  }

  /** The funnel, in pipeline order. */
  get funnel(): FunnelRow[] {
    const c = this.classes;
    return [
      { key: "ingested", label: "ingested — bound, lowered, verified", count: c.ingested },
      { key: "census-only", label: "census-only — an op is unbound", count: c.censusOnly },
      { key: "rejected", label: "rejected — torch.export raised", count: c.rejected },
    ];
  }

  /** Total op nodes the slice spells. */
  get nodes(): number {
    return INTAKE_WORKLOADS.reduce((s, w) => s + w.nodes, 0);
  }

  /** The corpus headline (plan §D1), not the slice. */
  get corpus(): { total: number; verified: number; defects: number } {
    return { ...INTAKE };
  }
}
