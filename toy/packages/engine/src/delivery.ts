/** Delivery — the chosen program must *run*.
 *
 * Mirrors `project/retros/retnet-external-baselines.md` and
 * `project/retros/gen-kernels.md`: one extracted term, delivered
 * several ways, runs at wildly different speeds.  The fused carrier
 * lowering sits at **parity with the paper's own chunked
 * implementation** (2.08 vs 2.12 ms at T = 2048), and the `chainfuse`
 * program — `x@A@B@C` plus a 6-op pointwise tail — reached **2.94× vs
 * `torch.compile`** only once the search minted the fusion kernel.
 *
 * The **modeled** column is computed live from a dispatch-aware model
 * (`work + launches × LAUNCH_US + capture`); the **measured** column
 * is illustrative and sourced from the retros.  Delivery is the
 * execution dimension: the same semantics, a different runtime.
 */

export interface Delivery {
  /** `generic` | `batched` | `fused` | `graph` | `minted`. */
  kind: string;
  label: string;
  /** Modeled: the materialized work this delivery does (µs). */
  workUs: number;
  /** Modeled: how many kernel launches it issues. */
  dispatches: number;
  /** Modeled: a one-off capture/compile cost (µs). */
  captureUs: number;
  /** Measured wall-clock (µs) — illustrative, sourced. */
  measuredUs: number;
  /** Whether `measuredUs` is a real measurement (vs illustrative). */
  sourced: boolean;
  source: string;
  note: string;
}

export interface DeliveryCase {
  id: string;
  /** The one term every delivery computes. */
  term: string;
  shape: string;
  deliveries: Delivery[];
}

/** The modeled per-launch cost (µs). */
export const LAUNCH_US = 6;

/** The probe's representative terms (RTX 2050; fp64/fp32 per source). */
export const DELIVERY_CASES: readonly DeliveryCase[] = [
  {
    id: "retention",
    term: "chunked retention — seq(pre, loop(step, K), post)",
    shape: "T = 2048, K = 8, C = 256",
    deliveries: [
      {
        kind: "generic",
        label: "spelled-eager",
        workUs: 19000,
        dispatches: 96,
        captureUs: 0,
        measuredUs: 21730,
        sourced: true,
        source: "retnet-external-baselines.md",
        note: "the optimize-input model: many small ops, materialized decay vectors",
      },
      {
        kind: "batched",
        label: "carrier (K-way)",
        workUs: 2600,
        dispatches: 8,
        captureUs: 0,
        measuredUs: 3140,
        sourced: true,
        source: "retnet-external-baselines.md",
        note: "the extracted carrier, lowered to a batched scan — 5.66× memory, 6.9× speed",
      },
      {
        kind: "fused",
        label: "fused carrier lowering",
        workUs: 1700,
        dispatches: 8,
        captureUs: 0,
        measuredUs: 2080,
        sourced: true,
        source: "retnet-external-baselines.md",
        note: "parity with the paper's own chunked loop (2.08 vs 2.12 ms)",
      },
      {
        kind: "graph",
        label: "CUDA-graph replay",
        workUs: 19000,
        dispatches: 1,
        captureUs: 0,
        measuredUs: 20500,
        sourced: false,
        source: "illustrative",
        note: "one replay kills per-launch CPU overhead, not the asymptotics",
      },
      {
        kind: "minted",
        label: "minted scan kernel (FLA)",
        workUs: 150,
        dispatches: 4,
        captureUs: 0,
        measuredUs: 280,
        sourced: true,
        source: "retnet-external-baselines.md (fla-chunk, external)",
        note: "a hand-tuned kernel — the lowering-quality gap our general lowering does not close",
      },
    ],
  },
  {
    id: "chainfuse",
    term: "x@A@B@C + a 6-op pointwise tail",
    shape: "256 × 512, d = 512",
    deliveries: [
      {
        kind: "generic",
        label: "eager",
        workUs: 250,
        dispatches: 40,
        captureUs: 0,
        measuredUs: 315,
        sourced: true,
        source: "gen-kernels.md",
        note: "every op its own kernel — launch-count bound",
      },
      {
        kind: "batched",
        label: "torch.compile",
        workUs: 200,
        dispatches: 1,
        captureUs: 0,
        measuredUs: 253,
        sourced: true,
        source: "gen-kernels.md",
        note: "Inductor fuses the pointwise chain — but keeps all three GEMMs",
      },
      {
        kind: "fused",
        label: "laws only (fold + spelled tail)",
        workUs: 160,
        dispatches: 1,
        captureUs: 0,
        measuredUs: 181,
        sourced: true,
        source: "gen-kernels.md",
        note: "associativity folds the weight chain to one GEMM — already 1.34× vs compile",
      },
      {
        kind: "graph",
        label: "CUDA-graph replay",
        workUs: 250,
        dispatches: 1,
        captureUs: 0,
        measuredUs: 300,
        sourced: false,
        source: "illustrative",
        note: "replay trims launches; the fused tail is still 40 separate ops",
      },
      {
        kind: "minted",
        label: "laws + minted kernel (Triton)",
        workUs: 70,
        dispatches: 1,
        captureUs: 0,
        measuredUs: 86,
        sourced: true,
        source: "gen-kernels.md",
        note: "the search minted the fused tail kernel — 2.94× vs compile, verified 1.7e-05",
      },
    ],
  },
];

/** The delivery lab, playable: the modeled column is computed live. */
export class DeliveryLab {
  sel = 0;
  picked = 0;

  get cases(): readonly DeliveryCase[] {
    return DELIVERY_CASES;
  }

  get deliveryCase(): DeliveryCase {
    return DELIVERY_CASES[this.sel];
  }

  select(i: number): void {
    this.sel = Math.max(0, Math.min(DELIVERY_CASES.length - 1, i));
    this.picked = 0;
  }

  choose(k: number): void {
    this.picked = Math.max(0, Math.min(this.deliveryCase.deliveries.length - 1, k));
  }

  get delivery(): Delivery {
    return this.deliveryCase.deliveries[this.picked];
  }

  /** The dispatch-aware model's price — live. */
  modeledUs(d: Delivery): number {
    return d.workUs + d.dispatches * LAUNCH_US + d.captureUs;
  }

  /** The modeled pick and the measured winner. */
  get modelPick(): Delivery {
    return this.deliveryCase.deliveries.reduce((best, d) =>
      this.modeledUs(d) < this.modeledUs(best) ? d : best,
    );
  }

  get winner(): Delivery {
    return this.deliveryCase.deliveries.reduce((best, d) =>
      d.measuredUs < best.measuredUs ? d : best,
    );
  }

  /** The model's pick matches the clock. */
  get agrees(): boolean {
    return this.modelPick.kind === this.winner.kind;
  }

  /** The measured win of the best delivery over the generic one. */
  get speedup(): number {
    const generic = this.deliveryCase.deliveries.find((d) => d.kind === "generic")!;
    return generic.measuredUs / this.winner.measuredUs;
  }

  /** One row per delivery: the modeled price, the measured clock, the flags. */
  get rows(): { delivery: Delivery; modeledUs: number; modelPick: boolean; winner: boolean }[] {
    const pick = this.modelPick;
    const win = this.winner;
    return this.deliveryCase.deliveries.map((delivery) => ({
      delivery,
      modeledUs: this.modeledUs(delivery),
      modelPick: delivery.kind === pick.kind,
      winner: delivery.kind === win.kind,
    }));
  }
}
