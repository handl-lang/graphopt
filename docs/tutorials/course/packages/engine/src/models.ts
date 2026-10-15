/** The cost-model family — one model is not enough.
 *
 * Plan 0028 §D2: the real system prices a term with a **family** of
 * models, each answering a different question — static `features`
 * describe, a `roofline` bounds, an `executor` adds the launch cost of
 * the route it would take, and **delivered** pricing bills the term
 * under the lowering it would actually be delivered by.  A measured
 * `TargetProfile` re-targets all of them to a machine.
 *
 * The live demonstration is the executor-routing inversion
 * (`project/retros/measured-cost-model.md`): the uncalibrated delivered
 * model prices the batched carrier 20–30 % cheaper and measures it
 * 1.9–2.7× slower; feeding measured corrections back lands the
 * measured-fastest route.  The roofline/executor terms are the toy's
 * static model; the measured column is sourced from the retro.
 */

import { LAUNCH_US } from "./delivery.ts";

/** The model ids the family exposes, in the order the level shows them. */
export type ModelKey = "features" | "roofline" | "executor" | "delivered";

/** One delivery route a term can take, with its modeled and measured price. */
export interface ModelRoute {
  kind: "generic" | "batched";
  label: string;
  /** Modeled: the materialised work this route does (µs). */
  workUs: number;
  /** Modeled: how many launches it issues. */
  dispatches: number;
  /** Measured wall-clock (µs) — sourced from the retro. */
  measuredUs: number;
  /** The measured correction applied under calibration (measured / modeled). */
  correction: number;
  note: string;
}

/** One case: a term, its static features, and the routes it can take. */
export interface CostModelCase {
  id: string;
  term: string;
  shape: string;
  /** Static features — no execution. */
  flops: number;
  bytes: number;
  /** The roofline's two bounds (µs). */
  computeUs: number;
  memoryUs: number;
  routes: ModelRoute[];
}

/** The probe's representative cases (RTX 2050; `measured-cost-model.md`). */
export const MODEL_CASES: readonly CostModelCase[] = [
  {
    id: "ssm",
    term: "SelectiveSSM small",
    shape: "B = 4, T = 64, d = 32",
    flops: 2.1e8,
    bytes: 3.4e6,
    computeUs: 158,
    memoryUs: 40,
    routes: [
      {
        kind: "generic",
        label: "add-rooted → IRModule",
        workUs: 158,
        dispatches: 8,
        measuredUs: 158,
        correction: 0.85,
        note: "the serial IRModule eval — the measured-fastest member",
      },
      {
        kind: "batched",
        label: "apply[d] → BatchedScanModule",
        workUs: 90,
        dispatches: 12,
        measuredUs: 366,
        correction: 2.5,
        note: "the level-schedule's gathers/copies dominate at these dims",
      },
    ],
  },
  {
    id: "attn",
    term: "ManualSoftmaxAttn small",
    shape: "B = 2, T = 64, d = 64",
    flops: 8.0e7,
    bytes: 2.1e6,
    computeUs: 107,
    memoryUs: 30,
    routes: [
      {
        kind: "generic",
        label: "linear → IRModule",
        workUs: 107,
        dispatches: 9,
        measuredUs: 107,
        correction: 1.0,
        note: "the pipeline's real win — already measured-best",
      },
      {
        kind: "batched",
        label: "batched delivery",
        workUs: 60,
        dispatches: 11,
        measuredUs: 90,
        correction: 0.9,
        note: "the batched route is genuinely faster here",
      },
    ],
  },
];

/** The four models, in the order the level presents them. */
export const MODELS: readonly { key: ModelKey; label: string; question: string }[] = [
  { key: "features", label: "features", question: "what does the program look like?" },
  { key: "roofline", label: "roofline", question: "what bounds it — compute or memory?" },
  { key: "executor", label: "executor", question: "what does its route's launch cost add?" },
  { key: "delivered", label: "delivered", question: "what does it cost as it would run?" },
];

/** The cost-model family, playable: every price is computed live. */
export class CostModels {
  sel = 0;
  /** Whether the delivered model applies the measured corrections. */
  calibrated = false;

  get cases(): readonly CostModelCase[] {
    return MODEL_CASES;
  }

  get costCase(): CostModelCase {
    return MODEL_CASES[this.sel];
  }

  select(i: number): void {
    this.sel = Math.max(0, Math.min(MODEL_CASES.length - 1, i));
  }

  toggle(): void {
    this.calibrated = !this.calibrated;
  }

  /** The static features — no execution. */
  features(c: CostModelCase = this.costCase): { flops: number; bytes: number; intensity: number } {
    return { flops: c.flops, bytes: c.bytes, intensity: c.flops / c.bytes };
  }

  /** The roofline bound — `max(compute, memory)`, route-independent. */
  roofline(c: CostModelCase = this.costCase): number {
    return Math.max(c.computeUs, c.memoryUs);
  }

  /** The executor route price — the route's work plus its launches. */
  executor(route: ModelRoute): number {
    return route.workUs + route.dispatches * LAUNCH_US;
  }

  /** The delivered price — the executor price, corrected by the measured factor. */
  delivered(route: ModelRoute, _c: CostModelCase = this.costCase): number {
    const factor = this.calibrated ? route.correction : 1;
    return this.executor(route) * factor;
  }

  /** A model's price for a route; the roofline does not see the route. */
  price(model: ModelKey, route: ModelRoute, c: CostModelCase = this.costCase): number {
    if (model === "executor") return this.executor(route);
    if (model === "delivered") return this.delivered(route, c);
    return this.roofline(c);
  }

  /** The route a model picks — `null` for the route-independent roofline. */
  pick(model: ModelKey, c: CostModelCase = this.costCase): ModelRoute | null {
    if (model === "features" || model === "roofline") return null;
    return c.routes.reduce((best, r) =>
      this.price(model, r, c) < this.price(model, best, c) ? r : best,
    );
  }

  /** The measured-fastest route. */
  get measuredWinner(): ModelRoute {
    return this.costCase.routes.reduce((best, r) => (r.measuredUs < best.measuredUs ? r : best));
  }

  /** The route delivered pricing picks right now. */
  get deliveredPick(): ModelRoute {
    return this.pick("delivered")!;
  }

  /** Whether the (uncalibrated) delivered pick disagrees with the clock. */
  get inverted(): boolean {
    return !this.calibrated && this.deliveredPick.kind !== this.measuredWinner.kind;
  }

  /** Whether the delivered pick agrees with the clock. */
  get correct(): boolean {
    return this.deliveredPick.kind === this.measuredWinner.kind;
  }

  /** One row per route: every model's price, and the picks. */
  get rows(): {
    route: ModelRoute;
    roofline: number;
    executor: number;
    delivered: number;
    deliveredPick: boolean;
    winner: boolean;
  }[] {
    const pick = this.deliveredPick;
    const win = this.measuredWinner;
    return this.costCase.routes.map((route) => ({
      route,
      roofline: this.roofline(),
      executor: this.executor(route),
      delivered: this.delivered(route),
      deliveredPick: route.kind === pick.kind,
      winner: route.kind === win.kind,
    }));
  }
}
