import { MODELS, CostModels as Engine } from "engine";
import type { CostModelCase, ModelKey, ModelRoute } from "engine";

/** The cost-model family, playable: every price is computed live. */
export class CostModelFamily {
  private m = new Engine();

  get cases(): readonly CostModelCase[] {
    return this.m.cases;
  }

  get sel(): number {
    return this.m.sel;
  }

  get costCase(): CostModelCase {
    return this.m.costCase;
  }

  get calibrated(): boolean {
    return this.m.calibrated;
  }

  get models(): readonly { key: ModelKey; label: string; question: string }[] {
    return MODELS;
  }

  get rows(): {
    route: ModelRoute;
    roofline: number;
    executor: number;
    delivered: number;
    deliveredPick: boolean;
    winner: boolean;
  }[] {
    return this.m.rows;
  }

  get roofline(): number {
    return this.m.roofline();
  }

  get features(): { flops: number; bytes: number; intensity: number } {
    return this.m.features();
  }

  get measuredWinner(): ModelRoute {
    return this.m.measuredWinner;
  }

  get deliveredPick(): ModelRoute {
    return this.m.deliveredPick;
  }

  get inverted(): boolean {
    return this.m.inverted;
  }

  get correct(): boolean {
    return this.m.correct;
  }

  select(i: number): void {
    this.m.select(i);
  }

  toggle(): void {
    this.m.toggle();
  }

  /** A model's pick label — `—` for the route-independent roofline. */
  pick(model: ModelKey): string {
    const r = this.m.pick(model);
    return r ? r.label : "— (route-independent)";
  }
}
