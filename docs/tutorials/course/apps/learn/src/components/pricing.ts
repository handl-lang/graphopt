import { Pricing as Engine } from "engine";
import type { Route, RouteCase } from "engine";

/** The route table: the model's pick vs the measured winner, live. */
export class Pricing {
  private p = new Engine();

  get cases(): readonly RouteCase[] {
    return this.p.cases;
  }

  get sel(): number {
    return this.p.sel;
  }

  get routeCase(): RouteCase {
    return this.p.routeCase;
  }

  get calibrated(): boolean {
    return this.p.calibrated;
  }

  get factors(): Record<string, number> {
    return this.p.factors;
  }

  get rows(): {
    route: Route;
    price: number;
    modelPick: boolean;
    winner: boolean;
    factor: number;
  }[] {
    return this.p.rows;
  }

  get pick(): Route {
    return this.p.pick();
  }

  get best(): Route {
    return this.p.best();
  }

  get inverted(): boolean {
    return this.p.inverted;
  }

  get speedup(): number {
    return this.p.speedup;
  }

  get modeledGap(): number {
    return this.p.modeledGap;
  }

  get correct(): boolean {
    return this.p.correct;
  }

  get correctCount(): { right: number; total: number } {
    return this.p.correctCount;
  }

  select(i: number): void {
    this.p.select(i);
  }

  toggle(): void {
    this.p.toggle();
  }
}
