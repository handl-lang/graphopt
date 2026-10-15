/** Pricing steers selection — and can be confidently wrong.
 *
 * Mirrors `project/retros/executor-pricing-fix.md` (+
 * `measured-cost-model.md`): the routed executor model priced the
 * *batched* carrier route as 20–30 % cheaper than the *generic* route,
 * while the measured wall-clock said the carrier was **1.9–2.7×
 * slower**.  The fix wired the measured profile back into the
 * delivered-price comparison — a pooled per-family correction
 * (`measured / modeled`) — and landed the measured-fastest member on
 * **12/12** probe cases.
 *
 * The route table's modeled and measured columns are the probe's
 * representative cases; the model's pick, the measured winner, the
 * pooled per-family correction and the calibrated pick are all
 * computed *live* here.  ADR 0003 holds: the cost model ranks, it
 * never decides equivalence.
 */

export interface Route {
  /** The executor family: `generic` | `batched` | `compiled`. */
  name: string;
  label: string;
  /** The uncalibrated routed model's price (µs). */
  modeled: number;
  /** The measured wall-clock (µs). */
  measured: number;
}

export interface RouteCase {
  id: string;
  shape: string;
  note: string;
  routes: Route[];
}

/** The probe's representative cases (RTX 2050, fp64). */
export const ROUTE_CASES: readonly RouteCase[] = [
  {
    id: "selective-ssm-small",
    shape: "SelectiveSSM · small",
    note: "an `add`-rooted IRModule (generic) vs the batched scan the model prefers",
    routes: [
      { name: "generic", label: "add / IRModule", modeled: 100, measured: 127 },
      { name: "batched", label: "apply / scan", modeled: 75, measured: 313 },
      { name: "compiled", label: "torch.compile", modeled: 90, measured: 236 },
    ],
  },
  {
    id: "diagonal-ssm-small",
    shape: "DiagonalSSM · small",
    note: "the same inversion on the diagonal recurrence",
    routes: [
      { name: "generic", label: "add / IRModule", modeled: 100, measured: 133 },
      { name: "batched", label: "applyd / scan", modeled: 70, measured: 290 },
      { name: "compiled", label: "torch.compile", modeled: 95, measured: 250 },
    ],
  },
  {
    id: "manual-softmax-small",
    shape: "ManualSoftmaxAttn · small",
    note: "a non-SSM case: the model and the clock already agree",
    routes: [
      { name: "generic", label: "add / IRModule", modeled: 100, measured: 130 },
      { name: "batched", label: "batched", modeled: 140, measured: 250 },
      { name: "compiled", label: "torch.compile", modeled: 120, measured: 205 },
    ],
  },
];

const argmin = (rs: readonly Route[], price: (r: Route) => number): Route =>
  rs.reduce((best, r) => (price(r) < price(best) ? r : best), rs[0]);

/** The route table, playable: the inversion is computed live. */
export class Pricing {
  sel = 0;
  calibrated = false;

  get cases(): readonly RouteCase[] {
    return ROUTE_CASES;
  }

  get routeCase(): RouteCase {
    return ROUTE_CASES[this.sel];
  }

  select(i: number): void {
    this.sel = Math.max(0, Math.min(ROUTE_CASES.length - 1, i));
  }

  toggle(): void {
    this.calibrated = !this.calibrated;
  }

  /**
   * The pooled per-family correction the probe emits: the geometric
   * mean of `measured / modeled` across every case — one factor per
   * executor family, not per case.  Computed live.
   */
  get factors(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const r of ROUTE_CASES[0].routes) {
      const ratios = ROUTE_CASES.map(
        (c) =>
          c.routes.find((x) => x.name === r.name)!.measured /
          c.routes.find((x) => x.name === r.name)!.modeled,
      );
      out[r.name] = Math.exp(ratios.reduce((p, q) => p + Math.log(q), 0) / ratios.length);
    }
    return out;
  }

  /** The price the model compares on — corrected when calibrated. */
  price(r: Route): number {
    return this.calibrated ? r.modeled * this.factors[r.name] : r.modeled;
  }

  /** The model's pick — the member the search would select. */
  pick(): Route {
    return argmin(this.routeCase.routes, (r) => this.price(r));
  }

  /** The measured winner — what actually ran fastest. */
  best(): Route {
    return argmin(this.routeCase.routes, (r) => r.measured);
  }

  /** The uncalibrated model's pick, for the before/after contrast. */
  modeledPick(): Route {
    return argmin(this.routeCase.routes, (r) => r.modeled);
  }

  /** True when the uncalibrated model's pick is not the measured winner. */
  get inverted(): boolean {
    return this.modeledPick().name !== this.best().name;
  }

  /** How much slower the model's pick runs than the measured winner. */
  get speedup(): number {
    return this.modeledPick().measured / this.best().measured;
  }

  /** How much cheaper the model *thinks* its pick is, in percent. */
  get modeledGap(): number {
    const pick = this.modeledPick();
    const best = this.best();
    return 100 * (1 - pick.modeled / best.modeled);
  }

  /** The current pick agrees with the clock. */
  get correct(): boolean {
    return this.pick().name === this.best().name;
  }

  /** One row per route: the two prices and the two verdicts. */
  get rows(): {
    route: Route;
    price: number;
    modelPick: boolean;
    winner: boolean;
    factor: number;
  }[] {
    const pick = this.pick();
    const best = this.best();
    return this.routeCase.routes.map((route) => ({
      route,
      price: this.price(route),
      modelPick: route.name === pick.name,
      winner: route.name === best.name,
      factor: this.factors[route.name],
    }));
  }

  /** How many of the probe's cases the current model gets right. */
  get correctCount(): { right: number; total: number } {
    const saved = this.sel;
    let right = 0;
    for (let i = 0; i < ROUTE_CASES.length; i++) {
      this.sel = i;
      if (this.correct) right += 1;
    }
    this.sel = saved;
    return { right, total: ROUTE_CASES.length };
  }
}
