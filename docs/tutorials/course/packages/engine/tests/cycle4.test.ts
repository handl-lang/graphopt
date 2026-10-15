import { describe, expect, it } from "vite-plus/test";

import {
  Bounded,
  DELIVERY_CASES,
  DeliveryLab,
  GAPS,
  Gaps,
  LAUNCH_US,
  MEMBERS,
  Pricing,
  ROUTE_CASES,
} from "../src/index.ts";

describe("pricing — the model can be wrong", () => {
  it("computes the routing inversion live on the SSM cases", () => {
    const p = new Pricing();
    expect(p.modeledPick().name).toBe("batched");
    expect(p.best().name).toBe("generic");
    expect(p.inverted).toBe(true);
    // measured 313 vs 127 — inside the probe's 1.9–2.7× band
    expect(p.speedup).toBeGreaterThan(1.9);
    expect(p.speedup).toBeLessThan(2.7);
    // the model thinks its pick is ~25 % cheaper
    expect(p.modeledGap).toBeCloseTo(25, 6);
  });

  it("pools one correction per family and flips the pick when calibrated", () => {
    const p = new Pricing();
    expect(p.factors.generic).toBeCloseTo(1.3, 1);
    expect(p.factors.batched).toBeGreaterThan(3);
    expect(p.correct).toBe(false);
    p.toggle();
    expect(p.calibrated).toBe(true);
    expect(p.pick().name).toBe("generic");
    expect(p.correct).toBe(true);
  });

  it("keeps the non-SSM case already-correct", () => {
    const p = new Pricing();
    p.select(2);
    expect(p.routeCase.id).toBe("manual-softmax-small");
    expect(p.modeledPick().name).toBe("generic");
    expect(p.inverted).toBe(false);
    expect(p.correct).toBe(true);
  });

  it("counts how many probe cases the model gets right", () => {
    const p = new Pricing();
    expect(p.correctCount).toEqual({ right: 1, total: ROUTE_CASES.length });
    p.toggle();
    expect(p.correctCount).toEqual({ right: ROUTE_CASES.length, total: ROUTE_CASES.length });
  });

  it("clamps the case selection and reads one row per route", () => {
    const p = new Pricing();
    p.select(99);
    expect(p.sel).toBe(ROUTE_CASES.length - 1);
    p.select(-3);
    expect(p.sel).toBe(0);
    expect(p.rows).toHaveLength(3);
    expect(p.rows.filter((r) => r.modelPick)).toHaveLength(1);
    expect(p.rows.filter((r) => r.winner)).toHaveLength(1);
  });
});

describe("delivery — the program must run", () => {
  it("computes the modeled column live from the dispatch-aware model", () => {
    const d = new DeliveryLab();
    const generic = d.deliveryCase.deliveries[0];
    expect(d.modeledUs(generic)).toBe(generic.workUs + generic.dispatches * LAUNCH_US);
    expect(LAUNCH_US).toBeGreaterThan(0);
  });

  it("agrees with the clock on both terms and prices the win", () => {
    const d = new DeliveryLab();
    expect(d.modelPick.kind).toBe("minted");
    expect(d.winner.kind).toBe("minted");
    expect(d.agrees).toBe(true);
    expect(d.speedup).toBeGreaterThan(3);
    d.select(1); // chainfuse
    expect(d.deliveryCase.id).toBe("chainfuse");
    expect(d.agrees).toBe(true);
    expect(d.speedup).toBeGreaterThan(3.5);
  });

  it("marks the illustrative deliveries and sources the measured ones", () => {
    const d = new DeliveryLab();
    const graph = d.deliveryCase.deliveries.find((x) => x.kind === "graph")!;
    expect(graph.sourced).toBe(false);
    expect(graph.source).toBe("illustrative");
    expect(d.deliveryCase.deliveries.filter((x) => x.sourced).length).toBe(4);
    expect(DELIVERY_CASES).toHaveLength(2);
  });

  it("flags the model's pick and the measured winner per row", () => {
    const d = new DeliveryLab();
    expect(d.rows).toHaveLength(5);
    expect(d.rows.filter((r) => r.modelPick)).toHaveLength(1);
    expect(d.rows.filter((r) => r.winner)).toHaveLength(1);
    d.choose(2);
    expect(d.delivery.kind).toBe("fused");
  });
});

describe("bounded — a certificate and an economics test", () => {
  it("certifies offers the pricing then refuses", () => {
    const b = new Bounded();
    b.select(5); // 3e-2
    expect(b.offered).toBe(56);
    expect(b.certified).toBe(56);
    expect(b.delivered).toBe(0);
    expect(b.pricedRefused).toBe(56);
    expect(b.gateClean).toBe(true);
    expect(b.verdict).toContain("pricing refused");
  });

  it("offers nothing at or below the smallest budgets", () => {
    const b = new Bounded();
    for (const i of [0, 1, 2, 3]) {
      b.select(i);
      expect(b.offered).toBe(0);
      expect(b.verdict).toContain("no offer");
    }
    b.select(4); // 1e-2 — exactly the qkv class
    expect(b.offered).toBe(20);
    expect(b.delivered).toBe(0);
  });

  it("prices the bounded member 1.32× the dense one", () => {
    const b = new Bounded();
    b.select(5);
    expect(b.rows[0].ratio).toBeCloseTo(1.32, 2);
    expect(b.rows[0].delivered).toBe(false);
    expect(MEMBERS[0].dedup).toBeLessThan(0.01);
  });

  it("delivers once the dedup is thick — the next target", () => {
    const b = new Bounded();
    b.select(5);
    b.toggle();
    expect(b.thick).toBe(true);
    expect(b.delivered).toBe(56);
    expect(b.pricedRefused).toBe(0);
    expect(b.verdict).toContain("beat dense");
    // the bound gate is clean in both regimes — it is the pricing that moves
    expect(b.gateClean).toBe(true);
  });
});

describe("improve — the payoff", () => {
  it("maps each gap to a blocker, a next measurement and a level", () => {
    expect(GAPS.length).toBeGreaterThanOrEqual(6);
    for (const g of GAPS) {
      expect(g.blocker.length).toBeGreaterThan(0);
      expect(g.next.length).toBeGreaterThan(0);
      expect(g.levelNo).toBeGreaterThanOrEqual(0);
    }
  });

  it("reads the three-step connection for the picked gap", () => {
    const g = new Gaps();
    expect(g.gap.level).toBe("mint");
    const c = g.connection;
    expect(c).toHaveLength(3);
    expect(c[0].next).toContain("level 15");
    g.pick(99);
    expect(g.gap.id).toBe("weight-space");
    expect(g.gap.level).toBe("honest");
  });
});
