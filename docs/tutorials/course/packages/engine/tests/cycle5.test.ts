import { describe, expect, it } from "vite-plus/test";

import {
  ARTIFACTS,
  Artifacts,
  BASE_FRAGMENTS,
  CellStoreLab,
  CostModels,
  DEFECTS,
  DOMAINS,
  EPISODE,
  GATES,
  GameLab,
  GateLab,
  INTAKE,
  INTAKE_WORKLOADS,
  Intake,
  LAUNCH_US,
  MODEL_CASES,
  MODELS,
  NO_SUPPRESSIONS,
  ORIGINS,
  PLAYERS,
  cellKey,
  handlKind,
} from "../src/index.ts";

describe("corpus — the intake funnel", () => {
  it("computes the three-way class split live over the slice", () => {
    const i = new Intake();
    expect(i.classes.total).toBe(INTAKE_WORKLOADS.length);
    expect(i.classes.ingested).toBe(6);
    expect(i.classes.censusOnly).toBe(5);
    expect(i.classes.rejected).toBe(1);
    expect(i.classes.ingested + i.classes.censusOnly + i.classes.rejected).toBe(i.classes.total);
    expect(i.funnel.map((f) => f.count)).toEqual([6, 5, 1]);
  });

  it("cites the plan's corpus totals and the three lowering defects", () => {
    const i = new Intake();
    expect(i.corpus.total).toBe(166);
    expect(i.corpus.verified).toBe(114);
    expect(i.corpus.defects).toBe(3);
    expect(DEFECTS).toHaveLength(3);
    expect(DEFECTS.map((d) => d.model)).toContain("nn.MultiheadAttention");
  });

  it("selects a workload and clamps the index", () => {
    const i = new Intake();
    expect(i.picked.status).toBe("ingested");
    i.pick(99);
    expect(i.picked.status).toBe("rejected");
    i.pick(-4);
    expect(i.picked.id).toBe("encoder-layer");
    expect(i.nodes).toBeGreaterThan(0);
  });
});

describe("models — the cost-model family", () => {
  it("prices the roofline, the executor route and the delivered route", () => {
    const m = new CostModels();
    const generic = m.costCase.routes[0];
    expect(m.roofline()).toBe(Math.max(m.costCase.computeUs, m.costCase.memoryUs));
    expect(m.executor(generic)).toBe(generic.workUs + generic.dispatches * LAUNCH_US);
    expect(MODELS).toHaveLength(4);
    expect(MODEL_CASES).toHaveLength(2);
  });

  it("shows the inversion: the uncalibrated delivered model disagrees with the clock", () => {
    const m = new CostModels();
    expect(m.costCase.id).toBe("ssm");
    expect(m.deliveredPick.kind).toBe("batched");
    expect(m.measuredWinner.kind).toBe("generic");
    expect(m.inverted).toBe(true);
    expect(m.correct).toBe(false);
  });

  it("flips the pick to the measured winner when calibrated", () => {
    const m = new CostModels();
    m.toggle();
    expect(m.calibrated).toBe(true);
    expect(m.deliveredPick.kind).toBe("generic");
    expect(m.correct).toBe(true);
    expect(m.inverted).toBe(false);
    expect(m.rows.filter((r) => r.deliveredPick)).toHaveLength(1);
    expect(m.rows.filter((r) => r.winner)).toHaveLength(1);
  });

  it("keeps the already-correct case correct", () => {
    const m = new CostModels();
    m.select(1);
    expect(m.costCase.id).toBe("attn");
    expect(m.measuredWinner.kind).toBe("batched");
    expect(m.inverted).toBe(false);
    expect(m.correct).toBe(true);
  });
});

describe("artifacts — what 'runnable' means", () => {
  it("lists every delivery artifact and what each needs to run", () => {
    const a = new Artifacts();
    expect(a.artifacts.length).toBeGreaterThanOrEqual(6);
    expect(a.needs.length).toBeGreaterThan(0);
    for (const art of ARTIFACTS) {
      expect(art.needs.length).toBeGreaterThan(0);
      expect(art.note.length).toBeGreaterThan(0);
    }
  });

  it("marks the standalone formats — `.pt2` and torchscript", () => {
    const a = new Artifacts();
    expect(a.portable.id).toBe("pt2");
    expect(a.portable.standalone).toBe(true);
    expect(a.standaloneCount).toBe(2);
    expect(ARTIFACTS.find((x) => x.id === "safetensors")!.standalone).toBe(false);
    expect(ARTIFACTS.find((x) => x.id === "cuda-graph")!.needs).toContain("a CUDA device");
  });

  it("selects an artifact and clamps the index", () => {
    const a = new Artifacts();
    a.pick(99);
    expect(a.artifact.id).toBe("native");
    a.pick(-1);
    expect(a.artifact.id).toBe("module");
  });
});

describe("cellstore — one record per idea", () => {
  it("assembles the store from fragments, one cell per item", () => {
    const c = new CellStoreLab();
    expect(c.cells.length).toBe(BASE_FRAGMENTS.flatMap((f) => f.cells).length);
    expect(c.size).toBe(c.cells.length); // no duplicates in the base
    expect(c.conflict).toBeNull();
    expect(new Set(c.cells.map(cellKey)).size).toBe(c.size);
  });

  it("raises on a duplicate (role, name) rather than silently overriding", () => {
    const c = new CellStoreLab();
    const before = c.size;
    c.addDuplicate();
    expect(c.conflict).not.toBeNull();
    expect(c.conflict?.name).toBe("assoc");
    expect(c.conflict?.role).toBe("law");
    expect(c.size).toBe(before); // the conflicting fragment was not merged
  });

  it("reads the provenance origins and the handl row per role", () => {
    const c = new CellStoreLab();
    expect(c.origins).toContain("shipped");
    expect(c.origins).toContain("minted");
    expect(c.origins).not.toContain("admitted");
    expect(ORIGINS).toHaveLength(6);
    expect(c.byRole.find((r) => r.role === "op")!.handl).toBe("AstShapeDef");
    expect(c.byRole.find((r) => r.role === "law")!.handl).toBe("TransformDef");
    expect(c.byRole.find((r) => r.role === "handle")!.handl).toBe("!Native");
    expect(handlKind("object")).toBe("TransformDef");
  });
});

describe("game — boards, players, priors", () => {
  it("steps the meta-arena episode to its terminal extract", () => {
    const g = new GameLab();
    expect(g.total).toBe(EPISODE.length);
    expect(g.at.move).toBe("saturate");
    expect(g.done).toBe(false);
    g.next();
    expect(g.at.move).toBe("declare");
    while (!g.done) g.next();
    expect(g.at.move).toBe("extract");
    expect(g.reward).toBeGreaterThan(0);
    g.reset();
    expect(g.step).toBe(0);
  });

  it("registers the six play domains", () => {
    const g = new GameLab();
    expect(g.domains).toEqual(DOMAINS);
    expect(g.domains).toContain("open");
  });

  it("reports what the learned prior buys over the heuristic", () => {
    const g = new GameLab();
    expect(PLAYERS.find((p) => p.id === "learned")!.ap).toBeCloseTo(0.133, 6);
    expect(PLAYERS.find((p) => p.id === "heuristic")!.ap).toBeCloseTo(0.045, 6);
    expect(g.best.id).toBe("learned");
    expect(g.priorGain).toBeGreaterThan(2);
    expect(g.priorComparison.learned.ap).toBeGreaterThan(g.priorComparison.heuristic.ap);
  });
});

describe("gates — how the project stays honest", () => {
  it("runs every gate green, then fails one when broken", () => {
    const g = new GateLab();
    expect(g.gates).toHaveLength(GATES.length);
    expect(g.passing).toBe(true);
    expect(g.report).toContain("all");
    g.break("coverage");
    expect(g.passing).toBe(false);
    expect(g.report).toContain("FAIL");
    expect(g.brokenGate?.id).toBe("coverage");
    expect(g.passingCount).toBe(GATES.length - 1);
    g.repair();
    expect(g.passing).toBe(true);
  });

  it("ignores an unknown gate and names the no-suppressions rule", () => {
    const g = new GateLab();
    g.break("not-a-gate");
    expect(g.passing).toBe(true);
    expect(NO_SUPPRESSIONS).toContain("never silenced");
  });

  it("lists the gates the project actually runs", () => {
    const ids = GATES.map((g) => g.id);
    for (const id of [
      "coverage",
      "ty",
      "ruff",
      "vulture",
      "import-linter",
      "bandit",
      "radon",
      "bench",
    ]) {
      expect(ids).toContain(id);
    }
    expect(INTAKE.total).toBeGreaterThan(0);
  });
});
