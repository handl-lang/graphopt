import { describe, expect, it } from "vite-plus/test";

import {
  CONCEPTS,
  LEVEL_ORDER,
  conceptById,
  conceptsAt,
  conceptsIntroducedAt,
  conceptsTaughtAt,
  firstUsed,
  forwardRefs,
  introductionIndex,
  levelIndex,
  taughtAt,
  undefinedConcepts,
} from "../src/index.ts";

describe("concept graph — shape", () => {
  it("runs the 22 levels, in course order, with unique ids", () => {
    expect(LEVEL_ORDER).toHaveLength(22);
    expect(LEVEL_ORDER[0]).toBe("tree");
    expect(LEVEL_ORDER.at(-1)).toBe("improve");
    expect(new Set(LEVEL_ORDER).size).toBe(LEVEL_ORDER.length);
  });

  it("gives every concept a unique id, a label and a one-line definition", () => {
    expect(CONCEPTS.length).toBeGreaterThan(30);
    expect(new Set(CONCEPTS.map((c) => c.id)).size).toBe(CONCEPTS.length);
    for (const c of CONCEPTS) {
      expect(c.label.length).toBeGreaterThan(0);
      expect(c.def.length).toBeGreaterThan(0);
    }
  });

  it("only names levels that exist, and every concept is used somewhere", () => {
    for (const c of CONCEPTS) {
      expect(c.used.length).toBeGreaterThan(0);
      for (const l of c.used) expect(LEVEL_ORDER).toContain(l);
      if (c.introduced !== null) expect(LEVEL_ORDER).toContain(c.introduced);
    }
  });

  it("uses each concept at the level that introduces it", () => {
    for (const c of CONCEPTS) {
      if (c.introduced !== null) expect(c.used).toContain(c.introduced);
    }
  });

  it("covers every level: each is used by at least one concept", () => {
    for (const l of LEVEL_ORDER) {
      expect(conceptsAt(l).length).toBeGreaterThan(0);
    }
  });

  it("introduces at least one concept at every level", () => {
    for (const l of LEVEL_ORDER) {
      expect(conceptsIntroducedAt(l).length).toBeGreaterThan(0);
    }
  });

  it("attributes every concept to exactly one teaching level", () => {
    const taught = CONCEPTS.map((c) => taughtAt(c));
    expect(taught.every((l) => l !== null && LEVEL_ORDER.includes(l))).toBe(true);
    for (const l of LEVEL_ORDER) {
      expect(conceptsTaughtAt(l).length).toBeGreaterThan(0);
    }
    const counted = LEVEL_ORDER.flatMap((l) => conceptsTaughtAt(l).map((c) => c.id));
    expect(counted).toHaveLength(CONCEPTS.length);
    expect(new Set(counted).size).toBe(CONCEPTS.length);
  });

  it("falls back to the first-use level when a concept has no introduction", () => {
    const closure = conceptById("closure")!;
    expect(taughtAt(closure)).toBe("reach");
    expect(conceptsTaughtAt("reach").map((c) => c.id)).toContain("closure");
  });
});

describe("concept graph — lookups", () => {
  it("finds a concept by id and a level by index", () => {
    expect(conceptById("reach")?.introduced).toBe("reach");
    expect(conceptById("nope")).toBeUndefined();
    expect(levelIndex("tree")).toBe(0);
    expect(levelIndex("improve")).toBe(21);
    expect(levelIndex("nope")).toBe(-1);
  });

  it("reports a concept's first use and introduction index", () => {
    const referee = conceptById("referee")!;
    expect(firstUsed(referee)).toBe("structure");
    expect(introductionIndex(referee)).toBe(4);
    const closure = conceptById("closure")!;
    expect(firstUsed(closure)).toBe("reach");
    expect(introductionIndex(closure)).toBe(Number.POSITIVE_INFINITY);
  });

  it("lists concepts a level uses and introduces", () => {
    expect(conceptsAt("structure").map((c) => c.id)).toContain("law");
    expect(conceptsIntroducedAt("substrate").map((c) => c.id)).toContain("morphism");
    expect(conceptsIntroducedAt("nope")).toHaveLength(0);
  });
});

describe("concept graph — forward references (the ordering bugs)", () => {
  it("flags the terms used before they are defined", () => {
    const ids = forwardRefs().map((r) => r.id);
    // the plan's audit, plus the ones the prose adds
    for (const id of [
      "closure",
      "monoid",
      "comonoid",
      "non-dominated-set",
      "critical-pair",
      "monoidal-product",
      "referee",
      "oracle",
      "cost-model",
      "features",
      "minting",
      "admission",
      "census",
      "provenance",
      "guard",
      "adversarial-sweep",
      "learned-policy",
      "dispatch",
      "e-graph",
      "carrier",
    ]) {
      expect(ids).toContain(id);
    }
  });

  it("records the plan's five terms as used at their audited levels", () => {
    const by = new Map(forwardRefs().map((r) => [r.id, r]));
    expect(by.get("closure")?.firstUsedAt).toBe("reach");
    expect(by.get("monoid")?.firstUsedAt).toBe("structure");
    expect(by.get("non-dominated-set")?.firstUsedAt).toBe("choice");
    expect(by.get("critical-pair")?.firstUsedAt).toBe("coherence");
    expect(by.get("monoidal-product")?.firstUsedAt).toBe("substrate");
  });

  it("treats a term no level defines as introduced at infinity", () => {
    const closure = forwardRefs().find((r) => r.id === "closure")!;
    expect(closure.introducedAt).toBeNull();
    expect(closure.gap).toBe(Number.POSITIVE_INFINITY);
  });

  it("measures a defined-later term by its gap, worst first", () => {
    const refs = forwardRefs();
    for (let i = 1; i < refs.length; i++) {
      expect(refs[i - 1].gap).toBeGreaterThanOrEqual(refs[i].gap);
    }
    const dispatch = refs.find((r) => r.id === "dispatch")!;
    expect(dispatch.firstUsedAt).toBe("measure");
    expect(dispatch.introducedAt).toBe("deliver");
    expect(dispatch.gap).toBe(11);
  });

  it("never flags a concept that is defined before or at first use", () => {
    const flagged = new Set(forwardRefs().map((r) => r.id));
    expect(flagged.has("certificate")).toBe(false);
    expect(flagged.has("morphism")).toBe(false);
    expect(flagged.has("exposure")).toBe(false);
  });

  it("lists the concepts no level defines yet", () => {
    const ids = undefinedConcepts().map((c) => c.id);
    expect(ids).toContain("closure");
    expect(ids).toContain("e-graph");
    expect(ids).toContain("carrier");
    expect(ids).not.toContain("certificate");
  });
});
