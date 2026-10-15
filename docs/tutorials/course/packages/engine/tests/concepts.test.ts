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
  glossary,
  introductionIndex,
  levelIndex,
  prerequisites,
  taughtAt,
  undefinedConcepts,
} from "../src/index.ts";

/** The measured forward-reference count after step 2.  It may only go
 * down: every change to the graph or the level text must keep the graph
 * free of terms used before they are defined. */
const FORWARD_REF_CEILING = 0;

describe("concept graph — shape", () => {
  it("runs the 28 levels, in course order, with unique ids", () => {
    expect(LEVEL_ORDER).toHaveLength(28);
    expect(LEVEL_ORDER[0]).toBe("tree");
    expect(LEVEL_ORDER.at(-1)).toBe("improve");
    expect(new Set(LEVEL_ORDER).size).toBe(LEVEL_ORDER.length);
    // the six coverage levels (plan 0028 §D) are in their module's position
    for (const id of ["corpus", "models", "artifacts", "cellstore", "game", "gates"]) {
      expect(LEVEL_ORDER).toContain(id);
    }
    expect(LEVEL_ORDER.indexOf("corpus")).toBeLessThan(LEVEL_ORDER.indexOf("reach"));
    expect(LEVEL_ORDER.indexOf("models")).toBeGreaterThan(LEVEL_ORDER.indexOf("measure"));
    expect(LEVEL_ORDER.indexOf("models")).toBeLessThan(LEVEL_ORDER.indexOf("architecture"));
    expect(LEVEL_ORDER.indexOf("gates")).toBeLessThan(LEVEL_ORDER.indexOf("honest"));
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

  it("teaches every concept at its introducing level", () => {
    for (const c of CONCEPTS) {
      expect(taughtAt(c)).toBe(c.introduced);
    }
  });
});

describe("concept graph — lookups", () => {
  it("finds a concept by id and a level by index", () => {
    expect(conceptById("reach")?.introduced).toBe("reach");
    expect(conceptById("nope")).toBeUndefined();
    expect(levelIndex("tree")).toBe(0);
    expect(levelIndex("improve")).toBe(27);
    expect(levelIndex("nope")).toBe(-1);
  });

  it("reports a concept's first use and introduction index", () => {
    const referee = conceptById("referee")!;
    expect(firstUsed(referee)).toBe("referee");
    expect(introductionIndex(referee)).toBe(5);
    const closure = conceptById("closure")!;
    expect(firstUsed(closure)).toBe("reach");
    expect(introductionIndex(closure)).toBe(3);
  });

  it("lists concepts a level uses and introduces", () => {
    expect(conceptsAt("structure").map((c) => c.id)).toContain("law");
    expect(conceptsIntroducedAt("substrate").map((c) => c.id)).toContain("morphism");
    expect(conceptsIntroducedAt("nope")).toHaveLength(0);
  });
});

describe("concept graph — forward references (the ordering gate)", () => {
  it("has no concept used before it is introduced", () => {
    // 22 → 0 in step 2; the ceiling may only go down.
    expect(forwardRefs().length).toBeLessThanOrEqual(FORWARD_REF_CEILING);
    expect(forwardRefs()).toHaveLength(0);
  });

  it("never flags a term that used to be a forward reference", () => {
    const flagged = new Set(forwardRefs().map((r) => r.id));
    for (const id of [
      "closure",
      "monoid",
      "comonoid",
      "non-dominated-set",
      "critical-pair",
      "monoidal-product",
      "e-graph",
      "carrier",
      "ranking",
      "dispatch",
      "guard",
      "adversarial-sweep",
      "minting",
      "admission",
      "census",
      "provenance",
      "cost-model",
      "learned-policy",
      "legal-move",
      "referee",
      "oracle",
      "features",
    ]) {
      expect(flagged.has(id)).toBe(false);
    }
  });

  it("defines every concept at or before its first use", () => {
    for (const c of CONCEPTS) {
      const first = firstUsed(c)!;
      expect(introductionIndex(c)).toBeLessThanOrEqual(levelIndex(first));
    }
  });

  it("has no concept left undefined", () => {
    expect(undefinedConcepts()).toHaveLength(0);
    for (const c of CONCEPTS) expect(c.introduced).not.toBeNull();
  });
});

describe("concept graph — prerequisites (the ladder is not flat)", () => {
  it("gives the entry level no prerequisites", () => {
    expect(prerequisites("tree")).toHaveLength(0);
  });

  it("only ever depends on earlier levels", () => {
    for (const l of LEVEL_ORDER) {
      for (const p of prerequisites(l)) {
        expect(levelIndex(p)).toBeLessThan(levelIndex(l));
      }
    }
  });

  it("links a level to the levels that introduce the concepts it uses", () => {
    // the referee uses semantic-equality (introduced at `equal`)
    expect(prerequisites("referee")).toContain("equal");
    // `measure` introduces the cost model but uses features (introduced at learn)
    expect(prerequisites("measure")).toContain("learn");
    // `pricing` reaches back across the whole ladder
    expect(prerequisites("pricing")).toContain("measure");
    expect(prerequisites("pricing")).toContain("deliver");
  });

  it("returns prerequisites in curriculum order, without duplicates", () => {
    const p = prerequisites("improve");
    expect(new Set(p).size).toBe(p.length);
    for (let i = 1; i < p.length; i++) {
      expect(levelIndex(p[i - 1])).toBeLessThan(levelIndex(p[i]));
    }
    expect(prerequisites("nope")).toHaveLength(0);
  });
});

describe("concept graph — the glossary", () => {
  it("lists every concept exactly once", () => {
    const rows = glossary();
    expect(rows).toHaveLength(CONCEPTS.length);
    expect(new Set(rows.map((r) => r.id)).size).toBe(CONCEPTS.length);
    expect(new Set(rows.map((r) => r.id))).toEqual(new Set(CONCEPTS.map((c) => c.id)));
  });

  it("gives every row a label, a definition and a real home level", () => {
    for (const r of glossary()) {
      expect(r.label.length).toBeGreaterThan(0);
      expect(r.def.length).toBeGreaterThan(0);
      expect(LEVEL_ORDER).toContain(r.introduced);
      expect(r.index).toBe(levelIndex(r.introduced));
      expect(r.used.length).toBeGreaterThan(0);
    }
  });

  it("orders rows by the level that introduces them", () => {
    const rows = glossary();
    for (let i = 1; i < rows.length; i++) {
      expect(rows[i - 1].index).toBeLessThanOrEqual(rows[i].index);
    }
    expect(rows[0].index).toBe(0);
  });
});
