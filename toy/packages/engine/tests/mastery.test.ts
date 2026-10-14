import { describe, expect, it } from "vite-plus/test";

import {
  CONCEPTS,
  MASTERED_AT,
  MASTERED_MIN,
  allFamiliar,
  conceptById,
  emptyMastery,
  masteredCount,
  record,
  recordAll,
  score,
  summary,
  tier,
  weakest,
} from "../src/index.ts";

/** Fold a list of right/wrong answers into a state for one concept. */
function tally(concept: string, answers: boolean[]) {
  return answers.reduce((s, ok) => record(s, concept, ok), emptyMastery());
}

describe("mastery — tallying", () => {
  it("starts empty and immutable", () => {
    const a = record(emptyMastery(), "law", true);
    const b = record(a, "law", false);
    expect(a.law).toEqual({ correct: 1, attempted: 1 });
    expect(b.law).toEqual({ correct: 1, attempted: 2 });
    expect(emptyMastery()).toEqual({});
  });

  it("records one answer across several concepts", () => {
    const s = recordAll(emptyMastery(), ["law", "reach"], true);
    expect(s.law).toEqual({ correct: 1, attempted: 1 });
    expect(s.reach).toEqual({ correct: 1, attempted: 1 });
  });

  it("scores a concept as its correctness ratio, zero with no attempts", () => {
    expect(score(undefined)).toBe(0);
    expect(score({ correct: 0, attempted: 0 })).toBe(0);
    expect(score({ correct: 3, attempted: 4 })).toBeCloseTo(0.75, 6);
  });
});

describe("mastery — tiers", () => {
  it("is weak until an answer is right", () => {
    expect(tier(undefined)).toBe("weak");
    expect(tier(tally("law", [false]).law)).toBe("weak");
  });

  it("is familiar after one right answer, mastered after two", () => {
    expect(tier(tally("law", [true]).law)).toBe("familiar");
    expect(tier(tally("law", [true, true]).law)).toBe("mastered");
  });

  it("needs both the ratio and the attempt floor to master", () => {
    expect(MASTERED_MIN).toBe(2);
    // 4/5 = 0.8 ratio, five attempts — mastered
    expect(tier(tally("law", [true, true, true, true, false]).law)).toBe("mastered");
    // 3/4 = 0.75 — below the mastered ratio, above familiar
    expect(tier(tally("law", [true, true, true, false]).law)).toBe("familiar");
    expect(score(tally("law", [true, true, true, false]).law)).toBeLessThan(MASTERED_AT);
    // 1/2 = 0.5 — exactly familiar, not weak
    expect(tier(tally("law", [true, false]).law)).toBe("familiar");
    // 1/3 — below familiar
    expect(tier(tally("law", [true, false, false]).law)).toBe("weak");
  });
});

describe("mastery — the weakest list (send you back)", () => {
  it("puts unattempted and wrong concepts ahead of mastered ones", () => {
    const s = recordAll(emptyMastery(), ["law", "reach"], true);
    const out = weakest(s, CONCEPTS, 3);
    expect(out.every((c) => c.tier === "weak")).toBe(true);
    expect(out.map((c) => c.id)).not.toContain("law");
  });

  it("returns the concepts to revisit with the levels that teach them", () => {
    const out = weakest(emptyMastery(), CONCEPTS, 4);
    expect(out).toHaveLength(4);
    for (const c of out) {
      expect(c.levels.length).toBeGreaterThan(0);
      const concept = conceptById(c.id)!;
      expect(c.levels).toContain(concept.used[0]);
    }
  });

  it("orders ties by curriculum order and clamps the count", () => {
    const out = weakest(emptyMastery(), CONCEPTS, 2);
    expect(out[0].id).toBe("program-tree");
    expect(out[1].id).toBe("work");
    expect(weakest(emptyMastery(), CONCEPTS, 0)).toHaveLength(0);
  });
});

describe("mastery — summary", () => {
  it("counts the graph by tier and averages the ratio", () => {
    const s = tally("law", [true, true]);
    const sum = summary(s, CONCEPTS);
    expect(sum.total).toBe(CONCEPTS.length);
    expect(sum.mastered).toBe(1);
    expect(sum.weak + sum.familiar + sum.mastered).toBe(sum.total);
    expect(sum.ratio).toBeCloseTo(1 / CONCEPTS.length, 6);
  });

  it("counts mastered concepts and reports whether none are weak", () => {
    // two right answers per concept clears the attempt floor for mastery
    const s = CONCEPTS.reduce((acc, c) => recordAll(acc, [c.id, c.id], true), emptyMastery());
    expect(masteredCount(s, CONCEPTS)).toBe(CONCEPTS.length);
    expect(allFamiliar(s, CONCEPTS)).toBe(true);
    expect(allFamiliar(emptyMastery(), CONCEPTS)).toBe(false);
  });
});
