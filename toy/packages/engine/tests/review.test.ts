import { describe, expect, it } from "vite-plus/test";

import {
  CONCEPTS,
  LEVEL_ORDER,
  MODULES,
  REVIEW_PASS,
  ReviewRound,
  conceptById,
  gradeReview,
  moduleCleared,
  moduleComplete,
  moduleConcepts,
  moduleOf,
  reviewAttempts,
  reviewLevels,
} from "../src/index.ts";
import type { Module } from "../src/index.ts";

/** A two-concept arc used to exercise the mechanism (step 3 supplies the real ones). */
const FIXTURE: Module = {
  id: "fixture",
  roman: "X",
  title: "Fixture arc",
  arc: "Recombines equality and associativity across two levels.",
  levels: ["tree", "equal"],
  questions: [
    {
      id: "q1",
      concept: "semantic-equality",
      q: "Same function, different tree — equal?",
      opts: ["yes", "no"],
      ok: "yes",
      why: "Semantic equality.",
    },
    {
      id: "q2",
      concept: "associativity",
      q: "Which law changes the cost, not the result?",
      opts: ["comm", "assoc"],
      ok: "assoc",
      why: "Associativity.",
    },
    {
      id: "q3",
      concept: "semantic-equality",
      q: "Is equality syntactic?",
      opts: ["no", "yes"],
      ok: "no",
      why: "Equality is semantic.",
    },
  ],
};

describe("modules — the provisional arcs", () => {
  it("has six modules with unique ids and roman numerals", () => {
    expect(MODULES).toHaveLength(6);
    expect(new Set(MODULES.map((m) => m.id)).size).toBe(6);
    expect(MODULES.map((m) => m.roman)).toEqual(["I", "II", "III", "IV", "V", "VI"]);
  });

  it("places every level in exactly one module", () => {
    const seen: string[] = [];
    for (const m of MODULES) for (const l of m.levels) seen.push(l);
    expect(seen).toHaveLength(LEVEL_ORDER.length);
    expect(new Set(seen).size).toBe(LEVEL_ORDER.length);
    for (const l of LEVEL_ORDER) expect(seen).toContain(l);
  });

  it("finds a level's module and reports completion", () => {
    expect(moduleOf("mint")?.id).toBe("discovery");
    expect(moduleOf("nope")).toBeUndefined();
    const found = moduleOf("tree")!;
    expect(moduleComplete(found, found.levels)).toBe(true);
    expect(moduleComplete(found, found.levels.slice(0, 1))).toBe(false);
    expect(moduleComplete({ ...found, levels: [] }, [])).toBe(false);
  });

  it("gates module completion on the review passing", () => {
    const found = moduleOf("tree")!;
    expect(moduleCleared(found, found.levels, [])).toBe(false);
    expect(moduleCleared(found, found.levels, [found.id])).toBe(true);
    expect(moduleCleared(found, found.levels.slice(0, 1), [found.id])).toBe(false);
  });

  it("collects a module's concepts from its levels", () => {
    const ids = moduleConcepts(FIXTURE, CONCEPTS).map((c) => c.id);
    expect(ids).toContain("semantic-equality");
    expect(ids).toContain("associativity");
    expect(ids).toContain("program-tree");
  });
});

describe("module review — the six real reviews", () => {
  it("gives every arc a multi-question review", () => {
    for (const m of MODULES) {
      expect(m.questions.length).toBeGreaterThanOrEqual(4);
      expect(new Set(m.questions.map((q) => q.id)).size).toBe(m.questions.length);
    }
  });

  it("tags every question with a concept from the graph", () => {
    for (const m of MODULES) {
      for (const q of m.questions) {
        expect(conceptById(q.concept)).toBeDefined();
        expect(q.opts).toContain(q.ok);
        expect(q.why.length).toBeGreaterThan(0);
      }
    }
  });

  it("recombines concepts across at least two levels of the arc", () => {
    for (const m of MODULES) {
      const levels = reviewLevels(m);
      expect(levels.length).toBeGreaterThanOrEqual(2);
      for (const l of levels) expect(m.levels).toContain(l);
    }
  });

  it("passes a perfect round and sends the weak concepts back", () => {
    const m = MODULES[2]; // evaluation
    const all = Object.fromEntries(m.questions.map((q) => [q.id, q.ok]));
    expect(gradeReview(m, all).passed).toBe(true);
    // three wrong of six is below the 0.7 threshold — the round fails
    const wrong = { ...all };
    for (const q of m.questions.slice(0, 3)) wrong[q.id] = "definitely not the answer";
    const r = gradeReview(m, wrong);
    expect(r.passed).toBe(false);
    expect(r.weak).toContain(m.questions[0].concept);
  });
});

describe("module review — grading", () => {
  it("passes a complete, correct round", () => {
    const r = gradeReview(FIXTURE, { q1: "yes", q2: "assoc", q3: "no" });
    expect(r.passed).toBe(true);
    expect(r.correct).toBe(3);
    expect(r.total).toBe(3);
    expect(r.weak).toEqual([]);
  });

  it("fails below the threshold and names the weak concepts", () => {
    expect(REVIEW_PASS).toBeCloseTo(0.7, 6);
    const r = gradeReview(FIXTURE, { q1: "yes", q2: "comm", q3: "no" });
    expect(r.correct).toBe(2);
    expect(r.passed).toBe(false);
    expect(r.weak).toEqual(["associativity"]);
  });

  it("does not pass an incomplete round even if every answer so far is right", () => {
    const r = gradeReview(FIXTURE, { q1: "yes", q2: "assoc" });
    expect(r.passed).toBe(false);
  });

  it("maps answers to per-concept mastery attempts", () => {
    const attempts = reviewAttempts(FIXTURE, { q1: "yes", q2: "comm", q3: "no" });
    expect(attempts).toEqual([
      { concept: "semantic-equality", correct: true },
      { concept: "associativity", correct: false },
      { concept: "semantic-equality", correct: true },
    ]);
  });
});

describe("module review — the playable round", () => {
  it("gates on completeness and scores the round", () => {
    const r = new ReviewRound(FIXTURE);
    expect(r.total).toBe(3);
    expect(r.gate).toBe(false);
    r.answer("q1", "yes");
    r.answer("q2", "comm");
    expect(r.answered).toBe(2);
    expect(r.correct).toBe(1);
    expect(r.gate).toBe(false);
    r.answer("q3", "no");
    expect(r.gate).toBe(true);
    expect(r.score).toBeCloseTo(2 / 3, 6);
    expect(r.passed).toBe(false);
    expect(r.weak).toEqual(["associativity"]);
    expect(r.result.passed).toBe(false);
    r.reset();
    expect(r.answers).toEqual({});
    expect(r.gate).toBe(false);
  });

  it("honours a custom threshold", () => {
    const r = new ReviewRound(FIXTURE, 0.5);
    r.answer("q1", "yes");
    r.answer("q2", "comm");
    r.answer("q3", "no");
    expect(r.passed).toBe(true);
  });

  it("stays locked for a question-less module", () => {
    const r = new ReviewRound({ ...MODULES[0], questions: [] });
    expect(r.total).toBe(0);
    expect(r.gate).toBe(false);
    expect(r.passed).toBe(false);
    expect(r.weak).toEqual([]);
  });

  it("plays the real foundations review to a verdict", () => {
    const r = new ReviewRound(MODULES[0]);
    expect(r.total).toBe(MODULES[0].questions.length);
    expect(r.gate).toBe(false);
    for (const q of MODULES[0].questions) r.answer(q.id, q.ok);
    expect(r.gate).toBe(true);
    expect(r.passed).toBe(true);
    expect(r.score).toBe(1);
    expect(r.weak).toEqual([]);
  });
});
