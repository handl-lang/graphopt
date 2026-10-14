/** Module reviews — the spaced-retrieval round between arcs.
 *
 * Plan 0028 §A/§C: the course is six arcs, each ending in a review —
 * a gated multi-question round that *recombines* the arc's concepts
 * (it must not merely repeat the level checkpoints).  Passing feeds
 * the concept mastery; failing reports the weak concepts so the
 * review can send you back to the levels that teach them.
 *
 * This is the **mechanism**, built in step 1.  The module→levels
 * mapping below is provisional (the existing 22 levels, placed by
 * plan 0028 §A's arc table); step 3 supplies the final mapping and
 * the review questions.  The shape — a {@link Module} of level ids
 * plus {@link ReviewQuestion}s — is all the mechanism needs.
 */

import type { Concept } from "./concepts.ts";

export interface ReviewQuestion {
  id: string;
  /** The concept this question exercises — feeds the mastery meter. */
  concept: string;
  q: string;
  opts: readonly string[];
  ok: string;
  why: string;
}

/** An arc: a module id, its levels, and its review questions. */
export interface Module {
  id: string;
  /** The roman numeral shown in the shell. */
  roman: string;
  title: string;
  /** One line describing the arc. */
  arc: string;
  /** The level ids in the arc, in order. */
  levels: readonly string[];
  /** The review questions.  Empty until step 3 supplies them. */
  questions: readonly ReviewQuestion[];
}

/** The six arcs.  Provisional: existing levels only (step 3 adds the new ones). */
export const MODULES: readonly Module[] = [
  {
    id: "foundations",
    roman: "I",
    title: "Foundations",
    arc: "What a program is, and what equality means.",
    levels: ["tree", "equal", "reach", "structure"],
    questions: [],
  },
  {
    id: "correctness",
    roman: "II",
    title: "Correctness & choice",
    arc: "Certificates, frontiers, search.",
    levels: ["referee", "choice", "search", "learn"],
    questions: [],
  },
  {
    id: "evaluation",
    roman: "III",
    title: "Evaluation & execution",
    arc: "What looks good versus what ran.",
    levels: ["measure", "architecture"],
    questions: [],
  },
  {
    id: "discovery",
    roman: "IV",
    title: "Discovery & minting",
    arc: "Growing the vocabulary.",
    levels: ["advanced", "coherence", "mint", "truthpay", "exposure"],
    questions: [],
  },
  {
    id: "substrate",
    roman: "V",
    title: "Substrate & backends",
    arc: "The presentation and the ports.",
    levels: ["substrate", "ports"],
    questions: [],
  },
  {
    id: "engineering",
    roman: "VI",
    title: "Engineering & payoff",
    arc: "How the project stays honest, and where to help.",
    levels: ["honest", "bounded", "deliver", "pricing", "improve"],
    questions: [],
  },
];

/** The module a level belongs to, or `undefined`. */
export function moduleOf(
  levelId: string,
  modules: readonly Module[] = MODULES,
): Module | undefined {
  return modules.find((m) => m.levels.includes(levelId));
}

/** A module is complete when it has levels and all of them are passed. */
export function moduleComplete(m: Module, passed: readonly string[]): boolean {
  return m.levels.length > 0 && m.levels.every((l) => passed.includes(l));
}

/** A module's concepts — the concepts its levels introduce or use. */
export function moduleConcepts(m: Module, concepts: readonly Concept[]): Concept[] {
  return concepts.filter(
    (c) =>
      (c.introduced !== null && m.levels.includes(c.introduced)) ||
      c.used.some((l) => m.levels.includes(l)),
  );
}

/** The fraction of a module's questions that must be right to pass. */
export const REVIEW_PASS = 0.7;

export interface ReviewResult {
  module: string;
  correct: number;
  total: number;
  passed: boolean;
  /** Concept ids of the wrong answers — the "send you back" list. */
  weak: string[];
}

/** Grade a module review from a question→option answer map. */
export function gradeReview(
  m: Module,
  answers: Readonly<Record<string, string>>,
  threshold = REVIEW_PASS,
): ReviewResult {
  let correct = 0;
  const weak: string[] = [];
  for (const q of m.questions) {
    if (answers[q.id] === q.ok) correct += 1;
    else weak.push(q.concept);
  }
  const total = m.questions.length;
  const complete = total > 0 && m.questions.every((q) => answers[q.id] !== undefined);
  const passed = complete && correct / total >= threshold;
  return { module: m.id, correct, total, passed, weak: [...new Set(weak)] };
}

/** The mastery attempts a review produces (right or wrong, per question). */
export function reviewAttempts(
  m: Module,
  answers: Readonly<Record<string, string>>,
): { concept: string; correct: boolean }[] {
  return m.questions.map((q) => ({ concept: q.concept, correct: answers[q.id] === q.ok }));
}

/** The playable round: answer every question, then read the verdict. */
export class ReviewRound {
  answers: Record<string, string> = {};
  module: Module;
  threshold: number;

  constructor(module: Module, threshold = REVIEW_PASS) {
    this.module = module;
    this.threshold = threshold;
  }

  get questions(): readonly ReviewQuestion[] {
    return this.module.questions;
  }

  get total(): number {
    return this.module.questions.length;
  }

  get answered(): number {
    return this.module.questions.filter((q) => this.answers[q.id] !== undefined).length;
  }

  get correct(): number {
    return this.module.questions.filter((q) => this.answers[q.id] === q.ok).length;
  }

  get score(): number {
    return this.total === 0 ? 0 : this.correct / this.total;
  }

  /** Every question has an answer — the round may be graded. */
  get gate(): boolean {
    return this.total > 0 && this.answered === this.total;
  }

  get passed(): boolean {
    return this.gate && this.score >= this.threshold;
  }

  /** The concept ids behind the wrong answers. */
  get weak(): string[] {
    return [
      ...new Set(
        this.module.questions.filter((q) => this.answers[q.id] !== q.ok).map((q) => q.concept),
      ),
    ];
  }

  get result(): ReviewResult {
    return gradeReview(this.module, this.answers, this.threshold);
  }

  answer(qid: string, opt: string): void {
    this.answers = { ...this.answers, [qid]: opt };
  }

  reset(): void {
    this.answers = {};
  }
}
