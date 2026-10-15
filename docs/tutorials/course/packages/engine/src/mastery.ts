/** Per-concept mastery — spaced retrieval over checkpoints and reviews.
 *
 * Plan 0028 §C: every concept accumulates correct and attempted
 * answers from *both* the level checkpoints and the module reviews,
 * and lands in one of three tiers — `weak` / `familiar` /
 * `mastered`.  {@link weakest} is the data the module-review boss
 * uses to send you back to the levels whose concepts were weak.
 *
 * All of it is pure: a {@link MasteryState} in, a new one out.
 */

import type { Concept } from "./concepts.ts";
import { CONCEPTS, LEVEL_ORDER, levelIndex } from "./concepts.ts";

/** One concept's running tally. */
export interface ConceptMastery {
  correct: number;
  attempted: number;
}

/** The whole mastery state, keyed by concept id. */
export type MasteryState = Readonly<Record<string, ConceptMastery>>;

export type Tier = "weak" | "familiar" | "mastered";

/** The correctness ratio at which a concept becomes `familiar`. */
export const FAMILIAR_AT = 0.5;
/** The correctness ratio at which a concept becomes `mastered`. */
export const MASTERED_AT = 0.8;
/** The attempts needed before `mastered` can be reached. */
export const MASTERED_MIN = 2;

export function emptyMastery(): MasteryState {
  return {};
}

/** Record one answer, returning a new state. */
export function record(state: MasteryState, concept: string, correct: boolean): MasteryState {
  const prev = state[concept] ?? { correct: 0, attempted: 0 };
  return {
    ...state,
    [concept]: {
      correct: prev.correct + (correct ? 1 : 0),
      attempted: prev.attempted + 1,
    },
  };
}

/** Record one answer against several concepts at once. */
export function recordAll(
  state: MasteryState,
  concepts: readonly string[],
  correct: boolean,
): MasteryState {
  let out = state;
  for (const c of concepts) out = record(out, c, correct);
  return out;
}

/** A concept's correctness ratio, 0…1; 0 when it has no attempts. */
export function score(m: ConceptMastery | undefined): number {
  if (!m || m.attempted === 0) return 0;
  return m.correct / m.attempted;
}

/** The tier a concept's tally has earned. */
export function tier(m: ConceptMastery | undefined): Tier {
  if (!m || m.attempted === 0) return "weak";
  const s = score(m);
  if (m.attempted >= MASTERED_MIN && s >= MASTERED_AT) return "mastered";
  if (s >= FAMILIAR_AT) return "familiar";
  return "weak";
}

const TIER_RANK: Record<Tier, number> = { weak: 0, familiar: 1, mastered: 2 };

/** A concept to revisit, with the levels that teach it. */
export interface WeakConcept {
  id: string;
  label: string;
  tier: Tier;
  score: number;
  attempts: number;
  /** The levels that teach it — where a review sends you back to. */
  levels: readonly string[];
}

/** The `n` weakest concepts — the module-review "send you back" data. */
export function weakest(
  state: MasteryState,
  concepts: readonly Concept[] = CONCEPTS,
  n = 5,
): WeakConcept[] {
  const rows = concepts.map((c) => {
    const m = state[c.id];
    const levels = [...new Set([...(c.introduced ? [c.introduced] : []), ...c.used])];
    return {
      id: c.id,
      label: c.label,
      tier: tier(m),
      score: score(m),
      attempts: m?.attempted ?? 0,
      levels,
    };
  });
  rows.sort(
    (a, b) =>
      TIER_RANK[a.tier] - TIER_RANK[b.tier] ||
      a.score - b.score ||
      a.attempts - b.attempts ||
      levelIndex(a.levels[0], LEVEL_ORDER) - levelIndex(b.levels[0], LEVEL_ORDER),
  );
  return rows.slice(0, Math.max(0, n));
}

/** A tally of the whole graph by tier. */
export interface MasterySummary {
  weak: number;
  familiar: number;
  mastered: number;
  total: number;
  /** Mean correctness ratio across every concept, 0…1. */
  ratio: number;
}

/** The graph's tier counts and overall ratio. */
export function summary(
  state: MasteryState,
  concepts: readonly Concept[] = CONCEPTS,
): MasterySummary {
  let weak = 0;
  let familiar = 0;
  let mastered = 0;
  let ratio = 0;
  for (const c of concepts) {
    const t = tier(state[c.id]);
    if (t === "mastered") mastered += 1;
    else if (t === "familiar") familiar += 1;
    else weak += 1;
    ratio += score(state[c.id]);
  }
  const total = concepts.length;
  return { weak, familiar, mastered, total, ratio: total === 0 ? 0 : ratio / total };
}

/** How many concepts are `mastered`. */
export function masteredCount(
  state: MasteryState,
  concepts: readonly Concept[] = CONCEPTS,
): number {
  return concepts.filter((c) => tier(state[c.id]) === "mastered").length;
}

/** Whether every concept has left `weak`. */
export function allFamiliar(state: MasteryState, concepts: readonly Concept[] = CONCEPTS): boolean {
  return concepts.every((c) => tier(state[c.id]) !== "weak");
}
