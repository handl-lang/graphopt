/** Badges — the course's milestones, each a predicate over progress.
 *
 * Plan 0028 §C.  A badge is data: an id, a name, a blurb, and a pure
 * `test` over a {@link BadgeState}.  The shell renders earned versus
 * locked from the same list, so adding a badge is adding one entry.
 */

import type { Concept } from "./concepts.ts";
import { CONCEPTS, LEVEL_ORDER } from "./concepts.ts";
import type { MasteryState } from "./mastery.ts";
import { allFamiliar } from "./mastery.ts";
import type { Module } from "./review.ts";
import { MODULES, moduleCleared } from "./review.ts";

/** Everything a badge may look at. */
export interface BadgeState {
  order: readonly string[];
  passed: readonly string[];
  engaged: readonly string[];
  reviewed: readonly string[];
  mastery: MasteryState;
  concepts: readonly Concept[];
  modules: readonly Module[];
}

export interface Badge {
  id: string;
  name: string;
  blurb: string;
  test: (s: BadgeState) => boolean;
}

/** The level ids that are passed, as a set. */
const has = (list: readonly string[], id: string): boolean => list.includes(id);

/** One badge per module, built from the module data so step 3 stays in sync. */
function moduleBadge(m: Module): Badge {
  return {
    id: `module-${m.id}`,
    name: `Module ${m.roman} Complete`,
    blurb: `Clear every level in ${m.title} and pass its review.`,
    test: (s) => {
      const mod = s.modules.find((x) => x.id === m.id);
      return mod ? moduleCleared(mod, s.passed, s.reviewed) : false;
    },
  };
}

/** The badge set. */
export const BADGES: readonly Badge[] = [
  {
    id: "first-steps",
    name: "First Steps",
    blurb: "Read a program as a tree and pass level 0.",
    test: (s) => has(s.passed, "tree"),
  },
  {
    id: "first-certificate",
    name: "First Certificate",
    blurb: "Clear the referee — a rewrite is legal only if the certificate replays.",
    test: (s) => has(s.passed, "referee"),
  },
  {
    id: "sweep-breaker",
    name: "Sweep Breaker",
    blurb: "Run level 13's adversarial sweep and break the naive guard.",
    test: (s) => has(s.engaged, "truthpay"),
  },
  {
    id: "negative-results-kept",
    name: "Negative Results Kept",
    blurb: "Clear the ledger of negatives — a measured refusal is a result.",
    test: (s) => has(s.passed, "honest"),
  },
  {
    id: "hands-on",
    name: "Hands On",
    blurb: "Engage the live widget on every level — points reward doing.",
    test: (s) => s.order.length > 0 && s.order.every((id) => has(s.engaged, id)),
  },
  ...MODULES.map(moduleBadge),
  {
    id: "curriculum-complete",
    name: "Modules I–VI Complete",
    blurb: "Clear all six arcs — every level and every review.",
    test: (s) =>
      s.modules.length > 0 && s.modules.every((m) => moduleCleared(m, s.passed, s.reviewed)),
  },
  {
    id: "course-cleared",
    name: "Course Cleared",
    blurb: "Pass every level's checkpoint.",
    test: (s) => s.order.length > 0 && s.order.every((id) => has(s.passed, id)),
  },
  {
    id: "full-comprehension",
    name: "Full Comprehension",
    blurb: "Every level cleared, every review passed, no concept left weak.",
    test: (s) =>
      s.order.length > 0 &&
      s.order.every((id) => has(s.passed, id)) &&
      s.modules.length > 0 &&
      s.modules.every((m) => has(s.reviewed, m.id)) &&
      allFamiliar(s.mastery, s.concepts),
  },
];

/** The badges earned under a progress state. */
export function earned(s: BadgeState): Badge[] {
  return BADGES.filter((b) => b.test(s));
}

/** The badges still locked. */
export function locked(s: BadgeState): Badge[] {
  return BADGES.filter((b) => !b.test(s));
}

/** Whether one badge is earned. */
export function isEarned(id: string, s: BadgeState): boolean {
  return BADGES.find((b) => b.id === id)?.test(s) ?? false;
}

/** A full progress state with the defaults filled in. */
export function badgeState(partial: Partial<BadgeState>): BadgeState {
  return {
    order: partial.order ?? LEVEL_ORDER,
    passed: partial.passed ?? [],
    engaged: partial.engaged ?? [],
    reviewed: partial.reviewed ?? [],
    mastery: partial.mastery ?? {},
    concepts: partial.concepts ?? CONCEPTS,
    modules: partial.modules ?? MODULES,
  };
}
