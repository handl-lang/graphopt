import { describe, expect, it } from "vite-plus/test";

import {
  BADGES,
  CONCEPTS,
  LEVEL_ORDER,
  MODULES,
  badgeState,
  earned,
  emptyMastery,
  isEarned,
  locked,
  recordAll,
} from "../src/index.ts";
import type { BadgeState } from "../src/index.ts";

/** The levels of the first arc — a small, complete module. */
const ARC1 = MODULES[0].levels;
/** Every level, in order. */
const ALL = MODULES.flatMap((m) => m.levels);

/** A mastery state where every concept has at least one right answer. */
function familiarAll(): ReturnType<typeof emptyMastery> {
  return CONCEPTS.reduce((s, c) => recordAll(s, [c.id], true), emptyMastery());
}

describe("badges — the set", () => {
  it("is non-empty, uniquely identified and blurbed", () => {
    expect(BADGES.length).toBeGreaterThan(8);
    expect(new Set(BADGES.map((b) => b.id)).size).toBe(BADGES.length);
    for (const b of BADGES) {
      expect(b.name.length).toBeGreaterThan(0);
      expect(b.blurb.length).toBeGreaterThan(0);
      expect(typeof b.test).toBe("function");
    }
  });

  it("carries the milestones plan 0028 names", () => {
    const ids = BADGES.map((b) => b.id);
    expect(ids).toContain("first-certificate");
    expect(ids).toContain("sweep-breaker");
    expect(ids).toContain("negative-results-kept");
    expect(ids).toContain("curriculum-complete");
    expect(ids).toContain("full-comprehension");
    expect(ids.filter((id) => id.startsWith("module-"))).toHaveLength(MODULES.length);
  });
});

describe("badges — predicates", () => {
  it("earns nothing on a fresh course and locks everything else", () => {
    const s = badgeState({});
    expect(earned(s)).toEqual([]);
    expect(locked(s)).toHaveLength(BADGES.length);
  });

  it("fills the default state from the graph", () => {
    const s = badgeState({});
    expect(s.order).toBe(LEVEL_ORDER);
    expect(s.concepts).toBe(CONCEPTS);
    expect(s.modules).toBe(MODULES);
    expect(s.passed).toEqual([]);
  });

  it("earns First Steps on the first level and First Certificate at the referee", () => {
    expect(isEarned("first-steps", badgeState({ passed: ["tree"] }))).toBe(true);
    expect(isEarned("first-certificate", badgeState({ passed: ["referee"] }))).toBe(true);
    expect(isEarned("first-certificate", badgeState({ passed: ["tree"] }))).toBe(false);
  });

  it("earns Sweep Breaker by engaging level 13's widget", () => {
    expect(isEarned("sweep-breaker", badgeState({ engaged: ["truthpay"] }))).toBe(true);
    expect(isEarned("sweep-breaker", badgeState({ passed: ["truthpay"] }))).toBe(false);
  });

  it("earns a module badge only when the whole arc is cleared *and* reviewed", () => {
    const mod = MODULES[0].id;
    expect(isEarned(`module-${mod}`, badgeState({ passed: [...ARC1], reviewed: [mod] }))).toBe(
      true,
    );
    expect(isEarned(`module-${mod}`, badgeState({ passed: [...ARC1] }))).toBe(false);
    expect(
      isEarned(`module-${mod}`, badgeState({ passed: ARC1.slice(0, 1), reviewed: [mod] })),
    ).toBe(false);
  });

  it("earns Modules I–VI Complete only when every arc is done and reviewed", () => {
    const allReviews = MODULES.map((m) => m.id);
    expect(
      isEarned("curriculum-complete", badgeState({ passed: [...ALL], reviewed: allReviews })),
    ).toBe(true);
    expect(isEarned("curriculum-complete", badgeState({ passed: [...ALL] }))).toBe(false);
    expect(isEarned("curriculum-complete", badgeState({ passed: ARC1 }))).toBe(false);
  });

  it("earns Course Cleared only when every level in order is passed", () => {
    expect(isEarned("course-cleared", badgeState({ order: ALL, passed: [...ALL] }))).toBe(true);
    expect(isEarned("course-cleared", badgeState({ order: ALL, passed: ALL.slice(0, 3) }))).toBe(
      false,
    );
  });

  it("earns Hands On only when every level's widget was engaged", () => {
    expect(isEarned("hands-on", badgeState({ order: ["tree", "equal"], engaged: ["tree"] }))).toBe(
      false,
    );
    expect(
      isEarned("hands-on", badgeState({ order: ["tree", "equal"], engaged: ["tree", "equal"] })),
    ).toBe(true);
  });

  it("earns Full Comprehension only with all levels, all reviews and no weak concept", () => {
    const done: BadgeState = badgeState({
      order: ALL,
      passed: [...ALL],
      reviewed: MODULES.map((m) => m.id),
      mastery: familiarAll(),
    });
    expect(isEarned("full-comprehension", done)).toBe(true);
    expect(isEarned("full-comprehension", { ...done, reviewed: [] })).toBe(false);
    expect(isEarned("full-comprehension", { ...done, mastery: emptyMastery() })).toBe(false);
    expect(isEarned("full-comprehension", { ...done, passed: ALL.slice(0, 5) })).toBe(false);
  });

  it("partitions the set into earned and locked", () => {
    const s = badgeState({ passed: ["tree", "referee"] });
    expect(earned(s).length + locked(s).length).toBe(BADGES.length);
    expect(earned(s).every((b) => b.test(s))).toBe(true);
  });
});
