import { describe, expect, it } from "vite-plus/test";

import {
  BASE_PASS,
  CURVE_STEP,
  LEVEL_ORDER,
  RANKS,
  REVIEW_AWARD,
  WIDGET_BONUS,
  forLevel,
  levelBase,
  nextRank,
  rank,
  rankProgress,
  totalXp,
} from "../src/index.ts";

const none = { passed: [], engaged: [], reviewed: [] };

describe("points — the level curve", () => {
  it("grows the base award by a fixed step", () => {
    expect(levelBase(0)).toBe(BASE_PASS);
    expect(levelBase(3)).toBe(BASE_PASS + 3 * CURVE_STEP);
    expect(levelBase(-4)).toBe(BASE_PASS);
  });

  it("breaks a level's award into base and widget bonus", () => {
    const lvl = forLevel("tree", { passed: ["tree"], engaged: ["tree"], reviewed: [] });
    expect(lvl.base).toBe(BASE_PASS);
    expect(lvl.bonus).toBe(WIDGET_BONUS);
    expect(lvl.total).toBe(BASE_PASS + WIDGET_BONUS);
    expect(lvl.passed).toBe(true);
    expect(lvl.engaged).toBe(true);
  });

  it("awards the widget bonus for engaging even before passing", () => {
    const lvl = forLevel("tree", { passed: [], engaged: ["tree"], reviewed: [] });
    expect(lvl.base).toBe(0);
    expect(lvl.bonus).toBe(WIDGET_BONUS);
    expect(lvl.total).toBe(WIDGET_BONUS);
  });

  it("awards nothing for an untouched level", () => {
    expect(forLevel("improve", none).total).toBe(0);
  });
});

describe("points — totals", () => {
  it("is zero at the start", () => {
    expect(totalXp(none)).toBe(0);
  });

  it("adds every level's award plus the reviews", () => {
    const all = { passed: [...LEVEL_ORDER], engaged: [...LEVEL_ORDER], reviewed: ["foundations"] };
    const levels = LEVEL_ORDER.reduce((s, _id, i) => s + levelBase(i), 0);
    expect(totalXp(all)).toBe(levels + 22 * WIDGET_BONUS + REVIEW_AWARD);
  });

  it("reaches the top rank when everything is done", () => {
    const all = {
      passed: [...LEVEL_ORDER],
      engaged: [...LEVEL_ORDER],
      reviewed: ["a", "b", "c", "d", "e", "f"],
    };
    expect(rank(totalXp(all)).id).toBe("comprehender");
  });
});

describe("points — ranks", () => {
  it("starts a newcomer at zero", () => {
    expect(rank(0).name).toBe("Newcomer");
    expect(rank(0).id).toBe("newcomer");
  });

  it("climbs through the named tiers at their thresholds", () => {
    for (const r of RANKS) {
      expect(rank(r.at).id).toBe(r.id);
      if (r.at > 0) expect(rank(r.at - 1).id).not.toBe(r.id);
    }
  });

  it("reports the next rank and progress toward it", () => {
    expect(nextRank(0)?.id).toBe("apprentice");
    expect(nextRank(0)?.at).toBe(RANKS[1].at);
    expect(rankProgress(0)).toBe(0);
    const mid = (RANKS[1].at - RANKS[0].at) / 2;
    expect(rankProgress(mid)).toBeCloseTo(0.5, 6);
    expect(nextRank(99999)).toBeNull();
    expect(rankProgress(99999)).toBe(1);
  });
});
