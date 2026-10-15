/** Points — the XP model, the level curve, and the named ranks.
 *
 * Plan 0028 §C: points reward *doing*.  A level awards a base for
 * passing its checkpoint plus a bonus for engaging its live widget;
 * a module review awards more.  Every function here is pure over a
 * {@link Progress} value, so the whole model is unit-testable.
 */

import { LEVEL_ORDER, levelIndex } from "./concepts.ts";

/** The progress a score is computed from. */
export interface Progress {
  /** Level ids whose checkpoint is passed. */
  passed: readonly string[];
  /** Level ids whose live widget was engaged. */
  engaged: readonly string[];
  /** Module ids whose review is passed. */
  reviewed: readonly string[];
}

/** The base award for passing a checkpoint. */
export const BASE_PASS = 100;
/** The bonus for engaging a level's live widget — reward the doing. */
export const WIDGET_BONUS = 60;
/** The award for passing a module review. */
export const REVIEW_AWARD = 400;
/** Each later level is worth more — the level curve's slope. */
export const CURVE_STEP = 10;

/** The base award for level `index` (0-based) — the level curve. */
export function levelBase(index: number): number {
  return BASE_PASS + Math.max(0, index) * CURVE_STEP;
}

/** One level's XP, broken down. */
export interface LevelXp {
  base: number;
  bonus: number;
  total: number;
  passed: boolean;
  engaged: boolean;
}

/** The XP a level is worth under a progress value. */
export function forLevel(id: string, p: Progress, order: readonly string[] = LEVEL_ORDER): LevelXp {
  const index = levelIndex(id, order);
  const passed = p.passed.includes(id);
  const engaged = p.engaged.includes(id);
  const base = passed ? levelBase(index) : 0;
  const bonus = engaged ? WIDGET_BONUS : 0;
  return { base, bonus, total: base + bonus, passed, engaged };
}

/** Total XP: every level's award plus the reviews. */
export function totalXp(p: Progress, order: readonly string[] = LEVEL_ORDER): number {
  let sum = p.reviewed.length * REVIEW_AWARD;
  for (const id of order) sum += forLevel(id, p, order).total;
  return sum;
}

/** A named rank tier. */
export interface Rank {
  id: string;
  name: string;
  /** The XP at which the rank is reached. */
  at: number;
}

/** The ranks, ascending — named for what the course asks of you. */
export const RANKS: readonly Rank[] = [
  { id: "newcomer", name: "Newcomer", at: 0 },
  { id: "apprentice", name: "Apprentice", at: 300 },
  { id: "journeyman", name: "Journeyman", at: 1000 },
  { id: "adept", name: "Adept", at: 2000 },
  { id: "expert", name: "Expert", at: 3200 },
  { id: "scholar", name: "Scholar", at: 4600 },
  { id: "master", name: "Master", at: 6200 },
  { id: "comprehender", name: "Comprehender", at: 8000 },
];

/** The rank an XP total has reached. */
export function rank(xp: number, ranks: readonly Rank[] = RANKS): Rank {
  let out = ranks[0];
  for (const r of ranks) if (xp >= r.at) out = r;
  return out;
}

/** The next rank above `xp`, or `null` at the top. */
export function nextRank(xp: number, ranks: readonly Rank[] = RANKS): Rank | null {
  return ranks.find((r) => r.at > xp) ?? null;
}

/** How far between the current and next rank, 0…1. */
export function rankProgress(xp: number, ranks: readonly Rank[] = RANKS): number {
  const now = rank(xp, ranks);
  const next = nextRank(xp, ranks);
  if (!next) return 1;
  return (xp - now.at) / (next.at - now.at);
}
