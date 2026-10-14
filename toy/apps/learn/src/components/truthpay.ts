import { GUARDS, Guards, unsq } from "engine";
import type { Guard, Site } from "engine";

export interface WitnessView {
  u: string;
  v: string;
  d: number;
  lhsU: string;
  lhsV: string;
  grid: string;
  gridEq: boolean;
  equal: boolean;
  bad: boolean;
  reason: string;
}

const shape = (s: number[]): string => `(${s.join(", ")})`;

/** The adversarial sweep, playable: break the naive guard, then see the exact one. */
export class Truthpay {
  private g = new Guards();

  get guards(): readonly Guard[] {
    return GUARDS;
  }

  get guard(): "naive" | "exact" {
    return this.g.guard;
  }

  get sweep(): {
    key: string;
    label: string;
    accepted: number;
    badAccept: number;
    missed: number;
  }[] {
    return this.g.sweep;
  }

  get domain(): {
    sites: number;
    evaluable: number;
    equal: number;
    unequal: number;
    illTyped: number;
  } {
    return this.g.domain;
  }

  get witnessCount(): number {
    return this.g.badAccepts.length;
  }

  get witnessIndex(): number {
    return this.witnessCount ? this.g.sel % this.witnessCount : 0;
  }

  get witness(): WitnessView | null {
    const s: Site | undefined = this.g.witness;
    if (!s) return null;
    return {
      u: shape(s.u),
      v: shape(s.v),
      d: s.d,
      lhsU: shape(unsq(s.u, s.d)),
      lhsV: shape(unsq(s.v, s.d)),
      grid: shape(s.grid),
      gridEq: s.gridEq,
      equal: s.equal,
      bad: s.gridEq && !s.equal,
      reason: s.reason,
    };
  }

  get synthesized(): string {
    return this.g.synthesized;
  }

  setGuard(k: "naive" | "exact"): void {
    this.g.setGuard(k);
  }

  next(): void {
    this.g.pick(this.g.sel + 1);
  }

  prev(): void {
    this.g.pick(Math.max(0, this.g.sel - 1));
  }
}
