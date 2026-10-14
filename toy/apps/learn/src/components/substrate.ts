import { PROGRAMS, Substrate as Engine, key, renderComb } from "engine";

/** Sharing is a morphism (Δ); recurrence is a `loop` — the bridge round-trips. */
export class Substrate {
  private s = new Engine(Object.entries(PROGRAMS).map(([id, term]) => ({ id, term })));

  get programs(): string[] {
    return this.s.programs.map((p) => p.id);
  }

  get sel(): number {
    return this.s.sel;
  }

  get programId(): string {
    return this.s.program.id;
  }

  get term(): string {
    return key(this.s.program.term);
  }

  get comb(): string {
    return renderComb(this.s.comb);
  }

  get back(): string {
    return key(this.s.back);
  }

  get roundTrips(): boolean {
    return this.s.roundTrips;
  }

  get dups(): number {
    return this.s.dups;
  }

  get loopComb(): string {
    return renderComb(this.s.loop.comb);
  }

  get loopBack(): string {
    return key(this.s.loop.back);
  }

  get loopRoundTrips(): boolean {
    return this.s.loop.roundTrips;
  }

  pick(i: number): void {
    this.s.pick(i);
  }
}
