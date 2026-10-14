import { Bounded as Engine } from "engine";
import type { Member } from "engine";

/** The bounded lane: a certificate *and* an economics test. */
export class Bounded {
  private b = new Engine();

  get budgets(): readonly number[] {
    return this.b.budgets;
  }

  get members(): readonly Member[] {
    return this.b.members;
  }

  get sel(): number {
    return this.b.sel;
  }

  get budgetLabel(): string {
    return this.b.budgetLabel;
  }

  get rows(): {
    member: Member;
    certified: boolean;
    memberFlops: number;
    ratio: number;
    delivered: boolean;
  }[] {
    return this.b.rows;
  }

  get offered(): number {
    return this.b.offered;
  }

  get certified(): number {
    return this.b.certified;
  }

  get delivered(): number {
    return this.b.delivered;
  }

  get pricedRefused(): number {
    return this.b.pricedRefused;
  }

  get gateClean(): boolean {
    return this.b.gateClean;
  }

  get thick(): boolean {
    return this.b.thick;
  }

  get verdict(): string {
    return this.b.verdict;
  }

  select(i: number): void {
    this.b.select(i);
  }

  toggle(): void {
    this.b.toggle();
  }
}
