import { Gaps as Engine } from "engine";
import type { Gap } from "engine";

/** The payoff: idea → blocker → next measurement. */
export class Improve {
  private g = new Engine();

  get gaps(): readonly Gap[] {
    return this.g.gaps;
  }

  get sel(): number {
    return this.g.sel;
  }

  get gap(): Gap {
    return this.g.gap;
  }

  get connection(): { idea: string; blocker: string; next: string }[] {
    return this.g.connection;
  }

  pick(i: number): void {
    this.g.pick(i);
  }
}
