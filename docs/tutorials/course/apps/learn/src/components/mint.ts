import { Minting } from "engine";
import type { Need, Stage } from "engine";

/** The minting funnel, playable: pick a need, read ADR 0007's four layers. */
export class Mint {
  private m = new Minting();

  get needs(): readonly Need[] {
    return this.m.needs;
  }

  get funnel(): Stage[] {
    return this.m.funnel;
  }

  get need(): Need {
    return this.m.need;
  }

  get layers(): { key: string; label: string; ok: boolean; detail: string }[] {
    return this.m.layers;
  }

  get verdict(): string {
    return this.m.verdict;
  }

  get verdictTone(): string {
    return this.m.verdictTone;
  }

  get totalFlops(): number {
    return this.m.totalFlops;
  }

  get sel(): number {
    return this.m.sel;
  }

  pick(i: number): void {
    this.m.pick(i);
  }
}
