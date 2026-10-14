import { CORPUS, Exposure as Screen, YIELD_PCT } from "engine";
import type { Bin, OpClass, Workload } from "engine";

/** The exposure screen, playable: the histogram is computed live. */
export class Expose {
  private e = new Screen();

  get mode(): "workloads" | "ops" {
    return this.e.mode;
  }

  get histogram(): Bin[] {
    return this.e.histogram;
  }

  get maxBin(): number {
    return this.e.maxBin;
  }

  get workloads(): readonly Workload[] {
    return this.e.workloads;
  }

  get ops(): readonly OpClass[] {
    return this.e.ops;
  }

  get modes(): { zero: number; high: number; total: number } {
    return this.e.modes;
  }

  get visibleButNull(): Workload[] {
    return this.e.visibleButNull;
  }

  get visibleOps(): OpClass[] {
    return this.e.ops.filter((o) => o.visible);
  }

  get invisibleOps(): OpClass[] {
    return this.e.ops.filter((o) => !o.visible);
  }

  get corpus(): { total: number; zero: number; high: number } {
    return CORPUS;
  }

  get yield(): number {
    return YIELD_PCT;
  }

  setMode(m: "workloads" | "ops"): void {
    this.e.setMode(m);
  }
}
