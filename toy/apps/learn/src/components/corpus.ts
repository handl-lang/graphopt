import { DEFECTS, Intake as Engine } from "engine";
import type { FunnelRow, IntakeWorkload } from "engine";

/** The intake browser, playable: the class counts are computed live. */
export class Corpus {
  private i = new Engine();

  get workloads(): readonly IntakeWorkload[] {
    return this.i.workloads;
  }

  get sel(): number {
    return this.i.sel;
  }

  get picked(): IntakeWorkload {
    return this.i.picked;
  }

  get funnel(): FunnelRow[] {
    return this.i.funnel;
  }

  get classes(): { ingested: number; censusOnly: number; rejected: number; total: number } {
    return this.i.classes;
  }

  get nodes(): number {
    return this.i.nodes;
  }

  get corpus(): { total: number; verified: number; defects: number } {
    return this.i.corpus;
  }

  get defects(): readonly { model: string; symptom: string; fix: string }[] {
    return DEFECTS;
  }

  pick(k: number): void {
    this.i.pick(k);
  }
}
