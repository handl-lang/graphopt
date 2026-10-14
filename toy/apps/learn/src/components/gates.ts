import { GateLab as Engine, NO_SUPPRESSIONS } from "engine";
import type { Gate } from "engine";

/** The gates, playable: break a gate and watch the run fail. */
export class Gates {
  private g = new Engine();

  get gates(): readonly Gate[] {
    return this.g.gates;
  }

  get broken(): string | null {
    return this.g.broken;
  }

  get brokenGate(): Gate | null {
    return this.g.brokenGate;
  }

  get results(): { gate: Gate; pass: boolean }[] {
    return this.g.results;
  }

  get passing(): boolean {
    return this.g.passing;
  }

  get passingCount(): number {
    return this.g.passingCount;
  }

  get report(): string {
    return this.g.report;
  }

  get rule(): string {
    return NO_SUPPRESSIONS;
  }

  break(id: string): void {
    this.g.break(id);
  }

  repair(): void {
    this.g.repair();
  }
}
