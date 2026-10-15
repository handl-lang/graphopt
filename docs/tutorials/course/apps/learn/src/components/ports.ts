import { SINKS, Ports as Engine, key } from "engine";
import type { Term } from "engine";

/** A fused pair — the member whose lowering differs between backends. */
const FUSED: Term = { op: "sqq", args: [{ leaf: "x" }, { leaf: "x" }] };

/** A backend is a port: swap the sink, watch core stay byte-identical. */
export class Ports {
  private p = new Engine(FUSED, ["fuse", "assoc"]);
  private initial = this.p.coreFingerprint;

  get sinks(): string[] {
    return SINKS.map((s) => s.name);
  }

  get sink(): string {
    return this.p.sink.name;
  }

  get supported(): string[] {
    return this.p.sink.supported;
  }

  get unsupported(): string[] {
    return this.p.unsupported;
  }

  get ir(): string {
    return key(this.p.ir);
  }

  get rules(): string {
    return this.p.core.rules.join(", ");
  }

  get coreMutations(): number {
    return this.p.coreMutations;
  }

  get unchanged(): boolean {
    return this.p.coreFingerprint === this.initial;
  }

  get swaps(): number {
    return this.p.swaps;
  }

  get lowerable(): boolean {
    return this.p.lowerable;
  }

  get artifact(): string {
    return this.p.lower();
  }

  swap(name: string): void {
    this.p.swap(name);
  }

  edit(): void {
    this.p.editCore("comm");
  }
}
