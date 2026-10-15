import { Artifacts as Engine } from "engine";
import type { Artifact } from "engine";

/** The delivery artifacts, playable: pick one, read what it needs to run. */
export class DeliveryArtifacts {
  private a = new Engine();

  get artifacts(): readonly Artifact[] {
    return this.a.artifacts;
  }

  get sel(): number {
    return this.a.sel;
  }

  get artifact(): Artifact {
    return this.a.artifact;
  }

  get needs(): readonly string[] {
    return this.a.needs;
  }

  get standalone(): boolean {
    return this.a.standalone;
  }

  get standaloneCount(): number {
    return this.a.standaloneCount;
  }

  get portable(): Artifact {
    return this.a.portable;
  }

  pick(i: number): void {
    this.a.pick(i);
  }
}
