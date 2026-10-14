import { CellStoreLab as Engine, ORIGINS } from "engine";
import type { Cell, CellRole, Origin } from "engine";

/** The cell store, playable: assemble fragments and watch a duplicate raise. */
export class CellStore {
  private c = new Engine();

  get cells(): Cell[] {
    return this.c.cells;
  }

  get size(): number {
    return this.c.size;
  }

  get origins(): Origin[] {
    return this.c.origins;
  }

  get allOrigins(): readonly Origin[] {
    return ORIGINS;
  }

  get byRole(): { role: CellRole; count: number; handl: string }[] {
    return this.c.byRole;
  }

  get conflict(): Cell | null {
    return this.c.conflict;
  }

  addDuplicate(): void {
    this.c.addDuplicate();
  }

  reset(): void {
    this.c.reset();
  }
}
