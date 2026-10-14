import { GameLab as Engine } from "engine";
import type { Player, Step } from "engine";

/** The game layer, playable: step an episode and compare the players. */
export class Game {
  private g = new Engine();

  get domains(): readonly string[] {
    return this.g.domains;
  }

  get step(): number {
    return this.g.step;
  }

  get total(): number {
    return this.g.total;
  }

  get at(): Step {
    return this.g.at;
  }

  get done(): boolean {
    return this.g.done;
  }

  get transcript(): Step[] {
    return this.g.transcript;
  }

  get reward(): number {
    return this.g.reward;
  }

  get players(): readonly Player[] {
    return this.g.players;
  }

  get best(): Player {
    return this.g.best;
  }

  get priorGain(): number {
    return this.g.priorGain;
  }

  get priorComparison(): { learned: Player; heuristic: Player } {
    return this.g.priorComparison;
  }

  next(): void {
    this.g.next();
  }

  reset(): void {
    this.g.reset();
  }
}
