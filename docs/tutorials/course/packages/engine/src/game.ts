/** The game layer — boards, players, and learned priors.
 *
 * Plan 0028 §D5.  `mlopt_discovery.engine` is the `Board` protocol
 * (a program, a legal move set, a referee) plus the generic
 * `run_episode` / `train_policy` / `evaluate` drivers;
 * `mlopt_discovery.players` is `LinearPolicy`, the learned arm; and
 * `python -m mlopt_discovery.play --domain meta|joint|search|torch|gen|open`
 * is the registry + CLI.  `mlopt_discovery.meta_arena` is the
 * mid-search construction board — `fire` / `saturate` / `declare` /
 * `claim` / `extract`.
 *
 * What the priors measurably buy (`project/adrs/0005-the-open-game.md`):
 * the baked learned prior's ordering **beats the cost-delta heuristic on
 * held-out pages, AP 0.133 vs 0.045**.  The episode transcript below is
 * the meta-arena's shape (`project/retros/meta-arena.md`); the move
 * rewards are the toy's static model, labelled.
 */

export type Move = "fire" | "saturate" | "declare" | "claim" | "extract";

/** One step of an episode. */
export interface Step {
  move: Move;
  reward: number;
  note: string;
}

/** The domains `play` registers. */
export const DOMAINS: readonly string[] = ["meta", "joint", "search", "torch", "gen", "open"];

/** A representative episode on the meta-arena board. */
export const EPISODE: readonly Step[] = [
  {
    move: "saturate",
    reward: 0,
    note: "a bounded closure step — the old game's whole move, now one option",
  },
  { move: "declare", reward: 0, note: "insert a construction (fold / lift / compose) mid-search" },
  { move: "fire", reward: 0, note: "one law application — the schedule/cost lever" },
  { move: "claim", reward: 0, note: "declare the handler's own pattern, bound to a kernel" },
  {
    move: "extract",
    reward: 0.5,
    note: "terminal: extract, replay the certificate, score the delta",
  },
];

/** A player, and its measured ordering quality. */
export interface Player {
  id: string;
  label: string;
  /** Average precision of the move ordering on held-out pages. */
  ap: number;
  note: string;
  sourced: boolean;
}

/** The measured players — AP from `project/adrs/0005-the-open-game.md`. */
export const PLAYERS: readonly Player[] = [
  {
    id: "learned",
    label: "learned prior (baked)",
    ap: 0.133,
    note: "the trained, baked prior — beats the heuristic on held-out pages",
    sourced: true,
  },
  {
    id: "heuristic",
    label: "cost-delta heuristic",
    ap: 0.045,
    note: "the ordering the prior is measured against",
    sourced: true,
  },
  {
    id: "scripted",
    label: "scripted (saturate → extract)",
    ap: 0.0,
    note: "the baseline to beat — no ordering signal",
    sourced: false,
  },
  {
    id: "random",
    label: "random",
    ap: 0.0,
    note: "a floor — legality only",
    sourced: false,
  },
];

/** The game layer, playable: step an episode and compare the players. */
export class GameLab {
  step = 0;

  get domains(): readonly string[] {
    return DOMAINS;
  }

  get episode(): readonly Step[] {
    return EPISODE;
  }

  get total(): number {
    return EPISODE.length;
  }

  /** The step reached — clamped to the last. */
  get at(): Step {
    return EPISODE[Math.min(this.step, EPISODE.length - 1)];
  }

  /** Whether the episode has reached the terminal `extract`. */
  get done(): boolean {
    return this.step >= EPISODE.length - 1;
  }

  next(): void {
    this.step = Math.min(EPISODE.length - 1, this.step + 1);
  }

  reset(): void {
    this.step = 0;
  }

  /** The steps taken so far — the transcript. */
  get transcript(): Step[] {
    return EPISODE.slice(0, this.step + 1);
  }

  /** The reward collected so far. */
  get reward(): number {
    return this.transcript.reduce((s, x) => s + x.reward, 0);
  }

  get players(): readonly Player[] {
    return PLAYERS;
  }

  /** The best-ordering player by measured AP. */
  get best(): Player {
    return PLAYERS.reduce((b, p) => (p.ap > b.ap ? p : b));
  }

  /** What the learned prior buys over the cost-delta heuristic. */
  get priorGain(): number {
    const learned = PLAYERS.find((p) => p.id === "learned")!;
    const heuristic = PLAYERS.find((p) => p.id === "heuristic")!;
    return learned.ap / heuristic.ap;
  }

  /** The learned prior and the heuristic, for the comparison view. */
  get priorComparison(): { learned: Player; heuristic: Player } {
    return {
      learned: PLAYERS.find((p) => p.id === "learned")!,
      heuristic: PLAYERS.find((p) => p.id === "heuristic")!,
    };
  }
}
