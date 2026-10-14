import { Board } from "./board.ts";
import { Bounded } from "./bounded.ts";
import { CellStore } from "./cellstore.ts";
import { Challenge } from "./challenge.ts";
import { Coherence } from "./coherence.ts";
import { Compare } from "./compare.ts";
import { Corpus } from "./corpus.ts";
import { CostModelFamily } from "./models.ts";
import { Course } from "./course.ts";
import { Deliver } from "./deliver.ts";
import { DeliveryArtifacts } from "./artifacts.ts";
import { Episode } from "./episode.ts";
import { Evaluation } from "./evaluation.ts";
import { Expose } from "./exposure.ts";
import { Frontier } from "./frontier.ts";
import { Game } from "./game.ts";
import { Gates } from "./gates.ts";
import { Honest } from "./honest.ts";
import { Improve } from "./improve.ts";
import { Inspect } from "./inspect.ts";
import { Leaks } from "./leaks.ts";
import { Mint } from "./mint.ts";
import { Players } from "./players.ts";
import { Ports } from "./ports.ts";
import { Pricing } from "./pricing.ts";
import { Reach } from "./reach.ts";
import { Structures } from "./structures.ts";
import { Substrate } from "./substrate.ts";
import { Swap } from "./swap.ts";
import { Tensions } from "./tensions.ts";
import { Truthpay } from "./truthpay.ts";

/** Every Alpine component the course registers. */
export const COMPONENTS: Record<string, () => object> = {
  course: () => new Course(),
  inspect: () => new Inspect(),
  compare: () => new Compare(),
  corpus: () => new Corpus(),
  reach: () => new Reach(),
  structures: () => new Structures(),
  board: () => new Board(),
  frontier: () => new Frontier(),
  players: () => new Players(),
  challenge: () => new Challenge(),
  episode: () => new Episode(),
  evaluation: () => new Evaluation(),
  costmodels: () => new CostModelFamily(),
  artifacts: () => new DeliveryArtifacts(),
  swap: () => new Swap(),
  leaks: () => new Leaks(),
  tensions: () => new Tensions(),
  coherence: () => new Coherence(),
  mint: () => new Mint(),
  truthpay: () => new Truthpay(),
  expose: () => new Expose(),
  substrate: () => new Substrate(),
  ports: () => new Ports(),
  cellstore: () => new CellStore(),
  game: () => new Game(),
  gates: () => new Gates(),
  honest: () => new Honest(),
  bounded: () => new Bounded(),
  deliver: () => new Deliver(),
  pricing: () => new Pricing(),
  improve: () => new Improve(),
};
