/** Module reviews — the spaced-retrieval round between arcs.
 *
 * Plan 0028 §A/§C: the course is six arcs, each ending in a review —
 * a gated multi-question round that *recombines* the arc's concepts
 * (it must not merely repeat the level checkpoints).  Passing feeds
 * the concept mastery; failing reports the weak concepts so the
 * review can send you back to the levels that teach them.
 *
 * Step 3 supplies the final module→levels mapping (28 levels) and the
 * review questions.  Every question is tagged with a concept from the
 * arc, and each arc's questions draw on **at least two** of its
 * levels — the recombination the plan's falsification demands.
 */

import type { Concept } from "./concepts.ts";
import { conceptById, LEVEL_ORDER, levelIndex } from "./concepts.ts";

export interface ReviewQuestion {
  id: string;
  /** The concept this question exercises — feeds the mastery meter. */
  concept: string;
  q: string;
  opts: readonly string[];
  ok: string;
  why: string;
}

/** An arc: a module id, its levels, and its review questions. */
export interface Module {
  id: string;
  /** The roman numeral shown in the shell. */
  roman: string;
  title: string;
  /** One line describing the arc. */
  arc: string;
  /** The level ids in the arc, in order. */
  levels: readonly string[];
  /** The review questions — the spaced-retrieval round. */
  questions: readonly ReviewQuestion[];
}

/** The six arcs (plan 0028 §A), each ending in its review. */
export const MODULES: readonly Module[] = [
  {
    id: "foundations",
    roman: "I",
    title: "Foundations",
    arc: "What a program is, where it comes from, and what equality means.",
    levels: ["tree", "equal", "corpus", "reach", "structure"],
    questions: [
      {
        id: "f1",
        concept: "semantic-equality",
        q: "Two trees compute the same function at different FLOP counts. Which idea makes that possible?",
        opts: [
          "associativity — bracketing changes the cost, not the result",
          "commutativity",
          "a bigger search budget",
        ],
        ok: "associativity — bracketing changes the cost, not the result",
        why: "Levels 1 + 2 + 4: equality is semantic (equal), the cost gap is associativity (equal), and the structure that licenses it is a monoid (structure).",
      },
      {
        id: "f2",
        concept: "reach",
        q: "Your law set has no associativity. Can a bigger search budget find a cheaper bracketing?",
        opts: ["yes", "no — reach is the closure of your laws", "only with a GPU"],
        ok: "no — reach is the closure of your laws",
        why: "Levels 3 + 4: reach is the closure (reach) of the law set, and laws come from structure (structure) — search only explores what the laws generate.",
      },
      {
        id: "f3",
        concept: "corpus",
        q: "Where do the programs the census counts come from?",
        opts: [
          "hand-written terms",
          "real `nn.Module`s through `torch.export`",
          "randomly generated trees",
        ],
        ok: "real `nn.Module`s through `torch.export`",
        why: "Levels 2 + 3: the corpus (corpus) is real exported programs, and its size bounds how much the closure (reach) can ever be asked to rewrite.",
      },
      {
        id: "f4",
        concept: "monoid",
        q: "Which structure licenses re-bracketing a chain — the law the whole course rests on?",
        opts: ["monoid", "comonoid", "the frontier"],
        ok: "monoid",
        why: "Levels 4 + 1: a monoid (structure) gives associativity, which is exactly the equality (equal) that lets two trees be one function at different costs.",
      },
      {
        id: "f5",
        concept: "work",
        q: "A four-matrix chain `a·b·c·d` has how many bracketings?",
        opts: ["5 — Catalan(3)", "1", "12"],
        ok: "5 — Catalan(3)",
        why: "Levels 0 + 3: the tree (tree) prices each bracketing's work (work), and the reach (reach) of one associativity law is all five bracketings.",
      },
      {
        id: "f6",
        concept: "intake",
        q: "A workload exports but binds an op the sink cannot lower. Which intake class does it land in?",
        opts: ["ingested", "census-only", "rejected"],
        ok: "census-only",
        why: "Levels 2 + 3: the intake (corpus) counts it in the census but cannot bind it, so it never enters the reachable set (reach) the laws act on.",
      },
    ],
  },
  {
    id: "correctness",
    roman: "II",
    title: "Correctness & choice",
    arc: "Certificates, frontiers, search.",
    levels: ["referee", "choice", "search", "learn"],
    questions: [
      {
        id: "c1",
        concept: "certificate",
        q: "A policy proposes a move. What decides whether it is legal?",
        opts: ["the cost model", "the certificate, replayed", "the search budget"],
        ok: "the certificate, replayed",
        why: "Levels 5 + 7: legality is semantic — the certificate (referee) replays the derivation; a policy (search) only orders moves.",
      },
      {
        id: "c2",
        concept: "policy",
        q: "What may a search policy never do?",
        opts: ["pick a bad move", "change what is legal", "stop early"],
        ok: "change what is legal",
        why: "Levels 7 + 5: a policy (search) can be wrong, but legality is fixed by the certificate (referee) — it may only reorder.",
      },
      {
        id: "c3",
        concept: "frontier",
        q: "Why keep a Pareto frontier rather than picking the cheapest at once?",
        opts: ["it is faster", "the hardware decides which region matters", "it uses less memory"],
        ok: "the hardware decides which region matters",
        why: "Levels 6 + 7: the frontier (choice) keeps the non-dominated set; the search (search) may only order within it, never collapse the trade-off early.",
      },
      {
        id: "c4",
        concept: "structural-encoding",
        q: "You add a rewrite rule at runtime. Must a shape-encoded policy be retrained?",
        opts: ["yes", "no — a rule is scored by its shape, not its index", "only on GPU"],
        ok: "no — a rule is scored by its shape, not its index",
        why: "Levels 8 + 7: the policy is learned (learn) over a structural encoding, so an unseen legal move (search) is still a point in the same space.",
      },
      {
        id: "c5",
        concept: "scalarization",
        q: "Collapsing flops and memory into one weighted number too early does what?",
        opts: ["hides the trade-off", "finds the optimum", "speeds the search"],
        ok: "hides the trade-off",
        why: "Levels 6 + 8: scalarizing (choice) bakes in one machine's trade-off before the learned policy (learn) can use the frontier.",
      },
      {
        id: "c6",
        concept: "reward",
        q: "What does learning the policy optimize?",
        opts: [
          "the cost improvement a step buys",
          "the number of rules fired",
          "the certificate length",
        ],
        ok: "the cost improvement a step buys",
        why: "Levels 8 + 6: the reward (learn) is the cost improvement — the same axis the frontier (choice) ranks, never the certificate, which is a hard filter.",
      },
    ],
  },
  {
    id: "evaluation",
    roman: "III",
    title: "Evaluation & execution",
    arc: "What looks good versus what ran.",
    levels: ["measure", "models", "artifacts", "architecture"],
    questions: [
      {
        id: "e1",
        concept: "cost-model",
        q: "What may a performance model never do?",
        opts: ["rank candidates", "prune the semantic space", "predict latency"],
        ok: "prune the semantic space",
        why: "Levels 9 + 12: a cost model (measure) may rank; feasibility is a hard bound, and pruning equivalence is the evaluation dimension (architecture) overstepping.",
      },
      {
        id: "e2",
        concept: "delivered-pricing",
        q: "Why is one cost model not enough to pick a program?",
        opts: [
          "the launch term dominates small programs and the roofline big ones",
          "models are always slow",
          "FLOPs are simply wrong",
        ],
        ok: "the launch term dominates small programs and the roofline big ones",
        why: "Levels 10 + 9: the family (models) answers four questions — features, roofline, executor, delivered (measure) — because no single model is right everywhere.",
      },
      {
        id: "e3",
        concept: "four-dimensions",
        q: "A cost model rejects a rewrite as unsound. Which dimension did it overstep?",
        opts: ["semantics", "evaluation", "execution"],
        ok: "evaluation",
        why: "Levels 12 + 9: evaluation (architecture) ranks; semantics decides equivalence. The model (measure) is the evaluation dimension, not the referee.",
      },
      {
        id: "e4",
        concept: "delivery-artifact",
        q: "Which export format round-trips to a standalone runnable module with no catopt at inference?",
        opts: ["state_dict", "`.pt2`", "safetensors"],
        ok: "`.pt2`",
        why: "Levels 11 + 12: the artifact (artifacts) is the execution half — a `.pt2` runs standalone, while execution stays a separate dimension (architecture) from the price.",
      },
      {
        id: "e5",
        concept: "calibration",
        q: "The routed model prices a carrier 20–30 % cheaper and measures it 1.9–2.7× slower. What fixes the pick?",
        opts: [
          "a bigger search budget",
          "measured corrections fed back (calibration)",
          "nothing — it is noise",
        ],
        ok: "measured corrections fed back (calibration)",
        why: "Levels 9 + 10: calibration (measure) re-targets the family (models) — the delivered price is only as good as the target it was measured against.",
      },
      {
        id: "e6",
        concept: "native-engine",
        q: "What is the opt-in Rust engine?",
        opts: ["a different IR", "a faster closure over the same IR", "a new law set"],
        ok: "a faster closure over the same IR",
        why: "Levels 11 + 12: the native engine (artifacts) swaps the execution of the search, not the semantics — the four dimensions (architecture) are untouched.",
      },
    ],
  },
  {
    id: "discovery",
    roman: "IV",
    title: "Discovery & minting",
    arc: "Growing the vocabulary.",
    levels: ["advanced", "coherence", "mint", "truthpay", "exposure"],
    questions: [
      {
        id: "d1",
        concept: "hard-filter",
        q: "Why is the certificate a hard filter rather than a reward term?",
        opts: [
          "it is cheaper",
          "a soft reward lets correctness be traded for cost",
          "rewards are hard to compute",
        ],
        ok: "a soft reward lets correctness be traded for cost",
        why: "Levels 13 + 16: as a reward, correctness becomes spendable; the hard filter (advanced) and the guard (truthpay) keep an unsound program unrepresentable.",
      },
      {
        id: "d2",
        concept: "critical-pair",
        q: "Two laws fire on one redex and their reducts never rejoin. What fills the gap?",
        opts: [
          "a bigger search budget",
          "a mediating law — often the inverse of the split",
          "delete one law",
        ],
        ok: "a mediating law — often the inverse of the split",
        why: "Levels 14 + 15: a critical pair (coherence) is filled by a mediating law — often one the minting loop (mint) then admits as a paying law in its own right.",
      },
      {
        id: "d3",
        concept: "guard",
        q: "A synthesized guard separates its measured domain perfectly. Is it sound?",
        opts: ["yes", "not necessarily — sweep past the measured domain", "only on GPU"],
        ok: "not necessarily — sweep past the measured domain",
        why: "Levels 16 + 15: the adversarial sweep (truthpay) exists because admission (mint) can admit a guard that is perfect on the measured envs and wrong just outside.",
      },
      {
        id: "d4",
        concept: "exposure",
        q: "Of 321 exported workloads, 81 score exactly 0 % FLOPs-weighted law coverage. What are they?",
        opts: [
          "broken exports",
          "the fused-op class — `nn.*` natives as one opaque node",
          "models with no matmuls",
        ],
        ok: "the fused-op class — `nn.*` natives as one opaque node",
        why: "Levels 17 + 15: the census (mint) counts the ops; exposure (exposure) is bimodal because the e-graph cannot see a fused op to rewrite it.",
      },
      {
        id: "d5",
        concept: "minting",
        q: "A minted atom verifies at 1.78e-15 and admits five certified laws. What did the pay probe find?",
        opts: [
          "it pays on the corpus",
          "zero certified-law fires on real terms",
          "the handler failed to verify",
        ],
        ok: "zero certified-law fires on real terms",
        why: "Levels 15 + 14: minting (mint) grows the vocabulary, and the coherence layer (coherence) measures whether the new laws actually rejoin and pay.",
      },
      {
        id: "d6",
        concept: "admission",
        q: "Which law was the first minted from a *corpus op-tuple* and shipped?",
        opts: ["select_mul", "mul_unsq_nat", "silu_fold"],
        ok: "mul_unsq_nat",
        why: "Levels 15 + 16: admission (mint) promoted it after the adversarial sweep (truthpay) replaced its naive guard with the exact `flat-map-unsq` clause.",
      },
    ],
  },
  {
    id: "substrate",
    roman: "V",
    title: "Substrate & backends",
    arc: "The presentation, the ports, and the store.",
    levels: ["substrate", "ports", "cellstore", "game"],
    questions: [
      {
        id: "s1",
        concept: "sharing",
        q: "In the combinator presentation, what becomes of a value shared between two parents?",
        opts: ["a pointer", "an explicit Δ morphism", "a second copy"],
        ok: "an explicit Δ morphism",
        why: "Levels 18 + 19: sharing is a morphism (substrate), and the port (ports) reads the same term — the presentation is a substrate, not the lowering surface.",
      },
      {
        id: "s2",
        concept: "port",
        q: "You swap the torch sink for the ONNX sink. What may change in core?",
        opts: ["the IR", "the law set", "nothing — only which forms extraction may select"],
        ok: "nothing — only which forms extraction may select",
        why: "Levels 19 + 18: a backend is a port (ports); the substrate (substrate) — IR and laws — is fixed, and only `supported_ops` re-bounds extraction.",
      },
      {
        id: "s3",
        concept: "cell",
        q: "How many records does a vocabulary item become in the cell store?",
        opts: ["one cell", "one per backend", "one per law"],
        ok: "one cell",
        why: "Levels 20 + 19: every op, law, object and handle is one cell (cellstore), whatever port (ports) lowers it.",
      },
      {
        id: "s4",
        concept: "fragment-merge",
        q: "Two fragments define the same `(role, name)`. What does `merge_fragments` do by default?",
        opts: ["keeps the last", "raises — a duplicate is a conflict", "silently overrides"],
        ok: "raises — a duplicate is a conflict",
        why: "Levels 20 + 21: the merge (cellstore) refuses a silent override, and the game (game) reads and writes the same store as an episode runs.",
      },
      {
        id: "s5",
        concept: "board",
        q: "A `Board` is which three things?",
        opts: [
          "a program, a legal move set, a referee",
          "a cost model and a law",
          "a graph and a sink",
        ],
        ok: "a program, a legal move set, a referee",
        why: "Levels 21 + 20: a board (game) is a program + moves + referee, and its moves read/write cells (cellstore) in the store.",
      },
      {
        id: "s6",
        concept: "handl-bridge",
        q: "A law cell becomes what in a handl fragment?",
        opts: ["an `AstShapeDef`", "a `TransformDef`", "a `!Native`"],
        ok: "a `TransformDef`",
        why: "Levels 20 + 19: the handl bridge (cellstore) maps a law cell to a `TransformDef`, and a handle cell to a `!Native` binding a kernel a port (ports) could lower.",
      },
    ],
  },
  {
    id: "engineering",
    roman: "VI",
    title: "Engineering & payoff",
    arc: "How the project stays honest, and where to help.",
    levels: ["gates", "honest", "bounded", "deliver", "pricing", "improve"],
    questions: [
      {
        id: "g1",
        concept: "ratchet",
        q: "A ratchet is a ceiling that may move which way?",
        opts: ["up", "one way — never backwards", "either way"],
        ok: "one way — never backwards",
        why: "Levels 22 + 23: the gates (gates) pin complexity and coverage one way; the ledger of negatives (honest) is the same discipline applied to results.",
      },
      {
        id: "g2",
        concept: "negative-result",
        q: "A verified minted atom fires nothing on the corpus. What is that?",
        opts: ["a bug", "a result — truth and pay are separate measurements", "a corpus error"],
        ok: "a result — truth and pay are separate measurements",
        why: "Levels 23 + 24: a measured refusal is a result (honest); the bounded lane (bounded) is the same — certified 56, delivered 0, recorded honestly.",
      },
      {
        id: "g3",
        concept: "economics-test",
        q: "56 bounded offers certify inside budget and 0 deliver. Why?",
        opts: [
          "the bound gate is broken",
          "the bounded member priced 1.32× dense, so pricing refused it",
          "the budget was too small",
        ],
        ok: "the bounded member priced 1.32× dense, so pricing refused it",
        why: "Levels 24 + 26: the economics test (bounded) is a second gate, and the cost model (pricing) is what refused — correctly, on a measured price.",
      },
      {
        id: "g4",
        concept: "delivery",
        q: "`chainfuse` ran 315 µs eager, 253 µs compiled, 86 µs after the search minted a kernel. Which dimension is that?",
        opts: ["semantics", "execution / delivery", "evaluation"],
        ok: "execution / delivery",
        why: "Levels 25 + 24: delivery (deliver) is the execution dimension — the semantics are unchanged; the bounded lane (bounded) would need the same delivery-aware price.",
      },
      {
        id: "g5",
        concept: "inversion",
        q: "The routed model prices a carrier 20–30 % cheaper and measures it 1.9–2.7× slower. What is that called?",
        opts: ["an inversion", "a calibration", "a certificate"],
        ok: "an inversion",
        why: "Levels 26 + 25: the inversion (pricing) is a cost model's error that flips its ranking — the fix is measured feedback on the delivered price (deliver).",
      },
      {
        id: "g6",
        concept: "gap",
        q: "A gap's next step is a what?",
        opts: ["a build", "a measurement", "a rewrite"],
        ok: "a measurement",
        why: "Levels 27 + 23: a gap (improve) is closed by a measurement, not a build — the same honest-negative discipline (honest) the course taught.",
      },
    ],
  },
];

/** The module a level belongs to, or `undefined`. */
export function moduleOf(
  levelId: string,
  modules: readonly Module[] = MODULES,
): Module | undefined {
  return modules.find((m) => m.levels.includes(levelId));
}

/** A module is complete when it has levels and all of them are passed. */
export function moduleComplete(m: Module, passed: readonly string[]): boolean {
  return m.levels.length > 0 && m.levels.every((l) => passed.includes(l));
}

/** A module is *cleared* when every level is passed **and** its review passed.
 *
 * Plan 0028 §C: the module review gates the arc — clearing every level
 * is not enough; the spaced-retrieval round must also pass.  This is
 * the completion the module badges and the shell read.
 */
export function moduleCleared(
  m: Module,
  passed: readonly string[],
  reviewed: readonly string[],
): boolean {
  return moduleComplete(m, passed) && reviewed.includes(m.id);
}

/** The levels a module's review questions draw on — the recombination check.
 *
 * A review must *recombine* concepts across its arc (plan 0028 §C): its
 * questions' concepts should come from at least two different levels.
 */
export function reviewLevels(m: Module, order: readonly string[] = LEVEL_ORDER): string[] {
  const levels = new Set<string>();
  for (const q of m.questions) {
    const c = conceptById(q.concept);
    const at = c?.introduced ?? null;
    if (at && m.levels.includes(at)) levels.add(at);
  }
  return [...levels].sort((a, b) => levelIndex(a, order) - levelIndex(b, order));
}

/** A module's concepts — the concepts its levels introduce or use. */
export function moduleConcepts(m: Module, concepts: readonly Concept[]): Concept[] {
  return concepts.filter(
    (c) =>
      (c.introduced !== null && m.levels.includes(c.introduced)) ||
      c.used.some((l) => m.levels.includes(l)),
  );
}

/** The fraction of a module's questions that must be right to pass. */
export const REVIEW_PASS = 0.7;

export interface ReviewResult {
  module: string;
  correct: number;
  total: number;
  passed: boolean;
  /** Concept ids of the wrong answers — the "send you back" list. */
  weak: string[];
}

/** Grade a module review from a question→option answer map. */
export function gradeReview(
  m: Module,
  answers: Readonly<Record<string, string>>,
  threshold = REVIEW_PASS,
): ReviewResult {
  let correct = 0;
  const weak: string[] = [];
  for (const q of m.questions) {
    if (answers[q.id] === q.ok) correct += 1;
    else weak.push(q.concept);
  }
  const total = m.questions.length;
  const complete = total > 0 && m.questions.every((q) => answers[q.id] !== undefined);
  const passed = complete && correct / total >= threshold;
  return { module: m.id, correct, total, passed, weak: [...new Set(weak)] };
}

/** The mastery attempts a review produces (right or wrong, per question). */
export function reviewAttempts(
  m: Module,
  answers: Readonly<Record<string, string>>,
): { concept: string; correct: boolean }[] {
  return m.questions.map((q) => ({ concept: q.concept, correct: answers[q.id] === q.ok }));
}

/** The playable round: answer every question, then read the verdict. */
export class ReviewRound {
  answers: Record<string, string> = {};
  module: Module;
  threshold: number;

  constructor(module: Module, threshold = REVIEW_PASS) {
    this.module = module;
    this.threshold = threshold;
  }

  get questions(): readonly ReviewQuestion[] {
    return this.module.questions;
  }

  get total(): number {
    return this.module.questions.length;
  }

  get answered(): number {
    return this.module.questions.filter((q) => this.answers[q.id] !== undefined).length;
  }

  get correct(): number {
    return this.module.questions.filter((q) => this.answers[q.id] === q.ok).length;
  }

  get score(): number {
    return this.total === 0 ? 0 : this.correct / this.total;
  }

  /** Every question has an answer — the round may be graded. */
  get gate(): boolean {
    return this.total > 0 && this.answered === this.total;
  }

  get passed(): boolean {
    return this.gate && this.score >= this.threshold;
  }

  /** The concept ids behind the wrong answers. */
  get weak(): string[] {
    return [
      ...new Set(
        this.module.questions.filter((q) => this.answers[q.id] !== q.ok).map((q) => q.concept),
      ),
    ];
  }

  get result(): ReviewResult {
    return gradeReview(this.module, this.answers, this.threshold);
  }

  answer(qid: string, opt: string): void {
    this.answers = { ...this.answers, [qid]: opt };
  }

  reset(): void {
    this.answers = {};
  }
}
