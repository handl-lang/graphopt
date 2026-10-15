import {
  CONCEPTS,
  GAPS,
  LEVEL_ORDER,
  MODULES,
  ReviewRound,
  conceptById,
  conceptsIntroducedAt,
  conceptsTaughtAt,
  earned,
  emptyMastery,
  forLevel,
  forwardRefs as conceptForwardRefs,
  glossary as conceptGlossary,
  levelBase,
  levelIndex,
  locked,
  masteredCount,
  moduleCleared,
  moduleComplete,
  nextRank,
  prerequisites as conceptPrerequisites,
  rank,
  rankProgress,
  record,
  summary,
  tier,
  totalXp,
  weakest,
  WIDGET_BONUS,
} from "engine";
import type {
  Badge,
  BadgeState,
  Concept,
  ForwardRef,
  Gap,
  GlossaryRow,
  MasteryState,
  MasterySummary,
  Module,
  Rank,
  ReviewResult,
  Tier,
  WeakConcept,
} from "engine";

/** One question in a level's multi-part checkpoint.
 *
 * Plan 0028 §B step 6: a checkpoint is *multi-part* — at least two
 * questions, at least one of which is answerable only from the live
 * interactive.  Each part carries the {@link concept} it exercises, so
 * the mastery meter is fed by the questions rather than by the level.
 */
export interface CheckpointPart {
  q: string;
  opts: string[];
  ok: string;
  why: string;
  /** The concept this question exercises — feeds the mastery meter. */
  concept: string;
  /** True when the answer is only visible in the live widget. */
  interactive?: boolean;
}

/** A level's checkpoint: the primary question's fields plus every part.
 *
 * `parts[0]` mirrors the top-level `q`/`opts`/`ok`/`why` exactly (built
 * by {@link checkpoint}), so the single-question shape stays readable
 * while the gate is the whole round.
 */
export interface Checkpoint extends CheckpointPart {
  parts: CheckpointPart[];
}

/** A file pointer into the real project — plan 0028 §B step 8. */
export interface RepoRef {
  path: string;
  note: string;
}

export interface Level {
  id: string;
  title: string;
  /** The ladder one-liner — the course's summary of itself. */
  idea: string;
  objective: string;
  /** §B 1 — why you are here: ties this level to the one before. */
  why: string;
  /** §B 2 — the concepts this level introduces (its new terms), in order. */
  terms: string[];
  /** §B 6 — the multi-part checkpoint. */
  checkpoint: Checkpoint;
  /** §B 7 — you can now: the outcome restated as a capability. */
  outcome: string;
  /** §B 8 — in the real project: file pointers + the retro/ADR. */
  repo: RepoRef[];
  /** Which arc (module id) this level belongs to. */
  module: string;
}

/** Build a checkpoint from its primary question plus the remaining parts.
 *
 * Keeps `parts[0]` identical to the top-level fields by construction —
 * no hand-duplication to drift.
 */
function checkpoint(primary: CheckpointPart, ...extras: CheckpointPart[]): Checkpoint {
  return { ...primary, parts: [primary, ...extras] };
}

/** One concept's row in the mastery meter. */
export interface ConceptRow {
  id: string;
  label: string;
  def: string;
  tier: Tier;
  /** The level that teaches it. */
  level: string;
  levelTitle: string;
}

/** A module and the concept rows it teaches. */
export interface ModuleMastery {
  module: Module;
  concepts: ConceptRow[];
}

/** A module's review state — what the review panel reads (plan 0028 §C). */
export interface ModuleReviewState {
  module: Module;
  /** Every level in the arc passed. */
  levelsCleared: boolean;
  /** The review passed. */
  reviewed: boolean;
  /** Levels cleared *and* review passed — the gated completion. */
  cleared: boolean;
  /** The review is unlocked (its levels are all cleared). */
  unlocked: boolean;
  /** How many review questions are answered. */
  answered: number;
  total: number;
  /** The round passes as answered so far. */
  passed: boolean;
  /** The concept ids behind the wrong answers. */
  weak: string[];
}

/** The completion state — plan 0028 §C's "completion that means something". */
export interface CompletionState {
  xp: number;
  rank: Rank;
  nextRank: Rank | null;
  levelsCleared: number;
  levelsTotal: number;
  reviewsPassed: number;
  reviewsTotal: number;
  weak: number;
  familiar: number;
  mastered: number;
  conceptsTotal: number;
  ratio: number;
  badgesEarned: number;
  badgesTotal: number;
  /** The project's open work, from the improve level's data. */
  gaps: Gap[];
}

/** Where progress lives.
 *
 * Plan 0028 step 3: the saved shape **changed** — module reviews add a
 * `reviews` field (moduleId → questionId → option) so spaced retrieval
 * survives a reload.  The key moves to `v5`; a `v4` blob is ignored,
 * so saved progress resets cleanly (announced in the settings text).
 */
export const STORAGE_KEY = "catopt-course-v5";

interface Saved {
  i: number;
  answers: Record<string, string>;
  engaged?: string[];
  reviewed?: string[];
  /** Module review answers, keyed moduleId → questionId → option. */
  reviews?: Record<string, Record<string, string>>;
}

/** The storage key for a checkpoint part — part 0 keeps the level id, so
 * saved progress from the single-question shape still restores. */
export function partKey(levelId: string, i: number): string {
  return i === 0 ? levelId : `${levelId}#${i}`;
}

/** A localStorage shim that no-ops when storage is unavailable. */
function store(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

/** The course shell: levels, gating, progress, and persistence. */
export class Course {
  levels: Level[] = [
    {
      id: "tree",
      title: "A program is a tree",
      idea: "ops over data, one root",
      objective: "Read a tensor program as a tree of operations over data.",
      module: "foundations",
      why: "The course opens here: nothing about optimization makes sense until a program is a concrete tree.",
      terms: ["program-tree", "work"],
      outcome:
        "read any tensor program as a tree of operations, name its root, and price its work node by node.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/ir.py",
          note: "the real IR — nodes, leaves, one root",
        },
        {
          path: "project/adrs/0003-evaluation-is-an-independent-dimension.md",
          note: "why 'work' is one dimension among four",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "In `(a·b)·c`, which node produces the final result?",
          opts: ["a", "the outer matmul", "c"],
          ok: "the outer matmul",
          why: "The root. Every other node feeds it. Optimization is choosing a different tree that computes the same thing — the root stays the answer.",
          concept: "program-tree",
        },
        {
          q: "In the live tree, switch to `a·b·c·d`. What total does the widget show?",
          opts: ["128", "180", "220"],
          ok: "220",
          why: "The widget sums every node's work: the worst bracketing of four matrices is 220 FLOPs — the tree the next level reassociates.",
          concept: "work",
          interactive: true,
        },
      ),
    },
    {
      id: "equal",
      title: "Two programs, one function",
      idea: "equality is not syntactic",
      objective: "Different trees can compute the same function — at different costs.",
      module: "foundations",
      why: "You can read one tree; now the observation the whole system rests on — two different trees can be one function.",
      terms: ["semantic-equality", "associativity"],
      outcome:
        "distinguish semantic from syntactic equality, and show that two equal programs can cost different amounts.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/egraph/certs.py",
          note: "the certificates that witness an equality",
        },
        {
          path: "packages/catopt-core/src/catopt_core/laws/ruleset.py",
          note: "the shipped law set (associativity among them)",
        },
        {
          path: "project/adrs/0002-categorical-re-expression-thesis.md",
          note: "equality as the object of the whole system",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "With a : 2×3, b : 3×4, c : 4×5, the two trees cost 180 and 128 FLOPs. Same function?",
          opts: ["yes", "no", "only if the shapes match"],
          ok: "yes",
          why: "Matrix multiplication is associative: bracketing changes the cost, never the result. That gap — same function, different cost — is the whole opportunity.",
          concept: "semantic-equality",
        },
        {
          q: "Reveal the costs: which bracketing is cheaper?",
          opts: ["a·(b·c)", "(a·b)·c", "they tie"],
          ok: "(a·b)·c",
          why: "128 vs 180 FLOPs for the same answer. Reassociating a matrix chain is classic (Sethi–Ullman); the point is that a rewriter that cannot reassociate is stuck with the tree it was handed.",
          concept: "associativity",
          interactive: true,
        },
      ),
    },
    {
      id: "corpus",
      title: "Where programs come from",
      idea: "real workloads through the export boundary",
      objective: "The system is measured on real exported programs, not hand-written ones.",
      module: "foundations",
      why: "You can tell two trees apart by function — but where do the trees the system is actually asked to rewrite come from?",
      terms: ["corpus", "export", "intake"],
      outcome:
        "trace a workload from a real `nn.Module` through `torch.export` and the intake, and read what each class of the funnel means.",
      repo: [
        {
          path: "python -m catopt_discovery.intake",
          note: "the intake path: export → classify → persist",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/zoo.py",
          note: "the held-out workload zoo",
        },
        {
          path: "project/retros/workload-intake.md",
          note: "166 workloads in, three lowering defects out",
        },
        {
          path: "project/retros/model-zoo-yield.md",
          note: "the held-out zoo: what fires, what pays",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "Where do the programs the census counts come from?",
          opts: [
            "hand-written terms",
            "real `nn.Module`s through `torch.export`",
            "randomly generated trees",
          ],
          ok: "real `nn.Module`s through `torch.export`",
          why: "The corpus is real code. torch's own `nn.*` modules turned out to be the richer source: `permute`, `unflatten`, packed-QKV attention — spellings the hand corpus never wrote.",
          concept: "corpus",
        },
        {
          q: "In the live funnel, how many of the twelve workloads land in `census-only`?",
          opts: ["1", "5", "6"],
          ok: "5",
          why: "Five of the twelve export but bind an op the sink cannot lower — the pooling trio, the loss family, `gru.input`, `col2im`. They count in the census but never enter the reachable set.",
          concept: "export",
          interactive: true,
        },
        {
          q: "The intake caught three real modules lowering *wrong*. What does that show?",
          opts: [
            "the e-graph is unsound",
            "the boundary can emit a term that was never the program — the verify gate catches it",
            "torch.export is broken",
          ],
          ok: "the boundary can emit a term that was never the program — the verify gate catches it",
          why: "Nothing in the e-graph, the laws or the executor was wrong: `export_to_ir` dropped an elided aten `dim` default (twice) and a `torch.dtype` kwarg. Lowering the ingested term and diffing fp64 against the original found all three.",
          concept: "intake",
        },
      ),
    },
    {
      id: "reach",
      title: "The reachable set",
      idea: "reach is the closure of your laws",
      objective: "Reach is the closure of your laws, not the power of your search.",
      module: "foundations",
      why: "If two trees can be equal, how many equal trees are there? The answer is a language, not a number.",
      terms: ["reach", "closure", "e-graph"],
      outcome:
        "explain reach as the closure of your laws, computed by saturating an e-graph, and show that more search over fewer laws finds nothing new.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/egraph/core.py",
          note: "equality saturation — the closure, computed",
        },
        {
          path: "packages/catopt-core/src/catopt_core/egraph/match.py",
          note: "where a law's LHS matches a redex",
        },
        {
          path: "project/retros/law-rediscovery.md",
          note: "what a bounded closure actually reaches",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "What bounds how many equivalent programs you can ever find?",
          opts: ["the search budget", "the laws you have", "the hardware"],
          ok: "the laws you have",
          why: "Search only explores what the laws generate. A bigger budget over a weak law set still finds nothing new.",
          concept: "reach",
        },
        {
          q: "With the six laws on, how many equivalent programs does the widget reach?",
          opts: ["2", "12", "40"],
          ok: "12",
          why: "The closure — the fixed point of applying the laws from the seeds — holds 12 forms, cheapest 32 FLOPs. The structure that computes it is the e-graph: it holds every equivalent form at once.",
          concept: "closure",
          interactive: true,
        },
        {
          q: "Flip the widget to 'no laws'. How many programs can the rewriter reach?",
          opts: ["1", "2", "12"],
          ok: "2",
          why: "Two seeds and no rewrites — the rewriter never leaves what it was handed, so tuning its schedule cannot find a different tree. Reach is the language, not the effort.",
          concept: "e-graph",
          interactive: true,
        },
      ),
    },
    {
      id: "structure",
      title: "Laws come from structure",
      idea: "monoid / comonoid / trace / product",
      objective: "A law is a consequence of algebraic structure, not an op pattern.",
      module: "foundations",
      why: "Reach is bounded by the laws you have — so where do laws come from?",
      terms: ["law", "algebraic-structure", "monoid", "comonoid"],
      outcome:
        "name the algebraic structure a program instantiates, and read the law it licenses — not from an op's name, but from the algebra.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/laws/tensor.py",
          note: "the hand-written tensor law kernel",
        },
        {
          path: "packages/catopt-core/src/catopt_core/laws/base.py",
          note: "the law record — pattern, guard, proof",
        },
        {
          path: "project/adrs/0002-categorical-re-expression-thesis.md",
          note: "structure as the source of laws",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "Why can't an op-level rule find `(Q·Kᵀ)·V → Q·(Kᵀ·V)`?",
          opts: [
            "it is not a valid rewrite",
            "it is a consequence of associativity plus shapes, not an op pattern",
            "it needs a special kernel",
          ],
          ok: "it is a consequence of associativity plus shapes, not an op pattern",
          why: "No op named 'attention' is involved. The law is associativity; the choice is driven by shape-aware costing. An op-pattern language cannot express that.",
          concept: "law",
        },
        {
          q: "Pick each structure in the widget. Which one licenses duplication, Δ(x) = (x, x)?",
          opts: ["monoid", "comonoid", "traced monoid"],
          ok: "comonoid",
          why: "A monoid gives associativity (re-bracketing); a comonoid is its dual — duplication, the law behind pairing k shared-input projections into one GEMM.",
          concept: "comonoid",
          interactive: true,
        },
      ),
    },
    {
      id: "referee",
      title: "Staying correct",
      idea: "the certificate is the referee",
      objective: "Free rewriting is dangerous. Certificates make it safe.",
      module: "correctness",
      why: "Laws generate reach — but free rewriting is dangerous. Something must decide what is true.",
      terms: ["certificate", "referee", "oracle"],
      outcome:
        "explain why legality is semantic, that a certificate is a replayable proof, and that a numeric oracle is what decides truth.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/egraph/certs.py",
          note: "certificate replay",
        },
        {
          path: "packages/catopt-core/src/catopt_core/egraph/proof.py",
          note: "the proof object a rewrite carries",
        },
        {
          path: "project/retros/matmul-unsound-fix.md",
          note: "a false equality the oracle caught",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "What makes a rewrite legal?",
          opts: [
            "the price approves it",
            "a law licenses it, and the certificate replays",
            "the search chose it",
          ],
          ok: "a law licenses it, and the certificate replays",
          why: "Legality is semantic. Cost and search have no vote — which is exactly why a heuristic proposer cannot produce a wrong program.",
          concept: "certificate",
        },
        {
          q: "Apply `assoc` on the chain, then replay the derivation. What does the referee report?",
          opts: ["certificate valid — max diff ≈ 0", "MISMATCH", "it refuses the move"],
          ok: "certificate valid — max diff ≈ 0",
          why: "The referee re-checks the derivation on real numbers (max|original − current|). It runs after the search, so a bad move is possible — an unsound one is not.",
          concept: "referee",
          interactive: true,
        },
      ),
    },
    {
      id: "choice",
      title: "Choosing among equals",
      idea: "keep the frontier, don't scalarize",
      objective: "Among many equals, do not collapse to one scalar too early.",
      module: "correctness",
      why: "You can now trust a class of equal programs — so which one do you ship?",
      terms: ["frontier", "non-dominated-set", "scalarization"],
      outcome:
        "keep the non-dominated set instead of scalarizing early, and explain why the target, not the search, picks the trade-off.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/pareto.py",
          note: "the Pareto frontier of equal programs",
        },
        {
          path: "project/adrs/0003-evaluation-is-an-independent-dimension.md",
          note: "evaluation ranks; it does not decide",
        },
        {
          path: "project/retros/eval-axis-selection.md",
          note: "which axis the target actually picks",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "Why keep a Pareto frontier instead of picking the cheapest immediately?",
          opts: [
            "it is faster",
            "the hardware decides which region matters",
            "it uses more memory",
          ],
          ok: "the hardware decides which region matters",
          why: "One scalar bakes in one machine's trade-off. The frontier keeps the options; the target picks.",
          concept: "frontier",
        },
        {
          q: "With both flops and memory on, how many of the six candidates stay on the frontier?",
          opts: ["1", "5", "6"],
          ok: "5",
          why: "Five candidates are non-dominated — nothing beats each on every axis. Only matmul drops out (dominated on both). Scalarizing early would keep just one and hide the trade-off.",
          concept: "non-dominated-set",
          interactive: true,
        },
      ),
    },
    {
      id: "search",
      title: "Searching the space",
      idea: "a policy orders legal moves",
      objective: "The space is exponential. A policy decides what to try.",
      module: "correctness",
      why: "The frontier can be astronomically large — you cannot enumerate it. Something must choose.",
      terms: ["policy", "legal-move", "search-space"],
      outcome:
        "describe a policy as an ordering over legal moves, and explain that search may be wrong but never illegal.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/policies.py",
          note: "the policy protocol — order legal moves",
        },
        {
          path: "packages/catopt-core/src/catopt_core/search_env.py",
          note: "the environment a policy steps through",
        },
        {
          path: "project/retros/contraction-policy.md",
          note: "a shipped policy, measured",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "What may a search policy never do?",
          opts: ["pick a bad move", "change what is legal", "stop early"],
          ok: "change what is legal",
          why: "A policy only orders legal moves. It can be wrong — it can pick a bad move — but it cannot make a move legal or illegal.",
          concept: "policy",
        },
        {
          q: "In 'Match the model', the policy's fixed three-move cost is shown. What is it?",
          opts: ["128", "180", "220"],
          ok: "128",
          why: "Three greedy moves from `a·(b·c)` reach 128 FLOPs — the same floor the comparator found. The policy orders the moves; it never changes which moves are legal.",
          concept: "legal-move",
          interactive: true,
        },
      ),
    },
    {
      id: "learn",
      title: "Learning the policy",
      idea: "trajectories, then reward",
      objective: "A policy can be learned: from trajectories, or from reward.",
      module: "correctness",
      why: "A policy decides what to try — but where does the policy itself come from?",
      terms: ["reward", "structural-encoding", "features", "learned-policy"],
      outcome:
        "frame optimization as a game (state = features, action = rule, reward = improvement) and explain why a shape-encoded action space generalizes to unseen rules.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/trajectories.py",
          note: "the trajectory data a policy learns from",
        },
        {
          path: "packages/catopt-torch/src/catopt_torch/learned_policy.py",
          note: "the shipped learned policy",
        },
        {
          path: "project/retros/contraction-player-shipped.md",
          note: "what the player buys, and where it loses",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "You add a new rewrite rule at runtime. Must the learned policy be retrained?",
          opts: ["yes", "no, if actions are encoded structurally", "only on GPU"],
          ok: "no, if actions are encoded structurally",
          why: "Encode a rule by its shape and an unseen rule is still a point in the same space. Encode it by index and you would have to retrain.",
          concept: "learned-policy",
        },
        {
          q: "Step the episode once from `a·(b·c)`. Which rule does it take?",
          opts: ["assoc", "comm", "cse"],
          ok: "assoc",
          why: "The state is the program's static features; the action is a rule; the reward is the cost improvement. Only `assoc` improves here, and the rule is encoded by shape, not index.",
          concept: "structural-encoding",
          interactive: true,
        },
      ),
    },
    {
      id: "measure",
      title: "Measuring the machine",
      idea: "features describe, a model predicts",
      objective: "Static features describe; hardware is truth; a model predicts.",
      module: "evaluation",
      why: "You have a policy; now it needs a price to steer by. Static features describe; a model predicts.",
      terms: ["cost-model", "roofline", "ranking", "calibration"],
      outcome:
        "describe a program with static features, predict runtime from a model, and explain why that model may rank but never prune.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/features.py",
          note: "ProgramFeatures — static, no execution",
        },
        {
          path: "packages/catopt-core/src/catopt_core/perf_model.py",
          note: "the performance model",
        },
        {
          path: "packages/catopt-core/src/catopt_core/cost/roofline.py",
          note: "compute-or-memory bound",
        },
        {
          path: "project/retros/measured-cost-model.md",
          note: "a model corrected by measurement",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "What may a performance model never do?",
          opts: ["rank candidates", "prune the semantic space", "predict latency"],
          ok: "prune the semantic space",
          why: "Feasibility — can the backend lower this at all? — is a hard bound. Performance is a soft ranking. Pruning on a prediction loses certified alternatives.",
          concept: "cost-model",
        },
        {
          q: "Drag `launch µs` to its maximum on the small program. Which term dominates the prediction?",
          opts: ["the roofline (compute)", "the launch overhead", "neither — they cancel"],
          ok: "the launch overhead",
          why: "On a small program the per-op launch cost dominates; on a big one the roofline does. Which term wins is hardware — which is why one model is not enough.",
          concept: "roofline",
          interactive: true,
        },
        {
          q: "A prediction orders candidates — what may it never do with them?",
          opts: ["rank them", "prune the semantic space", "price them"],
          ok: "prune the semantic space",
          why: "A cost model is a ranking, not the truth. It may order; it may never remove a certified alternative — that is a hard feasibility bound, decided elsewhere.",
          concept: "ranking",
          interactive: true,
        },
      ),
    },
    {
      id: "models",
      title: "The cost-model family",
      idea: "features → roofline → executor → delivered",
      objective: "One model is not enough — a family prices a term, and a profile re-targets it.",
      module: "evaluation",
      why: "You met *a* cost model; the real system prices with a family, and calibrates it to a machine.",
      terms: ["cost-model-family", "delivered-pricing", "target-profile"],
      outcome:
        "explain why features, a roofline, an executor route and delivered pricing are four different questions, and how a `TargetProfile` re-targets them.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/perf_model.py",
          note: "the roofline prediction",
        },
        {
          path: "packages/catopt-orchestrator/src/catopt_orchestrator/optimize.py",
          note: "`delivered_cost_for` and the carrier-upgrade comparison",
        },
        {
          path: "packages/catopt-core/src/catopt_core/profile.py",
          note: "`TargetProfile` — the measured target",
        },
        {
          path: "project/retros/measured-cost-model.md",
          note: "the inversion, and the corrections that fix it",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "Which model bills a term under the lowering it would actually be delivered by?",
          opts: ["the roofline", "delivered pricing", "static features"],
          ok: "delivered pricing",
          why: "The roofline bounds the work; the executor adds the route's launches; delivered pricing bills the term as it would run — the only one that can catch a slower carrier winning a modeled comparison.",
          concept: "cost-model-family",
        },
        {
          q: "In the widget, switch on delivered pricing with the measured factors. Which route does the SSM case now pick?",
          opts: ["the batched carrier", "the generic route", "it ties"],
          ok: "the generic route",
          why: "Uncalibrated, the delivered model prices the batched carrier ~25 % cheaper and it measures 1.9–2.7× slower — the inversion. Feeding the measured per-family corrections back lands the generic route, the measured-fastest member.",
          concept: "delivered-pricing",
          interactive: true,
        },
        {
          q: "What does `TargetProfile` carry that the built-in constants do not?",
          opts: [
            "the law set",
            "measured constants plus per-family corrections",
            "the certificate",
          ],
          ok: "measured constants plus per-family corrections",
          why: "`scripts/calibrate_profile.py` micro-benchmarks the machine and writes the constants and the executor-family corrections, so extraction ranks by this machine's numbers, not the built-in ones.",
          concept: "target-profile",
        },
      ),
    },
    {
      id: "artifacts",
      title: "What 'runnable' means",
      idea: "module · safetensors · torchscript · `.pt2` · CUDA graph · native",
      objective: "A delivered term must run standalone — the artifact and the runner are choices.",
      module: "evaluation",
      why: "A price is not a program; something has to ship as a file that runs without catopt.",
      terms: ["delivery-artifact", "cuda-graph", "native-engine"],
      outcome:
        "name the artifact a delivered term ships as, and explain what each needs to run standalone.",
      repo: [
        {
          path: "packages/catopt-torch/src/catopt_torch/export.py",
          note: "`export_optimized` / `load_optimized` — the artifact formats",
        },
        {
          path: "packages/catopt-cuda/src/catopt_cuda/runners.py",
          note: "the CUDA-graph runner",
        },
        {
          path: "packages/catopt-native/",
          note: "the opt-in PyO3/Rust engine",
        },
        {
          path: "project/retros/onnx-package.md",
          note: "a second backend, verified through onnxruntime",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "Which export format round-trips to a standalone runnable module with no catopt at inference?",
          opts: ["state_dict", "`.pt2`", "safetensors"],
          ok: "`.pt2`",
          why: "`.pt2` roundtrips standalone (docs/api.md). A `state_dict` or `safetensors` is weights only — it needs the model class back; torchscript bakes the graph but still needs torch.",
          concept: "delivery-artifact",
        },
        {
          q: "In the widget, pick the CUDA-graph replay. What does it need to run?",
          opts: [
            "a CUDA device and the captured input shapes",
            "a bigger search budget",
            "a new law",
          ],
          ok: "a CUDA device and the captured input shapes",
          why: "`CudaGraphRunner` captures the executor into one replay — compile-free, but it needs the device and the exact shapes it was captured with, since the graph is fixed.",
          concept: "cuda-graph",
          interactive: true,
        },
        {
          q: "What is the opt-in Rust engine?",
          opts: ["a different IR", "a faster closure over the same IR", "a new law set"],
          ok: "a faster closure over the same IR",
          why: "`catopt-native` is the PyO3/Rust search engine (~17× on match-bound closures) — the same IR and laws, a faster closure. The pure-Python engine stays the reference.",
          concept: "native-engine",
        },
      ),
    },
    {
      id: "architecture",
      title: "The architecture",
      idea: "four dimensions, swappable parts",
      objective: "Everything so far is one of four jobs. Keeping them apart is the design.",
      module: "evaluation",
      why: "Every piece you have touched belongs to exactly one of four jobs — keeping them apart is the design.",
      terms: ["four-dimensions", "evaluation-dimension"],
      outcome:
        "assign any component to one of the four dimensions — semantics, search, evaluation, execution — and explain why that separation makes each replaceable.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/pipeline.py",
          note: "the Backend bundle the four dimensions compose",
        },
        {
          path: "packages/catopt-core/src/catopt_core/ports.py",
          note: "every adapter contract, named",
        },
        {
          path: "project/adrs/0003-evaluation-is-an-independent-dimension.md",
          note: "the ADR the whole architecture rests on",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "A cost model rejects a rewrite as unsound. Which rule did it break?",
          opts: [
            "search may not decide equivalence",
            "evaluation may not decide equivalence",
            "execution may not decide equivalence",
          ],
          ok: "evaluation may not decide equivalence",
          why: "EVALUATION ranks; SEMANTICS decides equivalence. The certificate is the referee — that separation is the whole architecture.",
          concept: "evaluation-dimension",
        },
        {
          q: "Swap the backend to a CPU sink. What stays true of the core?",
          opts: [
            "the law set changes",
            "supported_ops re-bounds extraction",
            "the certificate is recomputed",
          ],
          ok: "supported_ops re-bounds extraction",
          why: "Swapping a backend is a port change: the core counter stays at zero and only `supported_ops` moves, re-bounding what extraction may select. Each dimension is replaceable because they are kept apart.",
          concept: "four-dimensions",
          interactive: true,
        },
      ),
    },
    {
      id: "advanced",
      title: "Advanced tensions",
      idea: "where the design gets hard",
      objective: "Where the design actually gets hard.",
      module: "discovery",
      why: "You have the whole machine; these are the cases that separate a toy from a system.",
      terms: ["hard-filter"],
      outcome:
        "recognise where a design gets hard — reward versus filter, ranking versus bound — and say which dimension each tension belongs to.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/meta.py",
          note: "the rule matcher where a guard raising rejects a candidate",
        },
        {
          path: "project/adrs/0003-evaluation-is-an-independent-dimension.md",
          note: "the separation the tensions stress",
        },
        {
          path: "project/retros/four-dimension-recon.md",
          note: "an audit of the four dimensions in practice",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "Why is the certificate a hard filter rather than a reward term?",
          opts: [
            "it is cheaper to compute",
            "a soft reward lets the optimizer trade correctness for cost",
            "rewards are hard to compute",
          ],
          ok: "a soft reward lets the optimizer trade correctness for cost",
          why: "As a reward, correctness becomes spendable. As a hard filter, an unsound program is unrepresentable: search can only lose cost, never truth.",
          concept: "hard-filter",
        },
        {
          q: "Case 4 in the widget: the model predicts a form is slow and the search skips it. What went wrong?",
          opts: [
            "nothing",
            "a soft ranking was used as a hard bound",
            "the model was simply wrong",
          ],
          ok: "a soft ranking was used as a hard bound",
          why: "Feasibility may prune; a prediction may only rank. Skipping on a guess can lose the certified optimum — correctness is a filter, never a spendable reward.",
          concept: "hard-filter",
          interactive: true,
        },
      ),
    },
    {
      id: "coherence",
      title: "Laws about laws",
      idea: "derivable · inverse · divergent",
      objective:
        "Laws relate — derivable, inverse, divergent. Measure the relations; fill the gaps.",
      module: "discovery",
      why: "The law library is not a flat set — laws relate, and the relations are measurable.",
      terms: ["critical-pair", "confluence", "mediating-law"],
      outcome:
        "measure the relations between laws — derivable, inverse, divergent — and fill a divergence with a mediating law.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/laws/pairing.py",
          note: "the pairwise law-relation probe",
        },
        {
          path: "packages/catopt-core/src/catopt_core/laws/ruleset.py",
          note: "the axiom/lemma split over the shipped rules",
        },
        {
          path: "project/retros/law-coherence-catalogue.md",
          note: "the measured catalogue, and the one gap",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "Two laws fire on the same redex and their reducts never rejoin. What fills the gap?",
          opts: [
            "a bigger search budget",
            "a mediating law — often the definitional inverse of the rule that caused the split",
            "delete one of the two laws",
          ],
          ok: "a mediating law — often the definitional inverse of the rule that caused the split",
          why: "The real library's one measured divergence — silu_expand × swiglu_fuse — was filled by silu_fold, x·σ(x) → silu(x): the inverse of the expansion that killed the fuse redex. Divergent pairs went 2 → 0, and the mediator is a paying law in its own right.",
          concept: "critical-pair",
        },
        {
          q: "Pick the `expand × fuse` pair and switch the fold OFF. What verdict does the probe report?",
          opts: ["pair-confluent", "library-mediated", "divergent"],
          ok: "divergent",
          why: "Without the fold the two reducts never rejoin — a genuine critical pair. Bring the fold back and it becomes library-mediated: silu_fold is the expansion's definitional inverse, the same fix the real library shipped.",
          concept: "confluence",
          interactive: true,
        },
      ),
    },
    {
      id: "mint",
      title: "The vocabulary grows itself",
      idea: "need → handler → laws → admission",
      objective:
        "New meaning enters the object language by measurement, not by hand — need → handler → laws → admission.",
      module: "discovery",
      why: "So far the vocabulary was given — five ops, six laws. The real system grows its own.",
      terms: ["minting", "admission", "census", "provenance"],
      outcome:
        "trace an atom through the minting funnel — need → signature → handler → laws → admission — and read where each law came from.",
      repo: [
        {
          path: "packages/catopt-discovery/src/catopt_discovery/pipeline.py",
          note: "the composed need → handler → laws → admission loop",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/handler_synth.py",
          note: "the handler synthesizer",
        },
        {
          path: "project/adrs/0007-semantic-atom-minting.md",
          note: "the four layers, defined",
        },
        {
          path: "project/retros/atom-synth-loop.md",
          note: "the loop end to end, and its pay-null",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "The funnel mints `scan_fold`, verifies it at 1.78e-15, and admits five certified laws. What did the pay probe find?",
          opts: [
            "it pays on the corpus",
            "zero certified-law fires on every real term",
            "the handler failed to verify",
          ],
          ok: "zero certified-law fires on every real term",
          why: "Truth and pay are separate measurements, and disjoint here — the minted atom computes the total, the corpus emits the sequence. Minting works end to end; paying is a different question.",
          concept: "minting",
        },
        {
          q: "Across all six needs in the bank, how many pay on the corpus?",
          opts: ["0", "1", "5"],
          ok: "0",
          why: "The census counts the needs and the handler and laws verify — but the pay probe fires zero certified laws on every real term. Truth and pay are separate measurements, and disjoint here.",
          concept: "census",
          interactive: true,
        },
        {
          q: "Pick the `cumsum` need: how many certified law rows does it admit?",
          opts: ["0", "5", "15"],
          ok: "5",
          why: "Five certified laws (cum.add_same, cum.sub_same, cum.neg, cum.scale_same, cum.cat) — admitted through the gauntlet, and none fire on the corpus. Provenance records where each came from.",
          concept: "admission",
          interactive: true,
        },
      ),
    },
    {
      id: "truthpay",
      title: "True is not the same as paying",
      idea: "guard synthesis + adversarial sweep",
      objective:
        "A guard that separates its measured domain perfectly can be unsound just outside it — so sweep.",
      module: "discovery",
      why: "A minted atom can be true and still not pay — and its guard can be unsound just outside what you measured.",
      terms: ["guard", "adversarial-sweep"],
      outcome:
        "explain why a guard must be swept adversarially, and why a guard that is perfect on the measured domain can still be unsound.",
      repo: [
        {
          path: "packages/catopt-discovery/src/catopt_discovery/oracle.py",
          note: "the numeric oracle that decides truth",
        },
        {
          path: "project/adrs/0007-semantic-atom-minting.md",
          note: "admission, and the guard's role",
        },
        {
          path: "project/retros/mul-unsq-nat.md",
          note: "the shipped guard and the sibling it refused",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "The naive guard `grid-eq` separates its measured domain perfectly. Why is it not the shipped guard?",
          opts: [
            "it is slower",
            "the adversarial sweep found bad accepts past its measured domain",
            "the cost model rejects it",
          ],
          ok: "the adversarial sweep found bad accepts past its measured domain",
          why: "Perfect on the measured envs is not sound. The synthesized flat-map-unsq clause — grid coincidence plus both flat-read maps — has 0 bad accepts and 0 misses, and the apparent sibling was refused by the same guard.",
          concept: "guard",
        },
        {
          q: "With the naive `grid-eq` guard selected, how many bad accepts does the sweep show?",
          opts: ["none", "some", "exactly 1"],
          ok: "some",
          why: "The sweep enumerates shapes and finds counterexamples the measured domain never saw. Switch to the synthesized guard and the count drops to zero — that is why you sweep before shipping.",
          concept: "adversarial-sweep",
          interactive: true,
        },
      ),
    },
    {
      id: "exposure",
      title: "Where discovery can pay",
      idea: "exposure is bimodal",
      objective: "Exposure is bimodal: the e-graph can only rewrite math it can see.",
      module: "discovery",
      why: "Discovery can only rewrite math the IR can see — so measure what it can see.",
      terms: ["exposure", "bimodal"],
      outcome:
        "measure FLOPs-weighted exposure, explain its bimodal shape, and say why exposure is necessary but not sufficient.",
      repo: [
        {
          path: "packages/catopt-discovery/src/catopt_discovery/census.py",
          note: "the corpus census — where a need is counted",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/zoo.py",
          note: "the workload zoo the screen runs over",
        },
        {
          path: "project/retros/exposed-math.md",
          note: "the bimodal screen, measured",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "Of 321 exported workloads, 81 score exactly 0% FLOPs-weighted law coverage. What are they?",
          opts: [
            "broken exports",
            "the fused-op class — `nn.*` natives that export as one opaque node",
            "models with no matmuls",
          ],
          ok: "the fused-op class — `nn.*` natives that export as one opaque node",
          why: "The e-graph literally cannot see the computation: there is nothing to rewrite. Exposure is bimodal, and it is necessary but not sufficient — four fully-visible models pay nothing.",
          concept: "exposure",
        },
        {
          q: "How many of the 321 exported workloads score exactly 0% law coverage?",
          opts: ["8", "81", "140"],
          ok: "81",
          why: "The fused-op class. The other mode is the ≥95% spelled-math class (140 workloads). The e-graph cannot see the computation, so there is nothing to rewrite.",
          concept: "bimodal",
          interactive: true,
        },
      ),
    },
    {
      id: "substrate",
      title: "Sharing is a morphism",
      idea: "Δ sharing, `loop` recurrence",
      objective:
        "Re-present the IR as combinators: Δ makes sharing explicit, `loop` makes recurrence first-class.",
      module: "substrate",
      why: "The op-graph is the lowering surface — a second presentation re-reads it as typed arrows.",
      terms: ["morphism", "sharing", "loop", "monoidal-product"],
      outcome:
        "re-present a program as combinators — Δ makes sharing explicit, `loop` makes recurrence first-class, and the bridge round-trips.",
      repo: [
        {
          path: "packages/catopt-orchestrator/src/catopt_orchestrator/morphisms/",
          note: "the combinator presentation, as code",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/antiunify.py",
          note: "the definitional-atom bridge (layer 1)",
        },
        {
          path: "project/adrs/0008-combinator-presentation.md",
          note: "the bridge, and what it buys",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "In the combinator presentation, what becomes of a value the DAG shares between two parents?",
          opts: ["a pointer", "an explicit Δ morphism — `dup`", "a second copy"],
          ok: "an explicit Δ morphism — `dup`",
          why: "Sharing is a morphism, not a pointer convention, so the DAG-vs-tree ambiguity is impossible by construction. The bridge round-trips — `dag → comb → dag` is the identity — and a recurrence spells as `loop(step)`.",
          concept: "sharing",
        },
        {
          q: "Pick the `dup` term. How many explicit Δ sites does its combinator form have?",
          opts: ["0", "1", "2"],
          ok: "2",
          why: "Both `mul(x,x)` sites fan x out through Δ — sharing made explicit, so the DAG-vs-tree ambiguity is impossible by construction. The bridge round-trips.",
          concept: "morphism",
          interactive: true,
        },
        {
          q: "Which combinator runs two programs side by side?",
          opts: ["seq", "par (the monoidal product ⊗)", "loop"],
          ok: "par (the monoidal product ⊗)",
          why: "`seq` composes, `par` is the monoidal product ⊗, `dup` is Δ, and `loop` makes a recurrence a typed trace.",
          concept: "monoidal-product",
          interactive: true,
        },
      ),
    },
    {
      id: "ports",
      title: "A backend is a port",
      idea: "swap the sink, core never changes",
      objective: "Swap the backend and nothing in core changes — only what extraction may select.",
      module: "substrate",
      why: "The presentation is one substrate; a backend is a port — and swapping it must not touch core.",
      terms: ["port", "supported-ops"],
      outcome:
        "implement a backend as a Source/Sink port pair, and explain why swapping it changes only supported_ops, never core.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/ports.py",
          note: "the Source / Sink protocols",
        },
        {
          path: "packages/catopt-onnx/src/catopt_onnx/sink.py",
          note: "the ONNX sink that landed with zero core edits",
        },
        {
          path: "packages/catopt-torch/src/catopt_torch/backend.py",
          note: "the torch Backend bundle",
        },
        {
          path: "project/adrs/0006-onnx-backend.md",
          note: "a second backend as a port pair",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "You swap the torch sink for the ONNX sink. What may change in core?",
          opts: ["the IR", "the law set", "nothing — only which forms the extraction may select"],
          ok: "nothing — only which forms the extraction may select",
          why: "A backend is a port: the core-mutation counter stays at zero across the swap. What changes is `supported_ops`, which re-bounds extraction — the ONNX sink can't lower `sqq`, so that member drops out.",
          concept: "port",
        },
        {
          q: "Swap to the ONNX sink. Which op does it fail to lower?",
          opts: ["matmul", "sqq", "add"],
          ok: "sqq",
          why: "The ONNX sink's `supported_ops` lacks `sqq`, so the fused member drops out and the lowering is refused — while the core-mutation counter stays at zero. A backend is a port.",
          concept: "supported-ops",
          interactive: true,
        },
      ),
    },
    {
      id: "cellstore",
      title: "One record per idea",
      idea: "a cell, merged from fragments",
      objective:
        "Every vocabulary item — op, law, object, handle — is one cell, assembled from fragments.",
      module: "substrate",
      why: "The project spelled 'a cell as data' five ways; the cell store makes it one record, merged from fragments.",
      terms: ["cell", "fragment-merge", "provenance-origin", "handl-bridge"],
      outcome:
        "assemble a store from fragments, read a cell's provenance origin, and emit the store as a handl fragment.",
      repo: [
        {
          path: "packages/catopt-core/src/catopt_core/cells.py",
          note: "`Cell` / `merge_fragments` / `CellStore`",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/cellstore.py",
          note: "the ambient store and the machine fragment",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/handl.py",
          note: "the handl schema bridge",
        },
        {
          path: "project/plans/0022-cell-store.md",
          note: "one record, fragment-merged",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "How many records does a vocabulary item become in the store?",
          opts: ["one cell", "one per backend", "one per law"],
          ok: "one cell",
          why: "Laws, constructed objects, `OpDef`s and handler-table dicts all normalise to one `Cell{role, name, body, kind, provenance}` — the Sanada operations-and-handlers mapping made literal.",
          concept: "cell",
        },
        {
          q: "In the widget, add the rival fragment. What does the merge do?",
          opts: [
            "keeps the last",
            "raises — a duplicate `(role, name)` is a conflict",
            "silently overrides",
          ],
          ok: "raises — a duplicate `(role, name)` is a conflict",
          why: "Silent override is how vocabularies drift. `merge_fragments` raises by default; alpha-equal 2-cells across fragments are reported as a discovery fact, not a schema error.",
          concept: "fragment-merge",
          interactive: true,
        },
        {
          q: "Which origin is *not* one of the store's provenance origins?",
          opts: ["shipped", "minted", "guessed"],
          ok: "guessed",
          why: "The origins are `shipped` / `constructed` / `admitted` / `minted` / `declared` / `imported` — where a cell came from, not what it can do. 'Guessed' is not one.",
          concept: "provenance-origin",
        },
        {
          q: "A law cell becomes what in a handl fragment?",
          opts: ["an `AstShapeDef`", "a `TransformDef`", "a `!Native`"],
          ok: "a `TransformDef`",
          why: "The handl bridge maps op cells → `AstShapeDef`, law/object cells → `TransformDef`, handle cells → `!Native`. Every row carries a `catopt:` extension so the round-trip is lossless.",
          concept: "handl-bridge",
        },
      ),
    },
    {
      id: "game",
      title: "The game layer",
      idea: "boards, players, learned priors",
      objective:
        "The pipeline is also a game: boards, players, and learned priors over certified moves.",
      module: "substrate",
      why: "The same machinery is played as a game — a board, a player, and a prior that orders legal moves.",
      terms: ["board", "meta-arena", "play-domain"],
      outcome:
        "describe the game layer — a `Board` protocol, players, the meta-arena, and `play --domain` — and what a learned prior measurably buys.",
      repo: [
        {
          path: "packages/catopt-discovery/src/catopt_discovery/engine.py",
          note: "the `Board` protocol + episode/train/eval drivers",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/players.py",
          note: "`LinearPolicy`, the learned arm",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/meta_arena.py",
          note: "the mid-search construction board",
        },
        {
          path: "packages/catopt-discovery/src/catopt_discovery/play.py",
          note: "the domain registry + CLI",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "A `Board` is which three things?",
          opts: [
            "a program, a legal move set, a referee",
            "a cost model and a law",
            "a graph and a sink",
          ],
          ok: "a program, a legal move set, a referee",
          why: "`catopt_discovery.engine` factors the shared shape out: every domain is 'a program + a legal move set + a referee', and the reward is certified improvement.",
          concept: "board",
        },
        {
          q: "In the widget, run the learned prior. What does it buy over the cost-delta heuristic on held-out pages?",
          opts: ["nothing", "AP 0.133 vs 0.045", "a new law set"],
          ok: "AP 0.133 vs 0.045",
          why: "The baked learned prior's ordering beats the cost-delta heuristic on held-out pages (ADR 0005). It orders moves; it never changes what is legal.",
          concept: "play-domain",
          interactive: true,
        },
        {
          q: "Which meta-arena move mints a new abstraction mid-search?",
          opts: ["`fire`", "`declare`", "`extract`"],
          ok: "`declare`",
          why: "`fire` applies a law, `saturate` takes a bounded closure step, `declare` inserts a construction (fold/lift/compose) mid-search, and `extract` is terminal — extract, replay the certificate, score the delta.",
          concept: "meta-arena",
        },
      ),
    },
    {
      id: "gates",
      title: "How the project stays honest",
      idea: "coverage · ty · ratchets · bench gate",
      objective:
        "The project keeps itself honest with ratchets, gates, and a no-new-suppressions rule.",
      module: "engineering",
      why: "Every claim so far is only as good as the gates that keep the tree honest.",
      terms: ["coverage-floor", "ratchet", "no-suppressions", "bench-gate"],
      outcome:
        "name the gates the project enforces — coverage floors, the type/lint/security checks, the radon ratchet, the bench gate — and explain the no-new-suppressions rule.",
      repo: [
        {
          path: "AGENTS.md",
          note: "the verification commands every change must pass",
        },
        {
          path: "scripts/radon_ratchet.py",
          note: "the complexity ceiling, pinned per function",
        },
        {
          path: "bench/README.md",
          note: "the bench gate and pinned baselines",
        },
        {
          path: "project/retros/onnx-package.md",
          note: "a 75% → 100% coverage climb, zero suppressions",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "A ratchet is a ceiling that may move which way?",
          opts: ["up", "one way — never backwards", "either way"],
          ok: "one way — never backwards",
          why: "The radon ratchet pins every function's complexity; the coverage floor is pinned at 100% (shipped) / 99% (discovery). Both may only tighten — never loosen.",
          concept: "ratchet",
        },
        {
          q: "In the widget, break the coverage gate. What does the run report?",
          opts: ["a warning", "a failure — the gate is a floor, not a target", "nothing"],
          ok: "a failure — the gate is a floor, not a target",
          why: "A floor fails the run, it does not warn. The project ships at 100% statement+branch coverage on the six shipped packages; discovery holds a 99% ratchet that may only tighten.",
          concept: "coverage-floor",
          interactive: true,
        },
        {
          q: "A lint finding appears in new code. What is the rule?",
          opts: ["add a `# noqa`", "fix it — no new suppressions", "delete the file"],
          ok: "fix it — no new suppressions",
          why: "No new suppressions: no `# noqa`, no `# ty: ignore`, no baseline entry. The `ty` ratchet's exclude list is empty and stays empty — annotate, never opt out.",
          concept: "no-suppressions",
        },
        {
          q: "A suite regresses against its pinned baseline. What happens?",
          opts: ["the run fails", "it is ignored", "the baseline updates automatically"],
          ok: "the run fails",
          why: "The bench gate compares every run against the committed baseline; a regression fails. Regenerating a baseline is a reviewed, intentional act — never automatic.",
          concept: "bench-gate",
        },
      ),
    },
    {
      id: "honest",
      title: "The ledger of negatives",
      idea: "methodology as a result",
      objective: "Methodology as a result: the measured refusals are taught, not buried.",
      module: "engineering",
      why: "Every claim so far is bounded by what was measured — and the most instructive results are the refusals.",
      terms: ["negative-result", "falsification"],
      outcome:
        "read a negative result as a result, and state the falsification criteria that let a claim be killed by measurement.",
      repo: [
        {
          path: "project/adrs/0001-weight-space-structure-falsified.md",
          note: "the first measured refusal, recorded as an ADR",
        },
        {
          path: "project/retros/atom-synth-loop.md",
          note: "the atom pay-null, taught as a result",
        },
        {
          path: "project/retros/birefnet-bounded-lane.md",
          note: "the bounded negative — certified, then refused",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "The composed atom loop minted a verified atom and admitted four law sets — and fired nothing on the corpus. What does that refusal teach?",
          opts: [
            "the minting machinery is broken",
            "truth and pay are separate measurements — the negative is a result",
            "the corpus is wrong",
          ],
          ok: "truth and pay are separate measurements — the negative is a result",
          why: "Every claim in this course is bounded by what was measured. The ledger of negatives — the falsified weight hypothesis, the pay-null, the BiRefNet bounded negative, the antiunify reach-null — is the methodology, taught as a result rather than a footnote.",
          concept: "negative-result",
        },
        {
          q: "How many measured refusals does the widget put to you?",
          opts: ["3", "5", "6"],
          ok: "5",
          why: "Five measured negatives — the weight-space closure, the atom pay-null, the BiRefNet bounded negative, the antiunify reach-null, and enumeration-beats-learning. Each is a result, not a footnote.",
          concept: "falsification",
          interactive: true,
        },
      ),
    },
    {
      id: "bounded",
      title: "Approximation with a certificate",
      idea: "certificate AND an economics test",
      objective:
        "Approximation is allowed — but only with a certified bound and an economics test.",
      module: "engineering",
      why: "Exact rewriting is the default — but a near-equal program is sometimes allowed, if it carries a bound.",
      terms: ["bounded-approximation", "economics-test"],
      outcome:
        "explain the bounded lane as two gates — a certificate and an economics test — and why passing the first is not enough.",
      repo: [
        {
          path: "packages/catopt-orchestrator/src/catopt_orchestrator/criteria.py",
          note: "the acceptance criteria the lane must clear",
        },
        {
          path: "project/retros/birefnet-bounded-lane.md",
          note: "certified 20/56, delivered 0 — measured",
        },
        {
          path: "project/adrs/0003-evaluation-is-an-independent-dimension.md",
          note: "why pricing is a separate gate",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "56 bounded offers carry certified bounds inside a 3e-2 budget, the bound gate refuses nothing — and nothing is delivered. Why?",
          opts: [
            "the bound gate is broken",
            "the bounded member priced 1.32× the dense member, so the pricing refused it",
            "the budget was too small",
          ],
          ok: "the bounded member priced 1.32× the dense member, so the pricing refused it",
          why: "Approximation must clear two gates: a certificate (is it within ε?) and an economics test (is it cheaper?). BiRefNet's near-duplicate rows are real but thin — 0.78% of the qkv weight — so the gather/re-expand/bias-add wrapper cost 1.32× the dense member. The bound gate stayed bitwise-clean (fresh_rel = 0.0); the cost model refused, correctly.",
          concept: "economics-test",
        },
        {
          q: "At ε = 3e-2, how many offers does the lane certify, and how many does it deliver?",
          opts: [
            "56 certified · 0 delivered",
            "20 certified · 20 delivered",
            "0 certified · 0 delivered",
          ],
          ok: "56 certified · 0 delivered",
          why: "The bound gate is bitwise-clean (fresh_rel = 0.0) and certifies every offer; the economics test refuses all of them, because the member priced 1.32× dense. Approximation must clear both gates.",
          concept: "bounded-approximation",
          interactive: true,
        },
      ),
    },
    {
      id: "deliver",
      title: "The program must run",
      idea: "same term, different runtime",
      objective:
        "The same term, delivered differently, runs at wildly different speeds — delivery is its own dimension.",
      module: "engineering",
      why: "Search picks a term; something still has to run it — and how it runs is its own dimension.",
      terms: ["delivery", "dispatch", "carrier", "minted-kernel"],
      outcome:
        "explain delivery as an execution dimension — eager, carrier, fused, graph, or minted kernel — and why the same term runs at different speeds.",
      repo: [
        {
          path: "packages/catopt-torch/src/catopt_torch/executors/",
          note: "the executor tiers a term can run through",
        },
        {
          path: "packages/catopt-cuda/src/catopt_cuda/runners.py",
          note: "the CUDA-graph runner",
        },
        {
          path: "packages/catopt-carriers/src/catopt_carriers/machinery.py",
          note: "the carrier seam — whole-subgraph lowerers",
        },
        {
          path: "project/retros/gen-kernels.md",
          note: "the minted fusion kernel",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "`chainfuse` — `x@A@B@C` plus a 6-op pointwise tail — takes 315 µs eager and 253 µs under `torch.compile`. What delivered it at 86 µs (2.94× vs compile)?",
          opts: [
            "a bigger search budget",
            "the laws plus a minted fusion kernel (the Triton tier)",
            "a faster GPU",
          ],
          ok: "the laws plus a minted fusion kernel (the Triton tier)",
          why: "The reassociation laws fold the weight chain to one GEMM — 181 µs, already 1.34× vs compile — but the fused tail was unreachable: no shipped law names it. The search minted the kernel, claimed it, and the delivered module ran it: 86 µs, verified 1.7e-05. Delivery is the execution dimension; semantics is unchanged.",
          concept: "delivery",
        },
        {
          q: "For the `chainfuse` term, which delivery does the live model pick — and does the clock agree?",
          opts: [
            "the minted kernel, and the clock agrees",
            "eager, and the clock agrees",
            "the CUDA graph, and the clock disagrees",
          ],
          ok: "the minted kernel, and the clock agrees",
          why: "The dispatch-aware model (work + launches) picks the minted fusion kernel and the measured column agrees — 86 µs, 2.94× vs torch.compile. Delivery removes the per-op dispatch overhead.",
          concept: "dispatch",
          interactive: true,
        },
      ),
    },
    {
      id: "pricing",
      title: "The cost model can be wrong",
      idea: "the model ranks — and can be wrong",
      objective: "The cost model steers selection — and it can be confidently wrong.",
      module: "engineering",
      why: "Delivery needs a price, and a price can be wrong — confidently.",
      terms: ["inversion"],
      outcome:
        "explain that a cost model is a ranking that can be wrong, and that measured feedback (calibration) is the fix.",
      repo: [
        {
          path: "packages/catopt-torch/src/catopt_torch/calibrate.py",
          note: "the measured corrections fed back into the model",
        },
        {
          path: "packages/catopt-core/src/catopt_core/cost/executor.py",
          note: "the executor-route pricing",
        },
        {
          path: "project/retros/executor-pricing-fix.md",
          note: "the inversion, and its 12/12 fix",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "On the SSM cases the uncalibrated model picks the batched carrier route — which then runs 1.9–2.7× slower than the generic route it skipped. What fixed the pick?",
          opts: [
            "a better search policy",
            "measured corrections fed back into the cost model (calibration)",
            "nothing — it was noise",
          ],
          ok: "measured corrections fed back into the cost model (calibration)",
          why: "The routed model priced the carrier 20–30% cheaper; the clock disagreed. Wiring the measured profile into the delivered-price comparison — a pooled per-family correction — landed the measured-fastest member on 12/12 probe cases. A cost model is a ranking, not the truth: it must be measured, and it may be wrong (ADR 0003).",
          concept: "inversion",
        },
        {
          q: "With calibration OFF on the default case, what does 'probe cases correct' show?",
          opts: ["0/3", "1/3", "3/3"],
          ok: "1/3",
          why: "The routed model prices the batched carrier 20–30% cheaper and measures it 1.9–2.7× slower — the ranking is inverted. Feeding measured corrections back lands the measured-fastest member on 12/12 probe cases.",
          concept: "inversion",
          interactive: true,
        },
      ),
    },
    {
      id: "improve",
      title: "The payoff: where to improve",
      idea: "idea → blocker → next measurement",
      objective:
        "Map each idea to the project's open work: the blocker, the next measurement, and the level that teaches it.",
      module: "engineering",
      why: "The course is not the point — improving the project is.",
      terms: ["gap", "blocker", "next-measurement"],
      outcome:
        "map an idea to the project's open work — the blocker, the next measurement, and the level that teaches the idea — and read a closed family as a result.",
      repo: [
        {
          path: "project/plans/0025-minting-and-substrate.md",
          note: "the project's open work, as the course's gaps",
        },
        {
          path: "project/retros/atom-synth-loop.md",
          note: "the enumeration-ordering blocker, measured",
        },
      ],
      checkpoint: checkpoint(
        {
          q: "A gap's next step is a *measurement*, not a build. Which is the project's stated next step for the enumeration-ordering blocker?",
          opts: [
            "rewrite the proposer from scratch",
            "re-measure the SelectiveScan funnel after a canonical emitted-scan seed",
            "delete the fold atom",
          ],
          ok: "re-measure the SelectiveScan funnel after a canonical emitted-scan seed",
          why: "The paying `fold_emit` verifies at 6.7e-16 when reached directly — so the gap is ordering, not expressiveness, and the next step is a measurement: does the canonical seed make the minted atom's law set fire on a real corpus term? If not, the pay-null is semantic and the atom is done. Idea → blocker → next measurement is the whole course, applied.",
          concept: "gap",
        },
        {
          q: "Pick the precision gap in the widget. Which level does its connection point to?",
          opts: ["15 mint", "24 bounded", "26 pricing"],
          ok: "24 bounded",
          why: "Idea → blocker → next measurement: the precision gap's blocker is BiRefNet's thin dedup, its next step is a wide-MLP checkpoint, and it is taught by level 24 (approximation with a certificate). Every gap maps to a level.",
          concept: "next-measurement",
          interactive: true,
        },
      ),
    },
  ];

  i = 0;
  answers: Record<string, string> = {};
  /** Level ids whose live widget was engaged — what earns the XP bonus. */
  engaged: string[] = [];
  /** Module ids whose review is passed. */
  reviewed: string[] = [];
  /** Module review answers, keyed moduleId → questionId → option. */
  reviewAnswers: Record<string, Record<string, string>> = {};
  settings = false;
  confirming = false;
  masteryPanel = false;
  badgesPanel = false;
  glossaryPanel = false;
  reviewPanel = false;
  restored = false;

  /** Alpine calls this on mount: restore any saved progress. */
  init(): void {
    const s = store();
    if (!s) return;
    try {
      const raw = s.getItem(STORAGE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw) as Partial<Saved>;
      if (typeof data.i === "number") {
        this.i = Math.max(0, Math.min(this.n - 1, Math.trunc(data.i)));
      }
      if (data.answers && typeof data.answers === "object") {
        this.answers = { ...data.answers };
      }
      if (Array.isArray(data.engaged))
        this.engaged = data.engaged.filter((x) => typeof x === "string");
      if (Array.isArray(data.reviewed)) {
        this.reviewed = data.reviewed.filter((x) => typeof x === "string");
      }
      if (data.reviews && typeof data.reviews === "object") {
        const out: Record<string, Record<string, string>> = {};
        for (const [mid, ans] of Object.entries(data.reviews)) {
          if (ans && typeof ans === "object") out[mid] = { ...ans };
        }
        this.reviewAnswers = out;
      }
      this.restored = true;
    } catch {
      // corrupt or unreadable storage: start clean rather than crash
      s.removeItem(STORAGE_KEY);
    }
  }

  private save(): void {
    const s = store();
    if (!s) return;
    try {
      s.setItem(
        STORAGE_KEY,
        JSON.stringify({
          i: this.i,
          answers: this.answers,
          engaged: this.engaged,
          reviewed: this.reviewed,
          reviews: this.reviewAnswers,
        }),
      );
    } catch {
      // quota or private mode: progress simply does not persist
    }
  }

  get level(): Level {
    return this.levels[this.i];
  }

  get n(): number {
    return this.levels.length;
  }

  get done(): number {
    return this.levels.filter((l) => this.passed(l.id)).length;
  }

  get complete(): boolean {
    return this.done === this.n;
  }

  /** Cleared fraction, for the shell's progress meter. */
  get pct(): number {
    return this.n === 0 ? 0 : Math.round((100 * this.done) / this.n);
  }

  /** The ladder's one-liners in order — the completion recap. */
  get recap(): { id: string; title: string; idea: string; passed: boolean }[] {
    return this.levels.map((l) => ({
      id: l.id,
      title: l.title,
      idea: l.idea,
      passed: this.passed(l.id),
    }));
  }

  get savedLabel(): string {
    return this.restored ? "restored from this browser" : "saved as you go";
  }

  // ── the template (plan 0028 §B) ────────────────────────────────

  /** The arc (module) the current level belongs to. */
  get module(): Module | undefined {
    return MODULES.find((m) => m.id === this.level.module);
  }

  /** The current level's new terms, resolved from the concept graph. */
  get levelTerms(): ConceptRow[] {
    const m = this.mastery;
    return this.level.terms.flatMap((id) => {
      const c = conceptById(id);
      if (!c) return [];
      return [
        {
          id: c.id,
          label: c.label,
          def: c.def,
          tier: tier(m[c.id]),
          level: this.level.id,
          levelTitle: this.level.title,
        },
      ];
    });
  }

  /** The levels the current level depends on — its prerequisite ladder. */
  get prerequisites(): { id: string; title: string; index: number }[] {
    return conceptPrerequisites(this.level.id).map((id) => ({
      id,
      title: this.levels.find((l) => l.id === id)?.title ?? id,
      index: levelIndex(id, LEVEL_ORDER),
    }));
  }

  // ── the gamification layer (plan 0028 §C) ──────────────────────

  /** The progress value the points, mastery and badge models read. */
  get progress(): { passed: string[]; engaged: string[]; reviewed: string[] } {
    return {
      passed: this.levels.filter((l) => this.passed(l.id)).map((l) => l.id),
      engaged: [...this.engaged],
      reviewed: [...this.reviewed],
    };
  }

  /** Total XP — the level curve plus the widget bonuses plus the reviews. */
  get xp(): number {
    return totalXp(this.progress, LEVEL_ORDER);
  }

  /** The named rank the XP total has reached. */
  get rank(): Rank {
    return rank(this.xp);
  }

  /** The next rank up, or `null` at the top. */
  get nextRank(): Rank | null {
    return nextRank(this.xp);
  }

  /** Progress toward the next rank, as a percentage. */
  get rankPct(): number {
    return Math.round(100 * rankProgress(this.xp));
  }

  /** The current level's XP, broken into base and widget bonus. */
  get levelXp(): {
    base: number;
    bonus: number;
    total: number;
    passed: boolean;
    engaged: boolean;
  } {
    return forLevel(this.level.id, this.progress, LEVEL_ORDER);
  }

  /** Whether the current level's live widget has been engaged. */
  get engagedHere(): boolean {
    return this.engaged.includes(this.level.id);
  }

  /** The XP the current level is worth in full — base plus widget bonus. */
  get levelPotentialXp(): number {
    return levelBase(levelIndex(this.level.id, LEVEL_ORDER)) + WIDGET_BONUS;
  }

  /** Per-concept mastery, folded from every checkpoint *and* review answer.
   *
   * Plan 0028 §C: spaced retrieval — a concept's tally accumulates
   * correct answers from the level checkpoints and the module reviews
   * alike.  Only *answered* questions count (an untouched question is
   * not a wrong one).
   */
  get mastery(): MasteryState {
    let s = emptyMastery();
    for (const l of this.levels) {
      l.checkpoint.parts.forEach((part, i) => {
        const picked = this.answers[partKey(l.id, i)];
        if (picked === undefined) return;
        s = record(s, part.concept, picked === part.ok);
      });
    }
    for (const m of MODULES) {
      const answers = this.reviewAnswers[m.id];
      if (!answers) continue;
      for (const q of m.questions) {
        if (answers[q.id] === undefined) continue;
        s = record(s, q.concept, answers[q.id] === q.ok);
      }
    }
    return s;
  }

  /** Tier counts and the overall ratio across the concept graph. */
  get masterySummary(): MasterySummary {
    return summary(this.mastery, CONCEPTS);
  }

  /** How many concepts are mastered. */
  get conceptsMastered(): number {
    return masteredCount(this.mastery, CONCEPTS);
  }

  /** The concepts to revisit — the module-review "send you back" data. */
  get weakestConcepts(): WeakConcept[] {
    return weakest(this.mastery, CONCEPTS, 5);
  }

  /** The concept graph, grouped by the module whose levels teach it. */
  get masteryByModule(): ModuleMastery[] {
    const m = this.mastery;
    return MODULES.map((module) => ({
      module,
      concepts: module.levels.flatMap((lvl) =>
        conceptsTaughtAt(lvl).map((c) => ({
          id: c.id,
          label: c.label,
          def: c.def,
          tier: tier(m[c.id]),
          level: lvl,
          levelTitle: this.levels.find((l) => l.id === lvl)?.title ?? lvl,
        })),
      ),
    }));
  }

  /** The concepts the current level teaches — its new terms. */
  get levelConcepts(): ConceptRow[] {
    const m = this.mastery;
    return conceptsTaughtAt(this.level.id).map((c) => ({
      id: c.id,
      label: c.label,
      def: c.def,
      tier: tier(m[c.id]),
      level: this.level.id,
      levelTitle: this.level.title,
    }));
  }

  /** Every mastered concept, for the completion recap. */
  get masteredConcepts(): { id: string; label: string; def: string }[] {
    const m = this.mastery;
    return CONCEPTS.filter((c) => tier(m[c.id]) === "mastered").map((c) => ({
      id: c.id,
      label: c.label,
      def: c.def,
    }));
  }

  /** The mastery state a badge predicate reads. */
  get badgeState(): BadgeState {
    return {
      order: LEVEL_ORDER,
      passed: this.progress.passed,
      engaged: this.progress.engaged,
      reviewed: this.progress.reviewed,
      mastery: this.mastery,
      concepts: CONCEPTS,
      modules: MODULES,
    };
  }

  /** The badges earned so far. */
  get badgesEarned(): Badge[] {
    return earned(this.badgeState);
  }

  /** The badges still locked. */
  get badgesLocked(): Badge[] {
    return locked(this.badgeState);
  }

  /** The glossary: every concept, with its definition and home level. */
  get glossary(): (GlossaryRow & { title: string })[] {
    return conceptGlossary().map((row) => ({
      ...row,
      title: this.levels.find((l) => l.id === row.introduced)?.title ?? row.introduced,
    }));
  }

  /** Concepts used before they are introduced — the step-2 ordering gate. */
  get forwardRefs(): ForwardRef[] {
    return conceptForwardRefs();
  }

  /** The concepts a level introduces — the graph's view of its new terms. */
  graphTerms(levelId: string): Concept[] {
    return conceptsIntroducedAt(levelId);
  }

  // ── module reviews (plan 0028 §A/§C) ───────────────────────────

  /** The module the current level belongs to (never undefined). */
  get currentModule(): Module {
    return MODULES.find((m) => m.id === this.level.module) ?? MODULES[0];
  }

  /** Every module's review state — levels cleared, reviewed, cleared. */
  get moduleStates(): ModuleReviewState[] {
    const passed = this.progress.passed;
    return MODULES.map((module) => {
      const levelsCleared = moduleComplete(module, passed);
      const round = this.reviewFor(module.id);
      return {
        module,
        levelsCleared,
        reviewed: this.reviewed.includes(module.id),
        cleared: moduleCleared(module, passed, this.reviewed),
        unlocked: levelsCleared,
        answered: round.answered,
        total: round.total,
        passed: round.passed,
        weak: round.weak,
      };
    });
  }

  /** The current level's module review state — the review card reads this. */
  get currentModuleState(): ModuleReviewState {
    return (
      this.moduleStates.find((s) => s.module.id === this.currentModule.id) ?? this.moduleStates[0]
    );
  }

  /** The review round for a module, with the learner's answers applied. */
  reviewFor(moduleId: string): ReviewRound {
    const m = MODULES.find((x) => x.id === moduleId) ?? MODULES[0];
    const r = new ReviewRound(m);
    const answers = this.reviewAnswers[moduleId] ?? {};
    for (const [qid, opt] of Object.entries(answers)) r.answer(qid, opt);
    return r;
  }

  /** Answer one review question (moduleId, questionId, option). */
  answerReview(moduleId: string, qid: string, opt: string): void {
    const cur = this.reviewAnswers[moduleId] ?? {};
    this.reviewAnswers = { ...this.reviewAnswers, [moduleId]: { ...cur, [qid]: opt } };
    this.save();
  }

  /** The option picked for a review question. */
  reviewPicked(moduleId: string, qid: string): string | undefined {
    return this.reviewAnswers[moduleId]?.[qid];
  }

  /** Whether a review question's picked answer is right. */
  reviewPassed(moduleId: string, qid: string): boolean {
    const m = MODULES.find((x) => x.id === moduleId);
    const q = m?.questions.find((x) => x.id === qid);
    return q !== undefined && this.reviewAnswers[moduleId]?.[qid] === q.ok;
  }

  /** Submit a module review; a pass records it.  Returns the verdict. */
  submitReview(moduleId: string): ReviewResult {
    const result = this.reviewFor(moduleId).result;
    if (result.passed) this.markReviewed(moduleId);
    return result;
  }

  /** Reset a module's review — clear the answers and the pass, and retry. */
  resetReview(moduleId: string): void {
    const next = { ...this.reviewAnswers };
    delete next[moduleId];
    this.reviewAnswers = next;
    this.reviewed = this.reviewed.filter((x) => x !== moduleId);
    this.save();
  }

  /** How many module reviews are passed. */
  get reviewsPassed(): number {
    return this.reviewed.length;
  }

  /** The concepts a review sends you back to, with the levels that teach them. */
  reviewWeak(moduleId: string): WeakConcept[] {
    const weak = new Set(this.reviewFor(moduleId).weak);
    return weakest(this.mastery, CONCEPTS, CONCEPTS.length).filter((c) => weak.has(c.id));
  }

  /** Every level passed *and* every review passed — the full-comprehension state. */
  get fullyComplete(): boolean {
    return this.complete && this.reviewed.length === MODULES.length;
  }

  // ── the completion state (plan 0028 §C) ────────────────────────

  /** The final state: XP, rank, levels, mastery, badges, and the gaps to attack. */
  get completion(): CompletionState {
    const s = this.masterySummary;
    return {
      xp: this.xp,
      rank: this.rank,
      nextRank: this.nextRank,
      levelsCleared: this.done,
      levelsTotal: this.n,
      reviewsPassed: this.reviewsPassed,
      reviewsTotal: MODULES.length,
      weak: s.weak,
      familiar: s.familiar,
      mastered: s.mastered,
      conceptsTotal: s.total,
      ratio: s.ratio,
      badgesEarned: this.badgesEarned.length,
      badgesTotal: this.badgesEarned.length + this.badgesLocked.length,
      gaps: [...GAPS],
    };
  }

  // ── gating (plan 0028 §B step 6) ───────────────────────────────

  /** Whether every part of a level's checkpoint is answered correctly. */
  passed(id: string): boolean {
    const lvl = this.levels.find((l) => l.id === id);
    if (!lvl) return false;
    return lvl.checkpoint.parts.every((p, i) => this.answers[partKey(id, i)] === p.ok);
  }

  /** Whether any part of a level's checkpoint has been answered. */
  answered(id: string): boolean {
    const lvl = this.levels.find((l) => l.id === id);
    if (!lvl) return false;
    return lvl.checkpoint.parts.some((_p, i) => this.answers[partKey(id, i)] !== undefined);
  }

  /** How many parts of a level's checkpoint are answered correctly. */
  partsPassed(id: string): number {
    const lvl = this.levels.find((l) => l.id === id);
    if (!lvl) return 0;
    return lvl.checkpoint.parts.filter((p, i) => this.answers[partKey(id, i)] === p.ok).length;
  }

  /** The current level's checkpoint progress, for the shell. */
  get checkpointProgress(): { correct: number; total: number; done: boolean } {
    const total = this.level.checkpoint.parts.length;
    const correct = this.partsPassed(this.level.id);
    return { correct, total, done: correct === total };
  }

  /** The answer picked for a checkpoint part of the current level. */
  partPicked(i: number): string | undefined {
    return this.answers[partKey(this.level.id, i)];
  }

  /** Whether a checkpoint part of the current level is correct. */
  partPassed(i: number): boolean {
    return this.answers[partKey(this.level.id, i)] === this.level.checkpoint.parts[i].ok;
  }

  /** Whether a checkpoint part of the current level has been answered. */
  partAnswered(i: number): boolean {
    return this.answers[partKey(this.level.id, i)] !== undefined;
  }

  /** Answer checkpoint part `i` of the current level. */
  answerAt(i: number, opt: string): void {
    this.answers = { ...this.answers, [partKey(this.level.id, i)]: opt };
    this.save();
  }

  /** Answer the primary (first) checkpoint part — the legacy single call. */
  answer(opt: string): void {
    this.answerAt(0, opt);
  }

  /** Mark a level's live widget as engaged — it earns the XP bonus. */
  engage(id: string = this.level.id): void {
    if (this.engaged.includes(id)) return;
    this.engaged = [...this.engaged, id];
    this.save();
  }

  /** Record a passed module review (step 3 supplies the questions). */
  markReviewed(id: string): void {
    if (this.reviewed.includes(id)) return;
    this.reviewed = [...this.reviewed, id];
    this.save();
  }

  get picked(): string | undefined {
    return this.answers[partKey(this.level.id, 0)];
  }

  get canAdvance(): boolean {
    return this.passed(this.level.id);
  }

  next(): void {
    if (this.canAdvance && this.i < this.n - 1) {
      this.i += 1;
      this.save();
    }
  }

  prev(): void {
    if (this.i > 0) {
      this.i -= 1;
      this.save();
    }
  }

  goto(i: number): void {
    this.i = Math.max(0, Math.min(this.n - 1, i));
    this.save();
  }

  /** Jump to a level by id — the glossary's and prerequisite links. */
  gotoLevel(id: string): void {
    const i = this.levels.findIndex((l) => l.id === id);
    if (i >= 0) this.goto(i);
  }

  /** Clear answers and start over — keeps you on the course. */
  restart(): void {
    this.i = 0;
    this.answers = {};
    this.engaged = [];
    this.reviewed = [];
    this.reviewAnswers = {};
    this.restored = false;
    this.save();
  }

  /** Settings: wipe stored progress entirely. */
  resetProgress(): void {
    const s = store();
    try {
      s?.removeItem(STORAGE_KEY);
    } catch {
      // nothing stored, nothing to clear
    }
    this.i = 0;
    this.answers = {};
    this.engaged = [];
    this.reviewed = [];
    this.reviewAnswers = {};
    this.restored = false;
    this.confirming = false;
    this.settings = false;
  }

  toggleSettings(): void {
    this.settings = !this.settings;
    this.confirming = false;
    this.masteryPanel = false;
    this.badgesPanel = false;
    this.glossaryPanel = false;
    this.reviewPanel = false;
  }

  toggleMastery(): void {
    this.masteryPanel = !this.masteryPanel;
    this.settings = false;
    this.badgesPanel = false;
    this.glossaryPanel = false;
    this.reviewPanel = false;
  }

  toggleBadges(): void {
    this.badgesPanel = !this.badgesPanel;
    this.settings = false;
    this.masteryPanel = false;
    this.glossaryPanel = false;
    this.reviewPanel = false;
  }

  toggleGlossary(): void {
    this.glossaryPanel = !this.glossaryPanel;
    this.settings = false;
    this.masteryPanel = false;
    this.badgesPanel = false;
    this.reviewPanel = false;
  }

  toggleReview(): void {
    this.reviewPanel = !this.reviewPanel;
    this.settings = false;
    this.masteryPanel = false;
    this.badgesPanel = false;
    this.glossaryPanel = false;
  }
}
