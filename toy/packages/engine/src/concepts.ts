/** The concept graph — every idea the course teaches, as data.
 *
 * Plan 0028 §A: a learner cannot see what depends on what while the
 * levels are a flat list.  This module is the prerequisite structure:
 * for every concept, the level that *introduces* it and the levels
 * that *use* it, in curriculum order.
 *
 * Step 1 seeded the graph honestly from the 22 levels' text and
 * recorded the ordering bugs (22 forward references — terms used
 * before they are defined).  Step 2 (this revision) **fixes** them:
 * a term is now either introduced at or before its first use, or the
 * incidental prose mention was removed.  {@link forwardRefs} therefore
 * returns **zero** and stays a live gate — the test asserts the count
 * can only go down.
 *
 * Two kinds of fix were applied, and the graph records which:
 *
 * - **Promoted** — the first-use level genuinely teaches the term, so
 *   its `introduced` moved up to meet its first use: `closure` and
 *   `e-graph` → reach (2), `monoid`/`comonoid` → structure (3),
 *   `non-dominated-set` → choice (5), `ranking` → measure (8),
 *   `features` → learn (7), `critical-pair` → coherence (11),
 *   `monoidal-product` → substrate (15).
 * - **Removed** — the early mention was incidental foreshadowing, so
 *   the `used` list dropped it (and the prose was reworded): `referee`,
 *   `oracle`, `cost-model`, `minting`, `admission`, `census`,
 *   `provenance`, `guard`, `adversarial-sweep` (level 3's callout),
 *   `learned-policy`, `legal-move` (level 4), `dispatch` (level 8),
 *   `carrier` (level 15).
 *
 * The labels stay in data so the graph — and the glossary it drives —
 * can be rebuilt without re-reading the prose.
 */

export interface Concept {
  id: string;
  /** The name shown in the mastery meter and the glossary. */
  label: string;
  /** One line, in the course's voice. */
  def: string;
  /** The level that introduces (defines) the concept — `null` if none does yet. */
  introduced: string | null;
  /** The levels that use the concept, in curriculum order. */
  used: readonly string[];
}

/** The 22 level ids, in course order — the curriculum skeleton. */
export const LEVEL_ORDER: readonly string[] = [
  "tree",
  "equal",
  "reach",
  "structure",
  "referee",
  "choice",
  "search",
  "learn",
  "measure",
  "architecture",
  "advanced",
  "coherence",
  "mint",
  "truthpay",
  "exposure",
  "substrate",
  "ports",
  "honest",
  "bounded",
  "deliver",
  "pricing",
  "improve",
];

/** Every concept the course teaches, seeded from the level text. */
export const CONCEPTS: readonly Concept[] = [
  // ── 0 · a program is a tree ──────────────────────────────────
  {
    id: "program-tree",
    label: "Program as a tree",
    def: "Operations over data, one root; work flows upward to it.",
    introduced: "tree",
    used: ["tree", "equal", "reach", "substrate"],
  },
  {
    id: "work",
    label: "Work (flops)",
    def: "The arithmetic a node costs — the toy's static cost.",
    introduced: "tree",
    used: ["tree", "equal", "choice", "measure", "deliver"],
  },
  // ── 1 · two programs, one function ───────────────────────────
  {
    id: "semantic-equality",
    label: "Semantic equality",
    def: "Two programs are equal when they compute the same function, not when they look alike.",
    introduced: "equal",
    used: ["equal", "referee", "architecture", "coherence", "bounded"],
  },
  {
    id: "associativity",
    label: "Associativity",
    def: "Matmul is associative: bracketing changes cost, never the result.",
    introduced: "equal",
    used: ["equal", "structure", "coherence", "deliver"],
  },
  // ── 2 · the reachable set ────────────────────────────────────
  {
    id: "reach",
    label: "Reach",
    def: "The programs your laws can get you to — the closure of your law set.",
    introduced: "reach",
    used: ["reach", "exposure", "improve"],
  },
  {
    id: "closure",
    label: "Closure",
    def: "The fixed point of applying laws from a seed — everything reachable.",
    introduced: "reach",
    used: ["reach", "truthpay"],
  },
  {
    id: "e-graph",
    label: "E-graph",
    def: "The equality-saturation structure holding every equivalent form.",
    introduced: "reach",
    used: ["reach", "exposure"],
  },
  // ── 3 · laws come from structure ─────────────────────────────
  {
    id: "law",
    label: "Law",
    def: "A licensed rewrite — an equivalence you may apply.",
    introduced: "structure",
    used: ["structure", "referee", "choice", "coherence", "mint", "improve"],
  },
  {
    id: "algebraic-structure",
    label: "Algebraic structure",
    def: "The algebra a program instantiates that makes a law true.",
    introduced: "structure",
    used: ["structure", "substrate"],
  },
  {
    id: "monoid",
    label: "Monoid",
    def: "A set with an associative operation and an identity — it gives associativity.",
    introduced: "structure",
    used: ["structure"],
  },
  {
    id: "comonoid",
    label: "Comonoid",
    def: "The dual structure whose law is duplication.",
    introduced: "structure",
    used: ["structure"],
  },
  // ── 4 · staying correct ──────────────────────────────────────
  {
    id: "certificate",
    label: "Certificate",
    def: "The replayable proof that a rewrite preserves the function.",
    introduced: "referee",
    used: ["referee", "architecture", "bounded", "improve"],
  },
  {
    id: "referee",
    label: "Referee",
    def: "The arbiter that decides equivalence — semantics, never cost or search.",
    introduced: "referee",
    used: ["referee", "architecture", "mint"],
  },
  {
    id: "oracle",
    label: "Oracle",
    def: "The numeric check that a candidate rewrite is true.",
    introduced: "referee",
    used: ["referee", "learn"],
  },
  // ── 5 · choosing among equals ────────────────────────────────
  {
    id: "frontier",
    label: "Frontier",
    def: "The candidates no other candidate dominates.",
    introduced: "choice",
    used: ["choice", "pricing"],
  },
  {
    id: "non-dominated-set",
    label: "Non-dominated set",
    def: "The set kept because nothing beats it on every axis.",
    introduced: "choice",
    used: ["choice"],
  },
  {
    id: "scalarization",
    label: "Scalarization",
    def: "Collapsing a trade-off to one weighted number too early.",
    introduced: "choice",
    used: ["choice", "pricing"],
  },
  // ── 6 · searching the space ──────────────────────────────────
  {
    id: "policy",
    label: "Policy",
    def: "A rule for which legal move to try next.",
    introduced: "search",
    used: ["search", "learn"],
  },
  {
    id: "legal-move",
    label: "Legal move",
    def: "A move a law licenses — the only moves a policy may order.",
    introduced: "search",
    used: ["search"],
  },
  {
    id: "search-space",
    label: "Search space",
    def: "The (exponential) set of programs reachable by legal moves.",
    introduced: "search",
    used: ["search", "learn", "improve"],
  },
  // ── 7 · learning the policy ──────────────────────────────────
  {
    id: "reward",
    label: "Reward",
    def: "The cost improvement a step buys — what learning optimizes.",
    introduced: "learn",
    used: ["learn"],
  },
  {
    id: "structural-encoding",
    label: "Structural encoding",
    def: "Encoding a rule by its shape, not its index, so unseen rules still score.",
    introduced: "learn",
    used: ["learn"],
  },
  {
    id: "features",
    label: "Features",
    def: "Static numbers describing a program without running it.",
    introduced: "learn",
    used: ["learn", "measure"],
  },
  {
    id: "learned-policy",
    label: "Learned policy",
    def: "A policy learned from trajectories — it may be wrong, but never illegal.",
    introduced: "learn",
    used: ["learn"],
  },
  // ── 8 · measuring the machine ────────────────────────────────
  {
    id: "cost-model",
    label: "Cost model",
    def: "A static or fitted model that predicts runtime.",
    introduced: "measure",
    used: ["measure", "deliver", "pricing"],
  },
  {
    id: "roofline",
    label: "Roofline",
    def: "The compute-or-memory bound a program hits.",
    introduced: "measure",
    used: ["measure", "deliver"],
  },
  {
    id: "ranking",
    label: "Ranking",
    def: "What a cost model does — it orders; it does not decide truth.",
    introduced: "measure",
    used: ["measure", "pricing"],
  },
  {
    id: "calibration",
    label: "Calibration",
    def: "Correcting a model with measured per-family factors.",
    introduced: "measure",
    used: ["measure", "pricing"],
  },
  // ── 9 · the architecture ─────────────────────────────────────
  {
    id: "four-dimensions",
    label: "The four dimensions",
    def: "Semantics, search, evaluation, execution — kept apart so each is replaceable.",
    introduced: "architecture",
    used: ["architecture", "advanced", "ports"],
  },
  {
    id: "evaluation-dimension",
    label: "Evaluation",
    def: "The dimension that ranks — it may not decide equivalence.",
    introduced: "architecture",
    used: ["architecture", "advanced", "pricing"],
  },
  // ── 10 · advanced tensions ───────────────────────────────────
  {
    id: "hard-filter",
    label: "Hard filter",
    def: "Correctness as a filter, not a reward term — an unsound program is unrepresentable.",
    introduced: "advanced",
    used: ["advanced"],
  },
  // ── 11 · laws about laws ─────────────────────────────────────
  {
    id: "critical-pair",
    label: "Critical pair",
    def: "Two laws that fire on one redex and whose reducts may not rejoin.",
    introduced: "coherence",
    used: ["coherence"],
  },
  {
    id: "confluence",
    label: "Confluence",
    def: "When every divergent pair rejoins.",
    introduced: "coherence",
    used: ["coherence"],
  },
  {
    id: "mediating-law",
    label: "Mediating law",
    def: "A law that fills a divergence — often the inverse of the split.",
    introduced: "coherence",
    used: ["coherence"],
  },
  // ── 12 · the vocabulary grows itself ─────────────────────────
  {
    id: "minting",
    label: "Minting",
    def: "Growing the vocabulary by measurement: need → handler → laws → admission.",
    introduced: "mint",
    used: ["mint", "improve"],
  },
  {
    id: "admission",
    label: "Admission",
    def: "The gauntlet a minted atom clears to enter the library.",
    introduced: "mint",
    used: ["mint"],
  },
  {
    id: "census",
    label: "Census",
    def: "The measured count of op-and-shape patterns in the corpus.",
    introduced: "mint",
    used: ["mint", "exposure"],
  },
  {
    id: "provenance",
    label: "Provenance",
    def: "Where a law came from — minted, assisted, or hand-written.",
    introduced: "mint",
    used: ["mint"],
  },
  // ── 13 · true is not the same as paying ──────────────────────
  {
    id: "guard",
    label: "Guard",
    def: "The condition that licenses a rewrite on a shape.",
    introduced: "truthpay",
    used: ["truthpay"],
  },
  {
    id: "adversarial-sweep",
    label: "Adversarial sweep",
    def: "Enumerating shapes to break a guard before shipping it.",
    introduced: "truthpay",
    used: ["truthpay"],
  },
  // ── 14 · where discovery can pay ─────────────────────────────
  {
    id: "exposure",
    label: "Exposure",
    def: "The FLOPs-weighted share of a graph whose ops appear in some law.",
    introduced: "exposure",
    used: ["exposure", "improve"],
  },
  {
    id: "bimodal",
    label: "Bimodal exposure",
    def: "Exposure splits: a 0% fused-op mode and a ≥95% spelled-math mode.",
    introduced: "exposure",
    used: ["exposure"],
  },
  // ── 15 · sharing is a morphism ───────────────────────────────
  {
    id: "morphism",
    label: "Morphism",
    def: "A typed arrow — the combinator presentation reads a program as morphisms.",
    introduced: "substrate",
    used: ["substrate"],
  },
  {
    id: "sharing",
    label: "Sharing (Δ)",
    def: "A shared subterm made explicit as a Δ morphism.",
    introduced: "substrate",
    used: ["substrate"],
  },
  {
    id: "loop",
    label: "Loop (trace)",
    def: "A recurrence made first-class as a typed trace.",
    introduced: "substrate",
    used: ["substrate"],
  },
  {
    id: "monoidal-product",
    label: "Monoidal product",
    def: "The ⊗ that runs two programs side by side (`par`).",
    introduced: "substrate",
    used: ["substrate"],
  },
  // ── 16 · a backend is a port ─────────────────────────────────
  {
    id: "port",
    label: "Port",
    def: "A boundary contract: a Source lifts a model, a Sink lowers the IR.",
    introduced: "ports",
    used: ["ports", "improve"],
  },
  {
    id: "supported-ops",
    label: "supported_ops",
    def: "The ops a sink can lower — the extraction's hard bound.",
    introduced: "ports",
    used: ["ports"],
  },
  // ── 17 · the ledger of negatives ─────────────────────────────
  {
    id: "negative-result",
    label: "Negative result",
    def: "A measured refusal taught as a result, not buried.",
    introduced: "honest",
    used: ["honest", "improve"],
  },
  {
    id: "falsification",
    label: "Falsification",
    def: "The criteria that let a claim be killed by measurement.",
    introduced: "honest",
    used: ["honest", "improve"],
  },
  // ── 18 · approximation with a certificate ────────────────────
  {
    id: "bounded-approximation",
    label: "Bounded approximation",
    def: "A near-equal program carrying a certified error bound.",
    introduced: "bounded",
    used: ["bounded"],
  },
  {
    id: "economics-test",
    label: "Economics test",
    def: "The second gate: an approximation must also be cheaper.",
    introduced: "bounded",
    used: ["bounded", "pricing"],
  },
  // ── 19 · the program must run ────────────────────────────────
  {
    id: "delivery",
    label: "Delivery",
    def: "How a term runs: eager, carrier, fused, graph, or minted kernel.",
    introduced: "deliver",
    used: ["deliver", "improve"],
  },
  {
    id: "dispatch",
    label: "Dispatch",
    def: "One launch per op — the overhead delivery removes.",
    introduced: "deliver",
    used: ["deliver"],
  },
  {
    id: "carrier",
    label: "Carrier",
    def: "An executor-level lowerer for a whole subgraph, not one op.",
    introduced: "deliver",
    used: ["deliver", "pricing"],
  },
  {
    id: "minted-kernel",
    label: "Minted kernel",
    def: "A fusion kernel the search minted when no shipped law named it.",
    introduced: "deliver",
    used: ["deliver"],
  },
  // ── 20 · the cost model can be wrong ─────────────────────────
  {
    id: "inversion",
    label: "Inversion",
    def: "A cost model's error that flips its ranking — it prices the slower member cheaper.",
    introduced: "pricing",
    used: ["pricing"],
  },
  // ── 21 · the payoff ──────────────────────────────────────────
  {
    id: "gap",
    label: "Gap",
    def: "An idea mapped to the project's open work.",
    introduced: "improve",
    used: ["improve"],
  },
  {
    id: "blocker",
    label: "Blocker",
    def: "What stands between mints-and-ships and pays-at-scale.",
    introduced: "improve",
    used: ["improve"],
  },
  {
    id: "next-measurement",
    label: "Next measurement",
    def: "The measurement that would resolve a blocker.",
    introduced: "improve",
    used: ["improve"],
  },
];

/** A concept used before it is introduced — an ordering bug, recorded. */
export interface ForwardRef {
  id: string;
  label: string;
  /** The level that first uses the concept. */
  firstUsedAt: string;
  /** The level that introduces it — `null` when none of the levels does. */
  introducedAt: string | null;
  /** `introductionIndex − firstUseIndex`; `Infinity` when never introduced. */
  gap: number;
}

/** A level's position in the course — `-1` for an unknown id. */
export function levelIndex(id: string, order: readonly string[] = LEVEL_ORDER): number {
  return order.indexOf(id);
}

export function conceptById(
  id: string,
  concepts: readonly Concept[] = CONCEPTS,
): Concept | undefined {
  return concepts.find((c) => c.id === id);
}

/** The earliest level that uses a concept, or `null` if it uses none. */
export function firstUsed(c: Concept, order: readonly string[] = LEVEL_ORDER): string | null {
  let best: string | null = null;
  let bestIndex = Number.POSITIVE_INFINITY;
  for (const lvl of c.used) {
    const i = levelIndex(lvl, order);
    if (i >= 0 && i < bestIndex) {
      bestIndex = i;
      best = lvl;
    }
  }
  return best;
}

/** The index that introduces a concept — `Infinity` when none does. */
export function introductionIndex(c: Concept, order: readonly string[] = LEVEL_ORDER): number {
  if (c.introduced === null) return Number.POSITIVE_INFINITY;
  const i = levelIndex(c.introduced, order);
  return i < 0 ? Number.POSITIVE_INFINITY : i;
}

/** Concepts whose first use precedes their introduction, worst gap first. */
export function forwardRefs(
  concepts: readonly Concept[] = CONCEPTS,
  order: readonly string[] = LEVEL_ORDER,
): ForwardRef[] {
  const out: ForwardRef[] = [];
  for (const c of concepts) {
    const first = firstUsed(c, order);
    if (first === null) continue;
    const firstIndex = levelIndex(first, order);
    const intro = introductionIndex(c, order);
    if (intro > firstIndex) {
      out.push({
        id: c.id,
        label: c.label,
        firstUsedAt: first,
        introducedAt: c.introduced,
        gap: intro - firstIndex,
      });
    }
  }
  return out.sort(
    (a, b) => b.gap - a.gap || levelIndex(a.firstUsedAt, order) - levelIndex(b.firstUsedAt, order),
  );
}

/** Concepts a level uses — what its checkpoint exercises. */
export function conceptsAt(levelId: string, concepts: readonly Concept[] = CONCEPTS): Concept[] {
  return concepts.filter((c) => c.used.includes(levelId));
}

/** Concepts a level introduces (defines). */
export function conceptsIntroducedAt(
  levelId: string,
  concepts: readonly Concept[] = CONCEPTS,
): Concept[] {
  return concepts.filter((c) => c.introduced === levelId);
}

/** The level that teaches a concept — the introducer, or its first use when none defines it. */
export function taughtAt(c: Concept, order: readonly string[] = LEVEL_ORDER): string | null {
  return c.introduced ?? firstUsed(c, order);
}

/** Concepts a level teaches — what its checkpoint and mastery meter attribute to it. */
export function conceptsTaughtAt(
  levelId: string,
  concepts: readonly Concept[] = CONCEPTS,
  order: readonly string[] = LEVEL_ORDER,
): Concept[] {
  return concepts.filter((c) => taughtAt(c, order) === levelId);
}

/** Concepts with no introducing level yet — the glossary step 2 must write. */
export function undefinedConcepts(concepts: readonly Concept[] = CONCEPTS): Concept[] {
  return concepts.filter((c) => c.introduced === null);
}

/** The levels a level depends on — the introducers of the concepts it uses.
 *
 * Plan 0028 §A: the ladder is not flat.  A level `L` depends on
 * every level that *introduces* a concept `L` uses, when that
 * introduction precedes `L`.  Concepts introduced at `L` itself are
 * not prerequisites; a forward reference (introduction after `L`)
 * would be excluded here *and* flagged by {@link forwardRefs} — but
 * the graph has none.
 */
export function prerequisites(
  levelId: string,
  concepts: readonly Concept[] = CONCEPTS,
  order: readonly string[] = LEVEL_ORDER,
): string[] {
  const here = levelIndex(levelId, order);
  if (here < 0) return [];
  const out = new Set<string>();
  for (const c of conceptsAt(levelId, concepts)) {
    if (c.introduced === null) continue;
    const i = levelIndex(c.introduced, order);
    if (i >= 0 && i < here) out.add(c.introduced);
  }
  return [...out].sort((a, b) => levelIndex(a, order) - levelIndex(b, order));
}

/** One concept's row in the glossary — label, definition, and home level. */
export interface GlossaryRow {
  id: string;
  label: string;
  def: string;
  /** The level that introduces (or first uses) the concept. */
  introduced: string;
  /** Its curriculum index. */
  index: number;
  /** The levels that use it. */
  used: readonly string[];
}

/** The glossary: every concept, ordered by the level that introduces it.
 *
 * Plan 0028 §B step 2: the glossary panel is the concept graph made
 * browsable — one line per idea, the level that defines it, and a jump
 * to that level.  Derived from the graph, so it cannot drift from the
 * mastery meter or the prerequisite display.
 */
export function glossary(
  concepts: readonly Concept[] = CONCEPTS,
  order: readonly string[] = LEVEL_ORDER,
): GlossaryRow[] {
  return concepts
    .map((c) => {
      const introduced = c.introduced ?? firstUsed(c, order) ?? order[0];
      return {
        id: c.id,
        label: c.label,
        def: c.def,
        introduced,
        index: levelIndex(introduced, order),
        used: c.used,
      };
    })
    .sort((a, b) => a.index - b.index || a.label.localeCompare(b.label));
}
