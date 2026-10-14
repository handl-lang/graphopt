export interface Checkpoint {
  q: string;
  opts: string[];
  ok: string;
  why: string;
}

export interface Level {
  id: string;
  title: string;
  /** The ladder one-liner — the course's summary of itself. */
  idea: string;
  objective: string;
  checkpoint: Checkpoint;
}

/** Where progress lives. Bump the suffix if the level list changes shape. */
export const STORAGE_KEY = "catopt-course-v3";

interface Saved {
  i: number;
  answers: Record<string, string>;
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
      checkpoint: {
        q: "In `(a·b)·c`, which node produces the final result?",
        opts: ["a", "the outer matmul", "c"],
        ok: "the outer matmul",
        why: "The root. Every other node feeds it. Optimization is choosing a different tree that computes the same thing — the root stays the answer.",
      },
    },
    {
      id: "equal",
      title: "Two programs, one function",
      idea: "equality is not syntactic",
      objective: "Different trees can compute the same function — at different costs.",
      checkpoint: {
        q: "With a : 2×3, b : 3×4, c : 4×5, the two trees cost 180 and 128 FLOPs. Same function?",
        opts: ["yes", "no", "only if the shapes match"],
        ok: "yes",
        why: "Matrix multiplication is associative: bracketing changes the cost, never the result. That gap — same function, different cost — is the whole opportunity.",
      },
    },
    {
      id: "reach",
      title: "The reachable set",
      idea: "reach is the closure of your laws",
      objective: "Reach is the closure of your laws, not the power of your search.",
      checkpoint: {
        q: "What bounds how many equivalent programs you can ever find?",
        opts: ["the search budget", "the laws you have", "the hardware"],
        ok: "the laws you have",
        why: "Search only explores what the laws generate. A bigger budget over a weak law set still finds nothing new.",
      },
    },
    {
      id: "structure",
      title: "Laws come from structure",
      idea: "monoid / comonoid / trace / product",
      objective: "A law is a consequence of algebraic structure, not an op pattern.",
      checkpoint: {
        q: "Why can't an op-level rule find `(Q·Kᵀ)·V → Q·(Kᵀ·V)`?",
        opts: [
          "it is not a valid rewrite",
          "it is a consequence of associativity plus shapes, not an op pattern",
          "it needs a special kernel",
        ],
        ok: "it is a consequence of associativity plus shapes, not an op pattern",
        why: "No op named 'attention' is involved. The law is associativity; the choice is driven by a shape-aware cost model. An op-pattern language cannot express that.",
      },
    },
    {
      id: "referee",
      title: "Staying correct",
      idea: "the certificate is the referee",
      objective: "Free rewriting is dangerous. Certificates make it safe.",
      checkpoint: {
        q: "What makes a rewrite legal?",
        opts: [
          "the cost model approves it",
          "a law licenses it, and the certificate replays",
          "the search chose it",
        ],
        ok: "a law licenses it, and the certificate replays",
        why: "Legality is semantic. Cost and search have no vote — which is exactly why a learned player cannot produce a wrong program.",
      },
    },
    {
      id: "choice",
      title: "Choosing among equals",
      idea: "keep the frontier, don't scalarize",
      objective: "Among many equals, do not collapse to one scalar too early.",
      checkpoint: {
        q: "Why keep a Pareto frontier instead of picking the cheapest immediately?",
        opts: ["it is faster", "the hardware decides which region matters", "it uses more memory"],
        ok: "the hardware decides which region matters",
        why: "One scalar bakes in one machine's trade-off. The frontier keeps the options; the target picks.",
      },
    },
    {
      id: "search",
      title: "Searching the space",
      idea: "a policy orders legal moves",
      objective: "The space is exponential. A policy decides what to try.",
      checkpoint: {
        q: "What may a search policy never do?",
        opts: ["pick a bad move", "change what is legal", "stop early"],
        ok: "change what is legal",
        why: "A policy only orders legal moves. It can be wrong — it can pick a bad move — but it cannot make a move legal or illegal.",
      },
    },
    {
      id: "learn",
      title: "Learning the policy",
      idea: "trajectories, then reward",
      objective: "A policy can be learned: from trajectories, or from reward.",
      checkpoint: {
        q: "You add a new rewrite rule at runtime. Must the learned policy be retrained?",
        opts: ["yes", "no, if actions are encoded structurally", "only on GPU"],
        ok: "no, if actions are encoded structurally",
        why: "Encode a rule by its shape and an unseen rule is still a point in the same space. Encode it by index and you would have to retrain.",
      },
    },
    {
      id: "measure",
      title: "Measuring the machine",
      idea: "features describe, a model predicts",
      objective: "Static features describe; hardware is truth; a model predicts.",
      checkpoint: {
        q: "What may a performance model never do?",
        opts: ["rank candidates", "prune the semantic space", "predict latency"],
        ok: "prune the semantic space",
        why: "Feasibility — can the backend lower this at all? — is a hard bound. Performance is a soft ranking. Pruning on a prediction loses certified alternatives.",
      },
    },
    {
      id: "architecture",
      title: "The architecture",
      idea: "four dimensions, swappable parts",
      objective: "Everything so far is one of four jobs. Keeping them apart is the design.",
      checkpoint: {
        q: "A cost model rejects a rewrite as unsound. Which rule did it break?",
        opts: [
          "search may not decide equivalence",
          "evaluation may not decide equivalence",
          "execution may not decide equivalence",
        ],
        ok: "evaluation may not decide equivalence",
        why: "EVALUATION ranks; SEMANTICS decides equivalence. The certificate is the referee — that separation is the whole architecture.",
      },
    },
    {
      id: "advanced",
      title: "Advanced tensions",
      idea: "where the design gets hard",
      objective: "Where the design actually gets hard.",
      checkpoint: {
        q: "Why is the certificate a hard filter rather than a reward term?",
        opts: [
          "it is cheaper to compute",
          "a soft reward lets the optimizer trade correctness for cost",
          "rewards are hard to compute",
        ],
        ok: "a soft reward lets the optimizer trade correctness for cost",
        why: "As a reward, correctness becomes spendable. As a hard filter, an unsound program is unrepresentable: search can only lose cost, never truth.",
      },
    },
    {
      id: "coherence",
      title: "Laws about laws",
      idea: "derivable · inverse · divergent",
      objective:
        "Laws relate — derivable, inverse, divergent. Measure the relations; fill the gaps.",
      checkpoint: {
        q: "Two laws fire on the same redex and their reducts never rejoin. What fills the gap?",
        opts: [
          "a bigger search budget",
          "a mediating law — often the definitional inverse of the rule that caused the split",
          "delete one of the two laws",
        ],
        ok: "a mediating law — often the definitional inverse of the rule that caused the split",
        why: "The real library's one measured divergence — silu_expand × swiglu_fuse — was filled by silu_fold, x·σ(x) → silu(x): the inverse of the expansion that killed the fuse redex. Divergent pairs went 2 → 0, and the mediator is a paying law in its own right.",
      },
    },
    {
      id: "mint",
      title: "The vocabulary grows itself",
      idea: "need → handler → laws → admission",
      objective:
        "New meaning enters the object language by measurement, not by hand — need → handler → laws → admission.",
      checkpoint: {
        q: "The funnel mints `scan_fold`, verifies it at 1.78e-15, and admits five certified laws. What did the pay probe find?",
        opts: [
          "it pays on the corpus",
          "zero certified-law fires on every real term",
          "the handler failed to verify",
        ],
        ok: "zero certified-law fires on every real term",
        why: "Truth and pay are separate measurements, and disjoint here — the minted atom computes the total, the corpus emits the sequence. Minting works end to end; paying is a different question.",
      },
    },
    {
      id: "truthpay",
      title: "True is not the same as paying",
      idea: "guard synthesis + adversarial sweep",
      objective:
        "A guard that separates its measured domain perfectly can be unsound just outside it — so sweep.",
      checkpoint: {
        q: "The naive guard `grid-eq` separates its measured domain perfectly. Why is it not the shipped guard?",
        opts: [
          "it is slower",
          "the adversarial sweep found bad accepts past its measured domain",
          "the cost model rejects it",
        ],
        ok: "the adversarial sweep found bad accepts past its measured domain",
        why: "Perfect on the measured envs is not sound. The synthesized flat-map-unsq clause — grid coincidence plus both flat-read maps — has 0 bad accepts and 0 misses, and the apparent sibling was refused by the same guard.",
      },
    },
    {
      id: "exposure",
      title: "Where discovery can pay",
      idea: "exposure is bimodal",
      objective: "Exposure is bimodal: the e-graph can only rewrite math it can see.",
      checkpoint: {
        q: "Of 321 exported workloads, 81 score exactly 0% FLOPs-weighted law coverage. What are they?",
        opts: [
          "broken exports",
          "the fused-op class — `nn.*` natives that export as one opaque node",
          "models with no matmuls",
        ],
        ok: "the fused-op class — `nn.*` natives that export as one opaque node",
        why: "The e-graph literally cannot see the computation: there is nothing to rewrite. Exposure is bimodal, and it is necessary but not sufficient — four fully-visible models pay nothing.",
      },
    },
    {
      id: "substrate",
      title: "Sharing is a morphism",
      idea: "Δ sharing, `loop` recurrence",
      objective:
        "Re-present the IR as combinators: Δ makes sharing explicit, `loop` makes recurrence first-class.",
      checkpoint: {
        q: "In the combinator presentation, what becomes of a value the DAG shares between two parents?",
        opts: ["a pointer", "an explicit Δ morphism — `dup`", "a second copy"],
        ok: "an explicit Δ morphism — `dup`",
        why: "Sharing is a morphism, not a pointer convention, so the DAG-vs-tree ambiguity is impossible by construction. The bridge round-trips — `dag → comb → dag` is the identity — and a recurrence spells as `loop(step)`.",
      },
    },
    {
      id: "ports",
      title: "A backend is a port",
      idea: "swap the sink, core never changes",
      objective: "Swap the backend and nothing in core changes — only what extraction may select.",
      checkpoint: {
        q: "You swap the torch sink for the ONNX sink. What may change in core?",
        opts: ["the IR", "the law set", "nothing — only which forms the extraction may select"],
        ok: "nothing — only which forms the extraction may select",
        why: "A backend is a port: the core-mutation counter stays at zero across the swap. What changes is `supported_ops`, which re-bounds extraction — the ONNX sink can't lower `sqq`, so that member drops out.",
      },
    },
    {
      id: "honest",
      title: "The ledger of negatives",
      idea: "methodology as a result",
      objective: "Methodology as a result: the measured refusals are taught, not buried.",
      checkpoint: {
        q: "The composed atom loop minted a verified atom and admitted four law sets — and fired nothing on the corpus. What does that refusal teach?",
        opts: [
          "the minting machinery is broken",
          "truth and pay are separate measurements — the negative is a result",
          "the corpus is wrong",
        ],
        ok: "truth and pay are separate measurements — the negative is a result",
        why: "Every claim in this course is bounded by what was measured. The ledger of negatives — the falsified weight hypothesis, the pay-null, the BiRefNet bounded negative, the antiunify reach-null — is the methodology, taught as a result rather than a footnote.",
      },
    },
    {
      id: "bounded",
      title: "Approximation with a certificate",
      idea: "certificate AND an economics test",
      objective:
        "Approximation is allowed — but only with a certified bound and an economics test.",
      checkpoint: {
        q: "56 bounded offers carry certified bounds inside a 3e-2 budget, the bound gate refuses nothing — and nothing is delivered. Why?",
        opts: [
          "the bound gate is broken",
          "the bounded member priced 1.32× the dense member, so the pricing refused it",
          "the budget was too small",
        ],
        ok: "the bounded member priced 1.32× the dense member, so the pricing refused it",
        why: "Approximation must clear two gates: a certificate (is it within ε?) and an economics test (is it cheaper?). BiRefNet's near-duplicate rows are real but thin — 0.78% of the qkv weight — so the gather/re-expand/bias-add wrapper cost 1.32× the dense member. The bound gate stayed bitwise-clean (fresh_rel = 0.0); the cost model refused, correctly.",
      },
    },
    {
      id: "deliver",
      title: "The program must run",
      idea: "same term, different runtime",
      objective:
        "The same term, delivered differently, runs at wildly different speeds — delivery is its own dimension.",
      checkpoint: {
        q: "`chainfuse` — `x@A@B@C` plus a 6-op pointwise tail — takes 315 µs eager and 253 µs under `torch.compile`. What delivered it at 86 µs (2.94× vs compile)?",
        opts: [
          "a bigger search budget",
          "the laws plus a minted fusion kernel (the Triton tier)",
          "a faster GPU",
        ],
        ok: "the laws plus a minted fusion kernel (the Triton tier)",
        why: "The reassociation laws fold the weight chain to one GEMM — 181 µs, already 1.34× vs compile — but the fused tail was unreachable: no shipped law names it. The search minted the kernel, claimed it, and the delivered module ran it: 86 µs, verified 1.7e-05. Delivery is the execution dimension; semantics is unchanged.",
      },
    },
    {
      id: "pricing",
      title: "The cost model can be wrong",
      idea: "the model ranks — and can be wrong",
      objective: "The cost model steers selection — and it can be confidently wrong.",
      checkpoint: {
        q: "On the SSM cases the uncalibrated model picks the batched carrier route — which then runs 1.9–2.7× slower than the generic route it skipped. What fixed the pick?",
        opts: [
          "a better search policy",
          "measured corrections fed back into the cost model (calibration)",
          "nothing — it was noise",
        ],
        ok: "measured corrections fed back into the cost model (calibration)",
        why: "The routed model priced the carrier 20–30% cheaper; the clock disagreed. Wiring the measured profile into the delivered-price comparison — a pooled per-family correction — landed the measured-fastest member on 12/12 probe cases. A cost model is a ranking, not the truth: it must be measured, and it may be wrong (ADR 0003).",
      },
    },
    {
      id: "improve",
      title: "The payoff: where to improve",
      idea: "idea → blocker → next measurement",
      objective:
        "Map each idea to the project's open work: the blocker, the next measurement, and the level that teaches it.",
      checkpoint: {
        q: "A gap's next step is a *measurement*, not a build. Which is the project's stated next step for the enumeration-ordering blocker?",
        opts: [
          "rewrite the proposer from scratch",
          "re-measure the SelectiveScan funnel after a canonical emitted-scan seed",
          "delete the fold atom",
        ],
        ok: "re-measure the SelectiveScan funnel after a canonical emitted-scan seed",
        why: "The paying `fold_emit` verifies at 6.7e-16 when reached directly — so the gap is ordering, not expressiveness, and the next step is a measurement: does the canonical seed make the minted atom's law set fire on a real corpus term? If not, the pay-null is semantic and the atom is done. Idea → blocker → next measurement is the whole course, applied.",
      },
    },
  ];

  i = 0;
  answers: Record<string, string> = {};
  settings = false;
  confirming = false;
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
      s.setItem(STORAGE_KEY, JSON.stringify({ i: this.i, answers: this.answers }));
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

  passed(id: string): boolean {
    const lvl = this.levels.find((l) => l.id === id);
    return lvl ? this.answers[id] === lvl.checkpoint.ok : false;
  }

  answered(id: string): boolean {
    return this.answers[id] !== undefined;
  }

  answer(opt: string): void {
    this.answers = { ...this.answers, [this.level.id]: opt };
    this.save();
  }

  get picked(): string | undefined {
    return this.answers[this.level.id];
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

  /** Clear answers and start over — keeps you on the course. */
  restart(): void {
    this.i = 0;
    this.answers = {};
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
    this.restored = false;
    this.confirming = false;
    this.settings = false;
  }

  toggleSettings(): void {
    this.settings = !this.settings;
    this.confirming = false;
  }
}
