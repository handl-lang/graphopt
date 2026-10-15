# catopt toy — a course in structural optimization

An interactive course on catopt's architecture: **one idea per level**, a worked
example, an interactive, and a **multi-part checkpoint you must pass before
advancing**. Scaffolded as a **Vite+ monorepo** with `vp`.

```bash
vp install             # install the workspace
vp -C apps/learn dev   # http://localhost:5173
vp check               # format + lint + typecheck (oxfmt / oxlint / tsgo)
vp run -r test         # vitest across the workspace
vp run -r build        # engine → dist, app → dist
```

## The per-level template (plan 0028 §B)

Every level renders the same eight-part clarity contract, in order:

1. **Why you are here** — one sentence tying the level to the one before.
2. **New terms** — the concepts this level introduces, defined before first use.
3. **The idea** — short prose, one idea.
4. **See it** — the worked example (numbers on screen).
5. **Do it** — the live interactive.
6. **Check it** — a **multi-part** checkpoint (≥2 questions; at least one
   answerable only from the interactive). Every part is tagged with the concept
   it exercises, so it feeds both the XP award and the per-concept mastery meter.
7. **You can now** — the outcome restated as a capability.
8. **In the real project** — file pointers into `packages/` + the retro/ADR.

A level also shows its **module** (one of the six arcs) and its
**prerequisites** — the levels that introduce the concepts it uses, read
straight off the concept graph, so the ladder is no longer flat.

## The concept graph, the glossary, and the ordering gate

`packages/engine/src/concepts.ts` is the prerequisite structure as data: for
every concept, the level that introduces it and the levels that use it. It
drives three things:

- the **prerequisite display** on each level (`prerequisites()`),
- the **mastery meter** (`mastery.ts`), and
- the **glossary panel** (`glossary()`) — every concept, its one-line
  definition, its home level, and a jump to that level, reachable from the shell.

`forwardRefs()` returns every concept used before it is introduced. Step 1
seeded the graph and **recorded 22 ordering bugs**; step 2 **fixed them all —
22 → 0** — by either promoting a term's introduction to its first-use level
(the level genuinely teaches it) or removing the incidental prose mention.
Step 3 grew the ladder to **28 levels and 80 concepts**, and the forward-ref
count **stays 0**: every new term is defined at or before its first use. The
tests assert the count, so it can only go down.

## The ladder (basics → advanced)

The course is **28 levels in six modules (arcs)**, each module ending in a
**review** that recombines its concepts. A level also shows its **module** and its
**prerequisites** — the levels that introduce the concepts it uses, read straight
off the concept graph, so the ladder is no longer flat.

| module | #   | Level                            | Idea                                        |
| ------ | --- | -------------------------------- | ------------------------------------------- |
| I      | 0   | a program is a tree              | ops over data, one root                     |
| I      | 1   | two programs, one function       | equality is not syntactic                   |
| I      | 2   | **where programs come from**     | real workloads through the export boundary  |
| I      | 3   | the reachable set                | reach is the closure of your laws           |
| I      | 4   | laws come from structure         | monoid / comonoid / trace / product         |
| II     | 5   | staying correct                  | the certificate is the referee              |
| II     | 6   | choosing among equals            | keep the frontier, don't scalarize          |
| II     | 7   | searching the space              | a policy orders legal moves                 |
| II     | 8   | learning the policy              | trajectories, then reward                   |
| III    | 9   | measuring the machine            | features describe, a model predicts         |
| III    | 10  | **the cost-model family**        | features → roofline → executor → delivered  |
| III    | 11  | **what 'runnable' means**        | module · safetensors · torchscript · `.pt2` |
| III    | 12  | the architecture                 | four dimensions, swappable parts            |
| IV     | 13  | advanced tensions                | where the design gets hard                  |
| IV     | 14  | laws about laws                  | derivable · inverse · divergent             |
| IV     | 15  | the vocabulary grows itself      | need → handler → laws → admission           |
| IV     | 16  | true is not the same as paying   | guard synthesis + adversarial sweep         |
| IV     | 17  | where discovery can pay          | exposure is bimodal                         |
| V      | 18  | sharing is a morphism            | Δ sharing, `loop` recurrence                |
| V      | 19  | a backend is a port              | swap the sink, core never changes           |
| V      | 20  | **one record per idea**          | a cell, merged from fragments               |
| V      | 21  | **the game layer**               | boards, players, learned priors             |
| VI     | 22  | **how the project stays honest** | coverage · ty · ratchets · bench gate       |
| VI     | 23  | the ledger of negatives          | methodology as a result                     |
| VI     | 24  | approximation with a certificate | certificate **and** an economics test       |
| VI     | 25  | the program must run             | same term, different runtime                |
| VI     | 26  | the cost model can be wrong      | the model ranks — and can be wrong          |
| VI     | 27  | the payoff: where to improve     | idea → blocker → next measurement           |

## Modules, reviews, and the completion state (plan 0028 §A/§C)

`review.ts` is the arc data: the six modules (`foundations`, `correctness`,
`evaluation`, `discovery`, `substrate`, `engineering`) and, for each, a
**review round** of six questions. Every question is tagged with a concept
from the arc, and each arc's questions draw on **at least two of its levels** —
the recombination the plan's falsification demands (`reviewLevels` measures it).

- A module is **complete** when every level passes (`moduleComplete`) and
  **cleared** when the review also passes (`moduleCleared`). The module badges
  and _Modules I–VI Complete_ read `moduleCleared`.
- **Mastery** (`mastery.ts`) accumulates correct answers from the level
  checkpoints _and_ the reviews — spaced retrieval — into per-concept
  weak / familiar / mastered tiers. A failed review reports its **weak
  concepts** and links back to the levels that teach them.
- The **completion state** (`Course.completion`) reports XP + rank, levels
  cleared, concepts mastered (weak / familiar / mastered counts), badges earned,
  and the **gaps the learner is now equipped to attack** (the `improve` level's
  data).

Progress lives under `catopt-course-v5` — reviews add a saved field, so a `v4`
blob resets cleanly (announced in the settings text).

## Layout

```
apps/
  learn/                     the course
    index.html               the twenty-eight levels + checkpoint + review shell
    src/main.ts              Alpine bootstrap
    src/style.css            Tailwind v4 (@import + @theme tokens)
    src/components/course.ts the levels + template, gating, reviews, completion
    src/components/*.ts      one interactive per level
    tests/course.test.ts     gating, template, checkpoints, reviews, completion, …
packages/
  engine/                    the toy e-graph engine, as a library
    src/term.ts              the term language + cost model
    src/rules.ts             the six laws (+ rewritesAt)
    src/referee.ts           numeric equivalence on concrete matrices
    src/features.ts          ProgramFeatures, mirroring catopt_core.features
    src/programs.ts          the starting programs
    src/concepts.ts          the concept graph: introduce/use, prerequisites, glossary
    src/corpus.ts            the intake funnel + the workload browser (level 2)
    src/models.ts            the cost-model family — the inversion, live (level 10)
    src/artifacts.ts         delivery artifacts: what "runnable" needs (level 11)
    src/cellstore.ts         the cell store — fragments, origins, handl (level 20)
    src/game.ts              the game layer — boards, players, priors (level 21)
    src/gates.ts             the gates — break one, watch the run fail (level 22)
    src/points.ts            the XP model — the level curve and the named ranks
    src/mastery.ts           per-concept mastery (weak / familiar / mastered)
    src/review.ts            the six arcs + the review questions
    src/badges.ts            the badge predicates
    src/minting.ts           the ADR-0007 minting funnel
    src/guards.ts            a live guard-synthesis mini-sweep
    src/exposure.ts          the bimodal exposure screen (live histogram)
    src/combinators.ts       dag ↔ combinator bridge (Δ, loop)
    src/ports.ts             Source / Sink + a core-mutation counter
    src/pricing.ts           the executor route table — the inversion, live
    src/delivery.ts          one term, five deliveries (modeled vs measured)
    src/bounded.ts           the bounded lane — certificate × economics, live
    src/improve.ts           the project's open work as data (gap table)
    tests/engine.test.ts     vitest, incl. Catalan(3) = 5 bracketings
    tests/concepts.test.ts   the ordering gate, prerequisites and the glossary
    tests/cycle3.test.ts     vitest, the six cycle-3 modules
    tests/cycle4.test.ts     vitest, the four cycle-4 modules
    tests/cycle5.test.ts     vitest, the six cycle-5 coverage modules
```

## What is real, and what is a toy

The **board** is a real five-op algebra (`matmul` / `add` / `mul` / `sq` /
`sqq`) with six laws, a static cost model, a static feature profiler, and
a numeric referee. It is a faithful miniature of the four dimensions —
the shapes are chosen so bracketing genuinely changes the FLOP count, the
referee checks `max|original − current|` on concrete matrices, and the
level-14 probe computes real critical-pair verdicts (confluent /
library-mediated / divergent) by bounded saturation.

The **toy's own numbers** — costs, reach counts, verdicts — are computed
live by the engine. Several levels compute their idea live in TS: the
guard-synthesis sweep (level 16 enumerates a shape domain and _finds_ bad
accepts for the naive guard, 0 for the exact one), the exposure histogram
(level 17 buckets the workload slice on the fly), the combinator round
trip (level 18 checks `dag → comb → dag` is the identity), the bounded
lane's economics (level 24 prices the certified member against dense and
refuses it, live), the delivery model (level 25's modeled column is
`work + launches × overhead`), the executor route table (level 26
computes the routing inversion and its calibrated fix live), and the gap
table (level 27 is data). The six coverage levels are live too: the intake
funnel (level 2 buckets the workload slice and computes its three-way split),
the cost-model family (level 10 shows the delivered inversion and its
calibrated fix), the artifacts (level 11 reads what each needs to run), the
cell store (level 20 merges fragments and raises on a duplicate), the game
layer (level 21 steps a meta-arena episode and compares the players' measured
AP), and the gates (level 22 fails a run when you break one). The minting
funnel (level 15) and the backend swap (level 19) run live over a fixed bank
of measured stage outcomes; the negatives (level 23) are a gallery of
measured refusals.

The **measured numbers in the prose callouts** come from the real Python
system (`project/retros/`): the **11 machine-admitted laws** the library
ships (`catopt_core.laws.provenance.MACHINE_SHIPPED`; the newest is
`mul_unsq_nat` −12.3 % on its source workload — the first minted from a
corpus op-tuple, with four further laws machine-_assisted_ but
hand-integrated: `select_mul`, `softmax_fold`, `glu_fold`,
`rms_norm_fold`), the false-but-cost-lowering
candidate the oracle rejected, the **82-rule** basis catalogue (58 axioms
/ 22 lemmas / 2 redundant) whose one divergence `silu_fold` closed, the
exposure screen's bimodality (**81 of 321** workloads at exactly 0 %,
**140 at ≥95 %**) and its ~1.5 % corpus discovery yield, the fused
carrier lowering at **parity with the paper's own chunked implementation**
(2.08 vs 2.12 ms at T = 2048), the contraction player's ~1.04× parity with
best-of-N greedy at n = 40, the ~10× enumeration-vs-learned-proposer yield
gap, the executor-routing inversion (modeled −20…30 %, measured 1.9–2.7×
slower) and its measured-corrections fix (**12/12** probe cases), the
BiRefNet bounded lane (**20** certified offers at 1e-2, **56** at 3e-2,
**0** delivered — the member priced **1.32×** dense), the `chainfuse`
minted kernel (**2.94× vs `torch.compile`**, 86 vs 253 µs), and the atom
pay-null (a verified minted atom that fires zero certified laws on the
corpus).

## Stack

- [Alpine.js](https://alpinejs.dev) — the interactivity
- [Tailwind CSS v4](https://tailwindcss.com) — via `@tailwindcss/vite`
- [Vite+](https://viteplus.dev) — `vp` for install / check / test / build
