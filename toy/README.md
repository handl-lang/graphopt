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
The tests assert the count, so it can only go down.

## The ladder (basics → advanced)

| #   | Level                            | Idea                                  |
| --- | -------------------------------- | ------------------------------------- |
| 0   | a program is a tree              | ops over data, one root               |
| 1   | two programs, one function       | equality is not syntactic             |
| 2   | the reachable set                | reach is the closure of your laws     |
| 3   | laws come from structure         | monoid / comonoid / trace / product   |
| 4   | staying correct                  | the certificate is the referee        |
| 5   | choosing among equals            | keep the frontier, don't scalarize    |
| 6   | searching the space              | a policy orders legal moves           |
| 7   | learning the policy              | trajectories, then reward             |
| 8   | measuring the machine            | features describe, a model predicts   |
| 9   | the architecture                 | four dimensions, swappable parts      |
| 10  | advanced tensions                | where the design gets hard            |
| 11  | laws about laws                  | derivable · inverse · divergent       |
| 12  | the vocabulary grows itself      | need → handler → laws → admission     |
| 13  | true is not the same as paying   | guard synthesis + adversarial sweep   |
| 14  | where discovery can pay          | exposure is bimodal                   |
| 15  | sharing is a morphism            | Δ sharing, `loop` recurrence          |
| 16  | a backend is a port              | swap the sink, core never changes     |
| 17  | the ledger of negatives          | methodology as a result               |
| 18  | approximation with a certificate | certificate **and** an economics test |
| 19  | the program must run             | same term, different runtime          |
| 20  | the cost model can be wrong      | the model ranks — and can be wrong    |
| 21  | the payoff: where to improve     | idea → blocker → next measurement     |

## Layout

```
apps/
  learn/                     the course
    index.html               the twenty-two levels + multi-part checkpoint shell
    src/main.ts              Alpine bootstrap
    src/style.css            Tailwind v4 (@import + @theme tokens)
    src/components/course.ts the levels + template, gating, prerequisites, glossary
    src/components/*.ts      one interactive per level
    tests/course.test.ts     gating, template, multi-part checkpoints, glossary, …
packages/
  engine/                    the toy e-graph engine, as a library
    src/term.ts              the term language + cost model
    src/rules.ts             the six laws (+ rewritesAt)
    src/referee.ts           numeric equivalence on concrete matrices
    src/features.ts          ProgramFeatures, mirroring catopt_core.features
    src/programs.ts          the starting programs
    src/concepts.ts          the concept graph: introduce/use, prerequisites, glossary
    src/points.ts            the XP model — the level curve and the named ranks
    src/mastery.ts           per-concept mastery (weak / familiar / mastered)
    src/review.ts            the module arcs + the review round
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
```

## What is real, and what is a toy

The **board** is a real five-op algebra (`matmul` / `add` / `mul` / `sq` /
`sqq`) with six laws, a static cost model, a static feature profiler, and
a numeric referee. It is a faithful miniature of the four dimensions —
the shapes are chosen so bracketing genuinely changes the FLOP count, the
referee checks `max|original − current|` on concrete matrices, and the
level-11 probe computes real critical-pair verdicts (confluent /
library-mediated / divergent) by bounded saturation.

The **toy's own numbers** — costs, reach counts, verdicts — are computed
live by the engine. Several levels compute their idea live in TS: the
guard-synthesis sweep (level 13 enumerates a shape domain and _finds_ bad
accepts for the naive guard, 0 for the exact one), the exposure histogram
(level 14 buckets the workload slice on the fly), the combinator round
trip (level 15 checks `dag → comb → dag` is the identity), the bounded
lane's economics (level 18 prices the certified member against dense and
refuses it, live), the delivery model (level 19's modeled column is
`work + launches × overhead`), the executor route table (level 20
computes the routing inversion and its calibrated fix live), and the gap
table (level 21 is data). The minting funnel (level 12) and the backend
swap (level 16) run live over a fixed bank of measured stage outcomes;
the negatives (level 17) are a gallery of measured refusals.

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
