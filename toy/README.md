# catopt toy — a course in structural optimization

An interactive course on catopt's architecture: **one idea per level**, a worked
example, an interactive, and a **checkpoint you must pass before advancing**.
Scaffolded as a **Vite+ monorepo** with `vp`.

```bash
vp install             # install the workspace
vp -C apps/learn dev   # http://localhost:5173
vp check               # format + lint + typecheck (oxfmt / oxlint / tsgo)
vp run -r test         # vitest across the workspace
vp run -r build        # engine → dist, app → dist
```

## The ladder (basics → advanced)

| #   | Level                          | Idea                                |
| --- | ------------------------------ | ----------------------------------- |
| 0   | a program is a tree            | ops over data, one root             |
| 1   | two programs, one function     | equality is not syntactic           |
| 2   | the reachable set              | reach is the closure of your laws   |
| 3   | laws come from structure       | monoid / comonoid / trace / product |
| 4   | staying correct                | the certificate is the referee      |
| 5   | choosing among equals          | keep the frontier, don't scalarize  |
| 6   | searching the space            | a policy orders legal moves         |
| 7   | learning the policy            | trajectories, then reward           |
| 8   | measuring the machine          | features describe, a model predicts |
| 9   | the architecture               | four dimensions, swappable parts    |
| 10  | advanced tensions              | where the design gets hard          |
| 11  | laws about laws                | derivable · inverse · divergent     |
| 12  | the vocabulary grows itself    | need → handler → laws → admission   |
| 13  | true is not the same as paying | guard synthesis + adversarial sweep |
| 14  | where discovery can pay        | exposure is bimodal                 |
| 15  | sharing is a morphism          | Δ sharing, `loop` recurrence        |
| 16  | a backend is a port            | swap the sink, core never changes   |
| 17  | the ledger of negatives        | methodology as a result             |

## Layout

```
apps/
  learn/                     the course
    index.html               the eighteen levels + checkpoint shell
    src/main.ts              Alpine bootstrap
    src/style.css            Tailwind v4 (@import + @theme tokens)
    src/components/course.ts the level list, gating, progress
    src/components/*.ts      one interactive per level
    tests/course.test.ts     gating, reach, compare, tensions, coherence, …
packages/
  engine/                    the toy e-graph engine, as a library
    src/term.ts              the term language + cost model
    src/rules.ts             the six laws (+ rewritesAt)
    src/referee.ts           numeric equivalence on concrete matrices
    src/features.ts          ProgramFeatures, mirroring catopt_core.features
    src/programs.ts          the starting programs
    src/minting.ts           the ADR-0007 minting funnel
    src/guards.ts            a live guard-synthesis mini-sweep
    src/exposure.ts          the bimodal exposure screen (live histogram)
    src/combinators.ts       dag ↔ combinator bridge (Δ, loop)
    src/ports.ts             Source / Sink + a core-mutation counter
    tests/engine.test.ts     vitest, incl. Catalan(3) = 5 bracketings
    tests/cycle3.test.ts     vitest, the six new modules
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
live by the engine. Three of the new levels compute their idea live in TS:
the guard-synthesis sweep (level 13 enumerates a shape domain and _finds_
bad accepts for the naive guard, 0 for the exact one), the exposure
histogram (level 14 buckets the workload slice on the fly), and the
combinator round trip (level 15 checks `dag → comb → dag` is the identity).
The minting funnel (level 12) and the backend swap (level 16) run live
over a fixed bank of measured stage outcomes; the negatives (level 17) are
a gallery of measured refusals.

The **measured numbers in the prose callouts** come from the real Python
system (`project/retros/`): the four machine-discovered laws the library
ships (`select_mul` −25.9 % modeled, `softmax_fold`, `silu_fold`,
`mul_unsq_nat` −12.3 % on its source workload), the false-but-cost-lowering
candidate the oracle rejected, the **82-rule** basis catalogue (58 axioms
/ 22 lemmas / 2 redundant) whose one divergence `silu_fold` closed, the
exposure screen's bimodality (**81 of 321** workloads at exactly 0 %,
**140 at ≥95 %**) and its ~1.5 % corpus discovery yield, the fused
carrier lowering at **parity with the paper's own chunked implementation**
(2.08 vs 2.12 ms at T = 2048), the contraction player's ~1.04× parity with
best-of-N greedy at n = 40, the ~10× enumeration-vs-learned-proposer yield
gap, the executor-routing inversion (modeled −20…30 %, measured 1.9–2.7×
slower) and its measured-corrections fix, and the atom pay-null (a verified
minted atom that fires zero certified laws on the corpus).

## Stack

- [Alpine.js](https://alpinejs.dev) — the interactivity
- [Tailwind CSS v4](https://tailwindcss.com) — via `@tailwindcss/vite`
- [Vite+](https://viteplus.dev) — `vp` for install / check / test / build
