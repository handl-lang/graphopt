# Plan 0026 — the toy course, cycle 3 (minting · substrate · honesty)

Depends on: ADR 0007 (semantic-atom minting), ADR 0008 (combinator
presentation), ADR 0006 (ONNX backend), ADR 0003 (four dimensions);
the toy's existing 12-level ladder (`docs/tutorials/course/README.md`).

## Motivation

`docs/tutorials/course/` is the didactic surface — one idea per level, a worked example,
an interactive, a checkpoint you must pass before advancing.  It is
**stale by one cycle**: the ladder ends at "laws about laws" and its
prose still cites *two* machine-discovered laws and the 30-of-53
basis catalogue.  This cycle changed the story in a way the course
does not yet teach:

- the loop now **mints and ships** content (a law from a corpus
  op-tuple, a handler for a previously hand-written executor);
- **truth ≠ pay** — the same machinery honestly refused its apparent
  siblings, its first composed atom, and its own compression layer;
- **exposure** decides where discovery can pay at all (bimodal);
- the IR is being re-presented as **morphisms** (Δ sharing, `loop`
  traces);
- a **second backend** (ONNX) landed with zero core edits.

Goal: pure understanding, so the reader can improve the real project
after finishing the course.

## Design — six new levels (12 → 18)

Each follows the house format: `objective` (one sentence), a worked
example, an interactive component, a gated `checkpoint` with a `why`.

| # | id | idea | interactive |
|---|---|---|---|
| 12 | `mint` | the vocabulary grows itself — need → handler → laws → admission (ADR 0007's four layers) | pick a measured need from a small bank; watch the funnel: signature ✓/✗ → handler proposed+verified → laws synthesized → verdict |
| 13 | `truthpay` | **true is not the same as paying** — guard synthesis, the adversarial sweep, and the refusal | try to break a naive guard by choosing shapes (a live mini-sweep: find a bad accept, see the synthesized exact guard); then the shipped-vs-rejected pair (`mul_unsq_nat` admitted; its apparent sibling refused) |
| 14 | `exposure` | where discovery can pay — the screen is bimodal | classify ops visible/invisible to the law LHSs; see 81/321 vs 140/321 and the ~1.5% yield |
| 15 | `substrate` | sharing is a morphism, recurrence is a `loop` | translate a tiny DAG to a combinator term (Δ becomes explicit) and back; spell a recurrence as `loop(step)`; see the round-trip identity |
| 16 | `ports` | a backend is a port — nothing in core changes | swap the sink (torch ↔ ONNX-shaped) on a fixed core; the core panel stays byte-identical |
| 17 | `honest` | the ledger of negatives — methodology as a result | a gallery of measured refusals (ADR 0001 closed negative, the pay-null, the bounded-lane BiRefNet negative, the antiunify reach-null), each with its number; pick which claim each one kills |

### Engine additions (docs/tutorials/course/packages/engine)

Small, testable, faithful-to-the-real-thing:

- `minting.ts` — a tiny funnel: needs → handler templates → law rows →
  verdicts, with counts matching the real ratios (documented as
  illustrative, not live-computed, where they are not computable in TS).
- `guards.ts` — a real mini-sweep over a small shape domain: a naive
  guard predicate, an enumerator of shapes, a counterexample search,
  and the exact guard; verdicts computed live.
- `exposure.ts` — an op → law-LHS-visible classifier over a fixed op
  list with FLOPs weights; the bimodal histogram computed live.
- `combinators.ts` — the basis (`leaf`/`arr`/`seq`/`par`/`dup`/`loop`),
  `dagToComb`/`combToDag` on the toy term language, round-trip check.
- `ports.ts` — a minimal `Source`/`Sink` pair and a core-mutation
  counter that must stay zero across a sink swap.

### Updates to existing material

- README ladder table (12 → 18) and the "what is real" callouts:
  **four** machine-discovered laws (`select_mul`, `softmax_fold`,
  `silu_fold`, `mul_unsq_nat`); the basis catalogue's current numbers;
  the ~1.5% corpus yield; the fused-lowering parity note.
- Any level prose citing the two-law count or the 53-rule catalogue.
- Bump `STORAGE_KEY` (`catopt-course-v1` → `v2`) — the level list
  changes shape.

## Constraints

- **One idea per level** — if a level needs two paragraphs of setup,
  it is two levels.
- Interactives compute **live** wherever the toy can (guards, exposure
  histogram, round-trip); illustrative numbers are labelled as such
  and sourced from the retro they came from.
- The course must not overclaim: the real system's honest negatives
  are taught as *results*, not footnotes.
- `vp check` (oxfmt/oxlint/tsgo) and `vp run -r test` (vitest) must
  pass; new engine modules need tests (the existing suite covers
  gating, reach, compare, tensions, coherence — extend the pattern).

## Falsification

- A new level whose checkpoint can be passed without reading the
  interactive is a lecture, not a level — rewrite it.
- If a new engine module cannot compute its idea live in TS, the idea
  belongs in prose in an existing level, not a new one.
