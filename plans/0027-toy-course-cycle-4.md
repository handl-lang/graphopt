---
title: Plan 0027 — the toy course, cycle 4 (delivery · pricing · approximation · payoff)
type: plan
date: 2026-10-14
---

# Plan 0027 — the toy course, cycle 4 (delivery · pricing · approximation · payoff)

Depends on: plan 0026 (cycle 3 — minting/substrate/ports/honesty),
plan 0012 (bounded-error rewrites), plan 0010 (performance), the
`gen` kernel path (`catopt_discovery.genkernel`), ADR 0003 (four
dimensions).

## Motivation — measured gaps after cycle 3

Cycle 3 landed levels 12–18 (minting, truth-vs-pay, exposure,
substrate, ports, honesty).  Auditing the resulting 18-level ladder
against the real system leaves four **uncovered ideas**, each central:

1. **Approximation with certificates** (plan 0012 / `error_budget`).
   The word "bounded" appears only in prose callouts.  The ε axis —
   offering members, certifying a bound, and *refusing* when the
   economics do not work (the measured BiRefNet negative: 56 offers,
   0 delivered, 1.32× worse than dense) — is a whole lesson with no
   level.
2. **Delivery** (execution dimension).  Level 16 swaps a backend;
   nothing teaches that a chosen program must *run fast* — executor
   tables, the fused carrier lowering (parity with the paper's own
   chunked implementation), CUDA-graph capture, and minted kernels
   (`genkernel` — the `chainfuse` 2.94× vs `torch.compile`).
3. **Pricing steers selection — and can be wrong.**  The measured
   executor-routing inversion (modeled −20…30 %, measured 1.9–2.7×
   slower; fixed by calibrated corrections) is the sharpest lesson
   about why wins are regime-dependent, and the course never states
   it.
4. **The payoff chapter.**  The stated objective is *understanding so
   the reader can improve the project* — but the ladder ends without
   mapping its ideas to the project's actual open work (plan 0025:
   enumeration ordering, grammar gaps, gate 3, the three law
   families) and their measured blockers.

Plus one presentation gap: gamification is gating + persistence only
— no visible progress meter, no completion recap.

## Design

### Four new levels (19 → 22)

| # | id | idea | interactive |
|---|---|---|---|
| 19 | `bounded` | approximation is allowed — with a certificate and an economics test | pick a budget; watch members offered → certified → **delivered or refused**; see the bound gate stay bitwise-clean while the *pricing* refuses; the BiRefNet numbers as the worked case |
| 20 | `deliver` | the chosen program must run: same term, different deliveries | pick a delivery (generic / batched / fused / graph-captured / minted kernel) and see modeled cost vs the measured wall-clock story from the retros; the fused-lowering parity and `chainfuse` cases |
| 21 | `pricing` | the cost model steers selection and can be wrong | live: a small route table where the model's pick ≠ the measured winner (the inversion); toggle calibration and watch the pick flip |
| 22 | `improve` | the payoff: each idea → the project's open work | pick a named gap (from plan 0025), see what blocks it, what the next measurement is, and which level teaches the idea behind it |

### Gamification (presentation, not new engine)

- A **progress meter** in the shell (cleared / total), visible without
  scrolling.
- A **per-level cleared badge** once its checkpoint passes.
- A **completion recap** panel (after level 22): the ladder's
  one-liners in order — the course's own summary of itself.

### Engine additions

- `pricing.ts` — a small route table with modeled vs measured costs,
  an uncalibrated and a calibrated model, and `pick()`/`best()`
  computing the inversion live.
- `delivery.ts` — delivery variants for one term with (illustrative,
  sourced) measured wall-clocks and a live modeled-cost column.
- `bounded.ts` — budget → offered/certified/delivered tallies with
  the economics rule (`bounded member must beat dense`) computed
  live over a small bank.
- `improve.ts` — the gap table (id, blocker, next measurement,
  level-that-teaches-it) as data.

### Constraints

- Same house rules as cycles 1–3: one idea per level, live wherever
  computable, illustrative numbers labelled and sourced from
  `project/retros/`.
- No overclaiming: the refusal paths are the point of levels 19/21.
- Bump `STORAGE_KEY` to `v3`; `vp check` / `vp run -r test` /
  `vp run -r build` must pass; extend the engine + course tests.

## Falsification

- If level 19 can be passed without understanding *why* a certified
  member is refused, it teaches the wrong thing — rewrite around the
  economics.
- If level 21's inversion cannot be reproduced live in TS, it becomes
  a worked example in level 20, not its own level.
- If level 22 duplicates plan 0025 verbatim, it is documentation —
  it must teach the *connection* (idea → blocker → next measurement),
  not restate the plan.
