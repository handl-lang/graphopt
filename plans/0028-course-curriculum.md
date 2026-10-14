# Plan 0028 — the course as a curriculum (clarity, coverage, mastery)

Depends on: plans 0026/0027 (the toy course's cycles 3–4), the real
project's docs (`docs/`, `project/adrs/`, `project/retros/`),
ADR 0002/0003/0005/0006/0007/0008.

## Motivation — audited gaps

The course is now 22 gated levels with live interactives.  Audited
against the goal *"full comprehension of the project, built
progressively"*, four structural gaps stand out:

1. **No prerequisite structure.**  Levels are a flat list.  A learner
   cannot see what depends on what, and cannot tell whether they are
   ready for a level.  (Forward references exist: `closure` at 2,
   `monoid/comonoid` at 3, `non-dominated set` at 5, `critical pair`
   at 11, `monoidal product` at 15 — each used before it is defined.)
2. **No per-level outcomes or terms.**  Each level has an `objective`
   one-liner but not: *"by the end you can …"*, the **new terms** it
   introduces, or where the idea lives in the **real repo** (the whole
   point is to improve the project).
3. **Coverage holes.**  Missing as lessons: the corpus/intake
   (166 workloads), the bench harness + ledger + gate + pinned
   baselines, the cost-model *family* (roofline/executor/delivered +
   profiles), delivery artifacts (export formats, CUDA-graph runner,
   the native engine), the cell store / handl bridge, the game layer
   (boards, players, learned priors, meta-arena), the failure taxonomy,
   and the engineering ratchets (coverage/ty/radon gates — how the
   project keeps itself honest).
4. **Gamification is thin.**  Gating + a % meter + a cleared badge.
   No points, no mastery per concept, no review, no module-level
   checkpoints, no completion that *means* comprehension.

## Design

### A. Structure: modules, prerequisites, and outcomes

- **Six modules (arcs)**, each 4–6 levels, each ending in a **module
  review** (spaced retrieval over that module's concepts):

  | module | arc | levels (existing + new) |
  |---|---|---|
  | I · Foundations | what a program is, what equality means | 0 tree, 1 equal, **NEW: the corpus** (where programs come from), 2 reach, 3 structure |
  | II · Correctness & choice | certificates, frontiers, search | 4 referee, 5 choice, 6 searching, 7 learning |
  | III · Evaluation & execution | what looks good vs what ran | 8 measuring, **NEW: the cost-model family**, **NEW: delivery artifacts**, 9 architecture |
  | IV · Discovery & minting | growing the vocabulary | 10 tensions, 11 coherence, 12 mint, 13 truthpay, 14 exposure |
  | V · Substrate & backends | the presentation and the ports | 15 substrate, 16 ports, **NEW: the cell store**, **NEW: the game layer** |
  | VI · Engineering & payoff | how the project stays honest, and where to help | **NEW: the gates** (ratchets/coverage/bench), 17 negatives, 18 bounded, 19 deliver, 20 pricing, 21 improve |

- **Every level gains four fields**: `outcome` ("by the end you can
  …"), `terms` (new vocabulary, defined in-level), `repo` (the real
  files/paths that implement it), `module` (which arc).
- **A concept graph** (`concepts.ts`): every concept id, its label,
  which level introduces it, which levels use it — driving both the
  prerequisite display and the mastery meter.

### B. Per-level template (the clarity contract)

1. **Why you are here** — one sentence tying it to the previous level.
2. **New terms** — defined *before* first use, listed and highlighted.
3. **The idea** — short prose, one idea.
4. **See it** — the worked example (numbers on screen).
5. **Do it** — the interactive (live wherever TS can compute it).
6. **Check it** — a **multi-part** checkpoint (≥2 questions, one of
   which must be answered from the interactive).
7. **You can now** — the outcome restated as a capability.
8. **In the real project** — file pointers + the retro/ADR.

### C. Gamification (points, mastery, review, badges)

- **Points**: each level awards XP — a base for passing the checkpoint
  plus a bonus for interacting with the live widget (so points reward
  *doing*, not clicking next).  Module reviews award more.
- **Mastery per concept**: each concept accumulates correct answers
  across checkpoints *and* reviews (spaced retrieval) → a per-concept
  meter (weak / familiar / mastered) shown on the concept graph.
- **Badges**: e.g. *First Certificate*, *Sweep Breaker* (level 13),
  *Module I–VI Complete*, *Negative Results Kept* (level 17),
  *Full Comprehension* (all levels + all reviews ≥ threshold).
- **Module bosses**: the module review is a gated multi-question
  round; failing sends you back to the levels whose concepts were
  weak (using the mastery data) — progressive, not punitive.
- **Completion that means something**: the final state lists the
  concepts mastered, the levels cleared, and the gaps *you* are now
  equipped to attack (from level 21).

### D. Coverage: the new levels (six)

1. **the corpus** — where programs come from: export, the intake
   (166 workloads, 114 verified), the three lowering bugs it caught.
2. **the cost-model family** — features → roofline/executor/delivered
   → profiles/calibration; why one model is not enough.
3. **delivery artifacts** — what "runnable" means: module/safetensors/
   torchscript/.pt2, the CUDA-graph runner, the native Rust engine.
4. **the cell store** — one record per vocabulary item (op/law/object/
   handle), fragments, provenance origins, the handl bridge.
5. **the game layer** — boards, players, learned priors, the
   meta-arena, `play --domain`; what the priors actually buy.
6. **the gates** — how the project stays honest: coverage floors,
   ty/ruff/vulture/import-linter, the radon ratchet, bench gate +
   pinned baselines, and the no-new-suppressions rule.

## Constraints

- **No forward references.**  A term used in level N must be defined
  in level ≤ N; a test asserts it (grep the level text against the
  concept graph's introduction order).
- One idea per level still holds; comprehensiveness comes from
  *more levels*, not longer ones.
- Live wherever TS can compute; illustrative numbers labelled and
  sourced from `project/retros/`.
- `vp check` / `vp run -r test` / `vp run -r build` must pass;
  `STORAGE_KEY` → `v4` (shape changes); progress migrates or resets
  cleanly (a reset is acceptable if announced).
- Tests: prerequisite-order test, points/mastery unit tests, module
  gating test, and the existing gating/checkpoint tests updated.

## Falsification

- If a learner can reach the end without the mastery meter leaving
  "weak" on any concept, the checkpoints are too shallow — deepen
  them rather than add levels.
- If the concept graph cannot be built without hand-labelling every
  level's terms, the labels are the deliverable (keep them in data).
- If the module reviews merely repeat the level checkpoints, they
  are not reviews — they must *recombine* concepts across the arc.

## Order

1. **C + A-skeleton** (gamification engine: `points.ts`, `mastery.ts`,
   `concepts.ts`, badges, review round) over the existing 22 levels —
   self-contained, no content rewrite.
2. **B** (the per-level template fields + glossary + repo pointers)
   applied to the existing 22 levels.
3. **D** (the six new levels) + module boundaries + reviews.
4. Final: completion state + README/docs refresh.

## Status (2026-10-14) — implemented

All three steps landed on `main` (`toy/` only):

1. **Engine + shell** (`4cf9b8c`, salvaged from the OOM reboot):
   `concepts.ts` (concept graph), `points.ts`, `mastery.ts`, `badges.ts`,
   `review.ts` + shell chrome (XP/rank, mastery meter, badges, recap).
2. **Per-level template + forward-ref fixes** (`ace7270`, `c23fe72`,
   `97ff8f4`): all 22 levels carry `why`/`terms`/`outcome`/`repo`/
   `module` + multi-part checkpoints; **forward references 22 → 0**
   (8 promoted to their teaching level, 14 incidental mentions
   removed; `FORWARD_REF_CEILING = 0` test); glossary panel (60
   concepts); prerequisite display; multi-part checkpoints feed points
   and per-concept mastery.
3. **Coverage + modules + completion** (`27fc034`, `13979cf`): six new
   levels (22 → 28) — `corpus`, `models`, `artifacts`, `cellstore`,
   `game`, `gates`; six modules with reviews that recombine concepts
   across ≥2 levels of their arc (tested); module completion gated on
   the review; `Course.completion` reports XP + rank, levels/reviews
   cleared, concepts mastered, badges, and the gaps the learner is
   equipped to attack.  `STORAGE_KEY` → `v5`.

Measured: **forward refs 0** (28 levels, 80 concepts); tests
**174 → 215** (engine 143, app 72); `vp check` / `vp run -r test` /
`vp run -r build` all pass.
