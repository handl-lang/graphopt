---
title: 'Plan 0032 — repo split: current state → gamekit / eqsat / graphopt / kernelopt'
type: plan
date: 2026-10-27
---

# Plan 0032 — repo split: current state → gamekit / eqsat / graphopt / kernelopt

Depends on: plan 0031 (landed), ADR 0013, notes/repo-split-plan.md +
kernel-engineer-board.md.  Phased below; `[P]` marks lanes that run in
parallel (max 3 agents), `[S]` serial.  Docs-only until phase 4.

## Final state

| repo | owns |
|---|---|
| `gamekit` | Board protocol, players, episode/train/eval, calibrated-gate machinery, benchkit |
| `eqsat` | e-graph, certificates, admission gauntlet, `dag_cost`, Sink port, Rust core |
| `graphopt` (was research-mlopt) | tensor law bank, discovery machinery, sinks, intake/zoo, boards, bench suites, paper |
| `kernelopt` (**private**) | codegen rules, kernel sinks, kernel board, paper — absent from org docs |

Edges: research repos → `eqsat` (substrate) and → `gamekit` (play);
`gamekit` ⊥ `eqsat`.  Machinery up, contents down.

## Cut map (per-package destination)

| package | dest | why |
|---|---|---|
| `mlopt-core` egraph/meta/cost/attrs/typing + laws machinery (gauntlet, provenance) | `eqsat` | substrate |
| `mlopt-core` `laws.ALL_RULES` + `provenance.MACHINE_SHIPPED` contents | `graphopt` | domain bank |
| `mlopt-orchestrator` (search/lower/verify) | `eqsat` | generic pipeline — rename `pipeline` |
| `mlopt-native` (Rust) | `eqsat` | travels |
| `mlopt_discovery.engine`/`players`/`lawdata` (game weights, feature PARTS) | `gamekit` | game layer |
| `mlopt_discovery` mint machinery (antiunify/handler_synth/verifier/evidence/intake) | `graphopt` as `lawmint` | tensor-domain |
| boards (meta/joint/search/torch/gen/open/contraction) | `graphopt` | domains |
| `mlopt-torch`/`onnx`/`carriers`/`cuda`/`kit`/`handl-peer`/`experiments` | `graphopt` | domain sinks/runners |
| `bench/` suite infra + emitters | `gamekit` as `benchkit` | game artifacts |
| `bench/` cases + ex-lab drivers | `graphopt` | domain data/drivers |
| `tests/` / `paper/` | split by subject / `graphopt` | — |

Duplication audit (nothing may be forked): the `bench/discovery`
*runner* is generic → `benchkit` (cases stay per-repo); `meta`/`attrs`/
`dag_cost`/certs single-copy in `eqsat`; `lawdata` is the one file that
*splits* (game weights/PARTS → `gamekit`, `MACROS`/law tables →
`graphopt`) — disjoint contents, same registry.

## Phase 0 — settle `[S]` (hard gate, blocking)

- Other session's WIP lands or parks: no foreign `M`/`D`/`??` in
  `git status`.  Signals today: `open_game.py`, `learned_policy.py`,
  `GOALS.md`, deleted ADR-0012, untracked stop/calibrate drivers.
- One `MLOPT_GATES_LANE=full` green run on HEAD (post-dissolution 99%
  floor still unverified).
- `_member_key` repr-parity + `engine=` contract test-pinned before
  any native-crate move.

## Phase 1 — API freeze `[S]` (1 agent; serial — touches contracts)

- Declare the public surface: `players` registry/`PlayerSpec`, `Board`,
  `Sink` port, `EGraph`/`verify_certificate`/`EGraph.run`, `dag_cost`,
  `engine=`, `verifier.verify_law`, `provenance` registry, **scoring
  spec** (the referee's value function — see phase 2 lane B).
- Extend `lint-imports` contracts + an API-version constant; mark
  everything else internal.

## Phase 2 — restructure `[P:3 lanes, disjoint file sets]`

- **Lane A — gamekit extraction (in-repo)**: `players/` package +
  `bench/players` + `open_self`/`advantage`/`open_vlabel`/`calibrate` +
  `stop_*` drivers + game weight tables → `packages/gamekit/`.
- **Lane B — systematic eval**: referee registry
  (`bench/referees/{naive,oracle,pay,sweep,full}`) + scoring spec
  (`{axis, threshold}` — memory/speed/launch/params + weighting;
  value spec is gamekit's, measurement is gauntlet's — phase-1 seam)
  + matrix runner (`python -m bench.matrix`, designed subset +
  memoized verdicts + Pareto fronts → `docs/matrix.md`) + coverage
  audit/backfill + pin all generated tables into `bench gate`.
- **Lane C — substrate seams**: `laws` machinery↔bank split explicit;
  `lawdata` split; `bench/` drivers re-homed to `bench/drivers/`;
  `mlopt-core`→`egraph_core`, `mlopt_discovery`→`lawmint`,
  `mlopt_orchestrator`→`pipeline` renames (one family per commit —
  lanes sequence their rename commits to avoid collisions).

Ordering within phase 2 (hard): lane C splits `lawdata` **before**
lane A moves the game half; lane B lands referee arms in `bench/` as
*consumers* of `verify_law` — not inside the verifier while C is
relocating it; rename commits serialize across all lanes.

Explosion control (baked into lane B): memoized verdicts dedup the
expensive step across arms (in code today); Pareto fronts + designed
subsets give the shape (planned).  Frontier map, not factorial grid.

## Phase 2b — resilience lane `[P, queued — starts when a phase-2 lane frees; max 3 concurrent]`

Production-ready operational qualities — currently the real gaps:

- **Parallel-safe evaluation** — evidence store gets file-locking or
  per-run namespacing (dir-keyed cache can collide under parallel
  workers); matrix runner is multi-worker from day one (memoized
  verdicts already make parallel eval safe — nothing exploits it yet).
- **Checkpoint/resume** — episodes and matrix lanes persist progress
  (verdict store is already the natural checkpoint boundary); a crash
  costs a cell, not a run.
- **Uniform provenance contract** — seed/env/sha/device required on
  every emitted artifact (scorecard/leaderboard/limits/matrix) —
  enforced inside benchkit emitters, not by convention.
- **Determinism check** — a rerun of the same lane must produce
  byte-identical artifacts modulo timestamps; add a determinism test
  to the suite contract.
- **API versioning** — lands in phase 1 (public surface + version
  constant); everything downstream binds to it.
- **No handrolled primitives** — prefer vetted libraries where they
  replace custom machinery: `joblib.Memory` (disk memoization +
  parallel-safe verdict cache), `filelock`/`diskcache` (store
  locking), `joblib.Parallel`/`concurrent.futures` (worker lanes).
  Each new dep must clear the repo's dep bar (justified, pinned,
  minimal) — candidates reviewed per PR, not blanket-approved.
  Keep-or-swap flagged for the handrolled PAVA isotonic
  (`sklearn.isotonic` exists; ours is ~60 tested lines — decide once).

## Phase 3 — kernelopt spike `[S]` (private, time-boxed, 1 agent) — **DONE, GO**

Verdict recorded in `notes/kernel-engineer-board.md` in the
`kernelopt` repo (extracted in phase 4):
commit `3801504` (`mlopt-kernel` + `KernelSink` + `play --domain
kernel`) — (a) PASS verified-equal cheaper kernels (2.2–3.1× CUDA
fp32); (b) PASS trained player beats playbook/random (0.49/0.34/0.08).
**Private confirmed by user — no org docs, pitch, paper.**

Original spec below for the record:

Minimal surface: `mlopt-kernel/` (small codegen law bank),
`KernelSink` (Triton or torch composite first, verify-first — ONNX
precedent), one kernel board in `play`, 3–5 `bench/discovery` kernel
cases + `triton`/`torch_compile` baseline adapters.
**Go/no-go**: (a) verified-equal kernel at lower measured cost ≥1 case
→ board is real; (b) trained player beats random/playbook on kernel
boards → game transfers.  Fail → document, park.  Bonus: zero
`eqsat`/`gamekit` changes needed = seams proven.

## Phase 4 — extract `[S]` (triggered, not scheduled) — **DONE (part 2)**

`eqsat` (private) + `gamekit` (private) are live repos — filter-repo
history for each package; `mlopt-native` cargo-checks standalone;
graphopt consumes all three via git+ssh subdirectory pins (kernelopt
repointed the same way).  Lessons for the record:

* **Test split is import-driven, not name-driven** — 44 test files
  are substrate-only; every gamekit test in the monorepo reached
  back through `mlopt_discovery` shims, so the gamekit repo got a
  fresh stdlib-only suite (98% measured) with `sys.modules` stubs
  for the lazy domain seam.
* **Coverage floors are suite-membership properties** — the
  substrate's 100% lived mostly in domain-bound tests; eqsat
  re-measured at 56%, gamekit at 98%, graphopt keeps 100% on the
  domain wall + 99% discovery.  Cross-repo merged coverage is a
  phase-5 follow-up.
* **uv lockfiles pin git revs** — repointing sources mid-split
  leaves stale-URL conflicts (`gamekit` via old graphopt rev vs
  new repo); delete `uv.lock` and re-lock, don't `--refresh`.
* **Ambient deps are invisible imports** — one moved test depended
  on the carrier registry being installed, not on an import;
  carved like the import-bound ones.
* **Foreign WIP survives the cut** — an in-flight prior-art bank
  (uncommitted edits + untracked `priorart.py`) was ported to the
  eqsat worktree and committed there as a wip commit.

## Phase 5 — org docs rewrite `[P]` (`handl-lang/docs`, public repos only)

1. Analyze — inventory every page, classify Diataxis + owning repo
   (`gamekit`/`eqsat`/`graphopt`/`handl`; `kernelopt` excluded while
   private), find gaps.
2. Design — per-repo sections + cross-cutting (three-costs, the game,
   the gauntlet); reference bound to the phase-1 frozen surface.
3. Update — rewrite repo-by-repo as the split lands; per-page
   provenance so renames sweep cleanly.

## Phase exit criteria (every phase — the agent loop)

Each lane/phase runs a loop, not a single pass:

1. **Implement** the phase's items.
2. **Test** — new code ships with tests; shipped-package modules must
   meet the coverage floors (99% `lawmint` ratchet / 100% the six);
   bench/suite code gets contract tests like the existing suites.
   Target: **100% coverage on new code**, no exceptions at the floor.
3. **Review** — self-review + parent review the diff: conventions,
   no-dup, no handrolled primitives, honest rows.
4. **Adjust & loop** — fix findings, re-run gates, repeat until all
   exit criteria green.  A phase is not done on "code written" — it's
   done on criteria met.

## Cross-cutting invariants (every phase)

- README/AGENTS.md/docs/paper.qmd re-anchored per move; `paper.pdf`
  re-rendered last.
- Renames one family per commit; full gates per landing.
- Nothing gets duplicated — splits happen exactly once.
- Prefer vetted libs over custom primitives (dep bar applies).
