# GOALS — the standing goal ledger

**Why this file exists**: so the work is *auditable* rather than
asserted.  Every goal carries a pass/fail criterion and the artifact
that proves it.  **If a claim is not in this ledger with an artifact,
treat it as not done.**  Commit messages name the goal they serve
(`G0-Q0`, `G1-ports`, …), so a night's work can be audited from
`git log` alone.

## The enduring goal (G\*)

A **semantic categorical language + learned search that discovers new
algorithms and semantics nobody wrote** — where discoveries are
*certified* (an equation with a replayable derivation), and where the
object language itself grows.

Not a night's deliverable.  Tonight only *unblocks* it.  **No artifact
below may be presented as G\* being met.**

## G0 — the handl ↔ catopt boundary spike (plan 0029)

Answer four questions with measurements and produce a decision.

| q | question | pass criterion | artifact |
|---|---|---|---|
| **Q0** | can catopt's **language** be expressed as handl data? | one catopt law encoded as a `TransformDef` + one shape as `AstShapeDef`; merges; handl's `RewriteEngine` fires it; result matches catopt's own `apply_rule` on the same input | retro + the fragment files |
| **Q1** | can a **tensor value** cross the boundary? | one catopt op registered as a handl **host function**; a small tensor term evaluates through handl; dtype/shape/device intact and fp64 agreement with catopt's `eval_term` | retro with the numbers |
| **Q3** | which integration direction? | one of (A) catopt embeds handl (PyO3 peer) or (B) handl drives catopt is buildable in catopt's CI **and** has a named **seam test** | retro + the seam-test sketch |
| **Q2** | what does the boundary cost? | host-fn dispatch does **not** dominate **rewrite throughput** (same order of magnitude vs in-process); exec timing is informational | retro with both timings |

**Decision the spike must produce**: `go (A)` · `go (B)` · `split`
(handl owns language+rewrite+typecheck; catopt owns the tensor exec
interpretation) · `no-go`.  A failed question with a **named gap** is a
successful result; a vague failure is not.

## G1 — tonight's supporting work

| goal | pass criterion | artifact |
|---|---|---|
| land the gate-speedups agent's work | `scripts/gates.sh` runs the suite once under coverage; `NP` default justified by a measured serial-vs-`-n 2` table (or set to 1); `slow` markers placed with measured durations; AGENTS.md's stale "~7.5 min" replaced | commits + the table |
| **ADR 0009 stage 1** — six core ports (`Calculus`, `Engine`, `Proposer`, `Environment`+`Reward`, `Prover`/`Bound`) | each lands **with a conformance test an existing implementation already satisfies**; no behaviour change; `ty`/`ruff`/`lint-imports` green | commits + tests |
| **ADR 0011** — canonical representation | the decision is written with a falsification path (op-graph IR vs arrow calculus) | the ADR |
| final gates on the merged tree | the full pass green **through the new `scripts/gates.sh`** | the run's output in the retro |

## G2 — the architectural chain

`0010 (handl is the kernel)` → **boundary spike (G0)** → `0011
(canonical representation)` → `0009 stage 4 (the calculus package)`.
Each is a prerequisite of the next; none is skipped.  **Proven by**: the
spike's decision, plus docs updated in **both** repositories.

## Invariants (violating one is a failure regardless of results)

1. **No force-push**; **no remote-branch deletion** without explicit
   consent.
2. **No `rm -rf`** outside regenerable caches (`~/.cache`, venvs for
   dead worktrees).
3. **No widening of any ratchet** (coverage floors, `ty` excludes, ruff
   ignores, radon baseline, the procedural-hook count).
4. **One heavy suite at a time**; check `free -m` before each; abort
   below 2.5 GB available.
5. **No invented numbers.**  A number that cannot be sourced is marked
   `UNSOURCED` or dropped with the reason.
6. **Nothing irreversible without asking** (destructive ops, remote
   state, other repos' history).
7. **Leave the tree committed and pushed**, with docs/md updated in both
   repos — so the next session starts from a clean, documented state.

## If something fails

The ledger shows **which** goal is unmet and **why** — no silent drops.
Salvage beats abandon: uncommitted work in a dying worktree gets
committed to its branch before anything else.
