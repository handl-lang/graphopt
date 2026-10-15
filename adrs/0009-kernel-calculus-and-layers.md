# ADR 0009 — the kernel calculus as a package, and the layer ladder

**Status**: proposed.
**Date**: 2026-10-14.
**Depends on**: ADR 0002 (categorical re-expression), ADR 0003 (four
dimensions — *and* its lesson that a port with no consumer is
decoration), ADR 0005 (the open game), ADR 0007 (semantic-atom
minting), ADR 0008 (combinator presentation).
**Related**: `catopt_core.ports` (27 protocols), the `lab/bridge`
combinator prototype, plan 0025 (the open frontier), the import-linter
layer contract in `pyproject.toml`.

## Context

Measured, this week:

| package | lines | modules |
|---|---|---|
| **catopt-discovery** | **50,490** | 54 |
| catopt-core | 28,789 | 54 |
| catopt-orchestrator | 14,953 | 17 |
| catopt-carriers | 14,454 | 14 |
| catopt-torch | 10,670 | 24 |
| catopt-onnx | 4,882 | 11 |
| catopt-native | 403 | 2 |
| catopt-cuda | 107 | 2 |

Two structural facts follow.

1. **`catopt-discovery` is the largest package in the repository** and
   it holds four unrelated concerns: the discovery pipeline
   (`census`/`propose`/`oracle`/`impact`/`emit`/`evidence`/`coherence`/
   `vocab`), the **minting** stack (`antiunify`/`handler_synth`/
   `grammar`/`dspec_synth`/`object_synthesis`/`cellstore`/`handl`), the
   **game/RL** layer (`engine`/`players`/`play`/`arena`/`meta_arena`/
   `open_game`/`training`), and **Triton codegen** (`genkernel`).
   The hexagonal boundary is enforced *between* packages; inside this
   one there is no boundary at all.

2. **The ports cover execution and evaluation, and nothing else.**  The
   27 protocols in `catopt_core.ports` describe `Source`/`Sink`/
   `Binding`/`Executor`/`Composer`/`Meter`/`Runner`/`Criterion`/
   `Profiler`/`Policy`/`PerformanceModel`/`Verifier`.  There is **no
   port for the calculus, the search engine, the proposer, or the RL
   environment** — the four things the project's own thesis is about.

Meanwhile the thing that most deserves the name *kernel* exists only as
a prototype in `lab/bridge/combinator.py`: a monoidal basis with
explicit `dup` (Δ) and a bounded `loop` (trace), with `dag ↔ comb`
bridges.  ADR 0008 measured that it round-trips 321/321 corpus
workloads and that the RetNet carrier spells as a `loop` term — but
nothing in the package graph reflects that this is the project's core
object language.

### Why Lindley–Wadler–Yallop is the right basis

Hughes' arrows ("Generalising monads to arrows", 2000) are the
abstraction; **Lindley, Wadler & Yallop's *The Arrow Calculus*** (2008;
and *Idioms are Oblivious, Arrows are Meticulous, Monads are
Promiscuous*, ENTCS 2011) is the part we actually need, because it
supplies a **term calculus with an equational theory** rather than a
typeclass: commands, a typing discipline, and laws.  A kernel language
needs exactly that — a syntax to rewrite *over* and equations to
rewrite *with*.  Hughes' own framing is also the honest one for this
project: arrows are the *meticulous* notion — they make sharing and
sequencing explicit — which is what `dup`/`loop` in our basis are for.

## Decision

**Adopt an eight-layer ladder, promote the calculus to a package, and
give the calculus/search/proposal/environment axes ports in core.**

### The ladder

| layer | package | role | port consumed |
|---|---|---|---|
| L0 primitives | `catopt-core` | terms/IR, e-graph, cells, cost algebra, **all ports** | — |
| L1 calculus | **`catopt-kernel`** *(new)* | the arrow calculus: n-cells, the basis as data, `dag↔comb`, `loop`/`dup`/`Param`, interpreter + rewrite machine | `Calculus`, `Machine` |
| L2 search | `catopt_core.egraph` (reference) · `catopt-native` (fast) | saturation + extraction | `Engine` |
| L3 evaluation | `catopt_core.cost` + measured profiles | roofline/executor/delivered, calibration | `CostFn`, `PerformanceModel`, `Profiler` |
| L4 execution | `catopt-torch` · `catopt-onnx` · `catopt-cuda` · **`catopt-kernels`** *(new: `genkernel`)* | lower and run | `Sink`, `Binding`, `Executor`, `Runner` |
| L5 orchestration | `catopt-orchestrator` | pipeline, strategies, regimes | `Backend`, `Strategy` |
| L6 discovery | `catopt-discovery` *(slimmed)* | census → propose → verify → measure → emit; intake; evidence | `Proposer`, `Verifier` |
| L7 learning | **`catopt-game`** *(new)* | boards, players, priors, training, `play` CLI | `Environment`, `Reward`, `Policy` |
| L8 applications | `bench/` · `lab/` · `examples/` · `docs/tutorials/course` · `scripts/` | measurement, experiments, teaching | — |

### Ports to add in core

Each lands **with a conformance test that the current implementation
already satisfies** — ADR 0003's lesson: an unconsumed port is
decoration.

| port | shape | consumer today (unported) |
|---|---|---|
| `Calculus` | 0/1/2/3-cell structure | the kernel; `Rewrite`/`derivation` in core |
| `Machine` | `eval(term, env)`; `apply(cell, term)` | `catopt-kernel` (`Sink` keeps lowering) |
| `Engine` | `saturate(cells, budget) -> Graph` | `catopt-native` (already `engine=` opt-in) |
| `Proposer` | `propose(graph, budget) -> iterable[Candidate]` | `enumerate`/`forms`/`propose` |
| `Environment` + `Reward` | `reset/step/observe`; `reward(state, action)` | `engine.Board`, `arena`, `meta_arena`, `open_game` |
| `Prover` / `Bound` | per-artifact verification boundaries | `verify_certificate`; bounded members |

### The split of `catopt-discovery`

- → **`catopt-game`**: `engine`, `players`, `play`, `arena`,
  `arena_player`, `meta_arena`, `meta_game`, `meta_player`,
  `open_game`, `training`, `law_order`
- → **`catopt-kernels`** (an execution adapter, beside torch/onnx):
  `genkernel` — it mints handlers, i.e. it is a backend
- → **`catopt-kernel`**: the calculus (from `lab/bridge` + the n-cell
  representation)
- **stays**: `census`, `propose`/`proposal`, `enumerate`, `forms`,
  `families`, `oracle`, `impact`, `emit`, `evidence`, `gap_gen`,
  `coherence*`, `vocab`, `lemma_cert`, `machine_pack`, `intake/`,
  `lawdata/`, `verifier`, `zoo`, and the minting stack

### Staging, with the floor policy

1. **Ports + conformance tests** — no behaviour change, no import churn.
2. **`catopt-kernel`** — promote the lab bridge (measured **92%** under
   its test; needs work to clear the floor).
3. **`catopt-game`** — mechanical move of the RL layer (the largest
   navigability win).
4. **`catopt-kernels`** — move `genkernel` beside the other execution
   adapters.
5. Each stage updates the import-linter layer contract; **each new
   package gets its own coverage ratchet** (the discovery-style
   `--fail-under` climb), never a global widening.

## Falsification

1. **Gate 3 decides the kernel.**  If ADR 0008's open gate — does search
   *reach* `loop`-headed forms from a corpus graph? — measures negative,
   the calculus is a presentation win, not a discovery substrate, and
   stage 2 is not justified.  Measure first.
2. **A port with no consumer is decoration.**  If `Calculus`/`Machine`/
   `Engine`/`Proposer`/`Environment` cannot name an existing consumer
   and a conformance test, they do not land.
3. **The split must pay in navigability, not just in file counts.**  If
   moving the RL layer out of discovery does not measurably reduce the
   cost of a change (import surface, test selection, review size), the
   split was cosmetic.
4. **No layer inversion.**  The import-linter contract must stay
   acyclic; a kernel that imports an execution adapter is a failed
   kernel.

## Honest scars

- **Splitting 50k lines is a test project, not a `git mv` project.**  The
  coverage floors make each extraction cost tests; budget accordingly.
- **The kernel's value is unproven** — this ADR proposes a home for it,
  not evidence that it earns one.
- **`catopt-carriers` (14k lines, torch-coupled) sits awkwardly**: it is
  carrier *laws* (semantics) with *executors* (execution) in one
  package.  Left alone here; it is the next split candidate if this one
  pays.
- **The lab is not the kernel.**  `lab/` stays the promotion queue; this
  ADR is about what promotion *is* for the calculus.

## Consequences

- **The thesis becomes structural.**  "Programs are 1-cells, laws are
  2-cells, derivations are 3-cells" stops being prose and becomes a
  package with a port, an interpreter, and a rewrite machine.
- **Discovery gains a uniform enumeration space** — morphism shapes
  rather than op-patterns (ADR 0008's intent), which is what its gate 3
  is waiting on.
- **Execution adapters multiply cleanly.**  With `genkernel` beside
  torch/onnx/cuda, "a new backend implements `Sink`" extends to "a new
  *minting* backend implements `Machine`".
- **Learning becomes optional, not embedded.**  The RL layer can be
  dropped or replaced without touching the pipeline it steers.
