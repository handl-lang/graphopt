# ADR 0009 — the kernel calculus as a package, and the layer ladder

**Status**: proposed — **revised** (see *Revision*, which reorders the
stages and fixes the naming rule).  **Gated on ADR 0010**: the
interpretation model ("one language, many interpretations") is a
prerequisite — there is no bespoke VM, and the `Machine` port below is
superseded by 0010's `Calculus` + per-interpretation entry points.
**Date**: 2026-10-14.
**Depends on**: ADR 0002 (categorical re-expression), ADR 0003 (four
dimensions — *and* its lesson that a port with no consumer is
decoration), ADR 0004 (abstraction as move), ADR 0005 (the open game),
ADR 0007 (semantic-atom minting), ADR 0008 (combinator
presentation), **ADR 0010 (one language, many interpretations — the
prerequisite)**.
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

### What the language is for — a prerequisite, not an engine

State this plainly, because it bounds the claim.  A semantic
categorical language buys exactly three things:

1. **Uniformity** — laws become *equations over morphisms* (derivable
   from structure: monoid → associativity, comonoid → duplication,
   trace → fold), so the space is generatable rather than hand-listed;
   search becomes enumeration over one space.
2. **First-class sharing and iteration** — `dup` (Δ) and `loop`
   (trace) make recurrence forms *expressible* instead of hand-minted.
   This is what makes **algorithmic** (fold/scan) forms reachable, as
   opposed to algebraic identities.
3. **Soundness** — a discovery is an equation carrying a replayable
   certificate, so "found" and "correct" stay separable measurements.

Measured, that yields: **law-level discovery** (a corpus-minted law,
guard synthesized, adversarially swept, shipping in `DEFAULT`; 11
machine-admitted laws; the axiom/lemma split derived) and **equational
derivation** (derivations are data, `verify_certificate` replays them,
`lemma_cert` replays lemmas, the sweep falsifies).  It does **not**
yield: a smaller search space (a universal language enumerates *more* —
proposers, exposure screening and RL remain the bottleneck), new
*semantics* without a constructor, general (non-equational) proof
synthesis, or **algorithm-level novelty** — the largest result to date
(the RetNet carrier) is a **rediscovery** of a published duality, and
the measured yield of fresh paying structure is ~1.5% of the exposed
corpus.

## Decision

**Adopt a two-axis package organization, give the
calculus/search/proposal/environment axes ports in core, and promote
the calculus to a package only when its own gate says it earns one.**

### The naming rule (the two axes)

| axis | rule | packages |
|---|---|---|
| **dimension** | dimension-named | `catopt-core` (primitives + ports) · `catopt-<calculus>` (the kernel/laws) · `catopt-search` · `catopt-eval` · `catopt-orchestrator` · `catopt-discovery` · `catopt-game` |
| **domain** | `catopt-<domain>` | **`catopt-torch` · `catopt-onnx` · `catopt-cuda`** — each *is* the execution dimension for its domain |

`exec-` is the **role**, the domain name is the **package**; the domain
name wins.  `catopt-native` keeps its name: it is a *search*
implementation (`Engine`), not a domain.

**Where a thing goes:**

| if it is… | it lives in… |
|---|---|
| a law / equivalence / rewrite | the **dimension** package (semantic side) |
| how to *run* an op for a domain | the **domain** package |
| an engine for the search dimension | a **dimension** implementation (`catopt-native`) |
| a backend that mints code | a **domain** package (Triton is torch-adjacent) |

### Ports to add in core (see Revision 2 — most already exist)

Each lands **with a conformance test that the current implementation
already satisfies** — ADR 0003's lesson: an unconsumed port is
decoration.

| port | shape | consumer today (unported) |
|---|---|---|
| `Calculus` | the one object language: term/rule structure, shapes, transforms | ADR 0010; `Rewrite`/`derivation`/`typing` today — **interpretations** (eval/types/rewrite/lower/cost) read it; there is no separate `Machine` |
| `Engine` | `saturate(cells, budget) -> Graph` | `catopt-native` (already `engine=` opt-in) |
| `Proposer` | `propose(graph, budget) -> iterable[Candidate]` | `enumerate`/`forms`/`propose` |
| `Environment` + `Reward` | `reset/step/observe`; `reward(state, action)` | `engine.Board`, `arena`, `meta_arena`, `open_game` |
| `Prover` / `Bound` | per-artifact verification boundaries | `verify_certificate`; bounded members |

### Revision (2026-10-14) — what the first draft got wrong

Writing the first draft made two weaknesses visible, and one of them
reorders the whole plan.

1. **The deeper inversion is `catopt-carriers`, not `catopt-discovery`.**
   14k lines hold carrier **laws** (semantics) *and* executors
   (execution) inside a **torch-coupled** package — i.e. semantics
   depend on an execution library, which is the opposite of the thesis.
   The equational part belongs on the semantic side; the executors
   belong in the domain package.  **This is the first split**, because
   it is the one that makes "semantic categorical language" true rather
   than aspirational.
2. **A peer kernel package hides a decision instead of making it.**
   Two representations (op-graph IR + arrow calculus) mean every law
   must exist in both, or one is second-class — exactly the drift ADR
   0008 warned about.  The architectural question is **which
   representation is canonical**, and that is its own decision record,
   not a package move.  Until it is answered, the kernel stays a lab
   prototype.
3. **A package for one module is over-fragmentation.**  A
   `catopt-kernels` holding only `genkernel` means a pyproject, a floor,
   an import contract and a CI surface for one module.  Fold it into a
   domain package instead.
4. **Organizational resilience is weaker than verification
   resilience.**  Minted laws now sit in `DEFAULT`: a change to the bank
   silently changes extraction for every caller (AGENTS.md has to warn
   about it), and there is no versioning or compatibility story for the
   bank or for minted atoms.  No package split addresses this; it is
   recorded here as the standing gap.

### Revision 2 (2026-10-14) — most of the "ports to add" already exist

Checking the list against the code before writing any of it (ADR 0003's
lesson applied to this ADR's own proposal):

| proposed port | measured state |
|---|---|
| `Engine` | **already exists** — `catopt_core.ports.Engine` (plan 0010 lever 3): `add_term` / `find` / `run` / `extract_best`, with the pure-Python `EGraph` as the reference and `catopt-native` as the opt-in |
| `Environment` + `Reward` | **already exist** — `catopt_discovery.engine.Board` (`observe` / `step -> (state, report)`) plus the `Report` dataclass carrying `reward` + `terminal` |
| `Calculus` | its **surface is already named** — `RuleLike` / `RuleSetLike` (laws as data) and `ShapeRule` (shape inference). What is missing is not a protocol but a **`Language` *value*** — a data structure (ADR 0010), not a port |
| `Prover` / `Bound` | `Verifier` **exists** (the semantic gate). `Bound` (the certified-approximation boundary) is genuinely absent — and its consumer (the bounded-member path) lives in the orchestrator, so it is named there or not at all |
| `Proposer` | genuinely absent — and it cannot be a real port yet: its shape would be `propose(...) -> Iterable[Candidate]`, but `Candidate` lives in `catopt_discovery.proposal`, so a core-owned *view* (the way `VerifyResult` is the core-owned view of a verify report) must exist first |

**So one of the six already existed in core, two existed elsewhere, one
is a data structure rather than a port, and one needs a core-owned shape
before it can be honest.**  The ports are therefore **tied to the stages
that create their consumers**, not added up front:

* `Proposer` — when the proposal layer's candidate shape is promoted (a
  core-owned `Candidate` view), i.e. with stage 1's *data* work, not
  before.
* `Environment` / `Reward` — when the game layer is extracted into
  `catopt-game` (stage 5): a package cannot depend on a protocol it does
  not own, and the port belongs where the drivers live.
* `Bound` — with the bounded-member path, if it needs naming at all.
* `Calculus` — not a port: a `Language` value (ADR 0010).  Its tables'
  read surfaces are the ports that already exist.

### Staged plan (revised order)

1. **Ports + conformance tests** — cheap, no behaviour change, no import
   churn.
2. **Fix the carrier inversion** — carrier *laws* to the semantic side,
   carrier *executors* into the domain package.
3. **Adopt the interpretation model** (ADR 0010) — one language, many
   engines; and **decide the canonical representation** (its own ADR) —
   op-graph IR vs arrow calculus.  *Then*, and only then:
4. **Promote the calculus** to a dimension-named package — **gated on
   ADR 0008 gate 3** (does search *reach* `loop`-headed forms?).  If the
   gate measures negative, the calculus is a presentation win and this
   stage is not justified.  **MEASURED 2026-10-16: the gate is
   negative.**  A `loop`-proposing rule over the presentation was built
   (`lab/bridge/loop_propose.py`, catopt `b7ba9ff`); it works on the
   presentation's spelled scan, but the search's winner has no regular
   spelled-scan subterm and the presentation's spelled form is not even
   in the search's root e-class — so the rule fires 0× and no
   `loop`-headed winner is reached.  **Stage 4 is therefore not
   justified by the presentation's discovery value**; it may proceed
   only on a different argument, and a *form-level* proposer (which
   abstracts the step, rather than a syntactic e-graph rewrite) is the
   piece of work that would re-open the gate.  See
   `project/retros/gate3-loop-proposing-rule.md`.  **Re-measured
  2026-10-16 (later): that form-level proposer was built
  (`lab/bridge/index_abstract.py`, catopt `e4a193a`) — a parametric
  abstraction binding the index *and* the recurrence state as
  parameters.  It collapses a *regular* fold to one shared step and
  the closure reaches a `Loop` (control PASS), but the RetNet winner
  still does not collapse: its 7 blocks span two chunk sizes and
  extraction mixed two state spellings, so no constant-parameter
  abstraction can unite them.  Stage 4 is still not justified, and
  the missing piece is larger than this bullet implied.  See
  `project/retros/gate3-parametric-abstraction.md`.  **Re-measured
  2026-10-16 (later still): still not justified.**  The block width was
  abstracted (`514888c`) and the winner is still 7 distinct templates
  *even modulo the width and the index* — the residual is the state
  spelling (the cross term is the unfolded geometric sum), not a
  parameter.  See `project/retros/gate3-sizes-abstraction.md`.**
5. **Extract `catopt-game`** — the RL layer out of the 50k-line package
   (the clear navigability win).
6. **Fold `genkernel` into a domain package** — no `catopt-kernels`.
7. Each stage updates the import-linter layer contract; **each new
   package gets its own coverage ratchet** (the discovery-style
   `--fail-under` climb), never a global widening.

## Falsification

1. **Gate 3 decides the calculus.**  If search cannot reach
   `loop`-headed forms from a corpus graph, stage 4 is not justified.
   Measure before spending the coverage budget.  **Measured 2026-10-16:
   negative** — the proposing rule fires 0× on the RetNet export
   (the winner has no regular spelled-scan subterm; the presentation's
   spelled form is in a different e-class).  Stage 4 is not justified by
   the discovery value of the presentation.  See ADR 0008's gate-3 FAIL
   bullet and `project/retros/gate3-loop-proposing-rule.md`.
   **Re-measured 2026-10-16 (later): the form-level proposer was
   built and the gate is *still* negative** — the parametric
   abstraction collapses a regular fold but not the RetNet winner
   (two chunk sizes, two state spellings).  See ADR 0008's second
   gate-3 FAIL bullet and
   `project/retros/gate3-parametric-abstraction.md`.
   **Re-measured again 2026-10-16 (later still): still negative, and one
   candidate fix is retired.**  The "two state spellings" cause was
   measured to be a cost *tie* broken by op name, and a principled
   sharing-aware extraction recovers **1.01 %** of the winner's FLOPs
   (455,752 → **451,128**; ~~5.06 % / 432,696~~ was a hash-order artifact,
   corrected 2026-10-15 by `abf6114`; numerically equivalent to 2.2e-16)
   while leaving the gate FAIL.  The mix is a cascade of per-class ties,
   and the small-chunk family's unfactored spelling is a genuine cost
   optimum — so the spelling cannot be normalised by a tie-break.  Stage
   4 is still not justified by the presentation's discovery value.  See
   ADR 0008's third gate-3 FAIL bullet and
   `project/retros/gate3-extraction-state-spelling.md`.
   **Re-measured once more 2026-10-16: still negative, and the "two
   chunk widths" cause is retired.**  The block width (`sizes`) was
   abstracted in the move (ADR 0008's fourth gate-3 FAIL bullet, catopt
   `514888c`) and the winner's 7 blocks are still **7 distinct
   templates** even modulo the width *and* the index — the difference is
   structural (the cross term is the unfolded geometric sum), not a
   parameter.  Stage 4 is still not justified, and the remaining piece is
   the state spelling alone.  See
   `project/retros/gate3-sizes-abstraction.md`.
2. **A port with no consumer is decoration.**  If `Calculus`/`Machine`/
   `Engine`/`Proposer`/`Environment` cannot name an existing consumer and
   a conformance test, they do not land.
3. **The split must pay in *change surface*, not file counts.**  For a
   representative change (add a law; add a backend; retrain a prior),
   count the packages it touches before and after.  If that number does
   not fall, the split was cosmetic.  This is the **acceptance test**,
   not a footnote.
4. **No layer inversion.**  The import-linter contract stays acyclic; a
   calculus package that imports a domain package is a failed calculus.

## Honest scars

- **Splitting 14k–50k lines is a test project, not a `git mv` project.**
  The coverage floors make each extraction cost tests; budget
  accordingly.
- **The calculus's discovery value is unproven** — this ADR proposes a
  home for it, not evidence that it earns one.
- **The language is a prerequisite, not an engine** (above): it makes
  discoveries expressible, composable and checkable; it does not shrink
  the search space, invent semantics, or prove general theorems.
- **The bank has no versioning story** — minted laws in `DEFAULT` are
  load-bearing for every caller.

## Consequences

- **The thesis becomes structural.**  "Programs are 1-cells, laws are
  2-cells, derivations are 3-cells" stops being prose and becomes a
  package with a port, an interpreter and a rewrite machine.
- **Discovery gains a uniform enumeration space** — morphism shapes
  rather than op-patterns (ADR 0008's intent), which is what gate 3 is
  waiting on.
- **Execution adapters multiply cleanly.**  The domain axis stays
  `catopt-<domain>`; a new backend implements `Sink`, and a
  code-minting backend is just another domain package.
- **Learning becomes optional, not embedded.**  The RL layer can be
  dropped or replaced without touching the pipeline it steers.
