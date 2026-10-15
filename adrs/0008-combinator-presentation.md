# ADR 0008 — Combinator presentation of the IR (monoidal + trace)

**Status**: proposed.
**Date**: 2026-10-14.
**Depends on**: ADR 0004 (abstraction as move — minted atoms become
*definable* under this presentation), ADR 0005 (the open game),
ADR 0007 (semantic-atom minting — the runtime-assembly loop runs over
combinator terms).
**Related**: `pack` (already the proto-monoidal product), the carrier
atoms (`om_decayed`, `applyd`, `decay_vec` — hand-minted; this ADR
makes them definable), the sharing-ambiguity scars
(DAG-vs-tree semantics under e-graph congruence).
Literature basis: the categorical calculus for neural architectures
(Gavranović, Lessard, McCourt, de Jong van Lier, Sanada et al.,
"Categorical Deep Learning") — architectures as morphisms of a
monoidal category, parameters via `Param`, gradients via lenses —
plus the arrow/``loop`` line (Hughes) for first-class recurrence.

## Context

The recurring user question this week — *"why is X hardcoded?"* — has
a single root: **the object language is an op-graph, so anything that
isn't an op-node needs a hand-minted atom.**  Carrier atoms
(`om_decayed`, `applyd`, `decay_vec`, `aff_diag`) are authored
primitives; multi-output needed a bespoke `pack`; sharing is implicit
in the DAG and has bitten us (the inf-norm sharing divergence);
weight-space rewrites need a new law family because parameters are
leaves, not morphisms.  ADR-0007 retires part of this by minting
atoms mechanically — but minted atoms are still *atoms*: the object
language has no combinator structure of its own.

The categorical presentation offers the minimal universal grammar:
objects are types, morphisms are terms, and a small combinator basis
— composition `>>>`, monoidal product `⊗`, structural morphisms
(`id`, swap, assoc, unitors), diagonal `Δ` for sharing — presents
every computation graph.  Two extensions cover the neural specifics:

- **`Param` morphisms** (CDL): a weight is a first-class component of
  the morphism, not an input leaf — conv+BN folding becomes an
  ordinary equation between morphism expressions, no new law family.
- **`loop`/`trace`** (Hughes): recurrence is first-class — a scan is
  `loop (arr step)`, and the carrier atoms become *spellings* of the
  trace construction rather than primitives.

## Decision

**Add a combinator presentation layer over the IR — parallel, not a
replacement.**  The op-graph IR stays the lowering surface (sinks,
executors, `pack`); the combinator presentation is the
*discovery and semantics* surface.

### The presentation

- Terms are morphisms of a symmetric monoidal category with Δ
  (sharing is a morphism, not a pointer convention) plus `loop`
  (typed trace — state shape fixed).
- A bridge, both directions: `dag → combinator` (fan-out becomes
  explicit `Δ`; `pack` becomes `⊗`) and `combinator → dag` (lower
  via expansion of structure morphisms).
- Laws remain equations — now between combinator expressions.
  The same admission gauntlet applies (ADR 0004/0007): oracle →
  typed gate → certificate → adversarial sweep → measured pay.

### What becomes definable (the payoff)

- **Carrier atoms as `loop` instances.**  `applyd`/`om_decayed`/
  `decay_vec` cease to be primitives — the RetNet carrier is a
  spelled trace term.  The authored-atom problem dissolves.
- **Weight-space laws as ordinary rewrites.**  Params as morphisms
  → conv+BN folding, reparameterization, linear-chain folding are
  equations in the same object language as everything else.
- **Sharing as structure.**  `Δ` makes fan-out explicit — the
  DAG-vs-tree ambiguity class of bugs is impossible by construction.
- **A uniform discovery space.**  Proposers enumerate morphism
  shapes of the free category — naturality/factorization families
  stop being separately-coded generators.

## What this does NOT buy — stated so we don't overclaim

- **Search gets uniform, not smaller.**  A universal substrate
  enumerates an enormous space; proposer/guidance machinery remains
  the bottleneck.  The menus stop being authored *op-patterns* but
  guidance (which morphism families to enumerate, budget policy,
  measured need) is still a designed component.
- **`loop` has real typing subtleties.**  Trace requires a fixed
  state shape; dynamically-shaped recurrence is out of scope
  initially (constant shape per term is the checkable contract).
- **Verification machinery is unchanged.**  The oracle/cert/sweep
  gauntlet is representation-agnostic; the presentation changes what
  terms look like, not what "sound" means.

## Falsification path

The direction fails, measurably, if:

1. **The bridge loses structure.**  `dag → combinator → dag` does
   not round-trip on the corpus (survivors verified end-to-end),
   or the Δ-expansion blows term size past the enode budget on
   representative exports.
2. **Nothing authored becomes definable.**  If `applyd`/`om_decayed`
   cannot be respelled as trace terms within the basis (measured:
   a corpus workload saturates to the carrier *through* `loop`
   terms), the claimed payoff is empty.
3. **The combinator space is unusably large for discovery.**  If
   enumerating morphism shapes to depth ~6 yields nothing firing
   that the op-pattern enumerators already find cheaper, uniformity
   bought us elegance, not reach.
4. **Two-way drift.**  If maintaining the dual presentation forces
   semantic fixes to land twice or diverge, the parallel layer is
   worse than the problem.

## Honest scars carried forward

- **Authoring moves, not disappears.**  The combinator basis itself,
  the Δ/`loop` law sets, and the bridge are authored — same
  bootstrapping disclaimer as the law bank.  The claim is that the
  authored surface gets *smaller and more uniform*, not that it
  vanishes.
- **An authored thing that retires must retire measured.**  Carrier
  atoms remain until a corpus workload reaches the carrier via
  `loop` terms end-to-end — parity-or-better, verified.  Until then
  both spellings coexist (provenance-marked).
- **The sharing fix is the strongest immediate argument.**  If `Δ`
  only eliminates one historical bug class and nothing else follows,
  that alone may not justify the layer.
- **Representation upgrades have a graveyard.**  The prior exposure
  screens keep paying because they *measured*; this ADR's value is
  bounded by its falsification checks running, not by the elegance
  of the substrate.

## Consequences

- **The authored atom set shrinks to the combinator basis** — the
  smallest hand-written surface the system can have while remaining
  an object language at all.
- **Discovery targets morphisms, not op-tuples** — proposer
  grammars become "which combinator shapes," strictly more uniform.
- **A runtime is a diagram family.**  The ADR-0007 emergent-runtime
  loop mints combinator terms; a generated package = {basis
  extensions, laws, verified handlers} all in one representation.
- **Both surfaces coexist deliberately** — the op-graph stays for
  lowering (sinks already speak it); the combinator layer is where
  semantics, laws, and discovery live.  The bridge is the seam.

## Measurements (added post-landing)

Bridge prototype + falsification gates measured (`841b99c`, `481eec7`,
merged `32962d0`; `lab/bridge/combinator.py` + `lab/bridge/combinator_probe.py`).

- **Gate 1 — corpus round-trip: PASS.**  321/321 exported workloads
  (zoo + intake + model registries) round-trip *structurally exact*
  (hash-consed identity); the 2 rejections are upstream
  `torch.export` failures, not bridge gaps.  Mean blowup 2.61×
  (max 7.27× on a sharing-heavy case); largest morphism 1.1% of the
  enode budget.  **Zero ops refuse translation.**  10 sampled
  survivors lower through `TorchSink` bit-identical.
- **Gate 2 — carrier as a `loop` term: PASS.**  The shared-state fold
  spells as `seq(pre, loop(step, K), post)`, lowered and verified
  against `zoo._ChunkedRetention` at 1e-7–2.6e-7 (the documented
  γ⁻ᶜ conditioning bound).  Named extension: **bounded scan**
  `loop_n` — a bare trace `Tr^S` is insufficient; n-fold iteration
  over block inputs is the required combinator.  Bonus: the
  staged-saturation extract projects through the bridge round-trip
  EQUAL (1590-node morphism, 92 explicit Δ sites) — same recurrence
  modulo spelling.
- **Gate 3 — open, and now with a protocol** (added 2026-10-14; it is
  the last gate standing between this ADR and ADR 0009's stage 4, so
  "open" without a way to close it was the real problem).
  `loop` structure is erased at export: fused recurrences
  (`lstm.input`, `cumsum`) translate fine but the fold must be
  *proposed by search*, never recovered by the bridge.  Whether search
  reaches `loop` forms is the remaining substrate risk.

  **The protocol** (one command, one criterion):

  ```sh
  .venv/bin/python lab/bridge/combinator_probe.py --gate2 --extracted
  ```

  This runs one bounded staged saturation (`om_shared_state_stages`)
  over the exported RetNet graph and projects the extracted winner
  through the bridge.  Read the result as:

  * **PASS** — the extracted winner is `loop`-headed (or contains a
    `loop` at the shared-state position), and it round-trips through the
    bridge equal.  Then the presentation is a *discovery substrate*: the
    search reached a form the export erased, and ADR 0009's stage 4 is
    justified on this evidence.
  * **FAIL** — the winner is the spelled (unrolled) form again.  Then
    search did **not** reach `loop`, and the honest reading is that the
    bridge is a *presentation* (a way to say recurrences) rather than a
    *search space*, and stage 4 should not be justified by it.

  **Why this is the right criterion**: gate 2 already proves the
  *spelling* works (a hand-written `loop` term lowers and verifies at
  1e-7).  Gate 3 is the only question a presentation cannot answer about
  itself — whether the search, not the author, gets there.  A negative
  result is therefore as valuable as a positive one, and both are
  publishable statements about the substrate.

  **Prior evidence, for calibration** (from the staged-saturation run
  recorded above): the extract did reach the carrier at t512
  (`TestRealExportFold::test_extraction_prefers_the_chunked_carrier_at_t512`)
  and the staged extract round-tripped EQUAL with 92 explicit Δ sites —
  which is *encouraging but not the gate*: those wins are
  carrier-rooted extractions, not a `loop`-headed winner at the
  shared-state position.  The protocol above is what distinguishes them.
- **Cost scars.**  Permutation bookkeeping dominates morphism mass
  (Σperm 6057 > Σdag 3519) — the flat context encoding is honest but
  wasteful; a real layer wants a better normal form.  Also: the
  shared-state probe's `_honest_reference_cost` fold is numerically
  off (missing `/√T` on cross terms, a γ^C on ξ) — valid as a FLOP
  floor (attrs don't move FLOPs), but it was never a verified
  program; the spelled `loop` is.
