---
title: Combinator presentation of the IR (monoidal + trace)
type: adr
date: 2026-10-14
status: proposed
---

# ADR 0008 — Combinator presentation of the IR (monoidal + trace)

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
- **Gate 3 — MEASURED 2026-10-15: PARTIAL, and the partial is the
  useful part.**  Ran the protocol (`--gate2 --extracted`; staged arm
  T=64 K=4, 15 s, 9,734 enodes):

  ```
  extracted term: dag=139  reads=2 chained=True
  projects: round-trip EQUAL  morph=1590 dup=92 copies=184
  fold-chain depth: spelled=3  extract=2
  ```

  Read against the criterion: the search **does** reach the algorithmic
  content — a *chained* shared-state fold (`chained=True`, depth 2
  against the spelled 3) — and that term **projects through the
  presentation EQUAL**.  What it does **not** produce is a
  `loop`-**headed** term, and the probe says why in its own words:

  > the DAG bridge never mints `loop` — it is a *proposed* form, not a
  > recovered one.

  So gate 3 splits, and the split is actionable:

  * **as an expression** — PASS: the calculus spells recurrences that
    lower and verify (gate 2), and the search's winner round-trips
    through it.
  * **as a search space** — NOT YET: a presentation that can only
    *project* an extracted term cannot be reached *by* search.  Closing
    gate 3 needs a **proposing rule** — a rewrite whose right-hand side
    introduces `loop` — so the question becomes whether saturation
    *selects* it, which is a different (and now well-posed) experiment.

  **What this means for ADR 0009 stage 4**: a `catopt-kernel` is
  justified as the home of *rules over combinators* (the proposing rule
  is exactly such a rule), not as a projection target.  The stage-4
  framing should say that, and this measurement is why.  **(Measured
  2026-10-16 — the proposing rule was built; it works, but the search
  does not reach it.  See the FAIL bullet immediately below, which
  supersedes this paragraph's optimistic reading of stage 4.)**
- **Gate 3 — MEASURED 2026-10-16: FAIL.  The proposing rule works; the
  search does not reach it.**  Built the rule the PARTIAL bullet above
  asked for — `lab/bridge/loop_propose.py` (commit `b7ba9ff`): a
  rewrite `unroll(step, n) → Loop(step, n)` over the presentation, plus
  the smallest search that can carry it (a term-level rewrite closure
  with cost extraction).  `unroll(step, n)` is the presentation's
  *spelled* bounded scan — one shared `step`, explicit perms — and the
  tests prove it denotes the same morphism as `Loop(step, n)`
  (`comb_to_dag(unroll(step,n)) == comb_to_dag(Loop(step,n))`).  The
  gate-3 command is unchanged, now with the rule armed:

  ```sh
  .venv/bin/python lab/bridge/combinator_probe.py --gate2 --extracted --propose-loop
  ```

  ```
  extracted term: dag=139  reads=2 chained=True
  projects: round-trip EQUAL  morph=1590 dup=92 copies=184
  fold-chain depth: spelled=3  extract=2
  the presentation's spelled scan is in the search's root e-class: False
  -- gate 3: the `loop`-proposing rule --
     loop-propose: winner morph=1590 spelled-scan subterms (n>=2) found=0 ns=[]
        saturation: terms=1 fired={'loop_propose': 0, 'loop_expand': 0}
          loop-headed reachable=False  best_is_loop=False  best_cost=1590
     control (authored spelled scan): spelled-scan subterms=1 ns=[4]
        fired={'loop_propose': 1, 'loop_expand': 0}  loop-headed reachable=True
        best_is_loop=True  best_cost=69 (spelled 285)
  ```

  Two measurements, both negative for the gate:

  1. **The rule is well-formed.**  On the *authored* spelled scan
     (`unroll(retnet_step, 4)`) it fires once and the closure's
     cheapest term **is** `Loop`-headed (cost 69 against the spelled
     285).  That is the control — it proves the rewrite itself works,
     not that the search reaches it.
  2. **The search never reaches it.**  The winner's projection is a
     flat DAG compilation (`morph=1590`) with **zero**
     `unroll(step, n≥2)` subterms, so the rule fires 0× and the search
     stays un-`loop`-ed.  Independently, the presentation's spelled
     scan lands in a **different e-class** from the root (`False`) —
     the saturation never derives it, so no rule keyed on that
     spelling could ever fire.

  **Why** (measured, and deeper than "the rule was too narrow"): the
  export's unrolled recurrence is *parametrically* regular but not
  *syntactically* regular.  Each block's step differs by the block
  index (`split(…, index=i)`) and the reassociation re-spells the decay
  rate, so no fixed syntactic pattern — and in particular no
  presentation `unroll(step, n)`, whose whole point is a *single
  shared* `step` — matches it.  Reaching the loop therefore needs
  *abstraction over the step* (a form-level / parametric move), not an
  e-graph rewrite.  The presentation makes the loop **expressible**
  (gate 2, PASS) but does not by itself make it **findable**.

  **What this means for ADR 0009 stage 4**: the gate measured
  **negative**, so on this evidence a `catopt-kernel` is **not**
  justified as a discovery substrate — the presentation is a
  *presentation*, not a *search space*.  Stage 4 now rests on something
  else or on nothing; this is exactly the outcome ADR 0009's own
  falsification clause asked to be measured and written down ("if
  search cannot reach `loop`-headed forms, stage 4 is not justified").
  A real combinator search — a form-level proposer that abstracts the
  step — is the missing piece, and it is a *different and larger*
  piece of work than "promote the prototype to a package".
- **Gate 3 — MEASURED 2026-10-16 (later): still FAIL — the
  parametric-abstraction move was built, and it names a *deeper* cause.**
  The FAIL bullet above named the missing piece ("abstraction over the
  step"); this is that move, built and measured —
  `lab/bridge/index_abstract.py` (commit `e4a193a`), wired as
  `--abstract-index`.  It binds the per-block index (and the recurrence
  state) as **parameters** (a parameter is a wire — no new node kind),
  collapses the `K` blocks to one shared `step` when the family is
  regular, and emits `unroll(step, K)` for the proposing rule to fold.
  The gate-3 command is unchanged, both arms armed:

  ```sh
  .venv/bin/python lab/bridge/combinator_probe.py \
      --gate2 --extracted --propose-loop --abstract-index
  ```

  ```
  -- gate 3: parametric abstraction over the step --
     abstract-index: blocks=7 split-families=2
        family sizes/dim=('(16, 16, 16, 16)', -2) indices=[0, 1, 2, 3]
        family sizes/dim=('(4, 4, 4, 4, 16, 16, 16)', -2) indices=[0, 1, 2, 3]
        verdict=IRREGULAR — the 7 blocks do NOT collapse to one shared step
        first difference: block1/block2/add[1]/mul[1]: op om_decayed vs add
     control (regular fold): blocks=4 verdict=regular round-trip=True  closure: best_is_loop=True best_cost=109
  ```

  The control is the *presentation's own spelled fold*, expanded (the
  gate-2 program): the move collapses it to one shared step, round-trips,
  and the closure reaches a `Loop` (`best_is_loop=True`) — measured at
  `K = 2, 4, 8`.  So the move **works**; the winner still does not
  collapse.  **The index was not the whole story**: the winner's output is
  a `concat` of **7 blocks of two sizes** (4×size-4 + 3×size-16, from two
  `split` families), and extraction **mixed two state spellings** — the
  shared-state read `matmul(q, S)` and the intra re-expansion
  `om_decayed(matmul(q, k_prevᵀ), D, v_prev)`.  Canonicalising every block
  modulo the varying integers (`index`, `t`, `offset`, `sign`) leaves
  **7 distinct** blocks, so no constant-parameter abstraction can collapse
  them.  Removing the index is *necessary but not sufficient*; the
  remaining gap is the state spelling and the chunking.

  **What this means for ADR 0009 stage 4**: unchanged and strengthened —
  the gate is negative and the missing piece is *larger* than the previous
  bullet implied (a move that normalises the state spelling *and* the
  chunking, not just a form-level proposer).  See
  `project/retros/gate3-parametric-abstraction.md`.
- **Gate 3 — MEASURED 2026-10-16 (later still): still FAIL — cause 2
  (the mixed state spelling) is an extraction *tie*, and the fix is real
  but does not close the gate.**  The parametric-abstraction bullet
  above named "extraction mixed two state spellings" without measuring
  it; this is that measurement plus the principled fix —
  `lab/probes/gate3_state_spelling_probe.py` and
  `lab/bridge/sharing_extract.py` (commit `aa56e55`), wired as
  `--sharing-extract`.  Same staged arm (T=64 K=4, ~~9733~~ **9,784**
  enodes, `PYTHONHASHSEED=0`); the winner is unchanged (`dag=139`,
  **455,752 FLOPs**, `morph=1590`, round-trip EQUAL) and its read
  sequence is `reexp|reexp|fold|fold|reexp|shared|shared`.

  **The mechanism, measured**: 5 of the 7 state-read e-classes have
  ≥ 2 members at the class minimum (2-way in blocks 0–3, a **4-way tie
  at 232,000** in block 4), so `extract_best`'s tie-break — `nops`, then
  member sort order `(op, children, repr(attrs))`, i.e. **op name** —
  decides the spelling, arbitrarily with respect to *sharing*.  Forcing
  the factored `matmul(q, S)` read per class moves the whole term by
  `Δ ∈ {−528, 0, +16, +16, +144}` FLOPs: the model is **indifferent**
  (±0.03 %), and in block 4 it *strictly prefers* the factored read
  (which shares its state with block 5's read) — a non-local saving the
  greedy pass cannot see.  So the greedy extraction is **suboptimal**:

  ```
  greedy           cost 455,752   dag=139   morph=1590   round-trip EQUAL
  sharing-aware    cost 451,128   (was 432,696)         round-trip EQUAL
                   Δ = -4,624 (-1.01 %, was -23,056 / -5.06 %),
                   max|greedy - fix| = 2.220e-16
  ```

  **The fix** — `extract_sharing_aware`: a bounded, *monotone*
  coordinate descent (force one class at a time; keep only strict
  improvements to the whole term's `dag_cost`) — removes the arbitrary
  choice and adds the real sharing signal.  It is numerically verified
  equivalent to machine epsilon, and it **still leaves the gate FAIL**:
  the abstraction's verdict is `IRREGULAR` with the **same first
  difference** (`block1/block2/add[1]/mul[1]: op om_decayed vs add`).
  The mix is a *cascade* of per-class ties, and the small-chunk family
  (`C=4 < d=16`) genuinely prefers the unfactored spelling
  (`C²d < Cd²`) — so forcing it would be manufacturing a result, not a
  fix.  The remaining causes are the **two chunk widths** (the
  abstraction abstracts `index`/`t`/`offset`/`sign` but not `sizes`) and
  the abstraction's own structural state identification (per family it
  binds `state_in = None|unsqueeze|unsqueeze|unsqueeze` for the small
  family and `None|matmul|mul` for the large one).

  **What this means for ADR 0009 stage 4**: unchanged.  The gate is
  still negative, and this task *retires one candidate fix*: the state
  spelling cannot be normalised by a principled tie-break, because part
  of the mix is a genuine cost optimum.  See
  `project/retros/gate3-extraction-state-spelling.md`.  (**Correction
  2026-10-15**, `abf6114`: the recovery above is **451,128 (−1.01 %, 3
  overrides)**, not 432,696 (−5.06 %, 5 overrides) — 432,696 was a
  hash-order artifact of the pre-fix scan, and the read sequence is
  **unchanged** from greedy.  The *caveat* this bullet used to carry —
  the staged saturation was not reproducible across processes
  (9733/9752/9756 enodes) — is **resolved**: `abf6114` made the
  saturation and the sharing-aware optimum deterministic (canonical
  member order), reproducible in every process under `PYTHONHASHSEED`
  0–6 + unset.  Numbers reported before that fix may be seeded/arbitrary.
  The defect is real but **1.01 %**, a much weaker case for promotion.)
- **Gate 3 — MEASURED 2026-10-16 (later still again): still FAIL — the
  block width (`sizes`) is abstracted, and it is *not* the binding
  cause.**  The previous bullet named "the two chunk widths" first; this
  task built that abstraction — `lab/bridge/index_abstract.py` (commit
  `514888c`), the move now wires the previously-**dead** `INDEX_ATTRS`
  in and adds `WIDTH_ATTRS = ("sizes",)` as `PARAM_ATTRS`, with
  `templates_agree_mod` / `param_families` answering the retro's
  question: are the `K` blocks the same step once the block **index** and
  the block **width** are parameters?  (A block's *own* split already
  carried its width on the input wire; the width of a node that
  *survives* into a template — an inherited read, a `decay_vec` — is now
  abstracted by erasure, so it can be *recognised* but never *folded*:
  the IR is fixed-shape and has no width parameter.)  Gate-3 command
  unchanged:

  ```sh
  .venv/bin/python lab/bridge/combinator_probe.py \
      --gate2 --extracted --propose-loop --abstract-index
  ```

  ```
  verdict=IRREGULAR — the 7 blocks do NOT collapse to one shared step
  first difference: block1/block2/add[1]/mul[1]: op om_decayed vs add
  ... and they are NOT the same step even modulo the block width and
  index — the difference is structural, not a width/index parameter
  first difference mod width/index: block1/block2/add[1]/mul[1]: op om_decayed vs add
  ```

  The measurement: abstracting `sizes` (and `index`/`t`/`offset`/`sign`)
  leaves **7 distinct** templates — on the greedy winner and on the
  sharing-aware one alike — so the width was **never** the binding
  constraint.  The blocks differ *structurally*, at the cross position,
  in the **arity of the unfolded geometric sum**: block 1's cross is one
  `om_decayed`, block 2's is an `add` of two.  The remaining cause is the
  **state spelling** (the cross sum is unfolded and q-dependent, so there
  is no shared state subterm to bind), and the state-identification
  limitation is **not fixable in the move** — the strongest relaxation
  (allow any shared candidate) leaves the verdict and the first
  difference byte-identical.  Cost/dag/morph are unchanged
  (455,752 / 139 / 1590 and **451,128** (~~432,696~~) / 133 / 1509,
  round-trip EQUAL; the sharing arm's cost is corrected by `abf6114`,
  and its `dag`/`morph` were not re-reported).  **Stage 4 stays
  unjustified.**  See `project/retros/gate3-sizes-abstraction.md`.
  (Caveats: `PYTHONHASHSEED=0` throughout; the **determinism fix**
  (`abf6114`, a *concurrent* `catopt_core/egraph` edit) moved the staged
  arm's enode count 9733 → 9784 during this task — verified not to be
  this change, which never touches the saturation.)
- **Gate 3 — the protocol** (added 2026-10-14; it is
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
