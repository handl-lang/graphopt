# Catopt: discovery, verified transformation, and learned search over computation graphs

*Draft.  Every number in this paper is sourced to the retro, baseline or
artifact that produced it; where a claim is a rediscovery rather than a
discovery it is labelled as such.*

## Abstract

Catopt is a general engine for discovering better solutions by combining
mathematical reasoning, verified transformations, and learned search.  It
treats optimization as a game over a structured space of *equivalent*
solutions: a tensor program is a morphism of a category, the category's
equations are rewrite laws, equality saturation closes the space of
programs equivalent to the input, cost-based extraction picks the cheapest
member the target can run, and a replayable certificate certifies the step.
Reinforcement learning navigates the space; the formal layer fixes what is
*admissible*, which is what lets the learner explore aggressively.

We separate four questions that are usually conflated — **optimization**
(choosing a better implementation from *known* transformations),
**discovery** (finding laws and algorithms a human did not encode),
**verification** (establishing validity before acceptance), and **search**
(exploring the candidate space efficiently) — and we make three of them
measurable.  On *novelty*: the engine mints laws, handler bodies and atoms
from measured need, and the first law minted from a model's own operation
tuple now ships in the default rule set (`mul_unsq_nat`, −12.27 % on its
source workload); the yield of fresh paying structure in the exposed corpus
is, honestly, ~1.5 %.  The claim is decidable, not aspirational: *novel*
here means **machine-original against catalogued knowledge** — a candidate
qualifies iff its certificate replays standalone, its certified extraction
strictly beats a registered `bench/discovery` baseline on a derivation that
invokes it, and its instance is underivable from `known_bank()` (the shipped
rules plus the prior-art preset) minus alpha-siblings.  The sieve
(`bench/drivers/screen/novelty_sieve.py`, results in
`docs/results_novelty_sieve.md`) ran 24 candidates over the then-242-rule bank
(`known_bank()` now counts 265 — the shipped `prim:synop*` folds seated since):
the review queue is **empty** — the two earlier review items
(`mul_unsq_nat`, `softsign_fold`) demoted to derivable once their
catalogued duals were registered as prior art — six derivable,
six unpaid, twelve unmeasured.  "Novel to humanity" stays an open-world
literature claim this draft does not make.  On *soundness*: every move carries a replayable
certificate, candidates clear a numeric oracle, a typed-pay gate, an
adversarial sweep and a measured-pay gate before shipping, and the
adversarial sweep has repeatedly falsified the engine's own candidates — a
naive guard with 212 bad accepts, an apparent paying sibling with
semantically unequal sites, four deliberately-false twins.  On *value*: the
delivered wins are **structural and regime-bounded** — a weight-chain fold
at 14.92× vs `torch.compile` (Inductor keeps all k `mm`s), a spelled RetNet
export at 6.2–7.4× vs `torch.compile` — and the losses are real and kept in
the same table (a fused-op vision model at 0.80× vs compiled; a certified
negative on the precision lane; a minted atom that pays nothing).  Backends
are adapters: a second, non-torch backend (ONNX, both directions) landed
with **zero changes under `mlopt-core`**, verified through `onnxruntime` at
1.11e-16 (fp64).  The contribution is automatic discovery + certified
equivalence + per-shape selection; the performance story is narrow and the
methodology — the verifier that keeps catching would-be false claims — is as
much the result as the numbers.

## 1. The thesis

Catopt is **not** "a PyTorch optimizer."  It is:

> **Catopt discovers better solutions by combining mathematical reasoning,
> verified transformations, and learned search.**  It treats optimization as
> a game over a structured space of equivalent solutions, using reinforcement
> learning to navigate that space and formal foundations to preserve
> correctness.  Applications and backends are replaceable interfaces to this
> general engine.

Three pillars follow, and every capability in the system answers to one of
them.

1. **Mathematical discovery.**  Discover new laws, algorithms and equivalent
   representations instead of relying only on predefined transformations.
   The library is not fixed: the engine census-s a corpus, proposes
   candidates, verifies them, measures them, ranks them, and — when a
   candidate clears the admission gauntlet — ships it.  Discovery is
   *choosing the richest structural interpretation* of a graph, not applying
   laws inside a fixed one.
2. **Categorical soundness.**  The formal foundation for valid
   transformations and preserved semantics.  A program is a 1-morphism; a
   rewrite law is a 2-morphism between programs; a `derivation` — one law
   proven from premises — is a 3-morphism (a coherence).  The structure's
   axioms *generate* rewrites; the certificate *certifies* the step.  This is
   what makes the search space a space of *equivalent* programs rather than a
   space of guesses.
3. **Learned search.**  Optimization is a game over the space of *valid*
   moves.  A policy chooses which legal move to fire next; it may only
   *reorder* — every rule still runs, so the fixed point and the certificate
   are unchanged.  Because legality is fixed by the formal layer, the learner
   is free to explore aggressively.

### 1.1 Four questions, not one

The word "optimizer" hides four different questions.  We keep them apart,
because conflating them is the failure mode we most want to avoid.

| Question | What it asks | Who owns it here |
|---|---|---|
| **Optimization** | choose a better implementation from *known* transformations | cost-based extraction over the law closure |
| **Discovery** | find laws/algorithms humans did not encode | `mlopt_discovery` (census → propose → verify → measure → rank → emit) |
| **Verification** | establish validity before accepting | the admission gauntlet + certificate replay |
| **Search** | explore the candidate space efficiently | a `Policy` over an already-structured space |

The distinctions are load-bearing.  A discovery that does not pay is *data*,
not a result; an optimization that silently trades accuracy is a bug; a
search heuristic that decides equivalence is a category error.  The engine
keeps **four independent dimensions** — semantics, search, evaluation,
execution (ADR 0003) — and no layer is allowed to answer another's question:
a cost model ranks but never changes equivalence, a policy orders but never
changes what is legal, a profiler observes but never runs on the target.

### 1.2 Backends are adapters, not the core abstraction

PyTorch is *one application* of the engine.  `mlopt-core` is torch-free and
dependency-light (`import-linter` enforces that core imports no `torch`,
`numpy`, or GPU library), and a new backend implements the `Sink` port
without touching core.  This is not an assertion; it is measured twice.  The
ONNX backend — `OnnxSink` (IR → ONNX) and `OnnxSource` (ONNX → IR) — landed
with **zero edits under `mlopt-core`** (`git diff --name-only` against the
package list), and the fp64 round-trip through `onnxruntime` is 1.11e-16,
machine epsilon (ADR 0006; `project/retros/onnx-package.md`).  The boundary
is measured, not asserted.  Applications are replaceable interfaces: the same
engine runs across model architectures and, once evaluation is a port, across
hardware.

## 2. Mechanism — syntax, structure, search, certificate

The conceptual pipeline (ADR 0002; `docs/mechanism.md`) is:

```text
syntax → structure(signature + axioms) → match(signature)
       → rewrite(axioms) → saturate → extract → certify → evaluate
```

1. **Syntax.**  The graph as written — ops and wires.  Just a term, no
   meaning.  *(Example: a chain of `matmul(A, h)` + `add(x)`.)*
2. **Structure.**  A mathematical object with two halves: a **signature**
   (its operations — the *shape* of a morphism into the structure) and
   **axioms** (equations those operations obey).  *(Example: the monoid
   `aff(A, b)`.)*
3. **Match → the lift.**  Recognize that a piece of the graph has the
   signature's shape; that makes it a morphism *into* the structure.
   Matching is done on the **signature**, never on the axioms.
4. **Rewrite.**  The structure's **axioms** become rewrite rules.  Side
   conditions (`check` / `derive` / the declarative `cond` DSL) are the
   soundness boundary — the only layer that keeps a false member out of the
   e-graph.
5. **Saturate.**  Apply the axioms to a fixed point.  The e-graph *is* this
   closure — the consequences, not a search over them.  A saturated e-graph
   encodes all `Catalan(k−1)` bracketings of a k-chain (verified: 16,796
   programs in 231 e-nodes at k=11).
6. **Extract.**  Search the closure for the cheapest program the backend can
   run.  Pricing is *backend-aware* (`backend_cost(cost_fn,
   sink.supported_ops)`): a form using an op the sink cannot lower prices at
   `+inf` and is never selected.
7. **Certify.**  The path original → extracted is a replayable certificate (a
   2-cell).  `verify_certificate` re-checks derivational equivalence on real
   terms (fp64), not numerical spot-checks.
8. **Evaluate.**  Describe the extracted program's characteristics
   (torch-free `ProgramFeatures`) and, when a target is available, its
   runtime.  Evaluation **ranks** candidates; it never changes what is
   equivalent (ADR 0003).

A shortcut matters in practice: if the structure *names* the target directly,
**construct** it and skip steps 5–6.  Construction is the primary path;
saturation is the completeness fallback (ADR 0002 Rule 4).

### 2.1 Where discovery happens — the meta-layer

The loop above runs *inside* a fixed law library.  Two mechanisms operate on
the library itself.

**Coherence — the 3-cell layer.**  A rewrite is a 2-cell; a law *about*
rewrites is a 3-cell.  `mlopt_discovery.coherence` enumerates the pair layer
over the shipped library; `coherence2` measures depth 2 — stratification,
reach under removal, and the mediator table.  The measured structure is
honest: the derived layer of `ALL_RULES` is *flat* — zero laws derivable from
the primitives alone, every derivation a self-supporting cycle plus one
emergent law (`silu_mul_form`); the effective basis is 32 independent
primitives (27 derivable).  Two findings worth stating: **derivable ≠
disposable** (removing a derivable law costs real bounded reach —
`linear_to_matmul_t` is provable yet the most load-bearing derivable law
under removal), and **divergence is found and mediated** (`silu_fold`
shipped to restore confluence where `silu_expand` destroyed the `swiglu_fuse`
redex).

**Discovery — the closed loop.**  `mlopt_discovery.pipeline` runs
census → propose → verify → measure → rank → emit over a corpus of real
exported models.  Candidates are verified (numeric oracle, `sink.verify` on
lowered modules, derivability oracle), measured (fires / cost deltas / cert
replay / closure ratio), ranked, and emitted as a reviewable patch
(`--emit-admission`).  The loop audits itself: adversarial witness generation
(`gap_gen`) produces bindings the corpus cannot and has falsified
paying-but-false candidates and a shipped matmul-family hole.

**Minting — the object language grows (ADR 0007).**  The loop above admits
*laws over a fixed vocabulary*; the minting stack makes the vocabulary itself
a product of measurement, in four layers with one admission gauntlet:
definitional atoms (anti-unification — measured **null** on reach-expansion,
below), handler-interpreted atoms (a structured handler grammar, descriptors
as plain data, never `eval`'d), law-theoretic atoms (synthesize + certify a
law set from measured bindings), and induced atoms (the bounded-error lane,
fenced: an approximate atom may never appear on an exact law's RHS).

One layer sits *beneath* the syntax: the presentation the discovery machinery
could enumerate over.  Whether that presentation is a *search space* or only a
way to *say* things is §2.2.

### 2.2 The substrate beneath the syntax — a presentation, and a bounded search space (ADR 0008)

A prototype combinator presentation (`mlopt_discovery.combinator`) re-expresses
the IR as morphisms of a monoidal category with explicit `dup` (Δ, sharing as a
morphism) and `loop` (bounded trace).  Gates 1–2 pass: 321/321 corpus
workloads round-trip structurally exact (zero ops refuse translation, mean
blowup 2.61×), and the RetNet shared-state fold spells as
`seq(pre, loop(step, K), post)` and verifies at 1e-7 — the "carrier atoms
become *definable*, not primitive" claim, measured.

Gate 3 is the one question a presentation cannot answer about itself: does
*search*, not the author, reach a `loop`-headed form?  `loop` structure is
erased at export — a fused recurrence translates fine, but the fold must be
*proposed by search*, never recovered by the bridge.  The criterion (ADR
0008): PASS = a `loop`-headed winner that round-trips equal; FAIL = the
spelled/unrolled form again.  ADR 0009's stage 4 (promote the calculus to a
package) is gated on it, so a negative is as much a result as a positive would
have been.

**Four moves, four negatives.**  Each built the move the previous
negative named, and each move **worked** — on a control.  The search never had
the subterm to bind.

| attempt | move built | control (the move, on an authored fold) | the winner |
|---|---|---|---|
| 1 — proposing rule (`loop_propose.py`, `b7ba9ff`) | `unroll(step, n) → Loop(step, n)`, n≥2 | authored `unroll(step,4)`: fires 1×, cheapest reachable term `Loop`-headed (**69** vs spelled 285) | **0** `unroll(step, n≥2)` subterms → fires **0×**; spelled scan in a *different* e-class |
| 2 — parametric abstraction (`index_abstract.py`, `e4a193a`) | bind the block index and the recurrence state as parameters (a parameter is a wire) | presentation's spelled fold, expanded: collapses, round-trips, reaches a `Loop` (**109**) at K=2/4/8 | **7 blocks, 2 `split` families**, `IRREGULAR`: `block1/block2/add[1]/mul[1]: op om_decayed vs add` |
| 3 — the state spelling (`sharing_extract.py`, `aa56e55`) | sharing-aware coordinated extraction | — (a measurement) | the mix is a cost *tie*; the fix recovers **1.01 %** (~~5.06 %~~ — see the correction in §2.2) and the winner still does not collapse |
| 4 — the width abstraction (`index_abstract.py` `PARAM_ATTRS`, `514888c`) | `sizes` + `index`/`t`/`offset`/`sign` as step *parameters* (`agree_mod` — recognition only, never a verdict) | surviving-width fold recognised `agree_mod=True`; a structural fold `False` | **7 distinct templates even modulo width + index** — the difference is structural (the unfolded cross sum's *arity*), not a parameter |

The reason is one sentence, measured four ways: **the presentation's whole
advantage — a single shared `step` — is exactly what the flat DAG erased**,
and the erasure is not one thing but three.  The recurrence is *parametrically*
regular (each block's step differs by `split(…, index=i)`) but not
*syntactically* regular, so there is no repeated `step` subterm to bind.  The
chunking is not uniform (7 blocks: four of size 4 and three of size 16, from
two `split` families).  And extraction mixed the state spelling — three ways,
`reexp | reexp | fold | fold | reexp | shared | shared`.

**The fifth move reached the gate — with one named residual**
(`form_propose.py`, `9ad7c3f`).  Every earlier move looked for the shared step
in the *syntax* (it is not there) or in *parameters* (no parameter absorbs an
arity).  The move that worked asked the **saturated e-graph** instead: each
block's state-read e-class already contains a factored `matmul(q̃, M)` member —
the q-free accumulated state extraction declined to choose — and those state
classes are already linked by `add(mul(w,·), B)` update members.  The
recurrence was in the graph; only the pick was missing.  The proposer
anti-unifies the block family through those derived members into **one step
schema** `pack(S′, o)` — `S′ = w·S + (k ⊙ η)ᵀ v`,
`o = om_decayed(qkᵀ/c, D, v) + ξ ⊙ ((q/c) @ S)` — over **9 wires** (the
state, the block's own q/k/v splits, and five decay/scale constants), so
everything family-varying — index, chunk width, constants — is a wire and one
schema serves both chunk widths.  It emits `unroll(step, 3)` per state chain —
blocks 1–3 (the 4-wide family) and blocks 4–6 (the 16-wide family); block 0
is peeled — and `PROPOSE_LOOP` folds each to `Loop(step, 3)` (`loop_propose`
fired 2×; closure cost 91 each).  The assembled
`seq(pre, par(unroll, unroll, id), post)` is **e-graph certified**: every
expanded block lands in its own block e-class and `comb_to_dag(prog)` lands in
the winner's own root e-class (class 25).  The residual: the single-loop arm
carries one state across the family boundary and its per-block e-class checks
fail at 4, 5, 6 — `state-chain-broken:block3→block4` — so the certified
program holds **two loops under one step schema, not one**.

**The substrate conclusion — amended by the fifth move.**  The bridge *can*
function as a search substrate — through the e-graph, not through the syntax.
That is still a statement about *finding* versus *expressing*: gates 1–2
stand, the search's winner still *projects* through the presentation EQUAL
(`morph=1590`, 92 explicit Δ sites), and extraction still never *selects* a
`loop` member — the fold is **proposed** by anti-unification over the
saturated e-graph and **certified** by class-landing, not chosen on cost.
But the recurrences are now *found*: the step was synthesised from derived
members the saturation itself had produced and the extraction had not picked.
ADR 0009's stage-4 blocker — its own falsification clause — therefore no
longer fires; whether the promotion is now justified is a decision this
measurement re-opens rather than takes.

**The positive finding: an extraction defect — a real but *small* one.**  The
gate work forced the winner's spelling to be inspected, and the inspection
found a real defect in the *extractor*.  The state-read spelling is a cost tie
in **5 of the 7** blocks (2-way in blocks 0–3, a **4-way tie at 232,000** in
block 4), and `extract_best` breaks the tie by `nops` then member sort order —
i.e. by **op name**, arbitrarily with respect to *sharing*.  Forcing the
factored read per class moves the whole term by Δ ∈ {−528, 0, +16, +16, +144}
FLOPs (the model is indifferent to ±0.03 %), except that block 4 strictly
prefers the factored read because it shares its state with block 5's read — a
non-local saving the per-class greedy pass cannot see.  So on this workload the
greedy per-class extraction is off the cost its own model assigns: a
sharing-aware pass reaches **451,128 FLOPs against the greedy 455,752**
(Δ = −4,624; **−1.01 %**), numerically equivalent at **2.2e-16**.  The fix is
small and lives in `bench/` (`extract_sharing_aware`,
`bench/drivers/bridge/sharing_extract.py`): a bounded, *monotone* coordinate descent that
begins at `extract_best` and keeps only per-class overrides that strictly lower
the whole term's `dag_cost`.  Monotone means it can never manufacture a form
the model did not already prefer — it removes the arbitrary per-class choice
and adds the sharing signal.  It is the same non-local blindness
`extract_paired` exists for, here on the tie-break rather than on a group
decision; it is a `bench/` prototype rather than a change to
`mlopt_core.egraph.extract` because the same tie-break would change extraction
for every caller, a change the full suite must land.

> **Correction 2026-10-15** (`project/retros/saturation-determinism.md`,
> mlopt `abf6114`).  This paragraph previously read **432,696 FLOPs
> (Δ = −23,056; **−5.06 %**)**, and reported `dag` 139 → 133
> (re-derived deterministically: **139 → 135**, `morph` 1590 → 1554).  The staged
> saturation and the sharing-aware optimum were **hash-order dependent** — the
> extraction scanned `EClass.nodes`, a `set` whose iteration order follows
> `ENode`'s hash, so the local optimum moved with `PYTHONHASHSEED` — and
> **432,696 was a hash-order artifact**.  Both are now **deterministic**
> (canonical member order, `abf6114`); the reproducible reading is **451,128**
> with **3 overrides** in every process (`PYTHONHASHSEED` 0–6 + unset, all
> identical), the read sequence is **unchanged** from the greedy term
> (`reexp|reexp|fold|fold|reexp|shared|shared`), and the e-graph is **9,784**
> enodes (was ~~9,733~~).  Numbers reported before `abf6114` may be
> seeded/arbitrary.  The **defect is still real** — greedy per-class extraction
> *is* off its own cost model, and the monotone fix *does* recover it — but the
> magnitude is **1.01 %**, not 5.06 %.  That is a much weaker case for
> promoting the tie-break into `mlopt_core.egraph.extract`: at ~1 % on a
> single workload the promotion is a **judgement call**, not the obvious win
> the 5 % figure implied.  It is not a paper claim.

**What remains.**  One obstruction, named by the graph itself:
`state-chain-broken:block3→block4`.  The two families' boundary state — the
small family's terminal state and the large family's initial one — is the
same *value* (the single-loop arm carries it across the boundary and the
numeric check passes; the per-block *e-class* checks fail at 4, 5, 6) but the
staged saturation never merged the two spellings into one *class*.  Closing
it is a reach problem, not a step-schema problem — the schema already serves
both families because everything family-varying rides on wires; one
cross-family state-reindex rewrite would let a single `Loop` cover all six
read-bearing blocks.

The retros are `project/retros/gate3-loop-proposing-rule.md`,
`gate3-parametric-abstraction.md`, `gate3-extraction-state-spelling.md`,
`gate3-sizes-abstraction.md` and `gate3-form-level-proposer.md`;
the caveat they carried — that the staged saturation was not reproducible
across processes (`PYTHONHASHSEED` changed the e-graph, 9,733/9,752/9,756
enodes, and the sharing-aware local optimum) — is now **resolved**.  mlopt
`abf6114` made the saturation and the extraction **deterministic** (canonical
member order), and the reproducible runs (`PYTHONHASHSEED` 0–6 + unset) are all
identical at **9,784** enodes and **451,128** FLOPs.  Numbers reported before
that fix may be seeded/arbitrary.

## 3. Verification boundaries are per-artifact

A single correctness claim is not enough, because soundness of an individual
transformation does not establish that a *discovered law*, the *search
procedure*, or the *final delivered result* is correct.  Each needs its own
boundary.  Catopt's boundaries are:

| Artifact | Boundary | What it rules out |
|---|---|---|
| a **law** | numeric oracle + typed-pay gate + certificate replay + adversarial sweep of the guard's false region | a law true on observed sites but false off-corpus |
| the **search** procedure | a policy may only *reorder*; the fixed point and the certificate are unchanged | a learner that changes what is legal or certified |
| a **delivery** | the lowered module is numerically verified against the source | a form the backend cannot actually run, or runs differently |
| a **bound** | a certified error budget (`error_budget=`) with output-propagated verification | a silent approximation |

The **law** boundary is the one that earns its keep.  A candidate must clear
the ADR-0004 admission gauntlet — reconstruct, full-data, measure, truth,
novelty, typed-pay, closure, cert — *and* the adversarial sweep must probe
the guard's false region, not just its true region.  The standing lesson
(ADR 0004, from the `select_mul` unsoundness): "verified on all observed
sites" is **not** "sound."

The **search** boundary is structural, not statistical.  Because the
certificate is a hard filter (accept/reject), never a soft reward, a policy
cannot launder a wrong answer into a reward.  A profiler observes, a policy
orders, a model ranks — none of them decides equivalence.

The **delivery** boundary is the last line.  `sink.verify` compares the
lowered module against the source numerically (fp64 where stated), so an
extracted wrong member fails and the optimization declines.  But it protects
the *output*, not the *search*: an unsound member sat in the e-class as an
equal citizen could still win extraction and contaminate certificates that
replay its edge.  This is why the fix for an unsound law belongs in the law
(its `check`/`cond`), not in `verify` — and why `verify_certificate`
replaying a derivation is *not* the same as the derivation being *true*
(`project/retros/matmul-unsound-fix.md`).

## 4. The three questions, answered with measured evidence

### 4.1 Novelty — can it discover useful laws/algorithms not already in its library?

**Yes, once, and the yield is measurably sparse.**  The strongest single
result is `mul_unsq_nat`, the first law minted from a *corpus op-tuple*
rather than a hand-declared need.  The census saw `mul(unsqueeze, unsqueeze)`
in a DeltaNet-style block's own `β_t · k_t` write; the candidate
`mul(unsq U d, unsq V d) = unsq(mul U V, d)` was proposed; it is `relation=new`
(no shipped law carries this two-sided naturality) and **not derivable** from
`ALL_RULES`.  It fires 4× (all typed), drops the extracted cost **−12.27 %**
(992 496 → 870 722) against the `ALL_RULES` baseline with a replaying
certificate and a verified lowering, adds **+18 enodes** of closure, and ships
in `DEFAULT` with `MACHINE_SHIPPED` provenance.  A strict-match sweep of all
320 corpus terms shows the pattern appears **nowhere else** — the new
architecture contributed an op-tuple the proposal grammar could not have
minted from the corpus (`project/retros/exposed-math.md`,
`project/retros/mul-unsq-nat.md`).

The engine also **mints meaning**, not just spellings.  A structured handler
grammar re-minted a verified handler for the `om_decayed` element
(`matmul(mul(S,D),V)`) at **8.9e-16** fp64 — a pattern that previously
required a hand-written executor row — plus the masked inner-chunk (1.8e-15),
the closed-form fold (4.4e-16) and the `applyd` nest (0.0).  ADR 0007's
falsification clause #3 ("handler grammar too weak") did **not** fire: every
named non-elementwise shape produced a verified handler in bounded search
(`project/retros/handler-proposals.md`).  A law-synthesis loop over the
shipped carrier atoms derived and certified **25 law rows / 4 rejected /
0 refused**, including new primitives (`affd.compose` slot-*solved*, not
spelled; `dv.merge`; `omd.absorb_score`) (`project/retros/layer3-loop.md`).

**But the honest number is the yield.**  A 60-model sweep of the exposed
class (`bench/drivers/screen/discovery_scale.py`) minted 2,101 pipeline proposals, measured
527 new-and-true candidates, and fired on 44 models — and found **0 paying
new laws beyond `mul_unsq_nat`**.  All four paying rows landed on
*purpose-built* intake probes (re-derivations of two already-banked unshipped
machine laws plus one derivable rule).  On organic architectures: nothing new
pays.  The rate is **~1.5 %** of the exposed class (1 find in 68 swept
exposed models).  Exposure itself is **bimodal**: of 321 exported workloads,
**81 score exactly 0 %** FLOPs-weighted law coverage (the fused-op class:
`nn.Conv*`, `nn.*Pool*`, `sdpa` spellings) while 140 score ≥95 %
(`project/retros/exposed-math.md`, `project/retros/discovery-scale.md`).

**The real-corpus mint funnel admits paying folds — all known-art (Q1).**
The corpus-driven overnight mint (`bench/drivers/overnight.py --corpus
zoo-real --compose`) paged all **53 held-out zoo modules**, minting
candidates from each module's own op-tuple census: **102 viable mints
(20 composites, 10 guarded), 204 evaluated, 8 admitted** — and every
admitted law classified **known-art** post-registration: the funnel's
paying finds (the `instance_norm`/`group_norm`/`rms_norm` denotations,
the sdpa/glu folds) were already catalogued transforms, so the count of
confirmed machine-originals is **0**
(`docs/reports/results_overnight_real_v6.md`,
`docs/results_realcorpus_v2.md`).  The run series also surfaced a real
substrate bug — emit-time RHS sanity caught a bare `Exception`,
conflating *cannot-evaluate* with *evaluated-and-false* so
non-denoting composites were minted as candidates; fixed in the
substrate (eqsat `2d6e4eb`), the mint now emits only viable objects.

**The primitive arm breaks the known-art ceiling (Q1, continued).**  The
`zoo-prim` mint (`bench/drivers/overnight.py --corpus zoo-prim`, 68-module
corpus) synthesizes a new `synop_*` op for each manual subgraph shape
recurring across ≥2 held-out modules and denoted by no bank op — the fold
is novel *by construction*.  The depth-1 run admitted **21/24 folds, all
machine-original** — certified, paying (launch 0.36–0.80x), underivable —
and they now ship in `DEFAULT`; the widened `--prim-deep` hunt admitted
**48/48** deeper shapes (all machine-original, disjoint names); and
`--prim-bootstrap`, which seats the shipped folds in the corpus e-graph as
vocabulary, admitted **2 depth-2 `synop2_*` folds** — **71 distinct
machine-original folds across the runs** (`docs/reports/primitives_finds.md`,
`docs/reports/results_overnight_prim_deep.md`,
`docs/reports/results_overnight_prim_bootstrap.md`).  The honest content:
the shared "sum-of-gated-products" skeleton of the SSM family (DeltaNet /
GLA / Mamba / RG-LRU) plus the 14-module `a*b+c*d` shape — real
cross-architecture structure, named by the machine, **not** "unknown to
humanity" (elementwise fusers do this implicitly; what is new is the named
primitive + certified law).  The handler layer is meta-circular: refused
specs mint their own handler *family* (`--prim-family-mint`, default on —
`family_synth.mint_families`), `gather` itself was recovered as descriptor
`fam_8c9a7ed8` (parity-verified), and all **24/24 handler bodies** mint
(`docs/reports/prim_handlers.md`, `docs/reports/family_synth.md`).

**Rediscoveries, said plainly.**  The largest structural wins are *not*
discoveries.  The RetNet carrier the machine extracts is numerically
identical (**9.8e-9 fp64**) to the paper's published chunkwise duality — "the
extracted program IS the paper's chunkwise algorithm, not an approximation of
it" — and the projection-pairing laws on llama2.c *rediscover*
`MergedColumnParallelLinear`/`QKVParallelLinear` (`project/retros/retnet-external-baselines.md`).
The contribution there is automatic discovery + certified equivalence +
per-shape selection of a *known* linearization, not the linearization itself.

### 4.2 Soundness — can it *verify* discoveries rather than merely generate plausible candidates?

**Yes — this is the strongest part of the contribution.**  The engine's
verification layer has repeatedly falsified the engine's own candidates.

* **The naive guard was falsified by the adversarial sweep.**  `auto_cond` on
  the bare `mul_unsqueeze` pattern minted a single grid-equality clause that
  separated its measured domain perfectly and was **unsound** past it: an
  adversarial sweep (mod-normalized dims, numpy broadcast, ranks ≤3, d ∈
  −8..8) found **212 bad accepts**.  The synthesized exact guard
  (`flat-map-unsq`, instantiated with the partner's own `unsq-out` spec) has
  **0 bad accepts / 0 misses** (`project/retros/mul-unsq-nat.md`).
* **An apparent paying sibling was correctly rejected at admission.**
  `mul_unsqueeze_r_id` (`mul(V, unsq U) → mul(U,V)`) showed −14.3 % and
  strict-matched **12 models / 22 sites** — the largest "blocked structure"
  the corpus sweep found.  Its real sites are **semantically unequal** (all
  four `DiagDenseSSM` sites bind `V = b_eye`; `I·unsq(U,−1)` and `U·I`
  coincide *only* because the identity masks off-diagonals — a contextual
  equality, not a law).  The discovery machinery over-reported; the admission
  machinery caught it (`mul-unsq-nat.md`, `discovery-scale.md`).
* **Deliberately-false twins are rejected.**  Layer-3 law synthesis rejected
  `addmerge_wrongstate`, `absorb_value`, `keycat_wrong` and `merge_sign_s1`
  — every false twin caught; the loop found a **real lowering bug**
  (`decay_vec`'s `sign=0` silently coerced to 1) *because the oracle refused
  to certify rather than ship a wrong law* (`project/retros/layer3-loop.md`).
* **The certificate machinery has retracted would-be claims.**  In one cycle
  it caught a false-proof matcher bug, a shape misinference that fabricated a
  **1.98× "win"**, a well-typed wrong program (diff **9.83**), and a
  launch-time-vs-execution timing bug.  All regression-tested
  (`tests/test_cost.py`, `tests/test_torch_integration.py`).
* **Shipped laws were found unsound off-corpus and guarded.**  The
  weight-merge law rewrote a real, evaluable term into a numerically
  different one (max diff **12.83**); the false region lives in the two
  summed right-operand addends (exactly one rank-1, the other ≥2), and all
  four members of the equivalence class shared the hole.  `select_mul` was
  latently unsound (broadcasting `mul` along the *selected* axis changes the
  result).  Both were invisible to the corpus (no real export carries the
  shape) and were caught only by adversarial bindings; both are now guarded
  (`project/retros/matmul-unsound-fix.md`,
  `project/retros/select-mul-broadcast.md`).
* **The typed-pay gate suppressed inflated "pays".**  Of 33 firing
  candidates in one audit, **19 minted at least one ill-typed member** (49
  ill-typed fires); the gate suppressed **11 probe pays and 12 reach drops**
  that would have been counted as wins.  Cost-dropping an invalid program is
  not evidence (`project/retros/typed-pay-gate.md`).

The methodology is the result as much as the numbers: a verifier that
accepts everything is worthless, and this one does not.

### 4.3 Value — can it discover solutions that outperform the best human-designed or existing-compiler alternatives?

**Regime-bounded yes; the losses are real.**  The wins live where work is
*asymptotically reduced* or where the incumbent compiler cannot produce code
at all.

| Win | Measurement | Artifact |
|---|---|---|
| Weight-chain fold, `x@W₁@…@W_k` → 1 GEMM (Inductor keeps all k `mm`s) | **14.92× vs Inductor** at (k,d,B·T)=(16,512,4096); 17.40× vs eager | `bench/baselines/reassoc_scale.json` |
| Asymptotic reassoc `(QKᵀ)V → Q(KᵀV)` | **8.0× at T=2048** — *retired source, not pinned* | `docs/results_curated.md` §8 |
| Spelled RetNet export, T=2048 | **5.0–5.9× vs eager**, **6.2–7.4× vs `torch.compile`**; 7.30× less peak activation memory; verified 9.8e-9 | `docs/results.md` |
| Held-out zoo (18 deliverable workloads) | **11/18 beat `torch.compile`** (LoRAAdapter 2.18×, WaveNetGate 1.51×) | `project/retros/gen-kernels.md` |
| `chainfuse` (fold + minted kernel, Triton-bound) | **2.94× vs `torch.compile`** (86 µs vs 253 µs) | `gen-kernels.md` |
| Joint forward+backward under one shared memo | **+13 % vs `torch.autograd`** (68 µs vs 78 µs, verified 9.5e-07) | `project/retros/joint-training-graph.md` |
| Streaming attention (`om` monoid) | **260× vs sdpa-recompute** at 65 k cache — *retired source, not pinned* | `docs/results_curated.md` §8 |
| ONNX round-trip (portability, not throughput) | **1.11e-16 fp64**, zero core changes | `project/retros/onnx-package.md` |

Two rows here (asymptotic reassoc, streaming attention) come from a retired
writeup whose cited artifact — a "README results table" — never existed; they
are recorded in `docs/results_curated.md` §8 as **unsourced** and must not be
read as pinned-baseline evidence.

**The honest losses, in the same table.**  The project keeps them because a
value claim without its losses is not a measurement.

| Loss | Measurement | Artifact |
|---|---|---|
| Fused-op vision model (BiRefNet Swin block) | **0.80× vs compiled** — the e-graph could not see inside `sdpa` when measured; the opt-in `sdpa_expand` family (`sdpa_decomposed` scorecard row) has since made it reachable — honest parity, not a speedup | `docs/results.md` |
| The carrier below the crossover | **0.51× vs eager at T=512** (wins asymptotically) | `docs/results.md` |
| Pure pointwise fusion (Inductor's home turf) | **0.82–0.88× loss** | `gen-kernels.md` |
| Precision (bounded) lane on real BiRefNet (220 M) | **certified negative** — 20 offers at 1e-2, 56 at 3e-2, **0 delivered** (bounded member prices 1.32× worse than dense; 18/2304 rows dedup) | `project/retros/birefnet-bounded-lane.md` |
| The minted atom did not pay | **0 certified laws fire on real terms** — the minted total vs the corpus's emitted sequence | `project/retros/atom-synth-loop.md` |
| Anti-unification compression (ADR 0007 L1) | **null on reach-expansion** — identical cheapest member with/without atoms on all 3 workloads | ADR 0007 |
| CPU decode; launch-bound cells | 0.5–0.8×, all six cells LOSS | `docs/results_curated.md` §8 |
| FLA (tuned kernels) vs the delivered carrier | FLA ~3× less memory, ~11× faster (a *lowering* gap, not semantics) | `project/retros/retnet-external-baselines.md` |

**The shape of the value claim.**  Wins are structural, not codegen: folding
a chain of GEMMs into one, deriving a quadratic→linear recurrence, lifting a
scan.  Where the structure is already fused (a `scaled_dot_product_attention`
call, a dense conv), mlopt is honest about the ceiling — parity, and the
delivered graph can be handed to `torch.compile` for codegen.  The remaining
gap to tuned kernels is precisely attributed: executor leanness (per-leaf
dispatch, fp64 torch GEMMs, a materialized score tensor), not structure.
Nothing discovered is novel to practitioners; the contribution is automatic
discovery + certified equivalence + per-shape selection, with a measured
~1.5 % yield of fresh paying structure in the exposed corpus.

### 4.4 Learned search — what the game layer measured

The learner is not the mechanism of reach; it is a *policy* over an
already-structured space, and it may only reorder (ADR 0003, amending ADR
0002 Rule 3).  Measured results, stated honestly:

* **The meta-arena.**  A cold linear REINFORCE player beats the scripted
  playbook on mean reward (−0.03 vs −0.10 over 40 held-out boards) *and*
  produces 3 certified structural wins the playbook cannot reach — but only
  after the `claim(tag)` option move fused the paying `declare → handle` line
  into one refereed action.  The caveat is explicit: **the wins ride sampling
  variance, not learned preference** (`op:claim`'s weight sits ≈+0.04) —
  action-space structure is what the board was missing, not the learner's
  consolidation (`project/retros/meta-arena-player.md`).
* **The contraction player.**  A trained/distilled player beats
  `opt_einsum`'s randomised greedy at equal wall-clock at n=40 (0.89–1.01×
  across three seeds) and is 0.51–0.90× the deterministic greedy — a real but
  small edge; the headroom is *search cost*, not quality
  (`project/retros/contraction-player-shipped.md`).
* **The learned prior.**  The shipped prior ranks held-out law candidates at
  AP 0.210 vs 0.045 for the heuristic (~4.7×) and finds ~2× more useful laws
  per episode (`docs/results_richer_features.md`; ADR 0005).
* **Enabling chains are learnable — where chain density permits (Q3).**  On a
  curriculum board whose bank is deliberately incomplete (the win needs
  `mint` the premise at −0.05, then `apply` it for +1.0), the trained linear
  arm argmaxes the two-step chain on **100 % of held-out boards** and the
  neural PolicyNet converges to the scripted ceiling (1.15/1.00 win both
  conditions).  The binding constraint is *density*, not the credit
  mechanism: at k = 16 decoys and 200 training episodes, spontaneous chains
  thin below what linear REINFORCE can lift out of the noise and it degrades
  to exactly the myopic arm's signature (`docs/results_credit_chain.md`).
* **Strassen-class rediscovery is a NO-GO (Q2).**  The 2×2×2 bilinear board
  provably expresses rank 7 — the scripted line verifies 10/10 in exactly 29
  moves, on the canonical board and every scrambled case — so the move
  vocabulary is not the blocker.  Yet across 200+ episodes at budgets 48 and
  96, random, greedy-repair, and cold-linear players all stall at **rank 8**:
  greedy repair rediscovers the *naive* scheme every episode, and the
  productive path is a ~15-move coordinated sequence through invalid-space
  intermediates with no positive prefix.  Doubling the budget bought more
  valley, not more depth (`docs/results_strassen.md`).

The pair teaches where the frontier sits: Q2 fails where Q3 succeeds, on the
same board protocol, players and referee — an enabling chain is a short
mint → apply pair the episode stream generates often enough to learn from,
while the naive → Strassen path rewards nothing until it is nearly complete.
**Search depth and chain density, not expressibility and not the referee,
are the frontier.**

Two honest warnings the game layer produced.  First, **a policy is coupled to
the semantics of its features**: a correctness fix in `mlopt_core.features`
(view/fold-aware traffic accounting) silently invalidated a documented RL
policy benefit, with no change to the RL code — so any change to
`ProgramFeatures` must be treated as a change to every policy trained on it
(ADR 0003; `project/retros/stage7-multifamily-results.md`).  Second, the
naive path — asking a learned policy to invent laws *directly* — was measured
~10× worse than structured construction and was falsified, not reopened
(ADR 0004).

## 5. Related work

**E-graph superoptimization / equality saturation.**  TENSAT and the
equality-saturation line rewrite tensor programs with a *fixed* rule set;
the reachable set is bounded by the laws a human wrote.  Catopt's
distinction is threefold: it *mints* new laws and atoms from measured need
(§4.1); every move carries a replayable certificate and candidates clear an
adversarial sweep before shipping (§4.2); and the search is an RL
environment whose policy may only reorder (§4.4).  The e-graph itself is not
the contribution — it is the substrate the discovery loop writes into.

**The nearest neighbors: mint + verify, learned play, certified learned
play.**  TASO (Jia et al., SOSP 2019) auto-generates graph substitutions
for DNNs and verifies them formally with an automated theorem prover — the
strongest prior on the verify-what-you-mint leg; the generated bank ships
once and never re-enters subsequent discovery.  Isaria (Thomas & Bornholt,
ASPLOS 2024) synthesizes *sound* rewrite-rule systems from ISA
specifications to drive e-graph vectorizing compilers, and Ruler/Enumo
(Nandi et al., OOPSLA 2021; Pal et al., OOPSLA 2023) infer rulesets and make
their inference programmable — all offline, all static once shipped.
Glenside (Smith et al., MLSys 2021) is the closest precedent for
machine-derived tensor transformations: its e-graphs rediscover
access-pattern rewrites (im2col-style), the shape of `conv_bn_fold` here.
On the play side, CompilerGym (Cummins et al., CGO 2022) casts compiler
optimization as an RL environment over a fixed action space, and the
Cambridge line — Omelette (Singh, MPhil thesis 2022), MCTS-GEB (He, Singh &
Yoneki, EuroMLSys 2023), Aurora (Bărbulescu et al., arXiv 2024) — trains
RL/MCTS policies that steer e-graph construction: learned play exists, but
over a static bank.  AlphaGeometry (Trinh et al., Nature 2024) couples a
learned proposer with a certifying symbolic deducer — learned play inside a
verified loop, over a fixed deduction closure (AlphaTensor, Fawzi et al.,
Nature 2022, similarly searches one fixed structure).  Each neighbor holds
pieces of the loop; to our knowledge none makes the minted bank itself an
object of play — rules re-entering as moves inside the learned game is the
differentiation (§4.4).  The closure buys three capabilities the neighbors
lack: **two-engine certification** — a certified law must execute
identically through mlopt's e-graph *and* handl's independent rewrite
engine (`packages/mlopt-handl-peer`,
`bench/drivers/audit/handl_differential.py`); **laws as first-class
values** — minting constructs a `Value` in the same calculus, no codegen
step; and **language-agnostic exits** — one `Sink` port lowers the same
extracted graph to torch, ONNX, or `.hnl`
(`bench/drivers/kit/term_hnl.py`).

**Program synthesis / library learning.**  Anti-unification library learning
(the Babble line) is the direct ancestor of ADR 0007 layer 1: anti-unify
recurring subgraphs and mint a named atom whose semantics *is* its expansion.
Catopt's layer 1 measured **null** on reach-expansion — compression is real
(shorter spellings: 12 vs 26 ops) but naming structure did not expand reach on
any measured workload.  The engine's novelty is the layers *above*
definitional atoms: handler-interpreted atoms (semantics by denotation) and
law-theoretic atoms (equational grounding), each admitted through the same
gauntlet.

**Tuned-kernel baselines (FlashAttention / FLA).**  The delivered carrier
reaches parity with the RetNet paper's own chunked implementation (2.08 vs
2.12 ms at T=2048; faster under graph replay, 0.670 vs 0.822 ms at T=1024),
but FLA's fused fp32 Triton kernels are still ~11× ahead.  That gap is
*kernel/lowering* quality, not semantics — a separate dimension (ADR 0003).
This is the honest boundary of the "value" pillar: structural reach is
demonstrated; matching a hand-tuned kernel is a codegen problem the engine
does not yet solve.

**Compiler baselines.**  `torch.compile`/Inductor and TVM are the
incumbents throughout.  The comparison is fair by construction — same
backend, same device, fp64 verification where stated — and the losses are
kept alongside the wins.

**Differential oracle.**  `tests/test_egglog_oracle.py` cross-checks mlopt's
hand-rolled equality saturation against the Rust **egglog** library on a
small op/law subset (opt-in, test-only; core stays pure-Python).  It asserts
the extracted lowest-FLOPs terms match up to commutative-representative
choice and that the folded costs coincide — an independent check that the
saturation is not idiosyncratic.  The second, stronger oracle is the handl
peer (`packages/mlopt-handl-peer`): the structural bank fires through
handl's own rewrite engine and must produce the term mlopt's
`EGraph.apply_rule` produces (`bench/drivers/audit/handl_differential.py`)
— a wrong law must fool two implementations of the calculus, not one.

## 6. Limitations

* **Dev-box magnitudes.**  Measured on an RTX 2050 (4 GB) + i5-12500H; the
  magnitudes do not extrapolate to datacenter hardware.
  `scripts/calibrate_profile.py` makes re-targeting mechanical, but the numbers
  themselves are box-specific.
* **Toy dimensions.**  d ≤ 64, T ≤ 2048, B = 1 in the structural sweeps;
  real checkpoints (stories15M/110M) transform and verify but land at
  **parity** on wall time.
* **Inference only.**  Weight folding destroys per-layer gradients.
* **Compile-time search.**  Seconds per block; a 4-layer model went 768 s →
  4.9 s with incremental matching, but the spine walk + saturation at large T
  is the wall (215–262 s at T=2048, vs Inductor's 45 s compile where it
  succeeds).
* **Discovery yield is sparse (~1.5 %)** and exposure is bimodal — 81/321
  workloads expose nothing at all.  On the real held-out corpus the
  corpus-driven mint admits real paying folds but every one classifies
  **known-art** post-registration: 53 zoo modules → 102 viable mints,
  8 admitted, **0 confirmed machine-originals** in the rewrite arm (§4.1 —
  the primitive arm breaks that ceiling: 71 distinct machine-original
  `prim:synop*` folds across the prim runs); shallow
  single-step mints are exhausted — guarded composites are the open lever.
* **Strassen-class rediscovery is out of reach of per-step play.**  The
  bilinear board expresses rank 7 (verified scripted witness) but no player
  crosses the invalid-space valley in 200+ episodes — the blocker is a
  ~15-move coordinated trajectory with no positive prefix, i.e. search
  depth, not vocabulary or machinery (§4.4; `docs/results_strassen.md`).
* **The scale walls are now mapped (Q4).**  Saturation is T-invariant on
  symbolic attention terms (≲ 0.3 s through T=4096) but unrolled spellings
  hit a matcher wall at ~11k IR nodes (T=1024 exceeds a 480 s budget, still
  ~1.3 GB); the memory wall is the referee's verify-forward of the spelled
  program (a ~17 GB fp64 attention tensor kernel-OOM'd at T=4096 — the
  e-graph had saturated the same term in 0.2 s); and an unscaled `exp`
  attention overflows fp64 at T≥1024, which verify honestly refuses rather
  than certify (`docs/reports/results_scale_ladder.md`).
* **Cross-domain prior transfer measured null (Q5)** on a private
  codegen-domain measurement — cited here without numbers; whether a learned
  prior survives a minted move-set remains open.
* **Enumeration ordering is a measured blocker.**  The paying `fold_emit`
  descriptor sits >200 k deep in the round-robin; it verifies at 6.7e-16
  when reached directly, so the pay-null is *ordering*, not expressiveness.
* **Executor leanness.**  Per-leaf dispatch is the loss driver; the FLA gap
  is now precisely attributed there.
* **The precision lane is a certified negative on BiRefNet** for a measured
  reason (thin dedup: 18/2304 rows).
* **The value story is narrow.**  Wins where work is asymptotically reduced
  or where Inductor cannot codegen; losses wherever Inductor's fused kernels
  apply.
* **The substrate is a bounded search space at best.**  ADR 0008 gate 3
  measured negative four times (§2.2) before the form-level proposer reached
  it: `loop`-headed programs equal to the winner are now reachable — but only
  through anti-unification over the saturated e-graph (extraction itself still
  never picks a `loop` member), and the certified program holds two loops
  under one schema, not one, because the family-boundary state is never merged
  (a reach gap, `state-chain-broken:block3→block4`).  The gate work also
  exposed an extraction defect — greedy per-class extraction is **1.01 %** off
  its own cost model on that workload (~~5.06 %~~; corrected by the
  determinism fix, `abf6114`; fix in `bench/`, pending the full suite — at ~1 %
  the promotion is a judgement call).

## 7. What would falsify each pillar

Stated so the claims can be killed by measurement, not defended by rhetoric.

**Mathematical discovery.**  Falsified if proposal-space extension never
produces a paying admission beyond `mul_unsq_nat` — i.e., if the yield stays
~1.5 % and the form list stays hand-written (plan 0024/0025), or if every
"new" law turns out to be a re-spelling of a shipped one.  The first reading
already partly fired: over 12,000 candidates the form-level enumerator
admitted 7 paying laws from forms outside the shipped 27, but each is a
*composite re-spelling* of two shipped identities — "the form list is not the
whole grammar, but the forms it lacks are shallow" (plan 0024).  The decisive
test is whether a minted *atom* (vocabulary, not a re-spelling) pays where no
composition does.

**Categorical soundness.**  Falsified if the adversarial sweep admits a false
law, if a delivered module verifies but computes something else, or if a
certificate replays over an unsound rule.  The last is already a *measured
hazard*: `verify_certificate` replays the derivation, not its truth, so the
guard is the only layer that keeps a false member out of the e-graph
(`matmul-unsound-fix.md`).  The standing test: no false equality accepted
(0/51 in the rediscovery controls), every shipped law's false region swept.

**Learned search.**  Falsified if a learned policy changes the fixed point or
the certificate (it may only reorder), or if learning never beats the
scripted/random baselines on held-out boards beyond sampling variance.  The
meta-arena result sits exactly on this line: the learned player beats
scripted, but the wins ride sampling variance, not learned preference — so
the pillar is *demonstrated but not yet consolidated*, and a stronger policy
class is the open work.

## 8. Verdict

Publishable as a **discovery + verification** contribution, not as a speedup
paper.  What is certified and verified: the loop *mints content that ships* (a
law derived from a model's own op-tuple is in the default rule set with a
synthesized guard that survived an adversarial sweep; a handler grammar
re-minted a previously hand-written carrier executor), while the same
machinery honestly refused its apparent siblings, its first composed atom
(pay-null) and its own compression layer (reach-null).  The verification
boundaries are per-artifact and the adversarial layer keeps catching
would-be false claims.  The performance story is real but narrow, and the
losses are kept in the same table as the wins.  Framed as "verified
discovery with a regime-bounded performance win, and a loop that ships what
it certifies," this is a solid systems/PL workshop or short-paper result; a full
venue paper needs the scale validation and the executor work named in §6.

---

## Appendix A — evidence table (claim → measurement → artifact)

| Claim | Measurement | Artifact |
|---|---|---|
| **Novelty.** A corpus-minted law ships in the default rule set | `mul_unsq_nat`, `MACHINE_SHIPPED`, −12.27 % on DeltaRuleBlock, +18 enodes | `project/retros/mul-unsq-nat.md`; `mlopt_core.laws.provenance.MACHINE_SHIPPED` |
| machine-admitted laws ship beside the hand-written kernel | `MACHINE_SHIPPED` (127 rows — 13 laws + 57 minted `prim:synop*` folds + 57 `.expand` premises; 70 seated in `DEFAULT`) | `docs/api.md`; `project/retros/promoted-laws.md`; `docs/reports/primitives_finds.md` |
| A handler for a previously hand-written executor is machine-minted | `om_decayed` element at 8.9e-16 fp64 | `project/retros/handler-proposals.md`; ADR 0007 |
| An atom's law set is synthesized and certified | 25 certified / 4 rejected / 0 refused | `project/retros/layer3-loop.md` |
| Fresh paying structure is sparse | 60-model sweep: 0 paying new laws beyond `mul_unsq_nat`; ~1.5 % | `project/retros/discovery-scale.md` |
| Exposure is bimodal | 81/321 expose 0 %; 140 ≥95 % | `project/retros/exposed-math.md`; `bench/drivers/screen/exposure_screen.py` |
| A major win is a *rediscovery* | carrier ≡ the paper's chunkwise duality at 9.8e-9 fp64 | `project/retros/retnet-external-baselines.md` |
| **Soundness.** A naive guard was falsified by the adversarial sweep | 212 bad accepts | `project/retros/mul-unsq-nat.md` |
| The synthesized exact guard is clean | `flat-map-unsq`: 0 bad / 0 miss | `mul-unsq-nat.md` |
| An apparent paying sibling is rejected | `mul_unsqueeze_r_id` −14.3 %, 12 models/22 sites, unequal | `mul-unsq-nat.md`; `discovery-scale.md` |
| Deliberately-false twins are rejected | 4 rejected (`addmerge_wrongstate`, `absorb_value`, `keycat_wrong`, `merge_sign_s1`) | `layer3-loop.md` |
| The verifier retracts would-be claims | false-proof matcher; a 1.98× fabricated win; a wrong program (diff 9.83); a timing bug | `tests/test_cost.py`; `tests/test_torch_integration.py` |
| Shipped laws were unsound off-corpus and fixed | matmul false region (max diff 12.83); `select_mul` broadcast | `matmul-unsound-fix.md`; `select-mul-broadcast.md` |
| The loop caught a real lowering bug by refusing | `decay_vec` `sign=0` coercion | `layer3-loop.md` |
| The typed-pay gate suppressed inflated "pays" | 19/33 mint ill-typed; 11 pays + 12 drops suppressed | `typed-pay-gate.md` |
| **Substrate.** The combinator presentation is a bounded search space — `loop` is reached through the e-graph oracle, not by extraction | gate 3 FAIL ×4 (proposing rule fires 0×; 7 blocks / 2 families `IRREGULAR`; still 7 templates modulo width+index), then **reached**: two `Loop(step,3)` under one `pack(S′,o)` schema over 9 wires, `comb_to_dag` lands in the winner's root e-class; residual `state-chain-broken:block3→block4` | `project/retros/gate3-loop-proposing-rule.md`; `gate3-parametric-abstraction.md`; `gate3-extraction-state-spelling.md`; `gate3-sizes-abstraction.md`; `gate3-form-level-proposer.md`; ADR 0008 |
| Greedy per-class extraction is suboptimal on the gate-3 workload | 455,752 → 451,128 FLOPs (−1.01 %; ~~432,696 / −5.06 %~~ was a hash-order artifact, `abf6114`); max\|Δ\| 2.2e-16 | `project/retros/gate3-extraction-state-spelling.md` |
| **Value (wins).** Weight-chain fold | 14.92× vs Inductor (17.40× vs eager) | `bench/baselines/reassoc_scale.json` |
| Asymptotic reassoc | 8.0× at T=2048 (retired source — unsourced) | `docs/results_curated.md` §8 |
| Spelled RetNet export T=2048 | 5.0–5.9× vs eager; 6.2–7.4× vs compile; 7.30× memory | `docs/results.md` |
| Held-out zoo | 11/18 beat `torch.compile` | `gen-kernels.md` |
| `chainfuse` (fold + minted kernel) | 2.94× vs compile | `gen-kernels.md` |
| Joint fwd+bwd | +13 % vs autograd (9.5e-07) | `joint-training-graph.md` |
| Streaming attention | 260× vs sdpa-recompute at 65 k (retired source — unsourced) | `docs/results_curated.md` §8 |
| ONNX portability | 1.11e-16 fp64; zero core changes | `onnx-package.md`; ADR 0006 |
| **Value (losses).** Fused-op vision model | 0.80× vs compiled | `docs/results.md` |
| Carrier below crossover | 0.51× vs eager at T=512 | `docs/results.md` |
| Pointwise fusion | 0.82–0.88× loss | `gen-kernels.md` |
| Precision lane on BiRefNet | certified negative; 0 delivered; 1.32× worse | `birefnet-bounded-lane.md` |
| The minted atom pays nothing | 0 certified laws fire on real terms | `atom-synth-loop.md` |
| Anti-unification reach-expansion | null on all 3 workloads | ADR 0007 |
| FLA vs the delivered carrier | ~3× memory, ~11× speed (lowering gap) | `retnet-external-baselines.md` |
| **Frontier.** Real-corpus mint funnel (Q1) | 53 zoo modules → 102 viable mints (20 composites, 10 guarded), 204 evaluated → 8 admitted, all known-art post-registration / 0 machine-originals | `docs/reports/results_overnight_real_v6.md`; `docs/results_realcorpus_v2.md` |
| Primitive arm (Q1) | 21/24 depth-1 + 48/48 `--prim-deep` + 2 depth-2 `synop2_*` under `--prim-bootstrap` admitted, all machine-original (71 distinct); 23 folds seated in `DEFAULT`; 24/24 handler bodies minted, `gather` recovered as descriptor `fam_8c9a7ed8` | `docs/reports/primitives_finds.md`; `docs/reports/results_overnight_prim_deep.md`; `docs/reports/results_overnight_prim_bootstrap.md`; `docs/reports/family_synth.md` |
| Emit-time RHS-sanity substrate fix | bare-`Exception` catch conflated can't-eval with eval-false; mint now emits only viable candidates | eqsat `2d6e4eb` |
| Strassen rediscovery board (Q2) | NO-GO: scripted witness verifies rank 7 (29 moves, 10/10); all players 0 wins in 200+ episodes, stall at rank 8 | `docs/results_strassen.md` |
| Enabling-chain credit (Q3) | learned arms close mint → apply chains (linear-greedy 100 % held-out; neural at the scripted ceiling); density-bound at k=16 | `docs/results_credit_chain.md` |
| Scale ladder (Q4) | symbolic terms T-invariant ≲ 0.3 s; unrolled matcher wall ~11k nodes (480 s budget); verify-forward OOM ~17 GB at T=4096; fp64 overflow at T≥1024 | `docs/reports/results_scale_ladder.md` |
| Cross-domain prior transfer (Q5) | null — a private codegen-domain measurement, no numbers | (private) |

## Appendix B — reproduction

All measurements are reproducible from the committed tree; the harness
commands live with their artifacts:

```sh
uv run pytest                                   # full suite (serial — no -n auto)
.venv/bin/ty check && .venv/bin/ruff check && .venv/bin/ruff format --check
.venv/bin/vulture && .venv/bin/lint-imports
python -m mlopt_discovery.pipeline             # census → propose → verify → measure → rank → emit
python -m mlopt_discovery.play --domain gen --probe
python -m mlopt_discovery.play --zoo
python -m bench run <suite> --device cpu
```

The per-work records this paper summarises live under `project/retros/`; the
decisions under `project/adrs/`; the measured results under `docs/results.md`
and `docs/results_tooling.md`.  The lab notebook
(`project/research_lab_notebook.md`) is the chronological record of how the
numbers were found, including the dead ends.
