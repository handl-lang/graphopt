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
member the target can run, and a replayable certificate proves the step.
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
is, honestly, ~1.5 %.  On *soundness*: every move carries a replayable
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
with **zero changes under `catopt-core`**, verified through `onnxruntime` at
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
   axioms *generate* rewrites; the certificate *proves* the step.  This is
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
| **Discovery** | find laws/algorithms humans did not encode | `catopt_discovery` (census → propose → verify → measure → rank → emit) |
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

PyTorch is *one application* of the engine.  `catopt-core` is torch-free and
dependency-light (`import-linter` enforces that core imports no `torch`,
`numpy`, or GPU library), and a new backend implements the `Sink` port
without touching core.  This is not an assertion; it is measured twice.  The
ONNX backend — `OnnxSink` (IR → ONNX) and `OnnxSource` (ONNX → IR) — landed
with **zero edits under `catopt-core`** (`git diff --name-only` against the
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
rewrites is a 3-cell.  `catopt_discovery.coherence` enumerates the pair layer
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

**Discovery — the closed loop.**  `catopt_discovery.pipeline` runs
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

**The substrate beneath the syntax (ADR 0008).**  A prototype combinator
presentation (`lab/bridge/combinator.py`) re-expresses the IR as morphisms of a
monoidal category with explicit `dup` (Δ, sharing as a morphism) and `loop`
(bounded trace): 321/321 corpus workloads round-trip structurally exact,
zero ops refuse translation, mean blowup 2.61×.  The RetNet shared-state fold
spells as `seq(pre, loop(step, K), post)` and verifies at 1e-7 — the
"carrier atoms become *definable*, not primitive" claim, measured.  The open
gate: `loop` structure is erased at export, so folds must be *proposed by
search*.

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
class (`lab/screen/discovery_scale.py`) minted 2,101 pipeline proposals, measured
527 new-and-true candidates, and fired on 44 models — and found **0 paying
new laws beyond `mul_unsq_nat`**.  All four paying rows landed on
*purpose-built* intake probes (re-derivations of two already-banked unshipped
machine laws plus one derivable rule).  On organic architectures: nothing new
pays.  The rate is **~1.5 %** of the exposed class (1 find in 68 swept
exposed models).  Exposure itself is **bimodal**: of 321 exported workloads,
**81 score exactly 0 %** FLOPs-weighted law coverage (the fused-op class:
`nn.Conv*`, `nn.*Pool*`, `sdpa` spellings) while 140 score ≥95 %
(`project/retros/exposed-math.md`, `project/retros/discovery-scale.md`).

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
  (`RESEARCH_WRITEUP.md` §2).
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
| Asymptotic reassoc `(QKᵀ)V → Q(KᵀV)` | **8.0× at T=2048** | `RESEARCH_WRITEUP.md` §3 |
| Spelled RetNet export, T=2048 | **5.0–5.9× vs eager**, **6.2–7.4× vs `torch.compile`**; 7.30× less peak activation memory; verified 9.8e-9 | `docs/results.md` |
| Held-out zoo (18 deliverable workloads) | **11/18 beat `torch.compile`** (LoRAAdapter 2.18×, WaveNetGate 1.51×) | `project/retros/gen-kernels.md` |
| `chainfuse` (fold + minted kernel, Triton-bound) | **2.94× vs `torch.compile`** (86 µs vs 253 µs) | `gen-kernels.md` |
| Joint forward+backward under one shared memo | **+13 % vs `torch.autograd`** (68 µs vs 78 µs, verified 9.5e-07) | `project/retros/joint-training-graph.md` |
| Streaming attention (`om` monoid) | **260× vs sdpa-recompute** at 65 k cache | `RESEARCH_WRITEUP.md` §3 |
| ONNX round-trip (portability, not throughput) | **1.11e-16 fp64**, zero core changes | `project/retros/onnx-package.md` |

**The honest losses, in the same table.**  The project keeps them because a
value claim without its losses is not a measurement.

| Loss | Measurement | Artifact |
|---|---|---|
| Fused-op vision model (BiRefNet Swin block) | **0.80× vs compiled** — the e-graph cannot see inside `sdpa` | `docs/results.md` |
| The carrier below the crossover | **0.51× vs eager at T=512** (wins asymptotically) | `docs/results.md` |
| Pure pointwise fusion (Inductor's home turf) | **0.82–0.88× loss** | `gen-kernels.md` |
| Precision (bounded) lane on real BiRefNet (220 M) | **certified negative** — 20 offers at 1e-2, 56 at 3e-2, **0 delivered** (bounded member prices 1.32× worse than dense; 18/2304 rows dedup) | `project/retros/birefnet-bounded-lane.md` |
| The minted atom did not pay | **0 certified laws fire on real terms** — the minted total vs the corpus's emitted sequence | `project/retros/atom-synth-loop.md` |
| Anti-unification compression (ADR 0007 L1) | **null on reach-expansion** — identical cheapest member with/without atoms on all 3 workloads | ADR 0007 |
| CPU decode; launch-bound cells | 0.5–0.8×, all six cells LOSS | `RESEARCH_WRITEUP.md` §4 |
| FLA (tuned kernels) vs the delivered carrier | FLA ~3× less memory, ~11× faster (a *lowering* gap, not semantics) | `project/retros/retnet-external-baselines.md` |

**The shape of the value claim.**  Wins are structural, not codegen: folding
a chain of GEMMs into one, deriving a quadratic→linear recurrence, lifting a
scan.  Where the structure is already fused (a `scaled_dot_product_attention`
call, a dense conv), catopt is honest about the ceiling — parity, and the
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
  AP 0.133 vs 0.045 for the heuristic and finds ~2× more useful laws per
  episode (`README.md`; ADR 0005).

Two honest warnings the game layer produced.  First, **a policy is coupled to
the semantics of its features**: a correctness fix in `catopt_core.features`
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

**Differential oracle.**  `tests/test_egglog_oracle.py` cross-checks catopt's
hand-rolled equality saturation against the Rust **egglog** library on a
small op/law subset (opt-in, test-only; core stays pure-Python).  It asserts
the extracted lowest-FLOPs terms match up to commutative-representative
choice and that the folded costs coincide — an independent check that the
saturation is not idiosyncratic.

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
  workloads expose nothing at all.
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
paper.  What is proven and verified: the loop *mints content that ships* (a
law derived from a model's own op-tuple is in the default rule set with a
synthesized guard that survived an adversarial sweep; a handler grammar
re-minted a previously hand-written carrier executor), while the same
machinery honestly refused its apparent siblings, its first composed atom
(pay-null) and its own compression layer (reach-null).  The verification
boundaries are per-artifact and the adversarial layer keeps catching
would-be false claims.  The performance story is real but narrow, and the
losses are kept in the same table as the wins.  Framed as "verified
discovery with a regime-bounded performance win, and a loop that ships what
it proves," this is a solid systems/PL workshop or short-paper result; a full
venue paper needs the scale validation and the executor work named in §6.

---

## Appendix A — evidence table (claim → measurement → artifact)

| Claim | Measurement | Artifact |
|---|---|---|
| **Novelty.** A corpus-minted law ships in the default rule set | `mul_unsq_nat`, `MACHINE_SHIPPED`, −12.27 % on DeltaRuleBlock, +18 enodes | `project/retros/mul-unsq-nat.md`; `catopt_core.laws.provenance.MACHINE_SHIPPED` |
| 11 machine-admitted laws ship beside the hand-written kernel | `MACHINE_SHIPPED` (11) | `docs/api.md`; `project/retros/promoted-laws.md` |
| A handler for a previously hand-written executor is machine-minted | `om_decayed` element at 8.9e-16 fp64 | `project/retros/handler-proposals.md`; ADR 0007 |
| An atom's law set is synthesized and certified | 25 certified / 4 rejected / 0 refused | `project/retros/layer3-loop.md` |
| Fresh paying structure is sparse | 60-model sweep: 0 paying new laws beyond `mul_unsq_nat`; ~1.5 % | `project/retros/discovery-scale.md` |
| Exposure is bimodal | 81/321 expose 0 %; 140 ≥95 % | `project/retros/exposed-math.md`; `lab/screen/exposure_screen.py` |
| A major win is a *rediscovery* | carrier ≡ the paper's chunkwise duality at 9.8e-9 fp64 | `project/retros/retnet-external-baselines.md` |
| **Soundness.** A naive guard was falsified by the adversarial sweep | 212 bad accepts | `project/retros/mul-unsq-nat.md` |
| The synthesized exact guard is clean | `flat-map-unsq`: 0 bad / 0 miss | `mul-unsq-nat.md` |
| An apparent paying sibling is rejected | `mul_unsqueeze_r_id` −14.3 %, 12 models/22 sites, unequal | `mul-unsq-nat.md`; `discovery-scale.md` |
| Deliberately-false twins are rejected | 4 rejected (`addmerge_wrongstate`, `absorb_value`, `keycat_wrong`, `merge_sign_s1`) | `layer3-loop.md` |
| The verifier retracts would-be claims | false-proof matcher; a 1.98× fabricated win; a wrong program (diff 9.83); a timing bug | `RESEARCH_WRITEUP.md` §2 |
| Shipped laws were unsound off-corpus and fixed | matmul false region (max diff 12.83); `select_mul` broadcast | `matmul-unsound-fix.md`; `select-mul-broadcast.md` |
| The loop caught a real lowering bug by refusing | `decay_vec` `sign=0` coercion | `layer3-loop.md` |
| The typed-pay gate suppressed inflated "pays" | 19/33 mint ill-typed; 11 pays + 12 drops suppressed | `typed-pay-gate.md` |
| **Value (wins).** Weight-chain fold | 14.92× vs Inductor (17.40× vs eager) | `bench/baselines/reassoc_scale.json` |
| Asymptotic reassoc | 8.0× at T=2048 | `RESEARCH_WRITEUP.md` §3 |
| Spelled RetNet export T=2048 | 5.0–5.9× vs eager; 6.2–7.4× vs compile; 7.30× memory | `docs/results.md` |
| Held-out zoo | 11/18 beat `torch.compile` | `gen-kernels.md` |
| `chainfuse` (fold + minted kernel) | 2.94× vs compile | `gen-kernels.md` |
| Joint fwd+bwd | +13 % vs autograd (9.5e-07) | `joint-training-graph.md` |
| Streaming attention | 260× vs sdpa-recompute at 65 k | `RESEARCH_WRITEUP.md` §3 |
| ONNX portability | 1.11e-16 fp64; zero core changes | `onnx-package.md`; ADR 0006 |
| **Value (losses).** Fused-op vision model | 0.80× vs compiled | `docs/results.md` |
| Carrier below crossover | 0.51× vs eager at T=512 | `docs/results.md` |
| Pointwise fusion | 0.82–0.88× loss | `gen-kernels.md` |
| Precision lane on BiRefNet | certified negative; 0 delivered; 1.32× worse | `birefnet-bounded-lane.md` |
| The minted atom pays nothing | 0 certified laws fire on real terms | `atom-synth-loop.md` |
| Anti-unification reach-expansion | null on all 3 workloads | ADR 0007 |
| FLA vs the delivered carrier | ~3× memory, ~11× speed (lowering gap) | `retnet-external-baselines.md` |

## Appendix B — reproduction

All measurements are reproducible from the committed tree; the harness
commands live with their artifacts:

```sh
uv run pytest                                   # full suite (serial — no -n auto)
.venv/bin/ty check && .venv/bin/ruff check && .venv/bin/ruff format --check
.venv/bin/vulture && .venv/bin/lint-imports
python -m catopt_discovery.pipeline             # census → propose → verify → measure → rank → emit
python -m catopt_discovery.play --domain gen --probe
python -m catopt_discovery.play --zoo
python -m bench run <suite> --device cpu
```

The per-work records this paper summarises live under `project/retros/`; the
decisions under `project/adrs/`; the measured results under `docs/results.md`
and `docs/results_tooling.md`.  The lab notebook
(`project/research_lab_notebook.md`) is the chronological record of how the
numbers were found, including the dead ends.
