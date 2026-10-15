# How catopt works — syntax, structure, and the rewrite loop

The model graph is **syntax**; a category-theoretic **structure**
interprets it; the structure's **axioms** generate rewrites; **search**
picks the cheapest; the **certificate** proves it.

This is the conceptual pipeline behind ADR 0002.  It is deliberately
separate from the code walkthrough — it answers *why* the machinery
is shaped the way it is.

## The tower

The same picture in categorical vocabulary: a program is a
**1-morphism** between objects (tensor types); a rewrite law is a
**2-morphism** between programs; a `derivation=` — one law proven
from premises — is a **3-morphism** (a coherence; the catalogue
measures confluence and divergence between laws).  Declared objects
— folds, lifts, compositions — are machine-constructed cells written
as *data*, and `opdata` makes a new primitive declarable the same
way.  Two games are played on this tower: **search** applies 2-cells
inside one equivalence class (steps 5–6 below); **construction**
writes new cells (`catopt_discovery.object_synthesis` + `evidence`,
played as episodes on `catopt_discovery.arena` / `meta_arena` boards,
exposed as an RL environment by `catopt_discovery.play` — one CLI,
five domains, `--deliver`/`--probe`/`--zoo`).  In the `gen` domain the
board mints a handler for any unnamed elementwise subterm
(`genkernel`), `claim` declares the handler's own pattern mid-search,
and `deliver` lowers to a verified module calling generated
`triton.jit` source.  The codegen grammar covers three tiers —
pointwise fusion sites, diagonal-scan `applyd` carriers, and the
fused `om_decayed` contraction (the elementwise decay product folded
into `tl.dot`'s inner loop, never materializing the `(…,T,K)`
intermediate) — each declined spec keeps its torch path honestly.
`profitable`, the measured referee, times every tier on its real
delivery path (`kernel_for`), same device for both arms.  The
certificate is the replayable 2-cell; its replay is the referee.

The decayed-attention structure is fully equational end-to-end:
`decay(γ,t)` names the causal triangle `γ^{i−j}` and `decay_vec`
the `γ^{s·k+o}` vectors, so a spelled `tril(γ^{i−j})` lifts into the
generator (`decay_lift`), and the chunked form is a law closure —
`om_decayed_chunkify_{rows,keys}` bisect a fused site, `split_*`
laws push the partition through the score matmul, `decay_block_*`
rewrite each block (diagonal → `decay`; strict-lower → the rank-1
`ξ·ηᵀ`), `om_decayed_cross` distributes `(s ⊙ ξηᵀ)·v` so the block
never materializes, and the state-lift folds the weighted sum into
`applyd` scan carriers.  On ChunkedRetention's real
`torch.export` graph the saturation produces `decay`,
`decay_vec`, and `applyd`/`aff_diag` enodes itself — the RetNet
recurrence as an equational structure the machine derived, not a
hand-named algorithm.

## The loop, step by step

Running example throughout: the recurrence `h_t = A·h_{t-1} + x_t`.

**1. Syntax.**  The graph as written — ops and wires.  Just a term,
no meaning.  *Example: a chain of `matmul(A, h)` + `add(x)`.*

**2. Structure.**  A mathematical object with two halves:

- **Signature** — its operations, i.e. the *shape*: "a fold over a
  sequence of `(A, b)` pairs."
- **Axioms** — equations those operations obey: associativity + unit.

*Example: the monoid `aff(A, b)`.*

**3. Match → the lift.**  Recognize that a piece of the graph has
the **signature's shape**.  That makes it a morphism *into* the
structure.  *Example: recognize the recurrence spine as an `aff`
fold.*  Matching is done on the **signature**.

**4. Rewrite.**  Now the structure's **axioms** become rewrite
rules.  *Example: associativity lets you regroup the fold.*
Rewriting is done with the **axioms**.  Side conditions
(`check` / `derive`) are the soundness boundary — the only layer
that keeps a false member out of the e-graph: adversarial witness
generation found three shipped matmul laws false on mixed-rank
bindings (and a fuse law minting an unevaluable member); all
guarded now.

**5. Saturate.**  Apply the axioms to a fixed point — this
*generates* every equivalent program.  The e-graph **is** this
closure (the consequences, not a search over them).

**6. Extract.**  Search the closure for the cheapest program the
backend can run.  *Example: the Blelloch tree.*

**7. Certify.**  The path original → extracted is a replayable
certificate (a 2-cell).

**8. Evaluate.**  Describe the extracted program's *characteristics*
(torch-free `ProgramFeatures`) and, when a target is available, its
runtime.  Evaluation is a separate dimension (ADR 0003): it **ranks**
candidates, it never changes what is equivalent.  A profiler observes;
it does not decide.

*Shortcut:* if the structure names the target directly, **construct**
it and skip steps 5–6.

```text
syntax → structure(signature + axioms) → match(signature)
       → rewrite(axioms) → saturate → extract → certify → evaluate
```

## The split that trips people up

| Half | What it is | What you do with it |
|---|---|---|
| **Signature** | sorts + operations — the *shape* of a morphism into the structure | **match** against it |
| **Axioms** | equations the operations satisfy | **rewrite** with them |

Matching is **not** based on the axioms.  You match on the
*shape* (signature); the axioms license the rewrites afterwards.
The precise statement: matching a structure = **finding a functor**
from the subgraph's free category into that structure — i.e.
finding an interpretation.

Two more consequences worth stating plainly:

- **Laws are entailed by the structure**, in two tiers: *axioms*
  (defining equations) and *consequences* (theorems derived from
  them).  Few axioms generate many consequences; the reach is their
  closure.  The split is *measured on the shipped library*, not
  asserted: `rule.kind` marks each member `axiom` / `lemma` /
  `redundant` (55 / 14 / 2 over `ALL_RULES`), and `rule.derivation`
  records the premise(s) proving each non-axiom.
- **Choosing the interpretation is the discovery act.**  The same
  graph reads as plain composition, a comonoid (pairing), a
  low-rank approximation (bounded), or a contraction problem —
  each exposing different laws.  Discovery = picking the richest
  reading, not applying laws inside a fixed one.

## The meta-layer — laws about laws, and where laws come from

The loop above runs *inside* a fixed law library.  Two mechanisms
operate on the library itself.

**Coherence — the 3-cell layer.**  A rewrite is a 2-cell; a law
*about* rewrites is a 3-cell.  `catopt_discovery.coherence` enumerates
the pair layer over the shipped library (co-firing critical pairs,
direct derivability edges); `catopt_discovery.coherence2` measures
depth 2 — stratification, reach under removal, and the mediator
table ("the coherence of `A × B` requires `C`").  The measured
structure: the derived layer of `ALL_RULES` is *flat* — zero laws
derivable from the primitives alone, every derivation a
self-supporting cycle (ten inverse pairs plus the 4-member matmul
class) plus one emergent law (`silu_mul_form`); the effective
basis is 32 independent primitives (27 derivable), marked on the rules
by `--emit-basis`.  Two honest findings: **derivable ≠
disposable** — removing a derivable law costs real bounded reach
(the twin proves the *equality*; only this direction's LHS fires
on those spellings — `linear_to_matmul_t` is provable yet the most
load-bearing derivable law under removal); and **divergence is
found and mediated** — `silu_fold` shipped to restore confluence
where `silu_expand` destroyed the `swiglu_fuse` redex, and it pays.

**Discovery — the closed loop.**  `catopt_discovery.pipeline` runs the
whole pipeline on a corpus of real exported models (364 terms /
725 op-tuples today): **census** the op-shape tuples that occur →
**propose** candidates (census-naturality, mixed-view,
pattern-recognition and grammar schemas — and the proposers' op
alphabet is itself property-derived by `catopt_discovery.vocab`, not
hand-listed) → **verify** each candidate (numeric oracle,
`sink.verify` on lowered modules, derivability oracle) →
**measure** (fires / cost deltas / cert replay / closure ratio) →
**rank** → **emit** (`--emit-admission` writes the `R(...)` block
plus generated tests as a reviewable patch — the emitted law is
bytecode-identical to the hand-written one in the held-out check).
Two shipped laws came out of this loop (`select_mul`,
`softmax_fold` — `silu_fold` came out of the coherence layer
above), and the loop audits itself:
adversarial witness generation (`catopt_discovery.gap_gen`)
produces bindings the corpus cannot — it falsified paying-but-false
candidates and the shipped matmul-family hole; the sqlite evidence
store (`--evidence-db` / `catopt_discovery.evidence`) caches verdicts
keyed by corpus × rules × code revision (~19× on a warm run) so
"what did we measure, and did it change?" is a query, not a
re-run.

**Minting — the object language grows (ADR 0007).**  The loop above
admits *laws over a fixed vocabulary*; the minting stack makes the
vocabulary itself a product of measurement, in four layers with one
admission gauntlet.  **Layer 1 — definitional atoms**
(`catopt_discovery.antiunify`): anti-unify recurring subgraphs, mint
a named atom whose semantics *is* its expansion.  Sound by
construction; measured **NULL on reach-expansion** — compression is
real (shorter spellings, provenance) but naming structure did not
expand reach on any measured workload (recorded in the ADR).
**Layer 2 — handler-interpreted atoms**
(`catopt_discovery.handler_synth`): a structured handler grammar
(`ew`/`contract`/`fold`/`fold_emit`/`fold_zip` — descriptors are
plain data, interpreted, never `eval`'d) proposes handler bodies and
verifies each fp64 against the spelled spec; the machine minted a
verified handler for the `om_decayed` element itself (8.9e-16) —
a pattern that previously required a hand-written executor.
**Layer 3 — law-theoretic atoms** (`lab/mint/atom_law_loop.py`): for an
atom, enumerate measured binding environments, run `auto_cond` /
`dspec_synth` to synthesize the laws it satisfies, certify via
`verify_law` — 25 certified rows on the shipped carrier atoms,
deliberate false twins rejected.  **Layer 4 — induced atoms**
(bounded-error lane, fenced: an approximate atom may never appear on
an exact law's RHS).

The composed loop (`lab/mint/atom_synth_loop.py`) runs need → handler →
laws → admission → pay probe end to end; on the corpus it minted
`scan_fold` (the selective-scan recurrence, 1.78e-15) with a
certified law set, and measured the honest null that the minted
*total* does not pay where the corpus *emits* the sequence.

**The admission story — `mul_unsq_nat`.**  The first law minted from
a *corpus op-tuple* rather than a hand-declared need: the census
saw `mul(unsqueeze, unsqueeze)` in DeltaRuleBlock's own `β_t·k_t`
write; the candidate `mul(unsqueeze U d, unsqueeze V d) =
unsqueeze(mul U V, d)` was proposed; `auto_cond`'s first guard
(plain broadcast equality) was **falsified by the adversarial
sweep** (212 bad accepts); the exact `flat-map-unsq` guard was
synthesized instead — 0 bad accepts, 0 misses; closure cost +18
enodes; it pays −12.27% on its source workload and ships in
`DEFAULT` with `MACHINE_SHIPPED` provenance.  Its apparent sibling
(`mul_unsqueeze_r_id`, the largest "blocked structure" the corpus
sweep found) was **correctly rejected at admission** — its real
sites are semantically unequal, and the guard declines all of them.

**Exposure — where discovery can pay at all.**  The screen
(`lab/screen/exposure_screen.py`) scores every corpus workload by the
FLOPs-weighted share of ops any law's LHS can name: the distribution
is **bimodal** — 81/321 workloads expose literally nothing (the
fused-op class: `nn.Conv*`, `nn.*Pool*`, `sdpa` spellings) and 140
score ≥95%.  Exposure predicts *reachability*, not *payoff*: a
corpus-wide discovery sweep (60 models) found 0 paying new laws
beyond `mul_unsq_nat` — the yield of fresh paying structure in the
exposed class is ~1.5%, and it concentrates where a model's op-tuple
is itself novel.

**The substrate beneath the syntax (ADR 0008).**  A prototype
combinator presentation (`lab/bridge/combinator.py`) re-expresses the IR
as morphisms of a monoidal category with explicit `dup` (Δ, sharing
as a morphism) and `loop` (bounded trace): 321/321 corpus workloads
round-trip structurally exact, zero ops refuse translation, and the
RetNet shared-state fold spells as `seq(pre, loop(step, K), post)` —
verified at 1e-7 — which is the carrier-atoms-become-definable
claim measured.  The open gate: `loop` structure is erased at
export, so folds must be *proposed by search*, never recovered by
the bridge.

## The kernel, and the platform next door

The calculus above is not only a catopt idea: it is implemented, as a
language *and* a set of engines, by the **handl** platform
(`handl-lang/platform` — one `Language` whose tables are data: `eval`,
`types`, `ast`, `transforms`, `lexer_rules`; engines for lexing,
parsing, execution ×3, typechecking and **rewriting**).  ADR 0010 adopts
its model — *one language, many interpretations*: eval, typecheck and
rewrite are interpretations of the same language, differing in the
scheme table they read, not three languages.  There is no bespoke VM
here, and no second type language.

**What catopt contributes** (measured, spike `plan 0029`):

| catopt | handl |
|---|---|
| the **language fragments** — laws as `TransformDef` rows, op shapes as `AstShapeDef` | the language model, the rewrite engine (traversals, traces, stats), typechecking, parsing, the VM |
| **host functions** for tensor ops (a tensor crosses by reference as an opaque `ForeignResource` pointer) | the value channel and the FFI seam |
| the **domain adapters** (`catopt-torch`/`onnx`/`cuda`) and the **evaluation** dimension | — |

The seam is **data in, terms out**: `catopt_discovery.handl` emits
handl-shaped rows (every row carrying an authoritative `catopt:` record,
which handl's serde ignores, so the bridge is lossless), and handl
assembles a `Language` from them with `language_from_dir`.  Measured
end-to-end: a catopt law lands in `lang.transforms` with its fields
intact, and its condition deserializes as `And([KindAt, ChildCount])`
(`project/retros/handl-q0-language-as-data.md`; the handl-side test is
`crates/handl-kit-data-canonical/tests/catopt_fragments.rs`).

Two traps cost real time and are worth knowing: `TransformDef.when` is
**one** condition (a conjunction is spelled `And([...])`), and under
`serde_yaml` enum variants are **YAML tags** (`!Declarative`, `!Create`,
`!KeepPath`, `!KindAt`), not JSON maps.

**Which representation is canonical** (ADR 0011): the calculus for
*laws and rewriting* — a law is a data row; the op-graph IR for
*execution* — the sinks, the cost model and the certificate replay
speak it, and tensors are host resources.  The `catopt:` extension field
is the authority, so the two cannot drift silently: a drift is a failed
round-trip, which is testable.

**What this is not (yet).**  The tensor channel is *available* but
unused: Q1a found `TaggedValue::Resource(Rc<ForeignResource>)` — an
opaque pointer with a finalizer, `Rc`-held and not `Send`/`Sync` — so a
tensor crosses by reference, but no run yet demonstrates a tensor op
through exec (Q1b).  And the direction is decided (Q3: catopt embeds
handl as an *optional* peer, the `catopt-native` pattern), not built.

## Where this lives in the code

| Step | Code |
|---|---|
| Signature | `BlockSig` / `InputSig` / `WeightRef` (`catopt_orchestrator.morphisms`) |
| Axioms | `catopt_core.laws` (`tensor.py`, `scan.py`); morphism laws' `check` / `derive`; the axiom/lemma marks `Rewrite.derivation` / `kind` + `tags.REDUNDANT` |
| Match (lift) | `lift_graph`, carrier recognition (`lift_scan_to_applyd`) |
| Saturate | `EGraph` |
| Extract | `extract_best`, `backend_cost` (measured pricing: `delivered_cost_for(profile, x=…)` / `executor_cost_for`) |
| Certify | `verify_certificate` |
| Evaluate | `catopt_core.features` (`ProgramFeatures`, `StaticProfiler`) · `SearchResult.frontier` · `PredictedCriterion`; `Meter` / `TargetProfile` for measurements (`scripts/calibrate_profile.py`) |
| Discover (meta) | `catopt_discovery.pipeline` (census → propose → verify → measure → rank → emit) · `catopt_discovery.vocab` (property-derived op alphabet) · `catopt_discovery.gap_gen` (adversarial witnesses) · `catopt_discovery.emit` (admission patch) |
| Mint (meta) | `catopt_discovery.antiunify` (layer 1 — definitional atoms) · `catopt_discovery.handler_synth` (layer 2 — handler bodies) · `lab/mint/atom_law_loop.py` (layer 3 — law synthesis per atom) · `lab/mint/atom_synth_loop.py` (the composed loop) |
| Screen / sweep | `lab/screen/exposure_screen.py` (law-LHS exposure census) · `lab/screen/discovery_scale.py` (budgeted corpus sweep + novelty pass) · `lab/screen/exposed_discovery.py` (per-model discovery driver) |
| Substrate (prototype) | `lab/bridge/combinator.py` (`BASIS`, `dag_to_comb` / `comb_to_dag`) · `lab/bridge/combinator_probe.py` (round-trip + `loop`-spell gates) |
| Backends | `catopt_torch` (`TorchSource`/`TorchSink`) · `catopt_onnx` (`OnnxSource`/`OnnxSink`, ADR 0006) |
| Coherence (meta) | `catopt_discovery.coherence` (`--emit-basis`) · `catopt_discovery.coherence2` (stratification / reach / mediators) · `catopt_discovery.evidence` (sqlite verdict store) |

## See also

- `project/adrs/0002-categorical-re-expression-thesis.md` — the
  thesis this pipeline implements.
- `project/adrs/0003-evaluation-is-an-independent-dimension.md` — the
  evaluation dimension this pipeline appends.
- `project/adrs/0007-semantic-atom-minting.md` — the four-layer
  minting stack (with its measured L1 null recorded).
- `project/adrs/0008-combinator-presentation.md` — the substrate
  direction (bridge gates 1–2 measured PASS; gate 3 open).
- `project/plans/0014-certified-re-expression.md` — the staged
  path from proofs to delivery.
- `project/retros/axiom-lemma-split.md` and
  `project/retros/law-coherence-depth2.md` — the measured kernel
  and the depth-2 coherence structure.
- `project/retros/automated-admission.md` and
  `project/retros/law-gap-targeted-gen.md` — the emit step and
  the adversarial oracle half of the discovery loop.
- `project/retros/mul-unsq-nat.md` — the first corpus-minted law's
  admission evidence; `project/retros/layer3-loop.md` — the
  law-synthesis tally; `project/retros/exposed-math.md` and
  `project/retros/discovery-scale.md` — the exposure screen and the
  corpus-wide yield measurement; `project/retros/handler-proposals.md`
  — the handler-grammar results; `project/retros/atom-synth-loop.md`
  — the composed loop's funnel; `project/retros/birefnet-bounded-lane.md`
  — the certified negative on the precision lane.
- `project/adrs/0010-one-language-many-interpretations.md` — one
  language, many interpretations; handl *is* the kernel.
- `project/adrs/0011-canonical-representation.md` — the calculus is
  canonical for laws, the op-graph for execution.
- `project/retros/handl-q0-language-as-data.md`,
  `handl-q1-tensor-crossing.md`, `handl-q3-integration-direction.md` —
  the boundary spike's measurements.
