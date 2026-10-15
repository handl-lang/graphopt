# Research lab notebook

*The honest record of how this was found.  Chronological and exploratory,
not a polished report.  Plain prose, first-person plural.  The point of this
document is that a future collaborator reads it and does not repeat the
mistakes.  Measured numbers are given where they matter, each traceable to a
retro under `retros/`, a decision under `adrs/`, or a plan under `plans/`.*

*Read this with the paper draft (`../docs/paper_draft.md`) open: the paper
says what we claim; this says how we got there and what it cost.*

---

## 0. How to read this

We kept a retro per work item from early on, and the retros are the raw
material — 127 of them at the time of writing, each with the measurement,
the negative, and the honest limits.  This notebook is the spine through
them: the questions in the order we asked them, the doors that closed, the
process lessons that cost real time, the reframings, and what is still open.
Where a retro is the authority we name it; where we are summarising across
retros we say so.

Two conventions we adopted and now hold to.  First: **a negative is a
result** — a probe that refuses to derive something, with the gap named, is
written up exactly like a probe that succeeds.  Second: **no number without
its artifact** — every figure below has a retro, baseline or tool behind it.

---

## 1. The questions, in the order we asked them

### 1.1 2025-09 — "Is there structure in the weights themselves?"

The founding bet was weight-space compression: if trained weights hid exact
structure (duplicates, shared subspaces, symmetries, generators), the
equivalence machinery could shrink storage *losslessly*.  This is the
original motivation and it was the first thing we measured, on llama2.c
`stories15M` (15 M params) and `stories110M`.

Every hypothesis class we could name, measured (`retros/measure_weights.py`;
`adrs/0001-weight-space-structure-falsified.md`):

- **Phase 0** — Toeplitz/displacement rank: ~270–288 of 288, full rank.  No
  exact low-rank structure.
- **Phase 0b** — the structured-algebra family: Kronecker-sum beat SVD by
  only 15–31 %; H-matrix, sparse, monarch/butterfly, INR coordinate-fit all
  dead or inconclusive.  The one live signal was the token embedding:
  **rank 19 at 95 % Frobenius energy**, ~15× compressible.
- **Phase 5** — real perplexity on TinyStories (baseline ppl 5.03).  The
  embedding's top-19 directions are **not** the ones next-token prediction
  needs: every structural rank (r192 → ppl 214) killed quality.  Only int8
  RTN survived (4×, ppl 5.16).
- **Relational** — cross-layer bitwise dups / shared subspace / Procrustes
  alignment: **0 hits, full column rank, 0 aligned**, on *both* checkpoints.
- **Symmetry** — equivariance `WG=GW`: residual ≈ √2 (the random value).  No
  symmetry.
- **Polynomial** — low-degree `p(W)=0`: minimal polynomial = full degree
  (288 distinct eigenvalues).
- **Learned displacement** — rank(`AW−WB`) over structured `A,B`:
  displacement rank ~280/288 — full.

**The door closed** (ADR 0001).  Trained checkpoint weights are
information-dense at the bit level; no emergent exact structure exists to
exploit.  The lesson that outlived the measurement: **a norm bound does not
predict task quality.**  `‖ΔW‖` bounds do not predict `Δppl`.  Certified
compression is real but certifies the *wrong quantity* for weight space.
What survived is only the *architectural* corner — tying, materialized GQA
replication, dead params, merged adapters (measured 0–75 % by archetype,
`REPORT.md` §10.2).

This is the first dead end and it set the tone: we killed our own founding
bet with our own measurements and wrote it up as ADR 0001.

### 1.2 2026-09 — "What is the project actually about?"

With weight-space closed, the capabilities had grown independently —
carrier monoids, non-local lifts, the morphism engine, bounded rewrites,
exact weight sharing — each documented in a different place and each
justified on its own.  The pitch still led with "compression with a receipt"
after compression had been closed.  The docs were drifting.

So we wrote down one thesis and made every capability answer to it
(**ADR 0002**): *a tensor program is a morphism of a category; optimization
is search in the quotient of the free category by the laws of the structures
the program instantiates.*  Op-level compilers rewrite *generators*; this
rewrites *morphisms up to structure*, which is why it reaches programs they
provably cannot.  Six binding rules followed — name your structure; the
certificate is the artifact; construction primary, saturation fallback;
exact and bounded are one object; scope claims to structure.

The rule that caused the most subsequent trouble was Rule 3: *"Directed
search, not learning.  RL is a non-goal."*  That was right for **reach** —
structure is what makes search beat enumeration — but it wrongly foreclosed
**policy** (choosing which rewrite to fire next).  ADR 0003 amended it.

### 1.3 2026-10 — "Is evaluation a separate thing, and is RL allowed?"

**ADR 0003** split the pipeline into four independent dimensions —
semantics / search / evaluation / execution — each answering exactly one
question, with explicit ports.  It amended Rule 3 to allow a *learned
selector* over the structure while keeping the certificate constraint.  The
central invariant it protects: **feasibility is hard, performance is soft**.
`supported_ops` is a hard bound (an unlowerable form prices at `+inf`);
a performance model is a soft ranking that may never prune a certified
alternative.

The reason RL became allowed, stated plainly: the certificate is a **hard
filter** (accept/reject), never a soft reward.  Because the learner cannot
launder a wrong answer into a reward, it is free to explore aggressively.
This is the formal/RL interaction the paper now leads with.

**ADR 0004** ("abstraction as a legal move") opened the move set: constructed
objects — folds, bridges, carriers — are admissible *data*, not pre-approved
Python.  It also recorded the first measured negative about learning: asking
a learned policy to invent laws *directly* was **~10× worse** than
enumeration (`retros/learned-proposal`; the `meta_game` probe).  Direct
neural law invention does not scale; *structured construction* does.

**ADR 0005** ("the open game") made the frame explicit: every rule, object,
vocabulary, runtime and pricing inside the optimizer is machine-made data
admitted by anchored machinery — and the only fixed pieces are the
**substrate** (torch/GPU/Triton), the **correctness oracle**, and the
**measured objective** (wall-clock, memory).  A self-minted objective is a
rigged race; a machine that could redefine equivalence voids every
certificate.  Everything else is inside the game.

---

## 2. Dead ends and closed negatives

These are the results we are most glad to have.  Each one saved time later
and each one is a measurement, not an opinion.

### 2.1 The anti-unification reach-null (ADR 0007 layer 1)

ADR 0007 proposed four layers for minting new *meaning* into the object
language.  Layer 1 — definitional atoms by anti-unification (the
Babble-style library-learning move) — was the one we expected to pay.

It mints real atoms: provenance-carrying, unfoldable, verified fp64
end-to-end, and the compression is real (a deq winner is 12 ops spelled vs
26 unfolded).  But on all three workloads (delta×6, deq×6, retnet T=64/K=4)
the cheapest extractable member and the class count were **identical with
and without the atoms**.  Naming structure did not expand reach.  The strict
arm is *strictly worse* under tight `max_nodes` (intro members consume
enodes with no pay).  `fold_ruleset` found zero targets: minted motifs live
in exports, not in law outputs.

Per ADR 0007's own falsification clause #1, layer 1's reach-expansion claim
is **measured negative** on these workloads.  The machinery stays as
compression tooling (shorter spellings, provenance), not a reach mechanism.
The retro is the ADR's own measurements section; the tool is
`lab/probes/anti_unify_probe.py`.

### 2.2 The atom pay-null (the composed loop)

The composed loop (`lab/mint/atom_synth_loop.py`) runs need → handler → laws →
admission → pay probe end to end.  It minted one full semantic atom —
`scan_fold`, the selective-scan recurrence `h_t = a_t·h_{t-1}+b_t`, verified
at **1.78e-15** — with 5 certified primitive laws, plus law sets for
`cumsum`/`cumprod`/`logcumsumexp`.  It is a real end-to-end mint.

And it pays **nothing**.  **0 certified laws fire on real terms.**  The
minted atom computes the fold's *total*, while the corpus *emits* the
sequence per step.  The paying sibling (`fold_emit`) sits **>200 k deep** in
the round-robin enumeration; it verifies at 6.7e-16 when reached directly —
so the gap is enumeration *ordering*, not expressiveness.  The corpus's
actual `cumsum` usages (`cumsum(mul(x,x))`, `cumsum(elu(k)+1)`,
`cumsum(mul(unsq,unsq))`) all live in the guarded-out region: the guards
that made the laws certifiable are exactly what correctly stops them firing
on the real sites.

This is ADR 0007's falsification criteria working as designed: **truth and
pay are measured separately, and they were disjoint here**
(`retros/atom-synth-loop.md`).  The retro produced the ordering fix
(`feat/enum-order`) as a follow-up.

### 2.3 The BiRefNet bounded negative

Question: does the precision lane (`error_budget=`) buy headroom on a real
pretrained model where the exact lane was already FLOP-neutral?  Model: the
real `ZhengPeng7/BiRefNet` HF checkpoint, 220.18 M params, Swin-v1-L + an
ASPPDeformable decoder (`lab/probes/birefnet_bounded_probe.py`).

The bounded offers *exist*: 20 sites at budget 1e-2, 56 at 3e-2, certified
bounds inside budget.  The bound gate stayed honest.  And **nothing
delivered**: only **18 of 2304 rows** dedup at ≤9.9e-3 (0.78 % of the
weight), so the gather/re-expand/bias-add wrapper prices **1.32× the dense
member**.  Launch-aware pricing correctly declined every offer.  The fp32
gate stayed bitwise-clean (`fresh_rel = 0.0` on every cell).

**Verdict: the precision lane does not pay on BiRefNet — not because the
bound machinery failed, but because the real structure is too thin to
price.**  This is a *certified negative*: the machinery offered, the gate
refused, and the refusal was right.  Side findings worth keeping: 43
`nn.BatchNorm2d` modules (so conv+BN fusion targets do exist — `retros/birefnet-bounded-lane.md`,
`retros/weightspace-fold.md`), and `deform_conv2d` exported with **no
lowering binding** — a real coverage gap, since fixed.

### 2.4 The enumeration-vs-learning ledger

We measured the learned path against enumeration repeatedly, and the ledger
is mixed — which is the honest result.

- **Direct law invention by a learned policy: ~10× worse than enumeration.**
  Falsified, not reopened (ADR 0004).
- **The meta-arena:** a cold linear REINFORCE player beats the scripted
  playbook (−0.03 vs −0.10 over 40 held-out boards) and produces 3 certified
  structural wins — *but only after* the `claim(tag)` option move fused the
  composed `declare → handle` line into one refereed action.  The wins ride
  **sampling variance, not learned preference** (`op:claim`'s weight sits
  ≈+0.04).  Action-space structure, not the learner, was the fix
  (`retros/meta-arena-player.md`).
- **The contraction player:** beats `opt_einsum`'s randomised greedy at
  equal wall-clock at n=40 (0.89–1.01× across three seeds), 0.51–0.90× the
  deterministic greedy.  Real, but small; the headroom is *search cost*, not
  quality (`retros/contraction-player-shipped.md`).
- **The learned prior:** ranks held-out law candidates at AP 0.133 vs 0.045
  heuristic, ~2× more useful laws per episode (`adrs/0005`).
- **The RL mixture collapse:** a diagnosed reward-scale bug was fixed (a
  per-episode standardized advantage), but the mixture still collapses —
  a shared-net **winner-take-all** under a sparse reward, not a scale bias
  (`retros/stage7-multifamily-results.md`).

The pattern we extract: **learning beats baselines where paying moves are
dense; it fails where the paying move is composed or sparse.**  That is a
board-design lesson, not a learner-capacity one.

---

## 3. Process lessons that cost real time

### 3.1 Scoped gates missing cross-cutting regressions — three instances

We run a heavy gate suite (full pytest, `ty`, `ruff`, `vulture`,
import-linter, bandit, semgrep, a radon ratchet, 100 % coverage).  Three
times a gate went green while a real regression lived *outside its scope*.
The pattern is the same each time: the gate's scope (a package, a corpus, a
feature contract) was narrower than the regression's scope.

**(1) The unconsumed ports.**  Four evaluation ports (`Profiler`,
`PerformanceModel`, `Policy`, and the `ProgramFeatures` seam) shipped typed,
tested and green — and **no code in `packages/` consumed them**.  Every gate
passed, because *no gate catches an unconsumed port*.  Coverage counts lines
executed by tests, not lines reached by the pipeline.  The fix was a rule,
now in ADR 0003: **"a port is not done until something reaches it"** —
"wired" is a deliverable of its own.

**(2) A correctness fix silently invalidated a learned policy.**  A
view/fold-aware traffic-accounting fix in `catopt_core.features` (a fix that
was *right*) altered the state input of an RL policy and re-introduced a
family collapse in a policy whose documented benefit had previously been
recorded — **with no change to the RL code**.  The features tests passed
(features changed as intended); the RL tests passed (the RL code was
untouched); the *documented benefit* was silently gone.  The cause was found
only by an audit for a different bug class and by reverting `features.py` to
an earlier commit and reproducing the old table exactly.  The rule that
came out of it: **a policy is coupled to the semantics of its features** —
any change to `ProgramFeatures` must be treated as a change to every policy
trained on it (retrain, or version the contract).  ADR 0003.

**(3) Shipped laws unsound off-corpus.**  The weight-merge law
(`x@a + x@b → x@(a+b)`) rewrote a real, evaluable term into a numerically
different one (max diff **12.83**).  All four members of the equivalence
class shared the hole.  `select_mul` was latently unsound (broadcasting
`mul` along the *selected* axis changes the result).  **The corpus could
never surface either**: every real exported site was in the true region
(the 13 real `select_mul` sites all had `shape(u)[D] == shape(v)[D]`).  The
corpus-scoped gates — including `sink.verify`, which only ever saw the true
region — were green.  Both were caught only by an *adversarial witness
generator* that synthesizes bindings the corpus does not contain
(`retros/matmul-unsound-fix.md`, `retros/select-mul-broadcast.md`).

The general lesson, now a rule: **"verified on all observed sites" is not
"sound."**  The adversarial layer is the part of the referee that keeps
earning its keep, and every shipped law that lacks a `cond` should
eventually get a synthesis pass over its metavar bindings.

### 3.2 Three host OOMs and what caused them

The dev host is an 11 GB / 16-core box (RTX 2050, 4 GB VRAM).  Memory is
the binding constraint on this project, and we killed the host three times.

**(1) The quartic `dspec_synth` sweep.**  The conv+BN fold probe's depth-3
`dspec_synth` sweep over the fold's seven-attr bank is **quartic in leaf
count**.  It OOM-killed the process (measured: `MemoryError` under a 4 GB
cap after ~17 s; an OS OOM-kill without one) — and the reboot it caused
meant the probe work had to be salvaged from the wreckage (commit `40ea395`,
"salvaged from the OOM reboot").  The fix routes every fit through
`dspec_audit.bounded_synth` (same candidate space, deepest level streamed
with early bail under a wall-clock budget).  The general class:
**unbounded candidate sweeps**.  `retros/weightspace-fold.md`.

**(2) The arena oracle's unbounded viewed-binding enumeration.**  The
oracle's `_synth_bases` materialized one `_LazySeq` per binding — a 9-ary
kernel fold over a deep view chain opened **~10^13 rows** and OOM-killed
*every run*.  Fixed by capping at `_MAX_VIEWED_BINDINGS`, and by running
rows subprocess-isolated with a **7 GB `ulimit` + 10-min timeout** so a
wedged row is *recorded* rather than silently killed.  The dead rows (one
OOM, one timeout) are kept in the results table honestly.
`retros/arena-player.md`.

**(3) The `pytest -n auto` inductor fan-out.**  The suite is single-process
by design: the torch.compile-heavy tests spawn an inductor pool of **16
compile workers per pytest process**, so `-n N` multiplies the fan-out to
~`16·N` subprocesses.  On the 11 GB/16-core host, `-n auto` **OOM-froze the
machine**; even `-n 3` drove memory to **10.2 GB** near the suite's tail.
The warning is now a block in `AGENTS.md` and a commit (`6c00529`).

A fourth, related datum: the discovery tools carry their own budgets because
of this host — a 240 s `SIGALRM` per-model wall clock, a 3000 s total sweep
budget, and a **1.5 GB `MemAvailable` abort floor** (`retros/discovery-scale.md`);
and plan 0024 records "no comm/assoc in default saturation (measured OOM at
bounded budgets on the 11 GB host)."  The budget discipline is not caution
for its own sake; it is the scar tissue.

### 3.3 The boot-time checklist the recurring agent deaths produced

We did not lose work to the OOMs only; long-running agent sessions died
mid-task (host freezes, session gaps), and we lost uncommitted work more
than once.  What emerged — partly written down in `AGENTS.md`, partly in the
tools' conventions — is the checklist we now run at the start of any
non-trivial session.  There is no single committed checklist file; this is
the reconstruction, with the source of each item:

1. **Read `AGENTS.md` first — it is the boot document.**  It names the
   package layout, the "no `catopt` façade" rule, the 11 machine-admitted
   laws and the authorship split, and the full gate list.  A change to any
   default rule changes extraction behavior for every caller, so **run the
   full suite, not a slice, when touching the law bank**.
2. **Never run the suite under `pytest -n`.**  (OOM #3, above.)  Use a
   scoped `coverage run -m pytest tests/test_<file>.py` for iteration; the
   full coverage gate is a commit-time check (~20–25 min serial).
3. **Budget every sweep.**  Any tool that enumerates candidates gets a
   wall-clock budget, a memory abort floor, and (for row-parallel work)
   subprocess isolation with a `ulimit` + timeout.  (OOMs #1, #2.)
4. **Commit early and often, per work item.**  The retros and the
   `docs/tutorials/course/`-style single-path commits exist so a reboot
   costs one work item,
   not a session.  "Salvaged from the OOM reboot" appears in the log twice.
5. **Write the retro before you close the task.**  A negative with a named
   gap is a deliverable; an unrecorded negative is a future repeat.
6. **Do not add suppressions to make a gate pass.**  No `# type: ignore`,
   `# noqa`, `# nosec`, `# pragma: no cover`; fix the code or add a
   justified, documented config entry.  Ratchets tighten, never widen.
7. **Check what the gate does not cover.**  The unconsumed-port and
   features→policy lessons (§3.1): green does not mean wired, and green does
   not mean the documented behavior still holds.

If a future host is larger, items 2–3 relax; items 1, 4–7 are about
epistemics, not RAM.

---

## 4. The reframings

Two reframings matter enough to write down, because both were corrections of
how we *talked* about the work rather than what the code did.

### 4.1 Rediscovery vs discovery

The strongest structural result — the RetNet carrier — is a
**rediscovery**.  The extracted program is numerically identical
(**9.8e-9 fp64**) to the paper's published chunkwise duality: "the extracted
program IS the paper's chunkwise algorithm, not an approximation of it"
(`retros/retnet-external-baselines.md`).  The projection-pairing laws on
llama2.c rediscover `MergedColumnParallelLinear`/`QKVParallelLinear`.  We
stopped calling these "discoveries."  The contribution there is automatic
discovery + certified equivalence + per-shape selection of a *known*
linearization — which is a real contribution, just not the one the word
"discovery" implies.

The line between rediscovery and creation is precisely where ADR 0007 draws
it: the discovery machinery can mint new *spellings* (law instances, guards,
derives, paying forms) and new *operators-as-implementations* (`genkernel`
binds a measured elementwise pattern to a generated kernel) — but minting a
new **semantic atom** (an op whose *meaning* did not exist before) is the
harder thing.  That is the line the minting stack was built to cross, and the
one paying crossing so far is `mul_unsq_nat`.

### 4.2 "Not a PyTorch optimizer"

For most of the project the self-description was "a PyTorch graph
optimizer."  That was wrong on two counts.

First, **backends are adapters, not the core abstraction.**  `catopt-core`
is torch-free and `import-linter` enforces it; a new backend implements the
`Sink` port.  We *tested* this by building a second, non-torch backend (ONNX,
both directions) and measuring the boundary: **zero edits under
`catopt-core`**, fp64 round-trip 1.11e-16 through `onnxruntime`
(`retros/onnx-package.md`, ADR 0006).  PyTorch is *one application* of the
engine.  The boundary is measured, not asserted.

Second, and more importantly, the thing we built is a **general engine**:
Catopt discovers better solutions by combining mathematical reasoning,
verified transformations, and learned search.  Optimization is a game over a
structured space of equivalent solutions; RL navigates it; the formal layer
preserves correctness; applications and backends are replaceable interfaces.
That is the reframing the paper now leads with, and it is the one that
finally made the three pillars — mathematical discovery, categorical
soundness, learned search — line up with the three measurable questions
(novelty, soundness, value).

---

## 5. What is still open

### 5.1 The law families (plan 0025, section D)

The measured boundary: models whose math is already fused expose nothing to
the op-algebra laws.  Three families are named and untried, in cost order:

1. **Precision** — the bounded lane is built and was a certified negative on
   BiRefNet for a measured reason (thin dedup).  Next target: a model where
   the dedup is *thick* (a wide-MLP checkpoint), and weight-space members
   rather than row-gather ones.
2. **Data movement** — hoisting invariant structure (Swin masks recomputed
   per layer, identical).  Compete with Inductor here at your peril; measure
   before building.
3. **Weight space** — conv+BN fusion targets exist (43 `BatchNorm2d` modules
   measured in real BiRefNet).  Per the no-hardcoding rule the work is
   **derive-or-refuse**: feed measured `(W,b,μ,σ,γ,β) → (W′,b′)` bindings to
   `dspec_synth` and let the machinery attempt the fold spec; whatever it
   cannot express becomes a named grammar gap.  The probe already ran this:
   the fold's *structure* is derivable given a `scale` leaf, but the cond
   DSL has **no `sqrt`/`rsqrt`** and the derive value domain is scalar
   (`int | float`), so a per-channel `scale` is unspellable today.  Two
   ordered blockers, both named (`retros/weightspace-fold.md`).

### 5.2 ADR 0008 gate 3 — does search reach `loop` forms?

The combinator presentation (ADR 0008) makes carrier atoms *definable* as
`loop` terms rather than primitives, and gates 1–2 passed: 321/321 corpus
workloads round-trip structurally exact (zero ops refuse translation, mean
blowup 2.61×), and the RetNet shared-state fold spells as
`seq(pre, loop(step, K), post)` verified at 1e-7.  **Gate 3 is open**: `loop`
structure is *erased at export*, so a fused recurrence translates fine but
the fold must be **proposed by search**, never recovered by the bridge.
Whether search *reaches* `loop` forms is the remaining substrate risk, and it
decides whether the substrate is a discovery surface or only a presentation
win.  Plan 0025 section C states the falsification: a negative closes the
substrate's *discovery* value, and the presentation value (uniformity,
explicit Δ, the eliminated DAG-vs-tree bug class) must then carry the ADR
alone.

### 5.3 The enumeration-ordering blocker

The one measured gap between the minted atom and pay (§2.2): the paying
`fold_emit` descriptor is >200 k deep in the round-robin.  The fix
(`feat/enum-order`) is in flight.  The falsification is explicit: if the
canonical seed is reached and the laws *still* fire nowhere real, the
pay-null is semantic — record it and stop chasing the atom.

### 5.4 The form list and the mint target

Plan 0024 measured the form level and the honest reading was narrow: over
12,000 candidates the machine admitted **7 paying laws from forms outside the
shipped 27**, but each is a *composite re-spelling* of two shipped
identities.  "The form list is not the whole grammar, but the forms it lacks
are shallow."  The open question moves to part B — whether a *minted atom*
(vocabulary, not a re-spelling) can pay where no composition does.  Part B's
own measurement inverted the plan's assumption: the machine's need-detection
backlog named `mod` (6 rules unblocked), not `pow` (0 rules) — **the plan's
guess about which op was missing was wrong, and the machinery said so.**

### 5.5 Scale validation and executor leanness

Two things the paper explicitly defers: production dimensions (d ≥ 1024,
B ≥ 8, a datacenter GPU, trained linear-attention checkpoints) — because the
dev-box magnitudes do not extrapolate — and executor leanness, because
per-leaf dispatch is the loss driver and the FLA gap is now precisely
attributed there (fused fp32 Triton vs fp64 torch GEMMs + a materialized
score tensor).

---

## 6. The measured numbers that matter (quick index)

Kept here so a future collaborator does not have to reconstruct them from
127 retros.  Each is sourced.

| Number | Value | Source |
|---|---|---|
| Weight-space exact structure | absent (full rank everywhere, both checkpoints) | `adrs/0001` |
| Embedding compressibility | rank 19 @ 95 % energy (~15×); **quality-dead** | `REPORT.md` §3, §10 |
| `mul_unsq_nat` on its source workload | −12.27 % (992 496 → 870 722); +18 enodes | `retros/mul-unsq-nat.md` |
| Naive guard bad-accepts (falsified) | 212 | `retros/mul-unsq-nat.md` |
| Synthesized guard | 0 bad / 0 miss | `retros/mul-unsq-nat.md` |
| Rejected paying sibling | −14.3 %, 12 models / 22 sites, unequal | `retros/discovery-scale.md` |
| Discovery yield (exposed corpus) | ~1.5 % (1 find in 68) | `retros/discovery-scale.md` |
| Exposure bimodality | 81/321 expose 0 %; 140 ≥95 % | `retros/exposed-math.md` |
| Machine-minted `om_decayed` handler | 8.9e-16 fp64 | `retros/handler-proposals.md` |
| Layer-3 law synthesis | 25 certified / 4 rejected / 0 refused | `retros/layer3-loop.md` |
| Composed-loop mint | `scan_fold` 1.78e-15; **0 fires on real terms** | `retros/atom-synth-loop.md` |
| Anti-unification reach | null on all 3 workloads (12 vs 26 ops) | `adrs/0007` |
| Weight-chain fold | 14.92× vs Inductor (17.40× vs eager) | `bench/baselines/reassoc_scale.json` |
| Spelled RetNet T=2048 | 5.0–5.9× vs eager; 6.2–7.4× vs compile; 7.30× memory | `docs/results.md` |
| Carrier below crossover | 0.51× vs eager at T=512 | `docs/results.md` |
| Held-out zoo | 11/18 beat `torch.compile` | `retros/gen-kernels.md` |
| Joint fwd+bwd | +13 % vs autograd (9.5e-07) | `retros/joint-training-graph.md` |
| Precision lane on BiRefNet | certified negative; 0 delivered; 1.32× worse | `retros/birefnet-bounded-lane.md` |
| ONNX round-trip | 1.11e-16 fp64; zero core changes | `retros/onnx-package.md` |
| Combinator bridge | 321/321 round-trip; mean blowup 2.61× | `adrs/0008` |
| Unconsumed ports | 4 ports green, 0 consumers | `adrs/0003` |
| Features fix → RL regression | documented benefit lost, no RL change | `adrs/0003`, `retros/stage7-multifamily-results.md` |
| Host OOMs | quartic `dspec` sweep; arena 10^13 rows; `pytest -n` fan-out | `retros/weightspace-fold.md`, `retros/arena-player.md`, `AGENTS.md` |

---

## 7. Closing note

The most useful thing in this notebook is probably §3.  The measurements in
§6 are reproducible; the process lessons are what cost us days.  If you take
one rule away: **the gates tell you what they cover, and you must ask what
they do not.**  Every one of the three cross-cutting regressions in §3.1 was
green under every gate we had.  We found them because we built an adversary
that lives *outside* the corpus, an audit that asked what the feature
contract silently carried, and a rule that a port is not done until
something reaches it.

And the second rule: **write the negative down.**  The weight-space
falsification, the reach-null, the pay-null, the bounded negative — each one
is a retro that stopped someone (us, later) from re-running the same
experiment hoping for a different answer.
