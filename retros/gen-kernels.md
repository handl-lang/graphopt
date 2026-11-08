---
title: 'Retro — generated kernels: lowerings the search builds'
type: retro
date: 2026-10-14
---

# Retro — generated kernels: lowerings the search builds

Plan 0022, phase 6.  The product question the night session could
not answer: *can the tower produce a program faster than every
entry in its own handler table — not just a new spelling?*
Answer, measured: **yes, where fusion pays; honestly no where it
doesn't.**

## The machinery (genkernel.py)

No new moves — novelty enters through the *handler table*:

- `gen_handlers(terms)` mints `{tag: {pattern, kernel, args}}`
  for each elementwise subterm of the program not already named
  by the shipped `HANDLERS` (concrete-pattern equality).  The
  board's vocabulary literally grows with lowerings.
- `gen_bindings` produces each kernel's torch `Binding`: a
  callable that evaluates the spec body over its metavar leaves,
  `torch.compile`'d — the delivered module runs a fused inductor
  graph, generated code, not a table entry.
- `gen_sink` extends a private `OpTable` — gen ops appear in
  `supported_ops` (the feasibility bound) and lower like any op.
- `novel_cost` discounts novel-headed members — the extraction
  prefers the invented form at ties.
- `play --domain gen`: torch modules spelling fusion patterns
  (mulrelu/mulgelu/mulsoftplus/multanh/bigfuse + a stock mlp)
  played under the generated handler table; `gen_probe` plays a
  claim, delivers through the gen sink, verifies, and times both
  sides.

## The soundness fix this forced

Specs in the arena are *skeletons* — `_canon_spec` strips op
names along with leaf names (`('X1','X2',('X3','X2'))`).  The
shipped `claim`/`handle` checked coverage and occurrence at that
level, so `mul(x,relu(x))` and `mul(x,sigmoid(x))` were
indistinguishable: a skeleton collision could offer a dead claim
*and* — worse — pick a corpus spec concretely different from the
handler's pattern, binding mismatched semantics.

Now: `_canon_concrete` (leaf metas normalised, op names kept),
`_concrete_match`/`_concrete_occurs` (full pattern-vs-subterm
matching), `MetaState.claim_tags` = `(tag, spec_index)` pairs
where the concrete pattern occurs AND a mined spec is
concretely alpha-equal.  `handle`'s coverage check uses the same
concrete comparison — the binding is sound by construction.
Kernel *support* stays a pricing question, not legality —
unsupported claims bind and feasible pricing holds parity.

## Measured (RTX 2050, CUDA, verified)

| case | baseline | gen kernel | speedup |
|---|---|---|---|
| mulrelu / mulgelu / multanh (8×64) | ~14 µs | ~36 µs | **0.38× — loss** |
| 2-op fusions at (256,512) | ~29 µs | ~34 µs | **0.86× — loss** |
| bigfuse — 6-op pointwise chain, (256,512) | 126.7 µs | 44.1 µs | **2.87×** |
| bigfuse whole-site (256,1024) | 314.1 µs | 51.3 µs | **6.1×** |

``--probe`` on the play CLI prints the same table per eval case —
the product question as a command.

Every delivered module verifies (max_abs ≤ 9.5e-7).

## The incumbent comparison — the real headline

``gen_probe`` now times ``torch.compile(baseline)`` alongside —
the honest bar is the incumbent, not eager:

| case | eager | compiled | delivered | vs compile |
|---|---|---|---|---|
| chain_mlp (x@A@B@C, d=64) | 38 µs | 87 µs | **22 µs** | **3.96×** |
| swiglu_mlp (llama-MLP shape, d=64) | 52 µs | 90 µs | **43 µs** | **2.11×** |
| bigfuse (6-op pointwise) | 125 µs | 24 µs | 30 µs | 0.82× |
| 2-op fusions (256,512) | ~29 µs | ~25 µs | ~29 µs | 0.85–0.88× |

Read: **where the win is structural** (the reassoc laws fold the
weight chain to one GEMM — Inductor keeps all three; the swiglu
block's spelled shape resolves to better form), the delivered
module *beats the incumbent* 2–4×.  Where the win is pure
pointwise fusion — Inductor's home turf — compile wins.

Honest incumbent check: `torch._inductor.config.freezing = True`
does NOT fold the param-param chain (76 µs vs delivered 22 µs) —
the 3.96× survives the strongest Inductor arm.

## The win that needed a generated kernel

`chainfuse` (d=512, batch 256): `x@A@B@C` + a 6-op pointwise tail.
The tail's only fused expression is a minted `gen_0` handler —
``claim(gen_0)`` declares ``mul·relu·…(X) → gen_0_k(X)`` where
``X`` binds the whole folded matmul.

| variant | time |
|---|---|
| eager | 314 µs |
| `torch.compile` | 242 µs |
| laws only (fold + spelled tail) | 181 µs |
| **laws + claim(gen_0)** | **90 µs** |

Decomposition: the fold alone beats compile 1.34×; the minted
fusion kernel roughly doubles it to **2.7× vs the incumbent**
(3.5× vs eager), verified max_abs 1.7e-05.  Laws could not reach
the fused tail — the win *required* the generated kernel.

Two design changes fell out:

- ``_fusion_sites`` now abstracts non-elementwise children as
  bound metavariables (``mul(y,relu(y))`` over ``y = mm(…)`` →
  ``("mul","X1",("relu","X1"))``) — previously it required the
  whole subtree elementwise and missed every site below a matmul.
- ``claim`` now declares the handler's *own pattern* as the object
  body (canonicalised so metavar names stay positional) instead of
  hunting a corpus spec by concrete equality — simpler and it
  eliminates the skeleton-collision soundness hole by
  construction.  ``claim_tags`` is now just the occurring handler
  tags.

## Joint fwd+bwd, delivered as a step

``training.joint_step(forward, grads, bindings)`` — the joint
board's runnable artifact: forward and every derived gradient
evaluated under ONE shared memo, so the activations the VJP
re-spells are computed once.  On the tanh-linear archetype
(CPU, verified vs autograd at 9.5e-07): **joint step 68 µs vs
autograd fwd+bwd 78 µs (+13%)**; without the shared memo the
same program costs 197 µs.  First measured runtime expression
of the cross-boundary-sharing claim.

## The honest read

- **The claim is real**: `claim(gen_0)` on `bigfuse` folds the
  *entire* six-op chain into `gen_0_k(x)` — a runnable module
  calling a kernel nobody wrote, 2.9× faster, verified.
- **The bound is real too**: at 8×64 / 2 ops the compiled-call
  overhead swamps the fusion (~2.5× slower); at (256,512) a 2-op
  fusion still loses ~14% — generated kernels only pay where
  kernel-launch count dominates.  The probe reports both regimes;
  nothing is overstated.
- Inductor already fuses pointwise chains *inside* a compiled
  graph — the win is vs the *eager spelling* (and vs any IR
  whose fusion vocabulary is fixed).  The novelty is that the
  **search produced the lowering**: the board grew a vocabulary
  entry, claimed it, and the delivered module ran it.

## Open

- ~~The probe claims the first offer; multi-claim composition
  (fuse several sites) is unmeasured.~~ **Measured 2026-11-07
  (T-010): composition already works.**  ``_probe_play`` claims
  every offer; on disjoint sites (the new ``twosite`` gen case —
  two fusion islands separated by matmuls) one extraction delivers
  ``gen_0_k`` + ``gen_2_k`` in a single verified module
  (max_abs 1.9e-06, 1.34x vs ``torch.compile``).  A claimed op can
  even feed another claim's metavar — the inner fusion's e-class
  member is what the outer kernel receives: chaining, not just
  co-existence.  Nested sites still collapse to the outermost
  single kernel, which is correct — one fused call beats two.
  Pinned by ``test_twosite_multi_claim_composes``.
- Elementwise-only by design — reductions/matmul bodies need a
  different codegen contract (attr binding, shape rules).
- The 0.38× regime wants a size guard: only offer gen claims
  where the fused body plausibly pays (a ProgramFeatures lever).
  **Measured 2026-11-07: neither cheap guard separates.**  Spec
  op-count doesn't — the losing 2-op fusions and the paying
  sub-claims inside bigfuse/chainfuse share sizes, and composed
  2-op claims can pay where the singleton loses.  The measured
  referee ``profitable`` is itself noisy at the margin: identical
  2-op cases flip keep/drop across seeds at reps=10–20.  A guard
  that drops *subsumed* claims would be wrong too — the inner
  claim is the chaining mechanism (see above).  Honest status:
  open; the cost model's ``novel_cost`` discount remains the only
  offer-side pricing, and it can't see launch overhead.
- Learned play on the gen board is untrained — arms are
  scripted-level only.


## The real corpus — zoo probe + measured referee + Triton tier

``play --zoo`` runs the full loop over the held-out model zoo (22
architectures): ``_zoo_case`` exports each workload, mints gen
handlers, and ``gen_probe`` plays claims → saturates → delivers →
verifies → times vs eager AND ``torch.compile``.

Two hard-won measurement-integrity fixes: dynamo's
``recompile_limit`` (8) silently degrades late compiles to eager —
raised to 256 inside the probes — and fp64 feeds need
``model.double()``.

``gk.profitable`` — the measured-cost referee: for each minted
handler, locate its concrete occurrence, eval the bound subterms,
time the spelled body vs the compiled kernel on real values, and
keep only claims that actually pay.  On the zoo it dropped nearly
every gen claim — 2-op fusions don't cover dynamo overhead at
d=16 fp64 — which is the honest finding: **op-count pricing can't
see what the clock sees.**

``gk.triton_bindings`` / ``triton_sink`` / ``hybrid_sink`` — the
codegen tier Inductor doesn't own: generated ``triton.jit`` source
written to real module files (``/tmp/catopt_genkernels/*.py`` —
inspectable artifacts), imported through ``importlib`` (keeps the
repo's no-exec invariant).  ``_triton_expr`` covers the full
ELEMENTWISE alphabet incl. libdevice calls; attr-carrying or
non-elementwise specs decline honestly.  Hybrid prefers Triton,
falls back to ``torch.compile``-wrapped bindings.

### Corpus result (RTX 2050, fp64, verified)

**11/18 deliverable workloads beat ``torch.compile``** —
LoRAAdapter 2.18x, WaveNetGate 1.51x, DeepEquilibrium 1.35x,
HighwayGate 1.45x, FiLMHead 1.46x — every one with ``claims=[]``:
the wins are the shipped + machine-admitted laws, not minted
kernels.  Losses are honest (TalkingHeads/CosineAttention 0.64x,
ExpertChoiceMoE 0.80x).  On real models the semantic rewrites
carry the wins; minted fusion pays only where launch-count
dominates (the synthetic chainfuse: 2.94x via the Triton tier).

chainfuse final table (256x512, d=512):

| variant | time |
|---|---|
| eager | 315 µs |
| ``torch.compile`` | 253 µs |
| laws only | 181 µs |
| laws + claim (compile-bound) | 95 µs |
| **laws + claim (Triton-bound)** | **86 µs — 2.94x vs compile** |

The Triton tier also beats the compile-bound delivery (95 → 86 µs)
— no dynamo guard-check overhead per call.

## Lever result — the offer-side cost guard lands (2026-11-07, T-010)

**Lever 3 landed: claims no longer offer unpriced.**  `_zoo_case`
already ran `genkernel.profitable` at intake; the synthetic
`_gen_cases` stream minted every fusion site *unfiltered*, so
`claim` offers included kernels that lose to their spelled body —
and `novel_cost`'s 0.5 discount let a losing member win extraction
on cost while losing wall-clock.

`_claim_cost_gate` (play.py) applies the same measured-cost
referee — `profitable(reps=20, kernel_for=_triton_kernel_for)` —
once per case inside `_gen_board`, before the handler store is
built: an unprofitable claim is never offered, never claimed,
never delivered.  Scope is every minted tier whose spelled body is
an honest denotation: `gen_*` pointwise, spelled `decayed_*`
(`matmul(mul(s,d),v)` evaluates through ordinary bindings), and
unrolled `scan_*` compose-trees.  Excluded by design: sequence-form
scans (spelled eval is the map, not the fold) and `om_decayed`-
headed specs (no `_IR_TO_TORCH` binding — unmeasurable stays
unpriced rather than silently dropped).  `_zoo_case` marks its
cases pre-gated so the board doesn't pay the referee twice.

**Measured (paired, same case instances, reps=100, RTX 2050):**

| case | ungated claims → speedup / vs compile | gated claims → speedup / vs compile |
|---|---|---|
| chain_mlp | `[gen_0]` → 0.69 / 1.68 | `[]` → **1.51 / 3.44** |
| mulrelu, multanh, mulsoftplus, mulgelu | `[gen_0]` → 0.82–0.94 | `[]` → ~same (offers were noise; delivery already declined them) |
| bigfuse | 6 claims → 3.68 / 0.81 | `gen_0,1,2` → 3.76 / 0.83 |
| chainfuse | 8 claims → 3.59 / 2.75 | `gen_0,1,3`+shipped → win retained (fresh-process check; one contaminated sweep dropped all — see caveat) |
| twosite | `[decayed_0, gen_0..2]` → 0.35 / 0.62 | `[]` → **1.05 / 1.72** |

Two honest findings inside the measurement:

- **The unpriced `decayed_0` offer was the twosite poison.**  A
  carrier claim that concrete-matched a `matmul(mul(s,d),v)` site,
  won extraction on the novelty discount, and delivered a kernel
  that loses wall-clock.  Gating it (spelled decayed bodies ARE
  measurable) took twosite from 0.62× to 1.72× vs compile — laws
  alone beat both the claimed and unclaimed spellings.
- **`profitable` noise is now offer-visible.**  The referee flips
  keep/drop at the margin (documented above); the gate inherits
  that — one interleaved sweep dropped chainfuse's whole claim set
  (delivered 1.37× vs compile instead of 2.75×) while a fresh
  process kept `gen_0,1,3` and the win.  Same referee, same
  caveat, new blast radius: an offer set is now only as stable as
  the measurement window.  A two-sample or margin-guarded referee
  is the obvious hardening, unmeasured.

Test posture: the composition pins (`test_chainfuse_claim_binds_
gen_over_mm`, `test_twosite_multi_claim_composes`,
`test_gen_probe_delivers`) bypass the gate — they measure
claim→deliver machinery, not offer policy; the gate itself is
pinned by `test_claim_cost_gate_filters_measurable_tiers`
(monkeypatched referee, scope + idempotence).

**Still open:** attr-bearing bodies (the `attr_mvars` machinery +
`concretize_at_site` exist post-T-026; what blocks *gen-minted*
attr specs is site-collector attr preservation — unmeasured) and
the reduction-body question (scan/decayed tiers bind; seq-scan
claims still offer unpriced by necessity — a carrier-aware spelled
referee is the missing piece, not more timing).

## T-010 remainder levers (2026-11-07)

Three levers attempted; one real bug found and fixed, two honest
blocks documented.

**Attr-bearing bodies — machinery fixed, corpus has no sites.**
Attr-carrying specs canon to *unhashable* dicts, and
`gen_handlers`/`decayed_handlers` deduped with `canon in skip` —
`TypeError: unhashable type: 'dict'` on the first attr-bearing
site (the scan tier already used the equality-based
`_canon_taken`).  Both handlers now dedup through `_canon_taken`;
attr specs mint (`gen_0`/`gen_1` on a synthetic
`mul(softplus(β=2,τ=5), sigmoid)` board), and `gen_bindings`
evaluates them verbatim — a real-tensor probe matched the spelled
torch computation at **max diff 0.0**.  The remaining codegen gap
is Triton-only: `_triton_node_h` declines attr-carrying nodes, so
an attr body delivers through the `torch.compile`d spelled
fallback, never a fused pointwise kernel — concretizing attrs
before codegen is the unbuilt step.  **Census: zero.**  Across
all 68 zoo workloads the IR contains *no* attr-carrying
elementwise op at all (TorchSource emits attr-free elementwise
nodes; the 87 attr dicts the scan tier mints are the `applyd`
node's own `kind`/`axis` carrier attrs, a different slot).
Attr bodies are tractable end-to-end except Triton, and
occurrence-blocked in the current corpus — nothing to measure.

**Reduction bodies — no tier collects them; census confirms.**
No site collector produces a spec whose body *is* a reduction:
`gen` is elementwise-only (`elementwise_spec` rejects
`sum`/`mean`/`max`/…), `decayed` is the `matmul(mul(s,d),v)`
contraction spelling, `scan` is the carrier fold.  Zoo census:
4 `decayed_*` minted total (GatedAttentionUnit, GCNLayer,
ChunkedRetention, ChunkedSSD), 5 minted specs contain a
reduction op *inside* an unrolled scan map body — none is a
reduction-bodied claim.  The single "kept decayed" row from the
earlier sweep did not reproduce on board inspection (no
`decayed_*` claim tag, handlers absent) — referee noise at the
margin, not a win.  Reduction claims stay blocked at the
*collector* layer: minting one needs a new tier whose spelled
denotation (a `sum`/`mean`/`max` tree) the bindings already
evaluate, plus a kernel — the honest next step is a reduction
tier, not a pricing fix.

**Seq-scan pricing — blocked, and a latent soundness gap closed.**
A carrier-aware spelled referee is *not* implementable with what
exists: the ambient `applyd` binding is one map application
(`f[0]*h + f[1]`, `mlopt_torch/torch_bridge.py:1963`), the minted
seq op declares a fold over the step axis, and `_IR_TO_TORCH`
contains no fold/scan construct to spell it with — unrolling
into an `affd_compose` tree would need T state-shaped operands,
an IR rewrite, not a binding.  In practice the question is moot
today: `_scan_sites` pins `kind="unrolled"` on every minted spec
(zoo census: seq=0, unrolled=87), so no collector ever produces
the seq form.  **But the audit found the gates were inconsistent
with that fact:** `_zoo_case` ran `profitable` over *every*
minted tier — seq scans would have been timed fold-kernel vs
map-spelled (wrong denotation) — and `_claim_cost_gate` left
seq scans *offered unpriced* while the seated `gen_sink`'s
`gen_bindings` would bind them to exactly the wrong (map)
denotation — a silent semantic bug had one ever been minted.
Fixed: `_claim_measurable` is now the shared predicate, and
seq-form scans are **dropped** at both `_zoo_case` intake and
`_claim_cost_gate` (unpriced AND undeliverable through
`gen_sink`; `hybrid_sink` remains their only honest delivery).
Test updated to pin the drop.

## The reduction tier lands (2026-11-07, T-010)

The "next step is a reduction TIER" lever is implemented and
measured — `docs/reports/p31_reduction_tier.md` is the full record.

**Census first:** the zoo is NOT occurrence-blocked — 161
reduction-headed sites across 32 of 68 workloads (sum 60, softmax
50, mean 18, plus cumsum/topk/amax/norm outside the tier's
vocabulary).  What IS narrow is the *fusable* subset: the tier only
claims a reduction whose spelled body carries ≥1 elementwise op —
22 workloads mint specs, ~30 distinct `red_*` handlers after canon
dedup.

**Machinery:** `genkernel.reduce_heads()`
(`opmeta.REDUCE_DIM_OPS ∪ {softmax}`), `reduction_spec`
(head concrete + attrs, ≥2 ops, non-head ops all elementwise),
`reduction_handlers` minting `red_<i>`,
`Tier("reduction", …, None)` — no reduction Triton codegen, so
delivery IS the compiled spelled body and `profitable` prices
exactly what ships.  `play._reduction_sites` collects
(head + attr dict concrete, args via `_fusion_pattern`'s boundary
rule), `TIER_SITES["reduction"]` registers it, and
`_claim_measurable` admits `red_` — spelled eval through
`_IR_TO_TORCH` is the honest denotation.

**Measured:** at default zoo sizes (16×8×2) the referee keeps
**0/30** — compiled-spelled loses 1.4–2.8× to eager spelled in the
tiny-shape dynamo-overhead regime, and the gate drops every one.
At 64×64×8, RWKVMixBlock's two claims keep (0.56×/0.73×) and
`gen_probe` delivers a verified module at **1.206× vs eager,
0.892× vs whole-model torch.compile** — the incumbent already
fuses these bodies, so the ceiling is beating eager, not Inductor.

**Verdict:** real but narrow.  The tier closes the collector gap
the audit named; its payoff is compiled-spelled fusion vs eager at
paying shapes.  Beating `torch.compile` needs a reduction Triton
codegen (`tl.*`-axis kernel over the spelled body) — a codegen
project, not a tier.
