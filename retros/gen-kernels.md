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

- The probe claims the first offer; multi-claim composition
  (fuse several sites) is unmeasured.
- Elementwise-only by design — reductions/matmul bodies need a
  different codegen contract (attr binding, shape rules).
- The 0.38× regime wants a size guard: only offer gen claims
  where the fused body plausibly pays (a ProgramFeatures lever).
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
