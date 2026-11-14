# Fusion-floor feasibility probe — is there a fusible-subgraph prize?

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (shared), torch
2.14.0+cu130, fp32, CUDA.  Probe:
`bench/drivers/probes/fusion_floor_probe.py` (new file).  Target:
`tts_stt_coverage_map.md` item #1 — the conv/resblock/pointwise+norm
chains that launch one kernel per op where Inductor fuses.  Measured
on the whisper-base decode step (the better-characterised of the two
walls; Kokoro noted where it differs).

**Verdict up front: the pointwise-fusion-carrier thesis is wrong for
this workload.**  The gen-tier collector itself finds **zero**
fusible sites on the delivered decode term; the real ~1.4 ms/call
non-GEMM prize decomposes into **K/V-cache `concat` elimination
(~0.95 ms/call — the single biggest item)**, GEMM-adjacent
pointwise/LN-epilogue fusion (~0.4 ms/call), and attention
online-softmax (~0.1 ms/call).  None of these is served by a
pointwise-fusion carrier; two of the three need no new kernels at
all.

## 1. The decode-step tape and term

Same machinery as `decode_step_fusion.md`
(`whisper_decode_loop.py` wrap/harvest, `Optimizer.search` iters=3 →
`lower(verify=False)`).  This run: delivered step **4.71 ms**,
`torch.compile(wrap)` **2.98 ms** — a **1.73 ms/call gap** (consistent
with the ~1.2–1.7 ms spread across prior runs on this shared GPU).

Extracted term: **263 nodes** — `linear`×49, `reshape`×36,
`transpose`×36, `concat`×36, `dropout`×24 (identity binding),
`layer_norm`×19, `add`×18, `sdpa`×12, `mul`×12, `slice`×12,
`gelu`×6, `pack`×1 (+`detach_`/`contiguous`).  Plan tape: 418 slots /
261 steps — identical histogram (views and identity ops cost no
kernel).

## 2. Fusible-subgraph census — the honest answer is "almost none"

Elementwise vocabulary = `genkernel.ELEMENTWISE` (the exact set the
`gen` tier's `_fusion_sites` collector uses:
`discoverykit/.../genkernel.py:185`).  Census over `res.term`
(DAG view, `mlopt_core.traversal.ops` — `expand` is the exponential
tree spelling and was deliberately not used):

* **36 elementwise ops in 19 weakly-connected regions.**
* **region0 — 18 `add`s, one region.**  The residual stream:
  `h_{l+1} = add(h_l, sublayer(LN(h_l)))` chains every layer's
  post-attention and post-MLP adds into a single elementwise
  component (each add's output is the next add's input).
* **18 singleton regions** — 12 `mul` (boundary=1: `mul(x, const)`
  scales) and 6 `gelu` (the MLP activation between two linears).

**`_fusion_sites(term)` returns 0 sites** (`play/sites.py:159` — a
site needs an elementwise-headed subterm whose pattern has ≥2
*distinct* ops).  The add-chain spells `("add", …)` only; every
mul/gelu is a one-op pattern.  **There is no legitimate
pointwise-fusion site in this term.**

The add-chain is also *semantically* unfusible even if it had two
distinct ops: each intermediate `h_l` is consumed by that layer's
`layer_norm` → QKV projections, so the 18 adds cannot collapse into
one kernel — Inductor doesn't fuse across those consumers either
(its wins are per-layer epilogue/prologue groups, see §4).

### Measured per-region ceiling (gen-tier machinery, real values)

For each candidate region: spelled eval (one binding call per op —
the plan-tape dispatch shape) vs `genkernel.gen_bindings`
`torch.compile`d kernel (`genkernel.py:353` — the shipped fusion
delivery mechanism), on the harvested decode-step tensors:

| region | ops | spelled | fused | verdict |
|---|---|---|---|---|
| add-chain (18 adds, 19 leaves) | 18 | 98–115 µs | 33–52 µs | 2.2–3.0× — **but invalid**: intermediate `h_l` are live consumers; a real delivery would still need ~12 of the 18 outputs materialised. Upper bound only: ≤ ~65 µs/call ≈ 1.4% of step |
| singletons (mul×7 measured) | 1 | 9–29 µs | 29–119 µs | **fusion loses 3–4×** — one compiled-kernel call costs more than one eager op at these sizes |

Singleton fusion is a measured *negative*: the gen-tier compiled
binding has ~20–90 µs of call/dispatch overhead vs ~9–29 µs for the
raw op.  Fusion only pays when it eliminates ≥2 kernels *and* the
region is real — which this term does not have.

## 3. Routing constraint — carriers are root-only, confirmed

Read path (all in `eqsat` unless noted):

* `mlopt_orchestrator/pricing.py:268` `_route_spec` — iterates
  `sink.executors`, keeps only specs with `carrier is not None` and
  `s.accepts(best_term)`; `accepts` is documented in
  `mlopt_core/ports.py:294` as "the *term-level* probe: does ``term``
  carry this executor's native **root shape**".
* The five carrier specs (`graphopt/.../adapters.py:203-270`:
  `om_decayed_chunked`, `scan`, `om_batched`, `omd_batched`,
  `om_streaming`) probe `is_*_apply_term`, each of which checks
  `root.op == <carrier-op>` (`om_lower.py:106`,
  `scan_lower.py:200`, `omd_lower.py:154`,
  `decayed_lower.py:371`).  A carrier member *inside* a mixed term
  evaluates through the plan tape's per-op bindings — the
  `decode_step_fusion.md` §2 finding, re-verified.
* `_carrier_upgrade` (`pricing.py:194`) only re-examines the **root
  eclass** for carrier-apply enodes; `_carrier_plans`
  (`pricing.py:42`) keys plan builders by carrier-apply root op.
* `CarrierMachinery` (`orchestrator/carriers.py:31`) has no slot for
  subterm routing — plans/lifts/rules are all whole-term.

So "no carrier executor routes NON-ROOT subterms" is exactly right,
and a pointwise-fusion carrier would need: a new carrier family
(root op + plan builder + `ExecutorSpec` + `_carrier_plans` entry +
laws forming carrier members) **plus** orchestrator subterm routing —
the LARGE-effort path the coverage map already flagged.  Against §2's
zero-site census, building it for this workload buys ~nothing.

## 4. Where the ~1.4 ms/call actually goes — profiler attribution

torch.profiler, CUDA-kernel rows only (CPU op rows double-count —
fixed mid-probe).  Delivered `IRModule` vs `torch.compile(wrap)`,
same inputs, 10 reps:

| bucket (ms CUDA/call) | delivered | compile | Δ |
|---|---|---|---|
| kernels/call | 230 | 158 | −72 |
| **gemm** | **2.80** | **2.70** | −0.10 (the floor — ~120 MB fp32 weights) |
| **cat** | **0.95** | **~0.00** | **−0.95** |
| **pointwise** | **0.51** | **~0.00** | **−0.51** |
| layernorm | 0.06 | ~0 | −0.06 |
| sdpa | 0.04 | ~0 | −0.04 |
| triton_fused (all Inductor kernels) | — | 0.14 | +0.14 |
| memcpy/other | 0.04 | 0.04 | ~0 |
| **total** | **4.39** | **2.88** | **−1.51** |

Step wall: 4.71 → 2.98 ms (−1.73; profiler device-time delta −1.51 +
dispatch).

Top kernels tell the story:

* **Delivered**: `CatArrayBatchedCopy` ×24/call ≈ 0.92 ms/call — the
  36 `concat` steps materialise K/V cache appends; `vectorized_
  elementwise`/`AUnaryFunctor` ×24 ≈ pointwise residue; LN ×19;
  math-backend sdpa already cheap (`sdpa_backend_fix.md` landed —
  `fmha`/`softmax_warp` rows ≈ 0.04 ms/call).
* **Compile**: `aten::mm`/gemvx ≈ same GEMM time; then a handful of
  fused Triton kernels do ALL the rest:
  `triton_poi_fused__unsafe_view_cat_transpose_view` ×12/call
  @0.3 µs — **cat+transpose+view fused into the consumer's index
  space, zero materialised cat**;
  `triton_per_fused_add_addmm_native_layer_norm` ×~15/call —
  residual-add + next layer_norm + GEMM epilogue in ONE kernel (the
  "pointwise+norm chain" the coverage map hypothesised — it exists,
  but straddles the LN boundary, so it is *not* an elementwise
  region and `_fusion_sites` can't see it);
  `triton_poi_fused_addmm_mul_view`,
  `triton_red_fused__softmax_…_online` (fused attention).

### Prize decomposition (per decode step)

| class | ms/call | mechanism Inductor uses | our analogue |
|---|---|---|---|
| K/V `concat` ×36 | **~0.95** | cat→view: consumer reads (past,new) without materialising | none — see §5 |
| pointwise residue (add/mul/gelu ×36) | ~0.5 | fused into GEMM/LN epilogues + pointwise Triton | gen-tier (0 sites here) / kernelopt `kf_*` |
| LN-adjacent fusion | ~0.06 | `add_addmm_native_layer_norm` per-layer kernels | none — LN ∉ ELEMENTWISE, ∉ reduce_heads |
| attention online-softmax | ~0.04–0.1 | `triton_red_fused` | sdpa math already near-floor |
| **total non-GEMM** | **~1.5** | | |

## 5. kernelopt's role — viable seam, wrong target for the big prize

`kernelopt_triton_wiring.md` already measured the `kf_*`/Triton
route: pays only at small-M GEMM epilogues (`o_proj` 1.42× vs the
compiled composite, loses at corpus sizes).  For *this* wall:

* The pointwise residue (~0.5 ms) is exactly what Inductor's
  epilogue absorption already does well — hand-Triton is unlikely to
  beat it (the wiring report's own conclusion).
* The `bindings=`/`KernelSink` seam *is* viable for delivering fused
  subgraphs (zero source change, verify-first) — but there is no
  kernelopt kernel family for cat-view elimination or LN-adjacent
  fusion; `conv1d`/`sdpa`-shaped sites already decline honestly.
* Where kernelopt DOES fit: the GEMM-epilogue slice of the pointwise
  prize (gelu/bias/residual absorption at decode M=1) — the
  `mlp_block`/`o_proj` measurements say ~1.1–1.4× on those subterms,
  worth ~0.05–0.15 ms/call here.  Real but small.

## 6. Recommended path — ranked by prize/effort

1. **K/V-cache cat elimination (~0.95 ms/call, ~55% of the gap) —
   harness reformulation, not fusion.**  The decode wrap already
   uses the padded-fixed-capacity trick (mask-gated L slots).  The
   remaining `cat(past_L, knew_1)` exists only because `knew` is
   produced *inside* the module.  Two honest routes:
   (a) **caller-maintained full K/V buffers + in-place cache-write
   op**: module takes the L+1-slot buffers as inputs (alias-aware
   `CudaGraphRunner` already binds persistent buffers), outputs
   `knew/vnew`; a `cache_write(buf, knew, t)` binding does the slot
   write and attention reads the buffer directly — zero cats, zero
   new kernels, reuses the shipped alias-capture machinery.  Needs
   one new op + binding + a purity/verify story for in-place
   (mutation guard exists in `runners.py`).  **MEDIUM effort, biggest
   prize.**
   (b) sdpa-split-KV binding (past + current as separate operands,
   joint softmax via custom math path) — same prize, more algebra,
   fp32 math-attention already cheap so the win is the cat copy
   itself.
2. **GEMM-epilogue absorption (~0.4–0.5 ms/call) — kernelopt `kf_*`
   route table.**  Extend the wired Triton mm-epilogue bindings
   (`triton_mm_bias`/`mm_gelu` family) to the decode-shaped sites —
   the seam exists (`bindings=` override), the pay is measured
   (1.42× on `o_proj`-class sites), and `kern_fuse_mm_gelu`-class
   laws cover `gelu(add(mm,b))` if minted.  **MEDIUM effort,
   moderate prize**; honest caveat from the wiring report: pays at
   small-M only — decode IS the small-M regime, encoder/prefill must
   keep declining.
3. **LN-adjacent fusion (~0.06–0.1 ms/call) — defer.**  Needs a new
   fusion class (norm-boundary groups); neither `ELEMENTWISE` nor
   `reduce_heads` covers `layer_norm`.  Small prize; revisit only if
   1–2 land.
4. **Pointwise-fusion carrier / non-root carrier routing — DO NOT
   BUILD for this workload.**  `_fusion_sites` = 0 sites; singleton
   fusion measured 3–4× *slower*; the one multi-op region is
   semantically unfusible.  The machinery (`genkernel` gen tier)
   stays valuable for workloads with real elementwise chains
   (GLU/SiLU-mul spellings, Kokoro resblock pointwise stacks — see
   below), reached through the existing `gen` play domain / claim
   path, not through carrier executors.

### Kokoro caveat

Kokoro's ~53 ms gap is conv/resblock-shaped, not decode-step-shaped.
Its pointwise+norm chains (Snake/SiLU activations, `mul`-`sigmoid`
spellings, norm epilogues) plausibly *do* produce `_fusion_sites`
hits — the gen-tier collector should be run on a Kokoro decoder
subterm before generalising this verdict.  Not measured here (probe
budget); flagged as the honest open question.

## Reproduce

```sh
PYTHONPATH=/tmp/xfm_pkgs .venv/bin/python \
    bench/drivers/probes/fusion_floor_probe.py 3 12 40
```

Files added: `bench/drivers/probes/fusion_floor_probe.py` + this
report.  No package files changed; `ruff check`/`ruff format` clean.
GPU was shared — absolute ms swing ±30% run to run; all *deltas* and
orderings replicated across three runs.
