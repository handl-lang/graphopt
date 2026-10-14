# CatOpt — research writeup

Verified search over semantics-preserving rewrites of neural-network
computation graphs. Engine: `catopt-core` (torch-free e-graph, laws,
cost algebra); shipped backend: PyTorch via `Source`/`Sink` ports.
Baseline throughout: `torch.compile`/Inductor, RTX 2050 (4 GB) +
i5-12500H, synced timing, fp64 verification where stated.

## 1. Claim

A program optimizer's reachable set is bounded by its semantic
language, not its search strategy. Tensor-level IRs rewrite *ops*;
rewrites expressed over *categorical structure* — monoid carriers,
traced monoidal fixpoints, products as `⟨f₁,…,f_k⟩ = (×fᵢ)∘Δ` —
reach programs no op-level pattern composes to. Precisely: the
balanced scan of an unrolled affine recurrence `h_t = Ah_{t-1}+x_t`
saturates at depth ~1.5T under matmul/add term laws because the
partial-product pair `(ΠA, ΣAⁱxᵢ)` is a cross-class object; in the
affine carrier the *same* associativity law produces the Blelloch
tree (depth ~2·log₂T). The claim is about reachability under a
fixed law set, not about a smarter heuristic.

## 2. Mechanism

`model → typed IR → e-graph saturation (equational laws) →
cost-based extraction → verified program → backend lowering`.

- **Carrier monoids**: `aff(A,b)` (linear recurrences → scans),
  `om(m,l,a)` (online softmax — FlashAttention's combine *follows*
  from homomorphism + associativity), `trace^U` (JSV axioms as
  rewrites; loops → resolvents `P+Q(I−S)⁻¹R`), cross-carrier laws
  (`xcarrier.py`) pushing readouts through scans and scans inside
  softmax elements.
- **Nonlocal lifts**: `lift_scan_to_applyd` constructs the carrier
  member directly from the recognized recurrence spine — carrier-law
  saturation is combinatorially explosive at long T (358k enodes on
  retnet T=128 vs the 100k cap), so the fold is built, not
  saturated. Pointwise witnesses make constructed members replay
  standalone.
- **Extraction**: additive cost over the e-graph; `backend_cost`
  prices members using ops the sink can't lower at +inf. Carrier
  selection is nonlocal — additive extraction can't price a batched
  spine — so `_carrier_upgrade` force-extracts root-class carrier
  enodes and compares *delivered* prices per lowering.
- **Certificates**: every extracted program carries an ordered
  replayable derivation `original → optimized`; `verify_certificate`
  re-checks derivational equivalence on real terms (fp64), not
  numerical spot-checks. It has caught a false-proof matcher bug, a
  shape misinference that fabricated a 1.98× "win", a well-typed
  wrong program (diff 9.83), and a launch-time-vs-execution timing
  bug. All regression-tested.

## 3. Verified results

All rows fp64-verified where stated; bench artifacts named for
reproduction.

| Transform | Result | Artifact |
|---|---|---|
| Weights-first fold, k-chain `x@W₁@…@W₁₆` → 1 GEMM (Inductor keeps all k `mm`s) | **15.9× vs Inductor GPU** (k=16, d=512, R=65536); ≈k× scaling across 36-cell sweep | `reassoc_scale_20260926-142951.md`, `GPU_RUN.md` |
| Asymptotic reassoc `(QKᵀ)V → Q(KᵀV)`, O(T²d)→O(Td²) | 8.0× at T=2048 | README results table |
| Scan lift, retnet/GLA/delta blocks — `optimize_model` selects + delivers carrier end-to-end | root=`applyd`, fp64-exact (rel 8e-16), **3.36× vs eager** T=128 CUDA; 3.9× with `compile=True` | `real_linear_attn_20260926-235234.md` |
| Chunked decode carriers, CUDA-graph captured | **1.65–2.8× vs best non-carrier** (GPU); delta C=8: 29.7µs vs 49.1µs; un-captured carrier loses (dispatch) | `decode_scan_20260927-015244.md` |
| GLA T=2048 — Inductor compile exceeds 60s timeout | CatOpt delivers certified batched schedule (262s pipeline), runtime parity w/ eager | plan `0005-cost-model-fidelity.md` §scale |
| Streaming attention (om monoid) | 260× vs sdpa-recompute at 65k cache; 89 MiB flat at 2M keys | README results table |
| Projection pairing (QKV, gate·up) on llama2.c | rediscovers `MergedColumnParallelLinear`/`QKVParallelLinear`; 1.24× vs Inductor | `stories15m_bench` |
| FLOP reduction (reassoc/merge/factorize) | 2.51× GPU / 3.02× CPU, DeepParallel b=4096 | README results table |
| Autotuned selection (`optimize_model_autotuned`) | matrix_chain 2.24× CPU; picks honest about losses (linear_recurrence → eager) | `killer_demo_20260927-021522.md` |

**Cost-model fidelity** (`cost_fidelity_*`): ρ ≈ 0.95 in the win
regime (k-chain — correct ordering *and* pick). Scan regime was
inverted (ρ −0.24…−0.38): same term, same predicted cost, 6–30×
latency spread across lowerings. Fixes: executor-aware pricing
(`executor_cost_for`, calibrated `dispatch_us`/`leaf_eval_us`),
lowering routing (`stats["lowering"]`), honest param-only discount
(a `trace` resolvent hid a 14.5s `linalg.solve`), and
`fusion_regions` (predicted Inductor kernel count — the 128-leaf
`applyd` spine bills ~1 kernel, predicted 18.7µs ≈ measured, not
~256 phantom dispatches). Post-fix ρ: `lowering_min` 0.167→0.261,
`exec_generic` 0.070→0.316 — positive but not yet predictive; the
frontier on scan blocks is thin (4 near-identical alts).

**Meta-level results** (the game layer — `catopt_discovery`):
`joint_probe` puts a forward term and all derived VJP terms in one
e-graph — `linear_fwd` extracts at 13.0→10.0 certified (the `tanh`
VJP reuses the whole forward term, billed once), while
silu/softmax stay flat (single-use sharing ≡ fusion under
`count_cost`).  The meta-arena's learned player over the mixed
board answers plan-0021 Q3: cold `LinearPolicy` beats scripted on
mean reward (−0.03 vs −0.10, 40 held-out boards) *and* produces 3
certified structural wins the playbook can't reach — but only
after the `claim(tag)` option move fused the paying
declare→handle line into one refereed action; consolidation
remains sampling luck, not learned preference.

**Delivered-vs-incumbent** (`gen_probe`, RTX 2050, verified):
spelled matmul chain delivered at **3.96× vs `torch.compile`**
(22µs vs 87µs — the reassoc laws fold to one GEMM; Inductor keeps
three); spelled llama-MLP swiglu at **2.11×**; pure pointwise
fusion honestly loses (0.82–0.88×).  `joint_step` — forward +
derived grads under one shared memo — beats autograd's fwd+bwd
13% on the tanh-linear archetype, verified at 9.5e-07.  The gen
kernel machinery (`genkernel` — minted handlers, compiled
bindings, gen sink) closes the loop for sites no law names;
`claim` binds them with no new moves.
Artifacts: `project/retros/joint-training-graph.md`,
`project/retros/meta-arena-player.md`,
`project/retros/generic-game-layer.md`,
`project/retros/gen-kernels.md`.

**Machine-minted laws and atoms (this cycle).**  The loop no longer
only *audits* a hand-written bank — it grows the bank and ships into
it.  `mul_unsq_nat` (`mul(unsqueeze U d, unsqueeze V d) =
unsqueeze(mul U V, d)`) was minted from a corpus op-tuple (a
DeltaNet-style block's own `β·k` write): the naive broadcast guard
was **falsified** by the adversarial sweep (212 bad accepts), the
exact `flat-map-unsq` guard was synthesized instead (0 bad accepts /
0 misses), closure cost +18 enodes, **−12.27%** on its source
workload, provenance `MACHINE_SHIPPED` — it is in the default rule
set.  A structured handler grammar minted and fp64-verified handler
bodies (the `om_decayed` carrier element at **8.9e-16** — previously
a hand-written executor); a law-synthesis loop derived and certified
25 law rows around shipped atoms (false twins rejected), and caught
a real lowering bug (`decay_vec` `sign=0` coercion) because the
oracle refused to certify rather than ship it.  The composed loop
(`need → handler → laws → admission`) minted `scan_fold` (the
selective-scan recurrence, 1.78e-15) with a certified law set —
and measured its own honest null (below).
Artifacts: `project/retros/mul-unsq-nat.md`,
`project/retros/handler-proposals.md`,
`project/retros/layer3-loop.md`,
`project/retros/atom-synth-loop.md`; ADR 0007.

**Backend generality, measured twice.**  `catopt-onnx` landed both
directions (`OnnxSink` IR→ONNX, `OnnxSource` ONNX→IR) with **zero
edits under `catopt-core`** — the hexagonal boundary tested by a
second, non-torch backend and held.  fp64 round-trip through
`onnxruntime` at **1.11e-16** (machine epsilon); 107 input op types,
opsets 13–21; package coverage 100%.  Artifact:
`project/retros/onnx-package.md`; ADR 0006.

**The substrate prototype.**  A combinator presentation
(`tools/combinator.py` — monoidal basis with explicit `Δ` sharing
and a bounded `loop` trace) round-trips **321/321** corpus workloads
structurally exact with zero ops refusing translation (mean blowup
2.61×), and the RetNet shared-state fold spells as a `loop` term
verified at 1e-7–2.6e-7 — carrier atoms become *definable*, not
primitive.  The open gate: `loop` structure is erased at export, so
folds must be proposed by search.  Artifact: ADR 0008.

**Delivery caught up on speed.**  The fused carrier lowering
(`decayed_lower`) reaches **parity with the paper's own chunked
implementation** (2.08 vs 2.12 ms at T=2048) and beats it under
graph replay (0.670 vs 0.822 ms at T=1024), fp64-exact 1.4e-16 —
the remaining gap to FLA is kernel-level (fused fp32 Triton vs fp64
torch GEMMs + a materialized score tensor), not structural.
Artifact: `project/retros/retnet-external-baselines.md`.

## 4. Honest losses and boundaries

- **Inductor wins wherever it compiles the raw graph.** Executor
  per-leaf dispatch makes the uncompiled carrier lose (retnet T=128:
  0.184×; CPU chunked decode: 0.5–0.8×, all six cells LOSS in
  `decode_scan_20260927-010340.md`). The CUDA-graph capture is what
  amortizes the carrier's launch structure to zero.
- **CPU decode is a loss**; launch-bound cells (B=1, T≤64) lose
  4–15% — split-view copies cost more than saved launches
  (`decode_bench`, `GPU_RUN.md` §decode).
- **Pipeline cost**: 262s at T=2048 vs Inductor's 45s compile where
  Inductor succeeds (retnet T=2048: CatOpt 215.8s→1.04ms vs
  Inductor 45.4s→0.256ms — Inductor still faster at runtime).
- **Toy dims**: d≤64, T≤2048, B=1, RTX 2050 (4 GB, no
  max_autotune_gemm). Real checkpoints (stories15M/110M): all blocks
  transform and verify (~2e-5), −37% GEMM launches profiler-verified,
  **parity** on wall time.
- Chunked om-attention loses to fused SDPA head-to-head when K,V fit
  on device; the win is memory/incremental state, not throughput.
- **The minted atom did not pay.**  The composed loop's first full
  semantic atom (`scan_fold`) computes the fold's *total*, while the
  corpus *emits* the sequence — 0 certified laws fire on real terms.
  The paying sibling (`fold_emit`) sits >200k deep in the round-robin
  enumeration; it verifies at 6.7e-16 when reached directly, so the
  gap is enumeration *ordering*, not expressiveness.  Truth and pay
  are measured separately, and they were disjoint here.
- **Discovery yield is sparse.**  A 60-model sweep of the
  law-LHS-exposed class found **no paying new law beyond
  `mul_unsq_nat`** (~1.5% of the exposed class carries fresh paying
  unpublished structure), and exposure itself is bimodal — 81/321
  workloads expose literally nothing (the fused-op class: convs,
  pools, `sdpa` spellings).  Exposure predicts reachability, not
  payoff.
- **Anti-unification compression is a measured null.**  Definitional
  atoms mint soundly and shorten spellings (12 vs 26 ops on one
  winner) but did **not** expand reach on any of three workloads
  (identical cheapest extractable member with/without atoms); the
  strict arm is worse under tight node budgets.  Recorded as the ADR's
  falsification clause firing.
- **The precision lane is a certified negative on real BiRefNet.**
  On the 220M-parameter checkpoint the bounded machinery offered
  members (20 at budget 1e-2, 56 at 3e-2) and **correctly refused all
  of them**: thin dedup structure (18/2304 rows) makes the gather-
  wrapped bounded member price 1.32× *worse* than dense.  The
  fp32 gate stayed bitwise-clean (nothing wrong shipped), but no
  speedup exists on that model.  Side findings: 43 `BatchNorm2d`
  modules (so conv+BN fusion targets do exist), and `deform_conv2d`
  exported with no lowering binding — a real coverage gap since
  fixed.

## 5. Open work

- **Enumeration ordering**: reach the canonical emitted-scan seed
  early (the measured blocker between the minted atom and pay).
- **Proposal-space extension**: the substrate prototype (ADR 0008)
  makes the enumeration space uniform over morphisms; gate 3 (does
  search *reach* `loop` forms) is the open question that decides
  whether folds are discoverable rather than spellable.
- **Executor leanness**: per-leaf dispatch is the loss driver;
  fusion/tiling inside `scan_lower`/`om_lower` or codegen for the
  batched levels.  The FLA gap is now precisely attributed here
  (fused fp32 kernels vs fp64 torch GEMMs).
- **Fusion-aware extraction**: `fusion_regions` is landed for
  *reporting*; completing it for *selection* needs either additive
  decomposition that respects kernel boundaries or more
  `_carrier_upgrade`-style coordinated passes.
- **Pipeline scalability**: spine walk + saturation at large T is
  the compile-time wall (215–262s); incremental matching already
  took 4-layer models 768s→4.9s, same direction needed for T.
- **Scale validation**: production dims (d≥1024, B≥8), datacenter
  GPU, trained linear-attention checkpoints. `calibrate()` makes
  re-targeting mechanical; magnitudes don't extrapolate.
- Thicker frontiers: scan-block alternates are near-identical;
  discrimination currently rides on executor variants, not form
  diversity.

## 6. Verdict

Publishable as a **reachability + verification** contribution, not
as a speedup paper. What is proven and verified: categorical
semantics expose transforms outside the tensor-IR reachable set
(Blelloch scan from associativity alone; om-combine = FlashAttention
derived, not encoded; closed-form resolvents via `trace`), the
pipeline *delivers* them end-to-end under certificates (not just
finds them in an e-graph), and the verifier has repeatedly caught
would-be false claims — the methodology is the result as much as
the numbers.  This cycle strengthens the claim in one specific way:
**the loop now mints content that ships** — a law derived from a
model's own op-tuple is in the default rule set with a synthesized
guard that survived an adversarial sweep, and a handler grammar
re-minted a previously hand-written carrier executor — while the
same machinery honestly refused its apparent siblings, its first
composed atom (pay-null), and its own compression layer (reach-null).
The performance story is real but narrow: wins where
work is asymptotically reduced (fold, reassoc, log-depth scan) or
where Inductor cannot produce code at all (GLA T=2048), and losses
wherever Inductor's fused kernels apply — executor overhead, not
search, is the gap. Nothing discovered is novel to practitioners;
the contribution is automatic discovery + certified equivalence +
per-shape selection, with a measured ~1.5% yield of fresh paying
structure in the exposed corpus. Framed as "verified reachability
with a real, regime-bounded performance win, and a discovery loop
that ships what it proves," it is a solid systems/PL workshop
or short-paper result; a full venue paper needs the scale
validation and executor work above.
