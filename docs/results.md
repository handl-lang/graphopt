# mlopt — measured results

Generated from the pinned baselines in `bench/baselines/` —
regenerate with `python -m bench results --out docs/results.md`.
Each entry states what the suite asked, what it expected, what
it measured, and the machine it measured on.  The full
per-suite reports (timings, plots, raw metrics) are produced by
`python -m bench report <baseline.json>`.

## Certified bounded rewrites

*Do error-budget rewrites buy wall-time on a real checkpoint?*

**Expected** — WIN — error-budget rewrites deliver bounded members and a measured wall-time win vs Inductor on stories15M (see docs/results.md).

| verdict | finding | headline |
|---|---|---|
| **WIN** | bounded rewrites (error_budget) buy wall-time on a real checkpoint | 4/5 budgets delivered a bounded member; best 1.410× vs eager |
| **WIN** | every budget's optimized module stays within the fp32 gate | 5/5 budgets optimized cleanly |

Measured 2026-10-14T06:54:12+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `2c782dd` (dirty)) — baseline [`bench/baselines/bounded_e2e.json`](../bench/baselines/bounded_e2e.json).

## MiniGPT end-to-end

*Does composition (pairing + fold) beat plain Inductor end-to-end?*

**Expected** — PARITY — pairing fires per block, verified fp64-exact; wall time is ~parity.

| verdict | finding | headline |
|---|---|---|
| **WIN** | the composed mlopt+Inductor module beats plain Inductor end-to-end | best 1.06× vs Inductor on tiny_llama@4x128 |
| **WIN** | the optimized module verifies against eager | 5/5 cells verified |

Measured 2026-10-14T06:33:13+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `2c782dd` (dirty)) — baseline [`bench/baselines/e2e_model.json`](../bench/baselines/e2e_model.json).

## Compositional pairing demo

*Does per-model lowering autotune pick the measured-fastest variant?*

**Expected** — WIN — the reported pick is the measured winner, never a static choice.

| verdict | finding | headline |
|---|---|---|
| **WIN** | the Autotuned lowering beats eager on at least one model | best 2.69× vs eager on matrix_chain |
| **WIN** | every model's optimized lowering verifies | 4/4 models verified |

Measured 2026-10-14T06:18:43+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `2c782dd` (dirty)) — baseline [`bench/baselines/killer_demo.json`](../bench/baselines/killer_demo.json).

## Rewrite-law effect bench

*Does each registered rewrite law fire, get picked by extraction, and lower to a verified term?*

**Expected** — WIN — every registered law fires/picks/verifies; non-firing laws report honestly.

| verdict | finding | headline |
|---|---|---|
| **WIN** | every registered law fires, is picked by extraction, and lowers to a verified term | 11 verified / 11 picked / 13 fired across 13 cells |

Measured 2026-10-14T06:13:49+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `2c782dd`) — baseline [`bench/baselines/law_bench.json`](../bench/baselines/law_bench.json).

## Morphism-window composition

*Do morphism windows convert term-flops into wall time?*

**Expected** — WIN — term-FLOP reduction converts to measured wall time at GEMM-bound sizes.

| verdict | finding | headline |
|---|---|---|
| **WIN** | morphism windows convert term-level flops headroom into measured wall time | best 12.85× vs eager on chain_x4@4096x128 |
| **WIN** | every morphism rewrite verifies equivalent | 14/14 cells verified |

Measured 2026-10-14T06:17:47+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `2c782dd` (dirty)) — baseline [`bench/baselines/morphism_e2e.json`](../bench/baselines/morphism_e2e.json).

## Weight-chain reassociation vs Inductor

*Can the e-graph find a form Inductor's post-grad graph cannot express?*

**Expected** — WIN — the e-graph reaches a weights-first form Inductor's post-grad graph cannot express (see docs/results.md).

| verdict | finding | headline |
|---|---|---|
| **WIN** | mlopt reaches a weight-folded form Inductor's post-grad graph cannot express | 14.92× vs Inductor at (k,d,B·T)=(16,512,4096) (17.40× vs eager) |
| **WIN** | every timed cell passes the fp32 equivalence gate | 12/12 cells verified |

Measured 2026-10-14T06:16:44+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `2c782dd` (dirty)) — baseline [`bench/baselines/reassoc_scale.json`](../bench/baselines/reassoc_scale.json).

## Tooling-level measurements (hand-maintained)

This file is appended verbatim to `docs/results.md` by
`bench results` / `render_results_doc` — edit it here, never in the
generated file.  These measurements live in `bench/` and `scripts/` and are
recorded in the retros, not in `bench/baselines/`.

All measured on the dev box (RTX 2050 / i5-12500H, fp64 — same
hardware class as the baselines); magnitudes do not extrapolate to
datacenter hardware:

| measurement | result | tool | retro |
|---|---|---|---|
| `softmax_fold` on `ManualSoftmaxAttention` | **+13–27% eager, +29–48% CUDA-graph** vs raw | `bench/drivers/probes/law_wallclock.py` | [law-wallclock-verification](../project/retros/law-wallclock-verification.md) |
| `select_mul` marginal (optimized vs law-ablated) | **faster in 19/20 measurements** | `bench/drivers/probes/law_wallclock.py` | [law-wallclock-verification](../project/retros/law-wallclock-verification.md) |
| executor routing after measured pricing | **12/12 cases ship the measured-fastest member** (uncorrected picks ran 2–3× slower) | `bench/drivers/probes/executor_cost_probe.py` | [executor-pricing-fix](../project/retros/executor-pricing-fix.md) |
| `silu_fold` on the bench case | **−53–55% modeled cost, 1.43–2.62× wall-clock** | `bench run law_bench` | [three-cell-mediator](../project/retros/three-cell-mediator.md) |
| contraction player, n=40 vs `opt_einsum` | **0.89–1.01× of randomised greedy at equal wall-clock** (0.91–0.98 at T≤1, three seeds, fed budgets); **0.51–0.90× deterministic greedy** | `bench/drivers/probes/contraction/contraction_guided_restart.py` | [contraction-synthesis](../project/retros/contraction-synthesis.md), [contraction-player-shipped](../project/retros/contraction-player-shipped.md) |
| pipeline held-out rediscovery | **winner re-ranks #1, SHIP, every run** | `mlopt_discovery.pipeline --holdout` | [law-proposer-extensions](../project/retros/law-proposer-extensions.md), [groupnorm-convnext-corpus](../project/retros/groupnorm-convnext-corpus.md) |
| coherence catalogue over `ALL_RULES` (84, re-run 2026-10-18) | **60 axioms / 22 lemmas / 2 redundant; 20 no-instance; divergence 1** (`rms_norm_fold` × `_nogain` — measured, unresolved) | `mlopt_discovery.coherence --emit-basis` | [axiom-lemma-split](../project/retros/axiom-lemma-split.md) |
| evidence store, second run | **50/50 verdicts cached; 128 s → 6.6 s (~19×)** | `mlopt_discovery.pipeline --evidence-db` | [evidence-store](../project/retros/evidence-store.md) |
| `chainfuse` — laws + `claim(gen_0)`, Triton-bound | synthetic composite (`x@A@B@C` + 6-op pointwise tail, 256×512): **2.94× vs `torch.compile`** (86µs vs 253µs), verified 1.7e-05; laws alone reach 181µs — the minted kernel is required for the full win | `python -m mlopt_discovery.play --domain gen --probe` | [gen-kernels](../project/retros/gen-kernels.md) |
| delivered module vs `torch.compile` (`gen` probe) | spelled matmul chain **3.96×** (22µs vs 87µs; Inductor keeps all three GEMMs even under `freezing`); swiglu **2.11×**; pointwise fusion **0.82–0.88× loss** | `python -m mlopt_discovery.play --domain gen --probe` | [gen-kernels](../project/retros/gen-kernels.md) |
| held-out zoo probe | **11/18 delivered verified modules beat `torch.compile`** — LoRAAdapter 2.18×, WaveNetGate 1.51×, DeepEquilibrium 1.35×; every winner ran `claims=[]` (the `gk.profitable` measured referee filtered unprofitable minted claims) | `python -m mlopt_discovery.play --zoo` | [gen-kernels](../project/retros/gen-kernels.md) |
| joint fwd+bwd step vs `torch.autograd` | **+13%** (68µs vs 78µs, verified 9.5e-07) — one shared memo across forward + derived VJP; unshared costs 197µs | `mlopt_discovery.training.joint_step` | [joint-training-graph](../project/retros/joint-training-graph.md) |
| machine law pack | **16/16 store objects deployable** as an opt-in `RuleSet`; machine-only arm wins 3/22 zoo models the human library misses (`linear_chain_t` owns LoRAAdapter's −20%) | `mlopt_discovery.machine_pack` | [human-bar](../project/retros/human-bar.md) |
| `mul_unsq_nat` — first corpus-minted law, shipped in `DEFAULT` | **−12.27% on its source workload** (DeltaRuleBlock, 992496→870722), closure **+18 enodes**; exact `flat-map-unsq` guard after the naive broadcast guard was falsified (212 bad accepts); apparent sibling correctly rejected at admission | `bench/drivers/screen/exposed_discovery.py` + admission gate | [mul-unsq-nat](../project/retros/mul-unsq-nat.md) |
| exposure screen (321 workloads) | **bimodal: 81 score 0% law-LHS coverage, 140 score ≥95%**; exposure predicts reachability, not payoff (100%-exposed models fire nothing) | `bench/drivers/screen/exposure_screen.py` | [exposed-math](../project/retros/exposed-math.md) |
| corpus-wide discovery sweep (60 models) | **0 paying new laws beyond `mul_unsq_nat`**; fresh-paying yield ~1.5% of the exposed class; enum arm 0→198 fires after the attr-gap fix but 0 paying | `bench/drivers/screen/discovery_scale.py` | [discovery-scale](../project/retros/discovery-scale.md) |
| handler synthesis (layer 2) | **machine-minted verified handler for the `om_decayed` element** (8.9e-16 fp64) — previously a hand-written executor; contraction/fold/fold_emit/fold_zip all verify in bounded search; reductions/softmax refuse (named grammar gaps) | `mlopt_discovery.handler_synth` | [handler-proposals](../project/retros/handler-proposals.md) |
| law synthesis per atom (layer 3) | **25 certified rows / 4 rejected / 0 refused** on the shipped carrier atoms; new primitives incl. `affd.compose` (slot-solved), `dv.merge`, `omd.absorb_score`; false twins all caught; found a real executor bug (`decay_vec` `sign=0` coercion) | `bench/drivers/mint/atom_law_loop.py` | [layer3-loop](../project/retros/layer3-loop.md) |
| composed atom-synthesis loop | one atom minted through all four layers (`scan_fold`, 1.78e-15) + certified law sets for `cumsum`/`cumprod`/`logcumsumexp`; **0 certified laws fire on real terms** (the minted total vs the corpus's emitted sequence; the paying `fold_emit` sibling is enumeration-ordered >200k deep) | `bench/drivers/mint/atom_synth_loop.py` | [atom-synth-loop](../project/retros/atom-synth-loop.md) |
| anti-unification compression (layer 1) | **NULL on reach-expansion** — identical cheapest extractable member with/without atoms on all 3 workloads; compression real (12 vs 26 ops), strict arm worse under tight `max_nodes` | `bench/drivers/probes/anti_unify_probe.py` | [ADR 0007](../project/adrs/0007-semantic-atom-minting.md) |
| combinator bridge (ADR 0008) | **321/321 corpus workloads round-trip structurally exact**, zero ops refuse translation, mean blowup **2.61×**; RetNet shared-state fold spells as `loop` term and verifies at 1e-7–2.6e-7; extract projects round-trip-equal (92 Δ sites) | `bench/drivers/bridge/combinator_probe.py` | [ADR 0008](../project/adrs/0008-combinator-presentation.md) |
| fused carrier lowering (`decayed_lower`) | **parity with the paper's own chunked impl** (2.08 vs 2.12 ms T=2048; 0.838 vs 0.822 ms T=1024 eager), **beats it under graph replay** (0.670 ms); fp64-exact 1.4e-16; FLA still ~7–11× ahead (kernel-level: fp64 GEMMs + materialized score tensor) | `bench/drivers/probes/retnet_baseline_probe.py` | [retnet-external-baselines](../project/retros/retnet-external-baselines.md) |
| bounded lane on real BiRefNet (220M, HF weights) | **certified negative** — 20 offers at 1e-2, 56 at 3e-2, all correctly refused (thin dedup: bounded member prices 1.32× worse than dense); fp32 gate bitwise-clean; **BN census: 43 `BatchNorm2d`** (conv+BN targets exist) | `bench/drivers/probes/birefnet_bounded_probe.py` | [birefnet-bounded-lane](../project/retros/birefnet-bounded-lane.md) |
| ONNX backend, both directions (ADR 0006) | **zero changes under `mlopt-core`** for `OnnxSink` and `OnnxSource`; fp64 round-trip **1.11e-16** (machine eps) through `onnxruntime`; 107 op types / opsets 13–21 in, 169 ops out; package coverage 100% | `mlopt_onnx` + `tests/test_onnx_*.py` | [onnx-package](../project/retros/onnx-package.md) |

Two reconciliation notes:

- **The contraction "~1.8×" claim is stale.**  Early numbers had
  the learned policy at 1.54–1.78× of `opt_einsum`'s randomised
  greedy at n=40; the RL curriculum regime narrowed the gap to
  ~1.04×, and the oe-all **distilled** player that now ships as
  `mlopt_torch.load_contraction_policy()`'s default *beats* the
  teacher at equal wall-clock (the figure in the table above).  The
  curriculum-RL weights remain bundled as an alternate artifact.
- **The `bounded_e2e` baseline's headline is vs eager** (best
  1.322×, above).  Its per-cell `speedup_vs_inductor` spans
  0.94–1.25 (1.16–1.25 over the budgets that delivered a bounded
  member) — so the README's "1.15–1.32× vs Inductor" is not the
  pinned baseline's literal column: the 1.32 is vs eager and the
  vs-Inductor delivered range tops out at 1.25 on this baseline.

<!-- Hand-maintained curated results.  Appended verbatim to
     `docs/results.md` by `bench.benchkit.render.results`
     (`python -m bench results`) — edit it HERE, never in the
     generated file, or the doc-sync test will fail. -->

## Machine-derived structure — results by kind (2026-10-15)

*What the derivation machinery achieved, separated by **what was
measured on what**.*  Scope legend:

- **[model]** — a real `nn.Module`, delivered + fp64-verified
- **[form]** — a synthetic comparison of two computational forms
  (random tensors; measures the asymptote, not a model)
- **[symbolic]** — a cost-model count over the extracted program
  (`flops_cost` / `param_bytes_cost`), not wall time

### 1. Optimized models, end-to-end [model]

| model | T | vs eager | vs `torch.compile` | max_abs |
|---|---|---|---|---|
| `ChunkedRetention` (RetNet-style), T=512 — fused + minted kernel | 512 | — | **1.82–2.33×** | 1.3e-8 |
| `ChunkedRetention`, T=512 — **the derived carrier itself** | 512 | 0.51× (loss) | 0.48× | 1.6e-8 |
| `ChunkedRetention` | 2048 | **5.0–5.9×** | **6.2–7.4×** | 9.8e-9 |
| `chainfuse` (x@A@B@C + pointwise tail) | 256×512 | 3.71× | 2.90× | 1.7e-5 |
| `swiglu_mlp` (llama-style) | — | 1.17× | 2.00–2.04× | 6e-8 |
| `chain_mlp` | — | 1.06× | 2.52× | 4e-6 |
| `mulrelu` | — | 1.22× | 1.39× | **0.0** |
| `mulsoftplus` | — | 1.31× | 1.46× | 1e-6 |
| `mlp` (base) | — | 1.13× | 2.82× | 0.0 |

`gen` probe re-run 2026-10-15: **10/10 delivered and verified**;
9/10 faster than eager; all faster than `torch.compile`.

**The crossover is the honest shape of the carrier claim:** the
derived carrier *loses* at T=512 and wins asymptotically.

### 2. Peak activation memory [model] vs [form]

**Real module** — `_ChunkedRetention(16)` forward vs its extracted
carrier lowered through the ambient sink, peak
`torch.cuda.max_memory_allocated`, fp64, B=2:

| T | model | carrier | saved | max_abs |
|---|---|---|---|---|
| 512 | 18.52 MiB | 10.91 MiB | 1.70× | 1.6e-08 |
| 1024 | 48.90 MiB | 14.39 MiB | 3.40× | 1.7e-16 |
| 2048 | 169.66 MiB | 23.26 MiB | **7.30×** | 9.8e-09 |

**Synthetic proxy** (same shapes, random tensors — extrapolates the
asymptote past the model's feasible range):

| T | `(q@kᵀ)@v` | C=16 chunked | saved |
|---|---|---|---|
| 512 | 12.6 MiB | 8.8 MiB | 1.4× |
| 1024 | 25.1 MiB | 9.4 MiB | 2.7× |
| 2048 | 74.1 MiB | 10.6 MiB | 7.0× |
| 4096 | **268.1 MiB** | **13.1 MiB** | **20.4×** |

The carrier's peak is near-flat (8.8 → 13.1 MiB across an 8× T range)
while the fused form grows 21×.  O(T·C) against O(T²), as memory.

### 3. Structural savings [symbolic]

**Compute — FLOPs** (instrument `flops_cost`; ratio = before ÷ after):

| optimization | before (FLOPs) | after (FLOPs) | **compute saved** |
|---|---|---|---|
| weight fold, k=8 (d=512) | 8,388,608 | 1,048,576 | **8.0×** |
| weight fold, k=16 (d=512) | 16,777,216 | 1,048,576 | **16.0×** |
| K-way chunk vs bisection-chunked (T=512) | 29,386,752 | 6,700,800 | **4.4×** |
| K-way chunk vs fused quadratic (T=512) | 39,059,456 | 6,700,800 | **5.83×** |
| shared-state reassociation (T=512, K=8) | 8,572,928 | 7,643,912 | **10.8%** (24.5% at T=128) |

**Storage — parameter bytes** (instrument `param_bytes_cost`; d=512,
fp32, `x@W₀@…@W_{k-1}` vs the folded `x@W_cat`):

| k | before (stored) | after (stored) | **storage saved** |
|---|---|---|---|
| 2 | 524,288 B | 262,144 B | **262,144 B** (2.0×) |
| 4 | 1,048,576 B | 262,144 B | **786,432 B** (4.0×) |
| 8 | 2,097,152 B | 262,144 B | **1,835,008 B** (8.0×) |
| 16 | 4,194,304 B | 262,144 B | **3,932,160 B** (16.0×) |

Plus the shared-input MoE fold: **196,608 B** of stored parameters
eliminated (pinned by test).

### 4. Oracle-evaluation savings [symbolic]

| quantity | before | after | **saved** |
|---|---|---|---|
| programs encoded per e-node (k=11) | — | 16,796 programs in 231 e-nodes | **72.7 programs/e-node** |
| ranking quality (AP, held-out pages) | heuristic 0.045 | trained prior **0.133** | **3.0×** |
| useful laws found per episode | random 0.083 | trained prior **0.167** | **2.0×** |
| wall time, second run (evidence store) | 128 s | 6.6 s | **19.4×** |

### 5. Honest losses

| finding | measured |
|---|---|
| the carrier below the crossover | **0.51×** vs eager at T=512 |
| the *folded* carrier vs the spelled form | a fold is a **cost-neutral relabelling** — the gap is search, not pricing |
| pure pointwise fusion | **0.82× loss** — Inductor's home turf |
| shared-state on the real export | ~~unreached~~ — **reached**: the staged extract carries the chained shared recurrence (7 fold reads + 7 single-computed state blocks at K=8; 6,943,236 vs the like-for-like floor 7,082,496 — ~2% under). The earlier "0/13 fold" probe counted `applyd`-spelled members, which extraction never prefers; the like-for-like reference also omitted projections + in-graph score construction. fp64-verified via `TorchSink` (6.7e-16) |
| the composed atom-synthesis loop's first atom | **0 certified laws fire on real terms** — `scan_fold` computes the fold's *total*, the corpus *emits* the sequence; the paying `fold_emit` sibling is enumeration-ordered >200k deep (it verifies at 6.7e-16 when reached directly) |
| anti-unification compression (ADR 0007 layer 1) | **NULL on reach-expansion** — identical cheapest extractable member with/without definitional atoms on all three workloads; compression real (12 vs 26 ops) |
| discovery yield in the exposed corpus | **~1.5%** — a 60-model sweep found no paying new law beyond `mul_unsq_nat`; exposure itself is bimodal (81/321 workloads expose 0%) |
| the precision (bounded) lane on real BiRefNet (220M) | **certified negative** — 20 offers at 1e-2, 56 at 3e-2, all correctly refused (bounded member prices 1.32× worse than dense); fp32 gate bitwise-clean |
| the ONNX backend's memory/speed | **not measured as a win** — the value is portability + verified round-trip (1.11e-16 fp64), not throughput |

### 6. The discovery loop — minted and shipped (2026-10-14) [model] + [tooling]

| result | measured |
|---|---|
| **first corpus-minted law shipped in `DEFAULT`** | `mul_unsq_nat` — minted from DeltaRuleBlock's own `β·k` op-tuple; naive guard **falsified by the adversarial sweep** (212 bad accepts), exact `flat-map-unsq` guard synthesized (0 bad / 0 miss); closure **+18 enodes**; pays **−12.27%**; provenance `MACHINE_SHIPPED` |
| its apparent sibling | **correctly rejected** at admission — `mul_unsqueeze_r_id` (−14.3%, 12 models / 22 sites) has semantically unequal sites; the guard declines all of them |
| handler synthesis (ADR 0007 layer 2) | machine-minted verified handler for the `om_decayed` element at **8.9e-16** — previously a hand-written executor; every named non-elementwise shape verifies in bounded search |
| law synthesis per atom (layer 3) | **25 certified / 4 rejected / 0 refused** rows around shipped atoms; new primitives (`affd.compose` slot-solved, `dv.merge`, `omd.absorb_score`); found a real executor bug (`decay_vec` `sign=0`) by refusing to certify |
| fused carrier lowering vs the paper's own chunked impl | **speed parity+** — 2.08 vs 2.12 ms (T=2048), **faster under graph replay** (0.670 vs 0.822 ms at T=1024); fp64 1.4e-16; FLA still ahead (kernel-level) |
| ONNX backend, both directions | **zero edits under `mlopt-core`**; fp64 round-trip **1.11e-16** via `onnxruntime`; 107 in-op types (opsets 13–21), 169 out-ops; package coverage 100% |
| combinator bridge (ADR 0008 prototype) | **321/321** corpus workloads round-trip structurally exact, zero ops refuse translation, mean blowup 2.61×; the RetNet carrier spells as a `loop` term verified at 1e-7–2.6e-7 |

## 7. Retired: the weight-space / ε axis (from the retired REPORT)

*Historical record of a closed axis.*  These rows are migrated verbatim
from the retired REPORT (the weight-space / ε-axis phases 0–5), which
was itself the historical record; the file has been deleted and this is its
measurements' new home.  They are **not** pinned baselines — the source was
the report's own measurement session on llama2.c checkpoints
(`retros/measure_weights.py`, archived) — so they are kept as the record of
a *falsified* axis, not as live evidence.  The decision is **ADR 0001**
(`project/adrs/0001-weight-space-structure-falsified.md`); the narrative is
the lab notebook §1.1.

### 7.1 Phase 0 — linear falsification (stories15M, 15 M params)

| probe | result | verdict |
|---|---|---|
| numerical rank @1e-2 | ~270–288/288 | full — no exact low-rank |
| displacement rank | ~280/288 | no Toeplitz/generator |
| symmetry defect | ~1.41 | fully asymmetric |
| low-rank @99% | 96.3% of params | meaningless saving |

Gate outcome: **exact linear structure is absent.**  Phase 0 tested only
one algebra, so the axis stayed open to Phase 0b.

### 7.2 Phase 0b — the structured-algebra family

Gated as *storage vs relative Frobenius error against plain SVD*:

| probe | result | verdict |
|---|---|---|
| **token embedding** (60% of params) | **rank 19 @ 95% energy** | **~15× — live** |
| Kronecker-sum (attn/MLP) | 1.3× @ 35% residual | marginal — real, small |
| H-matrix off-diagonals | rel. rank 1.0 | dead |
| sparse concentration | top 10% ≈ 46% energy | mild |
| Monarch/butterfly ALS | diverged | inconclusive |
| INR coordinate-fit | rel err ≈ 1.0 | dead (no smooth manifold) |

Correction recorded honestly: the earlier "4 terms" was the best-e1 split's
rank; the storage-optimal split needs K~28, giving ~1.3× at 35% residual.

### 7.3 Phase 1/2 — ε pipeline rate–distortion (2-layer MLP, fp64)

| method | stored | ratio | measured err | certified bound |
|---|---|---|---|---|
| int8 quant | 16,384 B | 8.0× | 5.3e-3 | 0.076 (site) / 0.79 (model) |
| low-rank @10% | — | no offer | — | (weights are full-rank) |

Composite (low-rank embedding + int8 quant on all params, one e-graph, one
extraction): **577 KB → 16.7 KB (34.6×)**, measured err 2.8e-2 within the
certified model bound.  Other ε-machinery probes: `low_rank_gather` on the
real stories15M embedding — **rank 2 at 5% spectral = 142× storage**;
Kronecker-structured 64×64 weight → K=4 terms, **32,768 → 4,096 bytes
(8×)**, certified bound, replayable certificate.  `model_bound`
(output-level certificates) verified on a quantized 2-layer model: err
0.0034 vs certified 0.397 (spectral products ~100× conservative).

### 7.4 Phase 5 — real perplexity (TinyStories, stories15M, 2819 tokens, fp64; baseline ppl 5.03)

| config | storage | perplexity | verdict |
|---|---|---|---|
| emb rank-2 / r19 / r64 / r128 / r192 | 2.5–1.2× | 6108 → 214 | **dead at every rank** |
| Kronecker-sum (35% resid) | 1.1× | 143 | dead |
| **int8 RTN quant (certified, mlopt)** | **4.0×** | **5.16** | **works, bounded** |
| int8 per-channel RTN (GPTQ-class baseline) | 4.0× | 5.00 | works, no bound |
| int4 g128 RTN (GPTQ-class) | 8.0× | 7.01 | works, no bound |

Honest #2 verdict: at matched ratio, certified per-tensor int8 (ppl 5.16)
≈ per-channel int8 baseline (ppl 5.00) — the certificate costs ~0.16 ppl,
not bytes.  `quant_params(per_channel=True)` closes the quality gap at the
same ratio with the bound kept.  The differentiator: **compression parity
with the RTN-class baseline, plus a certificate** — not a ratio win.  The
embedding's top-19 singular directions are not the ones next-token
prediction needs: `‖ΔW‖` bounds do not predict `Δppl`.

### 7.5 §10.1 relational probe — the last untested class

Exact **cross-layer** structure on stories15M **and** stories110M:

| probe | stories15M | stories110M |
|---|---|---|
| 1. bitwise cross-layer dup rows/cols/head-blocks | **0 hits** | **0 hits** |
| 2. shared subspace — rank(stack) vs ambient | **full rank everywhere** (288/288, 768/768, 2048/2048) | **full rank everywhere** |
| 3. after Procrustes alignment | 0 aligned dup rows, still full rank | 0, still full rank |
| 4. stacked spectrum (diagnostic) | r95 ≈ 242–265/288 (~85–92%) | r95 ≈ 655–714/768 (~85–93%) |
| 5. shared dictionary (diagnostic) | k=64 → resid 0.70 @ 0.26× storage | k=64 → resid 0.74–0.93 @ 0.09× |

Verdict: **negative on exact — the door closes.**  Two independently-trained
checkpoints carry zero bitwise-shared cross-layer structure, and the
gauge-alignment test finds nothing the permutation/rotation freedom was
hiding.  Mild *approximate* redundancy exists (85–93% of dims for 95%
energy), but Phase 5 already showed approximate ≠ quality-preserving.

### 7.6 §10.1b symmetry probes (stories15M)

| probe | hypothesis | result |
|---|---|---|
| A. equivariance `WG=GW` | circulant / reversal / block-shift groups | **resid ≈ 1.41 everywhere** (=√2, the random value) — no symmetry |
| C. polynomial identity | low-degree `p(W)=0`, minimal poly | **minimal poly = full degree**: 288/288 distinct eigenvalues, Krylov rank unsaturated |
| B. learned displacement | rank(`AW−WB`) over learned diag/circulant `A,B` | residual ~0.12 but **disp-rank ≈ 280/288 — full** |

With Phase 0/0b and §10.1 all negative, **every known hypothesis class for
exact weight-space structure is closed on trained checkpoints.**

### 7.7 §10.2 the exact corner, measured across archetypes

Under `param_bytes_cost_for` with fp64 output equality verified:

| archetype | saved | mechanism |
|---|---:|---|
| stories15M (real, dense) | 0.00% | nothing to dedup (46 tensors probed) |
| dense transformer (synth) | 0.08% | `linear_channel_scale` fold |
| GQA 8q/2kv (repeat_kv materialised) | **37.50%** | `share_duplicate_param_slices` |
| adapter-merged (W+B·A stored) | 11.03% | `_fold_weight_chains` param-only fold |
| tied embed/classifier stored twice | **50.00%** | `share_duplicate_params` |
| MoE: 4 weight-tied routed experts | **75.00%** | `share_duplicate_params` |
| MoE: 4 weight-tied shared-input | 75.00% | tying found; concat copies billed per-occurrence |
| composed linears w2(w1·x), no act | 50.00% | fold exact: fused W₂·W₁ materialises at d² < 2d² |
| dead param (unused tensor) | 98.46% | unreachable leaf dropped |
| adapter UNmerged (Wx+BAx) | 0.00% | pairing honestly declines: concat costs more than it saves |

The claim, plainly: **the exact corner is ~0% on dense trained checkpoints
and real on structured ones** — GQA replication, double-stored ties,
weight-tied experts, merged adapters.  Every row fp64-verified
output-equal.  Archived evidence: `retros/exact_probe.py`,
`retros/weight-as-programs.md`; decision ADR 0001.

### 7.8 §11 ε-axis limits (retained)

- **`omd` at realistic scale** (`bench_omd2.py`): MQA fires (`omd_applym`)
  but the dense-fiber numerator is ~dv/dim× oversized — batched executor
  only **1.15–1.5× vs generic on CUDA**, loses to Inductor **5–20×**
  everywhere.  MHA now lifts via view-commute laws
  (`xc_reshape_apply`/`xc_transpose_apply`).
- **`model_bound` is known unsound for spectral sites at activation
  positions** (low-rank `eps_lr` members: it misses the ‖activation‖
  factor; reported **0.14** vs measured **0.246**).  Use
  `ibp.tight_model_bound` instead — it flags `spectral_unsafe` and gives
  sound bounds (**118×→3×** on quant sites via realized-delta propagation).
- **Monolithic saturation** previously didn't scale past ~2 blocks —
  **fixed**: incremental matching (dirty frontier) + member-resolution
  memoization + bounded expansive-rule saturation put a 4-layer model at
  **768 s → 4.9 s**, reaching ≥8 blocks monolithically.  (Also in
  `docs/paper_draft.md` §6.)
- **`omd` executor landed** (`omd_lower.py`): **1.2–3.3× vs generic eval**,
  **2.4–5.8× CUDA-graph**, fp64-exact.
- **Monarch ALS was inconclusive** — deeper butterfly structure may exist
  that naive alternating least squares can't reach.
- ε-passes are opt-in (`eps_rtol`); default extraction is exact-only.

### 7.9 §1 the original unified thesis (the founding bet)

The report opened on: *a PyTorch model and its weights are one semantic
program; optimizing that program — with certificates — covers computation
structure and parameter storage in a single search.*  Mechanism:
parameters are `Param` leaves in the same term language as compute, so
*elimination is a corollary* (a leaf no extracted member references never
reaches the state dict) and composition, tying, slice-sharing and bounded
factorization are the same event at different certificate bound values.
**Outcome:** the compute half survived; the weight half was falsified
(ADR 0001).  The ε machinery survives as an opt-in toolkit where a norm
bound IS the contract (activation paths, verification) — not as a
weight-compression mechanism.

## 8. Retired baselines (from the retired RESEARCH_WRITEUP)

*The research writeup has been retired (deleted); these are the rows it
recorded that no surviving doc carried.*  **The first three are unsourced**:
the writeup cited a "README results table" for them, and **no such table
exists** (the README never carried one) — so their provenance is the
retired writeup's own measurement session, and they must **not** be quoted
as pinned-baseline evidence.  The rest are retained with their config
named; where a pinned baseline supersedes a row, that is noted.

| measurement | result | source / note |
|---|---|---|
| asymptotic reassoc `(QKᵀ)V → Q(KᵀV)`, O(T²d)→O(Td²) | **8.0× at T=2048** | retired RESEARCH_WRITEUP §3 — **UNSOURCED** (claimed "README results table", which does not exist) |
| streaming attention (`om` monoid) | **260× vs sdpa-recompute at 65 k cache**; 89 MiB flat at 2 M keys | retired RESEARCH_WRITEUP §3 — **UNSOURCED** (same) |
| FLOP reduction (reassoc/merge/factorize) | **2.51× GPU / 3.02× CPU**, DeepParallel b=4096 | retired RESEARCH_WRITEUP §3 — **UNSOURCED** (same) |
| weights-first fold, k-chain → 1 GEMM | 15.9× vs Inductor GPU (k=16, d=512, R=65536); ≈k× scaling across a 36-cell sweep | retired RESEARCH_WRITEUP §3 — superseded by the pinned `reassoc_scale.json` (14.92× at B·T=4096) |
| scan lift (retnet/GLA/delta blocks) | 3.36× vs eager at T=128 CUDA; 3.9× with `compile=True` | retired RESEARCH_WRITEUP §3 |
| chunked decode carriers, CUDA-graph captured | 1.65–2.8× vs best non-carrier; delta C=8: 29.7 µs vs 49.1 µs | retired RESEARCH_WRITEUP §3 — un-captured carrier loses (dispatch) |
| projection pairing (QKV, gate·up) on llama2.c | 1.24× vs Inductor | retired RESEARCH_WRITEUP §3 — rediscovers `MergedColumnParallelLinear`/`QKVParallelLinear` |
| autotuned selection (`optimize_model_autotuned`) | matrix_chain 2.24× CPU | retired RESEARCH_WRITEUP §3 |
| the carrier below the crossover | 0.184× at retnet T=128 (uncompiled) | retired RESEARCH_WRITEUP §4 — superseded by `docs/results.md`'s 0.51× at T=512 |
| pipeline cost at T=2048 | MLOpt 215.8 s → 1.04 ms vs Inductor 45.4 s → 0.256 ms | retired RESEARCH_WRITEUP §4; also `project/plans/0005-cost-model-fidelity.md` — Inductor still faster at runtime |
| real checkpoints (stories15M/110M) | all blocks transform + verify (~2e-5); **−37% GEMM launches** profiler-verified; **parity** on wall time | retired RESEARCH_WRITEUP §4 |
| cost-model fidelity, post-fix ρ | `lowering_min` 0.167→0.261, `exec_generic` 0.070→0.316 | retired RESEARCH_WRITEUP §3 — positive but not yet predictive (pre-fix: `project/plans/0005-cost-model-fidelity.md`) |

## Learned models (2026-10-15)

*What the project trains, and how each performs.*  Controlled
same-budget baseline + ablations (2026-10-19):
[`docs/results_learned_baseline.md`](results_learned_baseline.md) /
[`docs/results_learned_ablations.md`](results_learned_ablations.md)
(`bench/drivers/policy/policy_baseline.py`, `bench/drivers/policy/policy_ablations.py`); the
feature-schema leaderboard — subsets × orderings of the shipped
featurizer parts, cold-trained at equal budget —
[`docs/results_schema_leaderboard.md`](results_schema_leaderboard.md)
(`bench/drivers/policy/schema_leaderboard.py`).  Separate from the
architectures above: these **steer** the search; the models in §1 are
what gets optimized.

### Neural — contraction ordering

| model | training | metric | value (vs teacher = 1.0) |
|---|---|---|---|
| `PairPolicyNet` | **RL** (curriculum REINFORCE), 25 KB | equal-rollouts cost ratio | n=20 **0.96** / n=30 **0.94** ✓; n=40 1.26 ✗ |
| `PairPolicyNet` **distilled** (`oe-all`, *shipped default*) | supervised (teacher's trial distribution) | equal-rollouts cost ratio | n=40 **0.91** ✓ — first arm to beat the teacher |
| `oe-best` (distilled, selected order) | supervised | equal-rollouts cost ratio | n=40 1.17 ✗ |
| `dp` (imitation of the RL curriculum) | supervised | equal-rollouts cost ratio | n=40 **127×** — the curriculum is a bad teacher |

Player is `best_order` (best-of-N sampled rollouts); the single argmax
is brittle at n=40 for every arm (RL 4.15×).  Not wired into
`Optimizer.optimize` — a standalone research artifact.

### Neural — rule value (supervised)

| model | metric | value | baseline | verdict |
|---|---|---|---|---|
| `RuleValueNet` (MLP 32) | ranking AP (`rank_pool`) | 0.023 | heuristic 0.045 | **loses** |

> Re-measured 2026-10-19 trained on on-distribution held-out candidates: AP **0.041** — heuristic parity (0.045). The 0.023 pin trained on off-distribution corpus samples. See `docs/results_learned_baseline.md`.
>
> Later still: the loss-form probe (`docs/results_neural_rank.md`) shows the residual gap was ordering credit — a pairwise RankNet objective on the reward's label reaches AP **0.1941** (above the trained-linear anchor 0.128, under the page-aware 0.210), and a ListNet-graded arm earns heuristic-parity episode reward **0.3549** vs 0.3595. Parity, not a win.

### Linear — the discovery game (RL, baked presets)

| model | metric | random | cold `{}` | trained | baseline |
|---|---|---|---|---|---|
| **open-game prior** (`OPEN_GAME_PLAYER_WEIGHTS`) | reward / episode | 0.134 | 0.143 | **0.304** | — |
| | useful laws / episode | 0.083 | 0.083 | **0.167** | — |
| | ranking AP | 0.012 | 0.012 | **0.133** | heuristic 0.045 |
| **meta-arena prior** (`META_ARENA_PLAYER_WEIGHTS`) | reward / episode | −0.57 | −0.49 | **−0.06** | scripted −0.10 |

Regenerate with `scripts/train_open_prior.py` / `scripts/train_meta_prior.py`.

Macro-action minting over `Trajectory` sequences (2026-10-19):
[`docs/results_macro_moves.md`](results_macro_moves.md)
(`scripts/macro_miner.py`) — mined `claim→extract` /
`declare→saturate→handle→extract` replayed as single refereed moves;
scripted+macros beats the base move set +0.19 reward/ep on the
held-out meta stream. **Shipped:** the 8-macro table is
`lawdata.MACROS` and `meta_arena.legal_actions` offers it to every
player (`mlopt_discovery.macro_moves`); the macro-trained prior's
bake gate is still unearned — sampled decode +0.007 < +0.088,
`v2-base` −0.072 < −0.056
([`docs/results_macro_bake2.md`](results_macro_bake2.md)).

**Honest note:** in *both* families the best result is not the neural
one — distillation beats neural RL at n=40, and the linear prior beats
the neural rule-value net.
