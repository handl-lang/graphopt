# catopt — measured results

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
| **WIN** | bounded rewrites (error_budget) buy wall-time on a real checkpoint | 4/5 budgets delivered a bounded member; best 1.322× vs eager |
| **WIN** | every budget's optimized module stays within the fp32 gate | 5/5 budgets optimized cleanly |

Measured 2026-10-02T01:00:43+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `b63355d` (dirty)) — baseline [`bench/baselines/bounded_e2e.json`](../bench/baselines/bounded_e2e.json).

## MiniGPT end-to-end

*Does composition (pairing + fold) beat plain Inductor end-to-end?*

**Expected** — PARITY — pairing fires per block, verified fp64-exact; wall time is ~parity.

| verdict | finding | headline |
|---|---|---|
| **WIN** | the composed catopt+Inductor module beats plain Inductor end-to-end | best 1.03× vs Inductor on tiny_llama@4x256 |
| **WIN** | the optimized module verifies against eager | 5/5 cells verified |

Measured 2026-10-02T00:47:33+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `f1e5d1a` (dirty)) — baseline [`bench/baselines/e2e_model.json`](../bench/baselines/e2e_model.json).

## Compositional pairing demo

*Does per-model lowering autotune pick the measured-fastest variant?*

**Expected** — WIN — the reported pick is the measured winner, never a static choice.

| verdict | finding | headline |
|---|---|---|
| **WIN** | the Autotuned lowering beats eager on at least one model | best 2.55× vs eager on matrix_chain |
| **WIN** | every model's optimized lowering verifies | 4/4 models verified |

Measured 2026-10-02T00:19:09+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/killer_demo.json`](../bench/baselines/killer_demo.json).

## Rewrite-law effect bench

*Does each registered rewrite law fire, get picked by extraction, and lower to a verified term?*

**Expected** — WIN — every registered law fires/picks/verifies; non-firing laws report honestly.

| verdict | finding | headline |
|---|---|---|
| **WIN** | every registered law fires, is picked by extraction, and lowers to a verified term | 11 verified / 11 picked / 13 fired across 13 cells |

Measured 2026-10-02T00:18:07+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/law_bench.json`](../bench/baselines/law_bench.json).

## Morphism-window composition

*Do morphism windows convert term-flops into wall time?*

**Expected** — WIN — term-FLOP reduction converts to measured wall time at GEMM-bound sizes.

| verdict | finding | headline |
|---|---|---|
| **WIN** | morphism windows convert term-level flops headroom into measured wall time | best 14.83× vs eager on chain_x4@4096x128 |
| **WIN** | every morphism rewrite verifies equivalent | 14/14 cells verified |

Measured 2026-10-02T00:40:26+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `f1e5d1a` (dirty)) — baseline [`bench/baselines/morphism_e2e.json`](../bench/baselines/morphism_e2e.json).

## Weight-chain reassociation vs Inductor

*Can the e-graph find a form Inductor's post-grad graph cannot express?*

**Expected** — WIN — the e-graph reaches a weights-first form Inductor's post-grad graph cannot express (see docs/results.md).

| verdict | finding | headline |
|---|---|---|
| **WIN** | catopt reaches a weight-folded form Inductor's post-grad graph cannot express | 17.12× vs Inductor at (k,d,B·T)=(16,512,4096) (18.63× vs eager) |
| **WIN** | every timed cell passes the fp32 equivalence gate | 12/12 cells verified |

Measured 2026-10-02T00:22:06+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/reassoc_scale.json`](../bench/baselines/reassoc_scale.json).

## Tooling-level measurements (hand-maintained)

This file is appended verbatim to `docs/results.md` by
`bench results` / `render_results_doc` — edit it here, never in the
generated file.  These measurements live in `tools/` and are
recorded in the retros, not in `bench/baselines/`.

All measured on the dev box (RTX 2050 / i5-12500H, fp64 — same
hardware class as the baselines); magnitudes do not extrapolate to
datacenter hardware:

| measurement | result | tool | retro |
|---|---|---|---|
| `softmax_fold` on `ManualSoftmaxAttention` | **+13–27% eager, +29–48% CUDA-graph** vs raw | `tools/law_wallclock.py` | [law-wallclock-verification](../project/retros/law-wallclock-verification.md) |
| `select_mul` marginal (optimized vs law-ablated) | **faster in 19/20 measurements** | `tools/law_wallclock.py` | [law-wallclock-verification](../project/retros/law-wallclock-verification.md) |
| executor routing after measured pricing | **12/12 cases ship the measured-fastest member** (uncorrected picks ran 2–3× slower) | `tools/executor_cost_probe.py` | [executor-pricing-fix](../project/retros/executor-pricing-fix.md) |
| `silu_fold` on the bench case | **−53–55% modeled cost, 1.43–2.62× wall-clock** | `bench run law_bench` | [three-cell-mediator](../project/retros/three-cell-mediator.md) |
| contraction player, n=40 vs `opt_einsum` | **0.89–1.01× of randomised greedy at equal wall-clock** (0.91–0.98 at T≤1, three seeds, fed budgets); **0.51–0.90× deterministic greedy** | `tools/contraction_guided_restart.py` | [contraction-synthesis](../project/retros/contraction-synthesis.md), [contraction-player-shipped](../project/retros/contraction-player-shipped.md) |
| pipeline held-out rediscovery | **winner re-ranks #1, SHIP, every run** | `catopt_discovery.pipeline --holdout` | [law-proposer-extensions](../project/retros/law-proposer-extensions.md), [groupnorm-convnext-corpus](../project/retros/groupnorm-convnext-corpus.md) |
| coherence catalogue over `ALL_RULES` (82) | **47 axioms / 13 lemmas / 2 redundant; 9 no-instance; divergence 1** (`rms_norm_fold` × `_nogain` — measured, unresolved) | `catopt_discovery.coherence --emit-basis` | [axiom-lemma-split](../project/retros/axiom-lemma-split.md) |
| evidence store, second run | **50/50 verdicts cached; 128 s → 6.6 s (~19×)** | `catopt_discovery.pipeline --evidence-db` | [evidence-store](../project/retros/evidence-store.md) |
| `chainfuse` — laws + `claim(gen_0)`, Triton-bound | synthetic composite (`x@A@B@C` + 6-op pointwise tail, 256×512): **2.94× vs `torch.compile`** (86µs vs 253µs), verified 1.7e-05; laws alone reach 181µs — the minted kernel is required for the full win | `python -m catopt_discovery.play --domain gen --probe` | [gen-kernels](../project/retros/gen-kernels.md) |
| delivered module vs `torch.compile` (`gen` probe) | spelled matmul chain **3.96×** (22µs vs 87µs; Inductor keeps all three GEMMs even under `freezing`); swiglu **2.11×**; pointwise fusion **0.82–0.88× loss** | `python -m catopt_discovery.play --domain gen --probe` | [gen-kernels](../project/retros/gen-kernels.md) |
| held-out zoo probe | **11/18 delivered verified modules beat `torch.compile`** — LoRAAdapter 2.18×, WaveNetGate 1.51×, DeepEquilibrium 1.35×; every winner ran `claims=[]` (the `gk.profitable` measured referee filtered unprofitable minted claims) | `python -m catopt_discovery.play --zoo` | [gen-kernels](../project/retros/gen-kernels.md) |
| joint fwd+bwd step vs `torch.autograd` | **+13%** (68µs vs 78µs, verified 9.5e-07) — one shared memo across forward + derived VJP; unshared costs 197µs | `catopt_discovery.training.joint_step` | [joint-training-graph](../project/retros/joint-training-graph.md) |
| machine law pack | **16/16 store objects deployable** as an opt-in `RuleSet`; machine-only arm wins 3/22 zoo models the human library misses (`linear_chain_t` owns LoRAAdapter's −20%) | `catopt_discovery.machine_pack` | [human-bar](../project/retros/human-bar.md) |

Two reconciliation notes:

- **The contraction "~1.8×" claim is stale.**  Early numbers had
  the learned policy at 1.54–1.78× of `opt_einsum`'s randomised
  greedy at n=40; the RL curriculum regime narrowed the gap to
  ~1.04×, and the oe-all **distilled** player that now ships as
  `catopt_torch.load_contraction_policy()`'s default *beats* the
  teacher at equal wall-clock (the figure in the table above).  The
  curriculum-RL weights remain bundled as an alternate artifact.
- **The `bounded_e2e` baseline's headline is vs eager** (best
  1.322×, above).  Its per-cell `speedup_vs_inductor` spans
  0.94–1.25 (1.16–1.25 over the budgets that delivered a bounded
  member) — so the README's "1.15–1.32× vs Inductor" is not the
  pinned baseline's literal column: the 1.32 is vs eager and the
  vs-Inductor delivered range tops out at 1.25 on this baseline.

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
| `ChunkedRetention` (RetNet-style) | 512 | — | **1.82–2.33×** | 1.3e-8 |
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

**Compute — FLOPs:**

| optimization | saved | instrument |
|---|---|---|
| weight fold (k GEMMs → 1) | **k×** (2–16×) | `flops_cost` |
| K-way chunk construct | **4.4×** vs bisection; 5.83× vs fused | `flops_cost` |
| shared-state reassociation | **10.8–24.5%** | `flops_cost` |

**Storage — parameter bytes** (`param_bytes_cost`, d=512, fp32,
`x@W₀@…@W_{k-1}` vs the folded `x@W_cat`):

| k | before | after | saved |
|---|---|---|---|
| 2 | 512 KB | 256 KB | 256 KB (2.0×) |
| 4 | 1.0 MB | 256 KB | 768 KB (4.0×) |
| 8 | 2.0 MB | 256 KB | 1.75 MB (8.0×) |
| 16 | 4.0 MB | 256 KB | **3.75 MB (16.0×)** |

Plus the shared-input MoE fold: **196,608 B (192 KB)** eliminated
(pinned by test).

### 4. Oracle-evaluation savings [symbolic]

| | saved |
|---|---|
| saturating e-graph encodes every bracketing | 16,796 programs in **231 e-nodes** (k=11); ≥10^5.6 in ~0.5 s (k=24) |
| trained prior vs cost-delta heuristic | AP **0.133 vs 0.045** (~3×), ~2× more useful laws per eval |
| evidence store, second run | 50/50 verdicts cached: **128 s → 6.6 s (19×)** |

### 5. Honest losses

| finding | measured |
|---|---|
| the carrier below the crossover | **0.51×** vs eager at T=512 |
| the *folded* carrier vs the spelled form | a fold is a **cost-neutral relabelling** — the gap is search, not pricing |
| pure pointwise fusion | **0.82× loss** — Inductor's home turf |
| shared-state on the real export | 10.8–24.5% [symbolic]; the ~589,952-scale target is unreached (a rule-set loop blocks the bridge) |

## Learned models (2026-10-15)

*What the project trains, and how each performs.*  Separate from the
architectures above: these **steer** the search; the models in §1 are
what gets optimized.

### Neural — contraction ordering

| model | training | result |
|---|---|---|
| `PairPolicyNet` | **RL** (curriculum REINFORCE), 25 KB | equal-rollouts n=20 **0.96** / n=30 **0.94** (beats the teacher); n=40 1.26; wall-clock parity 1.00–1.02 at n≤30 |
| `PairPolicyNet` **distilled** (`oe-all`, *shipped default*) | supervised (teacher's trial distribution) | n=40 **0.91** — the first arm to beat the teacher |
| `oe-best` (distilled, selected order) | supervised | n=40 1.17 |
| `dp` (imitation of the RL curriculum) | supervised | **127×** at n=40 — the curriculum is a bad teacher |

Player is `best_order` (best-of-N sampled rollouts); the single argmax
is brittle at n=40 for every arm (RL 4.15×).  Not wired into
`Optimizer.optimize` — a standalone research artifact.

### Neural — rule value (supervised)

| model | result |
|---|---|
| `RuleValueNet` (MLP 32) | `rank_pool` AP ~0.023 vs the cost-delta heuristic's 0.045 — **loses** |

### Linear — the discovery game (RL, baked presets)

| model | result |
|---|---|
| **open-game prior** (`OPEN_GAME_PLAYER_WEIGHTS`) | reward/ep **0.304** vs random 0.134 / cold 0.143; useful/ep 0.167 vs 0.083; **AP 0.133 vs heuristic 0.045** |
| **meta-arena prior** (`META_ARENA_PLAYER_WEIGHTS`) | **−0.06** vs random −0.57, cold −0.49, hand-written `scripted` −0.10 |

Regenerate with `tools/train_open_prior.py` / `tools/train_meta_prior.py`.

**Honest note:** in *both* families the best result is not the neural
one — distillation beats neural RL at n=40, and the linear prior beats
the neural rule-value net.
